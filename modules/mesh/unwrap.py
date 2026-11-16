"""Cutting parts into charts, flattening them and packing them into one texture atlas."""
# Faces cluster into charts whose normals stay within an angle of the chart's direction, which
# flatten by least squares conformal maps (Levy et al. 2002), optionally relax as rigidly as
# possible (Liu et al. 2008), and pack at one texel density.

from __future__ import annotations

__all__ = [
    "QUALITIES",
    "UnwrapSettings",
    "Unwrapped",
    "distortion",
    "flatten",
    "segment",
    "unwrap_parts",
]

import math
from dataclasses import dataclass, field

import torch

from ..log import get_logger
from ..model3d.scene import Part
from . import device as mesh_device
from . import bake, pack, topology
from .tangents import split_part

logger = get_logger("mesh.unwrap")

#: Unwrap qualities, fastest first.
QUALITIES = ("fast", "balanced", "best")

#: Conformal solve tolerance and iteration cap, rigid relaxation passes and packing grid, per quality.
QUALITY_STEPS = {
    "fast": {"tolerance": 1e-6, "iterations": 300, "rigid": 0, "grid": 1024},
    "balanced": {"tolerance": 1e-7, "iterations": 1000, "rigid": 3, "grid": 1024},
    "best": {"tolerance": 1e-8, "iterations": 3000, "rigid": 10, "grid": 2048},
}

#: Share of the chart angle used when faces at fault are cut into charts of their own.
CUT_ANGLE = 0.5

#: Largest chart angle, in degrees.
MAX_CHART_ANGLE = 85.0

#: Degrees a single face may stray past the chart angle while the chart's area stays within it.
SPREAD_SLACK = 25.0

#: Largest angle any face of a chart may make with the chart's direction, in degrees.
MAX_SPREAD = 88.0

#: Largest stretch ratio one triangle adds to the angle distortion figure.
ANGLE_CAP = 10.0

#: Most merge rounds of the face clustering.
MERGE_ROUNDS = 1000

#: Weight of a merged chart's distance from a disc in the merge cost.
ROUNDNESS_WEIGHT = 0.25

#: Weight of the seam a merge removes, as a share of the smaller chart's border.
SEAM_WEIGHT = 0.5

#: Rounds of splitting charts that overlap or fold themselves.
REPAIR_ROUNDS = 6

#: Widest grid charts are drawn on when looking for self-overlaps.
CHECK_SIDE = 1024

#: Repair rounds that cut out only the faces at fault before whole charts are halved.
LOCAL_ROUNDS = 3

#: Most triangles one part may hold.
MAX_TRIANGLES = 4_000_000

#: Cost added to merging across hard edges, by the share of the shared border they make up.
HARD_EDGE_COST = 1.0

#: Share of a chart's texture area that may lie in flipped slivers before the chart is cut.
FOLD_SHARE = 1e-3

#: Largest cotangent weight in the flattening solves.
COT_CLAMP = 50.0

#: A solved chart smaller than this share of its projected area is replaced by the projection.
COLLAPSED = 1e-2

#: Relative residual and step cap of each rigid relaxation solve.
RIGID_TOLERANCE = 1e-7
RIGID_STEPS = 300

#: Largest block of unknowns solved by a dense factorisation instead of conjugate gradients.
DENSE_UNKNOWNS = 2400

#: Most bytes of dense blocks factored at once.
DENSE_BYTES = 256 * 1024 * 1024


@dataclass
class UnwrapSettings:
    """How an unwrap runs.

    Attributes:
        chart_angle: Degrees a face's normal may turn from its chart's direction.
        resolution: Atlas width and height in pixels.
        padding: Pixels kept free between charts.
        quality: One of `QUALITIES`.
    """

    chart_angle: float = 60.0
    resolution: int = 2048
    padding: int = 4
    quality: str = "balanced"


@dataclass
class Unwrapped:
    """What an unwrap produced for one group of parts.

    Attributes:
        parts: The parts on their new vertices, carrying the new texture coordinates.
        sources: Per part, ``(V2,)`` int64 original vertex of each new vertex.
        charts: Number of charts.
        face_charts: Per part, ``(F,)`` int64 chart of each triangle, numbered across all parts.
        stats: Named figures: efficiency, distortion, flips, overlaps.
    """

    parts: list[Part]
    sources: list[torch.Tensor]
    charts: int
    face_charts: list[torch.Tensor] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Segmentation


def _face_links(vertices: torch.Tensor, faces: torch.Tensor):
    """Face pairs across two-face edges wound the same way, with edge ids, lengths and face perimeters.

    Returns:
        ``(ids, pairs, lengths, perimeter)``: ``(P,)`` edge ids, ``(P, 2)`` faces, ``(P,)``
        float64 edge lengths and ``(F,)`` float64 perimeter of each face.
    """
    topo = topology.edge_topology(faces, vertices.shape[0])
    ids, pairs = topo.manifold_face_pairs()
    reversed_flat = topo.face_edge_reversed
    consistent = (reversed_flat[pairs[:, 0], topo.edge_corners[ids, 0]]
                  != reversed_flat[pairs[:, 1], topo.edge_corners[ids, 1]])
    ids, pairs = ids[consistent], pairs[consistent]
    every = topology.edge_lengths(vertices.double(), topo.edges)
    perimeter = every[topo.face_edges].sum(dim=1)
    return ids, pairs, every[ids], perimeter


def _adjacency(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed face adjacency across two-face edges wound the same way."""
    _, pairs, _, _ = _face_links(vertices, faces)
    return torch.cat([pairs[:, 0], pairs[:, 1]]), torch.cat([pairs[:, 1], pairs[:, 0]])


def _relabel(chart: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Charts split into edge-connected pieces and renumbered from 0."""
    same = chart[source] == chart[target]
    roots = topology.union_find(chart.shape[0], source[same], target[same])
    _, labels = torch.unique(roots, return_inverse=True)
    return labels


def _mean_normals(chart: torch.Tensor, count: int, normals: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Area-weighted unit normal of each chart."""
    total = torch.zeros(count, 3, dtype=normals.dtype, device=normals.device)
    total.index_add_(0, chart, normals * areas[:, None])
    return torch.nn.functional.normalize(total, dim=1)


def _angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Angle in radians between rows of two unit vector arrays."""
    return torch.acos((first * second).sum(dim=-1).clamp(-1.0, 1.0))


def _roundness(area: torch.Tensor, perimeter: torch.Tensor) -> torch.Tensor:
    """How far a region is from a disc: 0 for a disc, towards 1 for a long or ragged one."""
    return (1.0 - 4.0 * math.pi * area / (perimeter * perimeter).clamp_min(1e-300)).clamp(0.0, 1.0)


def segment(vertices: torch.Tensor, faces: torch.Tensor, chart_angle: float,
            hard_edges: torch.Tensor | None = None, polygon: torch.Tensor | None = None) -> torch.Tensor:
    """Cut a mesh into charts whose face normals stay within an angle of the chart's direction.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices, welded where positions match.
        chart_angle: Degrees a face's normal may turn from its chart's direction.
        hard_edges: ``(E,)`` bool per edge of ``edge_topology(faces)``, edges preferred as seams.
        polygon: ``(F,)`` polygon of each triangle, kept in one chart, or None.

    Returns:
        ``(F,)`` int64 chart of each face, ``0`` to ``C - 1``, each one connected piece.
    """
    # Every face starts as a chart; each round merges neighbouring charts that pick each other as
    # their cheapest merge, as hierarchical face clustering does (Garland et al. 2001). The cost
    # weighs the normal spread, the seam a merge removes, the chart's roundness and hard edges.
    device = faces.device
    face_count = int(faces.shape[0])
    if face_count == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)
    angle = min(max(float(chart_angle), 1.0), MAX_CHART_ANGLE)
    limit = math.radians(min(angle + SPREAD_SLACK, MAX_SPREAD))
    # A spherical cap of half-angle `angle` has this mean of 1 - cos over its area.
    deviation_limit = (1.0 - math.cos(math.radians(angle))) / 2.0
    work = vertices.double()
    normals = topology.face_normals(work, faces)
    areas = topology.face_areas(work, faces)
    counted = normals.norm(dim=1) > 0
    ids, pairs, lengths, face_perimeter = _face_links(work, faces)
    hard = hard_edges[ids].double() if hard_edges is not None else torch.zeros_like(lengths)
    source = torch.cat([pairs[:, 0], pairs[:, 1]])
    target = torch.cat([pairs[:, 1], pairs[:, 0]])
    index = torch.arange(face_count, device=device)
    if polygon is not None and polygon.numel() == face_count:
        polygon = polygon.long().to(device)
        first_face = torch.full((int(polygon.max()) + 1,), face_count, dtype=torch.int64, device=device)
        first_face.scatter_reduce_(0, polygon, index, reduce="amin")
        chart = first_face[polygon]
        inside = polygon[source] == polygon[target]
        source, target = torch.cat([source, source[inside]]), torch.cat([target, target[inside]])
    else:
        chart = index.clone()

    total = torch.zeros(face_count, 3, dtype=torch.float64, device=device)
    total.index_add_(0, chart, normals * areas[:, None])
    area = torch.zeros(face_count, dtype=torch.float64, device=device).index_add_(0, chart, areas)
    perimeter = torch.zeros(face_count, dtype=torch.float64, device=device).index_add_(0, chart, face_perimeter)
    link_a, link_b = chart[pairs[:, 0]], chart[pairs[:, 1]]
    inner = link_a == link_b
    perimeter.index_add_(0, link_a[inner], -2.0 * lengths[inner])
    link_a, link_b = link_a[~inner], link_b[~inner]
    link_length, link_hard = lengths[~inner], lengths[~inner] * hard[~inner]

    for _ in range(MERGE_ROUNDS):
        if link_a.numel() == 0:
            break
        low, high = torch.minimum(link_a, link_b), torch.maximum(link_a, link_b)
        keys, inverse = torch.unique(low * face_count + high, return_inverse=True)
        a, b = keys // face_count, keys % face_count
        shared = torch.zeros(keys.shape[0], dtype=torch.float64, device=device).index_add_(0, inverse, link_length)
        shared_hard = torch.zeros_like(shared).index_add_(0, inverse, link_hard)
        link_a, link_b, link_length, link_hard = a, b, shared, shared_hard

        length = total.norm(dim=1)
        mean = total / length.clamp_min(1e-300)[:, None]
        face_turn = torch.where(counted, _angle(normals, mean[chart]), torch.zeros_like(areas))
        spread = torch.zeros(face_count, dtype=torch.float64, device=device)
        spread.scatter_reduce_(0, chart, face_turn, reduce="amax")
        joined = total[a] + total[b]
        joined_length = joined.norm(dim=1)
        union = joined / joined_length.clamp_min(1e-300)[:, None]
        turn_a = torch.where(length[a] > 0, _angle(mean[a], union), torch.zeros_like(shared))
        turn_b = torch.where(length[b] > 0, _angle(mean[b], union), torch.zeros_like(shared))
        # The merged chart's normals lie within each side's spread plus the turn of its mean.
        bound = torch.where(joined_length > 0, torch.maximum(spread[a] + turn_a, spread[b] + turn_b),
                            torch.zeros_like(shared))
        merged_area = area[a] + area[b]
        deviation = torch.where(merged_area > 0, 1.0 - joined_length / merged_area.clamp_min(1e-300),
                                torch.zeros_like(shared))
        allowed = (bound <= limit) & (deviation <= deviation_limit)
        if not bool(allowed.any()):
            break
        merged_perimeter = (perimeter[a] + perimeter[b] - 2.0 * shared).clamp_min(0.0)
        roundness = _roundness(merged_area, merged_perimeter)
        smaller = torch.minimum(perimeter[a], perimeter[b]).clamp_min(1e-300)
        cost = (deviation / deviation_limit + ROUNDNESS_WEIGHT * roundness - SEAM_WEIGHT * (shared / smaller).clamp_max(1.0)
                + HARD_EDGE_COST * shared_hard / shared.clamp_min(1e-300))
        cost = torch.where(allowed, cost, torch.full_like(cost, math.inf))
        best = torch.full((face_count,), math.inf, dtype=torch.float64, device=device)
        best.scatter_reduce_(0, a, cost, reduce="amin")
        best.scatter_reduce_(0, b, cost, reduce="amin")
        partner = torch.full((face_count,), face_count, dtype=torch.int64, device=device)
        fits_a = torch.isfinite(cost) & (cost <= best[a])
        fits_b = torch.isfinite(cost) & (cost <= best[b])
        partner.scatter_reduce_(0, a[fits_a], b[fits_a], reduce="amin")
        partner.scatter_reduce_(0, b[fits_b], a[fits_b], reduce="amin")
        chosen = partner.clamp_max(face_count - 1)
        mutual = (partner < face_count) & (partner[chosen] == index) & (index < partner)
        if not bool(mutual.any()):
            break
        keep, gone = index[mutual], partner[mutual]
        at = torch.searchsorted(keys, keep * face_count + gone)
        perimeter[keep] = perimeter[keep] + perimeter[gone] - 2.0 * shared[at]
        total[keep] += total[gone]
        area[keep] += area[gone]
        into = index.clone()
        into[gone] = keep
        chart = into[chart]
        link_a, link_b = into[link_a], into[link_b]
        across = link_a != link_b
        link_a, link_b, link_length, link_hard = link_a[across], link_b[across], link_length[across], link_hard[across]
    return _relabel(chart, source, target)


def split_charts(vertices: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, which: torch.Tensor,
                 polygon: torch.Tensor | None = None) -> torch.Tensor:
    """Cut charts in two across their longest direction.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        chart: ``(F,)`` chart of each face.
        which: ``(C,)`` bool, the charts to cut.
        polygon: ``(F,)`` polygon of each triangle, kept on one side, or None.

    Returns:
        ``(F,)`` chart labels, renumbered from 0.
    """
    count = int(chart.max()) + 1
    centres = vertices[faces].double().mean(dim=1)
    if polygon is not None and polygon.numel() == chart.numel():
        polygon = polygon.long()
        total = torch.zeros(int(polygon.max()) + 1, 3, dtype=torch.float64, device=vertices.device)
        total.index_add_(0, polygon, centres)
        tally = torch.bincount(polygon, minlength=total.shape[0]).clamp_min(1).double()
        centres = (total / tally[:, None])[polygon]
    tally = torch.bincount(chart, minlength=count).clamp_min(1).double()
    mean = torch.zeros(count, 3, dtype=torch.float64, device=vertices.device)
    mean.index_add_(0, chart, centres)
    mean = mean / tally[:, None]
    offset = centres - mean[chart]
    spread = torch.zeros(count, 3, 3, dtype=torch.float64, device=vertices.device)
    spread.index_add_(0, chart, offset[:, :, None] * offset[:, None, :])
    _, vectors = torch.linalg.eigh(spread + torch.eye(3, dtype=torch.float64, device=vertices.device) * 1e-30)
    axis = vectors[:, :, 2]
    side = (offset * axis[chart]).sum(dim=1) > 0
    moved = side & which[chart]
    return torch.where(moved, chart + count, chart)


# ---------------------------------------------------------------------------------------------
# Flattening


def _plane_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit axes spanning the plane of each normal, right-handed with it."""
    axis = torch.zeros_like(normals)
    axis.scatter_(1, normals.abs().argmin(dim=1, keepdim=True), 1.0)
    first = torch.nn.functional.normalize(torch.cross(axis, normals, dim=1), dim=1)
    second = torch.cross(normals, first, dim=1)
    return first, second


def _flat_area(uvs: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, count: int) -> torch.Tensor:
    """Unsigned texture-space area of each chart."""
    total = torch.zeros(count, dtype=torch.float64, device=uvs.device)
    total.index_add_(0, chart, _signed_areas(uvs, faces).abs().double())
    return total


def _signed_areas(uvs: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Signed area of each triangle in texture space, positive counter-clockwise."""
    corners = uvs[faces]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _pins(projected: torch.Tensor, vertex_chart: torch.Tensor, count: int) -> torch.Tensor:
    """Two vertices per chart, far apart along its main direction."""
    device = projected.device
    tally = torch.bincount(vertex_chart, minlength=count).clamp_min(1).double()
    mean = torch.zeros(count, 2, dtype=torch.float64, device=device)
    mean.index_add_(0, vertex_chart, projected.double())
    mean = mean / tally[:, None]
    offset = projected.double() - mean[vertex_chart]
    moments = torch.zeros(count, 3, dtype=torch.float64, device=device)
    moments.index_add_(0, vertex_chart, torch.stack([offset[:, 0] ** 2, offset[:, 1] ** 2,
                                                     offset[:, 0] * offset[:, 1]], dim=1))
    angle = 0.5 * torch.atan2(2 * moments[:, 2], moments[:, 0] - moments[:, 1])
    along = offset[:, 0] * torch.cos(angle)[vertex_chart] + offset[:, 1] * torch.sin(angle)[vertex_chart]
    index = torch.arange(projected.shape[0], device=device)
    picks = []
    for value in (along, -along):
        top = torch.full((count,), -math.inf, dtype=torch.float64, device=device)
        top.scatter_reduce_(0, vertex_chart, value, reduce="amax")
        hit = value >= top[vertex_chart]
        first = torch.full((count,), projected.shape[0], dtype=torch.int64, device=device)
        first.scatter_reduce_(0, vertex_chart[hit], index[hit], reduce="amin")
        picks.append(first)
    low, high = picks
    clash = low == high
    if bool(clash.any()):
        other = torch.full((count,), projected.shape[0], dtype=torch.int64, device=device)
        spare = index != low[vertex_chart]
        other.scatter_reduce_(0, vertex_chart[spare], index[spare], reduce="amin")
        high = torch.where(clash & (other < projected.shape[0]), other, high)
    return torch.cat([low, high])


def _conformal(vertices: torch.Tensor, faces: torch.Tensor, start: torch.Tensor, pins: torch.Tensor,
               vertex_chart: torch.Tensor, tolerance: float, iterations: int) -> tuple[torch.Tensor, int]:
    """Least squares conformal coordinates with the pinned vertices held at their start values.

    Returns:
        ``(uvs, iterations)``: ``(N, 2)`` float64 coordinates and the solver steps taken.
    """
    count = int(vertices.shape[0])
    device = vertices.device
    laplacian_coo = topology.cotangent_laplacian(vertices, faces, clamp=COT_CLAMP)
    laplacian = laplacian_coo.to_sparse_csr()
    edges = topology.edge_topology(faces, count)
    rim = torch.nonzero(edges.boundary).squeeze(1)
    face = edges.edge_faces[rim, 0]
    side = edges.edge_corners[rim, 0]
    tail = faces[face, side]
    head = faces[face, (side + 1) % 3]
    rows = torch.cat([tail, head])
    cols = torch.cat([head, tail])
    values = torch.cat([torch.full_like(tail, 1, dtype=torch.float64) * 0.5,
                        torch.full_like(tail, 1, dtype=torch.float64) * -0.5])
    area = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (count, count),
                                   check_invariants=False).coalesce().to_sparse_csr()
    free = torch.ones(count, dtype=torch.bool, device=device)
    free[pins] = False
    free_index = torch.nonzero(free).squeeze(1)
    held = start.double().clone()
    held[free_index] = 0.0
    free_count = int(free_index.shape[0])
    position = torch.full((count,), -1, dtype=torch.int64, device=device)
    position[free_index] = torch.arange(free_count, device=device)

    def apply_full(full: torch.Tensor) -> torch.Tensor:
        smooth = laplacian @ full
        turn = area @ full
        return torch.stack([smooth[:, 0] - turn[:, 1], turn[:, 0] + smooth[:, 1]], dim=1)

    def apply(values_free: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(count, 2, dtype=torch.float64, device=device)
        full[free_index, 0] = values_free[:free_count, 0]
        full[free_index, 1] = values_free[free_count:, 0]
        result = apply_full(full)[free_index]
        return torch.cat([result[:, 0], result[:, 1]])[:, None]

    smooth_rows, smooth_cols = laplacian_coo.indices()
    smooth_values = laplacian_coo.values()
    area_coo = area.to_sparse_coo().coalesce()
    turn_rows, turn_cols = area_coo.indices()
    turn_values = area_coo.values()
    keep = (position[smooth_rows] >= 0) & (position[smooth_cols] >= 0)
    a, b, w = position[smooth_rows[keep]], position[smooth_cols[keep]], smooth_values[keep]
    rows_m = [a, a + free_count]
    cols_m = [b, b + free_count]
    values_m = [w, w]
    keep = (position[turn_rows] >= 0) & (position[turn_cols] >= 0)
    a, b, q = position[turn_rows[keep]], position[turn_cols[keep]], turn_values[keep]
    rows_m += [a, a + free_count]
    cols_m += [b + free_count, b]
    values_m += [-q, q]
    vertex_chart = vertex_chart[free_index]
    solver = _BlockSolver(torch.cat(rows_m), torch.cat(cols_m), torch.cat(values_m),
                          torch.cat([vertex_chart, vertex_chart]))
    rhs_full = -apply_full(held)[free_index]
    rhs = torch.cat([rhs_full[:, 0], rhs_full[:, 1]])
    guess = torch.cat([start[free_index, 0], start[free_index, 1]]).double()
    solution, steps = solver.solve(rhs[:, None], guess[:, None], apply, tolerance, iterations)
    result = start.double().clone()
    result[free_index, 0] = solution[:free_count, 0]
    result[free_index, 1] = solution[free_count:, 0]
    return result, int(steps)


class _BlockSolver:
    """Solves of a sparse symmetric positive definite system made of independent blocks."""
    # Blocks of up to `DENSE_UNKNOWNS` unknowns are factored densely, larger ones are left to
    # conjugate gradients.

    def __init__(self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, block: torch.Tensor):
        device = block.device
        self.unknowns = int(block.shape[0])
        self.block = block
        block_count = int(block.max()) + 1 if self.unknowns else 0
        sizes = torch.bincount(block, minlength=block_count)
        order = torch.argsort(block, stable=True)
        starts = torch.cumsum(sizes, 0) - sizes
        self.local = torch.empty_like(block)
        self.local[order] = torch.arange(self.unknowns, device=device) - starts[block[order]]
        dense = sizes <= DENSE_UNKNOWNS
        self.dense = dense[block]
        self.failed = torch.zeros(block_count, dtype=torch.bool, device=device)
        self.diagonal = torch.zeros(self.unknowns, dtype=values.dtype, device=device)
        on = rows == cols
        self.diagonal.index_add_(0, rows[on], values[on])
        self.groups = []
        padded = torch.zeros_like(sizes)
        ladder = 8
        while ladder <= DENSE_UNKNOWNS * 3 // 2:
            padded = torch.where((padded == 0) & (sizes <= ladder) & dense & (sizes > 0), ladder, padded)
            ladder = max(ladder + 1, ladder * 3 // 2)
        entry_block = block[rows]
        for width in torch.unique(padded[padded > 0]).tolist():
            members = torch.nonzero(padded == width).squeeze(1)
            per_batch = max(1, DENSE_BYTES // (width * width * 8))
            for begin in range(0, int(members.shape[0]), per_batch):
                ids = members[begin:begin + per_batch]
                slot = torch.full((block_count,), -1, dtype=torch.int64, device=device)
                slot[ids] = torch.arange(ids.shape[0], device=device)
                matrix = torch.zeros(ids.shape[0], width, width, dtype=values.dtype, device=device)
                spare = torch.arange(width, device=device)[None, :] >= sizes[ids][:, None]
                matrix.diagonal(dim1=1, dim2=2)[spare] = 1.0
                mine = slot[entry_block] >= 0
                matrix.index_put_((slot[entry_block[mine]], self.local[rows[mine]], self.local[cols[mine]]),
                                  values[mine], accumulate=True)
                factor, info = torch.linalg.cholesky_ex(matrix)
                self.failed[ids[info > 0]] = True
                self.groups.append((ids, slot, width, factor))

    def solve(self, rhs: torch.Tensor, guess: torch.Tensor, apply, tolerance: float, iterations: int):
        """Solve for every block, densely where factored and by conjugate gradients elsewhere.

        Args:
            rhs: ``(U, K)`` right-hand sides.
            guess: ``(U, K)`` starting values for the blocks left to conjugate gradients.
            apply: Callable applying the whole matrix to ``(U, K)`` values.
            tolerance: Relative residual the iterative part stops at.
            iterations: Step cap of the iterative part.

        Returns:
            ``(solution, steps)``; unknowns of a block that would not factor are NaN.
        """
        solution = guess.clone()
        for ids, slot, width, factor in self.groups:
            mine = slot[self.block] >= 0
            local_rhs = torch.zeros(ids.shape[0], width, rhs.shape[1], dtype=rhs.dtype, device=rhs.device)
            local_rhs[slot[self.block[mine]], self.local[mine]] = rhs[mine]
            local = torch.cholesky_solve(local_rhs, factor)
            solution[mine] = local[slot[self.block[mine]], self.local[mine]]
        steps = 0
        if not bool(self.dense.all()):
            inverse = torch.where(self.diagonal.abs() > 0, 1.0 / self.diagonal, torch.ones_like(self.diagonal))
            solution, steps, _ = topology.solve_cg(apply, rhs, x0=solution, tol=tolerance, max_iter=iterations,
                                                   preconditioner=inverse, return_info=True)
        failed = self.failed[self.block]
        solution[failed] = math.nan
        return solution, steps


def _local_frames(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Each triangle laid flat in its own plane, ``(F, 3, 2)``, first corner at the origin."""
    corners = vertices[faces]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    length = first.norm(dim=1).clamp_min(1e-30)
    axis = first / length[:, None]
    normal = torch.cross(first, second, dim=1)
    up = torch.nn.functional.normalize(torch.cross(normal, axis, dim=1), dim=1)
    flat = torch.zeros(faces.shape[0], 3, 2, dtype=vertices.dtype, device=vertices.device)
    flat[:, 1, 0] = length
    flat[:, 2, 0] = (second * axis).sum(dim=1)
    flat[:, 2, 1] = (second * up).sum(dim=1)
    return flat


def _rigid(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor, anchors: torch.Tensor,
           vertex_chart: torch.Tensor, passes: int, tolerance: float) -> torch.Tensor:
    """Relax coordinates towards rigid copies of each triangle, local and global steps alternating."""
    count = int(vertices.shape[0])
    device = vertices.device
    flat = _local_frames(vertices, faces)
    corners = vertices[faces]
    ahead = corners[:, [1, 2, 0]] - corners
    behind = corners[:, [2, 0, 1]] - corners
    cot = ((ahead * behind).sum(-1) / torch.cross(ahead, behind, dim=-1).norm(dim=-1).clamp_min(1e-30))
    cot = cot.clamp(0.0, COT_CLAMP)
    laplacian_coo = topology.cotangent_laplacian(vertices, faces, clamp=COT_CLAMP, nonnegative=True)
    laplacian = laplacian_coo.to_sparse_csr()
    free = torch.ones(count, dtype=torch.bool, device=device)
    free[anchors] = False
    free_index = torch.nonzero(free).squeeze(1)
    position = torch.full((count,), -1, dtype=torch.int64, device=device)
    position[free_index] = torch.arange(free_index.shape[0], device=device)
    rows, cols = laplacian_coo.indices()
    keep = (position[rows] >= 0) & (position[cols] >= 0)
    solver = _BlockSolver(position[rows[keep]], position[cols[keep]], laplacian_coo.values()[keep],
                          vertex_chart[free_index])
    # Edge k of a triangle runs from corner k + 1 to corner k + 2, opposite corner k.
    tail = [1, 2, 0]
    head = [2, 0, 1]
    current = uvs.double().clone()
    for _ in range(int(passes)):
        mapped = current[faces]
        cross = torch.zeros(faces.shape[0], 2, 2, dtype=torch.float64, device=device)
        for k in range(3):
            du = mapped[:, tail[k]] - mapped[:, head[k]]
            dx = flat[:, tail[k]] - flat[:, head[k]]
            cross += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        theta = torch.atan2(cross[:, 1, 0] - cross[:, 0, 1], cross[:, 0, 0] + cross[:, 1, 1])
        cos, sin = torch.cos(theta), torch.sin(theta)
        rhs = torch.zeros(count, 2, dtype=torch.float64, device=device)
        for k in range(3):
            dx = flat[:, tail[k]] - flat[:, head[k]]
            turned = torch.stack([cos * dx[:, 0] - sin * dx[:, 1], sin * dx[:, 0] + cos * dx[:, 1]], dim=1)
            weighted = 0.5 * cot[:, k, None] * turned
            rhs.index_add_(0, faces[:, tail[k]], weighted)
            rhs.index_add_(0, faces[:, head[k]], -weighted)

        def apply(values_free: torch.Tensor) -> torch.Tensor:
            full = torch.zeros(count, values_free.shape[1], dtype=torch.float64, device=device)
            full[free_index] = values_free
            return (laplacian @ full)[free_index]

        held = current.clone()
        held[free_index] = 0.0
        right = rhs[free_index] - (laplacian @ held)[free_index]
        solved, _ = solver.solve(right, current[free_index], apply, tolerance, RIGID_STEPS)
        current = current.clone()
        current[free_index] = torch.where(torch.isfinite(solved), solved, current[free_index])
    return current


def _vertex_chart(faces: torch.Tensor, chart: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Chart of each vertex of a mesh whose vertices each belong to one chart."""
    vertex_chart = torch.zeros(vertex_count, dtype=torch.int64, device=faces.device)
    vertex_chart[faces.reshape(-1)] = chart.repeat_interleave(3)
    return vertex_chart


def fold_share(uvs: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, count: int) -> torch.Tensor:
    """Share of each chart's texture area lying in triangles turned back to front.

    Args:
        uvs: ``(N, 2)`` chart coordinates, counter-clockwise for a front face.
        faces: ``(F, 3)`` vertex indices.
        chart: ``(F,)`` chart of each face.
        count: Number of charts.

    Returns:
        ``(C,)`` float64 shares from 0 to 1.
    """
    signed = _signed_areas(uvs.double(), faces)
    flipped = torch.zeros(count, dtype=torch.float64, device=uvs.device)
    flipped.index_add_(0, chart, torch.where(signed < 0, -signed, torch.zeros_like(signed)))
    return flipped / _flat_area(uvs, faces, chart, count).clamp_min(1e-300)


def flatten(vertices: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, quality: str = "balanced"):
    """Flatten every chart of a split mesh into the plane at true scale.

    Args:
        vertices: ``(N, 3)`` positions, each vertex in one chart.
        faces: ``(F, 3)`` vertex indices.
        chart: ``(F,)`` chart of each face, ``0`` to ``C - 1``.
        quality: One of `QUALITIES`.

    Returns:
        ``(uvs, folded, pins, steps)``: ``(N, 2)`` float64 coordinates, ``(C,)`` bool charts that
        fold or collapse, ``(2C,)`` pinned vertices, and conformal solver steps taken.
    """
    steps = QUALITY_STEPS.get(quality, QUALITY_STEPS["balanced"])
    device = vertices.device
    count = int(chart.max()) + 1 if chart.numel() else 0
    work = vertices.double()
    vertex_chart = _vertex_chart(faces, chart, work.shape[0])
    normals = topology.face_normals(work, faces)
    areas = topology.face_areas(work, faces)
    direction = _mean_normals(chart, count, normals, areas)
    blank = direction.norm(dim=1) == 0
    direction[blank] = torch.tensor([0.0, 0.0, 1.0], dtype=direction.dtype, device=device)
    first, second = _plane_basis(direction)
    projected = torch.stack([(work * first[vertex_chart]).sum(1), (work * second[vertex_chart]).sum(1)], dim=1)
    pins = _pins(projected, vertex_chart, count)
    solved, taken = _conformal(work, faces, projected, pins, vertex_chart, steps["tolerance"],
                               steps["iterations"])
    broken = torch.zeros(count, dtype=torch.int64, device=device)
    broken.index_add_(0, vertex_chart, (~torch.isfinite(solved).all(dim=1)).long())
    broken = broken > 0
    solved = torch.where(broken[vertex_chart][:, None], projected, solved)
    solved_folds = fold_share(solved, faces, chart, count)
    projected_folds = fold_share(projected, faces, chart, count)
    use_projection = (solved_folds > 0) & (projected_folds < solved_folds)
    use_projection |= _flat_area(solved, faces, chart, count) < COLLAPSED * _flat_area(projected, faces, chart, count)
    uvs = torch.where(use_projection[vertex_chart][:, None], projected, solved)
    surface = torch.zeros(count, dtype=torch.float64, device=device)
    surface.index_add_(0, chart, areas)
    collapsed = _flat_area(uvs, faces, chart, count) < COLLAPSED * surface
    uvs = _true_scale(work, faces, uvs, chart, count, vertex_chart)
    return uvs, (fold_share(uvs, faces, chart, count) > FOLD_SHARE) | collapsed, pins, taken


def relax(vertices: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, uvs: torch.Tensor,
          anchors: torch.Tensor, passes: int) -> torch.Tensor:
    """Relax flattened charts towards rigid copies of their triangles, at true scale.

    Args:
        vertices: ``(N, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        chart: ``(F,)`` chart of each face.
        uvs: ``(N, 2)`` flattened coordinates.
        anchors: One vertex per chart held in place.
        passes: Local and global step pairs.

    Returns:
        ``(N, 2)`` float64 coordinates.
    """
    count = int(chart.max()) + 1
    work = vertices.double()
    vertex_chart = _vertex_chart(faces, chart, work.shape[0])
    relaxed = _rigid(work, faces, uvs, anchors, vertex_chart, passes, RIGID_TOLERANCE)
    return _true_scale(work, faces, relaxed, chart, count, vertex_chart)


def _true_scale(vertices, faces, uvs, chart, count, vertex_chart) -> torch.Tensor:
    """Scale each chart so its flattened area equals its surface area."""
    surface = torch.zeros(count, dtype=torch.float64, device=uvs.device)
    surface.index_add_(0, chart, topology.face_areas(vertices, faces).double())
    flat = torch.zeros(count, dtype=torch.float64, device=uvs.device)
    flat.index_add_(0, chart, _signed_areas(uvs, faces).abs())
    factor = torch.where(flat > 0, (surface / flat.clamp_min(1e-300)).sqrt(), torch.ones_like(flat))
    return uvs * factor[vertex_chart][:, None]


def _singular_values(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor):
    """Singular values of each triangle's map from surface to texture space, and surface areas."""
    flat = _local_frames(vertices.double(), faces)
    mapped = uvs.double()[faces]
    source = torch.stack([flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0]], dim=2)
    target = torch.stack([mapped[:, 1] - mapped[:, 0], mapped[:, 2] - mapped[:, 0]], dim=2)
    det = source[:, 0, 0] * source[:, 1, 1] - source[:, 0, 1] * source[:, 1, 0]
    usable = det.abs() > 1e-300
    inverse = torch.stack([torch.stack([source[:, 1, 1], -source[:, 0, 1]], 1),
                           torch.stack([-source[:, 1, 0], source[:, 0, 0]], 1)], 1) / torch.where(
        usable, det, torch.ones_like(det))[:, None, None]
    jacobian = target @ inverse
    values = torch.linalg.svdvals(jacobian)
    return values, 0.5 * det.abs(), usable


def _area_error(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor, chart: torch.Tensor,
                count: int) -> torch.Tensor:
    """Area-weighted mean absolute log of each triangle's area ratio, per chart, at the chart's own scale."""
    surface = topology.face_areas(vertices.double(), faces)
    mapped = _signed_areas(uvs.double(), faces).abs()
    surface_total = torch.zeros(count, dtype=torch.float64, device=uvs.device).index_add_(0, chart, surface)
    mapped_total = torch.zeros(count, dtype=torch.float64, device=uvs.device).index_add_(0, chart, mapped)
    scale = (surface_total / mapped_total.clamp_min(1e-300))[chart]
    error = (mapped * scale / surface.clamp_min(1e-300)).clamp_min(1e-12).log().abs()
    total = torch.zeros(count, dtype=torch.float64, device=uvs.device).index_add_(0, chart, error * surface)
    return total / surface_total.clamp_min(1e-300)


def distortion(vertices: torch.Tensor, faces: torch.Tensor, uvs: torch.Tensor) -> dict[str, float]:
    """How far a texture mapping stretches the surface, its overall scale matched to the surface.

    Args:
        vertices: ``(N, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        uvs: ``(N, 2)`` texture coordinates, ``v`` down the image.

    Returns:
        ``stretch`` (root mean square, 1 is none), ``area`` (mean absolute log of the area
        ratio, 0 is none), ``angle`` (mean ratio of the larger to the smaller stretch, capped at
        `ANGLE_CAP` per triangle, 1 is conformal) and ``flipped`` (triangles mirrored in the image).
    """
    values, area, usable = _singular_values(vertices, faces, uvs)
    flat = _signed_areas(uvs.double(), faces)
    total = float(area.sum())
    mapped = float(flat.abs().sum())
    if not total > 0 or not mapped > 0:
        return {"stretch": 0.0, "area": 0.0, "angle": 0.0, "flipped": 0}
    scale = math.sqrt(total / mapped)
    values = values * scale
    weight = torch.where(usable & (area > 0), area, torch.zeros_like(area))
    weight = weight / weight.sum().clamp_min(1e-300)
    product = (values[:, 0] * values[:, 1]).clamp_min(1e-300)
    stretch = float(((values ** 2).sum(1) / 2 * weight).sum().sqrt())
    area_error = float((product.log().abs() * weight).sum())
    angle = float((values[:, 0] / values[:, 1].clamp_min(1e-300)).clamp_max(ANGLE_CAP).mul(weight).sum())
    flipped = int(((flat >= 0) & (area > 0)).sum())
    return {"stretch": stretch, "area": area_error, "angle": angle, "flipped": flipped}


# ---------------------------------------------------------------------------------------------
# Parts


def _attribute_key(part: Part) -> torch.Tensor:
    """One row per vertex of every attribute but texture coordinates and tangents."""
    columns = [part.vertices.double()]
    for name in ("normals", "colors", "joints", "weights"):
        value = getattr(part, name)
        if value is not None:
            columns.append(value.double().reshape(part.vertex_count, -1))
    for morph in part.morphs.values():
        columns.append(morph.positions.double())
        if morph.normals is not None:
            columns.append(morph.normals.double())
    return torch.cat(columns, dim=1)


def _hard_edges(part: Part, welded_faces: torch.Tensor, welded_count: int) -> torch.Tensor | None:
    """Welded edges where the part's own normals split."""
    if part.normals is None:
        return None
    key = torch.cat([part.vertices.double(), part.normals.double()], dim=1)
    _, by_normal = torch.unique(key, dim=0, return_inverse=True)
    split_faces = by_normal[part.faces.long()]
    remap = torch.zeros(int(by_normal.max()) + 1, dtype=torch.int64)
    remap[split_faces.reshape(-1)] = welded_faces.reshape(-1)
    return topology.seam_edges(split_faces, remap, topology.edge_topology(welded_faces, welded_count))


@dataclass
class _Flat:
    """One part's charts while they are flattened."""

    part: Part
    attribute_id: torch.Tensor
    position_faces: torch.Tensor
    positions: torch.Tensor
    chart: torch.Tensor
    links: tuple = ()
    param_faces: torch.Tensor = None
    param_source: torch.Tensor = None
    uvs: torch.Tensor = None
    steps: int = 0


def _param_mesh(flat: _Flat) -> None:
    """Split a part's welded vertices once per chart they touch."""
    count = int(flat.chart.max()) + 1
    corner_vertex = flat.position_faces.reshape(-1)
    corner_chart = flat.chart.repeat_interleave(3)
    key = corner_vertex * count + corner_chart
    keys, inverse = torch.unique(key, return_inverse=True)
    flat.param_faces = inverse.reshape(-1, 3)
    flat.param_source = keys // count


def _cut_out(flat: _Flat, marked: torch.Tensor, rings: int, chart_angle: float,
             polygon: torch.Tensor | None) -> torch.Tensor:
    """Give marked faces and the rings of faces around them charts of their own, cut strictly."""
    source, target = flat.links
    for _ in range(int(rings)):
        spread = marked[source] & (flat.chart[source] == flat.chart[target])
        marked = marked.clone()
        marked[target[spread]] = True
    if polygon is not None:
        whole = torch.zeros(int(polygon.max()) + 1, dtype=torch.int64, device=marked.device)
        whole.index_add_(0, polygon, marked.long())
        marked = whole[polygon] > 0
    pieces = segment(flat.positions, flat.position_faces[marked], chart_angle * CUT_ANGLE, None,
                     polygon[marked] if polygon is not None else None)
    chart = flat.chart.clone()
    chart[marked] = int(flat.chart.max()) + 1 + pieces
    return chart


def _overlaps(uvs: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, count: int, fresh: torch.Tensor,
              density: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-overlapping pixels of the fresh charts only, and the faces covering them."""
    overlaps = torch.zeros(count, dtype=torch.int64, device=chart.device)
    involved = torch.zeros(chart.shape[0], dtype=torch.bool, device=chart.device)
    chosen = fresh[chart]
    if not bool(chosen.any()):
        return overlaps, involved
    ids = torch.nonzero(fresh).squeeze(1)
    local = torch.full((count,), -1, dtype=torch.int64, device=chart.device)
    local[ids] = torch.arange(ids.shape[0], device=chart.device)
    counts, hit = pack.self_overlaps(uvs.float(), faces[chosen], local[chart[chosen]], int(ids.shape[0]), density)
    overlaps[ids] = counts
    involved[chosen] = hit
    return overlaps, involved


def _repair(flat: _Flat, settings: UnwrapSettings, device: torch.device) -> dict:
    """Flatten a part's charts, cutting out the faces that fold or overlap until none do."""
    polygon = flat.part.polygons.triangle_polygon.to(device) if flat.part.polygons is not None else None
    passes = QUALITY_STEPS[settings.quality]["rigid"]
    folded_total = overlap_total = 0
    fresh = torch.ones(int(flat.chart.max()) + 1, dtype=torch.bool, device=device)
    overlaps = torch.zeros_like(fresh, dtype=torch.int64)
    for round_index in range(REPAIR_ROUNDS + 1):
        _param_mesh(flat)
        vertices = flat.positions[flat.param_source].double()
        faces = flat.param_faces
        uvs, folded, pins, steps = flatten(vertices, faces, flat.chart, settings.quality)
        flat.steps += steps
        count = int(flat.chart.max()) + 1
        area = float(topology.face_areas(vertices, faces).sum())
        side = min(int(settings.resolution), CHECK_SIDE)
        density = math.sqrt(pack.FIRST_FILL * side * side / max(area, 1e-30))
        found, involved = _overlaps(uvs, faces, flat.chart, count, fresh, density)
        overlaps = torch.where(fresh, found, overlaps)
        flat.uvs = uvs
        bad = folded | (overlaps > 0)
        folded_total, overlap_total = int(folded.sum()), int((overlaps > 0).sum())
        if not bool(bad.any()) or round_index == REPAIR_ROUNDS:
            break
        sizes = torch.bincount(flat.chart, minlength=count)
        bad = bad & (sizes > 1)
        if not bool(bad.any()):
            break
        changed = bad[flat.chart]
        kept_overlaps = overlaps[flat.chart]
        flipped = _signed_areas(uvs, faces) <= 0
        marked = (involved | flipped) & changed
        if round_index < LOCAL_ROUNDS:
            lost = bad & (torch.zeros(count, dtype=torch.int64, device=device).index_add_(
                0, flat.chart, marked.long()) == 0)
            marked |= lost[flat.chart]
            flat.chart = _cut_out(flat, marked, round_index + 1, settings.chart_angle, polygon)
        elif round_index < REPAIR_ROUNDS - 1:
            flat.chart = split_charts(flat.positions, flat.position_faces, flat.chart, bad, polygon)
        else:
            # The faces still at fault become charts of their own, which cannot overlap themselves.
            flat.chart = flat.chart.clone()
            flat.chart[marked] = count + torch.arange(int(marked.sum()), device=device)
        flat.chart = _relabel(flat.chart, *flat.links)
        count = int(flat.chart.max()) + 1
        fresh = torch.zeros(count, dtype=torch.int64, device=device).index_add_(0, flat.chart, changed.long()) > 0
        overlaps = torch.zeros(count, dtype=torch.int64, device=device)
        overlaps.scatter_reduce_(0, flat.chart, kept_overlaps, reduce="amax")
        overlaps = torch.where(fresh, torch.zeros_like(overlaps), overlaps)
    if passes:
        # Each chart keeps its relaxed shape only where that adds no overlap or fold and lowers stretch.
        everyone = torch.ones(count, dtype=torch.bool, device=device)
        relaxed = relax(vertices, faces, flat.chart, uvs, pins[:count], passes)
        relaxed_overlaps, _ = _overlaps(relaxed, faces, flat.chart, count, everyone, density)
        better = ((relaxed_overlaps <= overlaps)
                  & (fold_share(relaxed, faces, flat.chart, count) <= fold_share(uvs, faces, flat.chart, count))
                  & (_area_error(vertices, faces, relaxed, flat.chart, count)
                     < _area_error(vertices, faces, uvs, flat.chart, count)))
        vertex_chart = _vertex_chart(faces, flat.chart, vertices.shape[0])
        flat.uvs = torch.where(better[vertex_chart][:, None], relaxed, uvs)
    return {"folded_charts": folded_total, "overlapping_charts": overlap_total}


def unwrap_parts(parts: list[Part], settings: UnwrapSettings) -> Unwrapped:
    """Give parts that share one texture new texture coordinates in one atlas.

    Args:
        parts: Parts with triangles, to share one atlas.
        settings: The `UnwrapSettings`.

    Returns:
        The `Unwrapped` parts, vertices split along the new seams, tangents dropped.

    Raises:
        ValueError: A part has no triangles, too many triangles or positions that are not
            finite numbers.
    """
    if settings.quality not in QUALITIES:
        raise ValueError(f"quality must be one of {', '.join(QUALITIES)}, not {settings.quality!r}")
    device = mesh_device.compute_device()

    def work(run_device: torch.device) -> Unwrapped:
        return _unwrap(parts, settings, run_device)

    return mesh_device.with_cpu_fallback(work, device, label="UV unwrap")


def _unwrap(parts: list[Part], settings: UnwrapSettings, device: torch.device) -> Unwrapped:
    """`unwrap_parts` on one device."""
    flats: list[_Flat] = []
    totals = {"folded_charts": 0, "overlapping_charts": 0}
    for part in parts:
        if part.triangle_count == 0:
            raise ValueError(f"part '{part.name}' has no triangles to unwrap")
        if part.triangle_count > MAX_TRIANGLES:
            raise ValueError(f"part '{part.name}' has {part.triangle_count} triangles, more than the "
                             f"{MAX_TRIANGLES} an unwrap takes; decimate it first")
        if not bool(torch.isfinite(part.vertices).all()):
            raise ValueError(f"part '{part.name}' has vertex positions that are not finite numbers; "
                             "clean the mesh first")
        faces = part.faces.long().to(device)
        _, attribute_id = torch.unique(_attribute_key(part).to(device), dim=0, return_inverse=True)
        positions, position_faces, _ = topology.weld(part.vertices.to(device).double(), faces, tolerance=0.0)
        hard = _hard_edges(part, position_faces.cpu(), int(positions.shape[0]))
        polygon = part.polygons.triangle_polygon.to(device) if part.polygons is not None else None
        chart = segment(positions, position_faces, settings.chart_angle,
                        hard.to(device) if hard is not None else None, polygon)
        source, target = _adjacency(positions, position_faces)
        if polygon is not None:
            inside = polygon[source] == polygon[target]
            source, target = torch.cat([source, source[inside]]), torch.cat([target, target[inside]])
        flat = _Flat(part, attribute_id, position_faces, positions, chart, (source, target))
        found = _repair(flat, settings, device)
        for key in totals:
            totals[key] += found[key]
        flats.append(flat)

    offsets, chart_offsets = [], []
    vertex_total = chart_total = 0
    for flat in flats:
        offsets.append(vertex_total)
        chart_offsets.append(chart_total)
        vertex_total += int(flat.param_source.shape[0])
        chart_total += int(flat.chart.max()) + 1
    all_uvs = torch.cat([flat.uvs for flat in flats]).float()
    # Charts are mirrored in v so front faces run the way glTF's downward v expects.
    all_uvs[:, 1] = -all_uvs[:, 1]
    all_faces = torch.cat([flat.param_faces + offset for flat, offset in zip(flats, offsets)])
    all_charts = torch.cat([flat.chart + offset for flat, offset in zip(flats, chart_offsets)])
    packing = pack.pack_charts(all_uvs, all_faces, all_charts, chart_total, int(settings.resolution),
                               int(settings.padding), grid=QUALITY_STEPS[settings.quality]["grid"])

    new_parts, sources = [], []
    stats_faces, stats_vertices, stats_uvs = [], [], []
    for flat, offset in zip(flats, offsets):
        atlas = packing.uvs[offset:offset + int(flat.param_source.shape[0])]
        count = int(flat.chart.max()) + 1
        corner_key = flat.attribute_id[flat.part.faces.long().to(device).reshape(-1)] * count + \
            flat.chart.repeat_interleave(3)
        keys, inverse = torch.unique(corner_key, return_inverse=True)
        faces_out = inverse.reshape(-1, 3)
        source = torch.zeros(keys.shape[0], dtype=torch.int64, device=device)
        source[inverse] = flat.part.faces.long().to(device).reshape(-1)
        uv_out = torch.zeros(keys.shape[0], 2, dtype=torch.float32, device=device)
        uv_out[inverse] = atlas[flat.param_faces.reshape(-1)]
        new_part = split_part(flat.part, faces_out.cpu(), source.cpu(), uvs=uv_out.cpu(), tangents=None)
        if flat.part.uvs is not None and (flat.part.material.textures() or not bake.has_previous(flat.part)):
            bake.keep_previous(new_part, flat.part)
        new_parts.append(new_part)
        sources.append(source.cpu())
        stats_faces.append(flat.param_faces + sum(int(v.shape[0]) for v in stats_vertices))
        stats_vertices.append(flat.positions[flat.param_source].double())
        stats_uvs.append(atlas.double())
    measured = distortion(torch.cat(stats_vertices), torch.cat(stats_faces), torch.cat(stats_uvs))
    efficiency, overlap = _coverage(packing.uvs, all_faces, all_charts, int(settings.resolution))
    stats = {
        "charts": chart_total,
        "efficiency": efficiency,
        "overlapping_texels": overlap,
        "stretch": measured["stretch"],
        "area_distortion": measured["area"],
        "angle_distortion": measured["angle"],
        "flipped": measured["flipped"],
        "texels_per_unit": packing.scale,
        "solver_steps": sum(flat.steps for flat in flats),
        **totals,
    }
    logger.info("unwrapped %d part(s) into %d charts, %.0f%% of the atlas used", len(parts), chart_total,
                efficiency * 100)
    face_charts = [(flat.chart + offset).cpu() for flat, offset in zip(flats, chart_offsets)]
    return Unwrapped(new_parts, sources, chart_total, face_charts, stats)


def _coverage(uvs: torch.Tensor, faces: torch.Tensor, chart: torch.Tensor, side: int) -> tuple[float, int]:
    """Share of atlas texels covered, and texels claimed by two different charts."""
    corners = uvs[faces] * side
    low = torch.full((side * side,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=uvs.device)
    high = torch.full((side * side,), -1, dtype=torch.int32, device=uvs.device)
    for piece in pack.raster.coverage_chunks(corners, side, side):
        inside = piece.barycentric.amin(dim=1) > pack.INSIDE
        pixel = piece.pixel[inside]
        label = chart[piece.face[inside]].to(torch.int32)
        low.scatter_reduce_(0, pixel, label, reduce="amin")
        high.scatter_reduce_(0, pixel, label, reduce="amax")
    covered = high >= 0
    return float(covered.float().mean()), int((covered & (low != high)).sum())
