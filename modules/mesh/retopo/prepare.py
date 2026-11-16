"""Preparing a surface for retopology: welding, cutting at a plane, the scale field, splitting
long edges and finding feature lines.

Vertices are ``(V, 3)`` float, faces ``(F, 3)`` int64, scales are lattice spacings in model units.
"""

from __future__ import annotations

__all__ = [
    "Features", "clean_faces", "close_small_holes", "compact", "cut_half", "features", "orient", "scale_field", "subdivide",
    "surface", "thickness",
]

import math
from dataclasses import dataclass

import torch

from .. import topology
from ..bvh import TriangleBVH

#: Turn in degrees past which a vertex on a feature line is treated as a corner.
CORNER_TURN_DEG = 50.0

#: Feature chains shorter than this many lattice spacings are ignored.
MIN_FEATURE_SPACINGS = 1.5

#: Edges longer than this many lattice spacings are split.
SPLIT_SPACINGS = 0.75

#: Most edge splitting passes.
MAX_SPLIT_PASSES = 24

#: Lattice spacing allowed on a thin part, as a share of its thickness.
THIN_SPACING = 0.8

#: Smallest spacing full adaptivity may reach, as a share of the even spacing.
ADAPTIVE_FLOOR = 0.25

#: Closed borders shorter than this many even lattice spacings are covered before retopology.
SMALL_HOLE_SPACINGS = 6.0

#: Most vertices a thickness ray is cast from.
THICKNESS_SAMPLES = 200_000

#: Rounds rescaling the spacing until the face count meets the target.
SPACING_ROUNDS = 6

#: Largest change of spacing per unit of distance.
MAX_SPACING_SLOPE = 0.2

#: Most passes spreading the spacing limit from cell to cell.
MAX_SLOPE_PASSES = 32


@dataclass
class Features:
    """Feature lines of a prepared surface.

    Attributes:
        edges: ``(K, 2)`` int64 vertex pairs of the feature edges.
        line: ``(V,)`` bool, vertices inside a feature line.
        corner: ``(V,)`` bool, ends, junctions and sharp turns of feature lines.
        direction: ``(V, 3)`` unit tangent of the line at each line vertex, zero elsewhere.
        cut: ``(K,)`` bool, feature edges lying on the symmetry cut.
    """

    edges: torch.Tensor
    line: torch.Tensor
    corner: torch.Tensor
    direction: torch.Tensor
    cut: torch.Tensor

    @property
    def count(self) -> int:
        """Number of feature edges."""
        return int(self.edges.shape[0])


def surface(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A triangle soup welded on exact positions, without faces that repeat a vertex.

    Args:
        vertices: ``(V, 3)`` positions, split where attributes differ.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(vertices, faces, remap)``: welded positions, every face on them, and ``(V,)`` int64
        welded index of each input vertex. Faces keep their order and count.
    """
    welded, welded_faces, remap = topology.weld(vertices.float(), faces.long(), tolerance=0.0)
    return welded, welded_faces, remap


def clean_faces(faces: torch.Tensor) -> torch.Tensor:
    """Faces without a repeated vertex.

    Args:
        faces: ``(F, 3)`` vertex indices.

    Returns:
        The faces whose three corners differ.
    """
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return faces[keep]


def orient(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Faces wound alike across every shared edge: closed pieces outward, open pieces by majority.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(F, 3)`` int64 faces.
    """
    count = int(vertices.shape[0])
    if faces.shape[0] == 0:
        return faces
    oriented = topology.orient_faces(faces, count, vertices, outward="closed")
    turned = (oriented != faces).any(1)
    if not bool(turned.any()):
        return oriented
    topo = topology.edge_topology(faces, count)
    piece = topology.face_components(faces, count)
    pieces = int(piece.max()) + 1
    open_piece = torch.zeros(pieces, dtype=torch.bool, device=faces.device)
    open_piece[piece[(topo.boundary | topo.non_manifold)[topo.face_edges].any(1)]] = True
    share = torch.zeros(pieces, dtype=torch.float32, device=faces.device).index_add_(0, piece, turned.float())
    share /= torch.bincount(piece, minlength=pieces).clamp_min(1).float()
    undo = (open_piece & (share > 0.5))[piece]
    return torch.where(undo[:, None], oriented[:, [0, 2, 1]], oriented)


def close_small_holes(vertices: torch.Tensor, faces: torch.Tensor,
                      max_length: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cover closed borders shorter than ``max_length`` with a fan around their centre.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        max_length: Longest border loop covered, in model units.

    Returns:
        ``(vertices, faces, closed)``: one new vertex per covered loop, the fan triangles
        appended after the given faces, and how many loops were covered.
    """
    count = int(vertices.shape[0])
    chains, closed = topology.boundary_chains(faces, count)
    if not chains or max_length <= 0:
        return vertices, faces, 0
    device = faces.device
    sizes = torch.tensor([int(chain.numel()) for chain in chains], dtype=torch.int64, device=device)
    corners = torch.cat(chains)
    loop = torch.repeat_interleave(torch.arange(sizes.shape[0], device=device), sizes)
    start = torch.cumsum(sizes, 0) - sizes
    following = torch.arange(corners.shape[0], device=device) + 1
    following = torch.where(following == (start + sizes)[loop], start[loop], following)
    step = (vertices[corners[following]] - vertices[corners]).norm(dim=1).double()
    length = torch.zeros(sizes.shape[0], dtype=torch.float64, device=device).index_add_(0, loop, step)
    small = closed.to(device) & (length <= float(max_length)) & (sizes >= 2)
    if not bool(small.any()):
        return vertices, faces, 0
    chosen = small[loop]
    loop_index = torch.cumsum(small.long(), 0) - 1
    centre_of = count + loop_index[loop[chosen]]
    centres = torch.zeros(int(small.sum()), 3, dtype=vertices.dtype, device=device)
    centres.index_add_(0, loop_index[loop[chosen]], vertices[corners[chosen]])
    centres /= sizes[small].to(vertices.dtype)[:, None]
    fan = torch.stack([corners[following][chosen], corners[chosen], centre_of], dim=1)
    return torch.cat([vertices, centres]), torch.cat([faces, fan]), int(small.sum())


def compact(vertices: torch.Tensor, faces: torch.Tensor, *extra: torch.Tensor):
    """Drop vertices no face uses.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        *extra: Per-vertex tensors kept in step with the vertices.

    Returns:
        ``(vertices, faces, *extra)`` on the used vertices only.
    """
    used = torch.zeros(vertices.shape[0], dtype=torch.bool, device=vertices.device)
    used[faces.reshape(-1)] = True
    index = torch.full((vertices.shape[0],), -1, dtype=torch.int64, device=vertices.device)
    index[used] = torch.arange(int(used.sum()), device=vertices.device)
    return (vertices[used], index[faces], *[value[used] for value in extra])


def cut_half(vertices: torch.Tensor, faces: torch.Tensor, axis: int = 0,
             tolerance: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The part of a surface on the positive side of a coordinate plane through the origin.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        axis: 0, 1 or 2 for the plane ``x = 0``, ``y = 0`` or ``z = 0``.
        tolerance: Distance from the plane within which a vertex lies on it; None uses a
            millionth of the box diagonal.

    Returns:
        ``(vertices, faces)``. Vertices on the cut have that coordinate exactly 0.
    """
    device = vertices.device
    count = int(vertices.shape[0])
    if tolerance is None:
        tolerance = 1e-6 * float((vertices.amax(0) - vertices.amin(0)).norm()) if count else 0.0
    coord = vertices[:, axis]
    side = torch.where(coord > tolerance, 1, torch.where(coord < -tolerance, -1, 0))
    vertices = vertices.clone()
    vertices[side == 0, axis] = 0.0
    corner_side = side[faces]
    keep_whole = (corner_side >= 0).all(1) & (corner_side > 0).any(1)
    mixed = (corner_side > 0).any(1) & (corner_side < 0).any(1)
    mixed_faces = faces[mixed]
    mixed_side = corner_side[mixed]
    start, end = mixed_faces, mixed_faces[:, [1, 2, 0]]
    crossing = mixed_side * mixed_side[:, [1, 2, 0]] < 0
    low = torch.minimum(start, end)[crossing]
    high = torch.maximum(start, end)[crossing]
    keys, inverse = torch.unique(low * max(count, 1) + high, return_inverse=True)
    first, second = keys // max(count, 1), keys % max(count, 1)
    a, b = vertices[first], vertices[second]
    t = (a[:, axis] / (a[:, axis] - b[:, axis])).clamp(0.0, 1.0)
    points = a + (b - a) * t[:, None]
    points[:, axis] = 0.0
    cross_index = torch.full(crossing.shape, -1, dtype=torch.int64, device=device)
    cross_index[crossing] = count + inverse
    slots = torch.stack([start[:, 0], cross_index[:, 0], start[:, 1], cross_index[:, 1],
                         start[:, 2], cross_index[:, 2]], dim=1)
    valid = torch.stack([mixed_side[:, 0] >= 0, crossing[:, 0], mixed_side[:, 1] >= 0, crossing[:, 1],
                         mixed_side[:, 2] >= 0, crossing[:, 2]], dim=1)
    rank = (~valid).long() * 8 + torch.arange(6, device=device)
    order = torch.argsort(rank, dim=1)[:, :4]
    polygon = slots.gather(1, order)
    sizes = valid.sum(1)
    pieces = [faces[keep_whole], polygon[:, [0, 1, 2]], polygon[sizes == 4][:, [0, 2, 3]]]
    every = torch.cat([vertices, points], dim=0)
    kept = clean_faces(torch.cat(pieces, dim=0))
    if kept.shape[0] == 0:
        return every[:0], kept
    out_vertices, out_faces = compact(every, kept)
    return out_vertices, out_faces


def _voxel_average(points: torch.Tensor, values: torch.Tensor, weights: torch.Tensor,
                   cell: float) -> torch.Tensor:
    """Weighted mean of a value over cubic cells, read back at each point from two offset grids."""
    low = points.amin(0)
    result = torch.zeros_like(values)
    for shift in (0.0, 0.5):
        cells = torch.floor((points - low) / cell + shift).long()
        _, group = torch.unique(cells, dim=0, return_inverse=True)
        size = int(group.max()) + 1
        total = torch.zeros(size, dtype=values.dtype, device=values.device).index_add_(0, group, values * weights)
        mass = torch.zeros(size, dtype=values.dtype, device=values.device).index_add_(0, group, weights)
        result += total[group] / mass[group].clamp_min(1e-30)
    return result / 2.0


def thickness(vertices: torch.Tensor, faces: torch.Tensor, normals: torch.Tensor, cell: float,
              generator: torch.Generator, samples: int = THICKNESS_SAMPLES) -> torch.Tensor:
    """Distance through the surface behind each vertex, averaged over cubic cells.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        normals: ``(V, 3)`` outward unit normals.
        cell: Side of the averaging cells.
        generator: CPU random generator choosing the sampled vertices.
        samples: Most vertices a ray is cast from.

    Returns:
        ``(V,)`` thickness, infinite where no ray found a far side.
    """
    count = int(vertices.shape[0])
    device = vertices.device
    pick = torch.arange(count, device=device)
    if count > samples:
        pick = torch.randperm(count, generator=generator)[:samples].to(device)
    tree = TriangleBVH(vertices, faces, device=device)
    reach, _, _ = tree.ray_cast(vertices[pick], -normals[pick], t_min=1e-3 * cell)
    found = torch.isfinite(reach) & (reach > 0)
    pick, reach = pick[found], reach[found]
    result = torch.full((count,), math.inf, dtype=torch.float64, device=device)
    if pick.numel() == 0:
        return result
    low = vertices.amin(0).double()
    cells = torch.floor((vertices.double() - low) / cell).long()
    _, group = torch.unique(cells, dim=0, return_inverse=True)
    size = int(group.max()) + 1
    total = torch.zeros(size, dtype=torch.float64, device=device).index_add_(0, group[pick], reach.double().log())
    hits = torch.zeros(size, dtype=torch.float64, device=device).index_add_(0, group[pick], torch.ones_like(reach, dtype=torch.float64))
    measured = hits[group] > 0
    result[measured] = (total[group][measured] / hits[group][measured]).exp()
    return result


def scale_field(vertices: torch.Tensor, faces: torch.Tensor, target_faces: float, generator: torch.Generator,
                adaptivity: float = 0.0, min_component_faces: float = 6.0) -> torch.Tensor:
    """Lattice spacing at each vertex for a target face count, finer on thin parts.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        target_faces: Quads wanted over the whole surface.
        generator: CPU random generator for the thickness samples.
        adaptivity: 0 = even spacing; 1 = spacing down to a quarter where the surface curves
            tightly or the model is thin.
        min_component_faces: Fewest quads a separate piece of the surface is given.

    Returns:
        ``(V,)`` spacings.

    Raises:
        ValueError: The surface has no area.
    """
    areas = topology.vertex_areas(vertices.double(), faces, kind="barycentric")
    total = float(areas.sum())
    if not total > 0 or not math.isfinite(total):
        raise ValueError("the mesh has no surface area to retopologise; check that it holds real triangles")
    target = max(float(target_faces), 1.0)
    even = math.sqrt(total / target)
    normals = topology.vertex_normals(vertices, faces)
    gain = torch.ones_like(areas)
    if adaptivity > 0 and faces.shape[0]:
        topo = topology.edge_topology(faces, vertices.shape[0])
        first, second = topo.edges[:, 0], topo.edges[:, 1]
        length = topology.edge_lengths(vertices.double(), topo.edges).clamp_min(1e-12)
        unit = normals.double()
        bend = torch.acos((unit[first] * unit[second]).sum(1).clamp(-1.0, 1.0)) / length
        bend_sum = torch.zeros_like(areas).index_add_(0, first, bend).index_add_(0, second, bend)
        incident = torch.bincount(topo.edges.reshape(-1), minlength=vertices.shape[0]).double().clamp_min(1)
        curvature = _voxel_average(vertices.double(), bend_sum / incident, areas, even)
        gain = 1.0 / (1.0 + (1.0 / ADAPTIVE_FLOOR - 1.0) * float(adaptivity) * (curvature * even).clamp(0.0, 1.0))
    floor = 1.0 - (1.0 - ADAPTIVE_FLOOR) * min(max(float(adaptivity), 0.0), 1.0)
    thin = torch.full_like(areas, math.inf)
    if adaptivity > 0:
        thin = THIN_SPACING * thickness(vertices, faces, normals, even / 2.0, generator)
    labels = topology.connected_components(faces, vertices.shape[0])
    piece_area = torch.zeros(int(labels.max()) + 1 if labels.numel() else 0, dtype=areas.dtype,
                             device=areas.device).index_add_(0, labels, areas)
    piece_cap = (piece_area / max(float(min_component_faces), 1.0)).sqrt()[labels].clamp_min(1e-12)
    cells = torch.floor((vertices.double() - vertices.double().amin(0)) / even).long()
    cell_keys, group = torch.unique(cells, dim=0, return_inverse=True)
    factor = math.sqrt(float((areas / gain ** 2).sum()) / target)
    spacing = gain * factor
    for _ in range(SPACING_ROUNDS):
        free = gain * factor
        wanted = torch.minimum(free, torch.maximum(thin, floor * free))
        per_cell = torch.full((cell_keys.shape[0],), math.inf, dtype=wanted.dtype, device=wanted.device)
        per_cell.scatter_reduce_(0, group, wanted, reduce="amin")
        per_cell = _limit_gradient(cell_keys, per_cell, MAX_SPACING_SLOPE * even)
        spacing = torch.minimum(torch.clamp(per_cell[group], min=floor * free, max=free), piece_cap)
        made = float((areas / spacing ** 2).sum())
        factor *= math.sqrt(max(made, 1e-30) / target)
    return spacing.float()


def _limit_gradient(cells: torch.Tensor, values: torch.Tensor, step: float) -> torch.Tensor:
    """Lower cell values until neighbouring cells differ by at most ``step``.

    Args:
        cells: ``(U, 3)`` int64 cell coordinates, sorted and unique.
        values: ``(U,)`` values per cell.
        step: Largest rise from one cell to the next.

    Returns:
        ``(U,)`` limited values.
    """
    span = cells.amax(0) + 3
    shifted = cells + 1
    keys = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]
    keys, order = torch.sort(keys)
    values = values[order].clone()
    offsets = [(span[1] * span[2]), span[2], 1]
    for _ in range(MAX_SLOPE_PASSES):
        lowest = values.clone()
        for offset in offsets:
            for sign in (-1, 1):
                wanted = keys + sign * offset
                at = torch.searchsorted(keys, wanted).clamp_max(keys.shape[0] - 1)
                found = keys[at] == wanted
                lowest[found] = torch.minimum(lowest[found], values[at[found]] + step)
        if torch.equal(lowest, values):
            break
        values = lowest
    result = torch.empty_like(values)
    result[order] = values
    return result


def subdivide(vertices: torch.Tensor, faces: torch.Tensor, scales: torch.Tensor,
              max_vertices: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edges longer than `SPLIT_SPACINGS` local lattice spacings until none remains.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        scales: ``(V,)`` lattice spacings.
        max_vertices: Most vertices the result may hold.

    Returns:
        ``(vertices, faces, scales)``.

    Raises:
        ValueError: Splitting would pass ``max_vertices``.
    """
    device = vertices.device
    for _ in range(MAX_SPLIT_PASSES):
        count = int(vertices.shape[0])
        topo = topology.edge_topology(faces, count)
        first, second = topo.edges[:, 0], topo.edges[:, 1]
        length = topology.edge_lengths(vertices, topo.edges)
        long = length > SPLIT_SPACINGS * torch.minimum(scales[first], scales[second])
        added = int(long.sum())
        if added == 0:
            break
        if count + added > max_vertices:
            raise ValueError(
                f"the mesh is too coarse for this many faces: it would need more than {max_vertices} "
                "working vertices; lower target_faces or retopologise fewer parts at once"
            )
        mid = torch.full((topo.edge_count,), -1, dtype=torch.int64, device=device)
        mid[long] = count + torch.arange(added, device=device)
        vertices = torch.cat([vertices, (vertices[first[long]] + vertices[second[long]]) / 2])
        scales = torch.cat([scales, torch.minimum(scales[first[long]], scales[second[long]])])
        side_mid = mid[topo.face_edges]
        marked = (side_mid >= 0).long()
        split_count = marked.sum(1)
        pieces = [faces[split_count == 0]]
        if bool((split_count == 3).any()):
            f, m = faces[split_count == 3], side_mid[split_count == 3]
            pieces += [torch.stack([f[:, 0], m[:, 0], m[:, 2]], 1), torch.stack([m[:, 0], f[:, 1], m[:, 1]], 1),
                       torch.stack([m[:, 2], m[:, 1], f[:, 2]], 1), torch.stack([m[:, 0], m[:, 1], m[:, 2]], 1)]
        roll = torch.arange(3, device=device)
        if bool((split_count == 1).any()):
            rows = split_count == 1
            start = marked[rows].argmax(1)
            turn = (start[:, None] + roll) % 3
            f, m = faces[rows].gather(1, turn), side_mid[rows].gather(1, turn)
            pieces += [torch.stack([f[:, 0], m[:, 0], f[:, 2]], 1), torch.stack([m[:, 0], f[:, 1], f[:, 2]], 1)]
        if bool((split_count == 2).any()):
            rows = split_count == 2
            start = ((1 - marked[rows]).argmax(1) + 1) % 3
            turn = (start[:, None] + roll) % 3
            f, m = faces[rows].gather(1, turn), side_mid[rows].gather(1, turn)
            pieces.append(torch.stack([f[:, 1], m[:, 1], m[:, 0]], 1))
            short = (vertices[f[:, 0]] - vertices[m[:, 1]]).norm(dim=1) <= (vertices[m[:, 0]] - vertices[f[:, 2]]).norm(dim=1)
            pieces.append(torch.where(short[:, None], torch.stack([f[:, 0], m[:, 0], m[:, 1]], 1),
                                      torch.stack([f[:, 0], m[:, 0], f[:, 2]], 1)))
            pieces.append(torch.where(short[:, None], torch.stack([f[:, 0], m[:, 1], f[:, 2]], 1),
                                      torch.stack([m[:, 0], m[:, 1], f[:, 2]], 1)))
        faces = torch.cat(pieces, dim=0)
    return vertices, faces, scales


def features(vertices: torch.Tensor, faces: torch.Tensor, scales: torch.Tensor, crease_angle: float,
             keep_borders: bool, cut_axis: int | None = None) -> Features:
    """Feature lines: creases, borders and the symmetry cut, without short stray chains.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        scales: ``(V,)`` lattice spacings.
        crease_angle: Dihedral angle in degrees from which an edge is a crease; 0 = none.
        keep_borders: Whether open borders are feature lines.
        cut_axis: Axis whose zero plane the surface was cut at, or None.

    Returns:
        The `Features`.
    """
    device = vertices.device
    count = int(vertices.shape[0])
    topo = topology.edge_topology(faces, count)
    first, second = topo.edges[:, 0], topo.edges[:, 1]
    marked = torch.zeros(topo.edge_count, dtype=torch.bool, device=device)
    if crease_angle > 0:
        angle = topology.dihedral_angles(vertices, faces, topo)
        marked |= (angle >= math.radians(float(crease_angle))) & (topo.edge_face_count == 2)
    if keep_borders:
        marked |= topo.boundary
    cut = torch.zeros_like(marked)
    if cut_axis is not None:
        cut = topo.boundary & (vertices[first, cut_axis] == 0) & (vertices[second, cut_axis] == 0)
        marked |= cut
    ids = torch.nonzero(marked).squeeze(1)
    if ids.numel():
        a, b = first[ids], second[ids]
        group = topology.union_find(count, a, b)[a]
        length = topology.edge_lengths(vertices, topo.edges[ids])
        spacing = torch.minimum(scales[a], scales[b])
        size = count
        chain_length = torch.zeros(size, dtype=length.dtype, device=device).index_add_(0, group, length)
        chain_spacing = torch.zeros(size, dtype=length.dtype, device=device).index_add_(0, group, spacing)
        members = torch.bincount(group, minlength=size).clamp_min(1)
        long_enough = chain_length[group] >= MIN_FEATURE_SPACINGS * chain_spacing[group] / members[group]
        ids = ids[long_enough | cut[ids]]
    edges = topo.edges[ids]
    is_cut = cut[ids]
    degree = torch.bincount(edges.reshape(-1), minlength=count)
    direction = torch.zeros_like(vertices)
    corner = (degree == 1) | (degree >= 3)
    line = degree == 2
    if ids.numel():
        a, b = edges[:, 0], edges[:, 1]
        unit = torch.nn.functional.normalize(vertices[b] - vertices[a], dim=1)
        owner = torch.cat([a, b])
        outward = torch.cat([unit, -unit])
        half = torch.arange(owner.shape[0], device=device)
        first_half = torch.full((count,), owner.shape[0], dtype=torch.int64, device=device)
        first_half.scatter_reduce_(0, owner, half, reduce="amin")
        lead = outward[first_half.clamp_max(owner.shape[0] - 1)]
        sign = torch.where((outward * lead[owner]).sum(1) >= 0, 1.0, -1.0)
        direction.index_add_(0, owner, outward * sign[:, None])
        total = torch.zeros_like(vertices).index_add_(0, owner, outward)
        turn_cos = 1.0 - (total * total).sum(1) / 2.0
        sharp = line & (turn_cos < math.cos(math.radians(CORNER_TURN_DEG)))
        corner |= sharp
        line &= ~sharp
        direction = torch.nn.functional.normalize(direction, dim=1) * line[:, None]
    return Features(edges=edges, line=line, corner=corner, direction=direction, cut=is_cut)
