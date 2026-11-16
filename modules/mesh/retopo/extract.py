"""Turning solved orientation and position fields into a polygon mesh.

Graph vertices sharing a lattice point merge into one output vertex; neighbours one lattice
step apart give an edge. Faces are the cycles of the edges sorted around each vertex.
"""

from __future__ import annotations

__all__ = ["Extraction", "collapse", "extract", "skip_edges", "trace_faces"]

import math
from dataclasses import dataclass, field

import torch

from .fields import lattice_round, middle_point, project_tangent
from .hierarchy import Level, match_pairs
from .polymesh import PolyMesh, order_cycles

#: Lattice points standing for less than this share of the mean number of graph vertices are dropped.
MIN_WEIGHT_SHARE = 0.1

#: Edges longer than this many lattice spacings are dropped.
MAX_EDGE_SPACINGS = 2.5

#: Cycles longer than this are left as holes.
MAX_FACE_CORNERS = 64

#: Most passes removing vertices left with one edge.
MAX_PRUNE_PASSES = 256

#: Most rounds merging graph vertices into lattice points.
MAX_COLLAPSE_ROUNDS = 64

#: Most passes removing edges that a single cycle runs along both ways.
MAX_SPUR_PASSES = 6

#: Collapse edges whose two lattice points lie further apart than this many spacings are ignored.
MAX_COLLAPSE_ERROR = 0.5

#: Of two edges leaving one vertex closer than this many degrees apart, the longer is dropped.
MIN_EDGE_ANGLE_DEG = 30.0


@dataclass
class Extraction:
    """A polygon mesh read off the fields.

    Attributes:
        mesh: The polygons, marked ``"feature"``, ``"corner"`` and ``"scale"``.
        figures: Counts describing the extraction.
    """

    mesh: PolyMesh
    figures: dict[str, int] = field(default_factory=dict)


def _scatter_max_pick(group: torch.Tensor, value: torch.Tensor, count: int) -> torch.Tensor:
    """Member index with the largest value in each group, lowest index on a tie."""
    device = group.device
    best = torch.full((count,), -math.inf, dtype=value.dtype, device=device)
    best.scatter_reduce_(0, group, value, reduce="amax")
    index = torch.arange(group.shape[0], device=device)
    top = value >= best[group]
    chosen = torch.full((count,), group.shape[0], dtype=torch.int64, device=device)
    chosen.scatter_reduce_(0, group[top], index[top], reduce="amin")
    return chosen.clamp_max(max(group.shape[0] - 1, 0))


def _pair_keys(a: torch.Tensor, b: torch.Tensor, count: int) -> torch.Tensor:
    """Order-free key of each vertex pair."""
    return torch.minimum(a, b) * count + torch.maximum(a, b)


def collapse(count: int, first: torch.Tensor, second: torch.Tensor, error: torch.Tensor,
             unit_first: torch.Tensor, unit_second: torch.Tensor,
             rounds: int = MAX_COLLAPSE_ROUNDS) -> torch.Tensor:
    """Merge graph vertices along collapse edges where more collapse edges than unit edges join two groups.

    Args:
        count: Number of graph vertices.
        first: ``(C,)`` int64 one end of each collapse edge.
        second: ``(C,)`` int64 the other end.
        error: ``(C,)`` distance between the two ends' lattice points, in spacings.
        unit_first: ``(U,)`` int64 one end of each edge one lattice step long.
        unit_second: ``(U,)`` int64 its other end.
        rounds: Most merging rounds.

    Returns:
        ``(count,)`` int64 group of each vertex, the smallest vertex index in the group.
    """
    # Each round pairs groups by their lowest-error collapse edge and merges the pairs. Two groups
    # joined by as many unit edges as collapse edges stay apart.
    device = first.device
    label = torch.arange(count, device=device)
    for _ in range(rounds):
        a, b = label[first], label[second]
        live = a != b
        if not bool(live.any()):
            break
        a, b, err = a[live], b[live], error[live]
        keys = _pair_keys(a, b, count)
        pair_keys, inverse = torch.unique(keys, return_inverse=True)
        support = torch.bincount(inverse, minlength=pair_keys.shape[0])
        ua, ub = label[unit_first], label[unit_second]
        apart = ua != ub
        if bool(apart.any()):
            conflict, opposed = torch.unique(_pair_keys(ua[apart], ub[apart], count), return_counts=True)
            at = torch.searchsorted(conflict, pair_keys).clamp_max(conflict.shape[0] - 1)
            against = torch.where(conflict[at] == pair_keys, opposed[at], torch.zeros_like(opposed[at]))
            free = (support > against)[inverse]
            a, b, err, keys = a[free], b[free], err[free], keys[free]
            if keys.numel() == 0:
                break
            pair_keys, inverse = torch.unique(keys, return_inverse=True)
            support = torch.bincount(inverse, minlength=pair_keys.shape[0])
        best = torch.full((pair_keys.shape[0],), math.inf, dtype=err.dtype, device=device)
        best.scatter_reduce_(0, inverse, err, reduce="amin")
        score = torch.log1p(support.to(err.dtype)) - 4.0 * best
        pa, pb = pair_keys // count, pair_keys % count
        partner = match_pairs(count, torch.cat([pa, pb]), torch.cat([pb, pa]), torch.cat([score, score]))
        merged = partner >= 0
        if not bool(merged.any()):
            break
        index = torch.arange(count, device=device)
        root = torch.where(merged, torch.minimum(index, partner), index)
        label = root[label]
    return label


def _around(points: torch.Tensor, normals: torch.Tensor, frames: torch.Tensor, first: torch.Tensor,
            second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every edge in both directions, sorted by its start and then by angle around the start.

    Args:
        points: ``(V, 3)`` vertex positions.
        normals: ``(V, 3)`` unit normals the angles turn about.
        frames: ``(V, 3)`` unit tangents the angles are measured from.
        first: ``(E,)`` int64 edge starts.
        second: ``(E,)`` int64 edge ends.

    Returns:
        ``(src, dst, angle, edge, offsets)``: ``(2E,)`` starts, ends, angles in ``[-pi, pi]`` and
        undirected edge of each, and ``(V + 1,)`` start of each vertex's run.
    """
    count = int(points.shape[0])
    edge_count = int(first.shape[0])
    src = torch.cat([first, second])
    dst = torch.cat([second, first])
    vector = points[dst] - points[src]
    q = frames[src]
    t = torch.cross(normals[src], q, dim=1)
    angle = torch.atan2((vector * t).sum(1), (vector * q).sum(1)).double()
    order = torch.argsort(src.double() * 8.0 + (angle + math.pi))
    src = src[order]
    offsets = torch.zeros(count + 1, dtype=torch.int64, device=points.device)
    offsets[1:] = torch.cumsum(torch.bincount(src, minlength=count), 0)
    return src, dst[order], angle[order], order % max(edge_count, 1), offsets


def skip_edges(points: torch.Tensor, normals: torch.Tensor, frames: torch.Tensor, first: torch.Tensor,
               second: torch.Tensor, min_angle_deg: float = MIN_EDGE_ANGLE_DEG) -> torch.Tensor:
    """The longer of every two edges leaving one vertex less than ``min_angle_deg`` apart.

    Args:
        points: ``(V, 3)`` vertex positions.
        normals: ``(V, 3)`` unit normals.
        frames: ``(V, 3)`` unit tangents the angles are measured from.
        first: ``(E,)`` int64 edge starts.
        second: ``(E,)`` int64 edge ends.
        min_angle_deg: Smallest angle two edges at one vertex may make.

    Returns:
        ``(E,)`` bool.
    """
    edge_count = int(first.shape[0])
    drop = torch.zeros(edge_count, dtype=torch.bool, device=first.device)
    if edge_count == 0:
        return drop
    src, _, angle, edge, offsets = _around(points, normals, frames, first, second)
    index = torch.arange(src.shape[0], device=src.device)
    following = index + 1
    following = torch.where(following == offsets[src + 1], offsets[src], following)
    gap = torch.remainder(angle[following] - angle, 2.0 * math.pi)
    close = (gap < math.radians(float(min_angle_deg))) & (edge[following] != edge)
    if not bool(close.any()):
        return drop
    length = (points[first] - points[second]).norm(dim=1)
    a, b = edge[close], edge[following[close]]
    drop[torch.where(length[a] >= length[b], a, b)] = True
    return drop


def trace_faces(points: torch.Tensor, normals: torch.Tensor, frames: torch.Tensor, first: torch.Tensor,
                second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faces of an embedded graph, from its edges sorted by angle around each vertex.

    Args:
        points: ``(V, 3)`` vertex positions.
        normals: ``(V, 3)`` unit normals.
        frames: ``(V, 3)`` unit tangents the angles are measured from.
        first: ``(E,)`` int64 edge starts.
        second: ``(E,)`` int64 edge ends.

    Returns:
        ``(sizes, indices, facing, spur)``: corners per cycle, the corners in order, per cycle
        the dot product of its vector area with its vertex normals (positive for a face), and
        ``(E,)`` bool marking edges one cycle runs along in both directions.
    """
    device = points.device
    count = int(points.shape[0])
    edge_count = int(first.shape[0])
    if edge_count == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return (empty, empty, torch.zeros(0, dtype=points.dtype, device=device),
                torch.zeros(0, dtype=torch.bool, device=device))
    src, dst, _, edge, offsets = _around(points, normals, frames, first, second)
    degree = offsets[1:] - offsets[:-1]
    position = torch.arange(src.shape[0], device=device) - offsets[src]
    stride = max(count, 1)
    keys, key_order = torch.sort(src * stride + dst)
    twin = key_order[torch.searchsorted(keys, dst * stride + src).clamp_max(keys.shape[0] - 1)]
    successor = offsets[dst] + torch.remainder(position[twin] - 1, degree[dst].clamp_min(1))
    label, rank = order_cycles(successor)
    _, cycle = torch.unique(label, return_inverse=True)
    spur = torch.zeros(edge_count, dtype=torch.bool, device=device)
    spur[edge[cycle == cycle[twin]]] = True
    cycles = int(cycle.max()) + 1
    sizes = torch.bincount(cycle, minlength=cycles)
    sequence = torch.argsort(cycle * (src.shape[0] + 1) + rank)
    indices = src[sequence]
    owner = cycle[sequence]
    start = torch.zeros(cycles + 1, dtype=torch.int64, device=device)
    start[1:] = torch.cumsum(sizes, 0)
    anchor = points[indices[start[owner]]]
    following = torch.arange(indices.shape[0], device=device) + 1
    following = torch.where(following == start[owner + 1], start[owner], following)
    here = points[indices] - anchor
    there = points[indices[following]] - anchor
    area = torch.zeros(cycles, 3, dtype=points.dtype, device=device).index_add_(0, owner, torch.cross(here, there, dim=1))
    normal_sum = torch.zeros(cycles, 3, dtype=points.dtype, device=device).index_add_(0, owner, normals[indices])
    return sizes, indices, (area * normal_sum).sum(1), spur


def _prune(first: torch.Tensor, second: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges left once vertices joined to fewer than two others are removed, repeatedly."""
    for _ in range(MAX_PRUNE_PASSES):
        degree = torch.bincount(torch.cat([first, second]), minlength=count)
        loose = (degree[first] < 2) | (degree[second] < 2)
        if not bool(loose.any()):
            break
        first, second = first[~loose], second[~loose]
    return first, second


def extract(level: Level) -> Extraction:
    """Read a polygon mesh off a level's solved fields.

    Args:
        level: The level, with ``q`` and ``o`` solved.

    Returns:
        The `Extraction`.
    """
    device = level.vertices.device
    count = level.count
    vertices, normals, q, o, scales = level.vertices, level.normals, level.q, level.o, level.scales
    t = torch.cross(normals, q, dim=1)
    i, j = level.first, level.second
    qi, ti = q[i], t[i]
    qj, tj = q[j], t[j]
    match = torch.stack([(qj * qi).sum(1), (tj * qi).sum(1), -(qj * qi).sum(1), -(tj * qi).sum(1)], 1).argmax(1)
    middle = middle_point(vertices[i], normals[i], vertices[j], normals[j])
    mine, a1, a2 = lattice_round(o[i], qi, ti, scales[i], middle)
    theirs, b1, b2 = lattice_round(o[j], qj, tj, scales[j], mine)
    turned = torch.stack([torch.stack([b1, b2], 1), torch.stack([b2, -b1], 1),
                          torch.stack([-b1, -b2], 1), torch.stack([-b2, b1], 1)], 1)
    c = turned[torch.arange(match.shape[0], device=device), match]
    d1, d2 = a1 - c[:, 0], a2 - c[:, 1]
    error = (theirs - mine).norm(dim=1) / torch.minimum(scales[i], scales[j]).clamp_min(1e-30)
    together = (d1 == 0) & (d2 == 0) & (error <= MAX_COLLAPSE_ERROR)
    unit = (d1.abs() + d2.abs()) == 1
    roots = collapse(count, i[together], j[together], error[together], i[unit], j[unit])
    _, cluster = torch.unique(roots, return_inverse=True)
    clusters = int(cluster.max()) + 1
    members = torch.bincount(cluster, minlength=clusters)
    weight = torch.exp(-9.0 * ((o - vertices) ** 2).sum(1) / (scales ** 2).clamp_min(1e-30))
    mass = torch.zeros(clusters, dtype=weight.dtype, device=device).index_add_(0, cluster, weight)
    safe_mass = mass.clamp_min(1e-30)
    points = torch.zeros(clusters, 3, dtype=o.dtype, device=device).index_add_(0, cluster, o * weight[:, None])
    points /= safe_mass[:, None]
    lead = _scatter_max_pick(cluster, weight, clusters)
    weak = mass < 1e-12
    points[weak] = o[lead[weak]]
    cluster_normals = torch.zeros_like(points).index_add_(0, cluster, normals * weight[:, None])
    cluster_normals = torch.nn.functional.normalize(cluster_normals, dim=1)
    flat = cluster_normals.norm(dim=1) < 0.5
    cluster_normals[flat] = normals[lead[flat]]
    frames = project_tangent(q[lead], cluster_normals)
    cluster_scale = torch.zeros(clusters, dtype=scales.dtype, device=device).index_add_(0, cluster, scales)
    cluster_scale /= members.clamp_min(1).to(scales.dtype)
    given = level.constraints
    pinned = given.point_weight > 0
    on_line = pinned & (given.line.norm(dim=1) > 0)
    at_corner = pinned & ~on_line
    feature = torch.zeros(clusters, dtype=torch.bool, device=device)
    feature[cluster[pinned]] = True
    corner = torch.zeros(clusters, dtype=torch.bool, device=device)
    corner[cluster[at_corner]] = True
    if bool(at_corner.any()):
        corner_lead = _scatter_max_pick(cluster, torch.where(at_corner, weight, torch.full_like(weight, -1.0)), clusters)
        points[corner] = given.point[corner_lead[corner]].to(points.dtype)
    alive = members.to(weight.dtype) > MIN_WEIGHT_SHARE * float(members.float().mean())
    ci, cj = cluster[i[unit]], cluster[j[unit]]
    keep = (ci != cj) & alive[ci] & alive[cj]
    ci, cj = ci[keep], cj[keep]
    keys = torch.unique(_pair_keys(ci, cj, clusters))
    first, second = keys // clusters, keys % clusters
    length = (points[first] - points[second]).norm(dim=1)
    reach = MAX_EDGE_SPACINGS * torch.maximum(cluster_scale[first], cluster_scale[second])
    short = length <= reach
    first, second = first[short], second[short]
    skipping = skip_edges(points, cluster_normals, frames, first, second)
    first, second = _prune(first[~skipping], second[~skipping], clusters)
    spurs = 0
    for _ in range(MAX_SPUR_PASSES):
        sizes, indices, facing, spur = trace_faces(points, cluster_normals, frames, first, second)
        if not bool(spur.any()):
            break
        spurs += int(spur.sum())
        first, second = _prune(first[~spur], second[~spur], clusters)
    owner = torch.repeat_interleave(torch.arange(sizes.shape[0], device=device), sizes)
    unique_pairs = torch.unique(owner * clusters + indices)
    distinct = torch.bincount(unique_pairs // clusters, minlength=sizes.shape[0])
    face = (facing > 0) & (sizes >= 3) & (sizes <= MAX_FACE_CORNERS) & (distinct == sizes)
    mesh = PolyMesh(points, sizes, indices, {"feature": feature, "corner": corner, "scale": cluster_scale})
    mesh = mesh.select(face).compacted()
    figures = {
        "graph vertices": count,
        "lattice points": clusters,
        "lattice points dropped": int((~alive).sum()),
        "edges": int(first.shape[0]),
        "skip edges removed": int(skipping.sum()),
        "spur edges removed": spurs,
        "faces traced": int(face.sum()),
        "border cycles": int((facing <= 0).sum()),
        "cycles dropped": int((~face & (facing > 0)).sum()),
    }
    return Extraction(mesh=mesh, figures=figures)
