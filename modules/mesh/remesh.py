"""Isotropic and adaptive remeshing of a triangle mesh onto its own surface.

Edges split above 4/3 and collapse below 4/5 of the target length, flip towards valence 6, relax
and project back, in parallel rounds. Lengths are in model units.
"""

from __future__ import annotations

__all__ = [
    "MAX_FACES",
    "RemeshResult",
    "Source",
    "estimate_faces",
    "remesh",
    "remesh_part",
    "target_length",
]

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch

from ..model3d.scene import Part
from . import device as mesh_device
from . import topology
from .arrays import TriMesh, from_part, to_part
from .bvh import TriangleBVH, closest_point_on_triangles
from .retopo import prepare
from .subdivide import blend_skin
from .transfer import Samples, interpolate

#: Most faces a remesh may produce.
MAX_FACES = 4_000_000

#: Most passes of the remeshing loop.
MAX_ITERATIONS = 50

#: Faces an equilateral triangulation with unit edges puts on a unit of area.
EQUILATERAL_DENSITY = 4.0 / math.sqrt(3.0)

#: Measured faces per predicted face of a remesh at a given target length.
FACE_CALIBRATION = 0.92

#: Edges longer than this many target lengths are split.
SPLIT_RATIO = 4.0 / 3.0

#: Edges shorter than this many target lengths are collapsed.
COLLAPSE_RATIO = 4.0 / 5.0

#: Most rounds of independent collapses or flips per pass.
SUB_ROUNDS = 4

#: A round of collapses or flips doing fewer than this share of the first round's ends the stage.
ROUND_FLOOR = 0.25

#: Split passes run per remeshing pass.
SPLIT_PASSES = 3

#: Smallest cosine between a face normal before and after a collapse.
COLLAPSE_NORMAL_COSINE = 0.5

#: Smallest cosine between a flipped face's normal and the pair's mean normal.
FLIP_NORMAL_COSINE = 0.5

#: Smallest corner angle, in degrees, a collapse or flip may leave in a new face.
MIN_NEW_ANGLE = 2.0

#: Smallest cosine between a face normal before and after a relaxation step.
FOLD_COSINE = 0.2

#: Passes putting back vertices of faces a relaxation step turned over.
FOLD_PASSES = 4

#: Steps of the walk that projects a vertex onto the source near where it was.
WALK_STEPS = 4

#: Share of the wanted face count a remesh may miss by before its target length is retuned.
FACE_TOLERANCE = 0.02

#: Target lengths per edge length at which a regular fan can collapse that edge.
COLLAPSE_REACH = 1.5

#: Largest change of the target lengths in one retune.
MAX_RETARGET = 1.25

#: Passes at the end of a face-count remesh that keep the target length fixed.
SETTLE_PASSES = 2

#: Target lengths a crease chain must reach to be kept as a feature.
MIN_CHAIN_TARGETS = 2.0

#: Selection passes that grow one round's set of independent collapses.
SELECT_ROUNDS = 4

#: Faces per expected face above which the source is first coarsened by collapses alone.
COARSEN_FROM = 2.0

#: Most rounds of the coarsening that runs before the remeshing passes.
MAX_COARSEN_ROUNDS = 40

#: Share of the faces a coarsening round must remove before flips are tried to unblock it.
COARSEN_STALL = 0.08

#: Share of the faces a coarsening round must remove, after flipping, to go on.
COARSEN_FLOOR = 0.01

#: Kinds of working vertex.
FREE, CHAIN, LOCKED = 0, 1, 2


@dataclass
class RemeshResult:
    """A remeshed surface and where each of its vertices landed on the source.

    Attributes:
        mesh: The new mesh, its attributes sampled from the source.
        source_face: ``(N,)`` int64 source triangle each new vertex was sampled on.
        barycentric: ``(N, 3)`` weights of that triangle's corners.
        stats: Counts and timings of the run.
    """

    mesh: TriMesh
    source_face: torch.Tensor
    barycentric: torch.Tensor
    stats: dict = field(default_factory=dict)


@dataclass
class Source:
    """The surface being remeshed, welded, with its feature chains and lookup tables.

    Attributes:
        vertices: ``(V, 3)`` float64 welded positions.
        faces: ``(F, 3)`` int64 welded triangles.
        face_index: ``(F,)`` int64 row of each triangle in the split mesh's faces.
        region: ``(F,)`` int64 attribute region of each triangle.
        corner: ``(V,)`` bool vertices held still.
        edge_chain: ``(E,)`` int64 chain of each welded edge, ``-1`` off every feature.
        chain_seam: ``(C,)`` bool chains that are attribute seams.
        topology: Edge topology of the welded triangles.
        sizing: ``(V,)`` float32 relative target length at each vertex.
        ring_faces: ``(F, 3)`` int64 the triangles on the vertices a walk turns around: the split
            vertices when seams are kept, so a walk never crosses one, else the welded ones.
        face_offsets: CSR offsets of the triangles around each of those vertices.
        face_members: Triangles around each of those vertices.
        segments: ``(S, 3)`` int64 feature segments as degenerate triangles ``(a, b, b)``.
        segment_chain: ``(S,)`` int64 chain of each segment.
        segment_face: ``(S,)`` int64 a triangle beside each segment.
        segment_offsets: ``(V + 1,)`` CSR offsets of the segments at each vertex.
        segment_members: Segments at each vertex.
        spacing: Distance between the copies of the surface placed per region or chain.
        area: Surface area of the welded triangles.
        length: Uniform target edge length.
        turned: ``(F,)`` bool triangles whose second and third corners are swapped against the
            split mesh's face, so all are wound alike.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    face_index: torch.Tensor
    region: torch.Tensor
    corner: torch.Tensor
    edge_chain: torch.Tensor
    chain_seam: torch.Tensor
    topology: topology.EdgeTopology
    sizing: torch.Tensor
    ring_faces: torch.Tensor
    face_offsets: torch.Tensor
    face_members: torch.Tensor
    segments: torch.Tensor
    segment_chain: torch.Tensor
    segment_face: torch.Tensor
    segment_offsets: torch.Tensor
    segment_members: torch.Tensor
    spacing: float
    area: float = 0.0
    length: float = 0.0
    turned: torch.Tensor | None = None
    _trees: dict = field(default_factory=dict)

    @property
    def chain_count(self) -> int:
        """Number of feature chains."""
        return int(self.chain_seam.shape[0])

    @property
    def region_count(self) -> int:
        """Number of attribute regions."""
        return int(self.region.max()) + 1 if self.region.numel() else 1

    def tree(self, kind: str) -> tuple[TriangleBVH, torch.Tensor]:
        """A search tree over every region (``"region"``) or chain (``"chain"``), each moved to its
        own grid offset, and the offsets.

        Args:
            kind: ``"region"`` or ``"chain"``.

        Returns:
            ``(tree, offsets)``.
        """
        if kind not in self._trees:
            device = self.vertices.device
            if kind == "region":
                offsets = _grid_offsets(self.region_count, self.spacing, device)
                corners, label = self.faces, self.region
            else:
                offsets = _grid_offsets(max(self.chain_count, 1), self.spacing, device)
                corners, label = self.segments, self.segment_chain
            placed = self.vertices[corners] + offsets[label][:, None, :]
            index = torch.arange(corners.shape[0] * 3, device=device).reshape(-1, 3)
            self._trees[kind] = (TriangleBVH(placed.reshape(-1, 3), index, device=device), offsets)
        return self._trees[kind]


def target_length(area: float, faces: int) -> float:
    """Edge length that covers an area with about a given number of remeshed faces.

    Args:
        area: Surface area in square model units.
        faces: Faces wanted.

    Returns:
        Target edge length in model units.
    """
    return math.sqrt(EQUILATERAL_DENSITY * FACE_CALIBRATION * max(area, 0.0) / max(int(faces), 1))


def estimate_faces(area: float, length: float) -> int:
    """Faces a remesh at a target edge length puts on an area.

    Args:
        area: Surface area in square model units.
        length: Target edge length in model units.

    Returns:
        The expected face count.
    """
    if not length > 0:
        return 1 << 62
    return int(math.ceil(EQUILATERAL_DENSITY * FACE_CALIBRATION * max(area, 0.0) / (length * length)))


# ---------------------------------------------------------------------------------------------
# Small vectorised helpers


def _ranges(offsets: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand CSR rows into ``(owner, position)`` pairs, one per entry of each row."""
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    total = int(counts.sum())
    owner = torch.repeat_interleave(torch.arange(rows.shape[0], device=rows.device), counts)
    before = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=rows.device) - before[owner]
    return owner, starts[owner] + within


def _csr(items: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of the elements touching each vertex, from an ``(K, C)`` element table."""
    flat = items.reshape(-1)
    order = torch.argsort(flat, stable=True)
    offsets = torch.zeros(count + 1, dtype=torch.int64, device=items.device)
    offsets[1:] = torch.cumsum(torch.bincount(flat, minlength=count), 0)
    return offsets, order // items.shape[1]


def _edge_lookup(topo: topology.EdgeTopology, stride: int) -> torch.Tensor:
    """Sorted keys of the topology's edges."""
    return topo.edges[:, 0] * stride + topo.edges[:, 1]


def _has_edge(keys: torch.Tensor, first: torch.Tensor, second: torch.Tensor, stride: int) -> torch.Tensor:
    """Whether each ``(first, second)`` pair is an edge, looked up in `_edge_lookup` keys."""
    if keys.numel() == 0:
        return torch.zeros_like(first, dtype=torch.bool)
    query = torch.minimum(first, second) * stride + torch.maximum(first, second)
    at = torch.searchsorted(keys, query).clamp_max(keys.shape[0] - 1)
    return keys[at] == query


def _any_by(owner: torch.Tensor, flags: torch.Tensor, count: int) -> torch.Tensor:
    """Per owner, whether any of its flags is set."""
    total = torch.zeros(count, dtype=torch.int64, device=owner.device)
    total.index_add_(0, owner, flags.long())
    return total > 0


def _ranks(keys: torch.Tensor) -> torch.Tensor:
    """Rank of each key in ascending order, all distinct."""
    order = torch.argsort(keys, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.shape[0], device=keys.device)
    return rank


def _winners(rank: torch.Tensor, owner: torch.Tensor, items: torch.Tensor, item_count: int) -> torch.Tensor:
    """Candidates holding the best rank on every item they claim.

    Args:
        rank: ``(C,)`` distinct ranks, lower is better.
        owner: ``(K,)`` candidate of each claim.
        items: ``(K,)`` item each claim names.
        item_count: Number of items.

    Returns:
        ``(C,)`` bool.
    """
    best = torch.full((item_count,), rank.shape[0], dtype=torch.int64, device=rank.device)
    best.scatter_reduce_(0, items, rank[owner], reduce="amin")
    lost = best[items] != rank[owner]
    return ~_any_by(owner, lost, rank.shape[0])


def _grid_offsets(count: int, spacing: float, device) -> torch.Tensor:
    """Distinct offsets on a cubic grid, ``spacing`` apart, one per label."""
    side = max(1, math.ceil(round(max(count, 1) ** (1.0 / 3.0), 9)))
    index = torch.arange(max(count, 1), dtype=torch.int64, device=device)
    cells = torch.stack([index % side, (index // side) % side, index // (side * side)], dim=1)
    return cells.double() * spacing


def _normals(points: torch.Tensor) -> torch.Tensor:
    """Unnormalised normal of each ``(..., 3, 3)`` triangle."""
    return torch.cross(points[..., 1, :] - points[..., 0, :], points[..., 2, :] - points[..., 0, :], dim=-1)


def _smallest_angle_sine(points: torch.Tensor) -> torch.Tensor:
    """Sine of the smallest corner angle of each ``(..., 3, 3)`` triangle."""
    twice_area = _normals(points).norm(dim=-1)
    sides = torch.stack([(points[..., 1, :] - points[..., 0, :]).norm(dim=-1),
                         (points[..., 2, :] - points[..., 1, :]).norm(dim=-1),
                         (points[..., 0, :] - points[..., 2, :]).norm(dim=-1)], dim=-1)
    longest_pair = sides.prod(dim=-1) / sides.amin(dim=-1).clamp_min(1e-30)
    return twice_area / longest_pair.clamp_min(1e-30)


# ---------------------------------------------------------------------------------------------
# Source analysis


def _chains(vertices, faces, topo, feature_angle, seams, min_length: float = 0.0):
    """Feature chains of a welded mesh: corners, chain of each edge and seam flag per chain.

    Args:
        vertices: ``(V, 3)`` welded positions.
        faces: ``(F, 3)`` triangles.
        topo: Their edge topology.
        feature_angle: Dihedral angle in degrees from which an edge is a crease.
        seams: ``(E,)`` bool attribute seams.
        min_length: Crease chains shorter than this, with no border, seam or non-manifold edge, are dropped.

    Returns:
        ``(edge_chain, corner, chain_seam)``.
    """
    required = topo.boundary | topo.non_manifold | seams
    crease = torch.zeros_like(required)
    if feature_angle < 180.0:
        crease = (topology.dihedral_angles(vertices, faces, topo) >= math.radians(feature_angle)) & ~required
    edge_chain, corner, chain_seam = _link(vertices, faces, topo, required | crease, seams, feature_angle)
    if min_length > 0 and bool(crease.any()):
        on = edge_chain >= 0
        lengths = topology.edge_lengths(vertices, topo.edges)
        total = torch.zeros(chain_seam.shape[0], dtype=lengths.dtype, device=lengths.device)
        total.index_add_(0, edge_chain[on], lengths[on])
        held = torch.zeros(chain_seam.shape[0], dtype=torch.bool, device=lengths.device)
        held[edge_chain[on & required]] = True
        keep = held | (total >= float(min_length))
        if not bool(keep.all()):
            crease = crease & on & keep[edge_chain.clamp_min(0)]
            edge_chain, corner, chain_seam = _link(vertices, faces, topo, required | crease, seams, feature_angle)
    return edge_chain, corner, chain_seam


def _link(vertices, faces, topo, feature, seams, feature_angle):
    """Join feature edges into chains that break at corners, bends and seam ends."""
    device = faces.device
    vertex_count = int(vertices.shape[0])
    chosen = torch.nonzero(feature).squeeze(1)
    ends = topo.edges[chosen]
    feature_valence = torch.bincount(ends.reshape(-1), minlength=vertex_count)
    corner = (feature_valence > 0) & (feature_valence != 2)
    corner |= topology.non_manifold_vertices(faces, vertex_count, topo)
    corner[topo.edges[topo.non_manifold].reshape(-1)] = True
    edge_chain = torch.full((topo.edge_count,), -1, dtype=torch.int64, device=device)
    if chosen.numel() == 0:
        return edge_chain, corner, torch.zeros(0, dtype=torch.bool, device=device)
    vertex = ends.reshape(-1)
    owner_edge = chosen.repeat_interleave(2)
    other = ends[:, [1, 0]].reshape(-1)
    order = torch.argsort(vertex, stable=True)
    vertex, owner_edge, other = vertex[order], owner_edge[order], other[order]
    starts = torch.ones_like(vertex, dtype=torch.bool)
    starts[1:] = vertex[1:] != vertex[:-1]
    first = torch.nonzero(starts & (feature_valence[vertex] == 2)).squeeze(1)
    middle = vertex[first]
    edge_a, edge_b = owner_edge[first], owner_edge[first + 1]
    before, after = other[first], other[first + 1]
    incoming = vertices[middle] - vertices[before]
    outgoing = vertices[after] - vertices[middle]
    cosine = (incoming * outgoing).sum(1) / (incoming.norm(dim=1) * outgoing.norm(dim=1)).clamp_min(1e-30)
    side = (torch.cross(incoming, outgoing, dim=1) * topology.vertex_normals(vertices, faces)[middle]).sum(1)
    turn = torch.acos(cosine.clamp(-1.0, 1.0)) * torch.where(side < 0, -1.0, 1.0).to(cosine.dtype)
    # Turns of the chain neighbours, read in the same direction; zigzags of a dense mesh cancel out.
    signed = torch.zeros(vertex_count, dtype=turn.dtype, device=device)
    previous = torch.full((vertex_count,), -1, dtype=torch.int64, device=device)
    signed[middle], previous[middle] = turn, before
    behind = torch.where(previous[before] == middle, -signed[before], signed[before])
    ahead = torch.where(previous[after] == middle, signed[after], -signed[after])
    settled = turn + 0.5 * (behind + ahead)
    limit = math.radians(min(feature_angle, 179.0))
    bent = (turn.abs() > limit) & (settled.abs() > limit)
    mixed = seams[edge_a] != seams[edge_b]
    corner[middle[bent | mixed]] = True
    join = ~corner[middle]
    labels = topology.union_find(topo.edge_count, edge_a[join], edge_b[join])
    _, chain = torch.unique(labels[chosen], return_inverse=True)
    edge_chain[chosen] = chain
    chain_count = int(chain.max()) + 1
    chain_seam = torch.zeros(chain_count, dtype=torch.bool, device=device)
    chain_seam[chain[seams[chosen]]] = True
    return edge_chain, corner, chain_seam


def _curvature_sizing(vertices: torch.Tensor, faces: torch.Tensor, topo, adaptivity: float) -> torch.Tensor:
    """Relative edge length at each vertex from its largest principal curvature.

    Args:
        vertices: ``(V, 3)`` float64 positions.
        faces: ``(F, 3)`` triangles.
        topo: Their edge topology.
        adaptivity: ``0`` for a uniform field to ``1`` for lengths fully following curvature.

    Returns:
        ``(V,)`` float32 factors around 1.
    """
    count = int(vertices.shape[0])
    if adaptivity <= 0 or faces.shape[0] == 0:
        return torch.ones(count, dtype=torch.float32, device=vertices.device)
    area = topology.vertex_areas(vertices, faces).clamp_min(1e-30)
    laplace = topology.cotangent_laplacian(vertices, faces)
    mean = (laplace @ vertices).norm(dim=1) / (2.0 * area)
    angles = topology.corner_angles(vertices, faces)
    total = torch.zeros(count, dtype=vertices.dtype, device=vertices.device)
    total.index_add_(0, faces.reshape(-1), angles.reshape(-1))
    on_border = torch.zeros(count, dtype=torch.bool, device=vertices.device)
    on_border[topo.edges[topo.boundary].reshape(-1)] = True
    defect = torch.where(on_border, math.pi - total, 2.0 * math.pi - total)
    gauss = defect / area
    largest = mean + (mean * mean - gauss).clamp_min(0).sqrt()
    used = torch.zeros(count, dtype=torch.bool, device=vertices.device)
    used[faces.reshape(-1)] = True
    weights = area * used
    reference = torch.exp((weights * largest.clamp_min(1e-12).log()).sum() / weights.sum().clamp_min(1e-30))
    largest = largest.clamp(float(reference) * 1e-4, float(reference) * 1e4)
    neighbours = topology.uniform_laplacian(faces, count, topo, dtype=vertices.dtype)
    degree = torch.bincount(topo.edges.reshape(-1), minlength=count).to(vertices.dtype).clamp_min(1)
    log_curvature = largest.log()
    for _ in range(3):
        log_curvature = log_curvature - 0.5 * (neighbours @ log_curvature) / degree
    reference = float((weights * log_curvature).sum() / weights.sum().clamp_min(1e-30))
    bound = 1.0 + 3.0 * float(adaptivity)
    factor = torch.exp(-0.5 * float(adaptivity) * (log_curvature - reference))
    return factor.clamp(1.0 / bound, bound).float()


def _single_cover(faces: torch.Tensor) -> torch.Tensor:
    """Triangles kept so no two cover the same three vertices.

    Args:
        faces: ``(F, 3)`` int64 triangles.

    Returns:
        ``(F,)`` bool: the first of each group wound alike, none of a group wound both ways.
    """
    count = int(faces.shape[0])
    if count == 0:
        return torch.zeros(0, dtype=torch.bool, device=faces.device)
    ordered, order = faces.sort(1)
    parity = ((order[:, 0] > order[:, 1]).long() + (order[:, 0] > order[:, 2]).long()
              + (order[:, 1] > order[:, 2]).long()) % 2
    _, group = torch.unique(ordered, dim=0, return_inverse=True)
    groups = int(group.max()) + 1
    index = torch.arange(count, device=faces.device)
    first = torch.full((groups,), count, dtype=torch.int64, device=faces.device)
    first.scatter_reduce_(0, group, index, reduce="amin")
    high = torch.zeros(groups, dtype=torch.int64, device=faces.device).scatter_reduce_(0, group, parity, reduce="amax")
    low = torch.ones(groups, dtype=torch.int64, device=faces.device).scatter_reduce_(0, group, parity, reduce="amin")
    return (index == first[group]) & (high == low)[group]


def _source(mesh: TriMesh, adaptivity: float, feature_angle: float, keep_seams: bool, device,
            length: float | None = None, faces_wanted: int | None = None) -> Source:
    """Weld a split mesh, keep one triangle over any three vertices, wind them alike and gather
    everything the remesher reads from it."""
    split_vertices = mesh.vertices.to(device).double()
    split_faces = mesh.faces.to(device).long()
    diagonal = float((split_vertices.amax(0) - split_vertices.amin(0)).norm()) if split_vertices.shape[0] else 0.0
    welded, faces, remap = topology.weld(split_vertices, split_faces, tolerance=1e-7 * diagonal)
    sound = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    rows = torch.nonzero(sound).squeeze(1)
    faces, split_faces = faces[sound], split_faces[sound]
    single = _single_cover(faces)
    rows, faces, split_faces = rows[single], faces[single], split_faces[single]
    if faces.shape[0] == 0:
        raise ValueError("every triangle of the mesh has a zero-length side, so there is no surface to remesh")
    oriented = prepare.orient(welded, faces)
    turned = (oriented != faces).any(1)
    faces = oriented
    split_faces = torch.where(turned[:, None], split_faces[:, [0, 2, 1]], split_faces)
    vertex_count = int(welded.shape[0])
    topo = topology.edge_topology(faces, vertex_count)
    area = float(topology.face_areas(welded, faces).sum())
    uniform = target_length(area, faces_wanted) if faces_wanted is not None else float(length or 0.0)
    if keep_seams:
        seams = topology.seam_edges(split_faces, remap, topo)
        region = topology.face_components(split_faces, int(split_vertices.shape[0]))
    else:
        seams = torch.zeros(topo.edge_count, dtype=torch.bool, device=device)
        region = torch.zeros(faces.shape[0], dtype=torch.int64, device=device)
    edge_chain, corner, chain_seam = _chains(welded, faces, topo, feature_angle, seams,
                                             MIN_CHAIN_TARGETS * uniform)
    ring_faces = split_faces if keep_seams else faces
    face_offsets, face_members = _csr(ring_faces, int(split_vertices.shape[0]) if keep_seams else vertex_count)
    chosen = torch.nonzero(edge_chain >= 0).squeeze(1)
    pairs = topo.edges[chosen]
    segments = torch.stack([pairs[:, 0], pairs[:, 1], pairs[:, 1]], dim=1)
    segment_offsets, segment_members = _csr(pairs, vertex_count)
    return Source(
        vertices=welded, faces=faces, face_index=rows, region=region, turned=turned,
        corner=corner, edge_chain=edge_chain, chain_seam=chain_seam, topology=topo,
        sizing=_curvature_sizing(welded, faces, topo, adaptivity), ring_faces=ring_faces,
        face_offsets=face_offsets, face_members=face_members, segments=segments,
        segment_chain=edge_chain[chosen], segment_face=topo.edge_faces[chosen, 0],
        segment_offsets=segment_offsets, segment_members=segment_members,
        spacing=4.0 * max(diagonal, 1e-6) + 1.0, area=area, length=uniform,
    )


def _walk(points, start, label, ring, corners, element_label, offsets, members, vertices):
    """Closest element to each point among those sharing its label, found by walking from a
    starting element across the elements around its corners.

    Args:
        points: ``(Q, 3)`` float64 query points.
        start: ``(Q,)`` element to start from.
        label: ``(Q,)`` label the element must carry.
        ring: ``(K, 3)`` corners of each element in the numbering the walk turns around.
        corners: ``(K, 3)`` source vertices of each element, indexing ``vertices``.
        element_label: ``(K,)`` label of each element.
        offsets: CSR offsets of the elements around each ring vertex.
        members: Elements around each ring vertex.
        vertices: ``(V, 3)`` float64 source positions.

    Returns:
        ``(element, barycentric, settled)``: the element reached, the closest point's weights on
        it and whether the walk settled with a match.
    """
    device = points.device
    count = int(points.shape[0])
    element = start.clone()
    weights = torch.zeros(count, 3, dtype=points.dtype, device=device)
    settled = torch.zeros(count, dtype=torch.bool, device=device)
    active = torch.arange(count, device=device)
    for _ in range(WALK_STEPS):
        if active.numel() == 0:
            break
        around = ring[element[active]].reshape(-1)
        owner, position = _ranges(offsets, around)
        owner = owner // 3
        candidate = members[position]
        keep = element_label[candidate] == label[active][owner]
        owner, candidate = owner[keep], candidate[keep]
        triangle = vertices[corners[candidate]]
        distance, barycentric = closest_point_on_triangles(points[active][owner], triangle[:, 0], triangle[:, 1],
                                                           triangle[:, 2])
        best = torch.full((active.shape[0],), math.inf, dtype=distance.dtype, device=device)
        best.scatter_reduce_(0, owner, distance, reduce="amin")
        is_best = distance <= best[owner]
        slot = torch.arange(candidate.shape[0], device=device)
        pick = torch.full((active.shape[0],), candidate.shape[0], dtype=torch.int64, device=device)
        pick.scatter_reduce_(0, owner[is_best], slot[is_best], reduce="amin")
        found = pick < candidate.shape[0]
        rows = active[found]
        reached = candidate[pick[found]]
        moved = reached != element[rows]
        element[rows] = reached
        weights[rows] = barycentric[pick[found]].to(points.dtype)
        settled[rows[~moved]] = True
        active = rows[moved]
    return element, weights, settled


# ---------------------------------------------------------------------------------------------
# Working mesh


class _Work:
    """The mesh being remeshed: positions, triangles and what constrains each vertex."""

    def __init__(self, source: Source, scale: float):
        self.positions = source.vertices.float().clone()
        self.faces = source.faces.clone()
        self.region = source.region.clone()
        self.side_chain = source.edge_chain[source.topology.face_edges]
        count = int(self.positions.shape[0])
        device = self.positions.device
        self.kind = torch.zeros(count, dtype=torch.int8, device=device)
        self.chain = torch.full((count,), -1, dtype=torch.int64, device=device)
        on_chain = source.edge_chain >= 0
        ends = source.topology.edges[on_chain]
        self.chain[ends[:, 0]] = source.edge_chain[on_chain]
        self.chain[ends[:, 1]] = source.edge_chain[on_chain]
        self.kind[self.chain >= 0] = CHAIN
        self.kind[source.corner] = LOCKED
        self.chain[source.corner] = -1
        self.face_source = torch.arange(self.faces.shape[0], device=device)
        self.face_hint = torch.zeros(count, dtype=torch.int64, device=device)
        self.face_hint[self.faces.reshape(-1)] = self.face_source.repeat_interleave(3)
        self.segment_hint = torch.zeros(count, dtype=torch.int64, device=device)
        if source.segments.shape[0]:
            has = source.segment_offsets[1:] > source.segment_offsets[:-1]
            self.segment_hint[has] = source.segment_members[source.segment_offsets[:-1][has]]
        self.size = source.sizing.float() * float(scale)
        self.scale = float(scale)
        self.counts = {"splits": 0, "collapses": 0, "flips": 0, "held_back": 0}

    @property
    def vertex_count(self) -> int:
        """Number of vertex rows, referenced or not."""
        return int(self.positions.shape[0])

    def topology(self) -> tuple[topology.EdgeTopology, torch.Tensor]:
        """Edge topology of the current faces and the chain of each edge, sides made to agree."""
        topo = topology.edge_topology(self.faces, self.vertex_count)
        chain = torch.full((topo.edge_count,), -1, dtype=torch.int64, device=self.faces.device)
        chain.scatter_reduce_(0, topo.face_edges.reshape(-1), self.side_chain.reshape(-1), reduce="amax")
        self.side_chain = chain[topo.face_edges]
        return topo, chain

    def compact(self) -> None:
        """Drop vertices no face uses."""
        used = torch.zeros(self.vertex_count, dtype=torch.bool, device=self.faces.device)
        used[self.faces.reshape(-1)] = True
        index = torch.cumsum(used.long(), 0) - 1
        for name in ("positions", "kind", "chain", "size", "face_hint", "segment_hint"):
            setattr(self, name, getattr(self, name)[used])
        self.faces = index[self.faces]

    def refresh_sources(self) -> None:
        """Point each face at the source triangle under one of its free corners."""
        for k in range(3):
            corner = self.faces[:, k]
            free = self.kind[corner] == FREE
            self.face_source[free] = self.face_hint[corner[free]]

    def stretch(self, factor: float) -> None:
        """Scale every target length."""
        self.size = self.size * float(factor)
        self.scale *= float(factor)


def _split(work: _Work, budget: int) -> int:
    """Split every edge longer than the split ratio at its midpoint; returns the count."""
    topo, edge_chain = work.topology()
    edges = topo.edges
    a, b = edges[:, 0], edges[:, 1]
    target = 0.5 * (work.size[a] + work.size[b])
    excess = (work.positions[a] - work.positions[b]).norm(dim=1) / target
    marked = excess > SPLIT_RATIO
    count = int(marked.sum())
    if count == 0:
        return 0
    room = max(0, (budget - int(work.faces.shape[0])) // 3)
    if count > room:
        if room == 0:
            return 0
        keep = torch.topk(excess, room).indices
        marked = torch.zeros_like(marked)
        marked[keep] = True
        count = room
    device = work.faces.device
    old = work.vertex_count
    midpoint = torch.full((topo.edge_count,), -1, dtype=torch.int64, device=device)
    midpoint[marked] = old + torch.arange(count, device=device)
    chosen = torch.nonzero(marked).squeeze(1)
    ea, eb = edges[chosen, 0], edges[chosen, 1]
    work.positions = torch.cat([work.positions, 0.5 * (work.positions[ea] + work.positions[eb])])
    new_chain = edge_chain[chosen]
    work.kind = torch.cat([work.kind, torch.where(new_chain >= 0, CHAIN, FREE).to(torch.int8)])
    work.chain = torch.cat([work.chain, new_chain])
    work.size = torch.cat([work.size, 0.5 * (work.size[ea] + work.size[eb])])
    work.face_hint = torch.cat([work.face_hint, work.face_source[topo.edge_faces[chosen, 0]]])
    hint_from = torch.where(work.kind[ea] == CHAIN, ea, eb)
    work.segment_hint = torch.cat([work.segment_hint, work.segment_hint[hint_from]])

    faces, sides, region = work.faces, work.side_chain, work.region
    mids = midpoint[topo.face_edges]
    flags = mids >= 0
    marks = flags.sum(1)
    rows = torch.arange(faces.shape[0], device=device)
    pieces, piece_sides, piece_region = [faces[marks == 0]], [sides[marks == 0]], [region[marks == 0]]
    piece_source = [work.face_source[marks == 0]]

    def turned(select: torch.Tensor, shift: torch.Tensor):
        """Corners, sides and midpoints of the selected faces, turned to start at ``shift``."""
        order = (shift[:, None] + torch.arange(3, device=device)[None, :]) % 3
        picked = rows[select][:, None]
        piece_source.extend([work.face_source[select]] * (int(marks[select][0]) + 1))
        return faces[picked, order], sides[picked, order], mids[picked, order], region[select]

    one = marks == 1
    if bool(one.any()):
        corner, side, mid, reg = turned(one, flags[one].long().argmax(1))
        a0, b0, c0 = corner[:, 0], corner[:, 1], corner[:, 2]
        m0 = mid[:, 0]
        blank = torch.full_like(m0, -1)
        pieces += [torch.stack([a0, m0, c0], 1), torch.stack([m0, b0, c0], 1)]
        piece_sides += [torch.stack([side[:, 0], blank, side[:, 2]], 1),
                        torch.stack([side[:, 0], side[:, 1], blank], 1)]
        piece_region += [reg, reg]
    two = marks == 2
    if bool(two.any()):
        unmarked = (~flags[two]).long().argmax(1)
        corner, side, mid, reg = turned(two, (unmarked + 1) % 3)
        a0, b0, c0 = corner[:, 0], corner[:, 1], corner[:, 2]
        m0, m1 = mid[:, 0], mid[:, 1]
        blank = torch.full_like(m0, -1)
        positions = work.positions
        through_a = (positions[a0] - positions[m1]).norm(dim=1) <= (positions[m0] - positions[c0]).norm(dim=1)
        pieces.append(torch.stack([m0, b0, m1], 1))
        piece_sides.append(torch.stack([side[:, 0], side[:, 1], blank], 1))
        piece_region.append(reg)
        first = torch.where(through_a[:, None], torch.stack([a0, m0, m1], 1), torch.stack([a0, m0, c0], 1))
        first_sides = torch.where(
            through_a[:, None], torch.stack([side[:, 0], blank, blank], 1),
            torch.stack([side[:, 0], blank, side[:, 2]], 1))
        second = torch.where(through_a[:, None], torch.stack([a0, m1, c0], 1), torch.stack([m0, m1, c0], 1))
        second_sides = torch.where(
            through_a[:, None], torch.stack([blank, side[:, 1], side[:, 2]], 1),
            torch.stack([blank, side[:, 1], blank], 1))
        pieces += [first, second]
        piece_sides += [first_sides, second_sides]
        piece_region += [reg, reg]
    three = marks == 3
    if bool(three.any()):
        corner, side, mid, reg = turned(three, torch.zeros(int(three.sum()), dtype=torch.int64, device=device))
        a0, b0, c0 = corner[:, 0], corner[:, 1], corner[:, 2]
        m0, m1, m2 = mid[:, 0], mid[:, 1], mid[:, 2]
        blank = torch.full_like(m0, -1)
        pieces += [torch.stack([a0, m0, m2], 1), torch.stack([m0, b0, m1], 1),
                   torch.stack([m2, m1, c0], 1), torch.stack([m0, m1, m2], 1)]
        piece_sides += [torch.stack([side[:, 0], blank, side[:, 2]], 1),
                        torch.stack([side[:, 0], side[:, 1], blank], 1),
                        torch.stack([blank, side[:, 1], side[:, 2]], 1),
                        torch.stack([blank, blank, blank], 1)]
        piece_region += [reg, reg, reg, reg]
    work.faces = torch.cat(pieces)
    work.side_chain = torch.cat(piece_sides)
    work.region = torch.cat(piece_region)
    work.face_source = torch.cat(piece_source)
    work.counts["splits"] += count
    return count


def _may_collapse(work: _Work, u, v, feature, edge_chain):
    """Whether moving ``u`` onto ``v`` keeps every constraint on ``u``."""
    kind_u, kind_v = work.kind[u], work.kind[v]
    free = (kind_u == FREE) & ~feature
    same_chain = (kind_v == CHAIN) & (work.chain[v] == work.chain[u])
    along = (kind_u == CHAIN) & feature & (edge_chain == work.chain[u]) & ((kind_v == LOCKED) | same_chain)
    return free | along


def _collapse(work: _Work, generator: torch.Generator, midpoint: bool = True, prefilter: bool = False) -> int:
    """One round of independent short-edge collapses; returns the count.

    Args:
        work: The working mesh, changed in place.
        generator: Source of the tie-breaking jitter.
        midpoint: Move both ends of an unconstrained edge to its middle rather than one onto the other.
        prefilter: Check only the edges that are the shortest at one of their ends.
    """
    topo, edge_chain = work.topology()
    count = work.vertex_count
    device = work.faces.device
    edges = topo.edges
    a, b = edges[:, 0], edges[:, 1]
    positions, size = work.positions, work.size
    target = 0.5 * (size[a] + size[b])
    ratio = (positions[a] - positions[b]).norm(dim=1) / target
    short = (ratio < COLLAPSE_RATIO) & (topo.edge_face_count <= 2)
    if not bool(short.any()):
        return 0
    feature = edge_chain >= 0
    forward = _may_collapse(work, a, b, feature, edge_chain)
    backward = _may_collapse(work, b, a, feature, edge_chain)
    picked = torch.nonzero(short & (forward | backward)).squeeze(1)
    if picked.numel() == 0:
        return 0
    valence = torch.bincount(edges.reshape(-1), minlength=count)
    score = ratio[picked] + torch.rand(picked.shape[0], generator=generator).to(device) * 1e-3
    if prefilter:
        ends = edges[picked]
        best = torch.full((count,), math.inf, dtype=score.dtype, device=device)
        best.scatter_reduce_(0, ends.reshape(-1), score.repeat_interleave(2), reduce="amin")
        lead = (score <= best[ends[:, 0]]) | (score <= best[ends[:, 1]])
        picked, score = picked[lead], score[lead]
    a, b = a[picked], b[picked]
    forward, backward = forward[picked], backward[picked]
    first = forward & (~backward | (valence[a] <= valence[b]))
    u = torch.where(first, a, b)
    v = torch.where(first, b, a)
    middle = forward & backward & bool(midpoint)
    spot = torch.where(middle[:, None], 0.5 * (positions[u] + positions[v]), positions[v])
    candidates = int(picked.shape[0])
    keys = _edge_lookup(topo, count)

    border = torch.zeros(count, dtype=torch.bool, device=device)
    border[edges[topo.boundary].reshape(-1)] = True
    opposite = topo.edge_opposite[picked]
    present = opposite >= 0
    safe_opposite = opposite.clamp_min(0)
    need = torch.where(border[safe_opposite], 3, 4)
    ok = (~present | (valence[safe_opposite] >= need)).all(1)
    feature_keys = keys[feature]
    ok &= ~(present & _has_edge(feature_keys, u[:, None].expand(-1, 2), safe_opposite, count)).any(1)

    offsets, members = _csr(work.faces, count)
    star_owner, star_position = _ranges(offsets, u)
    star = members[star_position]
    keeper_owner, keeper_position = _ranges(offsets, v)
    keeper_star = members[keeper_position]
    corners = work.faces[star]
    other = corners.reshape(-1)
    other_owner = star_owner.repeat_interleave(3)
    keeper = v[other_owner]
    plain = (other != u[other_owner]) & (other != keeper)
    plain &= (other != opposite[other_owner, 0]) & (other != opposite[other_owner, 1])
    ok &= ~_any_by(other_owner, plain & _has_edge(keys, keeper, other, count), candidates)

    # Faces around each end that stay after the collapse, with that end moved to the new spot.
    moving = middle[keeper_owner]
    sides = [(star_owner, star, u, v), (keeper_owner[moving], keeper_star[moving], v, u)]
    floor = math.sin(math.radians(MIN_NEW_ANGLE))
    for owner, faces_around, mover, partner in sides:
        corners = work.faces[faces_around]
        mover, partner = mover[owner], partner[owner]
        kept = ~(corners == partner[:, None]).any(1)
        place = spot[owner]
        ends = corners.reshape(-1)
        end_owner = owner.repeat_interleave(3)
        reach = (place.repeat_interleave(3, dim=0) - positions[ends]).norm(dim=1)
        limit = SPLIT_RATIO * 0.5 * (size[v[end_owner]] + size[ends])
        far = (reach > limit) & (ends != mover.repeat_interleave(3)) & kept.repeat_interleave(3)
        ok &= ~_any_by(end_owner, far, candidates)
        before = positions[corners]
        after = torch.where((corners == mover[:, None])[:, :, None], place[:, None, :], before)
        old_normal, new_normal = _normals(before), _normals(after)
        cosine = (old_normal * new_normal).sum(1) / (new_normal.norm(dim=1) * old_normal.norm(dim=1)).clamp_min(1e-30)
        folded = kept & ((cosine < COLLAPSE_NORMAL_COSINE) | (_smallest_angle_sine(after) < floor))
        ok &= ~_any_by(owner, folded, candidates)

    alive = torch.nonzero(ok).squeeze(1)
    if alive.numel() == 0:
        return 0
    alive_rank = torch.full((candidates,), candidates, dtype=torch.int64, device=device)
    alive_rank[alive] = _ranks(score[alive])
    claim_owner = torch.cat([star_owner, keeper_owner])
    claim_face = torch.cat([star, keeper_star])
    live = ok[claim_owner]
    claim_owner, claim_face = claim_owner[live], claim_face[live]
    face_count = int(work.faces.shape[0])
    won = torch.zeros_like(ok)
    open_claims = ok.clone()
    taken = torch.zeros(face_count, dtype=torch.bool, device=device)
    for _ in range(SELECT_ROUNDS):
        live = open_claims[claim_owner]
        if not bool(live.any()):
            break
        chosen = _winners(alive_rank, claim_owner[live], claim_face[live], face_count) & open_claims
        if not bool(chosen.any()):
            break
        won |= chosen
        taken[claim_face[chosen[claim_owner]]] = True
        open_claims &= ~_any_by(claim_owner, taken[claim_face], candidates)
    winners = torch.nonzero(won).squeeze(1)
    if winners.numel() == 0:
        return 0
    remap = torch.arange(count, device=device)
    remap[u[winners]] = v[winners]
    shifted = winners[middle[winners]]
    work.positions[v[shifted]] = spot[shifted]
    work.size[v[shifted]] = 0.5 * (size[u[shifted]] + size[v[shifted]])
    faces = remap[work.faces]
    sound = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    work.faces, work.side_chain, work.region = faces[sound], work.side_chain[sound], work.region[sound]
    work.face_source = work.face_source[sound]
    work.counts["collapses"] += int(winners.shape[0])
    return int(winners.shape[0])


def _flip(work: _Work, generator: torch.Generator) -> int:
    """One round of independent valence-improving edge flips; returns the count."""
    topo, edge_chain = work.topology()
    count = work.vertex_count
    device = work.faces.device
    faces = work.faces
    picked = torch.nonzero((topo.edge_face_count == 2) & (edge_chain < 0)).squeeze(1)
    if picked.numel() == 0:
        return 0
    valence = torch.bincount(topo.edges.reshape(-1), minlength=count)
    border = torch.zeros(count, dtype=torch.bool, device=device)
    border[topo.edges[topo.boundary].reshape(-1)] = True
    ideal = torch.where(border, 4, 6)
    f0, f1 = topo.edge_faces[picked, 0], topo.edge_faces[picked, 1]
    k0, k1 = topo.edge_corners[picked, 0], topo.edge_corners[picked, 1]
    a, b, c = faces[f0, k0], faces[f0, (k0 + 1) % 3], faces[f0, (k0 + 2) % 3]
    d = faces[f1, (k1 + 2) % 3]
    consistent = (faces[f1, k1] == b) & (faces[f1, (k1 + 1) % 3] == a) & (work.region[f0] == work.region[f1])

    def deviation(vertex, change):
        return (valence[vertex] + change - ideal[vertex]) ** 2

    gain = (deviation(a, 0) + deviation(b, 0) + deviation(c, 0) + deviation(d, 0)
            - deviation(a, -1) - deviation(b, -1) - deviation(c, 1) - deviation(d, 1))
    ok = consistent & (gain > 0) & (c != d) & (valence[a] > 3) & (valence[b] > 3)
    ok &= ~_has_edge(_edge_lookup(topo, count), c, d, count)
    positions = work.positions
    pa, pb, pc, pd = positions[a], positions[b], positions[c], positions[d]
    n0 = torch.nn.functional.normalize(torch.cross(pb - pa, pc - pa, dim=1), dim=1)
    n1 = torch.nn.functional.normalize(torch.cross(pa - pb, pd - pb, dim=1), dim=1)
    mean = torch.nn.functional.normalize(n0 + n1, dim=1)
    first_face = torch.stack([pc, pa, pd], dim=1)
    second_face = torch.stack([pd, pb, pc], dim=1)
    floor = math.sin(math.radians(MIN_NEW_ANGLE))
    for triangle in (first_face, second_face):
        normal = _normals(triangle)
        length = normal.norm(dim=1)
        ok &= ((normal * mean).sum(1) > FLIP_NORMAL_COSINE * length) & (_smallest_angle_sine(triangle) > floor)
    alive = torch.nonzero(ok).squeeze(1)
    if alive.numel() == 0:
        return 0
    jitter = torch.rand(alive.shape[0], generator=generator).to(device) * 1e-3
    rank = _ranks(-(gain[alive].double() + jitter))
    owner = torch.arange(alive.shape[0], device=device).repeat(4)
    items = torch.cat([a[alive], b[alive], c[alive], d[alive]])
    won = alive[_winners(rank, owner, items, count)]
    if won.numel() == 0:
        return 0
    f0, f1, k0, k1 = f0[won], f1[won], k0[won], k1[won]
    a, b, c, d = a[won], b[won], c[won], d[won]
    sides = work.side_chain
    blank = torch.full_like(a, -1)
    first_sides = torch.stack([sides[f0, (k0 + 2) % 3], sides[f1, (k1 + 1) % 3], blank], 1)
    second_sides = torch.stack([sides[f1, (k1 + 2) % 3], sides[f0, (k0 + 1) % 3], blank], 1)
    work.faces = faces.clone()
    work.faces[f0] = torch.stack([c, a, d], 1)
    work.faces[f1] = torch.stack([d, b, c], 1)
    work.side_chain = sides.clone()
    work.side_chain[f0] = first_sides
    work.side_chain[f1] = second_sides
    work.counts["flips"] += int(won.shape[0])
    return int(won.shape[0])


def _rounds(step, work: _Work, generator: torch.Generator) -> None:
    """Run rounds of an operation until one does little."""
    first = 0
    for index in range(SUB_ROUNDS):
        done = step(work, generator)
        first = first or done
        if done == 0 or (index and done < ROUND_FLOOR * first):
            break


def _coarsen(work: _Work, wanted: int, generator: torch.Generator) -> int:
    """Collapse short edges of a source far denser than the target until the collapses run out.

    Returns:
        The number of rounds run.
    """
    rounds = 0
    while rounds < MAX_COARSEN_ROUNDS and int(work.faces.shape[0]) > COARSEN_FROM * wanted:
        before = int(work.faces.shape[0])
        _collapse(work, generator, midpoint=False, prefilter=True)
        work.compact()
        rounds += 1
        removed = before - int(work.faces.shape[0])
        if removed < COARSEN_STALL * before:
            _flip(work, generator)
            if removed < COARSEN_FLOOR * before:
                break
    return rounds


def _relax(work: _Work) -> None:
    """Move free vertices towards their area-weighted centre within the tangent plane, and chain
    vertices towards the middle of their two chain neighbours."""
    topo, edge_chain = work.topology()
    positions, faces = work.positions, work.faces
    corners = positions[faces]
    area = 0.5 * _normals(corners).norm(dim=1)
    centre = corners.mean(1)
    total = torch.zeros_like(positions)
    weight = torch.zeros(work.vertex_count, dtype=positions.dtype, device=positions.device)
    for k in range(3):
        total.index_add_(0, faces[:, k], centre * area[:, None])
        weight.index_add_(0, faces[:, k], area)
    target = total / weight.clamp_min(1e-30)[:, None]
    normal = topology.vertex_normals(positions, faces)
    step = target - positions
    step = step - (step * normal).sum(1, keepdim=True) * normal
    free = (work.kind == FREE) & (weight > 0)
    moved = positions.clone()
    moved[free] = positions[free] + step[free]
    feature = topo.edges[edge_chain >= 0]
    if feature.numel():
        along = torch.zeros_like(positions)
        hits = torch.zeros(work.vertex_count, dtype=positions.dtype, device=positions.device)
        along.index_add_(0, feature[:, 0], positions[feature[:, 1]])
        along.index_add_(0, feature[:, 1], positions[feature[:, 0]])
        hits.index_add_(0, feature.reshape(-1), torch.ones(feature.numel(), dtype=positions.dtype,
                                                           device=positions.device))
        chain = (work.kind == CHAIN) & (hits == 2)
        moved[chain] = along[chain] / 2.0
    work.positions = moved


def _smooth_step(work: _Work, source: Source, adaptive: bool) -> None:
    """Relax and project, then put back the vertices of any face the move turned over."""
    before = {name: getattr(work, name).clone() for name in ("positions", "face_hint", "segment_hint", "size")}
    old_normal = torch.nn.functional.normalize(_normals(work.positions[work.faces]), dim=1)
    _relax(work)
    _project(work, source, adaptive)
    for _ in range(FOLD_PASSES):
        new_normal = torch.nn.functional.normalize(_normals(work.positions[work.faces]), dim=1)
        turned = (new_normal * old_normal).sum(1) < FOLD_COSINE
        if not bool(turned.any()):
            break
        back = torch.unique(work.faces[turned].reshape(-1))
        for name, value in before.items():
            getattr(work, name)[back] = value[back]
        work.counts["held_back"] += int(back.numel())


def _retarget(work: _Work, wanted: int) -> None:
    """Scale the target lengths so the next pass splits or collapses about as many edges as the
    face count is away from ``wanted``."""
    count = int(work.faces.shape[0])
    gap = count - int(wanted)
    if abs(gap) <= FACE_TOLERANCE * wanted:
        return
    topo, edge_chain = work.topology()
    a, b = topo.edges[:, 0], topo.edges[:, 1]
    ratio = (work.positions[a] - work.positions[b]).norm(dim=1) / (0.5 * (work.size[a] + work.size[b]))
    if gap > 0:
        feature = edge_chain >= 0
        movable = _may_collapse(work, a, b, feature, edge_chain) | _may_collapse(work, b, a, feature, edge_chain)
        ratio = ratio[movable & (topo.edge_face_count <= 2)]
    edges = int(ratio.shape[0])
    if edges < 2:
        return
    needed = min(max(abs(gap) // 2, 1), edges - 1)
    if gap > 0:
        factor = float(torch.kthvalue(ratio, needed).values) * COLLAPSE_REACH
        factor = min(max(factor, math.sqrt(count / wanted), 1.0), MAX_RETARGET)
    else:
        factor = float(torch.kthvalue(ratio, edges - needed).values) / SPLIT_RATIO * 0.999
        factor = max(min(factor, math.sqrt(count / wanted), 1.0), 1.0 / MAX_RETARGET)
    work.stretch(factor)


def _vertex_region(work: _Work) -> torch.Tensor:
    """Region of a face at each vertex."""
    region = torch.zeros(work.vertex_count, dtype=torch.int64, device=work.faces.device)
    region[work.faces.reshape(-1)] = work.region.repeat_interleave(3)
    return region


def _land_faces(source: Source, points, hint, region):
    """Closest triangle in each point's region: a walk from the hint, the tree where it fails.

    Returns:
        ``(face, barycentric, closest)`` with ``closest`` in float64.
    """
    face, weights, settled = _walk(points, hint, region, source.ring_faces, source.faces, source.region,
                                   source.face_offsets, source.face_members, source.vertices)
    lost = torch.nonzero(~settled).squeeze(1)
    if lost.numel():
        tree, offsets = source.tree("region")
        offset = offsets[region[lost]]
        _, found, found_weights, _ = tree.closest_point(points[lost] + offset)
        hit = found >= 0
        face[lost[hit]] = found[hit]
        weights[lost[hit]] = found_weights[hit].to(weights.dtype)
    closest = (source.vertices[source.faces[face]] * weights[:, :, None]).sum(1)
    return face, weights, closest


def _project(work: _Work, source: Source, adaptive: bool) -> None:
    """Move free vertices onto their region of the source surface and chain vertices onto their
    chain, refreshing the target length where it varies."""
    positions = work.positions
    free = torch.nonzero(work.kind == FREE).squeeze(1)
    if free.numel():
        region = _vertex_region(work)[free]
        face, weights, closest = _land_faces(source, positions[free].double(), work.face_hint[free], region)
        positions[free] = closest.float()
        work.face_hint[free] = face
        if adaptive:
            work.size[free] = (source.sizing[source.faces[face]] * weights.float()).sum(1) * work.scale
    chained = torch.nonzero(work.kind == CHAIN).squeeze(1)
    if chained.numel() and source.segments.shape[0]:
        points = positions[chained].double()
        label = work.chain[chained]
        segment, weights, settled = _walk(points, work.segment_hint[chained], label, source.segments,
                                          source.segments, source.segment_chain, source.segment_offsets, source.segment_members,
                                          source.vertices)
        lost = torch.nonzero(~settled).squeeze(1)
        if lost.numel():
            tree, offsets = source.tree("chain")
            offset = offsets[label[lost]]
            _, found, found_weights, _ = tree.closest_point(points[lost] + offset)
            hit = found >= 0
            segment[lost[hit]] = found[hit]
            weights[lost[hit]] = found_weights[hit].to(weights.dtype)
        closest = (source.vertices[source.segments[segment]] * weights[:, :, None]).sum(1)
        positions[chained] = closest.float()
        work.segment_hint[chained] = segment
        work.face_hint[chained] = source.segment_face[segment]
        if adaptive:
            ends = source.segments[segment]
            share = weights.float()
            work.size[chained] = (source.sizing[ends[:, 0]] * share[:, 0]
                                  + source.sizing[ends[:, 1]] * (share[:, 1] + share[:, 2])) * work.scale
    work.positions = positions


# ---------------------------------------------------------------------------------------------
# Output


def _sample(work: _Work, source: Source, cut_seams: bool, feature_angle: float):
    """Split the welded result into fans and land each on the source.

    Returns:
        ``(vertices, faces, source_face, barycentric)`` for the split result.
    """
    topo, edge_chain = work.topology()
    device = work.faces.device
    faces = work.faces
    cut = torch.zeros(topo.edge_count, dtype=torch.bool, device=device)
    if cut_seams and source.chain_count:
        cut = (edge_chain >= 0) & source.chain_seam[edge_chain.clamp_min(0)]
    elif not cut_seams and feature_angle < 180.0:
        cut = topology.feature_edges(work.positions, faces, feature_angle, topo, include_boundary=False,
                                     include_non_manifold=False)
    marked = replace(topo, edge_face_count=torch.where(cut, torch.full_like(topo.edge_face_count, 3),
                                                       topo.edge_face_count))
    fans = topology.corner_fans(faces, work.vertex_count, marked)
    labels, fan_index = torch.unique(fans, return_inverse=True)
    split_faces = fan_index.reshape(-1, 3)
    split_count = int(labels.shape[0])
    welded_of = torch.zeros(split_count, dtype=torch.int64, device=device)
    welded_of[fan_index] = faces.reshape(-1)
    corner_face = torch.zeros(split_count, dtype=torch.int64, device=device)
    corner_face[fan_index] = torch.arange(faces.shape[0], device=device).repeat_interleave(3)
    copies = torch.bincount(welded_of, minlength=work.vertex_count)
    positions = work.positions.double()
    point = positions[welded_of]
    centre = positions[faces[corner_face]].mean(1)
    several = copies[welded_of] > 1
    query = torch.where(several[:, None], point + 0.25 * (centre - point), point)
    hint = torch.where(several, work.face_source[corner_face], work.face_hint[welded_of])
    face, weights, _ = _land_faces(source, query, hint, work.region[corner_face])
    rows = torch.nonzero(several).squeeze(1)
    if rows.numel():
        triangle = source.vertices[source.faces[face[rows]]]
        _, exact = closest_point_on_triangles(point[rows], triangle[:, 0], triangle[:, 1], triangle[:, 2])
        weights[rows] = exact.to(weights.dtype)
    if source.turned is not None:
        weights = torch.where(source.turned[face][:, None], weights[:, [0, 2, 1]], weights)
    return work.positions[welded_of], split_faces, source.face_index[face], weights.float()


def _run(mesh: TriMesh, length: float | None, faces: int | None, adaptivity: float, iterations: int,
         feature_angle: float, keep_seams: bool, max_faces: int, device, progress=None) -> RemeshResult:
    """The remesh on one device."""
    started = time.perf_counter()
    adaptive = adaptivity > 0
    source = _source(mesh, adaptivity, feature_angle, keep_seams, device, length, faces)
    areas = topology.face_areas(source.vertices, source.faces)
    area = source.area
    uniform = source.length
    expected = estimate_faces(area, uniform)
    if expected > max_faces:
        raise ValueError(
            f"a remesh at an edge length of {uniform:.4g} would make about {expected} faces, over the "
            f"limit of {max_faces}; raise edge_length or lower faces"
        )
    stretch = 1.0
    if adaptive:
        corner_sizing = source.sizing[source.faces].double().mean(1)
        stretch = math.sqrt(max(float((areas / corner_sizing.pow(2)).sum()), 1e-30) / max(area, 1e-30))
    work = _Work(source, uniform * stretch)
    generator = torch.Generator().manual_seed(0)
    budget = max(4 * max(expected, 1), int(work.faces.shape[0])) * 2
    coarsened = _coarsen(work, expected, generator) if int(iterations) > 0 else 0
    for index in range(int(iterations)):
        if progress is not None:
            progress(index / max(int(iterations), 1))
        for _ in range(SPLIT_PASSES):
            if _split(work, budget) == 0:
                break
        _rounds(_collapse, work, generator)
        work.compact()
        _rounds(_flip, work, generator)
        _smooth_step(work, source, adaptive)
        work.refresh_sources()
        if faces is not None and index < int(iterations) - SETTLE_PASSES:
            _retarget(work, int(faces))
    vertices, split_faces, face, weights = _sample(work, source, keep_seams, feature_angle)
    stats = {
        "target_length": work.scale / stretch,
        "regions": source.region_count if keep_seams else 0,
        "feature_chains": source.chain_count,
        "held_vertices": int((work.kind == LOCKED).sum()),
        "welded_vertices": work.vertex_count,
        "coarsening_rounds": coarsened,
        **work.counts,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return RemeshResult(TriMesh(vertices.cpu(), split_faces.cpu()), face.cpu(), weights.cpu(), stats)


def remesh(mesh: TriMesh, length: float | None = None, faces: int | None = None, adaptivity: float = 0.0,
           iterations: int = 8, feature_angle: float = 45.0, keep_seams: bool = True,
           max_faces: int = MAX_FACES, device: torch.device | str | None = None,
           progress: Callable[[float], None] | None = None) -> RemeshResult:
    """Remesh a surface towards even triangles of a target size, keeping its features.

    Args:
        mesh: The split source mesh with its per-vertex attributes.
        length: Target edge length in model units; ignored when ``faces`` is given.
        faces: Face count wanted instead of a length; the length is tuned towards it each pass.
        adaptivity: ``0`` uniform, up to ``1`` where lengths follow curvature.
        iterations: Passes of split, collapse, flip, relax and project.
        feature_angle: Dihedral angle in degrees from which an edge is a crease; ``180`` keeps
            only borders and seams.
        keep_seams: Keep attribute seams as edges and sample attributes per side.
        max_faces: Most faces the result may have.
        device: Device to run on; None means the mesh compute device.
        progress: Called with the share of passes done, from 0 to below 1, before each pass.

    Returns:
        The `RemeshResult`. Attributes the mesh carries are sampled onto the result; without
        ``keep_seams`` texture coordinates and tangents are left out and normals recomputed.

    Raises:
        ValueError: The mesh has no triangles, no size was given, or the result would pass
            ``max_faces``.
    """
    if mesh.face_count == 0:
        raise ValueError("the mesh has no triangles to remesh")
    if faces is None and (length is None or not length > 0):
        raise ValueError("give a face count or an edge length above 0 to remesh to")
    iterations = max(0, min(int(iterations), MAX_ITERATIONS))
    if faces is None:
        area = float(topology.face_areas(mesh.vertices.double(), mesh.faces.long()).sum())
        expected = estimate_faces(area, float(length))
        if expected > int(max_faces):
            raise ValueError(
                f"a remesh at an edge length of {float(length):.4g} would make about {expected} faces, over "
                f"the limit of {int(max_faces)}; raise edge_length or remesh fewer parts"
            )
    elif int(faces) > int(max_faces):
        raise ValueError(f"{int(faces)} faces were asked for, over the limit of {int(max_faces)}; lower faces")

    def work(on: torch.device) -> RemeshResult:
        return _run(mesh, length, faces, float(adaptivity), iterations, float(feature_angle),
                    bool(keep_seams), int(max_faces), on, progress)

    result = mesh_device.with_cpu_fallback(work, device, label="Mesh Remesh")
    names = [name for name in mesh.attributes if keep_seams or name not in ("uvs", "tangents")]
    source = TriMesh(mesh.vertices, mesh.faces, {name: mesh.attributes[name] for name in names})
    samples = Samples(result.source_face, result.barycentric,
                      torch.zeros(result.source_face.shape[0]), result.mesh.vertices)
    result.mesh.attributes = interpolate(source, samples, names) if names else {}
    if not keep_seams and "normals" in mesh.attributes:
        result.mesh.attributes["normals"] = topology.vertex_normals(result.mesh.vertices, result.mesh.faces)
    return result


def remesh_part(part: Part, joint_count: int = 0, **options) -> tuple[Part, dict]:
    """Remesh one model part, carrying its attributes, skin and shape keys.

    Args:
        part: The part to remesh.
        joint_count: Skeleton size, 0 for an unrigged model.
        **options: Keyword arguments of `remesh`.

    Returns:
        ``(part, stats)``: a new part and the run's counts. Without ``keep_seams`` a textured
        material is replaced by a copy without its textures, since the texture coordinates go.

    Raises:
        ValueError: As `remesh` does.
    """
    mesh = from_part(part, 0)
    result = remesh(mesh, **options)
    new_part = to_part(part, result.mesh)
    if joint_count and part.joints is not None and part.weights is not None:
        corners = mesh.faces[result.source_face]
        count = result.mesh.vertex_count
        rows = torch.arange(count).repeat_interleave(3)
        joints, weights = blend_skin(part.joints, part.weights, rows, corners.reshape(-1),
                                     result.barycentric.reshape(-1), count, joint_count)
        new_part = replace(new_part, joints=joints, weights=weights)
    if not options.get("keep_seams", True) and part.uvs is not None:
        material = replace(part.material, base_color_texture=None, metallic_roughness_texture=None,
                           normal_texture=None, occlusion_texture=None, emissive_texture=None)
        new_part = replace(new_part, material=material)
    return new_part, result.stats
