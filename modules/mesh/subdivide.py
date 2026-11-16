"""Subdivision of triangle and polygon meshes: Loop, Catmull-Clark and midpoint.

Each level is a sparse stencil from old vertices to new ones. Positions, skin and shape keys follow
the welded surface; texture coordinates, colours and tangents follow the split mesh.
"""

from __future__ import annotations

__all__ = [
    "LIMIT_FACES",
    "MAX_FACES",
    "SCHEMES",
    "Stencil",
    "SubdivisionResult",
    "apply_stencil",
    "blend_skin",
    "predicted_faces",
    "subdivide",
    "subdivide_part",
]

import math
from dataclasses import dataclass, replace

import torch

from ..model3d.scene import Part, Polygons
from . import device as mesh_device
from . import topology
from .arrays import MORPH_PREFIX, TriMesh, from_part, sparse_skin, to_part

#: Schemes `subdivide` offers.
SCHEMES = ("loop", "catmull_clark", "midpoint")

#: Default ceiling on the triangles a subdivision may produce.
MAX_FACES = 4_000_000

#: Highest ceiling a caller may set.
LIMIT_FACES = 16_000_000

#: Bytes of dense skin weights one blending chunk may hold.
SKIN_CHUNK_BYTES = 64 << 20

#: Attributes that follow the welded surface, so copies of one vertex never part.
WELDED_ATTRIBUTES = ("skin",)


@dataclass
class Stencil:
    """One subdivision level: new vertices as weighted sums of old ones, and the new faces.

    Attributes:
        rows: ``(K,)`` int64 new vertex of each weight.
        cols: ``(K,)`` int64 old vertex of each weight.
        values: ``(K,)`` float weights.
        count: Number of new vertices.
        sizes: ``(P,)`` int64 corner count of each new polygon.
        indices: ``(sizes.sum(),)`` int64 new vertex of each polygon corner.
        edges: ``(E, 2)`` int64 old edges, whose points follow the old vertices in new numbering.
        faces_first: Row of the first face point in new numbering, ``count`` without any.
    """

    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    count: int
    sizes: torch.Tensor
    indices: torch.Tensor
    edges: torch.Tensor
    faces_first: int


def _group_rank(keys_sorted: torch.Tensor) -> torch.Tensor:
    """Position of each element within its run of equal keys."""
    count = keys_sorted.shape[0]
    index = torch.arange(count, device=keys_sorted.device)
    start = torch.ones(count, dtype=torch.bool, device=keys_sorted.device)
    start[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first = torch.cummax(torch.where(start, index, torch.zeros_like(index)), 0).values
    return index - first


def _ranges(offsets: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand CSR rows into ``(owner, position)`` pairs, one per entry of each row."""
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    owner = torch.repeat_interleave(torch.arange(rows.shape[0], device=rows.device), counts)
    before = torch.cumsum(counts, 0) - counts
    within = torch.arange(int(counts.sum()), device=rows.device) - before[owner]
    return owner, starts[owner] + within


class _Polygons:
    """Corner and edge tables of a polygon mesh."""

    def __init__(self, sizes: torch.Tensor, indices: torch.Tensor, vertex_count: int):
        device = indices.device
        self.sizes = sizes.long()
        self.indices = indices.long()
        self.count = int(sizes.shape[0])
        self.vertex_count = int(vertex_count)
        self.offsets = torch.zeros(self.count + 1, dtype=torch.int64, device=device)
        self.offsets[1:] = torch.cumsum(self.sizes, 0)
        corners = int(self.indices.shape[0])
        self.polygon = torch.repeat_interleave(torch.arange(self.count, device=device), self.sizes)
        local = torch.arange(corners, device=device) - self.offsets[self.polygon]
        size = self.sizes[self.polygon]
        self.next = self.offsets[self.polygon] + (local + 1) % size
        self.previous = self.offsets[self.polygon] + (local - 1) % size
        start, end = self.indices, self.indices[self.next]
        stride = max(self.vertex_count, 1)
        keys = torch.minimum(start, end) * stride + torch.maximum(start, end)
        unique, self.edge_of, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        self.edges = torch.stack([unique // stride, unique % stride], dim=1)
        self.edge_faces = counts
        order = torch.argsort(self.edge_of, stable=True)
        rank = _group_rank(self.edge_of[order])
        self.edge_corner = torch.full((int(unique.shape[0]), 2), -1, dtype=torch.int64, device=device)
        for slot in range(2):
            chosen = order[rank == slot]
            self.edge_corner[self.edge_of[chosen], slot] = chosen

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        return int(self.edges.shape[0])

    def valence(self) -> torch.Tensor:
        """Edges at each vertex."""
        return torch.bincount(self.edges.reshape(-1), minlength=self.vertex_count)

    def corner_vertices(self, crease: torch.Tensor) -> torch.Tensor:
        """Vertices a subdivision holds still: on three or more creases, or not a manifold fan."""
        creases = torch.bincount(self.edges[crease].reshape(-1), minlength=self.vertex_count)
        held = creases >= 3
        held[self.edges[self.edge_faces > 2].reshape(-1)] = True
        faces_at = torch.bincount(self.indices, minlength=self.vertex_count)
        border = torch.bincount(self.edges[self.edge_faces == 1].reshape(-1), minlength=self.vertex_count)
        held |= (border == 0) & (faces_at != self.valence()) & (faces_at > 0)
        return held


def _vertex_rules(mesh: _Polygons, crease: torch.Tensor, corner: torch.Tensor):
    """Masks for the smooth, creased and held vertex rules, and the crease neighbour sums."""
    creases = torch.bincount(mesh.edges[crease].reshape(-1), minlength=mesh.vertex_count)
    held = corner | (creases >= 3)
    creased = (creases == 2) & ~held
    smooth = (creases <= 1) & ~held
    return smooth, creased, held


def _crease_rows(mesh: _Polygons, crease: torch.Tensor, creased: torch.Tensor):
    """Weights of the crease rule: three quarters of the vertex, an eighth of each crease neighbour."""
    lines = mesh.edges[crease]
    rows = torch.cat([lines[:, 0], lines[:, 1]])
    cols = torch.cat([lines[:, 1], lines[:, 0]])
    keep = creased[rows]
    centre = torch.nonzero(creased).squeeze(1)
    dtype = torch.float64
    return (torch.cat([rows[keep], centre]), torch.cat([cols[keep], centre]),
            torch.cat([torch.full((int(keep.sum()),), 0.125, dtype=dtype, device=rows.device),
                       torch.full((centre.shape[0],), 0.75, dtype=dtype, device=rows.device)]))


def _loop(mesh: _Polygons, crease: torch.Tensor, corner: torch.Tensor) -> Stencil:
    """The Loop stencil of a triangle mesh."""
    device = mesh.indices.device
    dtype = torch.float64
    vertices, edge_total = mesh.vertex_count, mesh.edge_count
    a, b = mesh.edges[:, 0], mesh.edges[:, 1]
    sharp = crease | (mesh.edge_faces != 2)
    edge_row = vertices + torch.arange(edge_total, device=device)
    half = torch.where(sharp, 0.5, 0.375).to(dtype)
    rows = [edge_row, edge_row]
    cols = [a, b]
    values = [half, half]
    smooth_edges = torch.nonzero(~sharp).squeeze(1)
    for slot in range(2):
        opposite = mesh.indices[mesh.previous[mesh.edge_corner[smooth_edges, slot]]]
        rows.append(edge_row[smooth_edges])
        cols.append(opposite)
        values.append(torch.full((smooth_edges.shape[0],), 0.125, dtype=dtype, device=device))
    smooth, creased, held = _vertex_rules(mesh, crease, corner)
    valence = mesh.valence().to(dtype)
    angle = 2.0 * math.pi / valence.clamp_min(1)
    beta = (0.625 - (0.375 + 0.25 * torch.cos(angle)) ** 2) / valence.clamp_min(1)
    around_rows = torch.cat([a, b])
    around_cols = torch.cat([b, a])
    keep = smooth[around_rows]
    rows += [around_rows[keep]]
    cols += [around_cols[keep]]
    values += [beta[around_rows[keep]]]
    centre = torch.nonzero(smooth).squeeze(1)
    rows.append(centre)
    cols.append(centre)
    values.append(1.0 - valence[centre] * beta[centre])
    crease_rows, crease_cols, crease_values = _crease_rows(mesh, crease, creased)
    still = torch.nonzero(held).squeeze(1)
    rows += [crease_rows, still]
    cols += [crease_cols, still]
    values += [crease_values, torch.ones(still.shape[0], dtype=dtype, device=device)]
    return Stencil(torch.cat(rows), torch.cat(cols), torch.cat(values), vertices + edge_total,
                   *_split_triangles(mesh), mesh.edges, vertices + edge_total)


def _midpoint_triangles(mesh: _Polygons) -> Stencil:
    """The midpoint stencil of a triangle mesh: every triangle into four, shape unchanged."""
    device = mesh.indices.device
    dtype = torch.float64
    vertices, edge_total = mesh.vertex_count, mesh.edge_count
    edge_row = vertices + torch.arange(edge_total, device=device)
    every = torch.arange(vertices, device=device)
    half = torch.full((edge_total,), 0.5, dtype=dtype, device=device)
    rows = torch.cat([every, edge_row, edge_row])
    cols = torch.cat([every, mesh.edges[:, 0], mesh.edges[:, 1]])
    values = torch.cat([torch.ones(vertices, dtype=dtype, device=device), half, half])
    return Stencil(rows, cols, values, vertices + edge_total, *_split_triangles(mesh), mesh.edges,
                   vertices + edge_total)


def _split_triangles(mesh: _Polygons) -> tuple[torch.Tensor, torch.Tensor]:
    """Each triangle into four on its edge points: ``(sizes, indices)``."""
    corners = mesh.indices.reshape(-1, 3)
    mid = (mesh.vertex_count + mesh.edge_of).reshape(-1, 3)
    faces = torch.stack([
        torch.stack([corners[:, 0], mid[:, 0], mid[:, 2]], 1),
        torch.stack([mid[:, 0], corners[:, 1], mid[:, 1]], 1),
        torch.stack([mid[:, 2], mid[:, 1], corners[:, 2]], 1),
        torch.stack([mid[:, 0], mid[:, 1], mid[:, 2]], 1),
    ], dim=1).reshape(-1)
    sizes = torch.full((corners.shape[0] * 4,), 3, dtype=torch.int64, device=corners.device)
    return sizes, faces


def _quads(mesh: _Polygons) -> tuple[torch.Tensor, torch.Tensor]:
    """Each polygon into one quad per corner around its face point: ``(sizes, indices)``."""
    device = mesh.indices.device
    corner = torch.arange(mesh.indices.shape[0], device=device)
    first_face = mesh.vertex_count + mesh.edge_count
    quads = torch.stack([
        mesh.indices,
        mesh.vertex_count + mesh.edge_of[corner],
        first_face + mesh.polygon,
        mesh.vertex_count + mesh.edge_of[mesh.previous],
    ], dim=1).reshape(-1)
    return torch.full((mesh.indices.shape[0],), 4, dtype=torch.int64, device=device), quads


def _face_points(mesh: _Polygons):
    """Weights of each polygon's face point: an equal share of every corner."""
    dtype = torch.float64
    first_face = mesh.vertex_count + mesh.edge_count
    rows = first_face + mesh.polygon
    values = 1.0 / mesh.sizes[mesh.polygon].to(dtype)
    return rows, mesh.indices, values


def _catmull_clark(mesh: _Polygons, crease: torch.Tensor, corner: torch.Tensor, linear: bool) -> Stencil:
    """The Catmull-Clark stencil of a polygon mesh, or its linear form when ``linear``."""
    device = mesh.indices.device
    dtype = torch.float64
    vertices, edge_total = mesh.vertex_count, mesh.edge_count
    first_face = vertices + edge_total
    count = first_face + mesh.count
    a, b = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_row = vertices + torch.arange(edge_total, device=device)
    face_rows, face_cols, face_values = _face_points(mesh)
    rows, cols, values = [face_rows], [face_cols], [face_values]
    if linear:
        every = torch.arange(vertices, device=device)
        half = torch.full((edge_total,), 0.5, dtype=dtype, device=device)
        rows += [every, edge_row, edge_row]
        cols += [every, a, b]
        values += [torch.ones(vertices, dtype=dtype, device=device), half, half]
    else:
        sharp = crease | (mesh.edge_faces != 2)
        half = torch.where(sharp, 0.5, 0.25).to(dtype)
        rows += [edge_row, edge_row]
        cols += [a, b]
        values += [half, half]
        smooth_edges = torch.nonzero(~sharp).squeeze(1)
        for slot in range(2):
            polygon = mesh.polygon[mesh.edge_corner[smooth_edges, slot]]
            owner, position = _ranges(mesh.offsets, polygon)
            rows.append(edge_row[smooth_edges][owner])
            cols.append(mesh.indices[position])
            values.append(0.25 / mesh.sizes[polygon][owner].to(dtype))
        smooth, creased, held = _vertex_rules(mesh, crease, corner)
        valence = mesh.valence().to(dtype).clamp_min(1)
        centre = torch.nonzero(smooth).squeeze(1)
        rows.append(centre)
        cols.append(centre)
        values.append((valence[centre] - 2.0) / valence[centre])
        around_rows = torch.cat([a, b])
        around_cols = torch.cat([b, a])
        keep = smooth[around_rows]
        rows.append(around_rows[keep])
        cols.append(around_cols[keep])
        values.append(1.0 / valence[around_rows[keep]] ** 2)
        touching = torch.nonzero(smooth[mesh.indices]).squeeze(1)
        vertex = mesh.indices[touching]
        polygon = mesh.polygon[touching]
        owner, position = _ranges(mesh.offsets, polygon)
        rows.append(vertex[owner])
        cols.append(mesh.indices[position])
        values.append(1.0 / (valence[vertex[owner]] ** 2 * mesh.sizes[polygon][owner].to(dtype)))
        crease_rows, crease_cols, crease_values = _crease_rows(mesh, crease, creased)
        still = torch.nonzero(held).squeeze(1)
        rows += [crease_rows, still]
        cols += [crease_cols, still]
        values += [crease_values, torch.ones(still.shape[0], dtype=dtype, device=device)]
    return Stencil(torch.cat(rows), torch.cat(cols), torch.cat(values), count, *_quads(mesh), mesh.edges,
                   first_face)


def apply_stencil(stencil: Stencil, values: torch.Tensor) -> torch.Tensor:
    """New per-vertex values from old ones.

    Args:
        stencil: The level's stencil.
        values: ``(old, K)`` or ``(old,)`` values.

    Returns:
        ``(stencil.count, K)`` or ``(stencil.count,)`` values in the input's dtype.
    """
    single = values.dim() == 1
    data = values[:, None] if single else values
    work = data.to(torch.float64) if data.dtype != torch.float64 else data
    out = torch.zeros(stencil.count, data.shape[1], dtype=torch.float64, device=data.device)
    step = max(1, (64 << 20) // max(8 * data.shape[1], 1))
    for start in range(0, int(stencil.rows.shape[0]), step):
        stop = start + step
        out.index_add_(0, stencil.rows[start:stop],
                       stencil.values[start:stop, None] * work[stencil.cols[start:stop]])
    out = out.to(values.dtype)
    return out[:, 0] if single else out


def blend_skin(joints: torch.Tensor, weights: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
               values: torch.Tensor, count: int, joint_count: int,
               influences: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Skin of new vertices made as weighted sums of old vertices' skin.

    Args:
        joints: ``(N, K)`` int64 joint indices of the old vertices.
        weights: ``(N, K)`` float weights of the old vertices.
        rows: ``(S,)`` new vertex of each term.
        cols: ``(S,)`` old vertex of each term.
        values: ``(S,)`` weight of each term.
        count: Number of new vertices.
        joint_count: Number of joints in the skeleton.
        influences: Influences kept per new vertex.

    Returns:
        ``(joints, weights)``: ``(count, influences)`` int64 and float32, rows summing to 1.
    """
    device = joints.device
    rows, cols, values = rows.to(device).long(), cols.to(device).long(), values.to(device).float()
    order = torch.argsort(rows, stable=True)
    rows, cols, values = rows[order], cols[order], values[order]
    joint_count = max(int(joint_count), int(joints.max()) + 1 if joints.numel() else 1)
    out_joints = torch.zeros(count, influences, dtype=torch.int64, device=device)
    out_weights = torch.zeros(count, influences, dtype=torch.float32, device=device)
    chunk = max(1, SKIN_CHUNK_BYTES // (4 * joint_count))
    width = joints.shape[1]
    bounds = torch.searchsorted(rows, torch.arange(0, count + chunk, chunk, device=device))
    for index, start in enumerate(range(0, count, chunk)):
        stop = min(start + chunk, count)
        first, last = int(bounds[index]), int(bounds[index + 1])
        dense = torch.zeros(stop - start, joint_count, dtype=torch.float32, device=device)
        local = (rows[first:last] - start)[:, None].expand(-1, width)
        source = cols[first:last]
        dense.index_put_((local.reshape(-1), joints[source].long().reshape(-1)),
                         (values[first:last, None] * weights[source].float()).reshape(-1), accumulate=True)
        out_joints[start:stop], out_weights[start:stop] = sparse_skin(dense, influences)
    return out_joints, out_weights


def _base(mesh: TriMesh, scheme: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The polygons a scheme starts from: ``(sizes, indices)`` on the split vertices."""
    if scheme != "loop" and mesh.polygons is not None and mesh.polygons.count:
        return mesh.polygons.sizes.long(), mesh.polygons.indices.long()
    faces = mesh.faces.long()
    return torch.full((faces.shape[0],), 3, dtype=torch.int64, device=faces.device), faces.reshape(-1)


def predicted_faces(mesh: TriMesh, scheme: str, levels: int) -> int:
    """Triangles a subdivision would produce.

    Args:
        mesh: The mesh to subdivide.
        scheme: One of `SCHEMES`.
        levels: Subdivision levels.

    Returns:
        The triangle count.
    """
    sizes, _ = _base(mesh, scheme)
    if levels <= 0:
        return mesh.face_count
    if scheme == "loop" or (scheme == "midpoint" and bool((sizes == 3).all())):
        return mesh.face_count * 4 ** int(levels)
    return 2 * int(sizes.sum()) * 4 ** (int(levels) - 1)


def _linear_stencil(mesh: _Polygons, scheme: str) -> Stencil:
    """The stencil that places a scheme's new vertices on the old faces, numbered as the scheme numbers them."""
    if scheme == "loop" or (scheme == "midpoint" and bool((mesh.sizes == 3).all())):
        return _midpoint_triangles(mesh)
    empty = torch.zeros(mesh.edge_count, dtype=torch.bool, device=mesh.indices.device)
    return _catmull_clark(mesh, empty, torch.zeros(mesh.vertex_count, dtype=torch.bool, device=empty.device),
                          linear=True)


def _stencil(mesh: _Polygons, scheme: str, crease: torch.Tensor, corner: torch.Tensor) -> Stencil:
    """The stencil of one level."""
    triangles = bool((mesh.sizes == 3).all())
    if scheme == "loop":
        return _loop(mesh, crease, corner)
    if scheme == "midpoint" and triangles:
        return _midpoint_triangles(mesh)
    return _catmull_clark(mesh, crease, corner, linear=scheme == "midpoint")


def _lookup(keys: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position of each query in sorted keys and whether it is there."""
    if keys.numel() == 0:
        return torch.zeros_like(query), torch.zeros_like(query, dtype=torch.bool)
    at = torch.searchsorted(keys, query).clamp_max(keys.shape[0] - 1)
    return at, keys[at] == query


def _edge_keys(edges: torch.Tensor, stride: int) -> torch.Tensor:
    """Key of each undirected edge."""
    return torch.minimum(edges[:, 0], edges[:, 1]) * stride + torch.maximum(edges[:, 0], edges[:, 1])


def _triangulate(sizes: torch.Tensor, indices: torch.Tensor, positions: torch.Tensor):
    """Triangles of quads and triangles, quads cut along their shorter diagonal.

    Returns:
        ``(faces, triangle_polygon)``.
    """
    device = indices.device
    offsets = torch.zeros(sizes.shape[0] + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(sizes, 0)
    triangles = torch.nonzero(sizes == 3).squeeze(1)
    quads = torch.nonzero(sizes == 4).squeeze(1)
    tri = indices[offsets[triangles][:, None] + torch.arange(3, device=device)]
    quad = indices[offsets[quads][:, None] + torch.arange(4, device=device)]
    across = ((positions[quad[:, 0]] - positions[quad[:, 2]]).norm(dim=1)
              <= (positions[quad[:, 1]] - positions[quad[:, 3]]).norm(dim=1))
    first = torch.where(across[:, None], quad[:, [0, 1, 2]], quad[:, [0, 1, 3]])
    second = torch.where(across[:, None], quad[:, [0, 2, 3]], quad[:, [1, 2, 3]])
    faces = torch.cat([tri, torch.stack([first, second], 1).reshape(-1, 3)])
    owner = torch.cat([triangles, quads.repeat_interleave(2)])
    order = torch.argsort(owner, stable=True)
    return faces[order], owner[order]


def _level(split: _Polygons, remap: torch.Tensor, welded: _Polygons, scheme: str,
           welded_crease: torch.Tensor, welded_corner: torch.Tensor):
    """One level on the welded and the split mesh together.

    Returns:
        ``(welded_stencil, split_stencil, new_remap, new_crease_keys, new_corner)``.
    """
    device = remap.device
    stride = welded.vertex_count
    crease_keys = torch.sort(_edge_keys(welded.edges[welded_crease], stride)).values
    split_welded = remap[split.edges]
    _, split_feature = _lookup(crease_keys, _edge_keys(split_welded, stride))
    split_crease = split_feature | (split.edge_faces != 2)
    split_corner = split.corner_vertices(split_crease) | welded_corner[remap]
    welded_stencil = _stencil(welded, scheme, welded_crease, welded_corner | welded.corner_vertices(welded_crease))
    split_stencil = _stencil(split, scheme, split_crease, split_corner)
    edge_keys = _edge_keys(welded.edges, stride)
    at, _ = _lookup(edge_keys, _edge_keys(split_welded, stride))
    new_remap = torch.cat([
        remap,
        welded.vertex_count + at,
        welded_stencil.faces_first + torch.arange(split.count, device=device)
        if split_stencil.faces_first < split_stencil.count else torch.zeros(0, dtype=torch.int64, device=device),
    ])
    edge_point = welded.vertex_count + torch.nonzero(welded_crease).squeeze(1)
    ends = welded.edges[welded_crease]
    new_stride = welded_stencil.count
    halves = torch.cat([torch.stack([ends[:, 0], edge_point], 1), torch.stack([edge_point, ends[:, 1]], 1)])
    new_crease_keys = torch.sort(_edge_keys(halves, new_stride)).values
    new_corner = torch.cat([welded_corner | welded.corner_vertices(welded_crease),
                            torch.zeros(welded_stencil.count - welded.vertex_count, dtype=torch.bool,
                                        device=device)])
    return welded_stencil, split_stencil, new_remap, new_crease_keys, new_corner


@dataclass
class SubdivisionResult:
    """A subdivided mesh and its skin.

    Attributes:
        mesh: The new mesh with its attributes; quads as polygons from Catmull-Clark, and from
            midpoint on quads.
        welded_of: ``(N,)`` int64 welded vertex each new vertex sits on.
        joints: ``(N, 4)`` int64 skin joints, or None.
        weights: ``(N, 4)`` float32 skin weights, or None.
    """

    mesh: TriMesh
    welded_of: torch.Tensor
    joints: torch.Tensor | None = None
    weights: torch.Tensor | None = None


def _run(mesh: TriMesh, scheme: str, levels: int, feature_angle: float, skin, smooth_uvs: bool,
         device) -> SubdivisionResult:
    """Subdivision on one device."""
    vertices = mesh.vertices.to(device).double()
    faces = mesh.faces.to(device).long()
    diagonal = float((vertices.amax(0) - vertices.amin(0)).norm()) if vertices.shape[0] else 0.0
    welded_positions, welded_faces, remap = topology.weld(vertices, faces, tolerance=1e-7 * diagonal)
    sizes, indices = _base(mesh, scheme)
    sizes, indices = sizes.to(device), indices.to(device)
    split = _Polygons(sizes, indices, vertices.shape[0])
    welded = _Polygons(sizes, remap[indices], welded_positions.shape[0])
    topo = topology.edge_topology(welded_faces, welded_positions.shape[0])
    if feature_angle < 180.0:
        sharp = topology.feature_edges(welded_positions, welded_faces, feature_angle, topo)
        sharp_keys = torch.sort(_edge_keys(topo.edges[sharp], welded.vertex_count)).values
    else:
        sharp_keys = torch.zeros(0, dtype=torch.int64, device=device)
    _, crease = _lookup(sharp_keys, _edge_keys(welded.edges, welded.vertex_count))
    crease = crease | (welded.edge_faces != 2)
    corner = topology.non_manifold_vertices(welded_faces, welded.vertex_count, topo)
    attributes = {name: value.to(device) for name, value in mesh.attributes.items()}
    welded_names = [name for name in attributes if name in WELDED_ATTRIBUTES or name.startswith(MORPH_PREFIX)]
    first_copy = torch.full((welded.vertex_count,), vertices.shape[0], dtype=torch.int64, device=device)
    first_copy.scatter_reduce_(0, remap, torch.arange(vertices.shape[0], device=device), reduce="amin")
    welded_values = {name: attributes[name][first_copy] for name in welded_names}
    split_values = {name: value for name, value in attributes.items() if name not in welded_names}
    linear_names = [] if smooth_uvs or scheme == "midpoint" else [name for name in split_values if name == "uvs"]
    joints = weights = None
    if skin is not None:
        joints, weights = skin[0].to(device)[first_copy], skin[1].to(device)[first_copy]
    positions = welded_positions
    for _ in range(int(levels)):
        welded_stencil, split_stencil, remap, crease_keys, corner = _level(
            split, remap, welded, scheme, crease, corner)
        linear_stencil = _linear_stencil(split, scheme) if linear_names else split_stencil
        positions = apply_stencil(welded_stencil, positions)
        welded_values = {name: apply_stencil(welded_stencil, value) for name, value in welded_values.items()}
        split_values = {name: apply_stencil(linear_stencil if name in linear_names else split_stencil, value)
                        for name, value in split_values.items()}
        if joints is not None:
            joints, weights = blend_skin(joints, weights, welded_stencil.rows, welded_stencil.cols,
                                         welded_stencil.values, welded_stencil.count, skin[2])
        split = _Polygons(split_stencil.sizes, split_stencil.indices, split_stencil.count)
        welded = _Polygons(split_stencil.sizes, remap[split_stencil.indices], welded_stencil.count)
        _, crease = _lookup(crease_keys, _edge_keys(welded.edges, welded.vertex_count))
        crease = crease | (welded.edge_faces != 2)
    out_positions = positions[remap].float()
    attributes = {name: value[remap] for name, value in welded_values.items()}
    attributes.update(split_values)
    tri_faces, owner = _triangulate(split.sizes, split.indices, out_positions)
    polygons = None
    if levels > 0 and bool((split.sizes == 4).all()):
        polygons = Polygons(split.sizes.cpu(), split.indices.cpu(), owner.cpu())
    result = TriMesh(out_positions.cpu(), tri_faces.cpu(),
                     {name: value.cpu() for name, value in attributes.items()}, polygons)
    if joints is not None:
        joints, weights = joints[remap].cpu(), weights[remap].cpu()
    return SubdivisionResult(result, remap.cpu(), joints, weights)


def subdivide(mesh: TriMesh, scheme: str = "loop", levels: int = 1, feature_angle: float = 70.0,
              max_faces: int = MAX_FACES, device: torch.device | str | None = None,
              skin: tuple[torch.Tensor, torch.Tensor, int] | None = None,
              smooth_uvs: bool = False) -> SubdivisionResult:
    """Subdivide a mesh, carrying its attributes.

    Args:
        mesh: The split mesh with per-vertex attributes; dense ``"skin"`` follows the surface.
        scheme: ``"loop"`` (smooth, triangles), ``"catmull_clark"`` (smooth, quads) or
            ``"midpoint"`` (shape kept).
        levels: Levels to apply, 0 to 3.
        feature_angle: Dihedral angle in degrees from which an edge is a crease; ``180`` creases
            borders only.
        max_faces: Most triangles the result may have.
        device: Device to run on; None means the mesh compute device.
        skin: ``(joints, weights, joint_count)`` sparse skin of the split vertices, or None.
        smooth_uvs: Smooth texture coordinates inside each chart, its border held; False places
            them on the old faces, so the texture stays where it was on every face.

    Returns:
        The `SubdivisionResult`.

    Raises:
        ValueError: For an unknown scheme, no triangles, or a result past ``max_faces``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"the subdivision scheme must be one of {', '.join(SCHEMES)}, not {scheme!r}")
    if mesh.face_count == 0:
        raise ValueError("the mesh has no triangles to subdivide")
    levels = max(0, min(int(levels), 3))
    expected = predicted_faces(mesh, scheme, levels)
    if expected > int(max_faces):
        raise ValueError(
            f"{levels} level(s) of {scheme} subdivision would make {expected} triangles, over the limit "
            f"of {int(max_faces)}; lower levels, raise max_faces, or choose fewer parts"
        )
    return mesh_device.with_cpu_fallback(
        lambda on: _run(mesh, scheme, levels, float(feature_angle), skin, bool(smooth_uvs), on), device,
        label="Mesh Subdivide")


def subdivide_part(part: Part, joint_count: int = 0, scheme: str = "loop", levels: int = 1,
                   feature_angle: float = 70.0, max_faces: int = MAX_FACES, smooth_uvs: bool = False) -> Part:
    """Subdivide one model part, carrying texture coordinates, colours, skin and shape keys.

    Args:
        part: The part to subdivide.
        joint_count: Skeleton size, 0 for an unrigged model.
        scheme: One of `SCHEMES`.
        levels: Levels to apply, 0 to 3.
        feature_angle: Crease angle in degrees.
        max_faces: Most triangles the part may have afterwards.
        smooth_uvs: Smooth texture coordinates inside each chart.

    Returns:
        A new part, normals recomputed and tangents kept at right angles to them; the part
        itself at 0 levels.

    Raises:
        ValueError: As `subdivide` does.
    """
    from .smooth import orthogonal_tangents, recompute_normals

    if levels <= 0:
        return part
    mesh = from_part(part, 0)
    rigged = bool(joint_count) and part.joints is not None and part.weights is not None
    skin = (part.joints, part.weights, int(joint_count)) if rigged else None
    result = subdivide(mesh, scheme, levels, feature_angle, max_faces, skin=skin, smooth_uvs=smooth_uvs)
    new_part = to_part(part, result.mesh)
    if part.normals is not None:
        normals = recompute_normals(result.mesh.vertices, result.mesh.faces, result.welded_of,
                                    result.mesh.attributes.get("normals"))
        new_part = replace(new_part, normals=normals)
        if new_part.tangents is not None:
            new_part = replace(new_part, tangents=orthogonal_tangents(new_part.tangents, normals))
    if rigged:
        new_part = replace(new_part, joints=result.joints, weights=result.weights)
    return new_part
