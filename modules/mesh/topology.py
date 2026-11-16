"""Triangle mesh topology, differential quantities and sparse operators in torch.

Vertices are ``(V, 3)`` floats, faces ``(F, 3)`` int64. Face side ``k`` joins corners ``k``
and ``(k + 1) % 3``. Results stay on the input device.
"""

from __future__ import annotations

__all__ = [
    "EdgeTopology",
    "boundary_chains",
    "boundary_loops",
    "connected_components",
    "corner_angles",
    "corner_fans",
    "cotangent_laplacian",
    "degenerate_faces",
    "dihedral_angles",
    "edge_lengths",
    "edge_topology",
    "face_areas",
    "face_components",
    "face_normals",
    "feature_edges",
    "mass_matrix",
    "non_manifold_vertices",
    "orient_faces",
    "seam_edges",
    "solve_cg",
    "split_non_manifold_vertices",
    "uniform_laplacian",
    "union_find",
    "valence",
    "vertex_areas",
    "vertex_face_csr",
    "vertex_neighbor_csr",
    "vertex_normals",
    "weld",
]

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import torch

#: Corner that ends edge ``k`` of a face.
NEXT = (1, 2, 0)


def _long(values: torch.Tensor) -> torch.Tensor:
    """Values as int64."""
    return values if values.dtype == torch.int64 else values.long()


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    """CSR offsets from per-row counts."""
    offsets = torch.zeros(counts.shape[0] + 1, dtype=torch.int64, device=counts.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return offsets


def _group_rank(keys_sorted: torch.Tensor) -> torch.Tensor:
    """Position of each element within its run of equal keys in a sorted tensor."""
    count = keys_sorted.shape[0]
    if count == 0:
        return keys_sorted.new_zeros(0)
    index = torch.arange(count, device=keys_sorted.device)
    start = torch.ones(count, dtype=torch.bool, device=keys_sorted.device)
    start[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first = torch.where(start, index, torch.zeros_like(index))
    first = torch.cummax(first, 0).values
    return index - first


# ---------------------------------------------------------------------------------------------
# Union find


def _compress(parent: torch.Tensor, parity: torch.Tensor | None = None):
    """Point every node at its root by pointer jumping, carrying parity along."""
    while True:
        grand = parent[parent]
        if torch.equal(grand, parent):
            return parent, parity
        if parity is not None:
            parity = parity ^ parity[parent]
        parent = grand


def union_find(count: int, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Merge nodes joined by pairs and label each node with the smallest node of its set.

    Args:
        count: Number of nodes.
        first: ``(P,)`` node indices.
        second: ``(P,)`` node indices paired with ``first``.

    Returns:
        ``(count,)`` int64 labels, each the smallest index in its set.
    """
    device = first.device
    parent = torch.arange(count, dtype=torch.int64, device=device)
    a, b = _long(first), _long(second)
    keep = a != b
    a, b = a[keep], b[keep]
    while a.numel():
        root_a, root_b = parent[a], parent[b]
        differ = root_a != root_b
        if not bool(differ.any()):
            break
        a, b, root_a, root_b = a[differ], b[differ], root_a[differ], root_b[differ]
        low = torch.minimum(root_a, root_b)
        high = torch.maximum(root_a, root_b)
        parent.scatter_reduce_(0, high, low, reduce="amin")
        parent, _ = _compress(parent)
    return parent


def _parity_union_find(count: int, first: torch.Tensor, second: torch.Tensor, relation: torch.Tensor):
    """Union find carrying a parity per node relative to its root.

    Args:
        count: Number of nodes.
        first: ``(P,)`` node indices.
        second: ``(P,)`` node indices.
        relation: ``(P,)`` int64 0 or 1, the parity each pair asks for between its nodes.

    Returns:
        ``(roots, parity)``: ``(count,)`` int64 smallest node of each set, and int64 parity
        of each node relative to that root.
    """
    device = first.device
    parent = torch.arange(count, dtype=torch.int64, device=device)
    parity = torch.zeros(count, dtype=torch.int64, device=device)
    a, b, rel = _long(first), _long(second), _long(relation)
    while a.numel():
        root_a, root_b = parent[a], parent[b]
        differ = root_a != root_b
        if not bool(differ.any()):
            break
        a, b, rel = a[differ], b[differ], rel[differ]
        root_a, root_b = root_a[differ], root_b[differ]
        between = parity[a] ^ parity[b] ^ rel
        low = torch.minimum(root_a, root_b)
        high = torch.maximum(root_a, root_b)
        key = torch.arange(count, dtype=torch.int64, device=device) * 2
        key.scatter_reduce_(0, high, low * 2 + between, reduce="amin")
        hooked = key < torch.arange(count, dtype=torch.int64, device=device) * 2
        parent = torch.where(hooked, key // 2, parent)
        parity = torch.where(hooked, key % 2, parity)
        parent, parity = _compress(parent, parity)
    return parent, parity


def _compact(labels: torch.Tensor) -> torch.Tensor:
    """Labels renumbered 0 to C - 1 in order of their smallest value."""
    _, inverse = torch.unique(labels, return_inverse=True)
    return inverse


# ---------------------------------------------------------------------------------------------
# Welding


def weld(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tolerance: float | None = None,
    relative: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge vertices closer than a tolerance on every axis into one vertex.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        tolerance: Absolute distance per axis; 0 merges exact duplicates only; None uses
            ``relative`` times the bounding box diagonal.
        relative: Tolerance as a fraction of the bounding box diagonal, used when
            ``tolerance`` is None.

    Returns:
        ``(vertices, faces, remap)``: welded positions (the mean of each group), faces
        indexing them, and ``(V,)`` int64 mapping each old vertex to its welded vertex.
        Welded vertices keep the order of their first old vertex.
    """
    device = vertices.device
    count = int(vertices.shape[0])
    faces = _long(faces)
    if count == 0:
        return vertices.clone(), faces.clone(), torch.zeros(0, dtype=torch.int64, device=device)
    exact, exact_inverse = torch.unique(vertices, dim=0, return_inverse=True)
    unique_count = int(exact.shape[0])
    rows = torch.nonzero(torch.isfinite(exact).all(1)).squeeze(1)
    finite = exact[rows].double()
    if tolerance is None:
        diagonal = float((finite.amax(0) - finite.amin(0)).norm()) if rows.numel() else 0.0
        tolerance = relative * diagonal
    tolerance = float(tolerance)
    labels = torch.arange(unique_count, dtype=torch.int64, device=device)
    if tolerance > 0 and rows.numel() > 1:
        cell = 2.0 * tolerance
        scaled = ((finite - finite.amin(0)) / cell).nan_to_num(0.0)
        first, second = [], []
        # A tolerance below the coordinates' float resolution merges exact duplicates only.
        for shift in range(8 if float(scaled.max()) < 2.0 ** 60 else 0):
            offset = torch.tensor(
                [(shift >> axis) & 1 for axis in range(3)], dtype=torch.float64, device=device
            ) * 0.5
            cells = torch.floor(scaled + offset).long()
            if int(cells.max()) + 1 < (1 << 21):
                keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
                _, group = torch.unique(keys, return_inverse=True)
            else:
                _, group = torch.unique(cells, dim=0, return_inverse=True)
            smallest = torch.full((int(group.max()) + 1,), unique_count, dtype=torch.int64, device=device)
            smallest.scatter_reduce_(0, group, rows, reduce="amin")
            first.append(rows)
            second.append(smallest[group])
        if first:
            labels = union_find(unique_count, torch.cat(first), torch.cat(second))
    groups = labels[exact_inverse]
    leader = torch.full((unique_count,), count, dtype=torch.int64, device=device)
    leader.scatter_reduce_(0, groups, torch.arange(count, device=device), reduce="amin")
    _, remap = torch.unique(leader[groups], return_inverse=True)
    welded_count = int(remap.max()) + 1
    total = torch.zeros(welded_count, 3, dtype=torch.float64, device=device)
    total.index_add_(0, remap, vertices.double())
    members = torch.bincount(remap, minlength=welded_count).clamp_min(1).double()
    welded = (total / members[:, None]).to(vertices.dtype)
    return welded, remap[faces], remap


# ---------------------------------------------------------------------------------------------
# Edges


@dataclass
class EdgeTopology:
    """Undirected edges of a triangle mesh and their incidence with faces.

    Attributes:
        edges: ``(E, 2)`` int64 vertex pairs, smaller index first, sorted.
        face_edges: ``(F, 3)`` int64 edge of each face side ``k`` (corners ``k`` to ``k + 1``).
        face_edge_reversed: ``(F, 3)`` bool, True where the face runs the edge from
            ``edges[e, 1]`` to ``edges[e, 0]``.
        edge_faces: ``(E, 2)`` int64 first two faces on each edge, -1 where absent.
        edge_corners: ``(E, 2)`` int64 side index of the edge within each of those faces.
        edge_opposite: ``(E, 2)`` int64 vertex opposite the edge in each of those faces.
        edge_face_count: ``(E,)`` int64 faces on each edge.
        boundary: ``(E,)`` bool, edges with one face.
        non_manifold: ``(E,)`` bool, edges with more than two faces.
    """

    edges: torch.Tensor
    face_edges: torch.Tensor
    face_edge_reversed: torch.Tensor
    edge_faces: torch.Tensor
    edge_corners: torch.Tensor
    edge_opposite: torch.Tensor
    edge_face_count: torch.Tensor
    boundary: torch.Tensor
    non_manifold: torch.Tensor

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        return int(self.edges.shape[0])

    def manifold_face_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Faces sharing an edge that has exactly two faces.

        Returns:
            ``(edge_ids, pairs)``: ``(P,)`` int64 edges and ``(P, 2)`` int64 face pairs.
        """
        ids = torch.nonzero(self.edge_face_count == 2).squeeze(1)
        return ids, self.edge_faces[ids]


def edge_topology(faces: torch.Tensor, vertex_count: int) -> EdgeTopology:
    """Build the edge table of a triangle mesh.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.

    Returns:
        The `EdgeTopology`.

    Raises:
        ValueError: For a negative vertex index.
    """
    faces = _long(faces)
    device = faces.device
    face_count = int(faces.shape[0])
    stride = max(int(vertex_count), 1)
    if face_count:
        if int(faces.min()) < 0:
            raise ValueError("the mesh has faces with negative vertex indices; repair the mesh first")
        stride = max(stride, int(faces.max()) + 1)
    start = faces.reshape(-1)
    end = faces[:, list(NEXT)].reshape(-1)
    low = torch.minimum(start, end)
    high = torch.maximum(start, end)
    keys = low * stride + high
    unique_keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    edge_count = int(unique_keys.shape[0])
    edges = torch.stack([unique_keys // stride, unique_keys % stride], dim=1)
    order = torch.argsort(inverse, stable=True)
    rank = _group_rank(inverse[order])
    half = torch.arange(face_count * 3, dtype=torch.int64, device=device)
    edge_faces = torch.full((edge_count, 2), -1, dtype=torch.int64, device=device)
    edge_corners = torch.full((edge_count, 2), -1, dtype=torch.int64, device=device)
    edge_opposite = torch.full((edge_count, 2), -1, dtype=torch.int64, device=device)
    for slot in range(2):
        chosen = order[rank == slot]
        edge = inverse[chosen]
        face = half[chosen] // 3
        corner = half[chosen] % 3
        edge_faces[edge, slot] = face
        edge_corners[edge, slot] = corner
        edge_opposite[edge, slot] = faces[face, (corner + 2) % 3]
    return EdgeTopology(
        edges=edges,
        face_edges=inverse.reshape(face_count, 3),
        face_edge_reversed=(start > end).reshape(face_count, 3),
        edge_faces=edge_faces,
        edge_corners=edge_corners,
        edge_opposite=edge_opposite,
        edge_face_count=counts.long(),
        boundary=counts == 1,
        non_manifold=counts > 2,
    )


def vertex_face_csr(faces: torch.Tensor, vertex_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Faces around each vertex in compressed rows.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.

    Returns:
        ``(offsets, indices)``: ``(V + 1,)`` row offsets and int64 face indices, so the faces
        of vertex ``v`` are ``indices[offsets[v]:offsets[v + 1]]``.
    """
    faces = _long(faces)
    flat = faces.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=int(vertex_count))
    return _offsets(counts), order // 3


def vertex_neighbor_csr(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertices sharing an edge with each vertex in compressed rows.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(offsets, indices)``: ``(V + 1,)`` row offsets and int64 neighbour indices,
        ascending within each row.
    """
    topology = topology or edge_topology(faces, vertex_count)
    edges = topology.edges
    source = torch.cat([edges[:, 0], edges[:, 1]])
    target = torch.cat([edges[:, 1], edges[:, 0]])
    order = torch.argsort(source * max(int(vertex_count), 1) + target)
    counts = torch.bincount(source, minlength=int(vertex_count))
    return _offsets(counts), target[order]


def valence(faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None) -> torch.Tensor:
    """Number of edges at each vertex.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(V,)`` int64 edge counts.
    """
    topology = topology or edge_topology(faces, vertex_count)
    return torch.bincount(topology.edges.reshape(-1), minlength=int(vertex_count))


# ---------------------------------------------------------------------------------------------
# Boundaries


def _chain_order(successor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order the nodes of a partial injection into chains.

    Args:
        successor: ``(N,)`` int64 next node, -1 at a chain end; no node has two predecessors.

    Returns:
        ``(order, chain, closed)``: node indices grouped by chain and in walking order, the
        chain id of each node in ``order``, and a bool per chain id, True for a cycle.
    """
    count = int(successor.shape[0])
    device = successor.device
    index = torch.arange(count, dtype=torch.int64, device=device)
    terminal = successor < 0
    jump = torch.where(terminal, index, successor)
    steps = max(1, math.ceil(math.log2(max(count, 2)))) + 1
    reach = jump
    for _ in range(steps):
        reach = reach[reach]
    in_cycle = ~terminal[reach]
    low = index.clone()
    pointer = jump
    for _ in range(steps):
        low = torch.minimum(low, low[pointer])
        pointer = pointer[pointer]
    head = in_cycle & (low == index)
    cut = successor.clone()
    predecessor = torch.full((count,), -1, dtype=torch.int64, device=device)
    linked = ~terminal
    predecessor[successor[linked]] = index[linked]
    cut_at = predecessor[head]
    cut[cut_at] = -1
    terminal = cut < 0
    jump = torch.where(terminal, index, cut)
    distance = (~terminal).long()
    for _ in range(steps):
        distance = distance + distance[jump]
        jump = jump[jump]
    end = jump
    chain_ids, chain = torch.unique(end, return_inverse=True)
    order = torch.argsort(chain * (count + 1) + (count - distance))
    closed = torch.zeros(int(chain_ids.shape[0]), dtype=torch.bool, device=device)
    closed[chain] = in_cycle
    return order, chain[order], closed


def boundary_chains(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Ordered boundary vertex chains, closed loops and open runs alike.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(chains, closed)``: int64 vertex tensors following the face winding, longest first,
        and a bool per chain. A closed loop does not repeat its first vertex; an open run,
        left where winding flips across the boundary, ends on its last vertex.
    """
    faces = _long(faces)
    device = faces.device
    topology = topology or edge_topology(faces, vertex_count)
    vertex_count = int(vertex_count)
    half_boundary = topology.boundary[topology.face_edges.reshape(-1)]
    half = torch.nonzero(half_boundary).squeeze(1)
    count = int(half.shape[0])
    if count == 0:
        return [], torch.zeros(0, dtype=torch.bool, device=device)
    face = half // 3
    corner = half % 3
    start = faces[face, corner]
    end = faces[face, (corner + 1) % 3]
    index = torch.arange(count, dtype=torch.int64, device=device)
    by_start = torch.argsort(start, stable=True)
    out_count = torch.bincount(start, minlength=vertex_count)
    out_offset = _offsets(out_count)
    successor = torch.full((count,), -1, dtype=torch.int64, device=device)
    single = out_count[end] == 1
    successor[single] = by_start[out_offset[end[single]]]

    several = torch.nonzero(out_count[end] > 1).squeeze(1)
    if several.numel():
        boundary_index = torch.full((faces.shape[0] * 3,), -1, dtype=torch.int64, device=device)
        boundary_index[half] = index
        walk_face = face[several]
        walk_corner = (corner[several] + 1) % 3
        found = torch.full((several.shape[0],), -1, dtype=torch.int64, device=device)
        alive = torch.ones(several.shape[0], dtype=torch.bool, device=device)
        limit = int(torch.bincount(faces.reshape(-1), minlength=vertex_count).max()) + 2
        for _ in range(limit):
            if not bool(alive.any()):
                break
            live = torch.nonzero(alive).squeeze(1)
            cur_face, cur_corner = walk_face[live], walk_corner[live]
            edge = topology.face_edges[cur_face, cur_corner]
            on_boundary = topology.boundary[edge]
            hit = live[on_boundary]
            found[hit] = boundary_index[cur_face[on_boundary] * 3 + cur_corner[on_boundary]]
            alive[hit] = False
            interior = ~on_boundary & (topology.edge_face_count[edge] == 2)
            alive[live[~on_boundary & ~interior]] = False
            live, cur_face, cur_corner, edge = (
                live[interior], cur_face[interior], cur_corner[interior], edge[interior]
            )
            side = (topology.edge_faces[edge, 0] == cur_face).long()
            other = topology.edge_faces[edge, side]
            other_corner = topology.edge_corners[edge, side]
            pivot = faces[cur_face, cur_corner]
            consistent = faces[other, other_corner] != pivot
            alive[live[~consistent]] = False
            walk_face[live[consistent]] = other[consistent]
            walk_corner[live[consistent]] = (other_corner[consistent] + 1) % 3
        successor[several] = found

    linked = successor >= 0
    claimant = torch.full((count,), count, dtype=torch.int64, device=device)
    claimant.scatter_reduce_(0, successor[linked], index[linked], reduce="amin")
    duplicate = linked & (claimant[successor.clamp_min(0)] != index)
    successor[duplicate] = -1
    claimed = torch.zeros(count, dtype=torch.bool, device=device)
    claimed[successor[successor >= 0]] = True
    loose_in = torch.nonzero(successor < 0).squeeze(1)
    loose_out = torch.nonzero(~claimed).squeeze(1)
    if loose_in.numel() and loose_out.numel():
        in_vertex = end[loose_in]
        out_vertex = start[loose_out]
        in_order = torch.argsort(in_vertex, stable=True)
        out_order = torch.argsort(out_vertex, stable=True)
        loose_in, in_vertex = loose_in[in_order], in_vertex[in_order]
        loose_out, out_vertex = loose_out[out_order], out_vertex[out_order]
        stride = count + 1
        in_key = in_vertex * stride + _group_rank(in_vertex)
        out_key = out_vertex * stride + _group_rank(out_vertex)
        at = torch.searchsorted(out_key, in_key).clamp_max(out_key.shape[0] - 1)
        match = out_key[at] == in_key
        successor[loose_in[match]] = loose_out[at[match]]

    order, chain, closed = _chain_order(successor)
    chain_count = int(closed.shape[0])
    lengths = torch.bincount(chain, minlength=chain_count).tolist()
    runs = torch.split(start[order], lengths)
    tails = torch.split(end[order], lengths)
    closed_list = closed.tolist()
    chains = [
        run if closed_list[item] else torch.cat([run, tails[item][-1:]])
        for item, run in enumerate(runs)
    ]
    ranking = sorted(range(chain_count), key=lambda item: -int(chains[item].shape[0]))
    flags = closed[torch.tensor(ranking, dtype=torch.int64, device=device)]
    return [chains[item] for item in ranking], flags


def boundary_loops(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> list[torch.Tensor]:
    """Ordered vertex loops along the mesh boundary.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        int64 vertex tensors following the face winding, longest first. See
        `boundary_chains` for open runs.
    """
    return boundary_chains(faces, vertex_count, topology)[0]


# ---------------------------------------------------------------------------------------------
# Components


def connected_components(faces: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Label each vertex with the connected piece of the mesh it belongs to.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices; vertices in no face get a label each.

    Returns:
        ``(V,)`` int64 labels from 0, numbered in order of each piece's smallest vertex.
    """
    faces = _long(faces)
    first = torch.cat([faces[:, 0], faces[:, 1]])
    second = torch.cat([faces[:, 1], faces[:, 2]])
    return _compact(union_find(int(vertex_count), first, second))


def face_components(faces: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Label each face with the connected piece of the mesh it belongs to.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.

    Returns:
        ``(F,)`` int64 labels from 0.
    """
    faces = _long(faces)
    labels = connected_components(faces, vertex_count)
    return _compact(labels[faces[:, 0]])


# ---------------------------------------------------------------------------------------------
# Geometry


def face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Normal of each triangle.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        normalize: Unit length when True, otherwise twice the area in length.

    Returns:
        ``(F, 3)`` normals; zero for a degenerate triangle.
    """
    faces = _long(faces)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    first, second = b - a, c - a
    if not normalize:
        return torch.cross(first, second, dim=1)
    # Each face's edges are scaled to a unit largest component.
    size = torch.maximum(first.abs().amax(1), second.abs().amax(1))
    size = torch.where(size > 0, size, torch.ones_like(size))[:, None]
    normal = torch.cross(first / size, second / size, dim=1)
    length = normal.norm(dim=1, keepdim=True)
    return torch.where(length > 0, normal / length.clamp_min(1e-30), torch.zeros_like(normal))


def face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area of each triangle.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(F,)`` areas.
    """
    return 0.5 * face_normals(vertices, faces, normalize=False).norm(dim=1)


def degenerate_faces(vertices: torch.Tensor, faces: torch.Tensor, area: float = 0.0) -> torch.Tensor:
    """Faces with a repeated vertex or an area at or below a threshold.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        area: Area threshold.

    Returns:
        ``(F,)`` bool.
    """
    faces = _long(faces)
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return repeated | (face_areas(vertices, faces) <= area)


def corner_angles(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Interior angle at each triangle corner.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(F, 3)`` angles in radians.
    """
    faces = _long(faces)
    corners = vertices[faces]
    ahead = corners[:, [1, 2, 0]] - corners
    behind = corners[:, [2, 0, 1]] - corners
    size = torch.maximum(ahead.abs().amax(-1), behind.abs().amax(-1))
    size = torch.where(size > 0, size, torch.ones_like(size))[..., None]
    ahead, behind = ahead / size, behind / size
    cosine = (ahead * behind).sum(-1)
    sine = torch.cross(ahead, behind, dim=-1).norm(dim=-1)
    return torch.atan2(sine, cosine)


def vertex_normals(
    vertices: torch.Tensor, faces: torch.Tensor, weighting: str = "angle"
) -> torch.Tensor:
    """Unit normal at each vertex from the faces around it.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        weighting: ``"angle"`` (corner angle), ``"area"`` (face area) or ``"uniform"``.

    Returns:
        ``(V, 3)`` normals; zero at a vertex in no face.

    Raises:
        ValueError: For an unknown weighting.
    """
    faces = _long(faces)
    if weighting == "area":
        weighted = face_normals(vertices, faces, normalize=False)[:, None, :].expand(-1, 3, -1)
    elif weighting == "angle":
        weighted = face_normals(vertices, faces)[:, None, :] * corner_angles(vertices, faces)[..., None]
    elif weighting == "uniform":
        weighted = face_normals(vertices, faces)[:, None, :].expand(-1, 3, -1)
    else:
        raise ValueError(f"vertex normal weighting must be angle, area or uniform, not {weighting!r}")
    total = torch.zeros_like(vertices)
    total.index_add_(0, faces.reshape(-1), weighted.reshape(-1, 3))
    length = total.norm(dim=1, keepdim=True)
    return torch.where(length > 0, total / length.clamp_min(1e-30), torch.zeros_like(total))


def edge_lengths(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Length of each edge.

    Args:
        vertices: ``(V, 3)`` positions.
        edges: ``(E, 2)`` vertex indices.

    Returns:
        ``(E,)`` lengths.
    """
    edges = _long(edges)
    return (vertices[edges[:, 1]] - vertices[edges[:, 0]]).norm(dim=1)


def dihedral_angles(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    topology: EdgeTopology | None = None,
    signed: bool = False,
) -> torch.Tensor:
    """Angle between the normals of the first two faces on each edge.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        topology: The mesh's `EdgeTopology`, built when None.
        signed: Negative across a concave edge when True.

    Returns:
        ``(E,)`` radians, 0 for a flat or boundary edge, up to pi for a folded edge.
    """
    faces = _long(faces)
    topology = topology or edge_topology(faces, vertices.shape[0])
    normals = face_normals(vertices, faces)
    paired = topology.edge_face_count >= 2
    first = topology.edge_faces[:, 0].clamp_min(0)
    second = topology.edge_faces[:, 1].clamp_min(0)
    n0, n1 = normals[first], normals[second]
    cosine = (n0 * n1).sum(1)
    sine = torch.cross(n0, n1, dim=1).norm(dim=1)
    angle = torch.atan2(sine, cosine)
    if signed:
        apex = vertices[topology.edge_opposite[:, 1].clamp_min(0)] - vertices[topology.edges[:, 0]]
        concave = (apex * n0).sum(1) > 0
        angle = torch.where(concave, -angle, angle)
    return torch.where(paired, angle, torch.zeros_like(angle))


def feature_edges(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    angle_deg: float = 30.0,
    topology: EdgeTopology | None = None,
    include_boundary: bool = True,
    include_non_manifold: bool = True,
) -> torch.Tensor:
    """Edges whose dihedral angle reaches a threshold, plus boundary and non-manifold edges.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        angle_deg: Dihedral angle threshold in degrees.
        topology: The mesh's `EdgeTopology`, built when None.
        include_boundary: Mark boundary edges.
        include_non_manifold: Mark edges with more than two faces.

    Returns:
        ``(E,)`` bool aligned with ``topology.edges``.
    """
    faces = _long(faces)
    topology = topology or edge_topology(faces, vertices.shape[0])
    marked = dihedral_angles(vertices, faces, topology) >= math.radians(float(angle_deg))
    if include_boundary:
        marked = marked | topology.boundary
    if include_non_manifold:
        marked = marked | topology.non_manifold
    return marked


def seam_edges(
    faces_split: torch.Tensor, remap: torch.Tensor, topology: EdgeTopology | None = None
) -> torch.Tensor:
    """Welded edges where the faces on either side use different split vertices.

    Args:
        faces_split: ``(F, 3)`` faces indexing attribute-split vertices.
        remap: ``(V,)`` split vertex to welded vertex, as returned by `weld`.
        topology: `EdgeTopology` of ``remap[faces_split]``, built when None.

    Returns:
        ``(E,)`` bool aligned with that welded topology's edges.
    """
    faces_split = _long(faces_split)
    remap = _long(remap)
    welded = remap[faces_split]
    welded_count = int(remap.max()) + 1 if remap.numel() else 0
    topology = topology or edge_topology(welded, welded_count)
    split_start = faces_split.reshape(-1)
    split_end = faces_split[:, list(NEXT)].reshape(-1)
    reversed_ = topology.face_edge_reversed.reshape(-1)
    low_split = torch.where(reversed_, split_end, split_start)
    high_split = torch.where(reversed_, split_start, split_end)
    edge = topology.face_edges.reshape(-1)
    first_face = topology.edge_faces[edge, 0]
    first_corner = topology.edge_corners[edge, 0]
    first_half = first_face * 3 + first_corner
    differs = (low_split != low_split[first_half]) | (high_split != high_split[first_half])
    seam = torch.zeros(topology.edge_count, dtype=torch.bool, device=welded.device)
    seam[edge[differs]] = True
    return seam


# ---------------------------------------------------------------------------------------------
# Orientation and fans


def orient_faces(
    faces: torch.Tensor,
    vertex_count: int,
    vertices: torch.Tensor | None = None,
    outward: str = "closed",
) -> torch.Tensor:
    """Faces rewound to agree in winding with their neighbours across every two-face edge.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        vertices: ``(V, 3)`` positions, needed to turn pieces outward.
        outward: ``"closed"`` turns closed pieces outward by signed volume, ``"all"`` turns
            open pieces too, ``"none"`` keeps the winding of each piece's first face.

    Returns:
        ``(F, 3)`` int64 faces.

    Raises:
        ValueError: For an unknown ``outward``.
    """
    if outward not in ("closed", "all", "none"):
        raise ValueError(f"outward must be closed, all or none, not {outward!r}")
    faces = _long(faces)
    face_count = int(faces.shape[0])
    if face_count == 0:
        return faces.clone()
    topology = edge_topology(faces, vertex_count)
    ids, pairs = topology.manifold_face_pairs()
    reversed_flat = topology.face_edge_reversed.reshape(-1)
    first_half = pairs[:, 0] * 3 + topology.edge_corners[ids, 0]
    second_half = pairs[:, 1] * 3 + topology.edge_corners[ids, 1]
    relation = (reversed_flat[first_half] == reversed_flat[second_half]).long()
    roots, flip = _parity_union_find(face_count, pairs[:, 0], pairs[:, 1], relation)
    if outward != "none" and vertices is not None:
        oriented = torch.where(flip.bool()[:, None], faces[:, [0, 2, 1]], faces)
        corners = vertices[oriented].double()
        centroid_sum = torch.zeros(face_count, 3, dtype=torch.float64, device=faces.device)
        centroid_sum.index_add_(0, roots, corners.mean(1))
        members = torch.bincount(roots, minlength=face_count).clamp_min(1).double()
        centre = (centroid_sum / members[:, None])[roots]
        a, b, c = corners[:, 0] - centre, corners[:, 1] - centre, corners[:, 2] - centre
        volume = torch.zeros(face_count, dtype=torch.float64, device=faces.device)
        volume.index_add_(0, roots, (a * torch.cross(b, c, dim=1)).sum(1))
        turn = volume < 0
        if outward == "closed":
            open_edges = topology.boundary | topology.non_manifold
            open_piece = torch.zeros(face_count, dtype=torch.bool, device=faces.device)
            touching = open_edges[topology.face_edges].any(1)
            open_piece[roots[touching]] = True
            turn = turn & ~open_piece
        flip = flip ^ turn[roots].long()
    return torch.where(flip.bool()[:, None], faces[:, [0, 2, 1]], faces)


def corner_fans(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> torch.Tensor:
    """Group the corners at each vertex into fans joined through two-face edges or a shared face.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(F * 3,)`` int64 fan label of corner ``f * 3 + k``, the smallest corner of its fan.
    """
    faces = _long(faces)
    topology = topology or edge_topology(faces, vertex_count)
    ids, pairs = topology.manifold_face_pairs()
    links_a, links_b = [], []
    for endpoint in range(2):
        vertex = topology.edges[ids, endpoint]
        corners = []
        for slot in range(2):
            face = pairs[:, slot]
            side = topology.edge_corners[ids, slot]
            at_start = faces[face, side] == vertex
            corners.append(face * 3 + torch.where(at_start, side, (side + 1) % 3))
        links_a.append(corners[0])
        links_b.append(corners[1])
    # Corners of one face on the same vertex always share a fan.
    for side in range(3):
        same = torch.nonzero(faces[:, side] == faces[:, NEXT[side]]).squeeze(1)
        links_a.append(same * 3 + side)
        links_b.append(same * 3 + NEXT[side])
    return union_find(int(faces.shape[0]) * 3, torch.cat(links_a), torch.cat(links_b))


def non_manifold_vertices(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> torch.Tensor:
    """Vertices whose faces form more than one fan or that touch a non-manifold edge.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(V,)`` bool.
    """
    faces = _long(faces)
    vertex_count = int(vertex_count)
    topology = topology or edge_topology(faces, vertex_count)
    fans = corner_fans(faces, vertex_count, topology)
    vertex = faces.reshape(-1)
    pairs = torch.unique(vertex * (faces.shape[0] * 3) + fans)
    owners = pairs // (faces.shape[0] * 3)
    marked = torch.bincount(owners, minlength=vertex_count) > 1
    marked[topology.edges[topology.non_manifold].reshape(-1)] = True
    return marked


def split_non_manifold_vertices(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every fan at a vertex its own vertex.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.

    Returns:
        ``(faces, source)``: faces on the split vertices and ``(V2,)`` int64 original vertex
        of each. Vertices keep their index for their first fan; extra fans are appended.
    """
    faces = _long(faces)
    vertex_count = int(vertex_count)
    device = faces.device
    corner_count = int(faces.shape[0]) * 3
    fans = corner_fans(faces, vertex_count, topology)
    vertex = faces.reshape(-1)
    keys, inverse = torch.unique(vertex * max(corner_count, 1) + fans, return_inverse=True)
    owners = keys // max(corner_count, 1)
    rank = _group_rank(owners)
    extra = rank > 0
    new_index = owners.clone()
    new_index[extra] = vertex_count + torch.arange(int(extra.sum()), dtype=torch.int64, device=device)
    source = torch.cat([torch.arange(vertex_count, dtype=torch.int64, device=device), owners[extra]])
    return new_index[inverse].reshape(-1, 3), source


# ---------------------------------------------------------------------------------------------
# Sparse operators


def _laplacian_from_weights(
    rows: torch.Tensor, cols: torch.Tensor, weights: torch.Tensor, vertex_count: int
) -> torch.Tensor:
    """Assemble ``D - W`` from symmetric off-diagonal weights as coalesced COO."""
    diagonal = torch.zeros(vertex_count, dtype=weights.dtype, device=weights.device)
    diagonal.index_add_(0, rows, weights)
    diagonal.index_add_(0, cols, weights)
    every = torch.arange(vertex_count, dtype=torch.int64, device=weights.device)
    index = torch.stack([torch.cat([rows, cols, every]), torch.cat([cols, rows, every])])
    values = torch.cat([-weights, -weights, diagonal])
    shape = (vertex_count, vertex_count)
    return torch.sparse_coo_tensor(index, values, shape, check_invariants=False).coalesce()


def uniform_laplacian(
    faces: torch.Tensor, vertex_count: int, topology: EdgeTopology | None = None, dtype=torch.float32
) -> torch.Tensor:
    """Graph Laplacian ``D - A`` with unit edge weights.

    Args:
        faces: ``(F, 3)`` vertex indices.
        vertex_count: Number of vertices.
        topology: The mesh's `EdgeTopology`, built when None.
        dtype: Value type.

    Returns:
        ``(V, V)`` coalesced sparse COO, positive semi-definite.
    """
    faces = _long(faces)
    topology = topology or edge_topology(faces, vertex_count)
    edges = topology.edges
    weights = torch.ones(edges.shape[0], dtype=dtype, device=faces.device)
    return _laplacian_from_weights(edges[:, 0], edges[:, 1], weights, int(vertex_count))


def _cotangents(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Cotangent of the angle at each corner, ``(F, 3)``."""
    corners = vertices[faces]
    ahead = corners[:, [1, 2, 0]] - corners
    behind = corners[:, [2, 0, 1]] - corners
    cosine = (ahead * behind).sum(-1)
    sine = torch.cross(ahead, behind, dim=-1).norm(dim=-1)
    return cosine / sine.clamp_min(torch.finfo(vertices.dtype).tiny)


def cotangent_laplacian(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    clamp: float | None = 1e4,
    nonnegative: bool = False,
) -> torch.Tensor:
    """Cotangent Laplacian ``D - W`` with ``W_ij = (cot a + cot b) / 2``.

    Args:
        vertices: ``(V, 3)`` positions; the value type follows them.
        faces: ``(F, 3)`` vertex indices.
        clamp: Bound on the magnitude of each cotangent, or None for no bound.
        nonnegative: Clamp each cotangent to 0 or more.

    Returns:
        ``(V, V)`` coalesced sparse COO, positive semi-definite when weights are nonnegative.
    """
    faces = _long(faces)
    cot = _cotangents(vertices, faces)
    if clamp is not None:
        cot = cot.clamp(-float(clamp), float(clamp))
    if nonnegative:
        cot = cot.clamp_min(0)
    rows = faces[:, [1, 2, 0]].reshape(-1)
    cols = faces[:, [2, 0, 1]].reshape(-1)
    return _laplacian_from_weights(rows, cols, 0.5 * cot.reshape(-1), int(vertices.shape[0]))


def vertex_areas(vertices: torch.Tensor, faces: torch.Tensor, kind: str = "voronoi") -> torch.Tensor:
    """Area attributed to each vertex.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        kind: ``"voronoi"`` (mixed Voronoi cells, obtuse triangles split by halves and
            quarters) or ``"barycentric"`` (a third of each face).

    Returns:
        ``(V,)`` areas summing to the surface area.

    Raises:
        ValueError: For an unknown kind.
    """
    faces = _long(faces)
    area = face_areas(vertices, faces)
    if kind == "barycentric":
        share = (area / 3.0)[:, None].expand(-1, 3)
    elif kind == "voronoi":
        corners = vertices[faces]
        cot = _cotangents(vertices, faces)
        side_sq = ((corners[:, [2, 0, 1]] - corners[:, [1, 2, 0]]) ** 2).sum(-1)
        voronoi = torch.stack([
            side_sq[:, 2] * cot[:, 2] + side_sq[:, 1] * cot[:, 1],
            side_sq[:, 0] * cot[:, 0] + side_sq[:, 2] * cot[:, 2],
            side_sq[:, 1] * cot[:, 1] + side_sq[:, 0] * cot[:, 0],
        ], dim=1) / 8.0
        obtuse = cot < 0
        any_obtuse = obtuse.any(1, keepdim=True)
        mixed = torch.where(obtuse, area[:, None] / 2.0, area[:, None] / 4.0)
        share = torch.where(any_obtuse, mixed, voronoi)
        share = torch.where(torch.isfinite(share), share, (area / 3.0)[:, None].expand(-1, 3))
    else:
        raise ValueError(f"vertex area kind must be voronoi or barycentric, not {kind!r}")
    total = torch.zeros(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
    total.index_add_(0, faces.reshape(-1), share.reshape(-1).to(vertices.dtype))
    return total


def mass_matrix(vertices: torch.Tensor, faces: torch.Tensor, kind: str = "voronoi") -> torch.Tensor:
    """Lumped mass matrix holding `vertex_areas` on its diagonal.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        kind: ``"voronoi"`` or ``"barycentric"``.

    Returns:
        ``(V, V)`` coalesced sparse COO diagonal matrix.
    """
    areas = vertex_areas(vertices, faces, kind)
    index = torch.arange(areas.shape[0], device=areas.device)
    shape = (areas.shape[0],) * 2
    return torch.sparse_coo_tensor(torch.stack([index, index]), areas, shape, check_invariants=False).coalesce()


def _diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Main diagonal of a square sparse or dense matrix."""
    if matrix.layout == torch.strided:
        return torch.diagonal(matrix)
    coo = matrix.to_sparse_coo().coalesce() if matrix.layout != torch.sparse_coo else matrix.coalesce()
    index = coo.indices()
    on = index[0] == index[1]
    diagonal = torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    diagonal.index_add_(0, index[0][on], coo.values()[on])
    return diagonal


def solve_cg(
    matrix: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    preconditioner: str | torch.Tensor | None = "jacobi",
    return_info: bool = False,
):
    """Solve a symmetric positive definite system by preconditioned conjugate gradients.

    Args:
        matrix: ``(N, N)`` sparse (COO or CSR) or dense matrix, or a callable applying it
            to an ``(N, K)`` tensor.
        rhs: ``(N,)`` or ``(N, K)`` right-hand sides, solved together column by column.
        x0: Starting guess shaped like ``rhs``, zero when None.
        tol: Relative residual ``|r| / |b|`` each column stops at.
        max_iter: Iteration limit.
        preconditioner: ``"jacobi"``, None, or an ``(N,)`` tensor of inverse diagonal values.
        return_info: Also return the iteration count and final relative residuals.

    Returns:
        The solution shaped like ``rhs``, or ``(solution, iterations, residuals)``.

    Raises:
        ValueError: For an unknown preconditioner, or Jacobi with a callable matrix.
    """
    single = rhs.dim() == 1
    b = rhs[:, None] if single else rhs
    if callable(matrix) and not isinstance(matrix, torch.Tensor):
        apply = matrix
        stored = None
    else:
        stored = matrix
        if matrix.layout == torch.sparse_coo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                operator = matrix.coalesce().to_sparse_csr()
        else:
            operator = matrix
        apply = lambda value: operator @ value  # noqa: E731
    if isinstance(preconditioner, torch.Tensor):
        inverse = preconditioner.to(b.dtype)[:, None]
    elif preconditioner == "jacobi":
        if stored is None:
            raise ValueError("a Jacobi preconditioner needs the matrix itself, not a callable")
        diagonal = _diagonal(stored).to(b.dtype)
        inverse = torch.where(diagonal.abs() > 0, 1.0 / diagonal, torch.ones_like(diagonal))[:, None]
    elif preconditioner is None:
        inverse = None
    else:
        raise ValueError(f"preconditioner must be 'jacobi', None or a tensor, not {preconditioner!r}")
    x = torch.zeros_like(b) if x0 is None else (x0[:, None] if single else x0).clone().to(b.dtype)
    r = b - apply(x) if x0 is not None else b.clone()
    z = r * inverse if inverse is not None else r
    p = z.clone()
    rz = (r * z).sum(0)
    scale = b.norm(dim=0).clamp_min(torch.finfo(b.dtype).tiny)
    iterations = 0
    residual = r.norm(dim=0) / scale
    for iterations in range(1, int(max_iter) + 1):
        active = residual > tol
        if not bool(active.any()):
            iterations -= 1
            break
        product = apply(p)
        curvature = (p * product).sum(0)
        usable = active & (curvature != 0)
        step = torch.where(usable, rz / torch.where(usable, curvature, torch.ones_like(curvature)), 0.0)
        x = x + step * p
        r = r - step * product
        z = r * inverse if inverse is not None else r
        rz_next = (r * z).sum(0)
        moving = active & (rz != 0)
        beta = torch.where(moving, rz_next / torch.where(moving, rz, torch.ones_like(rz)), 0.0)
        p = z + beta * p
        rz = rz_next
        residual = r.norm(dim=0) / scale
    solution = x[:, 0] if single else x
    if return_info:
        return solution, iterations, residual
    return solution
