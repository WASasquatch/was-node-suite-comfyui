"""Polygon meshes of mixed quads, triangles and n-gons, and the edits a retopology makes to them.
Polygons are stored as corner counts and a flat run of vertex indices, counter-clockwise seen from
the front."""
# Per-vertex marks travel with the vertices through every edit.

from __future__ import annotations

__all__ = [
    "PolyMesh", "corner_angles", "dissolve_valence_two", "drop_folds", "drop_fragments", "drop_non_manifold",
    "drop_stray_shells", "feature_pairs", "fill_holes", "flaws", "hole_loops", "merge_triangles", "open_pinches",
    "order_cycles", "polygon_normals", "relax", "repeats_polygon", "split_large", "subdivide_quads", "triangles_only",
    "twins",
]

import math
from dataclasses import dataclass, field, replace

import torch

from .. import topology
from ..bvh import TriangleBVH
from .hierarchy import match_pairs

#: Default value of each mark for a vertex an edit creates.
MARK_DEFAULTS = {"feature": False, "corner": False}

#: Sides with an opening of the source beyond them that make a hole part of that opening.
OPENING_SIDES = 2

#: Most polygons in a fan removed where fans meet at a vertex between open sides.
PINCH_POLYGONS = 4

#: Most rounds removing interior vertices with two neighbours.
DISSOLVE_PASSES = 4


@dataclass
class PolyMesh:
    """Polygons over shared vertices.

    Attributes:
        vertices: ``(V, 3)`` positions.
        sizes: ``(P,)`` int64 corners per polygon.
        indices: ``(sizes.sum(),)`` int64 vertex of each corner, polygon after polygon.
        marks: Per-vertex tensors by name: ``"feature"`` and ``"corner"`` (bool), ``"scale"``
            (lattice spacing) and any others.
    """

    vertices: torch.Tensor
    sizes: torch.Tensor
    indices: torch.Tensor
    marks: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def polygon_count(self) -> int:
        """Number of polygons."""
        return int(self.sizes.shape[0])

    def offsets(self) -> torch.Tensor:
        """``(P + 1,)`` start of each polygon's corners, and the end."""
        offsets = torch.zeros(self.polygon_count + 1, dtype=torch.int64, device=self.sizes.device)
        offsets[1:] = torch.cumsum(self.sizes, 0)
        return offsets

    def corner_polygon(self) -> torch.Tensor:
        """``(C,)`` polygon of each corner."""
        return torch.repeat_interleave(torch.arange(self.polygon_count, device=self.sizes.device), self.sizes)

    def next_corner(self) -> torch.Tensor:
        """``(C,)`` the following corner of the same polygon."""
        offsets = self.offsets()
        polygon = self.corner_polygon()
        corner = torch.arange(self.indices.shape[0], device=self.indices.device)
        following = corner + 1
        return torch.where(following == offsets[polygon + 1], offsets[polygon], following)

    def half_edges(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(src, dst, polygon)`` of every polygon side, one per corner."""
        return self.indices, self.indices[self.next_corner()], self.corner_polygon()

    def edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Undirected edges ``(E, 2)`` and the edge of each corner's side ``(C,)``."""
        src, dst, _ = self.half_edges()
        low, high = torch.minimum(src, dst), torch.maximum(src, dst)
        stride = max(self.count, 1)
        keys, inverse = torch.unique(low * stride + high, return_inverse=True)
        return torch.stack([keys // stride, keys % stride], 1), inverse

    def centroids(self) -> torch.Tensor:
        """``(P, 3)`` mean corner position of each polygon."""
        total = torch.zeros(self.polygon_count, 3, dtype=self.vertices.dtype, device=self.vertices.device)
        total.index_add_(0, self.corner_polygon(), self.vertices[self.indices])
        return total / self.sizes.clamp_min(1)[:, None].to(total.dtype)

    def triangulate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Triangles of every polygon: quads across a diagonal, others as fans.

        Returns:
            ``(faces, triangle_polygon)``: ``(T, 3)`` int64 and ``(T,)`` int64. A quad is cut
            across the diagonal whose two triangles both face the way the quad does, and across
            the shorter one where both or neither do.
        """
        device = self.indices.device
        offsets = self.offsets()
        pieces, owners = [], []
        quads = torch.nonzero(self.sizes == 4).squeeze(1)
        if quads.numel():
            corners = self.indices[offsets[quads][:, None] + torch.arange(4, device=device)]
            p = self.vertices[corners]
            facing = torch.cross(p, torch.roll(p, -1, dims=1), dim=-1).sum(1)

            def upright(a: int, b: int, c: int) -> torch.Tensor:
                return (torch.cross(p[:, b] - p[:, a], p[:, c] - p[:, a], dim=1) * facing).sum(1) > 0

            across = upright(0, 1, 2) & upright(0, 2, 3)
            other = upright(1, 2, 3) & upright(1, 3, 0)
            shorter = (p[:, 0] - p[:, 2]).norm(dim=1) <= (p[:, 1] - p[:, 3]).norm(dim=1)
            first = torch.where(across == other, shorter, across)
            a = torch.where(first[:, None], corners[:, [0, 1, 2]], corners[:, [1, 2, 3]])
            b = torch.where(first[:, None], corners[:, [0, 2, 3]], corners[:, [1, 3, 0]])
            pieces += [a, b]
            owners += [quads, quads]
        others = torch.nonzero((self.sizes != 4) & (self.sizes >= 3)).squeeze(1)
        if others.numel():
            fan = self.sizes[others] - 2
            owner = torch.repeat_interleave(others, fan)
            step = torch.arange(int(fan.sum()), device=device) - torch.repeat_interleave(
                torch.cumsum(fan, 0) - fan, fan)
            base = offsets[owner]
            pieces.append(torch.stack([self.indices[base], self.indices[base + step + 1],
                                       self.indices[base + step + 2]], 1))
            owners.append(owner)
        if not pieces:
            return torch.zeros(0, 3, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)
        faces = torch.cat(pieces)
        owner = torch.cat(owners)
        order = torch.argsort(owner, stable=True)
        return faces[order], owner[order]

    def vertex_normals(self) -> torch.Tensor:
        """``(V, 3)`` angle-weighted unit normals from the triangulated polygons."""
        faces, _ = self.triangulate()
        return topology.vertex_normals(self.vertices, faces)

    def select(self, keep: torch.Tensor) -> PolyMesh:
        """The mesh with only the polygons where ``keep`` is True."""
        corner_keep = torch.repeat_interleave(keep, self.sizes)
        return replace(self, sizes=self.sizes[keep], indices=self.indices[corner_keep])

    def compacted(self) -> PolyMesh:
        """The mesh without vertices no polygon uses."""
        used = torch.zeros(self.count, dtype=torch.bool, device=self.vertices.device)
        used[self.indices] = True
        index = torch.full((self.count,), -1, dtype=torch.int64, device=self.vertices.device)
        index[used] = torch.arange(int(used.sum()), device=self.vertices.device)
        return PolyMesh(self.vertices[used], self.sizes, index[self.indices],
                        {name: value[used] for name, value in self.marks.items()})

    def appended(self, points: torch.Tensor, sources: list[torch.Tensor] | torch.Tensor,
                 marks: dict[str, torch.Tensor] | None = None) -> PolyMesh:
        """The mesh with vertices added, their marks taken from source vertices.

        Args:
            points: ``(N, 3)`` new positions.
            sources: ``(N, K)`` int64 vertices each new one is made from; ``"scale"`` is their
                mean, other marks take their default unless given.
            marks: Explicit marks for the new vertices.

        Returns:
            A mesh with the same polygons.
        """
        given = marks or {}
        out = {}
        for name, value in self.marks.items():
            if name in given:
                extra = given[name].to(value.dtype)
            elif value.dtype == torch.bool:
                extra = torch.full((points.shape[0],), MARK_DEFAULTS.get(name, False), dtype=torch.bool,
                                   device=value.device)
            else:
                extra = value[sources].float().mean(1).to(value.dtype) if points.shape[0] else value[:0]
            out[name] = torch.cat([value, extra])
        return replace(self, vertices=torch.cat([self.vertices, points.to(self.vertices.dtype)]), marks=out)


def twins(src: torch.Tensor, dst: torch.Tensor, count: int) -> torch.Tensor:
    """Index of the opposite half-edge of each half-edge, -1 where it has none.

    Args:
        src: ``(H,)`` half-edge starts.
        dst: ``(H,)`` half-edge ends.
        count: Number of vertices.

    Returns:
        ``(H,)`` int64.
    """
    if src.numel() == 0:
        return src.clone()
    stride = max(int(count), 1)
    keys, order = torch.sort(src * stride + dst)
    wanted = dst * stride + src
    at = torch.searchsorted(keys, wanted).clamp_max(keys.shape[0] - 1)
    return torch.where(keys[at] == wanted, order[at], torch.full_like(at, -1))


def order_cycles(successor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Label the cycles of a permutation and rank each element along its cycle.

    Args:
        successor: ``(N,)`` int64 permutation.

    Returns:
        ``(label, rank)``: smallest element of each element's cycle, and steps from it.
    """
    count = int(successor.shape[0])
    device = successor.device
    index = torch.arange(count, device=device)
    label = topology.union_find(count, index, successor)
    previous = torch.empty_like(successor)
    previous[successor] = index
    previous = torch.where(label == index, torch.full_like(previous, -1), previous)
    rank = (previous >= 0).long()
    pointer = previous.clone()
    for _ in range(max(count, 1).bit_length() + 1):
        if not bool((pointer >= 0).any()):
            break
        live = pointer >= 0
        target = pointer.clamp_min(0)
        rank = torch.where(live, rank + rank[target], rank)
        pointer = torch.where(live, pointer[target], pointer)
    return label, rank


def polygon_normals(points: torch.Tensor) -> torch.Tensor:
    """Unit vector area of polygons of one size.

    Args:
        points: ``(P, K, 3)`` corner positions in order.

    Returns:
        ``(P, 3)`` unit normals, zero for a polygon without area.
    """
    area = torch.cross(points, torch.roll(points, -1, dims=1), dim=-1).sum(1)
    return torch.nn.functional.normalize(area, dim=-1)


def corner_angles(points: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
    """Interior angle at each corner of polygons of one size.

    Args:
        points: ``(P, K, 3)`` corner positions in order.
        normals: ``(P, 3)`` unit normals the turn is measured about, or None for unsigned
            angles in ``[0, pi]``.

    Returns:
        ``(P, K)`` angles in radians; with normals, reflex corners exceed ``pi``.
    """
    ahead = torch.roll(points, -1, dims=1) - points
    behind = torch.roll(points, 1, dims=1) - points
    cosine = (ahead * behind).sum(-1)
    cross = torch.cross(ahead, behind, dim=-1)
    if normals is None:
        return torch.atan2(cross.norm(dim=-1), cosine)
    angle = torch.atan2((cross * normals[:, None, :]).sum(-1), cosine)
    return torch.where(angle < 0, angle + 2.0 * math.pi, angle)


def _turn(behind: torch.Tensor, at: torch.Tensor, ahead: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Interior angle at ``at`` between the sides to ``behind`` and ``ahead``, about ``normals``.

    Args:
        behind: ``(P, K, 3)`` previous corners.
        at: ``(P, K, 3)`` corners measured.
        ahead: ``(P, K, 3)`` following corners.
        normals: ``(P, 3)`` unit normals.

    Returns:
        ``(P, K)`` angles in ``[0, 2 pi)``.
    """
    forward, backward = ahead - at, behind - at
    angle = torch.atan2((torch.cross(forward, backward, dim=-1) * normals[:, None, :]).sum(-1),
                        (forward * backward).sum(-1))
    return torch.where(angle < 0, angle + 2.0 * math.pi, angle)


def _edge_keys(mesh: PolyMesh) -> torch.Tensor:
    """Sorted order-free keys of every polygon side, ``low * V + high``."""
    src, dst, _ = mesh.half_edges()
    stride = max(mesh.count, 1)
    return torch.unique(torch.minimum(src, dst) * stride + torch.maximum(src, dst))


def _has_key(keys: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """Whether each wanted key is among the sorted keys."""
    if keys.numel() == 0:
        return torch.zeros_like(wanted, dtype=torch.bool)
    at = torch.searchsorted(keys, wanted).clamp_max(keys.shape[0] - 1)
    return keys[at] == wanted


def repeats_polygon(mesh: PolyMesh, sizes: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """Loops whose corners are exactly the corners of a polygon already in the mesh.

    Args:
        mesh: The mesh.
        sizes: ``(L,)`` int64 corners per loop.
        corners: ``(sizes.sum(),)`` int64 the loops' corners, loop after loop.

    Returns:
        ``(L,)`` bool.
    """
    device = sizes.device
    found = torch.zeros(sizes.shape[0], dtype=torch.bool, device=device)
    if sizes.numel() == 0 or mesh.polygon_count == 0:
        return found
    loop_start = torch.cumsum(sizes, 0) - sizes
    polygon_start = mesh.offsets()[:-1]
    for size in torch.unique(sizes).tolist():
        rows = torch.nonzero(mesh.sizes == size).squeeze(1)
        if rows.numel() == 0:
            continue
        loops = torch.nonzero(sizes == size).squeeze(1)
        span = torch.arange(size, device=device)
        theirs = mesh.indices[polygon_start[rows][:, None] + span].sort(1).values
        mine = corners[loop_start[loops][:, None] + span].sort(1).values
        _, group = torch.unique(torch.cat([theirs, mine]), dim=0, return_inverse=True)
        known = torch.zeros(int(group.max()) + 1, dtype=torch.bool, device=device)
        known[group[:rows.shape[0]]] = True
        found[loops] = known[group[rows.shape[0]:]]
    return found


def _quad_badness(points: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
    """Largest departure of a quad's corner angles from a right angle, in radians."""
    if normals is None:
        normals = polygon_normals(points)
    return (corner_angles(points, normals) - math.pi / 2).abs().amax(-1)


def split_large(mesh: PolyMesh, max_size: int = 4) -> PolyMesh:
    """Cut polygons above ``max_size`` corners into quads, and a triangle where the count is odd.

    Args:
        mesh: The mesh.
        max_size: Largest polygon left whole, 4 or more.

    Returns:
        The mesh with no polygon above ``max_size`` corners, but for polygons every cut of
        which would run along an existing side; those stay whole.
    """
    # Each cut takes off the quad of four consecutive corners whose angles come closest to right
    # angles, measured about the polygon's own normal.
    device = mesh.indices.device
    max_size = max(int(max_size), 4)
    if not bool((mesh.sizes > max_size).any()):
        return mesh
    stride = max(mesh.count, 1)
    taken = _edge_keys(mesh)
    offsets = mesh.offsets()
    keep = mesh.sizes <= max_size
    out_sizes = [mesh.sizes[keep]]
    out_indices = [mesh.indices[torch.repeat_interleave(keep, mesh.sizes)]]
    pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for size in torch.unique(mesh.sizes[~keep]).tolist():
        rows = torch.nonzero(mesh.sizes == size).squeeze(1)
        corners = mesh.indices[offsets[rows][:, None] + torch.arange(size, device=device)]
        pending.setdefault(size, []).append((corners, polygon_normals(mesh.vertices[corners])))
    for size in range(max(pending), max_size, -1):
        if size not in pending:
            continue
        corners = torch.cat([item[0] for item in pending[size]])
        normals = torch.cat([item[1] for item in pending[size]])
        points = mesh.vertices[corners]
        angles = corner_angles(points, normals)
        steps = torch.arange(4, device=device)
        window = (torch.arange(size, device=device)[:, None] + steps) % size
        quad_points = points[:, window]
        flat_normals = normals[:, None, :].expand(-1, size, -1).reshape(-1, 3)
        quad_angles = corner_angles(quad_points.reshape(-1, 4, 3), flat_normals).reshape(-1, size, 4)
        cost = (quad_angles - math.pi / 2).abs().sum(-1)
        cost = cost + (angles[:, window[:, 1]] - math.pi / 2).abs() + (angles[:, window[:, 2]] - math.pi / 2).abs()
        before = points[:, (torch.arange(size, device=device) - 1) % size]
        after = points[:, (torch.arange(size, device=device) + 4) % size]
        cut_start = _turn(before, points, quad_points[:, :, 3], normals)
        cut_end = _turn(points, quad_points[:, :, 3], after, normals)
        bent = (quad_angles >= math.pi).any(-1) | (cut_start >= math.pi) | (cut_end >= math.pi)
        cost = torch.where(bent, cost + 100.0, cost)
        ends = corners[:, window[:, 0]], corners[:, window[:, 3]]
        diagonal = torch.minimum(*ends) * stride + torch.maximum(*ends)
        cost = torch.where(_has_key(taken, diagonal), torch.full_like(cost, math.inf), cost)
        best = cost.argmin(1)
        blocked = torch.isinf(cost.gather(1, best[:, None]).squeeze(1))
        if bool(blocked.any()):
            out_sizes.append(torch.full((int(blocked.sum()),), size, dtype=torch.int64, device=device))
            out_indices.append(corners[blocked].reshape(-1))
            corners, normals = corners[~blocked], normals[~blocked]
            best, diagonal = best[~blocked], diagonal[~blocked]
            if corners.shape[0] == 0:
                continue
        taken = torch.unique(torch.cat([taken, diagonal.gather(1, best[:, None]).squeeze(1)]))
        order = (best[:, None] + torch.arange(size, device=device)) % size
        rolled = corners.gather(1, order)
        out_sizes.append(torch.full((rolled.shape[0],), 4, dtype=torch.int64, device=device))
        out_indices.append(rolled[:, :4].reshape(-1))
        rest = torch.cat([rolled[:, :1], rolled[:, 3:]], 1)
        if size - 2 <= max_size or size - 2 < 3:
            out_sizes.append(torch.full((rest.shape[0],), size - 2, dtype=torch.int64, device=device))
            out_indices.append(rest.reshape(-1))
        else:
            pending.setdefault(size - 2, []).append((rest, normals))
    return replace(mesh, sizes=torch.cat(out_sizes).long(), indices=torch.cat(out_indices).long())


def _border_vertices(mesh: PolyMesh) -> torch.Tensor:
    """``(V,)`` bool, vertices on a side with no polygon across it."""
    src, dst, _ = mesh.half_edges()
    border = twins(src, dst, mesh.count) < 0
    marked = torch.zeros(mesh.count, dtype=torch.bool, device=mesh.vertices.device)
    marked[src[border]] = True
    marked[dst[border]] = True
    return marked


def dissolve_valence_two(mesh: PolyMesh, passes: int = DISSOLVE_PASSES) -> PolyMesh:
    """Remove interior vertices joined to only two others, merging their two edges into one.

    Args:
        mesh: The mesh.
        passes: Most rounds, each removing the vertices the last round left with two neighbours.

    Returns:
        The mesh without those vertices; polygons left with fewer than three corners are dropped.
    """
    for _ in range(max(int(passes), 1)):
        edges, _ = mesh.edges()
        degree = torch.bincount(edges.reshape(-1), minlength=mesh.count)
        uses = torch.bincount(mesh.indices, minlength=mesh.count)
        dissolve = (degree == 2) & (uses == 2) & ~_border_vertices(mesh)
        if not bool(dissolve.any()):
            break
        keep_corner = ~dissolve[mesh.indices]
        sizes = torch.zeros_like(mesh.sizes).index_add_(0, mesh.corner_polygon(), keep_corner.long())
        result = replace(mesh, sizes=sizes, indices=mesh.indices[keep_corner])
        mesh = result.select(result.sizes >= 3).compacted()
    return mesh


def open_pinches(mesh: PolyMesh, max_polygons: int = PINCH_POLYGONS) -> tuple[PolyMesh, int]:
    """Remove the smaller fans of polygons at vertices where several fans meet between open sides.

    Args:
        mesh: The mesh.
        max_polygons: Most polygons a fan removed may hold; larger fans stay.

    Returns:
        ``(mesh, polygons removed)``. At each such vertex the fan with the most polygons stays.
    """
    if mesh.polygon_count == 0:
        return mesh, 0
    src, dst, polygon = mesh.half_edges()
    opened = twins(src, dst, mesh.count) < 0
    pinched = torch.nonzero(torch.bincount(dst[opened], minlength=mesh.count) > 1).squeeze(1)
    if pinched.numel() == 0:
        return mesh, 0
    previous = torch.empty_like(src)
    previous[mesh.next_corner()] = torch.arange(src.shape[0], device=src.device)
    at = torch.isin(src, pinched)
    drop = torch.zeros(mesh.polygon_count, dtype=torch.bool, device=src.device)
    corners = torch.nonzero(at).squeeze(1)
    rows = torch.stack([src[corners], polygon[corners], dst[corners], src[previous[corners]]], 1).tolist()
    by_vertex: dict[int, list[tuple[int, int, int]]] = {}
    for vertex, owner, ahead, behind in rows:
        by_vertex.setdefault(vertex, []).append((owner, ahead, behind))
    for around in by_vertex.values():
        fan = {owner: owner for owner, _, _ in around}

        def root(item: int) -> int:
            while fan[item] != item:
                fan[item] = fan[fan[item]]
                item = fan[item]
            return item

        reach: dict[int, int] = {}
        for owner, ahead, behind in around:
            for neighbour in (ahead, behind):
                if neighbour in reach:
                    fan[root(owner)] = root(reach[neighbour])
                else:
                    reach[neighbour] = owner
        groups: dict[int, list[int]] = {}
        for owner in fan:
            groups.setdefault(root(owner), []).append(owner)
        if len(groups) < 2:
            continue
        largest = max(groups.values(), key=len)
        for members in groups.values():
            if members is not largest and len(members) <= max_polygons:
                drop[members] = True
    if not bool(drop.any()):
        return mesh, 0
    return mesh.select(~drop).compacted(), int(drop.sum())


def _tangent_angles(vectors: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Angle of each vector about its normal, from a tangent fixed by the normal alone."""
    axis = torch.zeros_like(normals)
    axis[:, 0] = 1.0
    axis[normals[:, 0].abs() > 0.9] = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    q = torch.nn.functional.normalize(torch.cross(normals, axis, dim=1), dim=1)
    t = torch.cross(normals, q, dim=1)
    return torch.atan2((vectors * t).sum(1), (vectors * q).sum(1))


def hole_loops(mesh: PolyMesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The open sides of a mesh, joined into the loops round its holes.

    Args:
        mesh: The mesh.

    Returns:
        ``(sides, loop, rank, closed)``: ``(B,)`` corner of each open side, ``(B,)`` its loop,
        ``(B,)`` its step along the loop in the sides' winding, and ``(L,)`` bool for loops
        that close. At a vertex several holes touch, a side leads on to the open side next
        to it anticlockwise about the vertex normal.
    """
    src, dst, _ = mesh.half_edges()
    sides = torch.nonzero(twins(src, dst, mesh.count) < 0).squeeze(1)
    count = int(sides.shape[0])
    device = sides.device
    if count == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty, empty, torch.zeros(0, dtype=torch.bool, device=device)
    b_src, b_dst = src[sides], dst[sides]
    out_count = torch.bincount(b_src, minlength=mesh.count)
    by_src = torch.full((mesh.count,), -1, dtype=torch.int64, device=device)
    by_src[b_src] = torch.arange(count, device=device)
    successor = torch.where(out_count[b_dst] == 1, by_src[b_dst], torch.full_like(b_dst, -1))
    crowded = torch.nonzero(out_count[b_dst] > 1).squeeze(1)
    if crowded.numel():
        normals = mesh.vertex_normals()
        order = torch.argsort(b_src, stable=True)
        start = torch.cumsum(out_count, 0) - out_count
        at = b_dst[crowded]
        choices = out_count[at]
        owner = torch.repeat_interleave(torch.arange(crowded.shape[0], device=device), choices)
        step = torch.arange(int(choices.sum()), device=device) - torch.repeat_interleave(
            torch.cumsum(choices, 0) - choices, choices)
        candidate = order[start[at][owner] + step]
        centre = mesh.vertices[at]
        back = _tangent_angles(mesh.vertices[b_src[crowded]] - centre, normals[at])
        ahead = _tangent_angles(mesh.vertices[b_dst[candidate]] - centre[owner], normals[at][owner])
        turn = torch.remainder(ahead - back[owner], 2.0 * math.pi)
        turn = torch.where(turn <= 1e-9, torch.full_like(turn, 2.0 * math.pi), turn)
        least = torch.full((crowded.shape[0],), math.inf, dtype=turn.dtype, device=device)
        least.scatter_reduce_(0, owner, turn, reduce="amin")
        pick = torch.full((crowded.shape[0],), int(candidate.shape[0]), dtype=torch.int64, device=device)
        top = turn <= least[owner]
        pick.scatter_reduce_(0, owner[top], torch.nonzero(top).squeeze(1), reduce="amin")
        successor[crowded] = candidate[pick.clamp_max(candidate.shape[0] - 1)]
    index = torch.arange(count, device=device)
    linked = successor >= 0
    claims = torch.bincount(successor[linked], minlength=count)
    loop = topology.union_find(count, index[linked], successor[linked])
    _, loop = torch.unique(loop, return_inverse=True)
    loops = int(loop.max()) + 1
    closed = torch.ones(loops, dtype=torch.bool, device=device)
    closed[loop[~linked | (claims != 1)]] = False
    _, rank = order_cycles(torch.where(closed[loop], successor, index))
    return sides, loop, rank, closed


def fill_holes(mesh: PolyMesh, keep_open: torch.Tensor, max_corners: int = 256,
               opening: torch.Tensor | None = None) -> tuple[PolyMesh, int]:
    """Close holes in a mesh with one polygon each, leaving open the borders the source has.

    Args:
        mesh: The mesh.
        keep_open: ``(V,)`` bool, vertices lying on a border of the source surface. A hole
            with at least half its vertices marked stays open.
        max_corners: Holes with more corners stay open.
        opening: ``(C,)`` bool per corner's side, sides with an opening of the source beyond
            them. A hole bordered by `OPENING_SIDES` of them stays open.

    Returns:
        ``(mesh, holes filled)``. Holes are the loops of `hole_loops`; a loop passing a vertex
        twice, or running round exactly one polygon's corners, stays open.
    """
    sides, loop, rank, closed = hole_loops(mesh)
    if sides.numel() == 0:
        return mesh, 0
    src, dst, _ = mesh.half_edges()
    b_dst = dst[sides]
    device = sides.device
    loops = int(closed.shape[0])
    stride = max(mesh.count, 1)
    size = torch.bincount(loop, minlength=loops)
    distinct = torch.bincount(torch.unique(loop * stride + b_dst) // stride, minlength=loops)
    opened = torch.zeros(loops, dtype=torch.float32, device=device).index_add_(0, loop, keep_open[b_dst].float())
    fill = closed & (distinct == size) & (size >= 3) & (size <= max_corners) & (opened < 0.5 * size.float())
    if opening is not None:
        beyond = torch.zeros(loops, dtype=torch.int64, device=device).index_add_(0, loop, opening[sides].long())
        fill &= beyond < OPENING_SIDES
    chosen = fill[loop]
    if not bool(chosen.any()):
        return mesh, 0
    order = torch.argsort(loop[chosen] * (sides.shape[0] + 1) + (sides.shape[0] - rank[chosen]))
    corners = b_dst[chosen][order]
    sizes = size[fill]
    fresh = ~repeats_polygon(mesh, sizes, corners)
    if not bool(fresh.any()):
        return mesh, 0
    corners = corners[torch.repeat_interleave(fresh, sizes)]
    sizes = sizes[fresh]
    result = replace(mesh, sizes=torch.cat([mesh.sizes, sizes]), indices=torch.cat([mesh.indices, corners]))
    return result, int(sizes.shape[0])


def _pieces(mesh: PolyMesh) -> tuple[torch.Tensor, int]:
    """``(P,)`` int64 piece of each polygon, polygons sharing a side being one piece, and the piece count."""
    src, dst, polygon = mesh.half_edges()
    twin = twins(src, dst, mesh.count)
    across = twin >= 0
    piece = topology.union_find(mesh.polygon_count, polygon[across], polygon[twin[across]])
    _, piece = torch.unique(piece, return_inverse=True)
    return piece, int(piece.max()) + 1 if piece.numel() else 0


def _polygon_areas(mesh: PolyMesh) -> torch.Tensor:
    """``(P,)`` area of each polygon, summed over its triangles."""
    faces, owner = mesh.triangulate()
    area = torch.zeros(mesh.polygon_count, dtype=mesh.vertices.dtype, device=mesh.vertices.device)
    return area.index_add_(0, owner, topology.face_areas(mesh.vertices, faces).to(area.dtype))


def drop_fragments(mesh: PolyMesh, source_piece: torch.Tensor, max_share: float = 0.005) -> tuple[PolyMesh, int]:
    """Remove small pieces lying on a source piece that a larger piece of the mesh already covers.

    Args:
        mesh: The mesh.
        source_piece: ``(V,)`` int64 piece of the source surface under each vertex, -1 where none.
        max_share: Largest share of the mesh's area a piece removed may cover.

    Returns:
        ``(mesh, pieces removed)``.
    """
    if mesh.polygon_count == 0:
        return mesh, 0
    piece, pieces = _pieces(mesh)
    label = source_piece[mesh.indices]
    valid = label >= 0
    if pieces < 2 or not bool(valid.any()):
        return mesh, 0
    device = mesh.indices.device
    labels = int(label.max()) + 1
    piece_area = torch.zeros(pieces, dtype=mesh.vertices.dtype, device=device)
    piece_area.index_add_(0, piece, _polygon_areas(mesh))
    corner_piece = piece[mesh.corner_polygon()]
    keys, votes = torch.unique(corner_piece[valid] * labels + label[valid], return_counts=True)
    voter, choice = keys // labels, keys % labels
    top = torch.zeros(pieces, dtype=votes.dtype, device=device).scatter_reduce_(0, voter, votes, reduce="amax")
    winner = votes == top[voter]
    under = torch.full((pieces,), labels, dtype=torch.int64, device=device)
    under.scatter_reduce_(0, voter[winner], choice[winner], reduce="amin")
    placed = under < labels
    largest = torch.full((labels,), -math.inf, dtype=piece_area.dtype, device=device)
    largest.scatter_reduce_(0, under[placed], piece_area[placed], reduce="amax")
    covered = placed & (piece_area < largest[under.clamp_max(labels - 1)])
    fragment = covered & (piece_area < max_share * piece_area.sum())
    if not bool(fragment.any()):
        return mesh, 0
    return mesh.select(~fragment[piece]).compacted(), int(fragment.sum())


def drop_stray_shells(mesh: PolyMesh, keep_open: torch.Tensor, max_share: float = 0.005) -> tuple[PolyMesh, int]:
    """Remove small open pieces cut loose from the surface, keeping closed pieces and the largest.

    Args:
        mesh: The mesh.
        keep_open: ``(V,)`` bool, vertices lying on a border of the source surface; a side between
            two of them does not count as open.
        max_share: Largest share of the surface's area a piece removed may cover.

    Returns:
        ``(mesh, pieces removed)``.
    """
    if mesh.polygon_count == 0:
        return mesh, 0
    src, dst, polygon = mesh.half_edges()
    across = twins(src, dst, mesh.count) >= 0
    piece, pieces = _pieces(mesh)
    if pieces < 2:
        return mesh, 0
    piece_area = torch.zeros(pieces, dtype=mesh.vertices.dtype, device=mesh.vertices.device)
    piece_area.index_add_(0, piece, _polygon_areas(mesh))
    open_side = ~across & ~(keep_open[src] & keep_open[dst])
    opened = torch.zeros(pieces, dtype=torch.bool, device=piece_area.device)
    opened[piece[polygon[open_side]]] = True
    stray = opened & (piece_area < max_share * piece_area.sum())
    stray[int(piece_area.argmax())] = False
    if not bool(stray.any()):
        return mesh, 0
    return mesh.select(~stray[piece]).compacted(), int(stray.sum())


def drop_non_manifold(mesh: PolyMesh) -> tuple[PolyMesh, int]:
    """Remove polygons so no side is shared by more than two polygons or twice in one direction.

    Args:
        mesh: The mesh.

    Returns:
        ``(mesh, polygons removed)``.
    """
    # On each crowded side the polygons whose corners stray furthest from right angles go first.
    if mesh.polygon_count == 0:
        return mesh, 0
    device = mesh.indices.device
    src, dst, polygon = mesh.half_edges()
    stride = max(mesh.count, 1)
    undirected = torch.minimum(src, dst) * stride + torch.maximum(src, dst)
    directed = src * stride + dst
    _, edge, uses = torch.unique(undirected, return_inverse=True, return_counts=True)
    _, way, way_uses = torch.unique(directed, return_inverse=True, return_counts=True)
    crowded = (uses[edge] > 2) | (way_uses[way] > 1)
    if not bool(crowded.any()):
        return mesh, 0
    offsets = mesh.offsets()
    corner = torch.arange(mesh.indices.shape[0], device=device)
    behind = torch.where(corner == offsets[polygon], offsets[polygon + 1] - 1, corner - 1)
    nxt = mesh.next_corner()
    a, b, c = mesh.vertices[mesh.indices[behind]], mesh.vertices[mesh.indices], mesh.vertices[mesh.indices[nxt]]
    turn = torch.atan2(torch.cross(c - b, a - b, dim=1).norm(dim=1), ((c - b) * (a - b)).sum(1))
    target = torch.where(mesh.sizes[polygon] == 3, math.pi / 3, math.pi / 2)
    badness = torch.zeros(mesh.polygon_count, dtype=turn.dtype, device=device)
    badness.scatter_reduce_(0, polygon, (turn - target).abs(), reduce="amax")
    rows = torch.nonzero(crowded).squeeze(1)
    key = torch.where(way_uses[way[rows]] > 1, way[rows] * 2, edge[rows] * 2 + 1)
    order = torch.argsort(badness[polygon[rows]], stable=True)
    order = order[torch.argsort(key[order], stable=True)]
    rows, key = rows[order], key[order]
    start = torch.ones_like(key, dtype=torch.bool)
    start[1:] = key[1:] != key[:-1]
    first = torch.cummax(torch.where(start, torch.arange(key.shape[0], device=device), 0), 0).values
    rank = torch.arange(key.shape[0], device=device) - first
    allowed = torch.where(key % 2 == 0, 1, 2)
    drop = torch.zeros(mesh.polygon_count, dtype=torch.bool, device=device)
    drop[polygon[rows[rank >= allowed]]] = True
    return mesh.select(~drop).compacted(), int(drop.sum())


def _chords(mesh: PolyMesh) -> tuple[torch.Tensor, torch.Tensor]:
    """Order-free keys of every pair of corners of one polygon that no side joins, and their polygon."""
    device = mesh.indices.device
    offsets = mesh.offsets()
    stride = max(mesh.count, 1)
    keys, owners = [mesh.indices[:0]], [mesh.indices[:0]]
    for size in torch.unique(mesh.sizes[mesh.sizes >= 4]).tolist():
        rows = torch.nonzero(mesh.sizes == size).squeeze(1)
        corners = mesh.indices[offsets[rows][:, None] + torch.arange(size, device=device)]
        pairs = [(i, j) for i in range(size) for j in range(i + 2, size) if (i, j) != (0, size - 1)]
        first = corners[:, [i for i, _ in pairs]]
        second = corners[:, [j for _, j in pairs]]
        keys.append((torch.minimum(first, second) * stride + torch.maximum(first, second)).reshape(-1))
        owners.append(rows.repeat_interleave(len(pairs)))
    return torch.cat(keys), torch.cat(owners)


def flaws(mesh: PolyMesh) -> dict[str, int]:
    """Counts of what keeps a polygon mesh from being a clean surface.

    Args:
        mesh: The mesh.

    Returns:
        ``"duplicate polygons"`` (beyond the first over the same corners), ``"folded polygons"``
        (as `drop_folds` finds them), ``"non-manifold vertices"`` and ``"degenerate polygons"``
        (no area, or a corner repeated).
    """
    result = {"duplicate polygons": 0, "folded polygons": 0, "non-manifold vertices": 0, "degenerate polygons": 0}
    if mesh.polygon_count == 0:
        return result
    offsets = mesh.offsets()
    stride = max(mesh.count, 1)
    owner = mesh.corner_polygon()
    distinct = torch.bincount(torch.unique(owner * stride + mesh.indices) // stride, minlength=mesh.polygon_count)
    for size in torch.unique(mesh.sizes).tolist():
        rows = torch.nonzero(mesh.sizes == size).squeeze(1)
        corners = mesh.indices[offsets[rows][:, None] + torch.arange(size, device=rows.device)].sort(1).values
        result["duplicate polygons"] += int(rows.shape[0] - torch.unique(corners, dim=0).shape[0])
    chords, chord_owner = _chords(mesh)
    if chords.numel():
        src, dst, _ = mesh.half_edges()
        sides = torch.unique(torch.minimum(src, dst) * stride + torch.maximum(src, dst))
        _, inverse, spans = torch.unique(chords, return_inverse=True, return_counts=True)
        folded = _has_key(sides, chords) | (spans[inverse] > 1)
        result["folded polygons"] = int(torch.unique(chord_owner[folded]).shape[0])
    faces, _ = mesh.triangulate()
    if faces.shape[0]:
        result["non-manifold vertices"] = int(topology.non_manifold_vertices(faces, mesh.count).sum())
    flat = _polygon_areas(mesh) <= 1e-12 * float((mesh.vertices.amax(0) - mesh.vertices.amin(0)).norm()) ** 2
    result["degenerate polygons"] = int((flat | (distinct != mesh.sizes)).sum())
    return result


def drop_folds(mesh: PolyMesh) -> tuple[PolyMesh, int]:
    """Remove polygons folded over each other: two corners of one polygon joined by another's side,
    or the same two corners spanned by two polygons.

    Args:
        mesh: The mesh.

    Returns:
        ``(mesh, polygons removed)``. The polygon spanning the pair and every polygon with that
        pair as a side are removed.
    """
    if mesh.polygon_count == 0:
        return mesh, 0
    chords, owner = _chords(mesh)
    if chords.numel() == 0:
        return mesh, 0
    src, dst, polygon = mesh.half_edges()
    stride = max(mesh.count, 1)
    side_keys = torch.minimum(src, dst) * stride + torch.maximum(src, dst)
    sides = torch.unique(side_keys)
    _, inverse, spans = torch.unique(chords, return_inverse=True, return_counts=True)
    folded = _has_key(sides, chords) | (spans[inverse] > 1)
    if not bool(folded.any()):
        return mesh, 0
    drop = torch.zeros(mesh.polygon_count, dtype=torch.bool, device=src.device)
    drop[owner[folded]] = True
    drop[polygon[_has_key(torch.unique(chords[folded]), side_keys)]] = True
    return mesh.select(~drop).compacted(), int(drop.sum())


def merge_triangles(mesh: PolyMesh, max_deviation_deg: float = 40.0,
                    max_fold_deg: float = 60.0) -> PolyMesh:
    """Join pairs of neighbouring triangles into quads where the quad comes out close to square.

    Args:
        mesh: The mesh.
        max_deviation_deg: Most any corner of the joined quad may depart from 90 degrees.
        max_fold_deg: Most the two triangles may be folded against each other.

    Returns:
        The mesh with the joined pairs replaced by quads. A pair whose far corners an edge
        already joins stays apart.
    """
    triangle = mesh.sizes == 3
    if int(triangle.sum()) < 2:
        return mesh
    src, dst, polygon = mesh.half_edges()
    twin = twins(src, dst, mesh.count)
    nxt = mesh.next_corner()
    corner = torch.arange(src.shape[0], device=src.device)
    pair = (twin > corner) & triangle[polygon] & triangle[polygon[twin.clamp_min(0)]]
    pair &= polygon != polygon[twin.clamp_min(0)]
    h = torch.nonzero(pair).squeeze(1)
    if h.numel() == 0:
        return mesh
    other = twin[h]
    third_p = mesh.indices[nxt[nxt[h]]]
    third_q = mesh.indices[nxt[nxt[other]]]
    quad = torch.stack([dst[h], third_p, src[h], third_q], 1)
    points = mesh.vertices[quad]
    badness = _quad_badness(points)
    na = torch.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], dim=1)
    nb = torch.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0], dim=1)
    fold = torch.acos(torch.nn.functional.cosine_similarity(na, nb, dim=1).clamp(-1.0, 1.0))
    stride = max(mesh.count, 1)
    across = torch.minimum(third_p, third_q) * stride + torch.maximum(third_p, third_q)
    open_across = (third_p != third_q) & ~_has_key(_edge_keys(mesh), across)
    allowed = (badness <= math.radians(max_deviation_deg)) & (fold <= math.radians(max_fold_deg)) & open_across
    h, other, quad, badness = h[allowed], other[allowed], quad[allowed], badness[allowed]
    if h.numel() == 0:
        return mesh
    p, q = polygon[h], polygon[other]
    score = -badness
    partner = match_pairs(mesh.polygon_count, torch.cat([p, q]), torch.cat([q, p]), torch.cat([score, score]))
    chosen = (partner[p] == q) & (p < q)
    quad = quad[chosen]
    merged = torch.zeros(mesh.polygon_count, dtype=torch.bool, device=src.device)
    merged[p[chosen]] = True
    merged[q[chosen]] = True
    rest = mesh.select(~merged)
    return replace(rest, sizes=torch.cat([rest.sizes, torch.full((quad.shape[0],), 4, device=src.device)]),
                   indices=torch.cat([rest.indices, quad.reshape(-1)]))


def subdivide_quads(mesh: PolyMesh) -> PolyMesh:
    """Split every polygon of ``n`` corners into ``n`` quads through its edge midpoints and centre.

    Args:
        mesh: The mesh.

    Returns:
        An all-quad mesh. A midpoint carries a true/false mark other than ``"corner"`` when
        both its edge's ends do.
    """
    device = mesh.indices.device
    edges, side_edge = mesh.edges()
    nxt = mesh.next_corner()
    polygon = mesh.corner_polygon()
    previous = torch.empty_like(nxt)
    previous[nxt] = torch.arange(nxt.shape[0], device=device)
    edge_base = mesh.count
    centre_base = mesh.count + int(edges.shape[0])
    quads = torch.stack([
        mesh.indices,
        edge_base + side_edge,
        centre_base + polygon,
        edge_base + side_edge[previous],
    ], 1)
    marks = {name: value[edges[:, 0]] & value[edges[:, 1]] for name, value in mesh.marks.items()
             if value.dtype == torch.bool and name != "corner"}
    result = PolyMesh(mesh.vertices, torch.full((quads.shape[0],), 4, dtype=torch.int64, device=device),
                      quads.reshape(-1), dict(mesh.marks))
    result = result.appended(mesh.vertices[edges].mean(1), edges, marks)
    offsets = mesh.offsets()
    slot = torch.arange(int(mesh.sizes.max()) if mesh.polygon_count else 0, device=device)
    reach = (offsets[:-1, None] + slot).clamp_max(max(mesh.indices.shape[0] - 1, 0))
    inside = slot[None, :] < mesh.sizes[:, None]
    sources = torch.where(inside, mesh.indices[reach], mesh.indices[offsets[:-1]][:, None])
    return result.appended(mesh.centroids(), sources)


def triangles_only(mesh: PolyMesh) -> PolyMesh:
    """The same surface with every polygon cut into triangles."""
    faces, _ = mesh.triangulate()
    return replace(mesh, sizes=torch.full((faces.shape[0],), 3, dtype=torch.int64, device=faces.device),
                   indices=faces.reshape(-1))


def _project(points: torch.Tensor, bvh: TriangleBVH, normals: torch.Tensor | None = None,
             reach: torch.Tensor | None = None, facing: torch.Tensor | None = None) -> torch.Tensor:
    """Points moved onto a tree's surface.

    Args:
        points: ``(Q, 3)`` positions.
        bvh: The surface.
        normals: ``(Q, 3)`` unit directions searched along both ways first, or None.
        reach: ``(Q,)`` longest search along the normals.
        facing: ``(F, 3)`` unit normals of the tree's triangles; with ``normals``, only
            triangles facing the same way as the point are landed on.

    Returns:
        ``(Q, 3)`` the nearer hit along the normal, else the closest surface point, else the
        point unmoved.
    """
    if normals is None or reach is None or points.shape[0] == 0:
        _, face, _, closest = bvh.closest_point(points)
        return torch.where((face >= 0)[:, None], closest.to(points.dtype), points)

    def agrees(face: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        if facing is None:
            return face >= 0
        return (face >= 0) & ((facing[face.clamp_min(0)].to(points.dtype) * directions).sum(1) > 0)

    ahead, ahead_face, _ = bvh.ray_cast(points, normals, t_max=reach)
    behind, behind_face, _ = bvh.ray_cast(points, -normals, t_max=reach)
    ahead, behind = ahead.to(points.dtype), behind.to(points.dtype)
    ahead = torch.where(agrees(ahead_face, normals), ahead, torch.full_like(ahead, math.inf))
    behind = torch.where(agrees(behind_face, normals), behind, torch.full_like(behind, math.inf))
    use_ahead = torch.isfinite(ahead) & (ahead <= behind)
    use_behind = torch.isfinite(behind) & ~use_ahead
    result = torch.where(use_ahead[:, None], points + normals * ahead.nan_to_num(0.0, posinf=0.0)[:, None], points)
    result = torch.where(use_behind[:, None], points - normals * behind.nan_to_num(0.0, posinf=0.0)[:, None], result)
    missed = torch.nonzero(~use_ahead & ~use_behind).squeeze(1)
    if missed.numel():
        _, face, _, closest = bvh.closest_point(points[missed])
        good = agrees(face, normals[missed])
        result[missed] = torch.where(good[:, None], closest.to(points.dtype), points[missed])
    return result


def feature_pairs(mesh: PolyMesh, feature_bvh: TriangleBVH | None, tolerance: float = 0.25) -> torch.Tensor:
    """Edges of the mesh lying along a feature line.

    Args:
        mesh: The mesh, with a ``"feature"`` mark.
        feature_bvh: Tree over the feature segments, or None.
        tolerance: Largest midpoint distance from the features, as a share of the edge length.

    Returns:
        ``(K, 2)`` int64 vertex pairs.
    """
    edges, _ = mesh.edges()
    feature = mesh.marks.get("feature")
    if feature is None or feature_bvh is None or edges.shape[0] == 0:
        return edges[:0]
    both = feature[edges[:, 0]] & feature[edges[:, 1]]
    edges = edges[both]
    if edges.shape[0] == 0:
        return edges
    middle = mesh.vertices[edges].mean(1)
    length = (mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]).norm(dim=1)
    distance, _, _, _ = feature_bvh.closest_point(middle)
    return edges[distance.to(length.dtype) <= tolerance * length]


def relax(mesh: PolyMesh, surface_bvh: TriangleBVH, feature_bvh: TriangleBVH | None, iterations: int,
          step: float = 0.5, plane_axis: int | None = None,
          surface_normals: torch.Tensor | None = None) -> PolyMesh:
    """Even out vertex spacing along the surface and settle every vertex back onto it.

    Args:
        mesh: The mesh, with ``"feature"`` and ``"corner"`` marks.
        surface_bvh: Tree over the source surface.
        feature_bvh: Tree over the feature segments, or None.
        iterations: Smoothing passes; 0 only projects.
        step: Share of the way to the neighbour average moved each pass.
        plane_axis: Axis whose zero plane holds the vertices marked ``"centre"``, or None.
        surface_normals: ``(F, 3)`` unit normals of the surface triangles, so vertices land
            only on the side they face; None lands on either side.

    Returns:
        The mesh with new positions.
    """
    # Vertices marked ``"corner"`` stay put, vertices on a feature line slide along it and the rest
    # move in their tangent plane, each followed by a projection.
    device = mesh.vertices.device
    count = mesh.count
    corner = mesh.marks.get("corner", torch.zeros(count, dtype=torch.bool, device=device))
    feature = mesh.marks.get("feature", torch.zeros(count, dtype=torch.bool, device=device))
    centre = mesh.marks.get("centre", torch.zeros(count, dtype=torch.bool, device=device))
    pairs = feature_pairs(mesh, feature_bvh)
    along = torch.bincount(pairs.reshape(-1), minlength=count)
    slide = feature & ~corner & (along == 2)
    if feature_bvh is None:
        slide = torch.zeros_like(slide)
    fixed = corner | (feature & ~slide)
    free = ~slide & ~fixed
    edges, _ = mesh.edges()
    a = torch.cat([edges[:, 0], edges[:, 1]])
    b = torch.cat([edges[:, 1], edges[:, 0]])
    degree = torch.bincount(a, minlength=count).clamp_min(1).to(mesh.vertices.dtype)
    fa = torch.cat([pairs[:, 0], pairs[:, 1]])
    fb = torch.cat([pairs[:, 1], pairs[:, 0]])
    faces, owner = mesh.triangulate()
    scale = mesh.marks.get("scale")
    reach = (0.5 * scale).to(mesh.vertices.dtype) if scale is not None else None
    points = mesh.vertices.clone()
    points = _apart(points, _settle(points, free, slide, surface_bvh, feature_bvh, _normals(points, faces, count),
                                    reach, surface_normals))
    for _ in range(max(int(iterations), 0)):
        normals = _normals(points, faces, count)
        mean = torch.zeros_like(points).index_add_(0, a, points[b]) / degree[:, None]
        move = mean - points
        move = move - normals * (move * normals).sum(1, keepdim=True)
        moved = points + step * move
        if fa.numel():
            line_mean = torch.zeros_like(points).index_add_(0, fa, points[fb]) / along.clamp_min(1)[:, None].to(points.dtype)
            moved = torch.where(slide[:, None], points + step * (line_mean - points), moved)
        moved = torch.where(fixed[:, None], points, moved)
        if plane_axis is not None:
            moved[centre, plane_axis] = 0.0
        moved = _settle(moved, free, slide, surface_bvh, feature_bvh, normals, reach, surface_normals)
        if plane_axis is not None:
            moved[centre, plane_axis] = 0.0
        points = _unfold(points, _apart(points, moved), faces, owner, count)
    return replace(mesh, vertices=points)


def _apart(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """New positions, with the vertices a move put on the very spot of another put back."""
    _, inverse, counts = torch.unique(after, dim=0, return_inverse=True, return_counts=True)
    crowded = counts[inverse] > 1
    if not bool(crowded.any()):
        return after
    return torch.where(crowded[:, None], before, after)


def _normals(points: torch.Tensor, faces: torch.Tensor, count: int) -> torch.Tensor:
    """``(V, 3)`` area-weighted unit vertex normals of triangles."""
    face_normal = topology.face_normals(points, faces, normalize=False)
    total = torch.zeros(count, 3, dtype=points.dtype, device=points.device)
    total.index_add_(0, faces.reshape(-1), face_normal.repeat_interleave(3, dim=0))
    return torch.nn.functional.normalize(total, dim=1)


def _unfold(before: torch.Tensor, after: torch.Tensor, faces: torch.Tensor, owner: torch.Tensor,
            count: int) -> torch.Tensor:
    """New positions, with the vertices of polygons a move turned over put back."""
    polygons = int(owner.max()) + 1 if owner.numel() else 0
    def polygon_normal(points: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(polygons, 3, dtype=points.dtype, device=points.device)
        return total.index_add_(0, owner, topology.face_normals(points, faces, normalize=False))
    old = polygon_normal(before)
    turned = (polygon_normal(after) * old).sum(1) < 0
    if not bool(turned.any()):
        return after
    back = torch.zeros(count, dtype=torch.bool, device=before.device)
    back[faces[turned[owner]].reshape(-1)] = True
    return torch.where(back[:, None], before, after)


def _settle(points: torch.Tensor, free: torch.Tensor, slide: torch.Tensor, surface_bvh: TriangleBVH,
            feature_bvh: TriangleBVH | None, normals: torch.Tensor | None = None,
            reach: torch.Tensor | None = None, facing: torch.Tensor | None = None) -> torch.Tensor:
    """Project free vertices onto the surface and sliding vertices onto the feature lines."""
    points = points.clone()
    if bool(free.any()):
        points[free] = _project(points[free], surface_bvh, None if normals is None else normals[free],
                                None if reach is None else reach[free], facing)
    if feature_bvh is not None and bool(slide.any()):
        points[slide] = _project(points[slide], feature_bvh)
    return points
