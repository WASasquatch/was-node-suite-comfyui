"""Triangle bounding volume hierarchy for closest point queries and ray casts in torch.

Leaves hold a fixed triangle count, split at each node's widest axis median. Node ``i`` has
children ``2i + 1`` and ``2i + 2``.
"""

from __future__ import annotations

__all__ = ["TriangleBVH", "closest_point_on_triangles", "intersect_ray_triangles", "morton_codes"]

import math
from typing import Callable

import torch

from . import device as mesh_device

#: Bits per axis in a Morton code.
MORTON_BITS = 21

#: Most stack entries one query pops in a traversal step.
MAX_POPS = 8

#: Rows a traversal step aims to process when few queries remain.
MIN_ROWS = 16384

#: Relative size below which a triangle counts as degenerate in a closest point query.
DEGENERATE = 1e-6

#: Relative widening of box tests against float rounding.
BOX_PAD = 1e-6


def _expand_bits(value: torch.Tensor) -> torch.Tensor:
    """Spread the low 21 bits of an int64 so two zero bits follow each one."""
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def morton_codes(points: torch.Tensor, low: torch.Tensor | None = None, high: torch.Tensor | None = None) -> torch.Tensor:
    """63-bit Morton codes of points inside a box.

    Args:
        points: ``(N, 3)`` positions.
        low: ``(3,)`` box minimum, the points' minimum when None.
        high: ``(3,)`` box maximum, the points' maximum when None.

    Returns:
        ``(N,)`` int64 codes.
    """
    if points.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int64, device=points.device)
    low = points.amin(0) if low is None else low
    high = points.amax(0) if high is None else high
    span = (high - low).clamp_min(torch.finfo(points.dtype).tiny)
    top = (1 << MORTON_BITS) - 1
    cells = ((points - low) / span * top).nan_to_num(0.0).clamp(0, top).long()
    return (_expand_bits(cells[:, 0]) << 2) | (_expand_bits(cells[:, 1]) << 1) | _expand_bits(cells[:, 2])


def _dot(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Dot product over the last axis."""
    return (first * second).sum(-1)


def _segment_closest(points, start, end):
    """Squared distance and parameter of the closest point on segments."""
    direction = end - start
    length_sq = _dot(direction, direction)
    t = _dot(points - start, direction) / torch.where(length_sq > 0, length_sq, torch.ones_like(length_sq))
    t = t.clamp(0.0, 1.0)
    offset = points - (start + direction * t[..., None])
    return _dot(offset, offset), t


def closest_point_on_triangles(
    points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact closest point on triangles, by Voronoi region of the triangle's features.

    Args:
        points: ``(..., 3)`` query points, broadcast against the corners.
        a: ``(..., 3)`` first corners.
        b: ``(..., 3)`` second corners.
        c: ``(..., 3)`` third corners.

    Returns:
        ``(distance_sq, barycentric)``: squared distances ``(...)`` and weights ``(..., 3)``
        of ``a``, ``b`` and ``c`` at the closest point.
    """
    points, a, b, c = torch.broadcast_tensors(points, a, b, c)
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    one = torch.ones_like(d1)
    zero = torch.zeros_like(d1)

    def ratio(top, bottom):
        return top / torch.where(bottom != 0, bottom, one)

    denom = va + vb + vc
    v = ratio(vb, denom)
    w = ratio(vc, denom)
    bary = torch.stack([1.0 - v - w, v, w], dim=-1)

    regions = []
    w_bc = ratio(d4 - d3, (d4 - d3) + (d5 - d6))
    regions.append(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), torch.stack([zero, 1.0 - w_bc, w_bc], -1)))
    w_ac = ratio(d2, d2 - d6)
    regions.append(((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack([1.0 - w_ac, zero, w_ac], -1)))
    regions.append(((d6 >= 0) & (d5 <= d6), torch.stack([zero, zero, one], -1)))
    v_ab = ratio(d1, d1 - d3)
    regions.append(((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack([1.0 - v_ab, v_ab, zero], -1)))
    regions.append(((d3 >= 0) & (d4 <= d3), torch.stack([zero, one, zero], -1)))
    regions.append(((d1 <= 0) & (d2 <= 0), torch.stack([one, zero, zero], -1)))
    for mask, weights in regions:
        bary = torch.where(mask[..., None], weights, bary)

    # Degenerate and sliver triangles also try their three edges and keep the nearest point.
    thin = denom <= DEGENERATE * _dot(ab, ab) * _dot(ac, ac)
    if bool(thin.any()):
        p, ta, tb, tc = points[thin], a[thin], b[thin], c[thin]
        current = bary[thin]
        region_point = ta * current[:, 0:1] + tb * current[:, 1:2] + tc * current[:, 2:3]
        region_distance = _dot(p - region_point, p - region_point)
        candidates = [_segment_closest(p, ta, tb), _segment_closest(p, tb, tc), _segment_closest(p, tc, ta)]
        distance = torch.stack([item[0] for item in candidates], -1)
        nearest, pick = distance.min(-1)
        t = torch.stack([item[1] for item in candidates], -1).gather(-1, pick[:, None]).squeeze(-1)
        rows = torch.zeros(p.shape[0], 3, dtype=p.dtype, device=p.device)
        rows.scatter_(1, pick[:, None], (1.0 - t)[:, None])
        rows.scatter_add_(1, ((pick + 1) % 3)[:, None], t[:, None])
        bary[thin] = torch.where((nearest < region_distance)[:, None], rows, current)

    closest = a * bary[..., 0:1] + b * bary[..., 1:2] + c * bary[..., 2:3]
    offset = points - closest
    return _dot(offset, offset), bary


def intersect_ray_triangles(
    origins: torch.Tensor,
    directions: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    t_min: float | torch.Tensor = 0.0,
    t_max: float | torch.Tensor = math.inf,
    cull_backfaces: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Watertight ray and triangle intersection in a ray-sheared frame.

    Args:
        origins: ``(..., 3)`` ray origins, broadcast against the corners.
        directions: ``(..., 3)`` ray directions; length scales ``t``.
        a: ``(..., 3)`` first corners.
        b: ``(..., 3)`` second corners.
        c: ``(..., 3)`` third corners.
        t_min: Nearest accepted ray parameter.
        t_max: Farthest accepted ray parameter.
        cull_backfaces: Ignore triangles whose counter-clockwise side faces away from the ray.

    Returns:
        ``(t, barycentric)``: ray parameters ``(...)``, infinity on a miss, and weights
        ``(..., 3)`` of ``a``, ``b`` and ``c``. A ray through a shared edge hits both triangles.
    """
    origins, directions = torch.broadcast_tensors(origins, directions)
    # Axis order per ray: the dominant direction axis last, the other two kept right-handed.
    major = directions.abs().argmax(-1, keepdim=True)
    first = (major + 1) % 3
    second = (first + 1) % 3
    along = directions.gather(-1, major)
    flip = along < 0
    first, second = torch.where(flip, second, first), torch.where(flip, first, second)
    axes = torch.cat([first, second, major], -1)
    rank = max(origins.dim(), a.dim(), b.dim(), c.dim())
    axes = axes.reshape((1,) * (rank - axes.dim()) + tuple(axes.shape))
    safe = torch.where(along == 0, torch.ones_like(along), along)
    shear_x = (directions.gather(-1, first) / safe).squeeze(-1)
    shear_y = (directions.gather(-1, second) / safe).squeeze(-1)
    shear_z = (1.0 / safe).squeeze(-1)

    def sheared(corner):
        relative = torch.take_along_dim(corner - origins, axes, -1)
        depth = relative[..., 2]
        return relative[..., 0] - shear_x * depth, relative[..., 1] - shear_y * depth, shear_z * depth

    ax, ay, az = sheared(a)
    bx, by, bz = sheared(b)
    cx, cy, cz = sheared(c)
    u = cx * by - cy * bx
    v = ax * cy - ay * cx
    w = bx * ay - by * ax
    det = u + v + w
    inside = ((u >= 0) & (v >= 0) & (w >= 0)) | ((u <= 0) & (v <= 0) & (w <= 0))
    usable = (det > 0) if cull_backfaces else (det != 0)
    usable = usable & inside & (along.squeeze(-1) != 0)
    inverse = 1.0 / torch.where(usable, det, torch.ones_like(det))
    t = (u * az + v * bz + w * cz) * inverse
    hit = usable & (t >= t_min) & (t <= t_max)
    t = torch.where(hit, t, torch.full_like(t, math.inf))
    bary = torch.stack([u * inverse, v * inverse, w * inverse], dim=-1)
    return t, torch.where(hit[..., None], bary, torch.zeros_like(bary))


class TriangleBVH:
    """Linear bounding volume hierarchy over a triangle mesh.

    Attributes:
        device: Device the tree and queries run on.
        face_count: Number of triangles.
        leaf_size: Triangles per leaf.
        depth: Levels below the root.
        lower: ``(3,)`` mesh bounding box minimum.
        upper: ``(3,)`` mesh bounding box maximum.
        centre: ``(3,)`` float64 centre of the box the tree is stored relative to.
        scale: Half the box's longest side, the tree's unit length.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        device: torch.device | str | None = None,
        leaf_size: int = 16,
    ):
        """Build the tree.

        Args:
            vertices: ``(V, 3)`` positions.
            faces: ``(F, 3)`` vertex indices. Faces with a non-finite corner are never hit.
            device: Device for the tree; None means the mesh compute device.
            leaf_size: Triangles per leaf.

        Raises:
            ValueError: For faces indexing past the vertices.
        """
        self.device = torch.device(device) if device is not None else mesh_device.compute_device()
        vertices = vertices.to(self.device)
        if not vertices.is_floating_point():
            vertices = vertices.float()
        faces = faces.to(self.device).long()
        self.face_count = int(faces.shape[0])
        self.leaf_size = max(1, int(leaf_size))
        if self.face_count and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
            raise ValueError(
                "the mesh has faces that index past its vertices; repair or rebuild the mesh "
                "before querying it"
            )
        self._vertices = vertices
        self._faces = faces
        self.centre = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.scale = 1.0
        if self.face_count == 0:
            self.depth = 0
            self.lower = torch.zeros(3, device=self.device)
            self.upper = torch.zeros(3, device=self.device)
            return
        finite_vertex = torch.isfinite(vertices).all(1)
        usable = finite_vertex[faces].all(1)
        if bool(usable.any()):
            used = vertices[faces[usable].reshape(-1)].double()
            low, high = used.amin(0), used.amax(0)
        else:
            low = high = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.lower = low.float()
        self.upper = high.float()
        self.centre = (low + high) / 2
        scale = float((high - low).max()) / 2
        self.scale = scale if math.isfinite(scale) and scale > 0 else 1.0
        local = ((vertices.double() - self.centre) / self.scale).float()
        corners = local[faces]
        centroids = torch.where(usable[:, None], corners.mean(1), torch.zeros(3, device=self.device))
        leaves = math.ceil(self.face_count / self.leaf_size)
        self.leaf_count = 1 << max(0, (leaves - 1).bit_length())
        self.depth = self.leaf_count.bit_length() - 1
        self.first_leaf = self.leaf_count - 1
        slots = self.leaf_count * self.leaf_size
        self.slot_face = self._median_order(centroids, slots)
        self.slot_valid = (self.slot_face >= 0) & usable[self.slot_face.clamp_min(0)]
        slot_corners = corners[self.slot_face.clamp_min(0)]
        self.slot_a = slot_corners[:, 0].contiguous()
        self.slot_b = slot_corners[:, 1].contiguous()
        self.slot_c = slot_corners[:, 2].contiguous()

        valid = self.slot_valid[:, None, None]
        leaf_low = torch.where(valid, slot_corners, math.inf).reshape(self.leaf_count, -1, 3).amin(1)
        leaf_high = torch.where(valid, slot_corners, -math.inf).reshape(self.leaf_count, -1, 3).amax(1)
        empty = ~self.slot_valid.reshape(self.leaf_count, -1).any(1)
        leaf_low[empty] = math.nan
        leaf_high[empty] = math.nan
        node_count = 2 * self.leaf_count - 1
        self.node_low = torch.empty(node_count, 3, dtype=torch.float32, device=self.device)
        self.node_high = torch.empty(node_count, 3, dtype=torch.float32, device=self.device)
        self.node_low[self.first_leaf:] = leaf_low
        self.node_high[self.first_leaf:] = leaf_high
        size = self.leaf_count // 2
        while size >= 1:
            children = slice(2 * size - 1, 4 * size - 1)
            pair_low = self.node_low[children].reshape(size, 2, 3)
            pair_high = self.node_high[children].reshape(size, 2, 3)
            self.node_low[size - 1:2 * size - 1] = torch.fmin(pair_low[:, 0], pair_low[:, 1])
            self.node_high[size - 1:2 * size - 1] = torch.fmax(pair_high[:, 0], pair_high[:, 1])
            size //= 2

    def _median_order(self, centroids: torch.Tensor, slots: int) -> torch.Tensor:
        """Triangle of each leaf slot, splitting every node at the median of its widest axis.

        Args:
            centroids: ``(F, 3)`` triangle centroids.
            slots: Leaf slots in the tree.

        Returns:
            ``(slots,)`` int64 face per slot, -1 for an empty slot.
        """
        order = torch.full((slots,), -1, dtype=torch.int64, device=self.device)
        order[: self.face_count] = torch.argsort(morton_codes(centroids))
        for level in range(self.depth):
            rows = order.view(1 << level, -1)
            valid = rows >= 0
            points = centroids[rows.clamp_min(0)]
            low = torch.where(valid[..., None], points, math.inf).amin(1)
            high = torch.where(valid[..., None], points, -math.inf).amax(1)
            axis = (high - low).nan_to_num(-1.0, posinf=-1.0, neginf=-1.0).argmax(1)
            along = points.gather(2, axis[:, None, None].expand(-1, rows.shape[1], 1)).squeeze(2)
            along = torch.where(valid, along, math.inf)
            rank = torch.argsort(along, dim=1, stable=True)
            order = rows.gather(1, rank).reshape(-1)
        return order

    # -----------------------------------------------------------------------------------------

    def _chunk(self, chunk: int | None, per_query: int) -> int:
        """Queries per traversal pass."""
        if chunk is not None:
            return max(1, int(chunk))
        return mesh_device.chunk_size(per_query, self.device, minimum=4096, maximum=1 << 18)

    def _local_points(self, points: torch.Tensor) -> torch.Tensor:
        """Points moved into the tree's unit frame."""
        return ((points.double() - self.centre) / self.scale).float()

    def _box_distance_sq(self, nodes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Squared distance from points to node boxes, lowered slightly; infinity for an empty node."""
        gap = torch.maximum(self.node_low[nodes] - points, points - self.node_high[nodes]).clamp_min(0)
        return (_dot(gap, gap) * (1.0 - BOX_PAD)).nan_to_num(math.inf)

    def _box_entry(self, nodes, origins, inverse, t_min, t_max):
        """Ray parameter entering node boxes, widened slightly; NaN where a ray misses."""
        near = (self.node_low[nodes] - origins) * inverse
        far = (self.node_high[nodes] - origins) * inverse
        enter = torch.minimum(near, far).amax(-1)
        leave = torch.maximum(near, far).amin(-1)
        enter = torch.maximum(enter - enter.abs() * BOX_PAD, t_min)
        leave = torch.minimum(leave + leave.abs() * BOX_PAD, t_max)
        hit = enter <= leave
        return torch.where(hit, enter, torch.full_like(enter, math.nan))

    @staticmethod
    def _coherent_order(points: torch.Tensor) -> torch.Tensor:
        """Query order along a Morton curve."""
        if points.shape[0] < 2:
            return torch.arange(points.shape[0], device=points.device)
        finite = torch.isfinite(points).all(1)
        safe = torch.where(finite[:, None], points, torch.zeros_like(points))
        return torch.argsort(morton_codes(safe))

    def _leaf_slots(self, nodes: torch.Tensor) -> torch.Tensor:
        """Triangle slots of leaf nodes, ``(N, leaf_size)``."""
        offsets = torch.arange(self.leaf_size, device=self.device)
        return (nodes - self.first_leaf)[:, None] * self.leaf_size + offsets

    @staticmethod
    def _first_per_query(queries: torch.Tensor, holder: torch.Tensor) -> torch.Tensor:
        """Positions in ``queries`` that hold the first occurrence of each query.

        Args:
            queries: ``(N,)`` query indices, possibly repeated.
            holder: ``(Q,)`` int64 scratch indexed by query.

        Returns:
            ``(M,)`` int64 positions.
        """
        position = torch.arange(queries.shape[0], device=queries.device)
        holder[queries] = queries.shape[0]
        holder.scatter_reduce_(0, queries, position, reduce="amin")
        return position[holder[queries] == position]

    # -----------------------------------------------------------------------------------------

    def _traverse(
        self,
        best: torch.Tensor,
        root_key: torch.Tensor,
        strict: bool,
        visit: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None],
        child_key: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Depth-first traversal of every query at once, nearest child first.

        Args:
            best: ``(Q,)`` current bound per query, lowered in place by ``visit``.
            root_key: ``(Q,)`` key of the root for each query.
            strict: Enter a node only when its key is below the bound, else at or below it.
            visit: Called with ``(queries, slots)`` for leaves reached; returns queries that
                are finished, or None.
            child_key: Called with ``(queries, nodes)``; returns each node's key, NaN or
                infinity to skip it.
        """
        count = best.shape[0]
        device = self.device

        def passes(key, bound):
            return key < bound if strict else key <= bound

        capacity = self.depth + 2
        stack = torch.zeros(count, capacity, dtype=torch.int64, device=device)
        keys = torch.zeros(count, capacity, dtype=torch.float32, device=device)
        keys[:, 0] = root_key
        depth = passes(root_key, best).long()
        finished = torch.zeros(count, dtype=torch.bool, device=device)
        active = torch.nonzero(depth > 0).squeeze(1)
        target = max(count, MIN_ROWS)
        while active.numel():
            width = active.numel()
            pops = min(MAX_POPS, max(1, target // width))
            held = depth[active]
            if pops == 1:
                take = torch.ones_like(held)
                queries, position = active, held - 1
            else:
                take = held.clamp(max=pops)
                owner = torch.repeat_interleave(torch.arange(width, device=device), take)
                layer = torch.arange(owner.shape[0], device=device) - (torch.cumsum(take, 0) - take)[owner]
                queries, position = active[owner], held[owner] - 1 - layer
            node = stack[queries, position]
            key = keys[queries, position]
            depth[active] = held - take
            live = passes(key, best[queries])
            queries, node = queries[live], node[live]
            leaf = node >= self.first_leaf

            if bool(leaf.any()):
                done = visit(queries[leaf], self._leaf_slots(node[leaf]))
                if done is not None and done.numel():
                    depth[done] = 0
                    finished[done] = True

            inner = ~leaf & ~finished[queries]
            if not bool(inner.any()):
                active = active[depth[active] > 0]
                continue
            at = queries[inner]
            left = 2 * node[inner] + 1
            right = left + 1
            k_left = child_key(at, left)
            k_right = child_key(at, right)
            left_first = (k_left <= k_right) | torch.isnan(k_right)
            near = torch.where(left_first, left, right)
            far = torch.where(left_first, right, left)
            k_near = torch.where(left_first, k_left, k_right)
            k_far = torch.where(left_first, k_right, k_left)
            bound = best[at]
            push_far = passes(k_far, bound)
            push_near = passes(k_near, bound)
            pushes = push_far.long() + push_near.long()
            base = depth[at]
            if pops > 1:
                flip = torch.arange(at.shape[0] - 1, -1, -1, device=device)
                at_flipped, pushes_flipped = at[flip], pushes[flip]
                run = torch.cumsum(pushes_flipped, 0) - pushes_flipped
                index = torch.arange(at.shape[0], device=device)
                start = torch.ones(at.shape[0], dtype=torch.bool, device=device)
                start[1:] = at_flipped[1:] != at_flipped[:-1]
                first = torch.cummax(torch.where(start, index, torch.zeros_like(index)), 0).values
                offset = torch.empty_like(run)
                offset[flip] = run - run[first]
                base = base + offset
            needed = int((base + pushes).max())
            if needed > capacity:
                extra = needed - capacity + self.depth + 2
                stack = torch.cat([stack, stack.new_zeros(count, extra)], 1)
                keys = torch.cat([keys, keys.new_zeros(count, extra)], 1)
                capacity += extra
            row, slot = at[push_far], base[push_far]
            stack[row, slot] = far[push_far]
            keys[row, slot] = k_far[push_far]
            slot = (base + push_far.long())[push_near]
            row = at[push_near]
            stack[row, slot] = near[push_near]
            keys[row, slot] = k_near[push_near]
            depth.index_add_(0, at, pushes)
            active = active[depth[active] > 0]

    def _closest_pass(self, points: torch.Tensor, bound_sq: float):
        """Closest point traversal for one chunk of points in the tree's unit frame."""
        count = points.shape[0]
        start = torch.tensor(bound_sq, dtype=torch.float32, device=self.device)
        best = torch.nextafter(start, start.new_tensor(math.inf)).expand(count).clone()
        best_slot = torch.full((count,), -1, dtype=torch.int64, device=self.device)
        best_bary = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        holder = torch.empty(count, dtype=torch.int64, device=self.device)

        def visit(queries, slots):
            distance, bary = closest_point_on_triangles(
                points[queries][:, None, :], self.slot_a[slots], self.slot_b[slots], self.slot_c[slots]
            )
            distance = torch.where(self.slot_valid[slots], distance, math.inf)
            nearest, pick = distance.min(1)
            before = best[queries]
            best.scatter_reduce_(0, queries, nearest, reduce="amin")
            won = torch.nonzero((nearest < before) & (nearest == best[queries])).squeeze(1)
            if won.numel():
                won = won[self._first_per_query(queries[won], holder)]
                best_slot[queries[won]] = slots[won, pick[won]]
                best_bary[queries[won]] = bary[won, pick[won]]
            return None

        def child_key(queries, nodes):
            return self._box_distance_sq(nodes, points[queries])

        root = torch.zeros(count, dtype=torch.int64, device=self.device)
        self._traverse(best, self._box_distance_sq(root, points), True, visit, child_key)
        return best, best_slot, best_bary

    def closest_point(
        self, points: torch.Tensor, max_distance: float = math.inf, chunk: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closest point on the mesh to each query point.

        Args:
            points: ``(Q, 3)`` query points.
            max_distance: Points farther than this from the mesh report no face.
            chunk: Queries per traversal pass; chosen from free memory when None.

        Returns:
            ``(distance, face_index, barycentric, closest)`` on the points' device:
            ``(Q,)`` distances, ``(Q,)`` int64 faces, ``(Q, 3)`` weights of the face's
            corners and ``(Q, 3)`` closest points. A point with no face within
            ``max_distance`` gets infinity, -1, zero weights and its own position.
        """
        home = points.device
        query = points.to(self.device, torch.float32).reshape(-1, 3)
        count = query.shape[0]
        distance = torch.full((count,), math.inf, dtype=torch.float32, device=self.device)
        face = torch.full((count,), -1, dtype=torch.int64, device=self.device)
        bary = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        closest = query.clone()
        limit = float(max_distance)
        if self.face_count == 0 or count == 0 or not limit >= 0:
            return distance.to(home), face.to(home), bary.to(home), closest.to(home)
        reach = limit / self.scale
        bound_sq = reach * reach * (1.0 + 4.0 * BOX_PAD) if reach < 1e18 else math.inf
        local = self._local_points(query)
        order = self._coherent_order(local)
        step = self._chunk(chunk, (self.depth + 2) * 12 + self.leaf_size * 200)
        for begin in range(0, count, step):
            index = order[begin:begin + step]
            _best, slot, weights = self._closest_pass(local[index], bound_sq)
            found = slot >= 0
            hit, slot, weights = index[found], slot[found], weights[found]
            hit_face = self.slot_face[slot]
            corners = self._vertices[self._faces[hit_face]].double()
            point = (corners * weights.double()[:, :, None]).sum(1)
            gap = (query[hit].double() - point).norm(dim=1)
            keep = gap <= limit
            hit, hit_face, weights, point, gap = hit[keep], hit_face[keep], weights[keep], point[keep], gap[keep]
            face[hit] = hit_face
            bary[hit] = weights
            closest[hit] = point.float()
            distance[hit] = gap.float()
        return distance.to(home), face.to(home), bary.to(home), closest.to(home)

    # -----------------------------------------------------------------------------------------

    def _ray_pass(self, origins, directions, t_min, t_max, cull_backfaces, any_hit):
        """Ray traversal for one chunk of rays in the tree's unit frame."""
        count = origins.shape[0]
        tiny = 1e-30
        safe = torch.where(directions.abs() < tiny, torch.where(directions < 0, -tiny, tiny), directions)
        inverse = 1.0 / safe
        best = t_max.clone()
        best_slot = torch.full((count,), -1, dtype=torch.int64, device=self.device)
        best_bary = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        holder = torch.empty(count, dtype=torch.int64, device=self.device)

        def visit(queries, slots):
            t, bary = intersect_ray_triangles(
                origins[queries][:, None, :], directions[queries][:, None, :],
                self.slot_a[slots], self.slot_b[slots], self.slot_c[slots],
                t_min[queries][:, None], best[queries][:, None], cull_backfaces,
            )
            t = torch.where(self.slot_valid[slots], t, math.inf)
            nearest, pick = t.min(1)
            before = best[queries]
            best.scatter_reduce_(0, queries, nearest, reduce="amin")
            won = torch.isfinite(nearest) & (nearest <= before) & (nearest == best[queries])
            won = torch.nonzero(won).squeeze(1)
            if won.numel():
                won = won[self._first_per_query(queries[won], holder)]
                best_slot[queries[won]] = slots[won, pick[won]]
                best_bary[queries[won]] = bary[won, pick[won]]
            return queries[won] if any_hit else None

        def child_key(queries, nodes):
            return self._box_entry(nodes, origins[queries], inverse[queries], t_min[queries], best[queries])

        root = torch.zeros(count, dtype=torch.int64, device=self.device)
        root_key = self._box_entry(root, origins, inverse, t_min, t_max)
        self._traverse(best, root_key, False, visit, child_key)
        return best, best_slot, best_bary

    def _rays(self, origins, directions, t_min, t_max, cull_backfaces, chunk, any_hit):
        """Run ray traversal over every chunk."""
        home = origins.device
        o = self._local_points(origins.to(self.device).reshape(-1, 3))
        d = (directions.to(self.device).reshape(-1, 3).double() / self.scale).float()
        count = o.shape[0]
        low = torch.as_tensor(t_min, dtype=torch.float32, device=self.device).expand(count).contiguous()
        high = torch.as_tensor(t_max, dtype=torch.float32, device=self.device).expand(count).contiguous()
        t = torch.full((count,), math.inf, dtype=torch.float32, device=self.device)
        face = torch.full((count,), -1, dtype=torch.int64, device=self.device)
        bary = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        if self.face_count == 0 or count == 0:
            return t.to(home), face.to(home), bary.to(home)
        order = self._coherent_order(o)
        step = self._chunk(chunk, (self.depth + 2) * 12 + self.leaf_size * 200)
        for begin in range(0, count, step):
            index = order[begin:begin + step]
            best, slot, weights = self._ray_pass(
                o[index], d[index], low[index], high[index], cull_backfaces, any_hit
            )
            found = slot >= 0
            hit = index[found]
            t[hit] = best[found]
            face[hit] = self.slot_face[slot[found]]
            bary[hit] = weights[found]
        return t.to(home), face.to(home), bary.to(home)

    def ray_cast(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_min: float | torch.Tensor = 0.0,
        t_max: float | torch.Tensor = math.inf,
        cull_backfaces: bool = False,
        chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest triangle hit along each ray.

        Args:
            origins: ``(Q, 3)`` ray origins.
            directions: ``(Q, 3)`` ray directions; their length scales ``t``.
            t_min: Nearest accepted ray parameter, a number or ``(Q,)`` tensor.
            t_max: Farthest accepted ray parameter, a number or ``(Q,)`` tensor.
            cull_backfaces: Ignore triangles whose counter-clockwise side faces away.
            chunk: Rays per traversal pass; chosen from free memory when None.

        Returns:
            ``(t, face_index, barycentric)`` on the origins' device: ``(Q,)`` ray parameters,
            ``(Q,)`` int64 faces and ``(Q, 3)`` corner weights. A miss gets infinity, -1 and
            zero weights.
        """
        return self._rays(origins, directions, t_min, t_max, cull_backfaces, chunk, any_hit=False)

    def any_hit(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        t_min: float | torch.Tensor = 0.0,
        t_max: float | torch.Tensor = math.inf,
        cull_backfaces: bool = False,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Whether each ray hits any triangle between ``t_min`` and ``t_max``.

        Args:
            origins: ``(Q, 3)`` ray origins.
            directions: ``(Q, 3)`` ray directions; their length scales ``t``.
            t_min: Nearest accepted ray parameter, a number or ``(Q,)`` tensor.
            t_max: Farthest accepted ray parameter, a number or ``(Q,)`` tensor.
            cull_backfaces: Ignore triangles whose counter-clockwise side faces away.
            chunk: Rays per traversal pass; chosen from free memory when None.

        Returns:
            ``(Q,)`` bool on the origins' device.
        """
        _t, face, _bary = self._rays(origins, directions, t_min, t_max, cull_backfaces, chunk, any_hit=True)
        return face >= 0
