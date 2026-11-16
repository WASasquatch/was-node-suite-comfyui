"""Cross-sections of a triangle mesh and the body measurements taken from them. Lengths are in model
units. A humanoid stands Y-up, facing +Z, its left side on +X."""
# A plane loop runs counter-clockwise seen from the tip of its plane normal.

from __future__ import annotations

__all__ = [
    "Capsule",
    "HeadFrame",
    "Landmarks",
    "Level",
    "Limb",
    "Loop",
    "Neck",
    "Profile",
    "Slicer",
    "Sphere",
    "fit_capsule",
    "fit_sphere",
    "head_frame",
    "landmarks",
    "profile",
    "ring",
    "slice_plane",
]

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import torch

from . import device as mesh_device
from . import topology

#: Most levels one profile cuts.
MAX_LEVELS = 4096

#: Most triangle and level crossings one cut computes.
MAX_CROSSINGS = 40_000_000

#: Most points `ring` returns.
MAX_RING_POINTS = 1_000_000

#: Most cells in the occupancy grid `landmarks` builds.
MAX_GRID_CELLS = 40_000_000

#: Most points a fit reads; larger sets are thinned evenly.
MAX_FIT_POINTS = 100_000

#: Most Gauss-Newton steps a fit takes.
FIT_ITERATIONS = 60

Vector = Sequence[float] | torch.Tensor


# ---------------------------------------------------------------------------------------------
# Results


@dataclass
class Loop:
    """One polyline where a plane crosses a mesh.

    Attributes:
        points: ``(K, 3)`` positions.
        closed: True when the polyline returns to its first point.
        faces: ``(K,)`` int64 triangle crossed from point ``i`` to point ``i + 1``, wrapping
            on a closed loop; ``-1`` on the last point of an open one.
        edges: ``(K, 2)`` int64 vertex pair of the mesh edge each point lies on.
        weights: ``(K,)`` place along that edge, ``point = (1 - w) * a + w * b``.
        normal: ``(3,)`` unit normal of the cutting plane.
        area: Signed area about ``normal``, positive counter-clockwise; an open loop is
            measured as if closed by a straight segment.
        perimeter: Length of the polyline.
        centroid: ``(3,)`` centre of the enclosed area, or the mean point when it has none.
    """

    points: torch.Tensor
    closed: bool
    faces: torch.Tensor
    edges: torch.Tensor
    weights: torch.Tensor
    normal: torch.Tensor
    area: float
    perimeter: float
    centroid: torch.Tensor

    @property
    def count(self) -> int:
        """Number of points."""
        return int(self.points.shape[0])


@dataclass
class Profile:
    """Cross-sections of a mesh at evenly spaced levels along an axis.

    Attributes:
        axis: ``(3,)`` unit axis; a level is the plane ``dot(point, axis) = height``.
        basis: ``(2, 3)`` in-plane unit axes ``u`` and ``w`` with ``cross(u, w) = axis``.
        heights: ``(L,)`` float64 level heights, ascending.
        loop_count: ``(L,)`` int64 loops at each level.
        open_count: ``(L,)`` int64 open polylines at each level.
        area: ``(L,)`` float64 total signed area; holes subtract.
        perimeter: ``(L,)`` float64 total polyline length.
        loop_offsets: ``(L + 1,)`` int64; level ``l`` holds loops ``loop_offsets[l]`` to
            ``loop_offsets[l + 1]``, largest area first.
        loop_level: ``(N,)`` int64 level of each loop.
        loop_closed: ``(N,)`` bool.
        loop_area: ``(N,)`` float64 signed area.
        loop_perimeter: ``(N,)`` float64.
        loop_centroid: ``(N, 3)`` float64.
        loop_low: ``(N, 2)`` float64 smallest coordinate along ``u`` and ``w``.
        loop_high: ``(N, 2)`` float64 largest coordinate along ``u`` and ``w``.
        point_offsets: ``(N + 1,)`` int64; loop ``i`` holds points ``point_offsets[i]`` to
            ``point_offsets[i + 1]``.
        points: ``(M, 3)`` loop points.
        point_faces: ``(M,)`` int64, as `Loop.faces`.
        point_edges: ``(M, 2)`` int64, as `Loop.edges`.
        point_weights: ``(M,)``, as `Loop.weights`.
    """

    axis: torch.Tensor
    basis: torch.Tensor
    heights: torch.Tensor
    loop_count: torch.Tensor
    open_count: torch.Tensor
    area: torch.Tensor
    perimeter: torch.Tensor
    loop_offsets: torch.Tensor
    loop_level: torch.Tensor
    loop_closed: torch.Tensor
    loop_area: torch.Tensor
    loop_perimeter: torch.Tensor
    loop_centroid: torch.Tensor
    loop_low: torch.Tensor
    loop_high: torch.Tensor
    point_offsets: torch.Tensor
    points: torch.Tensor
    point_faces: torch.Tensor
    point_edges: torch.Tensor
    point_weights: torch.Tensor

    @property
    def level_count(self) -> int:
        """Number of levels."""
        return int(self.heights.shape[0])

    def loop(self, index: int) -> Loop:
        """Loop ``index`` as a `Loop`."""
        start, end = int(self.point_offsets[index]), int(self.point_offsets[index + 1])
        return Loop(
            points=self.points[start:end],
            closed=bool(self.loop_closed[index]),
            faces=self.point_faces[start:end],
            edges=self.point_edges[start:end],
            weights=self.point_weights[start:end],
            normal=self.axis.to(self.points.dtype),
            area=float(self.loop_area[index]),
            perimeter=float(self.loop_perimeter[index]),
            centroid=self.loop_centroid[index].to(self.points.dtype),
        )

    def loops(self, level: int) -> list[Loop]:
        """The loops at level ``level``, largest area first."""
        start, end = int(self.loop_offsets[level]), int(self.loop_offsets[level + 1])
        return [self.loop(index) for index in range(start, end)]


class Sphere(NamedTuple):
    """A sphere fitted to points.

    Attributes:
        centre: ``(3,)`` centre.
        radius: Radius.
        error: Root mean square distance of the points from the surface.
    """

    centre: torch.Tensor
    radius: float
    error: float


class Capsule(NamedTuple):
    """A capsule fitted to points: every point within ``radius`` of a segment.

    Attributes:
        start: ``(3,)`` first end of the segment.
        end: ``(3,)`` second end of the segment.
        radius: Radius.
        error: Root mean square distance of the points from the surface.
    """

    start: torch.Tensor
    end: torch.Tensor
    radius: float
    error: float


@dataclass
class Level:
    """A horizontal measuring height on a body and the torso section there.

    Attributes:
        height: Y of the level.
        centre: ``(3,)`` centre of the torso section.
        width: Torso extent across the body.
        depth: Torso extent front to back.
        area: Torso section area.
        measured: False when the level is set from body proportions.
    """

    height: float
    centre: torch.Tensor
    width: float
    depth: float
    area: float
    measured: bool = True


@dataclass
class Limb:
    """A leg or an arm as a straight axis.

    Attributes:
        root: ``(3,)`` hip or shoulder joint estimate.
        tip: ``(3,)`` end of the axis at the floor or the fingertips.
        axis: ``(3,)`` unit direction of the limb's longest spread, root end to tip end.
        length: Root to tip distance.
        radius: Median distance of the limb surface from its axis.
        centre: ``(3,)`` area centroid of the limb surface.
        vertices: ``(N,)`` int64 input vertices on the limb.
    """

    root: torch.Tensor
    tip: torch.Tensor
    axis: torch.Tensor
    length: float
    radius: float
    centre: torch.Tensor
    vertices: torch.Tensor


@dataclass
class Neck:
    """The narrowest ring between shoulders and head.

    Attributes:
        centre: ``(3,)`` centre of the ring's area.
        normal: ``(3,)`` unit normal of the cutting plane, pointing to the head.
        radius: Radius of the circle with the ring's area.
        loop: The ring.
        measured: False when the ring is cut at a height set from body proportions.
    """

    centre: torch.Tensor
    normal: torch.Tensor
    radius: float
    loop: Loop | None
    measured: bool = True


@dataclass
class Landmarks:
    """Body measurements of a standing humanoid, in the input's coordinates.

    Attributes:
        floor: Lowest Y.
        height: Floor to head top.
        heading: Degrees to turn about Y so the body faces +Z.
        head_top: ``(3,)`` highest point.
        head_centre: ``(3,)`` centre of the head above the neck.
        crotch: ``(3,)`` point where the legs meet.
        hips: Widest torso level just above the crotch.
        waist: Narrowest torso level between hips and chest.
        chest: Widest torso level just below the armpits.
        armpits: ``"left"`` and ``"right"`` points where the arms leave the torso.
        shoulders: ``"left"`` and ``"right"`` shoulder joint estimates.
        neck: The neck ring.
        legs: ``"left"`` and ``"right"`` legs found.
        arms: ``"left"`` and ``"right"`` arms found.
        arm_angle: Mean arm angle from hanging straight down, in degrees; 90 is level.
        arm_pose: ``"t-pose"``, ``"a-pose"`` or ``"down"``.
        legs_separable: True when two legs were traced from the floor to the crotch.
        arms_separable: True when both arms were traced apart from the torso.
        measured: Landmark name against True when found on the mesh, False when set from
            body proportions.
        notes: Plain sentences about what could not be measured.
    """

    floor: float
    height: float
    heading: float
    head_top: torch.Tensor
    head_centre: torch.Tensor
    crotch: torch.Tensor
    hips: Level
    waist: Level
    chest: Level
    armpits: dict[str, torch.Tensor]
    shoulders: dict[str, torch.Tensor]
    neck: Neck
    legs: dict[str, Limb] = field(default_factory=dict)
    arms: dict[str, Limb] = field(default_factory=dict)
    arm_angle: float = 0.0
    arm_pose: str = "down"
    legs_separable: bool = False
    arms_separable: bool = False
    measured: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class HeadFrame:
    """Position and facing of a head-only mesh.

    Attributes:
        neck: The neck ring: the open boundary, or the narrowest ring across the lower part.
        neck_centre: ``(3,)`` centre of the neck ring.
        neck_radius: Radius of the circle with the neck ring's area.
        open_neck: True when the neck ring is an open boundary of the mesh.
        up: ``(3,)`` unit normal of the neck ring, pointing into the head, unless given.
        forward: ``(3,)`` unit facing direction, at right angles to ``up``.
        left: ``(3,)`` unit ``cross(up, forward)``.
        centre: ``(3,)`` centre of the head's box in this frame.
        size: Extent along ``up`` from the neck ring to the crown.
        width: Extent along ``left``.
        depth: Extent along ``forward``.
        confidence: 0 to 1, how far the head's mass sits behind the neck, which decides the
            facing; 1 when ``forward`` was given.
    """

    neck: Loop | None
    neck_centre: torch.Tensor
    neck_radius: float
    open_neck: bool
    up: torch.Tensor
    forward: torch.Tensor
    left: torch.Tensor
    centre: torch.Tensor
    size: float
    width: float
    depth: float
    confidence: float


# ---------------------------------------------------------------------------------------------
# Small helpers


def _vector(value: Vector, device: torch.device, name: str, unit: bool = False) -> torch.Tensor:
    """A ``(3,)`` float64 vector from a sequence or tensor, optionally made unit length."""
    vector = torch.as_tensor(value, dtype=torch.float64).reshape(-1).to(device)
    if vector.shape[0] != 3 or not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must be three finite numbers, such as (0, 1, 0)")
    if unit:
        length = float(vector.norm())
        if length <= 1e-12:
            raise ValueError(f"{name} must not be the zero vector; give a direction such as (0, 1, 0)")
        vector = vector / length
    return vector


def _plane_basis(normal: torch.Tensor) -> torch.Tensor:
    """``(2, 3)`` in-plane unit axes ``u`` and ``w`` with ``cross(u, w) = normal``."""
    reference = torch.zeros(3, dtype=normal.dtype, device=normal.device)
    reference[0 if abs(float(normal[0])) < 0.9 else 2] = 1.0
    u = reference - (reference @ normal) * normal
    u = u / u.norm()
    w = torch.linalg.cross(normal, u)
    return torch.stack([u, w])


def _run_rank(keys_sorted: torch.Tensor) -> torch.Tensor:
    """Position of each element within its run of equal keys in a sorted tensor."""
    count = int(keys_sorted.shape[0])
    if count == 0:
        return keys_sorted.new_zeros(0)
    index = torch.arange(count, device=keys_sorted.device)
    start = torch.ones(count, dtype=torch.bool, device=keys_sorted.device)
    start[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first = torch.cummax(torch.where(start, index, torch.zeros_like(index)), 0).values
    return index - first


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    """CSR offsets from per-row counts."""
    offsets = torch.zeros(int(counts.shape[0]) + 1, dtype=torch.int64, device=counts.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return offsets


def _chain_rank(successor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain id and steps to the chain's end for each node of a partial injection.

    Args:
        successor: ``(N,)`` int64 next node, ``-1`` at a chain end; no node has two
            predecessors.

    Returns:
        ``(chain, distance)``: the id of each node's chain, and its steps to the chain's last
        node. A cycle is cut before its smallest node, which starts it.
    """
    count = int(successor.shape[0])
    index = torch.arange(count, device=successor.device)
    terminal = successor < 0
    step = torch.where(terminal, index, successor)
    rounds = max(1, math.ceil(math.log2(max(count, 2)))) + 1
    reach = step
    for _ in range(rounds):
        reach = reach[reach]
    cyclic = ~terminal[reach]
    low = index.clone()
    pointer = step
    for _ in range(rounds):
        low = torch.minimum(low, low[pointer])
        pointer = pointer[pointer]
    head = cyclic & (low == index)
    predecessor = torch.full_like(successor, -1)
    linked = ~terminal
    predecessor[successor[linked]] = index[linked]
    cut = successor.clone()
    cut[predecessor[head]] = -1
    terminal = cut < 0
    step = torch.where(terminal, index, cut)
    distance = (~terminal).long()
    for _ in range(rounds):
        distance = distance + distance[step]
        step = step[step]
    return step, distance


# ---------------------------------------------------------------------------------------------
# Cutting


@dataclass
class _Chains:
    """Plane cuts at several levels, chained into loops.

    Attributes:
        level: ``(N,)`` int64 level of each loop.
        closed: ``(N,)`` bool.
        offsets: ``(N + 1,)`` int64 point offsets.
        points: ``(M, 3)`` float64.
        segment_face: ``(M,)`` int64 face crossed towards the next point, -1 at an open end.
        edge_face: ``(M,)`` int64 face whose side holds the point.
        edge_side: ``(M,)`` int64 side of that face.
        edge_low: ``(M,)`` int64 smaller welded vertex of the edge.
        edge_weight: ``(M,)`` float64 place along the edge from ``edge_low``.
    """

    level: torch.Tensor
    closed: torch.Tensor
    offsets: torch.Tensor
    points: torch.Tensor
    segment_face: torch.Tensor
    edge_face: torch.Tensor
    edge_side: torch.Tensor
    edge_low: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def count(self) -> int:
        """Number of loops."""
        return int(self.level.shape[0])

    def loop_index(self) -> torch.Tensor:
        """``(M,)`` int64 loop of each point."""
        counts = self.offsets[1:] - self.offsets[:-1]
        return torch.repeat_interleave(torch.arange(self.count, device=counts.device), counts)

    def following(self) -> torch.Tensor:
        """``(M,)`` int64 index of the next point around each loop, wrapping."""
        loop = self.loop_index()
        index = torch.arange(int(self.points.shape[0]), device=loop.device)
        following = index + 1
        last = following == self.offsets[1:][loop]
        return torch.where(last, self.offsets[:-1][loop], following)


def _cut(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    usable: torch.Tensor,
    along: torch.Tensor,
    heights: torch.Tensor,
) -> _Chains:
    """Cut a mesh at several parallel levels and chain the crossings into loops.

    Args:
        vertices: ``(V, 3)`` float64 welded positions.
        faces: ``(F, 3)`` int64 welded faces.
        usable: ``(F,)`` bool, faces with finite corners.
        along: ``(V,)`` float64 coordinate of each vertex along the cutting axis.
        heights: ``(L,)`` float64 ascending level coordinates.

    Returns:
        The loops, unsorted. A vertex exactly on a level counts as above it.

    Raises:
        ValueError: When the cut would exceed `MAX_CROSSINGS`.
    """
    device = vertices.device
    vertex_count = int(vertices.shape[0])
    corner = along[faces]
    first = torch.searchsorted(heights, corner.amin(1).contiguous(), right=True)
    last = torch.searchsorted(heights, corner.amax(1).contiguous(), right=True)
    count = torch.where(usable, (last - first).clamp_min(0), torch.zeros_like(first))
    total = int(count.sum())
    if total > MAX_CROSSINGS:
        raise ValueError(
            f"cutting this mesh at {int(heights.shape[0])} levels crosses {total:,} triangles, "
            f"past the limit of {MAX_CROSSINGS:,}; lower the number of levels or simplify the mesh"
        )
    face = torch.repeat_interleave(torch.arange(int(faces.shape[0]), device=device), count)
    begin = torch.cumsum(count, 0) - count
    level = first[face] + torch.arange(total, device=device) - begin[face]
    above = (corner[face] - heights[level][:, None]) >= 0
    after = above[:, [1, 2, 0]]
    start_side = (above & ~after).to(torch.int32).argmax(1).long()
    end_side = (~above & after).to(torch.int32).argmax(1).long()
    corners = faces[face]

    def edge(side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = corners.gather(1, side[:, None]).squeeze(1)
        b = corners.gather(1, ((side + 1) % 3)[:, None]).squeeze(1)
        return torch.minimum(a, b), torch.maximum(a, b)

    start_low, start_high = edge(start_side)
    end_low, end_high = edge(end_side)
    # A face with a repeated vertex can cross one edge twice; it adds nothing.
    keep = (start_low != end_low) | (start_high != end_high)
    if not bool(keep.all()):
        face, level, start_side, end_side = face[keep], level[keep], start_side[keep], end_side[keep]
        start_low, start_high = start_low[keep], start_high[keep]
        end_low, end_high = end_low[keep], end_high[keep]
    pairs = int(face.shape[0])
    stride = max(vertex_count, 1)
    keys = torch.cat([start_low * stride + start_high, end_low * stride + end_high])
    edge_keys, edge_index = torch.unique(keys, return_inverse=True)
    edge_count = max(int(edge_keys.shape[0]), 1)
    node_keys, node_index = torch.unique(level.repeat(2) * edge_count + edge_index, return_inverse=True)
    node_edge = edge_keys[node_keys % edge_count]
    node_low, node_high = node_edge // stride, node_edge % stride
    node_level = node_keys // edge_count
    weight = (heights[node_level] - along[node_low]) / (along[node_high] - along[node_low])
    node_point = vertices[node_low] + weight[:, None] * (vertices[node_high] - vertices[node_low])

    # State 2i enters segment i at its start crossing, 2i + 1 at its end crossing.
    ends = torch.stack([node_index[:pairs], node_index[pairs:]], 1).reshape(-1)
    sides = torch.stack([start_side, end_side], 1).reshape(-1)
    states = 2 * pairs
    order = torch.argsort(ends, stable=True)
    sorted_ends = ends[order]
    rank = _run_rank(sorted_ends)
    at = torch.nonzero(rank % 2 == 0).squeeze(1)
    nxt = (at + 1).clamp_max(max(states - 1, 0))
    paired = (at + 1 < states) & (sorted_ends[nxt] == sorted_ends[at])
    partner = torch.full((states,), -1, dtype=torch.int64, device=device)
    a, b = order[at[paired]], order[nxt[paired]]
    partner[a] = b
    partner[b] = a
    index = torch.arange(states, device=device)
    successor = partner[index ^ 1]
    chain, distance = _chain_rank(successor)

    # Each chain is found twice, once each way; the copy agreeing with most faces is kept.
    natural = (index & 1) == 0
    length = torch.bincount(chain, minlength=states)
    agree = torch.bincount(chain[natural], minlength=states)
    twin = chain[index ^ 1]
    kept = (2 * agree[chain] > length[chain]) | ((2 * agree[chain] == length[chain]) & (chain < twin))
    kept_states = index[kept]
    order = torch.argsort(chain[kept_states] * (states + 1) + (states - distance[kept_states]))
    kept_states = kept_states[order]
    chain_ids, loop_of_state = torch.unique(chain[kept_states], return_inverse=True)
    loop_count = int(chain_ids.shape[0])
    state_count = torch.bincount(loop_of_state, minlength=loop_count)
    state_offsets = _offsets(state_count)
    first_state = kept_states[state_offsets[:-1]]
    last_state = kept_states[state_offsets[1:] - 1]
    closed = successor[last_state] == first_state
    point_count = state_count + (~closed).long()
    offsets = _offsets(point_count)
    within = torch.arange(int(kept_states.shape[0]), device=device) - state_offsets[:-1][loop_of_state]
    slot = offsets[:-1][loop_of_state] + within
    total_points = int(offsets[-1])
    point_node = torch.empty(total_points, dtype=torch.int64, device=device)
    segment_face = torch.empty(total_points, dtype=torch.int64, device=device)
    edge_face = torch.empty(total_points, dtype=torch.int64, device=device)
    edge_side = torch.empty(total_points, dtype=torch.int64, device=device)
    segment = kept_states >> 1
    point_node[slot] = ends[kept_states]
    segment_face[slot] = face[segment]
    edge_face[slot] = face[segment]
    edge_side[slot] = sides[kept_states]
    open_loops = torch.nonzero(~closed).squeeze(1)
    if open_loops.numel():
        tail = kept_states[state_offsets[1:][open_loops] - 1]
        tail_slot = offsets[1:][open_loops] - 1
        point_node[tail_slot] = ends[tail ^ 1]
        segment_face[tail_slot] = -1
        edge_face[tail_slot] = face[tail >> 1]
        edge_side[tail_slot] = sides[tail ^ 1]
    return _Chains(
        level=level[first_state >> 1],
        closed=closed,
        offsets=offsets,
        points=node_point[point_node],
        segment_face=segment_face,
        edge_face=edge_face,
        edge_side=edge_side,
        edge_low=node_low[point_node],
        edge_weight=weight[point_node],
    )


@dataclass
class _LoopStats:
    """Per-loop measurements of `_Chains`.

    Attributes:
        area: ``(N,)`` float64 signed area about the axis.
        perimeter: ``(N,)`` float64.
        centroid: ``(N, 3)`` float64.
        low: ``(N, 2)`` float64 smallest coordinate along the basis axes.
        high: ``(N, 2)`` float64 largest coordinate along the basis axes.
    """

    area: torch.Tensor
    perimeter: torch.Tensor
    centroid: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor


def _measure(chains: _Chains, axis: torch.Tensor, basis: torch.Tensor) -> _LoopStats:
    """Area, perimeter, centroid and extents of every loop."""
    device = chains.points.device
    count = chains.count
    loop = chains.loop_index()
    following = chains.following()
    points = chains.points
    origin = points[chains.offsets[:-1].clamp_max(max(int(points.shape[0]) - 1, 0))] if count else points[:0]
    p = points - origin[loop]
    q = points[following] - origin[loop]
    twice = (torch.linalg.cross(p, q, dim=1) * axis).sum(1)
    area = torch.zeros(count, dtype=torch.float64, device=device).index_add_(0, loop, twice) / 2
    moment = torch.zeros(count, 3, dtype=torch.float64, device=device)
    moment.index_add_(0, loop, twice[:, None] * (p + q))
    length = (q - p).norm(dim=1)
    closing = (following <= torch.arange(int(points.shape[0]), device=device)) & ~chains.closed[loop]
    perimeter = torch.zeros(count, dtype=torch.float64, device=device)
    perimeter.index_add_(0, loop, torch.where(closing, torch.zeros_like(length), length))
    sizes = (chains.offsets[1:] - chains.offsets[:-1]).clamp_min(1).double()
    mean = torch.zeros(count, 3, dtype=torch.float64, device=device).index_add_(0, loop, p) / sizes[:, None]
    span = perimeter.clamp_min(1e-300) ** 2
    flat = area.abs() <= 1e-12 * span
    centroid = origin + torch.where(flat[:, None], mean, moment / (6 * torch.where(flat, 1.0, area))[:, None])
    projected = points @ basis.T
    low = torch.full((count, 2), math.inf, dtype=torch.float64, device=device)
    high = torch.full((count, 2), -math.inf, dtype=torch.float64, device=device)
    low.scatter_reduce_(0, loop[:, None].expand(-1, 2), projected, reduce="amin")
    high.scatter_reduce_(0, loop[:, None].expand(-1, 2), projected, reduce="amax")
    return _LoopStats(area, perimeter, centroid, low, high)


def _reverse(chains: _Chains) -> _Chains:
    """The same loops walked the other way round."""
    loop = chains.loop_index()
    start = chains.offsets[:-1][loop]
    end = chains.offsets[1:][loop]
    index = torch.arange(int(chains.points.shape[0]), device=loop.device)
    closed = chains.closed[loop]
    # A closed loop keeps its first point; an open one swaps its ends.
    mirrored = torch.where(closed, torch.where(index == start, index, start + end - index), start + end - 1 - index)
    segment = torch.where(closed, start + end - 1 - index, start + end - 2 - index)
    crossed = closed | (index != end - 1)
    segment_face = torch.where(crossed, chains.segment_face[segment.clamp(min=0)], torch.full_like(index, -1))
    return _Chains(
        level=chains.level,
        closed=chains.closed,
        offsets=chains.offsets,
        points=chains.points[mirrored],
        segment_face=segment_face,
        edge_face=chains.edge_face[mirrored],
        edge_side=chains.edge_side[mirrored],
        edge_low=chains.edge_low[mirrored],
        edge_weight=chains.edge_weight[mirrored],
    )


def _nearest_ends(ends: torch.Tensor, level: torch.Tensor, tolerance: float) -> torch.Tensor:
    """For each polyline end, the nearest other end at the same level within ``tolerance``.

    Args:
        ends: ``(E, 3)`` float64 end points.
        level: ``(E,)`` int64 level of each end.
        tolerance: Largest distance searched.

    Returns:
        ``(E,)`` int64 index of the nearest other end, or -1.
    """
    device = ends.device
    count = int(ends.shape[0])
    rows = torch.cat([level[:, None], torch.floor(ends / tolerance).long()], 1)
    index = torch.arange(count, device=device)
    best = torch.full((count,), -1, dtype=torch.int64, device=device)
    best_gap = torch.full((count,), math.inf, dtype=torch.float64, device=device)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                query = rows + torch.tensor([0, dx, dy, dz], dtype=torch.int64, device=device)
                _, key = torch.unique(torch.cat([rows, query]), dim=0, return_inverse=True)
                own, wanted = key[:count], key[count:]
                order = torch.argsort(own)
                sorted_keys = own[order]
                low = torch.searchsorted(sorted_keys, wanted)
                high = torch.searchsorted(sorted_keys, wanted, right=True)
                for k in range(4):
                    slot = low + k
                    present = slot < high
                    candidate = order[slot.clamp_max(count - 1)]
                    gap = (ends[candidate] - ends).norm(dim=1)
                    better = present & (candidate != index) & (gap <= tolerance) & (gap < best_gap)
                    best = torch.where(better, candidate, best)
                    best_gap = torch.where(better, gap, best_gap)
    return best


def _bridge(chains: _Chains, tolerance: float) -> _Chains:
    """The closed loops of a cut, plus open polylines joined end to end into loops across gaps.

    Args:
        chains: Loops from one cut.
        tolerance: Widest gap bridged between two polyline ends.

    Returns:
        Closed loops only. Open polylines that no chain of bridges returns to its start are
        left out; a joined loop runs the way most of its pieces did.
    """
    # Ends are paired with their mutually nearest end at the same level, whichever way each polyline
    # runs, so a crack where the surface also turns over still closes.
    device = chains.points.device
    closed_ids = torch.nonzero(chains.closed).squeeze(1)
    open_ids = torch.nonzero(~chains.closed).squeeze(1)
    count = int(open_ids.shape[0])
    parts = closed_ids
    forward = torch.ones(int(closed_ids.shape[0]), dtype=torch.bool, device=device)
    sizes = torch.ones(int(closed_ids.shape[0]), dtype=torch.int64, device=device)
    if count and tolerance > 0:
        # End 2i is the start of open polyline i and end 2i + 1 its finish.
        ends = torch.stack(
            [chains.points[chains.offsets[:-1][open_ids]], chains.points[chains.offsets[1:][open_ids] - 1]], 1
        ).reshape(-1, 3)
        nearest = _nearest_ends(ends, chains.level[open_ids].repeat_interleave(2), tolerance)
        index = torch.arange(2 * count, device=device)
        mutual = (nearest >= 0) & (nearest[nearest.clamp_min(0)] == index)
        partner = torch.where(mutual, nearest, torch.full_like(nearest, -1))
        # State 2i walks polyline i forward and leaves by end 2i + 1; state 2i + 1 walks it
        # backward and leaves by end 2i. Entering by an end's partner picks the next state.
        leave = index ^ 1
        target = partner[leave]
        successor = torch.where(target >= 0, target, torch.full_like(target, -1))
        chain, distance = _chain_rank(successor)
        tails = distance == 0
        cyclic = torch.zeros(2 * count, dtype=torch.bool, device=device)
        cyclic[chain[tails & (successor >= 0)]] = True
        natural = (index & 1) == 0
        length = torch.bincount(chain, minlength=2 * count)
        agree = torch.bincount(chain[natural], minlength=2 * count)
        twin = chain[index ^ 1]
        kept = cyclic[chain] & (
            (2 * agree[chain] > length[chain]) | ((2 * agree[chain] == length[chain]) & (chain < twin))
        )
        states = torch.nonzero(kept).squeeze(1)
        if states.numel():
            states = states[torch.argsort(chain[states] * (2 * count + 1) + (2 * count - distance[states]))]
            _, joined = torch.unique_consecutive(chain[states], return_counts=True)
            parts = torch.cat([parts, open_ids[states >> 1]])
            forward = torch.cat([forward, (states & 1) == 0])
            sizes = torch.cat([sizes, joined])
    # Points of every part in loop order, reversed parts read backwards.
    counts = (chains.offsets[1:] - chains.offsets[:-1])[parts]
    part_offsets = _offsets(counts)
    part = torch.repeat_interleave(torch.arange(int(parts.shape[0]), device=device), counts)
    within = torch.arange(int(part.shape[0]), device=device) - part_offsets[:-1][part]
    within = torch.where(forward[part], within, counts[part] - 1 - within)
    source = chains.offsets[:-1][parts][part] + within
    loop_of_part = torch.repeat_interleave(torch.arange(int(sizes.shape[0]), device=device), sizes)
    loop_points = torch.zeros(int(sizes.shape[0]), dtype=torch.int64, device=device)
    loop_points.index_add_(0, loop_of_part, counts)
    first_part = _offsets(sizes)[:-1]
    # A reversed part crosses, from each point, the face its next point used to cross from.
    segment_face = chains.segment_face[source]
    backward = ~forward[part]
    if bool(backward.any()):
        previous = (source - 1).clamp_min(0)
        segment_face = torch.where(backward, torch.where(within > 0, chains.segment_face[previous], -1), segment_face)
    return _Chains(
        level=chains.level[parts[first_part]],
        closed=torch.ones(int(sizes.shape[0]), dtype=torch.bool, device=device),
        offsets=_offsets(loop_points),
        points=chains.points[source],
        segment_face=segment_face,
        edge_face=chains.edge_face[source],
        edge_side=chains.edge_side[source],
        edge_low=chains.edge_low[source],
        edge_weight=chains.edge_weight[source],
    )


def _permute(chains: _Chains, stats: _LoopStats, order: torch.Tensor) -> tuple[_Chains, _LoopStats]:
    """Loops reordered so loop ``i`` becomes the old loop ``order[i]``."""
    device = order.device
    counts = (chains.offsets[1:] - chains.offsets[:-1])[order]
    offsets = _offsets(counts)
    loop = torch.repeat_interleave(torch.arange(int(order.shape[0]), device=device), counts)
    index = torch.arange(int(loop.shape[0]), device=device)
    source = chains.offsets[:-1][order][loop] + index - offsets[:-1][loop]
    moved = _Chains(
        level=chains.level[order],
        closed=chains.closed[order],
        offsets=offsets,
        points=chains.points[source],
        segment_face=chains.segment_face[source],
        edge_face=chains.edge_face[source],
        edge_side=chains.edge_side[source],
        edge_low=chains.edge_low[source],
        edge_weight=chains.edge_weight[source],
    )
    measured = _LoopStats(
        stats.area[order], stats.perimeter[order], stats.centroid[order], stats.low[order], stats.high[order]
    )
    return moved, measured


# ---------------------------------------------------------------------------------------------
# Slicer


class Slicer:
    """A mesh prepared for repeated plane cuts.

    Attributes:
        vertex_count: Vertices in the input mesh.
        face_count: Triangles in the input mesh.
        device: Device the cuts run on.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        weld: bool = True,
        device: torch.device | str | None = None,
    ):
        """Prepare a mesh for cutting.

        Args:
            vertices: ``(V, 3)`` positions.
            faces: ``(F, 3)`` vertex indices.
            weld: Join vertices at exactly the same position first, so loops run on across
                texture seams.
            device: Device for the cuts; None means the mesh compute device.

        Raises:
            ValueError: For a mesh with no triangles, or faces indexing past the vertices.
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("the mesh vertices must be a (count, 3) table of positions")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("the mesh faces must be a (count, 3) table of triangle corners")
        self.vertex_count = int(vertices.shape[0])
        self.face_count = int(faces.shape[0])
        if self.face_count == 0:
            raise ValueError("the mesh has no triangles to cut; connect a model that has faces")
        if int(faces.min()) < 0 or int(faces.max()) >= self.vertex_count:
            raise ValueError("the mesh has faces that index past its vertices; repair the mesh first")
        self._output_device = vertices.device
        self._dtype = vertices.dtype if vertices.is_floating_point() else torch.float32

        def prepare(target: torch.device):
            positions = vertices.to(target)
            source = faces.to(target).long()
            if weld:
                welded, welded_faces, remap = topology.weld(positions, source, tolerance=0)
            else:
                welded, welded_faces = positions, source
                remap = torch.arange(self.vertex_count, device=target)
            return welded.double(), welded_faces, remap, source

        target = torch.device(device) if device is not None else mesh_device.compute_device()
        self._vertices, self._faces, self._remap, self._source = mesh_device.with_cpu_fallback(
            prepare, target, "mesh cross-sections"
        )
        self.device = self._vertices.device
        self._usable = torch.isfinite(self._vertices).all(1)[self._faces].all(1)
        if not bool(self._usable.any()):
            raise ValueError("every triangle of the mesh has a corner at an infinite or undefined position")

    def _on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Welded vertices, faces and the usable mask on ``device``."""
        return self._vertices.to(device), self._faces.to(device), self._usable.to(device)

    def extent(self, axis: Vector = (0.0, 1.0, 0.0)) -> tuple[float, float]:
        """Lowest and highest coordinate of the mesh along ``axis``.

        Args:
            axis: Direction to measure along; made unit length.

        Returns:
            ``(low, high)``.
        """
        direction = _vector(axis, self.device, "axis", unit=True)
        used = self._faces[self._usable].reshape(-1)
        along = self._vertices[used] @ direction
        return float(along.min()), float(along.max())

    def _chains(self, axis: torch.Tensor, heights: torch.Tensor) -> tuple[_Chains, _LoopStats, torch.Tensor]:
        """Loops at each height along ``axis``, wound so their total area is positive."""

        def work(target: torch.device):
            vertices, faces, usable = self._on(target)
            direction = axis.to(target)
            chains = _cut(vertices, faces, usable, vertices @ direction, heights.to(target))
            basis = _plane_basis(direction)
            stats = _measure(chains, direction, basis)
            if float(stats.area.sum()) < 0:
                chains = _reverse(chains)
                stats.area = -stats.area
            order = torch.argsort(-stats.area.abs(), stable=True)
            order = order[torch.argsort(chains.level[order], stable=True)]
            chains, stats = _permute(chains, stats, order)
            return chains, stats, basis

        return mesh_device.with_cpu_fallback(work, self.device, "mesh cross-sections")

    def _point_edges(self, chains: _Chains) -> tuple[torch.Tensor, torch.Tensor]:
        """Input vertex pairs and weights of the edge under each loop point."""
        source = self._source.to(chains.points.device)
        remap = self._remap.to(chains.points.device)
        first = source[chains.edge_face, chains.edge_side]
        second = source[chains.edge_face, (chains.edge_side + 1) % 3]
        weight = torch.where(remap[first] == chains.edge_low, chains.edge_weight, 1 - chains.edge_weight)
        return torch.stack([first, second], 1), weight

    def _loops(self, chains: _Chains, stats: _LoopStats, normal: torch.Tensor) -> list[Loop]:
        """`Loop` objects on the input's device."""
        edges, weights = self._point_edges(chains)
        out = self._output_device
        points = chains.points.to(out, self._dtype)
        faces = chains.segment_face.to(out)
        edges, weights = edges.to(out), weights.to(out, self._dtype)
        offsets = chains.offsets.tolist()
        closed = chains.closed.tolist()
        area, perimeter = stats.area.tolist(), stats.perimeter.tolist()
        centroid = stats.centroid.to(out, self._dtype)
        unit = normal.to(out, self._dtype)
        return [
            Loop(
                points=points[offsets[i]:offsets[i + 1]],
                closed=closed[i],
                faces=faces[offsets[i]:offsets[i + 1]],
                edges=edges[offsets[i]:offsets[i + 1]],
                weights=weights[offsets[i]:offsets[i + 1]],
                normal=unit,
                area=area[i],
                perimeter=perimeter[i],
                centroid=centroid[i],
            )
            for i in range(chains.count)
        ]

    def slice(self, origin: Vector, normal: Vector) -> list[Loop]:
        """Cut the mesh with one plane.

        Args:
            origin: A point on the plane.
            normal: The plane normal; made unit length.

        Returns:
            The loops, largest area first, wound so their total area is positive. A vertex
            exactly on the plane counts as above it.
        """
        direction = _vector(normal, self.device, "normal", unit=True)
        point = _vector(origin, self.device, "origin")
        heights = (point @ direction).reshape(1)
        chains, stats, _ = self._chains(direction, heights)
        return self._loops(chains, stats, direction)

    def heights(
        self,
        axis: Vector = (0.0, 1.0, 0.0),
        levels: int | Sequence[float] | torch.Tensor = 256,
        low: float | None = None,
        high: float | None = None,
    ) -> torch.Tensor:
        """Level heights a profile cuts at.

        Args:
            axis: Profile axis; made unit length.
            levels: A count of evenly spaced levels, or the heights themselves.
            low: Bottom of the measured span; None means the mesh's lowest point.
            high: Top of the measured span; None means the mesh's highest point.

        Returns:
            ``(L,)`` float64 ascending heights. A count places each level at the middle of one
            of ``levels`` equal bands between ``low`` and ``high``.

        Raises:
            ValueError: For no levels, more than `MAX_LEVELS`, or a height that is not finite.
        """
        if isinstance(levels, int) and not isinstance(levels, bool):
            if levels < 1 or levels > MAX_LEVELS:
                raise ValueError(f"levels must be between 1 and {MAX_LEVELS}, not {levels}")
            bottom, top = self.extent(axis)
            bottom = bottom if low is None else float(low)
            top = top if high is None else float(high)
            if not (math.isfinite(bottom) and math.isfinite(top)):
                raise ValueError("low and high must be finite numbers")
            step = (top - bottom) / levels
            return bottom + (torch.arange(levels, dtype=torch.float64, device=self.device) + 0.5) * step
        heights = torch.as_tensor(levels, dtype=torch.float64).reshape(-1).to(self.device)
        if heights.numel() < 1 or heights.numel() > MAX_LEVELS:
            raise ValueError(f"give between 1 and {MAX_LEVELS} level heights, not {heights.numel()}")
        if not bool(torch.isfinite(heights).all()):
            raise ValueError("every level height must be a finite number")
        return torch.sort(heights).values

    def profile(
        self,
        axis: Vector = (0.0, 1.0, 0.0),
        levels: int | Sequence[float] | torch.Tensor = 256,
        low: float | None = None,
        high: float | None = None,
    ) -> Profile:
        """Cut the mesh at many parallel levels in one pass.

        Args:
            axis: Profile axis; made unit length.
            levels: A count of evenly spaced levels, or the heights themselves.
            low: Bottom of the measured span; None means the mesh's lowest point.
            high: Top of the measured span; None means the mesh's highest point.

        Returns:
            The `Profile`, its loops wound so their total area over every level is positive.
        """
        direction = _vector(axis, self.device, "axis", unit=True)
        heights = self.heights(direction, levels, low, high)
        chains, stats, basis = self._chains(direction, heights)
        level_count = int(heights.shape[0])
        work = chains.points.device
        loop_count = torch.bincount(chains.level, minlength=level_count)
        open_count = torch.bincount(chains.level[~chains.closed], minlength=level_count)
        area = torch.zeros(level_count, dtype=torch.float64, device=work).index_add_(0, chains.level, stats.area)
        perimeter = torch.zeros(level_count, dtype=torch.float64, device=work)
        perimeter.index_add_(0, chains.level, stats.perimeter)
        edges, weights = self._point_edges(chains)
        out = self._output_device
        return Profile(
            axis=direction.to(out),
            basis=basis.to(out),
            heights=heights.to(out),
            loop_count=loop_count.to(out),
            open_count=open_count.to(out),
            area=area.to(out),
            perimeter=perimeter.to(out),
            loop_offsets=_offsets(loop_count).to(out),
            loop_level=chains.level.to(out),
            loop_closed=chains.closed.to(out),
            loop_area=stats.area.to(out),
            loop_perimeter=stats.perimeter.to(out),
            loop_centroid=stats.centroid.to(out),
            loop_low=stats.low.to(out),
            loop_high=stats.high.to(out),
            point_offsets=chains.offsets.to(out),
            points=chains.points.to(out, self._dtype),
            point_faces=chains.segment_face.to(out),
            point_edges=edges.to(out),
            point_weights=weights.to(out, self._dtype),
        )


def slice_plane(
    vertices: torch.Tensor, faces: torch.Tensor, origin: Vector, normal: Vector, weld: bool = True
) -> list[Loop]:
    """Cut a mesh with a plane and chain the crossings into polylines.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        origin: A point on the plane.
        normal: The plane normal.
        weld: Join vertices at exactly the same position first.

    Returns:
        The loops, largest area first. A vertex exactly on the plane counts as above it.
    """
    return Slicer(vertices, faces, weld=weld).slice(origin, normal)


def profile(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    axis: Vector = (0.0, 1.0, 0.0),
    levels: int | Sequence[float] | torch.Tensor = 256,
    low: float | None = None,
    high: float | None = None,
    weld: bool = True,
) -> Profile:
    """Cross-sections of a mesh at many levels along an axis, all cut in one pass.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        axis: Profile axis.
        levels: A count of evenly spaced levels, or the heights themselves.
        low: Bottom of the measured span; None means the mesh's lowest point.
        high: Top of the measured span; None means the mesh's highest point.
        weld: Join vertices at exactly the same position first.

    Returns:
        The `Profile`.
    """
    return Slicer(vertices, faces, weld=weld).profile(axis, levels, low, high)


# ---------------------------------------------------------------------------------------------
# Resampling


def ring(loop: Loop, count: int, start_direction: Vector = (0.0, 0.0, 1.0)) -> torch.Tensor:
    """Resample a loop to points evenly spaced along its length.

    Args:
        loop: The loop.
        count: Points to return.
        start_direction: A closed loop starts where a ray from its centroid in this
            direction, flattened onto the plane, last leaves it.

    Returns:
        ``(count, 3)`` points. A closed loop runs counter-clockwise seen from the tip of its
        normal; an open one runs from its first point to its last, both included.

    Raises:
        ValueError: For a loop with no points, or a count below 2 or above `MAX_RING_POINTS`.
    """
    if count < 2 or count > MAX_RING_POINTS:
        raise ValueError(f"count must be between 2 and {MAX_RING_POINTS}, not {count}")
    points = loop.points.double()
    size = int(points.shape[0])
    if size == 0:
        raise ValueError("the loop has no points to resample")
    if size == 1:
        return loop.points[:1].expand(count, 3).clone()
    device = points.device
    if loop.closed:
        normal = loop.normal.double().to(device)
        normal = normal / normal.norm().clamp_min(1e-300)
        if loop.area < 0:
            points = torch.cat([points[:1], points[1:].flip(0)])
        following = torch.roll(torch.arange(size, device=device), -1)
        centre = loop.centroid.double().to(device)
        direction = _vector(start_direction, device, "start_direction")
        direction = direction - (direction @ normal) * normal
        if float(direction.norm()) <= 1e-12:
            direction = points[0] - centre
            direction = direction - (direction @ normal) * normal
        if float(direction.norm()) <= 1e-300:
            direction = _plane_basis(normal)[0]
        first = direction / direction.norm()
        second = torch.linalg.cross(normal, first)
        relative = points - centre
        a, b = relative @ first, relative @ second
        a1, b1 = a[following], b[following]
        crossing = (b <= 0) != (b1 <= 0)
        fraction = torch.where(crossing, b / torch.where(crossing, b - b1, 1.0), torch.zeros_like(b))
        reach = a + fraction * (a1 - a)
        hit = crossing & (reach > 0)
        length = (points[following] - points).norm(dim=1)
        start = torch.cumsum(length, 0) - length
        total = float(length.sum())
        if total <= 0:
            return loop.points[:1].expand(count, 3).clone()
        if bool(hit.any()):
            best = int(torch.argmax(torch.where(hit, reach, torch.full_like(reach, -math.inf))))
            offset = float(start[best] + fraction[best] * length[best])
        else:
            best = int(torch.argmax(a / relative.norm(dim=1).clamp_min(1e-300)))
            offset = float(start[best])
        targets = torch.remainder(
            offset + torch.arange(count, dtype=torch.float64, device=device) * (total / count), total
        )
    else:
        following = torch.cat([torch.arange(1, size, device=device), torch.tensor([size - 1], device=device)])
        length = (points[following] - points).norm(dim=1)
        start = torch.cumsum(length, 0) - length
        total = float(length.sum())
        if total <= 0:
            return loop.points[:1].expand(count, 3).clone()
        targets = torch.linspace(0.0, total, count, dtype=torch.float64, device=device)
    segment = (torch.searchsorted(start, targets, right=True) - 1).clamp(0, size - 1)
    if not loop.closed:
        segment = segment.clamp_max(size - 2)
    span = length[segment]
    fraction = torch.where(span > 0, (targets - start[segment]) / torch.where(span > 0, span, 1.0), 0.0)
    fraction = fraction.clamp(0.0, 1.0)
    out = points[segment] + fraction[:, None] * (points[following[segment]] - points[segment])
    return out.to(loop.points.dtype)


# ---------------------------------------------------------------------------------------------
# Fitting


def _fit_points(points: torch.Tensor, name: str) -> torch.Tensor:
    """Finite float64 rows of ``points``, thinned evenly to `MAX_FIT_POINTS`."""
    points = torch.as_tensor(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} takes a (count, 3) table of points")
    points = points.double()
    points = points[torch.isfinite(points).all(1)]
    count = int(points.shape[0])
    if count == 0:
        raise ValueError(f"{name} needs at least one finite point; select a part of the mesh that has vertices")
    if count > MAX_FIT_POINTS:
        pick = torch.linspace(0, count - 1, MAX_FIT_POINTS, device=points.device).round().long()
        points = points[pick]
    return points


def _least_squares(residual, params: torch.Tensor, scale: float) -> torch.Tensor:
    """Levenberg-Marquardt on a function returning ``(residuals, jacobian)`` for parameters."""
    values, jacobian = residual(params)
    cost = float((values * values).sum())
    damping = 1e-3
    for _ in range(FIT_ITERATIONS):
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ values
        diagonal = torch.diagonal(normal).clone()
        floor = 1e-12 * float(diagonal.mean().clamp_min(1e-300))
        accepted = False
        step_size = 0.0
        improvement = 0.0
        for _ in range(8):
            system = normal + torch.diag(damping * (diagonal + floor))
            try:
                step = torch.linalg.solve(system, -gradient)
            except RuntimeError:
                damping *= 10
                continue
            trial = params + step
            trial_values, trial_jacobian = residual(trial)
            trial_cost = float((trial_values * trial_values).sum())
            if math.isfinite(trial_cost) and trial_cost <= cost:
                params, values, jacobian = trial, trial_values, trial_jacobian
                damping = max(damping / 3, 1e-9)
                accepted = True
                step_size = float(step.norm())
                improvement = cost - trial_cost
                cost = trial_cost
                break
            damping *= 4
        if not accepted or step_size <= 1e-12 * scale or improvement <= 1e-15 * max(cost, 1e-300):
            break
    return params


def fit_sphere(points: torch.Tensor) -> Sphere:
    """Fit a sphere to points by least squares on their distance from its surface.

    Args:
        points: ``(N, 3)`` positions.

    Returns:
        The `Sphere`, on the points' device.

    Raises:
        ValueError: For no finite points.
    """
    source = torch.as_tensor(points)
    data = _fit_points(source, "a sphere fit")
    device = data.device
    centre = data.mean(0)
    scale = float((data - centre).norm(dim=1).max().clamp_min(1e-300))
    radius = float((data - centre).norm(dim=1).mean())
    if data.shape[0] >= 4:
        local = (data - centre) / scale
        design = torch.cat([2 * local, torch.ones(local.shape[0], 1, dtype=torch.float64, device=device)], 1)
        target = (local * local).sum(1)
        system = design.T @ design + 1e-12 * torch.eye(4, dtype=torch.float64, device=device)
        solution = torch.linalg.solve(system, design.T @ target)
        squared = float(solution[3] + solution[:3] @ solution[:3])
        if squared > 0 and math.isfinite(squared) and float(solution[:3].norm()) < 1e3:
            centre = centre + solution[:3] * scale
            radius = math.sqrt(squared) * scale
    ones = torch.ones(data.shape[0], 1, dtype=torch.float64, device=device)

    def residual(params: torch.Tensor):
        offset = data - params[:3]
        distance = offset.norm(dim=1)
        direction = offset / distance.clamp_min(1e-300)[:, None]
        return distance - params[3], torch.cat([-direction, -ones], 1)

    start = torch.cat([centre, torch.tensor([radius], dtype=torch.float64, device=device)])
    params = _least_squares(residual, start, scale)
    values, _ = residual(params)
    dtype = source.dtype if source.is_floating_point() else torch.float32
    return Sphere(params[:3].to(dtype), abs(float(params[3])), float(values.pow(2).mean().sqrt()))


def _segment_distance(data: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
    """Distance of points from a segment, the unit offset direction and the segment parameter."""
    span = end - start
    squared = float(span @ span)
    if squared > 0:
        along = ((data - start) @ span / squared).clamp(0.0, 1.0)
    else:
        along = torch.zeros(data.shape[0], dtype=torch.float64, device=data.device)
    offset = data - (start + along[:, None] * span)
    distance = offset.norm(dim=1)
    direction = torch.where(distance[:, None] > 0, offset / distance.clamp_min(1e-300)[:, None], 0.0)
    return distance, direction, along


def fit_capsule(points: torch.Tensor, axis: Vector | None = None) -> Capsule:
    """Fit a capsule to points by least squares on their distance from its surface.

    Args:
        points: ``(N, 3)`` positions.
        axis: Direction the capsule must run along; None fits the direction too, starting
            from the points' longest spread.

    Returns:
        The `Capsule`, on the points' device. Start and end coincide when the points are
        rounder than long.

    Raises:
        ValueError: For no finite points, or a zero axis.
    """
    source = torch.as_tensor(points)
    data = _fit_points(source, "a capsule fit")
    device = data.device
    mean = data.mean(0)
    centred = data - mean
    scale = float(centred.norm(dim=1).max().clamp_min(1e-300))
    if axis is None:
        if data.shape[0] >= 2:
            direction = torch.linalg.eigh(centred.T @ centred)[1][:, -1]
        else:
            direction = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    else:
        direction = _vector(axis, device, "axis", unit=True)
    along = centred @ direction
    radial = (centred - along[:, None] * direction).norm(dim=1)
    radius = float(radial.median())
    middle = float(along.max() + along.min()) / 2
    half = max(float(along.max() - along.min()) / 2 - radius, 0.0)
    centre = mean + middle * direction
    ones = torch.ones(data.shape[0], 1, dtype=torch.float64, device=device)

    if axis is None:
        def residual(params: torch.Tensor):
            distance, offset, t = _segment_distance(data, params[:3], params[3:6])
            jacobian = torch.cat([-offset * (1 - t)[:, None], -offset * t[:, None], -ones], 1)
            return distance - params[6], jacobian

        first = torch.cat([
            centre - half * direction, centre + half * direction,
            torch.tensor([radius], dtype=torch.float64, device=device),
        ])
        params = _least_squares(residual, first, scale)
        values, _ = residual(params)
        start, end, radius = params[:3], params[3:6], abs(float(params[6]))
    else:
        def residual(params: torch.Tensor):
            middle_point, length = params[:3], params[3]
            distance, offset, t = _segment_distance(
                data, middle_point - length * direction, middle_point + length * direction
            )
            jacobian = torch.cat([-offset, -((offset @ direction) * (2 * t - 1))[:, None], -ones], 1)
            return distance - params[4], jacobian

        first = torch.cat([centre, torch.tensor([half, radius], dtype=torch.float64, device=device)])
        params = _least_squares(residual, first, scale)
        values, _ = residual(params)
        length = abs(float(params[3]))
        start, end = params[:3] - length * direction, params[:3] + length * direction
        radius = abs(float(params[4]))
    dtype = source.dtype if source.is_floating_point() else torch.float32
    return Capsule(start.to(dtype), end.to(dtype), radius, float(values.pow(2).mean().sqrt()))


# ---------------------------------------------------------------------------------------------
# Body measurements

#: Heights of body landmarks above the floor as fractions of standing height.
PROPORTIONS = {
    "crotch": 0.47,
    "hips": 0.52,
    "waist": 0.61,
    "chest": 0.71,
    "armpits": 0.74,
    "shoulders": 0.81,
    "neck": 0.865,
}

#: Most cells along one side of a level in the occupancy grid.
GRID_SIDE = 768

#: Polar bins around a head section, two degrees each.
POLAR_BINS = 180

#: Widest crack in a body's surface bridged when sections are filled, as a fraction of its height.
BRIDGE = 0.02

#: Least roundness, ``4 pi area / perimeter ** 2``, of a neck ring; a 3:1 ellipse has about 0.75.
RING_ROUNDNESS = 0.7

#: Surface points a head's facing is read from.
HEAD_SAMPLES = 3000

#: Cells along each side of the distance grid a head's mirror plane is scored on.
MIRROR_GRID = 48


def _rotation_y(degrees: float, device: torch.device) -> torch.Tensor:
    """``(3, 3)`` float64 turn about Y."""
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64, device=device)


def _resample(points: torch.Tensor, count: int) -> torch.Tensor:
    """A closed polyline resampled to ``count`` points evenly spaced along its length."""
    size = int(points.shape[0])
    if size < 2:
        return points[:1].expand(count, 3)
    following = torch.roll(torch.arange(size, device=points.device), -1)
    length = (points[following] - points).norm(dim=1)
    total = float(length.sum())
    if total <= 0:
        return points[:1].expand(count, 3)
    start = torch.cumsum(length, 0) - length
    targets = torch.arange(count, dtype=torch.float64, device=points.device) * (total / count)
    segment = (torch.searchsorted(start, targets, right=True) - 1).clamp(0, size - 1)
    span = length[segment]
    fraction = torch.where(span > 0, (targets - start[segment]) / torch.where(span > 0, span, 1.0), 0.0)
    return points[segment] + fraction.clamp(0, 1)[:, None] * (points[following[segment]] - points[segment])


def _contains(points: torch.Tensor, point: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> bool:
    """Whether a closed polyline, flattened onto two axes, encloses a point."""
    a = (points - point) @ first
    b = (points - point) @ second
    a1, b1 = torch.roll(a, -1), torch.roll(b, -1)
    crossing = (b <= 0) != (b1 <= 0)
    reach = a + torch.where(crossing, b / torch.where(crossing, b - b1, 1.0), 0.0) * (a1 - a)
    return int((crossing & (reach > 0)).sum()) % 2 == 1


def _polar(points: torch.Tensor, centre: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Outermost radius of a closed polyline in each of `POLAR_BINS` directions from a centre.

    Bin ``k`` looks along ``first`` turned ``2k`` degrees towards ``second``.
    """
    dense = _resample(points, POLAR_BINS * 8) - centre
    a, b = dense @ first, dense @ second
    angle = torch.remainder(torch.atan2(b, a), 2 * math.pi)
    bins = (angle / (2 * math.pi) * POLAR_BINS + 0.5).long() % POLAR_BINS
    radius = torch.full((POLAR_BINS,), -math.inf, dtype=torch.float64, device=points.device)
    radius.scatter_reduce_(0, bins, torch.hypot(a, b), reduce="amax")
    filled = torch.isfinite(radius)
    if not bool(filled.all()) and bool(filled.any()):
        index = torch.arange(POLAR_BINS, device=points.device)
        known = index[filled]
        values = radius[filled]
        wrapped = torch.cat([known - POLAR_BINS, known, known + POLAR_BINS])
        repeated = torch.cat([values, values, values])
        right = torch.searchsorted(wrapped, index.double().to(wrapped.dtype)).clamp(1, int(wrapped.shape[0]) - 1)
        left = right - 1
        span = (wrapped[right] - wrapped[left]).clamp_min(1)
        weight = (index - wrapped[left]) / span
        radius = torch.where(filled, radius, repeated[left] + weight * (repeated[right] - repeated[left]))
    return torch.where(torch.isfinite(radius), radius, torch.zeros_like(radius))


def _sections(slicer: Slicer, centre: torch.Tensor, up: torch.Tensor, first: torch.Tensor,
              heights: torch.Tensor, about_axis: bool) -> torch.Tensor:
    """Outermost polar radius of a body at several heights, one row per height that has a loop.

    Args:
        slicer: The prepared mesh.
        centre: ``(3,)`` float64 point on the axis.
        up: ``(3,)`` unit axis.
        first: ``(3,)`` unit direction of bin 0, at right angles to ``up``.
        heights: ``(H,)`` float64 heights along ``up``.
        about_axis: Measure from the axis line when True, from each loop's centroid when False.

    Returns:
        ``(rows, POLAR_BINS)`` float64 radii. At each height the closed loop enclosing the axis
        with the largest area is read, or the largest loop when none encloses it.
    """
    second = torch.linalg.cross(up, first)
    chains, stats, _ = slicer._chains(up, heights)
    points = chains.points
    rows = []
    if chains.count == 0:
        return torch.zeros(0, POLAR_BINS, dtype=torch.float64, device=points.device)
    offsets = chains.offsets.tolist()
    levels = chains.level.tolist()
    closed = chains.closed.tolist()
    areas = stats.area.abs().tolist()
    best: dict[int, tuple[bool, float, int]] = {}
    for index, level in enumerate(levels):
        if not closed[index]:
            continue
        axis_point = centre + (float(heights[level]) - float(centre @ up)) * up
        rank = (_contains(points[offsets[index]:offsets[index + 1]], axis_point, first, second), areas[index], index)
        if level not in best or rank > best[level]:
            best[level] = rank
    for level in sorted(best):
        index = best[level][2]
        axis_point = centre + (float(heights[level]) - float(centre @ up)) * up
        origin = axis_point if about_axis else stats.centroid[index].to(points.device)
        rows.append(_polar(points[offsets[index]:offsets[index + 1]], origin, first, second))
    if not rows:
        return torch.zeros(0, POLAR_BINS, dtype=torch.float64, device=points.device)
    return torch.stack(rows)


def _winding(points: torch.Tensor, loop: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Winding number of a closed polyline around each of several points, flattened on two axes.

    Args:
        points: ``(P, 3)`` query points.
        loop: ``(K, 3)`` polyline points.
        first: ``(3,)`` first in-plane axis.
        second: ``(3,)`` second in-plane axis; counter-clockwise runs from ``first`` to it.

    Returns:
        ``(P,)`` int64 winding numbers.
    """
    a = loop @ first
    b = loop @ second
    a1, b1 = torch.roll(a, -1), torch.roll(b, -1)
    x = (points @ first)[:, None]
    y = (points @ second)[:, None]
    side = (a1 - a)[None, :] * (y - b[None, :]) - (x - a[None, :]) * (b1 - b)[None, :]
    rising = (b[None, :] <= y) & (b1[None, :] > y) & (side > 0)
    falling = (b[None, :] > y) & (b1[None, :] <= y) & (side < 0)
    return rising.long().sum(1) - falling.long().sum(1)


def _smallest_ring(
    slicer: Slicer,
    centre: torch.Tensor,
    normals: list[torch.Tensor],
    shifts: list[float],
    smallest: float = 0.0,
    bridge: float = 0.0,
) -> tuple[Loop | None, torch.Tensor | None, torch.Tensor | None]:
    """The smallest closed loop enclosing an axis through a point, over several planes.

    Args:
        slicer: The prepared mesh.
        centre: ``(3,)`` float64 point the planes pass near.
        normals: Unit plane normals to try, the axis direction of each.
        shifts: Offsets of each plane from ``centre`` along its normal.
        smallest: Least area a ring may enclose.
        bridge: Widest crack in the surface bridged to close a loop.

    Returns:
        ``(loop, normal, origin)``, the loop wound counter-clockwise about ``normal``, or three
        Nones.
    """
    # A loop lying mostly inside another closed loop of its plane, such as one shell pushed into
    # another, is passed over, as is one less round than `RING_ROUNDNESS`.
    best = None
    for normal in normals:
        base = float(centre @ normal)
        heights = torch.tensor([base + shift for shift in shifts], dtype=torch.float64, device=slicer.device)
        cut, _, _ = slicer._chains(normal, heights)
        chains = _bridge(cut, bridge)
        basis = _plane_basis(normal)
        stats = _measure(chains, normal, basis)
        offsets = chains.offsets.tolist()
        levels = chains.level.tolist()
        areas = stats.area.abs().tolist()
        roundness = (4 * math.pi * stats.area.abs() / stats.perimeter.clamp_min(1e-300) ** 2).tolist()
        for index in range(chains.count):
            if areas[index] < smallest or roundness[index] < RING_ROUNDNESS:
                continue
            if best is not None and areas[index] >= best[0]:
                continue
            origin = centre + (float(heights[levels[index]]) - base) * normal
            points = chains.points[offsets[index]:offsets[index + 1]]
            if not _contains(points, origin.to(points.device), basis[0], basis[1]):
                continue
            pick = torch.linspace(0, int(points.shape[0]) - 1, min(16, int(points.shape[0])), device=points.device).long()
            samples = points[pick]
            total = torch.zeros(int(samples.shape[0]), dtype=torch.int64, device=points.device)
            for other in range(chains.count):
                if other != index and levels[other] == levels[index]:
                    total += _winding(samples, chains.points[offsets[other]:offsets[other + 1]], basis[0], basis[1])
            if 2 * int((total != 0).sum()) > int(samples.shape[0]):
                continue
            best = (areas[index], normal, origin, chains, stats, index)
    if best is None:
        return None, None, None
    _, normal, origin, chains, stats, index = best
    order = torch.tensor([index], dtype=torch.int64, device=chains.points.device)
    chains, stats = _permute(chains, stats, order)
    if float(stats.area[0]) < 0:
        chains = _reverse(chains)
        stats.area = -stats.area
    return slicer._loops(chains, stats, normal)[0], normal, origin


def _jaw_reach(slicer: Slicer, neck: torch.Tensor, up: torch.Tensor, forward: torch.Tensor,
               low: float, high: float) -> float:
    """How much further a head reaches along ``forward`` than behind it, just above the neck.

    Args:
        slicer: The prepared mesh.
        neck: ``(3,)`` float64 centre of the neck.
        up: ``(3,)`` unit axis of the neck.
        forward: ``(3,)`` unit candidate facing direction.
        low: Lowest height along ``up`` to read.
        high: Highest height along ``up`` to read.

    Returns:
        The difference in mean radius from the neck axis, front minus back, in mean radii.
    """
    if high <= low:
        return 0.0
    heights = torch.linspace(low, high, 8, dtype=torch.float64, device=slicer.device)
    radii = _sections(slicer, neck.to(slicer.device), up.to(slicer.device), forward.to(slicer.device), heights, True)
    if radii.shape[0] == 0:
        return 0.0
    window = list(range(-5, 6))
    front = radii[:, window].mean()
    back = radii[:, [POLAR_BINS // 2 + k for k in window]].mean()
    return float((front - back) / radii.mean().clamp_min(1e-12))


def _surface_samples(vertices: torch.Tensor, faces: torch.Tensor, count: int) -> torch.Tensor:
    """Points spread over a mesh's surface in proportion to area, the same on every run.

    Args:
        vertices: ``(V, 3)`` float64 positions.
        faces: ``(F, 3)`` int64 triangles with finite corners.
        count: Points to return.

    Returns:
        ``(count, 3)`` float64 points.
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * torch.linalg.cross(b - a, c - a, dim=1).norm(dim=1)
    total = torch.cumsum(area, 0)
    generator = torch.Generator().manual_seed(0)
    draw = torch.rand(count, 3, generator=generator, dtype=torch.float64).to(vertices.device)
    if float(total[-1]) <= 0:
        pick = (draw[:, 0] * int(faces.shape[0])).long().clamp_max(int(faces.shape[0]) - 1)
    else:
        pick = torch.searchsorted(total, draw[:, 0] * total[-1]).clamp_max(int(faces.shape[0]) - 1)
    s = draw[:, 1].sqrt()
    t = draw[:, 2]
    return a[pick] * (1 - s)[:, None] + b[pick] * (s * (1 - t))[:, None] + c[pick] * (s * t)[:, None]


def _distance_field(points: torch.Tensor, side: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Distance from each cell of a cubic grid around points to the nearest occupied cell.

    Args:
        points: ``(N, 3)`` float64 points.
        side: Cells along each side.

    Returns:
        ``(field, origin, cell)``: the ``(side, side, side)`` float64 distances, the corner of
        cell ``(0, 0, 0)`` and the cell side.
    """
    low, high = points.amin(0), points.amax(0)
    cell = float((high - low).max()) / (side - 4)
    if not cell > 0:
        cell = 1.0
    origin = low - 2 * cell
    index = torch.floor((points - origin) / cell).long().clamp(0, side - 1)
    field = torch.full((side, side, side), 1e30, dtype=torch.float64, device=points.device)
    field[index[:, 0], index[:, 1], index[:, 2]] = 0.0
    steps = torch.arange(side, dtype=torch.float64, device=points.device)
    cost = (steps[:, None] - steps[None, :]) ** 2
    # Exact squared distances, one axis at a time.
    for dim in range(3):
        moved = field.movedim(dim, -1)
        field = (moved[..., None, :] + cost).amin(-1).movedim(-1, dim)
    return field.sqrt() * cell, origin, cell


def _field_at(field: torch.Tensor, origin: torch.Tensor, cell: float, points: torch.Tensor) -> torch.Tensor:
    """Trilinear distance-field values at points; points off the grid read the largest value."""
    side = int(field.shape[0])
    grid = (points - origin) / cell - 0.5
    base = torch.floor(grid).long()
    fraction = grid - base
    value = torch.zeros(int(points.shape[0]), dtype=torch.float64, device=points.device)
    for corner in range(8):
        step = torch.tensor([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1], device=points.device)
        weight = torch.where(step.bool(), fraction, 1 - fraction).prod(1)
        index = (base + step).clamp(0, side - 1)
        value += weight * field[index[:, 0], index[:, 1], index[:, 2]]
    outside = ((grid < -0.5) | (grid > side - 0.5)).any(1)
    return torch.where(outside, torch.full_like(value, float(field.max())), value)


def _mirror_plane(
    points: torch.Tensor, centre: torch.Tensor, up: torch.Tensor, size: float
) -> tuple[torch.Tensor, float, float]:
    """The plane containing ``up`` that a set of surface points is most nearly mirror symmetric about.

    Args:
        points: ``(N, 3)`` float64 surface points.
        centre: ``(3,)`` float64 point the plane passes near.
        up: ``(3,)`` unit direction the plane contains.
        size: Length the plane's offset and the error are measured in.

    Returns:
        ``(normal, offset, error)``: the plane ``dot(p, normal) = offset`` and the trimmed mean
        distance of the mirrored points from the surface, in units of ``size``.
    """
    field, origin, cell = _distance_field(points, MIRROR_GRID)
    first = _plane_basis(up)[0]
    second = torch.linalg.cross(up, first)
    keep = max(1, int(0.6 * int(points.shape[0])))

    def score(angles: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        normals = torch.cos(angles)[:, None] * first + torch.sin(angles)[:, None] * second
        offsets = (normals @ centre)[:, None] + shifts[None, :] * size
        along = (points @ normals.T).T
        lift = along[:, None, :] - offsets[:, :, None]
        mirrored = points[None, None] - 2 * lift[..., None] * normals[:, None, None, :]
        distance = _field_at(field, origin, cell, mirrored.reshape(-1, 3)).reshape(lift.shape)
        return distance.sort(-1).values[..., :keep].mean(-1)

    device = points.device
    angles = torch.deg2rad(torch.arange(0.0, 180.0, 5.0, dtype=torch.float64, device=device))
    shifts = torch.arange(-4, 5, dtype=torch.float64, device=device) * 0.03
    errors = score(angles, shifts)
    best = int(torch.argmin(errors))
    angle, shift = angles[best // int(shifts.shape[0])], shifts[best % int(shifts.shape[0])]
    angles = angle + torch.deg2rad(torch.arange(-4, 5, dtype=torch.float64, device=device))
    shifts = shift + torch.arange(-3, 4, dtype=torch.float64, device=device) * 0.01
    errors = score(angles, shifts)
    best = int(torch.argmin(errors))
    angle, shift = angles[best // int(shifts.shape[0])], shifts[best % int(shifts.shape[0])]
    normal = torch.cos(angle) * first + torch.sin(angle) * second
    return normal, float(normal @ centre) + float(shift) * size, float(errors.min()) / size


def _head_facing(
    points: torch.Tensor, neck: torch.Tensor, axis: torch.Tensor, up: torch.Tensor, size: float
) -> tuple[torch.Tensor, float]:
    """Facing of a head from its mirror plane and the skull sitting behind the neck.

    Args:
        points: ``(N, 3)`` float64 points on the head's surface.
        neck: ``(3,)`` float64 neck ring centre.
        axis: ``(3,)`` unit normal of the neck ring, pointing into the head.
        up: ``(3,)`` unit direction the facing is at right angles to.
        size: Neck to crown distance.

    Returns:
        ``(forward, confidence)``: a unit vector at right angles to ``up`` in the mirror plane,
        on the side away from the head's mass measured across the neck's axis, and 0 to 1 for
        how far that mass sits off it.
    """
    normal, _, _ = _mirror_plane(points, neck, up, size)
    across = torch.linalg.cross(normal, up)
    across = across / across.norm()
    relative = points - neck
    above = (relative @ axis) > 0
    level = across - (across @ axis) * axis
    if not bool(above.any()) or float(level.norm()) <= 1e-9:
        return across, 0.0
    lean = float((relative[above] @ level).mean()) / (size * float(level.norm()))
    return (across if lean < 0 else -across), min(1.0, abs(lean) / 0.1)


@dataclass
class _Grid:
    """Horizontal sections of a body filled on a regular grid, split into pieces.

    Attributes:
        levels: Number of levels.
        rows: Cells along Z.
        columns: Cells along X.
        x0: X of the grid's first column edge.
        z0: Z of the grid's first row edge.
        cell: Cell side.
        labels: ``(L * rows * columns,)`` int64 piece of each cell, -1 when empty.
        level: Level of each piece.
        cells: Cell count of each piece.
        x: Centroid X of each piece.
        z: Centroid Z of each piece.
        x_low: Smallest X of each piece.
        x_high: Largest X of each piece.
        z_low: Smallest Z of each piece.
        z_high: Largest Z of each piece.
        at_level: Pieces of each level, as ranges.
        up: Piece above against the overlapping pieces and shared cells, most shared first.
        down: Piece below against the overlapping pieces and shared cells, most shared first.
    """

    levels: int
    rows: int
    columns: int
    x0: float
    z0: float
    cell: float
    labels: torch.Tensor
    level: list[int]
    cells: list[int]
    x: list[float]
    z: list[float]
    x_low: list[float]
    x_high: list[float]
    z_low: list[float]
    z_high: list[float]
    at_level: list[range]
    up: dict[int, list[tuple[int, int]]]
    down: dict[int, list[tuple[int, int]]]

    def area(self, piece: int) -> float:
        """Area of a piece."""
        return self.cells[piece] * self.cell * self.cell

    def above(self, piece: int) -> int:
        """The piece one level up sharing most cells with ``piece``, or -1."""
        found = self.up.get(piece)
        return found[0][0] if found else -1

    def below(self, piece: int) -> int:
        """The piece one level down sharing most cells with ``piece``, or -1."""
        found = self.down.get(piece)
        return found[0][0] if found else -1

    def lookup(self, level: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Piece under each point, searching the eight neighbouring cells when it is empty."""
        column = torch.floor((x - self.x0) / self.cell).long()
        row = torch.floor((z - self.z0) / self.cell).long()
        found = torch.full_like(column, -1)
        for dz, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            r, c = row + dz, column + dx
            inside = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.columns) & (level >= 0) & (level < self.levels)
            flat = ((level * self.rows + r.clamp(0, self.rows - 1)) * self.columns + c.clamp(0, self.columns - 1))
            piece = torch.where(inside, self.labels[flat.clamp(0, int(self.labels.shape[0]) - 1)], -1)
            found = torch.where(found >= 0, found, piece)
        return found


def _fill(
    points: torch.Tensor,
    following: torch.Tensor,
    point_level: torch.Tensor,
    point_sign: torch.Tensor,
    level_count: int,
    step: float,
) -> _Grid:
    """Fill closed horizontal loops onto a grid and split it into pieces.

    Args:
        points: ``(M, 3)`` float64 loop points in the body frame.
        following: ``(M,)`` int64 next point around each loop, wrapping.
        point_level: ``(M,)`` int64 level of each point.
        point_sign: ``(M,)`` int64 winding weight of each point's loop, +1 or -1; 0 leaves it out.
        level_count: Number of levels.
        step: Level spacing, the preferred cell side.

    Returns:
        The `_Grid`.
    """
    # A cell is filled where the loops wind around it a positive number of times.
    device = points.device
    x, z = points[:, 0], points[:, 2]
    span = max(float(x.max() - x.min()), float(z.max() - z.min()), step)
    cell = max(step, span / GRID_SIDE)
    while True:
        x0 = float(x.min()) - 2 * cell
        z0 = float(z.min()) - 2 * cell
        columns = int(math.ceil((float(x.max()) - x0) / cell)) + 3
        rows = int(math.ceil((float(z.max()) - z0) / cell)) + 3
        if level_count * rows * (columns + 1) <= MAX_GRID_CELLS:
            break
        cell *= 1.25
    gx = (x - x0) / cell - 0.5
    gz = (z - z0) / cell - 0.5
    ring = torch.nonzero(point_sign != 0).squeeze(1)
    gx0, gz0 = gx[ring], gz[ring]
    gx1, gz1 = gx[following[ring]], gz[following[ring]]
    low = torch.ceil(torch.minimum(gz0, gz1)).long()
    high = torch.ceil(torch.maximum(gz0, gz1)).long()
    count = (high - low).clamp_min(0)
    segment = torch.repeat_interleave(torch.arange(int(ring.shape[0]), device=device), count)
    begin = torch.cumsum(count, 0) - count
    row = low[segment] + torch.arange(int(segment.shape[0]), device=device) - begin[segment]
    a, b = gz0[segment], gz1[segment]
    fraction = (row.double() - a) / (b - a)
    crossing = gx0[segment] + fraction * (gx1[segment] - gx0[segment])
    column = torch.ceil(crossing).long().clamp(0, columns)
    sign = torch.where(b > a, 1, -1).to(torch.int32) * point_sign[ring[segment]].to(torch.int32)
    width = columns + 1
    delta = torch.zeros(level_count * rows * width, dtype=torch.int32, device=device)
    delta.index_add_(0, (point_level[ring[segment]] * rows + row.clamp(0, rows - 1)) * width + column, sign)
    winding = torch.cumsum(delta.reshape(level_count, rows, width), 2, dtype=torch.int32)[:, :, :columns]
    occupied = (winding > 0).reshape(-1)
    del delta, winding
    cells = torch.nonzero(occupied).squeeze(1)
    total = int(cells.shape[0])
    plane = rows * columns
    first, second = [], []
    for offset, valid in ((1, (cells % columns) < columns - 1), (columns, ((cells // columns) % rows) < rows - 1)):
        neighbour = cells + offset
        joined = valid & occupied[neighbour.clamp_max(int(occupied.shape[0]) - 1)]
        first.append(torch.nonzero(joined).squeeze(1))
        second.append(torch.searchsorted(cells, neighbour[joined]))
    roots = topology.union_find(total, torch.cat(first), torch.cat(second))
    _, piece = torch.unique(roots, return_inverse=True)
    pieces = int(piece.max()) + 1 if total else 0
    labels = torch.full((level_count * plane,), -1, dtype=torch.int64, device=device)
    labels[cells] = piece
    cell_level = cells // plane
    cell_x = x0 + ((cells % columns).double() + 0.5) * cell
    cell_z = z0 + (((cells // columns) % rows).double() + 0.5) * cell
    counts = torch.bincount(piece, minlength=pieces)
    sums_x = torch.zeros(pieces, dtype=torch.float64, device=device).index_add_(0, piece, cell_x)
    sums_z = torch.zeros(pieces, dtype=torch.float64, device=device).index_add_(0, piece, cell_z)

    def reduce(values: torch.Tensor, how: str) -> list[float]:
        start = math.inf if how == "amin" else -math.inf
        out = torch.full((pieces,), start, dtype=torch.float64, device=device)
        return out.scatter_reduce_(0, piece, values, reduce=how).tolist()

    piece_level = torch.zeros(pieces, dtype=torch.int64, device=device).scatter_reduce_(
        0, piece, cell_level, reduce="amax", include_self=False
    )
    upper = cells + plane
    inside = upper < level_count * plane
    shared = torch.zeros_like(inside)
    shared[inside] = occupied[upper[inside]]
    pairs_low = piece[shared]
    pairs_high = labels[upper[shared]]
    keys, overlap = torch.unique(pairs_low * max(pieces, 1) + pairs_high, return_counts=True)
    up: dict[int, list[tuple[int, int]]] = {}
    down: dict[int, list[tuple[int, int]]] = {}
    for key, size in zip(keys.tolist(), overlap.tolist()):
        lower_piece, upper_piece = divmod(key, max(pieces, 1))
        up.setdefault(lower_piece, []).append((upper_piece, size))
        down.setdefault(upper_piece, []).append((lower_piece, size))
    for table in (up, down):
        for entries in table.values():
            entries.sort(key=lambda item: -item[1])
    level_list = piece_level.tolist()
    per_level = torch.bincount(piece_level, minlength=level_count).tolist()
    at_level, start = [], 0
    for size in per_level:
        at_level.append(range(start, start + size))
        start += size
    count_list = counts.tolist()
    return _Grid(
        levels=level_count,
        rows=rows,
        columns=columns,
        x0=x0,
        z0=z0,
        cell=cell,
        labels=labels,
        level=level_list,
        cells=count_list,
        x=(sums_x / counts.clamp_min(1)).tolist(),
        z=(sums_z / counts.clamp_min(1)).tolist(),
        x_low=reduce(cell_x - cell / 2, "amin"),
        x_high=reduce(cell_x + cell / 2, "amax"),
        z_low=reduce(cell_z - cell / 2, "amin"),
        z_high=reduce(cell_z + cell / 2, "amax"),
        at_level=at_level,
        up=up,
        down=down,
    )


class _Body:
    """The measuring pass behind `landmarks`."""

    def __init__(self, slicer: Slicer, levels: int, facing: str):
        if levels < 64 or levels > MAX_LEVELS:
            raise ValueError(f"levels must be between 64 and {MAX_LEVELS}, not {levels}")
        if facing not in ("auto", "+z"):
            raise ValueError(f"facing must be 'auto' or '+z', not {facing!r}")
        self.slicer = slicer
        self.device = slicer.device
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=self.device)
        self.floor, self.top = slicer.extent(self.up)
        self.height = self.top - self.floor
        if self.height <= 0:
            raise ValueError("the mesh is flat along Y; a standing figure needs height along Y")
        self.count = levels
        self.step = self.height / levels
        heights = slicer.heights(self.up, levels)
        self.heights = heights.tolist()
        cut, _, _ = slicer._chains(self.up, heights)
        # Polylines broken by a crack in the surface are rejoined; hair cards and other open
        # sheets stay open and fill nothing.
        chains = _bridge(cut, BRIDGE * self.height)
        if chains.count == 0:
            raise ValueError(
                "no horizontal section of the mesh encloses anything, so it has no body to measure; "
                "connect a closed mesh, or close its holes first"
            )
        area = _measure(chains, self.up, _plane_basis(self.up)).area
        loop = chains.loop_index()
        self.points = chains.points
        self.following = chains.following()
        self.point_level = chains.level[loop]
        # Each shell's loops are wound so the shell's total section area is positive, keeping holes
        # within a shell while a shell wound inside out still adds.
        shell = topology.face_components(slicer._faces, int(slicer._vertices.shape[0]))
        loop_shell = shell[chains.segment_face[chains.offsets[:-1]].clamp_min(0)]
        shell_area = torch.zeros(int(shell.max()) + 1, dtype=torch.float64, device=self.device)
        shell_area.index_add_(0, loop_shell, area)
        self.point_sign = torch.where(shell_area[loop_shell] < 0, -1, 1)[loop]
        self.facing = facing
        self.notes: list[str] = []
        self.measured: dict[str, bool] = {}

    # Grid helpers -----------------------------------------------------------------------

    def level_at(self, fraction: float) -> int:
        """Level at a fraction of the standing height."""
        return min(self.count - 1, max(0, int(fraction * self.count)))

    def grid(self, heading: float) -> _Grid:
        """The occupancy grid in the frame turned ``heading`` degrees about Y."""
        turn = _rotation_y(heading, self.device)
        return _fill(self.points @ turn.T, self.following, self.point_level, self.point_sign, self.count, self.step)

    # Legs -------------------------------------------------------------------------------

    def legs(self, grid: _Grid) -> dict | None:
        """Two pieces traced from near the floor up to where they join."""
        smallest = (0.015 * self.height) ** 2
        start = None
        for level in range(self.level_at(0.02), self.level_at(0.32) + 1):
            pieces = sorted(
                (piece for piece in grid.at_level[level] if grid.area(piece) >= smallest),
                key=lambda piece: -grid.cells[piece],
            )[:5]
            best = None
            for i, a in enumerate(pieces):
                for b in pieces[i + 1:]:
                    ratio = grid.cells[b] / grid.cells[a]
                    gap = math.hypot(grid.x[a] - grid.x[b], grid.z[a] - grid.z[b])
                    if ratio >= 0.3 and gap >= 0.05 * self.height:
                        score = grid.cells[a] + grid.cells[b]
                        if best is None or score > best[0]:
                            best = (score, a, b)
            if best is not None:
                start = (level, best[1], best[2])
                break
        if start is None:
            return None
        level, a, b = start
        chain_a, chain_b = {level: a}, {level: b}
        merged = None
        while level + 1 < self.count and self.heights[level + 1] <= self.floor + 0.66 * self.height:
            a, b = grid.above(a), grid.above(b)
            if a < 0 or b < 0:
                return None
            level += 1
            if a == b:
                merged = level
                break
            chain_a[level], chain_b[level] = a, b
        if merged is None:
            return None
        a, b = chain_a[start[0]], chain_b[start[0]]
        for level in range(start[0] - 1, -1, -1):
            a, b = grid.below(a), grid.below(b)
            if a < 0 or b < 0:
                break
            chain_a[level], chain_b[level] = a, b
        return {"start": start[0], "merged": merged, "a": chain_a, "b": chain_b}

    def mean_xz(self, grid: _Grid, pieces: list[int]) -> tuple[float, float]:
        """Cell-weighted centroid of several pieces."""
        weight = sum(grid.cells[piece] for piece in pieces)
        if not weight:
            return 0.0, 0.0
        return (
            sum(grid.x[piece] * grid.cells[piece] for piece in pieces) / weight,
            sum(grid.z[piece] * grid.cells[piece] for piece in pieces) / weight,
        )

    def feet_forward(self, grid: _Grid, legs: dict) -> tuple[float, float]:
        """Mean shift from shin to foot over both legs, in the grid frame."""
        shift_x = shift_z = 0.0
        found = 0
        for chain in (legs["a"], legs["b"]):
            feet = [chain[level] for level in range(0, self.level_at(0.035) + 1) if level in chain]
            shins = [chain[level] for level in range(self.level_at(0.08), self.level_at(0.16) + 1) if level in chain]
            if feet and shins:
                fx, fz = self.mean_xz(grid, feet)
                sx, sz = self.mean_xz(grid, shins)
                shift_x += fx - sx
                shift_z += fz - sz
                found += 1
        return (shift_x / found, shift_z / found) if found else (0.0, 0.0)

    # Torso ------------------------------------------------------------------------------

    def torso(self, grid: _Grid, legs: dict | None) -> dict[int, int]:
        """The trunk piece at each level from the crotch, or mid height, to the top."""
        if legs is not None:
            level = legs["merged"]
            piece = grid.above(legs["a"][level - 1])
        else:
            level = self.level_at(PROPORTIONS["crotch"] + 0.03)
            pieces = list(grid.at_level[level])
            if not pieces:
                return {}
            piece = max(pieces, key=lambda item: grid.cells[item])
        chain = {}
        centre_x, centre_z = grid.x[piece], grid.z[piece]
        width = grid.x_high[piece] - grid.x_low[piece]
        while True:
            chain[level] = piece
            span = grid.x_high[piece] - grid.x_low[piece]
            # The centre line follows the trunk only where no arm widens it.
            if span <= 1.5 * width:
                centre_x = 0.7 * centre_x + 0.3 * grid.x[piece]
                centre_z = 0.7 * centre_z + 0.3 * grid.z[piece]
                width = 0.8 * width + 0.2 * span
            found = self.trunk_above(grid, level, piece, centre_x, centre_z, width)
            if found is None:
                return chain
            level, piece = found

    def trunk_above(
        self, grid: _Grid, level: int, piece: int, centre_x: float, centre_z: float, width: float
    ) -> tuple[int, int] | None:
        """The level and piece that continue the trunk upward, or None at its top."""
        # First choice is the nearest to the centre line of the pieces within a fifth of the trunk's
        # width of it, up to 6% of the height ahead, and of the next level's pieces sharing at least
        # a tenth of ``piece``'s cells; the search ahead crosses a collar with the neck rising
        # through its opening. Failing that, the nearest of the next levels' pieces that overlap
        # ``piece`` or lie within half the trunk's width, up to 1.5% of the height ahead.
        smallest = max(1.0, ((0.01 * self.height) / grid.cell) ** 2)
        shared = dict(grid.up.get(piece, []))

        def offset(item: int) -> float:
            return math.hypot(grid.x[item] - centre_x, grid.z[item] - centre_z)

        for reach, admit in (
            (self.level_at(0.06), lambda item, ahead: offset(item) <= 0.2 * width
                or (ahead == 1 and shared.get(item, 0) >= 0.1 * grid.cells[piece])),
            (self.level_at(0.015), lambda item, ahead: offset(item) <= 0.5 * width
                or (ahead == 1 and item in shared)),
        ):
            for ahead in range(1, max(1, reach) + 1):
                if level + ahead >= self.count:
                    break
                candidates = [
                    item for item in grid.at_level[level + ahead]
                    if grid.cells[item] >= smallest and admit(item, ahead)
                ]
                if candidates:
                    return level + ahead, min(candidates, key=offset)
        return None

    # Heading ----------------------------------------------------------------------------

    def heading(self) -> tuple[float, bool]:
        """Degrees about Y that turn the body to face +Z, and whether a facing cue was found."""
        grid = self.grid(0.0)
        legs = self.legs(grid)
        if legs is not None:
            level = legs["start"]
            a, b = legs["a"][level], legs["b"][level]
            side = (grid.x[a] - grid.x[b], grid.z[a] - grid.z[b])
        else:
            plane = grid.rows * grid.columns
            flat = torch.nonzero(grid.labels >= 0).squeeze(1)
            level = flat // plane
            flat = flat[(level >= self.level_at(0.45)) & (level <= self.level_at(0.8))]
            if flat.numel() < 3:
                return 0.0, False
            xs = grid.x0 + ((flat % grid.columns).double() + 0.5) * grid.cell
            zs = grid.z0 + (((flat // grid.columns) % grid.rows).double() + 0.5) * grid.cell
            spread = torch.stack([xs - xs.mean(), zs - zs.mean()])
            vector = torch.linalg.eigh(spread @ spread.T)[1][:, -1]
            side = (float(vector[0]), float(vector[1]))
        length = math.hypot(*side)
        if length <= 0:
            return 0.0, False
        # Front or back, a quarter turn from the line across the body.
        sagittal = (-side[1] / length, side[0] / length)
        vote = 0.0
        if legs is not None:
            shift = self.feet_forward(grid, legs)
            vote += (shift[0] * sagittal[0] + shift[1] * sagittal[1]) / self.height / 0.01
        chain = self.torso(grid, legs)
        neck_level = self.narrowest(grid, chain, self.level_at(0.6))
        if neck_level is not None:
            piece = chain[neck_level]
            neck = torch.tensor([grid.x[piece], self.heights[neck_level], grid.z[piece]], dtype=torch.float64, device=self.device)
            forward = torch.tensor([sagittal[0], 0.0, sagittal[1]], dtype=torch.float64, device=self.device)
            size = self.top - self.heights[neck_level]
            low = self.heights[neck_level]
            vote += _jaw_reach(self.slicer, neck, self.up, forward, low + 0.02 * size, low + 0.18 * size) / 0.1
        found = abs(vote) >= 0.5
        if not found:
            # Without a clear cue the facing nearer the input's +Z is kept.
            vote = sagittal[1] if abs(sagittal[1]) > 1e-9 else sagittal[0]
        front = sagittal if vote > 0 else (-sagittal[0], -sagittal[1])
        return -math.degrees(math.atan2(front[0], front[1])), found

    def narrowest(self, grid: _Grid, chain: dict[int, int], low: int) -> int | None:
        """Level of the smallest trunk section from ``low`` to near the head top."""
        candidates = [level for level in range(low, self.level_at(0.93) + 1) if level in chain]
        if not candidates:
            return None
        return min(candidates, key=lambda level: grid.cells[chain[level]])

    # Arms -------------------------------------------------------------------------------

    def branch(self, grid: _Grid, piece: int, blocked: set[int]) -> list[int]:
        """Pieces below ``piece`` traced down while they stay apart and no wider than a hand."""
        pieces = [piece]
        while True:
            below = grid.below(piece)
            if below < 0 or below in blocked or grid.cells[below] > 3 * grid.cells[piece] + 4:
                return pieces
            pieces.append(below)
            piece = below

    def arm_branch(
        self, grid: _Grid, torso: dict[int, int], blocked: set[int], start: int, stop: int, side: int
    ) -> dict | None:
        """The longest piece beside the torso that joins it from below on one side."""
        smallest = (0.012 * self.height) ** 2
        best = None
        for level in range(start + 1, stop):
            previous, current = torso.get(level - 1), torso.get(level)
            if previous is None or current is None:
                continue
            half = (grid.x_high[previous] - grid.x_low[previous]) / 2
            for piece in grid.at_level[level - 1]:
                if piece == previous or piece in blocked or grid.area(piece) < smallest:
                    continue
                if grid.above(piece) != current:
                    continue
                across = (grid.x[piece] - grid.x[previous]) * side
                if across < 0.5 * half or across < abs(grid.z[piece] - grid.z[previous]):
                    continue
                pieces = self.branch(grid, piece, blocked)
                if best is None or len(pieces) > len(best["pieces"]):
                    best = {"kind": "branch", "level": level, "piece": piece, "pieces": pieces}
        if best is None or len(best["pieces"]) * self.step < 0.06 * self.height:
            return None
        return best

    def arm_level(self, grid: _Grid, torso: dict[int, int], start: int, stop: int, side: int) -> dict | None:
        """The lowest level where the torso suddenly widens on one side, as level arms join."""

        def edge(level: int) -> float | None:
            piece = torso.get(level)
            if piece is None:
                return None
            return grid.x_high[piece] if side > 0 else -grid.x_low[piece]

        jump = 0.04 * self.height
        for level in range(start + self.level_at(0.12), stop):
            below, here = edge(level - 1), edge(level)
            if below is None or here is None or here - below < jump:
                continue
            ahead = [edge(level + k) for k in (1, 2)]
            if all(value is not None and value - below >= jump for value in ahead):
                return {"kind": "level", "level": level, "edge": below * side}
        return None

    def arm_lateral(self, grid: _Grid, torso: dict[int, int], stop: int, side: int) -> dict | None:
        """An arm read from the surface reaching out beside the torso between mid height and the
        neck."""
        # It finds an arm held out level in a sleeve that widens the trunk gradually, or one that
        # meets the trunk where its sections are broken.
        held = [level for level in range(self.level_at(0.55), stop) if level in torso]
        if not held:
            return None
        widths = sorted(grid.x_high[torso[level]] - grid.x_low[torso[level]] for level in held)
        half = widths[len(widths) // 4] / 2
        middle = sorted(grid.x[torso[level]] for level in held)[len(held) // 2]
        centre = self.face_centre
        reach = side * (centre[:, 0] - middle)
        member = (
            (reach > half + 0.03 * self.height)
            & (centre[:, 1] > self.heights[held[0]])
            & (centre[:, 1] < self.heights[min(stop, self.count - 1)] + 0.02 * self.height)
            & (self.face_piece >= 0)
        )
        if int(member.sum()) < 20 or float(reach[member].max()) < half + 0.12 * self.height:
            return None
        root = member & (reach < half + 0.06 * self.height)
        lowest = float(centre[root if bool(root.any()) else member, 1].min())
        level = min(self.count - 1, max(1, int((lowest - self.floor) / self.step) + 1))
        return {"kind": "level", "level": level, "edge": middle + side * half, "member": member}

    # Limbs ------------------------------------------------------------------------------

    def face_data(self, grid: _Grid) -> None:
        """Centroid, area and grid piece of every usable face, in the body frame."""
        vertices, faces = self.slicer._vertices, self.slicer._faces
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        self.face_area = 0.5 * torch.linalg.cross(b - a, c - a, dim=1).norm(dim=1)
        self.face_area = torch.where(self.slicer._usable, self.face_area, torch.zeros_like(self.face_area))
        self.face_centre = ((a + b + c) / 3) @ self.turn.T
        level = torch.floor((self.face_centre[:, 1] - self.floor) / self.step).long().clamp(0, self.count - 1)
        self.face_piece = grid.lookup(level, self.face_centre[:, 0], self.face_centre[:, 2])
        self.face_piece = torch.where(self.slicer._usable, self.face_piece, torch.full_like(self.face_piece, -1))

    def members(self, grid: _Grid, pieces: set[int]) -> torch.Tensor:
        """Faces lying in any of ``pieces``."""
        table = torch.zeros(max(len(grid.cells), 1), dtype=torch.bool, device=self.device)
        if pieces:
            table[torch.tensor(sorted(pieces), dtype=torch.int64, device=self.device)] = True
        return (self.face_piece >= 0) & table[self.face_piece.clamp_min(0)]

    def limb(self, axis_faces: torch.Tensor, member_faces: torch.Tensor, kind: str, anchor: torch.Tensor) -> Limb | None:
        """A straight limb fitted to faces, rooted at the end nearer ``anchor``."""
        if int(axis_faces.sum()) < 3:
            return None
        weight = self.face_area[axis_faces]
        total = float(weight.sum())
        if total <= 0:
            return None
        points = self.face_centre[axis_faces]
        fit = weight
        for _ in range(3):
            centre = (points * fit[:, None]).sum(0) / fit.sum().clamp_min(1e-300)
            relative = points - centre
            spread = (relative * fit[:, None]).T @ relative / fit.sum().clamp_min(1e-300)
            direction = torch.linalg.eigh(spread)[1][:, -1]
            distance = (relative - (relative @ direction)[:, None] * direction).norm(dim=1)
            # Faces far off the axis, such as a hand resting on a thigh, stop counting.
            fit = torch.where(distance <= 1.8 * distance.median(), weight, torch.zeros_like(weight))
        if kind == "leg":
            if float(direction[1]) > 0:
                direction = -direction
        elif float((centre - anchor) @ direction) < 0:
            direction = -direction
        distance = (relative - (relative @ direction)[:, None] * direction).norm(dim=1)
        order = torch.argsort(distance)
        cumulative = torch.cumsum(weight[order], 0)
        radius = float(distance[order][int(torch.searchsorted(cumulative, cumulative[-1] / 2).clamp_max(order.shape[0] - 1))])
        corners = self.slicer._faces[member_faces].reshape(-1)
        reach = (self.slicer._vertices[corners] @ self.turn.T - centre) @ direction
        if kind == "leg" and abs(float(direction[1])) > 0.3:
            root = centre + (anchor[1] + radius - centre[1]) / direction[1] * direction
            tip = centre + (self.floor - centre[1]) / direction[1] * direction
        else:
            root = centre + (float((anchor - centre) @ direction) - radius) * direction
            tip = centre + float(reach.max()) * direction
        source = self.slicer._source[member_faces].reshape(-1)
        return Limb(
            root=self.back(root),
            tip=self.back(tip),
            axis=self.back(direction),
            length=float((tip - root).norm()),
            radius=radius,
            centre=self.back(centre),
            vertices=torch.unique(source).to(self.slicer._output_device),
        )

    # Output -----------------------------------------------------------------------------

    def back(self, value: torch.Tensor) -> torch.Tensor:
        """A body-frame vector in the input's frame, dtype and device."""
        return (self.turn.T @ value.double()).to(self.slicer._output_device, self.slicer._dtype)

    def point(self, x: float, y: float, z: float) -> torch.Tensor:
        """A body-frame point as a float64 tensor."""
        return torch.tensor([x, y, z], dtype=torch.float64, device=self.device)

    def level_of(self, grid: _Grid, torso: dict[int, int], level: int, measured: bool) -> Level:
        """The torso section at a level, or an estimate from the body's centre line."""
        piece = torso.get(level)
        if piece is None:
            near = min(torso, key=lambda item: abs(item - level)) if torso else None
            piece = torso[near] if near is not None else None
            measured = False
        if piece is None:
            return Level(self.heights[level], self.back(self.point(0.0, self.heights[level], 0.0)), 0.0, 0.0, 0.0, False)
        return Level(
            height=self.heights[level],
            centre=self.back(self.point(grid.x[piece], self.heights[level], grid.z[piece])),
            width=grid.x_high[piece] - grid.x_low[piece],
            depth=grid.z_high[piece] - grid.z_low[piece],
            area=grid.area(piece),
            measured=measured,
        )

    def neck_ring(self, centre: torch.Tensor, smallest: float) -> tuple[Loop | None, torch.Tensor]:
        """The smallest ring of at least ``smallest`` area around the neck, over planes tilted and shifted around ``centre``."""
        normals = []
        for pitch in (-20, -10, 0, 10, 20, 30, 40):
            for roll in (-10, 0, 10):
                a, b = math.radians(pitch), math.radians(roll)
                tilted = self.point(-math.sin(b) * math.cos(a), math.cos(b) * math.cos(a), math.sin(a))
                normals.append(self.turn.T @ tilted)
        shifts = [k * 0.006 * self.height for k in (-2, -1, 0, 1, 2)]
        # Cracks are bridged only where they are small beside the neck itself.
        bridge = 0.25 * math.sqrt(smallest / (0.55 * math.pi)) if smallest > 0 else BRIDGE * self.height
        loop, normal, _ = _smallest_ring(self.slicer, self.turn.T @ centre, normals, shifts, smallest, bridge)
        if loop is None:
            return None, self.turn.T @ self.up
        return loop, normal

    def measure(self) -> Landmarks:
        """Every landmark of the body."""
        heading, cued = (0.0, True) if self.facing == "+z" else self.heading()
        if not cued:
            self.notes.append("No feet or jaw showed which way the body faces, so it is taken as facing +Z.")
        self.turn = _rotation_y(heading, self.device)
        grid = self.grid(heading)
        height, floor = self.height, self.floor
        legs = self.legs(grid)
        if legs is not None and self.heights[legs["merged"]] < floor + 0.3 * height:
            legs = None
            self.notes.append("The legs join below 30% of the height, as under a skirt or long coat.")
        if legs is not None:
            start = legs["start"]
            if grid.x[legs["a"][start]] < grid.x[legs["b"][start]]:
                legs["a"], legs["b"] = legs["b"], legs["a"]
        else:
            self.notes.append("The legs could not be traced apart, so the crotch is set from body proportions.")
        self.measured["legs"] = legs is not None

        torso = self.torso(grid, legs)
        if not torso:
            raise ValueError("no trunk was found in the mesh; check it is a standing figure, Y up")
        plane = grid.rows * grid.columns
        if legs is not None:
            merged = legs["merged"]
            left, right = legs["a"][merged - 1], legs["b"][merged - 1]
            here = grid.labels[merged * plane:(merged + 1) * plane]
            below = grid.labels[(merged - 1) * plane:merged * plane]
            bridge = torch.nonzero((here == torso[merged]) & (below != left) & (below != right)).squeeze(1)
            if bridge.numel():
                x = float(grid.x0 + ((bridge % grid.columns).double() + 0.5).mean() * grid.cell)
                z = float(grid.z0 + (((bridge // grid.columns) % grid.rows).double() + 0.5).mean() * grid.cell)
            else:
                x, z = (grid.x[left] + grid.x[right]) / 2, (grid.z[left] + grid.z[right]) / 2
            crotch_level = merged
            crotch = self.point(x, self.heights[merged] - self.step / 2, z)
        else:
            crotch_level = self.level_at(PROPORTIONS["crotch"])
            piece = torso.get(crotch_level, next(iter(torso.values())))
            crotch = self.point(grid.x[piece], floor + PROPORTIONS["crotch"] * height, grid.z[piece])
        self.measured["crotch"] = legs is not None

        # The neck is the trunk's narrowest section above 60% of the height; arms join below it.
        neck_level = self.narrowest(grid, torso, self.level_at(0.6))
        stop = neck_level if neck_level is not None else self.level_at(PROPORTIONS["neck"])
        blocked = set(torso.values())
        if legs is not None:
            blocked |= set(legs["a"].values()) | set(legs["b"].values())
        start = crotch_level
        found: dict[str, dict | None] = {}
        self.face_data(grid)
        for name, side in (("left", 1), ("right", -1)):
            found[name] = (
                self.arm_branch(grid, torso, blocked, start, stop, side)
                or self.arm_level(grid, torso, start, stop, side)
                or self.arm_lateral(grid, torso, stop, side)
            )
        armpits: dict[str, torch.Tensor] = {}
        arms: dict[str, Limb] = {}
        arm_faces = torch.zeros_like(self.face_piece, dtype=torch.bool)
        armpit_levels = []
        for name, side in (("left", 1), ("right", -1)):
            arm = found[name]
            if arm is None:
                level = self.level_at(PROPORTIONS["armpits"])
                piece = torso.get(level)
                if piece is None:
                    piece = torso[min(torso, key=lambda item: abs(item - level))]
                edge = grid.x_high[piece] if side > 0 else grid.x_low[piece]
                armpits[name] = self.point(edge, floor + PROPORTIONS["armpits"] * height, grid.z[piece])
                self.measured["armpit_" + name] = False
                continue
            level = arm["level"]
            previous = torso.get(level - 1)
            if previous is None:
                previous = torso[min(torso, key=lambda item: abs(item - level))]
            y = self.heights[level] - self.step / 2
            if arm["kind"] == "branch":
                piece = arm["piece"]
                inner = grid.x_low[piece] if side > 0 else grid.x_high[piece]
                edge = grid.x_high[previous] if side > 0 else grid.x_low[previous]
                anchor = self.point((inner + edge) / 2, y, (grid.z[piece] + grid.z[previous]) / 2)
                member = self.members(grid, set(arm["pieces"]))
            else:
                edge = arm["edge"]
                anchor = self.point(edge, y, grid.z[previous])
                centre = self.face_centre
                member = arm.get("member")
                if member is None:
                    member = (
                        (side * (centre[:, 0] - edge) > 0.02 * height)
                        & (centre[:, 1] > y - 0.03 * height)
                        & (centre[:, 1] < y + 0.2 * height)
                        & (self.face_piece >= 0)
                    )
            armpits[name] = anchor
            armpit_levels.append(level)
            self.measured["armpit_" + name] = True
            limb = self.limb(member, member, "arm", anchor)
            if limb is not None:
                # The joint sits above the armpit by more when the arm hangs lower.
                hang = max(0.0, -float(limb.axis[1]))
                joint = self.point(
                    float(anchor[0]) - side * 0.5 * limb.radius,
                    float(anchor[1]) + limb.radius * (1 + 1.5 * hang),
                    float(anchor[2]),
                )
                limb.root = self.back(joint)
                limb.length = float((limb.tip.double() - limb.root.double()).norm())
                arms[name] = limb
                arm_faces |= member

        legs_out: dict[str, Limb] = {}
        if legs is not None:
            bottom = self.level_at(0.06)
            for name, chain in (("left", legs["a"]), ("right", legs["b"])):
                member = self.members(grid, set(chain.values())) & ~arm_faces
                upper = member & (self.face_centre[:, 1] > floor + 0.06 * height)
                upper &= self.members(grid, {piece for level, piece in chain.items() if level >= bottom})
                limb = self.limb(upper, member, "leg", crotch)
                if limb is not None:
                    legs_out[name] = limb

        armpit_level = min(armpit_levels) if armpit_levels else self.level_at(PROPORTIONS["armpits"])
        span = max(1, self.level_at(0.1))

        def extreme(levels: range, largest: bool) -> int | None:
            held = [level for level in levels if level in torso]
            if not held:
                return None
            key = (lambda level: grid.cells[torso[level]])
            return max(held, key=key) if largest else min(held, key=key)

        hips_level = extreme(range(crotch_level, crotch_level + span + 1), True)
        chest_level = extreme(range(max(crotch_level + 1, armpit_level - span), armpit_level), True)
        waist_level = None
        if hips_level is not None and chest_level is not None and chest_level - hips_level > 1:
            waist_level = extreme(range(hips_level + 1, chest_level), False)
        named = {}
        for name, level in (("hips", hips_level), ("chest", chest_level), ("waist", waist_level)):
            self.measured[name] = level is not None
            if level is None:
                level = self.level_at(PROPORTIONS[name])
            named[name] = self.level_of(grid, torso, level, self.measured[name])

        neck_measured = neck_level is not None
        if neck_level is None:
            neck_level = self.level_at(PROPORTIONS["neck"])
            self.notes.append("No narrow neck was found above the shoulders; it is cut at 86.5% of the height.")
        piece = torso.get(neck_level)
        neck_point = self.point(grid.x[piece], self.heights[neck_level], grid.z[piece]) if piece is not None \
            else self.point(float(crotch[0]), self.heights[neck_level], float(crotch[2]))
        smallest = 0.55 * grid.area(piece) if piece is not None else 0.0
        loop, normal = self.neck_ring(neck_point, smallest)
        if loop is not None:
            area = abs(loop.area)
            neck = Neck(loop.centroid.clone(), normal.to(self.slicer._output_device, self.slicer._dtype),
                        math.sqrt(area / math.pi), loop, neck_measured)
        else:
            neck = Neck(self.back(neck_point), self.back(self.up), 0.0, None, False)
        self.measured["neck"] = neck.measured

        head_levels = [level for level in torso if level > neck_level]
        weight = sum(grid.cells[torso[level]] for level in head_levels)
        if weight:
            head = self.point(
                sum(grid.x[torso[level]] * grid.cells[torso[level]] for level in head_levels) / weight,
                sum(self.heights[level] * grid.cells[torso[level]] for level in head_levels) / weight,
                sum(grid.z[torso[level]] * grid.cells[torso[level]] for level in head_levels) / weight,
            )
        else:
            head = neck_point
        heights = torch.where(torch.isfinite(self.slicer._vertices[:, 1]), self.slicer._vertices[:, 1], -math.inf)
        top = self.slicer._vertices[int(torch.argmax(heights))]

        shoulders: dict[str, torch.Tensor] = {}
        chest = named["chest"]
        for name, side in (("left", 1), ("right", -1)):
            if name in arms:
                shoulders[name] = arms[name].root
                self.measured["shoulder_" + name] = True
            else:
                centre = self.turn @ chest.centre.double().to(self.device)
                shoulders[name] = self.back(self.point(
                    float(centre[0]) + side * 0.4 * chest.width, floor + PROPORTIONS["shoulders"] * height, float(centre[2])
                ))
                self.measured["shoulder_" + name] = False
        angles = [math.degrees(math.acos(max(-1.0, min(1.0, -float(arm.axis[1]))))) for arm in arms.values()]
        angle = sum(angles) / len(angles) if angles else 0.0
        pose = "t-pose" if angle >= 65 else "a-pose" if angle >= 20 else "down"
        if len(arms) < 2:
            self.notes.append("Not both arms could be traced apart from the torso; missing armpits and shoulders are set from body proportions.")
        return Landmarks(
            floor=floor,
            height=height,
            heading=heading,
            head_top=top.to(self.slicer._output_device, self.slicer._dtype),
            head_centre=self.back(head),
            crotch=self.back(crotch),
            hips=named["hips"],
            waist=named["waist"],
            chest=named["chest"],
            armpits={name: self.back(value) for name, value in armpits.items()},
            shoulders=shoulders,
            neck=neck,
            legs=legs_out,
            arms=arms,
            arm_angle=angle,
            arm_pose=pose,
            legs_separable=legs is not None,
            arms_separable=len(arms) == 2,
            measured=self.measured,
            notes=self.notes,
        )


def landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    levels: int = 320,
    facing: str = "auto",
    weld: bool = True,
) -> Landmarks:
    """Measure a standing humanoid: crotch, hips, waist, chest, armpits, shoulders, neck, head, limbs.

    Args:
        vertices: ``(V, 3)`` positions, Y up.
        faces: ``(F, 3)`` vertex indices.
        levels: Horizontal sections from floor to head top, 64 to `MAX_LEVELS`.
        facing: ``"auto"`` reads which way the body faces from its feet and jaw; ``"+z"``
            takes it as facing +Z.
        weld: Join vertices at exactly the same position first.

    Returns:
        The `Landmarks`, in the input's coordinates.

    Raises:
        ValueError: For a mesh with no triangles, no height along Y, or no enclosed section.
    """
    return _Body(Slicer(vertices, faces, weld=weld), levels, facing).measure()


def _boundary_neck(
    slicer: Slicer, samples: torch.Tensor, diagonal: float, prior: torch.Tensor
) -> Loop | None:
    """The open boundary a head's neck ends in, or None when the mesh has none.

    Args:
        slicer: The prepared head mesh.
        samples: ``(N, 3)`` float64 points on the head's surface.
        diagonal: Diagonal of the head's box.
        prior: ``(3,)`` float64 unit direction the neck-to-crown axis most likely takes.

    Returns:
        The loop, its normal pointing into the head and wound counter-clockwise about it; its
        ``faces`` are the rim triangles and its ``edges`` repeat the boundary vertex, weight 0.
    """
    # Candidates are round closed boundary loops at least a tenth of ``diagonal`` long with the head
    # on one side of them and at most 12% of ``diagonal`` of it past their plane. Those facing
    # within 60 degrees of ``prior`` are preferred and the longest of those within 5% of
    # ``diagonal`` of the lowest along it is taken; when none faces that way, the longest of those
    # whose rim runs on as a tube. A candidate it encloses, as a neck inside a rim of hair, is taken
    # in its place.
    usable = torch.nonzero(slicer._usable).squeeze(1)
    faces = slicer._faces[usable]
    vertices = slicer._vertices
    count = int(vertices.shape[0])
    edges = topology.edge_topology(faces, count)
    chains, closed = topology.boundary_chains(faces, count, edges)
    stride = max(count, 1)
    keys = edges.edges[:, 0] * stride + edges.edges[:, 1]
    key_order = torch.argsort(keys)
    sorted_keys = keys[key_order]
    face_normals = topology.face_normals(vertices, faces)
    inside = samples.mean(0)

    def rim(chain: torch.Tensor) -> torch.Tensor:
        following = torch.roll(chain, -1)
        low, high = torch.minimum(chain, following), torch.maximum(chain, following)
        slot = torch.searchsorted(sorted_keys, low * stride + high).clamp_max(int(keys.shape[0]) - 1)
        return edges.edge_faces[key_order[slot], 0].clamp_min(0)

    candidates = []
    for chain, is_closed in zip(chains, closed.tolist()):
        if not is_closed or chain.numel() < 3:
            continue
        points = vertices[chain]
        perimeter = float((torch.roll(points, -1, 0) - points).norm(dim=1).sum())
        if perimeter < 0.1 * diagonal:
            continue
        mean = points.mean(0)
        relative = points - mean
        vector = 0.5 * torch.linalg.cross(relative, torch.roll(relative, -1, 0), dim=1).sum(0)
        projected = float(vector.norm())
        if projected <= 0 or 4 * math.pi * projected / perimeter ** 2 < 0.3:
            continue
        normal = vector / projected
        if float((inside - mean) @ normal) < 0:
            normal = -normal
        lift = (samples - mean) @ normal
        if float((lift > 0).double().mean()) < 0.85 or float(-lift.min()) > 0.12 * diagonal:
            continue
        tube = float((face_normals[rim(chain)] @ normal).abs().mean())
        candidates.append((perimeter, chain, normal, tube, float(mean @ prior)))
    if not candidates:
        return None
    facing = [item for item in candidates if float(item[2] @ prior) >= 0.5]
    if facing:
        lowest = min(item[4] for item in facing)
        bottom = [item for item in facing if item[4] <= lowest + 0.05 * diagonal]
        perimeter, chain, normal, _, _ = max(bottom, key=lambda item: item[0])
    else:
        tubes = [item for item in candidates if item[3] <= 0.6] or candidates
        perimeter, chain, normal, _, _ = max(tubes, key=lambda item: item[0])
    while True:
        plane = _plane_basis(normal)
        ring_points = vertices[chain]
        inner = [
            item for item in candidates
            if item[0] < perimeter and _contains(ring_points, vertices[item[1]].mean(0), plane[0], plane[1])
        ]
        if not inner:
            break
        perimeter, chain, normal, _, _ = max(inner, key=lambda item: item[0])
    mean = vertices[chain].mean(0)
    relative = vertices[chain] - mean
    twice = torch.linalg.cross(relative, torch.roll(relative, -1, 0), dim=1) @ normal
    if float(twice.sum()) < 0:
        chain = torch.cat([chain[:1], chain[1:].flip(0)])
        relative = vertices[chain] - mean
        twice = torch.linalg.cross(relative, torch.roll(relative, -1, 0), dim=1) @ normal
    area = float(twice.sum()) / 2
    moment = (twice[:, None] * (relative + torch.roll(relative, -1, 0))).sum(0)
    centroid = mean + (moment / (6 * area) if area > 1e-30 else relative.mean(0))
    face = usable[rim(chain)]
    first = torch.full((count,), slicer.vertex_count, dtype=torch.int64, device=vertices.device)
    first.scatter_reduce_(0, slicer._remap, torch.arange(slicer.vertex_count, device=vertices.device), reduce="amin")
    out, dtype = slicer._output_device, slicer._dtype
    source = first[chain]
    return Loop(
        points=vertices[chain].to(out, dtype),
        closed=True,
        faces=face.to(out),
        edges=torch.stack([source, source], 1).to(out),
        weights=torch.zeros(int(chain.shape[0]), dtype=dtype, device=out),
        normal=normal.to(out, dtype),
        area=area,
        perimeter=perimeter,
        centroid=centroid.to(out, dtype),
    )


def _steady_narrowest(
    slicer: Slicer, direction: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor, _Chains, _LoopStats, int] | None:
    """The narrowest section across the lower 60% of a closed mesh along ``direction``.

    Args:
        slicer: The prepared mesh.
        direction: ``(3,)`` float64 unit axis.

    Returns:
        ``(area, centre, line, chains, stats, index)``: the section's area and centre, the
        direction of the line through the centres of the steady sections around it (or
        ``direction``), and the cut holding it as loop ``index``; None when no level holds a loop.
    """
    # Only sections whose neighbours show the surface running on as a tube are compared, which
    # passes over a narrowing end cap.
    count = 60
    low, high = slicer.extent(direction)
    span = high - low
    heights = torch.linspace(low + 0.01 * span, low + 0.6 * span, count, dtype=torch.float64, device=slicer.device)
    cut, _, _ = slicer._chains(direction, heights)
    chains = _bridge(cut, 0.01 * span)
    if chains.count == 0:
        return None
    stats = _measure(chains, direction, _plane_basis(direction))
    areas = stats.area.abs()
    largest = torch.zeros(count, dtype=torch.float64, device=areas.device)
    largest.scatter_reduce_(0, chains.level, areas, reduce="amax")
    owner = torch.full((count,), -1, dtype=torch.int64, device=areas.device)
    top = areas == largest[chains.level]
    owner[chains.level[top]] = torch.nonzero(top).squeeze(1)
    section = largest.tolist()
    steady = [
        level for level in range(2, count - 2)
        if section[level] > 0 and section[level - 2] > 0 and section[level + 2] > 0
        and abs(section[level + 2] - section[level - 2]) <= 0.35 * section[level]
    ]
    held = steady or [level for level in range(count) if section[level] > 0]
    level = min(held, key=lambda item: section[item])
    centre = stats.centroid[int(owner[level])]
    line = direction
    near = [item for item in steady if abs(item - level) <= 6]
    if len(near) >= 4:
        centres = stats.centroid[owner[torch.tensor(near, device=owner.device)]]
        spread = centres - centres.mean(0)
        fitted = torch.linalg.eigh(spread.T @ spread)[1][:, -1]
        fitted = fitted if float(fitted @ direction) >= 0 else -fitted
        if float(fitted @ direction) >= math.cos(math.radians(30)):
            line = fitted
            centre = centres.mean(0) + float((centre - centres.mean(0)) @ fitted) * fitted
    return section[level], centre, line, chains, stats, int(owner[level])


def _closed_neck(slicer: Slicer, axis: torch.Tensor) -> Loop | None:
    """The narrowest ring across the lower part of a closed head standing near ``axis``.

    Args:
        slicer: The prepared head mesh.
        axis: ``(3,)`` float64 unit neck-to-crown direction.

    Returns:
        The ring, wound counter-clockwise about its normal, or None.
    """
    # The narrowest steady section is sought along ``axis`` and along axes tilted 15 and 30 degrees
    # from it, and the smallest round ring is then sought over planes tilted up to 22 degrees from
    # the best of them.
    basis = _plane_basis(axis)
    directions = [axis]
    for tilt in (15, 30):
        for turn in range(0, 360, 45):
            t, a = math.radians(tilt), math.radians(turn)
            directions.append(math.cos(t) * axis + math.sin(t) * (math.cos(a) * basis[0] + math.sin(a) * basis[1]))
    best = None
    for direction in directions:
        found = _steady_narrowest(slicer, direction / direction.norm())
        if found is not None and (best is None or found[0] < best[0]):
            best = found
    if best is None:
        return None
    area, centre, line, chains, stats, index = best
    low, high = slicer.extent(line)
    span = high - low
    plane = _plane_basis(line)
    normals = []
    for pitch in (-16, -8, 0, 8, 16):
        for roll in (-16, -8, 0, 8, 16):
            p, r = math.radians(pitch), math.radians(roll)
            normal = math.cos(p) * math.cos(r) * line + math.sin(p) * plane[0] + math.sin(r) * plane[1]
            normals.append(normal / normal.norm())
    ring, _, _ = _smallest_ring(slicer, centre, normals, [0.0, 0.01 * span], 0.55 * area, 0.01 * span)
    if ring is not None:
        return ring
    order = torch.tensor([index], dtype=torch.int64, device=chains.points.device)
    single, single_stats = _permute(chains, stats, order)
    if float(single_stats.area[0]) < 0:
        single = _reverse(single)
        single_stats.area = -single_stats.area
    return slicer._loops(single, single_stats, line)[0]


def head_frame(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    up: Vector | None = None,
    forward: Vector | None = None,
    weld: bool = True,
) -> HeadFrame:
    """Find the neck ring, up direction, facing and size of a head-only mesh.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        up: Neck-to-crown direction to use; None reads it from the neck ring. An open neck
            facing near +Y is preferred among several boundaries, and a closed head is read
            standing along +Y.
        forward: Facing direction to use; None reads it from the head's mirror plane, the face
            lying on the side away from the skull's mass behind the neck.
        weld: Join vertices at exactly the same position first.

    Returns:
        The `HeadFrame`, in the input's coordinates.

    Raises:
        ValueError: For a mesh with no triangles, no section across its lower part, or a
            ``forward`` along ``up``.
    """
    slicer = Slicer(vertices, faces, weld=weld)
    device = slicer.device
    points = slicer._vertices
    faces_used = slicer._faces[slicer._usable]
    corners = points[faces_used.reshape(-1)]
    diagonal = float((corners.amax(0) - corners.amin(0)).norm())
    out, dtype = slicer._output_device, slicer._dtype

    given_up = _vector(up, device, "up", unit=True) if up is not None else None
    samples = _surface_samples(points, faces_used, HEAD_SAMPLES)
    prior = given_up if given_up is not None else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
    neck = _boundary_neck(slicer, samples, diagonal, prior)
    open_neck = neck is not None
    if neck is None:
        axis = given_up if given_up is not None else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)
        low, high = slicer.extent(axis)
        if high - low <= 0:
            raise ValueError("the head mesh is flat along its up direction; give the up direction it stands along")
        neck = _closed_neck(slicer, axis)
        if neck is None:
            raise ValueError("no closed section was found across the head; check the mesh is a head, crown up")
    axis = given_up if given_up is not None else neck.normal.double().to(device)
    axis = axis / axis.norm()
    neck_centre = neck.centroid.double().to(device)
    height = float(neck_centre @ axis)
    top = float((corners @ axis).max())
    size = top - height
    if size <= 0:
        raise ValueError("the neck ring is at the top of the mesh; give the up direction from neck to crown")
    confidence = 1.0
    if forward is not None:
        facing = _vector(forward, device, "forward", unit=True)
        facing = facing - (facing @ axis) * axis
        if float(facing.norm()) <= 1e-9:
            raise ValueError("forward runs along the up direction; give a facing direction across it")
        facing = facing / facing.norm()
    else:
        ring_axis = neck.normal.double().to(device)
        ring_axis = ring_axis / ring_axis.norm()
        facing, confidence = _head_facing(samples, neck_centre, ring_axis, axis, size)
    left = torch.linalg.cross(axis, facing)
    frame = torch.stack([left, axis, facing])
    local = corners @ frame.T
    low, high = local.amin(0), local.amax(0)
    low[1] = height
    middle = (low + high) / 2
    centre = middle @ frame
    return HeadFrame(
        neck=neck,
        neck_centre=neck_centre.to(out, dtype),
        neck_radius=math.sqrt(abs(neck.area) / math.pi),
        open_neck=open_neck,
        up=axis.to(out, dtype),
        forward=facing.to(out, dtype),
        left=left.to(out, dtype),
        centre=centre.to(out, dtype),
        size=size,
        width=float(high[0] - low[0]),
        depth=float(high[2] - low[2]),
        confidence=confidence,
    )
