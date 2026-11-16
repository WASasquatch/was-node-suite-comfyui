"""Multiresolution graph hierarchy of a surface, with alignment constraints and vertex colouring.

Each level halves the vertex count by merging matched neighbour pairs. Vertices of one colour
share no edge, so a colour is updated in one parallel step.
"""

from __future__ import annotations

__all__ = ["Constraints", "Level", "build", "colour", "match_pairs", "mean_edge_length", "smooth_normals"]

from dataclasses import dataclass, field

import torch

#: Proposal rounds per level when matching vertex pairs.
MATCH_ROUNDS = 8

#: Levels stop shrinking below this many vertices.
MIN_VERTICES = 8

#: Most levels built.
MAX_LEVELS = 48

#: A level is kept only when it holds at most this share of the finer level's vertices.
MIN_SHRINK = 0.97

#: Colours tracked per vertex in the colouring; a vertex needing more takes the overflow colour.
TRACKED_COLOURS = 64


@dataclass
class Constraints:
    """Alignment constraints per vertex.

    Attributes:
        direction: ``(V, 3)`` tangent the orientation field is pulled to, or zero.
        direction_weight: ``(V,)`` pull strength, 0 to 1.
        point: ``(V, 3)`` point a lattice line or lattice point is pulled to.
        line: ``(V, 3)`` unit direction of that line, or zero to pull a lattice point onto ``point``.
        point_weight: ``(V,)`` pull strength, 0 to 1.
    """

    direction: torch.Tensor
    direction_weight: torch.Tensor
    point: torch.Tensor
    line: torch.Tensor
    point_weight: torch.Tensor

    @classmethod
    def empty(cls, vertices: torch.Tensor) -> Constraints:
        """No constraint on any vertex."""
        zero = torch.zeros_like(vertices)
        none = torch.zeros(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
        return cls(zero, none, vertices.clone(), zero.clone(), none.clone())


@dataclass
class Level:
    """One level of the hierarchy.

    Attributes:
        vertices: ``(V, 3)`` positions.
        normals: ``(V, 3)`` unit normals.
        areas: ``(V,)`` surface area each vertex stands for.
        scales: ``(V,)`` lattice spacing.
        first: ``(E,)`` int64 one end of each undirected edge.
        second: ``(E,)`` int64 the other end.
        weights: ``(E,)`` edge weights.
        constraints: The level's `Constraints`.
        parent: ``(V,)`` int64 vertex of the next coarser level, or None on the coarsest.
        order: Vertices sorted by colour, once coloured.
        src: Directed edges sorted by the colour of their source, once coloured.
        dst: Their targets.
        src_local: Position of each source within its colour's run of ``order``.
        edge_weight: Weight of each directed edge.
        vertex_offsets: Start of each colour in ``order``, and the end.
        edge_offsets: Start of each colour's edges, and the end.
        q: ``(V, 3)`` orientation field, once solved.
        o: ``(V, 3)`` position field, once solved.
    """

    vertices: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    scales: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    weights: torch.Tensor
    constraints: Constraints
    parent: torch.Tensor | None = None
    order: torch.Tensor | None = None
    src: torch.Tensor | None = None
    dst: torch.Tensor | None = None
    src_local: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    vertex_offsets: list[int] = field(default_factory=list)
    edge_offsets: list[int] = field(default_factory=list)
    q: torch.Tensor | None = None
    o: torch.Tensor | None = None

    @property
    def count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def colours(self) -> int:
        """Number of colours, 0 before colouring."""
        return max(len(self.vertex_offsets) - 1, 0)


def mean_edge_length(level: Level) -> float:
    """Mean length of a level's edges, 0 for a level without edges."""
    if level.first.numel() == 0:
        return 0.0
    return float((level.vertices[level.first] - level.vertices[level.second]).norm(dim=1).mean())


def match_pairs(count: int, src: torch.Tensor, dst: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Pair vertices by mutual best proposals.

    Args:
        count: Number of vertices.
        src: ``(M,)`` directed edge sources, every edge present both ways.
        dst: ``(M,)`` directed edge targets.
        score: ``(M,)`` symmetric preference of each edge, higher first.

    Returns:
        ``(count,)`` int64 partner of each vertex, -1 when unmatched.
    """
    device = src.device
    partner = torch.full((count,), -1, dtype=torch.int64, device=device)
    index = torch.arange(count, device=device)
    for _ in range(MATCH_ROUNDS):
        free = partner < 0
        live = free[src] & free[dst]
        if not bool(live.any()):
            break
        s, a, b = score[live], src[live], dst[live]
        best = torch.full((count,), -torch.inf, dtype=s.dtype, device=device)
        best.scatter_reduce_(0, a, s, reduce="amax")
        top = s >= best[a]
        choice = torch.full((count,), count, dtype=torch.int64, device=device)
        choice.scatter_reduce_(0, a[top], b[top], reduce="amin")
        has = choice < count
        pick = torch.where(has, choice, index)
        mutual = has & (pick[pick] == index)
        partner[mutual] = pick[mutual]
    return partner


def _pick_child(parent: torch.Tensor, priority: torch.Tensor, coarse_count: int) -> torch.Tensor:
    """The child of each coarse vertex with the highest priority, lowest index on a tie."""
    device = parent.device
    best = torch.full((coarse_count,), -torch.inf, dtype=priority.dtype, device=device)
    best.scatter_reduce_(0, parent, priority, reduce="amax")
    index = torch.arange(parent.shape[0], device=device)
    top = priority >= best[parent]
    chosen = torch.full((coarse_count,), parent.shape[0], dtype=torch.int64, device=device)
    chosen.scatter_reduce_(0, parent[top], index[top], reduce="amin")
    return chosen.clamp_max(parent.shape[0] - 1)


def _downsample(level: Level, generator: torch.Generator) -> Level | None:
    """The next coarser level, or None when matching no longer shrinks the graph."""
    count = level.count
    device = level.vertices.device
    a, b = level.first, level.second
    if a.numel() == 0:
        return None
    jitter = 1.0 + 1e-3 * torch.rand(a.shape[0], generator=generator).to(device)
    mean_area = level.areas.mean().clamp_min(1e-30)
    dot = (level.normals[a] * level.normals[b]).sum(1)
    score = (1.0 + dot) * mean_area / (level.areas[a] + level.areas[b]).clamp_min(1e-30) * jitter
    partner = match_pairs(count, torch.cat([a, b]), torch.cat([b, a]), torch.cat([score, score]))
    index = torch.arange(count, device=device)
    leader = torch.where(partner >= 0, torch.minimum(index, partner), index)
    _, parent = torch.unique(leader, return_inverse=True)
    coarse_count = int(parent.max()) + 1
    if coarse_count > MIN_SHRINK * count:
        return None
    weight = level.areas.clamp_min(1e-30)
    mass = torch.zeros(coarse_count, dtype=weight.dtype, device=device).index_add_(0, parent, weight)
    vertices = torch.zeros(coarse_count, 3, dtype=level.vertices.dtype, device=device)
    vertices.index_add_(0, parent, level.vertices * weight[:, None])
    vertices /= mass[:, None]
    normals = torch.zeros_like(vertices).index_add_(0, parent, level.normals * weight[:, None])
    normals = torch.nn.functional.normalize(normals, dim=1)
    flat = normals.norm(dim=1) < 0.5
    if bool(flat.any()):
        lead = _pick_child(parent, weight, coarse_count)
        normals[flat] = level.normals[lead[flat]]
    scales = torch.zeros(coarse_count, dtype=level.scales.dtype, device=device)
    scales.index_add_(0, parent, level.scales * weight)
    scales /= mass
    areas = torch.zeros(coarse_count, dtype=level.areas.dtype, device=device).index_add_(0, parent, level.areas)
    pa, pb = parent[a], parent[b]
    keep = pa != pb
    low = torch.minimum(pa[keep], pb[keep])
    high = torch.maximum(pa[keep], pb[keep])
    keys, inverse = torch.unique(low * coarse_count + high, return_inverse=True)
    weights = torch.zeros(keys.shape[0], dtype=level.weights.dtype, device=device)
    weights.index_add_(0, inverse, level.weights[keep])
    given = level.constraints
    orient_pick = _pick_child(parent, given.direction_weight, coarse_count)
    point_priority = given.point_weight * 2.0 + ((given.line.norm(dim=1) == 0) & (given.point_weight > 0)).to(given.point_weight.dtype)
    point_pick = _pick_child(parent, point_priority, coarse_count)
    direction = given.direction[orient_pick]
    direction = direction - normals * (direction * normals).sum(1, keepdim=True)
    constraints = Constraints(
        direction=torch.nn.functional.normalize(direction, dim=1),
        direction_weight=given.direction_weight[orient_pick],
        point=given.point[point_pick],
        line=given.line[point_pick],
        point_weight=given.point_weight[point_pick],
    )
    level.parent = parent
    return Level(
        vertices=vertices, normals=normals, areas=areas, scales=scales,
        first=keys // coarse_count, second=keys % coarse_count, weights=weights,
        constraints=constraints,
    )


def build(vertices: torch.Tensor, normals: torch.Tensor, areas: torch.Tensor, scales: torch.Tensor,
          edges: torch.Tensor, constraints: Constraints, generator: torch.Generator) -> list[Level]:
    """Build the hierarchy from the finest graph down to a handful of vertices.

    Args:
        vertices: ``(V, 3)`` positions.
        normals: ``(V, 3)`` unit normals.
        areas: ``(V,)`` vertex areas.
        scales: ``(V,)`` lattice spacings.
        edges: ``(E, 2)`` int64 undirected edges.
        constraints: Constraints on the finest level.
        generator: CPU random generator for tie breaking.

    Returns:
        Levels, finest first.
    """
    weights = torch.ones(edges.shape[0], dtype=vertices.dtype, device=vertices.device)
    levels = [Level(vertices=vertices, normals=normals, areas=areas, scales=scales,
                    first=edges[:, 0].contiguous(), second=edges[:, 1].contiguous(), weights=weights,
                    constraints=constraints)]
    while levels[-1].count > MIN_VERTICES and len(levels) < MAX_LEVELS:
        coarse = _downsample(levels[-1], generator)
        if coarse is None:
            break
        levels.append(coarse)
    return levels


def smooth_normals(levels: list[Level], finest: int, source: int, passes: int = 2) -> None:
    """Replace the normals of the levels finer than ``source`` by its normals, carried down
    and averaged with their neighbours at each level.

    Args:
        levels: The hierarchy, changed in place.
        finest: Index of the finest level changed.
        source: Index of the level whose normals are carried down.
        passes: Neighbour averaging passes per level.
    """
    for index in range(source - 1, finest - 1, -1):
        level = levels[index]
        normals = levels[index + 1].normals[level.parent]
        a, b = level.first, level.second
        for _ in range(passes):
            total = normals.clone()
            total.index_add_(0, a, normals[b] * level.weights[:, None])
            total.index_add_(0, b, normals[a] * level.weights[:, None])
            normals = torch.nn.functional.normalize(total, dim=1)
        weak = normals.norm(dim=1) < 0.5
        level.normals = torch.where(weak[:, None], level.normals, normals)


def colour(level: Level, generator: torch.Generator) -> None:
    """Colour a level's vertices so no edge joins two of one colour, and sort its edges by colour.

    Args:
        level: The level, filled in place.
        generator: CPU random generator for the priorities.
    """
    count = level.count
    device = level.vertices.device
    src = torch.cat([level.first, level.second])
    dst = torch.cat([level.second, level.first])
    weight = torch.cat([level.weights, level.weights])
    priority = torch.randperm(count, generator=generator).to(device)
    colours = torch.full((count,), -1, dtype=torch.int64, device=device)
    used = torch.zeros(count, TRACKED_COLOURS, dtype=torch.bool, device=device)
    s, d = src, dst
    for _ in range(count + 1):
        uncoloured = colours < 0
        if not bool(uncoloured.any()):
            break
        live = uncoloured[s] & uncoloured[d]
        s, d = s[live], d[live]
        top = torch.full((count,), -1, dtype=torch.int64, device=device)
        top.scatter_reduce_(0, s, priority[d], reduce="amax")
        rows = torch.nonzero(uncoloured & (priority > top)).squeeze(1)
        taken = used[rows]
        free = (~taken).long().argmax(1)
        colours[rows] = torch.where(taken.all(1), TRACKED_COLOURS, free)
        chosen = torch.zeros(count, dtype=torch.bool, device=device)
        chosen[rows] = True
        hit = chosen[d] & (colours[d] < TRACKED_COLOURS)
        used[s[hit], colours[d[hit]]] = True
    colour_count = int(colours.max()) + 1 if count else 0
    order = torch.argsort(colours * max(count, 1) + torch.arange(count, device=device))
    rank = torch.empty_like(order)
    rank[order] = torch.arange(count, device=device)
    per_colour = torch.bincount(colours, minlength=colour_count)
    vertex_offsets = [0] + torch.cumsum(per_colour, 0).tolist()
    edge_order = torch.argsort(rank[src])
    src, dst, weight = src[edge_order], dst[edge_order], weight[edge_order]
    edge_counts = torch.bincount(colours[src], minlength=colour_count)
    edge_offsets = [0] + torch.cumsum(edge_counts, 0).tolist()
    starts = torch.tensor(vertex_offsets[:-1], dtype=torch.int64, device=device)
    level.order = order
    level.src = src
    level.dst = dst
    level.src_local = rank[src] - starts[colours[src]]
    level.edge_weight = weight
    level.vertex_offsets = vertex_offsets
    level.edge_offsets = edge_offsets
