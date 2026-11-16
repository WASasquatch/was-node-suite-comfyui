"""Joining a character's head and body at the neck into one surface along a ring of shared vertices.

The seam is the level cut Character Assemble leaves. Angles are taken about the neck's axis in X and Z.
"""

from __future__ import annotations

__all__ = ["BAND", "MAX_BAND", "join_neck"]

import bisect
import copy
import math
from dataclasses import dataclass

import torch

from ..mesh import arrays, clean, topology
from ..model3d import merge, selection
from ..model3d.scene import Model3D, MorphTarget, Part, Polygons
from . import assemble as character_assemble

#: Height either side of the seam, in neck radii, over which each side eases into the shared ring.
BAND = 0.5
MAX_BAND = 3.0

#: Share of the model's height two heights may differ by and still lie on the seam.
LEVEL_SHARE = 1e-4

#: Fewest level border edges of the head a seam is read from.
MIN_RIM_EDGES = 12

#: Least cosine between a flat face on the seam and up for it to belong to the body's neck cap.
CAP_FLAT = 0.99

#: Cap pieces on the seam level tried against the head's edge, largest first.
MAX_CAPS = 16

#: Farthest the head's neck edge may lie from the body's at the seam, in neck radii.
GAP_LIMIT = 0.5

#: Farthest a vertex of the head's edge may lie from the body's ring and be joined to it, in neck radii.
RIM_GIVE = 0.05

#: Least share of an edge's length that runs along the ring, for it to be part of the head's neck edge.
ALONG = 0.5

#: Most a face on the head's edge may reach away from the ring for each unit it rises, to be part of the neck.
RISE = 2.0

#: Neck radii back along the ring past which a stretch of the head's edge folds over one it already ran.
FOLD = 0.02

#: Share of the shorter neighbouring edge within which a vertex of one ring is paired with one of the other.
SNAP_SHARE = 0.35

#: Neck radii in or out from the neck's line a vertex in the band may lie and still be eased as neck.
TUBE_GIVE = 0.25

#: Share of the seam's height tolerance within which a vertex lies on the cut itself and keeps its place.
ON_CUT = 1e-3

#: Sectors round the neck in which each side's slope is read, and the angle at the middle of each.
SECTORS = 48
SECTOR_MIDDLES = (torch.arange(SECTORS, dtype=torch.float64) + 0.5) / SECTORS * 2 * math.pi - math.pi

#: Share of the band, from the seam, below which a side's slope is not read.
SLOPE_FROM = 0.3

#: Share of the band, from the seam, over which each side's shape is drawn onto the line both necks share.
FAIR_SHARE = 0.4

#: Degrees past an end of the head's edge over which its corrections fade out.
END_FADE = 12.0

#: Degrees round the ring over which the skin weights and shape keys the two sides share are smoothed.
RING_ARC = 4.0

#: Least gap between neighbouring shared vertices, in degrees, read as an end of the head's edge.
OPEN_ARC = 10.0

#: Points per block when the nearest point of a ring is found.
BLOCK = 512


@dataclass
class _Side:
    """One side of the seam: its parts as one vertex and face table, welded where positions repeat.

    Attributes:
        indices: Model part indices.
        vertex_starts: First row of each part in the table, and the total last.
        face_starts: First face of each part, and the total last.
        faces: ``(F, 3)`` rows of the table.
        welded: ``(W, 3)`` float64 welded positions.
        wfaces: ``(F, 3)`` welded vertex of each corner.
        remap: ``(N,)`` welded vertex of each row.
    """

    indices: list[int]
    vertex_starts: list[int]
    face_starts: list[int]
    faces: torch.Tensor
    welded: torch.Tensor
    wfaces: torch.Tensor
    remap: torch.Tensor


def _chosen(model: Model3D, text: str, input_name: str) -> list[int]:
    """Indices of the parts with triangles a selection names."""
    text = str(text or "").strip()
    if not text:
        raise ValueError(f"{input_name} is empty; name the parts by label or name, such as head")
    found = [index for index in selection.selected(model, text) if model.parts[index].triangle_count]
    if not found:
        names = ", ".join(f"{part.name or '(unnamed)'} [{part.label or 'no label'}]" for part in model.parts[:12])
        raise ValueError(f"{input_name} '{text[:60]}' matches none of the model's parts ({names}); use a label "
                         "or name from that list, or label the parts with Model 3D Parts")
    return found


def _side(model: Model3D, indices: list[int]) -> _Side:
    """The parts of one side as a welded table."""
    parts = [model.parts[index] for index in indices]
    vertex_starts, face_starts = [0], [0]
    for part in parts:
        vertex_starts.append(vertex_starts[-1] + part.vertex_count)
        face_starts.append(face_starts[-1] + part.triangle_count)
    vertices = torch.cat([part.vertices.float() for part in parts])
    faces = torch.cat([part.faces.long() + start for part, start in zip(parts, vertex_starts)])
    welded, wfaces, remap = topology.weld(vertices, faces, tolerance=0)
    return _Side(indices, vertex_starts, face_starts, faces, welded.double(), wfaces, remap)


def _head_rim(side: _Side, tolerance: float) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head's cut neck edge: ``(level, start, end, third)``, its level border edges and each face's third corner.

    Raises:
        ValueError: The head has no level border at all.
    """
    count = int(side.welded.shape[0])
    found = topology.edge_topology(side.wfaces, count)
    ids = torch.nonzero(found.boundary).squeeze(1)
    face, corner = found.edge_faces[ids, 0], found.edge_corners[ids, 0]
    start, end = side.wfaces[face, corner], side.wfaces[face, (corner + 1) % 3]
    ys, ye = side.welded[start, 1], side.welded[end, 1]
    flat = (ys - ye).abs() <= tolerance
    if int(flat.sum()) < MIN_RIM_EDGES:
        raise ValueError("the head has no cut neck edge to join; wire a model from Character Assemble whose head "
                         "was fitted and joined (head_fit other than keep)")
    ordered = ((ys + ye) / 2)[flat].sort().values
    low = torch.searchsorted(ordered, ordered - tolerance)
    high = torch.searchsorted(ordered, ordered + tolerance, right=True)
    best = int((high - low).argmax())
    if int(high[best] - low[best]) < MIN_RIM_EDGES:
        raise ValueError("the head's neck edge is not level, so it is not the cut Character Assemble makes; wire "
                         "a model from Character Assemble whose head was fitted and joined")
    level = float(ordered[int(low[best]):int(high[best])].median())
    on = flat & ((ys - level).abs() <= tolerance) & ((ye - level).abs() <= tolerance)
    third = side.wfaces[face, (corner + 2) % 3]
    return level, start[on], end[on], third[on]


def _body_rings(side: _Side, level: float, tolerance: float) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Closed loops of the body on the seam level: ``(loop, cap faces)``, no cap faces for an open loop."""
    count = int(side.welded.shape[0])
    near = (side.welded[:, 1] - level).abs() <= tolerance
    touching = torch.nonzero(near[side.wfaces].any(dim=1)).squeeze(1)
    if not touching.numel():
        return []
    sub = side.wfaces[touching]
    found = []
    flat = near[sub].all(dim=1) & (topology.face_normals(side.welded, sub)[:, 1].abs() >= CAP_FLAT)
    if bool(flat.any()):
        caps = touching[flat]
        cap_faces = side.wfaces[caps]
        pieces = topology.face_components(cap_faces, count)
        for piece in torch.bincount(pieces).argsort(descending=True)[:MAX_CAPS].tolist():
            chosen = pieces == piece
            chains, closed = topology.boundary_chains(cap_faces[chosen], count)
            loops = [chain for chain, is_closed in zip(chains, closed.tolist()) if is_closed and chain.numel() >= 3]
            if loops:
                found.append((loops[0], caps[chosen]))
    chains, closed = topology.boundary_chains(sub, count)
    for chain, is_closed in zip(chains, closed.tolist()):
        if is_closed and chain.numel() >= 3 and bool(near[chain].all()):
            found.append((chain, touching[:0]))
    return found


def _polygon(points: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Area centre of a closed polygon in X, Z and the radius of a circle of its area."""
    a = points.double()
    b = a.roll(-1, dims=0)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area = float(cross.sum()) / 2
    if abs(area) < 1e-18:
        centre = a.mean(dim=0)
        return centre, float((a - centre).norm(dim=1).mean())
    return ((a + b) * cross[:, None]).sum(dim=0) / (6 * area), math.sqrt(abs(area) / math.pi)


def _chains(start: torch.Tensor, end: torch.Tensor) -> list[tuple[list[int], bool]]:
    """Directed border edges walked into chains of welded vertices: ``(vertices, closed)``."""
    after: dict[int, list[int]] = {}
    arriving: dict[int, int] = {}
    for a, b in zip(start.tolist(), end.tolist()):
        after.setdefault(a, []).append(b)
        arriving[b] = arriving.get(b, 0) + 1
    used: set[tuple[int, int]] = set()
    chains = []
    for first in [vertex for vertex in after if not arriving.get(vertex)] + list(after):
        while any((first, ahead) not in used for ahead in after[first]):
            chain, vertex, closed = [first], first, False
            while True:
                ahead = next((b for b in after.get(vertex, []) if (vertex, b) not in used), None)
                if ahead is None:
                    break
                used.add((vertex, ahead))
                if ahead == first:
                    closed = True
                    break
                chain.append(ahead)
                vertex = ahead
            chains.append((chain, closed))
    return chains


def _nearest_on_ring(points: torch.Tensor, ring: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The nearest point of a closed ring, in X and Z, to each point: ``(segment, share, distance)``."""
    a = ring[:, [0, 2]].double()
    e = a.roll(-1, dims=0) - a
    length = (e * e).sum(dim=1).clamp_min(1e-30)
    flat = points[:, [0, 2]].double()
    count = int(flat.shape[0])
    segment = torch.zeros(count, dtype=torch.long)
    share = torch.zeros(count, dtype=torch.float64)
    distance = torch.zeros(count, dtype=torch.float64)
    for begin in range(0, count, BLOCK):
        chunk = flat[begin:begin + BLOCK]
        along = (((chunk[:, None] - a[None]) * e[None]).sum(dim=2) / length[None]).clamp(0.0, 1.0)
        apart = (chunk[:, None] - (a[None] + along[..., None] * e[None])).norm(dim=2)
        best = apart.argmin(dim=1)
        rows = torch.arange(best.shape[0])
        segment[begin:begin + BLOCK] = best
        share[begin:begin + BLOCK] = along[rows, best]
        distance[begin:begin + BLOCK] = apart[rows, best]
    return segment, share, distance


def _on_vertex(points: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """``(N,)`` bool: points standing exactly on one of the ring's vertices."""
    found = torch.zeros(points.shape[0], dtype=torch.bool)
    for begin in range(0, int(points.shape[0]), BLOCK):
        apart = (points[begin:begin + BLOCK, None, :].double() - ring[None].double()).abs().amax(dim=2)
        found[begin:begin + BLOCK] = (apart == 0).any(dim=1)
    return found


def _match(head: _Side, rim_start: torch.Tensor, rim_end: torch.Tensor, rim_third: torch.Tensor, body: _Side,
           loop: torch.Tensor, level: float, radius: float) -> dict:
    """Pair the head's neck edge with the body's ring by distance along the ring.

    Args:
        head: The head side.
        rim_start: Welded start of each level border edge of the head.
        rim_end: Welded end of each.
        rim_third: Welded third corner of the face on each.
        body: The body side.
        loop: The body's ring, welded vertices in order.
        level: Height of the seam.
        radius: The neck's radius.

    Returns:
        ``head_at`` and ``body_at``, ``(S, 3)`` where each side holds each shared vertex; ``head_moves`` and
        ``body_moves``, the shared vertex of each welded vertex or -1; ``head_inserts`` and ``body_inserts``,
        vertices added along edges; ``full``, the ring joined all round; ``turned``, radians joined; ``rejoined``,
        the edge already shared its vertices with the ring.

    Raises:
        ValueError: The head's edge does not lie along the ring.
    """
    ring = body.welded[loop]
    lengths = (ring.roll(-1, dims=0) - ring).norm(dim=1).clamp_min(1e-12)
    total = float(lengths.sum())
    starts = torch.cumsum(lengths, 0) - lengths
    shortest = torch.minimum(lengths, lengths.roll(1))
    count = int(loop.shape[0])
    a, b, c = head.welded[rim_start], head.welded[rim_end], head.welded[rim_third]
    give = RIM_GIVE * radius
    found = [_nearest_on_ring(point, ring) for point in (a, b, (a + b) / 2, c)]
    near = (found[0][2] <= give) & (found[1][2] <= give) & (found[2][2] <= give)
    near &= found[3][2] <= torch.clamp(RISE * (c[:, 1] - level), min=give)
    run = [starts[segment] + share * lengths[segment] for segment, share, _ in found[:2]]
    along = ((run[1] - run[0] + total / 2) % total - total / 2).abs()
    near &= along >= ALONG * (b - a).norm(dim=1)
    shared = _on_vertex(a, ring) & _on_vertex(b, ring)
    rejoined = int((near & shared).sum()) >= MIN_RIM_EDGES
    if rejoined:
        near &= shared
    if int(near.sum()) < MIN_RIM_EDGES:
        raise ValueError(f"the head's neck edge does not lie on the body's neck, within {give * 1000:.1f} mm; "
                         "run Character Assemble again so the head is joined, then Character Neck Join")
    rim_start, rim_end = rim_start[near], rim_end[near]

    head_at, body_at = [], []
    head_moves = torch.full((head.welded.shape[0],), -1, dtype=torch.long)
    body_moves = torch.full((body.welded.shape[0],), -1, dtype=torch.long)
    head_inserts, body_inserts = [], []
    ring_station = torch.full((count,), -1, dtype=torch.long)
    covered = 0.0
    pieces = []
    for chain, closed in _chains(rim_start, rim_end):
        if len(chain) < 2:
            continue
        vertices = torch.tensor(chain, dtype=torch.long)
        segment, share, _ = _nearest_on_ring(head.welded[vertices], ring)
        param = starts[segment] + share * lengths[segment]
        steps = (param.diff() + total / 2) % total - total / 2
        turn = float(steps.sum())
        if closed:
            turn += float((param[0] - param[-1] + total / 2) % total - total / 2)
        if turn < 0:
            vertices, param = vertices.flip(0), param.flip(0)
            steps = (param.diff() + total / 2) % total - total / 2
        param = torch.cat([param[:1], param[0] + torch.cumsum(steps, 0)])
        folded = ((torch.cummax(param, 0).values - param > FOLD * radius)
                  & (param - torch.cummin(param.flip(0), 0).values.flip(0) > FOLD * radius))
        if not bool(folded.any()):
            pieces.append((vertices, param, closed))
            continue
        kept = torch.nonzero(~folded).squeeze(1)
        for run in torch.tensor_split(kept, (torch.nonzero(kept.diff() > 1).squeeze(1) + 1).tolist()):
            if run.numel() >= 2:
                pieces.append((vertices[run], param[run], False))
    for vertices, param, closed in pieces:
        points = head.welded[vertices]
        param = torch.cummax(param, 0).values
        low = float(param[0])
        if closed:
            param = param.clamp_max(low + total * (1 - 1e-9))
        high = low + total if closed else float(param[-1])
        covered += high - low

        first = torch.cat([torch.tensor([True]), param.diff() > 1e-9 * total])
        group_of = torch.cumsum(first.long(), 0) - 1
        groups = int(group_of[-1]) + 1
        group_param = param[first]
        group_first = torch.nonzero(first).squeeze(1)
        group_last = torch.cat([group_first[1:] - 1, torch.tensor([len(vertices) - 1])])
        group_at = torch.zeros(groups, 3, dtype=torch.float64).index_add_(0, group_of, points)
        group_at /= torch.bincount(group_of, minlength=groups).double()[:, None]
        edge = (points.roll(-1, dims=0) - points).norm(dim=1)
        before = torch.where(group_first > 0, edge[(group_first - 1).clamp_min(0)],
                             edge[-1] if closed else torch.full_like(group_param, math.inf))
        after = torch.where(group_last < len(vertices) - 1, edge[group_last],
                            edge[-1] if closed else torch.full_like(group_param, math.inf))
        group_edge = torch.minimum(before, after)

        lifted = starts + torch.ceil((low - starts) / total) * total
        inside = (lifted < high) if closed else (lifted <= high)
        ring_ids = torch.nonzero(inside & (ring_station < 0)).squeeze(1).tolist()
        candidates = []
        for j in ring_ids:
            where = int(torch.searchsorted(group_param, lifted[j]))
            for g in (where - 1, where):
                wrapped = g
                offset = 0.0
                if g >= groups and closed:
                    wrapped, offset = 0, total
                if not 0 <= wrapped < groups:
                    continue
                apart = abs(float(group_param[wrapped]) + offset - float(lifted[j]))
                if apart <= SNAP_SHARE * min(float(shortest[j]), float(group_edge[wrapped])):
                    candidates.append((apart, wrapped, j))
        group_ring: dict[int, int] = {}
        taken: set[int] = set()
        for _, g, j in sorted(candidates):
            if g not in group_ring and j not in taken:
                group_ring[g] = j
                taken.add(j)

        events = [(float(group_param[g]), 0, g) for g in range(groups)]
        events += [(float(lifted[j]), 1, j) for j in ring_ids if j not in taken]
        for value, kind, which in sorted(events):
            station = len(head_at)
            if kind == 0:
                head_moves[vertices[group_first[which]:group_last[which] + 1]] = station
                head_at.append(group_at[which])
                j = group_ring.get(which)
                if j is not None:
                    body_at.append(ring[j])
                    body_moves[loop[j]] = station
                    ring_station[j] = station
                    continue
                along = value % total
                i = min(max(int(torch.searchsorted(starts, torch.tensor(along, dtype=torch.float64), right=True)) - 1,
                            0), count - 1)
                part_share = min(max((along - float(starts[i])) / float(lengths[i]), 0.0), 1.0)
                body_at.append(ring[i] + (ring[(i + 1) % count] - ring[i]) * part_share)
                body_inserts.append((int(loop[i]), int(loop[(i + 1) % count]), part_share, station))
                continue
            k = int(torch.searchsorted(group_param, torch.tensor(value, dtype=torch.float64), right=True)) - 1
            if k < 0:
                continue
            following, offset = k + 1, 0.0
            if following >= groups:
                if not closed:
                    continue
                following, offset = 0, total
            u, v = int(vertices[group_last[k]]), int(vertices[group_first[following]])
            reach = float(group_param[following]) + offset - float(group_param[k])
            part_share = min(max((value - float(group_param[k])) / max(reach, 1e-18), 0.0), 1.0)
            head_at.append(head.welded[u] + (head.welded[v] - head.welded[u]) * part_share)
            body_at.append(ring[which])
            body_moves[loop[which]] = station
            ring_station[which] = station
            head_inserts.append((u, v, part_share, station))
    if not head_at:
        raise ValueError("the head's neck edge and the body's neck share no stretch to join")
    return {"head_at": torch.stack(head_at), "body_at": torch.stack(body_at), "head_moves": head_moves,
            "body_moves": body_moves, "head_inserts": head_inserts, "body_inserts": body_inserts,
            "full": bool((ring_station >= 0).all()), "turned": min(covered / total, 1.0) * 2 * math.pi,
            "rejoined": rejoined}


def _angle(points: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """Angle of each point about the neck's axis."""
    offset = points[:, [0, 2]].double() - centre
    return torch.atan2(offset[:, 1], offset[:, 0])


def _field(angles: torch.Tensor, values: torch.Tensor, query: torch.Tensor, fade: bool = True) -> torch.Tensor:
    """Values held at the shared vertices, read at other angles; past an end of the edge they fade out.

    Args:
        angles: ``(S,)`` angles of the shared vertices.
        values: ``(S, K)`` values at them.
        query: ``(Q,)`` angles to read at.
        fade: Fade past an open end; False reads the nearer end's value.

    Returns:
        ``(Q, K)`` values.
    """
    order = angles.argsort()
    held, known = angles[order].double(), values[order].double()
    count = int(held.shape[0])
    if count == 1:
        distance = (query.double() - held[0] + math.pi) % (2 * math.pi) - math.pi
        weight = (1 - distance.abs() / math.radians(END_FADE)).clamp(0.0, 1.0) if fade else torch.ones_like(distance)
        return known[0][None] * weight[:, None]
    gaps = ((held.roll(-1) - held) % (2 * math.pi)).clamp_min(1e-12)
    spacing = float(gaps.median())
    open_gap = gaps > max(5 * spacing, math.radians(OPEN_ARC))
    position = torch.searchsorted(held, query.double().contiguous(), right=True) - 1
    before = position % count
    after = (before + 1) % count
    into = (query.double() - held[before]) % (2 * math.pi)
    span = gaps[before].clamp_min(1e-12)
    share = (into / span).clamp(0.0, 1.0)
    inside = known[before] * (1 - share[:, None]) + known[after] * share[:, None]
    to_after = (span - into).clamp_min(0.0)
    nearer_before = into <= to_after
    nearest = torch.where(nearer_before[:, None], known[before], known[after])
    distance = torch.minimum(into, to_after)
    weight = (1 - distance / math.radians(END_FADE)).clamp(0.0, 1.0) if fade else torch.ones_like(distance)
    return torch.where(open_gap[before][:, None], nearest * weight[:, None], inside)


def _smooth(share: torch.Tensor) -> torch.Tensor:
    """Smoothstep of a share clamped to ``[0, 1]``."""
    share = share.clamp(0.0, 1.0)
    return share * share * (3 - 2 * share)


def _edge_faces(side: _Side, level: float, tolerance: float) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Faces round each welded edge lying on the seam level: ``(low, high)`` against ``(face, side)`` pairs."""
    near = (side.welded[:, 1] - level).abs() <= tolerance
    touching = torch.nonzero(near[side.wfaces].any(dim=1)).squeeze(1)
    table: dict[tuple[int, int], list[tuple[int, int]]] = {}
    corners = side.wfaces[touching].tolist()
    for face, (a, b, c) in zip(touching.tolist(), corners):
        for k, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            if bool(near[u]) and bool(near[v]):
                table.setdefault((min(u, v), max(u, v)), []).append((face, k))
    return table


def _split(side: _Side, model: Model3D, inserts: list[tuple[int, int, float, int]], targets: torch.Tensor,
           level: float, tolerance: float) -> dict[int, dict]:
    """Add the shared vertices that fall along a side's edges, splitting the faces on those edges.

    Args:
        side: The side.
        model: The model holding its parts.
        inserts: ``(welded a, welded b, share from a, station)`` per vertex added.
        targets: ``(S, 3)`` float64 position of each shared vertex.
        level: Height of the seam.
        tolerance: Height difference still on the seam.

    Returns:
        Per side part position: ``rows`` (station of each added row), ``sources``, ``shares``, ``faces`` (the
        rebuilt face table), ``origin`` (the face each appended face came from) and ``splits`` (polygon edits).
    """
    table = _edge_faces(side, level, tolerance)
    on_face: dict[int, list[tuple[int, bool, float, int]]] = {}
    for a, b, share, station in inserts:
        for face, k in table.get((min(a, b), max(a, b)), []):
            on_face.setdefault(face, []).append((k, int(side.wfaces[face, k]) == a, share, station))
    edits: dict[int, dict] = {}
    made: dict[tuple, int] = {}
    for face in sorted(on_face):
        position = bisect.bisect_right(side.face_starts, face) - 1
        part = model.parts[side.indices[position]]
        edit = edits.setdefault(position, {"rows": [], "sources": [], "shares": [], "replaced": {}, "splits": []})
        local = face - side.face_starts[position]
        rows = [int(row) - side.vertex_starts[position] for row in side.faces[face].tolist()]
        triangles = [tuple(rows)]
        by_side: dict[int, list[tuple[float, int]]] = {}
        for k, forward, share, station in on_face[face]:
            by_side.setdefault(k, []).append((share if forward else 1.0 - share, station))
        for k in sorted(by_side):
            u, v = rows[k], rows[(k + 1) % 3]
            added = []
            for share, station in sorted(by_side[k]):
                key = (position, min(u, v), max(u, v), station)
                if key not in made:
                    made[key] = part.vertex_count + len(edit["rows"])
                    edit["rows"].append(station)
                    edit["sources"].append((u, v))
                    edit["shares"].append(share)
                added.append(made[key])
            for index, triangle in enumerate(triangles):
                turned = next(((triangle[j], triangle[(j + 1) % 3], triangle[(j + 2) % 3]) for j in range(3)
                               if triangle[j] == u and triangle[(j + 1) % 3] == v), None)
                if turned is None:
                    continue
                chain = [u, *added, v]
                fan = [(chain[i], chain[i + 1], turned[2]) for i in range(len(chain) - 1)]
                triangles[index:index + 1] = fan
                break
            edit["splits"].append((local, u, v, added))
        edit["replaced"][local] = triangles
    for position, edit in edits.items():
        part = model.parts[side.indices[position]]
        faces = part.faces.long().clone()
        appended, origin = [], []
        for local, triangles in edit["replaced"].items():
            faces[local] = torch.tensor(triangles[0], dtype=torch.long)
            appended += triangles[1:]
            origin += [local] * (len(triangles) - 1)
        if appended:
            faces = torch.cat([faces, torch.tensor(appended, dtype=torch.long)])
        edit["faces"] = faces
        edit["origin"] = torch.tensor(origin, dtype=torch.long)
    return edits


def _split_polygons(part: Part, splits: list, origin: torch.Tensor) -> Polygons | None:
    """A part's polygons with the added vertices set into the edges they split, or None when they cannot follow."""
    source = part.polygons
    if source is None:
        return None
    sizes = source.sizes.long().tolist()
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + size)
    flat = source.indices.long().tolist()
    corners = {}
    for local, u, v, added in splits:
        polygon = int(source.triangle_polygon[local])
        ring = corners.get(polygon) or flat[starts[polygon]:starts[polygon + 1]]
        for j in range(len(ring)):
            if ring[j] == u and ring[(j + 1) % len(ring)] == v:
                ring = ring[:j + 1] + list(added) + ring[j + 1:]
                break
            if ring[j] == v and ring[(j + 1) % len(ring)] == u:
                ring = ring[:j + 1] + list(reversed(added)) + ring[j + 1:]
                break
        else:
            return None
        corners[polygon] = ring
    new_sizes, new_indices = [], []
    for polygon, size in enumerate(sizes):
        ring = corners.get(polygon, flat[starts[polygon]:starts[polygon + 1]])
        new_sizes.append(len(ring))
        new_indices += ring
    triangle_polygon = torch.cat([source.triangle_polygon.long(), source.triangle_polygon.long()[origin]])
    return Polygons(torch.tensor(new_sizes, dtype=torch.long), torch.tensor(new_indices, dtype=torch.long),
                    triangle_polygon)


def _rebuild(part: Part, edit: dict, targets: torch.Tensor) -> Part:
    """A part with the added rows, their attributes blended from the edge they split, and the new faces."""
    rebuilt = copy.copy(part)
    rebuilt.morphs = dict(part.morphs)
    count = len(edit["rows"])
    if count:
        sources = torch.tensor(edit["sources"], dtype=torch.long)
        shares = torch.tensor(edit["shares"], dtype=torch.float64)
        coefficients = torch.stack([1 - shares, shares], dim=1)
        blended = clean.blend(part.vertex_attributes(), sources, coefficients)
        for name, value in blended.items():
            setattr(rebuilt, name, torch.cat([getattr(part, name), value.to(getattr(part, name).dtype)]))
        stations = torch.tensor(edit["rows"], dtype=torch.long)
        rebuilt.vertices = torch.cat([part.vertices, targets[stations].to(part.vertices.dtype)])
        for key, morph in part.morphs.items():
            mixed = (morph.positions[sources].float() * coefficients.float()[:, :, None]).sum(dim=1)
            normals = None
            if morph.normals is not None:
                normals = torch.cat([morph.normals, (morph.normals[sources].float()
                                                     * coefficients.float()[:, :, None]).sum(dim=1)])
            rebuilt.morphs[key] = MorphTarget(torch.cat([morph.positions, mixed]), normals)
    rebuilt.faces = edit["faces"]
    if part.polygons is not None:
        rebuilt.polygons = _split_polygons(part, edit["splits"], edit["origin"])
    return rebuilt


def _vertex_normals(part: Part) -> torch.Tensor:
    """A part's normals from its faces, shared across rows that repeat a position."""
    welded, faces, remap = topology.weld(part.vertices, part.faces, tolerance=0)
    return topology.vertex_normals(welded, faces)[remap].float()


def _station_sums(values: torch.Tensor, stations: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum and count of rows per shared vertex: ``(count, K)`` and ``(count,)``."""
    chosen = stations >= 0
    total = torch.zeros(count, values.shape[1], dtype=torch.float64)
    seen = torch.zeros(count, dtype=torch.float64)
    total.index_add_(0, stations[chosen], values[chosen].double())
    seen.index_add_(0, stations[chosen], torch.ones(int(chosen.sum()), dtype=torch.float64))
    return total, seen


def _side_normals(model: Model3D, edited: dict, stations: int) -> dict[str, torch.Tensor]:
    """Each side's mean unit normal at every shared vertex, read from rows that draw the surface."""
    sums: dict[str, list] = {}
    for index, entry in edited.items():
        normals = torch.nn.functional.normalize(model.parts[index].normals.double(), dim=1)
        total, seen = _station_sums(normals, torch.where(entry["surface"], entry["stations"], -1), stations)
        held = sums.setdefault(entry["role"], [0, 0])
        held[0], held[1] = held[0] + total, held[1] + seen
    return {role: torch.nn.functional.normalize(total / seen.clamp_min(1)[:, None], dim=1)
            for role, (total, seen) in sums.items()}


def _shared_normals(model: Model3D, edited: dict) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Normals from the faces of both sides round the band, rows at one position sharing one normal.

    Returns:
        Per part index: ``(rows, normals)`` for the rows of the faces round the band and the shared ring.
    """
    pieces = []
    for index, entry in edited.items():
        part = model.parts[index]
        faces = part.faces.long()
        wanted = (entry["ease"] > 0) | (entry["stations"] >= 0)
        pieces.append((index, faces[~entry["caps"] & ~entry["collapsed"] & wanted[faces].any(dim=1)]))
    rows = [torch.unique(faces) for _, faces in pieces]
    positions = torch.cat([model.parts[index].vertices[found] for (index, _), found in zip(pieces, rows)])
    if not positions.shape[0]:
        return {index: (torch.zeros(0, dtype=torch.long), torch.zeros(0, 3, dtype=torch.float64)) for index in edited}
    _, ids = torch.unique(positions, dim=0, return_inverse=True)
    total = torch.zeros(int(ids.max()) + 1, 3, dtype=torch.float64)
    starts = [0]
    for found in rows:
        starts.append(starts[-1] + int(found.shape[0]))
    for (index, faces), found, start in zip(pieces, rows, starts):
        if not faces.shape[0]:
            continue
        part = model.parts[index]
        local = torch.full((part.vertex_count,), -1, dtype=torch.long)
        local[found] = ids[start:start + found.shape[0]]
        normal = topology.face_normals(part.vertices.double(), faces)
        weight = topology.corner_angles(part.vertices.double(), faces)
        total.index_add_(0, local[faces].reshape(-1), (normal[:, None, :] * weight[..., None]).reshape(-1, 3))
    length = total.norm(dim=1, keepdim=True)
    normals = torch.where(length > 1e-30, total / length.clamp_min(1e-30), torch.full_like(total, math.nan))
    return {index: (found, normals[ids[start:start + found.shape[0]]])
            for (index, _), found, start in zip(pieces, rows, starts)}


def _along_ring(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Values at the shared vertices averaged with their neighbours round the ring, over `RING_ARC` degrees."""
    spread = math.radians(RING_ARC)
    result = torch.empty_like(values, dtype=torch.float64)
    for begin in range(0, int(angles.shape[0]), BLOCK):
        apart = (angles[begin:begin + BLOCK, None] - angles[None, :] + math.pi) % (2 * math.pi) - math.pi
        weight = torch.exp(-0.5 * (apart / spread) ** 2)
        result[begin:begin + BLOCK] = (weight @ values.double()) / weight.sum(dim=1, keepdim=True)
    return result


def _toward_ring(values: torch.Tensor, entry: dict, held: torch.Tensor, angles: torch.Tensor, centre: torch.Tensor,
                 positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A part's per-row values eased toward the ring's at each row's angle; rows on the ring take them outright.

    Args:
        values: ``(N, K)`` float64 values of the part's rows, changed in place.
        entry: The part's seam rows and ease.
        held: ``(S, K)`` values at the shared vertices.
        angles: ``(S,)`` angles of the shared vertices.
        centre: The neck's axis in X, Z.
        positions: ``(N, 3)`` the part's vertices.

    Returns:
        ``(values, changed)``: the values and the rows that changed.
    """
    rows = entry["stations"]
    chosen = rows >= 0
    moving = (entry["ease"] > 0) & ~chosen
    if bool(moving.any()):
        share = entry["ease"][moving, None]
        ring = _field(angles, held, _angle(positions[moving], centre), fade=False)
        values[moving] = values[moving] * (1 - share) + ring * share
    values[chosen] = held[rows[chosen]]
    return values, chosen | moving


def _reached(part: Part, candidate: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """``(N,)`` bool: candidate rows joined to a seed row through faces whose corners are all candidates."""
    faces = part.faces.long()
    inside = candidate[faces].all(dim=1)
    result = torch.zeros_like(candidate)
    if not bool(inside.any()) or not bool(seeds.any()):
        return result
    chosen = faces[inside]
    rows = torch.unique(chosen)
    _, welded = torch.unique(part.vertices[rows], dim=0, return_inverse=True)
    local = torch.full((part.vertex_count,), -1, dtype=torch.long)
    local[rows] = welded
    labels = topology.connected_components(local[chosen], int(welded.max()) + 1)
    seeded = torch.zeros(int(labels.max()) + 1, dtype=torch.bool)
    seeded[labels[local[rows[seeds[rows]]]]] = True
    result[rows] = seeded[labels[local[rows]]]
    return result


def _sector_fit(heights: torch.Tensor, offsets: torch.Tensor,
                angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """How fast a side's neck widens with height in each sector round it, from a straight line fitted in each.

    Returns:
        ``(slope, known)``, each ``(SECTORS,)``; sectors read from too few vertices are filled from their neighbours.
    """
    sector = ((angles + math.pi) / (2 * math.pi) * SECTORS).long().clamp(0, SECTORS - 1)

    def summed(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(SECTORS, dtype=torch.float64).index_add_(0, sector, values.double())

    count = summed(torch.ones_like(heights))
    sx, so = summed(heights), summed(offsets)
    sxx, sxo = summed(heights * heights), summed(heights * offsets)
    spread = count * sxx - sx * sx
    known = (count >= 8) & (spread > 1e-18)
    slope = torch.where(known, (count * sxo - sx * so) / spread.clamp_min(1e-18), torch.zeros_like(spread))
    return _filled(slope, known), known


def _filled(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """``(SECTORS,)`` values with the unknown sectors read between the known ones round the circle, 0 with none."""
    if not bool(known.any()):
        return torch.zeros(SECTORS, dtype=torch.float64)
    return _field(SECTOR_MIDDLES[known], values[known][:, None], SECTOR_MIDDLES, fade=False)[:, 0]


def join_neck(model: Model3D, head_parts: str = "head", body_parts: str = "body", band: float = BAND,
              merge_parts: bool = False) -> tuple[Model3D, dict]:
    """Join the head's neck edge and the body's neck ring into one ring of shared vertices.

    Args:
        model: The character, changed in place.
        head_parts: Parts above the seam, by label or name, wildcards allowed.
        body_parts: Parts below the seam, the same way.
        band: Height either side of the seam, in neck radii, over which each side eases into the ring.
        merge_parts: Join head and body parts drawn with one material into one part.

    Returns:
        ``(model, figures)``.

    Raises:
        ValueError: No parts match, the head has no cut neck edge, or the body has no neck ring beside it.
    """
    heads = _chosen(model, head_parts, "head_parts")
    bodies = _chosen(model, body_parts, "body_parts")
    shared = sorted(set(heads) & set(bodies))
    if shared:
        raise ValueError(f"head_parts and body_parts both name the part {model.parts[shared[0]].name!r}; "
                         "each part belongs to one side of the neck")
    band = min(max(float(band), 0.0), MAX_BAND)
    tolerance = LEVEL_SHARE * max(model.height(), 1e-6)
    notes: list[str] = []

    head = _side(model, heads)
    level, rim_start, rim_end, rim_third = _head_rim(head, tolerance)
    body = _side(model, bodies)
    candidates = _body_rings(body, level, tolerance)
    if not candidates:
        raise ValueError(f"the body has no neck edge at the head's seam, {level:.3f} m up; wire the model "
                         "Character Assemble made, before any node that moves, reshapes or reduces the head or the "
                         "body")
    rim_points = head.welded[torch.unique(torch.cat([rim_start, rim_end]))][:, [0, 2]]
    best = None
    for loop, caps in candidates:
        points = body.welded[loop][:, [0, 2]]
        _, radius = _polygon(points)
        distance = torch.cdist(rim_points, points).min(dim=1).values
        score = int((distance <= GAP_LIMIT * radius).sum())
        if best is None or score > best[0]:
            best = (score, loop, caps)
    score, loop, caps = best
    if score < MIN_RIM_EDGES:
        raise ValueError(f"the head's neck edge at {level:.3f} m does not meet the body's neck; wire the model "
                         "Character Assemble made, before any node that moves, reshapes or reduces the head or the "
                         "body")
    centre, radius = _polygon(body.welded[loop][:, [0, 2]])
    matched = _match(head, rim_start, rim_end, rim_third, body, loop, level, radius)
    head_at, body_at = matched["head_at"], matched["body_at"]
    targets = ((head_at + body_at) / 2).float().double()
    stations = int(targets.shape[0])
    angles = _angle(targets, centre)
    head_moves, body_moves = matched["head_moves"], matched["body_moves"]
    head_inserts, body_inserts = matched["head_inserts"], matched["body_inserts"]
    full_ring, turned = matched["full"], matched["turned"]
    ring_points = body.welded[loop]

    edited: dict[int, dict] = {}
    for side, moves, inserts, role in ((head, head_moves, head_inserts, "head"),
                                       (body, body_moves, body_inserts, "body")):
        edits = _split(side, model, inserts, targets, level, tolerance)
        cap_rows = torch.zeros(side.faces.shape[0], dtype=torch.bool)
        if role == "body" and caps.numel():
            cap_rows[caps] = True
        for position, index in enumerate(side.indices):
            part = model.parts[index]
            edit = edits.get(position)
            rows = moves[side.remap[side.vertex_starts[position]:side.vertex_starts[position + 1]]]
            caps_here = cap_rows[side.face_starts[position]:side.face_starts[position + 1]]
            if part.normals is None:
                part.normals = _vertex_normals(part)
            if edit is not None:
                part = _rebuild(part, edit, targets)
                rows = torch.cat([rows, torch.tensor(edit["rows"], dtype=torch.long)])
                caps_here = torch.cat([caps_here, caps_here[edit["origin"]]])
            model.parts[index] = part
            corner = rows[part.faces.long()]
            collapsed = (((corner[:, 0] == corner[:, 1]) & (corner[:, 0] >= 0))
                         | ((corner[:, 1] == corner[:, 2]) & (corner[:, 1] >= 0))
                         | ((corner[:, 2] == corner[:, 0]) & (corner[:, 2] >= 0)))
            surface = torch.zeros(part.vertex_count, dtype=torch.bool)
            surface[part.faces.long()[~caps_here & ~collapsed].reshape(-1)] = True
            edited[index] = {"role": role, "stations": rows, "caps": caps_here, "collapsed": collapsed,
                             "surface": surface, "added": len(edit["rows"]) if edit is not None else 0}

    first_normals = _side_normals(model, edited, stations)
    band_height = band * radius
    ring_reach = (ring_points[:, [0, 2]] - centre).norm(dim=1)
    ring_angles = _angle(ring_points, centre)
    covered = torch.ones(stations, 1, dtype=torch.float64)
    for index, entry in edited.items():
        part, rows = model.parts[index], entry["stations"]
        points = part.vertices.double()
        chosen = rows >= 0
        points[chosen] = targets[rows[chosen]]
        part.vertices = points.to(part.vertices.dtype)
        entry["ease"] = torch.zeros(part.vertex_count, dtype=torch.float64)
        if band_height <= 0:
            continue
        height = points[:, 1] - level
        up = height if entry["role"] == "head" else -height
        turn = _angle(points, centre)
        ring_at = _field(ring_angles, ring_reach[:, None], turn, fade=False)[:, 0]
        reach = (points[:, [0, 2]] - centre).norm(dim=1)
        cover = _field(angles, covered, turn)[:, 0]
        seeds = chosen & entry["surface"]
        candidate = (entry["surface"] & (up >= 0) & (up <= band_height) & (cover > 0)
                     & ((reach - ring_at).abs() <= TUBE_GIVE * radius)) | seeds
        reached = _reached(part, candidate, seeds) & ~chosen
        ease = ((1 - _smooth(up / band_height)) * cover * reached
                * _smooth(1 - (reach - ring_at).abs() / (TUBE_GIVE * radius)))
        on_seam = (entry["surface"] & ~chosen & (height.abs() <= ON_CUT * tolerance)
                   & ((reach - ring_at).abs() <= RIM_GIVE * radius))
        entry.update(height=height, up=up, turn=turn, ring_at=ring_at, reach=reach,
                     ease=torch.where(on_seam, torch.maximum(ease, cover), ease))

    eased = 0.0
    if matched["rejoined"]:
        notes.append("the neck was joined before, so its shape was kept and its shading and skin weights matched again")
    elif band_height > 0:
        fits = {}
        for role in ("head", "body"):
            pick = [(entry, (entry["ease"] > 0) & (entry["up"] >= SLOPE_FROM * band_height))
                    for entry in edited.values() if entry["role"] == role and "up" in entry]
            empty = [torch.zeros(0, dtype=torch.float64)]
            fits[role] = _sector_fit(torch.cat([entry["height"][chosen] for entry, chosen in pick] or empty),
                                     torch.cat([(entry["reach"] - entry["ring_at"])[chosen] for entry, chosen in pick]
                                               or empty),
                                     torch.cat([entry["turn"][chosen] for entry, chosen in pick] or empty))
        both = fits["head"][1] & fits["body"][1]
        slope = torch.where(both, (fits["head"][0] + fits["body"][0]) / 2,
                            torch.where(fits["head"][1], fits["head"][0], fits["body"][0]))
        slope = _filled(slope, fits["head"][1] | fits["body"][1])
        slope = (slope.roll(1) + slope + slope.roll(-1)) / 3
        for index, entry in edited.items():
            if "up" not in entry or not bool((entry["ease"] > 0).any()):
                continue
            part = model.parts[index]
            slope_at = _field(SECTOR_MIDDLES, slope[:, None], entry["turn"], fade=False)[:, 0]
            line = entry["ring_at"] + slope_at * entry["height"]
            fair = ((1 - _smooth(entry["up"] / (FAIR_SHARE * band_height))) * (entry["ease"] > 0)
                    * _smooth(1 - (entry["reach"] - line).abs() / (TUBE_GIVE * radius)))
            moving = (fair > 0) & (entry["height"].abs() > ON_CUT * tolerance)
            if not bool(moving.any()):
                continue
            before = part.vertices.clone()
            points = part.vertices.double()
            reach = entry["reach"][moving].clamp_min(1e-12)
            wanted = reach + (line[moving] - reach) * fair[moving]
            eased = max(eased, float((wanted - reach).abs().max()))
            offset = points[moving][:, [0, 2]] - centre
            points[moving, 0] = centre[0] + offset[:, 0] * wanted / reach
            points[moving, 2] = centre[1] + offset[:, 1] * wanted / reach
            part.vertices = points.to(part.vertices.dtype)
            character_assemble._bend_normals(part, before, moving | (entry["stations"] >= 0))

    step = 0.0
    if "head" in first_normals and "body" in first_normals:
        cosine = (first_normals["head"] * first_normals["body"]).sum(dim=1).clamp(-1.0, 1.0)
        step = math.degrees(float(torch.acos(cosine).max()))
    shared = _shared_normals(model, edited)
    for index, entry in edited.items():
        part, rows = model.parts[index], entry["stations"]
        found, held = shared[index]
        surface = entry["surface"]
        surface[found[~torch.isfinite(held).all(dim=1)]] = False
        geometric = torch.zeros(part.vertex_count, 3, dtype=torch.float64)
        known = torch.zeros(part.vertex_count, dtype=torch.bool)
        geometric[found], known[found] = held, True
        normals = part.normals.double().clone()
        chosen = (rows >= 0) & surface & known
        normals[chosen] = geometric[chosen]
        moving = (entry["ease"] > 0) & surface & known & ~chosen
        if bool(moving.any()):
            share = entry["ease"][moving, None]
            normals[moving] = torch.nn.functional.normalize(normals[moving] * (1 - share) + geometric[moving] * share,
                                                            dim=1)
        changed = chosen | moving
        part.normals = normals.to(part.normals.dtype)
        if part.tangents is not None and bool(changed.any()):
            tangents = part.tangents.double().clone()
            along, normal = tangents[changed, :3], normals[changed]
            tangents[changed, :3] = torch.nn.functional.normalize(
                along - normal * (along * normal).sum(dim=1, keepdim=True), dim=1)
            part.tangents = tangents.to(part.tangents.dtype)

    gap = (head_at - body_at).norm(dim=1)
    figures = {
        "seam height m": round(level, 4), "neck radius m": round(radius, 4), "shared vertices": stations,
        "head vertices added": sum(entry["added"] for entry in edited.values() if entry["role"] == "head"),
        "body vertices added": sum(entry["added"] for entry in edited.values() if entry["role"] == "body"),
        "gap closed max mm": round(float(gap.max()) * 1000, 3),
        "gap closed mean mm": round(float(gap.mean()) * 1000, 3),
        "normal step closed degrees": round(step, 2), "ring joined degrees": round(math.degrees(turned), 1),
        "shape eased max mm": round(eased * 1000, 3),
        "weights": "", "cap": "", "parts merged": 0, "notes": notes,
    }

    skinned = [model.parts[index] for index in edited]
    rigged = model.skeleton is not None and all(part.joints is not None and part.weights is not None
                                                for part in skinned)
    if rigged:
        joints = model.skeleton.count
        sums: dict[str, list] = {}
        for index, entry in edited.items():
            part = model.parts[index]
            dense = arrays.dense_skin(part.joints, part.weights, joints).double()
            total_w, seen = _station_sums(dense, entry["stations"], stations)
            held = sums.setdefault(entry["role"], [0, 0])
            held[0], held[1] = held[0] + total_w, held[1] + seen
        side_skin = {role: held[0] / held[1].clamp_min(1)[:, None] for role, held in sums.items()}
        target_skin = _along_ring((side_skin["head"] + side_skin["body"]) / 2, angles)
        figures["weight step closed"] = round(float(0.5 * (side_skin["head"] - side_skin["body"]).abs().sum(dim=1)
                                                    .max()), 3)
        for index, entry in edited.items():
            part = model.parts[index]
            width = int(part.joints.shape[1])
            dense, changed = _toward_ring(arrays.dense_skin(part.joints, part.weights, joints).double(), entry,
                                          target_skin, angles, centre, part.vertices)
            if not bool(changed.any()):
                continue
            picked_joints, picked_weights = arrays.sparse_skin(dense[changed].float(), influences=width)
            part_joints, part_weights = part.joints.long().clone(), part.weights.float().clone()
            part_joints[changed], part_weights[changed] = picked_joints, picked_weights
            part.joints, part.weights = part_joints.to(part.joints.dtype), part_weights.to(part.weights.dtype)
        figures["weights"] = f"matched at {stations} shared vertices"
    elif model.skeleton is not None:
        notes.append("some parts at the neck have no skin weights, so the weights were left as they were")
        figures["weights"] = "left: a part at the neck has none"
    else:
        figures["weights"] = "not rigged"

    names = set()
    for index in edited:
        names |= set(model.parts[index].morphs)
    for name in sorted(names):
        sides: dict[str, list] = {}
        for index, entry in edited.items():
            part = model.parts[index]
            morph = part.morphs.get(name)
            offsets = morph.positions.double() if morph is not None else torch.zeros(part.vertex_count, 3,
                                                                                    dtype=torch.float64)
            total_m, seen = _station_sums(offsets, entry["stations"], stations)
            held = sides.setdefault(entry["role"], [0, 0, False])
            held[0], held[1], held[2] = held[0] + total_m, held[1] + seen, held[2] or morph is not None
        side_offset = {role: held[0] / held[1].clamp_min(1)[:, None] for role, held in sides.items()}
        both = all(held[2] for held in sides.values()) and len(sides) == 2
        target_offset = _along_ring((side_offset["head"] + side_offset["body"]) / 2, angles) if both \
            else torch.zeros(stations, 3, dtype=torch.float64)
        for index, entry in edited.items():
            part = model.parts[index]
            morph = part.morphs.get(name)
            if morph is None:
                continue
            offsets, _ = _toward_ring(morph.positions.double().clone(), entry, target_offset, angles, centre,
                                      part.vertices)
            morph.positions = offsets.to(morph.positions.dtype)

    removed = 0
    for index, entry in edited.items():
        dropped = entry["collapsed"] | (entry["caps"] if full_ring else torch.zeros_like(entry["caps"]))
        if bool(dropped.any()):
            removed += int((entry["caps"] & dropped).sum())
            model.parts[index] = merge.subset_part(model.parts[index], torch.nonzero(~dropped).squeeze(1))
    figures["faces collapsed"] = sum(int(entry["collapsed"].sum()) for entry in edited.values())
    if full_ring and caps.numel():
        figures["cap"] = f"removed, {removed} faces"
    elif caps.numel():
        figures["cap"] = "kept under the hair"
    else:
        figures["cap"] = "none"

    if merge_parts:
        figures["parts merged"] = merge.merge_by_material(model, sorted(edited))
        if not figures["parts merged"]:
            notes.append("the head and body are drawn with different materials, so they stay separate parts")
    return model, figures
