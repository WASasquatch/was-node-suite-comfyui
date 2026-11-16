"""Cutting a surface with a plane: splitting the triangles it crosses, keeping a side, capping the openings.

A plane is a point and a unit normal; the side the normal points to is above. Caps are flat.
"""

from __future__ import annotations

__all__ = [
    "CAP_UV_MODES", "cap_openings", "cut_model", "cut_surface", "plane_basis", "split_by_plane",
    "triangulate_polygon",
]

import math

import torch

from ..model3d.merge import reindex_parts
from ..model3d.scene import Model3D
from . import topology
from .clean import (
    Surface, append_rows, blend, compact, part_of, relax_patch, remove_faces, surface_of, take,
)
from .device import with_cpu_fallback

#: Ear clipping rounds one polygon may take before it is given up.
MAX_EAR_ROUNDS = 20_000

#: Corners one cap polygon may have, holes included.
MAX_CAP_CORNERS = 20_000

#: Point and triangle tests one ear clipping pass makes at once.
EAR_TESTS = 4_000_000

#: Segment tests one hole bridge makes at once.
BRIDGE_TESTS = 4_000_000

#: Edge flip rounds that even out a cap's triangles.
CAP_RELAX = True

#: Width of the texture square an ``"edge"`` cap is projected into.
CAP_UV_SPAN = 1.0 / 8192.0

#: Cap texture modes: a speck at a border corner, or the whole unit square.
CAP_UV_MODES = ("edge", "planar")


def plane_basis(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit axes spanning a plane, ``u x v = normal``.

    Args:
        normal: ``(3,)`` unit normal.

    Returns:
        ``(u, v)``, each ``(3,)`` float64.
    """
    normal = torch.nn.functional.normalize(normal.double(), dim=0)
    helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=normal.device)
    if abs(float(normal[1])) > 0.9:
        helper = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=normal.device)
    u = torch.nn.functional.normalize(torch.cross(helper, normal, dim=0), dim=0)
    v = torch.cross(normal, u, dim=0)
    return u, v


def split_by_plane(surface: Surface, point: torch.Tensor, normal: torch.Tensor,
                   snap: float = 0.0) -> tuple[Surface, torch.Tensor]:
    """Split every triangle the plane crosses so each face lies on one side.

    Args:
        surface: The surface.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal; its side is above.
        snap: Vertices this close to the plane move onto it.

    Returns:
        ``(surface, side)``: the split surface and ``(F,)`` int64 side of each face, 1 above and -1 below.
    """
    device = surface.device
    point = point.to(device).double()
    normal = torch.nn.functional.normalize(normal.to(device).double(), dim=0)
    distance = (surface.vertices.double() - point) @ normal
    near = distance.abs() <= float(snap)
    vertices = surface.vertices
    if bool(near.any()):
        vertices = vertices.clone()
        vertices[near] = (surface.vertices[near].double() - distance[near, None] * normal).float()
        distance = torch.where(near, torch.zeros_like(distance), distance)
        surface = Surface(vertices, surface.faces, surface.attributes, surface.source, surface.flipped, surface.copy)
    ids, positions = surface.topology()
    sign = torch.sign(distance).long()
    faces = surface.faces
    corner_sign = sign[faces]
    above = (corner_sign > 0).sum(dim=1)
    below = (corner_sign < 0).sum(dim=1)
    crossing = (above > 0) & (below > 0)
    face_normal = topology.face_normals(surface.vertices.double(), faces)
    side = torch.where(above > 0, 1, torch.where(below > 0, -1, torch.where((face_normal @ normal) > 0, -1, 1)))
    if not bool(crossing.any()):
        return surface, side
    rows = torch.nonzero(crossing).squeeze(1)
    signs = corner_sign[rows]
    zeros = (signs == 0).sum(dim=1)
    lone_above = (signs > 0).sum(dim=1) == 1
    special = torch.where(zeros == 1, (signs == 0).long().argmax(dim=1),
                          torch.where(lone_above, (signs > 0).long().argmax(dim=1), (signs < 0).long().argmax(dim=1)))
    turn = (special[:, None] + torch.arange(3, device=device)[None, :]) % 3
    corner = faces[rows][torch.arange(rows.numel(), device=device)[:, None], turn]
    one_zero = zeros == 1
    first_a = torch.where(one_zero, corner[:, 1], corner[:, 0])
    first_b = torch.where(one_zero, corner[:, 2], corner[:, 1])
    second_a = corner[:, 0]
    second_b = corner[:, 2]
    pairs = torch.cat([torch.stack([first_a, first_b], 1), torch.stack([second_a[~one_zero], second_b[~one_zero]], 1)])
    low = torch.where(ids[pairs[:, 0]] <= ids[pairs[:, 1]], pairs[:, 0], pairs[:, 1])
    high = torch.where(ids[pairs[:, 0]] <= ids[pairs[:, 1]], pairs[:, 1], pairs[:, 0])
    count = surface.vertex_count
    unique_key, inverse = torch.unique(low * count + high, return_inverse=True)
    start, end = unique_key // count, unique_key % count
    d0, d1 = distance[start], distance[end]
    t = (d0 / (d0 - d1)).clamp(0.0, 1.0)
    p0, p1 = surface.vertices[start].double(), surface.vertices[end].double()
    crossing_point = (p0 + (p1 - p0) * t[:, None]).float()
    added = blend(surface.attributes, torch.stack([start, end], 1), torch.stack([1.0 - t, t], 1).float())
    new_id = count + torch.arange(unique_key.numel(), device=device)
    first_cross = new_id[inverse[:rows.numel()]]
    second_cross = torch.full_like(first_cross, -1)
    second_cross[~one_zero] = new_id[inverse[rows.numel():]]
    s0 = sign[corner[:, 0]]
    s1 = sign[corner[:, 1]]
    s2 = sign[corner[:, 2]]
    zero_faces = torch.cat([torch.stack([corner[:, 0], corner[:, 1], first_cross], 1),
                            torch.stack([corner[:, 0], first_cross, corner[:, 2]], 1)])[
        torch.cat([one_zero, one_zero])]
    zero_side = torch.cat([s1[one_zero], s2[one_zero]])
    lone = ~one_zero
    x1, x2 = first_cross[lone], second_cross[lone]
    l, o1, o2 = corner[lone, 0], corner[lone, 1], corner[lone, 2]
    lone_faces = torch.cat([torch.stack([l, x1, x2], 1), torch.stack([x1, o1, o2], 1), torch.stack([x1, o2, x2], 1)])
    lone_side = torch.cat([s0[lone], s1[lone], s1[lone]])
    new_faces = torch.cat([zero_faces, lone_faces])
    new_side = torch.cat([zero_side, lone_side])
    keep = ~crossing
    added_count = int(new_faces.shape[0])
    result = Surface(
        torch.cat([surface.vertices, crossing_point]),
        torch.cat([faces[keep], new_faces]),
        append_rows(surface.attributes, added),
        torch.cat([surface.source[keep], torch.full((added_count,), -1, dtype=torch.int64, device=device)]),
        torch.cat([surface.flipped[keep], torch.zeros(added_count, dtype=torch.bool, device=device)]),
        torch.cat([surface.copy[keep], torch.zeros(added_count, dtype=torch.int64, device=device)]),
    )
    return result, torch.cat([side[keep], new_side])


def _segments_cross(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Whether 2D segments ``ab`` and ``cd`` cross at a point inside both."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    return ((orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0))


def _bridge(points: torch.Tensor, ring: list[int], hole: list[int], others: list[list[int]]) -> list[int] | None:
    """Join a hole into the outer ring through the nearest outer corner it can see."""
    device = points.device
    hole_t = torch.tensor(hole, device=device)
    start = int(hole_t[points[hole_t, 0].argmax()])
    at_hole = hole.index(start)
    ring_t = torch.tensor(ring, device=device)
    origin = points[start]
    loops = [ring, hole] + others
    edge_a = torch.cat([torch.tensor(loop, device=device) for loop in loops])
    edge_b = torch.cat([torch.tensor(loop[1:] + loop[:1], device=device) for loop in loops])
    pa, pb = points[edge_a], points[edge_b]
    order = torch.argsort((points[ring_t] - origin).norm(dim=1))
    width = max(1, BRIDGE_TESTS // max(int(edge_a.numel()), 1))
    for begin in range(0, order.numel(), width):
        chosen = order[begin:begin + width]
        candidates = points[ring_t[chosen]]
        crosses = _segments_cross(origin[None, None, :], candidates[:, None, :], pa[None, :, :], pb[None, :, :])
        corner = ring_t[chosen][:, None]
        touching = ((edge_a[None, :] == corner) | (edge_b[None, :] == corner)
                    | (edge_a[None, :] == start) | (edge_b[None, :] == start))
        clear = torch.nonzero(~(crosses & ~touching).any(dim=1)).squeeze(1)
        if clear.numel():
            pick = int(chosen[clear[0]])
            rotated = hole[at_hole:] + hole[:at_hole]
            return ring[:pick + 1] + rotated + [start] + ring[pick:]
    return None


def _side(p: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Twice the signed area of the 2D triangle ``p, q, s``."""
    return (q[..., 0] - p[..., 0]) * (s[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (s[..., 0] - p[..., 0])


def triangulate_polygon(points: torch.Tensor, outer: list[int], holes: list[list[int]] | None = None) -> torch.Tensor | None:
    """Ear-clip a counter-clockwise polygon with clockwise holes in the plane.

    Args:
        points: ``(N, 2)`` float64 corner positions.
        outer: Corner indices of the outer boundary, counter-clockwise.
        holes: Corner indices of each hole, clockwise.

    Returns:
        ``(T, 3)`` int64 counter-clockwise triangles of corner indices, or None when the polygon cannot be cut.
    """
    device = points.device
    ring = list(outer)
    pending = sorted(holes or [], key=lambda hole: -float(points[torch.tensor(hole, device=device), 0].max()))
    for number, hole in enumerate(pending):
        joined = _bridge(points, ring, hole, pending[number + 1:])
        if joined is None:
            return None
        ring = joined
    ring_t = torch.tensor(ring, dtype=torch.int64, device=device)
    triangles = []
    for _ in range(MAX_EAR_ROUNDS):
        size = ring_t.numel()
        if size < 3:
            break
        if size == 3:
            triangles.append(ring_t[None, :])
            break
        prev, nxt = ring_t.roll(1), ring_t.roll(-1)
        a, b, c = points[prev], points[ring_t], points[nxt]
        turn = (b[:, 0] - a[:, 0]) * (c[:, 1] - b[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - b[:, 0])
        scale = ((c - a) ** 2).sum(1).clamp_min(1e-300)
        convex = turn > 1e-12 * scale
        reflex = torch.nonzero(~convex).squeeze(1)
        ear = convex.clone()
        candidates = torch.nonzero(convex).squeeze(1)
        if reflex.numel() and candidates.numel():
            r = points[ring_t[reflex]][None, :, :]
            corner = ring_t[reflex][None, :]
            width = max(1, EAR_TESTS // reflex.numel())
            for begin in range(0, candidates.numel(), width):
                rows = candidates[begin:begin + width]
                pa, pb, pc = a[rows][:, None, :], b[rows][:, None, :], c[rows][:, None, :]
                inside = (_side(pa, pb, r) >= 0) & (_side(pb, pc, r) >= 0) & (_side(pc, pa, r) >= 0)
                same = (corner == prev[rows][:, None]) | (corner == ring_t[rows][:, None]) | (corner == nxt[rows][:, None])
                ear[rows] = ~(inside & ~same).any(dim=1)
        ears = torch.nonzero(ear).squeeze(1)
        if not ears.numel():
            return None
        chosen = ears[ears % 2 == 0]
        if not chosen.numel():
            chosen = ears[:1]
        if chosen.numel() > 1 and int(chosen[-1]) == size - 1 and int(chosen[0]) == 0:
            chosen = chosen[:-1]
        triangles.append(torch.stack([prev[chosen], ring_t[chosen], nxt[chosen]], dim=1))
        keep = torch.ones(size, dtype=torch.bool, device=device)
        keep[chosen] = False
        ring_t = ring_t[keep]
    if not triangles:
        return None
    return torch.cat(triangles)


def cap_openings(surface: Surface, point: torch.Tensor, normal: torch.Tensor,
                 tolerance: float, uv_mode: str = "edge") -> tuple[Surface, int, int]:
    """Close the openings a cut left in the plane with flat caps facing along ``normal``.

    Args:
        surface: The kept side of a cut.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal the caps face.
        tolerance: Distance from the plane a border corner may have.
        uv_mode: ``"edge"`` projects each cap into a speck of texture at one of its border corners;
            ``"planar"`` projects all caps across the whole unit square.

    Returns:
        ``(surface, capped, failed)``: openings capped and openings left open.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    device = surface.device
    point = point.to(device).double()
    normal = torch.nn.functional.normalize(normal.to(device).double(), dim=0)
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    chains, closed = topology.boundary_chains(welded, point_count, edges)
    on_plane = ((positions.double() - point) @ normal).abs() <= tolerance
    loops = [chain for chain, is_closed in zip(chains, closed.tolist())
             if is_closed and chain.numel() >= 3 and bool(on_plane[chain].all())]
    if not loops:
        return surface, 0, 0
    half = torch.nonzero(edges.boundary[edges.face_edges.reshape(-1)]).squeeze(1)
    start_of = torch.full((point_count,), -1, dtype=torch.int64, device=device)
    start_of[welded.reshape(-1)[half]] = surface.faces.reshape(-1)[half]
    u_axis, v_axis = plane_basis(normal)
    flat = [loop.flip(0) for loop in loops]
    corners = torch.cat(flat)
    plane_points = torch.stack([(positions[corners].double() - point) @ u_axis,
                                (positions[corners].double() - point) @ v_axis], dim=1)
    offsets, at = [], 0
    for loop in flat:
        offsets.append(at)
        at += loop.numel()
    area = []
    for begin, loop in zip(offsets, flat):
        p = plane_points[begin:begin + loop.numel()]
        q = p.roll(-1, 0)
        area.append(0.5 * float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum()))
    outers = [k for k, value in enumerate(area) if value > 0]
    holes = [k for k, value in enumerate(area) if value < 0]
    owner: dict[int, list[int]] = {k: [] for k in outers}
    failed = sum(1 for value in area if value == 0)
    for hole in holes:
        probe = plane_points[offsets[hole]]
        best, best_area = None, math.inf
        for outer in outers:
            p = plane_points[offsets[outer]:offsets[outer] + flat[outer].numel()]
            q = p.roll(-1, 0)
            crosses = ((p[:, 1] > probe[1]) != (q[:, 1] > probe[1]))
            x = p[:, 0] + (probe[1] - p[:, 1]) * (q[:, 0] - p[:, 0]) / (q[:, 1] - p[:, 1]).where(
                (q[:, 1] - p[:, 1]) != 0, torch.ones_like(q[:, 1]))
            if int((crosses & (x > probe[0])).sum()) % 2 == 1 and area[outer] < best_area:
                best, best_area = outer, area[outer]
        if best is None:
            failed += 1
        else:
            owner[best].append(hole)
    cap_faces, capped = [], 0
    cap_of_corner = torch.full((corners.numel(),), -1, dtype=torch.int64, device=device)
    cap_first: list[int] = []
    for outer, inner in owner.items():
        outer_index = list(range(offsets[outer], offsets[outer] + flat[outer].numel()))
        hole_index = [list(range(offsets[k], offsets[k] + flat[k].numel())) for k in inner]
        if len(outer_index) + sum(len(h) for h in hole_index) > MAX_CAP_CORNERS:
            failed += 1 + len(inner)
            continue
        triangles = triangulate_polygon(plane_points, outer_index, hole_index)
        if triangles is None:
            failed += 1 + len(inner)
            continue
        cap_faces.append(triangles)
        for member in (outer, *inner):
            cap_of_corner[offsets[member]:offsets[member] + flat[member].numel()] = len(cap_first)
        cap_first.append(offsets[outer])
        capped += 1 + len(inner)
    if not cap_faces:
        return surface, 0, failed
    local_faces = torch.cat(cap_faces)
    used = torch.unique(local_faces)
    renumber = torch.full((corners.numel(),), -1, dtype=torch.int64, device=device)
    renumber[used] = torch.arange(used.numel(), device=device)
    local_faces = renumber[local_faces]
    if CAP_RELAX:
        flat_points = torch.cat([plane_points[used], torch.zeros(used.numel(), 1, dtype=torch.float64, device=device)], 1)
        boundary = topology.edge_topology(local_faces, used.numel())
        fixed = boundary.edges[boundary.edge_face_count == 1]
        count = used.numel()
        local_faces = relax_patch(flat_points, local_faces, fixed[:, 0] * count + fixed[:, 1],
                             torch.zeros(0, dtype=torch.int64, device=device))
    source_vertex = start_of[corners[used]]
    attributes = take(surface.attributes, source_vertex.clamp_min(0))
    if "normals" in attributes:
        attributes["normals"] = normal.float().expand(used.numel(), 3).clone().to(attributes["normals"].dtype)
    if "tangents" in attributes:
        tangent = torch.cat([u_axis.float(), torch.ones(1, dtype=torch.float32, device=device)])
        attributes["tangents"] = tangent.expand(used.numel(), 4).clone().to(attributes["tangents"].dtype)
    if "uvs" in attributes:
        plane = plane_points[used]
        if uv_mode == "planar":
            low = plane.amin(dim=0)
            extent = float((plane.amax(dim=0) - low).max())
            projected = (plane - low) / max(extent, 1e-30)
            projected[:, 1] = 1.0 - projected[:, 1]
        else:
            cap = cap_of_corner[used]
            caps = len(cap_first)
            low = torch.full((caps, 2), math.inf, dtype=torch.float64, device=device)
            high = torch.full((caps, 2), -math.inf, dtype=torch.float64, device=device)
            low.scatter_reduce_(0, cap[:, None].expand(-1, 2), plane, reduce="amin")
            high.scatter_reduce_(0, cap[:, None].expand(-1, 2), plane, reduce="amax")
            extent = (high - low).amax(dim=1).clamp_min(1e-30)
            first = torch.tensor(cap_first, dtype=torch.int64, device=device)
            anchor = surface.attributes["uvs"][start_of[corners[first]].clamp_min(0)].double()
            offset = (plane - low[cap]) / extent[cap, None] * CAP_UV_SPAN
            offset[:, 1] = -offset[:, 1]
            projected = anchor[cap] + offset
        attributes["uvs"] = projected.to(attributes["uvs"].dtype)
    base = surface.vertex_count
    new_faces = local_faces + base
    added = int(new_faces.shape[0])
    result = Surface(
        torch.cat([surface.vertices, positions[corners[used]]]),
        torch.cat([surface.faces, new_faces]),
        append_rows(surface.attributes, attributes),
        torch.cat([surface.source, torch.full((added,), -1, dtype=torch.int64, device=device)]),
        torch.cat([surface.flipped, torch.zeros(added, dtype=torch.bool, device=device)]),
        torch.cat([surface.copy, torch.zeros(added, dtype=torch.int64, device=device)]),
    )
    return result, capped, failed


def cut_surface(surface: Surface, point: torch.Tensor, normal: torch.Tensor, keep: str = "above",
                cap: bool = True, snap: float = 0.0,
                uv_mode: str = "edge") -> tuple[list[tuple[str, Surface]], dict]:
    """Cut a surface with a plane, keep one side or both, and cap what was opened.

    Args:
        surface: The surface.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal; its side is above.
        keep: ``"above"``, ``"below"`` or ``"both"``.
        cap: Close the openings with flat caps.
        snap: Vertices this close to the plane move onto it.
        uv_mode: Texture coordinates of the caps, one of `CAP_UV_MODES`.

    Returns:
        ``(pieces, counts)``: ``("above" | "below", surface)`` pairs with faces, and named counts.

    Raises:
        ValueError: For an unknown ``keep`` or ``uv_mode``.
    """
    if keep not in ("above", "below", "both"):
        raise ValueError(f"keep must be above, below or both, not {keep!r}")
    if uv_mode not in CAP_UV_MODES:
        raise ValueError(f"the cap texture mode must be edge or planar, not {uv_mode!r}")
    faces_before = surface.face_count
    split, side = split_by_plane(surface, point, normal, snap)
    counts = {"split_faces": split.face_count - faces_before, "capped": 0, "uncapped": 0}
    normal = torch.nn.functional.normalize(normal.double(), dim=0)
    scale = float((split.vertices.amax(dim=0) - split.vertices.amin(dim=0)).norm()) if split.vertex_count else 1.0
    tolerance = max(float(snap), 1e-6 * scale)
    pieces = []
    for name, wanted, facing in (("above", 1, -normal), ("below", -1, normal)):
        if keep not in (name, "both"):
            continue
        piece = remove_faces(split, side == wanted)
        piece, _removed = compact(piece)
        if piece.face_count == 0:
            continue
        if cap:
            piece, capped, failed = cap_openings(piece, point, facing, tolerance, uv_mode)
            counts["capped"] += capped
            counts["uncapped"] += failed
        pieces.append((name, piece))
    return pieces, counts


def cut_model(model: Model3D, chosen: list[int], point: torch.Tensor, normal: torch.Tensor,
              keep: str = "above", cap: bool = True, snap: float = 0.0, uv_mode: str = "edge",
              device: torch.device | str | None = None) -> dict:
    """Cut the chosen parts with one plane in place; ``"both"`` makes ``<name>_above`` and ``<name>_below``.

    Args:
        model: The model, changed in place.
        chosen: Indices of the parts to cut.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal; its side is above.
        keep: ``"above"``, ``"below"`` or ``"both"``.
        cap: Close the openings with flat caps.
        snap: Vertices this close to the plane move onto it.
        uv_mode: Texture coordinates of the caps, one of `CAP_UV_MODES`.
        device: The first device to try; None means the mesh compute device.

    Returns:
        Counts: ``cut_parts``, ``removed_parts``, ``split_faces``, ``capped`` and ``uncapped``.

    Raises:
        ValueError: For an unknown ``keep`` or ``uv_mode``, or a normal of zero length.
    """
    if keep not in ("above", "below", "both"):
        raise ValueError(f"keep must be above, below or both, not {keep!r}")
    if uv_mode not in CAP_UV_MODES:
        raise ValueError(f"the cap texture mode must be edge or planar, not {uv_mode!r}")
    normal = normal.double().cpu()
    if not bool(torch.isfinite(normal).all()) or float(normal.norm()) <= 1e-12:
        raise ValueError("the cut plane's normal has no length; set at least one of normal_x, normal_y "
                         "and normal_z to a value other than 0")
    normal = normal / normal.norm()
    point = point.double().cpu()
    wanted = set(chosen)
    totals = {"cut_parts": 0, "removed_parts": 0, "split_faces": 0, "capped": 0, "uncapped": 0}
    parts, source = [], []
    for index, part in enumerate(model.parts):
        if index not in wanted or part.triangle_count == 0:
            parts.append(part)
            source.append([index])
            continue
        surface = surface_of(part)

        def work(target: torch.device, surface=surface):
            pieces, counts = cut_surface(surface.to(target), point.to(target), normal.to(target), keep, cap,
                                         snap, uv_mode)
            return [(name, piece.to("cpu")) for name, piece in pieces], counts

        pieces, counts = with_cpu_fallback(work, device, label="mesh cut")
        for name in ("split_faces", "capped", "uncapped"):
            totals[name] += counts[name]
        if counts["split_faces"] or len(pieces) != 1:
            totals["cut_parts"] += 1
        if not pieces:
            totals["removed_parts"] += 1
        for side, piece in pieces:
            result = part_of(part, piece)
            if len(pieces) > 1:
                result.name = f"{part.name}_{side}"
            parts.append(result)
            source.append([index])
    reindex_parts(model, parts, source)
    return totals
