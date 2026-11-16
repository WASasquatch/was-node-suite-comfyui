"""Mirror symmetry: finding a model's mirror plane, mirroring one side onto the other, pairing left and right names.

A plane is a point and a unit normal; the side the normal points to is the positive side.
"""

from __future__ import annotations

__all__ = [
    "AXES", "PLANES", "fit_plane", "joint_pairs", "mirror_error", "mirror_names", "name_pairs", "reflect",
    "symmetrize_model", "symmetrize_skeleton", "symmetrize_surface", "symmetry_error",
]

import math
import re

import torch

from ..model3d.merge import reindex_parts
from ..model3d.scene import Model3D, Skeleton
from . import arrays, topology
from .bvh import TriangleBVH
from .clean import (
    MORPH_NORMAL, Surface, append_rows, blend_skin, compact, part_of, remove_degenerate, remove_faces, surface_of,
    take,
)
from .cut import split_by_plane
from .device import compute_device, with_cpu_fallback

#: Starting plane normals by name.
AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

#: Surface points the mirrored samples are matched against while the plane is refined.
SAMPLES = 12_000

#: Surface points mirrored in each refinement round against the matched samples.
ROUND_SAMPLES = 4_000

#: Surface points mirrored to compare candidate planes.
CANDIDATE_SAMPLES = 2_000

#: Most rounds of the plane refinement.
ICP_ROUNDS = 40

#: Refinement rounds given to each of the best candidate planes.
CANDIDATE_ROUNDS = 6

#: Candidate planes refined before the best is chosen.
CANDIDATES_KEPT = 3

#: A round improving the mirror distance by less than this share ends the refinement.
SETTLED = 1e-4

#: Surface points mirrored in the closing rounds, which measure exact distances to the surface.
POLISH_SAMPLES = 2_000

#: Closing rounds against the exact surface.
POLISH_ROUNDS = 3

#: Distances worked out per pass of the nearest-sample search.
MATCH_CHUNK = 4_000_000

#: Share of mirrored samples whose distance sets the outlier limit of a refinement round.
KEEP_QUANTILE = 0.9

#: Mirrored samples farther than this many times that distance are left out of a refinement round.
OUTLIER_FACTOR = 1.5

#: Left and right spellings swapped to find a name's mirror, as ``(left pattern, right pattern, left, right)``.
_SIDE_PATTERNS = (
    (re.compile(r"Left"), re.compile(r"Right"), "Left", "Right"),
    (re.compile(r"left"), re.compile(r"right"), "left", "right"),
    (re.compile(r"LEFT"), re.compile(r"RIGHT"), "LEFT", "RIGHT"),
    (re.compile(r"(?<![A-Za-z0-9])L(?=[^A-Za-z0-9]|$)"), re.compile(r"(?<![A-Za-z0-9])R(?=[^A-Za-z0-9]|$)"), "L", "R"),
    (re.compile(r"(?<![A-Za-z0-9])l(?=[^A-Za-z0-9]|$)"), re.compile(r"(?<![A-Za-z0-9])r(?=[^A-Za-z0-9]|$)"), "l", "r"),
    (re.compile(r"(?<=[a-z0-9])L$"), re.compile(r"(?<=[a-z0-9])R$"), "L", "R"),
)


def reflect(points: torch.Tensor, point: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Points mirrored across a plane.

    Args:
        points: ``(N, 3)`` positions.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal.

    Returns:
        ``(N, 3)`` mirrored positions in the points' dtype.
    """
    p = points.double()
    n = normal.to(p.device).double()
    distance = (p - point.to(p.device).double()) @ n
    return (p - 2.0 * distance[:, None] * n).to(points.dtype)


def _mirror_vectors(vectors: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Directions mirrored across a plane through the origin."""
    n = normal.to(vectors.device).to(torch.float64)
    v = vectors.double()
    return (v - 2.0 * (v @ n)[:, None] * n).to(vectors.dtype)


def _samples(vertices: torch.Tensor, faces: torch.Tensor,
             count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Area-weighted random surface points, the area-weighted centroid and each point's face, from a fixed seed."""
    area = topology.face_areas(vertices.double(), faces)
    total = float(area.sum())
    corners = vertices[faces].double()
    centroid = (corners.mean(dim=1) * area[:, None]).sum(dim=0) / max(total, 1e-300)
    generator = torch.Generator(device="cpu").manual_seed(0)
    if total <= 0:
        return vertices.double()[:count], centroid, torch.zeros(min(count, vertices.shape[0]), dtype=torch.int64)
    face = torch.multinomial(area.float().cpu(), count, replacement=True, generator=generator).to(vertices.device)
    r = torch.rand(count, 2, generator=generator, dtype=torch.float64).to(vertices.device)
    root = r[:, 0].sqrt()
    weights = torch.stack([1.0 - root, root * (1.0 - r[:, 1]), root * r[:, 1]], dim=1)
    return (corners[face] * weights[:, :, None]).sum(dim=1), centroid, face


def symmetry_error(tree: TriangleBVH, samples: torch.Tensor, point: torch.Tensor, normal: torch.Tensor,
                   keep: float = 1.0) -> float:
    """Mean distance from mirrored surface samples to the surface.

    Args:
        tree: Tree over the surface.
        samples: ``(N, 3)`` surface points.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal.
        keep: Share of the nearest samples averaged; 1 averages all of them.

    Returns:
        The mean distance in model units.
    """
    distance, *_ = tree.closest_point(reflect(samples, point, normal).float())
    distance = distance[torch.isfinite(distance)].double()
    if keep < 1.0 and distance.numel():
        distance = distance.sort().values[:max(1, int(distance.numel() * keep))]
    return float(distance.mean()) if distance.numel() else math.inf


def _nearest(reference: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance to and index of the nearest reference point of each query point."""
    rows = max(1, MATCH_CHUNK // max(1, reference.shape[0]))
    distance = torch.empty(queries.shape[0], dtype=torch.float32, device=queries.device)
    index = torch.empty(queries.shape[0], dtype=torch.int64, device=queries.device)
    reference = reference.float()
    for start in range(0, queries.shape[0], rows):
        block = torch.cdist(queries[start:start + rows].float(), reference)
        distance[start:start + rows], index[start:start + rows] = block.min(dim=1)
    return distance, index


def _measure(match, samples: torch.Tensor, point: torch.Tensor,
             normal: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Mirror distance on a plane, and the Gauss-Newton step towards a better plane.

    Args:
        match: Called with ``(Q, 3)`` mirrored points; returns their distance to the surface, the matched surface
            points and the surface normals there, ``inf`` distance for a point with no match.
        samples: ``(Q, 3)`` float64 surface points.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal.

    Returns:
        ``(error, point, normal)``: the robust mean mirror distance and the stepped plane.
    """
    mirrored = reflect(samples, point, normal)
    distance, closest, surface_normal = match(mirrored)
    usable = torch.isfinite(distance)
    found = distance[usable].double().sort().values
    if found.numel() < 16:
        return math.inf, point, normal
    error = float(found[:max(1, int(found.numel() * KEEP_QUANTILE))].mean())
    limit = OUTLIER_FACTOR * float(found[min(found.numel() - 1, int(found.numel() * KEEP_QUANTILE))]) + 1e-12
    usable &= distance <= limit
    p = samples[usable]
    m = surface_normal[usable]
    residual = ((mirrored[usable] - closest[usable].double()) * m).sum(dim=1)
    helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=normal.device)
    if abs(float(normal[1])) > 0.9:
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=normal.device)
    e1 = torch.nn.functional.normalize(torch.cross(normal, helper, dim=0), dim=0)
    e2 = torch.cross(normal, e1, dim=0)
    local = p - point
    height = local @ normal
    facing = m @ normal
    jacobian = torch.stack([
        -2.0 * ((local @ e1) * facing + height * (m @ e1)),
        -2.0 * ((local @ e2) * facing + height * (m @ e2)),
        2.0 * facing,
    ], dim=1)
    system = jacobian.T @ jacobian + 1e-9 * torch.eye(3, dtype=torch.float64, device=normal.device)
    step = -torch.linalg.solve(system, jacobian.T @ residual)
    turned = torch.nn.functional.normalize(normal + step[0] * e1 + step[1] * e2, dim=0)
    moved = point + step[2] * normal
    return error, moved, turned


def _settle(match, points: torch.Tensor, point: torch.Tensor, normal: torch.Tensor,
            rounds: int) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Gauss-Newton rounds on the plane, halving a step that makes the mirror distance worse.

    Returns:
        ``(point, normal, error, rounds)``: the plane, its mirror distance and the rounds kept.
    """
    error, moved, turned = _measure(match, points, point, normal)
    used = 0
    for _ in range(rounds):
        trial, next_moved, next_turned = _measure(match, points, moved, turned)
        shrink = 0
        while trial > error and shrink < 3:
            moved = 0.5 * (moved + point)
            turned = torch.nn.functional.normalize(0.5 * (turned + normal), dim=0)
            trial, next_moved, next_turned = _measure(match, points, moved, turned)
            shrink += 1
        if trial > error:
            break
        used += 1
        gain = error - trial
        point, normal, error = moved, turned, trial
        moved, turned = next_moved, next_turned
        if gain <= SETTLED * error:
            break
    return point, normal, error, used


def fit_plane(vertices: torch.Tensor, faces: torch.Tensor, axis: str = "X", refine: bool = True,
              device: torch.device | str | None = None) -> dict:
    """Find the plane a surface is most nearly mirrored across.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        axis: ``"X"``, ``"Y"`` or ``"Z"`` to start from that axis, ``"auto"`` to try the axes and principal directions.
        refine: Turn and move the plane until mirrored surface points land on the surface.
        device: Device for the work; None means the mesh compute device.

    Returns:
        ``point`` and ``normal`` (``(3,)`` float64), ``error_before`` and ``error`` (mean mirror distance of the
        nearest `KEEP_QUANTILE` share of samples, on the starting and the found plane), and ``rounds``.

    Raises:
        ValueError: For an unknown axis or a surface with no area.
    """
    if axis not in (*AXES, "auto"):
        raise ValueError(f"the plane must be X, Y, Z or auto, not {axis!r}")
    faces = faces.long()
    area = topology.face_areas(vertices.double(), faces)
    if faces.shape[0] == 0 or float(area.sum()) <= 0:
        raise ValueError("the selected parts have no surface to find a mirror plane on; choose parts that have faces")
    target = torch.device(device) if device is not None else compute_device()
    vertices, faces = vertices.to(target), faces.to(target)
    tree = TriangleBVH(vertices.float(), faces, device=target)
    face_normal = topology.face_normals(vertices.double(), faces)
    samples, centroid, sample_face = _samples(vertices, faces, SAMPLES)
    sample_normal = face_normal[sample_face]

    def by_sample(mirrored):
        distance, index = _nearest(samples, mirrored)
        return distance, samples[index], sample_normal[index]

    def by_tree(mirrored):
        distance, face, _bary, closest = tree.closest_point(mirrored.float())
        hit = face >= 0
        distance = torch.where(hit, distance, torch.full_like(distance, math.inf))
        return distance, closest.double(), face_normal[face.clamp_min(0)]

    if axis == "auto":
        centred = samples - centroid
        _values, vectors = torch.linalg.eigh(centred.T @ centred)
        candidates = [torch.tensor(AXES[name], dtype=torch.float64, device=samples.device) for name in AXES]
        candidates += [vector.to(samples.device) for vector in vectors.T]
        probe = samples[:CANDIDATE_SAMPLES]
        scored = sorted((_measure(by_sample, probe, centroid, candidate)[0], number)
                        for number, candidate in enumerate(candidates))
        tried = []
        for _score, number in scored[:CANDIDATES_KEPT if refine else 1]:
            found = _settle(by_sample, probe, centroid.clone(), candidates[number], CANDIDATE_ROUNDS if refine else 0)
            tried.append((found[2], found[0], found[1]))
        _score, point, normal = min(tried, key=lambda item: item[0])
        normal = normal * (1.0 if float(normal[normal.abs().argmax()]) >= 0 else -1.0)
    else:
        normal = torch.tensor(AXES[axis], dtype=torch.float64, device=samples.device)
        point = centroid.clone()
    if not refine:
        point = centroid.clone()
    reference = normal.clone()
    exact = samples[:POLISH_SAMPLES]
    error_before = error = symmetry_error(tree, exact, point, normal, KEEP_QUANTILE)
    rounds = 0
    if refine:
        point, normal, _error, rounds = _settle(by_sample, samples[:ROUND_SAMPLES], point, normal, ICP_ROUNDS)
        point, normal, error, polished = _settle(by_tree, exact, point, normal, POLISH_ROUNDS)
        rounds += polished
        if float(normal @ reference) < 0:
            normal = -normal
    return {"point": point.double().cpu(), "normal": normal.double().cpu(), "error_before": error_before,
            "error": error, "rounds": rounds}


def mirror_error(vertices: torch.Tensor, faces: torch.Tensor, point: torch.Tensor, normal: torch.Tensor,
                 device: torch.device | str | None = None) -> float:
    """Mean distance from mirrored surface points to the surface, for one plane.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal.
        device: Device for the tree; None means the mesh compute device.

    Returns:
        The mean distance of the nearest `KEEP_QUANTILE` share of samples in model units, ``inf`` for a
        surface with no area.
    """
    faces = faces.long()
    if faces.shape[0] == 0 or float(topology.face_areas(vertices.double(), faces).sum()) <= 0:
        return math.inf
    tree = TriangleBVH(vertices.float(), faces, device=device)
    samples, _, _ = _samples(vertices, faces, ROUND_SAMPLES)
    return symmetry_error(tree, samples, point.double().to(samples.device), normal.double().to(samples.device),
                          KEEP_QUANTILE)


def mirror_names(name: str) -> list[str]:
    """Spellings of a name with its side swapped, one per left and right pattern it holds.

    Args:
        name: A joint or shape key name.

    Returns:
        Candidate names, without the name itself.
    """
    found = []
    for left_pattern, right_pattern, left, right in _SIDE_PATTERNS:
        for pattern, replacement in ((left_pattern, right), (right_pattern, left)):
            if pattern.search(name):
                candidate = pattern.sub(replacement, name)
                if candidate != name and candidate not in found:
                    found.append(candidate)
    return found


def name_pairs(names: list[str]) -> list[int]:
    """The index of each name's mirror partner by spelling, its own index when it has none.

    Args:
        names: Names, such as joint or shape key names.

    Returns:
        Partner index per name; pairs are mutual.
    """
    where = {name: index for index, name in enumerate(names)}
    partner = list(range(len(names)))
    for index, name in enumerate(names):
        if partner[index] != index:
            continue
        for candidate in mirror_names(name):
            other = where.get(candidate)
            if other is not None and other != index and partner[other] == other:
                partner[index], partner[other] = other, index
                break
    return partner


def joint_pairs(skeleton: Skeleton, point: torch.Tensor, normal: torch.Tensor,
                reach: float) -> tuple[torch.Tensor, int, int]:
    """Pair each joint with its mirror: humanoid bones first, then left and right spellings, then position.

    Args:
        skeleton: The skeleton.
        point: ``(3,)`` a point on the mirror plane.
        normal: ``(3,)`` unit normal.
        reach: Largest distance between a mirrored joint and the joint it pairs with by position.

    Returns:
        ``(partner, by_name, by_position)``: ``(J,)`` int64 partner of each joint, its own index for a joint on the
        plane or with no partner, and how many pairs each rule found.
    """
    from ..character.bones import mirror

    count = skeleton.count
    partner = list(range(count))
    for bone, joint in skeleton.humanoid.items():
        other = skeleton.humanoid.get(mirror(bone))
        if other is not None and 0 <= joint < count and 0 <= other < count and other != joint \
                and partner[joint] == joint and partner[other] == other:
            partner[joint], partner[other] = other, joint
    by_name_list = name_pairs(skeleton.names)
    for joint, other in enumerate(by_name_list):
        if other != joint and partner[joint] == joint and partner[other] == other:
            partner[joint], partner[other] = other, joint
    by_name = sum(1 for joint in range(count) if partner[joint] != joint) // 2
    heads = skeleton.heads().double()
    if count:
        side = (heads - point.double()) @ normal.double()
        lonely = torch.tensor([partner[j] == j for j in range(count)]) & (side.abs() > reach)
        mirrored = reflect(heads, point, normal)
        gap = (mirrored[:, None, :] - heads[None, :, :]).norm(dim=2)
        gap[:, ~lonely] = math.inf
        gap[~lonely, :] = math.inf
        gap.fill_diagonal_(math.inf)
        nearest = gap.argmin(dim=1)
        for joint in torch.nonzero(lonely).squeeze(1).tolist():
            other = int(nearest[joint])
            if float(gap[joint, other]) <= reach and int(nearest[other]) == joint \
                    and partner[joint] == joint and partner[other] == other:
                partner[joint], partner[other] = other, joint
    by_position = sum(1 for joint in range(count) if partner[joint] != joint) // 2 - by_name
    return torch.tensor(partner, dtype=torch.int64), by_name, by_position


def symmetrize_skeleton(skeleton: Skeleton, partner: torch.Tensor, point: torch.Tensor,
                        normal: torch.Tensor, centre_reach: float = 0.0) -> int:
    """Move the joint of each pair lying further on the negative side to its partner's mirror, keeping binds.

    Args:
        skeleton: The skeleton, changed in place.
        partner: ``(J,)`` partner of each joint.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal pointing to the kept side.
        centre_reach: Joints with no partner closer than this to the plane move onto it; 0 leaves them.

    Returns:
        How many joints moved.
    """
    heads = skeleton.heads().double()
    point = point.double().to(heads.device)
    normal = normal.double().to(heads.device)
    side = (heads - point) @ normal
    partner = partner.to(heads.device)
    index = torch.arange(skeleton.count)
    moving = (partner != index) & (side < side[partner]) & (side[partner] > 0)
    centring = (partner == index) & (side.abs() <= float(centre_reach)) & (side != 0)
    rows = torch.nonzero(moving | centring).squeeze(1)
    if not rows.numel():
        return 0
    rest = skeleton.rest.double().clone()
    bind = skeleton.inverse_bind.double().clone()
    target = torch.where(moving[rows, None], reflect(heads[partner[rows]], point, normal),
                         heads[rows] - side[rows, None] * normal)
    moved_rest = rest[rows].clone()
    moved_rest[:, :3, 3] = target
    bind[rows] = torch.linalg.inv(moved_rest) @ rest[rows] @ bind[rows]
    rest[rows] = moved_rest
    skeleton.rest = rest.float()
    skeleton.inverse_bind = bind.float()
    return int(rows.numel())


def symmetrize_surface(surface: Surface, point: torch.Tensor, normal: torch.Tensor, snap: float,
                       joint_partner: torch.Tensor | None = None,
                       morph_partner: dict[str, str] | None = None) -> tuple[Surface, dict]:
    """Replace the negative side of a surface with a mirror image of its positive side, sharing the seam.

    Args:
        surface: The surface.
        point: ``(3,)`` a point on the plane.
        normal: ``(3,)`` unit normal pointing to the side kept.
        snap: Vertices this close to the plane join the seam.
        joint_partner: ``(J,)`` mirror partner of each joint, for the skin of mirrored vertices.
        morph_partner: Mirror partner of each shape key name, such as ``blinkLeft`` for ``blinkRight``.

    Returns:
        ``(surface, counts)``: the symmetric surface and counts of removed and mirrored faces, seam vertices, and
        faces dropped for collapsing onto the seam.
    """
    device = surface.device
    point = point.to(device).double()
    normal = torch.nn.functional.normalize(normal.to(device).double(), dim=0)
    split, side = split_by_plane(surface, point, normal, snap)
    distance = (split.vertices.double() - point) @ normal
    tolerance = max(float(snap), 1e-7)
    flat = (distance.abs() <= tolerance)[split.faces].all(dim=1)
    keep = (side > 0) & ~flat
    kept, _ = compact(remove_faces(split, keep))
    removed = int((~keep).sum())
    if kept.face_count == 0:
        return kept, {"removed_faces": removed, "mirrored_faces": 0, "seam_vertices": 0, "collapsed_faces": 0}
    distance = (kept.vertices.double() - point) @ normal
    seam = distance.abs() <= tolerance
    vertices = kept.vertices.clone()
    vertices[seam] = (kept.vertices[seam].double() - distance[seam, None] * normal).float()
    count = kept.vertex_count
    outside = torch.nonzero(~seam).squeeze(1)
    mirror_index = torch.arange(count, device=device)
    mirror_index[outside] = count + torch.arange(outside.numel(), device=device)
    mirrored = _mirrored_attributes(kept.attributes, normal, joint_partner, morph_partner)
    seam_rows = torch.nonzero(seam).squeeze(1)
    attributes = {name: value.clone() for name, value in kept.attributes.items()}
    if seam_rows.numel():
        own_and_mirror = torch.stack([seam_rows, seam_rows + count], dim=1)
        both = append_rows(kept.attributes, mirrored)
        for name, value in attributes.items():
            if name in ("joints",):
                continue
            if name == "weights" and "joints" in attributes:
                joints, weights = blend_skin(both["joints"], both["weights"], own_and_mirror,
                                              torch.full((seam_rows.numel(), 2), 0.5, device=device))
                attributes["joints"][seam_rows] = joints.to(attributes["joints"].dtype)
                attributes["weights"][seam_rows] = weights.to(value.dtype)
                continue
            if name == "uvs" or name == "colors":
                continue
            mixed = 0.5 * (both[name][seam_rows].float() + both[name][seam_rows + count].float())
            if name == "normals":
                mixed = torch.nn.functional.normalize(mixed, dim=1)
            elif name == "tangents":
                mixed = torch.cat([torch.nn.functional.normalize(mixed[:, :3], dim=1), value[seam_rows, 3:].float()], 1)
            attributes[name][seam_rows] = mixed.to(value.dtype)
    new_vertices = reflect(vertices[outside], point, normal)
    new_attributes = take(mirrored, outside)
    faces = mirror_index[kept.faces][:, [0, 2, 1]]
    total_faces = kept.face_count
    result = Surface(
        torch.cat([vertices, new_vertices]),
        torch.cat([kept.faces, faces]),
        append_rows(attributes, new_attributes),
        torch.cat([kept.source, kept.source]),
        torch.cat([kept.flipped, ~kept.flipped]),
        torch.cat([kept.copy, torch.ones(total_faces, dtype=torch.int64, device=device)]),
    )
    result, collapsed, _duplicate = remove_degenerate(result)
    result, _unused = compact(result)
    return result, {"removed_faces": removed, "mirrored_faces": total_faces, "seam_vertices": int(seam_rows.numel()),
                    "collapsed_faces": collapsed}


def _mirrored_attributes(attributes: dict[str, torch.Tensor], normal: torch.Tensor,
                         joint_partner: torch.Tensor | None,
                         morph_partner: dict[str, str] | None) -> dict[str, torch.Tensor]:
    """Attributes as the mirror image needs them: vectors reflected, handedness flipped, skin and shape keys paired."""
    mirrored: dict[str, torch.Tensor] = {}
    for name, value in attributes.items():
        if name == "normals":
            mirrored[name] = _mirror_vectors(value, normal)
        elif name == "tangents":
            mirrored[name] = torch.cat([_mirror_vectors(value[:, :3], normal), -value[:, 3:]], dim=1)
        elif name == "joints" and joint_partner is not None:
            table = joint_partner.to(value.device)
            mirrored[name] = table[value.long().clamp(0, table.numel() - 1)].to(value.dtype)
        elif name.startswith(arrays.MORPH_PREFIX) or name.startswith(MORPH_NORMAL):
            prefix = arrays.MORPH_PREFIX if name.startswith(arrays.MORPH_PREFIX) else MORPH_NORMAL
            key = name[len(prefix):]
            partner = (morph_partner or {}).get(key, key)
            source = attributes.get(prefix + partner, value)
            mirrored[name] = _mirror_vectors(source, normal)
        else:
            mirrored[name] = value
    return mirrored


#: Planes `symmetrize_model` accepts: fitted to the surface, through its centre, or through the origin.
PLANES = ("fit", "centre", "origin")

#: Joints pair by position, and unpaired joints are centred, within this share of the model's diagonal.
JOINT_REACH = 0.03


def symmetrize_model(model: Model3D, chosen: list[int], axis: str = "X", plane: str = "fit",
                     keep: str = "positive", snap: float = 1e-4, skeleton: bool = True,
                     device: torch.device | str | None = None) -> dict:
    """Mirror one side of the chosen parts onto the other in place, skin by joint pairs, shape keys by side names.

    Args:
        model: The model, changed in place.
        chosen: Indices of the parts to mirror; the plane is found on all of them together.
        axis: ``"X"``, ``"Y"``, ``"Z"`` or ``"auto"``.
        plane: One of `PLANES`.
        keep: ``"positive"`` keeps the side the axis points to, ``"negative"`` the other.
        snap: Vertices this close to the plane join the seam.
        skeleton: Move joints on the mirrored side to mirror their partners, and centre unpaired joints near the plane.
        device: The first device to try; None means the mesh compute device.

    Returns:
        ``point`` and ``normal`` lists, ``error_before`` (mean mirror distance), and counts of removed and
        mirrored faces, seam vertices, joints paired by name and by position, joints moved, shape keys paired
        and parts removed.

    Raises:
        ValueError: For an unknown axis, plane or side, or when the chosen parts have no surface.
    """
    if plane not in PLANES:
        raise ValueError(f"the plane must be fit, centre or origin, not {plane!r}")
    if keep not in ("positive", "negative"):
        raise ValueError(f"the side kept must be positive or negative, not {keep!r}")
    surfaces = [index for index in chosen if model.parts[index].triangle_count]
    if not surfaces:
        raise ValueError("the chosen parts have no triangles to mirror; check the parts input or wire a model "
                         "with faces")
    offsets, pieces_v, pieces_f = 0, [], []
    for index in surfaces:
        part = model.parts[index]
        pieces_v.append(part.vertices.float())
        pieces_f.append(part.faces.long() + offsets)
        offsets += part.vertex_count
    vertices, faces = torch.cat(pieces_v), torch.cat(pieces_f)

    def fit(target: torch.device) -> dict:
        found = fit_plane(vertices.to(target), faces.to(target), axis, plane == "fit", target)
        if plane == "origin":
            found["point"] = torch.zeros(3, dtype=torch.float64, device=target)
            found["error"] = mirror_error(vertices.to(target), faces.to(target), found["point"], found["normal"],
                                          target)
        return {name: value.cpu() if isinstance(value, torch.Tensor) else value for name, value in found.items()}

    found = with_cpu_fallback(fit, device, label="mesh symmetrize")
    point, normal = found["point"].double(), found["normal"].double()
    if keep == "negative":
        normal = -normal
    low, high = vertices.amin(dim=0), vertices.amax(dim=0)
    diagonal = float((high - low).norm())
    snap = max(float(snap), 1e-7 * diagonal)
    reach = JOINT_REACH * diagonal
    joint_partner = None
    counts = {"removed_faces": 0, "mirrored_faces": 0, "seam_vertices": 0, "joints_by_name": 0,
              "joints_by_position": 0, "joints_moved": 0, "shape_keys_paired": 0, "removed_parts": 0}
    if model.skeleton is not None and model.skeleton.count:
        joint_partner, counts["joints_by_name"], counts["joints_by_position"] = joint_pairs(
            model.skeleton, point, normal, reach)
    morph_names = sorted({name for index in surfaces for name in model.parts[index].morphs})
    morph_partner_index = name_pairs(morph_names)
    morph_partner = {name: morph_names[other] for name, other in zip(morph_names, morph_partner_index)}
    counts["shape_keys_paired"] = sum(1 for name, other in morph_partner.items() if name != other) // 2
    wanted = set(surfaces)
    parts, source = [], []
    for index, part in enumerate(model.parts):
        if index not in wanted:
            parts.append(part)
            source.append([index])
            continue
        surface = surface_of(part)

        def work(target: torch.device, surface=surface):
            partner = joint_partner.to(target) if joint_partner is not None else None
            result, made = symmetrize_surface(surface.to(target), point.to(target), normal.to(target), snap,
                                              partner, morph_partner)
            return result.to("cpu"), made

        result, made = with_cpu_fallback(work, device, label="mesh symmetrize")
        for name in ("removed_faces", "mirrored_faces", "seam_vertices"):
            counts[name] += made[name]
        if result.face_count == 0:
            counts["removed_parts"] += 1
            continue
        parts.append(part_of(part, result))
        source.append([index])
    reindex_parts(model, parts, source)
    if skeleton and joint_partner is not None:
        counts["joints_moved"] = symmetrize_skeleton(model.skeleton, joint_partner, point, normal, reach)
    return {"point": point.tolist(), "normal": normal.tolist(), "error_before": float(found["error"]), **counts}
