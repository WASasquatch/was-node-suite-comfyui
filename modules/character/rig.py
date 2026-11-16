"""Rigging a standing character: a VRM humanoid skeleton fitted to the mesh, skin weights, look-at and blink.

Joints sit at measured features; weights are bone heat over a reduced copy, and over each hand for its
fingers.
"""

from __future__ import annotations

__all__ = ["BONES", "LOOK_RANGES", "REST_POSES", "fit_skeleton", "heat_weights", "rig"]

import math

import torch

from ..log import get_logger
from ..mesh import arrays, bvh, sections, topology, transfer
from ..mesh import decimate as mesh_decimate
from ..model3d import orient
from ..model3d.scene import Expression, Model3D, Skeleton
from . import assemble, face, hands

logger = get_logger("character.rig")

#: The humanoid bones fitted, parents first, each with its parent.
BONES = (
    ("hips", None), ("spine", "hips"), ("chest", "spine"), ("upperChest", "chest"), ("neck", "upperChest"),
    ("head", "neck"),
    ("leftShoulder", "upperChest"), ("leftUpperArm", "leftShoulder"), ("leftLowerArm", "leftUpperArm"),
    ("leftHand", "leftLowerArm"),
    ("rightShoulder", "upperChest"), ("rightUpperArm", "rightShoulder"), ("rightLowerArm", "rightUpperArm"),
    ("rightHand", "rightLowerArm"),
    ("leftUpperLeg", "hips"), ("leftLowerLeg", "leftUpperLeg"), ("leftFoot", "leftLowerLeg"),
    ("leftToes", "leftFoot"),
    ("rightUpperLeg", "hips"), ("rightLowerLeg", "rightUpperLeg"), ("rightFoot", "rightLowerLeg"),
    ("rightToes", "rightFoot"),
)

#: ``T-pose`` turns the arms level, as VRM expects; ``as modelled`` keeps the pose the mesh has.
REST_POSES = ("T-pose", "as modelled")

#: Spine joints as shares of the span from the leg roots to the neck joint.
SPINE_SHARES = {"hips": 0.145, "spine": 0.30, "chest": 0.52, "upperChest": 0.64}

#: Head joint above the neck ring, as a share of the ring's height below the top.
HEAD_ABOVE_RING = 0.27

#: Labels of parts that move with the head: weighted to the head and neck only.
HEAD_LABELS = ("head", "hair")

#: A head part's vertex with less than this weight on the head and neck is bound to the head.
HEAD_WEIGHT_FLOOR = 0.25

#: Shoulder joints below the neck joint, as a share of the span; and their reach towards the arm.
SHOULDER_DROP = 0.13
CLAVICLE_REACH = 0.41

#: Shoulder joint of a level arm as a share of its line from the body's middle to the fingertips; elbow
#: and wrist as shares from the shoulder joint to the fingertips.
SHOULDER_SHARE, ELBOW_SHARE, WRIST_SHARE = 0.17, 0.36, 0.71

#: An arm whose line rises or falls more steeply than this sine finds its shoulder at shoulder height.
SLOPED_ARM = 0.34

#: Share of the arm's reach from the middle where tracing in from the fingertips stops.
ARM_TRACE_FROM = 0.45

#: Largest step between neighbouring arm sections, as a share of the figure's height.
ARM_JUMP = 0.04

#: Knee and ankle heights as shares of the leg root's height above the floor; toes above the floor
#: as a share of the figure, and along the foot from heel to tip.
KNEE_SHARE, ANKLE_SHARE = 0.57, 0.152
TOES_HEIGHT, TOES_ALONG = 0.025, 0.7

#: Cross-sections per side when tracing an arm.
ARM_LEVELS = 96

#: Triangles of the reduced copy the skeleton is fitted to and the weights are solved on.
PROXY_TRIANGLES = 60000

#: Bones a vertex may be anchored to when the nearest is hidden behind the surface.
VISIBLE_CANDIDATES = 4

#: Weights under this are dropped before the strongest four are kept.
WEIGHT_FLOOR = 0.02

#: Arms within this many degrees of level count as level already.
LEVEL_TOLERANCE = 8.0

#: Triangles of the reduced copy of a hand its finger weights are worked out on.
HAND_TRIANGLES = 30000

#: Least hand weight a vertex holds to be measured as part of the hand when fingers are fitted.
HAND_MEMBER = 0.5

#: Head and body vertices this close, as a share of the figure's height, are one shared vertex.
SEAM_TOLERANCE = 1e-6

#: Distance from a shared neck ring, as a share of its radius, over which weights ease back to their own.
SEAM_BAND = 0.6

#: VRM look-at range maps for eye bones: degrees of gaze against degrees the eyes turn.
LOOK_RANGES = {
    "rangeMapHorizontalInner": {"inputMaxValue": 90.0, "outputScale": 8.0},
    "rangeMapHorizontalOuter": {"inputMaxValue": 90.0, "outputScale": 12.0},
    "rangeMapVerticalDown": {"inputMaxValue": 90.0, "outputScale": 10.0},
    "rangeMapVerticalUp": {"inputMaxValue": 90.0, "outputScale": 10.0},
}

#: Expressions a blink is written as, and the shape keys each binds.
BLINK_EXPRESSIONS = {
    "blink": ("blinkLeft", "blinkRight"),
    "blinkLeft": ("blinkLeft",),
    "blinkRight": ("blinkRight",),
}


def _proxy(model: Model3D, triangles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every part's triangles welded into one surface, reduced to at most ``triangles``."""
    vertices, faces = assemble._geometry(model)
    welded, wfaces, _ = topology.weld(vertices, faces, tolerance=0)
    if wfaces.shape[0] > triangles:
        reduced, _ = mesh_decimate.decimate([arrays.TriMesh(welded, wfaces)], target_triangles=triangles,
                                            max_error=math.inf)
        welded, wfaces = reduced[0].vertices, reduced[0].faces
    return welded.float(), wfaces.long()


def _arm_line(vertices: torch.Tensor, faces: torch.Tensor, side: float, centre_x: float, low_y: float,
              high_y: float, height: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A straight line through one arm, traced in from the fingertips: ``(point, outward direction, tip)``."""
    reach = float(vertices[:, 0].max()) - centre_x if side > 0 else centre_x - float(vertices[:, 0].min())
    if reach <= 0.05 * height:
        return None
    offsets = torch.linspace(ARM_TRACE_FROM * reach, reach - 0.002 * height, ARM_LEVELS)
    levels = sorted((centre_x + side * offsets).tolist())
    found = sections.profile(vertices, faces, axis=(1.0, 0.0, 0.0), levels=levels)
    order = list(reversed(range(found.level_count))) if side > 0 else list(range(found.level_count))
    points, previous = [], None
    for level in order:
        loops = [loop for loop in found.loops(level) if abs(loop.area) > 0
                 and low_y <= float(loop.centroid[1]) <= high_y]
        if not loops:
            continue
        if previous is None:
            chosen = max(loops, key=lambda loop: float(loop.centroid[1]))
        else:
            chosen = min(loops, key=lambda loop: float((loop.centroid - previous)[1:].norm()))
            if float((chosen.centroid - previous)[1:].norm()) > ARM_JUMP * height:
                continue
        previous = chosen.centroid
        points.append(chosen.centroid.float())
    if len(points) < 6:
        return None
    line = torch.stack(points)
    middle = line.mean(dim=0)
    direction = torch.linalg.svd(line - middle, full_matrices=False).Vh[0]
    if float(direction[0]) * side < 0:
        direction = -direction
    tip_x = centre_x + side * reach
    tip = middle + direction * ((tip_x - float(middle[0])) / float(direction[0]))
    return middle, direction, tip


def _side_loop(vertices: torch.Tensor, faces: torch.Tensor, y: float, side: float, centre_x: float):
    """The loop of one leg at height ``y``: the closed loop on that side nearest the centre line."""
    loops = sections.slice_plane(vertices, faces, (centre_x, y, 0.0), (0.0, 1.0, 0.0))
    loops = [loop for loop in loops if abs(loop.area) > 0 and (float(loop.centroid[0]) - centre_x) * side > 0]
    if not loops:
        return None
    return max(loops, key=lambda loop: abs(loop.area))


def fit_skeleton(vertices: torch.Tensor, faces: torch.Tensor,
                 neck_level: float | None = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Joint heads for `BONES` on a figure standing on Y facing +Z, and where each bone ends.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` triangles.
        neck_level: Height of the neck ring a separate head was joined at, or None to measure it.

    Returns:
        ``(heads, tips)``: bone name against ``(3,)`` position, and the end of each bone's segment.

    Raises:
        ValueError: The mesh does not read as a standing figure with two legs.
    """
    low, high = vertices.min(dim=0).values, vertices.max(dim=0).values
    height = float(high[1] - low[1])
    floor = float(low[1])
    try:
        found = sections.landmarks(vertices, faces, facing="+z")
    except ValueError as error:
        raise ValueError(f"the model does not read as a standing figure ({error}); stand it on the floor "
                         "facing forward before rigging") from None
    if not found.legs_separable or "left" not in found.legs or "right" not in found.legs:
        raise ValueError("the legs could not be told apart, as under a long skirt or robe; Character Rig needs "
                         "two legs that separate below the hips")
    centre_x = float(found.crotch[0])
    torso_z = float(found.hips.centre[2])
    if neck_level is not None:
        neck_y = float(neck_level)
    else:
        neck_ring = assemble._axis_neck(vertices, faces, assemble.BODY_NECK_BAND)
        neck_y = float(neck_ring[0][1]) if neck_ring is not None else float(found.neck.centre[1])
    head_y = neck_y + HEAD_ABOVE_RING * (float(high[1]) - neck_y)
    root_y = 0.5 * (float(found.legs["left"].root[1]) + float(found.legs["right"].root[1]))
    span = max(neck_y - root_y, 0.05 * height)

    def torso_point(y: float) -> torch.Tensor:
        loops = sections.slice_plane(vertices, faces, (centre_x, y, 0.0), (0.0, 1.0, 0.0))
        holding = [loop for loop in loops if loop.closed and abs(loop.area) > 0
                   and assemble._inside(loop.points[:, [0, 2]], torch.tensor([centre_x, torso_z]))]
        z = float(min(holding, key=lambda loop: abs(loop.area)).centroid[2]) if holding else torso_z
        return torch.tensor([centre_x, y, z])

    heads = {name: torso_point(root_y + share * span) for name, share in SPINE_SHARES.items()}
    heads["neck"] = torso_point(neck_y)
    heads["head"] = torso_point(head_y)
    tips = {"head": torch.tensor([centre_x, float(high[1]), float(heads["head"][2])])}

    for prefix, side in (("left", 1.0), ("right", -1.0)):
        arm = _arm_line(vertices, faces, side, centre_x, floor + 0.35 * height, head_y, height)
        if arm is None:
            raise ValueError(f"the {prefix} arm could not be traced; Character Rig needs both arms held away "
                             "from the body, in a T-pose or an A-pose")
        middle, direction, tip = arm
        shoulder_y = neck_y - SHOULDER_DROP * span
        if abs(float(direction[1])) > SLOPED_ARM:
            shoulder_joint = middle + direction * ((shoulder_y - float(middle[1])) / float(direction[1]))
        else:
            base = middle + direction * ((centre_x - float(middle[0])) / float(direction[0]))
            shoulder_joint = base + (tip - base) * SHOULDER_SHARE
        reach = tip - shoulder_joint
        heads[f"{prefix}UpperArm"] = shoulder_joint
        heads[f"{prefix}LowerArm"] = shoulder_joint + reach * ELBOW_SHARE
        heads[f"{prefix}Hand"] = shoulder_joint + reach * WRIST_SHARE
        tips[f"{prefix}Hand"] = tip
        clavicle = heads["neck"] + (shoulder_joint - heads["neck"]) * CLAVICLE_REACH
        clavicle[1] = neck_y - SHOULDER_DROP * span
        heads[f"{prefix}Shoulder"] = clavicle

        leg = found.legs[prefix]
        hip = leg.root.float().clone()
        heads[f"{prefix}UpperLeg"] = hip
        knee_y = floor + KNEE_SHARE * (float(hip[1]) - floor)
        ankle_y = floor + ANKLE_SHARE * (float(hip[1]) - floor)
        for name, y in ((f"{prefix}LowerLeg", knee_y), (f"{prefix}Foot", ankle_y)):
            loop = _side_loop(vertices, faces, y, side, centre_x)
            if loop is None:
                along = (y - float(hip[1])) / min(float(leg.tip[1]) - float(hip[1]), -1e-6)
                heads[name] = hip + (leg.tip.float() - hip) * along
            else:
                heads[name] = loop.centroid.float()
        toe_loop = _side_loop(vertices, faces, floor + TOES_HEIGHT * height, side, centre_x)
        if toe_loop is not None:
            heel, front = float(toe_loop.points[:, 2].min()), float(toe_loop.points[:, 2].max())
            toes = toe_loop.centroid.float().clone()
            toes[2] = heel + TOES_ALONG * (front - heel)
            tips[f"{prefix}Toes"] = torch.tensor([float(toes[0]), float(toes[1]), front])
        else:
            toes = heads[f"{prefix}Foot"] + torch.tensor([0.0, -0.5 * (ankle_y - floor), 0.05 * height])
            tips[f"{prefix}Toes"] = toes + torch.tensor([0.0, 0.0, 0.03 * height])
        heads[f"{prefix}Toes"] = toes
    return heads, tips


def _segments(heads: dict[str, torch.Tensor], tips: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(start, end)`` of each bone's segment, ``(J, 3)`` each, in `BONES` order."""
    children: dict[str, list[str]] = {}
    for name, parent in BONES:
        if parent is not None:
            children.setdefault(parent, []).append(name)
    first_child = {"hips": "spine", "upperChest": "neck"}
    starts, ends = [], []
    for name, _ in BONES:
        start = heads[name]
        if name in tips:
            end = tips[name]
        else:
            child = first_child.get(name) or children[name][0]
            end = heads[child]
        starts.append(start)
        ends.append(end)
    return torch.stack(starts), torch.stack(ends)


def _segment_distance(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance from each point to each segment and the closest point: ``(V, J)`` and ``(V, J, 3)``."""
    direction = end - start
    length_sq = (direction * direction).sum(dim=1).clamp_min(1e-12)
    offset = points[:, None, :] - start[None, :, :]
    t = ((offset * direction[None]).sum(dim=2) / length_sq[None]).clamp(0, 1)
    closest = start[None] + t[..., None] * direction[None]
    return (points[:, None, :] - closest).norm(dim=2), closest


def heat_weights(vertices: torch.Tensor, faces: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Bone heat weights: each vertex heated by the nearest bone it can see, then diffused.

    Args:
        vertices: ``(V, 3)`` positions of a welded surface.
        faces: ``(F, 3)`` triangles.
        start: ``(J, 3)`` bone segment starts.
        end: ``(J, 3)`` bone segment ends.

    Returns:
        ``(V, J)`` weights, rows summing to 1.
    """
    low, high = vertices.min(dim=0).values, vertices.max(dim=0).values
    scale = float((high - low).max()) or 1.0
    points = (vertices - low) / scale
    start, end = (start - low) / scale, (end - low) / scale
    count, bones = points.shape[0], start.shape[0]
    distance, closest = _segment_distance(points, start, end)
    order = distance.argsort(dim=1)[:, :VISIBLE_CANDIDATES]
    tree = bvh.TriangleBVH(points, faces, device="cpu")
    chosen = torch.full((count,), -1, dtype=torch.long)
    rows = torch.arange(count)
    for rank in range(order.shape[1]):
        open_rows = rows[chosen < 0]
        if open_rows.numel() == 0:
            break
        bone = order[open_rows, rank]
        origin = closest[open_rows, bone]
        offset = points[open_rows] - origin
        length = offset.norm(dim=1).clamp_min(1e-9)
        blocked = tree.any_hit(origin, offset / length[:, None], t_min=1e-6, t_max=length * (1 - 1e-3) - 1e-5)
        chosen[open_rows[~blocked]] = bone[~blocked]
    nearest = distance.gather(1, chosen.clamp_min(0)[:, None]).squeeze(1).clamp_min(1e-4)
    heat = torch.where(chosen >= 0, 1.0 / nearest ** 2, torch.zeros_like(nearest))
    area = topology.vertex_areas(points.double(), faces).float()
    target = torch.zeros(count, bones)
    target[rows[chosen >= 0], chosen[chosen >= 0]] = 1.0
    laplacian = topology.cotangent_laplacian(points, faces).coalesce()
    diagonal = area * heat + 1e-6 * area.mean()
    indices = torch.arange(count)
    system = torch.sparse_coo_tensor(
        torch.cat([laplacian.indices(), torch.stack([indices, indices])], dim=1),
        torch.cat([laplacian.values().float(), diagonal]), (count, count), check_invariants=False).coalesce()
    weights = topology.solve_cg(system, (area * heat)[:, None] * target, tol=1e-7, max_iter=3000)
    weights = weights.clamp_min(0)
    weights = torch.where(weights < WEIGHT_FLOOR, torch.zeros_like(weights), weights)
    total = weights.sum(dim=1, keepdim=True)
    fallback = torch.zeros_like(weights)
    fallback[rows, distance.argmin(dim=1)] = 1.0
    return torch.where(total > 0, weights / total.clamp_min(1e-12), fallback)


def _level_arms(vertices: torch.Tensor, faces: torch.Tensor, heads: dict[str, torch.Tensor],
                tips: dict[str, torch.Tensor], weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Turn both arms level about the shoulder joints, moving the skin with them.

    Returns:
        ``(vertices, degrees)``: the posed proxy vertices and the larger turn applied.
    """
    names = [name for name, _ in BONES]
    turned = 0.0
    moved = vertices.clone()
    for prefix, side in (("left", 1.0), ("right", -1.0)):
        pivot = heads[f"{prefix}UpperArm"]
        reach = tips[f"{prefix}Hand"] - pivot
        angle = math.degrees(math.atan2(float(reach[1]), abs(float(reach[0])) + 1e-9))
        if abs(angle) <= LEVEL_TOLERANCE:
            continue
        turn = orient.rotation_matrix(0.0, 0.0, side * -angle).float()
        chain = [names.index(f"{prefix}{bone}") for bone in ("UpperArm", "LowerArm", "Hand")]
        share = weights[:, chain].sum(dim=1, keepdim=True)
        rotated = (vertices - pivot) @ turn.T + pivot
        moved = moved + share * (rotated - vertices)
        for bone in ("LowerArm", "Hand"):
            heads[f"{prefix}{bone}"] = (heads[f"{prefix}{bone}"] - pivot) @ turn.T + pivot
        tips[f"{prefix}Hand"] = (tips[f"{prefix}Hand"] - pivot) @ turn.T + pivot
        turned = max(turned, abs(angle))
    return moved, turned


def _joined_neck(model: Model3D) -> float | None:
    """Height of the neck ring a separate head was set on: as recorded, else the body's top, or None."""
    heads = [part for part in model.parts if part.label == "head" and part.vertex_count]
    bodies = [part for part in model.parts if part.label == "body" and part.vertex_count]
    if not heads or not bodies:
        return None
    share = model.meta.get(assemble.NECK_RING_KEY)
    if isinstance(share, (int, float)) and 0.0 < float(share) < 1.0:
        low, _ = model.bounds()
        return float(low[1]) + float(share) * model.height()
    body_top = max(float(part.vertices[:, 1].max()) for part in bodies)
    head_low = min(float(part.vertices[:, 1].min()) for part in heads)
    head_top = max(float(part.vertices[:, 1].max()) for part in heads)
    return body_top if head_low < body_top < head_top else None


def _head_skin(skin: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Weights of a part that moves with the head, kept to the head and neck."""
    columns = [names.index("neck"), names.index("head")]
    kept = torch.zeros_like(skin)
    kept[:, columns] = skin[:, columns]
    total = kept.sum(dim=1, keepdim=True)
    bound = torch.zeros_like(skin)
    bound[:, columns[1]] = 1.0
    return torch.where(total >= HEAD_WEIGHT_FLOOR, kept / total.clamp_min(1e-12), bound)


def _join_neck(model: Model3D, skins: list[torch.Tensor]) -> int:
    """Give head and body vertices at one position the ring's mean weights, easing back to their own.

    Args:
        model: The character.
        skins: ``(V, J)`` weights of each part, changed in place.

    Returns:
        Positions the head and body share.
    """
    height = max(model.height(), 1e-9)
    head = [index for index, part in enumerate(model.parts) if part.label in HEAD_LABELS and part.vertex_count]
    body = [index for index, part in enumerate(model.parts) if part.label not in HEAD_LABELS and part.vertex_count]
    if not head or not body:
        return 0

    def keys(indices: list[int]) -> torch.Tensor:
        points = torch.cat([model.parts[index].vertices.double() for index in indices])
        return torch.round(points / (SEAM_TOLERANCE * height)).long()

    head_keys, body_keys = keys(head), keys(body)
    _, inverse = torch.unique(torch.cat([head_keys, body_keys]), dim=0, return_inverse=True)
    head_ids, body_ids = inverse[:head_keys.shape[0]], inverse[head_keys.shape[0]:]
    in_body = torch.zeros(int(inverse.max()) + 1, dtype=torch.bool)
    in_body[body_ids] = True
    in_head = torch.zeros_like(in_body)
    in_head[head_ids] = True
    shared_ids = in_body & in_head
    if not bool(shared_ids.any()):
        return 0
    body_skin = torch.cat([skins[index] for index in body])
    ring_weight = body_skin[shared_ids[body_ids]].mean(dim=0)
    ring = torch.cat([model.parts[index].vertices.double() for index in head])[shared_ids[head_ids]]
    middle = ring.mean(dim=0)
    radius = float((ring - middle)[:, [0, 2]].norm(dim=1).mean())
    band = max(SEAM_BAND * radius, 1e-9)
    for index in head + body:
        points = model.parts[index].vertices.double()
        near = ((points - middle).norm(dim=1) < radius + band)
        if not bool(near.any()):
            continue
        rows = torch.nonzero(near).squeeze(1)
        distance = torch.cat([torch.cdist(points[chunk], ring).min(dim=1).values
                              for chunk in rows.split(65536)])
        share = (distance / band).clamp(0, 1)
        share = (share * share * (3 - 2 * share)).float()[:, None]
        skins[index][rows] = ring_weight[None] * (1 - share) + skins[index][rows] * share
    return int(shared_ids.sum())


def _picked(vertices: torch.Tensor, faces: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The triangles whose every corner is in ``keep``, with only the vertices they use."""
    picked = faces[keep[faces].all(dim=1)]
    used = picked.unique()
    index = torch.full((vertices.shape[0],), -1, dtype=torch.long)
    index[used] = torch.arange(used.numel())
    return vertices[used].float(), index[picked]


def _hand_skin(vertices: torch.Tensor, faces: torch.Tensor, elbow: torch.Tensor, wrist: torch.Tensor,
               tip: torch.Tensor, hand: hands.Hand, prefix: str) -> tuple[arrays.TriMesh, list[str]] | None:
    """A hand's surface carrying, as ``"skin"``, each vertex's split of its forearm and hand weight.

    Args:
        vertices: ``(V, 3)`` every vertex with some of the hand's weight.
        faces: ``(F, 3)`` the triangles between them.
        elbow: ``(3,)`` elbow joint.
        wrist: ``(3,)`` wrist joint.
        tip: ``(3,)`` end of the hand bone when the hand has no fingers.
        hand: The fingers fitted to the hand.
        prefix: ``"left"`` or ``"right"``.

    Returns:
        ``(surface, bones)``: the columns are the lower arm, the hand and each finger bone fitted,
        or None when the hand has no fingers.
    """
    if faces.shape[0] == 0 or not hand.heads:
        return None
    points, triangles, _ = topology.weld(vertices.float(), faces, tolerance=0)
    if triangles.shape[0] > HAND_TRIANGLES:
        reduced, _ = mesh_decimate.decimate([arrays.TriMesh(points, triangles)], target_triangles=HAND_TRIANGLES,
                                            max_error=math.inf)
        points, triangles = reduced[0].vertices.float(), reduced[0].faces.long()
    rows = [(name, parent) for name, parent in hands.finger_bones(prefix) if name in hand.heads]
    child = {parent: name for name, parent in rows}
    names = [f"{prefix}LowerArm", f"{prefix}Hand"] + [name for name, _ in rows]
    starts = [elbow, wrist] + [hand.heads[name] for name, _ in rows]
    ends = [wrist, hand.knuckles if hand.knuckles is not None else tip]
    ends += [hand.heads[child[name]] if name in child else hand.tips[name] for name, _ in rows]
    weights = heat_weights(points, triangles, torch.stack(starts).float(), torch.stack(ends).float())
    return arrays.TriMesh(points, triangles, {"skin": weights}), names


def _share_hand(skin: torch.Tensor, vertices: torch.Tensor, names: list[str], surface: arrays.TriMesh,
                bones: list[str]) -> None:
    """Share each hand vertex's lower arm and hand weight among the surface's bones by its split.

    Args:
        skin: ``(V, J)`` weights, changed in place.
        vertices: ``(V, 3)`` positions.
        names: Bone name of each column.
        surface: The hand's surface from `_hand_skin`.
        bones: Its columns: lower arm, hand, then the finger bones.
    """
    forearm, palm = names.index(bones[0]), names.index(bones[1])
    rows = skin[:, palm] > 0
    if not bool(rows.any()):
        return
    carried, _ = transfer.transfer(surface, vertices[rows].float(), ["skin"])
    split = carried["skin"]
    block = skin[rows]
    held = block[:, forearm] + block[:, palm]
    missed = split.sum(dim=1) < 0.5
    split[missed] = 0.0
    split[missed, 0] = block[missed, forearm] / held[missed]
    split[missed, 1] = block[missed, palm] / held[missed]
    block[:, forearm] = 0.0
    block[:, palm] = 0.0
    block[:, [names.index(bone) for bone in bones]] += held[:, None] * split
    skin[rows] = block


def _keep_strongest(skin: torch.Tensor, influences: int = arrays.SKIN_INFLUENCES) -> torch.Tensor:
    """Weights cut to ``influences`` per vertex smoothly: the next strongest is taken off every weight.

    Args:
        skin: ``(V, J)`` weights, rows summing to 1.
        influences: Bones a vertex keeps.

    Returns:
        ``(V, J)`` weights with at most ``influences`` above zero per row, rows summing to 1.
    """
    if skin.shape[1] <= influences:
        return skin
    cut = skin.topk(influences + 1, dim=1).values[:, -1:]
    kept = (skin - cut).clamp_min(0)
    total = kept.sum(dim=1, keepdim=True)
    return torch.where(total > 1e-6, kept / total.clamp_min(1e-12), skin)


def _unique_key(name: str, taken) -> str:
    """``name``, or ``name_2`` and so on when ``taken`` holds it."""
    if name not in taken:
        return name
    number = 2
    while f"{name}_{number}" in taken:
        number += 1
    return f"{name}_{number}"


def _hand_fact(hand: hands.Hand) -> str:
    """How one hand's fingers were found, in a few words."""
    if not hand.heads:
        return "no fingers found"
    found = []
    if hand.apart:
        found.append(f"{hand.apart} apart")
    if hand.joined:
        found.append(f"{hand.joined} joined")
    found.append("thumb" if hand.thumb else "no thumb")
    return ", ".join(found)


def rig(model: Model3D, rest_pose: str = "T-pose", proxy_triangles: int = PROXY_TRIANGLES,
        facing: str = "auto", fingers: bool = True, eyes: bool = True, blink: bool = True) -> tuple[Model3D, dict]:
    """Give a standing character a VRM humanoid skeleton, skin weights, look-at and a blink.

    Args:
        model: The character, changed in place; any skeleton it has is replaced.
        rest_pose: One of `REST_POSES`.
        proxy_triangles: Triangles of the reduced copy the work is done on.
        facing: One of `assemble.FACINGS`; ``"auto"`` keeps a model marked as facing +Z.
        fingers: Fit finger bones and share the hands' weight with them.
        eyes: Place eye bones and a VRM look-at; eyeball parts are bound to them.
        blink: Add shape keys and ``blink``, ``blinkLeft`` and ``blinkRight`` expressions that close
            painted open eyes, unless the model has a blink already.

    Returns:
        ``(model, figures)``.

    Raises:
        ValueError: The model has no triangles or does not read as a standing figure.
    """
    vertices, faces = assemble._geometry(model)
    turn, how = assemble._body_turn(model, facing, vertices, faces)
    if abs(turn) > 1e-6:
        low, high = model.bounds()
        pivot = ((low + high) / 2).double()
        pivot[1] = float(low[1])
        orient.transform(model, assemble._about(pivot, orient.similarity(orient.rotation_matrix(0.0, turn, 0.0))))
    if not how.startswith("taken as"):
        model.meta[orient.FACES_FORWARD_KEY] = True
    for part in model.parts:
        part.joints = part.weights = None
    model.skeleton = None
    proxy_v, proxy_f = _proxy(model, max(5000, int(proxy_triangles)))
    heads, tips = fit_skeleton(proxy_v, proxy_f, _joined_neck(model))
    start, end = _segments(heads, tips)
    weights = heat_weights(proxy_v, proxy_f, start, end)
    turned = 0.0
    if rest_pose == "T-pose":
        posed, turned = _level_arms(proxy_v, proxy_f, heads, tips, weights)
        if turned:
            source = arrays.TriMesh(proxy_v, proxy_f, {"offset": posed - proxy_v})
            for part in model.parts:
                moved, _ = transfer.transfer(source, part.vertices.float(), ["offset"])
                part.vertices = part.vertices.float() + moved["offset"]
                part.normals = None
                part.tangents = None
            proxy_v = posed
    bones = list(BONES)
    body = [name for name, _ in BONES]
    proxy = arrays.TriMesh(proxy_v, proxy_f, {"skin": weights})
    skins = []
    for part in model.parts:
        carried, _ = transfer.transfer(proxy, part.vertices.float(), ["skin"])
        skin = carried["skin"]
        skins.append(_head_skin(skin, body) if part.label in HEAD_LABELS else skin)
    hand_skins = []
    facts: dict[str, str] = {}
    shared = _join_neck(model, skins)
    if shared:
        facts["neck"] = f"head and body weights matched at {shared} shared vertices"
    if fingers:
        vertices, faces = assemble._geometry(model)
        for prefix in ("left", "right"):
            column = body.index(f"{prefix}Hand")
            held = torch.cat([skin[:, column] for part, skin in zip(model.parts, skins)
                              if part.vertex_count and part.triangle_count])
            wrist, tip = heads[f"{prefix}Hand"], tips[f"{prefix}Hand"]
            hand = hands.fit_hand(*_picked(vertices, faces, held >= HAND_MEMBER), wrist, tip, prefix)
            facts[f"{prefix} fingers"] = _hand_fact(hand)
            for name, parent in hands.finger_bones(prefix):
                if name in hand.heads:
                    bones.append((name, parent))
                    heads[name] = hand.heads[name]
            surface = _hand_skin(*_picked(vertices, faces, held > 0), heads[f"{prefix}LowerArm"], wrist, tip,
                                 hand, prefix)
            if surface is not None:
                hand_skins.append(surface)
    found = None
    if eyes:
        found = face.find_eyes(model, float(heads["head"][0]), float(heads["neck"][1]), float(model.bounds()[1][1]))
        if found is None:
            facts["eyes"] = "no face found above the neck"
        else:
            heads["leftEye"], heads["rightEye"] = found.left.float(), found.right.float()
            bones += [("leftEye", "head"), ("rightEye", "head")]
            facts["eyes"] = {"eyeballs": "eyeball parts, bound to the eye bones",
                             "painted": "painted open; eye bones move no vertices",
                             "proportions": "none painted open; placed from the face, moving no vertices"}[found.kind]
    names = [name for name, _ in bones]
    for number, (part, skin) in enumerate(zip(model.parts, skins)):
        full = torch.zeros(part.vertex_count, len(names))
        full[:, :len(BONES)] = skin
        for surface, finger_columns in hand_skins:
            _share_hand(full, part.vertices, names, surface, finger_columns)
        if found is not None:
            for side, pieces in found.eyeballs.items():
                for index, held in pieces:
                    if index == number:
                        full[held] = 0.0
                        full[held, names.index(f"{side}Eye")] = 1.0
        part.joints, part.weights = arrays.sparse_skin(_keep_strongest(full))
    rest = torch.eye(4).repeat(len(names), 1, 1)
    rest[:, :3, 3] = torch.stack([heads[name] for name in names])
    parents = torch.tensor([names.index(parent) if parent else -1 for _, parent in bones], dtype=torch.long)
    model.skeleton = Skeleton(names=list(names), parents=parents, rest=rest,
                              inverse_bind=torch.linalg.inv(rest), humanoid={name: i for i, name in enumerate(names)})
    if found is not None:
        between = (found.left.double() + found.right.double()) / 2 - heads["head"].double()
        model.extras["vrm_look_at"] = {"type": "bone", "offsetFromHeadBone": [float(value) for value in between],
                                       **{key: dict(value) for key, value in LOOK_RANGES.items()}}
    else:
        model.extras.pop("vrm_look_at", None)
    made_blink = False
    if blink:
        if any(model.expressions.get(name) is not None and model.expressions[name].binds for name in BLINK_EXPRESSIONS):
            facts["blink"] = "kept the model's own"
        elif found is None:
            facts["blink"] = "not made: no eyes"
        else:
            shapes = face.blink_shapes(model, found)
            if not shapes:
                facts["blink"] = "not made: no eyes painted open with lids above them"
            else:
                keys: dict[int, dict[str, str]] = {}
                for index, targets in shapes.items():
                    part = model.parts[index]
                    for shape, target in targets.items():
                        key = _unique_key(shape, part.morphs)
                        part.morphs[key] = target
                        keys.setdefault(index, {})[shape] = key
                for expression, wanted in BLINK_EXPRESSIONS.items():
                    binds = [(index, held[shape], 1.0) for index, held in keys.items() for shape in wanted if shape in held]
                    if binds:
                        model.expressions[expression] = Expression(name=expression, binds=binds)
                made_blink = True
                facts["blink"] = "made"
    figures = {"joints": len(names), "turned to face +Z": round(turn, 2), "arms levelled by": round(turned, 1),
               "proxy triangles": int(proxy_f.shape[0]), "facing": how,
               "finger bones": sum(1 for name, _ in bones if any(finger in name for finger in hands.FINGERS)),
               "eye bones": 2 if found is not None else 0, "blink": made_blink, "notes": facts}
    return model, figures
