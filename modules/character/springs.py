"""Spring bone chains for a rigged character's hanging hair and cloth, and the colliders they swing against."""
# Hair is found by colour from the crown; the hair hanging below the head joint is split into strands,
# each given a chain of joints from its root to its tip and skinned along it. A cloth part named by the
# caller gets chains spaced round the torso bone it hangs from, its vertices blended between the two
# nearest. Colliders are sized from the surface each bone carries.

from __future__ import annotations

__all__ = ["HAIR_SOURCES", "add_springs"]

import math

import torch

from ..mesh import topology
from ..model3d import selection
from ..model3d.scene import Collider, Model3D, Skeleton, SpringChain, Springs
from . import assemble, bones

#: Where hair is looked for: colour on head and hair parts, hair parts alone, or nowhere.
HAIR_SOURCES = ("auto", "hair parts", "off")

#: Joint name prefixes of the chains this module adds.
PREFIX = "hair_"
CLOTH_PREFIX = "cloth_"

#: Torso bones a cloth part may hang from; it hangs from the highest one below its top.
CLOTH_ROOTS = ("hips", "spine", "chest", "upperChest", "neck")

#: Degrees between neighbouring chains round a cloth part.
CLOTH_SECTOR = 45.0

#: Fewest vertices a stretch of cloth needs for a chain of its own; less shares its neighbour's.
CLOTH_FEWEST = 30

#: Largest hit radius of a cloth chain's joints, in metres.
CLOTH_HIT = 0.03

#: A hanging strand shorter than this share of the head's height is left to move with the head.
SHORTEST_STRAND = 0.25

#: A strand smaller than this share of all hanging hair faces is left to move with the head.
SMALLEST_STRAND = 0.01

#: Below the head joint, a head part's surface farther than this many neck radii from the neck's axis
#: counts as hanging hair whatever its colour.
NECK_CLEARANCE = 1.6

#: Lab distance from the face's skin within which surface clear of the neck still counts as skin,
#: such as the chin and jaw.
SKIN_DISTANCE = 18.0

#: A strand spanning more than this many degrees round the head is cut into pieces of `SECTOR`.
CURTAIN_SPAN, SECTOR = 70.0, 45.0

#: Collider radius as a share of the median distance of a bone's surface from the bone.
COLLIDER_SHARE = 0.8

#: A chain joint's hit radius as a share of the strand's half thickness, and its bounds in metres.
HIT_SHARE, HIT_RANGE = 0.6, (0.005, 0.06)

#: Bones colliders are put on: a sphere on the head, capsules along the others.
COLLIDER_BONES = (("head", "sphere"), ("neck", "capsule"), ("upperChest", "capsule"), ("chest", "capsule"),
                  ("leftUpperArm", "capsule"), ("rightUpperArm", "capsule"), ("hips", "sphere"),
                  ("leftUpperLeg", "capsule"), ("rightUpperLeg", "capsule"))

#: The bone each capsule runs to.
COLLIDER_ENDS = {"neck": "head", "upperChest": "neck", "chest": "upperChest", "leftUpperArm": "leftLowerArm",
                 "rightUpperArm": "rightLowerArm", "leftUpperLeg": "leftLowerLeg", "rightUpperLeg": "rightLowerLeg"}


def _dominant(model: Model3D) -> list[torch.Tensor | None]:
    """The joint each vertex of each part follows most, or None for a part with no skin."""
    found = []
    for part in model.parts:
        if part.joints is None or part.weights is None or not part.vertex_count:
            found.append(None)
            continue
        found.append(part.joints.long().gather(1, part.weights.float().argmax(dim=1, keepdim=True)).squeeze(1))
    return found


def _head_parts(model: Model3D, head: int, dominant: list) -> list[int]:
    """Parts labelled head or hair, else the parts most of whose vertices follow the head bone."""
    labelled = [index for index, part in enumerate(model.parts) if part.label in ("head", "hair") and part.vertex_count]
    if labelled:
        return labelled
    return [index for index, joints in enumerate(dominant)
            if joints is not None and float((joints == head).double().mean()) > 0.5]


def _hair_faces(model: Model3D, indices: list[int], source: str, level: float) -> dict[int, torch.Tensor]:
    """Hair face masks for the given parts: whole parts labelled hair, colour on the rest when ``auto``."""
    masks: dict[int, torch.Tensor] = {}
    coloured = []
    for index in indices:
        part = model.parts[index]
        if part.label == "hair":
            masks[index] = torch.ones(part.triangle_count, dtype=torch.bool)
        elif source == "auto":
            coloured.append(index)
    if coloured:
        found = assemble.head_hair(Model3D(parts=[model.parts[index] for index in coloured]), level)
        if found is not None:
            for index, mask in zip(coloured, found):
                if mask is not None and bool(mask.any()):
                    masks[index] = mask
    return masks


def _strands(part, mask: torch.Tensor, hang: float, axis: torch.Tensor, neck: float,
             skin: torch.Tensor | None) -> list[torch.Tensor]:
    """Vertex sets of one part's hanging strands: hair by colour, or below the head joint, clear of the neck
    and not skin."""
    middle = part.vertices[part.faces].float().mean(dim=1)
    clear = (middle[:, [0, 2]].double() - axis).norm(dim=1) > NECK_CLEARANCE * neck
    colours = assemble._face_colours(part) if skin is not None else None
    if colours is not None:
        clear &= (colours - skin).norm(dim=1) > SKIN_DISTANCE
    hanging = (mask | clear) & (middle[:, 1] < hang)
    if not bool(hanging.any()):
        return []
    welded, faces, remap = topology.weld(part.vertices, part.faces, tolerance=0)
    chosen = torch.nonzero(hanging).squeeze(1)
    labels = topology.face_components(faces[chosen], welded.shape[0])
    strands = []
    for label in labels.unique().tolist():
        picked = chosen[labels == label]
        rows = torch.unique(part.faces[picked].reshape(-1)).long()
        offset = part.vertices[rows, :][:, [0, 2]].double() - axis
        angle = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0]))
        span = float(_span(angle))
        if span <= CURTAIN_SPAN:
            strands.append(rows)
            continue
        start = float(_start(angle))
        turned = torch.remainder(angle - start, 360.0)
        pieces = max(2, math.ceil(span / SECTOR))
        for piece in range(pieces):
            low, high = piece * span / pieces, (piece + 1) * span / pieces
            inside = (turned >= low) & (turned < high + (1e-6 if piece == pieces - 1 else 0.0))
            if bool(inside.any()):
                strands.append(rows[inside])
    return strands


def _span(angle: torch.Tensor) -> float:
    """Degrees round a circle a set of angles covers, across its widest gap."""
    ordered = torch.sort(torch.remainder(angle, 360.0)).values
    gaps = torch.cat([ordered[1:] - ordered[:-1], (ordered[:1] + 360.0) - ordered[-1:]])
    return 360.0 - float(gaps.max())


def _start(angle: torch.Tensor) -> float:
    """The angle a set of angles starts at, just past its widest gap."""
    ordered = torch.sort(torch.remainder(angle, 360.0)).values
    gaps = torch.cat([ordered[1:] - ordered[:-1], (ordered[:1] + 360.0) - ordered[-1:]])
    return float(ordered[(int(gaps.argmax()) + 1) % ordered.shape[0]])


def _chain_points(points: torch.Tensor, segments: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Joint positions down a hanging strand, each point's place along it, and its half thickness."""
    top, bottom = float(points[:, 1].max()), float(points[:, 1].min())
    along = ((top - points[:, 1]) / max(top - bottom, 1e-9)).clamp(0.0, 1.0)
    joints = []
    for step in range(segments + 1):
        place = step / segments
        band = (along - place).abs() <= 0.5 / segments
        if not bool(band.any()):
            band = (along - place).abs() <= (along - place).abs().min() + 1e-9
        joints.append(points[band].mean(dim=0))
    joints = torch.stack(joints)
    nearest = torch.cdist(points.double(), joints.double()).min(dim=1).values
    return joints, along, float(nearest.median())


def _collider_radius(model: Model3D, dominant: list, joint: int, start: torch.Tensor, end: torch.Tensor) -> float:
    """How far a bone's own surface sits from its segment, scaled by `COLLIDER_SHARE`."""
    distances = []
    direction = end - start
    length_sq = float(direction @ direction)
    for part, joints in zip(model.parts, dominant):
        if joints is None:
            continue
        rows = part.vertices[joints == joint].double()
        if not rows.shape[0]:
            continue
        along = ((rows - start) @ direction / max(length_sq, 1e-12)).clamp(0.0, 1.0)
        distances.append((rows - (start + along[:, None] * direction)).norm(dim=1))
    if not distances:
        return 0.0
    return COLLIDER_SHARE * float(torch.cat(distances).median())


def _colliders(model: Model3D, skeleton: Skeleton, dominant: list) -> tuple[list[Collider], list[str]]:
    """Colliders on the head, neck, chest and upper arms, in each bone's own space."""
    heads = skeleton.heads().double()
    names = {name: index for name, index in skeleton.humanoid.items()}
    children: dict[int, int] = {}
    for joint in range(skeleton.count):
        parent = int(skeleton.parents[joint])
        if parent >= 0 and parent not in children:
            children[parent] = joint
    colliders, placed = [], []
    for bone, shape in COLLIDER_BONES:
        joint = names.get(bone)
        if joint is None:
            continue
        child = names.get(COLLIDER_ENDS[bone]) if bone in COLLIDER_ENDS else None
        start = heads[joint]
        end = heads[child] if child is not None else start
        if shape == "sphere":
            rows = [part.vertices[joints == joint].double() for part, joints in zip(model.parts, dominant)
                    if joints is not None]
            rows = torch.cat(rows) if rows else torch.zeros(0, 3, dtype=torch.float64)
            if not rows.shape[0]:
                continue
            centre = rows.mean(dim=0)
            radius = COLLIDER_SHARE * float((rows - centre).norm(dim=1).median())
            start = end = centre
        else:
            radius = _collider_radius(model, dominant, joint, start, end)
        if radius <= 0:
            continue
        inverse = torch.linalg.inv(skeleton.rest[joint].double())
        local = lambda point: (inverse[:3, :3] @ point + inverse[:3, 3]).float()
        colliders.append(Collider(joint, shape, local(start), local(end), radius))
        placed.append(bone)
    return colliders, placed


def _cloth_plan(part, skeleton: Skeleton, segments: int) -> dict | None:
    """Chains round one cloth part: the bone it hangs from, its stretches by angle and their joints."""
    points = part.vertices.double()
    heads = skeleton.heads().double()
    top = float(points[:, 1].max())
    roots = [skeleton.humanoid[name] for name in CLOTH_ROOTS if name in skeleton.humanoid]
    if not roots:
        return None
    below = [joint for joint in roots if float(heads[joint][1]) <= top] or roots[:1]
    root = max(below, key=lambda joint: float(heads[joint][1]))
    axis = heads[root][[0, 2]]
    offset = points[:, [0, 2]] - axis
    angle = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0]))
    span = _span(angle)
    cyclic = span > 360.0 - CLOTH_SECTOR
    if cyclic:
        count, start = max(3, round(360.0 / CLOTH_SECTOR)), 0.0
        width = 360.0 / count
    else:
        count, start = max(1, math.ceil(span / CLOTH_SECTOR)), _start(angle)
        width = max(span / count, 1e-6)
    turned = torch.remainder(angle - start, 360.0)
    sector = (turned / width).floor().long().clamp(0, count - 1)
    stretches = []
    for number in range(count):
        rows = torch.nonzero(sector == number).squeeze(1)
        if rows.shape[0] < CLOTH_FEWEST:
            stretches.append(None)
            continue
        joints, _, half = _chain_points(points[rows], segments)
        stretches.append({"joints": joints, "half": half, "top": float(points[rows, 1].max()),
                          "bottom": float(points[rows, 1].min())})
    if all(stretch is None for stretch in stretches):
        return None
    return {"root": root, "axis": axis, "start": start, "width": width, "count": count, "cyclic": cyclic,
            "stretches": stretches}


def _nearest_stretch(stretches: list, number: int, cyclic: bool) -> int:
    """The nearest stretch with a chain of its own to stretch ``number``."""
    count = len(stretches)
    for distance in range(count):
        for step in (number - distance, number + distance):
            index = step % count if cyclic else min(max(step, 0), count - 1)
            if stretches[index] is not None:
                return index
    return number


def _cloth_skin(part, plan: dict, firsts: list[int], segments: int) -> None:
    """Skin a cloth part between the two nearest chains round it and along each, easing into its root."""
    points = part.vertices.double()
    offset = points[:, [0, 2]] - plan["axis"]
    angle = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0]))
    place = torch.remainder(angle - plan["start"], 360.0) / plan["width"] - 0.5
    count = plan["count"]
    if plan["cyclic"]:
        low = place.floor()
        share = (place - low).float()
        near = [torch.remainder(low.long(), count), torch.remainder(low.long() + 1, count)]
    else:
        place = place.clamp(0.0, count - 1)
        low = place.floor()
        share = (place - low).float()
        near = [low.long(), (low.long() + 1).clamp_max(count - 1)]
    owner = torch.tensor([_nearest_stretch(plan["stretches"], number, plan["cyclic"]) for number in range(count)])
    tops = torch.tensor([plan["stretches"][index]["top"] for index in owner.tolist()], dtype=torch.float64)
    bottoms = torch.tensor([plan["stretches"][index]["bottom"] for index in owner.tolist()], dtype=torch.float64)
    first_of = torch.tensor([firsts[index] for index in owner.tolist()])
    segment_ids = torch.arange(-1, segments, dtype=torch.float32)
    columns, values = [], []
    root_weight = torch.zeros(points.shape[0])
    for side, mix in ((near[0], 1.0 - share), (near[1], share)):
        along = ((tops[side] - points[:, 1]) / (tops[side] - bottoms[side]).clamp_min(1e-9)).clamp(0.0, 1.0)
        tent = (1.0 - (along.float()[:, None] * segments - (segment_ids[None, :] + 0.5)).abs()).clamp_min(0.0)
        tent = tent / tent.sum(dim=1, keepdim=True).clamp_min(1e-12)
        root_weight += tent[:, 0] * mix
        columns.append(first_of[side][:, None] + torch.arange(segments)[None, :])
        values.append(tent[:, 1:] * mix[:, None])
    columns = torch.cat([torch.full((points.shape[0], 1), plan["root"], dtype=torch.long)] + columns, dim=1)
    values = torch.cat([root_weight[:, None]] + values, dim=1)
    strongest, which = values.topk(min(4, values.shape[1]), dim=1)
    width = part.joints.shape[1]
    take = min(width, strongest.shape[1])
    joints = torch.zeros_like(part.joints, dtype=torch.long)
    weights = torch.zeros_like(part.weights, dtype=torch.float32)
    joints[:, :take] = columns.gather(1, which[:, :take])
    weights[:, :take] = strongest[:, :take] / strongest[:, :take].sum(dim=1, keepdim=True).clamp_min(1e-12)
    part.joints, part.weights = joints, weights


def _extend(skeleton: Skeleton, positions: list[torch.Tensor], parents: list[int], names: list[str]) -> Skeleton:
    """A skeleton with joints added at ``positions``, unturned, under the given parents."""
    count = len(positions)
    rest = torch.eye(4, dtype=skeleton.rest.dtype).repeat(count, 1, 1)
    rest[:, :3, 3] = torch.stack(positions).to(skeleton.rest.dtype)
    return Skeleton(
        names=list(skeleton.names) + names,
        parents=torch.cat([skeleton.parents.long(), torch.tensor(parents, dtype=torch.long)]),
        rest=torch.cat([skeleton.rest, rest]),
        inverse_bind=torch.cat([skeleton.inverse_bind, torch.linalg.inv(rest)]),
        humanoid=dict(skeleton.humanoid),
    )


def add_springs(model: Model3D, hair: str = "auto", joints_per_chain: int = 4, max_chains: int = 16,
                stiffness: float = 0.8, drag: float = 0.4, gravity: float = 0.25,
                colliders: bool = True, cloth: str = "") -> tuple[Model3D, dict]:
    """Give a rigged character spring chains for its hanging hair and named cloth parts, in place.

    Args:
        model: A model with a humanoid skeleton and skin.
        hair: One of `HAIR_SOURCES`.
        joints_per_chain: Segments in each chain; the chain holds one more joint than this.
        max_chains: Most chains made; the largest strands are kept.
        stiffness: Pull of each chain back to its rest shape.
        drag: Share of its speed a chain loses each step, 0 to 1.
        gravity: Strength of gravity on each chain.
        colliders: Put colliders on the head, neck, chest, upper arms, hips and upper legs.
        cloth: Comma-separated names or labels of parts that swing like cloth, wildcards allowed;
            empty for none.

    Returns:
        ``(model, figures)``.

    Raises:
        ValueError: The model has no humanoid head bone or no skin, or already has hair chains.
    """
    skeleton = model.skeleton
    if skeleton is not None and "head" not in skeleton.humanoid:
        try:
            skeleton.humanoid = bones.guess_humanoid(skeleton.names, skeleton.parents, skeleton.heads())
        except ValueError:
            pass
    if skeleton is None or "head" not in skeleton.humanoid:
        raise ValueError("the model has no humanoid skeleton with a head bone; rig it first with Character Rig "
                         "or Tripo's Rig model on vrm")
    if any(name.startswith(PREFIX) or name.startswith(CLOTH_PREFIX) for name in skeleton.names):
        raise ValueError("the model already has hair chains from Character Springs; wire the model from before "
                         "that node")
    dominant = _dominant(model)
    if all(joints is None for joints in dominant):
        raise ValueError("the model has a skeleton but no skin; rig it with Character Rig first")
    head = skeleton.humanoid["head"]
    heads = skeleton.heads().double()
    neck = skeleton.humanoid.get("neck", head)
    hang = float(heads[head][1])
    axis = heads[head][[0, 2]]
    top = max(float(part.vertices[:, 1].max()) for part in model.parts if part.vertex_count)
    head_height = max(top - float(heads[neck][1]), 1e-6)
    neck_radius = _collider_radius(model, dominant, neck, heads[neck], heads[head]) / COLLIDER_SHARE
    neck_radius = neck_radius if neck_radius > 0 else 0.1 * head_height
    figures = {"chains": 0, "joints added": 0, "swinging vertices": 0, "colliders": 0, "notes": []}
    strands = []
    if hair != "off":
        indices = _head_parts(model, head, dominant)
        masks = _hair_faces(model, indices, hair, float(heads[neck][1]))
        if hair == "auto":
            for index in indices:
                masks.setdefault(index, torch.zeros(model.parts[index].triangle_count, dtype=torch.bool))
        if not masks:
            figures["notes"].append("no hair was found; label the hair part `hair` with 3D Model Parts")
        samples = assemble._face_skin(Model3D(parts=[model.parts[index] for index in indices]), float(heads[neck][1]))
        skin = samples.median(dim=0).values if samples.shape[0] >= 16 else None
        for index, mask in masks.items():
            part = model.parts[index]
            for rows in _strands(part, mask, hang, axis, neck_radius, skin):
                points = part.vertices[rows].double()
                if float(points[:, 1].max() - points[:, 1].min()) < SHORTEST_STRAND * head_height:
                    continue
                strands.append((index, rows))
    if strands:
        sizes = torch.tensor([float(rows.shape[0]) for _, rows in strands])
        keep = sizes >= SMALLEST_STRAND * float(sizes.sum())
        strands = [strand for strand, kept in zip(strands, keep.tolist()) if kept]
        strands.sort(key=lambda strand: -strand[1].shape[0])
        if len(strands) > max_chains:
            figures["notes"].append(f"{len(strands)} strands were found; the largest {max_chains} swing")
            strands = strands[:max_chains]
    segments = max(2, int(joints_per_chain))
    positions, parents, names, chains, spans = [], [], [], [], []
    base = skeleton.count
    for number, (index, rows) in enumerate(strands):
        points = model.parts[index].vertices[rows].double()
        joints, along, half = _chain_points(points, segments)
        first = base + len(positions)
        for step in range(segments + 1):
            positions.append(joints[step])
            parents.append(head if step == 0 else first + step - 1)
            names.append(f"{PREFIX}{number}_{step}")
        hit = min(max(HIT_SHARE * half, HIT_RANGE[0]), HIT_RANGE[1])
        chains.append(SpringChain(name=f"hair_{number}", joints=list(range(first, first + segments + 1)),
                                  stiffness=float(stiffness), drag=float(drag), gravity_power=float(gravity),
                                  hit_radius=hit))
        spans.append((index, rows, along, first))
    cloth_plans = []
    if str(cloth or "").strip():
        hair_parts = {index for index, _ in strands}
        for index in selection.selected(model, cloth):
            part = model.parts[index]
            if index in hair_parts or not part.vertex_count or part.joints is None:
                continue
            plan = _cloth_plan(part, skeleton, segments)
            if plan is None:
                figures["notes"].append(f"{part.name or part.label} has no torso bone above it to hang from")
                continue
            firsts = []
            for number, stretch in enumerate(plan["stretches"]):
                if stretch is None:
                    firsts.append(-1)
                    continue
                first = base + len(positions)
                firsts.append(first)
                for step in range(segments + 1):
                    positions.append(stretch["joints"][step])
                    parents.append(plan["root"] if step == 0 else first + step - 1)
                    names.append(f"{CLOTH_PREFIX}{len(cloth_plans)}_{number}_{step}")
                hit = min(max(HIT_SHARE * stretch["half"], HIT_RANGE[0]), CLOTH_HIT)
                chains.append(SpringChain(name=f"cloth_{len(cloth_plans)}_{number}",
                                          joints=list(range(first, first + segments + 1)),
                                          stiffness=float(stiffness), drag=float(drag),
                                          gravity_power=float(gravity), hit_radius=hit))
            cloth_plans.append((index, plan, firsts))
        if not cloth_plans:
            figures["notes"].append(f"no part matching '{str(cloth)[:60]}' could hang as cloth")
    if positions:
        model.skeleton = skeleton = _extend(skeleton, positions, parents, names)
    swinging = 0
    for index, rows, along, first in spans:
        part = model.parts[index]
        place = along.float() * segments
        segment_ids = torch.arange(-1, segments, dtype=torch.float32)
        tent = (1.0 - (place[:, None] - (segment_ids[None, :] + 0.5)).abs()).clamp_min(0.0)
        tent = tent / tent.sum(dim=1, keepdim=True).clamp_min(1e-12)
        joint_ids = torch.cat([torch.tensor([head]), torch.arange(first, first + segments)])
        strongest, which = tent.topk(min(4, tent.shape[1]), dim=1)
        strongest = strongest / strongest.sum(dim=1, keepdim=True).clamp_min(1e-12)
        joints = part.joints.clone().long()
        weights = part.weights.clone().float()
        width = joints.shape[1]
        joints[rows] = 0
        weights[rows] = 0.0
        take = min(width, strongest.shape[1])
        joints[rows, :take] = joint_ids[which[:, :take]]
        weights[rows, :take] = strongest[:, :take] / strongest[:, :take].sum(dim=1, keepdim=True).clamp_min(1e-12)
        part.joints, part.weights = joints, weights
        swinging += int(rows.shape[0])
    for index, plan, firsts in cloth_plans:
        _cloth_skin(model.parts[index], plan, firsts, segments)
        swinging += model.parts[index].vertex_count
    springs = model.springs if model.springs is not None else Springs()
    if colliders and springs.collider_groups:
        for chain in chains:
            chain.collider_groups = list(range(len(springs.collider_groups)))
        figures["notes"].append("the model's own colliders were kept and the new chains collide with them")
    elif colliders:
        placed_colliders, placed = _colliders(model, skeleton, _dominant(model))
        if placed_colliders:
            first_collider = len(springs.colliders)
            springs.colliders.extend(placed_colliders)
            springs.collider_groups.append(list(range(first_collider, first_collider + len(placed_colliders))))
            springs.group_names.append("body")
            group = len(springs.collider_groups) - 1
            for chain in chains:
                chain.collider_groups = [group]
            figures["colliders"] = len(placed_colliders)
            figures["collider bones"] = ", ".join(placed)
    settings = model.extras.get("vrm_spring_settings")
    if isinstance(settings, list) and len(settings) == len(springs.chains):
        # Each new chain gets a neutral entry, so the settings stay paired with their chains.
        settings.extend({"center": -1, "joints": None} for _ in chains)
    springs.chains.extend(chains)
    if chains or springs.colliders:
        model.springs = springs
    figures.update({"chains": len(chains), "joints added": len(positions), "swinging vertices": swinging})
    return model, figures
