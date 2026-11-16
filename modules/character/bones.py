"""VRM humanoid bones, name maps from common rigs and a guesser that labels any skeleton.

Joint heads are metres, Y-up. T-pose directions are unit vectors for a character facing +Z with
its left on +X.
"""

from __future__ import annotations

__all__ = [
    "BONE_INFO", "HEADING_ONLY", "HUMANOID_BONES", "HumanBone", "MAX_GUESS_JOINTS",
    "REQUIRED_BONES", "REQUIRED_BONES_VRM0", "RIG_MAPS", "TPOSE_DIRECTIONS", "from_vrm0",
    "guess_humanoid", "map_by_name", "mirror", "to_vrm0", "tpose_directions",
    "validate_humanoid",
]

import math
import re
from dataclasses import dataclass

from ..log import get_logger

logger = get_logger("character.bones")

MAX_GUESS_JOINTS = 4096

_SIDES = (("left", "L", "Left", "l"), ("right", "R", "Right", "r"))
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


@dataclass(frozen=True)
class HumanBone:
    """One VRM humanoid bone.

    Attributes:
        name: VRM 1.0 bone name.
        parent: VRM 1.0 parent bone, ``None`` for ``hips``.
        required: Required by VRM 1.0.
        required_vrm0: Required by VRM 0.x.
        needs_parent: The bone may only exist when its parent bone exists.
        vrm0: VRM 0.x name of the same bone.
    """

    name: str
    parent: str | None
    required: bool
    required_vrm0: bool
    needs_parent: bool
    vrm0: str


def _segments(finger: str) -> tuple[str, str, str]:
    """VRM 1.0 segment names of a finger, root first."""
    if finger == "Thumb":
        return ("Metacarpal", "Proximal", "Distal")
    return ("Proximal", "Intermediate", "Distal")


def _build_bones() -> tuple[HumanBone, ...]:
    """Every VRM 1.0 humanoid bone in parent-first order."""
    rows = [
        ("hips", None, True, True, False),
        ("spine", "hips", True, True, False),
        ("chest", "spine", False, True, False),
        ("upperChest", "chest", False, False, True),
        ("neck", "upperChest", False, True, False),
        ("head", "neck", True, True, False),
        ("leftEye", "head", False, False, False),
        ("rightEye", "head", False, False, False),
        ("jaw", "head", False, False, False),
    ]
    for side, *_ in _SIDES:
        rows += [
            (f"{side}UpperLeg", "hips", True, True, False),
            (f"{side}LowerLeg", f"{side}UpperLeg", True, True, False),
            (f"{side}Foot", f"{side}LowerLeg", True, True, False),
            (f"{side}Toes", f"{side}Foot", False, False, False),
        ]
    for side, *_ in _SIDES:
        rows += [
            (f"{side}Shoulder", "upperChest", False, False, False),
            (f"{side}UpperArm", f"{side}Shoulder", True, True, False),
            (f"{side}LowerArm", f"{side}UpperArm", True, True, False),
            (f"{side}Hand", f"{side}LowerArm", True, True, False),
        ]
    for side, *_ in _SIDES:
        for finger in _FINGERS:
            parent = f"{side}Hand"
            for i, segment in enumerate(_segments(finger)):
                name = f"{side}{finger}{segment}"
                rows.append((name, parent, False, False, i > 0))
                parent = name
    bones = []
    for name, parent, required, required_vrm0, needs_parent in rows:
        vrm0 = name
        # Thumbs take their VRM 0.x Proximal, Intermediate and Distal names.
        if "Thumb" in name:
            vrm0 = name.replace("Proximal", "Intermediate").replace("Metacarpal", "Proximal")
        bones.append(HumanBone(name, parent, required, required_vrm0, needs_parent, vrm0))
    return tuple(bones)


HUMANOID_BONES: tuple[HumanBone, ...] = _build_bones()
BONE_INFO: dict[str, HumanBone] = {bone.name: bone for bone in HUMANOID_BONES}
REQUIRED_BONES: tuple[str, ...] = tuple(b.name for b in HUMANOID_BONES if b.required)
REQUIRED_BONES_VRM0: tuple[str, ...] = tuple(b.name for b in HUMANOID_BONES if b.required_vrm0)
_ORDER = {bone.name: i for i, bone in enumerate(HUMANOID_BONES)}
_FROM_VRM0 = {bone.vrm0: bone.name for bone in HUMANOID_BONES}


def to_vrm0(bone: str) -> str:
    """VRM 0.x name of a VRM 1.0 bone.

    Args:
        bone: VRM 1.0 bone name.

    Returns:
        The VRM 0.x name, or ``bone`` unchanged when it is not a humanoid bone.
    """
    info = BONE_INFO.get(bone)
    return info.vrm0 if info else bone


def from_vrm0(bone: str) -> str:
    """VRM 1.0 name of a VRM 0.x bone.

    Args:
        bone: VRM 0.x bone name.

    Returns:
        The VRM 1.0 name, or ``bone`` unchanged when it is not a humanoid bone.
    """
    return _FROM_VRM0.get(bone, bone)


def mirror(bone: str) -> str:
    """The same bone on the other side.

    Args:
        bone: VRM bone name.

    Returns:
        ``leftX`` for ``rightX`` and the reverse, or ``bone`` for a centre bone.
    """
    if bone.startswith("left"):
        return "right" + bone[4:]
    if bone.startswith("right"):
        return "left" + bone[5:]
    return bone


# Rig name maps: joint name to VRM bone, None for a known joint that is not humanoid.

def _finger_map(m: dict, side: str, pattern: str, names: tuple[str, ...]) -> None:
    """Add three segments for each finger, ``pattern`` formatted with finger and 1-based index."""
    for finger, label in zip(_FINGERS, names):
        for i, segment in enumerate(_segments(finger), 1):
            m[pattern.format(f=label, i=i)] = f"{side}{finger}{segment}"


def _tripo_map() -> dict[str, str | None]:
    """Tripo native rig, which is the Character Creator rig without its prefix."""
    m: dict[str, str | None] = {
        "BoneRoot": None, "Hip": "hips", "Pelvis": None, "Waist": "spine", "Spine01": "chest",
        "Spine02": "upperChest", "NeckTwist01": "neck", "NeckTwist02": None, "Head": "head",
        "FacialBone": None, "JawRoot": "jaw",
    }
    for side, s, _, _ in _SIDES:
        m.update({
            f"{s}_Eye": f"{side}Eye", f"{s}_Clavicle": f"{side}Shoulder",
            f"{s}_Upperarm": f"{side}UpperArm", f"{s}_Forearm": f"{side}LowerArm",
            f"{s}_Hand": f"{side}Hand", f"{s}_Thigh": f"{side}UpperLeg",
            f"{s}_Calf": f"{side}LowerLeg", f"{s}_Foot": f"{side}Foot",
            f"{s}_ToeBase": f"{side}Toes",
        })
        _finger_map(m, side, s + "_{f}{i}", ("Thumb", "Index", "Mid", "Ring", "Pinky"))
        for n, finger in enumerate(_FINGERS):
            for i, segment in enumerate(_segments(finger)):
                m[f"{s}_Finger{n}{i}"] = f"{side}{finger}{segment}"
        for extra in ("UpperarmTwist01", "UpperarmTwist02", "ForearmTwist01", "ForearmTwist02",
                      "ElbowShareBone", "ThighTwist01", "ThighTwist02", "CalfTwist01",
                      "CalfTwist02", "KneeShareBone"):
            m[f"{s}_{extra}"] = None
    return m


def _mixamo_map() -> dict[str, str | None]:
    """Mixamo rig, keys without the ``mixamorig:`` prefix."""
    m: dict[str, str | None] = {
        "Hips": "hips", "Spine": "spine", "Spine1": "chest", "Spine2": "upperChest",
        "Neck": "neck", "Head": "head", "HeadTop_End": None,
    }
    for side, _, S, _ in _SIDES:
        m.update({
            f"{S}Eye": f"{side}Eye", f"{S}Shoulder": f"{side}Shoulder",
            f"{S}Arm": f"{side}UpperArm", f"{S}ForeArm": f"{side}LowerArm",
            f"{S}Hand": f"{side}Hand", f"{S}UpLeg": f"{side}UpperLeg",
            f"{S}Leg": f"{side}LowerLeg", f"{S}Foot": f"{side}Foot",
            f"{S}ToeBase": f"{side}Toes", f"{S}Toe_End": None,
        })
        _finger_map(m, side, S + "Hand{f}{i}", ("Thumb", "Index", "Middle", "Ring", "Pinky"))
        for label in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            m[f"{S}Hand{label}4"] = None
    return m


def _humanik_map() -> dict[str, str | None]:
    """Autodesk HumanIK rig."""
    m = _mixamo_map()
    m["Reference"] = None
    for i in range(3, 10):
        m[f"Spine{i}"] = None
    for i in range(1, 10):
        m[f"Neck{i}"] = None
    for _, _, S, _ in _SIDES:
        for extra in ("ArmRoll", "ForeArmRoll", "UpLegRoll", "LegRoll", "InHandThumb",
                      "InHandIndex", "InHandMiddle", "InHandRing", "InHandPinky", "FingerBase",
                      "HandIndex0", "HandMiddle0", "HandRing0", "HandPinky0"):
            m[f"{S}{extra}"] = None
    return m


def _vroid_map() -> dict[str, str | None]:
    """VRoid Studio and UniVRM ``J_Bip`` rig."""
    m: dict[str, str | None] = {
        "Root": None, "J_Bip_C_Hips": "hips", "J_Bip_C_Spine": "spine",
        "J_Bip_C_Chest": "chest", "J_Bip_C_UpperChest": "upperChest",
        "J_Bip_C_Neck": "neck", "J_Bip_C_Head": "head",
    }
    for side, s, _, _ in _SIDES:
        m.update({
            f"J_Adj_{s}_FaceEye": f"{side}Eye", f"J_Bip_{s}_Shoulder": f"{side}Shoulder",
            f"J_Bip_{s}_UpperArm": f"{side}UpperArm", f"J_Bip_{s}_LowerArm": f"{side}LowerArm",
            f"J_Bip_{s}_Hand": f"{side}Hand", f"J_Bip_{s}_UpperLeg": f"{side}UpperLeg",
            f"J_Bip_{s}_LowerLeg": f"{side}LowerLeg", f"J_Bip_{s}_Foot": f"{side}Foot",
            f"J_Bip_{s}_ToeBase": f"{side}Toes",
        })
        _finger_map(m, side, "J_Bip_" + s + "_{f}{i}", _FINGERS)
    return m


def _unreal_map(ue5: bool) -> dict[str, str | None]:
    """Unreal Engine mannequin, UE4 or UE5, also the MakeHuman game engine rig."""
    m: dict[str, str | None] = {
        "root": None, "pelvis": "hips", "spine_01": "spine", "neck_01": "neck", "head": "head",
        "ik_foot_root": None, "ik_hand_root": None, "ik_hand_gun": None,
    }
    if ue5:
        m.update({"spine_02": None, "spine_03": "chest", "spine_04": None,
                  "spine_05": "upperChest", "neck_02": None})
    else:
        m.update({"spine_02": "chest", "spine_03": "upperChest"})
    for side, _, _, s in _SIDES:
        m.update({
            f"clavicle_{s}": f"{side}Shoulder", f"upperarm_{s}": f"{side}UpperArm",
            f"lowerarm_{s}": f"{side}LowerArm", f"hand_{s}": f"{side}Hand",
            f"thigh_{s}": f"{side}UpperLeg", f"calf_{s}": f"{side}LowerLeg",
            f"foot_{s}": f"{side}Foot", f"ball_{s}": f"{side}Toes",
            f"ik_foot_{s}": None, f"ik_hand_{s}": None,
        })
        _finger_map(m, side, "{f}_0{i}_" + s, ("thumb", "index", "middle", "ring", "pinky"))
        for i in ("01", "02") if ue5 else ("01",):
            for limb in ("upperarm", "lowerarm", "thigh", "calf"):
                m[f"{limb}_twist_{i}_{s}"] = None
        if ue5:
            for label in ("index", "middle", "ring", "pinky"):
                m[f"{label}_metacarpal_{s}"] = None
    return m


def _makehuman_map() -> dict[str, str | None]:
    """MakeHuman and MPFB default rig."""
    m: dict[str, str | None] = {
        "root": "hips", "spine05": "spine", "spine04": None, "spine03": "chest",
        "spine02": None, "spine01": "upperChest", "neck01": "neck", "neck02": None,
        "neck03": None, "head": "head", "jaw": "jaw",
    }
    for side, s, _, _ in _SIDES:
        m.update({
            f"eye.{s}": f"{side}Eye", f"clavicle.{s}": f"{side}Shoulder",
            f"shoulder01.{s}": None, f"upperarm01.{s}": f"{side}UpperArm",
            f"upperarm02.{s}": None, f"lowerarm01.{s}": f"{side}LowerArm",
            f"lowerarm02.{s}": None, f"wrist.{s}": f"{side}Hand", f"pelvis.{s}": None,
            f"upperleg01.{s}": f"{side}UpperLeg", f"upperleg02.{s}": None,
            f"lowerleg01.{s}": f"{side}LowerLeg", f"lowerleg02.{s}": None,
            f"foot.{s}": f"{side}Foot", f"toe1-1.{s}": f"{side}Toes", f"breast.{s}": None,
        })
        _finger_map(m, side, "finger{f}-{i}." + s, ("1", "2", "3", "4", "5"))
        for i in range(1, 5):
            m[f"metacarpal{i}.{s}"] = None
    return m


def _rigify_map() -> dict[str, str | None]:
    """Blender Rigify deform bones and metarig bones."""
    m: dict[str, str | None] = {
        "DEF-spine": "hips", "DEF-spine.001": "spine", "DEF-spine.002": "chest",
        "DEF-spine.003": "upperChest", "DEF-spine.004": "neck", "DEF-spine.005": None,
        "DEF-spine.006": "head", "DEF-jaw": "jaw",
    }
    for side, s, _, _ in _SIDES:
        m.update({
            f"DEF-shoulder.{s}": f"{side}Shoulder", f"DEF-upper_arm.{s}": f"{side}UpperArm",
            f"DEF-upper_arm.{s}.001": None, f"DEF-forearm.{s}": f"{side}LowerArm",
            f"DEF-forearm.{s}.001": None, f"DEF-hand.{s}": f"{side}Hand",
            f"DEF-thigh.{s}": f"{side}UpperLeg", f"DEF-thigh.{s}.001": None,
            f"DEF-shin.{s}": f"{side}LowerLeg", f"DEF-shin.{s}.001": None,
            f"DEF-foot.{s}": f"{side}Foot", f"DEF-toe.{s}": f"{side}Toes",
            f"DEF-pelvis.{s}": None, f"DEF-breast.{s}": None,
        })
        _finger_map(m, side, "DEF-{f}.0{i}." + s,
                    ("thumb", "f_index", "f_middle", "f_ring", "f_pinky"))
        for i in range(1, 5):
            m[f"DEF-palm.0{i}.{s}"] = None
    # Metarig names: deform names without ``DEF-``, except bare ``spine`` and deform-only segments.
    for key, bone in list(m.items()):
        if key != "DEF-spine" and not re.search(r"\.[LR]\.\d{3}$", key):
            m[key[4:]] = bone
    for side, s, _, _ in _SIDES:
        m[f"eye.{s}"] = f"{side}Eye"
    return m


def _biped_map() -> dict[str, str | None]:
    """Autodesk 3ds Max Biped rig, any ``Bip`` number."""
    m: dict[str, str | None] = {
        "Bip01": None, "Bip01 Pelvis": "hips", "Bip01 Spine": "spine",
        "Bip01 Spine1": "chest", "Bip01 Spine2": "upperChest", "Bip01 Neck": "neck",
        "Bip01 Head": "head", "Bip01 HeadNub": None,
    }
    for side, s, _, _ in _SIDES:
        m.update({
            f"Bip01 {s} Clavicle": f"{side}Shoulder", f"Bip01 {s} UpperArm": f"{side}UpperArm",
            f"Bip01 {s} Forearm": f"{side}LowerArm", f"Bip01 {s} Hand": f"{side}Hand",
            f"Bip01 {s} Thigh": f"{side}UpperLeg", f"Bip01 {s} Calf": f"{side}LowerLeg",
            f"Bip01 {s} Foot": f"{side}Foot", f"Bip01 {s} Toe0": f"{side}Toes",
            f"Bip01 {s} Toe0Nub": None,
        })
        for n, finger in enumerate(_FINGERS):
            for i, segment in enumerate(_segments(finger)):
                m[f"Bip01 {s} Finger{n}{i if i else ''}"] = f"{side}{finger}{segment}"
            m[f"Bip01 {s} Finger{n}Nub"] = None
    return m


def _vrm_names_map(vrm0_thumbs: bool) -> dict[str, str | None]:
    """VRM and Unity humanoid names, either thumb convention."""
    m: dict[str, str | None] = {}
    for bone in HUMANOID_BONES:
        m[bone.vrm0 if vrm0_thumbs else bone.name] = bone.name
    return m


RIG_MAPS: dict[str, dict[str, str | None]] = {
    "tripo": _tripo_map(),
    "character_creator": {"CC_Base_" + k: v for k, v in _tripo_map().items()},
    "mixamo": _mixamo_map(),
    "humanik": _humanik_map(),
    "vroid": _vroid_map(),
    "unreal": _unreal_map(False),
    "unreal5": _unreal_map(True),
    "makehuman": _makehuman_map(),
    "rigify": _rigify_map(),
    "biped": _biped_map(),
    "vrm": _vrm_names_map(False),
    "vrm0": _vrm_names_map(True),
}

_MIXAMO_PREFIX = re.compile(r"^mixamorig\d*_", re.IGNORECASE)
_BIPED_NUMBER = re.compile(r"^bip\d+(?=_|$)")
_LOOSE_RIGS = frozenset(("vrm", "vrm0"))
# Fewest required VRM bones a rig table places to be chosen.
_MIN_REQUIRED_HITS = 8


def _key(name: str) -> str:
    """Name with namespace, Mixamo prefix and Biped number removed, lower case, ``_`` for spaces."""
    base = re.split(r"[:|]", name.strip())[-1]
    key = _MIXAMO_PREFIX.sub("", base).strip().lower().replace(" ", "_")
    return _BIPED_NUMBER.sub("bip", key)


def _loose(name: str) -> str:
    """Name reduced to lower case letters and digits."""
    return re.sub(r"[^a-z0-9]", "", _key(name))


_TABLES: dict[str, dict[str, str | None]] = {}


def _table(rig: str) -> dict[str, str | None]:
    """Lookup table of a rig keyed by normalised joint name."""
    if rig not in _TABLES:
        fold = _loose if rig in _LOOSE_RIGS else _key
        _TABLES[rig] = {fold(k): v for k, v in RIG_MAPS[rig].items()}
    return _TABLES[rig]


def map_by_name(names: list[str]) -> tuple[str, dict[str, int]]:
    """Map joints to VRM bones with the rig table that matches the most names.

    Args:
        names: Joint names.

    Returns:
        The rig name and a VRM bone to joint index map, or ``""`` and ``{}`` when no table places
        at least eight of the required bones.
    """
    keys = [_key(n) for n in names]
    loose = [_loose(n) for n in names]
    best, best_score, best_hits = "", 0, {}
    for rig in RIG_MAPS:
        table = _table(rig)
        folded = loose if rig in _LOOSE_RIGS else keys
        score, hits = 0, {}
        for joint, key in enumerate(folded):
            if key in table:
                score += 1
                bone = table[key]
                if bone is not None and bone not in hits:
                    hits[bone] = joint
        covered = sum(1 for bone in REQUIRED_BONES if bone in hits)
        if covered >= _MIN_REQUIRED_HITS and score > best_score:
            best, best_score, best_hits = rig, score, hits
    return best, best_hits


def _known_none(names: list[str], rig: str) -> set[int]:
    """Joints the rig table names as not humanoid."""
    if not rig:
        return set()
    table = _table(rig)
    fold = _loose if rig in _LOOSE_RIGS else _key
    return {j for j, n in enumerate(names) if fold(n) in table and table[fold(n)] is None}


class _Tree:
    """Joint hierarchy with depth, preorder and subtree lookups."""

    def __init__(self, parents: list[int]):
        n = len(parents)
        self.count = n
        self.parent = [p if 0 <= p < n and p != i else -1 for i, p in enumerate(parents)]
        self.children: list[list[int]] = [[] for _ in range(n)]
        for joint, parent in enumerate(self.parent):
            if parent >= 0:
                self.children[parent].append(joint)
        self.depth = [-1] * n
        self.pre = [-1] * n
        self.size = [0] * n
        self.order: list[int] = []
        for root in range(n):
            if self.parent[root] >= 0:
                continue
            self.depth[root] = 0
            stack = [root]
            while stack:
                joint = stack.pop()
                self.pre[joint] = len(self.order)
                self.order.append(joint)
                for child in reversed(self.children[joint]):
                    self.depth[child] = self.depth[joint] + 1
                    stack.append(child)
        # Subtree sizes accumulate from the deepest preorder entries upward.
        for joint in reversed(self.order):
            self.size[joint] += 1
            if self.parent[joint] >= 0:
                self.size[self.parent[joint]] += self.size[joint]

    def reached(self, joint: int) -> bool:
        """Whether the joint hangs from a root."""
        return self.depth[joint] >= 0

    def above(self, a: int, b: int) -> bool:
        """Whether ``a`` is ``b`` or one of its ancestors."""
        if not (self.reached(a) and self.reached(b)):
            return False
        return self.pre[a] <= self.pre[b] < self.pre[a] + self.size[a]

    def subtree(self, joint: int) -> list[int]:
        """The joint and all its descendants."""
        return self.order[self.pre[joint]:self.pre[joint] + self.size[joint]]

    def lca(self, a: int, b: int) -> int:
        """Lowest common ancestor, or ``-1`` across separate trees."""
        if not (self.reached(a) and self.reached(b)):
            return -1
        while self.depth[a] > self.depth[b]:
            a = self.parent[a]
        while self.depth[b] > self.depth[a]:
            b = self.parent[b]
        while a != b:
            a, b = self.parent[a], self.parent[b]
        return a

    def path(self, top: int, bottom: int) -> list[int]:
        """Joints below ``top`` down to ``bottom``, top first."""
        out = []
        while bottom != top and bottom >= 0:
            out.append(bottom)
            bottom = self.parent[bottom]
        return out[::-1] if bottom == top else []

    def nearest(self, joint: int, marked: dict[int, str]) -> int:
        """Closest strict ancestor that is in ``marked``, or ``-1``."""
        joint = self.parent[joint]
        while joint >= 0 and joint not in marked:
            joint = self.parent[joint]
        return joint


def _effective_parent(bone: str, assigned: dict[str, int]) -> str | None:
    """Closest assigned VRM ancestor of a bone."""
    parent = BONE_INFO[bone].parent
    while parent is not None and parent not in assigned:
        parent = BONE_INFO[parent].parent
    return parent


def _compact(mapping: dict[str, int]) -> dict[str, int]:
    """Move bones down onto missing parents that a child bone needs."""
    out = dict(mapping)
    if "upperChest" in out and "chest" not in out:
        out["chest"] = out.pop("upperChest")
    for side, *_ in _SIDES:
        for finger in _FINGERS:
            names = [f"{side}{finger}{s}" for s in _segments(finger)]
            joints = [out.pop(n) for n in names if n in out]
            for name, joint in zip(names, joints):
                out[name] = joint
    return out


def validate_humanoid(mapping: dict[str, int], parents) -> dict[str, int]:
    """Keep the bones of a mapping that form a valid VRM humanoid hierarchy.

    Args:
        mapping: VRM bone name to joint index.
        parents: ``(J,)`` parent index of each joint, ``-1`` for a root.

    Returns:
        A new mapping in VRM bone order. Unknown names, repeated joints and bones whose nearest
        assigned ancestor joint is not their VRM parent are removed, a spine joint holding both
        legs passes its name up the trunk, and a neck joint holding both arms is dropped.
    """
    parent_list = _int_list(parents)
    tree = _Tree(parent_list)
    clean: dict[str, int] = {}
    used: set[int] = set()
    for bone in sorted((b for b in mapping if b in BONE_INFO), key=_ORDER.__getitem__):
        joint = int(mapping[bone])
        if 0 <= joint < tree.count and tree.reached(joint) and joint not in used:
            clean[bone] = joint
            used.add(joint)
    clean = _compact(clean)
    clean = _repair_neck(_repair_trunk(clean, tree), tree)
    for _ in range(len(BONE_INFO) + 1):
        marked = {j: b for b, j in clean.items()}
        dropped = False
        for bone in sorted(clean, key=_ORDER.__getitem__):
            joint = clean[bone]
            expected = _effective_parent(bone, clean)
            if expected is None:
                continue
            if tree.nearest(joint, marked) != clean[expected]:
                del clean[bone]
                del marked[joint]
                dropped = True
        clean = _compact(clean)
        if not dropped:
            break
    return {b: clean[b] for b in sorted(clean, key=_ORDER.__getitem__)}


def _repair_trunk(mapping: dict[str, int], tree: _Tree) -> dict[str, int]:
    """Shift the trunk up one bone when the legs hang from the spine joint."""
    if "spine" not in mapping or "hips" not in mapping:
        return mapping
    legs = [mapping[f"{s}UpperLeg"] for s, *_ in _SIDES if f"{s}UpperLeg" in mapping]
    if not legs:
        return mapping
    marked = {j: b for b, j in mapping.items()}
    spine = mapping["spine"]
    if not all(tree.nearest(j, marked) == spine for j in legs):
        return mapping
    if tree.nearest(spine, marked) != mapping["hips"]:
        return mapping
    out = dict(mapping)
    trunk = [out.pop(b) for b in ("spine", "chest", "upperChest") if b in out][1:]
    for bone, joint in zip(("spine", "chest", "upperChest"), trunk):
        out[bone] = joint
    return out


def _repair_neck(mapping: dict[str, int], tree: _Tree) -> dict[str, int]:
    """Drop the neck when both arms hang from the neck joint."""
    if "neck" not in mapping:
        return mapping
    roots = []
    for side, *_ in _SIDES:
        root = next((mapping[side + p] for p in ("Shoulder", "UpperArm") if side + p in mapping),
                    None)
        if root is None:
            return mapping
        roots.append(root)
    marked = {j: b for b, j in mapping.items()}
    if all(tree.nearest(j, marked) == mapping["neck"] for j in roots):
        return {b: j for b, j in mapping.items() if b != "neck"}
    return mapping


def _int_list(values) -> list[int]:
    """Plain list of ints from a list, array or tensor."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def _positions(heads, count: int) -> list[tuple[float, float, float]] | None:
    """Plain list of ``(x, y, z)`` from a ``(J, 3)`` list, array or tensor."""
    if heads is None:
        return None
    rows = heads.tolist() if hasattr(heads, "tolist") else list(heads)
    if len(rows) != count or any(len(r) != 3 for r in rows):
        raise ValueError(
            f"Joint heads hold {len(rows)} rows for {count} joints. Pass one (x, y, z) per joint."
        )
    out = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    if not all(math.isfinite(v) for p in out for v in p):
        return None
    return out


# Name tokens.

_PREFIX = re.compile(
    r"^(?:mixamorig\d*[_:]?|cc_base_|j_bip_|j_adj_|def[-_.]|bip0*\d+[ _]|"
    r"valvebiped[._]bip0*\d+[ _])", re.IGNORECASE,
)
_CAMEL = re.compile(
    r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_NOISE = frozenset((
    "joint", "jnt", "bone", "bn", "bip", "b", "j", "c", "f", "skeleton", "armature", "def",
    "mixamorig", "character", "rig", "deform",
))
_EXCLUDE = frozenset((
    "twist", "roll", "share", "helper", "ik", "pole", "target", "tgt", "ctrl", "control",
    "mch", "org", "end", "nub", "tip", "top", "sec", "bust", "breast", "hair", "skirt", "tail",
    "ear", "cape", "cloth", "weapon", "prop", "socket", "attach", "corrective", "dummy", "null",
    "offset", "lid", "brow", "lash", "tongue", "teeth", "tooth", "lip", "cheek", "nose",
    "ponytail", "bag", "sleeve", "hood", "wing", "reference", "palm", "adjust", "physics",
    "spring", "dyn", "jiggle", "volume", "bend", "scale", "aim", "lookat", "goal",
))
_FINGER_WORDS = {
    "thumb": "Thumb", "index": "Index", "middle": "Middle", "mid": "Middle", "ring": "Ring",
    "pinky": "Little", "pinkie": "Little", "little": "Little",
}
_RULES = (
    ("hand", re.compile(r"hand|wrist")),
    ("lowerArm", re.compile(r"(fore|lower|low)arm|elbow")),
    ("upperArm", re.compile(r"(upper|up)arm|humerus")),
    ("arm", re.compile(r"arm")),
    ("shoulder", re.compile(r"shoulder|clavicle|clav|collar|collarbone")),
    ("toes", re.compile(r"toes?|toebase|ball")),
    ("foot", re.compile(r"foot|ankle")),
    ("lowerLeg", re.compile(r"calf|shin|knee|(lower|low)leg|crus")),
    ("upperLeg", re.compile(r"thigh|(upper|up)leg|femur")),
    ("leg", re.compile(r"leg")),
    ("neck", re.compile(r"neck")),
    ("head", re.compile(r"head")),
    ("jaw", re.compile(r"jaw(root|bone)?")),
    ("eye", re.compile(r"(face)?eye(ball)?")),
    ("hips", re.compile(r"hips?|pelvis")),
    ("upperChest", re.compile(r"upperchest")),
    ("chest", re.compile(r"chest|ribcage")),
    ("trunk", re.compile(r"(lower|mid|middle|upper)?"
                         r"(spine|torso|waist|abdomen|belly|back|upperbody|lowerback|stomach)"
                         r"(lower|mid|middle|upper)?")),
)


@dataclass
class _Token:
    """Parsed joint name."""

    side: str | None
    kind: str
    finger: str = ""
    meta: bool = False


def _words(name: str) -> tuple[str | None, list[str], list[str]]:
    """Side, lower case words and digit runs of a joint name, prefix and noise words removed."""
    text = _PREFIX.sub("", re.split(r"[:|]", name.strip())[-1])
    parts: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        parts += [p for p in _CAMEL.split(chunk) if p]
    side = None
    words: list[str] = []
    digits: list[str] = []
    for part in parts:
        token = part.lower()
        if token in ("l", "left", "r", "right"):
            side = side or ("left" if token[0] == "l" else "right")
            continue
        for word, label in (("left", "left"), ("right", "right")):
            if token.startswith(word) and len(token) > len(word) and token[len(word):].isalpha():
                side = side or label
                token = token[len(word):]
        if token.isdigit():
            digits.append(token)
        elif token not in _NOISE:
            words.append(token)
    return side, words, digits


def _parse(name: str) -> _Token | None:
    """Side and body part read from a joint name, ``None`` when it names no humanoid part."""
    side, words, digits = _words(name)
    if any(w in _EXCLUDE for w in words):
        return _Token(side, "excluded")
    if not words:
        return None
    finger = next((_FINGER_WORDS[w] for w in words if w in _FINGER_WORDS), "")
    if not finger and "finger" in words and digits:
        finger = _finger_from_digits(digits)
    meta = any("carpal" in w for w in words)
    if finger and side is not None:
        if meta and finger != "Thumb":
            return None
        return _Token(side, "finger", finger, meta or (bool(digits) and digits[-1] == "0"))
    core = "".join(words)
    for kind, rule in _RULES:
        if rule.fullmatch(core):
            centre = kind in ("hips", "trunk", "chest", "upperChest", "neck", "head", "jaw")
            if centre and side is not None:
                return None
            if not centre and side is None:
                return None
            return _Token(side, kind)
    return None


def _finger_from_digits(digits: list[str]) -> str:
    """Finger from numbered names, ``finger1-2`` counting from 1 or ``Finger01`` from 0."""
    if len(digits) >= 2:
        n = int(digits[0]) - 1
    else:
        n = int(digits[0][0])
    return _FINGERS[n] if 0 <= n < 5 else ""


def _spread(chain: list[int]) -> list[int]:
    """Spine, chest and upper chest picked from a trunk chain, hips end first."""
    if len(chain) <= 2:
        return list(chain)
    return [chain[0], chain[(len(chain) - 1) // 2], chain[-1]]


def _fill(cands: list[tuple[int, int, int | None]], slots: int, generic_start: int,
          spill: int | None = None) -> list[int | None]:
    """Assign depth-sorted ``(depth, joint, slot)`` candidates to ordered slots."""
    out: list[int | None] = [None] * slots
    pointer = 0
    for _, joint, slot in cands:
        if slot is None:
            k = max(pointer, generic_start)
            if k < slots:
                out[k] = joint
                pointer = k + 1
        elif slot >= pointer and out[slot] is None:
            out[slot] = joint
            pointer = slot + 1
        elif slot == spill and pointer == slot + 1 and pointer < slots and out[pointer] is None:
            out[pointer] = joint
            pointer += 1
    return out


def _by_names(names: list[str], tree: _Tree, skip: set[int]) -> tuple[dict[str, int], set[int]]:
    """VRM bones read from joint name tokens and the hierarchy, plus joints the names exclude."""
    tokens: dict[int, _Token] = {}
    excluded: set[int] = set()
    for joint, name in enumerate(names):
        if joint in skip or not tree.reached(joint):
            continue
        token = _parse(name)
        if token is not None and token.kind == "excluded":
            excluded.add(joint)
        elif token is not None:
            tokens[joint] = token

    def of(kind: str, side: str | None = None) -> list[int]:
        found = [j for j, t in tokens.items() if t.kind == kind and t.side == side]
        return sorted(found, key=lambda j: (tree.depth[j], j))

    out: dict[str, int] = {}
    legs = {side: of("upperLeg", side) + of("leg", side) for side, *_ in _SIDES}
    hips = of("hips")
    if hips:
        others = [j for j in tokens if j not in hips]
        out["hips"] = max(hips, key=lambda h: (sum(tree.above(h, o) for o in others),
                                                -tree.depth[h]))
    else:
        firsts = [min(v, key=tree.depth.__getitem__) for v in legs.values() if v]
        uppers = of("neck") + of("head") + of("trunk") + of("chest")
        if len(firsts) == 2 and uppers:
            joint = tree.lca(firsts[0], firsts[1])
            joint = tree.lca(joint, min(uppers, key=tree.depth.__getitem__))
            if joint >= 0:
                out["hips"] = joint
    hip = out.get("hips", -1)

    heads = [j for j in of("head") if hip < 0 or tree.above(hip, j)]
    necks = [j for j in of("neck") if hip < 0 or tree.above(hip, j)]
    if heads:
        out["head"] = heads[0]
        necks = [j for j in necks if tree.above(j, heads[0]) and j != heads[0]]
    elif len(necks) >= 2 and tree.above(necks[0], necks[-1]):
        out["head"] = necks[-1]
        necks = necks[:-1]
    if necks:
        out["neck"] = necks[0]
    anchor = out.get("neck", out.get("head", -1))

    trunk = [(tree.depth[j], j, None) for j in of("trunk")]
    trunk += [(tree.depth[j], j, 1) for j in of("chest")]
    trunk += [(tree.depth[j], j, 2) for j in of("upperChest")]
    trunk = sorted(c for c in trunk if c[1] != hip and (hip < 0 or tree.above(hip, c[1]))
                   and (anchor < 0 or tree.above(c[1], anchor)))
    if trunk and all(slot is None for _, _, slot in trunk):
        slots: list[int | None] = list(_spread([j for _, j, _ in trunk]))
    else:
        slots = _fill(trunk, 3, 0)
    for bone, joint in zip(("spine", "chest", "upperChest"), slots):
        if joint is not None:
            out[bone] = joint

    head = out.get("head", -1)
    jaws = [j for j in of("jaw") if head < 0 or tree.above(head, j)]
    if jaws:
        out["jaw"] = jaws[0]
    for side, *_ in _SIDES:
        eyes = [j for j in of("eye", side) if head < 0 or tree.above(head, j)]
        if eyes:
            out[f"{side}Eye"] = eyes[0]
        cands = [(tree.depth[j], j, slot) for slot, kind in
                 enumerate(("upperLeg", "lowerLeg", "foot", "toes")) for j in of(kind, side)]
        cands += [(tree.depth[j], j, None) for j in of("leg", side)]
        for bone, joint in zip(("UpperLeg", "LowerLeg", "Foot", "Toes"),
                               _fill(sorted(cands), 4, 0, spill=2)):
            if joint is not None:
                out[side + bone] = joint
        cands = [(tree.depth[j], j, slot) for slot, kind in
                 enumerate(("shoulder", "upperArm", "lowerArm", "hand")) for j in of(kind, side)]
        cands += [(tree.depth[j], j, None) for j in of("arm", side)]
        for bone, joint in zip(("Shoulder", "UpperArm", "LowerArm", "Hand"),
                               _fill(sorted(cands), 4, 1)):
            if joint is not None:
                out[side + bone] = joint
        for finger in _FINGERS:
            chain = sorted((j for j, t in tokens.items()
                            if t.kind == "finger" and t.side == side and t.finger == finger),
                           key=lambda j: (tree.depth[j], j))
            if len(chain) > 3 and not tree.children[chain[-1]]:
                chain = chain[:-1]
            if len(chain) > 3 and tokens[chain[0]].meta:
                chain = chain[1:]
            for segment, joint in zip(_segments(finger), chain[:3]):
                out[f"{side}{finger}{segment}"] = joint
    return out, excluded


# Geometry and hierarchy.

def _sub(a, b):
    """Vector ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v) -> float:
    """Length of a vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _flat(a, b) -> float:
    """Horizontal distance between two points."""
    return math.hypot(a[0] - b[0], a[2] - b[2])


def _leg(chain: list[int], pos, hip: int) -> dict[str, int] | None:
    """Upper leg, lower leg, foot and toes along a chain from the hips to the lowest joint."""
    y = [pos[j][1] for j in chain]
    top, floor = pos[hip][1], y[-1]
    length = top - floor
    if length <= 0 or len(chain) < 3:
        return None
    upper = 0
    for i, value in enumerate(y):
        if value >= top - 0.18 * length:
            upper = i
    rest = list(range(upper + 1, len(chain)))
    if len(rest) < 2:
        return None
    target = floor + 0.5 * (y[upper] - floor)
    knee = min(rest[:-1], key=lambda i: abs(y[i] - target))
    threshold = y[knee] - 0.6 * (y[knee] - floor)
    foot = next((i for i in range(knee + 1, len(chain)) if y[i] <= threshold), None)
    if foot is None:
        return None
    out = {"UpperLeg": chain[upper], "LowerLeg": chain[knee], "Foot": chain[foot]}
    if foot + 1 < len(chain):
        out["Toes"] = chain[foot + 1]
    return out


def _arm(tree: _Tree, pos, base: int, root: int, skip: set[int]) -> dict[str, int] | None:
    """Shoulder, arm, hand and fingers of the subtree that leaves the spine at ``base``."""
    sub = [j for j in tree.subtree(root) if j not in skip]
    if not sub:
        return None
    tip = max(sub, key=lambda j: _norm(_sub(pos[j], pos[base])))
    path = [root] + tree.path(root, tip)
    hand_at = None
    for i in range(1, len(path)):
        joint = path[i]
        branches = [k for k in tree.children[joint] if k not in skip and tree.children[k]]
        if len(branches) >= 2:
            reach = max(_norm(_sub(pos[k], pos[joint])) for k in tree.subtree(joint))
            if reach < 0.5 * _norm(_sub(pos[joint], pos[base])):
                hand_at = i
                break
    fan = hand_at is not None
    if hand_at is None:
        hand_at = len(path) - 1
    before = path[:hand_at]
    if len(before) < 2:
        return None
    hand = path[hand_at]
    out: dict[str, int] = {}
    if len(before) >= 3 and _flat(pos[base], pos[before[0]]) < 0.5 * _flat(pos[base], pos[before[1]]):
        out["Shoulder"] = before[0]
        before = before[1:]
    upper = before[0]
    lower = min(before[1:], key=lambda j: abs(_norm(_sub(pos[j], pos[hand]))
                                              - _norm(_sub(pos[j], pos[upper]))))
    out.update({"UpperArm": upper, "LowerArm": lower, "Hand": hand})
    if fan:
        out.update(_fingers(tree, pos, hand, skip))
    return out


def _fingers(tree: _Tree, pos, hand: int, skip: set[int]) -> dict[str, int]:
    """Five finger chains of a hand, ordered from the thumb, when the hand has exactly five."""
    chains = []
    for start in tree.children[hand]:
        if start in skip:
            continue
        chain = [start]
        while True:
            kids = [k for k in tree.children[chain[-1]] if k not in skip]
            if len(kids) != 1:
                break
            chain.append(kids[0])
        if len(chain) >= 2:
            chains.append(chain)
    if len(chains) != 5:
        return {}
    gaps = [_norm(_sub(pos[c[0]], pos[hand])) for c in chains]
    ranked = sorted(range(5), key=gaps.__getitem__)
    if gaps[ranked[0]] < 0.6 * gaps[ranked[1]]:
        thumb = ranked[0]
    else:
        lengths = [len(c) for c in chains]
        odd = [i for i in range(5) if lengths.count(lengths[i]) == 1]
        if len(odd) != 1:
            return {}
        thumb = odd[0]
    out: dict[str, int] = {}
    chain = chains[thumb]
    if len(chain) > 3 and not tree.children[chain[-1]]:
        chain = chain[:-1]
    for segment, joint in zip(_segments("Thumb"), chain[:3]):
        out[f"Thumb{segment}"] = joint
    rest = []
    for i in range(5):
        if i == thumb:
            continue
        chain = chains[i]
        if len(chain) > 3:
            span = _norm(_sub(pos[chain[-1]], pos[chain[0]]))
            chain = chain[1:] if gaps[i] < 0.3 * span else chain[:3]
        rest.append(chain)
    anchor = pos[chains[thumb][0]]
    rest.sort(key=lambda c: _norm(_sub(pos[c[0]], anchor)))
    for finger, chain in zip(_FINGERS[1:], rest):
        for segment, joint in zip(_segments(finger), chain[:3]):
            out[f"{finger}{segment}"] = joint
    return out


def _by_geometry(tree: _Tree, pos, skip: set[int]) -> dict[str, int]:
    """VRM bones read from joint heads and the hierarchy alone."""
    live = [j for j in tree.order if j not in skip]
    if len(live) < 8:
        return {}
    ys = [pos[j][1] for j in live]
    floor, height = min(ys), max(ys) - min(ys)
    if height <= 1e-9:
        return {}
    leaves = sorted((j for j in live if not any(k not in skip for k in tree.children[j])),
                    key=lambda j: pos[j][1])
    if len(leaves) < 3:
        return {}
    first = leaves[0]
    second = None
    for leaf in leaves[1:]:
        if pos[leaf][1] > floor + 0.25 * height:
            break
        joint = tree.lca(first, leaf)
        if joint >= 0 and pos[joint][1] > pos[first][1] + 0.25 * height:
            second = leaf
            break
    if second is None:
        return {}
    top = max(live, key=lambda j: pos[j][1])
    hip = tree.lca(tree.lca(first, second), top)
    if hip < 0 or pos[top][1] <= pos[hip][1] or pos[hip][1] - floor < 0.25 * height:
        return {}
    legs = [_leg([j for j in tree.path(hip, leaf) if j not in skip], pos, hip)
            for leaf in (first, second)]
    if not all(legs):
        return {}
    heading = [0.0, 0.0]
    for leg in legs:
        if "Toes" in leg:
            step = _sub(pos[leg["Toes"]], pos[leg["Foot"]])
            heading[0] += step[0]
            heading[1] += step[2]
    size = math.hypot(*heading)
    forward = (heading[0] / size, heading[1] / size) if size > 1e-9 else (0.0, 1.0)
    left = (forward[1], -forward[0])

    def side_of(joint: int, origin: int) -> float:
        offset = _sub(pos[joint], pos[origin])
        return offset[0] * left[0] + offset[2] * left[1]

    out = {"hips": hip}
    signs = [side_of(leg["UpperLeg"], hip) for leg in legs]
    if signs[0] * signs[1] < 0:
        for leg, sign in zip(legs, signs):
            side = "left" if sign > 0 else "right"
            out.update({side + k: v for k, v in leg.items()})
    spine = [j for j in tree.path(hip, top) if j not in skip]
    branches = []
    for i, base in enumerate(spine):
        following = spine[i + 1] if i + 1 < len(spine) else -1
        for child in tree.children[base]:
            if child == following or child in skip:
                continue
            if tree.above(child, first) or tree.above(child, second):
                continue
            sub = tree.subtree(child)
            reach = max(_norm(_sub(pos[k], pos[base])) for k in sub)
            if reach > 0.2 * height:
                cx = sum(pos[k][0] for k in sub) / len(sub) - pos[base][0]
                cz = sum(pos[k][2] for k in sub) / len(sub) - pos[base][2]
                branches.append((reach, i, child, (cx, cz)))
    branches.sort(key=lambda b: -b[0])
    if not branches:
        return out
    one = branches[0]
    two = next((b for b in branches[1:]
                if b[3][0] * one[3][0] + b[3][1] * one[3][1] < 0), None)
    if two is None:
        return out
    branch_at = max(one[1], two[1])
    out.update(zip(("spine", "chest", "upperChest"), _spread(spine[:branch_at + 1])))
    upper = spine[branch_at + 1:]
    if upper:
        head = upper[-1]
        for joint in upper:
            head = joint
            kids = [k for k in tree.children[joint] if k not in skip]
            if len(kids) != 1:
                break
            # Stops where the chain turns more forward than up.
            step = _sub(pos[kids[0]], pos[joint])
            if abs(step[1]) < math.hypot(step[0], step[2]):
                break
        at = upper.index(head)
        # Steps back from a crown tip: a leaf whose bone is over 1.5 times its parent's.
        if at >= 2 and not any(k not in skip for k in tree.children[head]):
            last = _norm(_sub(pos[head], pos[upper[at - 1]]))
            if last > 1.5 * _norm(_sub(pos[upper[at - 1]], pos[upper[at - 2]])):
                head = upper[at - 1]
        out["head"] = head
        if head != upper[0]:
            out["neck"] = upper[0]
    arms = []
    for branch in (one, two):
        base = spine[branch[1]]
        arm = _arm(tree, pos, base, branch[2], skip)
        if arm is None:
            return out
        arms.append((arm, side_of(arm["Hand"], base)))
    if arms[0][1] * arms[1][1] < 0:
        for arm, sign in arms:
            side = "left" if sign > 0 else "right"
            out.update({side + k: v for k, v in arm.items()})
    return out


def _merge(base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    """Add bones from ``extra`` that ``base`` lacks, on joints ``base`` does not use."""
    out = dict(base)
    used = set(out.values())
    for bone, joint in extra.items():
        if bone not in out and joint not in used:
            out[bone] = joint
            used.add(joint)
    return out


def _bone_side(bone: str) -> str | None:
    """``"left"``, ``"right"`` or ``None`` for a centre bone."""
    if bone.startswith("left"):
        return "left"
    if bone.startswith("right"):
        return "right"
    return None


def _orient(found: dict[str, int], known: dict[str, int],
            sides: list[str | None]) -> dict[str, int]:
    """Swap left and right in ``found`` when known bones and name sides mostly read it mirrored."""
    same = flipped = 0
    for bone, joint in found.items():
        side = _bone_side(bone)
        if side is None:
            continue
        if known.get(bone) == joint or sides[joint] == side:
            same += 1
        elif known.get(mirror(bone)) == joint or sides[joint] is not None:
            flipped += 1
    if flipped > same:
        return {mirror(b): j for b, j in found.items()}
    return found


def _agree(mapping: dict[str, int], sides: list[str | None]) -> dict[str, int]:
    """Drop bones placed on a joint whose name states the other side."""
    return {b: j for b, j in mapping.items()
            if _bone_side(b) is None or sides[j] is None or sides[j] == _bone_side(b)}


def guess_humanoid(names: list[str], parents, heads=None) -> dict[str, int]:
    """Label joints with VRM humanoid bones by rig table, name tokens, then geometry.

    Args:
        names: Joint names.
        parents: ``(J,)`` parent index of each joint, ``-1`` for a root.
        heads: ``(J, 3)`` joint heads in metres, Y-up, or ``None`` to use names alone.
            Skeletons over ``MAX_GUESS_JOINTS`` joints are matched by rig table only.

    Returns:
        VRM bone name to joint index, in VRM bone order, each joint used at most once. A joint
        whose name states a side never holds a bone of the other side.

    Raises:
        ValueError: When ``parents`` or ``heads`` do not hold one row per joint.
    """
    names = [str(n) for n in names]
    parent_list = _int_list(parents)
    if len(parent_list) != len(names):
        raise ValueError(
            f"{len(parent_list)} parents for {len(names)} joints. Pass one parent per joint."
        )
    pos = _positions(heads, len(names))
    rig, found = map_by_name(names)
    if len(names) <= MAX_GUESS_JOINTS:
        tree = _Tree(parent_list)
        skip = _known_none(names, rig)
        sides = [_words(n)[0] for n in names]
        found = validate_humanoid(_agree(found, sides), parent_list)
        tokens, excluded = _by_names(names, tree, skip)
        found = validate_humanoid(_merge(found, validate_humanoid(tokens, parent_list)),
                                  parent_list)
        if pos is not None:
            shape = _by_geometry(tree, pos, skip | excluded)
            shape = _agree(_orient(shape, found, sides), sides)
            found = validate_humanoid(_merge(found, validate_humanoid(shape, parent_list)),
                                      parent_list)
        found = _agree(found, sides)
    found = validate_humanoid(found, parent_list)
    logger.debug("Humanoid guess: rig %s, %d bones.", rig or "unknown", len(found))
    return found


# VRM T-pose directions from each bone's head toward its child.

def _tpose() -> dict[str, tuple[float, float, float]]:
    """Unit direction of each bone in the VRM T-pose."""
    up, down, ahead = (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)
    half = math.sqrt(0.5)
    out = {b: up for b in ("hips", "spine", "chest", "upperChest", "neck", "head")}
    for (side, *_), x in zip(_SIDES, (1.0, -1.0)):
        out.update({side + "UpperLeg": down, side + "LowerLeg": down,
                    side + "Foot": ahead, side + "Toes": ahead})
        for part in ("Shoulder", "UpperArm", "LowerArm", "Hand"):
            out[side + part] = (x, 0.0, 0.0)
        for finger in _FINGERS:
            way = (x * half, 0.0, half) if finger == "Thumb" else (x, 0.0, 0.0)
            for segment in _segments(finger):
                out[f"{side}{finger}{segment}"] = way
    return out


TPOSE_DIRECTIONS: dict[str, tuple[float, float, float]] = _tpose()
HEADING_ONLY: frozenset[str] = frozenset(("leftFoot", "rightFoot", "leftToes", "rightToes"))


def tpose_directions(bones: list[str]):
    """T-pose directions of several bones as a tensor.

    Args:
        bones: VRM bone names.

    Returns:
        ``(N, 3)`` float32 CPU tensor, zero rows for bones with no fixed direction (eyes, jaw).
    """
    import torch

    rows = [TPOSE_DIRECTIONS.get(b, (0.0, 0.0, 0.0)) for b in bones]
    return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), 3)
