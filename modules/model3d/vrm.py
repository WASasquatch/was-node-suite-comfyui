"""VRM 0.x and 1.0 avatars read into `scene.Model3D`, and VRM 1.0 written from one.

VRM 0.x models turn 180 degrees about Y on read to face +Z. Data with no model field sits in
``model.extras`` under ``vrm_`` keys.
"""

from __future__ import annotations

__all__ = [
    "EXPRESSION_PRESETS", "HUMAN_BONES", "LICENSE_URL", "META_KEYS", "REQUIRED_BONES", "VrmError",
    "is_vrm", "missing_humanoid", "read", "write",
]

import dataclasses
import math

import numpy as np
import torch

from ..log import get_logger
from . import gltf
from .gltf import GltfError
from .scene import Collider, Expression, Material, Model3D, Part, Skeleton, SpringChain, Springs, Texture

logger = get_logger("model3d.vrm")


class VrmError(GltfError):
    """A VRM file that cannot be read, or a model that cannot be written as VRM."""


def _bone_table() -> dict[str, str | None]:
    """Each VRM 1.0 humanoid bone against its parent bone, parents first."""
    table: dict[str, str | None] = {
        "hips": None, "spine": "hips", "chest": "spine", "upperChest": "chest",
        "neck": "upperChest", "head": "neck", "leftEye": "head", "rightEye": "head", "jaw": "head",
    }
    for side in ("left", "right"):
        table.update({
            f"{side}UpperLeg": "hips", f"{side}LowerLeg": f"{side}UpperLeg",
            f"{side}Foot": f"{side}LowerLeg", f"{side}Toes": f"{side}Foot",
        })
    for side in ("left", "right"):
        table.update({
            f"{side}Shoulder": "upperChest", f"{side}UpperArm": f"{side}Shoulder",
            f"{side}LowerArm": f"{side}UpperArm", f"{side}Hand": f"{side}LowerArm",
        })
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            segments = (("Metacarpal", "Proximal", "Distal") if finger == "Thumb"
                        else ("Proximal", "Intermediate", "Distal"))
            parent = f"{side}Hand"
            for segment in segments:
                table[f"{side}{finger}{segment}"] = parent
                parent = f"{side}{finger}{segment}"
    return table


_PARENT_BONE = _bone_table()

#: Every VRM 1.0 humanoid bone, parents before children.
HUMAN_BONES = tuple(_PARENT_BONE)

#: Bones a VRM 1.0 humanoid must map.
REQUIRED_BONES = (
    "hips", "spine", "head",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot",
    "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
)

#: VRM 0.x thumb bones against their VRM 1.0 names.
_VRM0_BONES = {
    f"{side}Thumb{old}": f"{side}Thumb{new}"
    for side in ("left", "right")
    for old, new in (("Proximal", "Metacarpal"), ("Intermediate", "Proximal"))
}

#: VRM 1.0 expression preset names.
EXPRESSION_PRESETS = (
    "happy", "angry", "sad", "relaxed", "surprised", "aa", "ih", "ou", "ee", "oh",
    "blink", "blinkLeft", "blinkRight", "lookUp", "lookDown", "lookLeft", "lookRight", "neutral",
)

#: VRM 0.x blend shape presets against VRM 1.0 expression presets.
_VRM0_PRESETS = {
    "neutral": "neutral", "a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh",
    "blink": "blink", "joy": "happy", "angry": "angry", "sorrow": "sad", "fun": "relaxed",
    "lookup": "lookUp", "lookdown": "lookDown", "lookleft": "lookLeft", "lookright": "lookRight",
    "blink_l": "blinkLeft", "blink_r": "blinkRight",
}

#: VRM 0.x material properties against VRM 1.0 material colour bind types.
_VRM0_COLOR_BINDS = {
    "_Color": "color", "_ShadeColor": "shadeColor", "_EmissionColor": "emissionColor",
    "_RimColor": "rimColor", "_OutlineColor": "outlineColor",
}

_COLOR_BIND_TYPES = ("color", "emissionColor", "shadeColor", "matcapColor", "rimColor", "outlineColor")
_OVERRIDES = ("none", "block", "blend")
_FIRST_PERSON = ("auto", "both", "thirdPersonOnly", "firstPersonOnly")
_VRM0_FIRST_PERSON = {
    "auto": "auto", "both": "both", "thirdpersononly": "thirdPersonOnly",
    "firstpersononly": "firstPersonOnly",
}
_RANGE_MAPS = (
    ("rangeMapHorizontalInner", "lookAtHorizontalInner"),
    ("rangeMapHorizontalOuter", "lookAtHorizontalOuter"),
    ("rangeMapVerticalDown", "lookAtVerticalDown"),
    ("rangeMapVerticalUp", "lookAtVerticalUp"),
)

#: Licence document VRM 1.0 meta points to when none is given.
LICENSE_URL = "https://vrm.dev/licenses/1.0/"

_META_STRINGS = (
    "name", "version", "copyrightInformation", "contactInformation", "licenseUrl",
    "thirdPartyLicenses", "otherLicenseUrl",
)
_META_LISTS = ("authors", "references")
_META_FLAGS = (
    "allowExcessivelyViolentUsage", "allowExcessivelySexualUsage", "allowPoliticalOrReligiousUsage",
    "allowAntisocialOrHateUsage", "allowRedistribution",
)
#: Allowed values of each VRM 1.0 meta choice, most restrictive first.
_META_CHOICES = {
    "avatarPermission": ("onlyAuthor", "onlySeparatelyLicensedPerson", "everyone"),
    "commercialUsage": ("personalNonProfit", "personalProfit", "corporation"),
    "creditNotation": ("required", "unnecessary"),
    "modification": ("prohibited", "allowModification", "allowModificationRedistribution"),
}

#: VRM 1.0 meta fields as they are kept in ``Model3D.meta``.
META_KEYS = _META_STRINGS + _META_LISTS + _META_FLAGS + tuple(_META_CHOICES)

#: VRM 0.x licence names against the VRM 1.0 permissions they grant.
_VRM0_LICENSES = {
    "CC0": {"allowRedistribution": True, "modification": "allowModificationRedistribution",
            "creditNotation": "unnecessary"},
    "CC_BY": {"allowRedistribution": True, "modification": "allowModificationRedistribution"},
    "CC_BY_SA": {"allowRedistribution": True, "modification": "allowModificationRedistribution"},
    "CC_BY_NC": {"allowRedistribution": True, "modification": "allowModificationRedistribution",
                 "commercialUsage": "personalNonProfit"},
    "CC_BY_NC_SA": {"allowRedistribution": True, "modification": "allowModificationRedistribution",
                    "commercialUsage": "personalNonProfit"},
    "CC_BY_ND": {"allowRedistribution": True},
    "CC_BY_NC_ND": {"allowRedistribution": True, "commercialUsage": "personalNonProfit"},
}

#: Most morph bindings read across every expression of one file.
MAX_BINDS = 4_000_000

_SPRING_DEFAULTS = {"hitRadius": 0.0, "stiffness": 1.0, "gravityPower": 0.0,
                    "gravityDir": [0.0, -1.0, 0.0], "dragForce": 0.5}

#: VRM 0.x bone group settings a file leaves out, under their VRM 0.x names.
_VRM0_SPRING_DEFAULTS = {"hitRadius": 0.02, "stiffiness": 1.0, "gravityPower": 0.0, "dragForce": 0.4}

#: Length of the tip joint added past the last bone of a VRM 0.x spring, in that bone's units.
_VRM0_TIP = 0.07

#: Sign of each axis under a half turn about Y.
_TURN = np.array([-1.0, 1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------------------------


def _number(value, default: float) -> float:
    """A finite float from a JSON value, or ``default``."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _vector(value, default=(0.0, 0.0, 0.0)) -> list[float]:
    """Three numbers from a JSON list or an ``{x, y, z}`` object."""
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y"), value.get("z")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return [float(item) for item in default]
    return [_number(value[axis], float(default[axis])) for axis in range(3)]


def _color(value, default, width: int) -> list[float]:
    """``width`` finite numbers from a JSON list, padded from ``default``."""
    if not isinstance(value, (list, tuple)):
        return [float(item) for item in default[:width]]
    return [_number(value[axis], float(default[axis])) if axis < len(value) else float(default[axis])
            for axis in range(width)]


def _index(value) -> int | None:
    """A non-negative integer index from a JSON value, or None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _list(value) -> list:
    """A JSON list, or an empty list."""
    return value if isinstance(value, list) else []


def _items(value) -> list:
    """The dictionaries of a JSON list, or an empty list."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _mapping(value) -> dict:
    """A JSON object, or an empty dictionary."""
    return value if isinstance(value, dict) else {}


def _srgb_to_linear(value: float) -> float:
    """One sRGB code converted to linear light."""
    value = max(value, 0.0)
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _unique(name: str, taken) -> str:
    """``name``, or ``name_2`` and so on when ``taken`` holds it."""
    if name not in taken:
        return name
    suffix = 2
    while f"{name}_{suffix}" in taken:
        suffix += 1
    return f"{name}_{suffix}"


def is_vrm(json_dict) -> bool:
    """Whether a parsed glTF JSON document carries a VRM 0.x or 1.0 extension.

    Args:
        json_dict: The glTF JSON document.

    Returns:
        True for a VRM file.
    """
    if not isinstance(json_dict, dict):
        return False
    extensions = json_dict.get("extensions")
    return isinstance(extensions, dict) and ("VRMC_vrm" in extensions or "VRM" in extensions)


def missing_humanoid(model: Model3D) -> list[str]:
    """Required VRM 1.0 humanoid bones the model does not map to a joint.

    Args:
        model: The model.

    Returns:
        Bone names in `REQUIRED_BONES` order, empty when every required bone is mapped.
    """
    skeleton = model.skeleton
    if skeleton is None:
        return list(REQUIRED_BONES)
    count = skeleton.count
    mapped = {bone for bone, joint in skeleton.humanoid.items()
              if isinstance(joint, int) and not isinstance(joint, bool) and 0 <= joint < count}
    return [bone for bone in REQUIRED_BONES if bone not in mapped]


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


class _Context:
    """State shared while one VRM file becomes a model."""

    def __init__(self, doc: gltf.GltfDocument, model: Model3D, node_map: gltf.NodeMap,
                 graph) -> None:
        self.doc = doc
        self.model = model
        self.node_map = node_map
        self.graph = graph
        self.binds = 0
        self.materials: dict[int, Material] = {}
        for part, index in zip(model.parts, node_map.material_of_part):
            if index >= 0:
                self.materials.setdefault(index, part.material)

    def joint(self, node) -> int:
        """Skeleton joint of a glTF node, ``-1`` when it is not one."""
        index = _index(node)
        return self.node_map.joint_of_node.get(index, -1) if index is not None else -1

    def node_name(self, node) -> str:
        """Name of a glTF node, or empty."""
        index = _index(node)
        if index is None or index >= len(self.graph.nodes):
            return ""
        return str(self.graph.nodes[index].get("name") or "")

    def morph_binds(self, parts: list[int], morph, weight: float) -> list[tuple[int, str, float]]:
        """Expression binds for one glTF morph target index across the parts drawn from it."""
        index = _index(morph)
        binds = []
        if index is None:
            return binds
        for part_index in parts:
            names = list(self.model.parts[part_index].morphs)
            if index < len(names):
                binds.append((part_index, names[index], weight))
        self.binds += len(binds)
        if self.binds > MAX_BINDS:
            raise VrmError(f"the VRM file binds more than {MAX_BINDS:,} shape keys to its "
                           f"expressions. The file is damaged; re-export it.")
        return binds

    def material_name(self, index) -> str | None:
        """Name of the model material made from a glTF material index, or None."""
        index = _index(index)
        if index is None:
            return None
        if index in self.materials:
            return self.materials[index].name
        materials = self.doc.json.get("materials")
        if isinstance(materials, list) and index < len(materials) and isinstance(materials[index], dict):
            return str(materials[index].get("name") or f"material_{index}")
        return None

    def texture(self, index) -> Texture | None:
        """The texture at a glTF texture index, or None with a warning."""
        index = _index(index)
        if index is None:
            return None
        try:
            return self.doc.texture(index)
        except GltfError as error:
            logger.warning("a VRM texture was skipped: %s", error)
            return None


def _wanted_nodes(document: dict, version: str) -> set[int]:
    """glTF nodes the VRM extensions name, which must become skeleton joints."""
    nodes = document.get("nodes") if isinstance(document.get("nodes"), list) else []
    count = len(nodes)
    extensions = _mapping(document.get("extensions"))
    wanted: set[int] = set()

    def add(value) -> None:
        index = _index(value)
        if index is not None and index < count:
            wanted.add(index)

    if version == "1.0":
        vrm = _mapping(extensions.get("VRMC_vrm"))
        for spec in _mapping(_mapping(vrm.get("humanoid")).get("humanBones")).values():
            if isinstance(spec, dict):
                add(spec.get("node"))
        springs = _mapping(extensions.get("VRMC_springBone"))
        for collider in _items(springs.get("colliders")):
            add(collider.get("node"))
        for spring in _items(springs.get("springs")):
            for joint in _items(spring.get("joints")):
                add(joint.get("node"))
            add(spring.get("center"))
        for index, node in enumerate(nodes):
            constraint = _mapping(_mapping(_mapping(node).get("extensions")).get("VRMC_node_constraint"))
            for spec in _mapping(constraint.get("constraint")).values():
                if isinstance(spec, dict) and _index(spec.get("source")) is not None:
                    add(index)
                    add(spec.get("source"))
        return wanted
    vrm = _mapping(extensions.get("VRM"))
    for spec in _items(_mapping(vrm.get("humanoid")).get("humanBones")):
        add(spec.get("node"))
    secondary = _mapping(vrm.get("secondaryAnimation"))
    for group in _items(secondary.get("colliderGroups")):
        add(group.get("node"))
    stack = []
    for group in _items(secondary.get("boneGroups")):
        stack.extend(bone for bone in _list(group.get("bones")) if _index(bone) is not None)
        add(group.get("center"))
    seen: set[int] = set()
    while stack:
        index = stack.pop()
        if not isinstance(index, int) or not 0 <= index < count or index in seen:
            continue
        seen.add(index)
        node = _mapping(nodes[index])
        if "mesh" in node:
            continue
        wanted.add(index)
        stack.extend(child for child in _list(node.get("children")) if isinstance(child, int))
    add(_mapping(vrm.get("firstPerson")).get("firstPersonBone"))
    return wanted


def _attach_nodes(document: dict, wanted: set[int]) -> list[int]:
    """List wanted nodes outside every skin as a skin of their own, so they read as joints."""
    if not isinstance(document.get("skins"), list):
        document["skins"] = []
    joined: set[int] = set()
    for skin in _items(document["skins"]):
        joined.update(joint for joint in _list(skin.get("joints")) if isinstance(joint, int))
    extra = sorted(wanted - joined)
    if extra:
        document["skins"].append({"joints": extra})
    return extra


def _settle_joints(model: Model3D, node_map: gltf.NodeMap, graph, added: list[int]) -> None:
    """Give joints added from VRM nodes a rest pose relative to their parent and a matching bind."""
    skeleton = model.skeleton
    if skeleton is None or not added:
        return
    node_of_joint = {joint: node for node, joint in node_map.joint_of_node.items()}
    rest = skeleton.rest.double().numpy()
    inverse = skeleton.inverse_bind.double().numpy()
    old_inverse = inverse.copy()
    joints = sorted(node_map.joint_of_node[node] for node in added if node in node_map.joint_of_node)
    for joint in joints:
        node = node_of_joint[joint]
        parent = int(skeleton.parents[joint])
        if parent >= 0:
            relative = np.linalg.inv(graph.world[node_of_joint[parent]]) @ graph.world[node]
            rest[joint] = rest[parent] @ relative
        else:
            rest[joint] = graph.world[node]
        inverse[joint] = np.linalg.inv(rest[joint])
    skeleton.rest = torch.from_numpy(rest.astype(np.float32))
    skeleton.inverse_bind = torch.from_numpy(inverse.astype(np.float32))
    # Re-expresses parts bound rigidly to an added joint under that joint's new bind.
    settled = set(joints)
    for part in model.parts:
        if part.joints is None or part.weights is None or part.vertex_count == 0:
            continue
        owners = part.joints[:, 0].unique()
        if owners.numel() != 1 or int(owners[0]) not in settled or bool((part.weights[:, 1:] != 0).any()):
            continue
        joint = int(owners[0])
        _carry(part, rest[joint] @ old_inverse[joint])


def _carry(part: Part, matrix: np.ndarray) -> None:
    """Transform a part's vertices, normals, tangents and shape keys by a 4x4 matrix in place."""
    if np.abs(matrix - np.eye(4)).max() <= 1e-12:
        return
    linear = torch.from_numpy(matrix[:3, :3].copy())
    offset = torch.from_numpy(matrix[:3, 3].copy())
    normal_matrix = torch.linalg.inv(linear).T
    part.vertices = (part.vertices.double() @ linear.T + offset).float()
    if part.normals is not None:
        part.normals = torch.nn.functional.normalize(part.normals.double() @ normal_matrix.T, dim=1).float()
    if part.tangents is not None:
        turned = torch.nn.functional.normalize(part.tangents[:, :3].double() @ linear.T, dim=1).float()
        part.tangents = torch.cat([turned, part.tangents[:, 3:]], dim=1)
    for morph in part.morphs.values():
        morph.positions = (morph.positions.double() @ linear.T).float()
        if morph.normals is not None:
            morph.normals = (morph.normals.double() @ normal_matrix.T).float()


def _texture_info(ctx: _Context, info, base_transform) -> dict | None:
    """An MToon texture reference with its texture decoded under ``"texture"``, or None."""
    if not isinstance(info, dict):
        return None
    if info.get("texCoord", 0) != 0:
        logger.warning("an MToon texture samples a second UV set, which is not read; it was dropped")
        return None
    texture = ctx.texture(info.get("index"))
    if texture is None:
        return None
    kept = {key: value for key, value in info.items() if key not in ("index", "texCoord")}
    extensions = _mapping(kept.get("extensions"))
    transform = extensions.get("KHR_texture_transform")
    if transform is not None and transform == base_transform:
        extensions = {key: value for key, value in extensions.items() if key != "KHR_texture_transform"}
        if extensions:
            kept["extensions"] = extensions
        else:
            kept.pop("extensions", None)
    kept["texture"] = texture
    return kept


def _base_transform(document: dict, index: int):
    """The base colour texture transform of a glTF material, or None."""
    materials = document.get("materials")
    if not isinstance(materials, list) or index >= len(materials):
        return None
    base = _mapping(_mapping(_mapping(materials[index]).get("pbrMetallicRoughness")).get("baseColorTexture"))
    return _mapping(base.get("extensions")).get("KHR_texture_transform")


def _read_mtoon1(ctx: _Context) -> None:
    """Move each material's VRMC_materials_mtoon extension into ``extras["mtoon"]``."""
    for index, material in ctx.materials.items():
        extensions = material.extras.get("extensions")
        if not isinstance(extensions, dict) or not isinstance(extensions.get("VRMC_materials_mtoon"), dict):
            continue
        spec = extensions.pop("VRMC_materials_mtoon")
        if not extensions:
            material.extras.pop("extensions", None)
        base = _base_transform(ctx.doc.json, index)
        mtoon = {}
        for key, value in spec.items():
            if key.endswith("Texture") and isinstance(value, dict):
                info = _texture_info(ctx, value, base)
                if info is not None:
                    mtoon[key] = info
            else:
                mtoon[key] = value
        material.extras["mtoon"] = mtoon


def _mtoon_from_vrm0(ctx: _Context, props: dict) -> dict:
    """VRM 1.0 MToon settings from VRM 0.x material properties."""
    floats = _mapping(props.get("floatProperties"))
    vectors = _mapping(props.get("vectorProperties"))
    textures = _mapping(props.get("textureProperties"))
    keywords = _mapping(props.get("keywordMap"))

    def value(name: str, default: float) -> float:
        return _number(floats.get(name), default)

    def linear(name: str, default) -> list[float]:
        return [_srgb_to_linear(item) for item in _color(vectors.get(name), default, 3)]

    main = _color(vectors.get("_MainTex"), (0.0, 0.0, 1.0, 1.0), 4)
    transform = None
    if main != [0.0, 0.0, 1.0, 1.0]:
        transform = {"offset": [main[0], 1.0 - main[3] - main[1]], "scale": [main[2], main[3]]}

    def texture(name: str) -> dict | None:
        found = ctx.texture(textures.get(name))
        if found is None:
            return None
        info: dict = {"texture": found}
        if transform is not None:
            info["extensions"] = {"KHR_texture_transform": dict(transform)}
        return info

    transparent = bool(keywords.get("_ALPHABLEND_ON", False))
    zwrite = value("_ZWrite", 0.0) == 1.0
    queue = _number(props.get("renderQueue"), 2000.0)
    offset = 0
    if transparent:
        offset = int(min(max(queue - 2501, 0), 9)) if zwrite else int(min(max(queue - 3000, -9), 0))
    toony = value("_ShadeToony", 0.9)
    shift = value("_ShadeShift", 0.0)
    outline_mode = int(value("_OutlineWidthMode", 0.0))
    mtoon: dict = {
        "specVersion": "1.0",
        "transparentWithZWrite": transparent and zwrite,
        "renderQueueOffsetNumber": offset,
        "shadeColorFactor": linear("_ShadeColor", (0.97, 0.81, 0.86)),
        "shadingShiftFactor": -shift - (1.0 - toony),
        "shadingToonyFactor": toony + (1.0 - toony) * (0.5 + 0.5 * shift),
        "giEqualizationFactor": 1.0 - value("_IndirectLightIntensity", 0.1),
        "matcapFactor": [1.0, 1.0, 1.0] if textures.get("_SphereAdd") is not None else [0.0, 0.0, 0.0],
        "parametricRimColorFactor": linear("_RimColor", (0.0, 0.0, 0.0)),
        "parametricRimFresnelPowerFactor": value("_RimFresnelPower", 1.0),
        "parametricRimLiftFactor": value("_RimLift", 0.0),
        "rimLightingMixFactor": value("_RimLightingMix", 0.0),
        "outlineWidthMode": ("none", "worldCoordinates", "screenCoordinates")[outline_mode]
        if 0 <= outline_mode <= 2 else "none",
        "outlineWidthFactor": value("_OutlineWidth", 0.0) * 0.01,
        "outlineColorFactor": linear("_OutlineColor", (0.0, 0.0, 0.0)),
        "outlineLightingMixFactor": value("_OutlineLightingMix", 1.0)
        if value("_OutlineColorMode", 0.0) == 1.0 else 0.0,
        "uvAnimationScrollXSpeedFactor": value("_UvAnimScrollX", 0.0),
        "uvAnimationScrollYSpeedFactor": -value("_UvAnimScrollY", 0.0),
        "uvAnimationRotationSpeedFactor": value("_UvAnimRotation", 0.0),
    }
    for key, name in (("shadeMultiplyTexture", "_ShadeTexture"), ("matcapTexture", "_SphereAdd"),
                      ("rimMultiplyTexture", "_RimTexture"),
                      ("outlineWidthMultiplyTexture", "_OutlineWidthTexture"),
                      ("uvAnimationMaskTexture", "_UvAnimMaskTexture")):
        info = texture(name)
        if info is not None:
            mtoon[key] = info
    return mtoon


def _read_mtoon0(ctx: _Context, properties) -> None:
    """VRM 0.x material properties into ``extras["mtoon"]`` and ``extras["vrm0_material"]``."""
    entries = properties if isinstance(properties, list) else []
    by_name = {str(entry.get("name")): entry for entry in _items(entries)}
    for index, material in ctx.materials.items():
        props = entries[index] if index < len(entries) and isinstance(entries[index], dict) else None
        if material.name in by_name and (props is None or props.get("name") != material.name):
            props = by_name[material.name]
        if not isinstance(props, dict):
            continue
        raw = {key: value for key, value in props.items() if key != "textureProperties"}
        material.extras["vrm0_material"] = gltf._json_safe(raw)
        if str(props.get("shader", "")) == "VRM/MToon":
            material.extras["mtoon"] = _mtoon_from_vrm0(ctx, props)


def _read_meta1(ctx: _Context, spec: dict) -> None:
    """VRM 1.0 meta into ``model.meta`` and the thumbnail into ``extras["vrm_thumbnail"]``."""
    meta = ctx.model.meta
    for key in _META_STRINGS:
        if isinstance(spec.get(key), str):
            meta[key] = spec[key]
    for key in _META_LISTS:
        if isinstance(spec.get(key), list):
            meta[key] = [str(item) for item in spec[key] if isinstance(item, str)]
    for key in _META_FLAGS:
        if isinstance(spec.get(key), bool):
            meta[key] = spec[key]
    for key, allowed in _META_CHOICES.items():
        if spec.get(key) in allowed:
            meta[key] = spec[key]
    image = _index(spec.get("thumbnailImage"))
    if image is not None:
        try:
            thumbnail = ctx.doc._image(image)
        except GltfError as error:
            logger.warning("the VRM thumbnail was skipped: %s", error)
            thumbnail = None
        if thumbnail is not None:
            ctx.model.extras["vrm_thumbnail"] = thumbnail


def _read_meta0(ctx: _Context, spec: dict) -> None:
    """VRM 0.x meta mapped onto VRM 1.0 field names in ``model.meta``."""
    meta = ctx.model.meta
    for old, new in (("title", "name"), ("version", "version"), ("contactInformation", "contactInformation")):
        if isinstance(spec.get(old), str) and spec[old]:
            meta[new] = spec[old]
    for old, new in (("author", "authors"), ("reference", "references")):
        if isinstance(spec.get(old), str) and spec[old]:
            meta[new] = [spec[old]]
    users = {"OnlyAuthor": "onlyAuthor", "ExplicitlyLicensedPerson": "onlySeparatelyLicensedPerson",
             "Everyone": "everyone"}
    meta["avatarPermission"] = users.get(spec.get("allowedUserName"), "onlyAuthor")
    meta["allowExcessivelyViolentUsage"] = spec.get("violentUssageName") == "Allow"
    meta["allowExcessivelySexualUsage"] = spec.get("sexualUssageName") == "Allow"
    commercial = spec.get("commercialUssageName") == "Allow"
    meta["commercialUsage"] = "personalProfit" if commercial else "personalNonProfit"
    meta["allowPoliticalOrReligiousUsage"] = False
    meta["allowAntisocialOrHateUsage"] = False
    meta["creditNotation"] = "required"
    meta["allowRedistribution"] = False
    meta["modification"] = "prohibited"
    meta.update(_VRM0_LICENSES.get(str(spec.get("licenseName", "")), {}))
    meta["licenseUrl"] = LICENSE_URL
    other = spec.get("otherLicenseUrl") or spec.get("otherPermissionUrl")
    if isinstance(other, str) and other:
        meta["otherLicenseUrl"] = other
    meta["vrm0_meta"] = {key: value for key, value in spec.items()
                         if key != "texture" and isinstance(value, (str, bool, int, float))}
    thumbnail = ctx.texture(spec.get("texture"))
    if thumbnail is not None:
        ctx.model.extras["vrm_thumbnail"] = thumbnail


def _read_humanoid(ctx: _Context, bones: list[tuple[str, object]]) -> None:
    """Humanoid ``(bone, node)`` pairs into ``skeleton.humanoid``."""
    skeleton = ctx.model.skeleton
    humanoid: dict[str, int] = {}
    unknown, unplaced = [], []
    for bone, node in bones:
        if bone not in _PARENT_BONE:
            unknown.append(bone[:40])
            continue
        joint = ctx.joint(node)
        if joint < 0:
            unplaced.append(bone)
            continue
        humanoid[bone] = joint
    if unknown:
        logger.warning("the VRM humanoid names %d unknown bone(s) such as %r; they were skipped",
                       len(unknown), unknown[0])
    if unplaced:
        logger.warning("the VRM humanoid bone(s) %s name no node in the file; they were skipped",
                       ", ".join(unplaced[:8]))
    if skeleton is not None:
        skeleton.humanoid = humanoid


def _expression(name: str, spec: dict, binds: list) -> Expression:
    """An expression from its VRM 1.0 flags and resolved binds."""
    def override(key: str) -> str:
        value = spec.get(key, "none")
        return value if value in _OVERRIDES else "none"

    return Expression(name=name, binds=binds, is_binary=bool(spec.get("isBinary", False)),
                      override_blink=override("overrideBlink"),
                      override_look_at=override("overrideLookAt"),
                      override_mouth=override("overrideMouth"))


def _read_expressions1(ctx: _Context, spec: dict) -> None:
    """VRM 1.0 preset and custom expressions into ``model.expressions``."""
    material_binds = {}
    for group in ("preset", "custom"):
        for name, entry in _mapping(spec.get(group)).items():
            if not isinstance(entry, dict) or not isinstance(name, str) or not name:
                continue
            binds = []
            for bind in _items(entry.get("morphTargetBinds")):
                node = _index(bind.get("node"))
                parts = ctx.node_map.parts_of_node.get(node, []) if node is not None else []
                weight = min(max(_number(bind.get("weight"), 0.0), 0.0), 1.0)
                binds.extend(ctx.morph_binds(parts, bind.get("index"), weight))
            ctx.model.expressions[name] = _expression(name, entry, binds)
            colors = []
            for bind in _items(entry.get("materialColorBinds")):
                material = ctx.material_name(bind.get("material"))
                if material is not None and bind.get("type") in _COLOR_BIND_TYPES:
                    colors.append({"material": material, "type": bind["type"],
                                   "targetValue": _color(bind.get("targetValue"), (0.0, 0.0, 0.0, 1.0), 4)})
            transforms = []
            for bind in _items(entry.get("textureTransformBinds")):
                material = ctx.material_name(bind.get("material"))
                if material is not None:
                    transforms.append({"material": material,
                                       "scale": _color(bind.get("scale"), (1.0, 1.0), 2),
                                       "offset": _color(bind.get("offset"), (0.0, 0.0), 2)})
            if colors or transforms:
                material_binds[name] = {"materialColorBinds": colors, "textureTransformBinds": transforms}
    if material_binds:
        ctx.model.extras["vrm_expression_binds"] = material_binds


def _read_expressions0(ctx: _Context, spec: dict) -> None:
    """VRM 0.x blend shape groups into ``model.expressions`` under VRM 1.0 names."""
    material_binds = {}
    for index, group in enumerate(_items(spec.get("blendShapeGroups"))):
        preset = _VRM0_PRESETS.get(str(group.get("presetName", "")).lower())
        name = preset or str(group.get("name") or f"expression_{index}")
        name = _unique(name, ctx.model.expressions)
        binds = []
        for bind in _items(group.get("binds")):
            mesh = _index(bind.get("mesh"))
            parts = ctx.node_map.parts_of_mesh.get(mesh, []) if mesh is not None else []
            weight = min(max(_number(bind.get("weight"), 100.0) / 100.0, 0.0), 1.0)
            binds.extend(ctx.morph_binds(parts, bind.get("index"), weight))
        ctx.model.expressions[name] = Expression(name=name, binds=binds,
                                                 is_binary=bool(group.get("isBinary", False)))
        colors, transforms = [], []
        for value in _items(group.get("materialValues")):
            material = value.get("materialName")
            if not isinstance(material, str):
                continue
            target = _color(value.get("targetValue"), (0.0, 0.0, 1.0, 1.0), 4)
            prop = value.get("propertyName")
            if prop in _VRM0_COLOR_BINDS:
                linear = [_srgb_to_linear(item) for item in target[:3]] + [target[3]]
                colors.append({"material": material, "type": _VRM0_COLOR_BINDS[prop], "targetValue": linear})
            elif prop == "_MainTex_ST":
                transforms.append({"material": material, "scale": target[:2],
                                   "offset": [target[2], 1.0 - target[1] - target[3]]})
        if colors or transforms:
            material_binds[name] = {"materialColorBinds": colors, "textureTransformBinds": transforms}
    if material_binds:
        ctx.model.extras["vrm_expression_binds"] = material_binds


def _range_map(spec, default_scale: float) -> dict:
    """A VRM 1.0 look-at range map with finite values."""
    spec = _mapping(spec)
    return {"inputMaxValue": max(_number(spec.get("inputMaxValue"), 90.0), 0.0),
            "outputScale": max(_number(spec.get("outputScale"), default_scale), 0.0)}


def _read_look1(ctx: _Context, vrm: dict) -> None:
    """VRM 1.0 look-at and first-person settings into ``model.extras``."""
    look = _mapping(vrm.get("lookAt"))
    if look:
        kind = look.get("type") if look.get("type") in ("bone", "expression") else "bone"
        scale = 10.0 if kind == "bone" else 1.0
        stored: dict = {"type": kind, "offsetFromHeadBone": _vector(look.get("offsetFromHeadBone"))}
        for key, _ in _RANGE_MAPS:
            stored[key] = _range_map(look.get(key), scale)
        ctx.model.extras["vrm_look_at"] = stored
    annotations = []
    for entry in _items(_mapping(vrm.get("firstPerson")).get("meshAnnotations")):
        kind = entry.get("type")
        node = _index(entry.get("node"))
        if kind in _FIRST_PERSON and node is not None:
            annotations.extend({"part": part, "type": kind}
                               for part in ctx.node_map.parts_of_node.get(node, []))
    if annotations:
        ctx.model.extras["vrm_first_person"] = annotations


def _read_look0(ctx: _Context, spec: dict) -> None:
    """VRM 0.x first-person and look-at settings into ``model.extras`` in VRM 1.0 form."""
    kind = "expression" if str(spec.get("lookAtTypeName", "Bone")) == "BlendShape" else "bone"
    offset = _vector(spec.get("firstPersonBoneOffset"))
    skeleton = ctx.model.skeleton
    source = ctx.joint(spec.get("firstPersonBone"))
    head = skeleton.humanoid.get("head", -1) if skeleton is not None else -1
    if source >= 0 and head >= 0 and source != head:
        # Re-expresses the offset from the first-person bone in the head's frame.
        rest = skeleton.rest.double()
        point = rest[source] @ torch.tensor([*offset, 1.0], dtype=torch.float64)
        offset = [float(value) for value in (torch.linalg.inv(rest[head]) @ point)[:3]]
    stored: dict = {"type": kind, "offsetFromHeadBone": offset}
    for key, old in _RANGE_MAPS:
        legacy = _mapping(spec.get(old))
        stored[key] = _range_map({"inputMaxValue": legacy.get("xRange"), "outputScale": legacy.get("yRange")},
                                 10.0 if kind == "bone" else 1.0)
    ctx.model.extras["vrm_look_at"] = stored
    annotations = []
    for entry in _items(spec.get("meshAnnotations")):
        kind = _VRM0_FIRST_PERSON.get(str(entry.get("firstPersonFlag", "")).lower())
        mesh = _index(entry.get("mesh"))
        if kind is not None and mesh is not None:
            annotations.extend({"part": part, "type": kind}
                               for part in ctx.node_map.parts_of_mesh.get(mesh, []))
    if annotations:
        ctx.model.extras["vrm_first_person"] = annotations


def _spring_params(spec: dict) -> dict:
    """One VRM 1.0 spring joint's settings with defaults filled in."""
    direction = _vector(spec.get("gravityDir"), _SPRING_DEFAULTS["gravityDir"])
    return {
        "hitRadius": max(_number(spec.get("hitRadius"), 0.0), 0.0),
        "stiffness": max(_number(spec.get("stiffness"), 1.0), 0.0),
        "gravityPower": max(_number(spec.get("gravityPower"), 0.0), 0.0),
        "gravityDir": direction,
        "dragForce": min(max(_number(spec.get("dragForce"), 0.5), 0.0), 1.0),
    }


def _chain(name: str, joints: list[int], params: dict, groups: list[int]) -> SpringChain:
    """A spring chain from one joint's settings."""
    return SpringChain(
        name=name, joints=joints, stiffness=params["stiffness"], drag=params["dragForce"],
        gravity_power=params["gravityPower"],
        gravity_dir=torch.tensor(params["gravityDir"], dtype=torch.float32),
        hit_radius=params["hitRadius"], collider_groups=groups,
    )


def _read_springs1(ctx: _Context, spec: dict) -> None:
    """VRMC_springBone colliders, groups and springs into ``model.springs``."""
    springs = Springs()
    collider_of: dict[int, int] = {}
    for index, entry in enumerate(spec.get("colliders", []) if isinstance(spec.get("colliders"), list) else []):
        if not isinstance(entry, dict):
            continue
        joint = ctx.joint(entry.get("node"))
        shape = _mapping(entry.get("shape"))
        if joint < 0:
            logger.warning("spring collider %d names no node in the file; it was skipped", index)
            continue
        if isinstance(shape.get("sphere"), dict):
            kind, body = "sphere", shape["sphere"]
        elif isinstance(shape.get("capsule"), dict):
            kind, body = "capsule", shape["capsule"]
        else:
            continue
        collider_of[index] = len(springs.colliders)
        springs.colliders.append(Collider(
            joint=joint, shape=kind,
            offset=torch.tensor(_vector(body.get("offset")), dtype=torch.float32),
            tail=torch.tensor(_vector(body.get("tail")) if kind == "capsule" else [0.0, 0.0, 0.0],
                              dtype=torch.float32),
            radius=max(_number(body.get("radius"), 0.0), 0.0),
        ))
    group_of: dict[int, int] = {}
    groups = spec.get("colliderGroups") if isinstance(spec.get("colliderGroups"), list) else []
    for index, entry in enumerate(groups):
        if not isinstance(entry, dict):
            continue
        members = []
        for collider in _list(entry.get("colliders")):
            mapped = collider_of.get(_index(collider), -1)
            if mapped >= 0 and mapped not in members:
                members.append(mapped)
        group_of[index] = len(springs.collider_groups)
        springs.collider_groups.append(members)
        springs.group_names.append(str(entry.get("name") or f"colliders_{index}"))
    settings = []
    for index, entry in enumerate(_items(spec.get("springs"))):
        joints, params = [], []
        for joint_spec in _items(entry.get("joints")):
            joint = ctx.joint(joint_spec.get("node"))
            if joint < 0:
                logger.warning("spring %d names a joint node missing from the file; the chain was cut "
                               "there", index)
                break
            joints.append(joint)
            params.append(_spring_params(joint_spec))
        if not joints:
            continue
        chain_groups = [group_of[group] for group in _list(entry.get("colliderGroups"))
                        if _index(group) in group_of]
        name = str(entry.get("name") or f"spring_{index}")
        springs.chains.append(_chain(name, joints, params[0], chain_groups))
        varied = any(item != params[0] for item in params[1:])
        settings.append({"center": ctx.joint(entry.get("center")), "joints": params if varied else None})
    if springs.chains or springs.colliders:
        ctx.model.springs = springs
        ctx.model.extras["vrm_spring_settings"] = settings


def _unity_vector(value, default=(0.0, 0.0, 0.0)) -> list[float]:
    """A VRM 0.x secondary animation vector, stored on Unity's axes, on the file's glTF axes."""
    x, y, z = _vector(value, default)
    return [x, y, 0.0 - z]


def _read_springs0(ctx: _Context, spec: dict) -> None:
    """VRM 0.x secondary animation into ``model.springs``, one chain per branch of each root bone."""
    springs = Springs()
    groups = spec.get("colliderGroups") if isinstance(spec.get("colliderGroups"), list) else []
    for index, group in enumerate(groups):
        group = _mapping(group)
        joint = ctx.joint(group.get("node"))
        members = []
        if joint >= 0:
            for collider in _items(group.get("colliders")):
                if "offset" not in collider or "radius" not in collider:
                    continue
                members.append(len(springs.colliders))
                springs.colliders.append(Collider(
                    joint=joint, shape="sphere",
                    offset=torch.tensor(_unity_vector(collider["offset"]), dtype=torch.float32),
                    tail=torch.zeros(3), radius=max(_number(collider["radius"], 0.0), 0.0),
                ))
        springs.collider_groups.append(members)
        springs.group_names.append(ctx.node_name(group.get("node")) or f"colliders_{index}")
    skeleton = ctx.model.skeleton
    tips: list[tuple[int, torch.Tensor]] = []
    claimed: set[int] = set()
    names: set[str] = set()
    settings = []
    group_count = len(springs.collider_groups)
    for index, group in enumerate(_items(spec.get("boneGroups"))):
        given = {key: group.get(key, default) for key, default in _VRM0_SPRING_DEFAULTS.items()}
        if "stiffiness" not in group and "stiffness" in group:
            given["stiffiness"] = group["stiffness"]
        # Gravity's length is folded into its power, leaving a unit direction.
        gravity = _unity_vector(group.get("gravityDir"), (0.0, -1.0, 0.0))
        strength = math.sqrt(sum(value * value for value in gravity))
        params = _spring_params({
            "hitRadius": given["hitRadius"], "stiffness": given["stiffiness"], "dragForce": given["dragForce"],
            "gravityPower": _number(given["gravityPower"], 0.0) * strength,
            "gravityDir": [value / strength for value in gravity] if strength > 1e-9 else None,
        })
        listed = (_index(item) for item in _list(group.get("colliderGroups")))
        chain_groups = sorted({item for item in listed if item is not None and item < group_count})
        comment = str(group.get("comment") or "")
        center = ctx.joint(group.get("center"))
        for root in _list(group.get("bones")):
            for joints, leaf in _branches(ctx, root, claimed):
                # A chain ending on a bone with no child gains a ``<bone>_end`` tip joint past it.
                offset = _tip_offset(ctx, leaf) if leaf is not None else None
                if offset is not None:
                    joints.append(skeleton.count + len(tips))
                    tips.append((joints[-2], offset))
                if len(joints) < 2:
                    continue
                name = _unique(comment or skeleton.names[joints[0]] or f"spring_{index}", names)
                names.add(name)
                springs.chains.append(_chain(name, joints, params, list(chain_groups)))
                settings.append({"center": center, "joints": None})
    if tips:
        _add_tips(skeleton, tips)
    if springs.chains or springs.colliders:
        ctx.model.springs = springs
        ctx.model.extras["vrm_spring_settings"] = settings


def _branches(ctx: _Context, root, claimed: set[int]) -> list[tuple[list[int], int | None]]:
    """Joint chains under a VRM 0.x spring root; first children continue a chain, other children start one.

    Returns:
        ``(joints, leaf)`` per chain: ``leaf`` is the glTF node of the childless bone it ends on, or
        None where it ends on a joint another chain holds.
    """
    start = _index(root)
    chains = []
    stack = [start] if start is not None else []
    while stack:
        node = stack.pop()
        if ctx.joint(node) < 0 or ctx.joint(node) in claimed:
            continue
        chain, leaf = [], None
        while True:
            claimed.add(ctx.joint(node))
            chain.append(ctx.joint(node))
            children = [child for child in ctx.graph.children[node] if ctx.joint(child) >= 0]
            stack.extend(children[1:][::-1])
            if not children:
                leaf = node
                break
            node = children[0]
            if ctx.joint(node) in claimed:
                chain.append(ctx.joint(node))
                break
        chains.append((chain, leaf))
    return chains


def _tip_offset(ctx: _Context, node: int) -> torch.Tensor | None:
    """`_VRM0_TIP` along a glTF node's own translation, in the node's frame; None when it has none."""
    offset = torch.from_numpy(np.asarray(ctx.graph.local[node], dtype=np.float64)[:3, 3].copy())
    length = float(offset.norm())
    return offset * (_VRM0_TIP / length) if length > 1e-9 else None


def _add_tips(skeleton: Skeleton, tips: list[tuple[int, torch.Tensor]]) -> None:
    """Append one ``<joint>_end`` joint per ``(joint, offset)``, at ``offset`` in that joint's rest frame."""
    rest = skeleton.rest.double()
    taken = set(skeleton.names)
    placed = []
    for joint, offset in tips:
        place = torch.eye(4, dtype=torch.float64)
        place[:3, 3] = offset
        placed.append(rest[joint] @ place)
        name = _unique(f"{skeleton.names[joint]}_end", taken)
        taken.add(name)
        skeleton.names.append(name)
    placed = torch.stack(placed)
    parents = torch.tensor([joint for joint, _ in tips], dtype=skeleton.parents.dtype)
    skeleton.parents = torch.cat([skeleton.parents, parents])
    skeleton.rest = torch.cat([skeleton.rest, placed.to(skeleton.rest.dtype)])
    inverse = torch.linalg.inv(placed).to(skeleton.inverse_bind.dtype)
    skeleton.inverse_bind = torch.cat([skeleton.inverse_bind, inverse])


def _read_constraints(ctx: _Context) -> None:
    """VRMC_node_constraint entries into ``extras["vrm_node_constraints"]`` by joint index."""
    found = []
    for index, node in enumerate(ctx.graph.nodes):
        spec = _mapping(_mapping(node.get("extensions")).get("VRMC_node_constraint"))
        for kind, body in _mapping(spec.get("constraint")).items():
            if kind not in ("roll", "aim", "rotation") or not isinstance(body, dict):
                continue
            joint, source = ctx.joint(index), ctx.joint(body.get("source"))
            if joint < 0 or source < 0:
                continue
            kept = gltf._json_safe(body)
            kept["source"] = source
            found.append({"joint": joint, "constraint": {kind: kept}})
    if found:
        ctx.model.extras["vrm_node_constraints"] = found


def _faces_negative_z(skeleton: Skeleton | None) -> bool:
    """Whether the humanoid's left side lies on -X, so the model faces -Z; True when unknown."""
    if skeleton is None:
        return True
    heads = skeleton.heads()
    for left, right in (("leftUpperLeg", "rightUpperLeg"), ("leftUpperArm", "rightUpperArm")):
        if left in skeleton.humanoid and right in skeleton.humanoid:
            side = heads[skeleton.humanoid[left]] - heads[skeleton.humanoid[right]]
            return bool(side[0] < 0)
    return True


def _turn(model: Model3D) -> None:
    """Rotate a whole model 180 degrees about Y, conjugating joint frames so their axes stay put."""
    sign = torch.tensor([-1.0, 1.0, -1.0])
    for part in model.parts:
        part.vertices = part.vertices * sign
        if part.normals is not None:
            part.normals = part.normals * sign
        if part.tangents is not None:
            part.tangents = torch.cat([part.tangents[:, :3] * sign, part.tangents[:, 3:]], dim=1)
        for morph in part.morphs.values():
            morph.positions = morph.positions * sign
            if morph.normals is not None:
                morph.normals = morph.normals * sign
    skeleton = model.skeleton
    if skeleton is not None:
        flip = torch.from_numpy(np.outer(_TURN, _TURN).astype(np.float32))
        skeleton.rest = skeleton.rest * flip
        skeleton.inverse_bind = skeleton.inverse_bind * flip
    if model.springs is not None:
        # Adding zero turns the -0.0 a sign flip leaves into 0.0.
        for collider in model.springs.colliders:
            collider.offset = collider.offset * sign + 0.0
            collider.tail = collider.tail * sign + 0.0
        for chain in model.springs.chains:
            chain.gravity_dir = chain.gravity_dir * sign + 0.0
    for settings in model.extras.get("vrm_spring_settings", []) or []:
        for params in settings.get("joints") or []:
            params["gravityDir"] = [0.0 - params["gravityDir"][0], params["gravityDir"][1],
                                    0.0 - params["gravityDir"][2]]
    look = model.extras.get("vrm_look_at")
    if isinstance(look, dict):
        offset = look["offsetFromHeadBone"]
        look["offsetFromHeadBone"] = [0.0 - offset[0], offset[1], 0.0 - offset[2]]
    quat_sign = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    for clip in model.animations:
        for channel in clip.channels:
            if channel.path == "translation":
                channel.values = channel.values * sign
            elif channel.path == "rotation":
                channel.values = channel.values * quat_sign


@gltf._damaged
def read(data: bytes, base_dir: str | None = None, name: str = "") -> Model3D:
    """Read a VRM 0.x or 1.0 file into a model facing +Z.

    Args:
        data: The file's bytes.
        base_dir: Folder holding the file, for buffers and images it names.
        name: Model name used when the file's meta gives none.

    Returns:
        The model, with ``source_format`` ``"vrm"``.

    Raises:
        GltfError: The file is not glTF 2.0, is damaged, or exceeds the model limits.
    """
    doc = gltf.parse(data, base_dir)
    document = doc.json
    extensions = _mapping(document.get("extensions"))
    if "VRMC_vrm" in extensions:
        version = "1.0"
    elif "VRM" in extensions:
        version = "0.x"
    else:
        logger.info("the file carries no VRM extension and was read as plain glTF")
        return gltf.to_model(doc, name)[0]
    added = _attach_nodes(document, _wanted_nodes(document, version))
    model, node_map = gltf.to_model(doc, name)
    if added:
        document["skins"].pop()
    graph = gltf._Graph(document)
    _settle_joints(model, node_map, graph, added)
    ctx = _Context(doc, model, node_map, graph)
    if version == "1.0":
        vrm = _mapping(extensions["VRMC_vrm"])
        _read_meta1(ctx, _mapping(vrm.get("meta")))
        human = _mapping(_mapping(vrm.get("humanoid")).get("humanBones"))
        _read_humanoid(ctx, [(str(bone), _mapping(spec).get("node")) for bone, spec in human.items()])
        _read_expressions1(ctx, _mapping(vrm.get("expressions")))
        _read_look1(ctx, vrm)
        if isinstance(extensions.get("VRMC_springBone"), dict):
            _read_springs1(ctx, extensions["VRMC_springBone"])
        _read_mtoon1(ctx)
        _read_constraints(ctx)
        model.extras["vrm_spec_version"] = str(vrm.get("specVersion") or "1.0")
    else:
        vrm = _mapping(extensions["VRM"])
        _read_meta0(ctx, _mapping(vrm.get("meta")))
        bones = []
        for spec in _items(_mapping(vrm.get("humanoid")).get("humanBones")):
            bone = str(spec.get("bone", ""))
            bones.append((_VRM0_BONES.get(bone, bone), spec.get("node")))
        _read_humanoid(ctx, bones)
        _read_expressions0(ctx, _mapping(vrm.get("blendShapeMaster")))
        _read_look0(ctx, _mapping(vrm.get("firstPerson")))
        _read_springs0(ctx, _mapping(vrm.get("secondaryAnimation")))
        _read_mtoon0(ctx, vrm.get("materialProperties"))
        model.extras["vrm_spec_version"] = str(vrm.get("specVersion") or "0.0")
        if _faces_negative_z(model.skeleton):
            _turn(model)
    if name and not model.meta.get("name"):
        model.meta["name"] = name
    model.source_format = "vrm"
    return model


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _checked_humanoid(skeleton: Skeleton) -> dict[str, int]:
    """The humanoid map with unknown bones dropped, refusing a joint mapped twice.

    Raises:
        VrmError: Two humanoid bones name the same joint.
    """
    humanoid: dict[str, int] = {}
    owner: dict[int, str] = {}
    for bone in HUMAN_BONES:
        joint = skeleton.humanoid.get(bone)
        if not isinstance(joint, int) or isinstance(joint, bool) or not 0 <= joint < skeleton.count:
            continue
        if joint in owner:
            raise VrmError(f"the humanoid bones {owner[joint]} and {bone} both name the joint "
                           f"{skeleton.names[joint]!r}; VRM needs one joint per bone. Map one of them "
                           f"to another joint before saving.")
        owner[joint] = bone
        humanoid[bone] = joint
    unknown = sorted(set(skeleton.humanoid) - set(_PARENT_BONE))
    if unknown:
        logger.warning("humanoid bones %s are not VRM 1.0 bones and were left out", ", ".join(unknown[:8]))
    return humanoid


def _check_layout(skeleton: Skeleton, humanoid: dict[str, int]) -> None:
    """Log humanoid bones that do not descend from their VRM parent bone, and a model not facing +Z."""
    parents = [int(value) for value in skeleton.parents]

    def descends(joint: int, ancestor: int) -> bool:
        current = parents[joint]
        while current >= 0:
            if current == ancestor:
                return True
            current = parents[current]
        return False

    misplaced = []
    for bone, joint in humanoid.items():
        parent = _PARENT_BONE[bone]
        while parent is not None and parent not in humanoid:
            parent = _PARENT_BONE[parent]
        if parent is not None and not descends(joint, humanoid[parent]):
            misplaced.append(bone)
    if misplaced:
        logger.warning("humanoid bones %s do not sit below their VRM parent bones in the skeleton; "
                       "VRM viewers may pose them wrongly. Check the humanoid mapping.",
                       ", ".join(misplaced[:8]))
    heads = skeleton.heads()
    side = heads[humanoid["leftUpperLeg"]] - heads[humanoid["rightUpperLeg"]]
    if float(side[0]) < 0.5 * float(side.norm()):
        logger.warning("the model's left leg is not on +X, so it does not face +Z; VRM viewers will "
                       "show it turned. Rotate the model to face +Z before saving.")


def _bake_part(part: Part, skin: torch.Tensor, count: int) -> Part:
    """A skinned part with its rest-pose skinning baked into its vertices, normals and shape keys."""
    vertices = part.vertices.double()
    joints = part.joints.long().clamp(0, count - 1)
    weights = part.weights.double().clamp_min(0)
    total = weights.sum(dim=1, keepdim=True)
    weights = torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))
    weights[:, 0] = torch.where(total[:, 0] > 0, weights[:, 0], torch.ones_like(weights[:, 0]))
    new_vertices = torch.empty_like(vertices)
    normals = part.normals.double() if part.normals is not None else None
    tangents = part.tangents.double().clone() if part.tangents is not None else None
    morphs = {name: [morph.positions.double().clone(),
                     morph.normals.double().clone() if morph.normals is not None else None]
              for name, morph in part.morphs.items()}
    new_normals = torch.empty_like(normals) if normals is not None else None
    step = 65536
    for start in range(0, vertices.shape[0], step):
        rows = slice(start, start + step)
        blend = (weights[rows, :, None, None] * skin[joints[rows]]).sum(dim=1)
        linear = blend[:, :3, :3]
        new_vertices[rows] = (linear @ vertices[rows, :, None])[..., 0] + blend[:, :3, 3]
        determinant = torch.linalg.det(linear)
        safe = torch.where(determinant.abs()[:, None, None] > 1e-12, linear,
                           torch.eye(3, dtype=linear.dtype).expand_as(linear))
        normal_matrix = torch.linalg.inv(safe).transpose(1, 2)
        if normals is not None:
            new_normals[rows] = torch.nn.functional.normalize(
                (normal_matrix @ normals[rows, :, None])[..., 0], dim=1)
        if tangents is not None:
            tangents[rows, :3] = torch.nn.functional.normalize(
                (linear @ tangents[rows, :3, None])[..., 0], dim=1)
        for pair in morphs.values():
            pair[0][rows] = (linear @ pair[0][rows, :, None])[..., 0]
            if pair[1] is not None:
                pair[1][rows] = (normal_matrix @ pair[1][rows, :, None])[..., 0]
    return dataclasses.replace(
        part, vertices=new_vertices.float(),
        normals=new_normals.float() if new_normals is not None else None,
        tangents=tangents.float() if tangents is not None else None,
        morphs={name: dataclasses.replace(part.morphs[name], positions=pair[0].float(),
                                          normals=pair[1].float() if pair[1] is not None else None)
                for name, pair in morphs.items()},
    )


def _flattened(skeleton: Skeleton, humanoid: dict[str, int]) -> torch.Tensor:
    """``(J,)`` bool, True for humanoid joints and every joint above one."""
    flat = torch.zeros(skeleton.count, dtype=torch.bool)
    for joint in humanoid.values():
        while joint >= 0 and not bool(flat[joint]):
            flat[joint] = True
            joint = int(skeleton.parents[joint])
    return flat


def _constraint(entry, count: int) -> tuple[int, str, dict, int] | None:
    """``(joint, kind, settings, source joint)`` of one kept node constraint, or None when malformed."""
    joint = entry.get("joint") if isinstance(entry, dict) else None
    constraint = _mapping(entry.get("constraint")) if isinstance(entry, dict) else {}
    if len(constraint) != 1 or not isinstance(joint, int) or not 0 <= joint < count:
        return None
    kind, body = next(iter(constraint.items()))
    source = _mapping(body).get("source")
    if kind not in ("roll", "aim", "rotation") or not isinstance(source, int) or not 0 <= source < count:
        return None
    return joint, kind, body, source


def _written_rest(model: Model3D, flat: torch.Tensor) -> torch.Tensor:
    """``(J, 4, 4)`` written rest: ``flat`` joints as translations, rotation-constraint targets turned with their source."""
    skeleton = model.skeleton
    rest = skeleton.rest.double()
    written = rest.clone()
    written[flat] = torch.eye(4, dtype=torch.float64)
    written[flat, :3, 3] = rest[flat, :3, 3]
    for entry in _items(model.extras.get("vrm_node_constraints")):
        found = _constraint(entry, skeleton.count)
        if found is None or found[1] != "rotation" or bool(flat[found[0]]) or not bool(flat[found[3]]):
            continue
        joint, _, _, source = found
        change = torch.linalg.inv(written[source, :3, :3]) @ rest[source, :3, :3]
        written[joint, :3, :3] = rest[joint, :3, :3] @ torch.linalg.inv(change)
    return written


def _bind_pose(model: Model3D, flat: torch.Tensor, written: torch.Tensor) -> tuple[Skeleton, list[Part]]:
    """A skeleton resting at ``written`` with ``flat`` joints exact, and parts skinned to it."""
    skeleton = model.skeleton
    count = skeleton.count
    rest = skeleton.rest.double()
    heads = rest[:, :3, 3]
    eye = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    new_rest = written
    new_inverse = torch.linalg.inv(new_rest)
    new_inverse[flat] = eye[flat]
    new_inverse[flat, :3, 3] = -heads[flat]
    skin = rest @ skeleton.inverse_bind.double()
    drift = (skin - torch.eye(4, dtype=torch.float64)).abs().amax(dim=(1, 2))
    tolerance = 1e-5 * max(1.0, float(heads.abs().max()) if count else 1.0)
    parts = []
    baked = 0
    for part in model.parts:
        if part.joints is None or part.weights is None or part.vertex_count == 0:
            parts.append(part)
            continue
        used = part.joints[part.weights > 0].long().clamp(0, count - 1)
        if used.numel() == 0 or float(drift[used].max()) <= tolerance:
            parts.append(part)
            continue
        parts.append(_bake_part(part, skin, count))
        baked += 1
    if baked:
        logger.info("%d part(s) were not in their bind pose and were baked into it for VRM", baked)
    normal = Skeleton(names=list(skeleton.names), parents=skeleton.parents.clone(),
                      rest=new_rest.float(), inverse_bind=new_inverse.float(),
                      humanoid=dict(skeleton.humanoid))
    return normal, parts


def _pin_joints(out: gltf.GltfBuild, skeleton: Skeleton, flat: torch.Tensor) -> None:
    """Write each ``flat`` joint node as a translation from its parent's head, with no rotation or scale."""
    heads = skeleton.rest.double()[:, :3, 3]
    nodes = out.json["nodes"]
    for joint, node_index in enumerate(out.joint_nodes):
        if not bool(flat[joint]):
            continue
        node = nodes[node_index]
        for key in ("matrix", "rotation", "scale", "translation"):
            node.pop(key, None)
        parent = int(skeleton.parents[joint])
        offset = heads[joint] - (heads[parent] if parent >= 0 else 0.0)
        if float(offset.abs().max()) > 0:
            node["translation"] = [float(value) for value in offset]


def _write_meta(out: gltf.GltfBuild, model: Model3D) -> dict:
    """VRM 1.0 meta from ``model.meta`` with every enum checked and restrictive defaults."""
    meta = model.meta
    result: dict = {}
    name = meta.get("name") or meta.get("title")
    result["name"] = str(name) if isinstance(name, str) and name else "Untitled"
    authors = meta.get("authors")
    if isinstance(authors, str):
        authors = [authors]
    if not isinstance(authors, list):
        authors = []
    authors = [str(item) for item in authors if isinstance(item, str) and item]
    if not authors and isinstance(meta.get("author"), str) and meta["author"]:
        authors = [meta["author"]]
    if not authors:
        logger.warning("the model's meta names no authors; VRM requires one, so 'Unknown' was written. "
                       "Set meta authors before saving.")
        authors = ["Unknown"]
    result["authors"] = authors
    for key in _META_STRINGS[1:]:
        if isinstance(meta.get(key), str) and meta[key]:
            result[key] = meta[key]
    if "copyrightInformation" not in result and isinstance(meta.get("copyright"), str) and meta["copyright"]:
        result["copyrightInformation"] = meta["copyright"]
    result.setdefault("licenseUrl", LICENSE_URL)
    references = meta.get("references")
    if isinstance(references, str):
        references = [references]
    if isinstance(references, list):
        kept = [str(item) for item in references if isinstance(item, str) and item]
        if kept:
            result["references"] = kept
    for key in _META_FLAGS:
        value = meta.get(key, False)
        if not isinstance(value, bool):
            logger.warning("meta %s must be true or false; false was written", key)
            value = False
        result[key] = value
    for key, allowed in _META_CHOICES.items():
        value = meta.get(key, allowed[0])
        if value not in allowed:
            logger.warning("meta %s %r is not a VRM 1.0 value (%s); %s was written", key, str(value)[:40],
                           ", ".join(allowed), allowed[0])
            value = allowed[0]
        result[key] = value
    thumbnail = model.extras.get("vrm_thumbnail")
    if isinstance(thumbnail, Texture):
        result["thumbnailImage"] = out.add_image(thumbnail)
    return result


def _material_indices(out: gltf.GltfBuild, model: Model3D) -> tuple[dict[str, int], dict[int, Material]]:
    """Written material index by material name, and the material behind each written index."""
    by_name: dict[str, int] = {}
    by_index: dict[int, Material] = {}
    for part, index in zip(model.parts, out.material_of_part):
        if index >= 0:
            by_name.setdefault(part.material.name, index)
            by_index.setdefault(index, part.material)
    return by_name, by_index


def _write_expressions(out: gltf.GltfBuild, model: Model3D) -> dict:
    """VRM 1.0 preset and custom expressions from ``model.expressions``."""
    by_name, _ = _material_indices(out, model)
    extra = _mapping(model.extras.get("vrm_expression_binds"))
    preset: dict = {}
    custom: dict = {}
    skipped = 0
    for name, expression in model.expressions.items():
        if not isinstance(name, str) or not name:
            continue
        entry: dict = {
            "isBinary": bool(expression.is_binary),
            "overrideBlink": expression.override_blink if expression.override_blink in _OVERRIDES else "none",
            "overrideLookAt": expression.override_look_at if expression.override_look_at in _OVERRIDES else "none",
            "overrideMouth": expression.override_mouth if expression.override_mouth in _OVERRIDES else "none",
        }
        morph_binds = []
        for part_index, morph_name, weight in expression.binds:
            if not isinstance(part_index, int) or not 0 <= part_index < len(model.parts):
                skipped += 1
                continue
            node = out.part_nodes[part_index]
            names = list(model.parts[part_index].morphs)
            if node < 0 or morph_name not in names:
                skipped += 1
                continue
            morph_binds.append({"node": node, "index": names.index(morph_name),
                                "weight": min(max(_number(weight, 0.0), 0.0), 1.0)})
        if morph_binds:
            entry["morphTargetBinds"] = morph_binds
        binds = _mapping(extra.get(name))
        colors = [{"material": by_name[bind["material"]], "type": bind["type"],
                   "targetValue": _color(bind.get("targetValue"), (0.0, 0.0, 0.0, 1.0), 4)}
                  for bind in _items(binds.get("materialColorBinds"))
                  if bind.get("material") in by_name and bind.get("type") in _COLOR_BIND_TYPES]
        transforms = [{"material": by_name[bind["material"]],
                       "scale": _color(bind.get("scale"), (1.0, 1.0), 2),
                       "offset": _color(bind.get("offset"), (0.0, 0.0), 2)}
                      for bind in _items(binds.get("textureTransformBinds")) if bind.get("material") in by_name]
        if colors:
            entry["materialColorBinds"] = colors
        if transforms:
            entry["textureTransformBinds"] = transforms
        (preset if name in EXPRESSION_PRESETS else custom)[name] = entry
    if skipped:
        logger.warning("%d expression bind(s) name a part or shape key the model does not hold and "
                       "were left out", skipped)
    result = {}
    if preset:
        result["preset"] = preset
    if custom:
        result["custom"] = custom
    return result


def _write_look_at(model: Model3D, skeleton: Skeleton, humanoid: dict[str, int],
                   original: Skeleton) -> dict:
    """VRM 1.0 look-at settings: kept ones when valid, otherwise defaults for bone or expression gaze."""
    stored = _mapping(model.extras.get("vrm_look_at"))
    eyes = "leftEye" in humanoid and "rightEye" in humanoid
    kind = stored.get("type")
    if kind not in ("bone", "expression"):
        kind = "bone" if eyes else "expression"
    if kind == "bone" and not eyes:
        kind = "expression"
    heads = skeleton.heads()
    if "offsetFromHeadBone" in stored:
        # Re-expresses the kept offset in the written head frame.
        change = _frame_changes(original, skeleton)[humanoid["head"]]
        turned = change @ torch.tensor(_vector(stored["offsetFromHeadBone"]), dtype=torch.float64)
        offset = [float(value) + 0.0 for value in turned]
    elif eyes:
        middle = (heads[humanoid["leftEye"]] + heads[humanoid["rightEye"]]) * 0.5 - heads[humanoid["head"]]
        offset = [float(value) for value in middle]
    else:
        offset = [0.0, 0.06, 0.0]
    scale = 10.0 if kind == "bone" else 1.0
    result: dict = {"type": kind, "offsetFromHeadBone": offset}
    for key, _ in _RANGE_MAPS:
        result[key] = _range_map(stored.get(key) if stored.get("type") == kind else None, scale)
    return result


def _write_first_person(out: gltf.GltfBuild, model: Model3D) -> dict:
    """VRM 1.0 first-person annotations: kept ones, else ``auto`` for skinned parts and ``both`` for others."""
    explicit: dict[int, str] = {}
    for entry in _items(model.extras.get("vrm_first_person")):
        part = entry.get("part")
        if isinstance(part, int) and entry.get("type") in _FIRST_PERSON:
            explicit[part] = entry["type"]
    annotations = []
    seen: set[int] = set()
    for index, part in enumerate(model.parts):
        node = out.part_nodes[index]
        if node < 0 or node in seen:
            continue
        seen.add(node)
        kind = explicit.get(index) or ("auto" if part.weights is not None else "both")
        annotations.append({"node": node, "type": kind})
    return {"meshAnnotations": annotations} if annotations else {}


def _frame_changes(skeleton: Skeleton, written: Skeleton) -> torch.Tensor:
    """``(J, 3, 3)`` maps from each joint's original rest frame into its written rest frame."""
    return torch.linalg.inv(written.rest.double()[:, :3, :3]) @ skeleton.rest.double()[:, :3, :3]


def _write_springs(out: gltf.GltfBuild, model: Model3D, skeleton: Skeleton,
                   written: Skeleton) -> dict | None:
    """VRMC_springBone from ``model.springs``, colliders moved into the written joint frames."""
    springs = model.springs
    if springs is None or not (springs.chains or springs.colliders):
        return None
    changes = _frame_changes(skeleton, written)
    parents = [int(value) for value in skeleton.parents]
    count = skeleton.count
    colliders = []
    collider_of: dict[int, int] = {}
    for index, collider in enumerate(springs.colliders):
        if not 0 <= collider.joint < count or collider.shape not in ("sphere", "capsule"):
            continue
        linear = changes[collider.joint]
        size = abs(float(torch.linalg.det(linear))) ** (1.0 / 3.0)
        body = {
            "offset": [float(value) for value in linear @ torch.as_tensor(collider.offset).double().reshape(3)],
            "radius": max(_number(collider.radius, 0.0), 0.0) * size,
        }
        if collider.shape == "capsule":
            body["tail"] = [float(value) for value in linear @ torch.as_tensor(collider.tail).double().reshape(3)]
        collider_of[index] = len(colliders)
        colliders.append({"node": out.joint_nodes[collider.joint], "shape": {collider.shape: body}})
    groups = []
    group_of: dict[int, int] = {}
    for index, members in enumerate(springs.collider_groups):
        kept = [collider_of[member] for member in members if member in collider_of]
        if not kept:
            continue
        group_of[index] = len(groups)
        name = springs.group_names[index] if index < len(springs.group_names) else ""
        groups.append({"name": str(name or f"colliders_{index}"), "colliders": kept})
    settings = model.extras.get("vrm_spring_settings")
    settings = settings if isinstance(settings, list) and len(settings) == len(springs.chains) else None
    written_springs = []
    dropped = []
    for index, chain in enumerate(springs.chains):
        joints = list(chain.joints)
        valid = bool(joints) and all(isinstance(joint, int) and 0 <= joint < count for joint in joints)
        valid = valid and all(parents[child] == parent for parent, child in zip(joints, joints[1:]))
        if not valid:
            dropped.append(chain.name)
            continue
        setting = _mapping(settings[index]) if settings is not None else {}
        per_joint = setting.get("joints")
        per_joint = per_joint if isinstance(per_joint, list) and len(per_joint) == len(joints) else None
        direction = torch.as_tensor(chain.gravity_dir).double().reshape(3)
        length = float(direction.norm())
        direction = direction / length if length > 1e-9 else torch.tensor([0.0, -1.0, 0.0])
        default = _spring_params({
            "hitRadius": chain.hit_radius, "stiffness": chain.stiffness, "gravityPower": chain.gravity_power,
            "gravityDir": [float(value) for value in direction], "dragForce": chain.drag,
        })
        entries = []
        for position, joint in enumerate(joints):
            params = _spring_params(per_joint[position]) if per_joint is not None else default
            entries.append({"node": out.joint_nodes[joint], **params})
        spring: dict = {"name": chain.name or f"spring_{index}", "joints": entries}
        chain_groups = [group_of[group] for group in chain.collider_groups if group in group_of]
        if chain_groups:
            spring["colliderGroups"] = chain_groups
        center = setting.get("center")
        if isinstance(center, int) and 0 <= center < count:
            spring["center"] = out.joint_nodes[center]
        written_springs.append(spring)
    if dropped:
        logger.warning("spring chains %s were left out: each joint in a chain must be the child of the "
                       "one before it", ", ".join(str(name) for name in dropped[:8]))
    result: dict = {"specVersion": "1.0"}
    if colliders:
        result["colliders"] = colliders
    if groups:
        result["colliderGroups"] = groups
    if written_springs:
        result["springs"] = written_springs
    return result if len(result) > 1 else None


def _write_constraints(out: gltf.GltfBuild, model: Model3D, skeleton: Skeleton, written: Skeleton) -> None:
    """VRMC_node_constraint on joints whose rest frames were written unchanged, so their axes still hold."""
    entries = _items(model.extras.get("vrm_node_constraints"))
    if not entries:
        return
    changes = _frame_changes(skeleton, written)
    eye = torch.eye(3, dtype=torch.float64)
    parents = [int(value) for value in skeleton.parents]

    def unchanged(*joints: int) -> bool:
        return all(joint < 0 or float((changes[joint] - eye).abs().max()) < 1e-4 for joint in joints)

    dropped = 0
    for entry in entries:
        found = _constraint(entry, skeleton.count)
        if found is None:
            continue
        joint, kind, body, source = found
        if kind == "rotation":
            usable = float((changes[joint] - changes[source]).abs().max()) < 1e-4
        elif kind == "aim":
            usable = unchanged(joint)
        else:
            usable = unchanged(joint, source, parents[joint], parents[source])
        if not usable:
            dropped += 1
            continue
        settings = gltf._json_safe(body)
        settings["source"] = out.joint_nodes[source]
        node = out.json["nodes"][out.joint_nodes[joint]]
        node.setdefault("extensions", {})["VRMC_node_constraint"] = {
            "specVersion": "1.0", "constraint": {kind: settings}}
        out.add_extension_used("VRMC_node_constraint")
    if dropped:
        logger.warning("%d node constraint(s) were left out: their joints were turned when humanoid "
                       "joints were written without rest rotation, so the constraint axes would change",
                       dropped)


def _mtoon_texture(out: gltf.GltfBuild, info: dict) -> dict | None:
    """An MToon texture reference with its texture written, or None."""
    texture = info.get("texture")
    if not isinstance(texture, Texture):
        return None
    written = {key: gltf._json_safe(value) for key, value in info.items() if key not in ("texture", "index")}
    written["index"] = out.add_texture(texture)
    if "KHR_texture_transform" in _mapping(written.get("extensions")):
        out.add_extension_used("KHR_texture_transform")
    return written


def _derived_mtoon(out: gltf.GltfBuild, material: Material) -> dict:
    """MToon settings derived from a PBR material: shade at 0.7 of base colour, default toon ramp."""
    base = [min(max(float(value), 0.0), 1.0) for value in material.base_color[:3]]
    mtoon: dict = {
        "specVersion": "1.0",
        "transparentWithZWrite": False,
        "renderQueueOffsetNumber": 0,
        "shadeColorFactor": [value * 0.7 for value in base],
        "shadingShiftFactor": 0.0,
        "shadingToonyFactor": 0.9,
        "giEqualizationFactor": 0.9,
        "matcapFactor": [0.0, 0.0, 0.0],
        "parametricRimColorFactor": [0.0, 0.0, 0.0],
        "parametricRimFresnelPowerFactor": 5.0,
        "parametricRimLiftFactor": 0.0,
        "rimLightingMixFactor": 1.0,
        "outlineWidthMode": "none",
        "outlineWidthFactor": 0.0,
        "outlineColorFactor": [0.0, 0.0, 0.0],
        "outlineLightingMixFactor": 1.0,
        "uvAnimationScrollXSpeedFactor": 0.0,
        "uvAnimationScrollYSpeedFactor": 0.0,
        "uvAnimationRotationSpeedFactor": 0.0,
    }
    if material.base_color_texture is not None:
        info: dict = {"texture": material.base_color_texture}
        transform = _mapping(material.extras.get("texture_transform")).get("base_color")
        if isinstance(transform, dict):
            info["extensions"] = {"KHR_texture_transform": transform}
        mtoon["shadeMultiplyTexture"] = _mtoon_texture(out, info)
    return mtoon


def _write_mtoon(out: gltf.GltfBuild, model: Model3D) -> None:
    """VRMC_materials_mtoon on every written material, kept settings first, else derived from PBR."""
    _, by_index = _material_indices(out, model)
    materials = out.json.get("materials") or []
    for index, material in sorted(by_index.items()):
        stored = material.extras.get("mtoon")
        if isinstance(stored, dict) and stored:
            mtoon: dict = {}
            for key, value in stored.items():
                if isinstance(value, dict) and "texture" in value:
                    info = _mtoon_texture(out, value)
                    if info is not None:
                        mtoon[key] = info
                elif not key.startswith("_"):
                    safe = gltf._json_safe(value)
                    if safe is not None:
                        mtoon[key] = safe
            mtoon["specVersion"] = "1.0"
        else:
            mtoon = _derived_mtoon(out, material)
        if mtoon.get("outlineWidthMode") not in ("none", "worldCoordinates", "screenCoordinates"):
            mtoon["outlineWidthMode"] = "none"
        queue = int(_number(mtoon.get("renderQueueOffsetNumber"), 0.0))
        if material.alpha_mode != "BLEND":
            queue = 0
            mtoon["transparentWithZWrite"] = False
        elif mtoon.get("transparentWithZWrite") is True:
            queue = min(max(queue, 0), 9)
        else:
            queue = min(max(queue, -9), 0)
        mtoon["renderQueueOffsetNumber"] = queue
        materials[index].setdefault("extensions", {})["VRMC_materials_mtoon"] = mtoon
        out.add_extension_used("VRMC_materials_mtoon")


def write(model: Model3D, mtoon: bool = True) -> bytes:
    """Write a model as VRM 1.0, humanoid joints and their ancestors rebuilt to rest without rotation.

    Args:
        model: A rigged model whose skeleton maps every bone in `REQUIRED_BONES`.
        mtoon: Write MToon toon shading on every material.

    Returns:
        The ``.vrm`` file's bytes.

    Raises:
        VrmError: Required humanoid bones are missing or two bones share a joint.
        GltfError: The model holds data that cannot be written.
    """
    missing = missing_humanoid(model)
    if missing:
        raise VrmError(f"the model cannot be saved as VRM: its skeleton maps no joint to the "
                       f"humanoid bone(s) {', '.join(missing)}. Map them, for example with a humanoid "
                       f"bone mapping step, before saving.")
    skeleton = model.skeleton
    humanoid = _checked_humanoid(skeleton)
    _check_layout(skeleton, humanoid)
    clips = len(model.animations) + len(model.extras.get("morph_animations", []) or [])
    if clips:
        logger.info("VRM 1.0 carries no animation clips; %d animation track(s) were left out. Save "
                    "animations as a separate file.", clips)
    flat = _flattened(skeleton, humanoid)
    normal, parts = _bind_pose(model, flat, _written_rest(model, flat))
    extras = {key: value for key, value in model.extras.items() if key != "morph_animations"}
    meta = {key: value for key, value in model.meta.items() if key not in META_KEYS and key != "vrm0_meta"}
    stand_in = Model3D(parts=parts, skeleton=normal, meta=meta, source_format=model.source_format,
                       extras=extras)
    out = gltf.build(stand_in, embed_images=True)
    _pin_joints(out, normal, flat)
    vrm: dict = {
        "specVersion": "1.0",
        "meta": _write_meta(out, model),
        "humanoid": {"humanBones": {bone: {"node": out.joint_nodes[joint]} for bone, joint in humanoid.items()}},
    }
    expressions = _write_expressions(out, model)
    if expressions:
        vrm["expressions"] = expressions
    vrm["lookAt"] = _write_look_at(model, normal, humanoid, skeleton)
    first_person = _write_first_person(out, model)
    if first_person:
        vrm["firstPerson"] = first_person
    extensions = out.json.setdefault("extensions", {})
    extensions["VRMC_vrm"] = vrm
    out.add_extension_used("VRMC_vrm")
    springs = _write_springs(out, model, skeleton, normal)
    if springs is not None:
        extensions["VRMC_springBone"] = springs
        out.add_extension_used("VRMC_springBone")
    _write_constraints(out, model, skeleton, normal)
    if mtoon:
        _write_mtoon(out, model)
    return out.to_glb()
