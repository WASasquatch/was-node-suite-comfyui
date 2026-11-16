"""Readying a model for a file: texture encoding and the humanoid a VRM needs.

Both change the model in place, so a node works on its own copy.
"""

from __future__ import annotations

__all__ = ["TEXTURE_FORMATS", "left_out", "prepare_vrm", "set_texture_format"]

from ..log import get_logger
from . import orient
from .scene import Model3D, Texture

logger = get_logger("model3d.export")

#: Texture encodings a save offers, against the media type written; ``keep`` uses each texture's own.
TEXTURE_FORMATS = {"keep": None, "png": "image/png", "jpeg": "image/jpeg"}


def _textures(model: Model3D) -> list[Texture]:
    """Every distinct texture the model's materials draw, MToon textures included."""
    found: dict[int, Texture] = {}
    for part in model.parts:
        material = part.material
        for texture in material.textures().values():
            found.setdefault(id(texture), texture)
        mtoon = material.extras.get("mtoon")
        if isinstance(mtoon, dict):
            for value in mtoon.values():
                held = value.get("texture") if isinstance(value, dict) else None
                if isinstance(held, Texture):
                    found.setdefault(id(held), held)
    return list(found.values())


def set_texture_format(model: Model3D, choice: str) -> int:
    """Set the encoding every texture is written in.

    Args:
        model: The model, changed in place.
        choice: A key of `TEXTURE_FORMATS`. A texture with transparency stays PNG under ``jpeg``.

    Returns:
        How many textures were set.

    Raises:
        ValueError: The choice is not a key of `TEXTURE_FORMATS`.
    """
    if choice not in TEXTURE_FORMATS:
        raise ValueError(f"texture_format '{choice}' is not one of {', '.join(TEXTURE_FORMATS)}")
    mime = TEXTURE_FORMATS[choice]
    if mime is None:
        return 0
    textures = _textures(model)
    for texture in textures:
        texture.mime = mime
    return len(textures)


def left_out(model: Model3D, fmt: str) -> list[str]:
    """What a format cannot hold of a model, as counted phrases.

    Args:
        model: The model about to be written.
        fmt: A format id such as ``"fbx"``.

    Returns:
        Phrases such as ``"2 animation clip(s)"``; empty when the format keeps everything.
    """
    missing = []
    clips = len(model.animations) + len(model.extras.get("morph_animations", []) or [])
    if fmt in ("fbx", "vrm", "obj") and clips:
        missing.append(f"{clips} animation clip(s)")
    if fmt == "obj":
        if model.skeleton is not None and model.skeleton.count:
            missing.append(f"a skeleton of {model.skeleton.count} joints")
        morphs = sum(len(part.morphs) for part in model.parts)
        if morphs:
            missing.append(f"{morphs} shape key(s)")
    return missing


def prepare_vrm(model: Model3D) -> dict[str, float | int]:
    """Map the humanoid bones a VRM needs and turn the model to face +Z.

    Args:
        model: A rigged model, changed in place. Bones it maps are kept; missing ones are
            guessed from joint names and positions.

    Returns:
        ``bones guessed`` and ``turned`` (degrees about Y).

    Raises:
        ValueError: The model has no skeleton or skinned part, or required bones stay unmapped.
    """
    from ..character import bones
    from . import vrm

    skeleton = model.skeleton
    if skeleton is None or not skeleton.count or not model.is_rigged:
        raise ValueError("this model has no skeleton, so it cannot be saved as VRM. Rig it with "
                         "Character Rig first, or save it as glb, fbx or obj")
    guessed = 0
    if vrm.missing_humanoid(model):
        count = skeleton.count
        held = {bone: joint for bone, joint in skeleton.humanoid.items()
                if isinstance(joint, int) and not isinstance(joint, bool) and 0 <= joint < count}
        try:
            found = bones.guess_humanoid(skeleton.names, skeleton.parents, skeleton.heads())
        except ValueError as error:
            logger.debug("the humanoid could not be guessed: %s", error)
            found = {}
        taken = set(held.values())
        for bone, joint in found.items():
            if bone not in held and joint not in taken:
                held[bone] = joint
                taken.add(joint)
        merged = bones.validate_humanoid(held, skeleton.parents)
        guessed = max(0, len(merged) - len(skeleton.humanoid))
        skeleton.humanoid = merged
    missing = vrm.missing_humanoid(model)
    if missing:
        raise ValueError(f"VRM needs the humanoid bone(s) {', '.join(missing)}, and this skeleton has "
                         "no joint that can be read as them. Rig the model with Character Rig, which "
                         "builds a full humanoid skeleton, then save it as VRM; or save it as glb or fbx")
    turned = orient.face_forward(model)
    return {"bones guessed": guessed, "turned": round(turned, 2)}
