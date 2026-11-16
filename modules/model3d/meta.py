"""VRM 1.0 avatar metadata on a model, its thumbnail, and a tally of what a model holds.

Meta fields keep their VRM 1.0 names in ``Model3D.meta``; the thumbnail is
``extras["vrm_thumbnail"]``.
"""

from __future__ import annotations

__all__ = [
    "AVATAR_PERMISSIONS", "COMMERCIAL_USAGES", "CREDIT_NOTATIONS", "FLAG_OPTIONS", "KEEP",
    "MAX_LIST_ITEMS", "MAX_TEXT_CHARS", "MAX_THUMBNAIL_SIDE", "MODIFICATIONS", "apply", "contents",
    "split_list", "thumbnail",
]

import torch

from .scene import Model3D, Texture

#: Option that leaves a field as the model already has it.
KEEP = "keep"

#: VRM 1.0 values of each meta choice, most restrictive first, after `KEEP`.
AVATAR_PERMISSIONS = ("keep", "onlyAuthor", "onlySeparatelyLicensedPerson", "everyone")
COMMERCIAL_USAGES = ("keep", "personalNonProfit", "personalProfit", "corporation")
CREDIT_NOTATIONS = ("keep", "required", "unnecessary")
MODIFICATIONS = ("keep", "prohibited", "allowModification", "allowModificationRedistribution")

#: Options of a VRM 1.0 yes-or-no permission.
FLAG_OPTIONS = ("keep", "false", "true")

#: Longest text kept in one meta field, in characters.
MAX_TEXT_CHARS = 2048

#: Most entries kept in a comma list such as the authors.
MAX_LIST_ITEMS = 64

#: Longest side of a stored thumbnail, in pixels.
MAX_THUMBNAIL_SIDE = 1024


def split_list(text: str) -> list[str]:
    """The non-empty entries of a comma-separated list, trimmed.

    Args:
        text: Such as ``"Ada, Grace"``.

    Returns:
        At most `MAX_LIST_ITEMS` entries of at most `MAX_TEXT_CHARS` characters.
    """
    pieces = [piece.strip()[:MAX_TEXT_CHARS] for piece in str(text or "")[:MAX_TEXT_CHARS * 4].split(",")]
    return [piece for piece in pieces if piece][:MAX_LIST_ITEMS]


def apply(model: Model3D, texts: dict[str, str], lists: dict[str, str], choices: dict[str, str],
          flags: dict[str, str]) -> list[str]:
    """Write VRM meta fields onto a model, leaving empty text and `KEEP` choices as they are.

    Args:
        model: The model, changed in place.
        texts: VRM field name against text; empty keeps the field.
        lists: VRM field name against a comma list; empty keeps the field.
        choices: VRM field name against one of its values or `KEEP`.
        flags: VRM field name against ``"true"``, ``"false"`` or `KEEP`.

    Returns:
        The field names written.
    """
    written = []
    for key, value in texts.items():
        text = str(value or "").strip()[:MAX_TEXT_CHARS]
        if text:
            model.meta[key] = text
            written.append(key)
    for key, value in lists.items():
        entries = split_list(value)
        if entries:
            model.meta[key] = entries
            written.append(key)
    for key, value in choices.items():
        if value and value != KEEP:
            model.meta[key] = str(value)
            written.append(key)
    for key, value in flags.items():
        if value in ("true", "false"):
            model.meta[key] = value == "true"
            written.append(key)
    return written


def thumbnail(image: torch.Tensor) -> Texture:
    """The first picture of an image batch as a VRM thumbnail texture.

    Args:
        image: ``(B, H, W, C)`` float image in ``[0, 1]``.

    Returns:
        A PNG texture no larger than `MAX_THUMBNAIL_SIDE` on its longest side.

    Raises:
        ValueError: The image is empty or not shaped like an image batch.
    """
    if not isinstance(image, torch.Tensor) or image.dim() != 4 or 0 in image.shape:
        raise ValueError("the thumbnail is not a picture; wire an IMAGE such as a render of the "
                         "avatar's face")
    picture = image[0, ..., :4].detach().float().cpu().clamp(0, 1)
    if picture.shape[-1] == 2:
        picture = picture[..., :1]
    if picture.shape[-1] == 1:
        picture = picture.expand(-1, -1, 3)
    height, width = int(picture.shape[0]), int(picture.shape[1])
    longest = max(height, width)
    if longest > MAX_THUMBNAIL_SIDE:
        factor = MAX_THUMBNAIL_SIDE / longest
        size = (max(1, round(height * factor)), max(1, round(width * factor)))
        picture = torch.nn.functional.interpolate(
            picture.permute(2, 0, 1).unsqueeze(0), size=size, mode="bilinear", antialias=True,
            align_corners=False,
        )[0].permute(1, 2, 0).clamp(0, 1)
    return Texture(image=picture.contiguous(), name="thumbnail", wrap_s="clamp", wrap_t="clamp",
                   mime="image/png")


def contents(model: Model3D) -> dict[str, int]:
    """What a model holds beyond its geometry, as counts.

    Returns:
        Materials, textures, animations, expressions, spring chains and skinned parts.
    """
    materials, textures = {}, {}
    for part in model.parts:
        materials[id(part.material)] = part.material
    for material in materials.values():
        for texture in material.textures().values():
            textures[id(texture)] = texture
    springs = model.springs
    return {
        "materials": len(materials),
        "textures": len(textures),
        "animations": len(model.animations),
        "expressions": len(model.expressions),
        "spring chains": len(springs.chains) if springs is not None else 0,
        "skinned parts": sum(1 for part in model.parts if part.weights is not None),
    }
