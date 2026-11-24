"""Correcting the surface settings of a model's materials, part by part.

Metalness is scaled by the factor and cleared per texel where the base colour is dark; a
roughness floor is written into the metallic-roughness texture.
"""

from __future__ import annotations

__all__ = ["DOUBLE_SIDED", "adjust"]

import copy

import torch
import torch.nn.functional as F

from .scene import Model3D, Texture

#: Choices for whether back faces are drawn.
DOUBLE_SIDED = ("keep", "on", "off")

#: Share of the brightness threshold below which metal is cleared completely; it fades back in above.
METAL_FADE_START = 0.6


def _retexture(texture: Texture, image: torch.Tensor) -> Texture:
    """A copy of ``texture`` holding ``image``."""
    return Texture(image=image, name=texture.name, wrap_s=texture.wrap_s, wrap_t=texture.wrap_t, mime=texture.mime)


def _mean_metalness(material) -> float:
    """Metalness averaged over the texture, factor included."""
    texture = material.metallic_roughness_texture
    if texture is None:
        return float(material.metallic)
    return float(material.metallic) * float(texture.image[..., 2].float().mean())


def _base_brightness(material, size: tuple[int, int] | None) -> torch.Tensor:
    """``(H, W)`` brightest channel of the base colour at ``size`` (width, height), or a 0-d value with no map."""
    factor = torch.tensor(material.base_color[:3], dtype=torch.float32).clamp(0, 1) ** (1 / 2.2)
    texture = material.base_color_texture
    if texture is None or size is None:
        return factor.max()
    rgb = texture.image[..., :3].float() * factor
    if texture.size != size:
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(size[1], size[0]), mode="bilinear",
                            align_corners=False, antialias=True)[0].permute(1, 2, 0)
    return rgb.amax(dim=-1)


def _clear_dark_metal(material, threshold: float) -> None:
    """Fade metalness out where the base colour is darker than ``threshold``, in place."""
    if float(material.metallic) <= 0:
        return
    texture = material.metallic_roughness_texture
    if texture is None and material.base_color_texture is not None:
        width, height = material.base_color_texture.size
        image = torch.ones(height, width, 3)
        image[..., 1] = float(material.roughness)
        image[..., 2] = float(material.metallic)
        texture = Texture(image=image, name="metallic_roughness")
        material.roughness = material.metallic = 1.0
    start = threshold * METAL_FADE_START
    if texture is None:
        keep = ((_base_brightness(material, None) - start) / (threshold - start)).clamp(0, 1)
        material.metallic = float(material.metallic) * float(keep)
        return
    keep = ((_base_brightness(material, texture.size) - start) / (threshold - start)).clamp(0, 1)
    image = texture.image.float().clone()
    image[..., 2] = image[..., 2] * keep
    material.metallic_roughness_texture = _retexture(texture, image)


def _floor_roughness(material, floor: float) -> None:
    """Raise every roughness value of a material to at least ``floor``, in place."""
    texture = material.metallic_roughness_texture
    if texture is None:
        material.roughness = max(float(material.roughness), floor)
        return
    image = texture.image.float().clone()
    factor = max(float(material.roughness), 1e-6)
    image[..., 1] = torch.maximum(image[..., 1] * factor, torch.tensor(floor)).clamp(0, 1)
    material.metallic_roughness_texture = _retexture(texture, image)
    material.roughness = 1.0


def adjust(model: Model3D, chosen: list[int], metalness: float = 1.0, min_roughness: float = 0.0,
           double_sided: str = "keep", metal_min_brightness: float = 0.0) -> dict[str, float]:
    """Change the materials of the chosen parts in place; parts sharing a material get their own copy.

    Args:
        model: The model.
        chosen: Indices of the parts to change.
        metalness: Multiplies each material's metalness; 0 makes it fully non-metal.
        min_roughness: Roughness every texel is raised to at least; 0 leaves roughness alone.
        double_sided: One of `DOUBLE_SIDED`.
        metal_min_brightness: Base colour brightness a texel needs to stay metal; 0 leaves metal alone.

    Returns:
        ``{"materials changed", "metalness before", "metalness after"}``, metalness averaged over each texture.
    """
    done: dict[int, object] = {}
    before = after = 0.0
    threshold = max(0.0, min(1.0, float(metal_min_brightness)))
    for index in chosen:
        part = model.parts[index]
        original = part.material
        if id(original) in done:
            part.material = done[id(original)]
            continue
        material = copy.copy(original)
        material.extras = dict(original.extras)
        before += _mean_metalness(material)
        material.metallic = float(material.metallic) * max(0.0, min(1.0, float(metalness)))
        if threshold > 0:
            _clear_dark_metal(material, threshold)
        if min_roughness > 0:
            _floor_roughness(material, max(0.0, min(1.0, float(min_roughness))))
        if double_sided in ("on", "off"):
            material.double_sided = double_sided == "on"
        after += _mean_metalness(material)
        done[id(original)] = material
        part.material = material
    count = len(done)
    return {"materials changed": count, "metalness before": round(before / max(count, 1), 3),
            "metalness after": round(after / max(count, 1), 3)}
