"""Correcting the surface settings of a model's materials, part by part.

Metalness is scaled through the material's factor. A roughness floor is written into the
metallic-roughness texture, so every texel is at least that rough.
"""

from __future__ import annotations

__all__ = ["DOUBLE_SIDED", "adjust"]

import copy

import torch

from .scene import Model3D, Texture

#: Choices for whether back faces are drawn.
DOUBLE_SIDED = ("keep", "on", "off")


def _floor_roughness(material, floor: float) -> None:
    """Raise every roughness value of a material to at least ``floor``, in place."""
    texture = material.metallic_roughness_texture
    if texture is None:
        material.roughness = max(float(material.roughness), floor)
        return
    image = texture.image.float().clone()
    factor = max(float(material.roughness), 1e-6)
    image[..., 1] = torch.maximum(image[..., 1] * factor, torch.tensor(floor)).clamp(0, 1)
    material.metallic_roughness_texture = Texture(image=image, name=texture.name, wrap_s=texture.wrap_s,
                                                  wrap_t=texture.wrap_t, mime=texture.mime)
    material.roughness = 1.0


def adjust(model: Model3D, chosen: list[int], metalness: float = 1.0, min_roughness: float = 0.0,
           double_sided: str = "keep") -> dict[str, float]:
    """Change the materials of the chosen parts in place; parts sharing a material get their own copy.

    Args:
        model: The model.
        chosen: Indices of the parts to change.
        metalness: Multiplies each material's metalness; 0 makes it fully non-metal.
        min_roughness: Roughness every texel is raised to at least; 0 leaves roughness alone.
        double_sided: One of `DOUBLE_SIDED`.

    Returns:
        ``{"materials changed", "metalness before", "metalness after"}``, metalness as the factor.
    """
    done: dict[int, object] = {}
    before = after = 0.0
    for index in chosen:
        part = model.parts[index]
        original = part.material
        if id(original) in done:
            part.material = done[id(original)]
            continue
        material = copy.copy(original)
        material.extras = dict(original.extras)
        before += float(material.metallic)
        material.metallic = float(material.metallic) * max(0.0, min(1.0, float(metalness)))
        if min_roughness > 0:
            _floor_roughness(material, max(0.0, min(1.0, float(min_roughness))))
        if double_sided in ("on", "off"):
            material.double_sided = double_sided == "on"
        after += float(material.metallic)
        done[id(original)] = material
        part.material = material
    count = len(done)
    return {"materials changed": count, "metalness before": round(before / max(count, 1), 3),
            "metalness after": round(after / max(count, 1), 3)}
