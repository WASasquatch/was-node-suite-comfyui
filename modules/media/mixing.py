"""Two pictures mixed by a weight, in picture codes or in light.

Frames are floats shaped ``(..., height, width, channels)``; a weight is a number or a tensor
shaped ``(..., height, width)``.
"""

from __future__ import annotations

import torch

__all__ = ["CODES", "LIGHT", "SPACES", "mix", "toward"]

#: Mix the values as they arrive.
CODES = "codes"

#: Decode sRGB picture codes to light, mix, and encode again.
LIGHT = "light"

#: The spaces a mix runs in, in menu order.
SPACES = (CODES, LIGHT)


def _mixed(a: torch.Tensor, b: torch.Tensor, weight, space: str) -> torch.Tensor:
    """The straight mix, before the two ends are pinned."""
    if space != LIGHT:
        return a * (1.0 - weight) + b * weight
    from ..image import raw

    colour = raw.encode(raw.linearise(a[..., :3]) * (1.0 - weight) + raw.linearise(b[..., :3]) * weight)
    if a.shape[-1] <= 3:
        return colour
    rest = a[..., 3:] * (1.0 - weight) + b[..., 3:] * weight
    return torch.cat([colour, rest], dim=-1)


def mix(a: torch.Tensor, b: torch.Tensor, weight, space: str = CODES) -> torch.Tensor:
    """``a`` where the weight is 0, ``b`` where it is 1, mixed between.

    Args:
        a: The picture at weight 0.
        b: The picture at weight 1, the same shape.
        weight: A number, or a tensor broadcasting against ``a`` without its channel axis.
        space: :data:`CODES` or :data:`LIGHT`.

    Returns:
        A new tensor. Every value at weight 0 is ``a``'s and at weight 1 is ``b``'s, bit for bit.
    """
    if not torch.is_tensor(weight):
        share = float(weight)
        if share <= 0.0:
            return a.clone()
        if share >= 1.0:
            return b.clone()
        return _mixed(a, b, share, space)
    share = weight.to(device=a.device, dtype=a.dtype).clamp(0.0, 1.0).unsqueeze(-1)
    out = _mixed(a, b, share, space)
    out = torch.where(share <= 0.0, a, out)
    return torch.where(share >= 1.0, b, out)


def toward(picture: torch.Tensor, level: float, weight, space: str = CODES) -> torch.Tensor:
    """A picture mixed toward a flat colour, opaque where it carries coverage.

    Args:
        picture: The picture at weight 0.
        level: The colour's value in every colour channel, 0 for black and 1 for white.
        weight: As :func:`mix` takes it.
        space: :data:`CODES` or :data:`LIGHT`.

    Returns:
        A new tensor, the flat colour at weight 1.
    """
    flat = torch.full_like(picture, float(level))
    if picture.shape[-1] > 3:
        flat[..., 3:] = 1.0
    return mix(picture, flat, weight, space)
