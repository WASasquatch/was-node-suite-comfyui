"""Fade and crossfade gain curves.

A curve maps progress ``0`` to ``1`` onto a gain ``0`` to ``1``. Fading out reads the same curve
backwards, so a crossfade on ``equal power`` keeps the summed power level.
"""

from __future__ import annotations

import math

import torch

__all__ = ["CURVES", "DEFAULT_CURVE", "curve_tooltip", "falling", "rising"]

#: The curves a fade can take, in menu order.
CURVES = ("equal power", "linear", "s-curve", "exponential", "logarithmic")

#: The curve a fresh node fades on.
DEFAULT_CURVE = "equal power"


def curve_tooltip(what: str) -> str:
    """The tooltip of a curve widget.

    Args:
        what: What the curve shapes, such as ``"The fade in"``.

    Returns:
        One tooltip naming every curve.
    """
    return (
        f"{what}'s shape. equal power = keeps the level steady through a crossfade; linear = "
        f"straight gain ramp; s-curve = soft at both ends; exponential = slow start, fast "
        f"finish; logarithmic = fast start, slow finish."
    )


def _shaped(progress: torch.Tensor, curve: str) -> torch.Tensor:
    """Gains for progress values on one curve."""
    if curve == "linear":
        return progress
    if curve == "s-curve":
        return 0.5 - 0.5 * torch.cos(math.pi * progress)
    if curve == "exponential":
        return progress ** 3
    if curve == "logarithmic":
        return 1.0 - (1.0 - progress) ** 3
    return torch.sin(0.5 * math.pi * progress)


def rising(count: int, curve: str = DEFAULT_CURVE) -> torch.Tensor:
    """Gains of a fade in, from silence up to full level.

    Args:
        count: Samples the fade lasts.
        curve: A name from :data:`CURVES`. Anything else reads as ``equal power``.

    Returns:
        A float32 tensor of ``count`` gains, the first near 0 and the last near 1, sampled at
        the middle of each sample's span.
    """
    count = max(0, int(count))
    if count == 0:
        return torch.zeros(0, dtype=torch.float32)
    progress = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    return _shaped(progress, curve).to(torch.float32)


def falling(count: int, curve: str = DEFAULT_CURVE) -> torch.Tensor:
    """Gains of a fade out, the mirror of :func:`rising`.

    Args:
        count: Samples the fade lasts.
        curve: A name from :data:`CURVES`.

    Returns:
        A float32 tensor of ``count`` gains from near 1 down to near 0.
    """
    return rising(count, curve).flip(0)
