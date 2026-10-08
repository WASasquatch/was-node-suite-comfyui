"""Correcting the colour of frames made from other frames, each against the frame it was made from.

Frames are ``(batch, height, width, 3)`` RGB; a frame and its source pair by batch index and may
differ in size.
"""

from __future__ import annotations

import math

import torch

from . import color_match
from .blend_modes import ceiling_of

__all__ = ["METHODS", "correct", "low_pass", "wavelet_levels"]

#: The methods, in menu order.
METHODS = ("wavelet", "reinhard", "mkl", "histogram")

#: The space each statistical method matches in.
SPACES = {"reinhard": "Lab", "mkl": "RGB", "histogram": "RGB"}

#: Pixels of the short side one level of the coarsest band spans, roughly.
COARSE_SPAN = 48


def wavelet_levels(height: int, width: int) -> int:
    """Blur levels below which detail is the frame's own, for a frame of this size.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        3 to 7; 5 for a 1536 pixel short side.
    """
    short = max(1, min(int(height), int(width)))
    return max(3, min(7, round(math.log2(max(2.0, short / COARSE_SPAN)))))


def low_pass(planes: torch.Tensor, levels: int) -> torch.Tensor:
    """``(batch, channels, height, width)`` blurred by ``levels`` dilated binomial passes."""
    taps = torch.tensor([0.25, 0.5, 0.25], dtype=planes.dtype, device=planes.device)
    channels = planes.shape[1]
    across = taps.view(1, 1, 1, 3).repeat(channels, 1, 1, 1)
    down = taps.view(1, 1, 3, 1).repeat(channels, 1, 1, 1)
    out = planes
    for level in range(int(levels)):
        step = 2 ** level
        reach_w = min(step, out.shape[3] - 1)
        reach_h = min(step, out.shape[2] - 1)
        if reach_w > 0:
            out = torch.nn.functional.pad(out, (reach_w, reach_w, 0, 0), mode="replicate")
            out = torch.nn.functional.conv2d(out, across, dilation=(1, reach_w), groups=channels)
        if reach_h > 0:
            out = torch.nn.functional.pad(out, (0, 0, reach_h, reach_h), mode="replicate")
            out = torch.nn.functional.conv2d(out, down, dilation=(reach_h, 1), groups=channels)
    return out


def _wavelet(frames: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Frames whose coarse bands are the sources' and whose fine bands are their own."""
    height, width = int(frames.shape[1]), int(frames.shape[2])
    levels = wavelet_levels(height, width)
    made = frames.permute(0, 3, 1, 2)
    source = sources.permute(0, 3, 1, 2).to(made)
    if tuple(source.shape[2:]) != (height, width):
        source = torch.nn.functional.interpolate(source, size=(height, width), mode="bicubic",
                                                 align_corners=False)
    fixed = made - low_pass(made, levels) + low_pass(source, levels)
    return fixed.permute(0, 2, 3, 1)


def correct(frames: torch.Tensor, sources: torch.Tensor, method: str) -> torch.Tensor:
    """Correct each frame's colour against the source frame at the same index.

    Args:
        frames: ``(batch, height, width, 3)`` to correct.
        sources: ``(batch, h, w, 3)``, the frames they were made from.
        method: One of :data:`METHODS`. ``wavelet`` takes the sources' colour and brightness at
            coarse scales and keeps the frames' fine detail; the others match each frame's
            colour statistics to its source's.

    Returns:
        The corrected frames, same shape, dtype and device as ``frames``, held inside the range
        the two arrived in.

    Raises:
        ValueError: ``method`` is not one of :data:`METHODS`, or the batches differ in length.
    """
    if method not in METHODS:
        raise ValueError(f"colour transfer method {method!r} is not one of {', '.join(METHODS)}")
    if int(frames.shape[0]) != int(sources.shape[0]):
        raise ValueError(
            f"{frames.shape[0]} frame(s) to correct against {sources.shape[0]} source frame(s)"
        )
    ceiling = ceiling_of(frames, sources)
    work = frames.float()
    reference = sources.to(device=work.device).float()
    if method == "wavelet":
        fixed = _wavelet(work, reference)
    else:
        fixed = torch.cat([
            color_match.color_match(work[i:i + 1], reference[i:i + 1], method, SPACES[method],
                                    1.0, False, 0.0)
            for i in range(int(work.shape[0]))
        ])
    return fixed.clamp(0.0, ceiling).to(frames.dtype)
