"""Holding a per-frame effect steady along a clip's measured motion.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. The effect is what the
processed frames add to the source; it is carried along the motion, forward and back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import optical_flow

__all__ = ["MATCH_LEVELS", "steady"]

#: Luminance difference, in levels of 255, left after following the motion at which a pixel
#: stops counting as the same surface.
MATCH_LEVELS = 12.0

#: Gaussian sigma, in pixels of the measured motion, applied to the carry weight.
WEIGHT_SOFTEN = 1.0


def _carry(motion, index: int, step: int, size, device):
    """Flow and carry weight from frame ``index`` onto its neighbour, at ``size``.

    Returns:
        ``(flow, weight)`` with ``weight`` in ``[0, 1]``, ``(1, 1, height, width)``, or ``None``
        where nothing is carried: no neighbour, or a cut.
    """
    found = motion.toward(index, step, device)
    if found is None:
        return None
    flow, seen = found
    here = motion.luma[index:index + 1].to(device=device, dtype=torch.float32)
    there = motion.luma[index + step:index + step + 1].to(device=device, dtype=torch.float32)
    gap = optical_flow.gaussian((optical_flow.warp(there, flow) - here).abs(), WEIGHT_SOFTEN)
    weight = seen * (1.0 - gap / MATCH_LEVELS).clamp(0.0, 1.0)
    weight = optical_flow.gaussian(weight, WEIGHT_SOFTEN)
    height, width = size
    return (
        optical_flow.resize_flow(flow, height, width),
        F.interpolate(weight, size=(height, width), mode="bilinear", align_corners=False),
    )


def _base(source, index: int, size, channels: int, device):
    """Source frame ``index`` at ``size``, ``(1, channels, height, width)``."""
    frame = source[index:index + 1, ..., :channels].to(device=device, dtype=torch.float32)
    return optical_flow.resize(frame.permute(0, 3, 1, 2), *size)


def steady(source, processed, motion, strength: float = 0.8, device=None, progress=None):
    """Carry the effect ``processed`` adds to ``source`` along the clip's motion.

    Args:
        source: The frames before the effect.
        processed: The same frames after it, any size, as many frames as ``source``.
        motion: A :class:`~.motion.Motion` measured from ``source``.
        strength: How much of each frame's effect is taken from its neighbours where they
            match, 0 to 1. 0 answers ``processed``.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame per pass.

    Returns:
        Frames shaped like ``processed``, on its device.
    """
    count, height, width, channels = (int(v) for v in processed.shape)
    device = processed.device if device is None else torch.device(device)
    strength = max(0.0, min(1.0, float(strength)))
    shared = min(channels, int(source.shape[-1]))
    size = (height, width)
    clip = bool(processed.min() >= 0.0 and processed.max() <= 1.0)
    out = processed.clone()
    if strength <= 0.0 or count < 2:
        if progress is not None:
            progress(2 * count)
        return out

    def effect(index):
        frame = processed[index:index + 1, ..., :shared].to(device=device, dtype=torch.float32)
        return frame.permute(0, 3, 1, 2) - _base(source, index, size, shared, device)

    # Forward: each frame's effect leans on the one before it; held in ``out`` as a residual.
    carried = None
    for index in range(count):
        residual = effect(index)
        link = _carry(motion, index, -1, size, device) if carried is not None else None
        if link is not None:
            flow, weight = link
            pulled = optical_flow.warp(carried, flow)
            residual = residual + strength * weight * (pulled - residual)
        carried = residual
        out[index, ..., :shared] = residual[0].permute(1, 2, 0).to(out.dtype).to(out.device)
        if progress is not None:
            progress(1)

    # Backward: the same leaning on the frame after, then the two passes averaged.
    carried = None
    for index in range(count - 1, -1, -1):
        residual = effect(index)
        link = _carry(motion, index, 1, size, device) if carried is not None else None
        if link is not None:
            flow, weight = link
            pulled = optical_flow.warp(carried, flow)
            residual = residual + strength * weight * (pulled - residual)
        carried = residual
        ahead = out[index, ..., :shared].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        both = 0.5 * (residual + ahead)
        frame = _base(source, index, size, shared, device) + both
        if clip:
            frame = frame.clamp(0.0, 1.0)
        out[index, ..., :shared] = frame[0].permute(1, 2, 0).to(out.dtype).to(out.device)
        if progress is not None:
            progress(1)
    return out
