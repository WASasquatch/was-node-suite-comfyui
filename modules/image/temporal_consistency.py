"""Holding a per-frame effect steady along a clip's measured motion.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. The effect is what the
processed frames add to the source; it is carried along the motion, back and forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import optical_flow, scratch

__all__ = ["LOOKAHEAD", "MATCH_LEVELS", "base", "passes", "steady"]

#: Luminance difference, in levels of 255, left after following the motion at which a pixel
#: stops counting as the same surface.
MATCH_LEVELS = 12.0

#: Gaussian sigma, in pixels of the measured motion, applied to the carry weight.
WEIGHT_SOFTEN = 1.0

#: Frames past a block's end its backward pass starts from, when a clip is steadied in blocks.
LOOKAHEAD = 48


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


def base(frames, index: int, size, channels: int, device):
    """Frame ``index`` of a batch at ``size``, ``(1, channels, height, width)`` float32.

    Args:
        frames: ``(frames, height, width, channels)``, float or uint8 codes.
        index: The frame.
        size: ``(height, width)`` wanted.
        channels: Leading channels kept.
        device: Where the answer is placed.

    Returns:
        The frame, resized with an antialiased bilinear filter.
    """
    frame = optical_flow.unit(frames[index:index + 1, ..., :channels], device)
    return optical_flow.resize(frame.permute(0, 3, 1, 2), *size)


def _lean(residual, carried, link, strength: float):
    """``residual`` moved toward ``carried`` pulled along ``link``, where one is given."""
    if link is None:
        return residual
    flow, weight = link
    pulled = optical_flow.warp(carried, flow)
    return residual + strength * weight * (pulled - residual)


def passes(count: int, residual, motion, strength: float, size, device, held=None, progress=None,
           block=None):
    """Every frame's residual held steady along the motion, back then forward, answered in order.

    Args:
        count: Frames in the clip.
        residual: Callable taking a frame number and answering its residual,
            ``(1, channels, height, width)`` float32 on ``device``; called on every sweep
            that crosses the frame.
        motion: A :class:`~.motion.Motion` with ``count`` frames.
        strength: How much of each frame's residual is taken from its neighbours where they
            match, 0 to 1. 0 answers each residual as it is, calling ``residual`` once.
        size: ``(height, width)`` of the residuals.
        device: Where the work runs.
        held: A :class:`~.scratch.FrameStore` with room for ``block`` frames shaped
            ``(height, width, channels)``, all of them when ``block`` is None, holding the
            backward pass. Each frame is released once answered. Required unless
            ``strength`` is 0.
        progress: Optional callable taking a step count, called once per frame per pass.
        block: Frames steadied together, or None for the whole clip. Each block's backward
            pass starts :data:`LOOKAHEAD` frames past its end; the forward pass runs on
            across blocks.

    Yields:
        ``(index, steadied)`` for every frame in order, ``steadied`` shaped like a residual.
    """
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0.0 or count < 2:
        for index in range(count):
            yield index, residual(index)
            if progress is not None:
                progress(2)
        return

    span = count if not block or int(block) >= count else max(1, int(block))
    ahead = None
    for start in range(0, count, span):
        stop = min(start + span, count)
        reach = count if stop >= count else min(count, stop + LOOKAHEAD)
        carried = None
        for index in range(reach - 1, start - 1, -1):
            link = _carry(motion, index, 1, size, device) if carried is not None else None
            carried = _lean(residual(index), carried, link, strength)
            if index < stop:
                held.write(index, carried[0].permute(1, 2, 0).to(held.dtype).contiguous())
                if progress is not None:
                    progress(1)
        for index in range(start, stop):
            link = _carry(motion, index, -1, size, device) if ahead is not None else None
            ahead = _lean(residual(index), ahead, link, strength)
            behind = held.read(index, device, int(ahead.shape[1]))
            held.release(index)
            yield index, 0.5 * (ahead + behind.permute(2, 0, 1).unsqueeze(0))
            if progress is not None:
                progress(1)


def steady(source, processed, motion, strength: float = 0.8, device=None, progress=None, node: str = ""):
    """Carry the effect ``processed`` adds to ``source`` along the clip's motion.

    Args:
        source: The frames before the effect.
        processed: The same frames after it, any size, as many frames as ``source``.
        motion: A :class:`~.motion.Motion` measured from ``source``.
        strength: How much of each frame's effect is taken from its neighbours where they
            match, 0 to 1. 0 answers ``processed``.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame per pass.
        node: Display name of the calling node, for the log and the refusal.

    Returns:
        Frames shaped like ``processed``, on the CPU, held in a scratch file where memory is
        short. Held to ``[0, 1]`` where ``processed`` was.

    Raises:
        MemoryError: Neither free memory nor a scratch drive can hold the result.
    """
    count, height, width, channels = (int(v) for v in processed.shape)
    device = processed.device if device is None else torch.device(device)
    shared = min(channels, int(source.shape[-1]))
    size = (height, width)
    strength = max(0.0, min(1.0, float(strength)))
    out = scratch.allocate(
        processed.shape,
        processed.dtype if processed.dtype.is_floating_point else torch.float32,
        node=node,
        advice="A shorter clip, or the effect run at a smaller size, also fits it.",
    )
    if strength <= 0.0 or count < 2 or channels > shared:
        for index in range(count):
            out[index].copy_(processed[index])
            scratch.trim(processed[index])
            scratch.trim(out[index])
    if strength <= 0.0 or count < 2:
        if progress is not None:
            progress(2 * count)
        return out
    low = torch.full((), float("inf"), device=device)
    high = torch.full((), float("-inf"), device=device)

    def residual(index):
        nonlocal low, high
        frame = processed[index:index + 1].to(device=device, dtype=torch.float32)
        scratch.trim(processed[index])
        low = torch.minimum(low, frame.amin())
        high = torch.maximum(high, frame.amax())
        frame = frame[..., :shared].permute(0, 3, 1, 2)
        return frame - base(source, index, size, shared, device)

    clip = None
    held = scratch.FrameStore(count, (height, width, channels), tensor=out)
    for index, steadied in passes(count, residual, motion, strength, size, device, held, progress):
        if clip is None:
            clip = bool(low >= 0.0) and bool(high <= 1.0)
        frame = base(source, index, size, shared, device) + steadied
        if clip:
            frame = frame.clamp(0.0, 1.0)
        out[index, ..., :shared].copy_(frame[0].permute(1, 2, 0).to(out.dtype).contiguous())
        scratch.trim(out[index])
    return out
