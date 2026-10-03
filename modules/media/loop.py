"""Finding where a clip can loop back on itself, and blending the join.

A loop plays frames ``start`` to ``stop - 1`` and returns to ``start``; ``stop`` is the frame
whose likeness to ``start`` makes the return seamless.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["BLEND_MODES", "CLEAN_LEVELS", "VISIBLE_LEVELS", "blend_for", "find", "loop_frames", "quality"]

#: Width, in pixels, of the thumbnails frames are compared at.
THUMB_WIDTH = 96

#: Frames either side of a join compared with their counterparts, so motion matches as well.
MATCH_SPAN = 2

#: Join difference, in levels of 255, below which the return is clean, and above which it shows.
CLEAN_LEVELS = 3.0
VISIBLE_LEVELS = 8.0

#: How far a longer loop is preferred over an equally good shorter one.
LENGTH_PREFERENCE = 0.05

#: Long side, in pixels, the blend's motion is measured at.
BLEND_SIDE = 768

#: How the frames across a join are blended: chosen by how far apart the ends are, carried along
#: the motion between them where it holds, or dissolved.
BLEND_MODES = ("auto", "motion", "crossfade")

#: Gaussian sigma, in pixels of the measured motion, softening where the motion is trusted.
TRUST_SOFTEN = 2.0


def _thumbs(frames, device):
    """Every frame as a flattened colour thumbnail in levels of 255, ``(frames, values)``."""
    count, height, width = (int(v) for v in frames.shape[:3])
    thumb_h = max(8, int(round(THUMB_WIDTH * height / max(width, 1))))
    out = []
    for start in range(0, count, 32):
        chunk = frames[start:start + 32, ..., :3].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        small = F.interpolate(chunk, size=(thumb_h, THUMB_WIDTH), mode="bilinear", align_corners=False, antialias=True)
        out.append((small * 255.0).reshape(small.shape[0], -1))
    return torch.cat(out)


def find(frames, min_frames: int, max_frames: int, blend: int, device=None, limit: float = 0.0):
    """The loop whose return is least visible, or the longest within a difference limit.

    Args:
        frames: ``(frames, height, width, channels)``.
        min_frames: Fewest frames the loop holds.
        max_frames: Most frames it holds; 0 for any.
        blend: Frames blended across the join, which must exist beside it.
        device: Where the comparison runs.
        limit: Most a join may differ, in levels of 255; above 0 the longest loop within it
            is taken.

    Returns:
        ``(start, stop, difference, closest)``: the loop, the mean difference in levels of 255
        between ``stop`` and ``start`` before blending, and the smallest difference any loop
        had. ``None`` for the loop where none fits, or none comes within ``limit``.
    """
    count = int(frames.shape[0])
    device = frames.device if device is None else torch.device(device)
    thumbs = _thumbs(frames, device)
    distance = torch.cdist(thumbs, thumbs, p=1) / thumbs.shape[1]
    longest = count - 1 if max_frames <= 0 else min(count - 1, int(max_frames))
    best = None
    longest_within = None
    closest = math.inf
    for length in range(max(1, int(min_frames)), longest + 1):
        line = distance.diagonal(length)
        window = 2 * MATCH_SPAN + 1
        cost = F.avg_pool1d(F.pad(line.view(1, 1, -1), (MATCH_SPAN, MATCH_SPAN), mode="replicate"), window, stride=1).view(-1)
        starts = torch.arange(line.numel(), device=device)
        stops = starts + length
        usable = (starts >= blend) | (stops + blend <= count - 1)
        if not bool(usable.any()):
            continue
        cost = torch.where(usable, cost, torch.full_like(cost, math.inf))
        score = cost * (1.0 + LENGTH_PREFERENCE * (1.0 - length / max(count - 1, 1)))
        at = int(torch.argmin(score))
        joined = float(line[at])
        closest = min(closest, float(line[usable].min()))
        if best is None or float(score[at]) < best[0]:
            best = (float(score[at]), at, at + length, joined)
        if limit > 0:
            fits = usable & (line <= float(limit))
            if bool(fits.any()):
                inside = torch.where(fits, cost, torch.full_like(cost, math.inf))
                at = int(torch.argmin(inside))
                longest_within = (at, at + length, float(line[at]))
    if limit > 0:
        if longest_within is None:
            return None, closest
        return (*longest_within, closest)
    if best is None:
        return None, closest
    return best[1], best[2], best[3], closest


def quality(difference: float) -> str:
    """How a join of ``difference`` levels reads: ``clean``, ``slight`` or ``visible``."""
    if difference < CLEAN_LEVELS:
        return "clean"
    return "slight" if difference < VISIBLE_LEVELS else "visible"


def blend_for(mode: str, difference: float) -> str:
    """The blend a join gets; ``auto`` carries ends within :data:`VISIBLE_LEVELS` along the motion.

    Args:
        mode: One of :data:`BLEND_MODES`.
        difference: The join's difference in levels of 255.

    Returns:
        ``"motion"`` or ``"crossfade"``.
    """
    if mode == "auto":
        return "motion" if difference < VISIBLE_LEVELS else "crossfade"
    return mode


def _ease(share: float) -> float:
    """A smooth step from 0 to 1."""
    return share * share * (3.0 - 2.0 * share)


def _morph(first, second, share: float, device, mode: str = "motion"):
    """``first`` carried ``share`` of the way to ``second`` along their motion, dissolved where it fails."""
    from ..image import motion as motion_field
    from ..image import optical_flow

    a = first.to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    b = second.to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    dissolved = (1.0 - share) * a + share * b
    if mode == "crossfade":
        return dissolved[0].permute(1, 2, 0)
    height, width = a.shape[-2:]
    size = motion_field.working_size(height, width, BLEND_SIDE)
    luma_a = optical_flow.resize(optical_flow.luminance(a[:, :3].permute(0, 2, 3, 1)), *size)
    luma_b = optical_flow.resize(optical_flow.luminance(b[:, :3].permute(0, 2, 3, 1)), *size)
    ahead = optical_flow.estimate(luma_a, luma_b)
    behind = optical_flow.estimate(luma_b, luma_a)
    trust = optical_flow.gaussian(optical_flow.consistent(ahead, behind).float(), TRUST_SOFTEN)
    trust = F.interpolate(trust, size=(height, width), mode="bilinear", align_corners=False)
    flow = optical_flow.resize_flow(ahead, height, width)
    carried = (1.0 - share) * optical_flow.warp(a, -share * flow) + share * optical_flow.warp(b, (1.0 - share) * flow)
    out = trust * carried + (1.0 - trust) * dissolved
    return out[0].permute(1, 2, 0)


def loop_frames(frames, start: int, stop: int, blend: int, device=None, mode: str = "motion"):
    """The loop's frames, the join blended along its motion.

    Args:
        frames: The clip.
        start: First frame of the loop.
        stop: The frame the loop returns in place of.
        blend: Frames blended; 0 cuts straight back.
        device: Where the blend runs.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(stop - start, height, width, channels)`` on the frames' device, and where the blend
        sits: ``"end"``, ``"start"`` or ``None``.
    """
    device = frames.device if device is None else torch.device(device)
    length = stop - start
    out = frames[start:stop].clone()
    count = int(frames.shape[0])
    blend = max(0, min(int(blend), length - 1))
    if blend == 0:
        return out, None
    if start >= blend:
        for j in range(blend):
            share = _ease((j + 1) / (blend + 1))
            made = _morph(frames[stop - blend + j], frames[start - blend + j], share, device, mode)
            out[length - blend + j] = made.to(out.dtype).to(out.device)
        return out, "end"
    if stop + blend <= count - 1:
        for j in range(blend):
            share = _ease((j + 1) / (blend + 1))
            made = _morph(frames[stop + j], frames[start + j], share, device, mode)
            out[j] = made.to(out.dtype).to(out.device)
        return out, "start"
    return out, None
