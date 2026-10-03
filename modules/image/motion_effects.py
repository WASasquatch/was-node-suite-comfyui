"""Effects drawn along a clip's measured motion: what moves, trails behind it and datamosh.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import camera_motion, optical_flow
from .motion import CUT_SHARE

__all__ = ["AT_CUTS", "BLENDS", "datamosh", "moving_mask", "trails"]

#: Share of the background's own speed added to the threshold, so a fast-sliding surface needs a
#: proportionally larger difference to count as moving.
BACKGROUND_SHARE = 0.2

#: How a trail is laid over the frame.
BLENDS = ("normal", "lighten", "add")

#: What datamosh does where the clip cuts: carry the old picture into the new shot, or start
#: clean.
AT_CUTS = ("carry", "reset")


def moving_mask(motion, index: int, frame, threshold: float, ignore_camera: bool = True,
                grow: int = 0, feather: float = 0.0, device=None):
    """Where a frame moves faster than ``threshold``, softened at the edge.

    Args:
        motion: A :class:`~.motion.Motion`.
        index: The frame.
        frame: ``(height, width)`` to answer at.
        threshold: Speed, in pixels per frame of the frame's own size, that counts as moving,
            raised by :data:`BACKGROUND_SHARE` of the background's own speed beneath.
        ignore_camera: Take the camera's own motion away first.
        grow: Pixels the mask is widened by.
        feather: Gaussian sigma, in pixels, of the softened edge.
        device: Where the work runs.

    Returns:
        ``(1, 1, height, width)`` in ``[0, 1]``.
    """
    speed, under = camera_motion.moving_speed(motion, index, frame, ignore_camera, device)
    needed = float(threshold) + BACKGROUND_SHARE * under
    ramp = (0.5 * needed).clamp(min=0.25)
    mask = ((speed - needed) / ramp).clamp(0.0, 1.0)
    changed = camera_motion.change_after_camera(motion, index, ignore_camera, device)
    if changed is not None:
        mask = mask * F.interpolate(changed, size=tuple(frame), mode="bilinear", align_corners=False)
    if grow > 0:
        mask = F.max_pool2d(mask, 2 * int(grow) + 1, stride=1, padding=int(grow))
    if feather > 0:
        mask = optical_flow.gaussian(mask, float(feather))
    return mask.clamp(0.0, 1.0)


def _blend(base, layer, alpha, mode: str, clip: bool):
    """``layer`` over ``base`` by ``alpha`` in one of :data:`BLENDS`."""
    if mode == "lighten":
        out = base + alpha * (torch.maximum(base, layer) - base)
    elif mode == "add":
        out = base + alpha * layer
    else:
        out = base + alpha * (layer - base)
    return out.clamp(0.0, 1.0) if clip else out


def trails(frames, motion, length: float = 8.0, spacing: int = 1, opacity: float = 0.6,
           threshold: float = 1.0, blend: str = "normal", ignore_camera: bool = True,
           device=None, progress=None):
    """Leave a fading trail behind everything that moves.

    Args:
        frames: The clip.
        motion: A :class:`~.motion.Motion` measured from it.
        length: Frames a trail takes to fade to about a third.
        spacing: Frames between the copies a trail is built from; 1 is a continuous smear.
        opacity: Strength of the trail, 0 to 1.
        threshold: Speed, in pixels per frame, that leaves a trail.
        blend: One of :data:`BLENDS`.
        ignore_camera: Leave the camera's own motion out, so a pan does not trail the scene.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame.

    Returns:
        The trailed clip, shaped like ``frames``.
    """
    count, height, width, channels = (int(v) for v in frames.shape)
    device = frames.device if device is None else torch.device(device)
    decay = math.exp(-1.0 / max(float(length), 0.1))
    spacing = max(1, int(spacing))
    clip = bool(frames.min() >= 0.0 and frames.max() <= 1.0)
    out = torch.empty_like(frames)
    gathered = weights = None
    for index in range(count):
        frame = frames[index:index + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        if index == 0 or float(motion.agreement[index - 1]) < CUT_SHARE:
            gathered = torch.zeros_like(frame)
            weights = torch.zeros_like(frame[:, :1])
        moving = moving_mask(motion, index, (height, width), threshold, ignore_camera, 1, 1.0, device)
        gathered = gathered * decay
        weights = weights * decay
        if index % spacing == 0:
            gathered = gathered + frame * moving
            weights = weights + moving
        ghost = gathered / weights.clamp(min=1e-6)
        alpha = float(opacity) * weights.clamp(0.0, 1.0) * (1.0 - moving)
        result = _blend(frame, ghost, alpha, str(blend), clip)
        out[index] = result[0].permute(1, 2, 0).to(out.dtype).to(out.device)
        if progress is not None:
            progress(1)
    return out


def _blocky(flow, block: int):
    """``flow`` averaged over ``block`` by ``block`` squares, or as it is for 0 or 1."""
    if block <= 1:
        return flow
    _, _, height, width = flow.shape
    pooled = F.avg_pool2d(flow, block, stride=block, ceil_mode=True)
    return F.interpolate(pooled, scale_factor=block, mode="nearest")[..., :height, :width]


def datamosh(frames, motion, start: int = 0, amplify: float = 1.0, refresh: float = 0.0,
             keyframe_every: int = 0, block: int = 16, at_cuts: str = "carry",
             residual: float = 1.0, device=None, progress=None):
    """Push each frame's pixels along the motion instead of showing the next frame.

    Args:
        frames: The clip.
        motion: A :class:`~.motion.Motion` measured from it.
        start: First frame that is moshed; the frames before it play as they are.
        amplify: How far the motion is exaggerated; 1 follows it as measured.
        refresh: Share of each real frame mixed back in, 0 to 1.
        keyframe_every: Frames between clean frames, counted from ``start``; 0 never.
        block: Side of the squares the motion moves in; 16 is codec-like, 0 per pixel.
        at_cuts: One of :data:`AT_CUTS`.
        residual: Share of each frame's own change carried in with the motion; 1 holds a shot
            together and moshes only across cuts, 0 melts.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame.

    Returns:
        The moshed clip, shaped like ``frames``.
    """
    count, height, width, channels = (int(v) for v in frames.shape)
    device = frames.device if device is None else torch.device(device)
    start = max(0, min(int(start), count - 1))
    clip = bool(frames.min() >= 0.0 and frames.max() <= 1.0)
    out = frames.clone()
    picture = frames[start:start + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    if progress is not None:
        progress(start + 1)
    previous = picture
    for index in range(start + 1, count):
        real = frames[index:index + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        if keyframe_every > 0 and (index - start) % int(keyframe_every) == 0:
            picture = real
        else:
            found = motion.toward(index, -1, device)
            if found is None:
                if at_cuts == "reset":
                    picture = real
            else:
                flow = _blocky(optical_flow.resize_flow(found[0], height, width) * float(amplify), int(block))
                picture = optical_flow.warp(picture, flow)
                if residual > 0:
                    picture = picture + float(residual) * (real - optical_flow.warp(previous, flow))
            if refresh > 0:
                picture = picture + float(refresh) * (real - picture)
        previous = real
        if clip:
            picture = picture.clamp(0.0, 1.0)
        out[index] = picture[0].permute(1, 2, 0).to(out.dtype).to(out.device)
        if progress is not None:
            progress(1)
    return out
