"""Film motion blur for a frame sequence, drawn along each pixel's measured path.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. A path is ``p + a t + c t^2``
for ``t`` in ``[-1, 1]``, in pixels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..media import clip as clips
from . import motion as motion_field
from . import optical_flow

__all__ = [
    "LAYERS",
    "MAX_SAMPLES",
    "MAX_SHUTTER",
    "blur_frames",
    "render",
]

#: Widest shutter accepted, in degrees. 360 exposes the whole frame interval.
MAX_SHUTTER = 720.0

#: Most samples taken along a path.
MAX_SAMPLES = 256

#: Longest half-path drawn, in pixels of the frame.
MAX_REACH = 160

#: Smallest tile the dominant motion is gathered over, in pixels.
MIN_TILE = 8

#: Nearness difference over which one surface goes from level with another to wholly in front.
SOFT_DEPTH = 0.05

#: How near a sample's own path must bring it to count as covering a pixel: pixels, plus a share
#: of the distance it was taken from.
LANDING = (1.0, 0.2)

#: Width, in pixels of the measured motion, of the band a mask edge refills.
MASK_BAND = 2

#: Which layers of a masked frame are blurred.
LAYERS = ("all", "background", "subject")


def _erode(mask, radius: int):
    """Shrink a ``(1, 1, height, width)`` mask by ``radius`` pixels."""
    if radius <= 0:
        return mask
    return -F.max_pool2d(-mask, 2 * radius + 1, stride=1, padding=radius)


def _refill(field, keep):
    """``field`` where ``keep`` holds, and the Gaussian-weighted mean of the kept part elsewhere."""
    total = optical_flow.gaussian(keep, 4.0)
    spread = optical_flow.gaussian(field * keep, 4.0) / total.clamp(min=1e-6)
    return torch.where(keep > 0.5, field, torch.where(total > 1e-3, spread, field))


def _jitter(height: int, width: int, device):
    """Interleaved gradient noise in ``[-0.5, 0.5)``, ``(1, 1, height, width)``."""
    ys = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    noise = torch.frac(52.9829189 * torch.frac(0.06711056 * xs + 0.00583715 * ys))
    return (noise - 0.5).view(1, 1, height, width)


def _neighbour_max(a, c, reach, tile: int):
    """The longest path among each pixel's surrounding tiles.

    Args:
        a: Linear path term.
        c: Curved path term.
        reach: ``|a| + |c|``, ``(1, 1, height, width)``.
        tile: Tile side in pixels.

    Returns:
        ``(a, c, reach)`` of the longest path within the 3 by 3 tiles around each pixel.
    """
    _, _, height, width = reach.shape
    pad_h, pad_w = (-height) % tile, (-width) % tile
    lengths = F.pad(reach, (0, pad_w, 0, pad_h), value=-1.0)
    paths = F.pad(torch.cat([a, c], 1), (0, pad_w, 0, pad_h))
    tile_reach, picked = F.max_pool2d(lengths, tile, stride=tile, return_indices=True)
    rows, columns = tile_reach.shape[-2:]
    tile_paths = paths.flatten(2).gather(2, picked.flatten(2).expand(-1, 4, -1)).view(1, 4, rows, columns)
    near_reach, near = F.max_pool2d(tile_reach, 3, stride=1, padding=1, return_indices=True)
    near_paths = tile_paths.flatten(2).gather(2, near.flatten(2).expand(-1, 4, -1)).view(1, 4, rows, columns)
    grown = torch.cat([near_paths, near_reach], 1)
    grown = grown.repeat_interleave(tile, 2).repeat_interleave(tile, 3)[..., :height, :width]
    return grown[:, 0:2], grown[:, 2:4], grown[:, 4:5]


def _base_grid(height: int, width: int, device):
    """Pixel centres as ``(1, height, width, 2)`` in ``grid_sample`` units."""
    ys = torch.linspace(-1.0, 1.0, height, device=device).view(1, height, 1).expand(1, height, width)
    xs = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, width).expand(1, height, width)
    return torch.stack([xs, ys], -1)


def _sample(pack, base, offset):
    """``pack`` read at ``p - offset``, bilinear, edges held."""
    _, _, height, width = pack.shape
    scale = torch.tensor([2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)], device=pack.device)
    grid = base - offset.permute(0, 2, 3, 1) * scale
    return F.grid_sample(pack, grid, mode="bilinear", padding_mode="border", align_corners=True)


def render(colour, a, c, near, samples: int):
    """Blur one frame along its paths.

    Args:
        colour: ``(1, channels, height, width)``.
        a: Linear path term, ``(1, 2, height, width)`` in pixels.
        c: Curved path term, shaped like ``a``.
        near: Nearness, ``(1, 1, height, width)``; higher is nearer the camera.
        samples: Points taken along each path.

    Returns:
        The blurred frame, shaped like ``colour``.
    """
    _, channels, height, width = colour.shape
    reach = a.norm(dim=1, keepdim=True) + c.norm(dim=1, keepdim=True)
    top = float(reach.max())
    if top < 0.5:
        return colour
    if top > MAX_REACH:
        shrink = (MAX_REACH / reach.clamp(min=MAX_REACH)).clamp(max=1.0)
        a, c, reach = a * shrink, c * shrink, reach * shrink
        top = MAX_REACH
    tile = int(min(MAX_REACH, max(MIN_TILE, math.ceil(top))))
    a_wide, c_wide, reach_wide = _neighbour_max(a, c, reach, tile)

    pack = torch.cat([colour, a, c, near], 1)
    at = channels
    base = _base_grid(height, width, colour.device)
    jitter = _jitter(height, width, colour.device)
    total = torch.zeros_like(colour)
    for index in range(samples):
        t = -1.0 + (2.0 * index + 1.0 + jitter) / samples
        t2 = t * t
        probes = []
        for path_a, path_c in ((a_wide, c_wide), (a, c)):
            offset = path_a * t + path_c * t2
            found = _sample(pack, base, offset)
            moved = found[:, at:at + 2] * t + found[:, at + 2:at + 4] * t2
            gap = (moved - offset).norm(dim=1, keepdim=True)
            tolerance = LANDING[0] + LANDING[1] * offset.norm(dim=1, keepdim=True)
            lands = (1.5 - gap / tolerance).clamp(0.0, 1.0)
            probes.append((found[:, :channels], found[:, at + 4:at + 5], lands))
        (wide, wide_near, wide_lands), (own, own_near, own_lands) = probes
        front = (0.5 + (wide_near - own_near) / (2.0 * SOFT_DEPTH)).clamp(0.0, 1.0)
        both = wide_lands * own_lands
        wide_seen = wide_lands - both + both * front
        own_seen = own_lands - both + both * (1.0 - front)
        hidden = (1.0 - wide_lands) * (1.0 - own_lands)
        wide_behind = (1.0 - (wide_near - near) / SOFT_DEPTH).clamp(0.0, 1.0)
        own_behind = (1.0 - (own_near - near) / SOFT_DEPTH).clamp(0.0, 1.0)
        background = (wide_behind * wide + own_behind * own + 0.05 * colour) / (wide_behind + own_behind + 0.05)
        total += wide_seen * wide + own_seen * own + hidden * background
    return torch.where(reach_wide > 0.5, total / samples, colour)


def _plane(batch, index: int, size, device):
    """One ``(1, 1, height, width)`` plane of a mask or greyscale batch, resized to ``size``."""
    plane = batch[min(index, batch.shape[0] - 1)].to(device=device, dtype=torch.float32)
    if plane.ndim == 3:
        plane = plane[..., :3].mean(-1) if plane.shape[-1] >= 3 else plane[..., 0]
    plane = plane.view(1, 1, *plane.shape[-2:])
    if tuple(plane.shape[-2:]) != tuple(size):
        plane = F.interpolate(plane, size=tuple(size), mode="bilinear", align_corners=False)
    return plane.clamp(0.0, 1.0)


def _depth_range(depth) -> tuple[float, float]:
    """The 1st and 99th percentile of a depth batch, read from a strided sample."""
    values = depth[..., :3].float().mean(-1) if depth.ndim == 4 else depth.float()
    flat = values.reshape(-1)
    step = max(1, flat.numel() // 1_000_000)
    sample = flat[::step]
    low = float(torch.quantile(sample, 0.01))
    high = float(torch.quantile(sample, 0.99))
    return low, max(high, low + 1e-6)


def blur_frames(
    frames,
    shutter=180.0,
    samples: int = 32,
    motion_side: int = motion_field.MOTION_SIDE,
    mask=None,
    depth=None,
    layers: str = "all",
    device=None,
    progress=None,
    motion=None,
):
    """Blur every frame of a sequence along the motion measured between its frames.

    Args:
        frames: ``(frames, height, width, channels)`` in ``[0, 1]``.
        shutter: Shutter angle in degrees, or one per frame; 360 spans one frame interval.
        samples: Points taken along each path.
        motion_side: Long side motion is measured at when ``motion`` is not given.
        mask: Optional ``(frames, height, width)`` subject mask, 1 on the subject.
        depth: Optional ``(frames, height, width, channels)`` depth, white nearest.
        layers: One of :data:`LAYERS`; which side of ``mask`` is blurred.
        device: Where the work runs. Defaults to the frames' own device.
        progress: Optional callable taking a step count, called as work completes.
        motion: A :class:`~.motion.Motion` measured from these frames, or ``None`` to
            measure here.

    Returns:
        ``(blurred, pictures)``: the blurred frames on the frames' device, and
        ``(frames, h, w, 3)`` pictures of the paths at the measured size.
    """
    count, height, width, channels = (int(v) for v in frames.shape)
    device = frames.device if device is None else torch.device(device)
    angles = clips.stretch(shutter if isinstance(shutter, (list, tuple)) else [shutter], count)
    samples = max(1, min(int(samples), MAX_SAMPLES))
    blurred = torch.empty_like(frames)
    if motion is None:
        if count < 2 or max(angles) <= 0.0:
            size = motion_field.working_size(height, width, int(motion_side))
            blurred.copy_(frames)
            if progress is not None:
                progress(max(count - 1, 0) + count)
            return blurred, torch.zeros(count, size[0], size[1], 3)
        motion = motion_field.measure(frames, int(motion_side), device, progress)
    size = motion.size
    pictures = torch.zeros(count, size[0], size[1], 3)
    if count < 2:
        blurred.copy_(frames)
        if progress is not None:
            progress(count)
        return blurred, pictures

    depth_range = _depth_range(depth) if depth is not None else None
    for index in range(count):
        half = 0.5 * max(0.0, angles[index]) / 360.0
        if half <= 0.0:
            blurred[index] = frames[index]
            if progress is not None:
                progress(1)
            continue
        a, c, occluded = motion.paths(index, device)
        subject = _plane(mask, index, size, device) if mask is not None else None
        if subject is not None:
            inside = _erode((subject > 0.5).float(), MASK_BAND)
            outside = _erode((subject <= 0.5).float(), MASK_BAND)
            a_in, c_in = _refill(a, inside), _refill(c, inside)
            a_out, c_out = _refill(a, outside), _refill(c, outside)
            if layers == "background":
                a_in, c_in = torch.zeros_like(a_in), torch.zeros_like(c_in)
            elif layers == "subject":
                a_out, c_out = torch.zeros_like(a_out), torch.zeros_like(c_out)
            full_mask = _plane(mask, index, (height, width), device)
            a = full_mask * optical_flow.resize_flow(a_in, height, width) + (1.0 - full_mask) * optical_flow.resize_flow(a_out, height, width)
            c = full_mask * optical_flow.resize_flow(c_in, height, width) + (1.0 - full_mask) * optical_flow.resize_flow(c_out, height, width)
        else:
            full_mask = None
            a = optical_flow.resize_flow(a, height, width)
            c = optical_flow.resize_flow(c, height, width)
        a = a * half
        c = c * (half * half)

        if depth is not None:
            low, high = depth_range
            near = ((_plane(depth, index, (height, width), device) - low) / (high - low)).clamp(0.0, 1.0)
            near = 0.5 * near + 0.5 * full_mask if full_mask is not None else near
        elif full_mask is not None:
            near = full_mask
        else:
            band = optical_flow.gaussian(occluded, 1.5)
            near = optical_flow.resize(0.5 - 0.5 * band.clamp(0.0, 1.0), height, width)

        colour = frames[index].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        out = render(colour, a, c, near, samples)
        blurred[index] = out[0].permute(1, 2, 0).to(blurred.dtype).to(blurred.device)
        small = F.interpolate(torch.cat([a, c], 1), size=size, mode="bilinear", align_corners=False)
        pictures[index] = motion_field.visualise(small[:, 0:2], small[:, 2:4]).cpu()
        if progress is not None:
            progress(1)
    return blurred, pictures
