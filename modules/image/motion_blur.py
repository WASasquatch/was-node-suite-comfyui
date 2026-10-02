"""Film motion blur for a frame sequence, drawn along each pixel's measured path.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. A path is ``p + a t + c t^2``
for ``t`` in ``[-1, 1]``, in pixels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import optical_flow

__all__ = [
    "LAYERS",
    "MAX_SAMPLES",
    "MAX_SHUTTER",
    "MOTION_SIDE",
    "blur_frames",
    "render",
    "visualise",
]

#: Widest shutter accepted, in degrees. 360 exposes the whole frame interval.
MAX_SHUTTER = 720.0

#: Most samples taken along a path.
MAX_SAMPLES = 256

#: Long side, in pixels, motion is measured at by default.
MOTION_SIDE = 768

#: Longest half-path drawn, in pixels of the frame.
MAX_REACH = 160

#: Smallest tile the dominant motion is gathered over, in pixels.
MIN_TILE = 8

#: Frame pairs measured together.
PAIR_BATCH = 8

#: Share of agreeing pixels below which a frame pair is treated as a cut.
CUT_SHARE = 0.4

#: Nearness difference over which one surface goes from level with another to wholly in front.
SOFT_DEPTH = 0.05

#: How near a sample's own path must bring it to count as covering a pixel: pixels, plus a share
#: of the distance it was taken from.
LANDING = (1.0, 0.2)

#: Distances, in pixels of the measured motion, a pixel looks for a motion that matches better.
SNAP_REACH = (4, 8, 16, 24)

#: Half the window a motion's match against the neighbouring frame is summed over, in pixels of
#: the measured motion.
SNAP_RADIUS = 3

#: Spread of motion, in pixels per frame of the measured motion, within :data:`SNAP_REACH` of a
#: pixel that marks it as near a motion edge.
SNAP_EDGE = 3.0

#: Share of its own match cost a neighbour's motion has to beat to be taken.
SNAP_BIAS = 0.85

#: How much worse, as a share plus a floor in levels of 255, one side's match may be than the
#: other's and still count as seen from that side.
SIDE_MATCH = (1.5, 2.0)

#: Width, in pixels of the measured motion, of the band a mask edge refills.
MASK_BAND = 2

#: Half-path length, in pixels, drawn at half brightness by :func:`visualise`.
VISUAL_REACH = 8.0

#: Which layers of a masked frame are blurred.
LAYERS = ("all", "background", "subject")


def _working_size(height: int, width: int, side: int) -> tuple[int, int]:
    """The size motion is measured at: long side ``side``, never above the frame's own, even sides."""
    longest = max(height, width)
    if side <= 0 or side >= longest:
        return height, width
    scale = side / longest
    return max(16, 2 * int(round(height * scale / 2))), max(16, 2 * int(round(width * scale / 2)))


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


def _pair_flows(frames, size, device, progress):
    """Forward and backward flow for every neighbouring pair, at ``size``.

    Returns:
        ``(forward, backward, cut, luma)``: flows ``(frames - 1, 2, h, w)``, a bool per pair that
        is True where the pair is treated as a cut, and ``(frames, 1, h, w)`` luminance, all on
        the CPU.
    """
    count = int(frames.shape[0])
    height, width = size
    forward = torch.zeros(max(count - 1, 0), 2, height, width)
    backward = torch.zeros_like(forward)
    cut = torch.zeros(max(count - 1, 0), dtype=torch.bool)
    luma = torch.zeros(count, 1, height, width)
    for start in range(0, count - 1, PAIR_BATCH):
        stop = min(start + PAIR_BATCH, count - 1)
        chunk = frames[start:stop + 1].to(device)
        planes = optical_flow.resize(optical_flow.luminance(chunk), height, width)
        del chunk
        first, second = planes[:-1], planes[1:]
        ahead = optical_flow.estimate(first, second)
        behind = optical_flow.estimate(second, first)
        share = optical_flow.consistent(ahead, behind).float().mean((1, 2, 3))
        forward[start:stop] = ahead.cpu()
        backward[start:stop] = behind.cpu()
        cut[start:stop] = (share < CUT_SHARE).cpu()
        luma[start:stop + 1] = planes.cpu()
        if progress is not None:
            progress(stop - start)
    return forward, backward, cut, luma


def _shifted(x, dx: int, dy: int):
    """``x`` read ``(dx, dy)`` pixels away, edges held."""
    height, width = x.shape[-2:]
    pad = max(abs(dx), abs(dy))
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return padded[..., pad + dy:pad + dy + height, pad + dx:pad + dx + width]


def _match_cost(here, flow, target):
    """Windowed mean difference between a frame and its neighbour pulled back along ``flow``.

    Returns:
        ``(1, 1, h, w)`` in levels of 255, averaged over a ``2 * SNAP_RADIUS + 1`` window.
    """
    mismatch = (optical_flow.warp(target, flow) - here).abs()
    window = 2 * SNAP_RADIUS + 1
    return F.avg_pool2d(F.pad(mismatch, (SNAP_RADIUS,) * 4, mode="replicate"), window, stride=1)


def _snap(here, flow, target):
    """Give pixels near a motion edge the motion, their own or a neighbour's, that best matches.

    Args:
        here: Luminance of the frame, ``(1, 1, h, w)``.
        flow: Flow from the frame onto ``target``, ``(1, 2, h, w)``.
        target: Luminance of the frame the flow maps onto.

    Returns:
        ``(flow, cost)``: the flow with its edges moved onto the edges the frames agree on, and
        its match cost from :func:`_match_cost`.
    """
    reach = max(SNAP_REACH)
    spread = (
        F.max_pool2d(flow, 2 * reach + 1, stride=1, padding=reach)
        + F.max_pool2d(-flow, 2 * reach + 1, stride=1, padding=reach)
    ).amax(1, keepdim=True)
    edge = spread > SNAP_EDGE
    own_cost = _match_cost(here, flow, target)
    if not bool(edge.any()):
        return flow, own_cost
    best = flow
    best_cost = own_cost * SNAP_BIAS
    for distance in SNAP_REACH:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            candidate = _shifted(flow, dx * distance, dy * distance)
            candidate_cost = _match_cost(here, candidate, target)
            take = (candidate_cost < best_cost) & edge
            best = torch.where(take, candidate, best)
            best_cost = torch.where(take, candidate_cost, best_cost)
    return best, torch.minimum(best_cost, own_cost)


def _frame_paths(index: int, forward, backward, cut, luma, device):
    """Path terms and the occlusion band for one frame, at the measured size, in frames.

    Returns:
        ``(a, c, occluded)``: ``a`` and ``c`` scaled to a whole frame interval either side, and
        a ``(1, 1, h, w)`` float that is 1 where the pixel is hidden in a neighbour.
    """
    count = forward.shape[0] + 1
    ahead = forward[index:index + 1].to(device) if index < count - 1 and not cut[index] else None
    behind = backward[index - 1:index].to(device) if index > 0 and not cut[index - 1] else None
    if ahead is None and behind is None:
        height, width = forward.shape[-2:]
        zero = torch.zeros(1, 2, height, width, device=device)
        return zero, zero, torch.zeros(1, 1, height, width, device=device)
    here = luma[index:index + 1].to(device)
    if behind is None:
        agree = optical_flow.consistent(ahead, backward[index:index + 1].to(device))
        ahead, _ = _snap(here, ahead, luma[index + 1:index + 2].to(device))
        return ahead, torch.zeros_like(ahead), (~agree).float()
    if ahead is None:
        agree = optical_flow.consistent(behind, forward[index - 1:index].to(device))
        behind, _ = _snap(here, behind, luma[index - 1:index].to(device))
        return -behind, torch.zeros_like(behind), (~agree).float()
    ahead_ok = optical_flow.consistent(ahead, backward[index:index + 1].to(device))
    behind_ok = optical_flow.consistent(behind, forward[index - 1:index].to(device))
    ahead, ahead_cost = _snap(here, ahead, luma[index + 1:index + 2].to(device))
    behind, behind_cost = _snap(here, behind, luma[index - 1:index].to(device))
    share, floor = SIDE_MATCH
    ahead_ok = ahead_ok & (ahead_cost <= share * behind_cost + floor)
    behind_ok = behind_ok & (behind_cost <= share * ahead_cost + floor)
    both = ahead_ok & behind_ok
    central = 0.5 * (ahead - behind)
    a = torch.where(both, central, torch.where(ahead_ok, ahead, torch.where(behind_ok, -behind, central)))
    c = torch.where(both, 0.5 * (ahead + behind), torch.zeros_like(ahead))
    return a, c, (~both).float()


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


def visualise(a, c):
    """A picture of one frame's paths: hue for direction, brightness for length.

    Args:
        a: Linear path term, ``(1, 2, height, width)`` in pixels of the frame.
        c: Curved path term, shaped like ``a``.

    Returns:
        ``(height, width, 3)`` in ``[0, 1]``.
    """
    reach = a.norm(dim=1)[0] + c.norm(dim=1)[0]
    hue = (torch.atan2(a[0, 1], a[0, 0]) / (2.0 * math.pi)) % 1.0
    value = reach / (reach + VISUAL_REACH)
    sector = hue * 6.0
    channels = []
    for shift in (5.0, 3.0, 1.0):
        k = (shift + sector) % 6.0
        channels.append(value * (1.0 - torch.clamp(torch.minimum(k, 4.0 - k), 0.0, 1.0)))
    return torch.stack(channels, -1)


def blur_frames(
    frames,
    shutter: float = 180.0,
    samples: int = 32,
    motion_side: int = MOTION_SIDE,
    mask=None,
    depth=None,
    layers: str = "all",
    device=None,
    progress=None,
):
    """Blur every frame of a sequence along the motion measured between its frames.

    Args:
        frames: ``(frames, height, width, channels)`` in ``[0, 1]``.
        shutter: Shutter angle in degrees; 360 spans one frame interval.
        samples: Points taken along each path.
        motion_side: Long side motion is measured at; 0 measures at the frame's own size.
        mask: Optional ``(frames, height, width)`` subject mask, 1 on the subject.
        depth: Optional ``(frames, height, width, channels)`` depth, white nearest.
        layers: One of :data:`LAYERS`; which side of ``mask`` is blurred.
        device: Where the work runs. Defaults to the frames' own device.
        progress: Optional callable taking a step count, called as work completes.

    Returns:
        ``(blurred, motion)``: the blurred frames on the frames' device, and
        ``(frames, h, w, 3)`` pictures of the paths at the measured size.
    """
    count, height, width, channels = (int(v) for v in frames.shape)
    device = frames.device if device is None else torch.device(device)
    size = _working_size(height, width, int(motion_side))
    exposure = max(0.0, float(shutter)) / 360.0
    samples = max(1, min(int(samples), MAX_SAMPLES))
    blurred = torch.empty_like(frames)
    motion = torch.zeros(count, size[0], size[1], 3)
    if count < 2 or exposure <= 0.0:
        blurred.copy_(frames)
        if progress is not None:
            progress(max(count - 1, 0) + count)
        return blurred, motion

    forward, backward, cut, luma = _pair_flows(frames, size, device, progress)
    depth_range = _depth_range(depth) if depth is not None else None
    half = 0.5 * exposure
    for index in range(count):
        a, c, occluded = _frame_paths(index, forward, backward, cut, luma, device)
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
        motion[index] = visualise(small[:, 0:2], small[:, 2:4]).cpu()
        if progress is not None:
            progress(1)
    return blurred, motion
