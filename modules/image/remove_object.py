"""Removing an object from a clip with background other frames show, carried along its motion.

Frames are ``(frames, height, width, channels)``; masks ``(frames, height, width)``, 1 on the
object. Positions are frame pixels, x first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .. import log
from . import motion as motion_field
from . import optical_flow

__all__ = [
    "MASK_LEVEL",
    "Removal",
    "check_masks",
    "fill_frames",
    "remove",
]

logger = log.get_logger("image.remove_object")

#: Mask value above which a pixel belongs to the object.
MASK_LEVEL = 0.05

#: Pixels, at the measured size, around the object whose measured motion is replaced by motion
#: completed from further out.
TRUST_MARGIN = 8

#: Pixels a source position keeps from the object in its own frame.
SOURCE_MARGIN = 2

#: Forward and backward motion a trace follows: share of the squared motion allowed as
#: disagreement, and a floor in squared pixels of the measured size.
CHAIN_AGREEMENT = (0.05, 4.0)

#: Mean difference, in 0 to 1, between the candidates from either side above which only the
#: nearer one is kept.
DISAGREE = 0.08

#: Solver passes at most, and the residual share at which it stops.
SOLVE_STEPS = 60
SOLVE_TOLERANCE = 1e-4

#: Root mean residual per pixel, in the values' own unit, at which a solve stops.
SOLVE_FLOOR = 1e-5

#: Passes between two convergence checks.
CHECK_EVERY = 4

#: Side, in pixels, at which the solver's pyramid stops halving.
COARSEST = 8

#: Smoothing sweeps either side of each coarser level, sweeps on the coarsest, and their damping.
SWEEPS = 2
COARSE_SWEEPS = 24
SMOOTHING = 0.8

#: Pixels, frames times crop, one batched solve holds at most.
SOLVE_PIXELS = 4_000_000

#: Frame pairs whose motion is completed together at most.
FLOW_BATCH = 16

#: Share of a frame the object may cover.
MOST_COVERED = 0.98

#: Share of free device memory the traced pixels of one window may take.
SPARE = 0.3

#: Device bytes one traced pixel takes with its candidates and its trace.
TRACE_BYTES = 256

#: Offsets, in pixels, of the samples beside each candidate.
TAP_ACROSS = (1.0, 0.0)
TAP_BELOW = (0.0, 1.0)

#: Traced pixels one window holds at most.
MOST_TRACED = 24_000_000

#: Share of the most pixels any frame has left that the frame offered for an edit holds at least.
EDIT_SHARE = 0.5


@dataclass
class Removal:
    """What a removal produced.

    Attributes:
        images: ``(frames, height, width, channels)`` with the object replaced.
        remaining: ``(frames, height, width)``, 1 where no frame and no fill showed the background.
        holes: Masked pixels per frame.
        left: Pixels per frame still without a source.
        engine: What measured the motion.
    """

    images: torch.Tensor
    remaining: torch.Tensor
    holes: list = field(default_factory=list)
    left: list = field(default_factory=list)
    engine: str = ""

    @property
    def edit_frame(self) -> int:
        """The middle in time of what is left, among frames holding :data:`EDIT_SHARE` of the most; 0 when none is."""
        if not self.left or max(self.left) <= 0:
            return 0
        total, running, middle = sum(self.left), 0, 0
        for index, value in enumerate(self.left):
            running += value
            if 2 * running >= total:
                middle = index
                break
        floor = EDIT_SHARE * max(self.left)
        held = [index for index, value in enumerate(self.left) if value >= floor]
        return int(min(held, key=lambda index: (abs(index - middle), index)))


def check_masks(mask, count: int, height: int, width: int, node: str):
    """The mask batch as one plane per frame at the frame's size, on the CPU.

    Args:
        mask: ``(masks, height, width)`` or ``(height, width)``.
        count: Frames in the clip.
        height: Their height.
        width: Their width.
        node: The node's name, for messages.

    Returns:
        ``(masks, height, width)`` float32, ``masks`` being 1 or ``count``.

    Raises:
        ValueError: The count matches neither, or the shape differs from the frames'.
    """
    held = mask
    if held.ndim == 2:
        held = held[None]
    if held.ndim == 4 and held.shape[-1] == 1:
        held = held[..., 0]
    if held.ndim != 3:
        raise ValueError(f"{node} needs a MASK batch on mask, and was given a tensor of {held.ndim} axes.")
    masks, mask_height, mask_width = (int(side) for side in held.shape)
    if masks not in (1, count):
        raise ValueError(
            f"{node} was given {masks} mask(s) for {count} frame(s). Connect one mask per frame, "
            f"such as Image Track Subject's, or a single mask for every frame."
        )
    shape = (mask_width / max(mask_height, 1)) / (width / max(height, 1))
    if abs(shape - 1.0) > 0.01:
        raise ValueError(
            f"{node} was given masks of {mask_width}x{mask_height} for frames of {width}x{height}. "
            f"Make the mask from these frames, or scale it to {width}x{height}."
        )
    held = held.float().cpu()
    if (mask_height, mask_width) != (height, width):
        held = F.interpolate(held[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return held


def fill_frames(fill, frames, count: int, node: str) -> list[int]:
    """Which clip frame each fill picture shows.

    Args:
        fill: The ``(pictures, height, width, channels)`` batch.
        frames: An int, or text such as ``"12"`` or ``"0, 40, 80"``.
        count: Frames in the clip.
        node: The node's name, for messages.

    Returns:
        One frame index per picture.

    Raises:
        ValueError: The text holds something other than whole numbers, a frame is outside the
            clip or named twice, or the count differs from the pictures'.
    """
    pictures = int(fill.shape[0])
    if isinstance(frames, bool):
        frames = int(frames)
    if isinstance(frames, (int, float)):
        found = [int(frames)]
    else:
        text = str(frames or "").replace(",", " ").replace(";", " ").split()
        try:
            found = [int(part) for part in text]
        except ValueError as error:
            raise ValueError(
                f"{node}: fill_frames takes whole frame numbers such as 12 or 0, 40, 80, and was "
                f"given {str(frames)[:40]!r}."
            ) from error
    if not found:
        found = [0]
    if len(found) != pictures:
        raise ValueError(
            f"{node} was given {pictures} fill picture(s) and {len(found)} frame number(s) in "
            f"fill_frames. Name the clip frame of each picture, in order, such as 0, 40, 80."
        )
    for index in found:
        if index < 0 or index >= count:
            raise ValueError(
                f"{node}: fill frame {index} is outside the clip, which holds frames 0 to {count - 1}. "
                f"Set fill_frames to the frame each fill picture was made from."
            )
    if len(set(found)) != len(found):
        raise ValueError(f"{node}: fill_frames names a frame twice; give each fill picture its own frame.")
    return found


def _dilate(plane: torch.Tensor, radius: int) -> torch.Tensor:
    """A boolean ``(height, width)`` plane grown by ``radius`` pixels, as a square."""
    if radius <= 0:
        return plane
    grown = plane.float()[None, None]
    grown = F.max_pool2d(grown, (2 * radius + 1, 1), stride=1, padding=(radius, 0))
    grown = F.max_pool2d(grown, (1, 2 * radius + 1), stride=1, padding=(0, radius))
    return grown[0, 0] > 0.5


def _object(masks: torch.Tensor, index: int, grow: int, device) -> torch.Tensor:
    """Frame ``index``'s object grown by ``grow`` pixels, ``(height, width)`` bool on ``device``."""
    plane = masks[index if int(masks.shape[0]) > 1 else 0].to(device, torch.float32)
    return _dilate(plane > MASK_LEVEL, grow)


def _box(plane: torch.Tensor, margin: int):
    """``(y0, y1, x0, x1)`` around the true pixels of a plane, grown by ``margin`` and held to it."""
    height, width = plane.shape
    rows = torch.nonzero(plane.any(dim=1)).flatten()
    columns = torch.nonzero(plane.any(dim=0)).flatten()
    y0 = max(int(rows[0]) - margin, 0)
    y1 = min(int(rows[-1]) + margin + 1, height)
    x0 = max(int(columns[0]) - margin, 0)
    x1 = min(int(columns[-1]) + margin + 1, width)
    return y0, y1, x0, x1


def _union(boxes) -> tuple:
    """The smallest box holding every ``(y0, y1, x0, x1)`` given."""
    boxes = list(boxes)
    return (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes))


def _neighbours(z: torch.Tensor) -> torch.Tensor:
    """Sum of the four neighbours of every pixel, nothing past the edges."""
    total = torch.zeros_like(z)
    total[..., :, 1:] += z[..., :, :-1]
    total[..., :, :-1] += z[..., :, 1:]
    total[..., 1:, :] += z[..., :-1, :]
    total[..., :-1, :] += z[..., 1:, :]
    return total


def _pushpull(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fill the pixels a weight leaves empty with a smooth spread of the rest.

    Args:
        values: ``(batch, channels, height, width)``.
        weight: ``(batch, 1, height, width)``, 1 where a value is known.

    Returns:
        ``(batch, channels, height, width)``.
    """
    levels = [(values * weight, weight)]
    while min(levels[-1][1].shape[-2:]) > 2:
        summed, held = levels[-1]
        levels.append((F.avg_pool2d(summed, 2, ceil_mode=True), F.avg_pool2d(held, 2, ceil_mode=True)))
    summed, held = levels[-1]
    mean = summed.sum((2, 3), keepdim=True) / held.sum((2, 3), keepdim=True).clamp_min(1e-8)
    filled = torch.where(held > 1e-8, summed / held.clamp_min(1e-8), mean.expand_as(summed))
    for summed, held in reversed(levels[:-1]):
        up = F.interpolate(filled, size=summed.shape[-2:], mode="bilinear", align_corners=False)
        filled = torch.where(held > 1e-8, summed / held.clamp_min(1e-8), up)
    return filled


def _divergence(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """What a guidance field asks of each pixel's four differences, summed."""
    total = -gx - gy
    total[..., :, 1:] += gx[..., :, :-1]
    total[..., 1:, :] += gy[..., :-1, :]
    return total


class _Level:
    """One level of the solver's pyramid: its unknown pixels and their neighbour counts."""

    def __init__(self, inside: torch.Tensor):
        self.inside = inside
        self.counts = _neighbours(torch.ones_like(inside))
        self.inverse = inside / self.counts.clamp_min(1.0)

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """The masked Laplacian of ``z``."""
        return (self.counts * z - _neighbours(z)) * self.inside


def _pyramid(inside: torch.Tensor) -> list:
    """Levels from the full plane down to :data:`COARSEST` pixels a side."""
    levels = [_Level(inside)]
    while min(levels[-1].inside.shape[-2:]) > COARSEST:
        levels.append(_Level(F.max_pool2d(levels[-1].inside, 2, ceil_mode=True)))
    return levels


def _cycle(levels: list, depth: int, residual: torch.Tensor) -> torch.Tensor:
    """One symmetric V-cycle from zero: an approximate inverse of the masked Laplacian."""
    level = levels[depth]
    z = SMOOTHING * residual * level.inverse
    if depth == len(levels) - 1:
        for _ in range(COARSE_SWEEPS):
            z = z + SMOOTHING * (residual - level.apply(z)) * level.inverse
        return z
    for _ in range(SWEEPS - 1):
        z = z + SMOOTHING * (residual - level.apply(z)) * level.inverse
    left = (residual - level.apply(z)) * level.inside
    lower = levels[depth + 1]
    up = _cycle(levels, depth + 1, F.avg_pool2d(left, 2, ceil_mode=True) * 2.0 * lower.inside)
    height, width = residual.shape[-2:]
    z = z + F.interpolate(up, scale_factor=2, mode="nearest")[..., :height, :width] * level.inside
    for _ in range(SWEEPS):
        z = z + SMOOTHING * (residual - level.apply(z)) * level.inverse
    return z


def _correction(residual: torch.Tensor, inside: torch.Tensor) -> torch.Tensor:
    """Solve the masked Laplacian for a residual: conjugate gradients with a V-cycle as preconditioner.

    Args:
        residual: ``(batch, channels, height, width)``, zero outside ``inside``.
        inside: ``(batch, 1, height, width)`` float, 1 on the unknown pixels.

    Returns:
        The correction, shaped like ``residual`` and zero outside ``inside``.
    """
    levels = _pyramid(inside)
    top = levels[0]
    x = torch.zeros_like(residual)
    left = residual.clone()
    floor = SOLVE_FLOOR * inside.sum((2, 3)).clamp_min(1.0).sqrt()
    scale = torch.maximum(residual.flatten(2).norm(dim=2), floor / SOLVE_TOLERANCE)
    z = _cycle(levels, 0, left)
    direction = z.clone()
    rz = (left * z).sum((2, 3), keepdim=True)
    for step in range(SOLVE_STEPS):
        pushed = top.apply(direction)
        alpha = rz / (direction * pushed).sum((2, 3), keepdim=True).clamp_min(1e-20)
        x = x + alpha * direction
        previous = left
        left = left - alpha * pushed
        if step % CHECK_EVERY == CHECK_EVERY - 1:
            if float((left.flatten(2).norm(dim=2) / scale).max()) < SOLVE_TOLERANCE:
                break
        z = _cycle(levels, 0, left)
        following = (z * (left - previous)).sum((2, 3), keepdim=True)
        direction = z + (following / rz.clamp_min(1e-30)) * direction
        rz = (left * z).sum((2, 3), keepdim=True)
    return x * inside


def _solve(values: torch.Tensor, unknown: torch.Tensor, start: torch.Tensor, guide=None) -> torch.Tensor:
    """Solve for the unknown pixels so their differences match a guidance field, edges held.

    Args:
        values: ``(batch, channels, height, width)``, read where ``unknown`` is False.
        unknown: ``(batch, height, width)`` bool.
        start: First guess for the unknown pixels, shaped like ``values``.
        guide: From :func:`_divergence`, shaped like ``values``, or None for the smoothest fill.

    Returns:
        Shaped like ``values``, equal to it outside ``unknown``.
    """
    inside = unknown[:, None].to(values.dtype)
    fixed = values * (1.0 - inside)
    first = start * inside + fixed
    counts = _neighbours(torch.ones_like(inside))
    residual = -(counts * first - _neighbours(first))
    if guide is not None:
        residual = residual + guide
    return first + _correction(residual * inside, inside)


def _complete_flows(flows: torch.Tensor, unknown: torch.Tensor) -> torch.Tensor:
    """Flows with their motion inside ``unknown`` replaced by the smoothest motion matching the rest.

    Args:
        flows: ``(batch, 2, h, w)``.
        unknown: ``(batch, h, w)`` bool.

    Returns:
        ``(batch, 2, h, w)``.
    """
    anywhere = unknown.any(dim=0)
    if not bool(anywhere.any()):
        return flows
    y0, y1, x0, x1 = _box(anywhere, 2)
    crop = flows[..., y0:y1, x0:x1]
    hole = unknown[:, y0:y1, x0:x1]
    start = _pushpull(crop, (~hole)[:, None].to(crop.dtype))
    completed = flows.clone()
    completed[..., y0:y1, x0:x1] = _solve(crop, hole, start)
    return completed


def _agreement(flow: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    """Where a flow and the flow back agree within :data:`CHAIN_AGREEMENT`, ``(1, 1, h, w)`` bool."""
    returned = optical_flow.warp(back, flow)
    gap = flow + returned
    share, floor = CHAIN_AGREEMENT
    allowed = share * ((flow * flow).sum(1, keepdim=True) + (returned * returned).sum(1, keepdim=True)) + floor
    return (gap * gap).sum(1, keepdim=True) < allowed


def _sample(plane: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of a ``(channels, height, width)`` plane at ``(n, 2)`` pixel positions, edges held.

    Returns:
        ``(n, channels)``.
    """
    channels, height, width = plane.shape
    grid = torch.empty_like(points)
    grid[:, 0] = points[:, 0] * (2.0 / max(width - 1, 1)) - 1.0
    grid[:, 1] = points[:, 1] * (2.0 / max(height - 1, 1)) - 1.0
    found = F.grid_sample(plane[None], grid.view(1, -1, 1, 2), mode="bilinear", padding_mode="border",
                          align_corners=True)
    return found[0, :, :, 0].t()


def _gather(picture: torch.Tensor, points: torch.Tensor, device) -> torch.Tensor:
    """Bilinear samples of a channels-last CPU picture at pixel positions, moving only the part they cover.

    Returns:
        ``(n, channels)`` float32 on ``device``.
    """
    height, width = int(picture.shape[0]), int(picture.shape[1])
    x0 = max(int(math.floor(float(points[:, 0].min()))) - 1, 0)
    x1 = min(int(math.ceil(float(points[:, 0].max()))) + 3, width)
    y0 = max(int(math.floor(float(points[:, 1].min()))) - 1, 0)
    y1 = min(int(math.ceil(float(points[:, 1].max()))) + 3, height)
    crop = optical_flow.unit(picture[y0:y1, x0:x1], device).permute(2, 0, 1)
    local = points - torch.tensor([x0, y0], dtype=points.dtype, device=points.device)
    return _taps(crop, local)


def _taps(plane: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Samples at positions and one pixel across and below them, from a ``(channels, height, width)`` plane.

    Returns:
        ``(n, 3, channels)``: here, across, below.
    """
    count = int(points.shape[0])
    across = torch.tensor(TAP_ACROSS, dtype=points.dtype, device=points.device)
    below = torch.tensor(TAP_BELOW, dtype=points.dtype, device=points.device)
    shifted = torch.cat([points, points + across, points + below])
    return _sample(plane, shifted).view(3, count, -1).permute(1, 0, 2)


@dataclass
class _Frame:
    """One frame's traced pixels inside a window: the object and a one-pixel ring around it.

    Attributes:
        index: The frame.
        box: ``(y0, y1, x0, x1)`` the pixels sit in.
        hole: ``(y1 - y0, x1 - x0)`` bool, the object.
        points: ``(n, 2)`` pixel positions of the traced pixels, row by row.
        offset: Where its candidates start in the window's buffers.
    """

    index: int
    box: tuple
    hole: torch.Tensor
    points: torch.Tensor
    offset: int


class _Clip:
    """The clip, its masks and its completed motion, as the trace reads them."""

    def __init__(self, frames, masks, motion, grow: int, reach: int, fills: dict, holes: list, device):
        self.frames = frames
        self.masks = masks
        self.motion = motion
        self.grow = grow
        self.holes = holes
        self.count = int(frames.shape[0])
        self.height, self.width = int(frames.shape[1]), int(frames.shape[2])
        self.channels = int(frames.shape[3])
        self.reach = reach if reach > 0 else self.count
        self.fills = fills
        self.device = device
        self.cuts = motion.cuts()
        h, w = motion.size
        self.to_measured = torch.tensor([w / self.width, h / self.height], device=device)
        self.to_frame = torch.tensor([self.width / w, self.height / h], device=device)

    def hole(self, index: int, extra: int = 0) -> torch.Tensor:
        """Frame ``index``'s object grown by ``grow`` and ``extra`` pixels, ``(height, width)`` bool."""
        return _object(self.masks, index, self.grow + extra, self.device)

    def pair(self, index: int, ahead: bool):
        """Motion of pair ``index`` read from frame ``index`` (ahead) or ``index + 1`` (behind).

        Returns:
            ``(flow, agree)``: ``(2, h, w)`` and ``(1, h, w)`` float, 1 where both ways agree.
        """
        forward = self.motion.forward[index].to(self.device, torch.float32)[None]
        backward = self.motion.backward[index].to(self.device, torch.float32)[None]
        flow, back = (forward, backward) if ahead else (backward, forward)
        return flow[0], _agreement(flow, back).float()[0]

    def step(self, points: torch.Tensor, flow: torch.Tensor, agree: torch.Tensor):
        """Move positions along a flow.

        Returns:
            ``(moved, kept)``: new positions, and which stay inside the frame on a motion both ways agree on.
        """
        measured = (points + 0.5) * self.to_measured - 0.5
        moved = points + _sample(flow, measured) * self.to_frame
        agreed = _sample(agree, measured)[:, 0] > 0.5
        inside = (
            (moved[:, 0] >= 0) & (moved[:, 0] <= self.width - 1)
            & (moved[:, 1] >= 0) & (moved[:, 1] <= self.height - 1)
        )
        return moved, agreed & inside


class _Candidates:
    """Background found for every traced pixel of a window, one set per direction, as :func:`_taps`."""

    def __init__(self, total: int, channels: int, device):
        self.colour = torch.zeros(2, total, 3, channels, device=device)
        self.source = torch.full((2, total), -1, dtype=torch.int32, device=device)
        self.fill_colour = torch.zeros(2, total, 3, channels, device=device)
        self.fill_source = torch.full((2, total), -1, dtype=torch.int32, device=device)
        self.fill_distance = torch.zeros(2, total, device=device)


def _trace(clip: _Clip, frames: list, found: _Candidates, ahead: bool, progress=None) -> None:
    """Follow every traced pixel of a window along the motion until a frame shows it.

    Args:
        clip: The :class:`_Clip`.
        frames: The window's :class:`_Frame` entries, in order.
        found: Where the candidates are written.
        ahead: True follows later frames, False earlier ones.
        progress: Optional callable taking a step count.
    """
    side = 1 if ahead else 0
    first, last = frames[0].index, frames[-1].index
    by_index = {entry.index: entry for entry in frames}
    fill_indices = sorted(clip.fills)
    order = range(first, clip.count) if ahead else range(last, -1, -1)
    device = clip.device
    points = torch.zeros(0, 2, device=device)
    slots = torch.zeros(0, dtype=torch.long, device=device)
    origins = torch.zeros(0, dtype=torch.long, device=device)
    for j in order:
        if progress is not None:
            progress(0)
        if points.shape[0]:
            distance = (origins - j).abs()
            within = distance <= clip.reach
            if clip.holes[j]:
                blocked = clip.hole(j, SOURCE_MARGIN)
                column = points[:, 0].round().long().clamp(0, clip.width - 1)
                row = points[:, 1].round().long().clamp(0, clip.height - 1)
                shown = ~blocked[row, column]
            else:
                shown = torch.ones_like(within)
            take = shown & within
            if bool(take.any()):
                chosen = slots[take]
                found.colour[side, chosen] = _gather(clip.frames[j], points[take], device)
                found.source[side, chosen] = j
            if j in clip.fills:
                open_fill = ~take & (found.fill_source[side, slots] < 0)
                if bool(open_fill.any()):
                    chosen = slots[open_fill]
                    found.fill_colour[side, chosen] = _taps(clip.fills[j], points[open_fill])
                    found.fill_source[side, chosen] = j
                    found.fill_distance[side, chosen] = distance[open_fill].float()
            keep = ~take
            points, slots, origins = points[keep], slots[keep], origins[keep]
        entry = by_index.get(j)
        if entry is not None and entry.points.shape[0]:
            count = int(entry.points.shape[0])
            new_slots = torch.arange(entry.offset, entry.offset + count, device=device)
            if j in clip.fills:
                found.fill_colour[side, new_slots] = _taps(clip.fills[j], entry.points)
                found.fill_source[side, new_slots] = j
                found.fill_distance[side, new_slots] = 0.0
            points = torch.cat([points, entry.points])
            slots = torch.cat([slots, new_slots])
            origins = torch.cat([origins, torch.full((count,), j, dtype=torch.long, device=device)])
        beyond = j >= last if ahead else j <= first
        if not points.shape[0] and beyond:
            break
        following = j + 1 if ahead else j - 1
        if following < 0 or following >= clip.count:
            break
        pair = j if ahead else j - 1
        if clip.cuts[pair] or not points.shape[0]:
            points = points[:0]
            slots, origins = slots[:0], origins[:0]
            continue
        flow, agree = clip.pair(pair, ahead)
        points, kept = clip.step(points, flow, agree)
        later = [index for index in fill_indices if (index > j if ahead else index < j)]
        if later:
            kept &= ((origins - following).abs() <= clip.reach) | (found.fill_source[side, slots] < 0)
        else:
            kept &= (origins - following).abs() <= clip.reach
        points, slots, origins = points[kept], slots[kept], origins[kept]


def _choose(found: _Candidates, entry: _Frame, count: int):
    """Each traced pixel's candidates from both sides: colour, source and weight.

    Returns:
        ``(colour, source, weight)``: ``(2, n, 3, channels)`` as :func:`_taps`, ``(2, n)`` int and
        ``(2, n)`` float, weight 0 where a side has nothing. Real background is preferred over a fill.
    """
    span = slice(entry.offset, entry.offset + int(entry.points.shape[0]))
    colour = found.colour[:, span]
    source = found.source[:, span].long()
    real = (source >= 0).any(dim=0)
    fill_source = found.fill_source[:, span].long()
    colour = torch.where(real[None, :, None, None], colour, found.fill_colour[:, span])
    distance = torch.where(real[None], (source - entry.index).abs().float(), found.fill_distance[:, span])
    source = torch.where(real[None], source, torch.where(fill_source >= 0, fill_source + count, fill_source))
    weight = torch.where(source >= 0, 1.0 / distance.clamp_min(1e-3), torch.zeros_like(distance))
    both = (weight > 0).all(dim=0)
    apart = (colour[0, :, 0] - colour[1, :, 0]).abs().mean(dim=-1) > DISAGREE
    split = both & apart
    nearer = distance[0] <= distance[1]
    weight[0] = torch.where(split & ~nearer, torch.zeros_like(weight[0]), weight[0])
    weight[1] = torch.where(split & nearer, torch.zeros_like(weight[1]), weight[1])
    source = torch.where(weight > 0, source, torch.full_like(source, -1))
    return colour, source, weight


def _guidance(sides: torch.Tensor, beside: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
              across: bool) -> torch.Tensor:
    """Differences to the next pixel across or down, as the frames the candidates came from show them.

    Args:
        sides: ``(batch, 2, channels, height, width)`` candidates per direction.
        beside: The same candidates' sources read one pixel across, or below.
        ids: ``(batch, 2, height, width)`` their sources, -1 for none.
        weights: ``(batch, 2, height, width)``.
        across: True for the next pixel across, False for the one below.

    Returns:
        ``(batch, channels, height, width)``, 0 on the last column or row.
    """
    if across:
        here, there = (..., slice(None), slice(None, -1)), (..., slice(None), slice(1, None))
    else:
        here, there = (..., slice(None, -1), slice(None)), (..., slice(1, None), slice(None))
    found = ids[here] >= 0
    same = found & (ids[here] == ids[there])
    strength = torch.where(same, (weights[here] + weights[there]) * 0.5, weights[here]) * found
    difference = torch.where(same[:, :, None], sides[there] - sides[here], beside[here] - sides[here])
    summed = (strength[:, :, None] * difference).sum(dim=1) / strength.sum(dim=1).clamp_min(1e-8)[:, None]
    padded = torch.zeros(sides.shape[0], *sides.shape[2:], device=sides.device)
    padded[here] = summed
    return padded


def _finish(clip: _Clip, found: _Candidates, entries: list, output, remaining) -> list:
    """Blend the candidates of a group of frames, solve their seams away and write them.

    Returns:
        Pixels of each frame left without a source.
    """
    device = clip.device
    y0, y1, x0, x1 = _union(entry.box for entry in entries)
    height, width = y1 - y0, x1 - x0
    batch, channels = len(entries), clip.channels
    hole = torch.zeros(batch, height, width, dtype=torch.bool, device=device)
    has = torch.zeros(batch, height * width, dtype=torch.bool, device=device)
    mean = torch.zeros(batch, channels, height * width, device=device)
    taps = torch.zeros(3, batch, 2, channels, height * width, device=device)
    ids = torch.full((batch, 2, height * width), -1, dtype=torch.long, device=device)
    weights = torch.zeros(batch, 2, height * width, device=device)
    picture = torch.empty(batch, channels, height, width, device=device)
    for b, entry in enumerate(entries):
        ey0, ey1, ex0, ex1 = entry.box
        hole[b, ey0 - y0:ey1 - y0, ex0 - x0:ex1 - x0] = entry.hole
        flat = (entry.points[:, 1].long() - y0) * width + (entry.points[:, 0].long() - x0)
        colour, source, weight = _choose(found, entry, clip.count)
        total = weight.sum(dim=0)
        has[b, flat] = total > 0
        blended = (weight[..., None] * colour[:, :, 0]).sum(dim=0) / total.clamp_min(1e-8)[:, None]
        mean[b, :, flat] = blended.t()
        taps[:, b, :, :, flat] = colour.permute(2, 0, 3, 1)
        ids[b, :, flat] = source
        weights[b, :, flat] = weight
        picture[b] = optical_flow.unit(clip.frames[entry.index, y0:y1, x0:x1], device).permute(2, 0, 1)
    has = has.view(batch, 1, height, width)
    mean = mean.view(batch, channels, height, width)
    taps = taps.view(3, batch, 2, channels, height, width)
    ids = ids.view(batch, 2, height, width)
    weights = weights.view(batch, 2, height, width)
    guide = _divergence(
        _guidance(taps[0], taps[1], ids, weights, True), _guidance(taps[0], taps[2], ids, weights, False)
    )
    del taps, ids, weights
    known = has | ~hole[:, None]
    start = _pushpull(torch.where(has, mean, picture), known.to(picture.dtype))
    start = torch.where(has, mean, start)
    solved = _solve(picture, hole, start, guide)
    left = hole & ~has[:, 0]
    counts = []
    for b, entry in enumerate(entries):
        window = output[entry.index, y0:y1, x0:x1]
        result = solved[b].permute(1, 2, 0).to(window.dtype).cpu()
        output[entry.index, y0:y1, x0:x1] = torch.where(hole[b].cpu()[..., None], result, window)
        remaining[entry.index, y0:y1, x0:x1] = torch.maximum(remaining[entry.index, y0:y1, x0:x1],
                                                             left[b].float().cpu())
        counts.append(int(left[b].sum()))
    return counts


def _groups(entries: list) -> list:
    """Runs of frames whose union crop, times their count, fits :data:`SOLVE_PIXELS`."""
    runs, current = [], []
    for entry in entries:
        trial = current + [entry]
        y0, y1, x0, x1 = _union(item.box for item in trial)
        if current and (y1 - y0) * (x1 - x0) * len(trial) > SOLVE_PIXELS:
            runs.append(current)
            trial = [entry]
        current = trial
    if current:
        runs.append(current)
    return runs


def _window_frames(clip: _Clip, indices: list) -> tuple[list, int]:
    """The traced pixels of a run of frames: each object and a one-pixel ring around it."""
    entries, offset = [], 0
    for index in indices:
        if clip.holes[index] <= 0:
            continue
        hole = clip.hole(index)
        region = _dilate(hole, 1)
        y0, y1, x0, x1 = _box(region, 1)
        flat = torch.nonzero(region[y0:y1, x0:x1].flatten()).flatten()
        width = x1 - x0
        points = torch.stack([(flat % width + x0).float(), (flat // width + y0).float()], dim=1)
        entries.append(_Frame(index, (y0, y1, x0, x1), hole[y0:y1, x0:x1], points, offset))
        offset += int(points.shape[0])
    return entries, offset


def _windows(counts: list, budget: int) -> list:
    """Runs of frames whose traced pixels fit ``budget``."""
    runs, current, held = [], [], 0
    for index, count in enumerate(counts):
        size = int(count * 1.3) + 1
        if current and held + size > budget:
            runs.append(current)
            current, held = [], 0
        current.append(index)
        held += size
    if current:
        runs.append(current)
    return runs


def _budget(device) -> int:
    """Traced pixels one window may hold on ``device``."""
    try:
        import comfy.model_management as management

        free = int(management.get_free_memory(device))
    except Exception:
        free = 0
    if free <= 0:
        return MOST_TRACED
    return max(1, min(MOST_TRACED, int(free * SPARE) // TRACE_BYTES))


def _complete_motion(motion, distrust: torch.Tensor, holes: list, device, progress=None) -> None:
    """Replace the motion inside every grown object with motion completed from around it, in place."""
    pairs = [
        index for index in range(motion.count - 1)
        if (holes[index] or holes[index + 1]) and float(motion.agreement[index]) >= motion_field.CUT_SHARE
    ]
    if progress is not None:
        progress(motion.count - 1 - len(pairs))
    for start in range(0, len(pairs), FLOW_BATCH):
        chunk = pairs[start:start + FLOW_BATCH]
        flows = torch.cat([
            motion.forward[chunk].to(device, torch.float32),
            motion.backward[chunk].to(device, torch.float32),
        ])
        unknown = torch.cat([distrust[chunk], distrust[[index + 1 for index in chunk]]]).to(device)
        completed = _complete_flows(flows, unknown).half().cpu()
        motion.forward[chunk] = completed[:len(chunk)]
        motion.backward[chunk] = completed[len(chunk):]
        if progress is not None:
            progress(len(chunk))


def remove(frames, masks, grow: int, reach: int = 0, network=None, fills=None, device=None,
           progress=None, node: str = "") -> Removal:
    """Replace the masked object in every frame with background other frames show.

    Args:
        frames: ``(frames, height, width, channels)``, two or more.
        masks: From :func:`check_masks`.
        grow: Pixels the object is grown by.
        reach: Frames searched each way; 0 searches the whole clip.
        network: A flow network to measure the motion with, or None for the texture flow.
        fills: ``{frame index: (height, width, channels) picture}`` carried into what no frame shows.
        device: Where the work runs.
        progress: Optional callable taking a step count.
        node: The node's name, for messages.

    Returns:
        The :class:`Removal`.

    Raises:
        ValueError: The masks cover nothing, or cover nearly all of a frame.
        MemoryError: The result fits neither in memory nor on a scratch drive.
    """
    from . import scratch

    count, height, width, channels = (int(side) for side in frames.shape)
    device = torch.device(device) if device is not None else torch.device("cpu")
    size = motion_field.working_size(height, width, motion_field.MOTION_SIDE)
    holes, distrust = [], torch.zeros(count, *size, dtype=torch.bool)
    for index in range(count):
        hole = _object(masks, index, grow, device)
        covered = int(hole.sum())
        holes.append(covered)
        if covered > MOST_COVERED * height * width:
            raise ValueError(
                f"{node}: the mask covers {100.0 * covered / (height * width):.0f}% of frame {index}, "
                f"leaving nothing to fill from. Mask the object alone on that frame, or lower grow."
            )
        if covered:
            small = F.interpolate(hole.float()[None, None], size=size, mode="area")[0, 0] > 0
            distrust[index] = _dilate(small, 1 + TRUST_MARGIN).cpu()
    if not any(holes):
        raise ValueError(
            f"{node}: the mask is empty on every frame, so there is nothing to remove. Connect the "
            f"object's matte, such as Image Track Subject's, white on the object."
        )
    motion = motion_field.measure(frames, motion_field.MOTION_SIDE, device, progress, network=network)
    _complete_motion(motion, distrust, holes, device, progress)
    del distrust
    pictures = {}
    for index, picture in (fills or {}).items():
        plane = optical_flow.unit(picture, device)[..., :channels].permute(2, 0, 1)
        if plane.shape[0] < channels:
            plane = torch.cat([plane, torch.ones(channels - plane.shape[0], *plane.shape[1:], device=device)])
        if tuple(plane.shape[1:]) != (height, width):
            plane = F.interpolate(plane[None], size=(height, width), mode="bilinear", align_corners=False)[0]
        pictures[int(index)] = plane
    clip = _Clip(frames, masks, motion, grow, reach, pictures, holes, device)
    advice = "A shorter clip or a smaller frame also fits it."
    output = scratch.allocate((count, height, width, channels), torch.float32, node=node, advice=advice)
    remaining = scratch.allocate((count, height, width), torch.float32, node=node, advice=advice)
    for index in range(count):
        output[index] = frames[index].to(torch.float32)
        remaining[index].zero_()
    left = [0] * count
    for run in _windows(holes, _budget(device)):
        entries, total = _window_frames(clip, run)
        if not entries:
            continue
        found = _Candidates(total, channels, device)
        _trace(clip, entries, found, ahead=False, progress=progress)
        _trace(clip, entries, found, ahead=True, progress=progress)
        for group in _groups(entries):
            for entry, value in zip(group, _finish(clip, found, group, output, remaining)):
                left[entry.index] = value
            if progress is not None:
                progress(len(group))
        del found, entries
    if progress is not None:
        progress(sum(1 for value in holes if not value))
    return Removal(images=output, remaining=remaining, holes=holes, left=left, engine=motion.engine)
