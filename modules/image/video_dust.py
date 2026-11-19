"""Dust and dirt that sits on one frame of a clip, found and repaired from its neighbours.

Frames are picture codes, a fourth channel carried through. Sizes are pixels; levels are on the
0 to 255 scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import dynamic, optical_flow, scratch, temporal_restore
from .temporal_restore import opponent

__all__ = ["POLARITIES", "Report", "remove_dust"]

#: Which specks are looked for, in the order the menu lists them.
POLARITIES = ("light and dark", "light", "dark")

#: Neighbour pairs tried for each frame, nearest first: one each side, then two on one side.
PAIRS = ((-1, 1), (1, 2), (-1, -2))

#: How far a speck's core must stand out, in noise deviations, at sensitivity 0 and at 1.
SEED_SIGMAS = (10.0, 4.0)

#: How far the rest of a speck must stand out, in noise deviations, at sensitivity 0 and at 1.
GROW_SIGMAS = (5.0, 2.0)

#: Levels a speck's core must stand out by at least, at sensitivity 0 and at 1.
SEED_LEVELS = (30.0, 8.0)

#: Levels the rest of a speck must stand out by at least, at sensitivity 0 and at 1.
GROW_LEVELS = (12.0, 4.0)

#: Share of the two neighbours' own disagreement taken off how far a pixel stands out.
AGREEMENT = 1.0

#: Width, in pixels, of the band outside a speck's repair both neighbours must match the frame on.
RING = 3

#: Mismatch a band pixel may hold, in noise deviations and in levels, and the share of the band
#: allowed past it.
RING_SIGMAS = 3.0
RING_LEVELS = 4.0
RING_OFF = 0.25

#: Share of the band whose motion both neighbours must agree on.
RING_SEEN = 0.5

#: Share of the long side a speck is looked for around where the motion puts it in each neighbour,
#: and the bounds of that reach in pixels.
SEARCH_SHARE = 0.03
SEARCH_BOUNDS = (6, 64)

#: Mismatch, in noise deviations and in levels, within which a speck counts as found in a neighbour.
FOUND_SIGMAS = 1.0
FOUND_LEVELS = 3.0

#: Share of a speck's mismatch where the motion points that the best match elsewhere must stay under.
FOUND_SHARE = 0.1

#: Candidates searched for together.
SEARCH_BATCH = 256

#: Share of the long side around a speck its motion is read over, and the share by which the
#: motion there may vary before the speck counts as on a moving edge and is left alone.
SPREAD_REACH = 1.0 / 96.0
SPREAD_LIMIT = 0.0026

#: Share of the long side a speck may move per frame before it counts as fast motion and is left alone.
SPEED_LIMIT = 0.015

#: Long side, in pixels, per pixel of margin the repair adds around a speck.
MARGIN_PER = 1500


class Report:
    """What a dust run found.

    Attributes:
        specks: Specks repaired on each frame.
        refused: Candidates turned down on each frame: too large, on motion the neighbours do
            not follow, or found again in a neighbour.
        area: Share of each frame repaired.
    """

    def __init__(self):
        self.specks: list[int] = []
        self.refused: list[int] = []
        self.area: list[float] = []


def _between(pair, share: float) -> float:
    """The value ``share`` of the way from ``pair[0]`` to ``pair[1]``."""
    return pair[0] + (pair[1] - pair[0]) * share


def _labels(mask, reach: int):
    """Label 8-connected regions of a ``(height, width)`` bool mask.

    Args:
        mask: The regions.
        reach: Steps the labels spread; a region still changing after them is answered as 0.

    Returns:
        ``(height, width)`` float64 labels, 0 outside every settled region.
    """
    height, width = mask.shape
    dtype = torch.float32 if height * width < (1 << 24) else torch.float64
    where = mask.to(dtype)
    labels = torch.arange(1, height * width + 1, device=mask.device, dtype=dtype).view(height, width) * where
    labels = labels.view(1, 1, height, width)
    where = where.view(1, 1, height, width)
    for _ in range(int(reach)):
        grown = F.max_pool2d(labels, 3, stride=1, padding=1) * where
        if torch.equal(grown, labels):
            break
        labels = grown
    grown = F.max_pool2d(labels, 3, stride=1, padding=1) * where
    moving = grown != labels
    if bool(moving.any()):
        unsettled = torch.unique(torch.cat([labels[moving], grown[moving]]))
        labels = labels * ~torch.isin(labels, unsettled)
    return labels[0, 0].double()


def _gather(index, values, count: int, reduce: str):
    """``values`` reduced per region into a ``(count,)`` float64 tensor."""
    out = torch.zeros(count, device=values.device, dtype=torch.float64)
    return out.scatter_reduce(0, index, values.double(), reduce=reduce, include_self=False)


def _spread(flow, reach: int):
    """How far the motion varies within ``reach`` pixels of each pixel, ``(height, width)``.

    Args:
        flow: ``(2, height, width)`` in pixels.
        reach: Half the window, in pixels.

    Returns:
        The larger of the x and y ranges, in pixels.
    """
    planes = F.avg_pool2d(flow[None], 5, stride=1, padding=2, count_include_pad=False)
    size = 2 * reach + 1
    ranges = F.max_pool2d(planes, size, stride=1, padding=reach) + F.max_pool2d(-planes, size, stride=1, padding=reach)
    return ranges[0].amax(0)


def _windows(plane, rows, columns):
    """Crops of a ``(height, width)`` plane at ``(count, size)`` row and column indices, edges held."""
    height, width = plane.shape
    rows = rows.clamp(0, height - 1)
    columns = columns.clamp(0, width - 1)
    return plane[rows[:, :, None], columns[:, None, :]]


def _found(here, labels, names, rows, columns, half, neighbours, deviation, reach: int):
    """Whether each candidate, with the frame just around it, sits within ``reach`` of where the
    motion puts it in either neighbour.

    Args:
        here: The frame's brightness, ``(height, width)``.
        labels: Region labels from :func:`_labels`.
        names: ``(count,)`` labels of the candidates.
        rows: ``(count,)`` centre rows.
        columns: ``(count,)`` centre columns.
        half: ``(count,)`` half sides of the boxes compared.
        neighbours: ``(brightness, flow)`` pairs, the neighbour's own brightness and the
            full-size flow from the frame onto it.
        deviation: ``(count,)`` noise deviation at each candidate.
        reach: Pixels searched around where the motion points.

    Returns:
        ``(count,)`` bool.
    """
    widest = int(half.max())
    steps = torch.arange(-widest, widest + 1, device=here.device)
    patch_rows = rows[:, None] + steps[None]
    patch_columns = columns[:, None] + steps[None]
    patch = _windows(here, patch_rows, patch_columns)
    box = ((steps.abs()[None, :, None] <= half[:, None, None]) & (steps.abs()[None, None, :] <= half[:, None, None]))
    core = (_windows(labels, patch_rows, patch_columns) == names[:, None, None]) & box
    allowed = (FOUND_SIGMAS * math.sqrt(2.0) * deviation + FOUND_LEVELS / 255.0).square()
    found = torch.zeros(rows.shape[0], dtype=torch.bool, device=here.device)
    span = torch.arange(-(widest + reach), widest + reach + 1, device=here.device)
    for plane, flow in neighbours:
        target_rows = rows + flow[1, rows, columns].round().long()
        target_columns = columns + flow[0, rows, columns].round().long()
        window = _windows(plane, target_rows[:, None] + span[None], target_columns[:, None] + span[None])
        worst = None
        for weight in (box.float(), core.float()):
            count = weight.sum((1, 2)).clamp(min=1.0)
            filters = weight * patch
            cross = F.conv2d(window[None], filters[:, None], groups=int(rows.shape[0]))[0]
            energy = F.conv2d(window.square()[None], weight[:, None], groups=int(rows.shape[0]))[0]
            own = (filters * patch).sum((1, 2))
            mismatch = ((own[:, None, None] - 2.0 * cross + energy) / count[:, None, None]).clamp(min=0.0)
            worst = mismatch / allowed[:, None, None] if worst is None else torch.maximum(worst, mismatch / allowed[:, None, None])
        best = worst.flatten(1).argmin(1)
        lowest = mismatch.flatten(1).gather(1, best[:, None])[:, 0]
        found |= (worst.flatten(1).amin(1) <= 1.0) & (lowest <= FOUND_SHARE * mismatch[:, reach, reach])
    return found


def _specks(colour, sides, level, thresholds, polarity: str, largest: int, margin: int, reach: int,
            raws, motion_limits):
    """Find the specks on one frame and repair them.

    Args:
        colour: The frame, ``(1, 3, height, width)``.
        sides: Two ``(aligned, difference, seen, spread, flow, step)`` tuples, one per neighbour.
        level: Brightness noise deviation at each pixel, ``(height, width)``.
        thresholds: ``(seed deviations, grow deviations, seed levels, grow levels)``.
        polarity: One of :data:`POLARITIES`.
        largest: Widest speck repaired, in pixels.
        margin: Pixels the repair reaches past a speck.
        reach: Pixels a speck is looked for around where the motion points in a neighbour.
        raws: The two neighbours' own brightness, ``(height, width)`` each.
        motion_limits: ``(reach, spread, speed)``: pixels around a speck its motion is read over,
            pixels the motion may vary by there, and pixels per frame it may move.

    Returns:
        ``(result, mask, specks, refused)``.
    """
    height, width = colour.shape[-2:]
    device = colour.device
    mask = torch.zeros(height, width, device=device)
    (first, first_d, first_seen, first_spread, first_flow, _), (second, second_d, second_seen, second_spread, second_flow, _) = sides
    ahead = -first_d[0, 0] / math.sqrt(3.0)
    behind = -second_d[0, 0] / math.sqrt(3.0)
    light = torch.minimum(ahead, behind)
    dark = torch.minimum(-ahead, -behind)
    spike = light if polarity == "light" else dark if polarity == "dark" else torch.maximum(light, dark)
    disagree = (first_d[0, 0] - second_d[0, 0]).abs() / math.sqrt(3.0)
    score = spike - AGREEMENT * disagree
    deviation = level * torch.sqrt(1.0 + torch.maximum(first_spread, second_spread)[0, 0])
    seed_sigmas, grow_sigmas, seed_levels, grow_levels = thresholds
    seeds = (score > seed_sigmas * deviation) & (score > seed_levels)
    grown = (score > grow_sigmas * deviation) & (score > grow_levels)
    if not bool(seeds.any()):
        return colour, mask, 0, 0
    labels = _labels(grown, 2 * largest + 2)
    inside = labels > 0
    names, slot = torch.unique(labels[inside], return_inverse=True)
    total = int(names.numel())
    if not total:
        return colour, mask, 0, 0
    ys, xs = torch.nonzero(inside, as_tuple=True)
    top, bottom = _gather(slot, ys, total, "amin"), _gather(slot, ys, total, "amax")
    left, right = _gather(slot, xs, total, "amin"), _gather(slot, xs, total, "amax")
    extent = torch.maximum(right - left, bottom - top) + 1
    seeded = _gather(slot, seeds[inside].double(), total, "amax") > 0

    plane = labels.view(1, 1, height, width)
    near = F.max_pool2d(plane, 2 * margin + 3, stride=1, padding=margin + 1)[0, 0]
    around = F.max_pool2d(plane, 2 * (margin + 1 + RING) + 1, stride=1, padding=margin + 1 + RING)[0, 0]
    band = (around > 0) & (near == 0) & ~grown
    band_slot = torch.searchsorted(names, around[band]).clamp(max=total - 1)
    band_found = names[band_slot] == around[band]
    band_slot = band_slot[band_found]

    def band_mean(values):
        sums = torch.zeros(total, device=device, dtype=torch.float64)
        sums.index_add_(0, band_slot, values[band][band_found].double())
        return sums

    cells = band_mean(torch.ones_like(ahead))
    filled = cells.clamp(min=1.0)
    allowed = RING_SIGMAS * deviation + RING_LEVELS / 255.0
    first_miss = band_mean(ahead.abs()) / filled
    second_miss = band_mean(behind.abs()) / filled
    keep = (
        seeded & (extent <= largest) & (cells > 0)
        & (band_mean(first_seen[0, 0]) / filled >= RING_SEEN)
        & (band_mean(second_seen[0, 0]) / filled >= RING_SEEN)
        & (band_mean((ahead.abs() > allowed).float()) / filled <= RING_OFF)
        & (band_mean((behind.abs() > allowed).float()) / filled <= RING_OFF)
    )
    chosen = keep.nonzero()[:, 0]
    if int(chosen.numel()):
        rows = ((top + bottom) / 2).round().long()[chosen]
        columns = ((left + right) / 2).round().long()[chosen]
        spread_reach, spread_limit, speed_limit = motion_limits
        spread = torch.maximum(_spread(first_flow, spread_reach), _spread(second_flow, spread_reach))
        steady = spread[rows, columns] <= spread_limit
        steps = (abs(sides[0][5]), abs(sides[1][5]))
        speed = torch.maximum(
            first_flow[:, rows, columns].norm(dim=0) / steps[0], second_flow[:, rows, columns].norm(dim=0) / steps[1]
        )
        steady &= speed <= speed_limit
        keep[chosen[~steady]] = False
        chosen, rows, columns = chosen[steady], rows[steady], columns[steady]
    if int(chosen.numel()):
        half = (extent[chosen] / 2).ceil().long() + 1
        order = torch.argsort(half)
        chosen, rows, columns, half = chosen[order], rows[order], columns[order], half[order]
        here = colour.mean(1)[0]
        neighbours = ((raws[0], first_flow), (raws[1], second_flow))
        for start in range(0, int(chosen.numel()), SEARCH_BATCH):
            part = slice(start, start + SEARCH_BATCH)
            found = _found(
                here, labels, names[chosen[part]], rows[part], columns[part], half[part], neighbours,
                deviation[rows[part], columns[part]], reach,
            )
            keep[chosen[part][found]] = False
    refused = int((seeded & ~keep).sum())
    specks = int(keep.sum())
    if not specks:
        return colour, mask, 0, refused
    source = torch.where(second_miss < first_miss, 2.0, 1.0)
    kept = torch.zeros(height, width, device=device, dtype=torch.float64)
    kept[inside] = torch.where(keep[slot], names[slot], torch.zeros_like(names[slot]))
    wide = F.max_pool2d(kept.view(1, 1, height, width), 2 * margin + 1, stride=1, padding=margin)[0, 0]
    cover = wide > 0
    picked = torch.zeros(height, width, device=device)
    picked[cover] = source[torch.searchsorted(names, wide[cover]).clamp(max=total - 1)].float()
    solid = cover.float().view(1, 1, height, width)
    soft = F.avg_pool2d(solid, 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    mask = torch.maximum((kept > 0).float(), soft * solid[0, 0])
    patch = torch.where((picked == 2.0).view(1, 1, height, width), second, first)
    return colour + mask.view(1, 1, height, width) * (patch - colour), mask, specks, refused


def remove_dust(frames, flows, sensitivity: float = 0.5, max_size: int = 24,
                polarity: str = "light and dark", device=None, progress=None, node: str = ""):
    """Find specks that sit on one frame only and repair them from its neighbours' own pixels.

    Args:
        frames: ``(frames, height, width, channels)`` picture codes, float or uint8.
        flows: A :class:`~.temporal_restore.Flows` over ``frames``.
        sensitivity: 0 = only bold specks, 1 = faint ones too.
        max_size: Widest speck repaired, in pixels.
        polarity: One of :data:`POLARITIES`.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame.
        node: Display name of the calling node, for the refusal.

    Returns:
        ``(frames, mask, report)``: the repaired frames shaped like ``frames`` in float32, a
        ``(frames, height, width)`` float32 mask of what was repaired, and a :class:`Report`.

    Raises:
        ValueError: ``polarity`` is not one of :data:`POLARITIES`.
        MemoryError: Neither free memory nor a scratch drive can hold the result.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"unknown polarity '{polarity}'; pick one of {', '.join(POLARITIES)}")
    count, height, width, channels = (int(v) for v in frames.shape)
    device = torch.device(device) if device is not None else frames.device
    advice = "A shorter clip, or the clip at a smaller size, also fits it."
    out = scratch.allocate((count, height, width, channels), torch.float32, node=node, advice=advice)
    repaired = scratch.allocate((count, height, width), torch.float32, node=node, advice=advice)
    share = max(0.0, min(1.0, float(sensitivity)))
    thresholds = (
        _between(SEED_SIGMAS, share), _between(GROW_SIGMAS, share),
        _between(SEED_LEVELS, share) / 255.0, _between(GROW_LEVELS, share) / 255.0,
    )
    largest = max(1, int(max_size))
    longest = max(height, width)
    margin = max(1, round(longest / MARGIN_PER))
    reach = int(max(SEARCH_BOUNDS[0], min(SEARCH_BOUNDS[1], round(SEARCH_SHARE * longest))))
    motion_limits = (
        max(3, round(SPREAD_REACH * longest)), max(1.0, SPREAD_LIMIT * longest), max(1.0, SPEED_LIMIT * longest)
    )
    hold = not dynamic.carries(frames) if frames.dtype.is_floating_point else True
    report = Report()
    held: dict = {}
    last_table = None

    def picture(index: int):
        found = held.get(index)
        if found is None:
            found = frames[index:index + 1].to(device=device)
            found = found.float() / 255.0 if found.dtype == torch.uint8 else found.float()
            found = found.permute(0, 3, 1, 2).contiguous()
            held[index] = found
        return found

    for index in range(count):
        centre = picture(index)
        result = centre[:, :3]
        mask = torch.zeros(height, width, device=device)
        specks = refused = 0
        low, high = max(0, index - 2), min(count - 1, index + 2)
        flows.measure([(i, i + 1) for i in range(low, high)] + [(i + 1, i) for i in range(low, high)])
        offsets = set(flows.reachable(index, 2))
        pair = next((p for p in PAIRS if p[0] in offsets and p[1] in offsets), None)
        if pair is not None:
            flows.measure([(index, index + k) for k in pair] + [(index + k, index) for k in pair])
            colour = centre[:, :3]
            centre_o = opponent(colour)
            sides, raws = [], []
            for k in pair:
                neighbour = picture(index + k)[:, :3]
                ahead = flows.flow(index, index + k)
                warped, seen, spread = temporal_restore.align(
                    neighbour, ahead, flows.flow(index + k, index), (height, width)
                )
                full = optical_flow.resize_flow(ahead, height, width)[0]
                sides.append((warped, opponent(warped) - centre_o, seen, spread, full, k))
                raws.append(neighbour.mean(1)[0])
            table = temporal_restore.noise_table(centre_o, [side[1:4] for side in sides])
            if table is None:
                table = last_table if last_table is not None else temporal_restore.spatial_noise(centre_o)
            last_table = table
            level = temporal_restore.noise_map(table, centre_o)[0, 0] / math.sqrt(3.0)
            result, mask, specks, refused = _specks(
                colour, sides, level, thresholds, polarity, largest, margin, reach, raws, motion_limits
            )
        if hold:
            result = result.clamp(0.0, 1.0)
        out[index, ..., :3].copy_(result[0].permute(1, 2, 0))
        if channels > 3:
            out[index, ..., 3:].copy_(centre[0, 3:].permute(1, 2, 0))
        repaired[index].copy_(mask)
        scratch.trim(out[index])
        scratch.trim(repaired[index])
        report.specks.append(specks)
        report.refused.append(refused)
        report.area.append(float(mask.mean()))
        for stale in [i for i in held if i <= index - 2]:
            del held[stale]
        flows.forget(index - 1)
        if progress is not None:
            progress(1)
    return out, repaired, report
