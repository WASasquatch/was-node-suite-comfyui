"""Motion-compensated temporal noise reduction for a clip.

Frames are ``(frames, height, width, channels)`` picture codes; a fourth channel is carried
through. Noise levels are standard deviations in levels of 255.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import dynamic, scratch, temporal_restore
from .temporal_restore import from_opponent, opponent

__all__ = ["NOISE_GAIN", "Report", "denoise", "strictness"]

#: Half the window a neighbour's match is averaged over, in pixels.
MATCH_RADIUS = 2

#: How far a neighbour's match may exceed the noise before its weight falls by ``e``, at
#: ``detail`` 0 and at ``detail`` 1, in multiples of the expected match.
STRICTNESS = (2.0, 0.2)

#: How much further a chroma match may stray than a brightness match.
CHROMA_LENIENCE = 2.0

#: Tolerance of the spatial pass as a share of the temporal one; 0 leaves it out.
SPATIAL = 0.5

#: Half the window the spatial pass searches, and half the patch it compares, in pixels.
SEARCH_RADIUS = 3
PATCH_RADIUS = 1

#: How much the removed-noise picture magnifies what was taken out.
NOISE_GAIN = 4.0


def strictness(detail: float) -> float:
    """How far a match may exceed the noise before its weight falls by ``e``.

    Args:
        detail: 0 = smoothest, 1 = keeps the finest texture.

    Returns:
        The tolerance, in multiples of the match expected from noise alone.
    """
    loose, tight = STRICTNESS
    share = max(0.0, min(1.0, float(detail)))
    return math.exp(math.log(loose) * (1.0 - share) + math.log(tight) * share)


def _box(plane, radius: int):
    """Mean over a ``2 * radius + 1`` square window, edges averaged over what is inside."""
    return F.avg_pool2d(plane, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)


def _shifted(planes, dx: int, dy: int):
    """``planes`` read ``(dx, dy)`` pixels away, edges held."""
    height, width = planes.shape[-2:]
    pad = max(abs(dx), abs(dy))
    padded = F.pad(planes, (pad, pad, pad, pad), mode="replicate")
    return padded[..., pad + dy:pad + dy + height, pad + dx:pad + dx + width]


def _spatial(planes, level, tolerance: float):
    """Non-local means over a small window, judged against the noise each pixel still holds.

    Args:
        planes: Opponent planes, ``(1, 3, height, width)``.
        level: Noise level left in each channel at each pixel, shaped like ``planes``.
        tolerance: How far a patch match may exceed the noise before its weight falls by ``e``.

    Returns:
        Planes shaped like ``planes``.
    """
    scale = 1.0 / (2.0 * level.square().clamp(min=1e-10))
    total = planes.clone()
    weight = torch.ones_like(planes[:, :1])
    for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            if dx == 0 and dy == 0:
                continue
            other = _shifted(planes, dx, dy)
            match = _box(((planes - other).square() * scale).mean(1, keepdim=True), PATCH_RADIUS)
            share = torch.exp(-(match - 1.0).clamp(min=0.0) / tolerance)
            total += share * other
            weight += share
    return total / weight


class Report:
    """What a denoise run measured.

    Attributes:
        levels: Noise level per frame, in levels of 255 per RGB channel, middle brightness band.
        used: Neighbours each frame took any weight from.
        weight: Mean total weight each frame's neighbours carried.
    """

    def __init__(self):
        self.levels: list[float] = []
        self.used: list[int] = []
        self.weight: list[float] = []


def denoise(frames, flows, radius: int = 2, luma: float = 1.0, chroma: float = 1.0,
            detail: float = 0.5, level: float = 0.0, device=None, progress=None, node: str = "",
            keep_noise: bool = True):
    """Average every frame with its neighbours along the motion, where they line up.

    Args:
        frames: ``(frames, height, width, channels)`` picture codes, float or uint8.
        flows: A :class:`~.temporal_restore.Flows` over ``frames``.
        radius: Neighbours averaged on each side.
        luma: Share of the brightness noise removed, 0 to 1.
        chroma: Share of the colour noise removed, 0 to 1.
        detail: 0 = smoothest, 1 = keeps the finest texture.
        level: Noise level in levels of 255; 0 measures it on every frame.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame.
        node: Display name of the calling node, for the refusal.
        keep_noise: Answer a picture of what was removed beside the frames.

    Returns:
        ``(frames, noise, report)``: the cleaned frames shaped like ``frames`` in float32; what
        was removed, magnified :data:`NOISE_GAIN` times about mid grey, or ``None``; and a
        :class:`Report`.

    Raises:
        MemoryError: Neither free memory nor a scratch drive can hold the result.
    """
    count, height, width, channels = (int(v) for v in frames.shape)
    device = torch.device(device) if device is not None else frames.device
    advice = "A shorter clip, or the clip at a smaller size, also fits it."
    out = scratch.allocate((count, height, width, channels), torch.float32, node=node, advice=advice)
    noise = (
        scratch.allocate((count, height, width, 3), torch.float32, node=node, advice=advice)
        if keep_noise else None
    )
    radius = max(0, int(radius))
    luma = max(0.0, min(1.0, float(luma)))
    chroma = max(0.0, min(1.0, float(chroma)))
    tolerance = strictness(detail)
    wanted = float(level) / 255.0
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
        low, high = max(0, index - radius), min(count - 1, index + radius)
        flows.measure([(i, i + 1) for i in range(low, high)] + [(i + 1, i) for i in range(low, high)])
        offsets = flows.reachable(index, radius) if radius else []
        flows.measure([(index, index + k) for k in offsets] + [(index + k, index) for k in offsets])
        centre = picture(index)
        colour = centre[:, :3]
        centre_o = opponent(colour)

        def aligned(k):
            warped, seen, spread = temporal_restore.align(
                picture(index + k)[:, :3], flows.flow(index, index + k), flows.flow(index + k, index),
                (height, width),
            )
            return opponent(warped) - centre_o, seen, spread

        nearest = {k: aligned(k) for k in (-1, 1) if k in offsets}
        table = temporal_restore.noise_table(centre_o, list(nearest.values()))
        if table is None:
            table = last_table if last_table is not None else temporal_restore.spatial_noise(centre_o)
        last_table = table
        if wanted > 0.0:
            table = table * (wanted / float(table.square().mean(0).sqrt().mean()))
        variance = temporal_restore.noise_map(table, centre_o).square()
        report.levels.append(float(table[:, temporal_restore.BINS // 2].square().mean().sqrt()) * 255.0)

        total_y = centre_o[:, :1].clone()
        total_c = centre_o[:, 1:].clone()
        weight_y = torch.ones_like(total_y)
        weight_c = torch.ones_like(total_y)
        square_y = torch.ones_like(total_y)
        square_c = torch.ones_like(total_y)
        used = 0
        for k in offsets:
            difference, seen, spread = nearest.pop(k) if k in nearest else aligned(k)
            match = _box((difference.square() / (variance * (1.0 + spread))).mean(1, keepdim=True), MATCH_RADIUS)
            excess = (match - 1.0).clamp(min=0.0)
            share_y = seen * torch.exp(-excess / tolerance)
            share_c = seen * torch.exp(-excess / (tolerance * CHROMA_LENIENCE))
            other = difference + centre_o
            total_y += share_y * other[:, :1]
            total_c += share_c * other[:, 1:]
            weight_y += share_y
            weight_c += share_c
            square_y += share_y.square() * spread
            square_c += share_c.square() * spread
            used += int(bool((share_y > 0.05).any()))
        report.used.append(used)
        report.weight.append(float((weight_y - 1.0).mean()))
        mean = torch.cat([total_y / weight_y, total_c / weight_c], 1)
        if SPATIAL > 0.0:
            left = torch.cat([square_y.sqrt() / weight_y, (square_c.sqrt() / weight_c).expand(-1, 2, -1, -1)], 1)
            mean = _spatial(mean, variance.sqrt() * left, tolerance * SPATIAL)
        moved = mean - centre_o
        cleaned = from_opponent(centre_o + torch.cat([luma * moved[:, :1], chroma * moved[:, 1:]], 1))
        if hold:
            cleaned = cleaned.clamp(0.0, 1.0)
        out[index, ..., :3].copy_(cleaned[0].permute(1, 2, 0))
        if channels > 3:
            out[index, ..., 3:].copy_(centre[0, 3:].permute(1, 2, 0))
        if noise is not None:
            shown = (0.5 + NOISE_GAIN * (colour - cleaned)).clamp(0.0, 1.0)
            noise[index].copy_(shown[0].permute(1, 2, 0))
            scratch.trim(noise[index])
        scratch.trim(out[index])
        for stale in [i for i in held if i <= index - radius]:
            del held[stale]
        flows.forget(index + 1 - radius)
        if progress is not None:
            progress(1)
    return out, noise, report
