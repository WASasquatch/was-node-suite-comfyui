"""Tiled video super-resolution with the Kandinsky 6 upscaler.

Frames are ``(T, H, W, 3)`` in ``[0, 1]``. A clip is cut into overlapping windows in time, each
window into overlapping tiles, enlarged in latent space, refined, decoded and blended.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .. import unpin_staged
from . import sr

#: Scale choices: total enlargement, the tiling factor and the pixel enlargement before tiling.
SCALES = {"2x": (2, 1.0), "2.25x": (2, 1.125), "4x": (4, 1.0)}

#: Frames per window by default: the five seconds at 24 fps the upscaler was trained on.
WINDOW_FRAMES = 121

#: Frames two neighbouring windows share by default, crossfaded.
WINDOW_OVERLAP = 8


@dataclass
class Plan:
    """How a clip is cut into tiles.

    Attributes:
        frames: Frames given; ``padded`` frames are processed, a ``1 + 4k`` count.
        size: ``(height, width)`` the clip is read at, multiples of 16.
        tile: ``(height, width)`` of a tile in that clip.
        refined: ``(height, width)`` each tile is refined at.
        factor: Tiling enlargement, 2 or 4.
        tops: Tile rows, in pixels of the read clip.
        lefts: Tile columns, in pixels of the read clip.
        seconds: Seconds spent on ``encode``, ``refine`` and ``decode``.
    """

    frames: int
    padded: int
    size: tuple
    tile: tuple
    refined: tuple
    factor: int
    tops: list = field(default_factory=list)
    lefts: list = field(default_factory=list)
    seconds: dict = field(default_factory=lambda: {"encode": 0.0, "refine": 0.0, "decode": 0.0})

    @property
    def tiles(self) -> list:
        return [(top, left) for top in self.tops for left in self.lefts]


def tile_positions(length: int, tile: int, min_overlap: float, snap: int) -> list:
    """Evenly spread, ``snap``-aligned starts covering ``length`` with at least ``min_overlap`` shared."""
    if tile >= length:
        return [0]
    span = (length - tile) // snap
    stride = max(1, math.floor(tile * (1.0 - min_overlap) / snap))
    count = math.ceil(span / stride) + 1
    return [round(index * span / (count - 1)) * snap for index in range(count)]


def hann_window(height: int, width: int) -> torch.Tensor:
    """2D Hann window with non-zero borders."""
    window_y = torch.hann_window(height + 2, periodic=True)[1:-1]
    window_x = torch.hann_window(width + 2, periodic=True)[1:-1]
    return window_y[:, None] * window_x[None, :]


def plan(frames: int, height: int, width: int, scale: str, overlap: float) -> Plan:
    """Cut a clip into tiles for ``scale``.

    Raises:
        ValueError: ``scale`` names nothing offered, or the clip is smaller than one tile.
    """
    if scale not in SCALES:
        raise ValueError(f"No scale {scale!r}. Choose one of: {', '.join(SCALES)}.")
    factor, pre = SCALES[scale]
    snap = sr.SPATIAL_FACTOR
    read = (max(snap, round(height * pre / snap) * snap), max(snap, round(width * pre / snap) * snap))
    if pre == 1.0:
        read = (max(snap, height // snap * snap), max(snap, width // snap * snap))
    refined = min(sr.TILE_SIZES, key=lambda hw: abs(hw[1] / hw[0] - read[1] / read[0]))
    tile = (refined[0] // factor, refined[1] // factor)
    if read[0] < tile[0] or read[1] < tile[1]:
        raise ValueError(
            f"A {scale} Kandinsky 6 upscale needs frames at least {tile[1]}x{tile[0]} pixels; these are "
            f"{width}x{height}. Enlarge them first or choose 2x."
        )
    padded = 1 + math.ceil(max(0, frames - 1) / sr.TEMPORAL_FACTOR) * sr.TEMPORAL_FACTOR
    result = Plan(frames, padded, read, tile, refined, factor)
    result.tops = tile_positions(read[0], tile[0], overlap, snap)
    result.lefts = tile_positions(read[1], tile[1], overlap, snap)
    return result


def _read(images: torch.Tensor, layout: Plan) -> torch.Tensor:
    clip = images[..., :3].float()
    if images.dtype == torch.uint8:
        clip /= 255.0
    if layout.padded > layout.frames:
        clip = torch.cat([clip, clip[-1:].expand(layout.padded - layout.frames, -1, -1, -1)], dim=0)
    if tuple(clip.shape[1:3]) != layout.size:
        clip = F.interpolate(clip.movedim(-1, 1), size=layout.size, mode="bilinear", align_corners=False).movedim(1, -1)
    return clip


def upscale(model, vae, upscaler, images: torch.Tensor, scale: str = "2.25x", seed: int = 0,
            steps: int = 0, noise: float = 0.7, overlap: float = 0.2, tiles_per_pass: int = 1,
            progress=None) -> tuple[torch.Tensor, Plan]:
    """Enlarge a clip with the Kandinsky 6 upscaler.

    Args:
        model: The super-resolution transformer's ``ModelPatcher``.
        vae: The K-VAE as a ComfyUI VAE.
        upscaler: A :class:`sr.LatentUpscaler` of the tiling factor, or ``None`` to enlarge tiles in pixels.
        images: Frames ``(T, H, W, C)`` in ``[0, 1]``.
        scale: A key of :data:`SCALES`.
        seed: Seed of the noise mixed into the tiles.
        steps: Model evaluations per tile, 0 for the checkpoint's own.
        noise: Share of noise mixed into the low-resolution latents, 0 to 1.
        overlap: Least share of a tile its neighbours cover, 0 to 0.9.
        tiles_per_pass: Tiles refined together.
        progress: Called with ``(done, total)`` after each tile.

    Returns:
        ``(frames, plan)``: ``(T, H * scale, W * scale, 3)`` frames in ``[0, 1]``, and how the clip was cut.
    """
    import comfy.model_management as mm
    import comfy.sample
    import comfy.samplers

    frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    layout = plan(frames, height, width, scale, overlap)
    clip = _read(images, layout)
    snap = sr.SPATIAL_FACTOR
    factor = layout.factor
    tiles = layout.tiles
    total = len(tiles)
    latent_format = model.model.latent_format

    if upscaler is not None and upscaler.scale != factor:
        raise ValueError(f"This scale tiles at x{factor} but the latent upscaler is x{upscaler.scale}.")

    low = []
    mark = time.perf_counter()
    if upscaler is not None:
        latents = vae.encode(clip)
        scaled = latent_format.process_in(latents.float())
        for top, left in tiles:
            mm.throw_exception_if_processing_interrupted()
            window = scaled[:, :, :, top // snap:(top + layout.tile[0]) // snap, left // snap:(left + layout.tile[1]) // snap]
            low.append(upscaler(window).cpu())
        del latents, scaled
    else:
        for top, left in tiles:
            mm.throw_exception_if_processing_interrupted()
            piece = clip[:, top:top + layout.tile[0], left:left + layout.tile[1]]
            piece = F.interpolate(piece.movedim(-1, 1), size=layout.refined, mode="bilinear", align_corners=False).movedim(1, -1)
            low.append(latent_format.process_in(vae.encode(piece).float()).cpu())
    unpin_staged()
    layout.seconds["encode"] = time.perf_counter() - mark
    mark = time.perf_counter()

    n_grid = int(getattr(model.model, "n_grid", 1))
    shift = float(getattr(model.model.model_sampling, "shift", sr.SHIFT))
    count = int(steps) if int(steps) > 0 else (2 if n_grid > 1 else 4)
    sigmas = sr.piflow_sigmas(count, shift) if n_grid > 1 else sr.flow_sigmas(count, shift)
    sampler = comfy.samplers.sampler_object("euler")
    blank = [[torch.zeros(1, 1, 64), {}]]
    keep = math.sqrt(max(0.0, 1.0 - noise * noise))

    refined = []
    for start in range(0, total, max(1, int(tiles_per_pass))):
        indices = list(range(start, min(total, start + max(1, int(tiles_per_pass)))))
        lows = torch.cat([low[index] for index in indices], dim=0)
        draws = torch.cat([
            torch.randn(low[index].shape, generator=torch.Generator().manual_seed(int(seed) + index))
            for index in indices
        ], dim=0)
        start_point = keep * lows + noise * draws
        out = comfy.sample.sample_custom(
            model, start_point, 1.0, sampler, sigmas, blank, blank, torch.zeros_like(lows), seed=int(seed),
        )
        refined.extend(out.float().cpu().split(1, dim=0))
        if progress is not None:
            progress(indices[-1] + 1, total * 2)
    del low
    unpin_staged()
    layout.seconds["refine"] = time.perf_counter() - mark
    mark = time.perf_counter()

    out_height, out_width = layout.size[0] * factor, layout.size[1] * factor
    canvas = torch.zeros((layout.padded, out_height, out_width, 3), dtype=torch.float32)
    weights = torch.zeros((out_height, out_width), dtype=torch.float32)
    window = hann_window(*layout.refined)
    for index, ((top, left), latent) in enumerate(zip(tiles, refined)):
        mm.throw_exception_if_processing_interrupted()
        pixels = vae.decode(latent)[0].float().cpu()
        top, left = top * factor, left * factor
        rows, cols = slice(top, top + layout.refined[0]), slice(left, left + layout.refined[1])
        canvas[:, rows, cols] += pixels[: layout.padded] * window[None, :, :, None]
        weights[rows, cols] += window
        if progress is not None:
            progress(total + index + 1, total * 2)
    canvas /= weights[None, :, :, None]
    canvas = canvas[: layout.frames].clamp_(0.0, 1.0)

    final = (round(height * factor * SCALES[scale][1]), round(width * factor * SCALES[scale][1]))
    if tuple(canvas.shape[1:3]) != final:
        canvas = F.interpolate(canvas.movedim(-1, 1), size=final, mode="bicubic", align_corners=False).movedim(1, -1).clamp_(0.0, 1.0)
    layout.seconds["decode"] = time.perf_counter() - mark
    return canvas, layout


def windows(frames: int, length: int = WINDOW_FRAMES, overlap: int = WINDOW_OVERLAP) -> list:
    """Frame ranges ``(start, stop)`` covering a clip, each neighbour sharing at least ``overlap``.

    Args:
        frames: Frames in the clip.
        length: Frames per window, at least 5.
        overlap: Frames two neighbours share, at most half of ``length``.

    Returns:
        The ranges in order, spread evenly from the clip's start to its end; one range where the
        clip fits a window.
    """
    frames, length = int(frames), max(1 + sr.TEMPORAL_FACTOR, int(length))
    if frames <= length:
        return [(0, frames)]
    overlap = min(max(0, int(overlap)), (length - 1) // 2)
    count = math.ceil((frames - overlap) / (length - overlap))
    starts = [round(index * (frames - length) / (count - 1)) for index in range(count)]
    return [(start, start + length) for start in starts]


def upscale_clip(model, vae, upscaler, read, frames: int, emit, scale: str = "2x", seed: int = 0,
                 steps: int = 0, noise: float = 0.7, overlap: float = 0.2, tiles_per_pass: int = 1,
                 window: int = WINDOW_FRAMES, window_overlap: int = WINDOW_OVERLAP, progress=None) -> dict:
    """Enlarge a clip window by window, handing finished frames on in order.

    Args:
        model: The super-resolution transformer's ``ModelPatcher``.
        vae: The K-VAE as a ComfyUI VAE.
        upscaler: A :class:`sr.LatentUpscaler`, or ``None`` to enlarge tiles in pixels.
        read: Called with ``(start, stop)``, answers those frames ``(T, H, W, C)``, float or uint8.
        frames: Frames in the clip.
        emit: Called with each run of finished frames ``(T, H', W', 3)`` in ``[0, 1]``.
        scale: A key of :data:`SCALES`.
        seed: Seed of the noise mixed into the tiles, the same in every window.
        steps: Model evaluations per tile, 0 for the checkpoint's own.
        noise: Share of noise mixed into the low-resolution latents.
        overlap: Least share of a tile its neighbours cover.
        tiles_per_pass: Tiles refined together.
        window: Frames per window.
        window_overlap: Frames neighbouring windows share, crossfaded.
        progress: Called with ``(done, total)``.

    Returns:
        ``{"windows", "plan", "seconds"}``: the frame ranges, the last window's :class:`Plan`, and
        seconds per stage summed over windows.
    """
    spans = windows(frames, window, window_overlap)
    seconds = {"encode": 0.0, "refine": 0.0, "decode": 0.0}
    tail, layout = None, None
    for number, (start, stop) in enumerate(spans):
        def step(done, total, number=number):
            if progress is not None:
                progress(number * total + done, len(spans) * total)

        out, layout = upscale(
            model, vae, upscaler, read(start, stop), scale=scale, seed=seed, steps=steps, noise=noise,
            overlap=overlap, tiles_per_pass=tiles_per_pass, progress=step,
        )
        for stage, spent in layout.seconds.items():
            seconds[stage] += spent
        if tail is not None:
            shared = int(tail.shape[0])
            weight = (torch.arange(1, shared + 1, dtype=torch.float32) / (shared + 1))[:, None, None, None]
            out[:shared].mul_(weight).add_(tail.mul_(1.0 - weight))
            tail = None
        keep = spans[number + 1][0] - start if number + 1 < len(spans) else int(out.shape[0])
        emit(out[:keep])
        if keep < int(out.shape[0]):
            tail = out[keep:].clone()
        del out
    return {"windows": spans, "plan": layout, "seconds": seconds}
