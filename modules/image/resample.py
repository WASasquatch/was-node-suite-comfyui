"""Separable image resampling on tensors with Pillow's filter weights.

Frames are ``(batch, height, width, channels)`` floats. Each axis is one weight matrix, so a
resize is two matrix products.
"""

from __future__ import annotations

import math

import torch

from . import scratch

__all__ = ["ADVICE", "FILTERS", "SUPPORT", "apply", "axis", "matrix", "resize", "supersampled"]

#: Closing sentence of the refusal when a resampled batch fits neither memory nor a drive.
ADVICE = "Lowering the target size or passing fewer frames also fits it."


def _bilinear(x: float) -> float:
    x = abs(x)
    return 1.0 - x if x < 1.0 else 0.0


def _bicubic(x: float) -> float:
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    x *= math.pi
    return math.sin(x) / x


def _lanczos(x: float) -> float:
    return _sinc(x) * _sinc(x / 3.0) if -3.0 < x < 3.0 else 0.0


#: Filter name to its kernel, for every filter but ``nearest``.
FILTERS = {"bilinear": _bilinear, "bicubic": _bicubic, "lanczos": _lanczos}

#: Filter name to the half width of its kernel in source pixels at a scale of 1.
SUPPORT = {"bilinear": 1.0, "bicubic": 2.0, "lanczos": 3.0}


def _fill(weights, size_in: int, size_out: int, name: str, box, rows: range, shift: int = 0):
    """Write rows of one axis's resample weights into a matrix.

    Args:
        weights: ``(rows, size_in)`` float64 tensor, written in place.
        size_in: Source length in pixels.
        size_out: Target length of the resample in pixels.
        name: ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos``.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.
        rows: Indices of the resample rows to write.
        shift: Added to a resample row index to give the row of ``weights`` it lands on.
    """
    start, end = box if box is not None else (0.0, float(size_in))
    scale = (end - start) / size_out
    if name not in FILTERS:
        for index in rows:
            source = int(start + (index + 0.5) * scale)
            weights[index + shift, min(max(source, 0), size_in - 1)] = 1.0
        return
    kernel = FILTERS[name]
    widen = max(scale, 1.0)
    support = SUPPORT[name] * widen
    for index in rows:
        centre = start + (index + 0.5) * scale
        first = max(math.trunc(centre - support + 0.5), 0)
        last = min(math.trunc(centre + support + 0.5), size_in)
        taps = [kernel((x + first - centre + 0.5) / widen) for x in range(last - first)]
        total = sum(taps)
        if total != 0.0:
            taps = [tap / total for tap in taps]
        if taps:
            weights[index + shift, first:last] = torch.tensor(taps, dtype=torch.float64)


def matrix(size_in: int, size_out: int, name: str, box: tuple[float, float] | None = None):
    """The weights that resample one axis.

    Args:
        size_in: Source length in pixels.
        size_out: Target length in pixels.
        name: ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos``.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.

    Returns:
        A ``(size_out, size_in)`` float64 tensor whose rows sum to 1.
    """
    weights = torch.zeros((size_out, size_in), dtype=torch.float64)
    _fill(weights, size_in, size_out, name, box, range(size_out))
    return weights


def resize(
    frames: torch.Tensor,
    width: int,
    height: int,
    name: str = "lanczos",
    box: tuple[float, float, float, float] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Resample a batch of frames to a size.

    Args:
        frames: ``(batch, height, width, channels)`` float tensor.
        width: Target width in pixels.
        height: Target height in pixels.
        name: ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos``.
        box: ``(left, upper, right, lower)`` of the source to resample, all of it when None.
        device: Where the products run, the frames' own device when None.

    Returns:
        ``(batch, height, width, channels)`` on the frames' own device and dtype.
    """
    rows, columns = int(frames.shape[1]), int(frames.shape[2])
    left, upper, right, lower = box if box is not None else (0.0, 0.0, float(columns), float(rows))
    across = axis(columns, width, width, 0, name, (left, right))
    down = axis(rows, height, height, 0, name, (upper, lower))
    return apply(frames, across, down, device=device, premultiply=name != "nearest")


def axis(
    size_in: int,
    scaled: int,
    target: int,
    offset: int,
    name: str,
    box: tuple[float, float] | None = None,
):
    """One axis of a resample followed by placement on a larger or smaller canvas.

    Args:
        size_in: Source length in pixels.
        scaled: Length the source is resampled to.
        target: Canvas length in pixels.
        offset: Where the resampled source starts on the canvas, negative to crop.
        name: A filter name.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.

    Returns:
        ``(weights, coverage)``: a ``(target, size_in)`` float64 matrix and a ``(target,)``
        vector, 1 where the canvas shows the source and 0 where it shows padding.
    """
    weights = torch.zeros((target, size_in), dtype=torch.float64)
    coverage = torch.zeros(target, dtype=torch.float64)
    first, last = max(offset, 0), min(offset + scaled, target)
    if first < last:
        _fill(weights, size_in, scaled, name, box, range(first - offset, last - offset), offset)
        # Adding zero turns a negative zero tap into a positive one.
        weights[first:last] += 0.0
        coverage[first:last] = 1.0
    return weights, coverage


def supersampled(plan, target: int, name: str):
    """Follow an axis plan built at a larger size with a resample down to the target.

    Args:
        plan: ``(weights, coverage)`` from :func:`axis` at the larger size.
        target: Final length in pixels.
        name: A filter name.

    Returns:
        ``(weights, coverage)`` at the target length.
    """
    weights, coverage = plan
    down = matrix(int(weights.shape[0]), target, name)
    return down @ weights, down @ coverage


def _resampled(frame: torch.Tensor, across: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """One ``(height, width, channels)`` frame through both axis matrices."""
    rows, columns, channels = (int(size) for size in frame.shape)
    width, height = int(across.shape[0]), int(down.shape[0])
    wide = across @ frame.permute(1, 0, 2).reshape(columns, rows * channels)
    wide = wide.reshape(width, rows, channels).permute(1, 0, 2).reshape(rows, width * channels)
    return (down @ wide).reshape(height, width, channels)


def apply(
    frames: torch.Tensor, across, down, pad=None, device=None, premultiply=True, budget=1 << 26
):
    """Resample and place a batch of frames by two axis plans.

    Args:
        frames: ``(batch, height, width, channels)`` float tensor.
        across: ``(weights, coverage)`` for the width.
        down: ``(weights, coverage)`` for the height.
        pad: Per-channel fill for uncovered canvas, as floats; zeros when None.
        device: Where the products run, the frames' own device when None.
        premultiply: Whether four channels are worked on premultiplied by alpha.
        budget: Elements of source and result held at once, which sets the chunk size.

    Returns:
        ``(batch, target height, target width, channels)`` on the frames' own device and
        dtype. A CPU result comes from :func:`modules.image.scratch.allocate`.

    Raises:
        MemoryError: Neither free memory nor any scratch drive has room for the result.
    """
    batch, rows, columns, channels = (int(size) for size in frames.shape)
    run = device or frames.device
    across_weights = across[0].to(run, torch.float32)
    down_weights = down[0].to(run, torch.float32)
    width, height = int(across_weights.shape[0]), int(down_weights.shape[0])
    uncovered = 1.0 - torch.outer(down[1], across[1]).to(run, torch.float32)
    fill = torch.zeros(channels, dtype=torch.float32, device=run)
    if pad is not None:
        fill = torch.tensor([float(value) for value in pad][:channels], dtype=torch.float32, device=run)
        if fill.numel() < channels:
            fill = torch.cat([fill, fill.new_zeros(channels - fill.numel())])
    premultiply = premultiply and channels == 4
    if premultiply:
        fill = torch.cat([fill[:3] * fill[3], fill[3:]])
    backdrop = uncovered[None, :, :, None] * fill
    per_frame = max(1, (rows * columns + height * width + rows * width) * channels)
    chunk = max(1, int(budget // per_frame))
    shape = (batch, height, width, channels)
    if frames.device.type == "cpu":
        result = scratch.allocate(shape, frames.dtype, advice=ADVICE)
    else:
        result = torch.empty(shape, dtype=frames.dtype, device=frames.device)
    direct = result.device == torch.device(run) and result.dtype == torch.float32
    for begin in range(0, batch, chunk):
        part = frames[begin:begin + chunk].to(run, torch.float32)
        if premultiply:
            part = torch.cat([part[..., :3] * part[..., 3:4], part[..., 3:4]], dim=-1)
        count = int(part.shape[0])
        held = result[begin:begin + count]
        done = held if direct else torch.empty(
            (count, height, width, channels), dtype=torch.float32, device=run
        )
        # Each frame is resampled by its own pair of products.
        for index in range(count):
            done[index] = _resampled(part[index], across_weights, down_weights)
        del part
        done += backdrop
        if premultiply:
            alpha = done[..., 3:4]
            safe = torch.where(alpha > 0, alpha, torch.ones_like(alpha))
            done[..., :3] = torch.where(alpha > 0, done[..., :3] / safe, 0.0)
        if not direct:
            held.copy_(done)
    return result
