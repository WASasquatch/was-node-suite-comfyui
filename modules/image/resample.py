"""Separable image resampling on tensors with Pillow's filter weights.

Frames are ``(batch, height, width, channels)`` floats. Each axis is one weight matrix, so a
resize is two matrix products.
"""

from __future__ import annotations

import math

import torch

from . import scratch, sizing

__all__ = [
    "ADVICE",
    "BAND_ROWS",
    "DENSE_WEIGHTS",
    "FILTERS",
    "MAX_WEIGHTS",
    "SUPPORT",
    "apply",
    "axis",
    "composed",
    "matrix",
    "resize",
    "supersampled",
]

#: Closing sentence of the refusal when a resampled batch fits neither memory nor a drive.
ADVICE = "Lowering the target size or passing fewer frames also fits it."

#: Most entries one weight matrix may hold, a 30000 pixel axis resampled to 30000 pixels.
MAX_WEIGHTS = sizing.MAX_CANVAS_SIDE * sizing.MAX_CANVAS_SIDE

#: Most entries either matrix :func:`composed` multiplies whole; past it the product is built
#: in bands of rows.
DENSE_WEIGHTS = 1 << 27

#: Target rows one band of :func:`composed` covers.
BAND_ROWS = 64


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


def _taps(size_in: int, size_out: int, name: str, box, index: int):
    """The source span one resample row reads, and its weights.

    Args:
        size_in: Source length in pixels.
        size_out: Target length of the resample in pixels.
        name: ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos``.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.
        index: The resample row.

    Returns:
        ``(first, last, taps)``: the row reads source pixels ``first`` to ``last - 1`` with
        the float weights ``taps``, which sum to 1. ``taps`` is empty where it reads none.
    """
    start, end = box if box is not None else (0.0, float(size_in))
    scale = (end - start) / size_out
    if name not in FILTERS:
        source = min(max(int(start + (index + 0.5) * scale), 0), size_in - 1)
        return source, source + 1, [1.0]
    kernel = FILTERS[name]
    widen = max(scale, 1.0)
    support = SUPPORT[name] * widen
    centre = start + (index + 0.5) * scale
    first = max(math.trunc(centre - support + 0.5), 0)
    last = min(math.trunc(centre + support + 0.5), size_in)
    taps = [kernel((x + first - centre + 0.5) / widen) for x in range(last - first)]
    total = sum(taps)
    if total != 0.0:
        taps = [tap / total for tap in taps]
    return first, last, taps


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
    for index in rows:
        first, last, taps = _taps(size_in, size_out, name, box, index)
        if taps:
            weights[index + shift, first:last] = torch.tensor(taps, dtype=torch.float64)


def _zeros(rows: int, columns: int):
    """An empty weight matrix.

    Args:
        rows: Target length in pixels.
        columns: Source length in pixels.

    Returns:
        A ``(rows, columns)`` float64 tensor of zeros.

    Raises:
        CanvasTooLarge: The matrix would hold more than :data:`MAX_WEIGHTS` entries.
    """
    if rows * columns > MAX_WEIGHTS:
        raise sizing.CanvasTooLarge(
            f"Resampling a {columns} pixel axis to {rows} pixels takes a {rows}x{columns} "
            f"weight matrix of {rows * columns * 8 / 1e9:.1f} GB, more than the "
            f"{MAX_WEIGHTS:,} entries one resample may hold. Nothing was allocated. Lower "
            f"the target size, or crop or shrink the source first."
        )
    return torch.zeros((rows, columns), dtype=torch.float64)


def matrix(size_in: int, size_out: int, name: str, box: tuple[float, float] | None = None):
    """The weights that resample one axis.

    Args:
        size_in: Source length in pixels.
        size_out: Target length in pixels.
        name: ``nearest``, ``bilinear``, ``bicubic`` or ``lanczos``.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.

    Returns:
        A ``(size_out, size_in)`` float64 tensor whose rows sum to 1.

    Raises:
        CanvasTooLarge: The matrix would hold more than :data:`MAX_WEIGHTS` entries.
    """
    weights = _zeros(size_out, size_in)
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

    Raises:
        CanvasTooLarge: The size is past the largest image
            :func:`modules.image.sizing.check_canvas` allows.
    """
    sizing.check_canvas(width, height, "The resampled image", "Lower the target width or height.")
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

    Raises:
        CanvasTooLarge: The matrix would hold more than :data:`MAX_WEIGHTS` entries.
    """
    weights = _zeros(target, size_in)
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

    Raises:
        CanvasTooLarge: The resample down would hold more than :data:`MAX_WEIGHTS` entries.
    """
    weights, coverage = plan
    down = matrix(int(weights.shape[0]), target, name)
    return down @ weights, down @ coverage


def composed(
    size_in: int,
    scaled: int,
    length: int,
    offset: int,
    name: str,
    box: tuple[float, float] | None,
    target: int,
):
    """One axis planned on a larger canvas and resampled down to the target.

    Args:
        size_in: Source length in pixels.
        scaled: Length the source is resampled to on the larger canvas.
        length: Larger canvas length in pixels.
        offset: Where the resampled source starts on the larger canvas, negative to crop.
        name: A filter name.
        box: ``(start, end)`` of the source span to resample, the whole axis when None.
        target: Final length in pixels.

    Returns:
        ``(weights, coverage)`` as :func:`supersampled` answers for :func:`axis`: a
        ``(target, size_in)`` float64 matrix and a ``(target,)`` vector. Where either matrix
        would pass :data:`DENSE_WEIGHTS` entries, the product is built :data:`BAND_ROWS`
        target rows at a time, over only the source pixels those rows reach.

    Raises:
        CanvasTooLarge: The answer would hold more than :data:`MAX_WEIGHTS` entries.
    """
    if length * max(size_in, target) <= DENSE_WEIGHTS:
        return supersampled(axis(size_in, scaled, length, offset, name, box), target, name)

    weights = _zeros(target, size_in)
    coverage = torch.zeros(target, dtype=torch.float64)
    covered = (max(offset, 0), min(offset + scaled, length))
    for begin in range(0, target, BAND_ROWS):
        rows = range(begin, min(target, begin + BAND_ROWS))
        spans = [_taps(length, target, name, None, index) for index in rows]
        first = min(span[0] for span in spans)
        last = max(first, max(span[1] for span in spans))
        low, high = max(covered[0], first), min(covered[1], last)
        if low >= high:
            continue
        down = torch.zeros((len(spans), last - first), dtype=torch.float64)
        for row, (start, end, taps) in enumerate(spans):
            if taps:
                down[row, start - first:end - first] = torch.tensor(taps, dtype=torch.float64)
        # Larger-canvas rows low to high, as axis() builds them, cut to the pixels they read.
        reads = [_taps(size_in, scaled, name, box, index - offset) for index in range(low, high)]
        left = min(span[0] for span in reads)
        right = max(left, max(span[1] for span in reads))
        band = torch.zeros((last - first, right - left), dtype=torch.float64)
        for place, (start, end, taps) in enumerate(reads, low - first):
            if taps:
                band[place, start - left:end - left] = torch.tensor(taps, dtype=torch.float64)
        band += 0.0
        shown = torch.zeros(last - first, dtype=torch.float64)
        shown[low - first:high - first] = 1.0
        weights[begin:begin + len(spans), left:right] = down @ band
        coverage[begin:begin + len(spans)] = down @ shown
    weights += 0.0
    return weights, coverage


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
        budget: Elements of source and result held at once, which sets the chunk size. A
            sixteenth of it is the most elements one band of pad fill holds.

    Returns:
        ``(batch, target height, target width, channels)`` on the frames' own device and
        dtype. A CPU result comes from :func:`modules.image.scratch.allocate`.

    Raises:
        CanvasTooLarge: The plans describe an image past the largest
            :func:`modules.image.sizing.check_canvas` allows.
        MemoryError: Neither free memory nor any scratch drive has room for the result.
    """
    batch, rows, columns, channels = (int(size) for size in frames.shape)
    width, height = int(across[0].shape[0]), int(down[0].shape[0])
    sizing.check_canvas(width, height, "The resampled image", "Lower the target width or height.")
    run = device or frames.device
    across_weights = across[0].to(run, torch.float32)
    down_weights = down[0].to(run, torch.float32)
    fill = torch.zeros(channels, dtype=torch.float32, device=run)
    if pad is not None:
        fill = torch.tensor([float(value) for value in pad][:channels], dtype=torch.float32, device=run)
        if fill.numel() < channels:
            fill = torch.cat([fill, fill.new_zeros(channels - fill.numel())])
    premultiply = premultiply and channels == 4
    if premultiply:
        fill = torch.cat([fill[:3] * fill[3], fill[3:]])
    band = max(1, (budget >> 4) // max(1, width * channels))
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
        # The pad fill is laid where the canvas shows no source, a band of rows at a time.
        for top in range(0, height, band):
            uncovered = 1.0 - torch.outer(down[1][top:top + band], across[1]).to(run, torch.float32)
            done[:, top:top + band] += uncovered[None, :, :, None] * fill
        if premultiply:
            alpha = done[..., 3:4]
            safe = torch.where(alpha > 0, alpha, torch.ones_like(alpha))
            done[..., :3] = torch.where(alpha > 0, done[..., :3] / safe, 0.0)
        if not direct:
            held.copy_(done)
    return result
