"""Rasterising triangles onto a pixel grid, sampling textures and filling the gutter between charts.
Pixel centres sit at half-integer coordinates."""
# A texture coordinate ``(u, v)`` lands at column ``u * width`` and row ``v * height``, so ``v``
# runs down the image.

from __future__ import annotations

__all__ = [
    "Coverage",
    "MAX_GRID_PIXELS",
    "coverage_chunks",
    "edge_pixels",
    "fill_gaps",
    "nearest_filled",
    "rasterize",
    "sample_texture",
    "texel_triangles",
    "winners",
]

import math
from dataclasses import dataclass
from typing import Iterator

import torch

#: Most pixels a grid handed to the rasteriser may hold.
MAX_GRID_PIXELS = 1 << 28

#: Pixels produced per chunk of coverage.
CHUNK_PIXELS = 1 << 22

#: Texels gathered per pass when sampling a texture.
SAMPLE_CHUNK = 1 << 21


@dataclass
class Coverage:
    """Pixel centres that fall inside triangles.

    Attributes:
        face: ``(K,)`` int64 index of the covering triangle.
        pixel: ``(K,)`` int64 ``row * width + column``.
        barycentric: ``(K, 3)`` float32 weights of the triangle's corners at the pixel centre.
    """

    face: torch.Tensor
    pixel: torch.Tensor
    barycentric: torch.Tensor

    @property
    def count(self) -> int:
        """Number of covered pixel centres."""
        return int(self.face.shape[0])

    def select(self, keep: torch.Tensor) -> Coverage:
        """The entries where ``keep`` is True."""
        return Coverage(self.face[keep], self.pixel[keep], self.barycentric[keep])


def _check_grid(width: int, height: int) -> None:
    """Refuse a grid past `MAX_GRID_PIXELS`."""
    if width < 1 or height < 1 or width * height > MAX_GRID_PIXELS:
        raise ValueError(f"a {width}x{height} grid is outside what the rasteriser takes; lower the "
                         "resolution or the samples per texel")


def _rows(corners: torch.Tensor, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """First covered row and row count of each triangle, zero rows for an unusable one."""
    finite = torch.isfinite(corners).all(dim=2).all(dim=1)
    safe = torch.where(finite[:, None, None], corners, torch.zeros_like(corners))
    first_edge = safe[:, 1] - safe[:, 0]
    second_edge = safe[:, 2] - safe[:, 0]
    area = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    usable = finite & (area != 0)
    y = safe[..., 1]
    first = torch.ceil(y.amin(dim=1) - 0.5).clamp(0, height).long()
    last = torch.floor(y.amax(dim=1) - 0.5).clamp(-1, height - 1).long()
    rows = torch.where(usable, (last - first + 1).clamp_min(0), torch.zeros_like(first))
    return first, rows


def _span_bounds(corners: torch.Tensor, rows: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """First and last covered column of each span, from where its row crosses the triangle.

    Args:
        corners: ``(S, 3, 2)`` corners of the triangle each span belongs to.
        rows: ``(S,)`` row of each span.
        width: Grid width.

    Returns:
        ``(first, last)`` int64 columns; ``last < first`` for an empty span.
    """
    centre = rows.to(corners.dtype) + 0.5
    low = torch.full_like(centre, math.inf)
    high = torch.full_like(centre, -math.inf)
    for side in range(3):
        start = corners[:, side]
        end = corners[:, (side + 1) % 3]
        y0, y1 = start[:, 1], end[:, 1]
        crosses = ((y0 <= centre) & (centre <= y1)) | ((y1 <= centre) & (centre <= y0))
        flat = y0 == y1
        slope_part = (centre - y0) / torch.where(flat, torch.ones_like(y0), y1 - y0)
        x = start[:, 0] + slope_part * (end[:, 0] - start[:, 0])
        sloped = crosses & ~flat
        low = torch.where(sloped, torch.minimum(low, x), low)
        high = torch.where(sloped, torch.maximum(high, x), high)
        level = crosses & flat
        low = torch.where(level, torch.minimum(low, torch.minimum(start[:, 0], end[:, 0])), low)
        high = torch.where(level, torch.maximum(high, torch.maximum(start[:, 0], end[:, 0])), high)
    first = torch.ceil(low - 0.5).nan_to_num(0, posinf=width, neginf=0).clamp(0, width).long()
    last = torch.floor(high - 0.5).nan_to_num(-1, posinf=width - 1, neginf=-1).clamp(-1, width - 1).long()
    return first, last


def _barycentric(corners: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Barycentric weights of points against triangles, both ``(K, ...)``."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    area = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    w1 = (ap[:, 0] * ac[:, 1] - ap[:, 1] * ac[:, 0]) / area
    w2 = (ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / area
    weights = torch.stack([1.0 - w1 - w2, w1, w2], dim=1)
    weights = weights.clamp_min(0.0)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def texel_triangles(uvs: torch.Tensor, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangles in texel units on a repeating texture, each kept whole across a tile border.

    Args:
        uvs: ``(F, 3, 2)`` texture coordinates of each triangle's corners.
        width: Texture width in texels.
        height: Texture height in texels.

    Returns:
        ``(corners, owner)``: ``(N, 3, 2)`` corners in texels, each triangle moved by whole tiles
        until its lowest corner lies in the first tile, plus a copy a tile back on every axis it
        runs past; and ``(N,)`` the index of the triangle each row came from.
    """
    work = uvs.float()
    base = work - torch.floor(work.min(dim=1).values)[:, None, :]
    over = base.max(dim=1).values > 1.0 + 1e-6
    every = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
    pieces, owners = [base], [torch.arange(base.shape[0], device=base.device)]
    for back_u, back_v in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        wanted = (over[:, 0] if back_u else every) & (over[:, 1] if back_v else every)
        if bool(wanted.any()):
            shift = torch.tensor([back_u, back_v], dtype=base.dtype, device=base.device)
            pieces.append(base[wanted] - shift)
            owners.append(torch.nonzero(wanted).squeeze(1))
    scale = torch.tensor([float(width), float(height)], dtype=base.dtype, device=base.device)
    return torch.cat(pieces) * scale, torch.cat(owners)


def coverage_chunks(corners: torch.Tensor, width: int, height: int,
                    chunk_pixels: int = CHUNK_PIXELS) -> Iterator[Coverage]:
    """Pixel centres inside triangles, in pieces of bounded size.

    Args:
        corners: ``(F, 3, 2)`` triangle corners in pixel units, ``x`` along a row.
        width: Grid width in pixels.
        height: Grid height in pixels.
        chunk_pixels: Most entries in one piece.

    Yields:
        `Coverage` pieces. A centre on an edge shared by two triangles appears once for each.

    Raises:
        ValueError: The grid is past `MAX_GRID_PIXELS`.
    """
    _check_grid(width, height)
    if corners.shape[0] == 0:
        return
    work = corners.float()
    first_row, rows = _rows(work, height)
    total = torch.cumsum(rows, 0)
    count = int(total[-1]) if total.numel() else 0
    if count == 0:
        return
    budget = max(1, int(chunk_pixels) // 16)
    face_start = 0
    face_count = int(corners.shape[0])
    while face_start < face_count:
        before = int(total[face_start - 1]) if face_start else 0
        face_stop = int(torch.searchsorted(total, torch.tensor(before + budget, device=total.device), right=True))
        face_stop = min(max(face_stop, face_start + 1), face_count)
        chunk_rows = rows[face_start:face_stop]
        if int(chunk_rows.sum()):
            span_face = torch.repeat_interleave(
                torch.arange(face_start, face_stop, device=work.device), chunk_rows)
            offsets = torch.cumsum(chunk_rows, 0) - chunk_rows
            local = torch.arange(span_face.shape[0], device=work.device) - torch.repeat_interleave(offsets, chunk_rows)
            span_row = first_row[span_face] + local
            yield from _span_pixels(work, span_face, span_row, width, chunk_pixels)
        face_start = face_stop


def _span_pixels(corners: torch.Tensor, span_face: torch.Tensor, span_row: torch.Tensor, width: int,
                 chunk_pixels: int) -> Iterator[Coverage]:
    """The pixels of spans, in pieces of at most ``chunk_pixels``."""
    first, last = _span_bounds(corners[span_face], span_row, width)
    lengths = (last - first + 1).clamp_min(0)
    ends = torch.cumsum(lengths, 0)
    span_count = int(span_face.shape[0])
    start = 0
    while start < span_count:
        before = int(ends[start - 1]) if start else 0
        stop = int(torch.searchsorted(ends, torch.tensor(before + chunk_pixels, device=ends.device), right=True))
        stop = min(max(stop, start + 1), span_count)
        piece = lengths[start:stop]
        pixel_count = int(piece.sum())
        if pixel_count:
            owner = torch.repeat_interleave(torch.arange(start, stop, device=ends.device), piece)
            offsets = torch.cumsum(piece, 0) - piece
            column = first[owner] + torch.arange(pixel_count, device=ends.device) - torch.repeat_interleave(offsets, piece)
            row = span_row[owner]
            face = span_face[owner]
            centre = torch.stack([column.float() + 0.5, row.float() + 0.5], dim=1)
            weights = _barycentric(corners[face], centre)
            yield Coverage(face, row * width + column, weights)
        start = stop


def rasterize(corners: torch.Tensor, width: int, height: int) -> Coverage:
    """Every pixel centre inside the triangles, in one piece.

    Args:
        corners: ``(F, 3, 2)`` triangle corners in pixel units.
        width: Grid width in pixels.
        height: Grid height in pixels.

    Returns:
        The `Coverage`, with a centre on a shared edge listed once per triangle.
    """
    pieces = list(coverage_chunks(corners, width, height))
    if not pieces:
        empty = torch.zeros(0, dtype=torch.int64, device=corners.device)
        return Coverage(empty, empty.clone(), torch.zeros(0, 3, device=corners.device))
    return Coverage(torch.cat([p.face for p in pieces]), torch.cat([p.pixel for p in pieces]),
                    torch.cat([p.barycentric for p in pieces]))


def winners(corners: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """The triangle each pixel centre belongs to, the highest index where several cover it.

    Args:
        corners: ``(F, 3, 2)`` triangle corners in pixel units.
        width: Grid width in pixels.
        height: Grid height in pixels.

    Returns:
        ``(height * width,)`` int32 triangle index, ``-1`` where nothing covers the centre.
    """
    _check_grid(width, height)
    owner = torch.full((width * height,), -1, dtype=torch.int32, device=corners.device)
    for piece in coverage_chunks(corners, width, height):
        owner.scatter_reduce_(0, piece.pixel, piece.face.to(torch.int32), reduce="amax")
    return owner


def edge_pixels(corners: torch.Tensor, width: int, height: int, spacing: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixels the edges of triangles pass through, sampled at most ``spacing`` pixels apart.

    Args:
        corners: ``(F, 3, 2)`` triangle corners in pixel units.
        width: Grid width in pixels.
        height: Grid height in pixels.
        spacing: Largest step between samples along an edge, in pixels.

    Returns:
        ``(face, pixel)`` int64, with repeats.
    """
    _check_grid(width, height)
    face_count = int(corners.shape[0])
    if face_count == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=corners.device)
        return empty, empty.clone()
    start = corners.float()
    end = start[:, [1, 2, 0]]
    finite = torch.isfinite(start).all(dim=2) & torch.isfinite(end).all(dim=2)
    length = (end - start).norm(dim=2)
    steps = torch.where(finite, torch.ceil(length / spacing).clamp(1, 4 * (width + height)), 0).long() + finite.long()
    flat_steps = steps.reshape(-1)
    owner = torch.repeat_interleave(torch.arange(face_count * 3, device=corners.device), flat_steps)
    offsets = torch.cumsum(flat_steps, 0) - flat_steps
    index = torch.arange(owner.shape[0], device=corners.device) - torch.repeat_interleave(offsets, flat_steps)
    fraction = index.float() / (flat_steps[owner] - 1).clamp_min(1).float()
    flat_start = start.reshape(-1, 2)[owner]
    flat_end = end.reshape(-1, 2)[owner]
    points = flat_start + (flat_end - flat_start) * fraction[:, None]
    column = torch.floor(points[:, 0]).long().clamp(0, width - 1)
    row = torch.floor(points[:, 1]).long().clamp(0, height - 1)
    return owner // 3, row * width + column


def _wrap(index: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    """Texel indices folded into ``[0, size)`` by a glTF wrap mode."""
    if mode == "clamp":
        return index.clamp(0, size - 1)
    if mode == "mirror":
        period = torch.remainder(index, 2 * size)
        return torch.where(period < size, period, 2 * size - 1 - period)
    return torch.remainder(index, size)


def sample_texture(image: torch.Tensor, uvs: torch.Tensor, wrap_s: str = "repeat", wrap_t: str = "repeat") -> torch.Tensor:
    """Bilinear samples of an image at texture coordinates.

    Args:
        image: ``(H, W, C)`` float image.
        uvs: ``(K, 2)`` texture coordinates; non-finite ones read texel 0.
        wrap_s: ``"repeat"``, ``"clamp"`` or ``"mirror"`` along ``u``.
        wrap_t: The same along ``v``.

    Returns:
        ``(K, C)`` float32 on the coordinates' device.
    """
    height, width, channels = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    flat = image.reshape(-1, channels).to(uvs.device, torch.float32)
    out = torch.empty(uvs.shape[0], channels, dtype=torch.float32, device=uvs.device)
    for begin in range(0, int(uvs.shape[0]), SAMPLE_CHUNK):
        part = uvs[begin:begin + SAMPLE_CHUNK].double().nan_to_num(0.0, posinf=0.0, neginf=0.0)
        x = part[:, 0] * width - 0.5
        y = part[:, 1] * height - 0.5
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        fx = (x - x0).float()[:, None]
        fy = (y - y0).float()[:, None]
        x0 = x0.clamp(-(1 << 40), 1 << 40).long()
        y0 = y0.clamp(-(1 << 40), 1 << 40).long()
        xa, xb = _wrap(x0, width, wrap_s), _wrap(x0 + 1, width, wrap_s)
        ya, yb = _wrap(y0, height, wrap_t), _wrap(y0 + 1, height, wrap_t)
        top = flat[ya * width + xa] * (1 - fx) + flat[ya * width + xb] * fx
        bottom = flat[yb * width + xa] * (1 - fx) + flat[yb * width + xb] * fx
        out[begin:begin + part.shape[0]] = top * (1 - fy) + bottom * fy
    return out


def nearest_filled(filled: torch.Tensor, radius: int) -> torch.Tensor:
    """For each pixel, the nearest filled pixel within a radius, by jump flooding.

    Args:
        filled: ``(H, W)`` bool, the pixels that hold values.
        radius: Largest distance searched, in pixels.

    Returns:
        ``(H, W)`` int64 flat index of the nearest filled pixel, ``-1`` past the radius.
    """
    height, width = int(filled.shape[0]), int(filled.shape[1])
    device = filled.device
    index = torch.arange(height * width, device=device).reshape(height, width)
    if radius <= 0 or not bool(filled.any()):
        return torch.where(filled, index, torch.full_like(index, -1))
    far = 1 << 20
    rows = torch.arange(height, device=device, dtype=torch.int32)[:, None].expand(height, width)
    cols = torch.arange(width, device=device, dtype=torch.int32)[None, :].expand(height, width)
    seed_row = torch.where(filled, rows, torch.full_like(rows, far))
    seed_col = torch.where(filled, cols, torch.full_like(cols, far))
    best = (seed_row - rows).float() ** 2 + (seed_col - cols).float() ** 2
    step = 1 << max(0, (int(radius) - 1).bit_length())
    while step >= 1:
        padded_row = torch.nn.functional.pad(seed_row[None], (step, step, step, step), value=far)[0]
        padded_col = torch.nn.functional.pad(seed_col[None], (step, step, step, step), value=far)[0]
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                candidate_row = padded_row[step + dy:step + dy + height, step + dx:step + dx + width]
                candidate_col = padded_col[step + dy:step + dy + height, step + dx:step + dx + width]
                distance = (candidate_row - rows).float() ** 2 + (candidate_col - cols).float() ** 2
                closer = distance < best
                seed_row = torch.where(closer, candidate_row, seed_row)
                seed_col = torch.where(closer, candidate_col, seed_col)
                best = torch.where(closer, distance, best)
        step //= 2
    found = (best <= float(radius) * float(radius)) & (seed_row < far)
    return torch.where(found, seed_row.long() * width + seed_col.long(), torch.full_like(index, -1))


def fill_gaps(image: torch.Tensor, nearest: torch.Tensor, background: torch.Tensor | None = None) -> torch.Tensor:
    """Copy each empty pixel's nearest filled value into it.

    Args:
        image: ``(H, W, C)`` values, read only where ``nearest`` points.
        nearest: ``(H, W)`` int64 from `nearest_filled`.
        background: ``(C,)`` value for pixels with no filled pixel in reach, or None to keep
            the image's own value there.

    Returns:
        A new ``(H, W, C)`` image.
    """
    height, width, channels = image.shape
    flat = image.reshape(-1, channels)
    reach = nearest.reshape(-1)
    found = reach >= 0
    out = flat.clone()
    out[found] = flat[reach[found]]
    if background is not None:
        out[~found] = background.to(out.dtype).to(out.device)
    return out.reshape(height, width, channels)
