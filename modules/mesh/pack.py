"""Packing flattened charts into one square texture atlas at one texel density."""
# Charts are rasterised into bitmaps, grown by half the padding, turned to their tightest bounding
# box and dropped into the atlas along their bitmap profiles in one of four quarter turns, Tetris
# fashion, as Levy et al. 2002 describe.

from __future__ import annotations

__all__ = ["Packing", "Bitmaps", "chart_bitmaps", "orient_charts", "pack_charts", "self_overlaps"]

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import raster

#: Turns tried when looking for a chart's tightest bounding box.
ORIENT_STEPS = 36

#: Share of the atlas the first density guess expects charts to fill.
FIRST_FILL = 0.6

#: Density trials when fitting the charts to the atlas.
FIT_TRIALS = 6

#: Stop refining the density once the bounds are this close.
FIT_RATIO = 1.03

#: Stop refining the density once a packing that fits reaches this share of the atlas height.
FIT_USED = 0.97

#: Widest mask tried at every column; wider ones are tried at every second or fourth, then refined.
STRIDE_WIDTH = 32

#: Longest side, in grid pixels, of a chart dropped as its whole box.
SMALL_MASK = 10

#: Share of its box a chart's bitmap fills above which only quarter turns that swap its sides are tried.
BOXY = 0.85

#: Strictest barycentric weight that still counts a pixel as inside, when looking for overlaps.
INSIDE = 1e-3

#: Widest grid charts are placed on by default; larger atlases place in blocks of several texels.
PACK_GRID = 1024


@dataclass
class Bitmaps:
    """Each chart drawn as a bitmap at one density.

    Attributes:
        masks: One ``(height, width)`` bool array per chart, grown by the margin.
        origin: ``(C, 2)`` float64 chart coordinate at each bitmap's left top corner.
        scale: Pixels per chart unit.
        margin: Pixels the masks were grown by.
    """

    masks: list[np.ndarray]
    origin: torch.Tensor
    scale: float
    margin: int


@dataclass
class Packing:
    """Where each chart landed in the atlas.

    Attributes:
        uvs: ``(N, 2)`` float32 atlas coordinates of every vertex, ``v`` down the image.
        scale: Pixels per chart unit.
        used_height: Rows of the atlas the charts reach.
        turns: ``(C,)`` quarter turns of each chart.
    """

    uvs: torch.Tensor
    scale: float
    used_height: int
    turns: torch.Tensor


def _chart_bounds(uvs: torch.Tensor, vertex_chart: torch.Tensor, chart_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(low, high)`` corners of each chart, ``(C, 2)``."""
    low = torch.full((chart_count, 2), math.inf, dtype=uvs.dtype, device=uvs.device)
    high = torch.full((chart_count, 2), -math.inf, dtype=uvs.dtype, device=uvs.device)
    index = vertex_chart[:, None].expand(-1, 2)
    low.scatter_reduce_(0, index, uvs, reduce="amin")
    high.scatter_reduce_(0, index, uvs, reduce="amax")
    low = torch.where(torch.isfinite(low), low, torch.zeros_like(low))
    high = torch.where(torch.isfinite(high), high, low)
    return low, high


def orient_charts(uvs: torch.Tensor, vertex_chart: torch.Tensor, chart_count: int) -> torch.Tensor:
    """Turn each chart to its tightest bounding box, longer side along ``u``.

    Args:
        uvs: ``(N, 2)`` chart coordinates.
        vertex_chart: ``(N,)`` chart of each vertex.
        chart_count: Number of charts.

    Returns:
        ``(N, 2)`` turned coordinates with each chart's box starting at zero.
    """
    work = uvs.double()
    best_area = torch.full((chart_count,), math.inf, dtype=torch.float64, device=uvs.device)
    best_angle = torch.zeros(chart_count, dtype=torch.float64, device=uvs.device)
    for step in range(ORIENT_STEPS):
        angle = step * (math.pi / 2) / ORIENT_STEPS
        cos, sin = math.cos(angle), math.sin(angle)
        turned = torch.stack([work[:, 0] * cos - work[:, 1] * sin, work[:, 0] * sin + work[:, 1] * cos], dim=1)
        low, high = _chart_bounds(turned, vertex_chart, chart_count)
        area = (high - low).prod(dim=1)
        better = area < best_area * (1 - 1e-9)
        best_area = torch.where(better, area, best_area)
        best_angle = torch.where(better, torch.full_like(best_angle, angle), best_angle)
    angle = best_angle[vertex_chart]
    cos, sin = torch.cos(angle), torch.sin(angle)
    turned = torch.stack([work[:, 0] * cos - work[:, 1] * sin, work[:, 0] * sin + work[:, 1] * cos], dim=1)
    low, high = _chart_bounds(turned, vertex_chart, chart_count)
    upright = (high[:, 1] - low[:, 1]) > (high[:, 0] - low[:, 0])
    flip = upright[vertex_chart]
    turned = torch.where(flip[:, None], torch.stack([turned[:, 1], -turned[:, 0]], dim=1), turned)
    low, _ = _chart_bounds(turned, vertex_chart, chart_count)
    return (turned - low[vertex_chart]).to(uvs.dtype)


def _shelves(widths: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Lay boxes out on shelves, tallest first, for a scratch canvas.

    Returns:
        ``(x, y, width, height)``: each box's left top corner and the canvas size.
    """
    count = len(widths)
    area = float((widths.astype(np.float64) * heights).sum())
    canvas = int(max(int(widths.max()) if count else 1, math.ceil(math.sqrt(max(area, 1.0)) * 1.1)))
    order = np.argsort(-heights, kind="stable")
    x = np.zeros(count, dtype=np.int64)
    y = np.zeros(count, dtype=np.int64)
    cursor_x = cursor_y = shelf = 0
    for index in order:
        if cursor_x + widths[index] > canvas:
            cursor_y += shelf
            cursor_x = shelf = 0
        x[index], y[index] = cursor_x, cursor_y
        cursor_x += int(widths[index])
        shelf = max(shelf, int(heights[index]))
    return x, y, canvas, max(cursor_y + shelf, 1)


def _canvas(uvs: torch.Tensor, faces: torch.Tensor, face_chart: torch.Tensor, chart_count: int, scale: float,
            margin: int):
    """Charts drawn at one density on a scratch canvas, one box each.

    Returns:
        ``(corners, boxes, origin, x, y, width, height)``: face corners in canvas pixels, each
        box's ``(height, width)``, each chart's left top in chart units, box offsets and the
        canvas size.
    """
    vertex_chart = torch.zeros(uvs.shape[0], dtype=torch.int64, device=uvs.device)
    vertex_chart[faces.reshape(-1)] = face_chart.repeat_interleave(3)
    work = uvs.double()
    low, high = _chart_bounds(work, vertex_chart, chart_count)
    extent = ((high - low) * scale).cpu().numpy()
    widths = (np.ceil(extent[:, 0]) + 2 * margin + 2).astype(np.int64)
    heights = (np.ceil(extent[:, 1]) + 2 * margin + 2).astype(np.int64)
    x, y, width, height = _shelves(widths, heights)
    if width * height > raster.MAX_GRID_PIXELS:
        raise ValueError("the charts are too large to draw at this texture resolution; lower the resolution")
    origin = low - (margin + 1.0) / scale
    shift = torch.from_numpy(np.stack([x, y], axis=1)).to(uvs.device).double()
    local = (work - origin[vertex_chart]) * scale + shift[vertex_chart]
    corners = local[faces].float()
    return corners, np.stack([heights, widths], axis=1), origin, x, y, width, height


def chart_bitmaps(uvs: torch.Tensor, faces: torch.Tensor, face_chart: torch.Tensor, chart_count: int,
                  scale: float, margin: int) -> Bitmaps:
    """Every chart drawn conservatively at one density and grown by a margin.

    Args:
        uvs: ``(N, 2)`` chart coordinates.
        faces: ``(F, 3)`` vertex indices.
        face_chart: ``(F,)`` chart of each face.
        chart_count: Number of charts.
        scale: Pixels per chart unit.
        margin: Pixels each chart grows by.

    Returns:
        The `Bitmaps`.
    """
    corners, boxes, origin, x, y, width, height = _canvas(uvs, faces, face_chart, chart_count, scale, margin)
    occupied = torch.zeros(height * width, dtype=torch.bool, device=uvs.device)
    for piece in raster.coverage_chunks(corners, width, height):
        occupied[piece.pixel] = True
    _, edge = raster.edge_pixels(corners, width, height)
    occupied[edge] = True
    grid = occupied.reshape(height, width)
    if margin > 0:
        grid = torch.nn.functional.max_pool2d(grid[None, None].float(), 2 * margin + 1, stride=1,
                                              padding=margin)[0, 0] > 0
    canvas = grid.cpu().numpy()
    masks = [canvas[y[c]:y[c] + boxes[c, 0], x[c]:x[c] + boxes[c, 1]].copy() for c in range(chart_count)]
    return Bitmaps(masks, origin, float(scale), int(margin))


def self_overlaps(uvs: torch.Tensor, faces: torch.Tensor, face_chart: torch.Tensor, chart_count: int,
                  scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixels each chart covers more than once, drawn at one density, and the faces covering them.

    Args:
        uvs: ``(N, 2)`` chart coordinates.
        faces: ``(F, 3)`` vertex indices.
        face_chart: ``(F,)`` chart of each face.
        chart_count: Number of charts.
        scale: Pixels per chart unit.

    Returns:
        ``(counts, involved)``: ``(C,)`` int64 overlapping pixel counts, and ``(F,)`` bool faces
        covering an overlapping pixel.
    """
    corners, _, _, _, _, width, height = _canvas(uvs, faces, face_chart, chart_count, scale, 0)
    hits = torch.zeros(height * width, dtype=torch.int32, device=uvs.device)
    for piece in raster.coverage_chunks(corners, width, height):
        inside = piece.barycentric.amin(dim=1) > INSIDE
        hits.index_add_(0, piece.pixel[inside], torch.ones(int(inside.sum()), dtype=torch.int32, device=uvs.device))
    doubled = hits > 1
    counts = torch.zeros(chart_count, dtype=torch.int64, device=uvs.device)
    involved = torch.zeros(faces.shape[0], dtype=torch.bool, device=uvs.device)
    if bool(doubled.any()):
        for piece in raster.coverage_chunks(corners, width, height):
            inside = piece.barycentric.amin(dim=1) > INSIDE
            over = inside & doubled[piece.pixel]
            involved[piece.face[over]] = True
            counts.index_add_(0, face_chart[piece.face[over]], torch.ones(int(over.sum()), dtype=torch.int64,
                                                                           device=uvs.device))
    return counts, involved


def _profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First occupied row, one past the last, and occupancy, per column of a mask."""
    filled = mask.any(axis=0)
    rows = mask.shape[0]
    top = np.where(filled, mask.argmax(axis=0), rows)
    bottom = np.where(filled, rows - mask[::-1].argmax(axis=0), 0)
    return top.astype(np.int64), bottom.astype(np.int64), filled


def _turn_profiles(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Column profiles of a mask in each quarter turn worth trying, as `_profiles` gives them."""
    rows = mask.shape[0]
    top, bottom, filled = _profiles(mask)
    profiles = [(top, bottom, filled)]
    boxy = mask.size and int(mask.sum()) >= BOXY * mask.size
    if boxy and mask.shape[0] == mask.shape[1]:
        return profiles
    turned = np.rot90(mask)
    side_top, side_bottom, side_filled = _profiles(turned)
    profiles.append((side_top, side_bottom, side_filled))
    if boxy:
        return profiles
    # A half turn reverses the columns and flips each one top to bottom.
    profiles.append((np.where(filled[::-1], rows - bottom[::-1], rows), np.where(filled[::-1], rows - top[::-1], 0),
                     filled[::-1]))
    columns = mask.shape[1]
    profiles.append((np.where(side_filled[::-1], columns - side_bottom[::-1], columns),
                     np.where(side_filled[::-1], columns - side_top[::-1], 0), side_filled[::-1]))
    return profiles


def _drop(skyline: np.ndarray, profiles: list) -> tuple[int, int, int] | None:
    """Best turn, column and row to drop a mask at, or None when no turn fits across.

    Args:
        skyline: ``(side,)`` first free row of each atlas column.
        profiles: `_turn_profiles` of the mask.
    """
    side = skyline.shape[0]
    widths = np.array([profile[0].shape[0] for profile in profiles], dtype=np.int64)
    if int(widths.min()) > side:
        return None
    turns = len(profiles)
    width = int(min(widths.max(), side))
    # Unfilled columns sit far below the atlas so they never decide the row.
    top = np.full((turns, width), 1 << 30, dtype=np.int32)
    filled = np.zeros((turns, width), dtype=np.int32)
    reach = np.zeros(turns, dtype=np.int64)
    filled_top = np.zeros(turns, dtype=np.int64)
    for turn, (turn_top, turn_bottom, turn_filled) in enumerate(profiles):
        used = min(turn_top.shape[0], width)
        top[turn, :used] = np.where(turn_filled[:used], turn_top[:used], 1 << 30)
        filled[turn, :used] = turn_filled[:used]
        reach[turn] = int(turn_bottom.max()) if turn_bottom.size else 0
        filled_top[turn] = int(turn_top[turn_filled].sum())
    padded = np.concatenate([skyline.astype(np.int32), np.zeros(width - 1, dtype=np.int32)])
    window = np.lib.stride_tricks.sliding_window_view(padded, width)
    # Gap under the mask: filled columns times the row, plus their tops, minus the skyline beneath.
    beneath = np.stack([np.correlate(padded.astype(np.int64), filled[turn].astype(np.int64), mode="valid")
                        for turn in range(turns)])
    counts = filled.sum(axis=1, dtype=np.int64)

    def costs(columns: np.ndarray, chosen: np.ndarray):
        """Row and cost of dropping each chosen turn at each of the columns."""
        row = np.maximum((window[columns][None] - top[chosen, None, :]).max(axis=2), 0).astype(np.int64)
        gap = counts[chosen, None] * row + filled_top[chosen, None] - beneath[chosen][:, columns]
        cost = ((row + reach[chosen, None]) * widths[chosen, None] + gap).astype(np.float64)
        cost[columns[None, :] > side - widths[chosen, None]] = np.inf
        return row, cost

    stride = 1 if width <= STRIDE_WIDTH else (2 if width <= 2 * STRIDE_WIDTH else 4)
    every = np.arange(turns)
    row, cost = costs(np.arange(0, side, stride), every)
    best = int(np.argmin(cost))
    turn, slot = divmod(best, cost.shape[1])
    if not np.isfinite(cost[turn, slot]):
        return None
    column = slot * stride
    if stride > 1:
        # The coarse best is refined over the columns between its neighbours.
        nearby = np.arange(max(0, column - stride + 1), min(side, column + stride))
        near_row, near_cost = costs(nearby, np.array([turn]))
        pick = int(np.argmin(near_cost[0]))
        return turn, int(nearby[pick]), int(near_row[0, pick])
    return turn, column, int(row[turn, slot])


def _drop_box(skyline: np.ndarray, height: int, width: int) -> tuple[int, int, int] | None:
    """Best turn, column and row to drop a mask's whole box at, upright or on its side."""
    side = skyline.shape[0]
    best = None
    for turn, (tall, wide) in enumerate(((height, width), (width, height))):
        if wide > side or (turn == 1 and tall == wide):
            continue
        window = np.lib.stride_tricks.sliding_window_view(skyline, wide)
        row = window.max(axis=1)
        cost = (row + tall) * wide + wide * row - window.sum(axis=1)
        column = int(np.argmin(cost))
        if best is None or cost[column] < best[0]:
            best = (int(cost[column]), turn, column, int(row[column]))
    return best[1:] if best is not None else None


def _skyline_pack(masks: list[np.ndarray], side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop masks into an atlas ``side`` pixels wide, largest first.

    Returns:
        ``(x, y, turns, height)`` per mask and the rows used; height is past ``side * 4`` when
        a mask does not fit across.
    """
    count = len(masks)
    skyline = np.zeros(side, dtype=np.int64)
    x = np.zeros(count, dtype=np.int64)
    y = np.zeros(count, dtype=np.int64)
    turns = np.zeros(count, dtype=np.int64)
    order = np.argsort([-(int(m.sum()) + 0.1 * max(m.shape)) for m in masks], kind="stable")
    for index in order:
        mask = masks[index]
        if max(mask.shape) <= SMALL_MASK:
            found = _drop_box(skyline, mask.shape[0], mask.shape[1])
            if found is None:
                return x, y, turns, side * 8
            turn, column, row = found
            height, width = mask.shape if turn == 0 else mask.shape[::-1]
            skyline[column:column + width] = row + height
            x[index], y[index], turns[index] = column, row, turn
            continue
        profiles = _turn_profiles(mask)
        found = _drop(skyline, profiles)
        if found is None:
            return x, y, turns, side * 8
        turn, column, row = found
        _, bottom, filled = profiles[turn]
        span = skyline[column:column + bottom.shape[0]]
        skyline[column:column + bottom.shape[0]] = np.where(filled, row + bottom, span)
        x[index], y[index], turns[index] = column, row, turn
    return x, y, turns, int(skyline.max()) if count else 0


def _place(uvs: torch.Tensor, vertex_chart: torch.Tensor, bitmaps: Bitmaps, x: np.ndarray, y: np.ndarray,
           turns: np.ndarray, side: int) -> torch.Tensor:
    """Atlas coordinates of every vertex once its chart is turned and placed."""
    device = uvs.device
    local = (uvs.double() - bitmaps.origin[vertex_chart]) * bitmaps.scale
    sizes = torch.tensor([m.shape for m in bitmaps.masks], dtype=torch.float64, device=device).reshape(-1, 2)
    height = sizes[vertex_chart, 0]
    width = sizes[vertex_chart, 1]
    turn = torch.from_numpy(turns).to(device)[vertex_chart]
    px, py = local[:, 0], local[:, 1]
    turned_x = torch.where(turn == 1, py, torch.where(turn == 2, width - px, torch.where(turn == 3, height - py, px)))
    turned_y = torch.where(turn == 1, width - px, torch.where(turn == 2, height - py, torch.where(turn == 3, px, py)))
    offset = torch.from_numpy(np.stack([x, y], axis=1)).to(device).double()[vertex_chart]
    atlas = torch.stack([turned_x, turned_y], dim=1) + offset
    return (atlas / side).float()


def pack_charts(uvs: torch.Tensor, faces: torch.Tensor, face_chart: torch.Tensor, chart_count: int,
                side: int, padding: int, grid: int = PACK_GRID) -> Packing:
    """Place every chart in a square atlas at the largest density that fits.

    Args:
        uvs: ``(N, 2)`` chart coordinates at true scale, every vertex in one chart.
        faces: ``(F, 3)`` vertex indices.
        face_chart: ``(F,)`` chart of each face, ``0`` to ``chart_count - 1``.
        chart_count: Number of charts.
        side: Atlas width and height in pixels.
        padding: Pixels kept free between charts.
        grid: Widest grid charts are placed on; a larger atlas places in blocks of texels.

    Returns:
        The `Packing`.

    Raises:
        ValueError: The charts have no area.
    """
    device = uvs.device
    vertex_chart = torch.zeros(uvs.shape[0], dtype=torch.int64, device=device)
    vertex_chart[faces.reshape(-1)] = face_chart.repeat_interleave(3)
    oriented = orient_charts(uvs, vertex_chart, chart_count)
    corners = oriented[faces].double()
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = float((first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]).abs().sum() / 2)
    if not area > 0:
        raise ValueError("the charts have no area to pack; the mesh may be flat or degenerate")
    block = max(1, math.ceil(side / max(int(grid), 64)))
    grid = side // block
    margin = math.ceil(((int(padding) + 1) // 2) / block)
    scale = math.sqrt(FIRST_FILL * grid * grid / area)
    low = high = None
    best = None
    for _ in range(FIT_TRIALS):
        bitmaps = chart_bitmaps(oriented, faces, face_chart, chart_count, scale, margin)
        x, y, turns, used = _skyline_pack(bitmaps.masks, grid)
        if used <= grid:
            best = (bitmaps, x, y, turns, used)
            low = scale
            if used >= FIT_USED * grid:
                break
        else:
            high = scale
        if low is not None and high is not None:
            if high / low < FIT_RATIO:
                break
            scale = math.sqrt(low * high)
        elif high is None:
            scale *= min(1.5, math.sqrt(grid / max(used, 1)) * 1.02)
        else:
            scale *= max(0.4, math.sqrt(grid / max(used, 1)) * 0.98)
    for _ in range(FIT_TRIALS * 3):
        if best is not None:
            break
        scale *= 0.7
        bitmaps = chart_bitmaps(oriented, faces, face_chart, chart_count, scale, margin)
        x, y, turns, used = _skyline_pack(bitmaps.masks, grid)
        if used <= grid:
            best = (bitmaps, x, y, turns, used)
    if best is None:
        raise ValueError(f"{chart_count} charts do not fit a {side} pixel atlas with {padding} pixels of "
                         "padding; raise the resolution, lower the padding or raise chart_angle")
    bitmaps, x, y, turns, used = best
    atlas = _place(oriented, vertex_chart, bitmaps, x, y, turns, grid) * (grid * block / side)
    return Packing(atlas, bitmaps.scale * block, int(used) * block, torch.from_numpy(turns))
