"""A Rec.709 vectorscope and ARRI exposure false colour, drawn as pictures.

Pictures are ``(height, width, 3)`` float codes. Y', Cb and Cr are BT.709 on the codes; signal
levels are Y' in percent, 0 to 100 for 0 to 1.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..data import paths

__all__ = [
    "FALSE_COLOUR_BANDS",
    "FULL_SCALE",
    "SKIN_ANGLE",
    "TARGETS",
    "TARGET_LEVELS",
    "draw_vectorscope",
    "false_colour",
    "luma_cbcr",
    "plot_position",
    "target_points",
    "vectorscope_density",
]

#: BT.709 luma weights for R', G' and B'.
LUMA = (0.2126, 0.7152, 0.0722)

#: Divisors taking B' - Y' and R' - Y' to Cb and Cr.
CB_DIVISOR = 2.0 * (1.0 - LUMA[2])
CR_DIVISOR = 2.0 * (1.0 - LUMA[0])

#: The six bar colours a vectorscope marks, as ``(label, (R', G', B'))`` at full amplitude.
TARGETS = (
    ("R", (1.0, 0.0, 0.0)),
    ("Mg", (1.0, 0.0, 1.0)),
    ("B", (0.0, 0.0, 1.0)),
    ("Cy", (0.0, 1.0, 1.0)),
    ("G", (0.0, 1.0, 0.0)),
    ("Yl", (1.0, 1.0, 0.0)),
)

#: Bar amplitudes a target is drawn for: a box at 75% and a ring at 100%.
TARGET_LEVELS = (0.75, 1.0)

#: Angle of the skin-tone line, in degrees anticlockwise from the +Cb axis.
SKIN_ANGLE = 123.0

#: Chroma, the length of (Cb, Cr), at the outer circle.
FULL_SCALE = 0.6

#: Outer circle radius as a share of half the scope's side.
CIRCLE_SHARE = 0.92

#: Factor the graticule is drawn larger by before it is reduced to the scope's size.
SUPERSAMPLE = 4

#: Half the side of a 75% target box, in chroma.
BOX_HALF = 0.018

#: Radius of a 100% target ring, in chroma.
RING_RADIUS = 0.011

#: Graticule colours, as 8-bit RGB.
GRID_COLOUR = (78, 78, 78)
AXIS_COLOUR = (58, 58, 58)
TARGET_COLOUR = (214, 170, 64)
SKIN_COLOUR = (232, 150, 112)
LABEL_TEXT = (232, 232, 232)
LABEL_PLATE = (14, 14, 14, 215)

#: Bundled face the graticule's labels are written in.
LABEL_FONT = "DejaVu Sans Bold"

#: Colour of a fully dense trace, as codes.
TRACE_COLOUR = (0.82, 1.0, 0.82)

#: ARRI false colour, as ``(name, low, high, (R', G', B'))`` with levels in percent of Y'.
#: The first band holds everything below its high edge and the last everything from its low
#: edge up; a level within :data:`EDGE_SLACK` under an edge counts as on it.
FALSE_COLOUR_BANDS = (
    ("black clip", 0.0, 2.5, (0.58, 0.0, 0.83)),
    ("just above black", 2.5, 4.0, (0.0, 0.35, 1.0)),
    ("18% grey", 38.0, 42.0, (0.0, 0.78, 0.0)),
    ("one stop over grey", 52.0, 56.0, (1.0, 0.41, 0.71)),
    ("just below white", 97.0, 99.0, (1.0, 0.9, 0.0)),
    ("white clip", 99.0, 100.0, (1.0, 0.0, 0.0)),
)

#: Percent of Y' an edge reaches down by, which absorbs float32 rounding of a level on it.
EDGE_SLACK = 1e-4


def luma_cbcr(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BT.709 Y', Cb and Cr of picture codes.

    Args:
        rgb: ``(..., 3)`` codes.

    Returns:
        ``(luma, cb, cr)``, each ``(...)`` float64.
    """
    codes = rgb[..., :3].to(torch.float64)
    red, green, blue = codes.unbind(-1)
    luma = LUMA[0] * red + LUMA[1] * green + LUMA[2] * blue
    return luma, (blue - luma) / CB_DIVISOR, (red - luma) / CR_DIVISOR


def target_points() -> list[tuple[str, float, float, float]]:
    """Every target the graticule marks.

    Returns:
        ``(label, level, cb, cr)`` per target, at each of :data:`TARGET_LEVELS`.
    """
    points = []
    for label, colour in TARGETS:
        _luma, cb, cr = luma_cbcr(torch.tensor(colour, dtype=torch.float64))
        for level in TARGET_LEVELS:
            points.append((label, level, float(cb) * level, float(cr) * level))
    return points


def _scale(size: int) -> float:
    """Pixels per unit of chroma on a scope of this side."""
    return (size / 2.0) * CIRCLE_SHARE / FULL_SCALE


def plot_position(cb, cr, size: int):
    """Where a Cb, Cr pair lands on a scope, in pixel units.

    Args:
        cb: Cb, a float or a tensor.
        cr: Cr, the same shape.
        size: The scope's side in pixels.

    Returns:
        ``(x, y)``, pixel ``i`` spanning ``i`` to ``i + 1``, +Cr pointing up.
    """
    scale = _scale(size)
    return size / 2.0 + cb * scale, size / 2.0 - cr * scale


def vectorscope_density(rgb: torch.Tensor, size: int) -> torch.Tensor:
    """Count where each pixel's Cb, Cr lands, split between the four nearest cells.

    Args:
        rgb: ``(height, width, 3)`` codes.
        size: The scope's side in pixels.

    Returns:
        ``(size, size)`` float64 counts. Pixels landing off the scope are dropped.
    """
    _luma, cb, cr = luma_cbcr(rgb.reshape(-1, rgb.shape[-1]))
    x, y = plot_position(cb, cr, size)
    x = x - 0.5
    y = y - 0.5
    left = torch.floor(x)
    top = torch.floor(y)
    fx = x - left
    fy = y - top
    left = left.to(torch.int64)
    top = top.to(torch.int64)
    density = torch.zeros(size * size, dtype=torch.float64, device=rgb.device)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy), (1, 1, fx * fy),
    ):
        column = left + dx
        row = top + dy
        inside = (column >= 0) & (column < size) & (row >= 0) & (row < size)
        density.index_add_(0, (row * size + column)[inside], weight[inside])
    return density.view(size, size)


def _font(size: int):
    """The face labels are drawn in, at a pixel size."""
    try:
        return ImageFont.truetype(str(paths.font_file(LABEL_FONT)), size=size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()


def _label(layer: ImageDraw.ImageDraw, text: str, x: float, y: float, font, bounds: int) -> None:
    """Write a label centred on a point, on a plate, kept inside the picture."""
    left, top, right, bottom = layer.textbbox((0, 0), text, font=font)
    width, height = right - left, bottom - top
    pad = max(2, height // 4)
    box_x = min(max(x - width / 2 - pad, 0), bounds - width - 2 * pad)
    box_y = min(max(y - height / 2 - pad, 0), bounds - height - 2 * pad)
    layer.rounded_rectangle(
        (box_x, box_y, box_x + width + 2 * pad, box_y + height + 2 * pad),
        radius=pad, fill=LABEL_PLATE,
    )
    layer.text((box_x + pad - left, box_y + pad - top), text, font=font, fill=LABEL_TEXT + (255,))


def _graticule(size: int) -> tuple[np.ndarray, np.ndarray]:
    """The scope's grid, targets and skin line, and its labels.

    Returns:
        ``(grid, labels)``: ``(size, size, 3)`` and ``(size, size, 4)`` float arrays in 0 to 1.
    """
    big = size * SUPERSAMPLE
    scale = _scale(size) * SUPERSAMPLE
    centre = big / 2.0
    line = SUPERSAMPLE
    grid = Image.new("RGB", (big, big), (0, 0, 0))
    pen = ImageDraw.Draw(grid)

    def at(cb: float, cr: float) -> tuple[float, float]:
        return centre + cb * scale, centre - cr * scale

    radius = FULL_SCALE * scale
    pen.line((centre - radius, centre, centre + radius, centre), fill=AXIS_COLOUR, width=line)
    pen.line((centre, centre - radius, centre, centre + radius), fill=AXIS_COLOUR, width=line)
    pen.ellipse((centre - radius, centre - radius, centre + radius, centre + radius),
                outline=GRID_COLOUR, width=line)
    for degree in range(0, 360, 2):
        reach = 0.035 if degree % 10 == 0 else 0.015
        angle = math.radians(degree)
        inner = radius * (1.0 - reach)
        pen.line(
            (centre + inner * math.cos(angle), centre - inner * math.sin(angle),
             centre + radius * math.cos(angle), centre - radius * math.sin(angle)),
            fill=GRID_COLOUR, width=line,
        )

    full = [(cb, cr) for _label_text, level, cb, cr in target_points() if level == 1.0]
    pen.polygon([at(cb, cr) for cb, cr in full], outline=GRID_COLOUR, width=line)

    skin = math.radians(SKIN_ANGLE)
    pen.line((centre, centre, centre + radius * math.cos(skin), centre - radius * math.sin(skin)),
             fill=SKIN_COLOUR, width=line)

    half_box = BOX_HALF * scale
    ring = RING_RADIUS * scale
    for _label_text, level, cb, cr in target_points():
        x, y = at(cb, cr)
        if level == 1.0:
            pen.ellipse((x - ring, y - ring, x + ring, y + ring), outline=TARGET_COLOUR, width=line)
        else:
            pen.rectangle((x - half_box, y - half_box, x + half_box, y + half_box),
                          outline=TARGET_COLOUR, width=line)

    labels = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    writer = ImageDraw.Draw(labels)
    font = _font(max(9, round(size / 40)) * SUPERSAMPLE)
    for label_text, level, cb, cr in target_points():
        if level != 1.0:
            continue
        length = math.hypot(cb, cr)
        reach = (length + 0.055) / length
        x, y = at(cb * reach, cr * reach)
        _label(writer, label_text, x, y, font, big)
    # Beside the skin line, on the side away from the red target.
    along = FULL_SCALE * 0.9
    aside = math.radians(SKIN_ANGLE + 90.0)
    skin_x, skin_y = at(
        math.cos(skin) * along + math.cos(aside) * 0.05,
        math.sin(skin) * along + math.sin(aside) * 0.05,
    )
    _label(writer, "skin", skin_x, skin_y, font, big)

    grid = grid.resize((size, size), Image.Resampling.BOX)
    labels = labels.resize((size, size), Image.Resampling.BOX)
    return (
        np.asarray(grid, dtype=np.float32) / 255.0,
        np.asarray(labels, dtype=np.float32) / 255.0,
    )


def draw_vectorscope(density: torch.Tensor) -> torch.Tensor:
    """Draw counts from :func:`vectorscope_density` under the graticule.

    Args:
        density: ``(size, size)`` counts.

    Returns:
        ``(size, size, 3)`` float32 codes in 0 to 1. The trace's brightness is the log of each
        cell's count over the log of the busiest cell's.
    """
    size = int(density.shape[0])
    grid, labels = _graticule(size)
    counts = density.detach().to("cpu", torch.float64).numpy()
    peak = float(counts.max())
    level = np.log1p(counts) / math.log1p(peak) if peak > 0 else np.zeros_like(counts)
    level = level.astype(np.float32)[..., None]
    trace = np.asarray(TRACE_COLOUR, dtype=np.float32)
    picture = grid * (1.0 - level) + trace * level
    alpha = labels[..., 3:4]
    picture = picture * (1.0 - alpha) + labels[..., :3] * alpha
    return torch.from_numpy(np.clip(picture, 0.0, 1.0).astype(np.float32))


def false_colour(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Paint each pixel by the ARRI band its Y' falls in, and the rest in grey.

    Args:
        rgb: ``(..., 3)`` codes.

    Returns:
        ``(picture, band)``: ``(..., 3)`` float32 codes, and ``(...)`` int64 holding each
        pixel's index into :data:`FALSE_COLOUR_BANDS`, or -1 for the grey rest.
    """
    luma, _cb, _cr = luma_cbcr(rgb)
    level = luma * 100.0 + EDGE_SLACK
    band = torch.full(level.shape, -1, dtype=torch.int64, device=rgb.device)
    last = len(FALSE_COLOUR_BANDS) - 1
    for index, (_name, low, high, _colour) in enumerate(FALSE_COLOUR_BANDS):
        above = torch.ones_like(level, dtype=torch.bool) if index == 0 else level >= low
        below = torch.ones_like(level, dtype=torch.bool) if index == last else level < high
        band = torch.where(above & below, torch.full_like(band, index), band)
    grey = luma.clamp(0.0, 1.0).to(torch.float32)
    picture = grey.unsqueeze(-1).expand(*grey.shape, 3).clone()
    for index, (_name, _low, _high, colour) in enumerate(FALSE_COLOUR_BANDS):
        paint = torch.tensor(colour, dtype=torch.float32, device=rgb.device)
        picture[band == index] = paint
    return picture, band
