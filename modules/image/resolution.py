"""Working a width and a height out of an aspect ratio and one measurement.

The ratio fixes the shape and the measurement fixes the scale, given as an edge in pixels
or as an area in megapixels.
"""

from __future__ import annotations

import math

__all__ = [
    "DRIVERS",
    "MEGAPIXEL",
    "RATIOS",
    "REALWORLD_MEGAPIXEL",
    "megapixel_unit",
    "parse_ratio",
    "resolve",
]

#: Every shape offered, width:height, sorted by the first number and then the second.
RATIOS: tuple[str, ...] = (
    "1:1", "1:1.43", "1:1.66", "1:1.85", "1:2", "1:2.2", "1:2.39", "1:3",
    "1.43:1", "1.66:1", "1.85:1",
    "2:1", "2:3", "2.2:1", "2.39:1",
    "3:1", "3:2", "3:4",
    "4:3", "4:5",
    "5:4",
    "9:16", "9:21",
    "10:16",
    "16:9", "16:10",
    "21:9",
)

#: Which measurement is being given, and so what the other one is worked out from. The area
#: comes first, then the edges set by hand.
DRIVERS: tuple[str, ...] = ("megapixels", "long edge", "short edge", "width", "height")

#: Largest side either edge may reach, which keeps a mistyped figure from asking for a
#: latent that cannot be allocated.
MAX_EDGE = 16384

#: Pixels in one megapixel, 1024 by 1024, as ComfyUI counts it.
MEGAPIXEL = 1024 * 1024

#: Pixels in one megapixel counted as a round million.
REALWORLD_MEGAPIXEL = 1_000_000


def megapixel_unit(realworld: bool) -> int:
    """Pixels in one megapixel: :data:`REALWORLD_MEGAPIXEL` when ``realworld``, else :data:`MEGAPIXEL`."""
    return REALWORLD_MEGAPIXEL if realworld else MEGAPIXEL


def parse_ratio(text: str) -> tuple[float, float]:
    """Read an aspect ratio written as two numbers.

    Args:
        text: ``"16:9"``, ``"1.85:1"``, ``"16/9"`` or a bare number meaning width over one.

    Returns:
        ``(width, height)`` as positive floats, ``(1.0, 1.0)`` for anything unreadable.
    """
    cleaned = str(text or "").strip().replace("x", ":").replace("/", ":").replace(",", ".")
    if not cleaned:
        return 1.0, 1.0
    parts = [part.strip() for part in cleaned.split(":") if part.strip()]
    if not parts:
        return 1.0, 1.0
    try:
        if len(parts) == 1:
            wide, high = float(parts[0]), 1.0
        else:
            wide, high = float(parts[0]), float(parts[1])
    except ValueError:
        return 1.0, 1.0
    if not (wide > 0 and high > 0) or math.isinf(wide) or math.isinf(high):
        return 1.0, 1.0
    return wide, high


def _snap(value: float, multiple: int) -> int:
    """The nearest whole step of ``multiple``, never below one step and never past the cap."""
    step = max(1, int(multiple))
    snapped = int(round(float(value) / step)) * step
    return max(step, min(snapped, (MAX_EDGE // step) * step))


def resolve(
    ratio: str,
    driver: str = "megapixels",
    size: int = 1024,
    megapixels: float = 1.0,
    multiple: int = 64,
    megapixel: int = MEGAPIXEL,
) -> tuple[int, int]:
    """The width and height a ratio and one measurement come to.

    Args:
        ratio: An entry from :data:`RATIOS`, or anything :func:`parse_ratio` reads, width first.
        driver: An entry from :data:`DRIVERS`, naming the measurement being given.
        size: The measurement in pixels, read by every driver but ``megapixels``.
        megapixels: The measurement in megapixels, read by ``megapixels`` alone.
        multiple: Step both sides land on, in pixels.
        megapixel: Pixels in one megapixel, from :func:`megapixel_unit`.

    Returns:
        ``(width, height)``, both a whole number of steps and both at least one step.
    """
    wide, high = parse_ratio(ratio)
    shape = wide / high

    if driver == "megapixels":
        area = max(float(megapixels), 0.0) * max(1, int(megapixel))
        width = math.sqrt(area * shape) if area > 0 else 0.0
        height = width / shape if shape > 0 else 0.0
    else:
        edge = max(float(size), 0.0)
        if driver == "width" or (driver == "long edge" and shape >= 1.0) \
                or (driver == "short edge" and shape < 1.0):
            width, height = edge, edge / shape
        else:
            height, width = edge, edge * shape

    return _snap(width, multiple), _snap(height, multiple)
