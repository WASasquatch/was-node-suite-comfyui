"""Palette reduction per frame with Pillow's median cut, frames run on a thread pool.

Frames are ``(height, width, 3)`` uint8 arrays. Dither is ``none``, ``floyd-steinberg`` or
``bayer-N`` for N in 2, 4, 8 and 16.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

#: Dither choices, in the order a menu lists them.
DITHERS = ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16")

#: The most frames quantized at once.
MAX_WORKERS = 8


def _bayer_matrix(level: int) -> np.ndarray:
    """The normalised ordered-dither matrix of side ``2 ** level``."""
    if level == 0:
        return np.zeros((1, 1), "float32")
    q = 4 ** level
    m = q * _bayer_matrix(level - 1)
    return np.block([[m - 1.5, m + 0.5], [m + 1.5, m - 0.5]]) / q


def _bayer(picture: Image.Image, palette: Image.Image, order: int) -> Image.Image:
    """Offset a picture by an ordered-dither matrix, then map it to the palette."""
    colours = len(palette.getpalette()) // 3
    matrix = (2 * 256 / colours) * _bayer_matrix(int(math.log2(order))) + 0.5
    values = np.asarray(picture).astype(np.float32)
    rows = math.ceil(values.shape[0] / matrix.shape[0])
    columns = math.ceil(values.shape[1] / matrix.shape[1])
    tiled = np.tile(matrix, (rows, columns))[: values.shape[0], : values.shape[1], None]
    offset = np.clip((values.astype(np.float64) + tiled).astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(offset).quantize(palette=palette, dither=Image.Dither.NONE)


def frame(values: np.ndarray, colours: int, dither: str) -> np.ndarray:
    """Reduce one frame to a palette of its own.

    Args:
        values: ``(height, width, 3)`` uint8.
        colours: Palette size, 1 to 256.
        dither: One of :data:`DITHERS`.

    Returns:
        ``(height, width, 3)`` uint8 in the palette's colours.
    """
    picture = Image.fromarray(values, mode="RGB")
    palette = picture.quantize(colors=colours)
    if dither == "floyd-steinberg":
        reduced = picture.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
    elif dither.startswith("bayer"):
        reduced = _bayer(picture, palette, int(dither.split("-")[-1]))
    else:
        reduced = picture.quantize(palette=palette, dither=Image.Dither.NONE)
    return np.asarray(reduced.convert("RGB"))


def frames(batch, colours: int, dither: str) -> list[np.ndarray]:
    """Reduce every frame of a batch, several at once.

    Args:
        batch: Sequence of ``(height, width, 3)`` uint8 arrays.
        colours: Palette size, 1 to 256.
        dither: One of :data:`DITHERS`.

    Returns:
        The reduced frames, in order.
    """
    workers = max(1, min(len(batch), os.cpu_count() or 1, MAX_WORKERS))
    if workers == 1:
        return [frame(values, colours, dither) for values in batch]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda values: frame(values, colours, dither), batch))
