"""The reference strip a LUT node draws itself on.

:func:`reference_strip` builds the chart, :func:`graded_strip` runs a LUT over it. The top
:data:`RAMP_ROWS` rows are a neutral ramp, so a row read back off it is the transfer
response.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "HUE_ROWS",
    "PATCH_ROWS",
    "RAMP_ROWS",
    "STRIP_HEIGHT",
    "STRIP_WIDTH",
    "graded_strip",
    "reference_strip",
]

#: One column per 8-bit level, so a row read back is a 256 point transfer curve.
STRIP_WIDTH = 256

#: Rows in each band: the neutral ramp the curve is read from, a hue sweep, and the patches.
RAMP_ROWS = 16
HUE_ROWS = 16
PATCH_ROWS = 16

STRIP_HEIGHT = RAMP_ROWS + HUE_ROWS + PATCH_ROWS

#: Skin tones and memory colours a look is judged on, as sRGB 0-255. Laid across the last
#: band in equal blocks, light to dark, then foliage and sky.
PATCHES = (
    (247, 216, 193),
    (231, 188, 160),
    (198, 148, 118),
    (150, 105, 79),
    (94, 62, 47),
    (108, 138, 74),
    (110, 160, 205),
    (200, 200, 200),
)


def _hue_row() -> np.ndarray:
    """One row sweeping hue at full saturation and value.

    Returns:
        ``(STRIP_WIDTH, 3)`` in 0-1, red through the wheel and back to red.
    """
    hue = np.linspace(0.0, 6.0, STRIP_WIDTH, endpoint=False)
    sector = np.floor(hue).astype(int) % 6
    rise = hue - np.floor(hue)
    fall = 1.0 - rise
    zero = np.zeros_like(rise)
    one = np.ones_like(rise)
    table = np.stack([
        np.stack([one, rise, zero], axis=-1),
        np.stack([fall, one, zero], axis=-1),
        np.stack([zero, one, rise], axis=-1),
        np.stack([zero, fall, one], axis=-1),
        np.stack([rise, zero, one], axis=-1),
        np.stack([one, zero, fall], axis=-1),
    ])
    return table[sector, np.arange(STRIP_WIDTH)]


def _patch_row() -> np.ndarray:
    """One row of the memory colours, in equal blocks.

    Returns:
        ``(STRIP_WIDTH, 3)`` in 0-1, one block per entry of :data:`PATCHES`.
    """
    block = STRIP_WIDTH // len(PATCHES)
    row = np.zeros((STRIP_WIDTH, 3), dtype=np.float32)
    for index, colour in enumerate(PATCHES):
        start = index * block
        end = STRIP_WIDTH if index == len(PATCHES) - 1 else start + block
        row[start:end] = np.array(colour, dtype=np.float32) / 255.0
    return row


def reference_strip() -> torch.Tensor:
    """The ungraded chart every LUT panel is drawn against.

    Returns:
        ``(1, STRIP_HEIGHT, STRIP_WIDTH, 3)`` in 0-1: a neutral ramp, a hue sweep and the
        memory colours, in that order down the picture.
    """
    ramp = np.repeat(np.linspace(0.0, 1.0, STRIP_WIDTH, dtype=np.float32)[:, None], 3, axis=1)
    bands = [
        np.repeat(ramp[None, :, :], RAMP_ROWS, axis=0),
        np.repeat(_hue_row().astype(np.float32)[None, :, :], HUE_ROWS, axis=0),
        np.repeat(_patch_row()[None, :, :], PATCH_ROWS, axis=0),
    ]
    return torch.from_numpy(np.concatenate(bands, axis=0)[None, ...].copy())


def graded_strip(table) -> torch.Tensor:
    """The reference chart with one LUT applied to it.

    Args:
        table: A :class:`lut.LUT`.

    Returns:
        ``(1, STRIP_HEIGHT, STRIP_WIDTH, 3)`` in 0-1. A table with nothing in it grades
        nothing and the chart is returned as it is.
    """
    from . import lut as lut_module

    strip = reference_strip()
    if table is None or table.size() == 0:
        return strip
    cube = table if table.table_3d is not None else lut_module.convert_to_3d(table, 33)
    graded = lut_module.apply_lut_3d(strip, cube.table_3d, cube.domain_min, cube.domain_max)
    return graded.clamp(0.0, 1.0)
