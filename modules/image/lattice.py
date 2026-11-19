"""Hald CLUT lattices with a probe band, built and read back as a 3D table.

A level ``L`` square is ``L**3`` pixels a side holding ``L**2`` steps, red fastest in raster
order, then green, then blue.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .delta_e import srgb_delta_e

__all__ = [
    "LEVELS",
    "MAX_READ_LEVEL",
    "MIN_READ_LEVEL",
    "PROBES",
    "Agreement",
    "band_height",
    "build",
    "cell_size",
    "check_probes",
    "layout_of",
    "read_table",
    "side",
    "steps",
]

#: Levels LUT Lattice builds.
LEVELS = (5, 6, 7, 8)

#: Smallest and largest level read back; level 16 holds the 256 steps a cube may have.
MIN_READ_LEVEL = 2
MAX_READ_LEVEL = 16

#: Probe colours, as ``(name, (red, green, blue))`` shares of each axis, snapped to the lattice.
PROBES = (
    ("black", (0.0, 0.0, 0.0)),
    ("white", (1.0, 1.0, 1.0)),
    ("mid grey", (0.5, 0.5, 0.5)),
    ("dark grey", (0.2, 0.2, 0.2)),
    ("light grey", (0.8, 0.8, 0.8)),
    ("skin", (0.85, 0.62, 0.5)),
    ("red", (0.8, 0.15, 0.15)),
    ("green", (0.2, 0.65, 0.25)),
    ("blue", (0.15, 0.25, 0.8)),
    ("yellow", (0.9, 0.85, 0.2)),
    ("cyan", (0.2, 0.75, 0.8)),
    ("magenta", (0.75, 0.2, 0.7)),
)

#: Rows of cells in the probe band: flat patches, single pixels, flat patches mirrored.
BAND_ROWS = 3

#: Grey the band's unused right-hand columns are filled with.
FILLER = 0.5


def steps(level: int) -> int:
    """Samples per channel a lattice of this level holds."""
    return int(level) * int(level)


def side(level: int) -> int:
    """Width and height in pixels of the Hald square at this level."""
    return int(level) ** 3


def cell_size(level: int) -> int:
    """Width and height in pixels of one probe cell at this level."""
    return side(level) // len(PROBES)


def band_height(level: int) -> int:
    """Height in pixels of the probe band under the square at this level."""
    return BAND_ROWS * cell_size(level)


def _probe_indices(level: int) -> list[tuple[int, int, int]]:
    """Each probe's lattice index per channel at this level."""
    last = steps(level) - 1
    return [tuple(int(round(share * last)) for share in shares) for _name, shares in PROBES]


def _probe_colours(level: int) -> torch.Tensor:
    """``(probes, 3)`` float32 codes of the probes at this level."""
    last = steps(level) - 1
    indices = torch.tensor(_probe_indices(level), dtype=torch.float64)
    return (indices / last).to(torch.float32)


def _lattice_position(level: int, index: tuple[int, int, int]) -> tuple[int, int]:
    """``(y, x)`` of one lattice sample in the Hald square."""
    count = steps(level)
    red, green, blue = index
    flat = red + green * count + blue * count * count
    return flat // side(level), flat % side(level)


def build(level: int, probes: bool = True) -> torch.Tensor:
    """Build the identity lattice picture.

    Args:
        level: Hald level, 2 to 16.
        probes: Add the probe band under the square.

    Returns:
        ``(1, height, width, 3)`` float32 codes in 0 to 1, ``width`` being ``side(level)`` and
        ``height`` that plus ``band_height(level)`` when probes are drawn.

    Raises:
        ValueError: ``level`` lies outside 2 to 16.
    """
    level = int(level)
    if not MIN_READ_LEVEL <= level <= MAX_READ_LEVEL:
        raise ValueError(
            f"a Hald lattice is built at level {MIN_READ_LEVEL} to {MAX_READ_LEVEL}, and level "
            f"{level} was asked for"
        )
    count = steps(level)
    width = side(level)
    axis = (torch.arange(count, dtype=torch.float64) / (count - 1)).to(torch.float32)
    blue, green, red = torch.meshgrid(axis, axis, axis, indexing="ij")
    square = torch.stack([red, green, blue], dim=-1).reshape(width, width, 3)
    if not probes:
        return square.unsqueeze(0).contiguous()

    cell = cell_size(level)
    band = torch.full((band_height(level), width, 3), FILLER, dtype=torch.float32)
    colours = _probe_colours(level)
    total = len(PROBES)
    centre = cell // 2
    for number, colour in enumerate(colours):
        left = number * cell
        band[0:cell, left:left + cell] = colour
        surround = 1.0 if float(colour.mean()) < 0.5 else 0.0
        band[cell:2 * cell, left:left + cell] = surround
        band[cell + centre, left + centre] = colour
        mirrored = (total - 1 - number) * cell
        band[2 * cell:3 * cell, mirrored:mirrored + cell] = colour
    return torch.cat([square, band], dim=0).unsqueeze(0).contiguous()


def layout_of(height: int, width: int) -> tuple[int, bool] | None:
    """Which lattice a picture of this size is.

    Args:
        height: Picture height in pixels.
        width: Picture width in pixels.

    Returns:
        ``(level, probes)``, or None when no lattice has that size.
    """
    for level in range(MIN_READ_LEVEL, MAX_READ_LEVEL + 1):
        if width != side(level):
            continue
        if height == side(level):
            return level, False
        if cell_size(level) > 0 and height == side(level) + band_height(level):
            return level, True
    return None


def read_table(picture: torch.Tensor, level: int) -> torch.Tensor:
    """Read the graded Hald square back as a cube.

    Args:
        picture: ``(height, width, 3)`` codes, the square in its top rows.
        level: The lattice's level.

    Returns:
        ``(steps, steps, steps, 3)`` float32 indexed ``[red, green, blue]``, holding the
        values as they arrived, outside 0 to 1 included.
    """
    count = steps(level)
    width = side(level)
    square = picture[:width, :width, :3].to(torch.float32)
    return square.reshape(count, count, count, 3).permute(2, 1, 0, 3).contiguous()


@dataclass
class Agreement:
    """How closely the probes of one graded lattice agree, in CIEDE2000.

    Attributes:
        worst: Largest difference of any probe pixel from its lattice sample.
        noise: Largest spread inside one flat patch, away from its edges.
        position: Largest difference between a probe's two patches and its lattice sample.
        neighbourhood: Largest difference of a single pixel in a contrasting surround from
            its flat patch.
        probe: Name of the probe the worst difference was found on.
    """

    worst: float
    noise: float
    position: float
    neighbourhood: float
    probe: str


def check_probes(picture: torch.Tensor, level: int) -> Agreement:
    """Measure how far each probe's copies came out from its lattice sample.

    Args:
        picture: ``(height, width, 3)`` codes of a lattice built with probes, after grading.
        level: The lattice's level.

    Returns:
        The agreement, every figure in CIEDE2000.
    """
    width = side(level)
    cell = cell_size(level)
    margin = max(1, cell // 4)
    centre = cell // 2
    total = len(PROBES)
    rgb = picture[..., :3].to(torch.float64)
    square = rgb[:width, :width]
    band = rgb[width:width + band_height(level), :width]

    worst, noise, position, neighbourhood = 0.0, 0.0, 0.0, 0.0
    worst_probe = PROBES[0][0]
    for number, ((name, _shares), index) in enumerate(zip(PROBES, _probe_indices(level))):
        y, x = _lattice_position(level, index)
        sample = square[y, x]
        left = number * cell
        mirrored = (total - 1 - number) * cell
        first = band[0:cell, left:left + cell].reshape(-1, 3)
        second = band[2 * cell:3 * cell, mirrored:mirrored + cell].reshape(-1, 3)
        single = band[cell + centre, left + centre].reshape(1, 3)
        inner_first = band[margin:cell - margin, left + margin:left + cell - margin].reshape(-1, 3)
        inner_second = band[
            2 * cell + margin:3 * cell - margin, mirrored + margin:mirrored + cell - margin
        ].reshape(-1, 3)
        middle_first = inner_first.median(dim=0).values
        middle_second = inner_second.median(dim=0).values

        copies = torch.cat([first, second, single], dim=0)
        found = float(srgb_delta_e(copies, sample).max())
        if found > worst:
            worst, worst_probe = found, name
        noise = max(
            noise,
            float(srgb_delta_e(inner_first, middle_first).max()),
            float(srgb_delta_e(inner_second, middle_second).max()),
        )
        position = max(
            position,
            float(srgb_delta_e(middle_first, sample)),
            float(srgb_delta_e(middle_second, sample)),
            float(srgb_delta_e(middle_first, middle_second)),
        )
        neighbourhood = max(neighbourhood, float(srgb_delta_e(single[0], middle_first)))
    return Agreement(worst, noise, position, neighbourhood, worst_probe)
