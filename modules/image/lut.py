"""Colour lookup tables: the model, the ``.cube`` format, and the grading maths.

A LUT is a cube of RGB samples. :func:`apply_lut_3d` reads the cube at the coordinate each
pixel's colour names, interpolating between the eight surrounding samples.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .. import log

__all__ = [
    "BUILTIN_PRESETS",
    "MAX_LUT_EDGE",
    "MIN_LUT_EDGE",
    "PRESET_DEFAULTS",
    "LUT",
    "apply_color_balance",
    "apply_contrast",
    "apply_exposure",
    "apply_gamma",
    "apply_lift",
    "apply_lut_3d",
    "apply_saturation",
    "apply_split_tone",
    "apply_vibrance",
    "apply_white_balance",
    "convert_to_3d",
    "cube_edge",
    "cube_files",
    "estimate_saturation",
    "find_cube",
    "identity_cube",
    "load_cube",
    "luma",
    "save_cube",
    "save_directory",
    "search_directories",
]

logger = log.get_logger("image.lut")

#: Subdirectory of every ComfyUI models root that cube files are read from.
#: Subdirectory of a ComfyUI models root that is also read, for an install holding its
#: tables there.
LUT_DIR_NAME = "LUT"

#: File extension a cube file carries, lower-cased for comparison.
CUBE_SUFFIX = ".cube"

#: Shortest and longest cube edge a table is built, read or resampled at, in samples.
MIN_LUT_EDGE = 2
MAX_LUT_EDGE = 256

#: Named looks that need no file, as
#: ``(exposure, contrast, saturation, vibrance, gamma, temperature, tint)``.
#: A look, as the grading settings that build it. Every key is optional and defaults to
#: doing nothing, so a preset names only what it changes. ``ev``, ``contrast``,
#: ``saturation``, ``vibrance``, ``gamma``, ``temperature`` and ``tint`` are the seven the
#: Custom widgets expose; ``lift`` raises the black point, ``shadows`` and ``highlights``
#: tint the two ends, and ``balance`` moves the crossover between them.
PRESET_DEFAULTS = {
    "ev": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "vibrance": 0.0,
    "gamma": 1.0,
    "temperature": 0.0,
    "tint": 0.0,
    "lift": 0.0,
    "shadows": (0.0, 0.0, 0.0),
    "highlights": (0.0, 0.0, 0.0),
    "balance": 0.0,
}

#: The looks the menu offers without a file. Each is built from grading settings rather
#: than shipped as data, so the set costs nothing on disk and carries no third-party terms.
#: The first five are the original set and their settings are unchanged, so a workflow
#: holding one renders as it always did.
BUILTIN_PRESETS = (
    ("Cinematic", {"contrast": 1.15, "saturation": 0.90, "vibrance": 0.35, "gamma": 0.95,
                   "temperature": 0.10, "tint": -0.05}),
    ("Vibrant", {"ev": 0.1, "contrast": 1.10, "saturation": 1.25, "vibrance": 0.30,
                 "gamma": 0.95, "temperature": 0.05}),
    ("Desaturated", {"contrast": 1.05, "saturation": 0.65, "vibrance": -0.10, "gamma": 1.05}),
    ("High Contrast", {"contrast": 1.35, "vibrance": 0.10, "gamma": 0.95}),
    ("Soft", {"ev": -0.05, "contrast": 0.90, "saturation": 0.95, "vibrance": -0.05,
              "gamma": 1.05}),

    # Split-toned looks. The crossover is what a three-way colour corrector is reached for,
    # and it is the difference between a colour cast and a grade.
    ("Teal and Orange", {"contrast": 1.18, "saturation": 0.95, "vibrance": 0.25,
                         "shadows": (-0.05, 0.01, 0.07), "highlights": (0.06, 0.02, -0.05)}),
    ("Golden Hour", {"ev": 0.05, "contrast": 1.08, "vibrance": 0.20, "temperature": 0.18,
                     "highlights": (0.07, 0.035, -0.03)}),
    ("Blue Hour", {"ev": -0.10, "contrast": 1.05, "saturation": 0.92, "temperature": -0.18,
                   "shadows": (-0.03, 0.0, 0.06)}),
    ("Day for Night", {"ev": -0.45, "contrast": 1.20, "saturation": 0.70, "temperature": -0.30,
                       "shadows": (-0.04, -0.01, 0.08)}),
    ("Cross Process", {"contrast": 1.30, "saturation": 1.15,
                       "shadows": (0.0, 0.05, 0.10), "highlights": (0.10, 0.05, -0.10)}),
    ("Cyberpunk", {"contrast": 1.22, "saturation": 1.20, "vibrance": 0.20,
                   "shadows": (0.02, -0.02, 0.10), "highlights": (0.09, -0.02, 0.06)}),
    ("Moody Green", {"contrast": 1.12, "saturation": 0.85, "temperature": -0.05,
                     "shadows": (-0.03, 0.05, 0.0), "highlights": (0.0, 0.03, -0.02)}),
    ("Autumn", {"contrast": 1.10, "vibrance": 0.30, "temperature": 0.14,
                "highlights": (0.06, 0.02, -0.04)}),
    ("Arctic", {"ev": 0.05, "contrast": 1.12, "saturation": 0.88, "temperature": -0.22,
                "highlights": (-0.02, 0.01, 0.05)}),

    # Print and stock looks. A lifted black is what separates a faded print from a low
    # contrast one, so these carry lift rather than only a gentler curve.
    ("Faded Film", {"contrast": 0.88, "saturation": 0.85, "lift": 0.09,
                    "shadows": (0.02, 0.01, 0.03)}),
    ("Bleach Bypass", {"contrast": 1.45, "saturation": 0.45, "gamma": 0.95,
                       "highlights": (0.02, 0.02, 0.02)}),
    ("Vintage Warm", {"contrast": 0.95, "saturation": 0.88, "lift": 0.06, "temperature": 0.16,
                      "shadows": (0.04, 0.02, -0.01)}),
    ("Muted Pastel", {"ev": 0.08, "contrast": 0.90, "saturation": 0.80, "lift": 0.07,
                      "highlights": (0.02, 0.01, 0.02)}),
    ("Sun Bleached", {"ev": 0.12, "contrast": 0.92, "saturation": 0.72, "lift": 0.10,
                      "temperature": 0.08}),

    # Portrait and neutral looks, where the point is that nothing draws attention.
    ("Warm Portrait", {"contrast": 1.06, "saturation": 0.98, "vibrance": 0.18,
                       "temperature": 0.09, "highlights": (0.03, 0.015, -0.01)}),
    ("Clean Neutral", {"contrast": 1.08, "vibrance": 0.12, "gamma": 0.98}),
    ("Punchy", {"ev": 0.05, "contrast": 1.28, "saturation": 1.18, "vibrance": 0.25,
                "gamma": 0.94}),

    # Monochrome. Saturation at zero, so the tint that follows is the whole look.
    ("Noir", {"contrast": 1.40, "saturation": 0.0, "gamma": 0.94}),
    ("Silver Gelatin", {"contrast": 1.10, "saturation": 0.0, "lift": 0.05,
                        "highlights": (0.01, 0.01, 0.02)}),
    ("Sepia", {"contrast": 1.05, "saturation": 0.0, "lift": 0.03,
               "shadows": (0.05, 0.02, -0.03), "highlights": (0.10, 0.06, -0.04)}),
)

#: ``((path, mtime), ...)`` the cube listing was built from, and the listing itself.
_listing: tuple[tuple[tuple[str, float], ...], list[Path]] | None = None


class LUT:
    """One colour lookup table, holding either a 1D curve set or a 3D cube.

    Attributes:
        title: Name carried in the file's ``TITLE`` line, or the preset's name.
        domain_min: Per-channel input value that maps to the first sample.
        domain_max: Per-channel input value that maps to the last sample.
        table_1d: ``(size, 3)`` array of one output triple per input level, or ``None``.
        table_3d: ``(size, size, size, 3)`` array indexed ``[r, g, b]``, or ``None``.
    """

    def __init__(
        self,
        title: str = "",
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        table_1d: np.ndarray | None = None,
        table_3d: np.ndarray | None = None,
    ):
        self.title = title
        self.domain_min = np.array(domain_min, dtype=np.float32)
        self.domain_max = np.array(domain_max, dtype=np.float32)
        self.table_1d = table_1d
        self.table_3d = table_3d

    def size(self) -> int:
        """Edge length of the table, or 0 when the table is empty."""
        if self.table_3d is not None:
            return int(self.table_3d.shape[0])
        if self.table_1d is not None:
            return int(self.table_1d.shape[0])
        return 0


def cube_edge(size) -> int:
    """A cube edge length as an int, held to :data:`MIN_LUT_EDGE` and :data:`MAX_LUT_EDGE`.

    Args:
        size: Edge length in samples.

    Returns:
        ``size`` as an int.

    Raises:
        ValueError: ``size`` is not a whole number, or lies outside that range.
    """
    edge = int(size)
    if edge < MIN_LUT_EDGE or edge > MAX_LUT_EDGE:
        raise ValueError(
            f"a LUT cube edge of {edge} samples was asked for, and an edge runs from "
            f"{MIN_LUT_EDGE} to {MAX_LUT_EDGE}. Set the size inside that range; 33 is the "
            f"usual choice."
        )
    return edge


def identity_cube(size: int) -> torch.Tensor:
    """Build the cube that maps every colour to itself.

    Args:
        size: Edge length in samples.

    Returns:
        A ``(1, size, size, size, 3)`` float32 tensor whose sample at ``[r, g, b]`` is the
        colour that indexes it. The leading axis is there so the grading operations, which
        are written for a batched image, apply to it unchanged.

    Raises:
        ValueError: ``size`` lies outside :data:`MIN_LUT_EDGE` to :data:`MAX_LUT_EDGE`.
    """
    size = cube_edge(size)
    grid = torch.linspace(0, 1, steps=size)
    red, green, blue = torch.meshgrid(grid, grid, grid, indexing="ij")
    return torch.stack([red, green, blue], dim=-1).unsqueeze(0).to(torch.float32)


def luma(x: torch.Tensor) -> torch.Tensor:
    """Perceived brightness of each pixel.

    Args:
        x: Tensor whose last axis is RGB.

    Returns:
        The Rec. 709 weighted sum, with the channel axis kept at length 1 so it broadcasts
        back over the colour it came from.
    """
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=x.dtype, device=x.device)
    return (x * weights.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True)


def apply_exposure(x: torch.Tensor, ev: float) -> torch.Tensor:
    """Scale brightness by a number of photographic stops.

    Args:
        x: Tensor whose last axis is RGB.
        ev: Stops. Each whole stop doubles or halves the value; 0.0 is a no-op.

    Returns:
        The scaled tensor, unclamped.
    """
    if ev == 0.0:
        return x
    return x * (2.0 ** ev)


def apply_contrast(x: torch.Tensor, c: float) -> torch.Tensor:
    """Push values away from or towards mid grey.

    Args:
        x: Tensor whose last axis is RGB.
        c: Multiplier around 0.5. Above 1.0 increases contrast, below 1.0 flattens it.

    Returns:
        The adjusted tensor, unclamped.
    """
    if abs(c - 1.0) < 1e-6:
        return x
    return (x - 0.5) * c + 0.5


def apply_saturation(x: torch.Tensor, s: float) -> torch.Tensor:
    """Scale the distance of each pixel from its own brightness.

    Args:
        x: Tensor whose last axis is RGB.
        s: Multiplier. 0.0 gives greyscale, 1.0 is a no-op, above 1.0 intensifies colour.

    Returns:
        The adjusted tensor, unclamped.
    """
    if abs(s - 1.0) < 1e-6:
        return x
    level = luma(x)
    return level + (x - level) * s


def estimate_saturation(x: torch.Tensor) -> torch.Tensor:
    """How colourful each pixel already is, as mean deviation from its channel average.

    Args:
        x: Tensor whose last axis is RGB.

    Returns:
        A tensor with the channel axis kept at length 1, near 0 for grey pixels.
    """
    mean = x.mean(dim=-1, keepdim=True)
    return (x - mean).abs().mean(dim=-1, keepdim=True)


def apply_vibrance(x: torch.Tensor, v: float) -> torch.Tensor:
    """Saturate muted colours more than colours that are already strong.

    Args:
        x: Tensor whose last axis is RGB.
        v: Strength. 0.0 is a no-op, positive lifts muted colour, negative drains it.

    Returns:
        The adjusted tensor, unclamped.
    """
    if abs(v) < 1e-6:
        return x
    saturation = estimate_saturation(x).clamp(0, 1)
    factor = 1.0 + v * (1.0 - saturation)
    level = luma(x)
    return level + (x - level) * factor


def apply_gamma(x: torch.Tensor, g: float) -> torch.Tensor:
    """Bend the tone curve between black and white.

    Args:
        x: Tensor whose last axis is RGB. Clamped to ``[0, 1]`` first, since a negative
            value has no real power.
        g: Gamma. Above 1.0 lifts midtones, below 1.0 deepens them, 1.0 is a no-op.

    Returns:
        The adjusted tensor, in ``[0, 1]``.
    """
    if abs(g - 1.0) < 1e-6:
        return x
    x = x.clamp(0.0, 1.0)
    return torch.pow(x, 1.0 / max(g, 1e-6))


def apply_white_balance(x: torch.Tensor, temp: float, tint: float) -> torch.Tensor:
    """Shift the colour of the light the picture appears to have been shot under.

    Args:
        x: Tensor whose last axis is RGB.
        temp: Temperature in ``[-1, 1]``. Positive warms towards red, negative cools
            towards blue.
        tint: Tint in ``[-1, 1]``. Positive pushes green, negative pushes magenta.

    Returns:
        The adjusted tensor, unclamped.
    """
    red_gain = 1.0 + 0.10 * temp - 0.10 * tint
    green_gain = 1.0 + 0.10 * tint
    blue_gain = 1.0 - 0.10 * temp - 0.10 * tint
    gains = torch.tensor([red_gain, green_gain, blue_gain], dtype=x.dtype, device=x.device)
    return x * gains.view(1, 1, 1, 3)


def apply_color_balance(x: torch.Tensor, r_bal: float, g_bal: float, b_bal: float) -> torch.Tensor:
    """Scale each channel on its own.

    Args:
        x: Tensor whose last axis is RGB.
        r_bal: Red offset in ``[-1, 1]``, applied as a gain of ``1 + r_bal``.
        g_bal: Green offset in ``[-1, 1]``.
        b_bal: Blue offset in ``[-1, 1]``.

    Returns:
        The adjusted tensor, unclamped.
    """
    gains = torch.tensor(
        [1.0 + r_bal, 1.0 + g_bal, 1.0 + b_bal], dtype=x.dtype, device=x.device
    )
    return x * gains.view(1, 1, 1, 3)


def apply_lift(x: torch.Tensor, amount: float) -> torch.Tensor:
    """Raise the black point, which is what makes a print look faded.

    Args:
        x: Tensor whose last axis is RGB.
        amount: How far black is lifted, 0 for none and 0.2 for a strongly faded print.

    Returns:
        The adjusted tensor, unclamped. White is held in place, so the range compresses
        from below rather than the whole picture brightening.
    """
    if not amount:
        return x
    return x * (1.0 - amount) + amount


def apply_split_tone(x: torch.Tensor, shadows, highlights, balance: float = 0.0) -> torch.Tensor:
    """Tint the dark end and the bright end towards different colours.

    Args:
        x: Tensor whose last axis is RGB.
        shadows: ``(r, g, b)`` offsets applied where the picture is dark, each in
            ``[-1, 1]``.
        highlights: ``(r, g, b)`` offsets applied where it is bright.
        balance: Moves the crossover, negative towards the shadows and positive towards the
            highlights, in ``[-1, 1]``.

    Returns:
        The adjusted tensor, unclamped. The two weights are a smoothstep of luma and its
        complement, so they sum to one everywhere and a flat grey is tinted once.
    """
    if not any(shadows) and not any(highlights):
        return x
    weight = luma(x).clamp(0.0, 1.0)
    pivot = min(max(0.5 + balance * 0.5, 0.05), 0.95)
    # Smoothstep about the pivot, so the crossover is gradual and the ends are unmixed.
    span = torch.clamp((weight - pivot + 0.5), 0.0, 1.0)
    high = span * span * (3.0 - 2.0 * span)
    low = 1.0 - high
    shadow = torch.tensor(list(shadows), dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
    bright = torch.tensor(list(highlights), dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
    return x + low * shadow + high * bright


def synthesize_builtin_lut(name: str, size: int = 33) -> LUT:
    """Bake one of :data:`BUILTIN_PRESETS` into a cube.

    Args:
        name: Preset name, matched exactly.
        size: Edge length of the cube in samples.

    Returns:
        A LUT holding the graded identity cube.

    Raises:
        ValueError: No preset carries that name.
    """
    params = None
    for preset_name, preset in BUILTIN_PRESETS:
        if preset_name == name:
            params = preset
            break
    if params is None:
        raise ValueError(f"there is no built-in look named {name!r}")
    settings = {**PRESET_DEFAULTS, **params}
    x = identity_cube(size)
    x = apply_exposure(x, settings["ev"])
    x = apply_contrast(x, settings["contrast"])
    x = apply_saturation(x, settings["saturation"])
    x = apply_vibrance(x, settings["vibrance"])
    x = apply_white_balance(x, settings["temperature"], settings["tint"])
    x = apply_gamma(x, settings["gamma"])
    x = apply_lift(x, settings["lift"])
    x = apply_split_tone(x, settings["shadows"], settings["highlights"], settings["balance"])
    table = x.squeeze(0).clamp(0, 1).cpu().numpy().astype(np.float32)
    return LUT(name, (0, 0, 0), (1, 1, 1), None, table)


def _oriented(data: list, size: int) -> np.ndarray:
    """Read a flat sample list as a cube, in the axis order closer to identity.

    Args:
        data: Sample triples in file order.
        size: Edge length of the cube.

    Returns:
        A ``(size, size, size, 3)`` float32 array indexed ``[r, g, b]``.
    """
    written = np.asarray(data, dtype=np.float32).reshape(size, size, size, 3)
    identity = identity_cube(size).squeeze(0).cpu().numpy().astype(np.float32)
    swapped = np.transpose(written, (2, 1, 0, 3))
    # A LUT is a modest departure from the identity cube, which is what makes the distance
    # from identity a usable test of the axis order.
    error_written = float(np.mean((written - identity) ** 2))
    error_swapped = float(np.mean((swapped - identity) ** 2))
    return swapped if (error_swapped + 1e-12) < error_written else written


def load_cube(path: Path) -> LUT:
    """Read a ``.cube`` file.

    Args:
        path: File to read. Undecodable bytes are ignored rather than raised on, since a
            grading application may have written a comment in any encoding.

    Returns:
        The table it holds, 1D or 3D. A file with neither size header is read as a cube
        when its sample count is a perfect cube and as a 1D curve set otherwise.

    Raises:
        ValueError: The sample count contradicts the declared size, the declared cube edge
            lies outside :data:`MIN_LUT_EDGE` to :data:`MAX_LUT_EDGE`, or the file holds
            nothing that can be read as a table.
        OSError: The file cannot be opened.
    """
    title = path.stem
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    size_1d: int | None = None
    size_3d: int | None = None
    data: list[tuple[float, float, float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("TITLE"):
                quote = stripped.find('"')
                title = stripped[quote + 1:stripped.rfind('"')].strip() if quote >= 0 else stripped
                continue
            if upper.startswith("DOMAIN_MIN"):
                parts = stripped.split()
                if len(parts) >= 4:
                    domain_min = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if upper.startswith("DOMAIN_MAX"):
                parts = stripped.split()
                if len(parts) >= 4:
                    domain_max = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if upper.startswith("LUT_1D_SIZE"):
                size_1d = int(stripped.split()[1])
                continue
            if upper.startswith("LUT_3D_SIZE"):
                size_3d = int(stripped.split()[1])
                if not MIN_LUT_EDGE <= size_3d <= MAX_LUT_EDGE:
                    raise ValueError(
                        f"{path.name} declares LUT_3D_SIZE {size_3d}, and a cube edge runs "
                        f"from {MIN_LUT_EDGE} to {MAX_LUT_EDGE}"
                    )
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if size_3d is not None:
        expected = size_3d ** 3
        if len(data) != expected:
            raise ValueError(
                f"{path.name} declares LUT_3D_SIZE {size_3d}, so it should hold {expected} "
                f"samples, and holds {len(data)}"
            )
        return LUT(title, domain_min, domain_max, None, _oriented(data, size_3d))

    if size_1d is not None:
        if len(data) != size_1d:
            raise ValueError(
                f"{path.name} declares LUT_1D_SIZE {size_1d}, so it should hold {size_1d} "
                f"samples, and holds {len(data)}"
            )
        table = np.asarray(data, dtype=np.float32).reshape(size_1d, 3)
        return LUT(title, domain_min, domain_max, table, None)

    count = len(data)
    edge = round(count ** (1 / 3))
    if edge * edge * edge == count and edge > 1:
        return LUT(title, domain_min, domain_max, None, _oriented(data, edge))

    table = np.asarray(data, dtype=np.float32)
    if table.ndim == 2 and table.shape[1] == 3:
        return LUT(title, domain_min, domain_max, table, None)

    raise ValueError(
        f"{path.name} carries no LUT_1D_SIZE or LUT_3D_SIZE line and its {count} sample(s) "
        f"are neither a cube nor a curve"
    )


def save_cube(path: Path, lut: LUT) -> None:
    """Write a 3D table as a ``.cube`` file.

    Samples are written with the blue axis fastest, which is the order :func:`load_cube`
    reads them back in.

    Args:
        path: File to write, replaced if it exists.
        lut: Table to write. Convert a 1D table with :func:`convert_to_3d` first.

    Raises:
        ValueError: The LUT holds no 3D table.
        OSError: The file cannot be written.
    """
    if lut.table_3d is None:
        raise ValueError("a .cube file is written from a 3D table; convert the LUT first")
    table = lut.table_3d
    edge = int(table.shape[0])
    low = np.asarray(lut.domain_min, dtype=np.float32).tolist()
    high = np.asarray(lut.domain_max, dtype=np.float32).tolist()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"TITLE \"{lut.title or path.stem}\"\n")
        handle.write(f"LUT_3D_SIZE {edge}\n")
        handle.write(f"DOMAIN_MIN {low[0]:.6f} {low[1]:.6f} {low[2]:.6f}\n")
        handle.write(f"DOMAIN_MAX {high[0]:.6f} {high[1]:.6f} {high[2]:.6f}\n")
        for red in range(edge):
            for green in range(edge):
                for blue in range(edge):
                    sample = table[red, green, blue]
                    handle.write(
                        f"{float(sample[0]):.6f} {float(sample[1]):.6f} {float(sample[2]):.6f}\n"
                    )


def _trilinear(table: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Read a cube at fractional coordinates, blending the eight surrounding samples.

    Args:
        table: ``(size, size, size, 3)`` tensor indexed ``[r, g, b]``.
        coordinates: Tensor whose last axis holds ``(r, g, b)`` in sample units, already
            inside the table.

    Returns:
        A tensor shaped like ``coordinates``, holding the interpolated colours.
    """
    size = table.shape[0]
    low = torch.floor(coordinates).to(torch.long).clamp(0, size - 1)
    high = torch.clamp(low + 1, max=size - 1)
    fraction = (coordinates - low.to(coordinates.dtype)).clamp(0, 1)

    r0, g0, b0 = low[..., 0], low[..., 1], low[..., 2]
    r1, g1, b1 = high[..., 0], high[..., 1], high[..., 2]
    dr, dg, db = fraction[..., 0], fraction[..., 1], fraction[..., 2]

    c00 = table[r0, g0, b0] * (1 - dr)[..., None] + table[r1, g0, b0] * dr[..., None]
    c01 = table[r0, g0, b1] * (1 - dr)[..., None] + table[r1, g0, b1] * dr[..., None]
    c10 = table[r0, g1, b0] * (1 - dr)[..., None] + table[r1, g1, b0] * dr[..., None]
    c11 = table[r0, g1, b1] * (1 - dr)[..., None] + table[r1, g1, b1] * dr[..., None]

    c0 = c00 * (1 - dg)[..., None] + c10 * dg[..., None]
    c1 = c01 * (1 - dg)[..., None] + c11 * dg[..., None]

    return c0 * (1 - db)[..., None] + c1 * db[..., None]


def convert_to_3d(lut: LUT, size: int) -> LUT:
    """Resample any table onto a cube of the given edge length.

    Args:
        lut: Table to convert. A 3D table already at ``size`` is returned unchanged.
        size: Edge length in samples.

    Returns:
        A LUT holding a ``(size, size, size, 3)`` table, keeping the source's title and
        domain.

    Raises:
        ValueError: The LUT holds no table at all, or ``size`` lies outside
            :data:`MIN_LUT_EDGE` to :data:`MAX_LUT_EDGE`.
    """
    size = cube_edge(size)
    if lut.table_3d is not None and lut.table_3d.shape[0] == size:
        return lut

    if lut.table_3d is not None:
        source = torch.from_numpy(lut.table_3d).to(torch.float32)
        coordinates = identity_cube(size).squeeze(0) * (source.shape[0] - 1)
        table = _trilinear(source, coordinates)
        return LUT(lut.title, lut.domain_min, lut.domain_max, None, table.numpy().astype(np.float32))

    if lut.table_1d is not None:
        levels = lut.table_1d.shape[0]
        cube = identity_cube(size).squeeze(0)
        table = torch.from_numpy(lut.table_1d).to(torch.float32)

        def sample(index: torch.Tensor, channel: int) -> torch.Tensor:
            low = torch.floor(index).to(torch.long)
            high = torch.clamp(low + 1, max=levels - 1)
            fraction = (index - low.to(index.dtype)).clamp(0, 1)
            return table[low, channel] * (1 - fraction) + table[high, channel] * fraction

        scaled = (cube * (levels - 1)).clamp(0, levels - 1)
        out = torch.stack(
            [sample(scaled[..., 0], 0), sample(scaled[..., 1], 1), sample(scaled[..., 2], 2)], -1
        )
        return LUT(lut.title, lut.domain_min, lut.domain_max, None, out.numpy().astype(np.float32))

    raise ValueError("this LUT holds no table to convert")


def apply_lut_3d(
    image: torch.Tensor,
    table: np.ndarray,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> torch.Tensor:
    """Grade an image through a 3D table.

    Args:
        image: ``(batch, height, width, 3)`` tensor in ``[0, 1]``.
        table: ``(size, size, size, 3)`` array indexed ``[r, g, b]``.
        domain_min: Per-channel input value mapping to the first sample.
        domain_max: Per-channel input value mapping to the last sample.

    Returns:
        The graded image, in ``[0, 1]``, on the input's device and dtype.
    """
    size = table.shape[0]
    cube = torch.from_numpy(table).to(image.device, dtype=image.dtype)
    low = torch.tensor(domain_min, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
    high = torch.tensor(domain_max, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
    normalized = ((image - low) / torch.clamp(high - low, min=1e-8)).clamp(0.0, 1.0)
    return _trilinear(cube, normalized * (size - 1)).clamp(0.0, 1.0)


def save_directory() -> Path:
    """Where this pack reads and writes LUTs: ``paths.luts``, or one in the config directory.

    Returns:
        The directory. It is not created here.
    """
    from ..config import luts_directory

    return luts_directory()


def search_directories() -> list[Path]:
    """Directories cube files are read from, in listing order.

    Returns:
        The pack's own LUT directory first, which is where :func:`save_cube` writes and
        what ``paths.luts`` relocates, then every ComfyUI models root holding a ``LUT``
        subdirectory. Existing directories only, deduplicated, and empty outside ComfyUI.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    # Ahead of the models roots: a lookup table is small user data rather than a weight, and
    # this is the directory the pack itself writes to, so what Save LUT wrote is what the
    # menu offers first. A models/LUT directory is still read, for an install that put them
    # there before this was the pack's own directory.
    try:
        own = save_directory()
    except Exception as error:
        logger.debug("the pack's own LUT directory is unavailable (%s)", error)
    else:
        if own.is_dir():
            seen.add(own)
            found.append(own)

    try:
        from folder_paths import folder_names_and_paths
    except ImportError:
        folder_names_and_paths = {}

    for value in folder_names_and_paths.values():
        # The registered paths are read by position, and a value shaped otherwise is skipped.
        try:
            registered = list(value[0])
        except (TypeError, KeyError, IndexError) as error:
            logger.debug("a folder_paths entry was not a path list (%s)", error)
            continue
        for entry in registered:
            try:
                path = Path(entry)
            except TypeError:
                continue
            models = path.parent
            if models.name != "models":
                for ancestor in path.parents:
                    if ancestor.name == "models":
                        models = ancestor
                        break
            if models.name != "models":
                continue
            directory = models / LUT_DIR_NAME
            if directory.is_dir() and directory not in seen:
                seen.add(directory)
                found.append(directory)

    return found


def _stamps(directories: list[Path]) -> tuple[tuple[str, float], ...]:
    """``(path, mtime)`` for each directory that can be stated, in search order."""
    stamps = []
    for directory in directories:
        try:
            stamps.append((str(directory), os.path.getmtime(directory)))
        except OSError:
            continue
    return tuple(stamps)


def cube_files() -> list[Path]:
    """Every ``.cube`` file in the search directories, sorted by name.

    Returns:
        One path per file name, case-insensitively sorted. A name held by two directories
        is listed once, from the earlier directory in :func:`search_directories`, since the
        menu and :func:`find_cube` both key on the name alone and two entries reading alike
        would resolve to one table with no way to tell which.
    """
    global _listing

    # A node's define_schema runs again for every /object_info request, so an unmemoized
    # scan here is paid on every browser refresh.
    signature = _stamps(search_directories())
    if _listing is not None and _listing[0] == signature:
        return list(_listing[1])

    found: list[Path] = []
    for directory, _mtime in signature:
        try:
            with os.scandir(directory) as entries:
                found += [
                    Path(entry.path)
                    for entry in entries
                    if entry.is_file() and entry.name.lower().endswith(CUBE_SUFFIX)
                ]
        except OSError as error:
            logger.warning("the LUT directory %s could not be listed: %s", directory, error)

    by_name: dict[str, Path] = {}
    for path in found:
        key = path.name.lower()
        if key in by_name:
            logger.debug(
                "%s is shadowed by %s, which is earlier in the search order",
                path, by_name[key],
            )
            continue
        by_name[key] = path
    listed = sorted(by_name.values(), key=lambda path: path.name.lower())
    _listing = (signature, listed)
    return list(listed)


def find_cube(name: str) -> Path:
    """The path of a listed cube file, looked up by its file name.

    Args:
        name: File name, as it appears in the load menu.

    Returns:
        The path of that file.

    Raises:
        ValueError: No search directory holds a file of that name, which is what a
            workflow saved against a LUT since removed or renamed lands on.
    """
    for path in cube_files():
        if path.name == name:
            return path
    searched = ", ".join(str(directory) for directory in search_directories()) or "no directory"
    raise ValueError(f"there is no LUT file named {name!r}; searched {searched}")
