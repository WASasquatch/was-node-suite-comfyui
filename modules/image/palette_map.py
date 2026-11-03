"""Palette matching and gradient mapping for RGB images.

Images are float arrays in ``[0, 1]`` shaped ``(height, width, 3)``. Palettes are uint8
arrays shaped ``(n, 3)``. Distances are measured in Oklab.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BAYER_8",
    "CUBE_STEPS",
    "DITHERS",
    "MODES",
    "RAMP",
    "apply_palette",
    "nearest_indices",
    "oklab",
    "parse_palette",
    "ramp_map",
    "sorted_by_lightness",
]

#: Palette-matching modes this module implements, as they are named on a widget.
MODES = ("Perceptual", "Luminance Ramp")

#: The subset of :data:`MODES` that treats the palette as an ordered gradient rather than
#: as a set of colours to choose from.
RAMP = ("Luminance Ramp",)

#: Dither methods, as they are named on a widget.
DITHERS = ("none", "FloydSteinberg", "Bayer")

#: Levels per channel in the nearest-colour lookup cube. 32 gives 32768 entries, which is
#: built in one vectorised pass against a 256-colour palette and indexed in constant time
#: from the serial diffusion loop.
CUBE_STEPS = 32

#: Normalised 8x8 Bayer threshold matrix, the ordered-dither pattern. Values are the
#: standard recursive construction divided by 64 and centred on zero, so adding it to an
#: image shifts each pixel up or down by less than one palette step before matching.
BAYER_8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
) - 0.5

#: Pixels matched per block when the distance table is built. The table is one float per
#: pixel per palette entry, so a whole 512x512 image against 256 colours would be 268 MB;
#: a block of 65536 keeps it to 64 MB whatever the image size.
_BLOCK = 65536

#: Linear sRGB to the LMS cone responses Oklab is built on.
_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)

#: The cube-rooted cone responses to Oklab's lightness and two opponent axes.
_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function.

    Args:
        rgb: Values in ``[0, 1]``, any shape.

    Returns:
        Linear-light values of the same shape.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function.

    Args:
        linear: Linear-light values, any shape.

    Returns:
        Values in ``[0, 1]``, clipped.
    """
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055
    )
    return np.clip(encoded, 0.0, 1.0)


def oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to Oklab.

    Args:
        rgb: Array whose last axis is ``(red, green, blue)`` in ``[0, 1]``.

    Returns:
        An array of the same shape whose last axis is ``(L, a, b)``. ``L`` runs 0 to 1 for
        black to white; ``a`` and ``b`` are the green-red and blue-yellow axes and are
        roughly bounded by ±0.4.
    """
    linear = _srgb_to_linear(rgb)
    cones = linear @ _LMS.T
    # The cube root is the whole of Oklab's non-linearity. Negative cone responses come
    # only from rounding at the extremes, and np.cbrt takes them without complaint.
    return np.cbrt(cones) @ _OKLAB.T


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab back to sRGB in ``[0, 1]``.

    Args:
        lab: Array whose last axis is ``(L, a, b)``.

    Returns:
        An array of the same shape whose last axis is ``(red, green, blue)``. Colours
        outside the sRGB gamut are clipped per channel.
    """
    cones = lab @ np.linalg.inv(_OKLAB).T.astype(np.float32)
    return _linear_to_srgb((cones**3) @ np.linalg.inv(_LMS).T.astype(np.float32))


def _split_entries(text: str) -> list[str]:
    """Split palette text into the entries it holds.

    Args:
        text: One string holding entries separated by line breaks or by commas.

    Returns:
        The entries, in order. A line break always separates, at every separator
        :meth:`str.splitlines` breaks on. A comma separates only outside parentheses, so
        the commas inside ``rgb(255, 136, 0)`` belong to the colour. A line whose
        parentheses do not close splits at every comma instead, which keeps a colour typed
        after a half-written function form on the same line.
    """
    entries: list[str] = []
    for line in text.splitlines():
        fields: list[str] = []
        depth = 0
        start = 0
        for position, character in enumerate(line):
            if character == "(":
                depth += 1
            elif character == ")" and depth:
                depth -= 1
            elif character == "," and depth == 0:
                fields.append(line[start:position])
                start = position + 1
        fields.append(line[start:])
        entries.extend(line.split(",") if depth else fields)
    return entries


def parse_palette(entries) -> np.ndarray:
    """Read a palette out of whatever a widget or a LIST socket carries.

    Args:
        entries: An iterable of colour strings, or one string holding them separated by
            line breaks or commas. Every spelling PIL understands is accepted:
            ``#RRGGBB``, ``#RGB``, ``red``, ``rgb(255, 0, 0)``, ``rgb(100%, 53%, 0%)``,
            ``rgba(255, 136, 0, 128)``, ``hsl(30, 100%, 50%)`` and ``hsv(30, 100%, 100%)``.
            A comma inside parentheses is part of the colour and does not separate one
            entry from the next. An entry that is not a colour is skipped rather than
            failing the palette, since a palette pasted from elsewhere routinely carries a
            heading or a stray line.

    Returns:
        An ``(n, 3)`` uint8 array, in the order the entries were given, holding each colour
        once. A later entry resolving to a colour already in the palette is dropped
        wherever it sits, so ``#f80`` under ``#ff8800`` adds nothing.

    Raises:
        ValueError: Nothing in ``entries`` was a colour.
    """
    from PIL import ImageColor

    if isinstance(entries, str):
        entries = _split_entries(entries)

    colors: list[tuple[int, int, int]] = []
    for entry in entries or ():
        text = str(entry).strip()
        if not text:
            continue
        try:
            resolved = ImageColor.getcolor(text, "RGB")
        except ValueError:
            continue
        if resolved not in colors:
            colors.append(resolved)

    if not colors:
        raise ValueError(
            "No colours could be read from the palette. Entries look like '#ff8800', "
            "'#f80', 'orange' or 'rgb(255, 136, 0)', one per line."
        )
    return np.array(colors, dtype=np.uint8)


def sorted_by_lightness(palette: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Order a palette dark to light.

    Args:
        palette: An ``(n, 3)`` uint8 array.
        reverse: Order light to dark instead.

    Returns:
        The same colours, reordered. Sorting is on Oklab lightness, so a saturated yellow
        sorts above a saturated blue as it looks rather than as its channel sum suggests.
    """
    lightness = oklab(palette.astype(np.float32) / 255.0)[:, 0]
    order = np.argsort(lightness, kind="stable")
    return palette[order[::-1] if reverse else order]


def nearest_indices(pixels: np.ndarray, palette: np.ndarray, perceptual: bool = True) -> np.ndarray:
    """Index of the closest palette entry for every pixel.

    Args:
        pixels: An ``(n, 3)`` array of colours in ``[0, 1]``.
        palette: An ``(m, 3)`` uint8 array.
        perceptual: Measure in Oklab rather than in RGB.

    Returns:
        An ``(n,)`` array of indices into ``palette``. Ties go to the earliest entry.
    """
    reference = palette.astype(np.float32) / 255.0
    if perceptual:
        pixels = oklab(pixels)
        reference = oklab(reference)

    chosen = np.empty(len(pixels), dtype=np.intp)
    # Squared distance expanded to |p|^2 - 2p.q + |q|^2, so the working array is one float
    # per pixel per entry rather than three.
    reference_sq = (reference**2).sum(axis=1)
    for start in range(0, len(pixels), _BLOCK):
        block = pixels[start : start + _BLOCK]
        distances = (block**2).sum(axis=1)[:, None] - 2.0 * (block @ reference.T) + reference_sq
        chosen[start : start + _BLOCK] = distances.argmin(axis=1)
    return chosen


def _lookup_cube(palette: np.ndarray, perceptual: bool) -> np.ndarray:
    """Nearest palette index for every colour of a coarse RGB cube.

    Args:
        palette: An ``(m, 3)`` uint8 array.
        perceptual: Measure in Oklab rather than in RGB.

    Returns:
        A ``(CUBE_STEPS, CUBE_STEPS, CUBE_STEPS)`` array of indices into ``palette``.
    """
    axis = np.linspace(0.0, 1.0, CUBE_STEPS, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return nearest_indices(grid, palette, perceptual).reshape(CUBE_STEPS, CUBE_STEPS, CUBE_STEPS)


def ramp_map(
    pixels: np.ndarray,
    palette: np.ndarray,
    smooth: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Map each pixel's lightness onto a palette read as a gradient.

    Args:
        pixels: An ``(n, 3)`` array of colours in ``[0, 1]``.
        palette: An ``(m, 3)`` uint8 array, in the order it is to be read.
        smooth: Interpolate between the two entries a pixel falls between. ``False``
            snaps to the nearer, posterising the gradient into ``m`` bands.
        normalize: Stretch the image's own darkest and lightest values across the ramp.
            ``False`` reads lightness absolutely.

    Returns:
        An ``(n, 3)`` array of colours in ``[0, 1]``.
    """
    lightness = oklab(pixels)[:, 0]
    reference = oklab(palette.astype(np.float32) / 255.0)

    if normalize:
        span = float(lightness.max() - lightness.min())
        # A flat plate has no range to stretch; without this every pixel would divide by
        # zero and land on the same entry.
        position = (
            (lightness - lightness.min()) / span if span > 1e-6 else np.zeros_like(lightness)
        )
    else:
        position = np.clip(lightness, 0.0, 1.0)
    scaled = position * (len(palette) - 1)

    if not smooth:
        return palette[np.rint(scaled).astype(np.intp)].astype(np.float32) / 255.0

    lower = np.floor(scaled).astype(np.intp)
    upper = np.minimum(lower + 1, len(palette) - 1)
    weight = (scaled - lower)[:, None]
    # Interpolating in Oklab rather than in sRGB is what keeps a ramp from darkening
    # through its middle, which is what a straight channel blend between two hues does.
    return _from_oklab(reference[lower] * (1.0 - weight) + reference[upper] * weight)


def _diffuse(pixels: np.ndarray, palette: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion against a palette.

    Args:
        pixels: An ``(height, width, 3)`` array of colours in ``[0, 1]``.
        palette: An ``(m, 3)`` uint8 array.
        cube: The lookup cube from :func:`_lookup_cube`.

    Returns:
        An ``(height, width, 3)`` array of palette colours in ``[0, 1]``.
    """
    working = pixels.astype(np.float32).copy()
    height, width = working.shape[:2]
    reference = palette.astype(np.float32) / 255.0
    result = np.empty_like(working)
    last = CUBE_STEPS - 1
    rows = np.arange(height)

    # Pixel (y, x) is taken at step x + 2y, after every pixel that passes it an error.
    for step in range(width + 2 * (height - 1)):
        ys = rows[max(0, (step - width + 2) // 2) : min(height - 1, step // 2) + 1]
        xs = step - 2 * ys
        old = working[ys, xs]
        index = np.clip(np.round(old * last), 0, last).astype(np.intp)
        new = reference[cube[index[:, 0], index[:, 1], index[:, 2]]]
        result[ys, xs] = new
        error = old - new
        # Each pixel receives its shares in the order a row-by-row scan adds them.
        below = ys + 1 < height
        right = xs + 1 < width
        left = below & (xs > 0)
        working[ys[left] + 1, xs[left] - 1] += error[left] * (3 / 16)
        working[ys[right], xs[right] + 1] += error[right] * (7 / 16)
        working[ys[below] + 1, xs[below]] += error[below] * (5 / 16)
        corner = below & right
        working[ys[corner] + 1, xs[corner] + 1] += error[corner] * (1 / 16)
    return result


def _bayer(pixels: np.ndarray, palette: np.ndarray, perceptual: bool) -> np.ndarray:
    """Ordered dithering against a palette.

    Args:
        pixels: An ``(height, width, 3)`` array of colours in ``[0, 1]``.
        palette: An ``(m, 3)`` uint8 array.
        perceptual: Measure in Oklab rather than in RGB.

    Returns:
        An ``(height, width, 3)`` array of palette colours in ``[0, 1]``.
    """
    height, width = pixels.shape[:2]
    tiled = np.tile(BAYER_8, (height // 8 + 1, width // 8 + 1))[:height, :width]
    # The threshold is scaled by the average gap between palette entries, so the pattern
    # nudges a pixel far enough to reach its neighbour in the palette and no further.
    lightness = np.sort(oklab(palette.astype(np.float32) / 255.0)[:, 0])
    gap = float(np.mean(np.diff(lightness))) if len(lightness) > 1 else 0.0
    shifted = np.clip(pixels + tiled[..., None] * gap, 0.0, 1.0)

    flat = shifted.reshape(-1, 3)
    chosen = nearest_indices(flat, palette, perceptual)
    return (palette[chosen].astype(np.float32) / 255.0).reshape(pixels.shape)


def apply_palette(
    image: np.ndarray,
    palette: np.ndarray,
    mode: str = "Perceptual",
    dither: str = "none",
    smooth: bool = True,
    reverse: bool = False,
    blend: float = 1.0,
    normalize: bool = False,
) -> np.ndarray:
    """Repaint an image in a palette's colours.

    Args:
        image: An ``(height, width, 3)`` array of colours in ``[0, 1]``.
        palette: An ``(n, 3)`` uint8 array.
        mode: ``Perceptual`` matches each pixel to its closest palette entry in Oklab.
            ``Luminance Ramp`` maps lightness along the palette instead.
        dither: ``none``, ``FloydSteinberg`` or ``Bayer``. Read only by ``Perceptual``.
        smooth: Interpolate along the ramp. Read only by ``Luminance Ramp``.
        reverse: Read the palette in the opposite direction.
        blend: How much of the result replaces the original, 0.0 to 1.0.
        normalize: Fit the ramp to the image's own range. Read only by ``Luminance Ramp``.

    Returns:
        An ``(height, width, 3)`` array of colours in ``[0, 1]``.
    """
    if mode in RAMP:
        ordered = sorted_by_lightness(palette, reverse)
    elif reverse:
        ordered = palette[::-1]
    else:
        ordered = palette

    if mode in RAMP:
        mapped = ramp_map(image.reshape(-1, 3), ordered, smooth, normalize).reshape(image.shape)
    elif dither == "FloydSteinberg":
        mapped = _diffuse(image, ordered, _lookup_cube(ordered, True))
    elif dither == "Bayer":
        mapped = _bayer(image, ordered, True)
    else:
        chosen = nearest_indices(image.reshape(-1, 3), ordered, True)
        mapped = (ordered[chosen].astype(np.float32) / 255.0).reshape(image.shape)

    if blend >= 1.0:
        return mapped
    factor = max(0.0, blend)
    return np.clip(image * (1.0 - factor) + mapped * factor, 0.0, 1.0)
