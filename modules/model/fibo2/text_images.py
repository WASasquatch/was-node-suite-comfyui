"""Pictures of a Fibo 2.0 caption's texts, drawn as training drew them.

Each text is one black line on white, in Pillow's built-in font at size 20, with missing
characters taken from the fallback fonts.
"""

from __future__ import annotations

import functools
import os

from PIL import Image, ImageDraw, ImageFont

#: Point size every text is drawn at.
FONT_SIZE = 20

#: Characters per line; a longer text wraps.
LINE_CHARS = 256

#: White border around the text, and the gap between lines, in pixels.
PADDING = 2
LINE_GAP = 4

#: The fallback fonts, in the order training tried them.
FALLBACK_FILES = ("Aileron-Regular.otf", "DejaVuSans.ttf")

#: A character no font draws, whose mask is a font's "missing glyph" box.
NOT_A_CHARACTER = "￾"


@functools.lru_cache(maxsize=None)
def builtin_font() -> ImageFont.FreeTypeFont:
    """Pillow's built-in font at :data:`FONT_SIZE`.

    Raises:
        ImportError: Pillow is older than 10.1 or built without FreeType.
    """
    try:
        font = ImageFont.load_default(FONT_SIZE)
    except TypeError:
        font = None
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise ImportError("Drawing a caption's text needs Pillow 10.1 or later, built with FreeType.")
    return font


@functools.lru_cache(maxsize=None)
def fallback_fonts(folder: str) -> tuple[ImageFont.FreeTypeFont, ...]:
    """The fallback fonts found in ``folder``, in training's order, set without complex layout."""
    return tuple(
        ImageFont.truetype(os.path.join(folder, name), FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
        for name in FALLBACK_FILES
        if os.path.isfile(os.path.join(folder, name))
    )


@functools.lru_cache(maxsize=4096)
def has_glyph(font: ImageFont.FreeTypeFont, character: str) -> bool:
    """Whether ``font`` draws ``character`` as something other than its missing-glyph box."""
    try:
        missing = font.getmask(NOT_A_CHARACTER)
        mask = font.getmask(character)
    except Exception:  # noqa: BLE001
        return True
    return not (mask.size == missing.size and bytes(mask) == bytes(missing))


def draw(text: str, fallbacks=()) -> Image.Image:
    """One caption text as training drew it.

    Args:
        text: The text of a ``text_render`` entry.
        fallbacks: Fonts tried, in order, for characters the built-in font lacks.

    Returns:
        An RGB picture.
    """
    font = builtin_font()
    line = " ".join(text.split())
    lines = [line[start:start + LINE_CHARS] for start in range(0, len(line), LINE_CHARS)] or [""]
    if all(ord(character) < 128 or has_glyph(font, character) for character in line):
        return _one_font(lines, font)
    return _mixed(lines, (font, *fallbacks))


def _one_font(lines: list[str], font) -> Image.Image:
    """Lines placed by their bounding boxes, each at least as tall as ``Ag``."""
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    sample = measure.textbbox((0, 0), "Ag", font=font)
    floor = max(1, sample[3] - sample[1])
    boxes = [measure.textbbox((0, 0), line or " ", font=font) for line in lines]
    heights = [max(floor, box[3] - box[1]) for box in boxes]
    width = max([1] + [box[2] - box[0] for box in boxes])
    height = sum(heights) + LINE_GAP * (len(lines) - 1)

    picture = Image.new("RGB", (width + 2 * PADDING, height + 2 * PADDING), "white")
    pen = ImageDraw.Draw(picture)
    top = PADDING
    for line, box, line_height in zip(lines, boxes, heights):
        pen.text((PADDING - box[0], top - box[1]), line, font=font, fill=(0, 0, 0))
        top += line_height + LINE_GAP
    return picture


def _runs(line: str, fonts) -> list[tuple[str, object]]:
    """The line split where the font drawing it changes."""
    runs = []
    for character in line:
        font = fonts[0]
        if not (ord(character) < 128 or has_glyph(fonts[0], character)):
            font = next((candidate for candidate in fonts[1:] if has_glyph(candidate, character)), fonts[0])
        if runs and runs[-1][1] is font:
            runs[-1][0] += character
        else:
            runs.append([character, font])
    return [(run, font) for run, font in runs]


def _mixed(lines: list[str], fonts) -> Image.Image:
    """Runs of several fonts on a shared baseline, each line the built-in font's ascent plus descent."""
    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    rows = [_runs(line or " ", fonts) for line in lines]
    width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in rows)
    height = line_height * len(lines) + LINE_GAP * (len(lines) - 1)

    picture = Image.new("RGB", (width + 2 * PADDING, height + 2 * PADDING), "white")
    pen = ImageDraw.Draw(picture)
    top = PADDING
    for runs in rows:
        left = float(PADDING)
        for run, font in runs:
            pen.text((left, top + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            left += font.getlength(run)
        top += line_height + LINE_GAP
    return picture
