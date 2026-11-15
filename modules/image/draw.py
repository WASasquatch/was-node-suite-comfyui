"""Text and rectangle drawing onto transparent RGBA layers.

Each drawing function renders onto a layer the size of the target image and returns it.
:func:`composite` lays a layer over an image. Arabic and Hebrew lines are joined and ordered.
"""

from __future__ import annotations

from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..data import paths
from . import rtl_text

__all__ = [
    "ANCHORS",
    "BITMAP_LINE_HEIGHT",
    "DEFAULT_FONT",
    "FALLBACK",
    "TRANSPARENT",
    "anchor_origin",
    "composite",
    "draw_boxes_layer",
    "draw_text_layer",
    "layer_mask",
    "load_font",
    "parse_color",
    "shown_line",
    "text_block",
    "wrap_lines",
]

#: Where a block sits inside the image it is drawn on, as ``(x, y)`` fractions of the space
#: left over once the block's own size is taken off. ``(0.0, 0.0)`` is the top left corner
#: and ``(1.0, 1.0)`` the bottom right.
ANCHORS = {
    "top left": (0.0, 0.0),
    "top center": (0.5, 0.0),
    "top right": (1.0, 0.0),
    "middle left": (0.0, 0.5),
    "middle center": (0.5, 0.5),
    "middle right": (1.0, 0.5),
    "bottom left": (0.0, 1.0),
    "bottom center": (0.5, 1.0),
    "bottom right": (1.0, 1.0),
}

#: Colour used when a string cannot be read as one.
FALLBACK = (255, 255, 255, 255)

#: Fully transparent, the value a layer starts at and an empty colour resolves to.
TRANSPARENT = (0, 0, 0, 0)

#: Line height assumed for a font that reports no metrics, which is the bitmap face PIL
#: falls back to when no TrueType file can be opened. It has one size and does not say
#: what it is.
BITMAP_LINE_HEIGHT = 11

#: Bundled font used when none is named. DejaVu Sans of the bundled faces covers the most
#: of what arbitrary text contains, Latin, Greek, Cyrillic, Arabic, Hebrew, currency, and the
#: em dashes and curly quotes ordinary prose is full of.
DEFAULT_FONT = "DejaVu Sans"

#: A character no face maps, whose glyph is the face's missing-glyph box.
UNMAPPED = "\U0010fffd"

#: Per face, whether each character checked has a glyph of its own, and under
#: :data:`UNMAPPED` the face's missing-glyph box.
_glyphs: dict[tuple, dict] = {}

#: DejaVu Sans per size, for a right-to-left line the chosen face has no letters for.
_stand_ins: dict[int, object] = {}


def parse_color(text: str, default: tuple[int, int, int, int] = FALLBACK) -> tuple[int, int, int, int]:
    """Read an RGBA colour out of a widget string.

    Args:
        text: A colour in any spelling PIL understands, with an eight-digit ``#RRGGBBAA``
            accepted as well for a colour that carries its own alpha. An empty or
            whitespace-only string resolves to fully transparent, which is how a widget
            spells "do not draw this part".
        default: Returned when the string holds something that is not a colour.

    Returns:
        ``(red, green, blue, alpha)``, each 0-255.
    """
    value = (text or "").strip()
    if not value:
        return TRANSPARENT
    try:
        resolved = ImageColor.getcolor(value, "RGBA")
    except ValueError:
        return default
    return (resolved[0], resolved[1], resolved[2], resolved[3])


def load_font(
    size: int,
    font_path: str | None = None,
    bundled: str | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Open a TrueType font at a size, falling back to PIL's built-in bitmap font.

    Args:
        size: Point size. Ignored by the bitmap fallback, which has one size.
        font_path: Path to a font file, already through the sandbox. Tried first, so a
            font of the user's own beats the chosen bundled one.
        bundled: A key of :data:`modules.data.paths.FONTS`. ``None`` selects DejaVu Sans,
            which of the bundled faces covers the most of what arbitrary text contains.

    Returns:
        A PIL font object. Each candidate is tried in turn and the bitmap fallback is
        returned when none opens, so a missing or unreadable font costs the typeface
        rather than the run.
    """
    candidates = []
    if font_path:
        candidates.append(font_path)
    for name in (bundled, DEFAULT_FONT):
        if not name:
            continue
        try:
            located = paths.font_file(name)
        except ValueError:
            continue
        if located.is_file():
            candidates.append(str(located))
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(1, int(size)))
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _glyph(font, character: str) -> tuple:
    """A character rendered alone in a face: its mask's size and bytes."""
    mask = font.getmask(character)
    return mask.size, bytes(mask)


def _covers(font, text: str) -> bool:
    """Whether a face has a glyph of its own for every visible character of a text.

    Args:
        font: A PIL font.
        text: The characters to draw.

    Returns:
        True when none of them draws as the face's missing-glyph box. A face that is not a
        TrueType face is taken to cover everything.
    """
    if not isinstance(font, ImageFont.FreeTypeFont):
        return True
    path = font.path if isinstance(font.path, str) else font.getname()
    known = _glyphs.setdefault((path, font.index, font.size), {})
    try:
        if UNMAPPED not in known:
            known[UNMAPPED] = _glyph(font, UNMAPPED)
        box = known[UNMAPPED]
        for character in set(text) - set(known):
            if not character.isspace():
                known[character] = _glyph(font, character) != box
    except (OSError, ValueError):
        return True
    return all(known.get(character, True) for character in set(text) if character != UNMAPPED)


def _stand_in(font):
    """DejaVu Sans at a face's size, or None when the face is DejaVu Sans or has no size."""
    size = getattr(font, "size", None)
    if not size:
        return None
    if size not in _stand_ins:
        _stand_ins[size] = load_font(size, None, DEFAULT_FONT)
    stand_in = _stand_ins[size]
    if stand_in is font or getattr(stand_in, "path", None) == getattr(font, "path", None):
        return None
    return stand_in


def shown_line(line: str, font) -> tuple[str, object]:
    """The characters a line is drawn as, and the face it is drawn in.

    Args:
        line: One line, in typed order.
        font: The face chosen for the text.

    Returns:
        ``(characters, face)``. A line holding Arabic or Hebrew is joined and put in drawing
        order when the face lays text out a character at a time, and drawn in DejaVu Sans when
        the face has no glyphs for it. Any other line is returned as it is, in ``font``.
    """
    if getattr(font, "layout_engine", ImageFont.Layout.BASIC) != ImageFont.Layout.BASIC:
        return line, font
    if not rtl_text.holds_rtl(line):
        return line, font
    shown = rtl_text.display_order(line)
    if _covers(font, shown):
        return shown, font
    stand_in = _stand_in(font)
    return shown, stand_in if stand_in is not None else font


def _line_width(draw: ImageDraw.ImageDraw, line: str, font, stroke_width: int) -> float:
    """Rendered width of one line, stroke included."""
    shown, face = shown_line(line, font)
    box = draw.textbbox((0, 0), shown, font=face, stroke_width=stroke_width)
    return box[2] - box[0]


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
    stroke_width: int = 0,
) -> list[str]:
    """Break text into lines that fit a width.

    Args:
        draw: Draw context the measurements are taken from, which must be the one the text
            is later drawn with.
        text: The text to lay out.
        font: The font it is measured in.
        max_width: Width to fit, in pixels. A value of 0 or less turns wrapping off and
            only the existing line breaks are honoured.
        stroke_width: Outline width, which widens every glyph and so has to be measured.

    Returns:
        The lines, in order.
    """
    paragraphs = text.split("\n")
    if max_width <= 0:
        return paragraphs

    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if current and _line_width(draw, candidate, font, stroke_width) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font,
    line_spacing: float = 1.0,
    stroke_width: int = 0,
) -> tuple[int, int, int]:
    """Measure a laid-out block of text.

    Args:
        draw: Draw context the measurements are taken from.
        lines: The lines, already wrapped.
        font: The font they are measured in.
        line_spacing: Multiplier on the font's own line height. 1.0 is single spaced.
        stroke_width: Outline width, which widens and heightens every glyph.

    Returns:
        ``(width, height, line_height)``, all in pixels. ``line_height`` is the step from
        one baseline to the next and is what a caller advances by when drawing.
    """
    # A TrueType face reports its own ascent and descent. The bitmap face PIL falls back
    # to on older releases reports neither, and its line height is its one fixed size.
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
    else:
        ascent, descent = getattr(font, "size", BITMAP_LINE_HEIGHT), 0
    natural = ascent + descent + (stroke_width * 2)
    line_height = max(1, int(round(natural * max(0.1, line_spacing))))
    width = max((_line_width(draw, line, font, stroke_width) for line in lines), default=0)
    height = line_height * max(1, len(lines))
    return int(round(width)), int(height), line_height


def anchor_origin(
    position: str,
    canvas: tuple[int, int],
    block: tuple[int, int],
    margin: int = 0,
) -> tuple[int, int]:
    """Place a block inside a canvas at a named position.

    Args:
        position: A key of :data:`ANCHORS`. An unknown name is treated as
            ``middle center``.
        canvas: ``(width, height)`` of the surface drawn on.
        block: ``(width, height)`` of what is being placed.
        margin: Pixels held back from every edge, so an edge-anchored block does not touch
            the border. Ignored on the centred axis of a centred anchor.

    Returns:
        ``(x, y)`` of the block's top left corner. Negative when the block is larger than
        the space it is placed in, which lets an oversized block overhang symmetrically
        rather than being pinned to one corner.
    """
    fraction_x, fraction_y = ANCHORS.get(position, ANCHORS["middle center"])
    free_x = canvas[0] - block[0] - (margin * 2)
    free_y = canvas[1] - block[1] - (margin * 2)
    return (
        int(round(margin + free_x * fraction_x)),
        int(round(margin + free_y * fraction_y)),
    )


def layer_mask(layer: Image.Image) -> Image.Image:
    """The alpha channel of an RGBA layer, as a greyscale image.

    Args:
        layer: An RGBA image.

    Returns:
        An ``L`` image, white where the layer is opaque.
    """
    return layer.getchannel("A")


def draw_text_layer(
    size: tuple[int, int],
    text: str,
    font,
    color: tuple[int, int, int, int],
    position: str = "middle center",
    align: str = "center",
    offset: tuple[int, int] = (0, 0),
    margin: int = 0,
    line_spacing: float = 1.0,
    wrap_width: int = 0,
    stroke_width: int = 0,
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    background: tuple[int, int, int, int] = TRANSPARENT,
    background_padding: int = 0,
) -> Image.Image:
    """Render text onto a transparent layer.

    Args:
        size: ``(width, height)`` of the layer.
        text: The text to draw. Line breaks are honoured.
        font: A PIL font, from :func:`load_font`.
        color: RGBA fill for the glyphs.
        position: A key of :data:`ANCHORS`.
        align: ``left``, ``center`` or ``right``, how the lines sit inside the block.
        offset: ``(x, y)`` pixels added to the anchored position.
        margin: Pixels held back from every edge by the anchor.
        line_spacing: Multiplier on the font's line height.
        wrap_width: Width in pixels to wrap at, or 0 for no wrapping.
        stroke_width: Outline width in pixels. 0 draws no outline.
        stroke_color: RGBA fill for the outline.
        background: RGBA fill for a panel behind the block. Transparent draws none.
        background_padding: Pixels the panel extends past the text on every side.

    Returns:
        An RGBA layer of ``size``, transparent where the text is not.
    """
    layer = Image.new("RGBA", size, TRANSPARENT)
    if not text:
        return layer

    draw = ImageDraw.Draw(layer)
    lines = wrap_lines(draw, text, font, wrap_width, stroke_width)
    block_width, block_height, line_height = text_block(draw, lines, font, line_spacing, stroke_width)
    origin_x, origin_y = anchor_origin(position, size, (block_width, block_height), margin)
    origin_x += offset[0]
    origin_y += offset[1]

    if background[3] > 0:
        pad = max(0, background_padding)
        draw.rectangle(
            (
                origin_x - pad,
                origin_y - pad,
                origin_x + block_width + pad,
                origin_y + block_height + pad,
            ),
            fill=background,
        )

    for index, line in enumerate(lines):
        width = _line_width(draw, line, font, stroke_width)
        if align == "center":
            x = origin_x + (block_width - width) / 2
        elif align == "right":
            x = origin_x + (block_width - width)
        else:
            x = origin_x
        shown, face = shown_line(line, font)
        draw.text(
            (x, origin_y + index * line_height),
            shown,
            font=face,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color if stroke_width else None,
        )
    return layer


def draw_boxes_layer(
    size: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int, int],
    thickness: int = 2,
    fill: tuple[int, int, int, int] = TRANSPARENT,
    labels: list[str] | None = None,
    font=None,
    label_color: tuple[int, int, int, int] = FALLBACK,
) -> Image.Image:
    """Render rectangles onto a transparent layer.

    Args:
        size: ``(width, height)`` of the layer.
        boxes: ``(left, top, right, bottom)`` rows in pixels, with both corners inside the
            rectangle. Rows are drawn in order, so a later one covers an earlier one.
        color: RGBA fill for the outline.
        thickness: Outline width in pixels. 0 draws no outline, which leaves ``fill``
            as the only mark and is how a bounds row becomes a solid mask.
        fill: RGBA fill for the rectangle's interior. Fully transparent leaves it empty.
        labels: One caption per box, drawn just inside its top left corner. ``None`` draws
            no captions, and a list shorter than ``boxes`` captions only the boxes it
            reaches.
        font: A PIL font for the captions. Required only when ``labels`` is given.
        label_color: RGBA fill for the captions.

    Returns:
        An RGBA layer the size of ``size``.
    """
    layer = Image.new("RGBA", size, TRANSPARENT)
    draw = ImageDraw.Draw(layer)
    captions = labels or []

    for index, (left, top, right, bottom) in enumerate(boxes):
        box = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        if fill[3] > 0:
            draw.rectangle(box, fill=fill)
        if thickness > 0:
            draw.rectangle(box, outline=color, width=thickness)
        if index < len(captions) and captions[index] and font is not None:
            shown, face = shown_line(captions[index], font)
            draw.text(
                (box[0] + thickness + 2, box[1] + thickness + 2),
                shown,
                font=face,
                fill=label_color,
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
            )
    return layer


def composite(base: Image.Image, layer: Image.Image, opacity: float = 1.0) -> Image.Image:
    """Lay a rendered layer over an image.

    Args:
        base: The image drawn on, in any mode. Converted to ``RGB`` on the way out.
        layer: An RGBA layer of the same size.
        opacity: How much of the layer shows, 0.0 to 1.0. Applied to the layer's alpha as
            a whole, so overlapping marks inside one layer do not compound.

    Returns:
        An ``RGB`` image.
    """
    if opacity < 1.0:
        alpha = layer.getchannel("A").point(lambda value: int(value * max(0.0, opacity)))
        layer = layer.copy()
        layer.putalpha(alpha)
    merged = base.convert("RGBA")
    merged.alpha_composite(layer)
    return merged.convert("RGB")
