"""Captions, timecode windows and name bars drawn as tiles and laid onto frames.

A :class:`Tile` holds premultiplied colour ``(height, width, 3)`` and coverage ``(height,
width)`` in 0 to 1. Sizes are pixels; colours are RGBA in 0 to 255.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from ..data import paths
from . import draw, rtl_text

__all__ = [
    "BOX_MODES",
    "Fonts",
    "Look",
    "Tile",
    "Typesetter",
    "cells",
    "lay",
    "GRAPHIC_STYLES",
    "lay_bar",
    "name_graphic",
    "paint",
    "set_cue",
    "timecode_tile",
]

#: How a caption's background box is drawn: none, one box per line, or one box round the block.
BOX_MODES = ("none", "lines", "block")

#: How a lower third is drawn.
GRAPHIC_STYLES = ("stripe", "split", "banner", "card", "underline", "minimal")

#: Soft glow behind a lower third with no backing: its blur as a share of the size, and its
#: strength as a share of the shadow colour's.
GLOW_RADIUS = 0.3
GLOW_STRENGTH = 0.6

#: Slant of a drawn italic, as the tangent of 12 degrees.
ITALIC_SLANT = 0.2126

#: Stroke a bold with no face of its own is thickened by, as a share of the size.
FAUX_BOLD = 0.03

#: Underline drop below the baseline and its weight, as shares of the size.
UNDERLINE_DROP = 0.11
UNDERLINE_WEIGHT = 0.06

#: Box padding beside and above a line, as shares of the size.
BOX_PAD_X = 0.32
BOX_PAD_Y = 0.06

#: Space between two captions shown at once, as a share of the size.
STACK_GAP = 0.25

#: Sizes a caption is tried at before more than two lines are allowed, as shares of its size.
SHRINK_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)

#: Smallest size a caption is drawn at, in pixels.
MIN_SIZE = 6

#: Words a line does not end on when a better break exists.
WEAK_WORDS = frozenset(
    "a an the of to in on at by for with from into onto over under about and or but nor so as "
    "if than that this these those my your his her its our their is are was were be".split()
)

#: Characters that end a clause, a good place for a line to break after.
CLAUSE_ENDS = tuple(",.;:!?\u2026")

#: Characters a spoken turn opens with.
DIALOGUE_DASHES = ("-", "\u2010", "\u2012", "\u2013", "\u2014")


@dataclass(frozen=True)
class Look:
    """How a caption is drawn.

    Attributes:
        fill: Glyph colour, where a cue sets none.
        outline: Outline width as a share of the size.
        outline_color: Outline colour.
        shadow: Shadow offset as a share of the size; its blur is half of it.
        shadow_color: Shadow colour.
        box: One of :data:`BOX_MODES`.
        box_color: Box colour.
    """

    fill: tuple = (255, 255, 255, 255)
    outline: float = 0.0
    outline_color: tuple = (0, 0, 0, 255)
    shadow: float = 0.0
    shadow_color: tuple = (0, 0, 0, 180)
    box: str = "none"
    box_color: tuple = (0, 0, 0, 160)


@dataclass
class Tile:
    """A drawn block, and the box it is placed by.

    Attributes:
        color: Premultiplied colour, ``(height, width, 3)``.
        cover: Coverage, ``(height, width)``.
        left: Left edge of the placing box, in tile pixels.
        top: Top edge of the placing box.
        right: Right edge of the placing box.
        bottom: Bottom edge of the placing box.
    """

    color: torch.Tensor
    cover: torch.Tensor
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Width of the placing box."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Height of the placing box."""
        return self.bottom - self.top


class Fonts:
    """A font menu value opened at any size, with its bold face where one exists."""

    def __init__(self, font: str):
        """Read a font menu value.

        Args:
            font: A name from the font menu, or a wired path to a font file.

        Raises:
            PathNotAllowed: A wired path lies outside the folders a font may be read from.
        """
        from ..interface import fonts
        from ..io import picker

        wired = picker.wired_file(font, paths.FONT_SUFFIXES, roots=fonts.font_roots())
        self.path = str(wired) if wired is not None else None
        self.name = None if wired is not None else (str(font or "").strip() or draw.DEFAULT_FONT)
        self.bold_name = None if self.name is None else _bold_sibling(self.name)
        self._held: dict = {}

    def faces(self, size: int) -> tuple:
        """``(regular, bold, faux)``: the faces at a size, and whether bold is thickened."""
        size = max(1, int(size))
        if size not in self._held:
            regular = draw.load_font(size, self.path, self.name)
            if self.bold_name and self.bold_name == self.name:
                self._held[size] = (regular, regular, False)
            elif self.bold_name:
                self._held[size] = (regular, draw.load_font(size, None, self.bold_name), False)
            else:
                self._held[size] = (regular, regular, True)
        return self._held[size]


def _bold_sibling(name: str) -> str | None:
    """The menu name of a font's bold face, the name itself when it is bold, or None."""
    if "bold" in name.lower():
        return name
    try:
        catalog = paths.font_catalog()
    except Exception:
        return None
    for candidate in (f"{name} Bold", name.replace("Regular", "Bold"), f"{name}-Bold", f"{name}Bold"):
        if candidate != name and candidate in catalog:
            return candidate
    return None


@dataclass(frozen=True)
class Run:
    """Characters drawn in one face and style.

    Attributes:
        text: The characters in drawing order.
        face: The PIL font.
        thicken: Stroke added for a thickened bold, in pixels.
        italic: Drawn slanted.
        underline: Drawn underlined.
        color: Glyph colour, or None for the look's.
        width: Advance in pixels.
    """

    text: str
    face: object
    thicken: int
    italic: bool
    underline: bool
    color: tuple | None
    width: float


def cells(spans) -> list:
    """A line's spans as ``(character, style)`` pairs, style ``(bold, italic, underline, color)``."""
    out = []
    for span in spans:
        style = (span.bold, span.italic, span.underline, span.color)
        out.extend((character, style) for character in span.text)
    return out


class Typesetter:
    """Measures and groups styled characters at one size."""

    def __init__(self, fonts: Fonts, size: int):
        """Open the faces at a size.

        Args:
            fonts: The font choice.
            size: Size in pixels.
        """
        self.size = max(1, int(size))
        self.regular, self.bold, faux = fonts.faces(self.size)
        self.thicken = max(1, round(FAUX_BOLD * self.size)) if faux else 0
        if hasattr(self.regular, "getmetrics"):
            self.ascent, self.descent = self.regular.getmetrics()
        else:
            self.ascent, self.descent = self.size, 0
        self._runs: dict = {}

    def runs(self, line) -> list:
        """The runs a line of cells is drawn as, right-to-left lines put in drawing order."""
        key = tuple(line)
        if key in self._runs:
            return self._runs[key]
        text = "".join(character for character, _style in line)
        groups: list = []
        if rtl_text.holds_rtl(text):
            styles = [style for character, style in line if not character.isspace()] or [line[0][1]]
            colors = {style[3] for style in styles}
            merged = (
                all(style[0] for style in styles),
                all(style[1] for style in styles),
                all(style[2] for style in styles),
                colors.pop() if len(colors) == 1 else None,
            )
            groups.append((text, merged))
        else:
            for character, style in line:
                if groups and groups[-1][1] == style:
                    groups[-1] = (groups[-1][0] + character, style)
                else:
                    groups.append((character, style))
        out = []
        for chunk, (bold, italic, underline, color) in groups:
            face = self.bold if bold else self.regular
            shown, face = draw.shown_line(chunk, face)
            thicken = self.thicken if bold else 0
            width = float(face.getlength(shown)) if hasattr(face, "getlength") else float(len(shown) * self.size)
            out.append(Run(shown, face, thicken, italic, underline, color, width + 2 * thicken))
        self._runs[key] = out
        return out

    def width(self, line) -> float:
        """Advance of a line of cells, in pixels."""
        return sum(run.width for run in self.runs(line))


def _words(line) -> list:
    """A line of cells split at plain spaces."""
    words, current = [], []
    for cell in line:
        if cell[0] == " ":
            if current:
                words.append(current)
            current = []
        else:
            current.append(cell)
    if current:
        words.append(current)
    return words


def _join(words) -> list:
    """Words joined back into one line of cells, each space styled as the word before it."""
    out: list = []
    for index, word in enumerate(words):
        if index:
            out.append((" ", out[-1][1]))
        out.extend(word)
    return out


def _plain(word) -> str:
    """The characters of a word."""
    return "".join(character for character, _style in word)


def _two_lines(words, typesetter: Typesetter, width: float):
    """The best place to break a line in two, or None where no break fits both halves.

    Args:
        words: The line's words.
        typesetter: Measures at the caption's size.
        width: Widest a line may be.

    Returns:
        Two lines of cells. Balanced lines win, then a bottom line at least as long as the top,
        then a break after a clause, never after an article, preposition or conjunction where
        another break fits.
    """
    best, best_cost = None, None
    for split in range(1, len(words)):
        top, bottom = _join(words[:split]), _join(words[split:])
        top_width, bottom_width = typesetter.width(top), typesetter.width(bottom)
        if top_width > width or bottom_width > width:
            continue
        cost = abs(top_width - bottom_width) / max(1.0, width)
        if top_width > bottom_width:
            cost += 0.1
        ending = _plain(words[split - 1]).rstrip("\"'\u201d\u2019)]")
        if ending.endswith(CLAUSE_ENDS):
            cost -= 0.35
        elif ending.lower().strip(".,;:!?\"'") in WEAK_WORDS:
            cost += 0.5
        if len(words) - split == 1 and split > 1:
            cost += 0.15
        if best_cost is None or cost < best_cost:
            best, best_cost = [top, bottom], cost
    return best


def _greedy(words, typesetter: Typesetter, width: float) -> list:
    """A line's words filled into as many lines of ``width`` as they need."""
    lines, current = [], []
    for word in words:
        candidate = current + [word]
        if current and typesetter.width(_join(candidate)) > width:
            lines.append(_join(current))
            current = [word]
        else:
            current = candidate
    if current:
        lines.append(_join(current))
    return lines


def _lay_out(paragraphs, typesetter: Typesetter, width: float) -> list:
    """The lines a cue is drawn as at one size, breaking only between words."""
    if all(typesetter.width(line) <= width for line in paragraphs):
        return list(paragraphs)
    if len(paragraphs) == 1:
        two = _two_lines(_words(paragraphs[0]), typesetter, width)
        if two:
            return two
    elif len(paragraphs) == 2 and not any(_plain(line).startswith(DIALOGUE_DASHES) for line in paragraphs):
        two = _two_lines(_words(paragraphs[0]) + _words(paragraphs[1]), typesetter, width)
        if two:
            return two
    lines = []
    for line in paragraphs:
        words = _words(line)
        if typesetter.width(line) <= width:
            lines.append(line)
            continue
        lines.extend(_two_lines(words, typesetter, width) or _greedy(words, typesetter, width))
    return lines


def set_cue(spans_lines, fonts: Fonts, size: int, width_at) -> tuple:
    """Lay a cue out in at most two lines, shrinking it only where it cannot fit.

    Args:
        spans_lines: The cue's lines, each a sequence of spans.
        fonts: The font choice.
        size: The caption size in pixels.
        width_at: Called with a size, answers the widest a line may be at that size.

    Returns:
        ``(lines, typesetter, shrunk, over)``: the lines of cells, the typesetter they were
        set with, whether the size was lowered, and whether more than two lines were needed.
        A cue the source wrote in more lines keeps them.
    """
    paragraphs = [cells(spans) for spans in spans_lines]
    allowed = max(2, len(paragraphs))
    for step in SHRINK_STEPS:
        typesetter = Typesetter(fonts, max(MIN_SIZE, round(size * step)))
        width = width_at(typesetter.size)
        lines = _lay_out(paragraphs, typesetter, width)
        if len(lines) <= allowed and all(typesetter.width(line) <= width for line in lines):
            return lines, typesetter, step < 1.0, False
    typesetter = Typesetter(fonts, max(MIN_SIZE, round(size * SHRINK_STEPS[-1])))
    width = width_at(typesetter.size)
    words = [word for line in paragraphs for word in _words(line)]
    widest = max((typesetter.width(word) for word in words), default=0.0)
    if widest > width > 0:
        typesetter = Typesetter(fonts, max(MIN_SIZE, math.floor(typesetter.size * width / widest)))
        width = width_at(typesetter.size)
    lines = []
    for line in paragraphs:
        lines.extend(_greedy(_words(line), typesetter, width))
    return lines, typesetter, True, len(lines) > allowed


def _over(layers, height: int, width: int) -> tuple:
    """Premultiplied colour and coverage of masks laid over one another in order.

    Args:
        layers: ``(rgba, mask)`` pairs, ``mask`` an ``L`` image of the tile's size.
        height: Tile height.
        width: Tile width.
    """
    color = torch.zeros((height, width, 3), dtype=torch.float32)
    cover = torch.zeros((height, width), dtype=torch.float32)
    for rgba, mask in layers:
        if rgba[3] <= 0:
            continue
        alpha = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).float() * (rgba[3] / 65025.0)
        if not bool(alpha.any()):
            continue
        tint = torch.tensor([rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0], dtype=torch.float32)
        keep = (1.0 - alpha).unsqueeze(-1)
        color = tint * alpha.unsqueeze(-1) + color * keep
        cover = alpha + cover * (1.0 - alpha)
    return color, cover


def _draw_run(masks: list, run: Run, x: float, baseline: float, grow: int, slant: float) -> None:
    """Draw a run's glyphs into masks, slanted when the run is italic.

    Args:
        masks: ``L`` images to draw into.
        run: The run.
        x: Left of its advance.
        baseline: Its baseline.
        grow: Stroke width added round every glyph.
        slant: Horizontal shift per pixel above the baseline.
    """
    if not run.text.strip():
        return
    if not run.italic:
        for mask in masks:
            ImageDraw.Draw(mask).text(
                (x, baseline), run.text, font=run.face, fill=255, anchor="ls",
                stroke_width=grow, stroke_fill=255,
            )
        return
    ascent, descent = run.face.getmetrics() if hasattr(run.face, "getmetrics") else (run.face.size, 0)
    pad = grow + 4
    lean = math.ceil(slant * (ascent + pad))
    width = int(math.ceil(run.width)) + 2 * pad + lean
    height = ascent + descent + 2 * pad
    own = Image.new("L", (width, height), 0)
    local = pad + ascent
    ImageDraw.Draw(own).text(
        (pad, local), run.text, font=run.face, fill=255, anchor="ls", stroke_width=grow, stroke_fill=255
    )
    sheared = own.transform(
        own.size, Image.AFFINE, (1.0, slant, -slant * local, 0.0, 1.0, 0.0), resample=Image.BICUBIC
    )
    left, top = int(round(x)) - pad, int(round(baseline)) - local
    for mask in masks:
        box = (left, top, left + width, top + height)
        region = mask.crop(box)
        mask.paste(ImageChops.lighter(region, sheared), box)


def paint(lines, typesetter: Typesetter, look: Look, align: str = "center") -> Tile:
    """Draw laid-out lines with their outline, shadow and box.

    Args:
        lines: Lines of cells, from :func:`set_cue`.
        typesetter: The typesetter they were set with.
        look: How they are drawn.
        align: ``left``, ``center`` or ``right``, how the lines sit against each other.

    Returns:
        The :class:`Tile`. Its placing box runs round every line's box, whether or not the box
        is drawn, and round the shadow's offset.
    """
    size = typesetter.size
    outline = round(look.outline * size)
    shadow = round(look.shadow * size)
    blur = shadow / 2.0
    step = typesetter.ascent + typesetter.descent + outline
    runs = [typesetter.runs(line) for line in lines]
    widths = [sum(run.width for run in line) for line in runs]
    block = max(widths, default=0.0)
    pad_x = round(BOX_PAD_X * size) + outline
    pad_y = round(BOX_PAD_Y * size)

    starts, boxes = [], []
    for index, width in enumerate(widths):
        start = 0.0 if align == "left" else block - width if align == "right" else (block - width) / 2.0
        starts.append(start)
        boxes.append((start - pad_x, index * step - pad_y, start + width + pad_x, (index + 1) * step + pad_y))
    left = math.floor(min((box[0] for box in boxes), default=0))
    top = math.floor(min((box[1] for box in boxes), default=0))
    right = math.ceil(max((box[2] for box in boxes), default=0))
    bottom = math.ceil(max((box[3] for box in boxes), default=0))
    box_right, box_bottom = right, bottom
    if shadow and look.shadow_color[3] > 0:
        right += shadow + math.ceil(blur)
        bottom += shadow + math.ceil(blur)

    margin = outline + typesetter.thicken + shadow + math.ceil(3 * blur) + math.ceil(0.3 * size)
    margin += math.ceil(ITALIC_SLANT * (typesetter.ascent + outline))
    shift_x, shift_y = margin - left, margin - top
    width = right - left + 2 * margin
    height = bottom - top + 2 * margin

    box_mask = Image.new("L", (width, height), 0)
    if look.box != "none" and look.box_color[3] > 0:
        pen = ImageDraw.Draw(box_mask)
        rows = boxes if look.box == "lines" else [(left, top, box_right, box_bottom)]
        for x0, y0, x1, y1 in rows:
            pen.rectangle(
                (round(x0 + shift_x), round(y0 + shift_y), round(x1 + shift_x) - 1, round(y1 + shift_y) - 1),
                fill=255,
            )

    edge = Image.new("L", (width, height), 0)
    fills: dict = {}
    for index, line in enumerate(runs):
        x = shift_x + starts[index]
        baseline = shift_y + index * step + outline / 2.0 + typesetter.ascent
        for run in line:
            key = run.color if run.color is not None else look.fill
            fill = fills.setdefault(key, Image.new("L", (width, height), 0))
            if outline:
                _draw_run([edge], run, x + run.thicken, baseline, outline + run.thicken, ITALIC_SLANT)
            _draw_run([fill], run, x + run.thicken, baseline, run.thicken, ITALIC_SLANT)
            if run.underline and run.text.strip():
                drop = baseline + round(UNDERLINE_DROP * size)
                weight = max(1, round(UNDERLINE_WEIGHT * size))
                bar = (round(x), round(drop), round(x + run.width) - 1, round(drop) + weight - 1)
                ImageDraw.Draw(fill).rectangle(bar, fill=255)
                if outline:
                    ImageDraw.Draw(edge).rectangle(
                        (bar[0] - outline, bar[1] - outline, bar[2] + outline, bar[3] + outline), fill=255
                    )
            x += run.width

    layers = [(look.box_color, box_mask)]
    if shadow and look.shadow_color[3] > 0:
        silhouette = edge.copy()
        for mask in fills.values():
            silhouette = ImageChops.lighter(silhouette, mask)
        cast = Image.new("L", (width, height), 0)
        cast.paste(silhouette, (shadow, shadow))
        if blur > 0:
            cast = cast.filter(ImageFilter.GaussianBlur(blur))
        layers.append((look.shadow_color, cast))
    if outline:
        layers.append((look.outline_color, edge))
    layers.extend((key, mask) for key, mask in fills.items())
    color, cover = _over(layers, height, width)
    return Tile(color, cover, shift_x + left, shift_y + top, shift_x + right, shift_y + bottom)


def lay(frame: torch.Tensor, tile: Tile, x: int, y: int, alpha: float = 1.0) -> None:
    """Composite a tile onto one frame in place.

    Args:
        frame: ``(height, width, channels)``, 3 or 4 channels, values in 0 to 1.
        tile: The tile.
        x: Frame column of the tile's left edge.
        y: Frame row of its top edge.
        alpha: How much of the tile shows, 0.0 to 1.0.
    """
    if alpha <= 0.0:
        return
    height, width = int(frame.shape[0]), int(frame.shape[1])
    rows, cols = int(tile.cover.shape[0]), int(tile.cover.shape[1])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + cols), min(height, y + rows)
    if x1 <= x0 or y1 <= y0:
        return
    cover = tile.cover[y0 - y:y1 - y, x0 - x:x1 - x].to(frame.device, frame.dtype)
    color = tile.color[y0 - y:y1 - y, x0 - x:x1 - x].to(frame.device, frame.dtype)
    if alpha < 1.0:
        cover = cover * alpha
        color = color * alpha
    region = frame[y0:y1, x0:x1]
    region[..., :3] = region[..., :3] * (1.0 - cover).unsqueeze(-1) + color
    if frame.shape[-1] > 3:
        region[..., 3] = cover + region[..., 3] * (1.0 - cover)


def timecode_tile(text: str, template: str, typesetter: Typesetter, fill: tuple, box_color: tuple) -> Tile:
    """A timecode or frame number drawn in a box sized for ``template``.

    Args:
        text: What to draw.
        template: The widest text the box must hold, so the box keeps one size.
        typesetter: Measures at the window's size.
        fill: Text colour.
        box_color: Box colour; fully transparent draws none.

    Returns:
        The :class:`Tile`, its placing box the drawn box.
    """
    size = typesetter.size
    face = typesetter.regular
    pad_x, pad_y = round(0.35 * size), round(0.2 * size)
    wide = math.ceil(max(face.getlength(template), face.getlength(text)))
    width = wide + 2 * pad_x
    height = typesetter.ascent + typesetter.descent + 2 * pad_y
    box = Image.new("L", (width, height), 255)
    glyphs = Image.new("L", (width, height), 0)
    shown, shown_face = draw.shown_line(text, face)
    ImageDraw.Draw(glyphs).text(
        (width / 2.0, pad_y + typesetter.ascent), shown, font=shown_face, fill=255, anchor="ms"
    )
    color, cover = _over([(box_color, box), (fill, glyphs)], height, width)
    return Tile(color, cover, 0, 0, width, height)


def _line_x(align: str, left: float, right: float, width: float) -> float:
    """Left edge of a line ``width`` wide set inside ``[left, right]``."""
    if align == "left":
        return left
    if align == "right":
        return right - width
    return (left + right - width) / 2.0


def _write(mask, runs, x: float, baseline: float) -> None:
    """Draw a line's runs into a mask, its advance starting at ``x``."""
    for run in runs:
        _draw_run([mask], run, x + run.thicken, baseline, run.thicken, ITALIC_SLANT)
        x += run.width


def name_graphic(
    style: str,
    name: str,
    title: str,
    name_set: Typesetter,
    title_set: Typesetter,
    align: str,
    fill: tuple,
    title_fill: tuple,
    back: tuple,
    accent: tuple,
    span: int = 0,
) -> tuple:
    """A lower third drawn as a backing tile and a text tile of one size.

    Args:
        style: One of :data:`GRAPHIC_STYLES`.
        name: The first line, drawn bold.
        title: The second line; empty draws the name alone.
        name_set: Typesetter for the name.
        title_set: Typesetter for the title.
        align: ``left``, ``center`` or ``right``, the edge the text and any stripe sit against.
        fill: Name colour.
        title_fill: Title colour.
        back: Bar, box or card colour; the shadow colour for ``underline`` and ``minimal``.
        accent: Stripe, box, rule or title colour, by style.
        span: Width of a ``banner`` in pixels; 0 fits it to its text.

    Returns:
        ``(back, text, lead)``: the two tiles, their placing box the graphic without its
        shadow, and the width of the stripe a wipe opens from, 0 for none.
    """
    size = name_set.size
    name_runs = name_set.runs([(character, (True, False, False, None)) for character in name])
    title_runs = title_set.runs([(character, (False, False, False, None)) for character in title]) if title else []
    name_width = sum(run.width for run in name_runs)
    title_width = sum(run.width for run in title_runs)
    text_width = math.ceil(max(name_width, title_width))
    name_height = name_set.ascent + name_set.descent
    title_height = title_set.ascent + title_set.descent if title else 0
    gap = round(0.1 * size)
    stripe = max(2, round(0.12 * size))
    lead = 0
    shadow = 0
    boxes: list = []
    lines: list = []

    if style == "split":
        pad_x, pad_y = round(0.45 * size), round(0.2 * size)
        title_pad = round(0.75 * pad_x)
        name_box = (math.ceil(name_width) + 2 * pad_x, name_height + 2 * pad_y)
        title_box = (math.ceil(title_width) + 2 * title_pad, title_height + 2 * round(0.16 * size)) if title else (0, 0)
        width = max(name_box[0], title_box[0])
        height = name_box[1] + title_box[1]
        name_left = _line_x(align, 0, width, name_box[0])
        boxes.append((accent, (name_left, 0, name_left + name_box[0], name_box[1]), 0))
        lines.append((name_runs, fill, name_left + pad_x, pad_y + name_set.ascent))
        if title:
            title_left = _line_x(align, 0, width, title_box[0])
            boxes.append((back, (title_left, name_box[1], title_left + title_box[0], height), 0))
            lines.append((title_runs, title_fill, title_left + title_pad, name_box[1] + round(0.16 * size) + title_set.ascent))
    elif style == "banner":
        pad_x, pad_y = round(0.5 * size), round(0.28 * size)
        block = math.ceil(name_width) + 2 * pad_x
        tail = math.ceil(title_width) + 2 * pad_x if title else 0
        width = max(int(span), block + tail)
        height = max(name_height, title_height) + 2 * pad_y
        middle = height / 2.0
        boxes.append((back, (0, 0, width, height), 0))
        boxes.append((accent, (0, height - stripe, width, height), 0))
        block_left = width - block if align == "right" else 0
        boxes.append((accent, (block_left, 0, block_left + block, height), 0))
        name_base = middle + (name_set.ascent - name_set.descent) / 2.0
        lines.append((name_runs, fill, block_left + pad_x, name_base))
        if title:
            title_base = middle + (title_set.ascent - title_set.descent) / 2.0
            title_x = block + pad_x if align != "right" else width - block - pad_x - title_width
            lines.append((title_runs, title_fill, title_x, title_base))
    elif style == "card":
        pad_x, pad_y = round(0.7 * size), round(0.42 * size)
        width = text_width + 2 * pad_x
        height = 2 * pad_y + name_height + (gap + title_height if title else 0)
        boxes.append((back, (0, 0, width, height), min(round(0.45 * size), height // 2)))
        lines.append((name_runs, fill, _line_x(align, pad_x, width - pad_x, name_width), pad_y + name_set.ascent))
        if title:
            title_color = accent[:3] + (round(accent[3] * title_fill[3] / 255.0),)
            lines.append((
                title_runs, title_color, _line_x(align, pad_x, width - pad_x, title_width),
                pad_y + name_height + gap + title_set.ascent,
            ))
    elif style in ("underline", "minimal"):
        rule = max(2, round(0.08 * size)) if style == "underline" else 0
        above = round(0.16 * size) if rule else 0
        below = round(0.22 * size) if rule else gap
        width = text_width
        height = name_height + (above + rule + below + title_height if title else above + rule)
        lines.append((name_runs, fill, _line_x(align, 0, width, name_width), name_set.ascent))
        if rule:
            length = max(math.ceil(name_width), round(0.5 * text_width))
            rule_left = _line_x(align, 0, width, length)
            boxes.append((accent, (rule_left, name_height + above, rule_left + length, name_height + above + rule), 0))
        if title:
            lines.append((
                title_runs, title_fill, _line_x(align, 0, width, title_width),
                name_height + above + rule + below + title_set.ascent,
            ))
        shadow = max(1, round(0.05 * size))
    else:
        lead = stripe + round(0.55 * size)
        trail = round(0.7 * size)
        top_pad, bottom_pad = round(0.32 * size), round(0.38 * size)
        if align == "center":
            lead = trail
        width = lead + text_width + trail
        height = top_pad + name_height + (gap + title_height if title else 0) + bottom_pad
        if align == "center":
            height += stripe
        boxes.append((back, (0, 0, width, height), 0))
        if align == "center":
            boxes.append((accent, (0, height - stripe, width, height), 0))
        else:
            stripe_x = 0 if align == "left" else width - stripe
            boxes.append((accent, (stripe_x, 0, stripe_x + stripe, height), 0))
        left, right = (lead, width - trail) if align == "left" else (trail, width - lead) if align == "right" else (lead, width - trail)
        lines.append((name_runs, fill, _line_x(align, left, right, name_width), top_pad + name_set.ascent))
        if title:
            lines.append((
                title_runs, title_fill, _line_x(align, left, right, title_width),
                top_pad + name_height + gap + title_set.ascent,
            ))
        lead = stripe if align != "center" else 0

    blur = 2.0 * shadow
    glow = GLOW_RADIUS * size
    margin = (shadow + math.ceil(3 * glow)) if shadow else 0
    margin += math.ceil(ITALIC_SLANT * name_set.ascent) if shadow else 0
    full = (int(width) + 2 * margin, int(height) + 2 * margin)
    back_layers = []
    for color, (x0, y0, x1, y1), radius in boxes:
        mask = Image.new("L", full, 0)
        corners = (round(x0) + margin, round(y0) + margin, round(x1) + margin - 1, round(y1) + margin - 1)
        if radius:
            ImageDraw.Draw(mask).rounded_rectangle(corners, radius=radius, fill=255)
        else:
            ImageDraw.Draw(mask).rectangle(corners, fill=255)
        back_layers.append((color, mask))
    text_layers = []
    for runs, color, x, baseline in lines:
        mask = Image.new("L", full, 0)
        _write(mask, runs, x + margin, baseline + margin)
        text_layers.append((color, mask))
    if shadow:
        silhouette = Image.new("L", full, 0)
        for _color, mask in back_layers + text_layers:
            silhouette = ImageChops.lighter(silhouette, mask)
        cast = Image.new("L", full, 0)
        cast.paste(silhouette, (shadow, shadow))
        cast = cast.filter(ImageFilter.GaussianBlur(blur))
        halo = silhouette.filter(ImageFilter.MaxFilter(2 * max(1, round(0.08 * size)) + 1))
        halo = halo.filter(ImageFilter.GaussianBlur(glow))
        back_layers[:0] = [(back[:3] + (round(back[3] * GLOW_STRENGTH),), halo), (back, cast)]
    back_color, back_cover = _over(back_layers, full[1], full[0])
    text_color, text_cover = _over(text_layers, full[1], full[0])
    box = (margin, margin, margin + int(width), margin + int(height))
    return Tile(back_color, back_cover, *box), Tile(text_color, text_cover, *box), lead


def _columns(width: int, start: float, stop: float) -> torch.Tensor:
    """Coverage of each column by the span ``[start, stop)``, fractional at both edges."""
    index = torch.arange(width, dtype=torch.float32)
    return ((torch.clamp(stop - index, 0.0, 1.0) - torch.clamp(start - index, 0.0, 1.0))).clamp_(0.0, 1.0)


def _shifted(tile: Tile, shift: float, axis: int = 1) -> Tile:
    """A tile's content moved by a fraction of a pixel or more along ``axis``, edges cut."""
    whole = math.floor(shift)
    part = shift - whole

    def moved(values: torch.Tensor, by: int) -> torch.Tensor:
        out = torch.zeros_like(values)
        length = values.shape[axis]
        if by >= 0:
            if by < length:
                out.narrow(axis, by, length - by).copy_(values.narrow(axis, 0, length - by))
        elif -by < length:
            out.narrow(axis, 0, length + by).copy_(values.narrow(axis, -by, length + by))
        return out

    color = moved(tile.color, whole) * (1.0 - part) + moved(tile.color, whole + 1) * part
    cover = moved(tile.cover, whole) * (1.0 - part) + moved(tile.cover, whole + 1) * part
    return Tile(color, cover, tile.left, tile.top, tile.right, tile.bottom)


def lay_bar(
    frame: torch.Tensor,
    bar: Tile,
    text: Tile,
    lead: int,
    x: int,
    y: int,
    progress: float,
    align: str,
    motion: str,
    rise: int = 1,
) -> None:
    """Composite a lower third onto one frame, part way through its entrance or exit.

    Args:
        frame: ``(height, width, channels)``, values in 0 to 1.
        bar: The backing tile, from :func:`name_graphic`.
        text: The text tile.
        lead: Width of the stripe a wipe opens from.
        x: Frame column of the placing box's left edge.
        y: Frame row of its top edge.
        progress: 0.0 hidden to 1.0 fully shown, already eased.
        align: ``left``, ``right`` or ``center``, the edge a wipe opens from; ``center`` opens
            from the middle.
        motion: ``slide`` wipes the graphic open and slides the text in with it; ``rise``
            moves it into place by half its height as it fades in; ``fade`` fades it.
        rise: 1 rises from below, -1 drops from above.
    """
    if progress <= 0.0:
        return
    left, top = x - bar.left, y - bar.top
    if motion == "fade" or progress >= 1.0:
        lay(frame, _stacked(bar, text), left, top, progress)
        return
    if motion == "rise":
        travel = (1.0 - progress) * 0.5 * bar.height * rise
        lay(frame, _shifted(_stacked(bar, text), travel, axis=0), left, top, progress)
        return
    width = int(bar.cover.shape[1])
    inner = bar.right - bar.left
    open_ = lead + (inner - lead) * progress
    if align == "left":
        window = _columns(width, 0.0, bar.left + open_)
        inside = _columns(width, float(bar.left + lead), bar.left + open_)
        travel = -(1.0 - progress) * min(inner * 0.35, 4.0 * max(lead, 2) + 0.25 * inner)
    elif align == "right":
        window = _columns(width, bar.right - open_, float(width))
        inside = _columns(width, bar.right - open_, float(bar.right - lead))
        travel = (1.0 - progress) * min(inner * 0.35, 4.0 * max(lead, 2) + 0.25 * inner)
    else:
        middle = (bar.left + bar.right) / 2.0
        window = _columns(width, middle - open_ / 2.0, middle + open_ / 2.0)
        inside = window
        travel = 0.0
    if bar.left:
        window = torch.where(torch.arange(width) < bar.left, window[min(width - 1, bar.left)], window)
        window = torch.where(torch.arange(width) >= bar.right, window[max(0, bar.right - 1)], window)
    moved = _shifted(text, travel) if travel else text
    whole = _stacked(_windowed(bar, window), _windowed(moved, inside))
    lay(frame, whole, left, top, min(1.0, progress * 1.6))


def _windowed(tile: Tile, window: torch.Tensor) -> Tile:
    """A tile with each column scaled by ``window``, 0 to 1 per column."""
    return Tile(tile.color * window.unsqueeze(-1), tile.cover * window, tile.left, tile.top, tile.right, tile.bottom)


def _stacked(under: Tile, over: Tile) -> Tile:
    """One tile laid over another of the same size, as one tile."""
    keep = 1.0 - over.cover
    return Tile(
        over.color + under.color * keep.unsqueeze(-1),
        over.cover + under.cover * keep,
        under.left, under.top, under.right, under.bottom,
    )
