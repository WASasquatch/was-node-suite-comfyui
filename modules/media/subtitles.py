"""SubRip and WebVTT subtitles read into timed, styled cues, and cue times placed on frames.

Times are exact fractions of a second. A cue covers frame ``n`` when
``start <= n / rate < end``.
"""

from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = [
    "EXTENSIONS",
    "MAX_CUES",
    "MAX_CUE_CHARS",
    "MAX_FILE_BYTES",
    "REASONS",
    "Cue",
    "Span",
    "Track",
    "clip_rate",
    "decode",
    "fade_alpha",
    "fade_frames",
    "frame_span",
    "markup",
    "parse",
    "rate_of",
    "read_file",
    "seconds",
]

#: Suffixes a subtitle file menu lists.
EXTENSIONS = (".srt", ".vtt")

#: Largest subtitle file or text read, in bytes. A feature film's SubRip file is about 100 KB.
MAX_FILE_BYTES = 4 * 1024 * 1024

#: Cues read from one file. Timing lines past this are counted and not read.
MAX_CUES = 20000

#: Characters of one cue's text, tags included. A longer cue is skipped.
MAX_CUE_CHARS = 2000

#: Lines of one cue's text. A cue with more is skipped.
MAX_CUE_LINES = 12

#: A timing line: start and end as ``[hh:]mm:ss,mmm`` or ``[hh:]mm:ss.mmm``, then WebVTT settings.
TIMING = re.compile(
    r"^\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*"
    r"(?:(\d{1,3}):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})(.*)$"
)

#: Arrows a mistyped timing line carries in place of ``-->``.
LOOSE_ARROWS = ("->", "=>", "\u2192", "\u2013>", "\u2014>")

#: Two clock times on one line, as a mistyped timing line holds.
LOOSE_TIMES = re.compile(r"\d{1,2}:\d{2}\D[^:]{0,40}\d{1,2}:\d{2}")

#: Longest line read as a mistyped timing line.
LOOSE_LENGTH = 120

#: An opening or closing tag such as ``<i>``, ``</b>``, ``<c.yellow>`` or ``<v Anna>``, or a
#: WebVTT timestamp tag such as ``<00:01.500>``.
TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:\.[^\s>.]+)*)([^>]*)>|<\d[^>]*>")

#: A SubRip override block such as ``{\an8}``.
OVERRIDE = re.compile(r"\{\\([^}]{0,200})\}")

#: The alignment inside an override block, numbered as a keypad: 7 to 9 top, 1 to 3 bottom.
ALIGNMENT = re.compile(r"(?:^|\\)an?(\d)")

#: A colour attribute inside a ``<font>`` tag.
FONT_COLOR = re.compile(r"""color\s*=\s*["']?\s*(#?[0-9A-Za-z]{1,32})""", re.IGNORECASE)

#: The colour classes WebVTT defines for ``<c.name>``.
CLASS_COLORS = {
    "white": (255, 255, 255, 255),
    "lime": (0, 255, 0, 255),
    "cyan": (0, 255, 255, 255),
    "red": (255, 0, 0, 255),
    "yellow": (255, 255, 0, 255),
    "magenta": (255, 0, 255, 255),
    "blue": (0, 0, 255, 255),
    "black": (0, 0, 0, 255),
}

#: Blocks a WebVTT file holds besides cues.
VTT_BLOCKS = ("NOTE", "STYLE", "REGION")

#: Why a cue was skipped, keyed by a short name.
REASONS = {
    "timing": "its timing line could not be read",
    "range": "a minute or second of its time is 60 or more",
    "order": "it ends before it starts",
    "long": f"its text is longer than {MAX_CUE_CHARS} characters or {MAX_CUE_LINES} lines",
    "empty": "it holds no text",
}


@dataclass(frozen=True)
class Span:
    """A piece of one line drawn in one style.

    Attributes:
        text: The characters, in typed order.
        bold: Drawn bold.
        italic: Drawn slanted.
        underline: Drawn underlined.
        color: ``(red, green, blue, alpha)`` set by the cue, or None for the node's colour.
    """

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: tuple | None = None


@dataclass(frozen=True)
class Cue:
    """One subtitle: when it shows, what it says and where it sits.

    Attributes:
        start: Seconds the cue appears at.
        end: Seconds the cue is gone by.
        lines: The lines, each a tuple of :class:`Span`.
        place: ``"top"``, ``"bottom"``, or ``""`` where the cue names no place.
        line_number: Line of the source its timing line sits on, counting from 1.
    """

    start: Fraction
    end: Fraction
    lines: tuple
    place: str = ""
    line_number: int = 0


@dataclass
class Track:
    """Every cue a source held, and what was skipped.

    Attributes:
        cues: The cues read, ordered by start time.
        skipped: ``(line number, reason key)`` for each cue left out, reason keys from
            :data:`REASONS`.
        unread: Timing lines past :data:`MAX_CUES`, counted and not read.
        kind: ``"SubRip"`` or ``"WebVTT"``.
        encoding: The text encoding the source was read in.
    """

    cues: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    unread: int = 0
    kind: str = "SubRip"
    encoding: str = "utf-8"


def decode(data: bytes) -> tuple[str, str]:
    """Text of a subtitle file, and the encoding it was read in.

    Args:
        data: The file's bytes.

    Returns:
        ``(text, encoding)``. UTF-16 with a byte order mark, then UTF-8, then Windows-1252,
        which reads any byte.
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "windows-1252"


def read_file(path: str) -> Track:
    """Read a subtitle file already resolved inside a permitted folder.

    Args:
        path: The file.

    Returns:
        The :class:`Track` it holds.

    Raises:
        ValueError: The file is larger than :data:`MAX_FILE_BYTES`.
        OSError: The file could not be read.
    """
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"`{os.path.basename(path)}` is {size / 1048576:.1f} MB, and a subtitle file is read "
            f"up to {MAX_FILE_BYTES // 1048576} MB. Split it into shorter files, one per reel or "
            f"episode."
        )
    with open(path, "rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(
            f"`{os.path.basename(path)}` grew past {MAX_FILE_BYTES // 1048576} MB while it was "
            f"read. Split it into shorter files."
        )
    text, encoding = decode(data)
    track = parse(text)
    track.encoding = encoding
    return track


def _time(hours, minutes, secs, fraction) -> Fraction | None:
    """Seconds from the four groups of a timing, or None when a field is out of range."""
    minutes, secs = int(minutes), int(secs)
    if minutes >= 60 or secs >= 60:
        return None
    millis = int(fraction.ljust(3, "0"))
    return Fraction(int(hours or 0) * 3600 + minutes * 60 + secs) + Fraction(millis, 1000)


def _vtt_place(settings: str) -> str:
    """``"top"`` or ``"bottom"`` from a WebVTT ``line:`` setting, or ``""`` where none is read."""
    for token in settings.split():
        if not token.startswith("line:"):
            continue
        value = token[5:].split(",")[0].strip()
        try:
            if value.endswith("%"):
                return "top" if float(value[:-1]) < 50.0 else "bottom"
            return "top" if int(value) >= 0 else "bottom"
        except ValueError:
            return ""
    return ""


def parse(text: str) -> Track:
    """Read SubRip or WebVTT text into cues.

    Args:
        text: The whole source. WebVTT is recognised by its ``WEBVTT`` first line.

    Returns:
        The :class:`Track`. Cues whose timing cannot be read, that end before they start or
        that hold no text are listed in ``skipped``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = text.split("\n")
    first = next((line.strip() for line in lines if line.strip()), "")
    vtt = first.startswith("WEBVTT")
    track = Track(kind="WebVTT" if vtt else "SubRip")

    open_cue = None
    held: list[str] = []
    between: list[str] = []
    state = "between"

    def close(trailing: list[str]) -> None:
        nonlocal open_cue
        if open_cue is None:
            return
        body = held + trailing
        start, end, place, number = open_cue
        open_cue = None
        joined = "\n".join(body)
        if len(joined) > MAX_CUE_CHARS or len(body) > MAX_CUE_LINES:
            track.skipped.append((number, "long"))
            return
        shown, override = markup(joined)
        if not shown:
            track.skipped.append((number, "empty"))
            return
        track.cues.append(Cue(start, end, shown, override or place, number))

    for index, raw in enumerate(lines):
        line = raw.strip()
        if "-->" in line or _mistyped(line):
            # Lines between two SubRip cues, bar the counter, continue the cue before them.
            trailing = []
            if not vtt:
                if state == "text" and len(held) > 1 and held[-1].isdigit():
                    held.pop()
                trailing = _continuation(between)
            close(trailing)
            held, between = [], []
            if len(track.cues) + len(track.skipped) >= MAX_CUES:
                track.unread += 1
                state = "skip"
                continue
            found = TIMING.match(line)
            if found is None:
                track.skipped.append((index + 1, "timing"))
                state = "skip"
                continue
            start = _time(*found.group(1, 2, 3, 4))
            end = _time(*found.group(5, 6, 7, 8))
            if start is None or end is None:
                track.skipped.append((index + 1, "range"))
                state = "skip"
                continue
            if end <= start:
                track.skipped.append((index + 1, "order"))
                state = "skip"
                continue
            place = _vtt_place(found.group(9)) if vtt else ""
            open_cue = (start, end, place, index + 1)
            state = "text"
            continue
        if state == "text":
            if line:
                held.append(line)
            else:
                state = "between"
        elif state == "skip":
            if not line:
                state = "between"
        elif line:
            if vtt and any(line.startswith(block) for block in VTT_BLOCKS):
                state = "skip"
            elif not vtt:
                between.append(line)
    close([] if vtt else _continuation(between))
    track.cues.sort(key=lambda cue: (cue.start, cue.line_number))
    return track


def _mistyped(line: str) -> bool:
    """Whether a line is a timing line written with the wrong arrow, such as ``->``."""
    if len(line) > LOOSE_LENGTH or not any(arrow in line for arrow in LOOSE_ARROWS):
        return False
    return LOOSE_TIMES.search(line) is not None


def _continuation(between: list[str]) -> list[str]:
    """Lines found between two SubRip cues, less the counter that opens the next one."""
    if between and between[-1].isdigit():
        return between[:-1]
    return list(between)


def _plain(text: str) -> str:
    """A text piece with entities read and SubRip escapes turned into breaks and spaces."""
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", "\u00a0")
    return html.unescape(text)


def _color(value: str):
    """``(red, green, blue, alpha)`` from a colour name or hex value, or None."""
    from PIL import ImageColor

    candidate = value.strip()
    if re.fullmatch(r"[0-9A-Fa-f]{6}", candidate):
        candidate = "#" + candidate
    try:
        return tuple(ImageColor.getcolor(candidate, "RGBA"))
    except ValueError:
        return None


def markup(text: str) -> tuple[tuple, str]:
    """Split a cue's text into styled lines.

    Args:
        text: The cue's text, line breaks included. ``<i>``, ``<b>``, ``<u>``,
            ``<font color>`` and WebVTT ``<c.colour>`` set the style; every other tag is
            dropped and its text kept, and ``<rt>`` text is dropped.

    Returns:
        ``(lines, place)``: the lines as tuples of :class:`Span`, empty lines removed, and
        ``"top"``, ``"bottom"`` or ``""`` from an ``{\\an}`` override.
    """
    place = ""
    for block in OVERRIDE.finditer(text):
        found = ALIGNMENT.search(block.group(1))
        if found:
            digit = int(found.group(1))
            place = "top" if digit >= 7 else "bottom" if digit <= 3 else place
    text = OVERRIDE.sub("", text)

    bold = italic = underline = ruby = 0
    colors: list = []
    pieces: list[tuple[str, tuple]] = []

    def style():
        return (bold > 0, italic > 0, underline > 0, colors[-1] if colors else None)

    position = 0
    for found in TAG.finditer(text):
        if found.start() > position and not ruby:
            pieces.append((_plain(text[position:found.start()]), style()))
        position = found.end()
        name = (found.group(2) or "").lower()
        if not name:
            continue
        closing = found.group(1) == "/"
        step = -1 if closing else 1
        if name == "b":
            bold = max(0, bold + step)
        elif name == "i":
            italic = max(0, italic + step)
        elif name == "u":
            underline = max(0, underline + step)
        elif name == "rt":
            ruby = max(0, ruby + step)
        elif name == "br":
            pieces.append(("\n", style()))
        elif name in ("font", "c"):
            if closing:
                if colors:
                    colors.pop()
                continue
            chosen = None
            if name == "font":
                attribute = FONT_COLOR.search(found.group(4) or "")
                chosen = _color(attribute.group(1)) if attribute else None
            else:
                for label in (found.group(3) or "").split("."):
                    if label.lower() in CLASS_COLORS:
                        chosen = CLASS_COLORS[label.lower()]
            colors.append(chosen if chosen is not None else (colors[-1] if colors else None))
    if position < len(text) and not ruby:
        pieces.append((_plain(text[position:]), style()))

    lines: list[list[Span]] = [[]]
    for piece, (is_bold, is_italic, is_underline, color) in pieces:
        for number, part in enumerate(piece.split("\n")):
            if number:
                lines.append([])
            if part:
                lines[-1].append(Span(part, is_bold, is_italic, is_underline, color))

    shown = []
    for spans in lines:
        merged: list[Span] = []
        for span in spans:
            if merged and _same(merged[-1], span):
                merged[-1] = Span(merged[-1].text + span.text, span.bold, span.italic, span.underline, span.color)
            else:
                merged.append(span)
        merged = _trimmed(merged)
        if merged:
            shown.append(tuple(merged))
    return tuple(shown), place


def _same(first: Span, second: Span) -> bool:
    """Whether two spans share a style."""
    return (first.bold, first.italic, first.underline, first.color) == (
        second.bold, second.italic, second.underline, second.color
    )


def _trimmed(spans: list) -> list:
    """A line's spans with the space at either end removed and inner runs of space collapsed."""
    cleaned = []
    for span in spans:
        collapsed = re.sub(r"[ \t]+", " ", span.text.replace("\t", " "))
        cleaned.append(Span(collapsed, span.bold, span.italic, span.underline, span.color))
    while cleaned and not cleaned[0].text.strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].text.strip():
        cleaned.pop()
    if not cleaned:
        return []
    head, tail = cleaned[0], cleaned[-1]
    cleaned[0] = Span(head.text.lstrip(), head.bold, head.italic, head.underline, head.color)
    tail = cleaned[-1]
    cleaned[-1] = Span(tail.text.rstrip(), tail.bold, tail.italic, tail.underline, tail.color)
    return [span for span in cleaned if span.text]


#: Frame rates of the 1000/1001 family, as the whole rate each one runs just under.
NTSC_BASES = (24, 30, 48, 60, 120)


def rate_of(fps: float) -> Fraction:
    """A frame rate as an exact fraction.

    Args:
        fps: Frames a second, such as ``25``, ``23.976`` or ``29.97``.

    Returns:
        ``24000/1001``, ``30000/1001`` and the rest of that family for the rates written
        23.976, 29.97, 47.952, 59.94 and 119.88; otherwise the value itself, to a millionth.

    Raises:
        ValueError: The rate is not above 0.
    """
    value = float(fps)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("a frame rate above 0 is needed")
    for base in NTSC_BASES:
        exact = Fraction(base * 1000, 1001)
        if abs(value - float(exact)) < 0.002:
            return exact
    return Fraction(value).limit_denominator(1000000)


#: What a refusal for a missing frame rate asks for.
RATE_FIX = (
    "Wire Video Metadata's fps into it (Video Metadata reads the metadata output of Load Video "
    "(Advanced)), or type the clip's rate, such as 23.976, 25 or 29.97."
)


def clip_rate(fps, node: str) -> Fraction:
    """A node's fps input as an exact frame rate.

    Args:
        fps: The input's value.
        node: The node's display name, for the refusal.

    Returns:
        The rate, from :func:`rate_of`.

    Raises:
        ValueError: fps is missing, 0 or not a number.
    """
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0.0:
        shown = "missing" if fps is None else f"{fps}"
        raise ValueError(f"{node} needs the clip's frame rate, and fps is {shown}. {RATE_FIX}")
    return rate_of(value)


def seconds(value: float) -> Fraction:
    """Seconds from a widget value, exact to a millionth.

    Args:
        value: Seconds, such as ``-10.01``.

    Returns:
        The nearest fraction with a denominator of a million or less.
    """
    return Fraction(float(value)).limit_denominator(1000000)


def frame_span(start: Fraction, end: Fraction, rate: Fraction, offset: Fraction = Fraction(0)) -> tuple[int, int]:
    """Frames a span of time covers.

    Args:
        start: Seconds the span opens at.
        end: Seconds it closes at.
        rate: Frames a second.
        offset: Seconds added to both ends.

    Returns:
        ``(first, stop)``: frame ``n`` is covered when ``first <= n < stop``, which is when
        ``start + offset <= n / rate < end + offset``.
    """
    first = math.ceil((start + offset) * rate)
    stop = math.ceil((end + offset) * rate)
    return first, max(first, stop)


def fade_frames(fade: float, rate: Fraction) -> int:
    """Frames a fade of ``fade`` seconds lasts at ``rate``, to the nearest frame."""
    return max(0, math.floor(seconds(max(0.0, float(fade))) * rate + Fraction(1, 2)))


def fade_alpha(frame: int, first: int, stop: int, fade: int) -> float:
    """How much of a span shows on one frame.

    Args:
        frame: The frame.
        first: The span's first frame.
        stop: The frame after its last.
        fade: Frames each fade lasts. 0 cuts in and out.

    Returns:
        0.0 outside the span. Inside it, ``(k + 1) / (fade + 1)`` on the ``k``-th frame from
        either end, and 1.0 past both fades.
    """
    if frame < first or frame >= stop:
        return 0.0
    if fade <= 0:
        return 1.0
    steps = fade + 1
    return min(1.0, (frame - first + 1) / steps, (stop - frame) / steps)
