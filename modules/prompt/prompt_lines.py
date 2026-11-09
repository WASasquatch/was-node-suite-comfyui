"""Prompt text written one part per line: which lines are kept, listed or joined.

Lines are prompts, notes, blanks, or off by a leading ``~``. Notes start with ``#`` or ``//``,
or sit in a ``/* ... */`` block.
"""

from __future__ import annotations

import re

__all__ = [
    "BLANK",
    "JOINS",
    "NOTE",
    "OFF",
    "OFF_MARK",
    "PROMPT",
    "classify",
    "join_lines",
    "kept_lines",
    "list_lines",
    "split_lines",
]

#: What a line is.
PROMPT = "prompt"
OFF = "off"
NOTE = "note"
BLANK = "blank"

#: The first character of a line that is switched off.
OFF_MARK = "~"

#: What a one-line note starts with.
LINE_NOTES = ("#", "//")

#: What opens and closes a note running over several lines.
BLOCK_OPEN = "/*"
BLOCK_CLOSE = "*/"

#: A line break as a text box writes one.
BREAKS = re.compile(r"\r\n|\r|\n")

#: How kept lines are joined into one text, by name.
JOINS = {"line break": "\n", "space": " ", "comma": ", "}


def split_lines(text: str) -> list[str]:
    """Cut text into its lines.

    Args:
        text: The text, with any of ``\\n``, ``\\r\\n`` or ``\\r`` between lines.

    Returns:
        One entry per line. The empty line after a final line break is not one.
    """
    lines = BREAKS.split(text or "")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines


def classify(text: str) -> list[tuple[str, str]]:
    """Name what each line is.

    Args:
        text: The text, one part per line.

    Returns:
        ``(kind, line)`` per line, ``kind`` one of :data:`PROMPT`, :data:`OFF`, :data:`NOTE` and
        :data:`BLANK`, and ``line`` as written.
    """
    kinds = []
    in_block = False
    for line in split_lines(text):
        stripped = line.strip()
        if in_block:
            kinds.append((NOTE, line))
            if BLOCK_CLOSE in line:
                in_block = False
        elif not stripped:
            kinds.append((BLANK, line))
        elif stripped.startswith(BLOCK_OPEN):
            kinds.append((NOTE, line))
            in_block = BLOCK_CLOSE not in stripped[len(BLOCK_OPEN):]
        elif stripped.startswith(LINE_NOTES):
            kinds.append((NOTE, line))
        elif stripped.startswith(OFF_MARK):
            kinds.append((OFF, line))
        else:
            kinds.append((PROMPT, line))
    return kinds


def kept_lines(text: str, skip_blank_lines: bool = True, skip_comment_lines: bool = True) -> list[str]:
    """The lines that are kept, as written.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines.
        skip_comment_lines: Leave out notes.

    Returns:
        The kept lines in order. A line switched off is never one.
    """
    kept = []
    for kind, line in classify(text):
        if kind == OFF:
            continue
        if kind == NOTE and skip_comment_lines:
            continue
        if kind == BLANK and skip_blank_lines:
            continue
        kept.append(line)
    return kept


def list_lines(text: str, skip_blank_lines: bool = True, skip_comment_lines: bool = True) -> list[str]:
    """The kept lines, each trimmed, one entry per line.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines rather than giving each as ``""``.
        skip_comment_lines: Leave out notes rather than giving each as written.

    Returns:
        The entries in line order.
    """
    return [line.strip() for line in kept_lines(text, skip_blank_lines, skip_comment_lines)]


def join_lines(
    text: str, skip_blank_lines: bool = False, skip_comment_lines: bool = True, join_with: str = "line break"
) -> str:
    """The kept lines as one text.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines.
        skip_comment_lines: Leave out notes.
        join_with: A key of :data:`JOINS`. ``line break`` keeps each line's indent and blank
            lines; ``space`` and ``comma`` trim each line and drop empty ones, and ``comma`` drops
            a line's own trailing comma first.

    Returns:
        The joined text, empty when nothing is kept.

    Raises:
        ValueError: ``join_with`` is not a key of :data:`JOINS`.
    """
    if join_with not in JOINS:
        raise ValueError(f"join_with is {join_with!r}; it takes one of {', '.join(JOINS)}.")
    kept = kept_lines(text, skip_blank_lines, skip_comment_lines)
    if join_with == "line break":
        return JOINS[join_with].join(line.rstrip() for line in kept)
    parts = [line.strip() for line in kept]
    if join_with == "comma":
        parts = [part.rstrip(",").rstrip() for part in parts]
    return JOINS[join_with].join(part for part in parts if part)
