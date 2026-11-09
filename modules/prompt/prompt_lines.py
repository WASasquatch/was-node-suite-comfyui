"""Prompt text written one part per line, resolved into the lines a node hands on.

Lines are prompts, ``$name = value`` terms, ``|`` versions, ``== name`` headings, notes and
blanks; ``~`` switches one off and ``{a|b}`` draws from a seed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import NamedTuple

__all__ = [
    "ALL",
    "BLANK",
    "COMBINES",
    "CYCLE",
    "DECREMENT",
    "EVERY",
    "FIXED",
    "INCREMENT",
    "JOINS",
    "LIST_LIMIT",
    "MAX_COMBINED",
    "MODES",
    "NOTE",
    "OFF_MARK",
    "PICKED_MARK",
    "PROMPT",
    "RANDOM",
    "SECTION",
    "SECTION_MARK",
    "TERM",
    "VERSION",
    "VERSION_MARK",
    "Choice",
    "Group",
    "Line",
    "classify",
    "combine",
    "draw",
    "expand",
    "glossary",
    "group_lines",
    "handed_on",
    "hidden_lines",
    "join_lines",
    "join_parts",
    "kept_lines",
    "list_entries",
    "list_lines",
    "list_seed",
    "parse_choices",
    "read_lists",
    "resolve",
    "split_lines",
    "stream",
    "tidy",
    "unescape",
]

#: What a line is.
PROMPT = "prompt"
TERM = "term"
VERSION = "version"
SECTION = "section"
NOTE = "note"
BLANK = "blank"

#: What a heading starts with.
SECTION_MARK = "=="

#: How several lists are put together into entries: line by line, the shorter ones starting
#: over, or every combination.
CYCLE = "cycle"
EVERY = "every combination"
COMBINES = (CYCLE, EVERY)

#: The most entries every combination may make.
MAX_COMBINED = 10000

#: The most lists one node reads, and the highest line a list's pick names.
LIST_LIMIT = 64
PICK_LIMIT = 999999

#: The first character of a line that is switched off, and the one after a version's mark.
OFF_MARK = "~"

#: What a version of the line above starts with, and what the version in use starts with.
VERSION_MARK = "|"
PICKED_MARK = ">"

#: How the versions of a line take turns from one run to the next.
FIXED = "fixed"
INCREMENT = "increment"
DECREMENT = "decrement"
RANDOM = "random"
ALL = "all"
MODES = (FIXED, INCREMENT, DECREMENT, RANDOM, ALL)

#: What a one-line note starts with.
LINE_NOTES = ("#", "//")

#: What opens and closes a note running over several lines.
BLOCK_OPEN = "/*"
BLOCK_CLOSE = "*/"

#: A line break as a text box writes one.
BREAKS = re.compile(r"\r\n|\r|\n")

#: How kept lines are joined into one text, by name.
JOINS = {"line break": "\n", "space": " ", "comma": ", "}

#: ``@mode`` and the space after it, at the start of a line.
MODE = re.compile(r"^@(fixed|increment|decrement|random|all)(?=\s|$)\s*")

#: ``$name = value``, the whole of a term's line after its marks.
TERM_LINE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(.*)$")

#: ``$name`` or ``${name}`` in a line, unless its dollar is escaped.
REFERENCE = re.compile(r"(?<!\\)\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")

#: ``2$$`` or ``1-3$$`` before a choice's first option.
COUNT = re.compile(r"^\s*([0-9]+)(?:\s*-\s*([0-9]+))?\s*\$\$")

#: ``3::`` or ``0.5::`` before an option.
WEIGHT = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)::")

#: ``{name}`` right after a dollar, which is a reference rather than a choice.
BRACED_NAME = re.compile(r"^\{[A-Za-z_][A-Za-z0-9_]*\}")

#: A character kept as text by a backslash.
ESCAPED = re.compile(r"\\([{}|$])")

#: What several drawn options are joined with when the choice names nothing else.
JOINER = ", "

#: How many terms deep a term's value is expanded.
TERM_DEPTH = 8

#: Arithmetic of the draw.
MASK = 0xFFFFFFFF
GOLDEN = 0x9E3779B9
SPAN = 4294967296


class Line(NamedTuple):
    """One line and what it is.

    Attributes:
        kind: One of :data:`PROMPT`, :data:`TERM`, :data:`VERSION`, :data:`SECTION`,
            :data:`NOTE` and :data:`BLANK`.
        line: The line as written.
        body: The text after its marks: a prompt, a term's value, a version or a heading.
        off: Switched off by :data:`OFF_MARK`.
        mode: One of :data:`MODES`, from the line's ``@mode``.
        picked: A version starting with :data:`PICKED_MARK`.
        name: A term's name.
        block: A line of a ``/* */`` note.
    """

    kind: str
    line: str
    body: str = ""
    off: bool = False
    mode: str = FIXED
    picked: bool = False
    name: str = ""
    block: bool = False


class Group(NamedTuple):
    """A prompt or term and the versions under it.

    Attributes:
        kind: :data:`PROMPT` or :data:`TERM`.
        versions: Line indices, the line itself first.
        mode: One of :data:`MODES`.
        off: The whole group is switched off.
        name: A term's name.
        pick: The position in ``versions`` in use.
        live: Positions that can be in use: the first, and every version not switched off.
    """

    kind: str
    versions: tuple
    mode: str
    off: bool
    name: str
    pick: int
    live: tuple


class Choice(NamedTuple):
    """One ``{a|b}`` choice.

    Attributes:
        low: Fewest options drawn.
        high: Most options drawn.
        joiner: What several drawn options are joined with.
        options: ``(weight, parts)`` per option, ``parts`` as :func:`parse_choices` gives them.
        literal: A brace pair with one option and no count, kept with its braces.
    """

    low: int
    high: int
    joiner: str
    options: tuple
    literal: bool


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


def _headed(line: str, stripped: str) -> Line:
    """Read a line that is a prompt or a term.

    Args:
        line: The line as written.
        stripped: The line without the space around it.

    Returns:
        The line as a :data:`PROMPT` or :data:`TERM`.
    """
    rest = stripped
    off = rest.startswith(OFF_MARK)
    if off:
        rest = rest[len(OFF_MARK):].lstrip()
    mode = FIXED
    found = MODE.match(rest)
    if found:
        mode = found.group(1)
        rest = rest[found.end():]
    term = TERM_LINE.match(rest)
    if term:
        return Line(TERM, line, term.group(2).strip(), off, mode, name=term.group(1))
    return Line(PROMPT, line, rest.strip(), off, mode)


def _section(line: str, stripped: str) -> Line | None:
    """Read a line that may be a heading.

    Args:
        line: The line as written.
        stripped: The line without the space around it.

    Returns:
        The line as a :data:`SECTION`, or None when it is not one.
    """
    rest = stripped
    off = rest.startswith(OFF_MARK)
    if off:
        rest = rest[len(OFF_MARK):].lstrip()
    if not rest.startswith(SECTION_MARK):
        return None
    return Line(SECTION, line, rest.strip("=").strip(), off)


def classify(text: str) -> list[Line]:
    """Name what each line is.

    Args:
        text: The text, one part per line.

    Returns:
        One :class:`Line` per line.
    """
    found = []
    in_block = False
    for line in split_lines(text):
        stripped = line.strip()
        if in_block:
            found.append(Line(NOTE, line, line, block=True))
            if BLOCK_CLOSE in line:
                in_block = False
        elif not stripped:
            found.append(Line(BLANK, line))
        elif stripped.startswith(BLOCK_OPEN):
            found.append(Line(NOTE, line, line, block=True))
            in_block = BLOCK_CLOSE not in stripped[len(BLOCK_OPEN):]
        elif stripped.startswith(LINE_NOTES):
            found.append(Line(NOTE, line, line))
        elif (section := _section(line, stripped)) is not None:
            found.append(section)
        elif stripped[0] in (VERSION_MARK, PICKED_MARK):
            rest = stripped[1:]
            off = rest.startswith(OFF_MARK)
            if off:
                rest = rest[len(OFF_MARK):]
            found.append(Line(VERSION, line, rest.strip(), off, picked=stripped[0] == PICKED_MARK))
        else:
            found.append(_headed(line, stripped))
    return found


def _group(lines: list[Line], indices: list[int]) -> Group:
    """Gather one line and its versions.

    Args:
        lines: Every line, from :func:`classify`.
        indices: The group's line indices, the line itself first.

    Returns:
        The group.
    """
    head = lines[indices[0]]
    live = tuple(
        position for position, index in enumerate(indices) if position == 0 or not lines[index].off
    )
    pick = next((position for position in live if position and lines[indices[position]].picked), 0)
    stray = head.kind == VERSION
    return Group(
        TERM if head.kind == TERM else PROMPT,
        tuple(indices),
        FIXED if stray else head.mode,
        head.off,
        head.name,
        pick,
        live,
    )


def group_lines(lines: list[Line]) -> list[Group]:
    """Gather each prompt and term with the versions written under it.

    Args:
        lines: Every line, from :func:`classify`.

    Returns:
        The groups in line order. A version with no prompt or term above it starts its own.
    """
    groups = []
    current: list[int] = []
    for index, line in enumerate(lines):
        if line.kind == VERSION and current:
            current.append(index)
            continue
        if current:
            groups.append(_group(lines, current))
            current = []
        if line.kind in (PROMPT, TERM, VERSION):
            current = [index]
    if current:
        groups.append(_group(lines, current))
    return groups


def _mix(value: int) -> int:
    """Scramble a 32-bit number.

    Args:
        value: Any integer; its low 32 bits are used.

    Returns:
        A 32-bit number.
    """
    value &= MASK
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & MASK
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & MASK
    value ^= value >> 16
    return value


def stream(seed: int, line: int) -> Callable[[], float]:
    """The draws one line takes, the same for the same seed and line.

    Args:
        seed: The seed; its low 32 bits are used.
        line: The line's index.

    Returns:
        A function answering the next draw in ``[0, 1)`` each call.
    """
    state = _mix((seed & MASK) ^ _mix(line + GOLDEN))

    def draw_next() -> float:
        nonlocal state
        state = (state + GOLDEN) & MASK
        return _mix(state) / SPAN

    return draw_next


def _closing(text: str, start: int) -> int:
    """Find the brace that closes the one at ``start``.

    Args:
        text: The text.
        start: Where a ``{`` stands.

    Returns:
        The index of its ``}``, or -1 when it is never closed.
    """
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _options(text: str) -> list[str]:
    """Split a choice's inside at each ``|`` not nested or escaped.

    Args:
        text: What stands between the braces.

    Returns:
        The options as written.
    """
    found = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "|" and depth == 0:
            found.append(text[start:index])
            start = index + 1
        index += 1
    found.append(text[start:])
    return found


def _choice(inside: str) -> Choice:
    """Read what stands between a choice's braces.

    Args:
        inside: The text between ``{`` and ``}``.

    Returns:
        The choice.
    """
    pieces = _options(inside)
    low = high = 1
    joiner = JOINER
    counted = COUNT.match(pieces[0])
    if counted:
        low = int(counted.group(1))
        high = int(counted.group(2) or counted.group(1))
        if high < low:
            low, high = high, low
        rest = pieces[0][counted.end():]
        named = rest.find("$$")
        if named >= 0:
            joiner = rest[:named]
            rest = rest[named + 2:]
        pieces[0] = rest
    options = []
    for piece in pieces:
        weighted = WEIGHT.match(piece)
        weight = float(weighted.group(1)) if weighted else 1.0
        options.append((weight, parse_choices(piece[weighted.end():] if weighted else piece)))
    return Choice(low, high, joiner, tuple(options), len(pieces) == 1 and not counted)


def parse_choices(text: str) -> list:
    """Find the ``{a|b}`` choices in a text.

    Args:
        text: One line, or one option of a choice.

    Returns:
        Its parts in order, each a string or a :class:`Choice`. A ``{`` never closed, an escaped
        one and the brace of ``${name}`` are text.
    """
    parts: list = []
    buffer = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            buffer.append(text[index:index + 2])
            index += 2
            continue
        if char == "{":
            after_dollar = index > 0 and text[index - 1] == "$" and BRACED_NAME.match(text[index:])
            close = -1 if after_dollar else _closing(text, index)
            if close > index:
                if buffer:
                    parts.append("".join(buffer))
                    buffer = []
                parts.append(_choice(text[index + 1:close]))
                index = close + 1
                continue
        buffer.append(char)
        index += 1
    if buffer:
        parts.append("".join(buffer))
    return parts


def _weighted(weights: list[float], draw_next: Callable[[], float]) -> int:
    """Draw one position, each as likely as its weight.

    Args:
        weights: One weight per option; zero is never drawn.
        draw_next: The line's draws.

    Returns:
        The position drawn, or -1 when no weight is above zero.
    """
    total = 0.0
    for weight in weights:
        if weight > 0:
            total += weight
    if total <= 0:
        return -1
    target = draw_next() * total
    reached = 0.0
    last = -1
    for position, weight in enumerate(weights):
        if weight <= 0:
            continue
        reached += weight
        last = position
        if target < reached:
            return position
    return last


def _drawn(parts: list, draw_next: Callable[[], float]) -> str:
    """Draw every choice among some parts.

    Args:
        parts: From :func:`parse_choices`.
        draw_next: The line's draws.

    Returns:
        The text with each choice replaced by what it drew.
    """
    out = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
            continue
        if part.literal:
            out.append("{" + _drawn(part.options[0][1], draw_next) + "}")
            continue
        count = part.low
        if part.high > part.low:
            count = part.low + int(draw_next() * (part.high - part.low + 1))
        weights = [weight for weight, _ in part.options]
        picks = []
        for _ in range(count):
            position = _weighted(weights, draw_next)
            if position < 0:
                break
            picks.append(position)
            weights[position] = 0.0
        out.append(part.joiner.join(_drawn(part.options[position][1], draw_next) for position in picks))
    return "".join(out)


def draw(text: str, seed: int, line: int) -> str:
    """Draw every ``{a|b}`` choice in one line.

    Args:
        text: The line's text.
        seed: The seed.
        line: The line's index, so two lines draw apart.

    Returns:
        The text with each choice drawn. Escapes are kept for :func:`unescape`.
    """
    if "{" not in text:
        return text
    return _drawn(parse_choices(text), stream(seed, line))


def expand(text: str, terms: dict, inside: tuple = ()) -> str:
    """Replace each ``$name`` and ``${name}`` with its term.

    Args:
        text: The text.
        terms: Name to value, from :func:`glossary`.
        inside: The names whose values this text is part of.

    Returns:
        The text with every known name replaced. An unknown name, and one naming a term it is
        already inside, is left as written.
    """
    if "$" not in text:
        return text

    def replace(found) -> str:
        name = found.group(1) or found.group(2)
        if name not in terms or name in inside or len(inside) >= TERM_DEPTH:
            return found.group(0)
        return expand(terms[name], terms, inside + (name,))

    return REFERENCE.sub(replace, text)


def tidy(text: str, written: str) -> str:
    """Close the gaps a choice or term left behind: empty list items and doubled spaces.

    Args:
        text: The line after drawing and expanding.
        written: The line as written, whose own leading or trailing comma is kept.

    Returns:
        The line tidied and trimmed.
    """
    text = re.sub(r"[ \t]*,(?:[ \t]*,)+", ",", text)
    text = re.sub(r"[ \t]+,", ",", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if not written.lstrip().startswith(","):
        text = re.sub(r"^[ \t]*,[ \t]*", "", text)
    if not written.rstrip().endswith(","):
        text = re.sub(r"[ \t]*,[ \t]*$", "", text)
    return text.strip()


def unescape(text: str) -> str:
    """Drop the backslash from each escaped brace, bar and dollar.

    Args:
        text: A finished line.

    Returns:
        The line as it is handed on.
    """
    return ESCAPED.sub(r"\1", text) if "\\" in text else text


def _finished(body: str, seed: int, line: int, terms: dict) -> str:
    """One line drawn, expanded, tidied and unescaped.

    Args:
        body: The text after its marks.
        seed: The seed.
        line: The line's index.
        terms: Name to value.

    Returns:
        The line as it is handed on.
    """
    done = expand(draw(body, seed, line), terms)
    return unescape(done if done == body else tidy(done, body))


def hidden_lines(lines: list[Line]) -> list[bool]:
    """Which lines sit in a section that is switched off.

    Args:
        lines: Every line, from :func:`classify`.

    Returns:
        One flag per line, True from a switched-off section's heading to the next heading.
    """
    hidden = []
    off = False
    for line in lines:
        if line.kind == SECTION:
            off = line.off
        hidden.append(off)
    return hidden


def glossary(lines: list[Line], groups: list[Group], seed: int) -> dict:
    """Every term switched on, by name, its choices drawn once.

    Args:
        lines: Every line, from :func:`classify`.
        groups: From :func:`group_lines`.
        seed: The seed.

    Returns:
        Name to value. A name written twice takes its last value.
    """
    hidden = hidden_lines(lines)
    terms = {}
    for group in groups:
        if group.kind == TERM and not group.off and not hidden[group.versions[0]]:
            index = group.versions[group.pick]
            terms[group.name] = draw(lines[index].body, seed, index)
    return terms


def resolve(text: str, seed: int = 0, terms: dict | None = None) -> list[tuple[str, str]]:
    """Every line a node could hand on, finished, before blanks and notes are left out.

    Args:
        text: The text, one part per line.
        seed: What the choices draw from.
        terms: Terms from elsewhere, such as another list; the text's own replace them.

    Returns:
        ``(kind, text)`` in order, ``kind`` one of :data:`PROMPT`, :data:`SECTION`, :data:`NOTE`
        and :data:`BLANK`, a heading giving its name. A prompt in mode :data:`ALL` gives each of
        its versions. A switched-off heading gives nothing, nor do the lines under it.
    """
    lines = classify(text)
    groups = group_lines(lines)
    known = {**(terms or {}), **glossary(lines, groups, seed)}
    hidden = hidden_lines(lines)
    heads = {group.versions[0]: group for group in groups}
    out: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if hidden[index]:
            continue
        if line.kind == SECTION:
            out.append((SECTION, line.body))
        elif line.kind == NOTE:
            out.append((NOTE, line.line.rstrip()))
        elif line.kind == BLANK:
            out.append((BLANK, ""))
        elif index in heads:
            group = heads[index]
            if group.off or group.kind == TERM:
                continue
            indent = line.line[:len(line.line) - len(line.line.lstrip())]
            chosen = group.live if group.mode == ALL else (group.pick,)
            for position in chosen:
                at = group.versions[position]
                out.append((PROMPT, indent + _finished(lines[at].body, seed, at, known)))
    return out


def handed_on(kind: str, skip_blank_lines: bool, skip_comment_lines: bool) -> bool:
    """Whether a finished line of a given kind is kept.

    Args:
        kind: :data:`PROMPT`, :data:`SECTION`, :data:`NOTE` or :data:`BLANK`.
        skip_blank_lines: Empty lines are left out.
        skip_comment_lines: Notes are left out.

    Returns:
        True when the line is kept. A section's heading never is.
    """
    if kind == SECTION:
        return False
    if kind == NOTE:
        return not skip_comment_lines
    if kind == BLANK:
        return not skip_blank_lines
    return True


def kept_lines(
    text: str, skip_blank_lines: bool = True, skip_comment_lines: bool = True, seed: int = 0
) -> list[str]:
    """The finished lines that are kept.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines.
        skip_comment_lines: Leave out notes.
        seed: What the choices draw from.

    Returns:
        The kept lines in order. Terms, versions not in use and lines switched off are never kept.
    """
    return [
        line for kind, line in resolve(text, seed) if handed_on(kind, skip_blank_lines, skip_comment_lines)
    ]


def list_lines(
    text: str, skip_blank_lines: bool = True, skip_comment_lines: bool = True, seed: int = 0
) -> list[str]:
    """The kept lines, each trimmed, one entry per line.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines rather than giving each as ``""``.
        skip_comment_lines: Leave out notes rather than giving each as written.
        seed: What the choices draw from.

    Returns:
        The entries in line order.
    """
    return [line.strip() for line in kept_lines(text, skip_blank_lines, skip_comment_lines, seed)]


def join_lines(
    text: str,
    skip_blank_lines: bool = False,
    skip_comment_lines: bool = True,
    join_with: str = "line break",
    seed: int = 0,
) -> str:
    """The kept lines as one text.

    Args:
        text: The text, one part per line.
        skip_blank_lines: Leave out empty lines.
        skip_comment_lines: Leave out notes.
        join_with: A key of :data:`JOINS`. ``line break`` keeps each line's indent and blank
            lines; ``space`` and ``comma`` trim each line and drop empty ones, and ``comma`` drops
            a line's own trailing comma first.
        seed: What the choices draw from.

    Returns:
        The joined text, empty when nothing is kept.

    Raises:
        ValueError: ``join_with`` is not a key of :data:`JOINS`.
    """
    if join_with not in JOINS:
        raise ValueError(f"join_with is {join_with!r}; it takes one of {', '.join(JOINS)}.")
    return join_parts(kept_lines(text, skip_blank_lines, skip_comment_lines, seed), join_with)


def join_parts(parts: list[str], join_with: str) -> str:
    """Join finished lines into one text.

    Args:
        parts: The lines.
        join_with: A key of :data:`JOINS`. ``line break`` keeps each part's indent and empty
            parts; ``space`` and ``comma`` trim each part and drop empty ones, and ``comma`` drops
            a part's own trailing comma first.

    Returns:
        The joined text.

    Raises:
        ValueError: ``join_with`` is not a key of :data:`JOINS`.
    """
    if join_with not in JOINS:
        raise ValueError(f"join_with is {join_with!r}; it takes one of {', '.join(JOINS)}.")
    if join_with == "line break":
        return JOINS[join_with].join(part.rstrip() for part in parts)
    trimmed = [part.strip() for part in parts]
    if join_with == "comma":
        trimmed = [part.rstrip(",").rstrip() for part in trimmed]
    return JOINS[join_with].join(part for part in trimmed if part)


def _one_list(entry) -> dict | None:
    """One list read from what a node stored.

    Args:
        entry: A dict with any of ``name``, ``mode``, ``pick``, ``on`` and ``lines``, or a string
            of lines.

    Returns:
        ``{name, mode, pick, on, lines}`` with every field filled, or None for anything else.
    """
    if isinstance(entry, str):
        entry = {"lines": entry}
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    mode = entry.get("mode")
    pick = entry.get("pick")
    lines = entry.get("lines")
    counted = isinstance(pick, (int, float)) and not isinstance(pick, bool) and float(pick).is_integer()
    return {
        "name": name if isinstance(name, str) else "",
        "mode": mode if mode in MODES else ALL,
        "pick": int(pick) if counted and 1 <= pick <= PICK_LIMIT else 1,
        "on": entry.get("on") is not False,
        "lines": lines if isinstance(lines, str) else "",
    }


def _no_constant(name: str):
    """Refuse ``NaN`` and ``Infinity``, which JSON does not have.

    Args:
        name: The constant read.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"{name} is not JSON")


def read_lists(value: str) -> list[dict]:
    """The lists a String List (Sections) node stores, as JSON.

    Args:
        value: A JSON array of lists, each ``{name, mode, pick, on, lines}``. Text that is not
            one is read as a single list of lines.

    Returns:
        Up to :data:`LIST_LIMIT` lists, every field filled.
    """
    text = value or ""
    try:
        data = json.loads(text, parse_constant=_no_constant) if text.strip() else []
    except ValueError:
        return [_one_list(text)]
    if not isinstance(data, list):
        return [_one_list(text)]
    found = (_one_list(entry) for entry in data[:LIST_LIMIT])
    return [entry for entry in found if entry is not None]


def list_seed(seed: int, index: int) -> int:
    """The seed one list draws from, so two lists draw apart.

    Args:
        seed: The node's seed.
        index: The list's place, from zero.

    Returns:
        A 32-bit seed.
    """
    return (seed + index * GOLDEN) & MASK


def list_entries(lists: list[dict], seed: int = 0) -> list[list[str]]:
    """Each switched-on list's finished prompts, a list in a mode other than ``all`` giving one.

    Args:
        lists: From :func:`read_lists`.
        seed: The node's seed.

    Returns:
        One list of entries per switched-on list, in order. Terms from any list reach every list.
    """
    texts = [
        (entry, list_seed(seed, index)) for index, entry in enumerate(lists) if entry["on"]
    ]
    terms: dict = {}
    for entry, own in texts:
        lines = classify(entry["lines"])
        terms.update(glossary(lines, group_lines(lines), own))
    found = []
    for entry, own in texts:
        items = [
            line.strip() for kind, line in resolve(entry["lines"], own, terms) if kind == PROMPT and line.strip()
        ]
        if entry["mode"] != ALL and items:
            items = [items[(entry["pick"] - 1) % len(items)]]
        found.append(items)
    return found


def combine(lists: list[list[str]], how: str = CYCLE, join_with: str = "comma") -> list[str]:
    """Put several lists together into one, each entry taking one line from every list.

    Args:
        lists: The lists, in the order their lines are joined. An empty one is left out.
        how: :data:`CYCLE` = entry ``i`` takes line ``i`` of each list, a shorter list starting
            over, as many entries as the longest list; :data:`EVERY` = every combination, the
            last list changing fastest.
        join_with: A key of :data:`JOINS`.

    Returns:
        The entries.

    Raises:
        ValueError: ``how`` is not one of :data:`COMBINES`, ``join_with`` is not a key of
            :data:`JOINS`, or every combination would make more than :data:`MAX_COMBINED` entries.
    """
    if how not in COMBINES:
        raise ValueError(f"combine is {how!r}; it takes one of {', '.join(COMBINES)}.")
    if join_with not in JOINS:
        raise ValueError(f"join_with is {join_with!r}; it takes one of {', '.join(JOINS)}.")
    lists = [entries for entries in lists if entries]
    if not lists:
        return []
    if how == CYCLE:
        longest = max(len(entries) for entries in lists)
        rows = [[entries[index % len(entries)] for entries in lists] for index in range(longest)]
        return [join_parts(row, join_with) for row in rows]
    total = 1
    for entries in lists:
        total *= len(entries)
    if total > MAX_COMBINED:
        sizes = " x ".join(str(len(entries)) for entries in lists)
        raise ValueError(
            f"Every combination of these lists ({sizes}) makes {total} entries, more than the "
            f"{MAX_COMBINED} one run takes. Switch a list off or shorten one, or set combine "
            f"to {CYCLE}."
        )
    rows = [[]]
    for entries in lists:
        rows = [row + [entry] for row in rows for entry in entries]
    return [join_parts(row, join_with) for row in rows]
