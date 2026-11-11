"""Strict JSON checking, and repair of the faults language models and hand edits leave.

:func:`check` reads text as strict JSON; :func:`repair` fixes each fault in place, keeps the rest
as written and lists every change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = [
    "KINDS",
    "MAX_CHARS",
    "MAX_DEPTH",
    "Change",
    "Fault",
    "Repaired",
    "Unrepairable",
    "check",
    "repair",
]

#: Characters of text either function reads at most.
MAX_CHARS = 16 * 1024 * 1024

#: Brackets a repair follows inside one another at most.
MAX_DEPTH = 200

#: Brackets tried as the start of the JSON when words come before it.
MAX_STARTS = 8

#: Characters a key may run to when a quote is tested as the end of the value before it.
MAX_KEY_CHARS = 256

QUOTE_ESCAPED = "quote escaped"
QUOTES_STRAIGHTENED = "quotes made double"
ESCAPE_FIXED = "backslash fixed"
CONTROL_ESCAPED = "line break or tab escaped"
STRING_CLOSED = "string closed"
COMMA_REMOVED = "comma removed"
COMMA_ADDED = "comma added"
COLON_ADDED = "colon added"
KEY_QUOTED = "key quoted"
STRING_QUOTED = "word quoted"
LITERAL_FIXED = "true, false or null fixed"
NUMBER_FIXED = "number fixed"
VALUE_ADDED = "null added"
BRACKET_CLOSED = "bracket closed"
BRACKET_FIXED = "bracket fixed"
COMMENT_REMOVED = "comment removed"
STRAY_REMOVED = "stray character removed"
TEXT_BEFORE_REMOVED = "text before the JSON removed"
TEXT_AFTER_REMOVED = "text after the JSON removed"

#: Every kind of change, in the order a report lists them.
KINDS = (
    QUOTE_ESCAPED,
    QUOTES_STRAIGHTENED,
    ESCAPE_FIXED,
    CONTROL_ESCAPED,
    STRING_CLOSED,
    COMMA_REMOVED,
    COMMA_ADDED,
    COLON_ADDED,
    KEY_QUOTED,
    STRING_QUOTED,
    LITERAL_FIXED,
    NUMBER_FIXED,
    VALUE_ADDED,
    BRACKET_CLOSED,
    BRACKET_FIXED,
    COMMENT_REMOVED,
    STRAY_REMOVED,
    TEXT_BEFORE_REMOVED,
    TEXT_AFTER_REMOVED,
)

#: Each character that opens a string, and the one that closes it.
QUOTES = {
    '"': '"',
    "'": "'",
    "“": "”",
    "”": "”",
    "‘": "’",
    "’": "’",
}

#: Words read as JSON's three literals, and what each is written as.
LITERALS = {
    "true": "true",
    "false": "false",
    "null": "null",
    "True": "true",
    "TRUE": "true",
    "False": "false",
    "FALSE": "false",
    "None": "null",
    "NULL": "null",
    "Null": "null",
    "undefined": "null",
    "NaN": "null",
    "Infinity": "null",
    "-Infinity": "null",
    "+Infinity": "null",
}

#: Escapes for the control characters a string may not hold as they are.
CONTROL = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}

WHITESPACE = re.compile(r"[ \t\r\n]*")
SPACES = re.compile(r"[ \t]*")
WORD = re.compile(r"[-+]?[A-Za-z_$][\w$]*")
KEY_WORD = re.compile(r"[\w$][\w$-]*")
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
JSON_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
NUMBER_PARTS = re.compile(r"(\d*)(?:\.(\d*))?([eE][-+]?\d+)?")
HEX4 = re.compile(r"[0-9a-fA-F]{4}")
OPENERS = re.compile(r"[\[{]")
FENCE = re.compile(r"```[^\n`]*\n(.*?)(?:```|\Z)", re.DOTALL)
BARE = {
    "value": re.compile(r"[^,}\r\n]*"),
    "array": re.compile(r"[^,\]\r\n]*"),
    "top": re.compile(r"[^\r\n]*"),
}
SPECIAL = {
    closer: re.compile("[\\\\\"\\x00-\\x1f" + re.escape(closer) + "]")
    for closer in set(QUOTES.values())
}

#: The end of a model's thinking, when its answer carries it.
THINK_END = "</think>"


@dataclass(frozen=True)
class Change:
    """One repair: a span of the text as it came, what replaced it, and what kind of fix it was.

    Attributes:
        start: Index of the span's first character in the text as it came.
        end: Index one past its last; equal to ``start`` for an insertion.
        text: What the span became.
        kind: One of :data:`KINDS`.
    """

    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class Fault:
    """Why text is not JSON, and where.

    Attributes:
        message: The reader's complaint, such as ``Expecting ',' delimiter``.
        position: Index of the character it stopped at, or ``None``.
        line: That character's line, counted from 1, or ``None``.
        column: Its column, counted from 1, or ``None``.
    """

    message: str
    position: int | None = None
    line: int | None = None
    column: int | None = None

    def where(self) -> str:
        """The fault as one phrase, with its line and column when known."""
        if self.line is None:
            return self.message
        return f"{self.message} at line {self.line}, column {self.column}"


@dataclass(frozen=True)
class Repaired:
    """Repaired JSON text.

    Attributes:
        text: The JSON.
        value: What it parses to.
        changes: Every change, in the order they sit in the text.
        spans: Where each change sits in ``text``, as ``(start, end)``.
    """

    text: str
    value: object
    changes: tuple[Change, ...]
    spans: tuple[tuple[int, int], ...]


class Unrepairable(ValueError):
    """Text a repair could not make into JSON."""


class _Constant(ValueError):
    """NaN or Infinity, which strict JSON has no spelling for."""


def _refuse_constant(name: str):
    raise _Constant(name)


def check(text: str) -> tuple[object, Fault | None]:
    """Read text as strict JSON.

    Args:
        text: The text.

    Returns:
        ``(value, None)`` for JSON, ``(None, fault)`` otherwise.
    """
    if len(text) > MAX_CHARS:
        return None, Fault(f"the text holds {len(text):,} characters, more than the {MAX_CHARS:,} read")
    if not text.strip():
        return None, Fault("the text is empty")
    try:
        return json.loads(text, parse_constant=_refuse_constant), None
    except json.JSONDecodeError as error:
        return None, Fault(error.msg, error.pos, error.lineno, error.colno)
    except _Constant as error:
        return None, Fault(f"{error} is not a JSON value")
    except RecursionError:
        return None, Fault("brackets are nested too deeply to read")
    except ValueError as error:
        return None, Fault(str(error))


def _number(token: str) -> str:
    """A number written the way JSON spells it, as ``+.5`` becomes ``0.5``."""
    sign = "-" if token.startswith("-") else ""
    parts = NUMBER_PARTS.fullmatch(token.lstrip("+-"))
    whole = (parts.group(1) or "").lstrip("0") or "0"
    fraction = parts.group(2) or ""
    return sign + whole + ("." + fraction if fraction else "") + (parts.group(3) or "")


class _Scan:
    """One pass over the text up to ``stop``, gathering the changes that make it JSON."""

    def __init__(self, text: str, stop: int):
        self.text = text
        self.stop = stop
        self.changes: list[Change] = []

    def change(self, start: int, end: int, text: str, kind: str) -> None:
        self.changes.append(Change(start, end, text, kind))

    def skip(self, i: int, record: bool = True) -> int:
        """Index of the next character past whitespace and comments."""
        text, stop = self.text, self.stop
        while True:
            i = WHITESPACE.match(text, i, stop).end()
            if text.startswith("//", i, stop):
                end = text.find("\n", i, stop)
                end = stop if end == -1 else end
            elif text.startswith("/*", i, stop):
                end = text.find("*/", i + 2, stop)
                end = stop if end == -1 else end + 2
            else:
                return i
            if record:
                self.change(i, end, "", COMMENT_REMOVED)
            i = end

    def value(self, i: int, where: str, depth: int, closers: tuple[str, ...]) -> int:
        """Read one value starting at or after ``i``; answers the index past it."""
        text, stop = self.text, self.stop
        while True:
            i = self.skip(i)
            if i >= stop:
                self.change(stop, stop, "null", VALUE_ADDED)
                return stop
            c = text[i]
            if c == "{":
                return self.object(i, depth + 1, closers)
            if c == "[":
                return self.array(i, depth + 1, closers)
            if c in QUOTES:
                return self.string(i, where)
            if c in "-+.0123456789":
                end = self.number(i, where)
                if end is not None:
                    return end
            if c.isalpha() or c in "_$-+":
                end = self.word(i, where)
                if end is not None:
                    return end
            if c in ",}]":
                self.change(i, i, "null", VALUE_ADDED)
                return i
            self.change(i, i + 1, "", STRAY_REMOVED)
            i += 1

    def object(self, i: int, depth: int, closers: tuple[str, ...]) -> int:
        """Read an object opening at ``i``; answers the index past it."""
        if depth > MAX_DEPTH:
            raise Unrepairable(f"brackets are nested more than {MAX_DEPTH} deep")
        text, stop = self.text, self.stop
        inner = closers + ("}",)
        i += 1
        members = 0
        comma = None
        last = i
        while True:
            i = self.skip(i)
            if i >= stop or text[i] in "}]":
                if comma is not None:
                    self.change(comma, comma + 1, "", COMMA_REMOVED)
                if i >= stop:
                    self.change(stop, stop, "}", BRACKET_CLOSED)
                    return stop
                if text[i] == "}":
                    return i + 1
                if "]" in closers:
                    self.change(i, i, "}", BRACKET_CLOSED)
                    return i
                self.change(i, i + 1, "}", BRACKET_FIXED)
                return i + 1
            c = text[i]
            if c == ",":
                if comma is not None or not members:
                    self.change(i, i + 1, "", COMMA_REMOVED)
                else:
                    comma = i
                i += 1
                continue
            word = None if c in QUOTES else KEY_WORD.match(text, i, stop)
            if c not in QUOTES and word is None:
                self.change(i, i + 1, "", STRAY_REMOVED)
                i += 1
                continue
            if members and comma is None:
                self.change(last, last, ",", COMMA_ADDED)
            comma = None
            if word is None:
                i = self.string(i, "key")
            else:
                self.change(i, word.end(), json.dumps(word.group(), ensure_ascii=False), KEY_QUOTED)
                i = word.end()
            key_end = i
            i = self.skip(i)
            if i < stop and text[i] == ":":
                i += 1
            else:
                self.change(key_end, key_end, ":", COLON_ADDED)
            i = self.value(i, "value", depth, inner)
            members += 1
            last = i

    def array(self, i: int, depth: int, closers: tuple[str, ...]) -> int:
        """Read an array opening at ``i``; answers the index past it."""
        if depth > MAX_DEPTH:
            raise Unrepairable(f"brackets are nested more than {MAX_DEPTH} deep")
        text, stop = self.text, self.stop
        inner = closers + ("]",)
        i += 1
        items = 0
        comma = None
        last = i
        while True:
            i = self.skip(i)
            if i >= stop or text[i] in "]}":
                if comma is not None:
                    self.change(comma, comma + 1, "", COMMA_REMOVED)
                if i >= stop:
                    self.change(stop, stop, "]", BRACKET_CLOSED)
                    return stop
                if text[i] == "]":
                    return i + 1
                if "}" in closers:
                    self.change(i, i, "]", BRACKET_CLOSED)
                    return i
                self.change(i, i + 1, "]", BRACKET_FIXED)
                return i + 1
            c = text[i]
            if c == ",":
                if comma is not None or not items:
                    self.change(i, i + 1, "", COMMA_REMOVED)
                else:
                    comma = i
                i += 1
                continue
            if not (c in "{[" or c in QUOTES or c in "-+.0123456789" or c.isalpha() or c in "_$"):
                self.change(i, i + 1, "", STRAY_REMOVED)
                i += 1
                continue
            if items and comma is None:
                self.change(last, last, ",", COMMA_ADDED)
            comma = None
            i = self.value(i, "array", depth, inner)
            items += 1
            last = i

    def string(self, i: int, where: str) -> int:
        """Read a string opening at ``i``; answers the index past its closing quote."""
        text, stop = self.text, self.stop
        opener = text[i]
        closer = QUOTES[opener]
        if opener != '"':
            self.change(i, i + 1, '"', QUOTES_STRAIGHTENED)
        special = SPECIAL[closer]
        j = i + 1
        while True:
            found = special.search(text, j, stop)
            if found is None:
                self.change(stop, stop, '"', STRING_CLOSED)
                return stop
            k = found.start()
            c = text[k]
            if c == "\\":
                j = self.escape(k, closer)
            elif c < " ":
                if c in "\r\n" and where in ("value", "array") and self.line_ends(k, where):
                    self.change(k, k, '"', STRING_CLOSED)
                    return k
                self.change(k, k + 1, CONTROL.get(c) or f"\\u{ord(c):04x}", CONTROL_ESCAPED)
                j = k + 1
            elif c == closer and (where == "key" or self.closes(k + 1, where)):
                if c != '"':
                    self.change(k, k + 1, '"', QUOTES_STRAIGHTENED)
                return k + 1
            elif c == '"':
                self.change(k, k, "\\", QUOTE_ESCAPED)
                j = k + 1
            else:
                j = k + 1

    def escape(self, k: int, closer: str) -> int:
        """Check the backslash at ``k`` inside a string; answers where reading goes on."""
        text, stop = self.text, self.stop
        following = text[k + 1] if k + 1 < stop else ""
        if following and following in '"\\/bfnrt':
            return k + 2
        if following == "u" and HEX4.match(text, k + 2, stop):
            return k + 6
        if following == "'" or (following == closer and closer != '"'):
            self.change(k, k + 1, "", ESCAPE_FIXED)
            return k + 2
        self.change(k, k + 1, "\\\\", ESCAPE_FIXED)
        return k + 1

    def closes(self, p: int, where: str) -> bool:
        """Whether a quote ending just before ``p`` closes a string standing as ``where``."""
        text, stop = self.text, self.stop
        q = self.skip(p, record=False)
        if q >= stop:
            return True
        if where == "top":
            return False
        end = "}" if where == "value" else "]"
        c = text[q]
        if c == end:
            return True
        if c == ",":
            q = self.skip(q + 1, record=False)
            while q < stop and text[q] == ",":
                q = self.skip(q + 1, record=False)
            if q >= stop or text[q] == end:
                return True
            return self.key_ahead(q) if where == "value" else self.item_ahead(q)
        if c in QUOTES:
            return self.key_ahead(q) if where == "value" else self.quoted_item_ahead(q)
        return False

    def line_ends(self, k: int, where: str) -> bool:
        """Whether a line break at ``k`` inside a string sits where the string should have closed."""
        text, stop = self.text, self.stop
        q = self.skip(k, record=False)
        if q >= stop:
            return True
        c = text[q]
        if where == "value":
            return c == "}" or self.key_ahead(q)
        return c == "]" or self.item_ahead(q)

    def key_ahead(self, q: int) -> bool:
        """Whether an object key, quoted or bare, followed by a colon starts at ``q``."""
        text, stop = self.text, self.stop
        c = text[q]
        if c in QUOTES:
            end = text.find(QUOTES[c], q + 1, min(stop, q + 1 + MAX_KEY_CHARS))
            if end == -1 or "\n" in text[q:end]:
                return False
            q = self.skip(end + 1, record=False)
            return q >= stop or text[q] == ":"
        word = KEY_WORD.match(text, q, stop)
        if word is None:
            return False
        q = self.skip(word.end(), record=False)
        return q < stop and text[q] == ":"

    def item_ahead(self, q: int) -> bool:
        """Whether an array item starts at ``q``."""
        text, stop = self.text, self.stop
        c = text[q]
        if c in "{[-0123456789":
            return True
        if c in QUOTES:
            return self.quoted_item_ahead(q)
        word = WORD.match(text, q, stop)
        return word is not None and word.group() in LITERALS

    def quoted_item_ahead(self, q: int) -> bool:
        """Whether the quoted text at ``q`` is followed by a comma, a closing bracket or the end."""
        text, stop = self.text, self.stop
        end = text.find(QUOTES[text[q]], q + 1, stop)
        if end == -1:
            return False
        q = self.skip(end + 1, record=False)
        return q >= stop or text[q] in ",]"

    def number(self, i: int, where: str) -> int | None:
        """Read a number at ``i``; ``None`` when none starts there."""
        text, stop = self.text, self.stop
        found = NUMBER.match(text, i, stop)
        if found is None:
            return None
        end = found.end()
        if end < stop and (text[end].isalnum() or text[end] in "_$"):
            return self.bare(i, where)
        token = found.group()
        if not JSON_NUMBER.fullmatch(token):
            self.change(i, end, _number(token), NUMBER_FIXED)
        return end

    def word(self, i: int, where: str) -> int | None:
        """Read a literal or an unquoted string at ``i``; ``None`` when neither starts there."""
        text, stop = self.text, self.stop
        found = WORD.match(text, i, stop)
        if found is None:
            return None
        literal = LITERALS.get(found.group())
        if literal is not None:
            after = SPACES.match(text, found.end(), stop).end()
            if after >= stop or text[after] in ",}]\r\n/":
                if literal != found.group():
                    self.change(i, found.end(), literal, LITERAL_FIXED)
                return found.end()
        return self.bare(i, where)

    def bare(self, i: int, where: str) -> int:
        """Quote the unquoted string at ``i``, which runs to the end of its line or value."""
        found = BARE[where].match(self.text, i, self.stop)
        words = found.group().rstrip()
        self.change(i, i + len(words), json.dumps(words, ensure_ascii=False), STRING_QUOTED)
        return i + len(words)


def _windows(text: str) -> list[tuple[int, int]]:
    """Where the JSON may start in the text, and where reading it stops."""
    base = text.rfind(THINK_END)
    base = 0 if base == -1 else base + len(THINK_END)
    for fence in FENCE.finditer(text, base):
        start = WHITESPACE.match(text, fence.start(1), fence.end(1)).end()
        if start < fence.end(1) and text[start] in "{[":
            return [(start, fence.end(1))]
    starts = []
    for opener in OPENERS.finditer(text, base):
        starts.append((opener.start(), len(text)))
        if len(starts) == MAX_STARTS:
            break
    if starts:
        return starts
    start = WHITESPACE.match(text, base).end()
    if start < len(text):
        c = text[start]
        word = WORD.match(text, start)
        if c in QUOTES or c in "-0123456789" or (word is not None and word.group() in LITERALS):
            return [(start, len(text))]
    return []


def repair(text: str) -> Repaired:
    """Make text into strict JSON, changing only the spots that break it.

    Args:
        text: The text, which may carry words, a code fence or a model's thinking around the JSON.

    Returns:
        The repaired JSON, what it parses to, and every change made.

    Raises:
        Unrepairable: No JSON was found, it is too large or too deep, or the repair still fails.
    """
    if len(text) > MAX_CHARS:
        raise Unrepairable(f"the text holds {len(text):,} characters, more than the {MAX_CHARS:,} read")
    windows = _windows(text)
    if not windows:
        raise Unrepairable("no JSON object or array was found in the text")
    best = None
    covered = -1
    for start, stop in windows:
        if start < covered:
            continue
        scan = _Scan(text, stop)
        end = scan.value(start, "top", 0, ())
        covered = end
        rank = (end - start, -len(scan.changes))
        if best is None or rank > best[0]:
            best = (rank, start, end, scan.changes)
    _, start, end, changes = best
    changes = list(changes)
    if text[:start].strip():
        changes.append(Change(0, start, "", TEXT_BEFORE_REMOVED))
    if WHITESPACE.match(text, end).end() < len(text):
        changes.append(Change(end, len(text), "", TEXT_AFTER_REMOVED))
    ordered = [
        change for _, change in sorted(
            enumerate(changes), key=lambda pair: (pair[1].start, pair[1].end > pair[1].start, pair[0]),
        )
    ]
    pieces = []
    spans = []
    at = 0
    length = 0
    for change in ordered:
        pieces.append(text[at:change.start])
        length += change.start - at
        spans.append((length, length + len(change.text)))
        pieces.append(change.text)
        length += len(change.text)
        at = change.end
    pieces.append(text[at:])
    repaired = "".join(pieces)
    value, fault = check(repaired)
    if fault is not None:
        raise Unrepairable(f"the repaired text still fails: {fault.where()}")
    return Repaired(repaired, value, tuple(ordered), tuple(spans))
