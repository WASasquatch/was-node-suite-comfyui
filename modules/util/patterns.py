"""Regular expressions read from a workflow, matched under a time limit.

The ``regex`` package runs each search for at most :data:`MATCH_SECONDS`. Where it is absent,
``re`` runs a pattern :func:`screen` accepts. Matches, pieces and expansions are what ``re``
returns.
"""

from __future__ import annotations

import functools
import re
import time
from typing import Iterator

__all__ = [
    "MATCH_SECONDS",
    "MAX_GROWTH_CHARS",
    "MAX_SCREENED_CHARS",
    "expand",
    "finditer",
    "screen",
    "search",
    "split",
]

#: Seconds one search may run before it is stopped.
MATCH_SECONDS = 1.0

#: Age in seconds at which a scan restarts from its last match.
RENEW_SECONDS = MATCH_SECONDS / 4

#: Longest pattern ``re`` runs where the ``regex`` package is absent.
MAX_SCREENED_CHARS = 1000

#: Characters a run of substitutions may add to the text it was given.
MAX_GROWTH_CHARS = 16 * 1024 * 1024

#: A counted repeat, such as ``{2}``, ``{2,}`` or ``{,5}``.
_BRACES = re.compile(r"\{([0-9]*)(,?)([0-9]*)\}")


@functools.lru_cache(maxsize=1)
def _engine():
    """The ``regex`` module where it is installed and takes a timeout, else None."""
    from .. import deps

    module = deps.optional("regex")
    if module is None:
        return None
    try:
        module.compile("a", module.VERSION0).search("a", timeout=MATCH_SECONDS, concurrent=True)
    except Exception:
        return None
    return module


@functools.lru_cache(maxsize=64)
def _guarded(pattern: str):
    """The pattern compiled by ``regex``, or None where it is absent or refuses the pattern."""
    engine = _engine()
    if engine is None:
        return None
    try:
        return engine.compile(pattern, engine.VERSION0)
    except Exception:
        return None


def _prepared(pattern: str, where: str):
    """The pattern compiled for one run.

    Args:
        pattern: The regular expression.
        where: The node and input it came from, for a message.

    Returns:
        ``(compiled, guarded)``: a ``regex`` pattern and True, or a ``re`` pattern that
        :func:`screen` accepted and False.

    Raises:
        re.error: The pattern is not a valid regular expression.
        ValueError: ``regex`` is absent and :func:`screen` refused the pattern.
    """
    plain = re.compile(pattern)
    guarded = _guarded(pattern)
    if guarded is not None:
        return guarded, True
    screen(pattern, where)
    return plain, False


def _too_slow(where: str) -> str:
    """The message for a search stopped at :data:`MATCH_SECONDS`."""
    return (
        f"{where} took longer than {MATCH_SECONDS:g} second to match and was stopped. A "
        f"repeat that can match the same text more than one way, such as (a|aa)+ or (a+)+, "
        f"tries every split of the text and can run for hours. Rewrite it so each character "
        f"is matched one way only, such as a+ in place of (a|aa)+."
    )


def _install_hint() -> str:
    """The command that installs the ``regex`` package, ready to paste."""
    from .. import deps

    return deps.install_command("regex")


def _class_end(pattern: str, index: int) -> int:
    """Where a character class opened at ``index`` closes.

    Args:
        pattern: The regular expression.
        index: The position of the opening ``[``.

    Returns:
        The position after its closing ``]``, or the pattern's length where it never closes.
    """
    index += 1
    if pattern.startswith("^", index):
        index += 1
    if pattern.startswith("]", index):
        index += 1
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        index += 1
        if char == "]":
            return index
    return len(pattern)


def _quantifier(pattern: str, index: int) -> tuple[bool | None, int]:
    """The repeat written at ``index``, if any.

    Args:
        pattern: The regular expression.
        index: The position just after an atom or a group.

    Returns:
        ``(unbounded, end)``. ``unbounded`` is None where no repeat is written, True for
        ``*``, ``+`` and ``{n,}``, and False for ``?``, ``{n}`` and ``{n,m}``. ``end`` is the
        position after the repeat and any lazy or possessive mark.
    """
    if index >= len(pattern):
        return None, index
    char = pattern[index]
    if char in "*+":
        unbounded, index = True, index + 1
    elif char == "?":
        unbounded, index = False, index + 1
    elif char == "{":
        braces = _BRACES.match(pattern, index)
        if braces is None:
            return None, index
        low, comma, high = braces.groups()
        if not low and not comma:
            return None, index
        unbounded, index = bool(comma) and not high, braces.end()
    else:
        return None, index
    if pattern.startswith(("?", "+"), index):
        index += 1
    return unbounded, index


def _nested_repeat(pattern: str) -> bool:
    """Whether a pattern repeats without bound a group that ends in an unbounded repeat.

    Args:
        pattern: The regular expression, as ``(a+)+`` or ``(?:x|y*)*``.

    Returns:
        True for a group ending, on any branch, in ``*``, ``+`` or ``{n,}`` that is itself
        followed by one of those.
    """
    groups: list[bool] = []
    last = False
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "(":
            if pattern.startswith("(?#", index):
                close = pattern.find(")", index)
                index = len(pattern) if close < 0 else close + 1
                continue
            groups.append(False)
            last = False
            index += 1
            continue
        if char == "|":
            if groups:
                groups[-1] = groups[-1] or last
            last = False
            index += 1
            continue
        if char == ")":
            ends = (groups.pop() if groups else False) or last
            unbounded, index = _quantifier(pattern, index + 1)
            if unbounded and ends:
                return True
            last = ends or bool(unbounded)
            continue
        if char == "\\":
            index += 2
        elif char == "[":
            index = _class_end(pattern, index)
        else:
            index += 1
        unbounded, index = _quantifier(pattern, index)
        last = bool(unbounded)
    return False


def screen(pattern: str, where: str) -> None:
    """Refuse a pattern ``re`` could run for hours.

    Args:
        pattern: The regular expression.
        where: The node and input it came from, as ``Text Find's pattern``.

    Raises:
        ValueError: The pattern is longer than :data:`MAX_SCREENED_CHARS`, or repeats a group
            that already ends in an unbounded repeat.
    """
    if len(pattern) > MAX_SCREENED_CHARS:
        raise ValueError(
            f"{where} is {len(pattern):,} characters long. Without the regex package a "
            f"pattern runs with no time limit, so it is held to {MAX_SCREENED_CHARS:,} "
            f"characters. Shorten it, or install regex with: {_install_hint()}"
        )
    if _nested_repeat(pattern):
        raise ValueError(
            f"{where} repeats a group that already ends in a repeat, as (a+)+ does, which can "
            f"run for hours on text that nearly matches. Without the regex package a pattern "
            f"runs with no time limit, so this one is refused. Write the repeat once, as a+ "
            f"in place of (a+)+, or install regex with: {_install_hint()}"
        )


def search(pattern: str, text: str, where: str):
    """The first match of a pattern in a text, as ``re.search`` finds it.

    Args:
        pattern: The regular expression.
        text: The text searched.
        where: The node and input the pattern came from, as ``Text Find's pattern``.

    Returns:
        The match, or None.

    Raises:
        re.error: The pattern is not a valid regular expression.
        ValueError: The search ran past :data:`MATCH_SECONDS`, or :func:`screen` refused it.
    """
    compiled, guarded = _prepared(pattern, where)
    if not guarded:
        return compiled.search(text)
    try:
        return compiled.search(text, timeout=MATCH_SECONDS, concurrent=True)
    except TimeoutError as error:
        raise ValueError(_too_slow(where)) from error


def _scan(compiled, text: str, where: str) -> Iterator:
    """Every match of a ``regex`` pattern, each search held to :data:`MATCH_SECONDS`.

    Args:
        compiled: A pattern from :func:`_guarded`.
        text: The text searched.
        where: The node and input the pattern came from.

    Yields:
        The matches, leftmost first.

    Raises:
        ValueError: One search ran past :data:`MATCH_SECONDS`.
    """
    position = 0
    skip = None
    while True:
        opened = time.monotonic()
        try:
            for match in compiled.finditer(
                text, position, timeout=MATCH_SECONDS, concurrent=True
            ):
                # A restarted scan finds the empty match it restarted on a second time.
                if skip is not None:
                    empty, skip = skip, None
                    if match.start() == match.end() == empty:
                        continue
                yield match
                if time.monotonic() - opened > RENEW_SECONDS:
                    position = match.end()
                    skip = position if match.start() == position else None
                    break
            else:
                return
        except TimeoutError as error:
            raise ValueError(_too_slow(where)) from error


def finditer(pattern: str, text: str, where: str) -> Iterator:
    """Every match of a pattern in a text, as ``re.finditer`` yields them.

    Args:
        pattern: The regular expression.
        text: The text searched.
        where: The node and input the pattern came from.

    Returns:
        An iterator over the matches, leftmost first.

    Raises:
        re.error: The pattern is not a valid regular expression.
        ValueError: :func:`screen` refused the pattern. A search running past
            :data:`MATCH_SECONDS` raises the same while iterating.
    """
    compiled, guarded = _prepared(pattern, where)
    if not guarded:
        return compiled.finditer(text)
    return _scan(compiled, text, where)


def split(pattern: str, text: str, where: str) -> list:
    """A text cut at every match of a pattern, as ``re.split`` cuts it.

    Args:
        pattern: The regular expression.
        text: The text cut.
        where: The node and input the pattern came from.

    Returns:
        The pieces, with the pattern's groups between them and None for a group that took no
        part in a match.

    Raises:
        re.error: The pattern is not a valid regular expression.
        ValueError: A search ran past :data:`MATCH_SECONDS`, or :func:`screen` refused it.
    """
    compiled, guarded = _prepared(pattern, where)
    if not guarded:
        return compiled.split(text)
    pieces = []
    last = 0
    for match in _scan(compiled, text, where):
        pieces.append(text[last:match.start()])
        pieces.extend(match.groups())
        last = match.end()
    pieces.append(text[last:])
    return pieces


@functools.lru_cache(maxsize=64)
def _check_template(pattern: str, template: str) -> None:
    """Raise what ``re`` raises for a template the pattern's groups cannot fill.

    Args:
        pattern: The regular expression the template expands matches of.
        template: The replacement template.

    Raises:
        re.error: The template holds a bad escape or names a group the pattern lacks.
        IndexError: The template names a group by a name the pattern lacks.
    """
    compiled = re.compile(pattern)
    names = {index: name for name, index in compiled.groupindex.items()}
    shape = "".join(
        f"(?P<{names[index]}>)" if index in names else "()"
        for index in range(1, compiled.groups + 1)
    )
    re.compile(shape).sub(template, "")


@functools.lru_cache(maxsize=64)
def _references(template: str) -> int:
    """At least as many as the group references a replacement template holds.

    Args:
        template: The replacement template.

    Returns:
        The count of ``\\g`` and of a backslash before a digit from 1 to 9, an octal escape
        among them.
    """
    count = 0
    index = template.find("\\")
    while index >= 0:
        mark = template[index + 1:index + 2]
        if mark and mark in "g123456789":
            count += 1
        index = template.find("\\", index + 2)
    return count


def expand(match, pattern: str, template: str, room: int, where: str) -> str:
    """One match's replacement, as ``re`` expands it.

    Args:
        match: A match from :func:`finditer` for ``pattern``.
        pattern: The regular expression that produced the match.
        template: The replacement, where ``\\1`` and ``\\g<name>`` stand for groups.
        room: Characters the replacement may run to.
        where: The node and input the template came from, for a message.

    Returns:
        The replacement text.

    Raises:
        re.error: The template holds a bad escape or names a group the pattern lacks.
        IndexError: The template names a group by a name the pattern lacks.
        ValueError: The replacement could run past ``room`` characters.
    """
    _check_template(pattern, template)
    bound = len(template)
    references = _references(template)
    if references:
        bound += references * max(end - start for start, end in match.regs)
    if bound > room:
        raise ValueError(
            f"{where} would make the result more than {MAX_GROWTH_CHARS:,} characters longer "
            f"than the text it was given, and was stopped. A replacement repeating \\g<0> or a "
            f"group many times multiplies the text. Use fewer references in the replacement, "
            f"or split the text across several runs."
        )
    return match.expand(template)
