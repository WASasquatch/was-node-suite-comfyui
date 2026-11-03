"""Filtering and ordering the entries of a text list, with names kept in step.

:func:`entries` reads a list input handed over whole, a ``LIST`` wire or a ``STRING`` list,
into its strings. Filters and orders answer positions into that list.
"""

from __future__ import annotations

import random
import re
from typing import Callable

from ..util import patterns

__all__ = [
    "KEEPS",
    "MATCHES",
    "ORDERS",
    "entries",
    "filter_entries",
    "matcher",
    "natural_key",
    "paired_names",
    "pattern_error",
    "quoted",
    "single",
    "sort_entries",
]

#: How an entry is tested against ``find``.
MATCHES = ("contains", "whole words", "starts with", "ends with", "regex", "is empty")

#: Which side of a filter is kept.
KEEPS = ("matching", "not matching")

#: How a list is ordered.
ORDERS = ("as given", "a-z", "z-a", "natural", "shortest first", "longest first", "shuffle")

#: A run of digits, kept by :func:`natural_key` as a number.
_DIGITS = re.compile(r"(\d+)")

#: The flag prefix that makes a pattern ignore case.
_IGNORE_CASE = "(?i)"


def entries(value) -> list[str]:
    """A list input, handed over whole, as its strings in order.

    Args:
        value: What an ``is_input_list`` node receives: a ``STRING`` list as its strings, a
            ``LIST`` wire as a list holding one list, or None for an input not wired.

    Returns:
        Every entry as a string. A ``LIST`` is unpacked in place; None becomes ``""``.
    """
    if value is None:
        return []
    found: list[str] = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, (list, tuple)):
            found.extend("" if entry is None else str(entry) for entry in item)
        else:
            found.append("" if item is None else str(item))
    return found


def single(value, default):
    """The one value of a widget input that arrived as a list, or ``default``.

    Args:
        value: The input as an ``is_input_list`` node receives it.
        default: Answered for a missing or empty input.

    Returns:
        The first value.
    """
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def paired_names(names, count: int, node: str) -> list[str] | None:
    """The names running beside a list of texts, one per text.

    Args:
        names: The names input as the node received it, or None where it is not wired.
        count: How many texts arrived.
        node: The node's display name, for the message.

    Returns:
        One name per text, or None where no names were wired.

    Raises:
        ValueError: The names are not one per text.
    """
    if names is None:
        return None
    found = entries(names)
    if len(found) != count:
        raise ValueError(
            f"{node} got {len(found)} names for {count} texts. names runs beside texts and "
            f"pairs with them by position, so wire one name per text, such as the "
            f"filename_list from the same Load Image Batch, or leave names unwired."
        )
    return found


def quoted(text: str, limit: int = 120) -> str:
    """A text in single quotes for a message, cut to ``limit`` characters.

    Args:
        text: The text, such as a pattern as typed.
        limit: Characters kept, an ellipsis included.

    Returns:
        The text as written, backslashes left single, between quotes.
    """
    shown = text if len(text) <= limit else text[: max(0, limit - 3)] + "..."
    return f"'{shown}'"


def pattern_error(where: str, pattern: str, error: re.error, shift: int = 0) -> str:
    """The message for a regular expression that does not compile.

    Args:
        where: The node and input, as ``Text Extract's pattern``.
        pattern: The pattern as typed.
        error: What compiling it raised.
        shift: Characters prefixed to the pattern before it was compiled.

    Returns:
        A message naming the input, the pattern and the fault.
    """
    detail = getattr(error, "msg", None) or str(error)
    position = getattr(error, "pos", None)
    if position is not None:
        detail = f"{detail} at position {max(0, position - shift)}"
    return (
        f"{where} {quoted(pattern)} is not a valid regular expression ({detail}). Fix the "
        f"pattern, and put a backslash before a bracket, a dot or a plus meant literally, as "
        f"\\( or \\. or \\+."
    )


def _terms(find: str) -> list[str]:
    """The comma separated terms of ``find``, trimmed, empty ones left out."""
    return [term.strip() for term in (find or "").split(",") if term.strip()]


def _empty_span(text: str) -> tuple[int, int] | None:
    """``(0, 0)`` for an entry holding only whitespace, otherwise None."""
    return None if text.strip() else (0, 0)


def matcher(
    find: str, match: str, ignore_case: bool, where: str
) -> Callable[[str], tuple[int, int] | None]:
    """A test answering where an entry matches ``find``.

    Args:
        find: Comma separated terms, any of which matches, or one pattern for ``regex``.
        match: One of :data:`MATCHES`. An unknown name reads as ``contains``.
        ignore_case: Whether a case difference still matches.
        where: The node and input ``find`` came from, for a message.

    Returns:
        A function taking an entry and answering the ``(start, end)`` span it matches at, or
        None. With no terms, every entry matches at ``(0, 0)``.

    Raises:
        ValueError: ``find`` is not a valid regular expression, or the pattern was refused.
    """
    if match == "is empty":
        return _empty_span
    prefix = _IGNORE_CASE if ignore_case else ""
    if match == "regex":
        if not find:
            return lambda text: (0, 0)
        pattern, group = prefix + find, 0
    else:
        terms = _terms(find)
        if not terms:
            return lambda text: (0, 0)
        alternatives = "|".join(re.escape(term) for term in terms)
        pattern, group = {
            "whole words": (rf"{prefix}(?<!\w)(?:{alternatives})(?!\w)", 0),
            "starts with": (rf"{prefix}\A\s*({alternatives})", 1),
            "ends with": (rf"{prefix}({alternatives})\s*\Z", 1),
        }.get(match, (f"{prefix}(?:{alternatives})", 0))

    def test(text: str) -> tuple[int, int] | None:
        found = patterns.search(pattern, text, where)
        return None if found is None else found.span(group)

    try:
        test("")
    except re.error as error:
        raise ValueError(pattern_error(where, find, error, len(prefix))) from error
    return test


def filter_entries(
    texts: list[str], find: str, match: str, ignore_case: bool, keep: str, where: str
) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Which entries a filter keeps.

    Args:
        texts: The entries.
        find: What :func:`matcher` tests each entry against.
        match: One of :data:`MATCHES`.
        ignore_case: Whether a case difference still matches.
        keep: ``matching`` or ``not matching``.
        where: The node and input ``find`` came from, for a message.

    Returns:
        ``(kept, spans)``: the positions kept, in order, and each entry's matched span or None.

    Raises:
        ValueError: ``find`` is not a valid regular expression, or the pattern was refused.
    """
    test = matcher(find, match, ignore_case, where)
    wanted = keep != "not matching"
    kept: list[int] = []
    spans: list[tuple[int, int] | None] = []
    for index, text in enumerate(texts):
        span = test(text)
        spans.append(span)
        if (span is not None) == wanted:
            kept.append(index)
    return kept, spans


def natural_key(text: str) -> list[tuple]:
    """A sort key reading each run of digits as a number, so ``img2`` sorts before ``img10``.

    Args:
        text: One entry.

    Returns:
        The text cut into words and numbers, a number compared by its value.
    """
    key: list[tuple] = []
    for index, part in enumerate(_DIGITS.split(text)):
        if index % 2:
            digits = part.lstrip("0")
            key.append((0, len(digits), digits, part))
        else:
            key.append((1, part))
    return key


def _ordered(
    indices: list[int], texts: list[str], order: str, seed: int, ignore_case: bool
) -> list[int]:
    """Positions put in one of :data:`ORDERS`.

    Args:
        indices: Positions into ``texts``.
        texts: The entries.
        order: One of :data:`ORDERS`. An unknown name keeps the order given.
        seed: The shuffle's seed.
        ignore_case: Whether ``a-z``, ``z-a`` and ``natural`` ignore case.

    Returns:
        The positions in the new order. Equal entries keep the order they arrived in.
    """
    def fold(index: int) -> str:
        return texts[index].casefold() if ignore_case else texts[index]

    if order == "a-z":
        return sorted(indices, key=fold)
    if order == "z-a":
        return sorted(indices, key=fold, reverse=True)
    if order == "natural":
        return sorted(indices, key=lambda index: natural_key(fold(index)))
    if order == "shortest first":
        return sorted(indices, key=lambda index: len(texts[index]))
    if order == "longest first":
        return sorted(indices, key=lambda index: len(texts[index]), reverse=True)
    if order == "shuffle":
        shuffled = list(indices)
        random.Random(int(seed)).shuffle(shuffled)
        return shuffled
    return list(indices)


def sort_entries(
    texts: list[str],
    order: str,
    seed: int = 0,
    dedupe: bool = False,
    ignore_case: bool = True,
    drop_empty: bool = False,
) -> tuple[list[int], int, int]:
    """Which entries a sort keeps, and in what order.

    Args:
        texts: The entries.
        order: One of :data:`ORDERS`.
        seed: The shuffle's seed.
        dedupe: Whether an entry repeating an earlier one, surrounding space aside, is dropped.
        ignore_case: Whether ordering and dedupe ignore case.
        drop_empty: Whether entries holding only whitespace are dropped.

    Returns:
        ``(positions, empties, duplicates)``: the positions kept in their new order, and how
        many empty and repeated entries were dropped.
    """
    indices = [index for index, text in enumerate(texts) if not (drop_empty and not text.strip())]
    empties = len(texts) - len(indices)
    duplicates = 0
    if dedupe:
        seen: set[str] = set()
        unique = []
        for index in indices:
            key = texts[index].strip()
            key = key.casefold() if ignore_case else key
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(index)
        indices = unique
    return _ordered(indices, texts, order, seed, ignore_case), empties, duplicates
