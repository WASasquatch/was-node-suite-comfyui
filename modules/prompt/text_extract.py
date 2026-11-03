"""Pulling every match of a regular expression, or one of its groups, out of a text."""

from __future__ import annotations

import re

from ..util import patterns
from .text_lists import pattern_error, quoted

__all__ = ["extract", "group_count"]

#: The flag prefix that makes a pattern ignore case.
_IGNORE_CASE = "(?i)"

#: Appended to a pattern to give it an empty alternative, which matches no text.
_EMPTY_ALTERNATIVE = "\n|"


def group_count(pattern: str, where: str) -> int | None:
    """How many parenthesised groups a pattern holds, read off a match against no text.

    Args:
        pattern: The regular expression.
        where: The node and input it came from, for a message.

    Returns:
        The count, or None where neither the pattern nor it with an empty alternative appended
        matches an empty text.

    Raises:
        re.error: The pattern is not a valid regular expression.
        ValueError: The pattern was refused or ran past its time limit.
    """
    found = patterns.search(pattern, "", where)
    if found is None:
        try:
            found = patterns.search(pattern + _EMPTY_ALTERNATIVE, "", where)
        except re.error:
            return None
    return None if found is None else found.re.groups


def extract(text: str, pattern: str, group: int = 0, ignore_case: bool = True,
            where: str = "Text Extract's pattern") -> list[str]:
    """Every match of a pattern in a text, or of one of its groups, in order.

    Args:
        text: The text searched.
        pattern: The regular expression, as ``(\\d+) steps``. Empty finds nothing.
        group: 0 for the whole match, 1 for the first parenthesised group, and so on.
        ignore_case: Whether a case difference still matches.
        where: The node and input the pattern came from, for a message.

    Returns:
        One string per match, ``""`` for a group that took no part in it.

    Raises:
        ValueError: The pattern is not valid, holds fewer groups than ``group``, was refused,
            or ran past its time limit.
    """
    if not pattern:
        return []
    group = int(group)
    prefix = _IGNORE_CASE if ignore_case else ""
    full = prefix + pattern
    try:
        groups = group_count(full, where)
        if groups is not None and group > groups:
            raise ValueError(
                f"{where} {quoted(pattern)} holds {groups} parenthesised "
                f"group{'' if groups == 1 else 's'}, so group {group} is past its end. "
                f"0 = the whole match; 1 = the first group; {groups} is the highest this "
                f"pattern allows. Lower group, or add a group around the part wanted, as "
                f"(\\d+) steps."
            )
        found = []
        for match in patterns.finditer(full, text, where):
            if group > match.re.groups:
                raise ValueError(
                    f"{where} {quoted(pattern)} holds {match.re.groups} parenthesised groups, "
                    f"so group {group} is past its end. Lower group."
                )
            found.append(match.group(group) or "")
        return found
    except re.error as error:
        raise ValueError(
            pattern_error(where, pattern, error, len(prefix))
            + f" Group {group} cannot be read until it is."
        ) from error
