"""Comparing two values, and reducing several booleans to one."""

from __future__ import annotations

import math
import re

from ..util import patterns

__all__ = [
    "COMPARISONS",
    "MAX_CONDITIONS",
    "REDUCTIONS",
    "CONDITION_NAMES",
    "compare",
    "reduce_booleans",
    "to_boolean",
]

#: How two values may be compared, in the order the widget lists them. Saved workflows store
#: the chosen option by value, so entries are appended and never reordered.
COMPARISONS: tuple[str, ...] = (
    "equals",
    "does not equal",
    "less than",
    "less than or equals",
    "greater than",
    "greater than or equals",
    "contains",
    "does not contain",
    "starts with",
    "ends with",
    "matches regex",
    "is empty",
)

#: How several booleans reduce to one.
REDUCTIONS: tuple[str, ...] = ("all", "any", "none", "exactly one", "majority")

#: Conditions a chain reads, matching the switch slot count.
MAX_CONDITIONS = 26

#: The condition slot names, in the order they are drawn.
CONDITION_NAMES: tuple[str, ...] = tuple(
    f"condition_{letter}" for letter in "abcdefghijklmnopqrstuvwxyz"
)

#: Words a checkbox may arrive as from a text field or a JSON round trip.
TRUE_WORDS = ("true", "1", "yes", "y", "on", "t")
FALSE_WORDS = ("false", "0", "no", "n", "off", "f", "")


def to_boolean(value, default: bool = False) -> bool:
    """Read any value as true or false.

    Args:
        value: What the socket carried.
        default: Answer for a word that reads as neither.

    Returns:
        The truth of the value. A number is true when it is not zero. Text is read as a
        word, then as a number, so ``"2"`` is true and ``"0.0"`` false, and otherwise
        answers ``default``, or whether it holds anything when ``default`` is None. A
        container is true when it holds anything.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        word = value.strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        try:
            number = float(word)
        except ValueError:
            number = math.nan
        if math.isfinite(number):
            return number != 0
        return default if default is not None else bool(word)
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _numbers(left, right) -> tuple[float, float] | None:
    """Both values as floats, or None where either is not a number."""
    try:
        return float(left), float(right)
    except (TypeError, ValueError):
        return None


def compare(left, right, comparison: str) -> bool:
    """Whether two values stand in the given relation.

    Args:
        left: The left-hand value.
        right: The right-hand value, ignored by ``is empty``.
        comparison: One of :data:`COMPARISONS`.

    Returns:
        The answer. An ordering compares numbers where both read as numbers and text
        otherwise, so ``"apple" < "banana"`` answers as a dictionary would order them.

    Raises:
        ValueError: The comparison is unknown, or a regex could not be read or ran past
            :data:`modules.util.patterns.MATCH_SECONDS`.
    """
    if comparison not in COMPARISONS:
        raise ValueError(f"unknown comparison {comparison!r}, expected one of {COMPARISONS}")

    if comparison == "is empty":
        if left is None:
            return True
        if isinstance(left, str):
            return not left.strip()
        try:
            return len(left) == 0
        except TypeError:
            return False

    if comparison == "equals":
        pair = _numbers(left, right)
        return pair[0] == pair[1] if pair else str(left) == str(right)
    if comparison == "does not equal":
        return not compare(left, right, "equals")

    if comparison in ("less than", "less than or equals", "greater than", "greater than or equals"):
        pair = _numbers(left, right)
        first, second = pair if pair else (str(left), str(right))
        if comparison == "less than":
            return first < second
        if comparison == "less than or equals":
            return first <= second
        if comparison == "greater than":
            return first > second
        return first >= second

    text, needle = str(left), str(right)
    if comparison == "contains":
        return needle in text
    if comparison == "does not contain":
        return needle not in text
    if comparison == "starts with":
        return text.startswith(needle)
    if comparison == "ends with":
        return text.endswith(needle)
    try:
        return patterns.search(needle, text, "Compare's value_b") is not None
    except re.error as bad:
        raise ValueError(
            f"Compare's value_b `{needle}` is not a readable regular expression ({bad}). "
            f"Correct the pattern, or put a backslash before a special character to match it "
            f"literally."
        ) from bad


def reduce_booleans(values: list[bool], reduction: str) -> bool:
    """Reduce several booleans to one.

    Args:
        values: The booleans, in slot order. An empty list answers False for every
            reduction but ``none``, which is vacuously true.
        reduction: One of :data:`REDUCTIONS`.

    Returns:
        The reduced answer.

    Raises:
        ValueError: The reduction is unknown.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}, expected one of {REDUCTIONS}")
    true_count = sum(1 for value in values if value)
    if reduction == "all":
        return bool(values) and true_count == len(values)
    if reduction == "any":
        return true_count > 0
    if reduction == "none":
        return true_count == 0
    if reduction == "exactly one":
        return true_count == 1
    return bool(values) and true_count * 2 > len(values)
