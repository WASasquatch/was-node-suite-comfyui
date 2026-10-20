"""Reading numbers out of a widget value or a list entry, and stepping a span.

An entry is a finite number, a bool, or text holding one. :data:`UNREADABLE` holds the
options for one with no number: ``skip``, ``zero``, ``error``.
"""

from __future__ import annotations

import math
import re

from .. import log

__all__ = [
    "SEPARATORS",
    "STEP_TOLERANCE",
    "UNREADABLE",
    "WHOLE_TOLERANCE",
    "as_number",
    "as_numbers",
    "split_values",
    "whole",
    "whole_steps",
]

logger = log.get_logger("util.numbers")

#: What an entry holding no number does, as a combo's ``options``. Every node that reads
#: numbers out of entries offers these three under the same name and in this order.
UNREADABLE = ("skip", "zero", "error")

#: What separates one value from the next inside a multi-value string. No number is written
#: with a line break, a comma, a space or a tab inside it, so a value is whatever sits
#: between two runs of them.
SEPARATORS = re.compile(r"[\s,]+")

#: Slack allowed when a span is divided by a step, as a share of the quotient. A span and a
#: step are both written as decimals that binary floating point cannot hold exactly, so
#: ``1.0 // 0.1`` is 9 where ten steps fit end to end, and a quotient this close to a whole
#: number is read as that number.
STEP_TOLERANCE = 1e-9

#: Distance from a whole number within which :func:`whole` reads a value as that number, as a
#: share of the value above 1 and as an absolute amount below it.
WHOLE_TOLERANCE = 1e-9


def whole(value: float) -> int:
    """A value cut off at the decimal point.

    Args:
        value: The value to cut.

    Returns:
        The value with its decimal part dropped, towards zero, with a value within
        :data:`WHOLE_TOLERANCE` of a whole number read as that number, so 57.99999999999999
        answers 58.
    """
    nearest = round(value)
    if abs(value - nearest) <= WHOLE_TOLERANCE * max(1.0, abs(value)):
        return int(nearest)
    return int(value)


def whole_steps(span: float, step: float) -> int:
    """How many whole steps fit inside a span.

    Args:
        span: Distance to be covered, not negative.
        step: Length of one step, greater than 0.

    Returns:
        The count, with a quotient within :data:`STEP_TOLERANCE` of a whole number read as
        that number rather than as the number below it.
    """
    quotient = span / step
    nearest = math.floor(quotient + 0.5)
    if abs(quotient - nearest) <= STEP_TOLERANCE * max(1.0, quotient):
        return nearest
    return math.floor(quotient)


def split_values(value) -> list:
    """The entries one widget value or one wire carries.

    Args:
        value: A string typed into a widget, whatever arrived on a ``LIST`` socket, a single
            number, or ``None``. A string is cut on :data:`SEPARATORS`, and the empty pieces
            a blank line, a trailing comma or leading indentation leave behind are dropped.

    Returns:
        The entries, in order. Empty for ``None``, for an empty string and for a string
        holding nothing but separators.
    """
    from ..compat.lists import as_list

    if isinstance(value, str):
        return [piece for piece in SEPARATORS.split(value.strip()) if piece]
    return as_list(value)


def as_number(entry) -> float | None:
    """Read one entry as a finite number.

    Args:
        entry: A number, or text holding one. A bool is read as 1.0 or 0.0, since it arrives
            that way from the boolean nodes and refusing it would be a surprise.

    Returns:
        The value, or ``None`` when the entry holds no number, and ``None`` for ``nan``,
        ``inf`` and ``-inf`` however they are spelled, including the overflow an exponent
        such as ``1e400`` and an integer too large for a float both land on.
    """
    if isinstance(entry, (int, float)):
        try:
            number = float(entry)
        except OverflowError:
            return None
    else:
        try:
            number = float(str(entry).strip())
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def as_numbers(entries, unreadable: str = "skip", node: str = "") -> list[float]:
    """Read every entry as a number, answering for the ones that are not.

    Args:
        entries: Entries to read, from :func:`split_values` or from a ``LIST``.
        unreadable: One of :data:`UNREADABLE`. Any other value behaves as ``skip``.
        node: Display name of the node reading them, which the exception and the log line
            are written against.

    Returns:
        The values, in order. Shorter than ``entries`` where anything was skipped, and as
        long as it on ``zero``.

    Raises:
        ValueError: ``unreadable`` is ``error`` and an entry holds no number.
    """
    values: list[float] = []
    for position, entry in enumerate(entries):
        number = as_number(entry)
        if number is None:
            if unreadable == "error":
                raise ValueError(
                    f"{node} could not read entry {position} as a number: {entry!r}. Set "
                    f"unreadable to 'skip' or 'zero' to allow it."
                )
            if unreadable == "zero":
                values.append(0.0)
            else:
                logger.debug("%s skipped entry %d: %r", node, position, entry)
            continue
        values.append(number)
    return values
