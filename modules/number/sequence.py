"""Number sequences drawn through keyframes, shaped by a curve, repeated and looped.

One cycle passes through the keyframes at even spacing. A position within a cycle runs from 0
at the first keyframe to 1 at the last.
"""

from __future__ import annotations

import math

from ..util.easing import EASING_NAMES, ease

__all__ = [
    "CURVE_NAMES",
    "CURVE_TARGETS",
    "HOLD",
    "LOOPS",
    "MAX_COUNT",
    "MAX_REPEAT",
    "build",
    "cycle_value",
    "length_of",
    "positions",
    "shaped",
]

#: The curve that keeps each value until the next one is reached.
HOLD = "hold"

#: Curve names in the order a menu offers them.
CURVE_NAMES = (*EASING_NAMES, HOLD)

#: What a curve shapes: each move from one keyframe to the next, or the values themselves.
CURVE_TARGETS = ("transitions", "values")

#: How one cycle follows the one before it.
LOOPS = ("restart", "ping_pong", "continue")

#: Most entries one sequence holds, and most keyframes it passes through.
MAX_COUNT = 10000

#: Most cycles one sequence holds.
MAX_REPEAT = 1000


def shaped(curve: str, t: float) -> float:
    """One curve applied to a progress value.

    Args:
        curve: A name from :data:`CURVE_NAMES`.
        t: Progress, clamped to ``[0, 1]``.

    Returns:
        The shaped progress. ``hold`` answers 0 until ``t`` reaches 1.
    """
    if curve == HOLD:
        return 1.0 if t >= 1.0 else 0.0
    return ease(curve, t)


def cycle_value(
    keys: list[float],
    position: float,
    curve: str,
    target: str,
    low: float,
    high: float,
) -> float:
    """The value one cycle holds at a position.

    Args:
        keys: The keyframes, at least one.
        position: Where in the cycle, from 0 at the first keyframe to 1 at the last.
        curve: A name from :data:`CURVE_NAMES`.
        target: ``transitions`` eases between neighbouring keyframes; ``values`` reshapes the
            straight-line value between ``low`` and ``high``.
        low: The smallest keyframe.
        high: The largest keyframe.

    Returns:
        The value at that position.
    """
    last = len(keys) - 1
    if last == 0:
        return float(keys[0])
    scaled = position * last
    index = min(int(math.floor(scaled)), last - 1)
    fraction = scaled - index
    start = keys[index]
    span = keys[index + 1] - start
    if target == "transitions":
        return start + span * shaped(curve, fraction)
    value = start + span * fraction
    if high == low:
        return value
    return low + (high - low) * shaped(curve, (value - low) / (high - low))


def length_of(keyframes: int, count: int, repeat: int, loop: str) -> int:
    """How many entries a sequence holds.

    Args:
        keyframes: How many keyframes one cycle passes through.
        count: Entries asked for, or 0 for one entry per keyframe in every cycle.
        repeat: Cycles in the sequence.
        loop: A name from :data:`LOOPS`.

    Returns:
        ``count`` when it is above 0. Otherwise one entry per keyframe per cycle, with the
        keyframe two cycles share counted once under ``ping_pong`` and ``continue``.
    """
    if count > 0:
        return count
    if loop == "restart":
        return keyframes * repeat
    return (keyframes - 1) * repeat + 1


def positions(length: int, repeat: int, loop: str) -> list[tuple[int, float]]:
    """Where each entry sits: its cycle and its position within that cycle.

    Args:
        length: Entries in the sequence, at least 1.
        repeat: Cycles in the sequence, at least 1 and, under ``restart``, at most ``length``.
        loop: A name from :data:`LOOPS`.

    Returns:
        One ``(cycle, position)`` per entry. Under ``restart`` the entries are shared between
        the cycles as evenly as they divide, earlier cycles taking one more, and each cycle
        runs from 0 to 1. Otherwise the entries are spread evenly over all the cycles at once.
    """
    if loop == "restart":
        base, extra = divmod(length, repeat)
        placed = []
        for cycle in range(repeat):
            size = base + (1 if cycle < extra else 0)
            for step in range(size):
                placed.append((cycle, step / (size - 1) if size > 1 else 0.0))
        return placed
    if length == 1:
        return [(0, 0.0)]
    placed = []
    for step in range(length):
        travelled = step * repeat / (length - 1)
        cycle = min(int(math.floor(travelled)), repeat - 1)
        placed.append((cycle, travelled - cycle))
    return placed


def build(
    keys: list[float],
    count: int,
    curve: str = "linear",
    target: str = "transitions",
    repeat: int = 1,
    loop: str = "restart",
    reverse: bool = False,
) -> list[float]:
    """The sequence, unrounded.

    Args:
        keys: The keyframes, at least one.
        count: Entries, or 0 for one per keyframe in every cycle.
        curve: A name from :data:`CURVE_NAMES`.
        target: A name from :data:`CURVE_TARGETS`.
        repeat: Cycles, at least 1.
        loop: A name from :data:`LOOPS`. ``ping_pong`` runs every other cycle backwards and
            ``continue`` raises each cycle by the rise of the one before it.
        reverse: Whether the finished list runs back to front.

    Returns:
        The values, in order.
    """
    length = length_of(len(keys), count, repeat, loop)
    low = min(keys)
    high = max(keys)
    rise = (
        cycle_value(keys, 1.0, curve, target, low, high)
        - cycle_value(keys, 0.0, curve, target, low, high)
    )
    values = []
    for cycle, position in positions(length, repeat, loop):
        if loop == "ping_pong" and cycle % 2 == 1:
            position = 1.0 - position
        value = cycle_value(keys, position, curve, target, low, high)
        if loop == "continue":
            value += cycle * rise
        values.append(value)
    if reverse:
        values.reverse()
    return values
