"""Keyframes for an amount along a clip, read from ``frame:amount:easing`` text.

Frames count from 0, a negative frame from the end. Amounts run 0 to 1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..number.sequence import CURVE_NAMES, shaped

__all__ = [
    "DEFAULT_EASING",
    "Keyframe",
    "MAX_FRAME",
    "MAX_KEYFRAMES",
    "amount_at",
    "amounts",
    "parse",
    "placed",
    "read",
]

#: The easing of an entry that names none.
DEFAULT_EASING = "linear"

#: Most keyframes one text holds.
MAX_KEYFRAMES = 1000

#: Furthest a keyframe may sit from frame 0, either way.
MAX_FRAME = 10_000_000

#: What separates two entries.
_ENTRY = re.compile(r"[;\n\r]")

#: A whole frame number, signed, in ASCII digits.
_FRAME = re.compile(r"^[+-]?[0-9]+$")

#: A decimal amount in ASCII digits, as ``float()`` reads one without underscores.
_AMOUNT = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


@dataclass(frozen=True)
class Keyframe:
    """One keyframe as written.

    Attributes:
        frame: The frame, negative counting from the end.
        amount: 0 to 1.
        easing: A name from :data:`modules.number.sequence.CURVE_NAMES`.
    """

    frame: int
    amount: float
    easing: str


def _entry(text: str) -> Keyframe | None:
    """One entry read, or None where it does not read."""
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in (2, 3) or not _FRAME.match(parts[0]) or not _AMOUNT.match(parts[1]):
        return None
    frame = int(parts[0])
    amount = float(parts[1]) + 0.0
    easing = parts[2].lower() if len(parts) == 3 else DEFAULT_EASING
    if abs(frame) > MAX_FRAME or not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        return None
    if easing not in CURVE_NAMES:
        return None
    return Keyframe(frame, amount, easing)


def parse(text) -> tuple[list[Keyframe], list[str]]:
    """Every entry of a keyframes text.

    Args:
        text: Entries separated by ``;`` or a new line. Blank entries and entries starting
            with ``#`` are skipped.

    Returns:
        ``(keyframes, unreadable)``: the readable keyframes in written order, and the text of
        every entry that does not read.
    """
    found, refused = [], []
    for raw in _ENTRY.split(str(text or "")):
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        keyframe = _entry(entry)
        if keyframe is None:
            refused.append(entry)
        else:
            found.append(keyframe)
    return found, refused


def read(text, name: str) -> list[Keyframe]:
    """The keyframes of a text, refusing one that holds an unreadable entry.

    Args:
        text: As :func:`parse` reads it.
        name: The node reading it, for the message.

    Returns:
        The keyframes in written order, at least one.

    Raises:
        ValueError: An entry does not read, there is none, or there are more than
            :data:`MAX_KEYFRAMES`.
    """
    found, refused = parse(text)
    if refused:
        shown = ", ".join(f"'{entry[:40]}'" for entry in refused[:3])
        raise ValueError(
            f"{name} cannot read the keyframe(s) {shown}. Write each as frame:amount or "
            f"frame:amount:easing, such as 0:0 or 48:1:ease_in_out_sine, with the amount "
            f"between 0 and 1 and the easing one of the curve names, or hold."
        )
    if not found:
        raise ValueError(
            f"{name} has no keyframes. Write at least one as frame:amount, such as 0:0; -1:1 "
            f"to move across the whole clip."
        )
    if len(found) > MAX_KEYFRAMES:
        raise ValueError(f"{name} holds {len(found)} keyframes; it reads at most {MAX_KEYFRAMES}.")
    return found


def placed(keys, count: int) -> list[tuple[int, float, str]]:
    """Keyframes at the frames they land on in a clip, in frame order.

    Args:
        keys: :class:`Keyframe` entries in written order.
        count: Frames in the clip.

    Returns:
        ``(frame, amount, easing)`` sorted by frame. A negative frame lands at ``count`` plus
        it; where two land on one frame, the one written later is kept.
    """
    held = {}
    for key in keys:
        frame = key.frame + count if key.frame < 0 else key.frame
        held[frame] = (frame, key.amount, key.easing)
    return [held[frame] for frame in sorted(held)]


def amount_at(points, frame: float) -> float:
    """The amount at one frame.

    Args:
        points: What :func:`placed` returned, at least one.
        frame: The frame, whole or between two.

    Returns:
        The first keyframe's amount before it, the last one's after it, and between two the
        earlier one's easing from its amount to the later one's, held to 0 to 1.
    """
    if frame <= points[0][0]:
        return points[0][1]
    if frame >= points[-1][0]:
        return points[-1][1]
    low, high = 0, len(points) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if points[middle][0] <= frame:
            low = middle
        else:
            high = middle
    start, start_amount, easing = points[low]
    if frame == start:
        return start_amount
    end, end_amount, _ = points[high]
    if start_amount == end_amount:
        return start_amount
    progress = (frame - start) / (end - start)
    value = start_amount + (end_amount - start_amount) * shaped(easing, progress)
    return min(1.0, max(0.0, value))


def amounts(keys, count: int) -> list[float]:
    """The amount at every frame of a clip.

    Args:
        keys: :class:`Keyframe` entries in written order, at least one.
        count: Frames in the clip.

    Returns:
        ``count`` amounts, 0 to 1.
    """
    points = placed(keys, count)
    return [amount_at(points, frame) for frame in range(count)]
