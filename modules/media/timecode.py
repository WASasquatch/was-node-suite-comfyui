"""SMPTE timecode: frame counts written as ``HH:MM:SS:FF`` and read back.

Drop-frame timecode, written with ``;``, skips labels 00 and 01 (00 to 03 at 59.94) at the
start of every minute except each tenth.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

__all__ = ["DAY_HOURS", "dropped", "label", "nominal", "parse"]

#: Hours a timecode counts before it wraps to 00:00:00:00.
DAY_HOURS = 24

#: ``HH:MM:SS:FF`` with ``:``, ``;`` or ``.`` before the frames.
TIMECODE = re.compile(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})([:;.,])(\d{1,3})\s*$")

#: A plain frame number.
FRAME_NUMBER = re.compile(r"^\s*(\d{1,9})\s*$")


def nominal(rate: Fraction) -> int:
    """Frame labels a timecode second holds: 24 at 23.976, 30 at 29.97, 25 at 25."""
    return max(1, math.floor(rate + Fraction(1, 2)))


def dropped(rate: Fraction, drop_frame: bool) -> int:
    """Labels skipped at each minute: 2 at 29.97 and 4 at 59.94 with drop frame on, else 0."""
    if not drop_frame or rate.denominator != 1001:
        return 0
    return {30: 2, 60: 4}.get(nominal(rate), 0)


def _day(rate: Fraction, drop: int) -> int:
    """Frames in a timecode day at a rate."""
    base = nominal(rate)
    per_ten = base * 600 - drop * 9
    return per_ten * 6 * DAY_HOURS


def label(frame: int, rate: Fraction, drop_frame: bool = True) -> str:
    """The timecode of a frame count.

    Args:
        frame: Frames since 00:00:00:00. Wraps at 24 hours.
        rate: Frames a second.
        drop_frame: Write drop-frame timecode at 29.97 and 59.94.

    Returns:
        ``HH:MM:SS:FF``, or ``HH:MM:SS;FF`` for drop-frame: frame 1800 at 29.97 is
        ``00:01:00;02`` and frame 17982 is ``00:10:00;00``.
    """
    base = nominal(rate)
    drop = dropped(rate, drop_frame)
    frame = int(frame) % _day(rate, drop)
    if drop:
        per_ten = base * 600 - drop * 9
        per_minute = base * 60 - drop
        tens, rest = divmod(frame, per_ten)
        frame += drop * 9 * tens
        if rest > drop:
            frame += drop * ((rest - drop) // per_minute)
    frames = frame % base
    total = frame // base
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{';' if drop else ':'}{frames:02d}"


def parse(text: str, rate: Fraction, drop_frame: bool = True) -> int:
    """The frame count a timecode or a frame number names.

    Args:
        text: ``HH:MM:SS:FF`` (``;`` before the frames is accepted too), or a frame number
            such as ``1001``.
        rate: Frames a second.
        drop_frame: Read drop-frame timecode at 29.97 and 59.94.

    Returns:
        Frames since 00:00:00:00.

    Raises:
        ValueError: The text is neither, a field is out of range, or the label is one drop-frame
            timecode skips.
    """
    value = str(text or "").strip() or "00:00:00:00"
    number = FRAME_NUMBER.match(value)
    if number:
        return int(number.group(1))
    found = TIMECODE.match(value)
    if found is None:
        raise ValueError(
            f"`{value[:40]}` is not a timecode or a frame number. Write HH:MM:SS:FF, such as "
            f"01:00:00:00, or a frame number, such as 1001."
        )
    hours, minutes, secs, _mark, frames = found.groups()
    hours, minutes, secs, frames = int(hours), int(minutes), int(secs), int(frames)
    base = nominal(rate)
    if minutes >= 60 or secs >= 60 or hours >= DAY_HOURS:
        raise ValueError(
            f"`{value}` has a field out of range. Hours run 00 to 23, minutes and seconds 00 to 59."
        )
    if frames >= base:
        raise ValueError(
            f"`{value}` names frame {frames:02d}, and a second holds frames 00 to {base - 1:02d} "
            f"at {float(rate):.6g} fps."
        )
    drop = dropped(rate, drop_frame)
    if drop and secs == 0 and minutes % 10 and frames < drop:
        raise ValueError(
            f"`{value}` is a label drop-frame timecode skips: frames 00 to {drop - 1:02d} are left "
            f"out at the start of every minute except each tenth. Use "
            f"{hours:02d}:{minutes:02d}:00;{drop:02d}, or switch drop_frame off."
        )
    count = ((hours * 60 + minutes) * 60 + secs) * base + frames
    if drop:
        total_minutes = hours * 60 + minutes
        count -= drop * (total_minutes - total_minutes // 10)
    return count

