"""Low frequency oscillators, tempo divisions and envelope followers that move an effect's settings.

An LFO runs ``-1`` to ``1``; an envelope runs ``0`` upward in linear amplitude. Rates are hertz.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DIVISIONS",
    "LFO_SHAPES",
    "envelope",
    "lfo",
    "peak_hold",
    "rate_of_division",
    "seconds_of_division",
]

#: The shapes an LFO can take, in menu order.
LFO_SHAPES = ("sine", "triangle", "square", "saw up", "saw down", "random steps", "random smooth")

#: Note lengths a tempo-synced setting offers, in menu order: d is dotted, t is triplet.
DIVISIONS = (
    "1/4", "1/8", "1/16", "1/32", "1/2", "1/1", "2/1", "4/1",
    "1/4d", "1/8d", "1/16d", "1/4t", "1/8t", "1/16t",
)


def seconds_of_division(division: str, bpm: float) -> float:
    """How long a note lasts at a tempo.

    Args:
        division: A name from :data:`DIVISIONS`, such as ``"1/8d"``.
        bpm: Beats per minute, a beat being a quarter note.

    Returns:
        Seconds.

    Raises:
        ValueError: ``division`` is not a note length.
    """
    text = str(division).strip().lower()
    scale = 1.0
    if text.endswith("d"):
        scale, text = 1.5, text[:-1]
    elif text.endswith("t"):
        scale, text = 2.0 / 3.0, text[:-1]
    top, _, bottom = text.partition("/")
    if not top.isdigit() or not bottom.isdigit() or int(bottom) == 0:
        raise ValueError(f"{division!r} is not a note length; write 1/4, 1/8d or 1/16t.")
    whole = 240.0 / max(1e-6, float(bpm))
    return whole * int(top) / int(bottom) * scale


def rate_of_division(division: str, bpm: float) -> float:
    """The rate in hertz of one cycle per note.

    Args:
        division: A name from :data:`DIVISIONS`.
        bpm: Beats per minute.

    Returns:
        Hertz.
    """
    return 1.0 / seconds_of_division(division, bpm)


def lfo(count: int, sample_rate: int, rate: float, shape: str = "sine", phase: float = 0.0,
        seed: int = 0) -> np.ndarray:
    """An oscillator's value at every sample.

    Args:
        count: Samples to make.
        sample_rate: Samples per second.
        rate: Cycles per second.
        shape: A name from :data:`LFO_SHAPES`.
        phase: Where the first cycle starts, ``0`` to ``1`` of a cycle.
        seed: Seed of the random shapes.

    Returns:
        A float64 array of ``count`` values from ``-1`` to ``1``.
    """
    cycles = (np.arange(count, dtype=np.float64) * (float(rate) / sample_rate) + float(phase))
    fraction = cycles - np.floor(cycles)
    if shape == "triangle":
        return 1.0 - 4.0 * np.abs(fraction - 0.5)
    if shape == "square":
        return np.where(fraction < 0.5, 1.0, -1.0)
    if shape == "saw up":
        return 2.0 * fraction - 1.0
    if shape == "saw down":
        return 1.0 - 2.0 * fraction
    if shape in ("random steps", "random smooth"):
        steps = int(np.floor(cycles[-1])) + 2 if count else 1
        values = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, steps)
        index = np.floor(cycles).astype(np.int64)
        index -= index.min() if count else 0
        if shape == "random steps":
            return values[np.clip(index, 0, steps - 1)]
        here = values[np.clip(index, 0, steps - 1)]
        there = values[np.clip(index + 1, 0, steps - 1)]
        blend = 0.5 - 0.5 * np.cos(math.pi * fraction)
        return here + (there - here) * blend
    return np.sin(2.0 * math.pi * cycles)


def envelope(signal: np.ndarray, sample_rate: int, attack: float, release: float) -> np.ndarray:
    """The level of a signal as it rises and falls, from its loudest channel.

    Args:
        signal: ``(channels, samples)`` float64.
        sample_rate: Samples per second.
        attack: Seconds the envelope takes to rise about 63% of a step.
        release: Seconds it takes to fall about 63% of a step.

    Returns:
        A float64 array, one linear level per sample.
    """
    from scipy.signal import lfilter

    rectified = np.max(np.abs(signal), axis=0)
    rise = math.exp(-1.0 / max(1.0, attack * sample_rate))
    fall = math.exp(-1.0 / max(1.0, release * sample_rate))
    # Release as a falling peak hold, then attack as a one pole smoother over it.
    held = peak_hold(rectified, fall)
    return lfilter([1.0 - rise], [1.0, -rise], held)


def peak_hold(level: np.ndarray, fall: float) -> np.ndarray:
    """A level that jumps up at once and falls back by a fixed factor per sample.

    Args:
        level: Linear levels, one per sample.
        fall: Factor kept per sample while falling, below 1.

    Returns:
        ``y[n] = max(level[n], fall * y[n - 1])`` for every ``n``.
    """
    if level.size == 0:
        return level.astype(np.float64)
    with np.errstate(divide="ignore"):
        logged = np.log(np.maximum(level, 1e-30))
    step = math.log(max(1e-30, fall))
    ramp = np.arange(level.size, dtype=np.float64) * step
    # max over k of (log x[k] + (n - k) step) is a running maximum of (log x[k] - k step).
    return np.exp(np.maximum.accumulate(logged - ramp) + ramp)
