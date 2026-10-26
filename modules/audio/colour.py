"""Saturation curves, bit and rate reduction, slew limiting and ring modulation.

Signals are float64 numpy arrays shaped ``(channels, samples)`` in ``[-1, 1]`` full scale.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CRUSH_VOICINGS",
    "DITHERS",
    "RING_MODES",
    "SATURATIONS",
    "SLEW_MODES",
    "crushed",
    "ringed",
    "saturate",
    "slewed",
    "transfer_curve",
]

#: Saturation curves, in menu order.
SATURATIONS = ("tube", "tape", "transistor", "diode", "hard clip", "foldback", "wavefolder", "rectify")

#: How quantising error is spread, in menu order.
DITHERS = ("none", "tpdf", "noise shaped")

#: Sampler and console voicings: bits, sample rate, a cut before the hold, a cut after it.
CRUSH_VOICINGS = {
    "custom": None,
    "SP-1200": (12, 26040, None, None),
    "MPC60": (12, 40000, 18000.0, 15000.0),
    "S950": (12, 32000, 14000.0, 14000.0),
    "8-bit console": (8, 22050, None, None),
    "4-bit chip": (4, 11025, None, None),
    "telephone": (8, 8000, 3400.0, 3400.0),
}

#: Slew modes, in menu order.
SLEW_MODES = ("limit", "smooth")

#: Ring modulator modes, in menu order.
RING_MODES = ("ring", "amplitude", "frequency shift")

#: Samples a per-sample loop reads at once.
CHUNK = 1 << 16


def _curve(x: np.ndarray, kind: str) -> np.ndarray:
    """One saturation curve, mapping 0 to 0."""
    if kind == "tape":
        return (2.0 / math.pi) * np.arctan(0.5 * math.pi * x)
    if kind == "transistor":
        return x / (1.0 + np.abs(x))
    if kind == "diode":
        return np.where(x >= 0.0, 1.0 - np.exp(-x), -0.6 * (1.0 - np.exp(x / 0.6)))
    if kind == "hard clip":
        return np.clip(x, -1.0, 1.0)
    if kind == "foldback":
        return 1.0 - np.abs(np.mod(x + 1.0, 4.0) - 2.0)
    if kind == "wavefolder":
        return np.sin(0.5 * math.pi * x)
    if kind == "rectify":
        return np.abs(x)
    return np.tanh(x)


def saturate(x: np.ndarray, kind: str, bias: float) -> np.ndarray:
    """A saturation curve with its input offset, the offset taken back out of the result.

    Args:
        x: Driven samples.
        kind: A name from :data:`SATURATIONS`.
        bias: Input offset, ``-1`` to ``1``, which makes the curve lopsided and adds even harmonics.

    Returns:
        Saturated samples.
    """
    offset = float(bias)
    return _curve(x + offset, kind) - _curve(np.array(offset), kind)


def transfer_curve(kind: str, drive: float, bias: float, points: int = 129) -> list:
    """The curve a saturator draws, input -1 to 1 against output.

    Args:
        kind: A name from :data:`SATURATIONS`.
        drive: Linear input gain.
        bias: Input offset.
        points: Samples of the curve.

    Returns:
        Output values for evenly spaced inputs from -1 to 1.
    """
    x = np.linspace(-1.0, 1.0, points)
    return [round(float(v), 5) for v in saturate(x * drive, kind, bias)]


def crushed(signal: np.ndarray, rate: int, bits: float, target_rate: float, jitter: float,
            dither: str, seed: int) -> np.ndarray:
    """A signal held at a lower sample rate and quantised to fewer bits, with no filtering of its own.

    Args:
        signal: ``(channels, samples)``.
        rate: Its samples per second.
        bits: Bits per sample, 1 to 24; fractions give in-between level counts.
        target_rate: The rate samples are held at, in hertz.
        jitter: ``0`` to ``1``: how unevenly the holds are spaced.
        dither: A name from :data:`DITHERS`.
        seed: Seed of the jitter and the dither.

    Returns:
        ``(channels, samples)``.
    """
    channels, length = signal.shape
    rng = np.random.default_rng(int(seed))
    out = signal
    if target_rate < rate:
        period = rate / max(1.0, float(target_rate))
        count = int(math.ceil(length / period)) + 2
        spacing = period * (1.0 + float(jitter) * 0.5 * rng.uniform(-1.0, 1.0, count))
        starts = np.concatenate([[0.0], np.cumsum(spacing)])
        index = np.searchsorted(starts, np.arange(length), side="right") - 1
        held = np.clip(np.floor(starts[np.clip(index, 0, count)]).astype(np.int64), 0, length - 1)
        out = signal[:, held]
    steps = 2.0 ** (min(24.0, max(1.0, float(bits))) - 1.0)
    if dither == "tpdf":
        noise = (rng.uniform(-0.5, 0.5, out.shape) + rng.uniform(-0.5, 0.5, out.shape)) / steps
        return np.clip(np.round((out + noise) * steps) / steps, -1.0, 1.0)
    if dither == "noise shaped":
        shaped = np.empty_like(out)
        for channel in range(channels):
            error = 0.0
            for first in range(0, length, CHUNK):
                row = out[channel, first:first + CHUNK].tolist()
                jolt = (rng.uniform(-0.5, 0.5, len(row)) + rng.uniform(-0.5, 0.5, len(row))).tolist()
                result = [0.0] * len(row)
                for n, value in enumerate(row):
                    wanted = value - error
                    level = round((wanted + jolt[n] / steps) * steps) / steps
                    error = level - wanted
                    result[n] = level
                shaped[channel, first:first + len(row)] = result
        return np.clip(shaped, -1.0, 1.0)
    return np.clip(np.round(out * steps) / steps, -1.0, 1.0)


def slewed(signal: np.ndarray, rate: int, rise: float, fall: float, mode: str) -> np.ndarray:
    """A signal whose rate of change is limited, or smoothed at separate rise and fall speeds.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        rise: Seconds a full swing from -1 to 1 takes going up, or the rise time constant.
        fall: The same going down.
        mode: ``limit`` caps each step; ``smooth`` eases towards the input.

    Returns:
        ``(channels, samples)``.
    """
    channels, length = signal.shape
    out = np.empty_like(signal)
    if mode == "smooth":
        up = 1.0 - math.exp(-1.0 / max(1e-6, rise * rate))
        down = 1.0 - math.exp(-1.0 / max(1e-6, fall * rate))
    else:
        up = 2.0 / max(1e-9, rise * rate)
        down = 2.0 / max(1e-9, fall * rate)
    for channel in range(channels):
        row = signal[channel].tolist()
        result = [0.0] * length
        y = row[0] if row else 0.0
        if mode == "smooth":
            for n, x in enumerate(row):
                y += (x - y) * (up if x > y else down)
                result[n] = y
        else:
            for n, x in enumerate(row):
                step = x - y
                if step > up:
                    step = up
                elif step < -down:
                    step = -down
                y += step
                result[n] = y
        out[channel] = result
    return out


def ringed(signal: np.ndarray, rate: int, mode: str, frequency: np.ndarray, depth: float) -> np.ndarray:
    """A signal multiplied by a carrier, or every frequency in it moved by a set number of hertz.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        mode: A name from :data:`RING_MODES`.
        frequency: Carrier or shift in hertz, one per sample; a negative shift moves down.
        depth: For ``amplitude``, how deep the tremolo is, ``0`` to ``1``.

    Returns:
        ``(channels, samples)``.
    """
    phase = 2.0 * math.pi * np.cumsum(frequency) / rate
    if mode == "frequency shift":
        from scipy.signal import hilbert

        analytic = hilbert(signal, axis=-1)
        return np.real(analytic * np.exp(1j * phase)[None, :])
    carrier = np.sin(phase)[None, :]
    if mode == "amplitude":
        amount = min(1.0, max(0.0, float(depth)))
        return signal * (1.0 - amount * 0.5 * (1.0 - carrier))
    return signal * carrier
