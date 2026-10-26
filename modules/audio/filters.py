"""Filters and a feedback echo that run sample by sample, so cutoff and level may move every sample.

Signals are float64 numpy arrays shaped ``(channels, samples)``. Frequencies are hertz.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "BIQUAD_KINDS",
    "FILTER_KINDS",
    "biquad",
    "crossover",
    "cut",
    "echo",
    "response_db",
    "swept",
]

#: The responses :func:`swept` offers.
FILTER_KINDS = ("low", "high", "band")

#: The shapes :func:`biquad` designs, in menu order.
BIQUAD_KINDS = ("bell", "low shelf", "high shelf", "low cut", "high cut", "notch", "band pass", "all pass")


def biquad(kind: str, freq: float, q: float, gain_db: float, rate: int) -> np.ndarray:
    """One second order section from the Audio EQ Cookbook formulas.

    Args:
        kind: A name from :data:`BIQUAD_KINDS`.
        freq: Centre or corner frequency in hertz, held below the Nyquist limit.
        q: Quality factor; shelves read it as their slope.
        gain_db: Boost or cut of a bell or a shelf.
        rate: Samples per second.

    Returns:
        ``[b0, b1, b2, 1, a1, a2]``, normalised.
    """
    freq = min(max(1.0, float(freq)), 0.49 * rate)
    q = max(0.025, float(q))
    amp = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * freq / rate
    cos, sin = math.cos(w0), math.sin(w0)
    alpha = sin / (2.0 * q)
    if kind == "low shelf" or kind == "high shelf":
        root = 2.0 * math.sqrt(amp) * alpha
        sign = 1.0 if kind == "low shelf" else -1.0
        b0 = amp * ((amp + 1) - sign * (amp - 1) * cos + root)
        b1 = sign * 2 * amp * ((amp - 1) - sign * (amp + 1) * cos)
        b2 = amp * ((amp + 1) - sign * (amp - 1) * cos - root)
        a0 = (amp + 1) + sign * (amp - 1) * cos + root
        a1 = -sign * 2 * ((amp - 1) + sign * (amp + 1) * cos)
        a2 = (amp + 1) + sign * (amp - 1) * cos - root
    elif kind == "low cut":
        b0, b1, b2 = (1 + cos) / 2, -(1 + cos), (1 + cos) / 2
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "high cut":
        b0, b1, b2 = (1 - cos) / 2, 1 - cos, (1 - cos) / 2
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2 * cos, 1.0
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "band pass":
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "all pass":
        b0, b1, b2 = 1 - alpha, -2 * cos, 1 + alpha
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    else:
        b0, b1, b2 = 1 + alpha * amp, -2 * cos, 1 - alpha * amp
        a0, a1, a2 = 1 + alpha / amp, -2 * cos, 1 - alpha / amp
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def cut(kind: str, freq: float, slope: int, rate: int) -> np.ndarray:
    """A Butterworth low cut or high cut as second order sections.

    Args:
        kind: ``low cut`` or ``high cut``.
        freq: Corner frequency in hertz.
        slope: Steepness in dB per octave, 6 to 96 in steps of 6.
        rate: Samples per second.

    Returns:
        Sections shaped ``(n, 6)``.
    """
    from scipy.signal import butter

    order = max(1, min(16, int(round(float(slope) / 6.0))))
    corner = min(max(1.0, float(freq)), 0.49 * rate)
    return butter(order, corner, btype="highpass" if kind == "low cut" else "lowpass", fs=rate, output="sos")


def crossover(freq: float, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """A fourth order Linkwitz-Riley crossover, whose two halves sum back to a flat response.

    Args:
        freq: Crossover frequency in hertz.
        rate: Samples per second.

    Returns:
        ``(low, high)`` second order sections.
    """
    from scipy.signal import butter

    corner = min(max(1.0, float(freq)), 0.49 * rate)
    low = butter(2, corner, btype="lowpass", fs=rate, output="sos")
    high = butter(2, corner, btype="highpass", fs=rate, output="sos")
    return np.vstack([low, low]), np.vstack([high, high])


def response_db(sections: np.ndarray, freqs, rate: int) -> np.ndarray:
    """The gain of a chain of second order sections at given frequencies.

    Args:
        sections: ``(n, 6)`` second order sections.
        freqs: Frequencies in hertz.
        rate: Samples per second.

    Returns:
        Gains in dB, floored at -240.
    """
    from scipy.signal import sosfreqz

    if len(sections) == 0:
        return np.zeros(len(freqs))
    _, answer = sosfreqz(np.asarray(sections, dtype=np.float64), worN=np.asarray(freqs, dtype=np.float64), fs=rate)
    return 20.0 * np.log10(np.maximum(np.abs(answer), 1e-12))


def swept(signal: np.ndarray, rate: int, cutoff, resonance: float = 0.707, kind: str = "low") -> np.ndarray:
    """A signal through a state variable filter whose cutoff may change every sample.

    Args:
        signal: ``(channels, samples)`` float64.
        rate: Samples per second.
        cutoff: Cutoff in hertz, one number or one per sample. Held below the Nyquist limit.
        resonance: Quality factor; 0.707 is flat, higher rings at the cutoff.
        kind: ``low``, ``high`` or ``band``.

    Returns:
        The filtered signal, same shape.
    """
    channels, length = signal.shape
    hertz = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (length,))
    hertz = np.clip(hertz, 10.0, 0.49 * rate)
    g = np.tan(np.pi * hertz / rate)
    k = 1.0 / max(0.05, float(resonance))
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    a1, a2, a3 = a1.tolist(), a2.tolist(), a3.tolist()
    out = np.empty_like(signal)
    for channel in range(channels):
        samples = signal[channel].tolist()
        result = [0.0] * length
        ic1 = ic2 = 0.0
        for index in range(length):
            v0 = samples[index]
            v3 = v0 - ic2
            v1 = a1[index] * ic1 + a2[index] * v3
            v2 = ic2 + a2[index] * ic1 + a3[index] * v3
            ic1 = 2.0 * v1 - ic1
            ic2 = 2.0 * v2 - ic2
            if kind == "high":
                result[index] = v0 - k * v1 - v2
            elif kind == "band":
                result[index] = v1
            else:
                result[index] = v2
        out[channel] = result
    return out


def echo(signal: np.ndarray, rate: int, delay: float, feedback: float, tail: float,
         damping: float = 4000.0) -> np.ndarray:
    """The repeats of a feedback echo fed by a signal, without the signal itself.

    Args:
        signal: ``(channels, samples)`` float64 feeding the echo.
        rate: Samples per second.
        delay: Seconds between repeats, at least one millisecond.
        feedback: Level each repeat keeps of the one before, 0 to 0.95.
        tail: Seconds the repeats run past the end of the signal.
        damping: Hertz above which each repeat is softened.

    Returns:
        ``(channels, samples + tail samples)`` float64.
    """
    from scipy.signal import lfilter

    channels, length = signal.shape
    gap = max(1, int(round(max(0.001, float(delay)) * rate)))
    total = length + max(0, int(round(float(tail) * rate)))
    keep = min(0.95, max(0.0, float(feedback)))
    pole = math.exp(-2.0 * math.pi * min(float(damping), 0.49 * rate) / rate)
    source = np.zeros((channels, total))
    source[:, :length] = signal
    wet = np.zeros((channels, total + gap))
    state = np.zeros((channels, 1))
    # Each block reads only what was written at least one repeat earlier.
    for first in range(0, total, gap):
        last = min(total, first + gap)
        fed = source[:, first:last] + keep * wet[:, first:last]
        softened, state = lfilter([1.0 - pole], [1.0, -pole], fed, axis=-1, zi=state)
        wet[:, first + gap:last + gap] = softened
    return wet[:, :total]
