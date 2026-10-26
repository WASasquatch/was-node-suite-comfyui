"""Analog filter voicings run one sample at a time with saturation inside their loops.

Signals are lists or arrays of floats for one channel. ``coefficients`` holds one cutoff
coefficient per sample, from :func:`coefficient`; ``resonance`` runs ``0`` to ``1``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["MODELS", "MODES", "coefficient", "filtered", "modes_of", "response_db"]

#: The voicings, in menu order.
MODELS = ("ladder", "diode", "ms-20", "sem")

#: Responses, in menu order; not every voicing has every one.
MODES = ("low pass", "high pass", "band pass", "notch")

#: The responses each voicing offers.
VOICING_MODES = {
    "ladder": ("low pass", "high pass", "band pass"),
    "diode": ("low pass",),
    "ms-20": ("low pass", "high pass"),
    "sem": ("low pass", "high pass", "band pass", "notch"),
}


def modes_of(model: str) -> tuple[str, ...]:
    """The responses a voicing offers.

    Args:
        model: A name from :data:`MODELS`.

    Returns:
        Names from :data:`MODES`.
    """
    return VOICING_MODES.get(model, ("low pass",))


def coefficient(cutoff: np.ndarray, rate: int, model: str) -> np.ndarray:
    """Per-sample cutoff coefficients for a voicing.

    Args:
        cutoff: Cutoff in hertz, one per sample.
        rate: Samples per second the filter runs at.
        model: A name from :data:`MODELS`.

    Returns:
        ``g`` per sample: a one pole step for the ladders, a prewarped tangent for the SEM.
    """
    hertz = np.clip(cutoff, 10.0, 0.45 * rate)
    if model == "sem":
        return np.tan(np.pi * hertz / rate)
    return 1.0 - np.exp(-2.0 * np.pi * hertz / rate)


def _ladder(samples, coefficients, resonance: float, mode: str) -> list:
    """Four transistor stages, tanh in each, the last fed back."""
    tanh = math.tanh
    k = 4.0 * resonance
    y1 = y2 = y3 = y4 = 0.0
    t1 = t2 = t3 = 0.0
    out = [0.0] * len(samples)
    pick_low, pick_high = mode == "low pass", mode == "high pass"
    for index, (x, g) in enumerate(zip(samples, coefficients)):
        u = tanh(x - k * y4)
        y1 += g * (u - t1)
        t1 = tanh(y1)
        y2 += g * (t1 - t2)
        t2 = tanh(y2)
        y3 += g * (t2 - t3)
        t3 = tanh(y3)
        y4 += g * (t3 - tanh(y4))
        if pick_low:
            out[index] = y4
        elif pick_high:
            out[index] = u - 4.0 * y1 + 6.0 * y2 - 4.0 * y3 + y4
        else:
            out[index] = 4.0 * (y2 - 2.0 * y3 + y4)
    return out


def _diode(samples, coefficients, resonance: float, mode: str) -> list:
    """Three diode-saturated stages with a bass-thinned feedback path."""
    k = 3.6 * resonance
    y1 = y2 = y3 = 0.0
    hp = 0.0
    out = [0.0] * len(samples)
    for index, (x, g) in enumerate(zip(samples, coefficients)):
        # A fixed one pole high pass, near 60 Hz at 96 kHz, thins the feedback's bass.
        hp += 0.004 * (y3 - hp)
        fed = y3 - hp
        u = x - k * fed
        u = u / (1.0 + abs(u)) if u > 0 else u / (1.0 + 0.6 * abs(u))
        y1 += g * (u - y1 / (1.0 + abs(y1)))
        y2 += g * (y1 - y2 / (1.0 + abs(y2)))
        y3 += g * (y2 - y3 / (1.0 + abs(y3)))
        out[index] = y3
    return out


def _korg35(samples, coefficients, resonance: float, mode: str) -> list:
    """Two stages with a hard clipped positive feedback, the Sallen-Key scream."""
    k = 2.2 * resonance
    y1 = y2 = 0.0
    high = mode == "high pass"
    out = [0.0] * len(samples)
    for index, (x, g) in enumerate(zip(samples, coefficients)):
        band = y1 - y2
        clipped = band / (1.0 + abs(band) * 1.5)
        u = x + k * clipped
        y1 += g * (u - y1)
        y2 += g * (y1 - y2)
        out[index] = (u - 2.0 * y1 + y2) if high else y2
    return out


def _sem(samples, coefficients, resonance: float, mode: str) -> list:
    """A trapezoidal state variable filter with a soft saturated input."""
    tanh = math.tanh
    k = 2.0 - 1.96 * resonance
    ic1 = ic2 = 0.0
    out = [0.0] * len(samples)
    for index, (x, g) in enumerate(zip(samples, coefficients)):
        v0 = tanh(x)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        v3 = v0 - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + g * a2 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == "high pass":
            out[index] = v0 - k * v1 - v2
        elif mode == "band pass":
            out[index] = v1
        elif mode == "notch":
            out[index] = v0 - k * v1
        else:
            out[index] = v2
    return out


VOICINGS = {"ladder": _ladder, "diode": _diode, "ms-20": _korg35, "sem": _sem}


def filtered(signal: np.ndarray, coefficients: np.ndarray, resonance: float, model: str, mode: str) -> np.ndarray:
    """A signal through one voicing, channel by channel.

    Args:
        signal: ``(channels, samples)`` float64, already driven.
        coefficients: One coefficient per sample, from :func:`coefficient`.
        resonance: ``0`` to ``1``; near 1 the ladders ring and self-oscillate.
        model: A name from :data:`MODELS`.
        mode: A name from :func:`modes_of` for the model; others fall back to low pass.

    Returns:
        ``(channels, samples)`` float64.
    """
    run = VOICINGS.get(model, _ladder)
    response = mode if mode in modes_of(model) else "low pass"
    reso = min(1.0, max(0.0, float(resonance)))
    step = coefficients.tolist()
    out = np.empty_like(signal)
    for channel in range(signal.shape[0]):
        out[channel] = run(signal[channel].tolist(), step, reso, response)
    return out


def response_db(model: str, mode: str, cutoff: float, resonance: float, freqs) -> np.ndarray:
    """The small-signal response of a voicing, its saturation left out, for a panel.

    Args:
        model: A name from :data:`MODELS`.
        mode: A name from :data:`MODES`.
        cutoff: Hertz.
        resonance: ``0`` to ``1``.
        freqs: Frequencies in hertz.

    Returns:
        Gains in dB.
    """
    s = 1j * np.asarray(freqs, dtype=np.float64) / max(1.0, float(cutoff))
    reso = min(1.0, max(0.0, float(resonance)))
    response = mode if mode in modes_of(model) else "low pass"
    if model == "sem":
        k = 2.0 - 1.96 * reso
        den = s * s + k * s + 1.0
        num = {"low pass": 1.0, "high pass": s * s, "band pass": s, "notch": s * s + 1.0}[response]
    elif model == "ms-20":
        k = 2.2 * reso * 0.9
        den = s * s + (2.0 - k) * s + 1.0
        num = s * s if response == "high pass" else 1.0
    elif model == "diode":
        k = 3.6 * reso * 0.55
        den = (1.0 + s) ** 3 + k
        num = 1.0 + k
    else:
        k = 4.0 * reso
        den = (1.0 + s) ** 4 + k
        num = {"low pass": 1.0, "high pass": s ** 4, "band pass": 4.0 * s * s}[response]
    gain = np.abs(num / den)
    return 20.0 * np.log10(np.maximum(gain, 1e-9))
