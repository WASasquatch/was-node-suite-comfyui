"""Loudness and level measurement: ITU-R BS.1770 loudness, EBU loudness range and true peak.

Levels are dB: LUFS for loudness, LU for range, dBFS and dBTP for peaks. A silent clip reads
:data:`FLOOR`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["FLOOR", "SERIES_POINTS", "k_weighted", "measure", "true_peak", "weights_for"]

#: The level a silent clip or an unmeasurable figure reads, in dB.
FLOOR = -120.0

#: Block length and step of momentary loudness, and of the gated integrated measure.
MOMENTARY = 0.4
STEP = 0.1

#: Window of short-term loudness.
SHORT_TERM = 3.0

#: Absolute gate, and the relative gates of integrated loudness and loudness range.
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0
RANGE_GATE = -20.0

#: Most points of each loudness curve handed to a panel.
SERIES_POINTS = 1200

#: Samples a true peak pass reads at once.
PEAK_CHUNK = 1 << 21

#: Most samples, across channels, one measurement reads: 23 minutes of 48 kHz stereo.
MAX_MEASURED = 1 << 27


def k_weighted(signal: np.ndarray, rate: int) -> np.ndarray:
    """A signal through the BS.1770 K-weighting filter.

    Args:
        signal: ``(channels, samples)`` float64.
        rate: Samples per second.

    Returns:
        The weighted signal, same shape.
    """
    from scipy.signal import sosfilt

    # Stage one: the high shelf modelling the head.
    gain, centre, quality = 3.999843853973347, 1681.974450955533, 0.7071752369554196
    k = math.tan(math.pi * centre / rate)
    high = 10.0 ** (gain / 20.0)
    band = high ** 0.4996667741545416
    a0 = 1.0 + k / quality + k * k
    shelf = [
        (high + band * k / quality + k * k) / a0,
        2.0 * (k * k - high) / a0,
        (high - band * k / quality + k * k) / a0,
        1.0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / quality + k * k) / a0,
    ]
    # Stage two: the RLB high pass.
    centre, quality = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * centre / rate)
    a0 = 1.0 + k / quality + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / quality + k * k) / a0]
    return sosfilt(np.array([shelf, highpass]), signal, axis=-1)


def weights_for(channels: int) -> np.ndarray:
    """BS.1770 channel weights.

    Args:
        channels: Channel count.

    Returns:
        One weight per channel: 5.1 in film order weighs the surrounds 1.41 and leaves out
        the LFE; every other layout weighs each channel 1.
    """
    if channels == 6:
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.ones(channels)


def _powers(weighted: np.ndarray, rate: int, window: float) -> np.ndarray:
    """Mean square of each channel over windows a :data:`STEP` apart."""
    length = int(weighted.shape[-1])
    span = max(1, int(round(window * rate)))
    hop = max(1, int(round(STEP * rate)))
    if length < span:
        return np.zeros((0, weighted.shape[0]))
    running = np.concatenate(
        [np.zeros((weighted.shape[0], 1)), np.cumsum(weighted * weighted, axis=-1)], axis=-1
    )
    starts = np.arange(0, length - span + 1, hop)
    return ((running[:, starts + span] - running[:, starts]) / span).T


def _lufs(power: np.ndarray) -> np.ndarray:
    """Weighted mean squares as loudness."""
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(power)


def true_peak(signal: np.ndarray, rate: int) -> float:
    """The highest level between samples, read four times oversampled below 96 kHz.

    Args:
        signal: ``(channels, samples)`` float64.
        rate: Samples per second.

    Returns:
        The peak as a linear amplitude.
    """
    from scipy.signal import resample_poly

    factor = 4 if rate < 96000 else 2 if rate < 192000 else 1
    if factor == 1:
        return float(np.max(np.abs(signal))) if signal.size else 0.0
    peak = 0.0
    length = int(signal.shape[-1])
    margin = 64
    for first in range(0, length, PEAK_CHUNK):
        lo, hi = max(0, first - margin), min(length, first + PEAK_CHUNK + margin)
        up = resample_poly(signal[:, lo:hi], factor, 1, axis=-1)
        keep = up[:, (first - lo) * factor:(min(length, first + PEAK_CHUNK) - lo) * factor]
        if keep.size:
            peak = max(peak, float(np.max(np.abs(keep))))
    return max(peak, float(np.max(np.abs(signal))) if signal.size else 0.0)


def _db(value: float) -> float:
    """A linear amplitude in dB, floored."""
    return 20.0 * math.log10(value) if value > 0 else FLOOR


def _floored(value: float) -> float:
    """A level held at :data:`FLOOR` where it is not finite or below it."""
    return float(value) if math.isfinite(value) and value > FLOOR else FLOOR


def _curve(values: np.ndarray, window: float) -> dict:
    """A loudness curve for a panel, held to :data:`SERIES_POINTS` by keeping each stretch's loudest.

    Returns:
        ``offset``, the time the first value's window ends at, ``step`` between values, and
        ``values`` in LUFS.
    """
    values = np.where(np.isfinite(values), np.maximum(values, FLOOR), FLOOR)
    step = STEP
    if values.size > SERIES_POINTS:
        edges = np.linspace(0, values.size, SERIES_POINTS + 1).astype(int)
        step = STEP * values.size / SERIES_POINTS
        values = np.array([values[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    return {
        "offset": window,
        "step": round(step, 6),
        "values": [round(float(v), 2) for v in values],
    }


def measure(wave: torch.Tensor, rate: int) -> dict:
    """Every level figure of one clip.

    Args:
        wave: ``(channels, samples)`` float tensor.
        rate: Samples per second.

    Returns:
        ``integrated``, ``range``, ``momentary_max``, ``short_term_max`` (LUFS and LU),
        ``true_peak`` (dBTP), ``sample_peak`` and ``rms`` (dBFS), ``crest`` (dB), ``dc`` (the
        largest channel mean), ``correlation`` (between the first two channels, 1.0 for mono),
        ``clipped`` (samples at or past full scale), ``duration`` (seconds), and ``series``:
        the ``momentary`` and ``short_term`` curves for a panel.

    Raises:
        ValueError: The clip holds more than :data:`MAX_MEASURED` samples across its channels.
    """
    held = int(wave.shape[0]) * int(wave.shape[-1])
    if held > MAX_MEASURED:
        raise ValueError(
            f"this clip is too long to measure in one pass: {held:,} samples across its channels, "
            f"past {MAX_MEASURED:,}, about 23 minutes of 48 kHz stereo. Split it with Audio Split "
            f"or trim it with Audio Editor first."
        )
    signal = wave.detach().to(device="cpu", dtype=torch.float64).numpy()
    channels, length = int(signal.shape[0]), int(signal.shape[1])
    weighted = k_weighted(signal, rate)
    weights = weights_for(channels)

    blocks = _powers(weighted, rate, MOMENTARY) @ weights
    loudness = _lufs(blocks)
    gated = blocks[loudness > ABSOLUTE_GATE]
    integrated = FLOOR
    if gated.size:
        relative = _lufs(np.array([gated.mean()]))[0] + RELATIVE_GATE
        kept = blocks[(loudness > ABSOLUTE_GATE) & (loudness > relative)]
        if kept.size:
            integrated = float(_lufs(np.array([kept.mean()]))[0])

    short = _lufs(_powers(weighted, rate, SHORT_TERM) @ weights)
    loud_range = 0.0
    above = short[short > ABSOLUTE_GATE]
    if above.size:
        mean = _lufs(np.array([np.mean(10.0 ** ((above + 0.691) / 10.0))]))[0]
        kept = above[above > mean + RANGE_GATE]
        if kept.size > 1:
            loud_range = float(np.percentile(kept, 95) - np.percentile(kept, 10))

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    correlation = 1.0
    if channels >= 2:
        left, right = signal[0] - signal[0].mean(), signal[1] - signal[1].mean()
        spread = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
        correlation = float(np.dot(left, right)) / spread if spread > 0 else 1.0
    return {
        "integrated": _floored(integrated),
        "range": round(loud_range, 4),
        "momentary_max": _floored(float(loudness.max())) if loudness.size else FLOOR,
        "short_term_max": _floored(float(short.max())) if short.size else FLOOR,
        "true_peak": _db(true_peak(signal, rate)),
        "sample_peak": _db(peak),
        "rms": _db(rms),
        "crest": _db(peak) - _db(rms) if peak > 0 and rms > 0 else 0.0,
        "dc": float(np.max(np.abs(signal.mean(axis=-1)))) if length else 0.0,
        "correlation": round(correlation, 6),
        "clipped": int(np.count_nonzero(np.abs(signal) >= 1.0)),
        "duration": length / float(rate),
        "series": {
            "momentary": _curve(loudness, MOMENTARY),
            "short_term": _curve(short, SHORT_TERM),
        },
    }
