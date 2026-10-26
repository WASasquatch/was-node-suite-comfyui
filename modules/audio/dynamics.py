"""Level detection and gain computing for compressors, limiters, gates, expanders and transient shapers.

Signals are float64 numpy arrays shaped ``(channels, samples)``. Levels and gains are dB unless
named linear; channels are linked, so every channel takes one gain.
"""

from __future__ import annotations

import math

import numpy as np

from .loudness import FLOOR
from .modulation import peak_hold

__all__ = [
    "DETECTORS",
    "applied",
    "compressed_db",
    "curve_for_panel",
    "detected",
    "expanded_db",
    "gated_db",
    "limited",
    "looked_ahead",
    "sidechain",
    "smoothed",
    "summary",
    "to_db",
]

#: How a detector reads level, in menu order.
DETECTORS = ("peak", "rms")

#: Window the RMS detector averages over, in seconds.
RMS_WINDOW = 0.01

#: Most points of a gain curve handed to a panel.
PANEL_POINTS = 1200


def to_db(level: np.ndarray) -> np.ndarray:
    """Linear amplitudes in dB, floored.

    Args:
        level: Linear amplitudes.

    Returns:
        dB, never below the loudness floor.
    """
    with np.errstate(divide="ignore"):
        return np.maximum(20.0 * np.log10(np.maximum(level, 1e-30)), FLOOR)


def detected(signal: np.ndarray, rate: int, detector: str = "peak") -> np.ndarray:
    """The level of the loudest channel at every sample.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        detector: ``peak`` reads each sample; ``rms`` averages power over 10 ms.

    Returns:
        Linear levels, one per sample.
    """
    if detector != "rms":
        return np.max(np.abs(signal), axis=0)
    span = max(1, int(round(RMS_WINDOW * rate)))
    power = np.max(signal * signal, axis=0)
    running = np.concatenate([[0.0], np.cumsum(power)])
    index = np.arange(power.size)
    first = np.maximum(0, index - span + 1)
    mean = (running[index + 1] - running[first]) / (index + 1 - first)
    return np.sqrt(np.maximum(mean, 0.0))


def smoothed(level: np.ndarray, rate: int, attack: float, release: float) -> np.ndarray:
    """A level that rises over ``attack`` and falls over ``release`` seconds.

    Args:
        level: Linear levels, one per sample.
        rate: Samples per second.
        attack: Rise time constant in seconds.
        release: Fall time constant in seconds.

    Returns:
        Linear levels: a falling peak hold for the release, then a one pole smoother for the attack.
    """
    from scipy.signal import lfilter

    fall = math.exp(-1.0 / max(1.0, release * rate))
    rise = math.exp(-1.0 / max(1.0, attack * rate))
    held = peak_hold(level, fall)
    if rise <= 0.0:
        return held
    return lfilter([1.0 - rise], [1.0, -rise], held)


def compressed_db(level_db: np.ndarray, threshold: float, ratio: float, knee: float) -> np.ndarray:
    """Gain a compressor applies at each level, with a soft knee.

    Args:
        level_db: Detected levels in dBFS.
        threshold: Level reduction starts at, in dBFS.
        ratio: Input dB per output dB above the threshold; 1 is no change.
        knee: Width in dB over which the ratio eases in.

    Returns:
        Gains in dB, at or below 0.
    """
    over = level_db - float(threshold)
    slope = 1.0 / max(1.0, float(ratio)) - 1.0
    width = max(0.0, float(knee))
    gain = np.where(over > 0.0, slope * over, 0.0)
    if width > 0.0:
        inside = np.abs(over) <= width / 2.0
        gain = np.where(inside, slope * (over + width / 2.0) ** 2 / (2.0 * width), gain)
    return gain


def expanded_db(level_db: np.ndarray, threshold: float, ratio: float, depth: float, knee: float) -> np.ndarray:
    """Gain a downward expander applies at each level.

    Args:
        level_db: Detected levels in dBFS.
        threshold: Level below which the signal is pushed down, in dBFS.
        ratio: dB of fall per dB under the threshold; 1 is no change.
        depth: Most attenuation in dB.
        knee: Width in dB over which the ratio eases in.

    Returns:
        Gains in dB, from ``-depth`` to 0.
    """
    under = float(threshold) - level_db
    slope = max(1.0, float(ratio)) - 1.0
    width = max(0.0, float(knee))
    gain = np.where(under > 0.0, -slope * under, 0.0)
    if width > 0.0:
        inside = np.abs(under) <= width / 2.0
        gain = np.where(inside, -slope * (under + width / 2.0) ** 2 / (2.0 * width), gain)
    return np.maximum(gain, -abs(float(depth)))


def gated_db(level_db: np.ndarray, rate: int, threshold: float, hysteresis: float, depth: float,
             attack: float, hold: float, release: float) -> np.ndarray:
    """Gain of a gate that opens above a threshold and closes below it less a hysteresis.

    Args:
        level_db: Detected levels in dBFS, one per sample.
        rate: Samples per second.
        threshold: Level the gate opens at, in dBFS.
        hysteresis: dB below the threshold the gate waits for before closing.
        depth: Attenuation in dB while closed.
        attack: Seconds to open.
        hold: Seconds the gate stays open after the level drops.
        release: Seconds to close.

    Returns:
        Gains in dB, one per sample.
    """
    hop = max(1, int(round(rate * 0.001)))
    frames = max(1, math.ceil(level_db.size / hop))
    padded = np.full(frames * hop, FLOOR)
    padded[: level_db.size] = level_db
    peaks = padded.reshape(frames, hop).max(axis=1).tolist()
    opening = 1.0 - math.exp(-hop / max(1.0, attack * rate))
    closing = 1.0 - math.exp(-hop / max(1.0, release * rate))
    hold_frames = int(round(hold * rate / hop))
    close_at = float(threshold) - abs(float(hysteresis))
    floor = 10.0 ** (-abs(float(depth)) / 20.0)
    gains = [0.0] * frames
    is_open, held, gain = False, 0, floor
    for index, peak in enumerate(peaks):
        if peak >= threshold:
            is_open, held = True, hold_frames
        elif peak < close_at:
            if held > 0:
                held -= 1
            else:
                is_open = False
        target = 1.0 if is_open else floor
        gain += (target - gain) * (opening if target > gain else closing)
        gains[index] = gain
    centres = (np.arange(frames) + 0.5) * hop
    linear = np.interp(np.arange(level_db.size), centres, np.asarray(gains))
    return to_db(linear)


def looked_ahead(gain_db: np.ndarray, samples: int) -> np.ndarray:
    """Gains moved earlier so they act before the sound that caused them arrives.

    Args:
        gain_db: Gains in dB, one per sample.
        samples: How far ahead to look.

    Returns:
        At each sample, the lowest gain within the next ``samples`` samples.
    """
    samples = int(samples)
    if samples <= 0 or gain_db.size == 0:
        return gain_db
    from scipy.ndimage import minimum_filter1d

    return minimum_filter1d(gain_db, size=samples + 1, origin=-(samples // 2) - (samples % 2), mode="nearest")


def applied(signal: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """A signal with a gain in dB applied at every sample.

    Args:
        signal: ``(channels, samples)``.
        gain_db: Gains in dB, one per sample.

    Returns:
        ``(channels, samples)``.
    """
    return signal * (10.0 ** (gain_db / 20.0))[None, :]


def _limit_pass(signal: np.ndarray, rate: int, ceiling_db: float, release: float, lookahead: float,
                true_peak: bool) -> np.ndarray:
    """One pass of :func:`limited`."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    level = np.max(np.abs(signal), axis=0)
    if true_peak and rate < 192000:
        from scipy.signal import resample_poly

        factor = 4 if rate < 96000 else 2
        between = np.max(np.abs(resample_poly(signal, factor, 1, axis=-1)), axis=0)
        between = between[: level.size * factor].reshape(-1, factor).max(axis=1)
        level = np.maximum(level, np.pad(between, (0, level.size - between.size)))
    ceiling = 10.0 ** (float(ceiling_db) / 20.0)
    target = np.minimum(1.0, ceiling / np.maximum(level, 1e-30))
    span = max(1, int(round(lookahead * rate)))
    # The lowest gain over the next span samples, then averaged over the span before.
    ahead = minimum_filter1d(target, size=span + 1, origin=-(span // 2) - (span % 2), mode="nearest")
    eased = np.minimum(uniform_filter1d(ahead, size=span + 1, origin=span // 2, mode="nearest"), ahead)
    # Recovery capped at a fixed rate in dB, never above the eased gain.
    climb = 8.685889638 / max(1e-4, release * rate)
    logged = to_db(eased)
    ramp = np.arange(logged.size, dtype=np.float64) * climb
    return np.minimum(np.minimum.accumulate(logged - ramp) + ramp, logged)


def limited(signal: np.ndarray, rate: int, ceiling_db: float, release: float, lookahead: float,
            true_peak: bool) -> np.ndarray:
    """Gain of a brickwall limiter that keeps every peak at or under a ceiling.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        ceiling_db: Highest level let through, in dBFS, or dBTP with ``true_peak``.
        release: Seconds the gain takes to recover.
        lookahead: Seconds the gain starts falling before a peak.
        true_peak: Read peaks between samples, four times oversampled.

    Returns:
        Gains in dB, one per sample.
    """
    gain = _limit_pass(signal, rate, ceiling_db, release, lookahead, true_peak)
    # A second pass over the limited signal trims the peaks the first pass's gain changes made.
    trim = _limit_pass(applied(signal, gain), rate, ceiling_db, release, lookahead, true_peak)
    return gain + trim


def curve_for_panel(gain_db: np.ndarray, rate: int) -> dict:
    """A gain curve thinned for a panel, keeping each stretch's deepest reduction.

    Args:
        gain_db: Gains in dB, one per sample.
        rate: Samples per second.

    Returns:
        ``offset``, ``step`` in seconds and ``values`` in dB.
    """
    count = int(gain_db.size)
    if count == 0:
        return {"offset": 0.0, "step": 0.0, "values": []}
    points = min(PANEL_POINTS, count)
    edges = np.linspace(0, count, points + 1).astype(np.int64)
    values = [round(float(gain_db[a:b].min()), 2) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    return {"offset": 0.0, "step": count / points / rate, "values": values}


def sidechain(audio, rate: int, length: int, batch: int, index: int) -> np.ndarray | None:
    """One clip of a sidechain input, brought to the main clip's rate and length.

    Args:
        audio: An ``AUDIO`` mapping, or None for no sidechain.
        rate: The main clip's samples per second.
        length: The main clip's samples.
        batch: Clips in the main batch.
        index: Which of them this is.

    Returns:
        ``(channels, length)`` float64, silence past the sidechain's end, or None.

    Raises:
        ValueError: The sidechain's batch cannot be paired with the main batch.
    """
    if audio is None:
        return None
    from .clip import conformed, rate_of, waveform_of

    wave = waveform_of(audio, "sidechain")
    own = int(wave.shape[0])
    if own not in (1, batch):
        raise ValueError(f"sidechain holds {own} clips and audio {batch}; they pair one to one, or one to many.")
    picked = conformed(wave[min(index, own - 1):min(index, own - 1) + 1], rate_of(audio, "sidechain"), rate, int(wave.shape[1]))[0]
    out = np.zeros((int(picked.shape[0]), length))
    count = min(length, int(picked.shape[-1]))
    out[:, :count] = picked[:, :count].double().numpy()
    return out


def summary(kind: str, gain_db: np.ndarray, rate: int, **settings) -> dict:
    """The figures a dynamics panel draws for one clip.

    Args:
        kind: ``compressor``, ``limiter``, ``gate``, ``expander``, ``de-esser``, ``multiband``
            or ``transient``.
        gain_db: The gain applied, in dB, one per sample.
        rate: Samples per second.
        **settings: The settings the panel's curve is drawn from, such as ``threshold``.

    Returns:
        ``kind``, the settings, ``gr`` (the gain curve), ``max_gr`` and ``avg_gr`` in dB of
        reduction.
    """
    reduction = -np.minimum(gain_db, 0.0)
    return {
        "kind": kind,
        **{key: float(value) if isinstance(value, (int, float)) else value for key, value in settings.items()},
        "gr": curve_for_panel(gain_db, rate),
        "max_gr": round(float(reduction.max()) if reduction.size else 0.0, 3),
        "avg_gr": round(float(reduction.mean()) if reduction.size else 0.0, 3),
    }
