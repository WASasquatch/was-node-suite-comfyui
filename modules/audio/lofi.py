"""Worn media: tape, vinyl, radio, phone and VHS, as pitch wobble, noise, crackle, band limits and saturation.

Signals are float64 numpy arrays shaped ``(channels, samples)``. ``amounts`` run ``0`` to ``1``.
"""

from __future__ import annotations

import math

import numpy as np

from . import delays, filters, modulation

__all__ = ["MEDIA", "worn"]

#: Each medium: low cut and high cut in hertz, wow and flutter depth in milliseconds and rate in
#: hertz, hiss level in dBFS, crackle per second, rumble level in dBFS, saturation drive, mono.
MEDIA = {
    "cassette": dict(low=40.0, high=12000.0, wow=(0.6, 0.6), flutter=(0.08, 9.0), hiss=-52.0,
                     crackle=0.0, rumble=-90.0, drive=1.6, mono=False),
    "vinyl": dict(low=25.0, high=15000.0, wow=(0.35, 0.55), flutter=(0.02, 6.0), hiss=-62.0,
                  crackle=12.0, rumble=-48.0, drive=1.1, mono=False),
    "AM radio": dict(low=200.0, high=4500.0, wow=(0.0, 0.5), flutter=(0.0, 5.0), hiss=-40.0,
                     crackle=1.0, rumble=-90.0, drive=2.4, mono=True),
    "telephone": dict(low=300.0, high=3400.0, wow=(0.0, 0.5), flutter=(0.0, 5.0), hiss=-50.0,
                      crackle=0.0, rumble=-90.0, drive=3.0, mono=True),
    "VHS": dict(low=50.0, high=9000.0, wow=(0.9, 0.35), flutter=(0.12, 11.0), hiss=-46.0,
                crackle=0.0, rumble=-80.0, drive=1.8, mono=False),
    "worn tape": dict(low=60.0, high=7000.0, wow=(1.8, 0.4), flutter=(0.25, 8.0), hiss=-40.0,
                      crackle=0.5, rumble=-70.0, drive=2.6, mono=False),
}


def _pink(rng, shape) -> np.ndarray:
    """Pink-ish noise at about unit RMS."""
    from scipy.signal import lfilter

    white = rng.standard_normal(shape)
    pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                   [1.0, -2.494956002, 2.017265875, -0.522189400], white, axis=-1)
    return pink / max(1e-9, float(np.sqrt(np.mean(pink * pink))))


def worn(signal: np.ndarray, rate: int, medium: str, age: float, wow: float, flutter: float, noise: float,
         crackle: float, bandwidth: float, saturation: float, seed: int) -> np.ndarray:
    """A signal played back from a worn medium.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        medium: A name from :data:`MEDIA`.
        age: Scales every other amount, ``0`` to ``1``.
        wow: Slow pitch drift amount.
        flutter: Fast pitch flutter amount.
        noise: Hiss and rumble amount.
        crackle: Clicks and pops amount.
        bandwidth: How far the band narrows beyond the medium's own.
        saturation: Drive amount.
        seed: Seed of every random part.

    Returns:
        ``(channels, samples)``.
    """
    from scipy.signal import lfilter, sosfilt

    profile = MEDIA.get(medium, MEDIA["cassette"])
    rng = np.random.default_rng(int(seed))
    channels, length = signal.shape
    scale = 0.25 + 0.75 * float(age)
    out = signal.mean(axis=0, keepdims=True).repeat(channels, axis=0) if profile["mono"] else signal.copy()

    wow_ms, wow_hz = profile["wow"]
    flutter_ms, flutter_hz = profile["flutter"]
    depth_wow = wow_ms * float(wow) * scale * 2.0
    depth_flutter = flutter_ms * float(flutter) * scale * 2.0
    if depth_wow > 0.0 or depth_flutter > 0.0:
        drift = depth_wow * (0.7 * modulation.lfo(length, rate, wow_hz, "sine")
                             + 0.3 * modulation.lfo(length, rate, wow_hz * 0.37, "random smooth", seed=seed))
        shake = depth_flutter * modulation.lfo(length, rate, flutter_hz, "random smooth", seed=seed + 1)
        centre = depth_wow + depth_flutter + 0.5
        out = delays.read_delayed(out, (centre + drift + shake) * rate / 1000.0)
        # The wobble's average delay is taken back out, so the result lines up with the input.
        lead = int(round(centre * rate / 1000.0))
        if 0 < lead < length:
            out = np.concatenate([out[:, lead:], np.zeros((channels, lead))], axis=-1)

    narrow = 1.0 - 0.6 * float(bandwidth) * scale
    high = profile["high"] * max(0.15, narrow)
    low = profile["low"] / max(0.3, narrow)
    band = np.vstack([filters.cut("low cut", low, 12, rate), filters.cut("high cut", min(high, 0.45 * rate), 24, rate)])
    out = sosfilt(band, out, axis=-1)

    drive = 1.0 + (profile["drive"] - 1.0) * float(saturation) * 2.0 * scale
    if drive > 1.0001:
        out = np.tanh(out * drive) / math.tanh(drive)

    amount = float(noise) * scale
    if amount > 0.0:
        hiss = _pink(rng, out.shape) * 10.0 ** (profile["hiss"] / 20.0) * amount
        out = out + sosfilt(filters.cut("high cut", min(high * 1.2, 0.45 * rate), 12, rate), hiss, axis=-1)
        if profile["rumble"] > -89.0:
            rumble = rng.standard_normal(out.shape)
            rumble = sosfilt(filters.cut("high cut", 30.0, 24, rate), rumble, axis=-1)
            rumble *= 10.0 ** (profile["rumble"] / 20.0) * amount / max(1e-9, float(np.sqrt(np.mean(rumble * rumble))))
            out = out + rumble

    pops = profile["crackle"] * float(crackle) * scale
    if pops > 0.0:
        count = rng.poisson(pops * length / rate)
        clicks = np.zeros((channels, length))
        where = rng.integers(0, max(1, length), count)
        sizes = rng.uniform(0.02, 0.35, count) * rng.choice([-1.0, 1.0], count)
        for channel in range(channels):
            np.add.at(clicks[channel], where, sizes)
        decay = math.exp(-1.0 / max(1.0, 0.0004 * rate))
        clicks = lfilter([1.0], [1.0, -decay], clicks, axis=-1)
        out = out + sosfilt(filters.cut("high cut", min(9000.0, 0.45 * rate), 12, rate), clicks, axis=-1)
    return out
