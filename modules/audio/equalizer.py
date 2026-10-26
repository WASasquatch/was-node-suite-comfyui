"""Parametric EQ bands read from text, the filter they build, and averaged spectra for a panel.

A band is ``kind frequency gain q``, or ``kind frequency slope`` for a cut; ``off`` bypasses one.
"""

from __future__ import annotations

import math

import numpy as np

from . import filters

__all__ = ["CUTS", "KINDS", "MAX_BANDS", "Band", "parse_bands", "sections", "spectrum"]

#: Band kinds, in the order a panel offers them.
KINDS = ("bell", "low shelf", "high shelf", "low cut", "high cut", "notch", "band pass")

#: The kinds that take a slope rather than a gain and a Q.
CUTS = ("low cut", "high cut")

#: Most bands one EQ reads.
MAX_BANDS = 16

#: Longest band text read, in characters.
MAX_TEXT = 8192

#: Points of a panel's spectrum, log spaced from 20 Hz.
SPECTRUM_POINTS = 240


class Band:
    """One EQ band.

    Attributes:
        kind: A name from :data:`KINDS`.
        freq: Hertz.
        gain: dB, for a bell or a shelf.
        q: Quality factor, for every kind but a cut.
        slope: dB per octave, for a cut.
        on: False for a bypassed band.
    """

    __slots__ = ("kind", "freq", "gain", "q", "slope", "on")

    def __init__(self, kind, freq, gain=0.0, q=0.707, slope=12, on=True):
        self.kind, self.freq, self.gain, self.q, self.slope, self.on = kind, freq, gain, q, slope, on

    def as_dict(self) -> dict:
        """The band for a panel."""
        return {"kind": self.kind, "freq": self.freq, "gain": self.gain, "q": self.q, "slope": self.slope, "on": self.on}


def _number(text: str, what: str, low: float, high: float) -> float:
    """A number from a band entry, held to a range."""
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"bands: {what} {text!r} is not a number.") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"bands: {what} {text!r} is outside {low:g} to {high:g}.")
    return value


def parse_bands(text: str) -> list[Band]:
    """Bands written in a widget.

    Args:
        text: Entries such as ``"low cut 30 24; bell 2500 -3 1.2; high shelf 9000 2 0.7"``.

    Returns:
        The bands in order.

    Raises:
        ValueError: An entry names no known kind or holds an unreadable number, or there are
            more than :data:`MAX_BANDS`.
    """
    body = str(text or "")
    if len(body) > MAX_TEXT:
        raise ValueError(f"bands holds {len(body):,} characters; at most {MAX_TEXT:,} are read.")
    bands = []
    for line in body.replace(";", "\n").splitlines():
        entry = line.strip().lower()
        if not entry or entry.startswith("#"):
            continue
        on = True
        if entry.startswith("off "):
            on, entry = False, entry[4:].strip()
        kind = next((name for name in sorted(KINDS, key=len, reverse=True) if entry.startswith(name)), None)
        if kind is None:
            raise ValueError(f"bands: {line.strip()!r} names no band kind; use one of {', '.join(KINDS)}.")
        fields = entry[len(kind):].split()
        if kind in CUTS:
            if not 1 <= len(fields) <= 2:
                raise ValueError(f"bands: {line.strip()!r} is a cut; write `{kind} 80 24`, frequency then slope.")
            freq = _number(fields[0], "frequency", 10.0, 40000.0)
            slope = _number(fields[1], "slope", 6.0, 96.0) if len(fields) > 1 else 12.0
            bands.append(Band(kind, freq, slope=int(round(slope / 6.0)) * 6, on=on))
        else:
            if not 1 <= len(fields) <= 3:
                raise ValueError(f"bands: {line.strip()!r} reads frequency, gain then Q: `{kind} 2500 -3 1.2`.")
            freq = _number(fields[0], "frequency", 10.0, 40000.0)
            gain = _number(fields[1], "gain", -36.0, 36.0) if len(fields) > 1 else 0.0
            q = _number(fields[2], "Q", 0.05, 40.0) if len(fields) > 2 else 0.707
            bands.append(Band(kind, freq, gain=gain, q=q, on=on))
        if len(bands) > MAX_BANDS:
            raise ValueError(f"bands lists more than {MAX_BANDS}; one EQ reads at most {MAX_BANDS}.")
    return bands


def sections(bands, rate: int) -> np.ndarray:
    """Second order sections of every enabled band, cascaded.

    Args:
        bands: :class:`Band` records.
        rate: Samples per second.

    Returns:
        ``(n, 6)`` sections; ``(0, 6)`` when no band is enabled.
    """
    rows = []
    for band in bands:
        if not band.on:
            continue
        if band.kind in CUTS:
            rows.append(filters.cut(band.kind, band.freq, band.slope, rate))
        else:
            rows.append(filters.biquad(band.kind, band.freq, band.q, band.gain, rate)[None, :])
    return np.vstack(rows) if rows else np.zeros((0, 6))


def spectrum(signal: np.ndarray, rate: int) -> tuple[list, list]:
    """The average spectrum of a clip on a log frequency axis, for a panel.

    Args:
        signal: ``(channels, samples)``; channels are averaged.
        rate: Samples per second.

    Returns:
        ``(freqs, levels)``: :data:`SPECTRUM_POINTS` frequencies in hertz and levels in dB.
    """
    from scipy.signal import welch

    mono = signal.mean(axis=0)
    size = int(min(8192, max(256, 2 ** int(math.log2(max(256, mono.size))))))
    freqs, power = welch(mono, fs=rate, nperseg=min(size, mono.size), scaling="spectrum")
    targets = np.geomspace(20.0, rate / 2.0, SPECTRUM_POINTS)
    levels = 10.0 * np.log10(np.maximum(np.interp(targets, freqs, power), 1e-24))
    return [round(float(f), 2) for f in targets], [round(float(v), 2) for v in levels]
