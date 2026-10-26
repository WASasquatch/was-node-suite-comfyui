"""Mixer tracks and the mix they sum to.

A track is one clip with its fader, pan, mute, solo and start time. Pan runs ``-1`` (left)
to ``1`` (right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .clip import check_size, conformed, rate_of, waveform_of

__all__ = ["LAYOUTS", "LENGTHS", "MAX_TRACKS", "Track", "mixed", "pan_gains"]

#: Most tracks one mixer sums.
MAX_TRACKS = 64

#: How long a mix runs, in menu order.
LENGTHS = ("longest track", "first track", "fixed")

#: The channel layouts a mix is made in, in menu order.
LAYOUTS = ("stereo", "mono")


@dataclass(frozen=True)
class Track:
    """One clip on a mixer.

    Attributes:
        audio: The ``AUDIO`` mapping.
        gain_db: Fader level in dB.
        pan: ``-1`` left to ``1`` right.
        mute: Left out of the mix.
        solo: Only soloed tracks play while any track is soloed.
        start: Seconds into the mix the clip begins.
        name: The label a panel draws.
    """

    audio: dict
    gain_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    start: float = 0.0
    name: str = ""


def pan_gains(pan: float, channels: int) -> tuple[float, float]:
    """Left and right gains for a track placed in a stereo mix.

    Args:
        pan: ``-1`` left to ``1`` right.
        channels: The track's own channels. A mono track is placed at constant power,
            3 dB down each side at the centre; a stereo track is balanced, unchanged at the
            centre and the far side faded at constant power.

    Returns:
        ``(left, right)`` linear gains.
    """
    pan = max(-1.0, min(1.0, float(pan)))
    if channels == 1:
        angle = (pan + 1.0) * math.pi / 4.0
        return math.cos(angle), math.sin(angle)
    return math.cos(max(pan, 0.0) * math.pi / 2.0), math.cos(max(-pan, 0.0) * math.pi / 2.0)


def mixed(tracks, master_db: float, length: str, duration: float, layout: str) -> tuple[torch.Tensor, int, list[dict]]:
    """Every audible track summed into one clip.

    Args:
        tracks: :class:`Track` records in chain order.
        master_db: Master fader in dB.
        length: A name from :data:`LENGTHS`.
        duration: Seconds of a ``fixed`` mix.
        layout: A name from :data:`LAYOUTS`.

    Returns:
        ``(waveform, rate, report)``: the mix at the highest rate among the tracks, and one
        row per track with its ``name``, ``state``, ``gain_db``, ``pan``, ``start``,
        ``duration`` and ``peak_db`` after its fader and pan.

    Raises:
        ValueError: There are no tracks, too many, or their batches cannot be paired.
    """
    tracks = list(tracks or ())
    if not tracks:
        raise ValueError("the mixer has no tracks. Wire Audio Mixer Track nodes into tracks.")
    if len(tracks) > MAX_TRACKS:
        raise ValueError(f"the chain carries {len(tracks)} tracks; one mixer sums at most {MAX_TRACKS}.")
    names = [track.name or f"track {index + 1}" for index, track in enumerate(tracks)]
    waves = [waveform_of(track.audio, name) for track, name in zip(tracks, names)]
    rates = [rate_of(track.audio, name) for track, name in zip(tracks, names)]
    rate = max(rates)
    channels = 1 if layout == "mono" else 2
    batches = {int(wave.shape[0]) for wave in waves}
    batch = max(batches)
    if len(batches - {1}) > 1:
        raise ValueError(
            f"the tracks carry batches of {sorted(batches)} clips; a mix pairs clips one to one, "
            f"so every track holds the same count, or 1."
        )
    soloing = any(track.solo for track in tracks)
    starts = [max(0, int(round(float(track.start) * rate))) for track in tracks]
    lengths = [int(round(int(w.shape[-1]) * rate / r)) for w, r in zip(waves, rates)]
    if length == "fixed":
        total = max(1, int(round(float(duration) * rate)))
    elif length == "first track":
        total = starts[0] + lengths[0]
    else:
        total = max(start + count for start, count in zip(starts, lengths))
    check_size(batch, channels, total, "the track start times or the fixed duration")
    out = torch.zeros(batch, channels, total, dtype=torch.float32)
    report = []
    for track, wave, track_rate, start, name in zip(tracks, waves, rates, starts, names):
        state = "muted" if track.mute else "silent while others solo" if soloing and not track.solo else "playing"
        row = {
            "name": name, "state": state, "gain_db": round(float(track.gain_db), 2),
            "pan": round(float(track.pan), 3), "start": round(float(track.start), 3),
            "duration": round(int(wave.shape[-1]) / track_rate, 3), "peak_db": None,
        }
        report.append(row)
        if state != "playing" or start >= total:
            continue
        own = int(wave.shape[1])
        placed = conformed(wave, track_rate, rate, 1 if own == 1 else 2)
        gain = 10.0 ** (float(track.gain_db) / 20.0)
        if channels == 2:
            left, right = pan_gains(track.pan, int(placed.shape[1]))
            placed = conformed(placed, rate, rate, 2) * torch.tensor([left, right]).view(1, 2, 1)
        else:
            placed = placed.mean(dim=1, keepdim=True)
        placed = placed * gain
        count = min(int(placed.shape[-1]), total - start)
        out[..., start:start + count] += placed[..., :count]
        peak = float(placed[..., :count].abs().max()) if count else 0.0
        row["peak_db"] = round(20.0 * math.log10(peak), 2) if peak > 0 else None
    out *= 10.0 ** (float(master_db) / 20.0)
    return out, rate, report
