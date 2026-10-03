"""Changing a clip's speed: where each new frame falls in the source, and making it.

Positions count in source frames from 0; a position between two frames lands between them.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import clip as clips

__all__ = [
    "EXACT", "MAX_FRAMES", "MAX_SPEED", "MIN_SPEED", "MODES", "frames_at", "positions", "retimed_audio",
]

#: Slowest and fastest speed a clip plays at.
MIN_SPEED = 0.05
MAX_SPEED = 16.0

#: Most frames a retime produces.
MAX_FRAMES = 20000

#: How a frame between two source frames is made: drawn by the interpolation network, the two
#: mixed, or the nearer held.
MODES = ("interpolate", "blend", "hold")

#: Distance from a source frame within which that frame is used as it is.
EXACT = 1e-3


def positions(count: int, speeds) -> list[float]:
    """Where in the source each new frame falls.

    Args:
        count: Frames in the source.
        speeds: Speed across the source, one or more values spread evenly over it; 0.5 plays
            at half speed, 2 at double.

    Returns:
        Source positions, starting at 0, one per new frame.

    Raises:
        ValueError: The retime would produce more than :data:`MAX_FRAMES` frames.
    """
    along = clips.stretch([min(MAX_SPEED, max(MIN_SPEED, float(v))) for v in speeds], max(count, 2))
    found = [0.0]
    last = float(count - 1)
    while True:
        at = found[-1]
        low = min(int(at), len(along) - 2)
        share = at - low
        speed = along[low] * (1.0 - share) + along[low + 1] * share
        following = at + speed
        if following > last + EXACT:
            break
        found.append(min(following, last))
        if len(found) > MAX_FRAMES:
            raise ValueError(
                f"This retime would make more than {MAX_FRAMES} frames from {count}. Raise the "
                f"speed or shorten the clip."
            )
    return found


def frames_at(frames, where, mode: str, cuts, net=None, device=None, progress=None):
    """The frames at the given source positions.

    Args:
        frames: ``(frames, height, width, channels)``.
        where: Source positions, from :func:`positions`.
        mode: One of :data:`MODES`.
        cuts: One flag per neighbouring pair, True where the clip cuts; nothing is made across
            a cut.
        net: The interpolation network, for ``interpolate``.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame made.

    Returns:
        ``(len(where), height, width, channels)`` on the frames' device.
    """
    from ..model import frame_interpolation

    device = frames.device if device is None else torch.device(device)
    out = torch.empty((len(where),) + tuple(frames.shape[1:]), dtype=frames.dtype, device=frames.device)
    last = int(frames.shape[0]) - 1
    for index, at in enumerate(where):
        low = min(int(math.floor(at)), last)
        share = at - low
        if share < EXACT or low >= last:
            out[index] = frames[low]
        elif share > 1.0 - EXACT:
            out[index] = frames[low + 1]
        elif cuts[low] or mode == "hold":
            out[index] = frames[low if share < 0.5 else low + 1]
        elif mode == "blend":
            out[index] = frames[low] * (1.0 - share) + frames[low + 1] * share
        else:
            first = frames[low:low + 1, ..., :3].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            second = frames[low + 1:low + 2, ..., :3].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            made = frame_interpolation.interpolate(net, first, second, float(share))
            out[index, ..., :3] = made[0].permute(1, 2, 0).clamp(0.0, 1.0).to(out.dtype).to(out.device)
            if frames.shape[-1] > 3:
                out[index, ..., 3:] = frames[low, ..., 3:] * (1.0 - share) + frames[low + 1, ..., 3:] * share
        if progress is not None:
            progress(1)
    return out


def retimed_audio(audio, where, rate: float):
    """The source audio replayed along the new timeline, its pitch kept.

    Args:
        audio: The source ``AUDIO``, or ``None``.
        where: Source positions per new frame, from :func:`positions`.
        rate: Frames per second.

    Returns:
        ``AUDIO`` as long as the new clip, or ``None`` without audio.
    """
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None
    from .audio_timeline import retimed

    places = np.asarray(where, dtype=np.float64)
    steps = np.arange(len(places), dtype=np.float64)
    last_speed = float(places[-1] - places[-2]) if len(places) > 1 else 1.0

    def read_at(position):
        inside = np.interp(position, steps, places)
        beyond = places[-1] + (position - steps[-1]) * last_speed
        return np.where(position > steps[-1], beyond, inside)

    return retimed(audio, float(len(places)), read_at, float(rate))
