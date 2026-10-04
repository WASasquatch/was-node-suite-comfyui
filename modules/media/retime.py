"""Changing a clip's speed: where each new frame falls in the source, and making it.

Positions count in source frames from 0; a position between two frames lands between them.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import torch

from ..image import scratch
from . import clip as clips

__all__ = [
    "EXACT",
    "MAX_FRAMES",
    "MAX_SPEED",
    "MIN_SPEED",
    "MODES",
    "AUTO_UNEXPLAINED",
    "check_network",
    "drawable",
    "frames_at",
    "positions",
    "rate_positions",
    "retimed_audio",
]

#: Slowest and fastest speed a clip plays at.
MIN_SPEED = 0.05
MAX_SPEED = 16.0

#: Most frames a retime produces.
MAX_FRAMES = 20000

#: How a frame between two source frames is made: drawn by the interpolation network, the two
#: mixed, the nearer held, or drawn only where the motion accounts for the change.
MODES = ("interpolate", "blend", "hold", "auto")

#: Share of a pair's changed pixels the motion may leave unaccounted for and the pair still be
#: drawn on ``auto``.
AUTO_UNEXPLAINED = 0.75

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


def rate_positions(count: int, source_rate, target_rate) -> list[float]:
    """Where in the source each frame falls once the clip plays at another frame rate, its length kept.

    Args:
        count: Frames in the source.
        source_rate: The source's frames per second.
        target_rate: Frames per second wanted.

    Returns:
        Source positions, one per new frame; any past the last source frame hold on it.

    Raises:
        ValueError: The clip would hold more than :data:`MAX_FRAMES` frames.
    """
    step = Fraction(source_rate) / Fraction(target_rate)
    total = max(1, round(Fraction(count) / step))
    if total > MAX_FRAMES:
        raise ValueError(
            f"{count} frames at {float(target_rate):g} fps would make {total} frames, more than "
            f"{MAX_FRAMES}. Lower the frame rate or shorten the clip."
        )
    last = float(count - 1)
    return [min(float(index * step), last) for index in range(total)]


def check_network(name: str, where) -> None:
    """Raise unless an EMA-VFI checkpoint can land at every in-between position given.

    Args:
        name: The checkpoint's file name.
        where: Source positions, from :func:`positions` or :func:`rate_positions`.

    Raises:
        ValueError: A position falls off the halfway point and the checkpoint only lands there.
    """
    from ..model import frame_interpolation

    shares = [at - int(at) for at in where]
    between = [share for share in shares if EXACT < share < 1.0 - EXACT]
    if not any(abs(share - 0.5) > 1e-3 for share in between):
        return
    if frame_interpolation.spec_for(name).get("any_timestep", False):
        return
    choices = ", ".join(
        entry for entry, spec in frame_interpolation.CHECKPOINTS.items() if spec["any_timestep"]
    )
    raise ValueError(
        f"{name} only lands halfway between two frames, which suits half speed or twice the "
        f"frame rate. For these frames choose one of: {choices}."
    )


def drawable(motion) -> list[bool]:
    """Which neighbouring pairs ``auto`` draws new frames between.

    Args:
        motion: A :class:`~modules.image.motion.Motion` of the source clip.

    Returns:
        One flag per pair: True where the pair moves, is no cut, and its motion accounts for
        the change.
    """
    import comfy.model_management

    device = comfy.model_management.get_torch_device()
    cuts, holds = motion.cuts(), motion.holds()
    left = motion.unexplained(device)
    return [
        not cut and not held and share <= AUTO_UNEXPLAINED
        for cut, held, share in zip(cuts, holds, left)
    ]


def frames_at(
    frames, where, mode: str, cuts, net=None, device=None, progress=None, name: str = "",
    drawn=None,
):
    """The frames at the given source positions.

    Args:
        frames: ``(frames, height, width, channels)`` on the CPU, float or uint8 codes.
        where: Source positions, from :func:`positions`.
        mode: One of :data:`MODES`.
        cuts: One flag per neighbouring pair, True where the clip cuts; nothing is made across
            a cut.
        net: The interpolation network, for ``interpolate``.
        device: Where the work runs.
        progress: Optional callable taking a step count, called once per frame made.
        name: Display name of the calling node, for the log and the refusal.
        drawn: One flag per pair for ``auto``, from :func:`drawable`: the network draws where
            True and the nearer frame is held elsewhere.

    Returns:
        ``(len(where), height, width, channels)`` on the CPU, of the frames' own type.

    Raises:
        MemoryError: Neither free memory nor a scratch drive can hold the frames.
    """
    from ..image.optical_flow import unit
    from ..model import frame_interpolation

    codes = frames.dtype == torch.uint8

    def kept(value):
        if codes:
            return value.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        return value.to(frames.dtype)

    device = frames.device if device is None else torch.device(device)
    out = scratch.allocate(
        (len(where),) + tuple(frames.shape[1:]),
        frames.dtype,
        node=name,
        advice="A higher speed or a shorter clip also fits it.",
    )
    last = int(frames.shape[0]) - 1
    for index, at in enumerate(where):
        low = min(int(math.floor(at)), last)
        share = at - low
        if share < EXACT or low >= last:
            out[index] = frames[low]
        elif share > 1.0 - EXACT:
            out[index] = frames[low + 1]
        elif cuts[low] or mode == "hold" or (mode == "auto" and not (drawn and drawn[low])):
            out[index] = frames[low if share < 0.5 else low + 1]
        elif mode == "blend":
            out[index] = kept(unit(frames[low]) * (1.0 - share) + unit(frames[low + 1]) * share)
        else:
            first = unit(frames[low:low + 1, ..., :3], device).permute(0, 3, 1, 2)
            second = unit(frames[low + 1:low + 2, ..., :3], device).permute(0, 3, 1, 2)
            made = frame_interpolation.interpolate(net, first, second, float(share))
            out[index, ..., :3] = kept(made[0].permute(1, 2, 0).clamp(0.0, 1.0)).to(out.device)
            if frames.shape[-1] > 3:
                out[index, ..., 3:] = kept(
                    unit(frames[low, ..., 3:]) * (1.0 - share) + unit(frames[low + 1, ..., 3:]) * share
                )
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
