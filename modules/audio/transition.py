"""One clip handed over to the next across an overlap, in one of several styles.

Waveforms are ``(batch, channels, samples)`` float tensors at one rate and channel count.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import curves, filters
from .clip import check_size
from .resample import read_at

__all__ = ["STYLES", "transitioned"]

#: The styles a transition can take, in menu order.
STYLES = (
    "crossfade",
    "dip to silence",
    "cut",
    "filter sweep",
    "echo out",
    "tape stop",
    "backspin",
)

#: Seconds a hard cut is smoothed over so it does not click.
CUT_SMOOTHING = 0.005

#: Share of the overlap the next clip fades in over after a tape stop or a backspin.
DROP_SHARE = 0.12

#: Level of the repeats an echo out leaves, against the clip that feeds them.
ECHO_LEVEL = 0.5

#: Cutoffs the filter sweep runs between, in hertz.
SWEEP_HIGH = 18000.0
SWEEP_LOW = 120.0
SWEEP_ENTRY = 1500.0
SWEEP_FLOOR = 20.0


def _progress(count: int, curve: str) -> np.ndarray:
    """A curve's gains as numpy, rising."""
    return curves.rising(count, curve).double().numpy()


def _sweep(start: float, end: float, progress: np.ndarray) -> np.ndarray:
    """Frequencies moved between two ends on a log scale."""
    return np.exp(np.log(start) + (np.log(end) - np.log(start)) * progress)


def _spin_positions(count: int, origin: float) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and speeds of a record spun backwards and let run down."""
    unit = (np.arange(count) + 0.5) / max(1, count)
    kick = 0.15
    speed = np.where(
        unit < kick,
        1.0 - 4.0 * (unit / kick),
        -3.0 * (1.0 - (unit - kick) / (1.0 - kick)) ** 2,
    )
    return origin + np.cumsum(speed), speed


def _stop_positions(count: int, origin: float, curve: str) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and speeds of a tape slowing to a halt."""
    speed = 1.0 - _progress(count, curve)
    return origin + np.cumsum(speed), speed


def transitioned(a: torch.Tensor, b: torch.Tensor, rate: int, overlap: int, style: str,
                 curve: str) -> tuple[torch.Tensor, int, int]:
    """Two clips joined across an overlap.

    Args:
        a: The clip that ends, ``(batch, channels, samples)``.
        b: The clip that follows, the same batch and channels.
        rate: Samples per second of both.
        overlap: Samples the two share; held to the shorter clip.
        style: A name from :data:`STYLES`.
        curve: The fades' curve, from :data:`curves.CURVES`.

    Returns:
        ``(joined, seam_start, seam_end)``: the joined clip and where the overlap sits in it.
    """
    length_a, length_b = int(a.shape[-1]), int(b.shape[-1])
    o = max(0, min(int(overlap), length_a, length_b))
    batch, channels = int(a.shape[0]), int(a.shape[1])
    total = length_a + length_b - o
    seam = length_a - o
    tail_extra = 0
    echo_tail = None
    if style == "echo out" and o:
        delay = min(1.0, max(0.08, o / rate / 4.0))
        feedback = 0.55
        fed = a[..., seam:].double().numpy().reshape(batch * channels, o)
        fed = fed * _progress(o, curve)[::-1][None, :]
        tail = max(0.0, math.log(0.001) / math.log(feedback) * delay - o / rate)
        echo_tail = filters.echo(fed, rate, delay, feedback, tail)
        tail_extra = max(0, int(echo_tail.shape[-1]) - length_b)
    check_size(batch, channels, total + tail_extra, "the overlap")
    out = torch.zeros(batch, channels, total + tail_extra, dtype=torch.float32)
    out[..., :seam] = a[..., :seam]
    out[..., length_a:total] = b[..., o:]
    if o == 0:
        out[..., seam:seam + length_b] = b
        return out[..., :total].contiguous(), seam, seam
    a_tail = a[..., seam:].double()
    b_head = b[..., :o].double()
    rise = torch.from_numpy(_progress(o, curve))
    fall = rise.flip(0)

    if style == "dip to silence":
        half = o // 2
        mixed = torch.zeros_like(a_tail)
        mixed[..., :half] = a_tail[..., :half] * curves.falling(half, curve).double()
        mixed[..., half:] = b_head[..., half:] * curves.rising(o - half, curve).double()
    elif style == "cut":
        half = o // 2
        smooth = min(half, max(1, int(round(CUT_SMOOTHING * rate))))
        gate = torch.zeros(o, dtype=torch.float64)
        gate[half:] = 1.0
        gate[half - smooth:half + smooth] = curves.rising(2 * smooth, "equal power").double()[: o - half + smooth]
        mixed = a_tail * torch.sqrt(torch.clamp(1.0 - gate ** 2, min=0.0)) + b_head * gate
    elif style == "filter sweep":
        flat_a = a_tail.numpy().reshape(batch * channels, o)
        flat_b = b_head.numpy().reshape(batch * channels, o)
        progress = rise.numpy()
        lowered = filters.swept(flat_a, rate, _sweep(SWEEP_HIGH, SWEEP_LOW, progress), 0.9, "low")
        raised = filters.swept(flat_b, rate, _sweep(SWEEP_ENTRY, SWEEP_FLOOR, progress), 0.9, "high")
        mixed = (torch.from_numpy(lowered).reshape(a_tail.shape) * fall
                 + torch.from_numpy(raised).reshape(b_head.shape) * rise)
    elif style == "echo out":
        mixed = a_tail * fall * fall + b_head * rise
    elif style in ("tape stop", "backspin"):
        if style == "tape stop":
            where, speed = _stop_positions(o, float(seam), curve)
        else:
            where, speed = _spin_positions(o, float(seam))
        played = read_at(a, where, speed).double()
        drop = max(1, int(round(o * DROP_SHARE)))
        entry = torch.zeros(o, dtype=torch.float64)
        entry[o - drop:] = curves.rising(drop, curve).double()
        hush = torch.ones(o, dtype=torch.float64)
        hush[o - drop:] = curves.falling(drop, curve).double()
        mixed = played * hush + b_head * entry
    else:
        mixed = a_tail * fall + b_head * rise

    out[..., seam:length_a] = mixed.to(torch.float32)
    if echo_tail is not None:
        repeats = torch.from_numpy(echo_tail * ECHO_LEVEL).reshape(batch, channels, -1).to(torch.float32)
        out[..., seam:seam + repeats.shape[-1]] += repeats
    end = total if echo_tail is None else max(total, seam + int(echo_tail.shape[-1]))
    return out[..., :end].contiguous(), seam, length_a
