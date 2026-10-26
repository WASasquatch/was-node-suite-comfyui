"""Delay lines: a signal read back at a moving fractional delay, and a feedback loop run in blocks.

Signals are float64 numpy arrays shaped ``(channels, samples)``. Delays are in samples.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CHUNK", "feedback_loop", "read_delayed"]

#: Output samples one pass of :func:`read_delayed` reads.
CHUNK = 1 << 16


def read_delayed(signal: np.ndarray, delay, cubic: bool = True) -> np.ndarray:
    """A signal read back a fractional number of samples late, the delay free to move each sample.

    Args:
        signal: ``(channels, samples)``.
        delay: Samples of delay, one number or one per sample, at least 0.
        cubic: Interpolate on a Catmull-Rom cubic; False reads straight lines between samples.

    Returns:
        ``(channels, samples)``, silent where the read falls before the start.
    """
    channels, length = signal.shape
    lag = np.broadcast_to(np.maximum(0.0, np.asarray(delay, dtype=np.float64)), (length,))
    out = np.empty((channels, length))
    for first in range(0, length, CHUNK):
        last = min(length, first + CHUNK)
        where = np.arange(first, last, dtype=np.float64) - lag[first:last]
        base = np.floor(where).astype(np.int64)
        frac = where - base

        def tap(offset: int) -> np.ndarray:
            index = base + offset
            valid = (index >= 0) & (index < length)
            return np.where(valid[None, :], signal[:, np.clip(index, 0, length - 1)], 0.0)

        if not cubic:
            out[:, first:last] = tap(0) * (1.0 - frac) + tap(1) * frac
            continue
        before, here, after, beyond = tap(-1), tap(0), tap(1), tap(2)
        f2 = frac * frac
        f3 = f2 * frac
        out[:, first:last] = (
            here
            + 0.5 * frac * (after - before)
            + f2 * (before - 2.5 * here + 2.0 * after - 0.5 * beyond)
            + f3 * (1.5 * (here - after) + 0.5 * (beyond - before))
        )
    return out


def feedback_loop(source: np.ndarray, delay: int, feedback: float, tail: int = 0, shape=None) -> np.ndarray:
    """The repeats a feedback delay makes of a signal, the signal itself left out, a delay at a time.

    Args:
        source: ``(channels, samples)`` feeding the loop.
        delay: Samples between repeats, at least 1.
        feedback: Share of each repeat fed into the next, held to ``-0.99`` to ``0.99``.
        tail: Samples the repeats run past the end of ``source``.
        shape: Called as ``shape(block, start)`` on everything entering the line, a block of
            ``(channels, n)`` and the sample it starts at; answers the block to write. Filters
            and saturation in the loop go here, keeping their own state between calls.

    Returns:
        ``(channels, samples + tail)`` of repeats.
    """
    channels, length = source.shape
    gap = max(1, int(delay))
    keep = min(0.99, max(-0.99, float(feedback)))
    total = length + max(0, int(tail))
    fed = np.zeros((channels, total))
    fed[:, :length] = source
    wet = np.zeros((channels, total + gap))
    for first in range(0, total, gap):
        last = min(total, first + gap)
        block = fed[:, first:last] + keep * wet[:, first:last]
        if shape is not None:
            block = shape(block, first)
        wet[:, first + gap:last + gap] = block
    return wet[:, :total]
