"""Sample rate conversion, and reading a waveform at fractional positions.

Positions are in input samples. Band-limiting follows the read speed, so a faster read does
not alias.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["KERNEL_TAPS", "MAX_SPEED", "read_at", "resampled"]

#: Taps either side of a read position at normal speed.
KERNEL_TAPS = 24

#: Fastest read speed band-limited exactly; a faster one is filtered as if read at this speed.
MAX_SPEED = 8.0

#: Kaiser window shape of the reading kernel.
KAISER_BETA = 8.6

#: Largest up or down factor converted with a polyphase filter.
POLYPHASE_LIMIT = 1024

#: Output samples read per pass of :func:`read_at`.
CHUNK = 1 << 15


def resampled(wave: torch.Tensor, rate: int, target: int) -> torch.Tensor:
    """A waveform converted to another sample rate.

    Args:
        wave: A ``(batch, channels, samples)`` float tensor.
        rate: Its samples per second.
        target: The samples per second wanted.

    Returns:
        A float32 CPU tensor at ``target``, ``round(samples * target / rate)`` long.
    """
    rate, target = int(rate), int(target)
    if rate == target:
        return wave
    length = int(wave.shape[-1])
    out_length = max(1, int(round(length * target / rate)))
    common = math.gcd(rate, target)
    up, down = target // common, rate // common
    if max(up, down) <= POLYPHASE_LIMIT:
        from scipy.signal import resample_poly

        data = wave.detach().cpu().numpy().astype(np.float64, copy=False)
        out = resample_poly(data, up, down, axis=-1, window=("kaiser", 5.0))
        out = out[..., :out_length]
        if out.shape[-1] < out_length:
            out = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(0, out_length - out.shape[-1])])
        return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
    positions = np.arange(out_length, dtype=np.float64) * (rate / target)
    return read_at(wave, positions, speed=rate / target)


def read_at(wave: torch.Tensor, positions, speed=1.0) -> torch.Tensor:
    """A waveform read at fractional sample positions through a windowed sinc.

    Args:
        wave: A ``(batch, channels, samples)`` float tensor.
        positions: Input positions to read, in samples, one per output sample. Positions
            outside the waveform read silence.
        speed: How far apart neighbouring reads are, as one number or one per output sample.
            Above 1 the kernel is widened to filter what would alias.

    Returns:
        A float32 ``(batch, channels, len(positions))`` tensor.
    """
    wave = wave.detach().to(device="cpu", dtype=torch.float32)
    batch, channels, length = (int(size) for size in wave.shape)
    where = np.asarray(positions, dtype=np.float64).reshape(-1)
    count = int(where.shape[0])
    pace = np.broadcast_to(np.abs(np.asarray(speed, dtype=np.float64)), (count,))
    stretch = np.clip(pace, 1.0, MAX_SPEED)
    flat = wave.reshape(batch * channels, length)
    out = torch.zeros(batch * channels, count, dtype=torch.float32)
    if count == 0:
        return out.reshape(batch, channels, 0)
    reach = int(math.ceil(KERNEL_TAPS * float(stretch.max())))
    # A wider kernel reads fewer positions per pass, so one pass holds about the same memory.
    step = max(1024, (CHUNK * KERNEL_TAPS) // reach)
    offsets = torch.arange(-reach + 1, reach + 1, dtype=torch.float64)
    norm = torch.special.i0(torch.tensor(KAISER_BETA, dtype=torch.float64))
    for first in range(0, count, step):
        last = min(count, first + step)
        spot = torch.from_numpy(where[first:last])
        widen = torch.from_numpy(np.ascontiguousarray(stretch[first:last]))
        index = torch.floor(spot)[:, None] + offsets[None, :]
        distance = (spot[:, None] - index) / widen[:, None]
        ramp = torch.clamp(1.0 - (distance / KERNEL_TAPS) ** 2, min=0.0)
        shape = torch.special.i0(KAISER_BETA * torch.sqrt(ramp)) / norm
        shape = torch.where(distance.abs() < KERNEL_TAPS, shape, torch.zeros((), dtype=torch.float64))
        weight = torch.sinc(distance) * shape / widen[:, None]
        valid = (index >= 0) & (index < length)
        picks = index.clamp(0, length - 1).to(torch.long)
        weight = torch.where(valid, weight, torch.zeros((), dtype=torch.float64)).to(torch.float32)
        gathered = flat[:, picks.reshape(-1)].reshape(batch * channels, last - first, -1)
        out[:, first:last] = (gathered * weight[None]).sum(dim=-1)
    return out.reshape(batch, channels, count)
