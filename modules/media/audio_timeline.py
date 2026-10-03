"""Audio rebuilt on a new timeline with its pitch kept, by phase-locked vocoding.

Positions on both timelines count in frames of the clip the audio belongs to.
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = ["AUDIO_FFT", "AUDIO_HOP", "retimed"]

#: Short-time Fourier size and hop the audio is rebuilt with.
AUDIO_FFT = 2048
AUDIO_HOP = 512


def retimed(audio: dict, frames: float, read_at, fps: float) -> dict:
    """Audio rebuilt on a new timeline, each output moment read from a mapped input moment.

    Args:
        audio: ComfyUI audio.
        frames: Length of the new timeline in frames.
        read_at: Maps positions on the new timeline, in frames, to input positions in frames.
        fps: Frame rate the positions count in.

    Returns:
        ComfyUI audio ``frames`` long.
    """
    waveform = audio["waveform"].float()
    rate = int(audio["sample_rate"])
    batch, channels, length = waveform.shape
    flat = waveform.reshape(batch * channels, length)
    window = torch.hann_window(AUDIO_FFT, device=flat.device)
    spectrum = torch.stft(flat, AUDIO_FFT, AUDIO_HOP, window=window, center=True,
                          return_complex=True)
    columns = spectrum.shape[-1]

    out_samples = int(round(frames / float(fps) * rate))
    steps = out_samples // AUDIO_HOP + 1
    position = np.arange(steps) * AUDIO_HOP / rate * float(fps)
    column = np.clip(read_at(position) / float(fps) * rate / AUDIO_HOP, 0.0, columns - 1.001)

    index = torch.as_tensor(np.floor(column), dtype=torch.long, device=flat.device)
    frac = torch.as_tensor(column - np.floor(column), dtype=torch.float32, device=flat.device)
    later = torch.clamp(index + 1, max=columns - 1)
    size = spectrum.abs()
    magnitude = size[..., index] * (1 - frac) + size[..., later] * frac

    angle = spectrum.angle()
    bins = torch.arange(spectrum.shape[1], device=flat.device, dtype=torch.float32)
    expected = (2 * math.pi * AUDIO_HOP / AUDIO_FFT * bins)[None, :, None]
    drift = angle[..., later] - angle[..., index] - expected
    drift = drift - 2 * math.pi * torch.round(drift / (2 * math.pi))
    advance = expected + drift
    phase = angle[..., index[:1]] + torch.cumsum(
        torch.cat([torch.zeros_like(advance[..., :1]), advance[..., :-1]], dim=-1), dim=-1
    )
    rebuilt = torch.polar(magnitude, _locked(phase, angle[..., index], magnitude))
    stretched = torch.istft(rebuilt, AUDIO_FFT, AUDIO_HOP, window=window, center=True,
                            length=out_samples)
    return {"waveform": stretched.reshape(batch, channels, -1), "sample_rate": rate}


def _locked(phase: torch.Tensor, source: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Phases with every bin held at its source offset from the nearest spectral peak.

    Args:
        phase: Accumulated phase, ``[rows, bins, columns]``.
        source: Phase of the source column each output column reads.
        magnitude: Output magnitude.

    Returns:
        The locked phases.
    """
    bins = magnitude.shape[1]
    lower = torch.cat([magnitude[:, :1] - 1, magnitude[:, :-1]], dim=1)
    upper = torch.cat([magnitude[:, 1:], magnitude[:, -1:] - 1], dim=1)
    peaks = (magnitude >= lower) & (magnitude > upper)
    ladder = torch.arange(bins, device=magnitude.device).view(1, -1, 1).expand_as(magnitude)
    below = torch.cummax(torch.where(peaks, ladder, torch.full_like(ladder, -1)), dim=1).values
    flipped = torch.where(peaks, ladder, torch.full_like(ladder, bins)).flip(1)
    above = (-torch.cummax(-flipped, dim=1).values).flip(1)
    has_below, has_above = below >= 0, above < bins
    nearer_above = has_above & (~has_below | (above - ladder < ladder - below))
    peak = torch.where(nearer_above, above, torch.where(has_below, below, ladder))
    return (torch.gather(phase, 1, peak) + source - torch.gather(source, 1, peak))
