"""Pitch shifting at a fixed length, with the source's spectral envelope put back to keep formants.

Waveforms are ``(batch, channels, samples)`` float tensors; shifts are semitones.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import torch

from .clip import make

__all__ = ["MAX_SHIFTED", "formants_kept", "shifted"]

#: Most samples, across channels and the stretch, one shift reads.
MAX_SHIFTED = 1 << 26

#: Analysis window and hop of the formant correction.
FFT = 2048
HOP = 512

#: Largest correction the envelope may apply, in nepers (about 24 dB).
MAX_CORRECTION = 2.76


def _envelope(log_magnitude: torch.Tensor, keep: int) -> torch.Tensor:
    """The smooth spectral envelope under a log magnitude spectrum, by cepstral liftering."""
    cepstrum = torch.fft.irfft(log_magnitude, n=FFT, dim=-2)
    lifter = torch.zeros(FFT, 1, dtype=cepstrum.dtype)
    lifter[:keep] = 1.0
    lifter[FFT - keep + 1:] = 1.0
    return torch.fft.rfft(cepstrum * lifter, n=FFT, dim=-2).real


def formants_kept(reference: torch.Tensor, shifted_wave: torch.Tensor, rate: int) -> torch.Tensor:
    """A shifted signal with the reference's spectral envelope laid back over it.

    Args:
        reference: ``(channels, samples)``, the unshifted source.
        shifted_wave: ``(channels, samples)``, the same length, shifted.
        rate: Samples per second.

    Returns:
        ``(channels, samples)``.
    """
    window = torch.hann_window(FFT, dtype=torch.float32)
    keep = max(8, int(round(rate / 1500.0)))
    source = torch.stft(reference.float(), FFT, HOP, window=window, return_complex=True)
    moved = torch.stft(shifted_wave.float(), FFT, HOP, window=window, return_complex=True)
    gap = _envelope(torch.log(source.abs() + 1e-7), keep) - _envelope(torch.log(moved.abs() + 1e-7), keep)
    corrected = moved * torch.exp(gap.clamp(-MAX_CORRECTION, MAX_CORRECTION))
    return torch.istft(corrected, FFT, HOP, window=window, length=int(shifted_wave.shape[-1]))


def shifted(wave: torch.Tensor, rate: int, semitones: float, keep_formants: bool) -> torch.Tensor:
    """A waveform moved in pitch with its length kept.

    Args:
        wave: ``(batch, channels, samples)``.
        rate: Samples per second.
        semitones: Shift; 12 is an octave up.
        keep_formants: Lay the source's spectral envelope back over the result.

    Returns:
        A float32 tensor of the same shape.

    Raises:
        ValueError: The shift reads more than :data:`MAX_SHIFTED` samples.
    """
    from scipy.signal import resample_poly

    from ..media.audio_timeline import retimed

    factor = 2.0 ** (float(semitones) / 12.0)
    batch, channels, length = (int(size) for size in wave.shape)
    if abs(factor - 1.0) < 1e-9:
        return wave.clone()
    held = batch * channels * int(math.ceil(length * max(1.0, factor)))
    if held > MAX_SHIFTED:
        raise ValueError(
            f"this shift reads {held:,} samples across the clip's channels, past the {MAX_SHIFTED:,} it "
            f"shifts in one pass, about 11 minutes of 48 kHz stereo. Split the clip with Audio Split first."
        )
    stretched = retimed(make(wave, rate), length * factor / rate, lambda at: at / factor, 1.0)["waveform"]
    ratio = Fraction(factor).limit_denominator(1000)
    back = resample_poly(stretched.double().numpy(), ratio.denominator, ratio.numerator, axis=-1)
    out = np.zeros((batch, channels, length))
    count = min(length, back.shape[-1])
    out[..., :count] = back[..., :count]
    result = torch.from_numpy(out.astype(np.float32))
    if keep_formants:
        for index in range(batch):
            result[index] = formants_kept(wave[index], result[index], rate)
    return result
