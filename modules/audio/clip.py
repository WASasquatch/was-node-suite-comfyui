"""The ``AUDIO`` mapping read, checked and built, and the ceilings every audio node holds to.

A waveform is a float32 CPU tensor shaped ``(batch, channels, samples)``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "MAX_CHANNELS",
    "MAX_RATE",
    "MAX_SAMPLES",
    "MAX_SECONDS",
    "MIN_RATE",
    "check_size",
    "conformed",
    "duration",
    "make",
    "matched",
    "rate_of",
    "samples_at",
    "waveform_of",
    "with_channels",
]

#: Lowest and highest sample rate a clip may carry.
MIN_RATE = 1000
MAX_RATE = 384000

#: Most channels a clip may carry.
MAX_CHANNELS = 32

#: Most samples one waveform an audio node builds may hold, across batch and channels.
MAX_SAMPLES = 1 << 29

#: Longest duration any audio widget accepts, in seconds.
MAX_SECONDS = 10800.0


def rate_of(audio, name: str = "audio") -> int:
    """The sample rate an ``AUDIO`` carries.

    Args:
        audio: An ``AUDIO`` mapping.
        name: The input it arrived on, for the error.

    Returns:
        Samples per second.

    Raises:
        ValueError: The mapping names no usable rate.
    """
    try:
        rate = int((audio or {}).get("sample_rate", 0))
    except (TypeError, ValueError, OverflowError, AttributeError):
        rate = 0
    if not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(
            f"{name} carries no usable sample rate (read {rate!r}). Audio from Load Audio, "
            f"Load Video or any node with an AUDIO output carries one between {MIN_RATE} and "
            f"{MAX_RATE} Hz."
        )
    return rate


def waveform_of(audio, name: str = "audio") -> torch.Tensor:
    """The waveform an ``AUDIO`` carries, as float32 on the CPU.

    Args:
        audio: An ``AUDIO`` mapping.
        name: The input it arrived on, for the error.

    Returns:
        A contiguous ``(batch, channels, samples)`` tensor. A missing batch or channel axis is
        added.

    Raises:
        ValueError: There is no waveform, it is empty, or it holds values that are not finite.
    """
    wave = (audio or {}).get("waveform") if isinstance(audio, dict) else None
    if not isinstance(wave, torch.Tensor) or wave.is_complex():
        raise ValueError(
            f"{name} is not audio: it carries no waveform. Wire a node with an AUDIO output "
            f"into it."
        )
    if wave.ndim == 1:
        wave = wave[None, None]
    elif wave.ndim == 2:
        wave = wave[None]
    if wave.ndim != 3:
        raise ValueError(
            f"{name} has a waveform shaped {tuple(wave.shape)}; audio is shaped (batch, "
            f"channels, samples)."
        )
    batch, channels, samples = (int(size) for size in wave.shape)
    if batch < 1 or channels < 1 or samples < 1:
        raise ValueError(f"{name} holds no samples ({batch} x {channels} x {samples}).")
    if channels > MAX_CHANNELS:
        raise ValueError(
            f"{name} has {channels} channels; at most {MAX_CHANNELS} are read. A waveform "
            f"shaped (samples, channels) needs its axes swapped upstream."
        )
    check_size(batch, channels, samples, name)
    wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(wave).all()):
        raise ValueError(
            f"{name} holds samples that are not finite numbers (NaN or infinity), which a "
            f"decoder or an upstream node produced. Fix the source before editing it."
        )
    return wave


def check_size(batch: int, channels: int, samples: int, name: str) -> None:
    """Refuse a waveform past :data:`MAX_SAMPLES` before it is allocated.

    Args:
        batch: Clips in the batch.
        channels: Channels per clip.
        samples: Samples per channel.
        name: The input to lower, named in the error.

    Raises:
        ValueError: The waveform would hold more than :data:`MAX_SAMPLES` samples.
    """
    total = int(batch) * int(channels) * int(samples)
    if total > MAX_SAMPLES or samples < 0:
        raise ValueError(
            f"this would hold {total:,} samples ({batch} clip(s) x {channels} channel(s) x "
            f"{samples:,}), more than the {MAX_SAMPLES:,} one audio node builds. Lower "
            f"{name}, or split the audio first."
        )


def samples_at(seconds: float, rate: int) -> int:
    """A time as a whole sample count.

    Args:
        seconds: The time, from 0.
        rate: Samples per second.

    Returns:
        The nearest sample index, never below 0.
    """
    value = float(seconds)
    if not math.isfinite(value) or value <= 0.0:
        return 0
    return int(round(value * int(rate)))


def duration(wave: torch.Tensor, rate: int) -> float:
    """How long a waveform plays.

    Args:
        wave: A ``(batch, channels, samples)`` tensor.
        rate: Samples per second.

    Returns:
        Seconds.
    """
    return int(wave.shape[-1]) / float(rate)


def make(wave: torch.Tensor, rate: int) -> dict:
    """An ``AUDIO`` mapping.

    Args:
        wave: A ``(batch, channels, samples)`` tensor.
        rate: Samples per second.

    Returns:
        ``{"waveform": wave, "sample_rate": rate}``, the waveform float32 and contiguous.
    """
    return {"waveform": wave.to(torch.float32).contiguous(), "sample_rate": int(rate)}


def with_channels(wave: torch.Tensor, channels: int) -> torch.Tensor:
    """A waveform carrying a set number of channels.

    Args:
        wave: A ``(batch, channels, samples)`` tensor.
        channels: Channels wanted.

    Returns:
        The waveform as it is when the count already matches. Mono is copied to every channel;
        anything else going to mono is averaged; otherwise channels are dropped from the end, or
        the last one repeated.
    """
    have = int(wave.shape[1])
    channels = int(channels)
    if have == channels:
        return wave
    if have == 1:
        return wave.expand(-1, channels, -1).contiguous()
    if channels == 1:
        return wave.mean(dim=1, keepdim=True)
    if have > channels:
        return wave[:, :channels].contiguous()
    extra = wave[:, -1:].expand(-1, channels - have, -1)
    return torch.cat([wave, extra], dim=1)


def conformed(wave: torch.Tensor, rate: int, target_rate: int, channels: int) -> torch.Tensor:
    """A waveform brought to a rate and a channel count.

    Args:
        wave: A ``(batch, channels, samples)`` tensor.
        rate: Its samples per second.
        target_rate: The rate wanted.
        channels: The channel count wanted.

    Returns:
        The waveform at ``target_rate`` with ``channels`` channels.
    """
    from .resample import resampled

    if int(rate) != int(target_rate):
        wave = resampled(wave, int(rate), int(target_rate))
    return with_channels(wave, channels)


def matched(clips, names) -> tuple[list[torch.Tensor], int, int]:
    """Several ``AUDIO`` clips at one rate and one channel count.

    Args:
        clips: ``AUDIO`` mappings in order.
        names: The input each arrived on, for errors.

    Returns:
        ``(waveforms, rate, channels)``: every clip at the first clip's rate, carrying as many
        channels as the widest clip.
    """
    waves = [waveform_of(clip, name) for clip, name in zip(clips, names)]
    rates = [rate_of(clip, name) for clip, name in zip(clips, names)]
    rate = rates[0]
    channels = max(int(wave.shape[1]) for wave in waves)
    return [conformed(w, r, rate, channels) for w, r in zip(waves, rates)], rate, channels
