"""Beat, downbeat, tempo and loudness analysis of an audio track, at a video frame rate.

Onsets are log mel spectral flux; beats are tracked by dynamic programming. Times are seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

#: Sample rate the analysis runs at, and its hop and window in samples.
ANALYSIS_RATE = 22050
HOP = 512
WINDOW = 2048
MELS = 96

#: Tempo search range and the prior's centre and width in octaves.
MIN_BPM = 50.0
MAX_BPM = 220.0
PRIOR_BPM = 120.0
PRIOR_OCTAVES = 1.0

#: How strongly the tracker keeps beats at the tempo's period.
TIGHTNESS = 100.0

#: Mel bands at or below this many hertz count as bass for downbeats.
BASS_HZ = 200.0


@dataclass
class Beats:
    """A track's rhythm, at a video frame rate.

    Attributes:
        times: Beat times in seconds from the start.
        downbeats: The beats that open a bar.
        tempo: Beats per minute.
        duration: Length of the analysed audio in seconds.
        fps: Video frames per second the per-frame curves are sampled at.
        beats_per_bar: Beats in one bar.
        energy: Loudness per video frame, ``0`` to ``1``.
        onset: Onset strength per video frame, ``0`` to ``1``.
    """

    times: list[float]
    downbeats: list[float]
    tempo: float
    duration: float
    fps: float
    beats_per_bar: int
    energy: list[float] = field(default_factory=list)
    onset: list[float] = field(default_factory=list)

    @property
    def bar_seconds(self) -> float:
        """Length of one bar at the detected tempo."""
        return 60.0 / max(self.tempo, 1e-6) * self.beats_per_bar

    def renewals(self, which: str, every: int) -> list[float]:
        """Times the noise renews at.

        Args:
            which: ``beats`` or ``downbeats``.
            every: Keep every this many of them.

        Returns:
            Times in seconds, rising.
        """
        source = self.downbeats if which == "downbeats" else self.times
        return list(source[::max(1, int(every))])


def mono(audio: dict) -> tuple[np.ndarray, int]:
    """A ComfyUI audio's first item as a mono float array.

    Args:
        audio: ``{"waveform": [B, C, L], "sample_rate": int}``.

    Returns:
        ``(samples, rate)``.
    """
    waveform = audio["waveform"][0].float().mean(dim=0)
    return waveform.cpu().numpy(), int(audio["sample_rate"])


def trimmed(audio: dict, start: float, length: float) -> dict:
    """A ComfyUI audio cut to a span.

    Args:
        audio: ``{"waveform": [B, C, L], "sample_rate": int}``.
        start: Seconds to skip.
        length: Seconds to keep, ``0`` for the rest.

    Returns:
        The cut audio in the same format.
    """
    rate = int(audio["sample_rate"])
    first = max(0, int(round(float(start) * rate)))
    last = audio["waveform"].shape[-1] if float(length) <= 0 else first + int(round(length * rate))
    return {"waveform": audio["waveform"][..., first:last].contiguous(), "sample_rate": rate}


def _mel_frames(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Log mel spectrogram frames and each band's centre frequency."""
    import torchaudio

    wave = torch.from_numpy(samples).float()
    if rate != ANALYSIS_RATE:
        wave = torchaudio.functional.resample(wave, rate, ANALYSIS_RATE)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=ANALYSIS_RATE, n_fft=WINDOW, hop_length=HOP, n_mels=MELS, power=2.0)
    mel = transform(wave).clamp(min=1e-10)
    log_mel = 10.0 * torch.log10(mel)
    log_mel = torch.maximum(log_mel, log_mel.max() - 80.0)
    centres = torchaudio.functional.melscale_fbanks(
        WINDOW // 2 + 1, 0.0, ANALYSIS_RATE / 2, MELS, ANALYSIS_RATE)
    freqs = torch.linspace(0, ANALYSIS_RATE / 2, WINDOW // 2 + 1)
    band_hz = (centres * freqs[:, None]).sum(0) / centres.sum(0).clamp(min=1e-10)
    return log_mel.numpy(), band_hz.numpy()


def onset_envelope(log_mel: np.ndarray, bands: np.ndarray | None = None) -> np.ndarray:
    """Positive spectral flux summed over mel bands, one value per analysis frame.

    Args:
        log_mel: ``[mels, frames]`` in decibels.
        bands: Rows to sum over, every row when None.

    Returns:
        The envelope, mean-removed and clipped at zero.
    """
    rows = log_mel if bands is None else log_mel[bands]
    flux = np.maximum(0.0, np.diff(rows, axis=1, prepend=rows[:, :1])).mean(axis=0)
    flux = flux - np.convolve(flux, np.ones(16) / 16.0, mode="same")
    return np.maximum(flux, 0.0)


def estimate_tempo(envelope: np.ndarray, frame_rate: float) -> float:
    """Beats per minute from the envelope's autocorrelation under a log-normal prior.

    Args:
        envelope: From :func:`onset_envelope`.
        frame_rate: Envelope frames per second.

    Returns:
        The tempo, refined between lags.
    """
    centred = envelope - envelope.mean()
    size = 1 << int(math.ceil(math.log2(max(2, 2 * len(centred)))))
    spectrum = np.fft.rfft(centred, size)
    auto = np.fft.irfft(spectrum * np.conj(spectrum))[:len(centred)]
    lags = np.arange(len(auto), dtype=np.float64)
    low = int(math.floor(60.0 * frame_rate / MAX_BPM))
    high = min(len(auto) - 2, int(math.ceil(60.0 * frame_rate / MIN_BPM)))
    if high <= low + 1:
        return PRIOR_BPM
    bpm = 60.0 * frame_rate / np.maximum(lags[low:high], 1e-6)
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    weighted = np.maximum(auto[low:high], 0.0) * prior
    best = int(np.argmax(weighted)) + low
    if 0 < best < len(auto) - 1:
        a, b, c = auto[best - 1], auto[best], auto[best + 1]
        denominator = a - 2 * b + c
        shift = 0.5 * (a - c) / denominator if abs(denominator) > 1e-12 else 0.0
        best = best + float(np.clip(shift, -0.5, 0.5))
    return float(60.0 * frame_rate / best)


def track_beats(envelope: np.ndarray, frame_rate: float, tempo: float) -> np.ndarray:
    """Beat frames by dynamic programming over the envelope at a fixed tempo.

    Args:
        envelope: From :func:`onset_envelope`.
        frame_rate: Envelope frames per second.
        tempo: Beats per minute.

    Returns:
        Envelope frame indices of the beats, rising.
    """
    count = len(envelope)
    if count == 0 or envelope.max() <= 0:
        return np.array([], dtype=np.int64)
    period = 60.0 * frame_rate / tempo
    score = envelope / (envelope.std() + 1e-8)
    width = max(1, int(round(period / 32)))
    kernel = np.exp(-0.5 * (np.arange(-2 * width, 2 * width + 1) / width) ** 2)
    local = np.convolve(score, kernel, mode="same")
    cumulative = local.copy()
    backlink = np.full(count, -1, dtype=np.int64)
    back = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    cost = -TIGHTNESS * np.log(-back / period) ** 2
    for index in range(count):
        candidates = index + back
        valid = candidates >= 0
        if not valid.any():
            continue
        options = cumulative[candidates[valid]] + cost[valid]
        pick = int(np.argmax(options))
        cumulative[index] = local[index] + options[pick]
        backlink[index] = candidates[valid][pick]
    tail = cumulative[max(0, count - int(round(period))):]
    frame = int(np.argmax(tail)) + max(0, count - int(round(period)))
    beats = []
    while frame >= 0:
        beats.append(frame)
        frame = int(backlink[frame])
    beats = np.array(beats[::-1], dtype=np.int64)
    # Beats in stretches with no onset energy are trimmed from both ends.
    floor = 0.5 * np.median(local[beats]) if len(beats) else 0.0
    keep = np.nonzero(local[beats] >= floor)[0]
    return beats[keep[0]:keep[-1] + 1] if len(keep) else beats


def pick_downbeats(beat_frames: np.ndarray, bass: np.ndarray, beats_per_bar: int) -> np.ndarray:
    """The beats that open bars: the phase whose beats carry the most bass onset.

    Args:
        beat_frames: From :func:`track_beats`.
        bass: The bass band envelope.
        beats_per_bar: Beats in one bar.

    Returns:
        The downbeat frames.
    """
    if len(beat_frames) == 0:
        return beat_frames
    bar = max(1, int(beats_per_bar))
    strength = [bass[beat_frames[phase::bar]].mean() if len(beat_frames[phase::bar]) else 0.0
                for phase in range(bar)]
    return beat_frames[int(np.argmax(strength))::bar]


def per_frame(values: np.ndarray, source_rate: float, fps: float, frames: int,
              reduce=np.mean) -> list[float]:
    """A curve resampled to video frames and scaled so its 98th percentile is ``1``.

    Args:
        values: One value per source frame.
        source_rate: Source frames per second.
        fps: Video frames per second.
        frames: Video frames wanted.
        reduce: How the source frames inside one video frame combine.

    Returns:
        ``frames`` values clipped to ``[0, 1]``.
    """
    out = np.zeros(frames, dtype=np.float64)
    for frame in range(frames):
        first = int(frame / fps * source_rate)
        last = max(first + 1, int((frame + 1) / fps * source_rate))
        chunk = values[first:last]
        out[frame] = reduce(chunk) if len(chunk) else 0.0
    scale = np.percentile(out, 98) if frames else 1.0
    return np.clip(out / (scale + 1e-12), 0.0, 1.0).round(4).tolist()


def analyse(audio: dict, fps: float, beats_per_bar: int) -> Beats:
    """Beats, downbeats, tempo and per-frame curves of a track.

    Args:
        audio: A ComfyUI audio.
        fps: Video frames per second.
        beats_per_bar: Beats in one bar.

    Returns:
        The analysis.
    """
    samples, rate = mono(audio)
    duration = len(samples) / float(rate)
    log_mel, band_hz = _mel_frames(samples, rate)
    frame_rate = ANALYSIS_RATE / HOP
    envelope = onset_envelope(log_mel)
    bass = onset_envelope(log_mel, np.nonzero(band_hz <= BASS_HZ)[0])
    tempo = estimate_tempo(envelope, frame_rate)
    beat_frames = track_beats(envelope, frame_rate, tempo)
    down_frames = pick_downbeats(beat_frames, bass, beats_per_bar)
    power = np.mean(10.0 ** (log_mel / 10.0), axis=0)
    frames = int(math.floor(duration * fps))
    return Beats(
        times=(beat_frames / frame_rate).round(4).tolist(),
        downbeats=(down_frames / frame_rate).round(4).tolist(),
        tempo=round(tempo, 2),
        duration=round(duration, 4),
        fps=float(fps),
        beats_per_bar=int(beats_per_bar),
        energy=per_frame(np.sqrt(power), frame_rate, fps, frames),
        onset=per_frame(envelope, frame_rate, fps, frames, reduce=np.max),
    )


def plot(beats: Beats, width: int = 1280, height: int = 240) -> torch.Tensor:
    """A strip showing the energy curve, onsets, beats and downbeats over time.

    Args:
        beats: The analysis.
        width: Picture width.
        height: Picture height.

    Returns:
        An IMAGE tensor ``[1, height, width, 3]``.
    """
    from PIL import Image, ImageDraw

    picture = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(picture)
    span = max(beats.duration, 1e-6)
    base = height - 18

    def x_of(seconds: float) -> int:
        return int(seconds / span * (width - 1))

    count = len(beats.energy)
    for frame in range(count):
        x = int(frame / max(1, count) * (width - 1))
        top = base - int(beats.energy[frame] * (base - 10))
        draw.line([(x, base), (x, top)], fill=(52, 92, 140))
        spike = base - int(beats.onset[frame] * (base - 10))
        draw.point((x, spike), fill=(236, 190, 92))
    for seconds in beats.times:
        draw.line([(x_of(seconds), base - 26), (x_of(seconds), base)], fill=(210, 210, 220))
    for seconds in beats.downbeats:
        draw.line([(x_of(seconds), 6), (x_of(seconds), base)], fill=(232, 88, 88), width=2)
    step = 10 if span > 40 else 5
    for mark in range(0, int(span) + 1, step):
        draw.text((x_of(mark) + 2, base + 3), f"{mark}s", fill=(150, 150, 160))
    draw.text((8, 6), f"{beats.tempo:.1f} BPM, {len(beats.times)} beats", fill=(230, 230, 235))
    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]
