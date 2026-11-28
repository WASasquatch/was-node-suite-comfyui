"""Scopes measured along a clip, frame by frame: curves of every frame and the sound's spectrogram.

Frames are ``(T, H, W, C)``, float codes in ``[0, 1]`` or uint8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import scopes

#: Longest side a frame is measured at.
MEASURE_SIDE = 512

#: Frames read at once.
CHUNK = 16

#: Rows of the spectrogram, low frequencies at the bottom.
SPECTRUM_BINS = 128

#: Lowest frequency the spectrogram shows, in hertz.
SPECTRUM_LOW = 30.0

#: Spectrogram columns per second, and the most it holds.
SPECTRUM_RATE, SPECTRUM_MAX_COLUMNS = 60, 8192

#: Loudness the spectrogram's darkest level stands for, in decibels below the loudest.
SPECTRUM_FLOOR_DB = 80.0

#: A channel at or over this code counts as clipped white, at or under the next as crushed black.
CLIP_HIGH, CLIP_LOW = 254.5 / 255.0, 0.5 / 255.0

#: Per-frame figures, in the order the curves are drawn.
CURVES = ("luma", "highlights", "shadows", "saturation", "red", "green", "blue")

#: Side of the vectorscope the targets are placed on, as shares of it.
TARGET_SIDE = 1000


@dataclass
class Measured:
    """A clip's measurements.

    Attributes:
        frames: Frames in the clip.
        rate: Frames per second.
        size: ``(width, height)`` of the source.
        curves: Each name of :data:`CURVES` and ``bands`` to one value per frame; ``bands`` holds
            one share per false-colour band.
        spectrum: ``(SPECTRUM_BINS, columns)`` uint8 spectrogram, or None without sound.
        spectrum_seconds: Seconds the spectrogram spans.
    """

    frames: int
    rate: float
    size: tuple
    curves: dict = field(default_factory=dict)
    spectrum: np.ndarray | None = None
    spectrum_seconds: float = 0.0


def codes(batch: torch.Tensor, device, side: int | None = MEASURE_SIDE) -> torch.Tensor:
    """Frames as float codes on ``device``, shrunk so the longest side is at most ``side``."""
    batch = batch[..., :3].to(device)
    batch = batch.float() / 255.0 if batch.dtype == torch.uint8 else batch.float()
    height, width = int(batch.shape[1]), int(batch.shape[2])
    if side and max(height, width) > side:
        scale = side / max(height, width)
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        batch = F.interpolate(batch.movedim(-1, 1), size=size, mode="area").movedim(1, -1)
    return batch.clamp(0.0, 1.0)


def luma_of(frame: torch.Tensor) -> torch.Tensor:
    """BT.709 Y' of ``(..., 3)`` codes."""
    return scopes.LUMA[0] * frame[..., 0] + scopes.LUMA[1] * frame[..., 1] + scopes.LUMA[2] * frame[..., 2]


def figures(frame: torch.Tensor) -> dict:
    """One frame's curve values and false-colour band shares."""
    luma = luma_of(frame)
    high = frame.amax(-1)
    low = frame.amin(-1)
    saturation = torch.where(high > 1e-6, (high - low) / high.clamp_min(1e-6), torch.zeros_like(high))
    _picture, band = scopes.false_colour(frame)
    bands = torch.bincount((band.reshape(-1) + 1), minlength=len(scopes.FALSE_COLOUR_BANDS) + 1)[1:]
    total = float(luma.numel())
    return {
        "luma": float(luma.mean()),
        "highlights": float((frame >= CLIP_HIGH).any(-1).float().mean()),
        "shadows": float((frame <= CLIP_LOW).all(-1).float().mean()),
        "saturation": float(saturation.mean()),
        "red": float(frame[..., 0].mean()),
        "green": float(frame[..., 1].mean()),
        "blue": float(frame[..., 2].mean()),
        "bands": [float(count) / total for count in bands.tolist()],
    }


def spectrogram(audio) -> tuple[np.ndarray | None, float]:
    """A clip's sound as a spectrogram on a log-frequency axis.

    Args:
        audio: An ``AUDIO`` dict, or None.

    Returns:
        ``(picture, seconds)``: ``(SPECTRUM_BINS, columns)`` uint8 with low frequencies at the
        bottom and the loudest cell at 255, and the seconds it spans; ``(None, 0.0)`` without sound.
    """
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None, 0.0
    wave = audio["waveform"]
    rate = int(audio.get("sample_rate", 44100))
    signal = wave[0].float().mean(0) if wave.dim() == 3 else wave.float().reshape(-1)
    seconds = float(signal.shape[-1]) / rate
    if seconds <= 0.05:
        return None, 0.0
    columns = int(min(SPECTRUM_MAX_COLUMNS, max(16, round(seconds * SPECTRUM_RATE))))
    hop = max(1, signal.shape[-1] // columns)
    size = 2048 if rate >= 32000 else 1024
    window = torch.hann_window(size)
    power = torch.stft(signal, size, hop, window=window, center=True, return_complex=True).abs().pow(2)
    frequencies = torch.linspace(0.0, rate / 2.0, power.shape[0])
    edges = torch.logspace(math.log10(SPECTRUM_LOW), math.log10(rate / 2.0), SPECTRUM_BINS + 1)
    which = torch.bucketize(frequencies, edges) - 1
    rows = torch.zeros(SPECTRUM_BINS, power.shape[1])
    counts = torch.zeros(SPECTRUM_BINS)
    inside = (which >= 0) & (which < SPECTRUM_BINS)
    rows.index_add_(0, which[inside], power[inside])
    counts.index_add_(0, which[inside], torch.ones(int(inside.sum())))
    empty = counts == 0
    if empty.any():
        nearest = torch.bucketize((edges[:-1] + edges[1:]) / 2.0, frequencies).clamp(max=power.shape[0] - 1)
        rows[empty] = power[nearest[empty]]
        counts[empty] = 1.0
    rows = rows / counts[:, None]
    decibels = 10.0 * torch.log10(rows + 1e-12)
    level = ((decibels - (decibels.max() - SPECTRUM_FLOOR_DB)) / SPECTRUM_FLOOR_DB).clamp(0.0, 1.0)
    level = F.interpolate(level[None, None], size=(SPECTRUM_BINS, columns), mode="bilinear", align_corners=False)[0, 0]
    picture = (level.flip(0) * 255.0).round().to(torch.uint8).numpy()
    return picture, seconds


def measure(read, frames: int, rate: float, audio=None, device=None, progress=None) -> Measured:
    """Measure every frame of a clip.

    Args:
        read: Called with ``(start, stop)``, answers those frames ``(T, H, W, C)``.
        frames: Frames in the clip.
        rate: Frames per second.
        audio: The clip's ``AUDIO``, or None.
        device: Where frames are measured; ComfyUI's compute device when None.
        progress: Called with ``(done, total)`` after each chunk.

    Returns:
        The :class:`Measured` clip.
    """
    if device is None:
        import comfy.model_management as mm

        device = mm.get_torch_device()
    curves = {name: [] for name in CURVES}
    curves["bands"] = []
    size = (0, 0)
    with torch.no_grad():
        for start in range(0, frames, CHUNK):
            stop = min(frames, start + CHUNK)
            chunk = read(start, stop)
            size = (int(chunk.shape[2]), int(chunk.shape[1]))
            batch = codes(chunk, device)
            for offset in range(int(batch.shape[0])):
                for name, value in figures(batch[offset]).items():
                    curves[name].append(value)
            if progress is not None:
                progress(stop, frames)
    spectrum, seconds = spectrogram(audio)
    return Measured(frames, float(rate), size, curves, spectrum, seconds)


def summary(measured: Measured) -> dict:
    """Whole-clip figures: means, worst frames and the share of frames that clip."""
    curves = measured.curves
    frames = max(1, measured.frames)
    lumas, highs, lows = curves.get("luma") or [0.0], curves.get("highlights") or [0.0], curves.get("shadows") or [0.0]
    worst_high, worst_low = int(np.argmax(highs)), int(np.argmax(lows))
    clipping = sum(1 for value in highs if value > 0.001)
    return {
        "frames": measured.frames,
        "seconds": round(measured.frames / max(measured.rate, 1e-6), 3),
        "luma_mean": float(np.mean(lumas)),
        "luma_range": [float(np.min(lumas)), float(np.max(lumas))],
        "saturation_mean": float(np.mean(curves.get("saturation") or [0.0])),
        "frames_clipping_highlights": int(clipping),
        "frames_crushing_shadows": int(sum(1 for value in lows if value > 0.001)),
        "worst_highlights": {"frame": worst_high, "share": float(highs[worst_high])},
        "worst_shadows": {"frame": worst_low, "share": float(lows[worst_low])},
        "share_of_frames_clipping": round(clipping / frames, 4),
    }


def data_document(measured: Measured) -> dict:
    """What the panel reads besides the clip and the spectrogram: sizes, curves, targets and the summary."""
    return {
        "version": 2,
        "frames": measured.frames,
        "rate": measured.rate,
        "size": list(measured.size),
        "bands": [[name, low, high, list(colour)] for name, low, high, colour in scopes.FALSE_COLOUR_BANDS],
        "targets": [
            [label, level, *(round(float(value) / TARGET_SIDE, 4) for value in scopes.plot_position(cb, cr, TARGET_SIDE))]
            for label, level, cb, cr in scopes.target_points()
        ],
        "chroma_scale": round(scopes.CIRCLE_SHARE / scopes.FULL_SCALE / 2.0, 6),
        "curves": {name: [round(value, 5) for value in values] for name, values in measured.curves.items() if name != "bands"},
        "band_shares": [[round(value, 5) for value in row] for row in measured.curves.get("bands", [])],
        "spectrum": None if measured.spectrum is None else {
            "bins": SPECTRUM_BINS, "columns": int(measured.spectrum.shape[1]),
            "seconds": round(measured.spectrum_seconds, 4), "low": SPECTRUM_LOW, "floor_db": SPECTRUM_FLOOR_DB,
        },
        "summary": summary(measured),
    }
