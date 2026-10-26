"""A clip's spectrum over time, drawn as a picture.

Rows run from high frequencies at the top to low at the bottom. Levels are dB below a
full-scale sine. Viridis, magma and inferno are the CC0 maps from matplotlib.
"""

from __future__ import annotations

import math

import torch

__all__ = ["COLOURS", "FFT_SIZES", "SCALES", "spectrogram"]

#: Frequency axes, in menu order.
SCALES = ("log", "mel", "linear")

#: Analysis window sizes, in menu order.
FFT_SIZES = ("2048", "1024", "4096", "512")

#: Seventeen evenly spaced stops of each colour map, as 0 to 255 RGB.
COLOURS = {
    "magma": [(0, 0, 4), (10, 8, 34), (29, 17, 71), (54, 16, 107), (81, 18, 124),
              (106, 28, 129), (131, 38, 129), (156, 46, 127), (183, 55, 121), (208, 65, 111),
              (231, 82, 99), (245, 107, 92), (252, 137, 97), (254, 167, 114), (254, 196, 136),
              (253, 226, 163), (252, 253, 191)],
    "viridis": [(68, 1, 84), (72, 24, 106), (71, 45, 123), (66, 64, 134), (59, 82, 139),
                (51, 99, 141), (44, 114, 142), (38, 130, 142), (33, 145, 140), (31, 160, 136),
                (40, 174, 128), (63, 188, 115), (94, 201, 98), (132, 212, 75), (173, 220, 48),
                (216, 226, 25), (253, 231, 37)],
    "inferno": [(0, 0, 4), (11, 7, 36), (33, 12, 74), (61, 9, 101), (87, 16, 110),
                (113, 25, 110), (138, 34, 106), (163, 44, 97), (188, 55, 84), (210, 70, 68),
                (228, 90, 49), (241, 115, 29), (249, 142, 9), (252, 172, 17), (249, 203, 53),
                (242, 234, 105), (252, 255, 164)],
    "greyscale": [(0, 0, 0), (255, 255, 255)],
}

#: Frames transformed at once.
FRAME_CHUNK = 2048

#: Lowest frequency the log and mel axes start from.
LOW_HZ = 20.0


def _mel(hz: torch.Tensor) -> torch.Tensor:
    """Hertz on the mel scale."""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _hz(mel: torch.Tensor) -> torch.Tensor:
    """Mels back to hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _bank(rows: int, bins: int, rate: int, scale: str) -> torch.Tensor:
    """Triangular weights taking FFT bins to picture rows, the lowest row first."""
    nyquist = rate / 2.0
    if scale == "linear":
        edges = torch.linspace(0.0, nyquist, rows + 2, dtype=torch.float64)
    elif scale == "mel":
        edges = _hz(torch.linspace(float(_mel(torch.tensor(LOW_HZ))), float(_mel(torch.tensor(nyquist))),
                                   rows + 2, dtype=torch.float64))
    else:
        edges = torch.logspace(math.log10(LOW_HZ), math.log10(nyquist), rows + 2, dtype=torch.float64)
    centres = torch.linspace(0.0, nyquist, bins, dtype=torch.float64)
    lower, middle, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (centres[None] - lower) / torch.clamp(middle - lower, min=1e-9)
    fall = (upper - centres[None]) / torch.clamp(upper - middle, min=1e-9)
    weights = torch.clamp(torch.minimum(rise, fall), min=0.0)
    # A row narrower than one bin reads the bin nearest its centre.
    nearest = torch.round(middle[:, 0] / nyquist * (bins - 1)).long().clamp(0, bins - 1)
    empty = weights.sum(dim=1) <= 0
    weights[empty, nearest[empty]] = 1.0
    return weights / weights.sum(dim=1, keepdim=True)


def _coloured(levels: torch.Tensor, colours: str) -> torch.Tensor:
    """Levels 0 to 1 painted through a colour map, as ``(rows, columns, 3)``."""
    stops = torch.tensor(COLOURS.get(colours, COLOURS["magma"]), dtype=torch.float32) / 255.0
    position = levels.clamp(0.0, 1.0) * (stops.shape[0] - 1)
    low = position.floor().long().clamp(0, stops.shape[0] - 2)
    share = (position - low.float())[..., None]
    return stops[low] * (1.0 - share) + stops[low + 1] * share


def spectrogram(wave: torch.Tensor, rate: int, width: int, height: int, fft: int, scale: str,
                floor_db: float, colours: str) -> torch.Tensor:
    """A picture of one clip's spectrum over time.

    Args:
        wave: ``(channels, samples)``; channels are averaged.
        rate: Samples per second.
        width: Picture width; columns cover equal spans of time.
        height: Picture height; rows follow ``scale``.
        fft: Window size in samples; the hop is a quarter of it.
        scale: A name from :data:`SCALES`.
        floor_db: Level drawn darkest, in dB below a full-scale sine, such as ``-90``.
        colours: A name from :data:`COLOURS`.

    Returns:
        A float32 ``(height, width, 3)`` picture in ``[0, 1]``.
    """
    mono = wave.to(torch.float32).mean(dim=0)
    hop = fft // 4
    window = torch.hann_window(fft, dtype=torch.float32)
    reference = (float(window.sum()) / 2.0) ** 2
    bank = _bank(height, fft // 2 + 1, rate, scale).to(torch.float32)
    length = int(mono.shape[0])
    frames = max(1, 1 + length // hop)
    padded = torch.nn.functional.pad(mono, (fft // 2, fft // 2 + hop))
    columns = torch.full((height, width), 0.0, dtype=torch.float32)
    counts = torch.zeros(width, dtype=torch.float32)
    keep = torch.zeros((height, frames), dtype=torch.float32) if frames < width else None
    for first in range(0, frames, FRAME_CHUNK):
        last = min(frames, first + FRAME_CHUNK)
        starts = torch.arange(first, last) * hop
        index = starts[:, None] + torch.arange(fft)[None, :]
        segment = padded[index.clamp(max=padded.shape[0] - 1)] * window
        power = torch.fft.rfft(segment, dim=-1).abs().square() / reference
        rows = bank @ power.T
        if keep is not None:
            keep[:, first:last] = rows
            continue
        target = (torch.arange(first, last) * width // frames).clamp(max=width - 1)
        columns.index_add_(1, target, rows)
        counts.index_add_(0, target, torch.ones(last - first))
    if keep is not None:
        power = torch.nn.functional.interpolate(keep[None, None], size=(height, width), mode="bilinear",
                                                align_corners=False)[0, 0]
    else:
        power = columns / counts.clamp(min=1.0)[None, :]
    level = 10.0 * torch.log10(power.clamp(min=1e-20))
    shade = (level - float(floor_db)) / max(1e-6, -float(floor_db))
    return _coloured(shade.flip(0), colours)
