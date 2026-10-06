"""Holding a MiniMax H3 clip's fast frames for a refining pass, and removing them again.

A hold of ``n`` shows a frame ``n`` times. Latent rows cover ``1, 4, 4, 4, 4`` frames per 17.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..media import audio_timeline
from . import h3_extend

__all__ = [
    "AUDIO_MODES",
    "COVERAGE",
    "Plan",
    "aligned",
    "audio_row_strength",
    "decode_scenes",
    "encode_stretched",
    "fit",
    "latent_audio",
    "motion",
    "padding",
    "plan",
    "plot",
    "recover",
    "row_of_frame",
    "scene_frames",
    "scene_spans",
    "squeeze_audio",
    "stretch_audio",
    "stretch_frames",
    "stretched_starts",
    "video_of",
]

#: Coverage presets: ``(quantile, peak hold, bridge)``. Rows at or above the quantile of the
#: motion profile are held at the peak; gaps of up to ``bridge`` rows between them are filled.
COVERAGE = {
    "balanced": (0.75, 4, 8),
    "wide": (0.70, 4, 8),
    "economy": (0.85, 3, 8),
}

#: Audio presets: the share of each audio row the refining pass re-renders.
AUDIO_MODES = {
    "follow": 0.5,
    "loose": 0.7,
    "pin": 0.0,
    "fresh": 1.0,
}



@dataclass(frozen=True)
class Plan:
    """What H3 De-RoPE Stretch did to a clip, for H3 De-RoPE Recover to undo.

    Attributes:
        holds: Showings of each frame of the fitted clip.
        fps: Frame rate of the clip.
        audio: The source audio as wired, or ``None``.
        rows: Motion of each latent row, for the plot.
        row_holds: Hold of each latent row.
        source: Frames of the clip as wired.
        cuts: Latent rows each scene after the first opens on.
    """

    holds: tuple
    fps: float
    audio: dict | None
    rows: tuple
    row_holds: tuple
    source: int
    cuts: tuple = ()

    @property
    def frames(self) -> int:
        """Frames of the fitted clip."""
        return len(self.holds)

    @property
    def stretched(self) -> int:
        """Frames of the stretched clip."""
        return int(sum(self.holds))


def fit(images: torch.Tensor, audio: dict | None, fps: float):
    """A clip lengthened to the 17k+5 grid by repeating its last frame, its audio padded to match.

    Args:
        images: ``[F, H, W, C]`` frames.
        audio: ComfyUI audio, or ``None``.
        fps: Frame rate of the clip.

    Returns:
        ``(images, audio)``, both covering the fitted length.

    Raises:
        ValueError: The clip holds no frames.
    """
    count = int(images.shape[0])
    if count < 1:
        raise ValueError("H3 De-RoPE was given no frames")
    clips = max(0, math.ceil((count - h3_extend.CLIP_LEAD) / h3_extend.CLIP_FRAMES))
    fitted = h3_extend.CLIP_LEAD + clips * h3_extend.CLIP_FRAMES
    if fitted > count:
        images = torch.cat([images, images[-1:].expand(fitted - count, *images.shape[1:])])
    if audio is not None:
        rate = int(audio["sample_rate"])
        samples = int(round(fitted / float(fps) * rate))
        waveform = audio["waveform"][..., :samples]
        if waveform.shape[-1] < samples:
            waveform = torch.nn.functional.pad(waveform, (0, samples - waveform.shape[-1]))
        audio = {"waveform": waveform, "sample_rate": rate}
    return images, audio


def video_of(latent: dict) -> torch.Tensor:
    """The video half of an H3 latent, joint or video only.

    Args:
        latent: A LATENT whose ``samples`` is a joint pair or a ``[B, 24, T, H, W]`` tensor.

    Returns:
        The ``[B, 24, T, H, W]`` video latent.

    Raises:
        ValueError: The latent is not a MiniMax H3 video latent.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if getattr(samples, "tensors", None) is not None:
        return h3_extend.split(latent)[0]
    if isinstance(samples, torch.Tensor) and samples.ndim == 5 and samples.shape[1] == 24:
        return samples
    raise ValueError(
        "H3 De-RoPE Stretch needs a MiniMax H3 video latent, from an H3 sampler or VAE Encode "
        "with the H3 video VAE"
    )


def latent_audio(audio_vae, latent: dict) -> dict | None:
    """The audio half of a joint H3 latent, decoded.

    Args:
        audio_vae: The H3 audio VAE.
        latent: A LATENT.

    Returns:
        ComfyUI audio, or ``None`` when the latent carries no audio half.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    tensors = getattr(samples, "tensors", None)
    if tensors is None or len(tensors) < 2:
        return None
    waveform = audio_vae.decode(tensors[-1]).movedim(-1, 1)
    # Brought to the level VAE Decode Audio gives the same latent.
    scale = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    waveform = waveform / scale.clamp(min=1.0)
    rate = getattr(audio_vae, "audio_sample_rate_output",
                   getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": waveform, "sample_rate": int(rate)}


def scene_spans(starts, rows: int) -> list[tuple[int, int]]:
    """``(start, stop)`` latent rows of each scene of a clip.

    Args:
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        rows: Latent rows of the clip.

    Returns:
        Spans in order covering every row, one where ``starts`` names no cut inside the clip.
    """
    rows = int(rows)
    kept = sorted({int(start) for start in starts or ()
                   if 0 < int(start) < rows and int(start) % h3_extend.CLIP_TOKENS == 0})
    edges = [0, *kept, rows]
    return [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]


def scene_frames(starts, frames: int) -> list[tuple[int, int]]:
    """``(start, stop)`` frames of each scene of a clip on the 17k+5 grid.

    Args:
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        frames: Frames of the clip.

    Returns:
        Spans in order covering every frame.
    """
    spans = scene_spans(starts, h3_extend.tokens_for(int(frames)))
    edges = [start // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES for start, _ in spans]
    edges.append(int(frames))
    return [(edges[index], edges[index + 1]) for index in range(len(spans))]


def decode_scenes(vae, video: torch.Tensor, starts=None) -> torch.Tensor:
    """A clip's frames, each scene decoded on its own.

    Args:
        vae: The H3 video VAE.
        video: The clip's video latent, ``[B, 24, T, H, W]``.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        ``[F, H, W, C]`` frames.
    """
    spans = scene_spans(starts, video.shape[2])
    if len(spans) == 1:
        images = vae.decode(video)
        return images.reshape(-1, *images.shape[-3:])
    pieces = []
    for start, stop in spans:
        images = vae.decode(video[:, :, start:stop])
        pieces.append(images.reshape(-1, *images.shape[-3:]))
    return torch.cat(pieces)


def row_of_frame(frame: int) -> int:
    """The latent row a frame is encoded into.

    Args:
        frame: A frame index, from 0.

    Returns:
        A row index, from 0.
    """
    clip, offset = divmod(int(frame), h3_extend.CLIP_FRAMES)
    return clip * h3_extend.CLIP_TOKENS + (0 if offset == 0 else 1 + (offset - 1) // 4)


def _third(rows: torch.Tensor) -> np.ndarray:
    """Mean size of the third difference over time ending at each row of one scene.

    Args:
        rows: A float ``[B, 24, T, H, W]`` latent.

    Returns:
        ``T`` values, the first three taking the fourth's, or zeros under four rows.
    """
    count = rows.shape[2]
    if count < 4:
        return np.zeros(count)
    third = rows[:, :, 3:] - 3 * rows[:, :, 2:-1] + 3 * rows[:, :, 1:-2] - rows[:, :, :-3]
    values = third.abs().mean(dim=(0, 1, 3, 4)).cpu().numpy().astype(np.float64)
    return np.concatenate([np.full(3, values[0]), values])


def motion(video: torch.Tensor, starts=None) -> np.ndarray:
    """Motion per latent row, the mean size of its third difference over time inside its scene.

    Args:
        video: The clip's video latent, ``[B, 24, T, H, W]``.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        ``T`` values. The first three rows of each scene take its fourth row's value.
    """
    rows = video.float()
    spans = scene_spans(starts, rows.shape[2])
    if len(spans) == 1:
        return _third(rows)
    return np.concatenate([_third(rows[:, :, start:stop]) for start, stop in spans])


def _spread(rows: np.ndarray, peak: int, bridge: int, ramp: bool) -> np.ndarray:
    """Row holds with short gaps between held rows filled and each held span ramped down.

    Args:
        rows: Hold of each row of one scene, ``peak`` or ``1``.
        peak: Hold of the fastest rows.
        bridge: Longest gap in rows between held rows that is held too.
        ramp: Whether holds fall by one per row away from a held span.

    Returns:
        The holds.
    """
    hot = np.flatnonzero(rows == int(peak))
    for first, second in zip(hot[:-1], hot[1:]):
        if 1 < second - first <= int(bridge):
            rows[first:second + 1] = int(peak)
    if ramp:
        for _ in range(int(peak) - 1):
            left = np.concatenate([[1], rows[:-1]])
            right = np.concatenate([rows[1:], [1]])
            rows = np.maximum(rows, np.maximum(left, right) - 1)
    return rows


def plan(profile: np.ndarray, frames: int, quantile: float, peak: int, bridge: int,
         ramp: bool = True, starts=None):
    """The hold of every frame, from the motion of every row.

    Args:
        profile: Motion per latent row.
        frames: Frames of the clip.
        quantile: Share of rows below the hold threshold, ``0.5`` to ``0.99``.
        peak: Hold of the fastest rows, ``2`` to ``8``.
        bridge: Longest gap in rows between held rows that is held too.
        ramp: Whether holds fall by one per row away from a held span.
        starts: Latent rows each scene opens on, or ``None`` for one scene. Bridges and
            ramps stop at a scene's edges.

    Returns:
        ``(holds, row_holds)``, one hold per frame and one per row.
    """
    profile = np.asarray(profile, dtype=np.float64)
    rows = np.ones(len(profile), dtype=int)
    if len(profile) and float(profile.max()) > 0.0:
        rows[profile > np.quantile(profile, float(quantile))] = int(peak)
    spans = scene_spans(starts, len(rows))
    if len(spans) == 1:
        rows = _spread(rows, peak, bridge, ramp)
    else:
        rows = np.concatenate([_spread(rows[start:stop].copy(), peak, bridge, ramp)
                               for start, stop in spans])
    last = len(rows) - 1
    holds = [int(rows[min(row_of_frame(frame), last)]) for frame in range(int(frames))]
    return holds, [int(value) for value in rows]


def padding(holds) -> int:
    """Frames appended to a stretched clip to reach the 17k+5 grid.

    Args:
        holds: One hold per source frame.

    Returns:
        A count from 0 to 16.
    """
    total = int(sum(holds))
    clips = max(0, math.ceil((total - h3_extend.CLIP_LEAD) / h3_extend.CLIP_FRAMES))
    return h3_extend.CLIP_LEAD + clips * h3_extend.CLIP_FRAMES - total


def _lengthen(holds: list, extra: int) -> list:
    """Holds with ``extra`` showings added to the most held frames, spread evenly."""
    if not extra:
        return holds
    peak = max(holds)
    chosen = [frame for frame, hold in enumerate(holds) if hold == peak]
    for step in range(extra):
        holds[chosen[(step * len(chosen)) // extra]] += 1
    return holds


def aligned(holds, starts=None) -> list[int]:
    """Holds lengthened to put the stretched clip on the 17k+5 grid, every scene on whole clips.

    Extra showings go to each scene's most held frames.

    Args:
        holds: One hold per frame of a clip on the 17k+5 grid.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        The lengthened holds.
    """
    holds = [int(hold) for hold in holds]
    spans = scene_frames(starts, len(holds))
    if len(spans) == 1:
        return _lengthen(holds, padding(holds))
    out = []
    for index, (start, stop) in enumerate(spans):
        part = holds[start:stop]
        extra = (padding(part) if index == len(spans) - 1
                 else -sum(part) % h3_extend.CLIP_FRAMES)
        out.extend(_lengthen(part, extra))
    return out


def stretched_starts(holds, starts) -> list[int]:
    """Latent rows each scene opens on in the stretched clip.

    Args:
        holds: Holds from :func:`aligned`, one per source frame.
        starts: Latent rows each scene opens on in the source clip.

    Returns:
        Row numbers in order, the first ``0``.
    """
    return [int(sum(holds[:start])) // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS
            for start, _ in scene_frames(starts, len(holds))]


def stretch_frames(images: torch.Tensor, holds) -> torch.Tensor:
    """Each frame repeated by its hold.

    Args:
        images: ``[F, H, W, C]`` frames.
        holds: One hold per frame.

    Returns:
        The stretched frames.
    """
    order = [frame for frame, hold in enumerate(holds) for _ in range(int(hold))]
    return images[torch.tensor(order, dtype=torch.long, device=images.device)]


def encode_stretched(vae, images: torch.Tensor, holds, source: torch.Tensor):
    """The stretched clip's video latent.

    Chunks before the first held frame and after the last are copied from ``source``.

    Args:
        vae: The H3 video VAE.
        images: ``[F, H, W, C]`` frames of the fitted clip.
        holds: Aligned holds, one per frame.
        source: The fitted clip's video latent.

    Returns:
        ``(latent, encoded, chunks)``: the ``[B, 24, T, H, W]`` latent, the chunks encoded
        and the chunks it holds.
    """
    size, rows = h3_extend.CLIP_FRAMES, h3_extend.CLIP_TOKENS
    held = [frame for frame, hold in enumerate(holds) if hold > 1]
    total = int(sum(holds))
    chunks = math.ceil(total / size)
    if not held:
        return source, 0, chunks
    shift = total - len(holds)
    first = held[0] // size
    last_shown = int(sum(holds[:held[-1] + 1]))
    after = math.ceil(last_shown / size)
    frames = stretch_frames(images, holds)[first * size:after * size]
    if after >= chunks:
        middle = vae.encode(frames[..., :3])
    else:
        lead = frames[-1:].expand(h3_extend.CLIP_LEAD, *frames.shape[1:])
        middle = vae.encode(torch.cat([frames, lead])[..., :3])[:, :, :(after - first) * rows]
    head = source[:, :, :first * rows]
    tail = source[:, :, (after - shift // size) * rows:]
    latent = torch.cat([head.to(middle), middle, tail.to(middle)], dim=2)
    return latent, after - first, chunks


def recover(images: torch.Tensor, holds) -> torch.Tensor:
    """The first showing of each source frame, taken out of a stretched clip.

    Args:
        images: Decoded stretched frames.
        holds: One hold per source frame.

    Returns:
        One frame per hold.

    Raises:
        ValueError: The clip is shorter than the holds need.
    """
    starts = np.concatenate([[0], np.cumsum(holds)[:-1]]).astype(int)
    needed = int(sum(holds))
    if images.shape[0] < needed:
        raise ValueError(
            f"H3 De-RoPE Recover was given {images.shape[0]} frames and the stretch made "
            f"{needed}. Wire in the decoded output of the latent H3 De-RoPE Stretch built"
        )
    return images[torch.as_tensor(starts, dtype=torch.long, device=images.device)]


def stretch_audio(audio: dict, holds, fps: float) -> dict:
    """Audio slowed under each held frame by its hold, with its pitch kept.

    Args:
        audio: ComfyUI audio covering the source frames.
        holds: One hold per source frame.
        fps: Frame rate of the clip.

    Returns:
        ComfyUI audio as long as the stretched clip.
    """
    starts = np.concatenate([[0.0], np.cumsum(holds)[:-1]])
    spans = np.asarray(holds, dtype=np.float64)

    def source_of(position):
        frame = np.clip(np.searchsorted(starts, position, side="right") - 1, 0, len(holds) - 1)
        return frame + (position - starts[frame]) / spans[frame]

    return audio_timeline.retimed(audio, float(sum(holds)), source_of, fps)


def squeeze_audio(audio: dict, holds, frames: int, fps: float) -> dict:
    """Audio sped up under each held frame by its hold, with its pitch kept.

    Args:
        audio: ComfyUI audio covering the stretched clip.
        holds: One hold per source frame.
        frames: Source frames the result covers.
        fps: Frame rate of the clip.

    Returns:
        ComfyUI audio as long as ``frames`` source frames.
    """
    starts = np.concatenate([[0.0], np.cumsum(holds)[:-1]])
    spans = np.asarray(holds, dtype=np.float64)

    def stretched_of(position):
        frame = np.clip(np.floor(position).astype(int), 0, len(holds) - 1)
        return starts[frame] + (position - frame) * spans[frame]

    return audio_timeline.retimed(audio, float(frames), stretched_of, fps)


def audio_row_strength(strength: float, video, audio):
    """A joint noise mask: the video free, every audio row at ``strength``.

    Args:
        strength: Share of each audio row re-rendered, ``0`` to ``1``.
        video: The video latent, ``[B, 24, T, H, W]``.
        audio: The audio latent, ``[B, 32, 2, L]``.

    Returns:
        The joint mask, as ``join`` builds it.
    """
    video_mask = torch.ones([video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]],
                            dtype=torch.float32, device=video.device)
    audio_mask = torch.full([audio.shape[0], 1, audio.shape[2], audio.shape[3]],
                            float(strength), dtype=torch.float32, device=audio.device)
    return h3_extend.join(video_mask, audio_mask)["samples"]


def plot(result: Plan, width: int = 1280, height: int = 200) -> torch.Tensor:
    """A strip of motion per latent row, coloured by the hold each row got.

    Args:
        result: What H3 De-RoPE Stretch held.
        width: Picture width.
        height: Picture height.

    Returns:
        An IMAGE tensor ``[1, height, width, 3]``.
    """
    from PIL import Image, ImageDraw

    colours = {1: (58, 84, 120), 2: (214, 170, 72), 3: (230, 124, 60)}
    picture = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(picture)
    rows = np.asarray(result.rows, dtype=np.float64)
    base, top = height - 18, 26
    tallest = float(rows.max()) if len(rows) and rows.max() > 0 else 1.0
    span = max(1, len(rows))
    for row, value in enumerate(rows):
        left = int(row / span * width)
        right = max(left + 1, int((row + 1) / span * width) - 1)
        hold = result.row_holds[row]
        colour = colours.get(hold, (232, 72, 72))
        draw.rectangle([left, base - int(value / tallest * (base - top)), right, base],
                       fill=colour)
    for cut in result.cuts:
        x = int(cut / span * width)
        draw.line([(x, top - 4), (x, base)], fill=(236, 236, 242), width=2)
    seconds = result.frames / float(result.fps)
    step = 1 if seconds <= 12 else 5
    for mark in range(0, int(seconds) + 1, step):
        x = int(mark / max(seconds, 1e-6) * (width - 1))
        draw.text((x + 2, base + 3), f"{mark}s", fill=(150, 150, 160))
    held = sum(1 for hold in result.holds[:result.source] if hold > 1)
    heading = (f"{held} of {result.source} frames held, peak x{max(result.holds)}, "
               f"{result.source} -> {result.stretched} frames")
    if result.cuts:
        heading += f", {len(result.cuts)} {'cut' if len(result.cuts) == 1 else 'cuts'} marked"
    draw.text((8, 6), heading, fill=(230, 230, 235))
    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]
