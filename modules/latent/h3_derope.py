"""Holding a MiniMax H3 clip's fast frames for a refining pass, and removing them again.

A hold of ``n`` shows a frame ``n`` times. Rows cover ``1, 4, 4, 4, 4`` frames per 17. Kept clips
pass through as sampled.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..media import audio_timeline
from . import h3_extend, h3_references

__all__ = [
    "AUDIO_MODES",
    "COVERAGE",
    "Plan",
    "aligned",
    "audio_row_strength",
    "decode_scenes",
    "denoise_sigmas",
    "encode_scenes",
    "encode_stretched",
    "first_frame_of_row",
    "fit",
    "guide_frames",
    "kept_clips",
    "kept_rows",
    "latent_audio",
    "motion",
    "moved_keyframes",
    "nearest_showings",
    "pass_mask",
    "padding",
    "pinned_rows",
    "plan",
    "plot",
    "recover",
    "recovered_latent",
    "refine",
    "row_of_frame",
    "scene_frames",
    "scene_spans",
    "settled",
    "shot_cuts",
    "source_sound",
    "squeeze_audio",
    "stretch_audio",
    "stretch_frames",
    "stretched_frame",
    "stretched_starts",
    "video_of",
    "window_keyframes",
    "zeroed",
]

#: Motion a row needs to be held, as a multiple of its scene's typical motion.
FLOOR = 1.6

#: Coverage presets: ``(quantile, peak hold, bridge, floor)``. Rows above the quantile of their
#: shot's motion and at least ``floor`` times their scene's typical motion are held at the peak;
#: gaps of up to ``bridge`` rows between them are filled.
COVERAGE = {
    "balanced": (0.75, 4, 8, FLOOR),
    "wide": (0.70, 4, 8, FLOOR),
    "economy": (0.85, 3, 8, FLOOR),
}

#: A frame step this many times the clip's median step opens a new shot...
CUT_OVER_MEDIAN = 8.0

#: ...when it is also this many times the steps either side of it...
CUT_OVER_NEIGHBOURS = 3.0

#: ...and at least this mean grey change, from 0 to 1.
CUT_FLOOR = 0.05

#: Height and width of the thumbnails cuts are found on.
CUT_SIZE = (36, 64)

#: Rows a cut reaches in the motion measure: from this many before its row to one fewer after.
CUT_REACH = 2

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
        sound: The fitted clip's audio latent ``[B, 32, 2, L]``, or ``None``.
        joint: Whether a latent rebuilt from the recovered frames carries an audio half.
        extras: Entries of the source latent besides ``samples``, for the rebuilt latent.
        pinned: Opening latent rows the source latent held, kept out of every hold.
        guide: ``[F, 40, 72]`` normalised greyscale thumbnails of the fitted clip, or ``None``.
        shots: Frames of the fitted clip a new shot opens on, inside its scenes.
        video: The fitted clip's ``[B, 24, T, H, W]`` video latent on the CPU, or ``None``.
    """

    holds: tuple
    fps: float
    audio: dict | None
    rows: tuple
    row_holds: tuple
    source: int
    cuts: tuple = ()
    sound: object = None
    joint: bool = True
    extras: dict | None = None
    pinned: int = 0
    guide: object = None
    shots: tuple = ()
    video: object = None

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


def first_frame_of_row(row: int) -> int:
    """The first frame a latent row covers.

    Args:
        row: A row index, from 0.

    Returns:
        A frame index, from 0.
    """
    clip, offset = divmod(int(row), h3_extend.CLIP_TOKENS)
    return clip * h3_extend.CLIP_FRAMES + (0 if offset == 0 else 1 + (offset - 1) * 4)


def _mask_video(mask):
    """The video half of a noise mask, or ``None`` where it holds no ``[B, C, T, H, W]`` video."""
    tensors = getattr(mask, "tensors", None)
    if tensors is not None:
        mask = tensors[0] if len(tensors) else None
    if isinstance(mask, torch.Tensor) and mask.ndim == 5:
        return mask
    return None


def pinned_rows(latent: dict) -> int:
    """Opening latent rows a latent's noise mask holds fully, as a carried window's overlap.

    Args:
        latent: A LATENT, joint or video only.

    Returns:
        The count of leading rows whose video mask is ``0`` throughout, ``0`` without a mask.
    """
    mask = _mask_video(latent.get("noise_mask")) if isinstance(latent, dict) else None
    if mask is None:
        return 0
    held = mask.float().abs().amax(dim=(0, 1, 3, 4)) <= 0.0
    free = torch.nonzero(~held)
    return int(free[0, 0]) if free.numel() else int(held.shape[0])


def stretched_frame(frame: int, holds) -> int:
    """Where a source frame is first shown in the stretched clip; the last frame goes to its end.

    Args:
        frame: A frame of the fitted clip, from 0.
        holds: One hold per source frame.

    Returns:
        A frame of the stretched clip.
    """
    frame, count, total = int(frame), len(holds), int(sum(holds))
    if frame >= count - 1:
        return total - count + frame
    return int(sum(holds[:frame])) if frame > 0 else frame


def moved_keyframes(conditioning, holds):
    """A conditioning whose pinned frames sit where their frames land in the stretched clip.

    Args:
        conditioning: A CONDITIONING list, as a MiniMax H3 prompt.
        holds: One hold per source frame.

    Returns:
        ``(conditioning, moved)``: a copy with every ``minimax_keyframes`` entry moved, and how
        many entries changed frame.
    """
    out, moved = [], 0
    for entry in conditioning:
        values = dict(entry[1])
        blocks = values.get("minimax_keyframes")
        if blocks:
            shifted = []
            for block in blocks:
                block = dict(block)
                if "resolved_frame_index" in block:
                    was = int(block["resolved_frame_index"])
                    block["resolved_frame_index"] = stretched_frame(was, holds)
                    moved += int(block["resolved_frame_index"] != was)
                shifted.append(block)
            values["minimax_keyframes"] = shifted
        out.append([entry[0], values, *entry[2:]])
    return out, moved


def _dephased(values: np.ndarray) -> np.ndarray:
    """Motion values with the five-row rhythm of H3's clips divided out, row by row.

    Args:
        values: One value per row of a scene that opens on a clip boundary.

    Returns:
        The rescaled values.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if len(out) < 2 * h3_extend.CLIP_TOKENS:
        return out
    index = np.arange(len(out))
    # The first row and the last two repeat their neighbours and are left out of the levels.
    measured = (index >= 1) & (index < len(out) - 2)
    overall = float(np.median(out[measured]))
    phase = index % h3_extend.CLIP_TOKENS
    for place in range(h3_extend.CLIP_TOKENS):
        rows = (phase == place) & measured
        level = float(np.median(out[rows])) if rows.any() else 0.0
        if level > 0.0:
            out[phase == place] *= overall / level
    return out


def _third(rows: torch.Tensor) -> np.ndarray:
    """Motion of each row of one scene, the third difference over time with the clip rhythm removed.

    Args:
        rows: A float ``[B, 24, T, H, W]`` latent.

    Returns:
        ``T`` values, the first taking the second's and the last two the one before them, or
        zeros under four rows.
    """
    count = rows.shape[2]
    if count < 4:
        return np.zeros(count)
    third = rows[:, :, 3:] - 3 * rows[:, :, 2:-1] + 3 * rows[:, :, 1:-2] - rows[:, :, :-3]
    values = third.abs().mean(dim=(0, 1, 3, 4)).cpu().numpy().astype(np.float64)
    # The difference over rows k to k+3 belongs to row k+1, the middle of its span.
    return _dephased(np.concatenate([values[:1], values, np.repeat(values[-1:], 2)]))


def motion(video: torch.Tensor, starts=None) -> np.ndarray:
    """Motion per latent row, the mean size of its third difference over time inside its scene.

    Args:
        video: The clip's video latent, ``[B, 24, T, H, W]``.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        ``T`` values. Each scene's first row takes its second's, its last two the one before.
    """
    rows = video.float()
    spans = scene_spans(starts, rows.shape[2])
    if len(spans) == 1:
        return _third(rows)
    return np.concatenate([_third(rows[:, :, start:stop]) for start, stop in spans])


def shot_cuts(images: torch.Tensor) -> list[int]:
    """Frames that open a new shot: a jump in the picture far above the steps either side of it.

    Args:
        images: ``[F, H, W, C]`` frames.

    Returns:
        Frame indices in order, none of them ``0``.
    """
    count = int(images.shape[0])
    if count < 3:
        return []
    small = []
    for first in range(0, count, 16):
        grey = images[first:first + 16, ..., :3].float().mean(dim=-1, keepdim=True)
        small.append(torch.nn.functional.interpolate(grey.movedim(-1, 1), size=CUT_SIZE,
                                                     mode="area")[:, 0].cpu())
    small = torch.cat(small)
    steps = (small[1:] - small[:-1]).abs().mean(dim=(1, 2)).numpy().astype(np.float64)
    # Repeated closing frames step by nothing and are left out of the typical step.
    moving = steps[steps > 0.0]
    typical = float(np.median(moving)) if len(moving) else 0.0
    cuts = []
    for index, step in enumerate(steps):
        before = steps[index - 1] if index > 0 else 0.0
        after = steps[index + 1] if index + 1 < len(steps) else 0.0
        if (step >= CUT_FLOOR and step >= CUT_OVER_MEDIAN * typical
                and step >= CUT_OVER_NEIGHBOURS * max(before, after)):
            cuts.append(index + 1)
    return cuts


def _distance_to(flagged: np.ndarray) -> np.ndarray:
    """Rows from each row to the nearest flagged row, ``0`` on one, the row count with none."""
    spots = np.flatnonzero(flagged)
    index = np.arange(len(flagged))
    if not len(spots):
        return np.full(len(flagged), len(flagged), dtype=int)
    at = np.searchsorted(spots, index)
    before = spots[np.clip(at - 1, 0, len(spots) - 1)]
    after = spots[np.clip(at, 0, len(spots) - 1)]
    return np.minimum(np.abs(index - before), np.abs(after - index))


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
         ramp: bool = True, starts=None, pinned: int = 0, tail: int = 0, shots=(),
         floor: float = 0.0):
    """The hold of every frame, from the motion of every row, shot by shot.

    Args:
        profile: Motion per latent row.
        frames: Frames of the clip.
        quantile: Share of a shot's rows below its hold threshold, ``0.5`` to ``0.99``.
        peak: Hold of the fastest rows, ``2`` to ``8``.
        bridge: Longest gap in rows between held rows that is held too.
        ramp: Whether holds fall by one per row away from a held span.
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        pinned: Opening rows that are never held.
        tail: Closing rows that are never held.
        shots: Frames a new shot opens on inside a scene, from :func:`shot_cuts`. Rows the cut
            reaches are never held and holds step down to them.
        floor: Motion a row needs, as a multiple of its scene's median, ``0`` for none.

    Returns:
        ``(holds, row_holds, peaks)``: one hold per frame, one per row, and per shot its
        fastest row over its scene's median, ``0`` where none of its rows was measured.
    """
    profile = np.asarray(profile, dtype=np.float64)
    count = len(profile)
    rows = np.ones(count, dtype=int)
    blocked = np.zeros(count, dtype=bool)
    cut_rows = sorted({row_of_frame(frame) for frame in shots or ()} - {0})
    cut_rows = [row for row in cut_rows if row < count]
    for row in cut_rows:
        blocked[max(0, row - CUT_REACH):row + CUT_REACH] = True
    blocked[:max(0, int(pinned))] = True
    if int(tail) > 0:
        blocked[-int(tail):] = True
    spans, peaks = [], []
    for scene_start, scene_stop in scene_spans(starts, count):
        inside = np.arange(scene_start, scene_stop)
        clean = inside[~blocked[inside]]
        typical = float(np.median(profile[clean])) if len(clean) else 0.0
        edges = [scene_start, *[row for row in cut_rows if scene_start < row < scene_stop],
                 scene_stop]
        for start, stop in zip(edges[:-1], edges[1:]):
            spans.append((start, stop))
            measured = np.arange(start, stop)
            measured = measured[~blocked[measured]]
            if not len(measured) or typical <= 0.0:
                peaks.append(0.0)
                continue
            values = profile[measured]
            peaks.append(float(values.max()) / typical)
            hot = values > np.quantile(values, float(quantile))
            if float(floor) > 0.0:
                hot &= values >= float(floor) * typical
            rows[measured[hot]] = int(peak)
    for start, stop in spans:
        rows[start:stop] = _spread(rows[start:stop].copy(), peak, bridge, ramp)
    if blocked.any():
        rows = np.minimum(rows, 1 + _distance_to(blocked))
    last = count - 1
    holds = [int(rows[min(row_of_frame(frame), last)]) for frame in range(int(frames))]
    return holds, [int(value) for value in rows], peaks


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
    """Holds with ``extra`` showings added to the last frame."""
    if extra:
        holds[-1] += int(extra)
    return holds


def aligned(holds, starts=None) -> list[int]:
    """Holds lengthened to put the stretched clip on the 17k+5 grid, every scene on whole clips.

    Extra showings go to each scene's last frame.

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


def kept_clips(holds) -> list[bool]:
    """Whether each clip of a clip on the 17k+5 grid shows every frame once.

    Args:
        holds: One hold per frame.

    Returns:
        One flag per clip, the closing five-frame clip last.
    """
    size = h3_extend.CLIP_FRAMES
    return [all(int(hold) == 1 for hold in holds[start:start + size])
            for start in range(0, len(holds), size)]


def _runs(flags, starts, frames: int) -> list[tuple[int, int, bool]]:
    """``(first, stop, kept)`` runs of clips sharing a flag, broken where a scene opens."""
    breaks = {start // h3_extend.CLIP_TOKENS
              for start, _ in scene_spans(starts, h3_extend.tokens_for(int(frames)))[1:]}
    out, first = [], 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[first] or index in breaks:
            out.append((first, index, bool(flags[first])))
            first = index
    return out


def settled(holds, starts=None) -> list[int]:
    """Holds evened so every run of held clips stretches by whole clips, its held frames gaining
    or losing showings.

    Args:
        holds: One hold per frame of a clip on the 17k+5 grid.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        The evened holds.
    """
    size = h3_extend.CLIP_FRAMES
    holds = [int(hold) for hold in holds]
    for first, stop, kept in _runs(kept_clips(holds), starts, len(holds)):
        if kept:
            continue
        low, high = first * size, min(stop * size, len(holds))
        short = -(sum(holds[low:high]) - (high - low)) % size
        held = [frame for frame in range(low, high) if holds[frame] > 1]
        if not short or not held:
            continue
        spare = [frame for frame in held if holds[frame] > 2]
        over = size - short
        if over < short and over <= len(spare):
            for step in range(over):
                holds[spare[step * len(spare) // over]] -= 1
        else:
            for step in range(short):
                holds[held[step * len(held) // short]] += 1
    return holds


def kept_rows(holds, stretched: bool = False) -> np.ndarray:
    """Whether each latent row belongs to a kept clip, on the source or the stretched clip.

    Args:
        holds: Holds from :func:`settled`, one per frame of a clip on the 17k+5 grid.
        stretched: Answer for the stretched clip's rows rather than the source's.

    Returns:
        One flag per row.
    """
    size = h3_extend.CLIP_FRAMES
    frames = int(sum(holds)) if stretched else len(holds)
    out = np.zeros(h3_extend.tokens_for(frames), dtype=bool)
    shown = 0
    for clip, kept in enumerate(kept_clips(holds)):
        low, high = clip * size, min((clip + 1) * size, len(holds))
        at = shown if stretched else low
        if kept and at % size == 0:
            out[row_of_frame(at):row_of_frame(at + high - low - 1) + 1] = True
        shown += int(sum(holds[low:high]))
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


def _encode_run(vae, frames: torch.Tensor, closing: bool) -> torch.Tensor:
    """Whole clips of frames encoded, a closing run with its five-frame clip."""
    if closing:
        return vae.encode(frames[..., :3])
    lead = frames[-1:].expand(h3_extend.CLIP_LEAD, *frames.shape[1:])
    rows = frames.shape[0] // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS
    return vae.encode(torch.cat([frames, lead])[..., :3])[:, :, :rows]


def _joined(pieces: list) -> torch.Tensor:
    """Latent pieces joined over time, on the device and type of the first encoded one."""
    like = next((piece for piece, encoded in pieces if encoded), pieces[0][0])
    return torch.cat([piece.to(like) for piece, _ in pieces], dim=2)


def encode_stretched(vae, images: torch.Tensor, holds, source: torch.Tensor, pinned: int = 0,
                     starts=None):
    """The stretched clip's video latent: kept clips from ``source``, held runs encoded.

    Args:
        vae: The H3 video VAE.
        images: ``[F, H, W, C]`` frames of the fitted clip.
        holds: Holds from :func:`settled`, one per frame.
        source: The fitted clip's video latent.
        pinned: Opening rows whose frames are held once each, copied from ``source``.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        ``(latent, encoded, chunks)``: the ``[B, 24, T, H, W]`` latent, the chunks encoded
        and the chunks it holds.
    """
    size, per = h3_extend.CLIP_FRAMES, h3_extend.CLIP_TOKENS
    chunks = math.ceil(int(sum(holds)) / size)
    flags = kept_clips(holds)
    if all(flags):
        return source, 0, chunks
    pieces, encoded = [], 0
    for first, stop, kept in _runs(flags, starts, len(holds)):
        low, high = first * size, min(stop * size, len(holds))
        if kept:
            pieces.append((source[:, :, first * per:min(stop * per, source.shape[2])], False))
            continue
        frames = stretch_frames(images[low:high], holds[low:high])
        encoded += math.ceil(frames.shape[0] / size)
        pieces.append((_encode_run(vae, frames, high >= len(holds)), True))
    latent = _joined(pieces)
    pinned = min(int(pinned), latent.shape[2], source.shape[2])
    if pinned:
        latent[:, :, :pinned] = source[:, :, :pinned].to(latent)
    return latent, encoded, chunks


def source_sound(latent, audio, audio_vae, frames: int):
    """The audio latent of a fitted clip, from its joint latent or its wired audio.

    Args:
        latent: The clip's LATENT, or ``None``.
        audio: ComfyUI audio covering the fitted clip, or ``None``.
        audio_vae: The H3 audio VAE, or ``None``.
        frames: Frames of the fitted clip.

    Returns:
        ``(sound, joint)``: a ``[B, 32, 2, L]`` latent sized to ``frames`` or ``None`` where
        there is none, and whether a rebuilt latent carries an audio half.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    tensors = getattr(samples, "tensors", None)
    if samples is not None and tensors is None:
        return None, False
    sound = None
    if tensors is not None and len(tensors) > 1:
        sound = tensors[-1]
    elif audio is not None and audio_vae is not None:
        sound, _ = h3_references.audio_latent(audio_vae, audio)
    if sound is None:
        return None, True
    span = h3_extend.audio_span(int(frames))
    sound = sound[..., :span]
    if sound.shape[-1] < span:
        sound = torch.nn.functional.pad(sound, (0, span - sound.shape[-1]))
    return sound, True


def encode_scenes(vae, images: torch.Tensor, starts=None) -> torch.Tensor:
    """A clip's video latent, each scene encoded on its own.

    Args:
        vae: The H3 video VAE.
        images: ``[F, H, W, C]`` frames on the 17k+5 grid.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        The ``[B, 24, T, H, W]`` latent.
    """
    spans = scene_frames(starts, int(images.shape[0]))
    pieces = []
    for index, (start, stop) in enumerate(spans):
        frames = images[start:stop, ..., :3]
        if index == len(spans) - 1:
            pieces.append(vae.encode(frames))
            continue
        lead = frames[-1:].expand(h3_extend.CLIP_LEAD, *frames.shape[1:])
        rows = (stop - start) // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS
        pieces.append(vae.encode(torch.cat([frames, lead]))[:, :, :rows])
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)


def _mask_rows(mask) -> int | None:
    """Latent rows a noise mask covers, or ``None`` where it holds no video mask."""
    video = _mask_video(mask)
    return None if video is None else int(video.shape[2])


def _recovered_video(vae, images: torch.Tensor, result: Plan) -> tuple[torch.Tensor, int]:
    """The recovered video latent: kept clips from the source latent, held runs encoded.

    Returns:
        ``(video, encoded)``: the ``[B, 24, T, H, W]`` latent and the clips encoded.
    """
    size, per = h3_extend.CLIP_FRAMES, h3_extend.CLIP_TOKENS
    flags = kept_clips(result.holds)
    if result.video is None or not any(flags):
        return encode_scenes(vae, images, result.cuts), len(flags)
    source, frames = result.video, len(result.holds)
    pieces, encoded = [], 0
    for first, stop, kept in _runs(flags, result.cuts, frames):
        if kept:
            pieces.append((source[:, :, first * per:min(stop * per, source.shape[2])], False))
            continue
        high = min(stop * size, frames)
        pieces.append((_encode_run(vae, images[first * size:high], high >= frames), True))
        encoded += stop - first
    return _joined(pieces), encoded


def recovered_latent(vae, images: torch.Tensor, result: Plan) -> dict:
    """The recovered clip as a latent, kept clips from the source latent, beside its audio.

    Args:
        vae: The H3 video VAE.
        images: One frame per hold, as :func:`recover` answers them.
        result: What H3 De-RoPE Stretch held.

    Returns:
        A LATENT on the fitted clip's timeline: joint where the source was, carrying every
        entry the source latent did. A noise mask is kept only where it still fits.
    """
    video, _ = _recovered_video(vae, images, result)
    if result.joint:
        sound = result.sound
        span = h3_extend.audio_span(int(images.shape[0]))
        if sound is None:
            sound = torch.zeros([video.shape[0], 32, 2, span], dtype=video.dtype,
                                device=video.device)
        latent = h3_extend.join(video, sound.to(device=video.device))
    else:
        latent = {"samples": video}
    for key, value in (result.extras or {}).items():
        if key == "samples" or key in latent:
            continue
        if key == "noise_mask" and _mask_rows(value) != int(video.shape[2]):
            continue
        latent[key] = value
    if result.cuts:
        latent = h3_extend.with_scenes(latent, [0, *result.cuts])
    return latent


def window_keyframes(conditioning, first: int, frames: int):
    """A conditioning keeping only the pinned frames inside one window, counted from its start.

    Args:
        conditioning: A CONDITIONING list.
        first: The window's first frame in the clip.
        frames: Frames the window holds.

    Returns:
        A copy whose ``minimax_keyframes`` hold only the entries in the window.
    """
    out = []
    for entry in conditioning or []:
        values = dict(entry[1])
        blocks = values.get("minimax_keyframes")
        if blocks is not None:
            kept = []
            for block in blocks:
                index = int(block.get("resolved_frame_index", 0)) - int(first)
                if 0 <= index < int(frames):
                    kept.append({**block, "resolved_frame_index": index})
            if kept:
                values["minimax_keyframes"] = kept
            else:
                values.pop("minimax_keyframes")
        out.append([entry[0], values, *entry[2:]])
    return out


def zeroed(conditioning):
    """A conditioning with every embedding zeroed and its other entries kept, as a negative.

    Args:
        conditioning: A CONDITIONING list.

    Returns:
        The zeroed copy.
    """
    out = []
    for entry in conditioning or []:
        values = dict(entry[1])
        pooled = values.get("pooled_output")
        if isinstance(pooled, torch.Tensor):
            values["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(entry[0]), values, *entry[2:]])
    return out


def denoise_sigmas(model, scheduler: str, steps: int, denoise: float):
    """The sigmas KSampler runs for ``steps`` steps at ``denoise``.

    Args:
        model: The model patcher sampling.
        scheduler: Scheduler name.
        steps: Steps to run.
        denoise: Share of the schedule run, ``0`` to ``1``.

    Returns:
        ``steps + 1`` sigmas.
    """
    import comfy.samplers

    sampling = model.get_model_object("model_sampling")
    steps = max(1, int(steps))
    if float(denoise) >= 0.9999:
        return comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
    total = max(steps, int(steps / max(float(denoise), 1e-4)))
    return comfy.samplers.calculate_sigmas(sampling, scheduler, total)[-(steps + 1):]


#: Clips of held context a refined span takes either side of its free rows.
CONTEXT_CLIPS = 2


def _areas(rows: int, starts, free=None) -> list[tuple[int, int, int]]:
    """``(start, stop, bound)`` rows to refine, each free run with :data:`CONTEXT_CLIPS` clips
    either side; ``bound`` is its scene's end."""
    per = h3_extend.CLIP_TOKENS
    out = []
    for start, stop in scene_spans(starts, rows):
        if free is None:
            out.append((start, stop, stop))
            continue
        marks = np.flatnonzero(free[start:stop]) + start
        if not len(marks):
            continue
        breaks = np.flatnonzero(np.diff(marks) > 1)
        for low, high in zip(marks[np.concatenate([[0], breaks + 1])],
                             marks[np.concatenate([breaks, [len(marks) - 1]])]):
            low = max(start, (int(low) // per - CONTEXT_CLIPS) * per)
            high = min(stop, (int(high) // per + 1 + CONTEXT_CLIPS) * per)
            if out and out[-1][2] == stop and low <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], high), stop)
            else:
                out.append((low, high, stop))
    return out


def _windows(rows: int, starts, window_frames: int, overlap_frames: int, free=None) -> list:
    """The windows covering a stretched latent's free rows, none crossing a scene cut.

    Args:
        rows: Latent rows of the stretched clip.
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        window_frames: Frames one window holds at most.
        overlap_frames: Frames each window shares with the one before in its area.
        free: One flag per row, true where the noise mask lets it change, or ``None`` for
            every row.

    Returns:
        ``(start_row, stop_row, chunk)`` per window: its area's first row, the end of the
        area's scene, and its :class:`modules.latent.h3_upscale.Chunk` counted from the area's
        start.
    """
    from . import h3_upscale

    most = (h3_upscale.snap_chunk(window_frames) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES
    shared = h3_upscale.snap_overlap(overlap_frames, window_frames) // h3_extend.CLIP_FRAMES
    out = []
    for start, area_stop, stop in _areas(rows, starts, free):
        frames = h3_extend.frames_for(area_stop - start) if area_stop >= rows else (
            (area_stop - start) // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES
            + h3_extend.CLIP_LEAD)
        total = h3_upscale.clips_for(frames)
        chunk_frames = window_frames
        if total > most > shared:
            # Windows come out even in length rather than leaving a short last one.
            passes = math.ceil((total - shared) / (most - shared))
            per = math.ceil((total - shared) / passes) + shared
            chunk_frames = per * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD
        for chunk in h3_upscale.plan(frames, chunk_frames, shared * h3_extend.CLIP_FRAMES):
            out.append((start, stop, chunk))
    return out


def refine(model, positive, latent: dict, *, steps: int, cfg: float, sampler_name: str,
           scheduler: str, denoise: float, seed: int, window_frames: int, overlap_frames: int,
           negative=None, sampler=None) -> tuple[dict, list]:
    """A stretched H3 latent refined window by window, each holding the clips it shares.

    Args:
        model: The H3 model patcher.
        positive: The prompt, its pinned frames on the stretched clip.
        latent: The stretched joint latent, its noise mask holding any pinned rows.
        steps: Steps a window runs.
        cfg: Guidance scale.
        sampler_name: Solver name, read where ``sampler`` is ``None``.
        scheduler: Scheduler name.
        denoise: Share of the schedule each window runs, as KSampler reads it.
        seed: Noise seed of the first window; each later window adds its index.
        window_frames: Frames one window holds at most.
        overlap_frames: Frames each window shares with the one before.
        negative: The negative prompt, or ``None`` for the positive zeroed.
        sampler: A SAMPLER run in place of ``sampler_name``, or ``None``.

    Returns:
        ``(latent, lines)``: the refined latent carrying the input's entries, and one line per
        window.
    """
    import comfy.model_management
    import comfy.sample
    import comfy.utils

    from ..sampling import preview

    video, sound = h3_extend.split(latent)
    video_out, sound_out = video.clone(), sound.clone()
    mask = latent.get("noise_mask")
    tensors = getattr(mask, "tensors", None)
    video_mask = _mask_video(mask)
    sound_mask = tensors[-1] if tensors is not None and len(tensors) > 1 else None
    if negative is None:
        negative = zeroed(positive)
    steps = max(1, int(steps))
    free = None
    if video_mask is not None and int(video_mask.shape[2]) == int(video.shape[2]):
        free = (video_mask.float().amax(dim=(0, 1, 3, 4)) > 0.0).cpu().numpy()
    plan_ = _windows(video.shape[2], h3_extend.scene_starts(latent), window_frames,
                     overlap_frames, free)
    if not plan_:
        out = {key: value for key, value in latent.items() if key not in ("samples", "noise_mask")}
        out.update(h3_extend.join(video_out, sound_out))
        return out, ["nothing held, every clip kept as sampled"]
    callback = preview.prepare_callback(model, steps * len(plan_))
    sigmas = denoise_sigmas(model, scheduler, steps, denoise)
    lines = []
    for index, (scene_row, scene_end, chunk) in enumerate(plan_):
        comfy.model_management.throw_exception_if_processing_interrupted()
        began = time.monotonic()
        start = scene_row + chunk.start_row
        stop = min(start + chunk.rows, scene_end)
        rows = video_out[:, :, start:stop]
        padded = chunk.rows - rows.shape[2]
        if padded > 0:
            # A window closing on a cut repeats the scene's last row; those rows are not kept.
            rows = torch.cat([rows, rows[:, :, -1:].expand(-1, -1, padded, -1, -1)], dim=2)
        first = start // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES
        span = h3_extend.audio_span(chunk.frames)
        a0 = h3_extend.audio_span(first)
        heard = sound_out[..., a0:a0 + span]
        if heard.shape[-1] < span:
            heard = torch.nn.functional.pad(heard, (0, span - heard.shape[-1]))
        frame_mask = torch.ones([rows.shape[0], 1, *rows.shape[2:]], dtype=torch.float32)
        if video_mask is not None:
            given = video_mask[:, :1, start:stop].float().cpu()
            if tuple(given.shape[3:]) == tuple(frame_mask.shape[3:]):
                frame_mask[:, :, :given.shape[2]] = given
        frame_mask[:, :, :chunk.held] = 0.0
        writes = min(chunk.kept, stop - start)
        if writes <= chunk.held or float(frame_mask[:, :, chunk.held:writes].max()) <= 0.0:
            lines.append(f"window {index + 1} of {len(plan_)}: stretched frames {first} to "
                         f"{first + chunk.frames - 1}, every clip kept as sampled")
            continue
        heard_mask = torch.ones([heard.shape[0], 1, *heard.shape[2:]], dtype=torch.float32)
        if sound_mask is not None:
            given = sound_mask[..., a0:a0 + span].float().cpu()
            heard_mask[..., :given.shape[-1]] = given[:, :1]
        held_frames = chunk.held // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES
        heard_mask[..., :h3_extend.audio_span(held_frames)] = 0.0
        joined = h3_extend.join(rows, heard)
        samples = comfy.sample.fix_empty_latent_channels(model, joined["samples"])
        noise_mask = h3_extend.join(frame_mask, heard_mask)["samples"]
        noise = comfy.sample.prepare_noise(samples, int(seed) + index)
        pos = window_keyframes(positive, first, chunk.frames)
        neg = window_keyframes(negative, first, chunk.frames)
        offset = index * steps

        def stepped(step, x0, x, total, offset=offset):
            callback(offset + step, x0, x, steps * len(plan_))

        bar = not comfy.utils.PROGRESS_BAR_ENABLED
        if sampler is not None:
            result = comfy.sample.sample_custom(
                model, noise, float(cfg), sampler, sigmas, pos, neg, samples,
                noise_mask=noise_mask, callback=stepped, disable_pbar=bar, seed=int(seed) + index)
        else:
            result = comfy.sample.sample(
                model, noise, steps, float(cfg), sampler_name, scheduler, pos, neg, samples,
                noise_mask=noise_mask, sigmas=sigmas, callback=stepped, disable_pbar=bar,
                seed=int(seed) + index)
        refined, heard_refined = h3_extend.split({"samples": result})
        kept = min(chunk.kept, stop - start)
        fresh = refined[:, :, chunk.held:kept].to(video_out)
        # Rows the mask held fully stay exactly as they came in.
        frozen = (frame_mask[0, 0, chunk.held:kept].amax(dim=(1, 2)) <= 0.0).to(fresh.device)
        if bool(frozen.any()):
            fresh[:, :, frozen] = video_out[:, :, start + chunk.held:start + kept][:, :, frozen]
        video_out[:, :, start + chunk.held:start + kept] = fresh
        kept_frames = (h3_extend.frames_for(kept) if kept % h3_extend.CLIP_TOKENS
                       else kept // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES)
        a1 = a0 + h3_extend.audio_span(held_frames)
        a2 = min(sound_out.shape[-1], a0 + h3_extend.audio_span(kept_frames))
        if a2 > a1:
            sound_out[..., a1:a2] = heard_refined[..., a1 - a0:a2 - a0].to(sound_out)
        lines.append(
            f"window {index + 1} of {len(plan_)}: stretched frames {first} to "
            f"{first + chunk.frames - 1}, {held_frames} held, {time.monotonic() - began:.0f} s"
        )
        del result, samples, noise, joined
    out = {key: value for key, value in latent.items() if key not in ("samples", "noise_mask")}
    out.update(h3_extend.join(video_out, sound_out))
    return out, lines


#: Height and width of the thumbnails a recovered frame is matched on.
GUIDE_SIZE = (40, 72)

#: Source frames either side whose showings a nearest pick may reach.
GUIDE_REACH = 3


def guide_frames(images: torch.Tensor) -> torch.Tensor:
    """Small greyscale thumbnails of a clip, each scaled to zero mean and unit spread.

    Args:
        images: ``[F, H, W, C]`` frames.

    Returns:
        A float16 ``[F, 40, 72]`` tensor on the CPU.
    """
    out = []
    for first in range(0, int(images.shape[0]), 16):
        grey = images[first:first + 16, ..., :3].float().mean(dim=-1, keepdim=True)
        small = torch.nn.functional.interpolate(grey.movedim(-1, 1), size=GUIDE_SIZE,
                                                mode="area")[:, 0]
        small = small - small.mean(dim=(1, 2), keepdim=True)
        small = small / small.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
        out.append(small.half().cpu())
    return torch.cat(out)


def nearest_showings(guide: torch.Tensor, refined: torch.Tensor, holds,
                     reach: int = GUIDE_REACH) -> list[int]:
    """The stretched frame each source frame is taken from, nearest its thumbnail and in order.

    Args:
        guide: Thumbnails of the source frames, from :func:`guide_frames`.
        refined: Thumbnails of the refined stretched frames.
        holds: One hold per source frame.
        reach: Source frames either side whose showings may be chosen.

    Returns:
        One strictly increasing stretched frame index per source frame.
    """
    starts = np.concatenate([[0], np.cumsum(holds)[:-1]]).astype(int)
    count, total = len(holds), int(refined.shape[0])
    ends = np.concatenate([starts[1:], [int(sum(holds))]])
    candidates, costs = [], []
    for frame in range(count):
        low = int(starts[max(0, frame - reach)])
        high = min(total, int(ends[min(count - 1, frame + reach)]))
        span = np.arange(low, high)
        gap = (refined[low:high].float() - guide[frame].float()).abs().mean(dim=(1, 2))
        candidates.append(span)
        costs.append(gap.numpy().astype(np.float64))
    best = [costs[0]]
    back = [np.full(len(candidates[0]), -1)]
    for frame in range(1, count):
        previous, before = candidates[frame - 1], best[-1]
        prefix = np.minimum.accumulate(before)
        order = np.maximum.accumulate(np.where(before == prefix, np.arange(len(before)), 0))
        # The cheapest earlier pick at a strictly smaller frame index.
        cut = np.searchsorted(previous, candidates[frame], side="left") - 1
        reach = cut >= 0
        total_cost = np.full(len(candidates[frame]), np.inf)
        total_cost[reach] = costs[frame][reach] + prefix[cut[reach]]
        link = np.full(len(candidates[frame]), -1)
        link[reach] = order[cut[reach]]
        best.append(total_cost)
        back.append(link)
    chosen = [int(np.argmin(best[-1]))]
    for frame in range(count - 1, 0, -1):
        chosen.append(int(back[frame][chosen[-1]]))
    chosen.reverse()
    picks = [int(candidates[frame][index]) for frame, index in enumerate(chosen)]
    if any(index < 0 for index in chosen) or any(b <= a for a, b in zip(picks, picks[1:])):
        return [int(start) for start in starts]
    return picks


def recover(images: torch.Tensor, holds, picks=None) -> torch.Tensor:
    """One showing of each source frame, taken out of a stretched clip.

    Args:
        images: Decoded stretched frames.
        holds: One hold per source frame.
        picks: The stretched frame each source frame is taken from, or ``None`` for each
            frame's first showing.

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
    chosen = starts if picks is None else np.asarray(picks, dtype=int)
    return images[torch.as_tensor(chosen, dtype=torch.long, device=images.device)]


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


def audio_row_strength(strength: float, video, audio, pinned: int = 0):
    """A joint noise mask: the video free past its pinned rows, every audio row at ``strength``.

    Args:
        strength: Share of each audio row re-rendered, ``0`` to ``1``.
        video: The video latent, ``[B, 24, T, H, W]``.
        audio: The audio latent, ``[B, 32, 2, L]``.
        pinned: Opening video rows held as they are.

    Returns:
        The joint mask, as ``join`` builds it.
    """
    video_mask = torch.ones([video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]],
                            dtype=torch.float32, device=video.device)
    video_mask[:, :, :max(0, int(pinned))] = 0.0
    audio_mask = torch.full([audio.shape[0], 1, audio.shape[2], audio.shape[3]],
                            float(strength), dtype=torch.float32, device=audio.device)
    return h3_extend.join(video_mask, audio_mask)["samples"]


def pass_mask(strength: float, video, audio, holds, pinned: int = 0):
    """A joint noise mask for the refining pass: held runs free, kept clips and pinned rows held.

    Args:
        strength: Share of each audio row under a held run re-rendered, ``0`` to ``1``.
        video: The stretched video latent, ``[B, 24, T, H, W]``.
        audio: The stretched audio latent, ``[B, 32, 2, L]``.
        holds: Holds from :func:`settled`, one per source frame.
        pinned: Opening video rows held as they are.

    Returns:
        The joint mask, as ``join`` builds it, or ``None`` where every row is free at full
        strength.
    """
    count = int(video.shape[2])
    kept = kept_rows(holds, stretched=True)[:count]
    if not kept.any() and not int(pinned) and float(strength) >= 1.0:
        return None
    video_mask = torch.ones([video.shape[0], 1, count, video.shape[3], video.shape[4]],
                            dtype=torch.float32, device=video.device)
    video_mask[:, :, torch.as_tensor(np.flatnonzero(kept), dtype=torch.long)] = 0.0
    video_mask[:, :, :max(0, int(pinned))] = 0.0
    audio_mask = torch.full([audio.shape[0], 1, audio.shape[2], audio.shape[3]],
                            float(strength), dtype=torch.float32, device=audio.device)
    total = int(sum(holds))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], kept.astype(int), [0]])))
    for low, high in zip(edges[::2], edges[1::2]):
        first = first_frame_of_row(int(low))
        last = first_frame_of_row(int(high)) if int(high) < count else total
        audio_mask[..., h3_extend.audio_span(first):h3_extend.audio_span(last)] = 0.0
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
    # Scaled to the tallest row no shot cut reaches; a cut's rows are cut off at the top.
    reach = np.zeros(len(rows), dtype=bool)
    for frame in result.shots:
        row = row_of_frame(frame)
        reach[max(0, row - CUT_REACH):row + CUT_REACH] = True
    clean = rows[~reach] if (~reach).any() else rows
    tallest = float(clean.max()) if len(clean) and clean.max() > 0 else 1.0
    span = max(1, len(rows))
    for row, value in enumerate(rows):
        left = int(row / span * width)
        right = max(left + 1, int((row + 1) / span * width) - 1)
        hold = result.row_holds[row]
        colour = (96, 96, 108) if reach[row] else colours.get(hold, (232, 72, 72))
        draw.rectangle([left, base - int(min(1.0, value / tallest) * (base - top)), right, base],
                       fill=colour)
    for frame in result.shots:
        x = int(row_of_frame(frame) / span * width)
        draw.line([(x, top - 4), (x, base)], fill=(150, 150, 165), width=1)
    for cut in result.cuts:
        x = int(cut / span * width)
        draw.line([(x, top - 4), (x, base)], fill=(236, 236, 242), width=2)
    seconds = result.frames / float(result.fps)
    step = 1 if seconds <= 12 else 5
    for mark in range(0, int(seconds) + 1, step):
        x = int(mark / max(seconds, 1e-6) * (width - 1))
        draw.text((x + 2, base + 3), f"{mark}s", fill=(150, 150, 160))
    held = sum(1 for hold in result.holds[:result.source] if hold > 1)
    peak = max(result.row_holds) if result.row_holds else max(result.holds)
    flags = kept_clips(result.holds)
    heading = (f"{held} of {result.source} frames held, peak x{peak}, "
               f"{result.source} -> {result.stretched} frames, "
               f"{sum(flags)} of {len(flags)} clips kept")
    if result.shots:
        heading += f", {len(result.shots)} shot {'cut' if len(result.shots) == 1 else 'cuts'}"
    if result.cuts:
        heading += f", {len(result.cuts)} scene {'cut' if len(result.cuts) == 1 else 'cuts'}"
    draw.text((8, 6), heading, fill=(230, 230, 235))
    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]
