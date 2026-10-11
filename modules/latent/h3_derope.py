"""Holding a MiniMax H3 clip's fast frames for a refining pass, and removing them again.

A hold of ``n`` shows a frame ``n`` times. Latent rows cover ``1, 4, 4, 4, 4`` frames per 17.
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
    "latent_audio",
    "motion",
    "moved_keyframes",
    "nearest_showings",
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
        sound: The fitted clip's audio latent ``[B, 32, 2, L]``, or ``None``.
        joint: Whether a latent rebuilt from the recovered frames carries an audio half.
        extras: Entries of the source latent besides ``samples``, for the rebuilt latent.
        pinned: Opening latent rows the source latent held, kept out of every hold.
        guide: ``[F, 40, 72]`` normalised greyscale thumbnails of the fitted clip, or ``None``.
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
         ramp: bool = True, starts=None, pinned: int = 0, tail: int = 0):
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
        pinned: Opening rows that are never held.
        tail: Closing rows that are never held.

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
    rows[:max(0, int(pinned))] = 1
    if int(tail) > 0:
        rows[-int(tail):] = 1
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


def encode_stretched(vae, images: torch.Tensor, holds, source: torch.Tensor, pinned: int = 0):
    """The stretched clip's video latent.

    Chunks before the first held frame and after the last, and the pinned rows, come from
    ``source``.

    Args:
        vae: The H3 video VAE.
        images: ``[F, H, W, C]`` frames of the fitted clip.
        holds: Aligned holds, one per frame.
        source: The fitted clip's video latent.
        pinned: Opening rows whose frames are held once each.

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
    pinned = min(int(pinned), latent.shape[2], source.shape[2])
    if pinned > first * rows:
        latent[:, :, first * rows:pinned] = source[:, :, first * rows:pinned].to(latent)
    return latent, after - first, chunks


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


def recovered_latent(vae, images: torch.Tensor, result: Plan) -> dict:
    """The recovered frames encoded again, beside the source's own audio and records.

    Args:
        vae: The H3 video VAE.
        images: One frame per hold, as :func:`recover` answers them.
        result: What H3 De-RoPE Stretch held.

    Returns:
        A LATENT on the fitted clip's timeline: joint where the source was, carrying every
        entry the source latent did. A noise mask is kept only where it still fits.
    """
    video = encode_scenes(vae, images, result.cuts)
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


def _windows(rows: int, starts, window_frames: int, overlap_frames: int) -> list:
    """The windows covering a stretched latent, none crossing a scene cut.

    Args:
        rows: Latent rows of the stretched clip.
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        window_frames: Frames one window holds at most.
        overlap_frames: Frames each window shares with the one before in its scene.

    Returns:
        ``(start_row, stop_row, chunk)`` per window: its scene's first row and the row after
        its last, and its :class:`modules.latent.h3_upscale.Chunk` counted from the scene's
        start.
    """
    from . import h3_upscale

    most = (h3_upscale.snap_chunk(window_frames) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES
    shared = h3_upscale.snap_overlap(overlap_frames, window_frames) // h3_extend.CLIP_FRAMES
    out = []
    for start, stop in scene_spans(starts, rows):
        frames = h3_extend.frames_for(stop - start) if stop >= rows else (
            (stop - start) // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD)
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
    plan_ = _windows(video.shape[2], h3_extend.scene_starts(latent), window_frames,
                     overlap_frames)
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
        video_out[:, :, start + chunk.held:start + kept] = \
            refined[:, :, chunk.held:kept].to(video_out)
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
    peak = max(result.row_holds) if result.row_holds else max(result.holds)
    heading = (f"{held} of {result.source} frames held, peak x{peak}, "
               f"{result.source} -> {result.stretched} frames")
    if result.cuts:
        heading += f", {len(result.cuts)} {'cut' if len(result.cuts) == 1 else 'cuts'} marked"
    draw.text((8, 6), heading, fill=(230, 230, 235))
    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]
