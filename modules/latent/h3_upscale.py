"""Upscaling a long MiniMax H3 clip in overlapping chunks of one continuous latent.

A chunk is whole 17-frame clips plus the five frames after them; its opening rows are held as
the chunk before refined them.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

import torch

from .. import log
from . import h3_decode, h3_extend

__all__ = [
    "BICUBIC",
    "Chunk",
    "MANUAL",
    "PRESETS",
    "Preset",
    "Result",
    "auto_chunking",
    "auto_tokens",
    "clips_for",
    "plan",
    "render",
    "snap_chunk",
    "snap_overlap",
    "target_size",
]

logger = log.get_logger("latent.h3_upscale")

#: Bytes of float frames one VAE encode call is handed at most.
ENCODE_BYTES = 4 * 1024 ** 3

#: Frames corrected against their source frames at once.
COLOR_BATCH = 8

#: The upscale choice that resizes without a learned upscaler.
BICUBIC = "bicubic"

#: The mode whose settings are set by hand.
MANUAL = "manual"

#: Whole clips one pass holds at most: H3's longest window of 362 frames.
MOST_CLIPS = 21

#: Whole clips each pass shares with the one before when the chunks are sized automatically.
AUTO_SHARED = 1

#: Share of the card the learned upscaler may fill when chunks are sized automatically.
UPSCALER_SHARE = 0.6


class Preset(NamedTuple):
    """Sampler settings one mode stands for.

    Attributes:
        steps: Steps a pass runs.
        sampler_name: Solver name.
        scheduler: Scheduler name.
        strength: Sigma each pass starts from.
    """

    steps: int
    sampler_name: str
    scheduler: str
    strength: float


#: Sigma a light preset starts each chunk from.
LIGHT = 0.4

#: Sigma a creative preset starts each chunk from.
CREATIVE = 0.75


def _pair(steps: int, sampler_name: str, scheduler: str) -> dict:
    """The light and creative presets for one step count."""
    name = f"{'full ' if steps >= 25 else ''}{steps} step"
    return {
        f"{name} (light)": Preset(steps, sampler_name, scheduler, LIGHT),
        f"{name} (creative)": Preset(steps, sampler_name, scheduler, CREATIVE),
    }


#: The modes and their settings, in menu order; every one runs at cfg 1.
PRESETS = {
    **_pair(2, "euler", "simple"),
    **_pair(4, "euler", "simple"),
    **_pair(8, "euler", "beta"),
    **_pair(12, "euler", "beta"),
    **_pair(25, "res_multistep", "simple"),
}


class Chunk(NamedTuple):
    """One sampler pass over part of the clip.

    Attributes:
        index: Place among the chunks, from 0.
        first_clip: The 17-frame clip the chunk opens on.
        clips: Whole clips it refines, before the five frames that close it.
        held: Latent rows at its start held from the chunk before.
        last: Whether it closes the clip.
    """

    index: int
    first_clip: int
    clips: int
    held: int
    last: bool

    @property
    def start_row(self) -> int:
        """The chunk's first row in the whole clip's latent."""
        return self.first_clip * h3_extend.CLIP_TOKENS

    @property
    def rows(self) -> int:
        """Latent rows the chunk samples."""
        return self.clips * h3_extend.CLIP_TOKENS + h3_extend.TOKEN_LEAD

    @property
    def first_frame(self) -> int:
        """The chunk's first frame in the clip."""
        return self.first_clip * h3_extend.CLIP_FRAMES

    @property
    def frames(self) -> int:
        """Frames the chunk encodes."""
        return self.clips * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD

    @property
    def kept(self) -> int:
        """Rows of the chunk that stay in the clip, counted from its start."""
        return self.rows if self.last else self.clips * h3_extend.CLIP_TOKENS


class Result(NamedTuple):
    """What a render produced.

    Attributes:
        latent: The refined video latent of the whole clip, ``[1, 24, T, H, W]``.
        frames: Frames written to the cache.
        size: ``(width, height)`` of the frames in pixels.
        source: ``(width, height)`` of the frames read, in pixels.
        chunks: The chunks the render ran, in order.
        start: Sigma each pass started from.
        disagreement: Largest difference between two chunks' upscales of the rows they share, as
            a share of the latent's spread, or ``None`` for one chunk.
        lines: One line per chunk.
        chunk_frames: Frames one pass held at most.
        overlap_frames: Frames each pass shared with the one before.
        max_tokens: Most tokens one tile held.
    """

    latent: torch.Tensor
    frames: int
    size: tuple
    source: tuple
    chunks: list
    start: float
    disagreement: float | None
    lines: list
    chunk_frames: int
    overlap_frames: int
    max_tokens: int


def clips_for(frames: int) -> int:
    """Whole 17-frame clips before the five frames that close a clip of ``frames``.

    Args:
        frames: Frames in the clip.

    Returns:
        The clip count of the shortest H3 length holding them all.
    """
    return max(0, math.ceil((int(frames) - h3_extend.CLIP_LEAD) / h3_extend.CLIP_FRAMES))


def snap_chunk(frames: int) -> int:
    """``frames`` brought down to an H3 length of at least one whole clip.

    Args:
        frames: Frames asked for in one pass.

    Returns:
        ``17k + 5`` frames, ``k`` at least 1.
    """
    clips = max(1, (int(frames) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES)
    return clips * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD


def snap_overlap(frames: int, chunk_frames: int) -> int:
    """``frames`` brought down to whole clips, fewer than a chunk holds.

    Args:
        frames: Frames asked to be shared between chunks.
        chunk_frames: Frames in one pass, an H3 length.

    Returns:
        A multiple of 17.
    """
    per = (snap_chunk(chunk_frames) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES
    shared = max(0, min(per - 1, int(frames) // h3_extend.CLIP_FRAMES))
    return shared * h3_extend.CLIP_FRAMES


def plan(frames: int, chunk_frames: int, overlap_frames: int) -> list[Chunk]:
    """The passes that cover a clip of ``frames``.

    Args:
        frames: Frames in the clip.
        chunk_frames: Frames in one pass, snapped by :func:`snap_chunk`.
        overlap_frames: Frames each pass shares with the one before, snapped by
            :func:`snap_overlap`.

    Returns:
        The chunks in order, the last closing the clip.
    """
    total = clips_for(frames)
    per = (snap_chunk(chunk_frames) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES
    shared = snap_overlap(overlap_frames, chunk_frames) // h3_extend.CLIP_FRAMES
    chunks: list[Chunk] = []
    first = 0
    while True:
        end = min(first + per, total)
        held = 0
        if chunks:
            previous = chunks[-1]
            held = (previous.first_clip + previous.clips - first) * h3_extend.CLIP_TOKENS
        chunks.append(Chunk(len(chunks), first, end - first, held, end >= total))
        if end >= total:
            return chunks
        first = end - shared


def auto_chunking(frames: int, height: int, width: int, learned: bool,
                  card_bytes: int | None) -> tuple[int, int]:
    """Frames per pass and frames shared, sized to the clip, the output and the card.

    Args:
        frames: Frames in the clip.
        height: Target latent height.
        width: Target latent width.
        learned: Whether the learned upscaler runs, whose memory grows with a pass's length.
        card_bytes: Device memory in bytes, or ``None`` where nothing can say.

    Returns:
        ``(chunk_frames, overlap_frames)`` for :func:`plan`. Chunks come out even in length.
    """
    from ..model import h3_latent_upscaler

    total = clips_for(frames)
    most = MOST_CLIPS
    if learned and card_bytes:
        per_row = h3_latent_upscaler.working_bytes(1, height, width)
        rows = int(card_bytes * UPSCALER_SHARE) // max(1, per_row)
        most = max(AUTO_SHARED + 1, min(MOST_CLIPS, (rows - h3_extend.TOKEN_LEAD) // h3_extend.CLIP_TOKENS))
    if total <= most:
        return total * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD, 0
    passes = math.ceil((total - AUTO_SHARED) / (most - AUTO_SHARED))
    clips = math.ceil((total - AUTO_SHARED) / passes) + AUTO_SHARED
    return clips * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD, AUTO_SHARED * h3_extend.CLIP_FRAMES


def auto_tokens(card_bytes: int | None) -> int:
    """Most tokens one tile holds, from the card's memory.

    Args:
        card_bytes: Device memory in bytes, or ``None`` where nothing can say.

    Returns:
        ``64000`` from 22 GB, ``40000`` from 14 GB, else ``24000``.
    """
    gigabytes = (card_bytes or 0) / 1024 ** 3
    if gigabytes >= 22:
        return 64000
    if gigabytes >= 14:
        return 40000
    return 24000


def target_size(height: int, width: int, scale: float) -> tuple[int, int]:
    """Latent height and width ``scale`` times the source's, each even and never smaller.

    Args:
        height: Source latent height.
        width: Source latent width.
        scale: Size multiplier.

    Returns:
        ``(height, width)`` in latent pixels.
    """
    def side(length: int) -> int:
        return max(length + length % 2, 2 * round(length * float(scale) / 2))

    return side(int(height)), side(int(width))


def _pixels(frames, first: int, count: int) -> torch.Tensor:
    """Frames ``first`` to ``first + count`` as floats in 0 to 1, the last frame repeated past the end."""
    last = int(frames.shape[0]) - 1
    stop = min(first + count, last + 1)
    block = frames[first:stop, ..., :3]
    if stop - first < count:
        block = torch.cat([block, frames[last:last + 1, ..., :3].expand(count - (stop - first), -1, -1, -1)])
    if block.dtype == torch.uint8:
        return block.float().div_(255.0)
    return block.float()


def _encode(vae, frames, chunk: Chunk) -> torch.Tensor:
    """The chunk's rows of the whole clip's latent, encoded a few clips at a time."""
    height, width = int(frames.shape[1]), int(frames.shape[2])
    per_clip = h3_extend.CLIP_FRAMES * height * width * 3 * 4
    group = max(1, ENCODE_BYTES // max(1, per_clip))
    pieces = []
    clip = chunk.first_clip
    end = chunk.first_clip + chunk.clips
    while True:
        count = min(group, end - clip)
        first = clip * h3_extend.CLIP_FRAMES
        pixels = _pixels(frames, first, count * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD)
        latent = vae.encode(pixels)
        if clip + count >= end:
            pieces.append(latent)
            break
        pieces.append(latent[:, :, :count * h3_extend.CLIP_TOKENS])
        clip += count
    return torch.cat([piece.cpu() for piece in pieces], dim=2)


def _resized(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """A video latent resized to ``height`` x ``width`` a row at a time, bicubic."""
    batch, channels, rows = latent.shape[:3]
    flat = latent.permute(0, 2, 1, 3, 4).reshape(batch * rows, channels, *latent.shape[3:]).float()
    flat = torch.nn.functional.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False)
    return flat.reshape(batch, rows, channels, height, width).permute(0, 2, 1, 3, 4).to(latent.dtype)


def _release() -> None:
    """Hand back every loaded model before another takes the card."""
    from ..model import release_models

    release_models()


def render(clip, model, positive, negative, vae, cache, *, upscale_model: str = BICUBIC,
           scale: float = 2.0, seed: int = 0, steps: int = 8, cfg: float = 1.0,
           sampler_name: str = "euler", scheduler: str = "simple", strength: float = 0.5,
           chunk_frames: int | None = None, overlap_frames: int | None = None,
           max_tokens: int | None = None, anchor: str = "auto", audio_vae=None,
           color_method: str | None = None, node: str = "H3 Upscale Video") -> Result:
    """Upscale and refine a clip chunk by chunk, writing its frames to ``cache`` as they settle.

    Args:
        clip: A :class:`modules.media.clip.Clip`.
        model: The MiniMax H3 model patcher.
        positive: Positive conditioning.
        negative: Negative conditioning.
        vae: The H3 video VAE.
        cache: A :class:`modules.media.frame_cache.FrameCache` the frames are added to.
        upscale_model: A learned upscaler checkpoint name, or :data:`BICUBIC`.
        scale: Size multiplier.
        seed: Noise seed of the first chunk; each later chunk adds its index.
        steps: Steps a pass runs.
        cfg: Guidance scale.
        sampler_name: Solver name.
        scheduler: Scheduler name.
        strength: Sigma each pass starts from.
        chunk_frames: Frames in one pass, or ``None`` with ``overlap_frames`` to size both by
            :func:`auto_chunking`.
        overlap_frames: Frames each pass shares with the one before.
        max_tokens: Most tokens one tile holds, or ``None`` for :func:`auto_tokens`.
        anchor: ``auto``, ``on`` or ``off``, guiding tiles by the first frame of every clip.
        audio_vae: The H3 audio VAE that encodes the clip's sound for the passes, or ``None`` to
            hold silence.
        color_method: A method from :data:`modules.image.color_fix.METHODS` correcting each
            written frame against its source frame, or ``None`` to write frames as decoded.
        node: The node rendering, for messages.

    Returns:
        The :class:`Result`.
    """
    import comfy.model_management
    import comfy.sample
    import comfy.utils

    from ..image import color_fix
    from ..media import clip as clips
    from ..model import h3_latent_upscaler
    from ..model import h3_tiles_wip as h3_tiles
    from ..sampling import preview
    from . import h3_references

    frames = clip.frames
    total_frames = int(frames.shape[0])
    device = comfy.model_management.get_torch_device()

    card = int(comfy.model_management.get_total_memory(device)) if device.type != "cpu" else None
    learned = upscale_model != BICUBIC
    goal = target_size(int(frames.shape[1]) // 16, int(frames.shape[2]) // 16, scale)
    if chunk_frames is None or overlap_frames is None:
        chunk_frames, overlap_frames = auto_chunking(total_frames, goal[0], goal[1], learned, card)
    if max_tokens is None:
        max_tokens = auto_tokens(card)
    chunks = plan(total_frames, chunk_frames, overlap_frames)
    total_rows = chunks[-1].start_row + chunks[-1].rows
    text = max((int(cond[0].shape[1]) for cond in (positive or []) + (negative or [])
                if hasattr(cond[0], "shape") and cond[0].ndim >= 2), default=0)
    settings = ("auto", int(max_tokens))
    tiled = h3_tiles.tiled_model(model, settings, None)
    steps = max(1, int(steps))
    callback = preview.prepare_callback(tiled, steps * len(chunks))

    refined: list[torch.Tensor] = []
    kept_rows = 0
    decoded = 0
    written = 0
    tail = upscaled_tail = None
    disagreement = None
    start_sigma = None
    lines = []
    size = (0, 0)
    source = (int(frames.shape[2]) // 16 * 16, int(frames.shape[1]) // 16 * 16)

    def emit(images):
        nonlocal written
        room = total_frames - written
        if room <= 0:
            return
        images = images[:room]
        for first in range(0, int(images.shape[0]), COLOR_BATCH):
            part = images[first:first + COLOR_BATCH]
            if color_method:
                source = _pixels(frames, written, int(part.shape[0]))
                part = color_fix.correct(part.to(device), source.to(device), color_method)
            cache.append(part, scene=written == 0, device=device)
            written += int(part.shape[0])

    def decode_until(stop: int, last: int) -> None:
        nonlocal decoded
        if stop <= decoded:
            return
        whole = torch.cat(refined, dim=2)
        h3_decode.decode_scene(vae, whole, decoded, stop, emit, first=0, last=last)
        decoded = stop

    for chunk in chunks:
        began = time.monotonic()
        comfy.model_management.throw_exception_if_processing_interrupted()
        if chunk.index:
            _release()
            decode_until((kept_rows // h3_extend.CLIP_TOKENS - 1) * h3_extend.CLIP_TOKENS, total_rows)

        latent = _encode(vae, frames, chunk)
        height, width = goal
        size = (width * 16, height * 16)
        if learned and (height, width) != tuple(latent.shape[3:]):
            upscaled = h3_latent_upscaler.upscale(upscale_model, latent, height, width)
        elif (height, width) != tuple(latent.shape[3:]):
            upscaled = _resized(latent, height, width)
        else:
            upscaled = latent
        del latent

        fresh = upscaled
        mask = torch.ones([1, 1, chunk.rows, height, width])
        if chunk.held:
            spread = float(fresh.float().std()) or 1.0
            gap = float((fresh[:, :, :chunk.held].float() - upscaled_tail.float()).abs().mean()) / spread
            disagreement = gap if disagreement is None else max(disagreement, gap)
            upscaled = fresh.clone()
            upscaled[:, :, :chunk.held] = tail.to(upscaled)
            mask[:, :, :chunk.held] = 0.0

        sound = None
        if audio_vae is not None:
            heard = clips.slice_audio(clip.audio, chunk.first_frame, chunk.first_frame + chunk.frames,
                                      clip.rate)
            sound = h3_references.sound_latent(heard, audio_vae, node)
        _release()

        planned = h3_tiles.auto_tiling(chunk.rows, height, width, text, max_tokens=int(max_tokens))
        split = len(planned.heights) * len(planned.widths) > 1
        guided = positive
        if anchor == "on" or (anchor == "auto" and split):
            guided = h3_tiles.anchored(positive, upscaled, chunk.rows)
        joined = h3_extend.with_held_audio({"samples": upscaled, "noise_mask": mask}, sound)
        samples = comfy.sample.fix_empty_latent_channels(tiled, joined["samples"])
        noise = comfy.sample.prepare_noise(samples, int(seed) + chunk.index)
        sigmas = h3_tiles.start_sigmas(tiled, scheduler, steps, float(strength))
        start_sigma = float(sigmas[0])
        offset = chunk.index * steps

        def stepped(step, x0, x, total, offset=offset):
            callback(offset + step, x0, x, steps * len(chunks))

        result = comfy.sample.sample(
            tiled, noise, steps, float(cfg), sampler_name, scheduler, guided, negative, samples,
            noise_mask=joined["noise_mask"], sigmas=sigmas, callback=stepped,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed) + chunk.index,
        )
        video = h3_extend.split({"samples": result})[0].cpu()
        del result, samples, noise, joined

        refined.append(video[:, :, chunk.held:chunk.kept])
        kept_rows += chunk.kept - chunk.held
        if not chunk.last:
            end = chunk.clips * h3_extend.CLIP_TOKENS
            share = chunks[chunk.index + 1].held
            tail = video[:, :, end - share:end]
            upscaled_tail = fresh[:, :, end - share:end]
        across = len(planned.heights) * len(planned.widths)
        last_frame = min(total_frames, chunk.first_frame + chunk.frames) - 1
        held_frames = chunk.held // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES
        lines.append(
            f"chunk {chunk.index + 1} of {len(chunks)}: frames {chunk.first_frame} to {last_frame}, "
            f"{held_frames} held, {len(planned.rows)} window(s) x {across} tile(s), "
            f"{time.monotonic() - began:.0f} s"
        )
        logger.info("%s: %s", node, lines[-1])

    _release()
    decode_until(total_rows, total_rows)
    whole = torch.cat(refined, dim=2)
    return Result(whole, written, size, source, chunks, start_sigma or float(strength), disagreement,
                  lines, snap_chunk(chunk_frames), snap_overlap(overlap_frames, chunk_frames),
                  int(max_tokens))
