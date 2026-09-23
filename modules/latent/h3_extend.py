"""Extending a MiniMax H3 video by sampling a window that carries the finished tail.

Frames run on the model's 17k+5 grid and latent tokens on its 5k+2 grid. A window holds
the clip's last rows at sigma 0.
"""

from __future__ import annotations

__all__ = [
    "AUDIO_LATENT_FPS",
    "AUDIO_RELEASE",
    "CLIP_FRAMES",
    "CLIP_LEAD",
    "CLIP_TOKENS",
    "REFERENCE_FRAMES",
    "TOKEN_LEAD",
    "FPS",
    "append",
    "CONTINUITY",
    "audio_carry",
    "audio_lands_whole",
    "audio_lead",
    "bands",
    "audio_span",
    "cut_point",
    "empty_like",
    "extension_tokens",
    "frames_for",
    "join",
    "masked_window",
    "reference_canvas",
    "SEGMENT_GAIN",
    "video_reference",
    "snap_clip",
    "snap_extension",
    "snap_overlap",
    "settled",
    "split",
    "tail",
    "tokens_for",
    "window_frames",
]

#: Frames one clip of latent tokens covers, and the lead frame every sequence opens with.
CLIP_FRAMES = 17
CLIP_LEAD = 5

#: Latent tokens per clip, and the tokens a sequence opens with.
CLIP_TOKENS = 5
TOKEN_LEAD = 2

#: Video frames a second, and audio latent frames a second.
FPS = 24
AUDIO_LATENT_FPS = 40


def audio_lands_whole(frames: int) -> bool:
    """Whether an overlap covers a whole number of audio latent steps.

    Args:
        frames: A snapped overlap in frames.

    Returns:
        True where ``frames`` maps onto the audio grid exactly.
    """
    return (int(frames) * AUDIO_LATENT_FPS) % FPS == 0


#: Audio latent steps a carried soundtrack is released over at the seam.
AUDIO_RELEASE = 8


def audio_carry(frames: int) -> int:
    """Whole audio latent steps that fit inside an overlap.

    Args:
        frames: A snapped overlap in frames.

    Returns:
        The step count, rounded down so the carry ends at or before the seam.
    """
    return max(0, int(frames)) * AUDIO_LATENT_FPS // FPS


def audio_lead(frames: int) -> float:
    """Seconds between the end of a carried soundtrack and the seam it meets.

    Args:
        frames: A snapped overlap in frames.

    Returns:
        The gap in seconds, 0.0 where the overlap lands on a whole step.
    """
    whole = audio_carry(frames)
    return (max(0, int(frames)) * AUDIO_LATENT_FPS / FPS - whole) / AUDIO_LATENT_FPS


#: How a segment continues from the one before it.
CONTINUITY = ("carry", "refresh", "handoff", "reference", "cut")

#: Fine detail a segment is expected to add, which a refreshed carry is softened below.
SEGMENT_GAIN = 1.15

#: Fewest frames a continuation reference holds, 2.33 seconds on the grid.
REFERENCE_FRAMES = 56

#: Short edge a re-encoded reference is sized to, the largest area it is given, and the
#: multiple both sides take.
REFERENCE_EDGE = 768
REFERENCE_AREA = 768 * 1344
CANVAS_MULTIPLE = 32


def reference_canvas(width: int, height: int, edge: int = REFERENCE_EDGE):
    """A canvas for a re-encoded reference video, never larger than the source.

    Args:
        width: The clip's width in pixels.
        height: The clip's height in pixels.
        edge: Short edge to aim for.

    Returns:
        ``(width, height)``, each a multiple of :data:`CANVAS_MULTIPLE`, covering at most
        :data:`REFERENCE_AREA`.
    """
    width, height = max(1, int(width)), max(1, int(height))
    scale = min(1.0, edge / min(width, height))
    area = width * height * scale * scale
    if area > REFERENCE_AREA:
        scale *= (REFERENCE_AREA / area) ** 0.5
    sides = [max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
             for side in (width, height)]
    return sides[0], sides[1]


def video_reference(latent, audio_latent=None) -> dict:
    """One ``minimax_refs`` block carrying a clip the new segment continues.

    Args:
        latent: A ``[B, 24, T, H, W]`` reference latent.
        audio_latent: Its soundtrack, or ``None``.

    Returns:
        The reference block.
    """
    audio_rows = 0 if audio_latent is None else audio_latent.shape[-1]
    return {
        "kind": "video_audio" if audio_rows else "video",
        "latent_t": latent.shape[2],
        "latent_h": latent.shape[3],
        "latent_w": latent.shape[4],
        "ref_audio_t": audio_rows,
        "latent": latent,
        "audio_latent": audio_latent,
    }


def empty_like(video, audio, frames: int) -> dict:
    """A silent, empty latent shaped like an existing clip but a given length.

    Args:
        video: A clip's video half, for its channels and canvas.
        audio: A clip's audio half, for its channels.
        frames: Frames the new latent covers.

    Returns:
        The empty joint latent.
    """
    import torch

    return join(
        torch.zeros(
            [video.shape[0], video.shape[1], tokens_for(frames),
             video.shape[3], video.shape[4]],
            device=video.device, dtype=video.dtype,
        ),
        torch.zeros(
            [audio.shape[0], audio.shape[1], audio.shape[2], audio_span(frames)],
            device=audio.device, dtype=audio.dtype,
        ),
    )


def bands(rows):
    """The low and high spatial bands of a stretch of latent rows.

    Args:
        rows: Rows shaped ``[B, 24, t, H, W]``.

    Returns:
        ``(low, high)``, summing back to ``rows``.
    """
    import torch.nn.functional as functional

    low = functional.avg_pool3d(rows, (1, 3, 3), stride=1, padding=(0, 1, 1),
                                count_include_pad=False)
    return low, rows - low


def settled(video, rows: int, strength: float):
    """Carried rows brought back to the scale and fine detail of the clip's opening.

    Args:
        video: The clip's video half, whose first ``rows`` rows are the reference.
        rows: How many rows the carry takes from the end.
        strength: How much of the correction to apply, from 0.0 to 1.0.

    Returns:
        ``(carried, scale, gain)``, the corrected rows and the two factors applied.
    """
    taken = min(int(rows), video.shape[2])
    carried = video[:, :, -taken:].float()
    anchor = video[:, :, :taken].float()
    amount = max(0.0, min(1.0, float(strength)))
    scale, gain = 1.0, 1.0
    if amount <= 0.0 or taken <= 0:
        return video[:, :, -taken:].clone(), scale, gain

    target, current = float(anchor.std()), float(carried.std())
    if target > 0.0 and current > 0.0:
        scale = 1.0 + (target / current - 1.0) * amount
        carried = carried * scale

    low, high = bands(carried)
    _, anchor_high = bands(anchor)
    energy, wanted = float(high.pow(2).mean()), float(anchor_high.pow(2).mean())
    if energy > 0.0 and wanted > 0.0:
        # Clamped at 1.0, which softens and never sharpens.
        gain = max(0.5, min(1.0, (wanted / energy) ** 0.5))
        gain = 1.0 + (gain - 1.0) * amount
        carried = low + high * gain
    return carried.to(video.dtype), scale, gain


def masked_window(video_tail, audio_tail, video_tokens: int, audio_length: int,
                  audio_steps: int, release: int = AUDIO_RELEASE):
    """A window holding the finished tail, masked so the sampler leaves it alone.

    Args:
        video_tail: The carried video rows, shaped ``[B, 24, t, H, W]``.
        audio_tail: The carried audio, shaped ``[B, 32, 2, l]``.
        video_tokens: Rows the whole window holds.
        audio_length: Audio latent frames the whole window holds.
        audio_steps: Audio latent steps of the tail to hold, 0 to hold none.
        release: Steps at the end of the carry the mask opens over.

    Returns:
        ``(latent, mask, held)``, a joint latent, a joint mask and the audio steps held.
        The mask is 1 where the pass generates and 0 over the carried rows.
    """
    import math

    import torch

    video = torch.zeros(
        [video_tail.shape[0], video_tail.shape[1], video_tokens,
         video_tail.shape[3], video_tail.shape[4]],
        device=video_tail.device, dtype=video_tail.dtype,
    )
    audio = torch.zeros(
        [audio_tail.shape[0], audio_tail.shape[1], audio_tail.shape[2], audio_length],
        device=audio_tail.device, dtype=audio_tail.dtype,
    )
    video_mask = torch.ones(
        [video.shape[0], 1, video_tokens, video.shape[3], video.shape[4]],
        device=video.device, dtype=torch.float32,
    )
    audio_mask = torch.ones(
        [audio.shape[0], 1, audio.shape[2], audio_length],
        device=audio.device, dtype=torch.float32,
    )

    rows = min(video_tail.shape[2], video_tokens)
    video[:, :, :rows] = video_tail[:, :, -rows:]
    video_mask[:, :, :rows] = 0.0

    held = max(0, min(int(audio_steps), audio_tail.shape[-1], audio_length))
    audio[..., :held] = audio_tail[..., -held:] if held else audio_tail[..., :0]
    # The mask opens over a half cosine.
    opened = max(0, min(int(release), held))
    audio_mask[..., :held - opened] = 0.0
    for step in range(opened):
        audio_mask[..., held - opened + step] = 0.5 - 0.5 * math.cos(
            math.pi * (step + 1) / (opened + 1)
        )
    return join(video, audio), join(video_mask, audio_mask), held


def tokens_for(frames: int) -> int:
    """Latent tokens a frame count occupies.

    Args:
        frames: A frame count on the ``17k + 5`` grid.

    Returns:
        The token count, ``5k + 2``.
    """
    if frames <= CLIP_LEAD:
        return TOKEN_LEAD
    return ((frames - CLIP_LEAD) // CLIP_FRAMES) * CLIP_TOKENS + TOKEN_LEAD


def frames_for(tokens: int) -> int:
    """Frames a token count covers, the inverse of :func:`tokens_for`.

    Args:
        tokens: A token count on the ``5k + 2`` grid.

    Returns:
        The frame count.
    """
    if tokens <= TOKEN_LEAD:
        return CLIP_LEAD
    return ((tokens - TOKEN_LEAD) // CLIP_TOKENS) * CLIP_FRAMES + CLIP_LEAD


def cut_point(tokens: int) -> tuple:
    """Where a clip ends ahead of a cut, on the last whole clip it holds.

    Args:
        tokens: Rows the clip holds.

    Returns:
        ``(rows, frames)`` kept. A clip shorter than one whole clip is kept entire.
    """
    whole = int(tokens) - int(tokens) % CLIP_TOKENS
    if whole <= 0:
        return int(tokens), frames_for(tokens)
    return whole, whole // CLIP_TOKENS * CLIP_FRAMES


def snap_overlap(frames: int) -> int:
    """The nearest guide length at or below a frame count, never under the lead.

    Args:
        frames: Frames asked for.

    Returns:
        A count on the ``17k + 5`` grid, at least :data:`CLIP_LEAD`.
    """
    wanted = max(CLIP_LEAD, int(frames))
    while wanted % CLIP_FRAMES != CLIP_LEAD % CLIP_FRAMES:
        wanted -= 1
    return max(CLIP_LEAD, wanted)


def snap_clip(frames: int) -> int:
    """The nearest clip length at or above a frame count.

    Args:
        frames: Frames asked for.

    Returns:
        A count on the ``17k + 5`` grid, at least :data:`CLIP_LEAD`.
    """
    wanted = max(CLIP_LEAD, int(frames))
    while wanted % CLIP_FRAMES != CLIP_LEAD % CLIP_FRAMES:
        wanted += 1
    return wanted


def snap_extension(frames: int) -> int:
    """The nearest extension at or below a frame count, never under one clip.

    Args:
        frames: Frames asked for.

    Returns:
        A positive multiple of :data:`CLIP_FRAMES`.
    """
    clips = max(1, int(frames) // CLIP_FRAMES)
    return clips * CLIP_FRAMES


def window_frames(overlap: int, extension: int) -> int:
    """Frames one continuation pass samples, the overlap plus the extension.

    Args:
        overlap: A snapped overlap.
        extension: A snapped extension.

    Returns:
        The window's frame count, which lands back on the ``17k + 5`` grid.
    """
    return snap_overlap(overlap) + snap_extension(extension)


def extension_tokens(extension: int) -> int:
    """Tokens a snapped extension adds.

    Args:
        extension: A snapped extension.

    Returns:
        The token count, five per clip.
    """
    return (snap_extension(extension) // CLIP_FRAMES) * CLIP_TOKENS


def audio_span(frames: int) -> int:
    """Audio latent frames a video frame count covers.

    Args:
        frames: A video frame count.

    Returns:
        The audio latent length, rounded as the empty latent rounds it.
    """
    return round(int(frames) / FPS * AUDIO_LATENT_FPS)


def split(latent: dict):
    """The video and audio halves of an H3 joint latent.

    Args:
        latent: A latent whose ``samples`` is a nested pair.

    Returns:
        ``(video, audio)``, video shaped ``[B, 24, T, H, W]`` and audio ``[B, 32, 2, L]``.

    Raises:
        ValueError: The latent is not a MiniMax H3 joint latent.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    tensors = getattr(samples, "tensors", None)
    if tensors is None or len(tensors) != 2:
        raise ValueError(
            "this is not a MiniMax H3 video and audio latent. Build one with Empty MiniMax "
            "H3 AV Latent, or take the output of an H3 sampler"
        )
    video, audio = tensors[0], tensors[1]
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            f"the video half is shaped {tuple(video.shape)}, and a MiniMax H3 latent is "
            f"[batch, 24, tokens, height, width]"
        )
    return video, audio


def join(video, audio) -> dict:
    """One H3 joint latent from a video and an audio half.

    Args:
        video: A ``[B, 24, T, H, W]`` tensor.
        audio: A ``[B, 32, 2, L]`` tensor.

    Returns:
        A latent carrying both under ``samples``.
    """
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def tail(latent: dict, overlap: int):
    """The last stretch of a finished latent, for the next pass to be guided by.

    Args:
        latent: The finished joint latent.
        overlap: A snapped overlap in frames.

    Returns:
        ``(video_tail, audio_tail)``, each a contiguous copy.

    Raises:
        ValueError: The latent is shorter than the overlap.
    """
    video, audio = split(latent)
    tokens = tokens_for(snap_overlap(overlap))
    if video.shape[2] < tokens:
        raise ValueError(
            f"a {snap_overlap(overlap)} frame guide needs {tokens} of the clip's "
            f"{video.shape[2]} tokens. Ask for a shorter overlap, or extend a longer clip"
        )
    span = min(audio.shape[-1], audio_span(snap_overlap(overlap)))
    return video[:, :, -tokens:].clone(), audio[..., -span:].clone()


def append(done: dict, sampled: dict, overlap: int) -> dict:
    """Join a sampled window onto the clip it continues, dropping the carried head.

    Args:
        done: The clip so far.
        sampled: The window, whose opening rows are the carried tail, or a fresh scene
            where ``overlap`` is ``0``.
        overlap: The overlap in frames, ``0`` for a cut.

    Returns:
        The joined clip. A cut ends the clip at :func:`cut_point` before the new scene.
    """
    import torch

    done_video, done_audio = split(done)
    new_video, new_audio = split(sampled)
    if int(overlap) <= 0:
        rows, span = 0, 0
        # The new scene opens on a clip boundary, where the decoder starts a clip.
        kept, frames = cut_point(done_video.shape[2])
        done_video = done_video[:, :, :kept]
        done_audio = done_audio[..., :min(done_audio.shape[-1], audio_span(frames))]
    else:
        rows = min(tokens_for(snap_overlap(overlap)), done_video.shape[2], new_video.shape[2])
        span = min(audio_span(snap_overlap(overlap)), done_audio.shape[-1], new_audio.shape[-1])
    return join(
        torch.cat([done_video, new_video[:, :, rows:]], dim=2),
        torch.cat([done_audio, new_audio[..., span:]], dim=-1),
    )
