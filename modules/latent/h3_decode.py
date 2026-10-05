"""Decoding a joined MiniMax H3 clip one scene at a time, a long scene in overlapping pieces.

Token spans start on multiples of :data:`~.h3_extend.CLIP_TOKENS`. A piece is decoded with
:data:`LEAD_IN` and :data:`LOOKAHEAD` tokens around it, then trimmed.
"""

from __future__ import annotations

import torch

from . import h3_extend

__all__ = ["LEAD_IN", "LOOKAHEAD", "PIECE_CLIPS", "decode_audio", "decode_scene", "scenes"]

#: Clips decoded together at most, inside one scene.
PIECE_CLIPS = 12

#: Tokens of the clip before a piece decoded with it and dropped.
LEAD_IN = h3_extend.CLIP_TOKENS

#: Tokens after a piece decoded with it and dropped.
LOOKAHEAD = h3_extend.TOKEN_LEAD


def scenes(latent: dict) -> list[tuple[int, int]]:
    """The token span of every fresh scene in a joined clip.

    Args:
        latent: An H3 joint latent, from H3 Extend Append or a sampler.

    Returns:
        ``(start, stop)`` token spans in order, covering the whole clip. A clip with no
        record of its scenes is one span, unless its segment ends show where cuts trimmed it.
    """
    video, _ = h3_extend.split(latent)
    total = int(video.shape[2])
    starts = h3_extend.scene_starts(latent)
    if starts is None:
        starts = [0]
        for end in (h3_extend.segment_ends(latent) or [])[:-1]:
            if end % h3_extend.CLIP_FRAMES == 0:
                starts.append(end // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS)
    kept = sorted({start for start in starts
                   if 0 < start < total and start % h3_extend.CLIP_TOKENS == 0})
    edges = [0] + kept + [total]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _frames(vae, video) -> torch.Tensor:
    """One token span decoded, ``(frames, height, width, 3)``."""
    images = vae.decode(video)
    return images.reshape(-1, *images.shape[-3:])


def decode_scene(vae, video, start: int, stop: int, emit, clips: int = PIECE_CLIPS) -> int:
    """Decode one scene a piece at a time, handing each piece's frames on in order.

    Args:
        vae: The H3 video VAE.
        video: The clip's video half, ``[B, 24, T, H, W]``.
        start: The scene's first token, on the clip grid.
        stop: The token after its last.
        emit: Callable taking ``(frames, height, width, 3)`` for each piece.
        clips: Clips decoded together at most.

    Returns:
        Frames handed on.
    """
    step = max(1, int(clips)) * h3_extend.CLIP_TOKENS
    handed = 0
    piece = start
    while piece < stop:
        end = min(piece + step, stop)
        if stop - end < h3_extend.CLIP_TOKENS:
            end = stop
        lead = LEAD_IN if piece > start else 0
        ahead = LOOKAHEAD if end < stop else 0
        frames = _frames(vae, video[:, :, piece - lead:end + ahead])
        first = h3_extend.CLIP_FRAMES if lead else 0
        if ahead:
            frames = frames[first:first + (end - piece) // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES]
        else:
            frames = frames[first:]
        emit(frames)
        handed += int(frames.shape[0])
        piece = end
    return handed


def decode_audio(audio_vae, audio) -> dict:
    """The clip's sound decoded whole, level-matched as ComfyUI's audio decode leaves it.

    Args:
        audio_vae: The H3 audio VAE.
        audio: The clip's audio half, ``[B, 32, 2, L]``.

    Returns:
        An ``AUDIO`` dict.
    """
    sound = audio_vae.decode(audio).movedim(-1, 1)
    spread = torch.std(sound, dim=[1, 2], keepdim=True) * 5.0
    spread[spread < 1.0] = 1.0
    sound = sound / spread
    rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": sound, "sample_rate": int(rate)}
