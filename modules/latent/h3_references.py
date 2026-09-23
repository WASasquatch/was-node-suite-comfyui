"""Reference pictures, videos and audio for a MiniMax H3 ``ref2va`` prompt.

Tags count from 1 per kind: pictures, then videos, each soundtrack's ``<Audio j>`` just
before its ``<Video k>``, then standalone audio. Images are ``[B, H, W, C]``.
"""

from __future__ import annotations

from typing import NamedTuple

from . import h3_extend
from .h3_conditioning import fitted_batch

__all__ = [
    "IMAGE_EDGE",
    "IMAGE_SIZES",
    "QWEN_STRIDE",
    "REF_AUDIOS",
    "REF_IMAGES",
    "REF_VIDEOS",
    "References",
    "audio_name",
    "build",
    "collect",
    "image_canvas",
    "image_name",
    "soundtrack_name",
    "video_name",
]

#: Reference slots offered for each kind.
REF_IMAGES = 9
REF_VIDEOS = 3
REF_AUDIOS = 3

#: How a reference picture is sized: to the canvas's area, to a longest side in pixels, or
#: to :data:`IMAGE_EDGE` on its short side.
IMAGE_SIZES = ("match", "256", "512", "768", "1024", "1536", "2048", "max")

#: Short edge a ``max`` reference picture is brought down to.
IMAGE_EDGE = 2048

#: Fewest and most pixels a reference picture has on a side.
SMALLEST_SIDE = 256
LARGEST_SIDE = 5760

#: Frames between the reference video frames the text encoder reads, 2 a second.
QWEN_STRIDE = h3_extend.FPS // 2

#: Side multiple a reference picture is snapped to.
CANVAS_MULTIPLE = 32

#: Pixels per latent cell on each spatial axis.
SPATIAL_STRIDE = 16

#: Sample rate an audio VAE that names none encodes at.
AUDIO_RATE = 32000


class References(NamedTuple):
    """The references one prompt carries.

    Attributes:
        items: What the text encoder is shown, in presentation order.
        blocks: The ``minimax_refs`` blocks the model reads, in the same order.
        tags: One line per reference, naming its prompt tag and what was encoded.
    """

    items: list
    blocks: list
    tags: list


def image_name(slot: int) -> str:
    """The input name of a reference picture.

    Args:
        slot: Slot number, from 1.

    Returns:
        The input name.
    """
    return f"ref_image_{slot}"


def video_name(slot: int) -> str:
    """The input name of a reference video.

    Args:
        slot: Slot number, from 1.

    Returns:
        The input name.
    """
    return f"ref_video_{slot}"


def soundtrack_name(slot: int) -> str:
    """The input name of a reference video's soundtrack.

    Args:
        slot: Slot number, from 1, the same as its video's.

    Returns:
        The input name.
    """
    return f"ref_video_audio_{slot}"


def audio_name(slot: int) -> str:
    """The input name of a standalone reference audio.

    Args:
        slot: Slot number, from 1.

    Returns:
        The input name.
    """
    return f"ref_audio_{slot}"


def collect(values: dict) -> tuple[list, list, list]:
    """The wired references, in slot order.

    Args:
        values: A node's inputs, keyed by name.

    Returns:
        ``(images, videos, audios)``. Each video is ``(slot, frames, soundtrack)``, the
        soundtrack ``None`` where none is wired.

    Raises:
        ValueError: A soundtrack is wired where its video is not.
    """
    images = [values[image_name(slot)] for slot in range(1, REF_IMAGES + 1)
              if values.get(image_name(slot)) is not None]
    videos = []
    for slot in range(1, REF_VIDEOS + 1):
        frames, soundtrack = values.get(video_name(slot)), values.get(soundtrack_name(slot))
        if frames is None and soundtrack is not None:
            raise ValueError(
                f"{soundtrack_name(slot)} is wired and {video_name(slot)} is not, and a "
                f"soundtrack belongs to the video in the same slot. Wire the video, or move "
                f"the audio to a ref_audio input"
            )
        if frames is not None:
            videos.append((slot, frames, soundtrack))
    audios = [values[audio_name(slot)] for slot in range(1, REF_AUDIOS + 1)
              if values.get(audio_name(slot)) is not None]
    return images, videos, audios


def _snapped(value: float) -> int:
    """A side rounded to :data:`CANVAS_MULTIPLE`, never below one multiple."""
    return max(CANVAS_MULTIPLE, int(round(float(value) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def image_canvas(width: int, height: int, canvas_width: int, canvas_height: int,
                 size: str) -> tuple[int, int]:
    """The size a reference picture is encoded at.

    Args:
        width: The picture's width in pixels.
        height: The picture's height in pixels.
        canvas_width: The clip's width in pixels.
        canvas_height: The clip's height in pixels.
        size: An entry from :data:`IMAGE_SIZES`.

    Returns:
        ``(width, height)``, each a multiple of :data:`CANVAS_MULTIPLE`, scaled down only,
        except where a side would fall under :data:`SMALLEST_SIDE`, and never past
        :data:`LARGEST_SIDE`.
    """
    width, height = max(1, int(width)), max(1, int(height))
    if size == "max":
        scale = min(1.0, IMAGE_EDGE / min(width, height))
    elif size == "match":
        scale = min(1.0, (canvas_width * canvas_height / (width * height)) ** 0.5)
    else:
        scale = min(1.0, int(size) / max(width, height))
    scale = max(scale, SMALLEST_SIDE / min(width, height))
    scale = min(scale, LARGEST_SIDE / max(width, height))
    return _snapped(width * scale), _snapped(height * scale)


def _audio_latent(audio_vae, audio: dict):
    """One soundtrack encoded by the H3 audio VAE.

    Args:
        audio_vae: The H3 audio VAE.
        audio: A ComfyUI audio, ``{"waveform": [B, C, L], "sample_rate": int}``.

    Returns:
        ``(latent, steps)``, the latent shaped ``[1, 32, 2, steps]``.
    """
    import torchaudio

    waveform = audio["waveform"][:1]
    rate = int(audio["sample_rate"])
    wanted = int(getattr(audio_vae, "audio_sample_rate", AUDIO_RATE))
    if rate != wanted:
        waveform = torchaudio.functional.resample(waveform, rate, wanted)
    latent = audio_vae.encode(waveform.movedim(1, -1))
    return latent, int(latent.shape[-1])


def _seconds(audio: dict) -> float:
    """How long a ComfyUI audio runs for."""
    return audio["waveform"].shape[-1] / max(1, int(audio["sample_rate"]))


def build(vae, audio_vae, images: list, videos: list, audios: list, canvas_width: int,
          canvas_height: int, longest: int, size: str = "match") -> References:
    """Every reference encoded once, for each segment's prompt to carry.

    Args:
        vae: The H3 video VAE.
        audio_vae: The H3 audio VAE, or ``None`` where no audio is wired.
        images: Reference pictures, from :func:`collect`.
        videos: Reference videos, from :func:`collect`.
        audios: Standalone reference audio, from :func:`collect`.
        canvas_width: The clip's width in pixels.
        canvas_height: The clip's height in pixels.
        longest: Frames the longest segment runs for, which no video runs past.
        size: An entry from :data:`IMAGE_SIZES`.

    Returns:
        The references, in presentation order.

    Raises:
        ValueError: Audio is wired without an audio VAE, or a video is under 5 frames.
    """
    if audio_vae is None and (audios or any(sound is not None for _, _, sound in videos)):
        raise ValueError(
            "a reference audio or video soundtrack is wired and audio_vae is not, so "
            "there is nothing to encode it with. Wire the H3 audio VAE into audio_vae"
        )

    items, blocks, tags = [], [], []
    for number, picture in enumerate(images, start=1):
        wide, high = image_canvas(picture.shape[2], picture.shape[1], canvas_width,
                                  canvas_height, size)
        resized = fitted_batch(picture[:1], wide, high)
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": high // SPATIAL_STRIDE,
                       "latent_w": wide // SPATIAL_STRIDE, "latent": vae.encode(resized)})
        tags.append(f"<Picture {number}> {wide}x{high}")

    heard = 0
    for number, (slot, frames, soundtrack) in enumerate(videos, start=1):
        count = min(int(frames.shape[0]), max(h3_extend.CLIP_LEAD, int(longest)))
        if count < h3_extend.CLIP_LEAD:
            raise ValueError(
                f"{video_name(slot)} holds {frames.shape[0]} frame(s) and a MiniMax H3 "
                f"reference video needs at least {h3_extend.CLIP_LEAD}, about 0.2s at "
                f"{h3_extend.FPS} fps. Wire a longer clip, or wire the picture to a "
                f"ref_image input"
            )
        count = h3_extend.snap_overlap(count)
        wide, high = h3_extend.reference_canvas(frames.shape[2], frames.shape[1])
        clip = fitted_batch(frames[:count], wide, high)
        audio_latent, steps = None, 0
        label = ""
        if soundtrack is not None:
            heard += 1
            items.append({"type": "audio"})
            audio_latent, steps = _audio_latent(audio_vae, soundtrack)
            label = f" with <Audio {heard}> {_seconds(soundtrack):.1f}s"
        shown = list(range(0, count, QWEN_STRIDE))
        items.append({"type": "video", "data": clip[shown],
                      "timestamps": [index / 2.0 for index in range(len(shown))]})
        blocks.append(h3_extend.video_reference(vae.encode(clip), audio_latent))
        tags.append(f"<Video {number}> {count} frames at {wide}x{high}{label}")

    for audio in audios:
        heard += 1
        latent, steps = _audio_latent(audio_vae, audio)
        items.append({"type": "audio"})
        blocks.append({"kind": "audio", "ref_audio_t": steps, "audio_latent": latent})
        tags.append(f"<Audio {heard}> {_seconds(audio):.1f}s")
    return References(items, blocks, tags)
