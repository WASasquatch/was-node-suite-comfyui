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
    "MOST_AUDIOS",
    "MOST_IMAGES",
    "MOST_VIDEOS",
    "Picture",
    "QWEN_STRIDE",
    "REF_AUDIOS",
    "REF_IMAGES",
    "REF_VIDEOS",
    "References",
    "Sound",
    "Video",
    "assemble",
    "audio_latent",
    "audio_name",
    "build",
    "clip_items",
    "collect",
    "encode_picture",
    "encode_sound",
    "encode_video",
    "image_canvas",
    "image_name",
    "room",
    "sound_latent",
    "soundtrack_name",
    "video_name",
]

#: Reference slots offered for each kind.
REF_IMAGES = 9
REF_VIDEOS = 3
REF_AUDIOS = 3

#: Most references of each kind one prompt carries.
MOST_IMAGES = 9
MOST_VIDEOS = 3
MOST_AUDIOS = 3

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


class Picture(NamedTuple):
    """One reference picture, encoded.

    Attributes:
        item: What the text encoder is shown.
        block: The ``minimax_refs`` block.
        size: The size it was encoded at, as ``832x480``.
    """

    item: dict
    block: dict
    size: str


class Video(NamedTuple):
    """One reference clip, encoded, with its soundtrack where it has one.

    Attributes:
        items: What the text encoder is shown, the soundtrack's entry ahead of the clip's.
        block: The ``minimax_refs`` block.
        text: The frames and size it was encoded at, as ``124 frames at 832x480``.
        seconds: How long its soundtrack runs, or ``None`` for a silent clip.
    """

    items: list
    block: dict
    text: str
    seconds: float | None


class Sound(NamedTuple):
    """One reference sound, encoded.

    Attributes:
        item: What the text encoder is shown.
        block: The ``minimax_refs`` block.
        seconds: How long it runs.
    """

    item: dict
    block: dict
    seconds: float


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


def audio_latent(audio_vae, audio: dict):
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
    # Whole latent steps, so the encoder trims nothing off the start.
    hop = max(1, wanted // h3_extend.AUDIO_LATENT_FPS)
    short = (-int(waveform.shape[-1])) % hop
    if short:
        import torch

        waveform = torch.nn.functional.pad(waveform, (0, short))
    latent = audio_vae.encode(waveform.movedim(1, -1))
    return latent, int(latent.shape[-1])


def sound_latent(audio, audio_vae, node: str):
    """The H3 audio latent to hold: a latent as it came, a sound encoded, or None.

    Args:
        audio: An audio latent, a ComfyUI sound, or None.
        audio_vae: The H3 audio VAE, or None.
        node: The node asking, for the message.

    Returns:
        A latent dictionary, or None for silence.

    Raises:
        ValueError: A sound came with no audio_vae.
    """
    if not (isinstance(audio, dict) and "waveform" in audio):
        return audio
    if audio.get("waveform") is None:
        return None
    if audio_vae is None:
        raise ValueError(
            f"{node} has a sound and no audio_vae to encode it with. Wire the H3 audio VAE "
            "into audio_vae, or unwire the sound"
        )
    latent, _ = audio_latent(audio_vae, audio)
    return {"samples": latent}


def _seconds(audio: dict) -> float:
    """How long a ComfyUI audio runs for."""
    return audio["waveform"].shape[-1] / max(1, int(audio["sample_rate"]))


def encode_picture(vae, picture, canvas_width: int, canvas_height: int,
                   size: str = "match") -> Picture:
    """One reference picture encoded once.

    Args:
        vae: The H3 video VAE.
        picture: A ``[1, H, W, C]`` tensor.
        canvas_width: The clip's width in pixels.
        canvas_height: The clip's height in pixels.
        size: An entry from :data:`IMAGE_SIZES`.

    Returns:
        The encoded picture.
    """
    wide, high = image_canvas(picture.shape[2], picture.shape[1], canvas_width,
                              canvas_height, size)
    resized = fitted_batch(picture[:1], wide, high)
    return Picture(
        {"type": "image", "data": resized},
        {"kind": "image", "latent_h": high // SPATIAL_STRIDE, "latent_w": wide // SPATIAL_STRIDE,
         "latent": vae.encode(resized)},
        f"{wide}x{high}",
    )


def clip_items(clip, sounding: bool) -> list:
    """What the text encoder is shown for one reference clip, its sound's entry first.

    Args:
        clip: ``[T, H, W, C]`` frames fitted to the reference canvas, at 24 fps.
        sounding: Whether the clip carries a soundtrack.

    Returns:
        The entries, in presentation order.
    """
    shown = list(range(0, int(clip.shape[0]), QWEN_STRIDE))
    items = [{"type": "audio"}] if sounding else []
    items.append({"type": "video", "data": clip[shown],
                  "timestamps": [index / 2.0 for index in range(len(shown))]})
    return items


def encode_video(vae, audio_vae, frames, soundtrack, longest: int, named: str) -> Video:
    """One reference clip encoded once, with its soundtrack where it has one.

    Args:
        vae: The H3 video VAE.
        audio_vae: The H3 audio VAE, or ``None`` where the clip is silent.
        frames: A ``[T, H, W, C]`` tensor at 24 fps.
        soundtrack: The clip's sound, or ``None``.
        longest: Frames the longest segment runs for, which the clip is cut to.
        named: The clip's name, for the error.

    Returns:
        The encoded clip.

    Raises:
        ValueError: The clip is under five frames, or carries sound with no audio VAE.
    """
    count = min(int(frames.shape[0]), max(h3_extend.CLIP_LEAD, int(longest)))
    if count < h3_extend.CLIP_LEAD:
        raise ValueError(
            f"{named} holds {frames.shape[0]} frame(s) and a MiniMax H3 reference video "
            f"needs at least {h3_extend.CLIP_LEAD}, about 0.2s at {h3_extend.FPS} fps. Wire a "
            f"longer clip, or use the picture as a reference picture"
        )
    if soundtrack is not None and audio_vae is None:
        raise ValueError(
            f"{named} carries a soundtrack and audio_vae is not wired, so there is nothing to "
            f"encode it with. Wire the H3 audio VAE into audio_vae"
        )
    count = h3_extend.floor_overlap(count)
    wide, high = h3_extend.reference_canvas(frames.shape[2], frames.shape[1])
    clip = fitted_batch(frames[:count], wide, high)
    heard, seconds = None, None
    if soundtrack is not None:
        heard, _ = audio_latent(audio_vae, soundtrack)
        seconds = _seconds(soundtrack)
    items = clip_items(clip, soundtrack is not None)
    return Video(
        items, h3_extend.video_reference(vae.encode(clip), heard),
        f"{count} frames at {wide}x{high}", seconds,
    )


def encode_sound(audio_vae, audio: dict, named: str) -> Sound:
    """One reference sound encoded once.

    Args:
        audio_vae: The H3 audio VAE, or ``None``.
        audio: A ComfyUI audio.
        named: The sound's name, for the error.

    Returns:
        The encoded sound.

    Raises:
        ValueError: No audio VAE arrived.
    """
    if audio_vae is None:
        raise ValueError(
            f"{named} is a reference sound and audio_vae is not wired, so there is nothing to "
            f"encode it with. Wire the H3 audio VAE into audio_vae"
        )
    latent, steps = audio_latent(audio_vae, audio)
    return Sound(
        {"type": "audio"}, {"kind": "audio", "ref_audio_t": steps, "audio_latent": latent},
        _seconds(audio),
    )


def assemble(pictures: list, videos: list, sounds: list, named: str = "the prompt") -> References:
    """The references one prompt carries, tagged in presentation order.

    Args:
        pictures: Encoded pictures, from :func:`encode_picture`.
        videos: Encoded clips, from :func:`encode_video`.
        sounds: Encoded sounds, from :func:`encode_sound`.
        named: What the prompt is called, for the error.

    Returns:
        The references, in presentation order.

    Raises:
        ValueError: More of one kind than a prompt carries.
    """
    for count, most, kind in (
        (len(pictures), MOST_IMAGES, "pictures"),
        (len(videos), MOST_VIDEOS, "clips"),
        (len(sounds), MOST_AUDIOS, "sounds"),
    ):
        if count > most:
            raise ValueError(
                f"{named} references {count} {kind}, and a MiniMax H3 prompt carries at most "
                f"{most}. Take some off, or move them to the segments that need them"
            )
    items, blocks, tags = [], [], []
    for number, picture in enumerate(pictures, start=1):
        items.append(picture.item)
        blocks.append(picture.block)
        tags.append(f"<Picture {number}> {picture.size}")
    heard = 0
    for number, video in enumerate(videos, start=1):
        label = ""
        if video.seconds is not None:
            heard += 1
            label = f" with <Audio {heard}> {video.seconds:.1f}s"
        items.extend(video.items)
        blocks.append(video.block)
        tags.append(f"<Video {number}> {video.text}{label}")
    for sound in sounds:
        heard += 1
        items.append(sound.item)
        blocks.append(sound.block)
        tags.append(f"<Audio {heard}> {sound.seconds:.1f}s")
    return References(items, blocks, tags)


def room(conditioning, kind: str) -> int:
    """How many more references of one kind every entry of a prompt has space for.

    Args:
        conditioning: A conditioning list.
        kind: A ``minimax_refs`` block kind: ``image``, ``video``, ``video_audio`` or ``audio``.

    Returns:
        The fewest free places across the entries, ``0`` when one is full.
    """
    group = "video" if kind == "video_audio" else kind
    most = {"image": MOST_IMAGES, "video": MOST_VIDEOS, "audio": MOST_AUDIOS}[group]
    free = most
    for entry in conditioning or ():
        blocks = entry[1].get("minimax_refs") or ()
        held = sum(1 for block in blocks
                   if ("video" if block.get("kind") == "video_audio" else block.get("kind")) == group)
        free = min(free, most - held)
    return max(0, free)


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
    pictures = [encode_picture(vae, picture, canvas_width, canvas_height, size)
                for picture in images]
    clips = [encode_video(vae, audio_vae, frames, soundtrack, longest, video_name(slot))
             for slot, frames, soundtrack in videos]
    sounds = [encode_sound(audio_vae, audio, audio_name(number))
              for number, audio in enumerate(audios, start=1)]
    return assemble(pictures, clips, sounds)
