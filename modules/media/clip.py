"""Taking a clip apart into frames and putting it back together, for the nodes that work along its motion.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``; audio is a ComfyUI ``AUDIO``
dict, ``waveform`` shaped ``(batch, channels, samples)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .. import log

__all__ = [
    "Clip",
    "DEFAULT_RATE",
    "compare",
    "motion_for",
    "open_clip",
    "progress",
    "rebuild",
    "slice_audio",
    "stretch",
]

logger = log.get_logger("media.clip")

#: Frames per second given to a batch of images, which carries no rate of its own.
DEFAULT_RATE = Fraction(24)


@dataclass
class Clip:
    """One clip, taken apart.

    Attributes:
        frames: ``(frames, height, width, 3)``.
        alpha: ``(frames, height, width)`` or ``None``.
        audio: The ``AUDIO`` dict, or ``None``.
        rate: Frames per second.
        metadata: The container metadata, or ``None``.
        bit_depth: Bits per channel the clip is written with.
        color_space: Colour space the clip is written with.
    """

    frames: object
    alpha: object
    audio: object
    rate: Fraction
    metadata: object
    bit_depth: int
    color_space: str


def open_clip(video, name: str) -> Clip:
    """A ``VIDEO`` or an ``IMAGE`` batch, taken apart.

    Args:
        video: A ``VIDEO`` object, or an ``IMAGE`` tensor.
        name: The node reading it, for the message.

    Returns:
        The :class:`Clip`. An ``IMAGE`` batch is given :data:`DEFAULT_RATE` and no audio.

    Raises:
        ValueError: The clip holds no frames.
    """
    if hasattr(video, "get_components"):
        parts = video.get_components()
        clip = Clip(
            frames=parts.images,
            alpha=parts.alpha,
            audio=parts.audio,
            rate=Fraction(parts.frame_rate),
            metadata=parts.metadata,
            bit_depth=int(getattr(video, "get_bit_depth", lambda: 8)()),
            color_space=str(getattr(video, "get_color_space", lambda: "sRGB")()),
        )
    else:
        clip = Clip(video, None, None, DEFAULT_RATE, None, 8, "sRGB")
    if getattr(clip.frames, "ndim", 0) != 4 or int(clip.frames.shape[0]) < 1:
        raise ValueError(f"{name} needs a clip with at least one frame.")
    return clip


def rebuild(clip: Clip, frames, alpha=None, audio="same", rate=None):
    """A ``VIDEO`` from new frames, carrying the clip's audio, rate and format.

    Args:
        clip: The clip the frames came from.
        frames: The frames to carry.
        alpha: An alpha batch, or ``None``.
        audio: An ``AUDIO`` dict, ``None``, or ``"same"`` for the clip's own.
        rate: Frames per second, or ``None`` for the clip's own.

    Returns:
        A ``VIDEO`` object.
    """
    from comfy_api.latest import InputImpl, Types

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=frames,
            frame_rate=Fraction(rate) if rate is not None else clip.rate,
            audio=clip.audio if isinstance(audio, str) else audio,
            metadata=clip.metadata,
            alpha=alpha,
        ),
        bit_depth=clip.bit_depth,
        color_space=clip.color_space,
    )


def slice_audio(audio, start: int, stop: int, rate) -> dict | None:
    """The stretch of a clip's audio that plays under frames ``start`` to ``stop``.

    Args:
        audio: An ``AUDIO`` dict, or ``None``.
        start: First frame, counting from 0.
        stop: Frame after the last.
        rate: Frames per second.

    Returns:
        A new ``AUDIO`` dict, or ``None`` where there is no audio.
    """
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 44100))
    per_frame = sample_rate / float(rate)
    first = max(0, int(round(start * per_frame)))
    last = max(first, min(int(waveform.shape[-1]), int(round(stop * per_frame))))
    return {"waveform": waveform[..., first:last].clone(), "sample_rate": sample_rate}


def motion_for(clip: Clip, motion, name: str, device, step=None):
    """The motion a node works along: the one it was given, checked, or one measured here.

    Args:
        clip: The clip.
        motion: A ``MOTION`` from Video Motion, or ``None``.
        name: The node, for the message.
        device: Where a measurement runs.
        step: Optional progress callable, called once per frame pair measured.

    Returns:
        A :class:`modules.image.motion.Motion`.

    Raises:
        ValueError: The motion given was measured from a different clip.
    """
    from ..image import motion as motion_field

    count, height, width = (int(v) for v in clip.frames.shape[:3])
    if motion is not None:
        motion.check(count, height, width, name)
        if step is not None:
            step(max(count - 1, 0))
        return motion
    return motion_field.measure(clip.frames, motion_field.MOTION_SIDE, device, step)


def progress(total: int):
    """A progress callable for one node's run, stopping the run when it is cancelled.

    Args:
        total: Steps the run takes.

    Returns:
        A callable taking a step count.
    """
    import comfy.model_management
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def advance(steps=1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        bar.update(steps)

    return advance


def compare(before, after, prefix: str) -> dict:
    """Both sides of a node's clip written for the two-video player.

    Args:
        before: The ``VIDEO`` the node received.
        after: The ``VIDEO`` it answered.
        prefix: What the temp files are named under.

    Returns:
        The ``ui`` mapping the player reads, ``a_video`` and ``b_video``.
    """
    from .temp_video import to_temp

    sides = {"a_video": [], "b_video": []}
    for key, video, side in (("a_video", before, "before"), ("b_video", after, "after")):
        written = to_temp(video, f"{prefix}.{side}")
        if written is not None:
            sides[key].append(written)
    return sides


def stretch(values, count: int) -> list[float]:
    """One value per frame, stretched linearly from however many were given.

    Args:
        values: Numbers spread evenly across the clip, one or more.
        count: Frames in the clip.

    Returns:
        ``count`` numbers, interpolated between the given ones.
    """
    points = [float(value) for value in values]
    if not points:
        return [0.0] * count
    if len(points) == 1 or count <= 1:
        return [points[0]] * count
    out = []
    for index in range(count):
        at = index * (len(points) - 1) / (count - 1)
        low = min(int(at), len(points) - 2)
        share = at - low
        out.append(points[low] * (1.0 - share) + points[low + 1] * share)
    return out
