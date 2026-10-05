"""Writing a VIDEO to the temp folder so a node interface can play it back.

Files land in ComfyUI's temp directory, which is cleared on restart. Every side is written
as mp4, no longer than :data:`PREVIEW_SIDE` on its long side.
"""

from __future__ import annotations

import contextlib
import os

from .. import log
from ..util import sandbox

logger = log.get_logger("media.temp_video")

#: Container every side is written in.
CONTAINER = "mp4"

#: Codecs tried in order, the first that encodes winning.
CODECS = ("auto", "h264")

#: Longest side, in pixels, a side is played back at.
PREVIEW_SIDE = 2560

#: Encoder settings of a side made smaller for playback.
PREVIEW_OPTIONS = {"crf": "20", "preset": "veryfast"}


def preview_size(width: int, height: int) -> tuple[int, int]:
    """The size a clip is played back at: no longer than :data:`PREVIEW_SIDE`, even sides.

    Args:
        width: The clip's width.
        height: Its height.

    Returns:
        ``(width, height)``, the clip's own size rounded to even sides where it already fits.
    """
    scale = min(1.0, PREVIEW_SIDE / max(int(width), int(height), 1))
    return (
        max(2, 2 * int(round(int(width) * scale / 2))),
        max(2, 2 * int(round(int(height) * scale / 2))),
    )


def shrink(frame, size: tuple[int, int]):
    """One frame at a playback size, as the encoder takes it.

    Args:
        frame: ``(height, width, channels)`` or ``(1, channels, height, width)`` in ``[0, 1]``,
            on any device.
        size: ``(width, height)``.

    Returns:
        A ``(height, width, 3)`` uint8 numpy array.
    """
    import torch
    import torch.nn.functional as F

    if frame.ndim == 3:
        frame = frame[..., :3].permute(2, 0, 1).unsqueeze(0)
    width, height = size
    frame = frame[:, :3].float()
    if tuple(frame.shape[-2:]) != (height, width):
        frame = F.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    codes = frame[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0)
    return codes.contiguous().cpu().numpy()


def _smaller(video, target: str, size: tuple[int, int]) -> None:
    """Encode a VIDEO to ``target`` at ``size``, one frame at a time, without its audio.

    Args:
        video: A ``VIDEO`` object.
        target: The mp4 file written.
        size: ``(width, height)``, both even.
    """
    import av
    from av.video.reformatter import Interpolation
    from comfy_api.latest import InputImpl

    from ..image import scratch
    from .video import Encoder

    width, height = size
    lazy = getattr(video, "lazy_frames", None)
    if lazy is not None:
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        with Encoder(target, "h264", width, height, video.get_frame_rate(),
                     options=PREVIEW_OPTIONS, color_space="sRGB") as out:
            for frame in lazy():
                out.write_rgb(shrink(frame.to(device), size))
        return
    if isinstance(video, InputImpl.VideoFromFile):
        with av.open(video.get_stream_source()) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or 24
            with Encoder(target, "h264", width, height, rate, options=PREVIEW_OPTIONS,
                         color_space="sRGB") as out:
                for frame in container.decode(stream):
                    out.write_rgb(frame.reformat(width=width, height=height, format="rgb24",
                                                 interpolation=Interpolation.AREA).to_ndarray())
        return

    import comfy.model_management

    device = comfy.model_management.get_torch_device()
    parts = video.get_components()
    with Encoder(target, "h264", width, height, parts.frame_rate, options=PREVIEW_OPTIONS,
                 color_space="sRGB") as out:
        for index in range(int(parts.images.shape[0])):
            out.write_rgb(shrink(parts.images[index].to(device), size))
            scratch.trim(parts.images[index])


def even_sided(video):
    """A video whose frames have an even width and height.

    Args:
        video: A ``VIDEO`` object.

    Returns:
        The video as it stands where both sides are already even, otherwise a copy with its
        last row and column repeated.

    Raises:
        MemoryError: Neither free memory nor a scratch drive can hold the copy.
    """
    width, height = video.get_dimensions()
    if width % 2 == 0 and height % 2 == 0:
        return video

    from comfy_api.latest import InputImpl, Types

    from ..image import scratch

    def squared(frames, rows, columns):
        if frames is None:
            return None
        count, tall, wide = (int(side) for side in frames.shape[:3])
        padded = scratch.allocate(
            (count, tall + rows, wide + columns) + tuple(frames.shape[3:]), frames.dtype
        )
        padded[:, :tall, :wide].copy_(frames)
        if rows:
            padded[:, tall, :wide].copy_(frames[:, -1])
        if columns:
            padded[:, :, wide].copy_(padded[:, :, wide - 1])
        return padded

    parts = video.get_components()
    rows, columns = height % 2, width % 2
    logger.info("padding a %dx%d clip to an even size for playback", width, height)
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=squared(parts.images, rows, columns),
            frame_rate=parts.frame_rate,
            audio=parts.audio,
            metadata=parts.metadata,
            alpha=squared(parts.alpha, rows, columns),
        ),
        bit_depth=video.get_bit_depth(),
        color_space=video.get_color_space(),
    )


def to_temp(video, prefix: str) -> dict | None:
    """Write a VIDEO to the temp folder under a prefix.

    Args:
        video: A ``VIDEO`` object.
        prefix: What to name the file, such as ``"was.compare.a"``.

    Returns:
        ``filename``, ``subfolder`` and ``type`` for the written file, or None where none of
        :data:`CODECS` encoded it.
    """
    try:
        import folder_paths
        from comfy_api.latest._util.video_types import VideoCodec, VideoContainer

        width, height = video.get_dimensions()
        size = None
        if max(width, height) > PREVIEW_SIDE:
            size = preview_size(width, height)
        else:
            # An odd side is refused by the yuv420p encoders every codec here uses.
            video = even_sided(video)
        folder, name, counter, subfolder, _ = sandbox.save_image_path(
            prefix, folder_paths.get_temp_directory(), width, height
        )
        file = f"{name}_{counter:05}_.{VideoContainer.get_extension(CONTAINER)}"
        target = os.path.join(folder, file)
    except Exception as error:  # noqa: BLE001
        logger.warning("a video could not be prepared for playback (%s)", error)
        return None

    if size is not None:
        try:
            _smaller(video, target, size)
            return {"filename": file, "subfolder": subfolder, "type": "temp"}
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "the %dx%d clip could not be made smaller for playback (%s)", width, height, error
            )
            return None

    keeping = getattr(video, "keeping", None)
    for codec in CODECS:
        try:
            with keeping() if keeping is not None else contextlib.nullcontext():
                video.save_to(
                    target,
                    format=VideoContainer(CONTAINER),
                    codec=VideoCodec(codec),
                )
            return {"filename": file, "subfolder": subfolder, "type": "temp"}
        except Exception as error:  # noqa: BLE001
            logger.debug("codec %s did not encode the clip (%s)", codec, error)
    logger.warning(
        "no codec in %s encoded the %dx%d clip for playback", ", ".join(CODECS), width, height
    )
    return None
