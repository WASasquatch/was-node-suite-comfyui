"""Writing a VIDEO to the temp folder so a node interface can play it back.

Files land in ComfyUI's temp directory, which is cleared on restart. Every side is written
as mp4.
"""

from __future__ import annotations

import os

from .. import log

logger = log.get_logger("media.temp_video")

#: Container every side is written in.
CONTAINER = "mp4"

#: Codecs tried in order, the first that encodes winning.
CODECS = ("auto", "h264")


def even_sided(video):
    """A video whose frames have an even width and height.

    Args:
        video: A ``VIDEO`` object.

    Returns:
        The video as it stands where both sides are already even, otherwise a copy with its
        last row and column repeated.
    """
    width, height = video.get_dimensions()
    if width % 2 == 0 and height % 2 == 0:
        return video

    import torch
    from comfy_api.latest import InputImpl, Types

    def squared(frames, rows, columns):
        if frames is None:
            return None
        if rows:
            frames = torch.cat([frames, frames[:, -1:, ...]], dim=1)
        if columns:
            frames = torch.cat([frames, frames[:, :, -1:, ...]], dim=2)
        return frames

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

        # An odd side is refused by the yuv420p encoders every codec here uses.
        video = even_sided(video)
        width, height = video.get_dimensions()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_temp_directory(), width, height
        )
        file = f"{name}_{counter:05}_.{VideoContainer.get_extension(CONTAINER)}"
        target = os.path.join(folder, file)
    except Exception as error:  # noqa: BLE001
        logger.warning("a video could not be prepared for playback (%s)", error)
        return None

    for codec in CODECS:
        try:
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
