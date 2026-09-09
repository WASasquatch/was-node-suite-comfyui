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
    logger.warning("no codec in %s encoded the clip for playback", ", ".join(CODECS))
    return None
