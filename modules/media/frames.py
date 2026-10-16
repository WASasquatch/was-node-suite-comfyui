"""Decoding a video file into numbered image files.

One decode pass through PyAV, one image file per frame, written with Pillow. Colour leaves
the decoder as ``rgb24`` and is handed to Pillow in that order.
"""

from __future__ import annotations

import os

from PIL import Image

from .. import deps, log
from ..util import sandbox
from . import demux
from .video import progress_bar

__all__ = ["extract"]

logger = log.get_logger("media.frames")


def extract(
    video_file: str,
    output_folder: str,
    prefix: str = "frame_",
    extension: str = "png",
    zero_padding_digits: int = -1,
) -> int:
    """Write every frame of a video to ``output_folder`` as a numbered image.

    Args:
        video_file: Video to read. Any container and codec libavformat can demux.
        output_folder: Directory to write into. Created if it does not exist. Existing
            files with the same names are overwritten.
        prefix: Text before the frame number, such as ``"frame_"``. It names a file inside
            ``output_folder``, so it may name a sub-directory but not a drive, a root or a
            ``..`` segment.
        extension: File extension, which decides the image format: ``png``, ``jpg``,
            ``gif`` or ``tiff``.
        zero_padding_digits: Width the frame number is zero-padded to. Zero or less
            numbers the frames without padding, so they sort by name in the wrong order
            past frame 9.

    Returns:
        The number of frames written, counting from the first frame as 0.

    Raises:
        DependencyError: av is not installed.
        FileNotFoundError: ``video_file`` does not exist.
        IndexError: The file holds no video stream.
        PathNotAllowed: ``output_folder`` is outside every permitted write root, or
            ``prefix`` names a file outside it.
        ValueError: Pillow has no writer for ``extension``.
    """
    deps.require("av")

    import av

    destination = sandbox.resolve_write(output_folder)
    os.makedirs(destination, exist_ok=True)

    frame_number = 0
    with av.open(video_file, mode="r", container_options=demux.OPTIONS) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        progress = progress_bar(stream.frames or 0)

        for frame in video.decode(stream):
            if zero_padding_digits > 0:
                name = f"{prefix}{frame_number:0{zero_padding_digits}}.{extension}"
            else:
                name = f"{prefix}{frame_number}.{extension}"

            # A prefix such as "C:Windows/" would replace the destination outright if the
            # name were merely joined onto it.
            frame_path = sandbox.resolve_write_file(destination, name)
            Image.fromarray(frame.to_ndarray(format="rgb24")).save(frame_path)
            frame_number += 1
            progress.update()

    logger.info("saved %s frames to %s", frame_number, destination)

    return frame_number
