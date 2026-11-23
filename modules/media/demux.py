"""The containers a file is opened as for reading, and the options holding FFmpeg to them.

:data:`OPTIONS` is the ``container_options`` for ``av.open``; :func:`refuse_foreign_probe`
checks a file bound for an unrestricted reader.
"""

from __future__ import annotations

import os

__all__ = ["DEMUXERS", "OPTIONS", "refuse_foreign_probe"]

#: Demuxers a file read by a node may open as: single-file containers, raw streams and
#: pictures. A playlist, a manifest or a list of other files opens as none of these.
DEMUXERS = (
    "mov,mp4,m4a,3gp,3g2,mj2",
    "matroska,webm",
    "avi",
    "mpeg",
    "mpegts",
    "mpegvideo",
    "asf",
    "flv",
    "gif",
    "wav",
    "w64",
    "mp3",
    "flac",
    "ogg",
    "aac",
    "aiff",
    "caf",
    "mxf",
    "dv",
    "nut",
    "ivf",
    "h264",
    "hevc",
    "m4v",
    "png_pipe",
    "apng",
    "image2",
    "webp_pipe",
    "jpeg_pipe",
)

#: ``av.open`` container options: the demuxers above, and local files as the only protocol.
OPTIONS = {"format_whitelist": ",".join(DEMUXERS), "protocol_whitelist": "file"}


def refuse_foreign_probe(path: str, who: str) -> None:
    """Refuse a file that a reader probing every container would open as one outside :data:`DEMUXERS`.

    Args:
        path: The resolved file.
        who: The node's display name, for the message.

    Raises:
        ValueError: The probe chose a demuxer outside :data:`DEMUXERS`, or could not open the file.
    """
    import av

    allowed = {name for entry in DEMUXERS for name in entry.split(",")}
    try:
        with av.open(path, mode="r", container_options={"protocol_whitelist": "file"}) as container:
            names = set(str(container.format.name).split(","))
    except Exception:
        names = set()
    if not names & allowed:
        shown = ", ".join(sorted(names)) or "a container it could not open"
        raise ValueError(
            f"{who} refuses {os.path.basename(path)}: opened without limits it reads as {shown}, "
            f"which can name other files or web addresses. Re-save it as an .mp4, .mkv or .webm "
            f"in a video editor or with ffmpeg."
        )
