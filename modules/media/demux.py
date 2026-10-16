"""The containers a file is opened as for reading, and the options holding FFmpeg to them.

Pass :data:`OPTIONS` as ``container_options`` to ``av.open`` on any file a node reads.
"""

from __future__ import annotations

__all__ = ["DEMUXERS", "OPTIONS"]

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
