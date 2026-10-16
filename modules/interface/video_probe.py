"""What a video measures, as JSON a node interface draws its timeline against.

``GET /was/interface/api/video_probe?file=<name>`` answers the rate, the frame count, the
duration, the size and the bit depth of one video a file menu lists.
"""

from __future__ import annotations

import os

from .. import log
from .channel import NO_STORE

__all__ = ["PROBES", "ROUTE", "probe_payload", "register_routes", "video_path"]

logger = log.get_logger("interface.video_probe")

#: The one route serving the measurement.
ROUTE = "/was/interface/api/video_probe"

#: Videos measured at once, each on a worker thread.
PROBES = 4

_registered = False


def video_path(name: str) -> str | None:
    """The regular video file a menu label names.

    Args:
        name: A label the file listing offers, as ``clips/a.mp4 [output]``, or a bare
            name in the input folder.

    Returns:
        The resolved absolute path, or ``None`` where no listed video carries that label.

    Raises:
        PathNotAllowed: The listed file resolves outside every permitted read root.
    """
    from ..media.reader import VIDEO_EXTENSIONS
    from ..util import file_listing, sandbox

    text = (name or "").strip()
    if not text:
        return None
    entry = file_listing.find(text, VIDEO_EXTENSIONS, file_listing.ROOTS)
    if entry is None and not text.endswith("]"):
        entry = file_listing.find(
            f"{text} [{file_listing.INPUT}]", VIDEO_EXTENSIONS, (file_listing.INPUT,)
        )
    if entry is None:
        return None
    path = str(sandbox.resolve_read(entry.path))
    return path if os.path.isfile(path) else None


def probe_payload(name: str) -> dict:
    """What the named video measures.

    Never raises.

    Args:
        name: The file as the widget holds it: a menu label, or a bare name in the input
            folder.

    Returns:
        ``{"fps", "frame_count", "duration", "width", "height", "bit_depth", "read"}``.
        ``read`` is False where nothing could be measured.
    """
    # A file that cannot be read answers zeroes rather than an error, which leaves the player
    # on its own measurement rather than on nothing at all.
    empty = {
        "fps": 0.0, "frame_count": 0, "duration": 0.0, "width": 0, "height": 0,
        "bit_depth": 0, "read": False,
    }
    if not name:
        return empty
    try:
        path = video_path(name)
        if path is None:
            logger.debug("%s refused %r: no listed video carries that name", ROUTE, name)
            return empty
        from ..media import reader

        found = reader.probe(path)
    except Exception as error:
        logger.debug("%s could not measure %r (%s)", ROUTE, name, error)
        return empty
    return {
        "fps": float(found.fps),
        "frame_count": int(found.frame_count),
        "duration": float(found.duration),
        "width": int(found.width),
        "height": int(found.height),
        "bit_depth": int(found.bit_depth),
        "read": True,
    }


def register_routes() -> bool:
    """Register the route serving what a video measures.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a player asking for the measurement gets a
        failed request and counts frames from what it can present.
    """
    global _registered
    if _registered:
        return False
    try:
        import asyncio

        from aiohttp import web
        from server import PromptServer

        gate = asyncio.Semaphore(PROBES)

        @PromptServer.instance.routes.get(ROUTE)
        async def get_video_probe(request):
            async with gate:
                payload = await asyncio.to_thread(
                    probe_payload, request.query.get("file", "")
                )
            return web.json_response(payload, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a video player asking what a clip measures "
            "gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the rate and frame count of listed videos", ROUTE)
    return True
