"""A small picture of one file a menu lists, for a node interface to draw.

``GET /was/interface/api/file_thumbnail?label=<label>&edge=<pixels>`` answers an image, or a
video's first frame, fitted inside ``edge`` pixels a side. Anything else answers 204.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from collections import OrderedDict

from .. import log
from ..constants import ALLOWED_EXT
from ..media.reader import VIDEO_EXTENSIONS
from ..util import file_listing, sandbox
from .channel import NO_STORE

__all__ = ["DEFAULT_EDGE", "EXTENSIONS", "MAX_EDGE", "ROUTE", "register_routes", "thumbnail"]

logger = log.get_logger("interface.file_thumbnail")

#: The one route serving the pictures.
ROUTE = "/was/interface/api/file_thumbnail"

#: Files a picture is drawn for.
EXTENSIONS = (*ALLOWED_EXT, *VIDEO_EXTENSIONS)

#: Pixels a side a picture is fitted inside: the default, and the bounds a request is held to.
DEFAULT_EDGE = 192
MIN_EDGE = 32
MAX_EDGE = 512

#: How many pictures are held, and how many bytes of them.
MAX_ENTRIES = 512
MAX_BYTES = 48 * 1024 * 1024

#: Pixels a side a source picture is decoded at before it is fitted.
DECODE_EDGE = 2048

#: Pictures drawn at once, each on a worker thread.
DECODERS = 4

#: Headers for a picture, which changes URL whenever the file does.
CACHED = {"Cache-Control": "private, max-age=86400"}

_lock = threading.Lock()
_cache: OrderedDict[tuple, tuple[bytes, str]] = OrderedDict()
_held = 0
_registered = False


def _encoded(picture) -> tuple[bytes, str]:
    """A PIL image as JPEG, or as PNG where it carries transparency."""
    buffer = io.BytesIO()
    if "A" in picture.getbands():
        picture.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png"
    picture.convert("RGB").save(buffer, format="JPEG", quality=82)
    return buffer.getvalue(), "image/jpeg"


def _image(path: str, edge: int):
    """The picture a file holds, fitted inside ``edge`` pixels a side."""
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        opened.draft("RGB", (DECODE_EDGE, DECODE_EDGE))
        picture = ImageOps.exif_transpose(opened)
        picture.thumbnail((edge, edge))
        return picture.copy()


def _first_frame(path: str, edge: int):
    """A video's first frame, fitted inside ``edge`` pixels a side."""
    from PIL import Image

    from ..media import reader

    clip = reader.read(path, start=0, end=0, num_frames=1)
    frame = (clip.images[0, ..., :3].clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    picture = Image.fromarray(frame, "RGB")
    picture.thumbnail((edge, edge))
    return picture


def _remember(key: tuple, body: bytes, kind: str) -> None:
    """Hold one picture, dropping the oldest past the bounds."""
    global _held
    with _lock:
        _cache[key] = (body, kind)
        _held += len(body)
        while _cache and (len(_cache) > MAX_ENTRIES or _held > MAX_BYTES):
            _, (dropped, _) = _cache.popitem(last=False)
            _held -= len(dropped)


def thumbnail(label: str, edge: int = DEFAULT_EDGE) -> tuple[bytes, str, str] | None:
    """The picture one menu label names, fitted inside ``edge`` pixels a side.

    Never raises.

    Args:
        label: A label the file listing offers, as ``cast/alice.png [input]``.
        edge: Pixels a side the picture is fitted inside, held to the bounds.

    Returns:
        ``(bytes, content type, etag)``, or ``None`` where the label names no listed image
        or video, the file cannot be read, or the picture cannot be made.
    """
    edge = max(MIN_EDGE, min(MAX_EDGE, int(edge or DEFAULT_EDGE)))
    try:
        entry = file_listing.find(label, EXTENSIONS, file_listing.ROOTS)
        if entry is None:
            return None
        path = str(sandbox.resolve_read(entry.path))
        stat = os.stat(path)
        key = (entry.label, edge, int(stat.st_mtime_ns), int(stat.st_size))
        etag = '"' + hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:20] + '"'
        with _lock:
            held = _cache.get(key)
            if held is not None:
                _cache.move_to_end(key)
                return held[0], held[1], etag
        suffix = os.path.splitext(path)[1].lower()
        if suffix in VIDEO_EXTENSIONS and suffix not in ALLOWED_EXT:
            picture = _first_frame(path, edge)
        else:
            picture = _image(path, edge)
        body, kind = _encoded(picture)
        _remember(key, body, kind)
        return body, kind, etag
    except Exception as error:
        logger.debug("%s could not draw %r (%s: %s)", ROUTE, label, type(error).__name__, error)
        return None


def register_routes() -> bool:
    """Register the route serving file pictures.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a browser asking for a picture gets a
        failed request and draws a stand-in.
    """
    global _registered
    if _registered:
        return False
    try:
        import asyncio

        from aiohttp import web
        from server import PromptServer

        gate = asyncio.Semaphore(DECODERS)

        @PromptServer.instance.routes.get(ROUTE)
        async def get_file_thumbnail(request):
            try:
                edge = int(request.query.get("edge", DEFAULT_EDGE))
            except (TypeError, ValueError):
                edge = DEFAULT_EDGE
            async with gate:
                answer = await asyncio.to_thread(thumbnail, request.query.get("label", ""), edge)
            if answer is None:
                return web.Response(
                    status=204, headers={**NO_STORE, "X-WAS-Refusal": "nothing to draw"}
                )
            body, kind, etag = answer
            headers = {**CACHED, "ETag": etag}
            if request.headers.get("If-None-Match") == etag:
                return web.Response(status=304, headers=headers)
            return web.Response(body=body, content_type=kind, headers=headers)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a browser asking for a file picture gets a "
            "failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving pictures of listed files", ROUTE)
    return True
