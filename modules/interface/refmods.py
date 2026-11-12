"""The RefMods in models/refmods as JSON and small pictures a node interface draws.

``GET /was/interface/api/refmods`` answers each file's menu name and what it holds, read from
headers. ``GET /was/interface/api/refmod_thumbnail?name=<name>&edge=<pixels>`` answers its
picture fitted inside ``edge`` pixels, or 204.
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict

from .. import log
from .channel import NO_STORE

__all__ = [
    "DEFAULT_EDGE", "LIST_ROUTE", "MAX_EDGE", "MAX_FILES", "THUMBNAIL_ROUTE", "listing_payload",
    "register_routes", "thumbnail",
]

logger = log.get_logger("interface.refmods")

#: The route listing RefMods, and the one drawing one.
LIST_ROUTE = "/was/interface/api/refmods"
THUMBNAIL_ROUTE = "/was/interface/api/refmod_thumbnail"

#: Files one listing describes.
MAX_FILES = 2000

#: Pixels a side a picture is fitted inside: the default, and the bounds a request is held to.
DEFAULT_EDGE = 192
MIN_EDGE = 32
MAX_EDGE = 512

#: Pictures held, and their bytes together.
MAX_ENTRIES = 256
MAX_BYTES = 24 * 1024 * 1024

#: Requests read from disk at once.
READERS = 2

#: Headers for a picture, which changes URL whenever the file does.
CACHED = {"Cache-Control": "private, max-age=86400"}

_lock = threading.Lock()
_pictures: OrderedDict[tuple, tuple[bytes, str]] = OrderedDict()
_held = 0
_registered = False


def listing_payload() -> dict:
    """Every RefMod the folders hold and what each carries.

    Never raises.

    Returns:
        ``{"refmods", "truncated"}``. Each RefMod is ``{"name", "members", "stored", "mtime"}``,
        ``name`` as the menu lists it and each member ``{"kind", "label", "tokens"}``.
    """
    from ..latent import h3_refmods

    try:
        found = h3_refmods.names()
    except Exception as error:
        logger.debug("%s could not list the RefMods (%s)", LIST_ROUTE, error)
        return {"refmods": [], "truncated": False}
    described = []
    for name in found[:MAX_FILES]:
        entry = h3_refmods.describe_file(name)
        if entry is not None:
            described.append(entry)
    return {"refmods": described, "truncated": len(found) > MAX_FILES}


def thumbnail(name: str, edge: int) -> tuple[bytes, str] | None:
    """One RefMod's picture as JPEG, fitted inside ``edge`` pixels a side.

    Never raises.

    Args:
        name: A menu name.
        edge: The side to fit inside, held to :data:`MIN_EDGE` and :data:`MAX_EDGE`.

    Returns:
        ``(bytes, etag)``, or ``None`` where the RefMod is unknown or holds no picture.
    """
    global _held
    from ..latent import h3_refmods

    edge = max(MIN_EDGE, min(MAX_EDGE, int(edge)))
    try:
        if name not in h3_refmods.names():
            return None
        described = h3_refmods.describe_file(name)
        if described is None:
            return None
        key = (described["name"], described["mtime"], edge)
        with _lock:
            held = _pictures.get(key)
            if held is not None:
                _pictures.move_to_end(key)
                return held
        picture = h3_refmods.preview_picture(described["name"])
        if picture is None:
            return None
        picture.thumbnail((edge, edge))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=82)
        body = buffer.getvalue()
        etag = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
        with _lock:
            _pictures[key] = (body, etag)
            _held += len(body)
            while _pictures and (len(_pictures) > MAX_ENTRIES or _held > MAX_BYTES):
                _, (dropped, _) = _pictures.popitem(last=False)
                _held -= len(dropped)
        return body, etag
    except Exception as error:
        logger.debug("%s could not draw a RefMod (%s: %s)", THUMBNAIL_ROUTE, type(error).__name__, error)
        return None


def register_routes() -> bool:
    """Register the routes listing and drawing RefMods.

    Returns:
        True when the routes were registered. False when they were registered already, or when
        the server could not be reached, in which case a window asking for RefMods lists none.
    """
    global _registered
    if _registered:
        return False
    try:
        import asyncio

        from aiohttp import web
        from server import PromptServer

        gate = asyncio.Semaphore(READERS)

        @PromptServer.instance.routes.get(LIST_ROUTE)
        async def get_refmods(request):
            async with gate:
                payload = await asyncio.to_thread(listing_payload)
            return web.json_response(payload, headers=NO_STORE)

        @PromptServer.instance.routes.get(THUMBNAIL_ROUTE)
        async def get_refmod_thumbnail(request):
            try:
                edge = int(request.query.get("edge", DEFAULT_EDGE))
            except (TypeError, ValueError):
                edge = DEFAULT_EDGE
            async with gate:
                answer = await asyncio.to_thread(thumbnail, str(request.query.get("name", "")), edge)
            if answer is None:
                return web.Response(status=204, headers={**NO_STORE, "X-WAS-Refusal": "nothing to draw"})
            body, etag = answer
            headers = {**CACHED, "ETag": etag}
            if request.headers.get("If-None-Match") == etag:
                return web.Response(status=304, headers=headers)
            return web.Response(body=body, content_type="image/jpeg", headers=headers)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a window asking for RefMods lists none",
            LIST_ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", LIST_ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is listing RefMods", LIST_ROUTE)
    return True
