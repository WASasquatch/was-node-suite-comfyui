"""Bytes a Three.js node prepared, held for the browser to fetch.

Keyed by the bytes' own SHA-256. Bounded, oldest dropped first, and gone on restart.
"""

from __future__ import annotations

__all__ = ["ROUTE", "entry_for", "held", "keep", "read", "register_routes"]

import hashlib
from collections import OrderedDict
from threading import Lock

from ..log import get_logger

logger = get_logger("interface.three_asset")

#: Where the browser fetches a held asset from.
ROUTE = "/was/threejs/api/asset"

#: Most assets kept at once.
MAX_ENTRIES = 64

#: Most bytes kept at once, across every entry.
MAX_BYTES = 512 * 1024 * 1024

#: Characters of the digest used as a key.
KEY_CHARS = 32

_entries: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_total = 0
_lock = Lock()


def keep(body: bytes, content_type: str = "image/png") -> str:
    """Hold one asset and answer the key it is fetched by.

    Args:
        body: The bytes to hold.
        content_type: What to serve them as, such as ``image/png`` or ``model/gltf-binary``.

    Returns:
        The key, which is the first :data:`KEY_CHARS` characters of the SHA-256 digest.
    """
    global _total
    key = hashlib.sha256(body).hexdigest()[:KEY_CHARS]
    with _lock:
        if key in _entries:
            _entries.move_to_end(key)
            return key
        _entries[key] = (body, content_type)
        _total += len(body)
        while _entries and (len(_entries) > MAX_ENTRIES or _total > MAX_BYTES):
            _, (dropped, _kind) = _entries.popitem(last=False)
            _total -= len(dropped)
    return key


def read(key: str) -> bytes | None:
    """The bytes held under one key.

    Args:
        key: The key :func:`keep` answered.

    Returns:
        The bytes, or None where nothing is held under that key.
    """
    entry = entry_for(key)
    return None if entry is None else entry[0]


def entry_for(key: str) -> tuple[bytes, str] | None:
    """The bytes and their content type, held under one key.

    Args:
        key: The key :func:`keep` answered.

    Returns:
        ``(body, content_type)``, or None where nothing is held under that key.
    """
    with _lock:
        entry = _entries.get(key)
        if entry is not None:
            _entries.move_to_end(key)
    return entry


def held() -> tuple[int, int]:
    """How much is being held.

    Returns:
        ``(entries, bytes)``.
    """
    with _lock:
        return len(_entries), _total


def register_routes() -> bool:
    """Serve held textures over HTTP.

    Returns:
        True where the route was registered.
    """
    try:
        from aiohttp import web

        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_asset(request):
            key = str(request.query.get("key", "")).strip()
            entry = entry_for(key) if key else None
            if entry is None:
                return web.Response(
                    status=404,
                    text=(
                        "Nothing is held under that key. Queue the graph again: the store is "
                        "in memory and is cleared when ComfyUI restarts."
                    ),
                )
            body, content_type = entry
            return web.Response(
                body=body,
                content_type=content_type,
                headers={"Cache-Control": "private, max-age=31536000, immutable",
                         "X-Content-Type-Options": "nosniff"},
            )

    except Exception as error:
        logger.warning(
            "%s was not registered (%s), so a Three.js texture or model cannot be fetched "
            "and anything using one is drawn without it",
            ROUTE, type(error).__name__,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False

    logger.debug("%s is serving held assets", ROUTE)
    return True
