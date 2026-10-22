"""The system prompt library as JSON a node interface draws.

``GET /was/interface/api/system_prompts`` answers every entry's name, folder, families,
description and size. With ``key`` it answers that one entry and its text, cut to
:data:`MAX_TEXT_CHARS`.
"""

from __future__ import annotations

import asyncio

from .. import log
from ..prompt import system_prompts
from .channel import NO_STORE, REFUSAL_HEADER

__all__ = [
    "MAX_KEY_CHARS",
    "MAX_TEXT_CHARS",
    "REFUSAL",
    "ROUTE",
    "WORKERS",
    "entry_fields",
    "entry_payload",
    "listing_payload",
    "register_routes",
]

logger = log.get_logger("interface.system_prompts")

#: The one route serving the library.
ROUTE = "/was/interface/api/system_prompts"

#: Characters of a requested key that are read.
MAX_KEY_CHARS = 1024

#: Characters of an entry's text one answer holds.
MAX_TEXT_CHARS = system_prompts.MAX_FILE_BYTES

#: Requests walking or reading the library at once.
WORKERS = 2

#: The words a refusal carries.
REFUSAL = "not in the system prompt library"

_registered = False


def entry_fields(entry: system_prompts.Entry) -> dict:
    """One entry as the object the route answers with.

    Args:
        entry: An entry of the library.

    Returns:
        ``{"key", "name", "folder", "families", "description", "tags", "size", "modified"}``,
        ``modified`` in seconds since the epoch.
    """
    return {
        "key": entry.key,
        "name": entry.name,
        "folder": entry.folder,
        "families": list(entry.families),
        "description": entry.description,
        "tags": list(entry.tags),
        "size": int(entry.size),
        "modified": round(float(entry.modified), 3),
    }


def listing_payload() -> dict:
    """Every entry of the library. Never raises.

    Returns:
        ``{"entries", "total", "limit"}``, ``limit`` the most entries a listing holds.
    """
    try:
        entries = system_prompts.scan()
    except Exception as error:
        logger.debug("%s could not list the library (%s)", ROUTE, error)
        entries = ()
    return {
        "entries": [entry_fields(entry) for entry in entries],
        "total": len(entries),
        "limit": system_prompts.MAX_ENTRIES,
    }


def entry_payload(key) -> dict | None:
    """One entry and its text. Never raises.

    Args:
        key: A key of the listing.

    Returns:
        ``{"entry", "text", "truncated"}``, or ``None`` when no entry has that key or its
        file cannot be read.
    """
    wanted = str(key or "")[:MAX_KEY_CHARS]
    try:
        entry = system_prompts.find(wanted)
        if entry is None:
            return None
        document = system_prompts.read(entry)
    except Exception as error:
        logger.debug("%s could not read %s (%s)", ROUTE, wanted, error)
        return None
    body = document.body.strip()
    return {
        "entry": entry_fields(entry),
        "text": body[:MAX_TEXT_CHARS],
        "truncated": len(body) > MAX_TEXT_CHARS,
    }


def register_routes() -> bool:
    """Register the route serving the library.

    Returns:
        True when the route was registered. False when it was registered already, or when
        the server could not be reached.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        gate = asyncio.Semaphore(WORKERS)

        @PromptServer.instance.routes.get(ROUTE)
        async def get_system_prompts(request):
            key = request.query.get("key")
            async with gate:
                if key is None:
                    payload = await asyncio.to_thread(listing_payload)
                else:
                    payload = await asyncio.to_thread(entry_payload, key)
            if payload is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: REFUSAL})
            return web.json_response(payload, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so the system prompt library window gets a "
            "failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the system prompt library", ROUTE)
    return True
