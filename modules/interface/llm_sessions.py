"""The saved LLM conversations as JSON a chat panel lists, reads, renames and deletes.

``GET /was/interface/api/llm_sessions`` lists them, newest first, or with ``name`` answers one
transcript. ``POST`` with ``action`` ``rename`` or ``delete`` changes one. Registered while
``features.llm`` is on.
"""

from __future__ import annotations

import asyncio

from .. import log
from ..config import group_enabled
from .channel import NO_STORE, REFUSAL_HEADER

__all__ = [
    "FEATURE",
    "MAX_NAME_CHARS",
    "REFUSAL",
    "ROUTE",
    "WORKERS",
    "change",
    "listing_payload",
    "register_routes",
    "transcript_payload",
]

logger = log.get_logger("interface.llm_sessions")

#: The one route serving the saved conversations.
ROUTE = "/was/interface/api/llm_sessions"

#: The feature that registers the route.
FEATURE = "features.llm"

#: Characters of a requested name that are read.
MAX_NAME_CHARS = 128

#: Requests reading or changing the conversations at once.
WORKERS = 2

#: The words a refusal carries.
REFUSAL = "no conversation is saved under that name"

_registered = False


def listing_payload() -> dict:
    """Every saved conversation, newest first. Never raises.

    Returns:
        ``{"sessions", "limit"}``, each session ``{"name", "messages", "updated"}``.
    """
    from ..llm import storage

    try:
        rows = storage.session_rows()
    except Exception as error:
        logger.debug("%s could not list the conversations (%s)", ROUTE, error)
        rows = []
    return {"sessions": rows, "limit": storage.MAX_LISTED}


def transcript_payload(name) -> dict | None:
    """One saved conversation's transcript. Never raises.

    Args:
        name: A saved conversation's name.

    Returns:
        ``{"name", "transcript", "messages"}``, ``transcript`` JSON lines of ``{"role", "text",
        "pictures"}``; ``None`` when nothing is saved under the name or it cannot be read.
    """
    from ..llm import chat, messages, storage

    wanted = str(name or "")[:MAX_NAME_CHARS]
    try:
        saved = storage.read_session(wanted)
        if saved is None:
            return None
        held = messages.from_dicts(saved.get("archived")) + messages.from_dicts(saved.get("active"))
        return {"name": storage.check_name(wanted, "Session"), "transcript": chat.transcript_lines(held), "messages": len(held)}
    except Exception as error:
        logger.debug("%s could not read conversation %r (%s)", ROUTE, wanted, error)
        return None


def change(action, name, to="") -> tuple[int, dict]:
    """Rename or delete one saved conversation. Never raises.

    Args:
        action: ``"rename"`` or ``"delete"``.
        name: The conversation's name.
        to: Its new name, for ``rename``.

    Returns:
        ``(status, body)``: ``200`` and ``{"name"}`` or ``{"deleted"}``, or ``400`` and
        ``{"error"}`` naming what was refused.
    """
    from ..llm import storage

    name = str(name or "")[:MAX_NAME_CHARS]
    to = str(to or "")[:MAX_NAME_CHARS]
    try:
        if action == "rename":
            storage.rename_session(name, to)
            return 200, {"name": storage.check_name(to, "Session")}
        if action == "delete":
            return 200, {"deleted": storage.delete_session(name)}
    except ValueError as error:
        return 400, {"error": str(error)}
    except Exception as error:
        logger.warning("%s could not %s conversation %r (%s: %s)", ROUTE, action, name, type(error).__name__, error)
        return 400, {"error": f"The conversation could not be {'renamed' if action == 'rename' else 'deleted'}."}
    return 400, {"error": "Ask for action rename or delete."}


def register_routes() -> bool:
    """Register the route serving the saved conversations, while the LLM nodes are on.

    Returns:
        True when the route was registered. False when it was registered already, the LLM
        nodes are off, or the server could not be reached.
    """
    global _registered
    if _registered or not group_enabled(FEATURE):
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        gate = asyncio.Semaphore(WORKERS)

        @PromptServer.instance.routes.get(ROUTE)
        async def get_llm_sessions(request):
            name = request.query.get("name")
            async with gate:
                if name is None:
                    payload = await asyncio.to_thread(listing_payload)
                else:
                    payload = await asyncio.to_thread(transcript_payload, name)
            if payload is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: REFUSAL})
            return web.json_response(payload, headers=NO_STORE)

        @PromptServer.instance.routes.post(ROUTE)
        async def post_llm_sessions(request):
            query = request.query
            async with gate:
                status, payload = await asyncio.to_thread(
                    change, query.get("action", ""), query.get("name", ""), query.get("to", "")
                )
            return web.json_response(payload, status=status, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so the chat panel cannot list saved conversations",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the saved conversations", ROUTE)
    return True
