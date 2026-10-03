"""Holding a run still until the browser that queued it resumes it.

``POST /was/interface/api/pause`` takes ``node_id``, ``action`` (``resume`` or ``cancel``),
``value`` and the ``client_id`` the run was queued under. ``GET`` lists the waiting nodes.
"""

from __future__ import annotations

import threading
import time

from .. import log

__all__ = [
    "ROUTE",
    "TICK",
    "RESUMED",
    "CANCELLED",
    "TIMED_OUT",
    "waiting",
    "wait_for_resume",
    "register_routes",
]

logger = log.get_logger("interface.pause")

#: The one route the browser resumes through.
ROUTE = "/was/interface/api/pause"

#: Seconds between checks while a node is held.
TICK = 0.1

#: How a hold ended.
RESUMED = "resumed"
CANCELLED = "cancelled"
TIMED_OUT = "timed out"

#: Node id -> what it is waiting for, while it waits. Guarded by ``_lock``.
_holds: dict[str, dict] = {}
_lock = threading.Lock()

_registered = False


def waiting() -> list[dict]:
    """Every node holding a run still.

    Returns:
        One entry per held node, each ``{"node_id", "message", "waited"}``.
    """
    now = time.monotonic()
    with _lock:
        current = list(_holds.items())
    return [
        {"node_id": node_id, "message": hold.get("message", ""),
         "kind": hold.get("kind", "none"), "content": hold.get("content", ""),
         "timeout": hold.get("timeout", 0.0),
         "waited": round(now - hold["started"], 1)}
        for node_id, hold in current
    ]


def _queued_client() -> str:
    """The client id the running prompt was queued under, or ``""`` where it had none."""
    try:
        from server import PromptServer

        return str(getattr(PromptServer.instance, "client_id", "") or "")
    except Exception:
        return ""


def _announce(
    node_id: str, message: str, timeout: float, kind: str = "none", client: str = ""
) -> None:
    """Tell the browser a node is waiting.

    Args:
        node_id: The node holding the run.
        message: Text drawn beside the resume control.
        timeout: Seconds the hold lasts, or 0 for no limit.
        kind: What is on offer to edit.
        client: The client id to tell, or ``""`` to tell every open tab.
    """
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "was-pause",
            {"node_id": node_id, "message": message, "timeout": timeout, "kind": kind},
            client or None,
        )
    except Exception as error:
        logger.debug("a paused node could not be announced (%s)", error)


def _released(node_id: str, action: str, client: str = "") -> None:
    """Tell the browser a node is no longer waiting.

    Args:
        node_id: The node that held the run.
        action: How the hold ended.
        client: The client id to tell, or ``""`` to tell every open tab.
    """
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "was-pause-done", {"node_id": node_id, "action": action}, client or None
        )
    except Exception as error:
        logger.debug("a resumed node could not be announced (%s)", error)


def wait_for_resume(
    node_id: str, timeout: float = 0.0, message: str = "",
    kind: str = "none", content: str = "",
) -> tuple[str, str]:
    """Hold the run until the browser resumes it, cancels it, or the wait runs out.

    Args:
        node_id: The node holding the run, which is what the browser resumes by.
        timeout: Seconds to wait, or 0 to wait with no limit.
        message: Text drawn beside the resume control.
        kind: What is on offer to edit: ``"none"``, ``"text"`` or ``"canvas"``.
        content: What to edit, which the browser reads back from the route.

    Returns:
        How the hold ended, and the value the browser sent back, empty where it sent none.

    Raises:
        InterruptProcessingException: The run was cancelled, from the browser or from
            ComfyUI's own cancel.
        ValueError: The run was queued without a client id and ``timeout`` is 0.
    """
    import comfy.model_management

    key = str(node_id)
    client = _queued_client()
    if not client:
        if not timeout:
            raise ValueError(
                f"Node {key} would hold this run with no time limit, but the run was queued "
                f"without a client id, so no browser tab can resume it. Queue it from the "
                f"ComfyUI page, send the prompt with the client_id of an open tab, or give "
                f"the hold a timeout."
            )
        logger.warning(
            "%s is holding a run queued without a client id, so no browser tab can resume "
            "it; it carries on after %gs", key, timeout,
        )
    with _lock:
        _holds[key] = {"started": time.monotonic(), "message": message,
                       "kind": kind, "content": content, "timeout": timeout,
                       "client": client, "action": None, "value": ""}
    _announce(key, message, timeout, kind, client)
    told = False
    try:
        while True:
            # Raises on a cancelled run and clears ComfyUI's interrupt flag as it does.
            comfy.model_management.throw_exception_if_processing_interrupted()
            with _lock:
                hold = dict(_holds.get(key) or {}) or None
            if hold is None:
                return RESUMED, ""
            action = hold.get("action")
            if action == CANCELLED:
                _released(key, CANCELLED, client)
                told = True
                raise comfy.model_management.InterruptProcessingException()
            if action == RESUMED:
                _released(key, RESUMED, client)
                told = True
                return RESUMED, hold.get("value") or ""
            if timeout and (time.monotonic() - hold["started"]) >= timeout:
                logger.info("%s waited %.0fs and carried on", key, timeout)
                _released(key, TIMED_OUT, client)
                told = True
                return TIMED_OUT, ""
            time.sleep(TICK)
    except comfy.model_management.InterruptProcessingException:
        logger.info("%s was cancelled, so the hold ended and the run stopped", key)
        raise
    finally:
        with _lock:
            _holds.pop(key, None)
        # Tells the browser the hold ended, where nothing above already has.
        if not told:
            _released(key, CANCELLED, client)


def release(node_id: str, action: str, value: str = "", client: str = "") -> bool:
    """Let a held node carry on.

    Args:
        node_id: The node to release.
        action: ``"resumed"`` or ``"cancelled"``.
        value: What the user edited, for a node that asked for one.
        client: The ComfyUI client id of the browser asking.

    Returns:
        True when a node was waiting under that id.

    Raises:
        PermissionError: ``client`` is not the client id the held run was queued under.
    """
    asking = str(client or "").strip()
    with _lock:
        hold = _holds.get(str(node_id))
        if hold is None:
            return False
        owner = hold.get("client") or ""
        if not owner:
            raise PermissionError(
                f"Node {node_id} is holding a run that was queued without a client id, so no "
                f"browser tab can resume it. It carries on when its timeout runs out, or "
                f"stops with ComfyUI's own Cancel."
            )
        if asking != owner:
            sender = "carried no client_id" if not asking else "came from another tab"
            raise PermissionError(
                f"Node {node_id} is holding a run queued from one ComfyUI tab, and this "
                f"request {sender}. Resume or cancel it from the tab that queued it, or "
                f"with ComfyUI's own Cancel."
            )
        hold["value"] = value
        hold["action"] = action
    return True


def register_routes() -> bool:
    """Register the route the browser resumes a held run through.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a Pause node waits out its timeout.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        from .channel import NO_STORE

        @PromptServer.instance.routes.get(ROUTE)
        async def get_pause(request):
            return web.json_response({"waiting": waiting()}, headers=NO_STORE)

        @PromptServer.instance.routes.post(ROUTE)
        async def post_pause(request):
            try:
                body = await request.json()
            except Exception:
                body = None
            if not isinstance(body, dict):
                return web.json_response({"released": False, "error": "unreadable body"},
                                         status=400, headers=NO_STORE)
            action = CANCELLED if body.get("action") == "cancel" else RESUMED
            client = str(body.get("client_id") or request.query.get("client_id") or "")
            try:
                released = release(
                    body.get("node_id", ""), action, str(body.get("value") or ""), client
                )
            except PermissionError as error:
                return web.json_response(
                    {"released": False, "action": action, "error": str(error)},
                    status=403, headers=NO_STORE,
                )
            return web.json_response({"released": released, "action": action},
                                     headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a Pause node cannot be resumed from the "
            "browser and waits out its timeout",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is releasing held runs", ROUTE)
    return True
