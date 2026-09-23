"""What a Three.js module file declares, for the node that picked it.

The canvas labels a node's slots from this, so a module names its inputs once in its header.
"""

from __future__ import annotations

__all__ = ["ROUTE", "declaration", "register_routes"]

from ..log import get_logger

logger = get_logger("interface.three_module")

#: Where the canvas asks what a module declares.
ROUTE = "/was/threejs/api/module"


def declaration(label: str) -> dict:
    """What one module declares, as the canvas reads it.

    Args:
        label: The module menu's value.

    Returns:
        ``{"ok": True, "kind": ..., "wires": [...], "values": [...]}`` where the module was
        read, otherwise ``{"ok": False, "error": ...}``.
    """
    from ..threejs import module_file

    if not module_file.chosen(label):
        return {"ok": True, "kind": "", "wires": [], "values": []}
    try:
        declared = module_file.load(label)
    except Exception as error:
        return {"ok": False, "error": str(error)}
    return {
        "ok": True,
        "kind": declared.kind,
        "wires": [{"name": one.name, "kind": one.kind} for one in declared.wires],
        "values": [
            {"name": one.name, "kind": one.kind, "default": one.default}
            for one in declared.values
        ],
    }


def register_routes() -> bool:
    """Answer what a module declares.

    Returns:
        True where the route was registered.
    """
    try:
        from aiohttp import web

        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_three_module(request):
            label = str(request.query.get("label", ""))
            return web.json_response(
                declaration(label), headers={"Cache-Control": "no-store"}
            )

    except Exception as error:
        logger.debug("the Three.js module channel is unavailable (%s)", error)
        return False
    return True
