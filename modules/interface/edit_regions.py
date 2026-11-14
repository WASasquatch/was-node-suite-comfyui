"""The things Image Edit Regions found on its last run, for its panel.

``GET /was/interface/api/edit_regions?node_id=&workflow=&client_id=&node_type=`` answers the finds
as JSON with their label layers as PNG, or 204.
"""

from __future__ import annotations

import base64
import io
import threading
from collections import OrderedDict

from .. import log
from .channel import (
    NO_STORE,
    REFUSAL_HEADER,
    executing_class_type,
    executing_client_id,
    executing_node_id,
    executing_workflow_id,
    node_key,
    scoped_key,
    watching,
    workflow_param,
)

__all__ = ["MAX_ENTRIES", "ROUTE", "held", "publish", "register_routes"]

logger = log.get_logger("interface.edit_regions")

#: The route the panel reads.
ROUTE = "/was/interface/api/edit_regions"

#: Nodes whose finds are held, oldest dropped first.
MAX_ENTRIES = 16

#: The refusal sent when nothing is held for the node asked about.
NOTHING_HELD = "no finds"

_lock = threading.Lock()
_held: OrderedDict = OrderedDict()
_latest: dict = {}
_run = 0
_registered = False


def _png(planes) -> str:
    """Up to three ``uint8`` planes as the red, green and blue of one PNG, in base64."""
    import numpy as np
    from PIL import Image

    height, width = planes[0].shape
    packed = np.zeros((height, width, 3), dtype=np.uint8)
    for channel, plane in enumerate(planes[:3]):
        packed[..., channel] = plane.cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(packed, "RGB").save(buffer, format="PNG", optimize=False, compress_level=6)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def publish(segments, layers, size, width: int, height: int, notes=(), node_id=None) -> bool:
    """Hold one run's finds for the publishing node's panel. Never raises.

    Args:
        segments: The finds, as :func:`modules.image.edit_regions.find` answers them.
        layers: Their label layers, from :func:`modules.image.edit_regions.label_layers`.
        size: ``(width, height)`` the layers were drawn at.
        width: The picture's width.
        height: The picture's height.
        notes: One line per edit the run passed over.
        node_id: The node's graph id; left out, the executing node's.

    Returns:
        True when the finds were held.
    """
    global _run
    if not watching():
        return False
    try:
        node = node_key(node_id if node_id is not None else executing_node_id())
        if node is None:
            return False
        client = executing_client_id()
        key = (client, scoped_key(node, executing_workflow_id()))
        payload = {
            "width": int(width),
            "height": int(height),
            "map": [int(size[0]), int(size[1])],
            "layers": [_png(layers[start:start + 3]) for start in range(0, len(layers), 3)],
            "layer_count": len(layers),
            "segments": [
                {
                    "id": index + 1,
                    "label": segment.label,
                    "names": list(segment.names),
                    "score": round(segment.score, 3),
                    "box": [round(value, 5) for value in segment.box],
                    "area": round(segment.area, 6),
                    "colour": segment.colour,
                }
                for index, segment in enumerate(segments)
            ],
            "notes": [str(note)[:200] for note in list(notes)[:64]],
        }
        declared = executing_class_type(node_id)
        with _lock:
            _run += 1
            payload["run"] = _run
            if declared:
                payload["node_type"] = declared
            _held.pop(key, None)
            _held[key] = payload
            _latest[(client, node)] = key
            while len(_held) > MAX_ENTRIES:
                evicted, _ = _held.popitem(last=False)
                for name in [name for name, kept in _latest.items() if kept == evicted]:
                    del _latest[name]
        return True
    except Exception as error:
        logger.debug("no finds were held for node %s (%s)", node_id, error)
        return False


def held(node_id, workflow=None, client=None, node_type=None) -> dict | None:
    """The finds a node held on its last run.

    Args:
        node_id: The node's graph id.
        workflow: The workflow id the reader has open; left out, the node's latest from any.
        client: The reader's client id; a run queued under another client id is never answered.
        node_type: The reader's node class; finds held under another class are not answered.

    Returns:
        The held payload, or None.
    """
    node = node_key(node_id)
    if node is None:
        return None
    asked = client.strip() if isinstance(client, str) else ""
    owners = (asked, "") if asked else ("",)
    scope = workflow_param(workflow)
    with _lock:
        for owner in owners:
            named = (owner, scoped_key(node, scope)) if scope else _latest.get((owner, node))
            for key in (named, (owner, node)):
                payload = _held.get(key)
                if payload is None:
                    continue
                if node_type and payload.get("node_type") and payload["node_type"] != node_type:
                    return None
                _held.move_to_end(key)
                return payload
    return None


def register_routes() -> bool:
    """Register the route the panel reads its node's finds from.

    Returns:
        True when the route was registered; False when it was already, or when the server could
        not be reached, in which case the panel shows no finds.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_edit_regions(request):
            payload = held(
                request.query.get("node_id"),
                request.query.get("workflow"),
                request.query.get("client_id"),
                request.query.get("node_type"),
            )
            if payload is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: NOTHING_HELD})
            return web.json_response(payload, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so Image Edit Regions' panel shows no finds",
            ROUTE, type(error).__name__, error,
        )
        return False
    _registered = True
    return True
