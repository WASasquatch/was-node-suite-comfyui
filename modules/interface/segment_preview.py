"""Each segment of a run as one sheet of small frames, replaced as it samples.

``GET /was/interface/api/segment_preview?node_id=<id>&segment=<n>`` answers a sheet as JPEG;
``/index?node_id=<id>`` lists the segments held. :data:`EVENT` announces each new sheet.
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections import OrderedDict

from .. import log
from .channel import NO_STORE, executing_prompt_id

__all__ = [
    "CELL_WIDTH",
    "EVENT",
    "INDEX_ROUTE",
    "MAX_BYTES",
    "MAX_SHEET",
    "OWNER_PROPERTY",
    "ROUTE",
    "clear",
    "held",
    "layout",
    "owner_of",
    "publish",
    "register_routes",
    "sheet",
]

logger = log.get_logger("interface.segment_preview")

#: The route serving one sheet, and the one listing a node's sheets.
ROUTE = "/was/interface/api/segment_preview"
INDEX_ROUTE = "/was/interface/api/segment_preview/index"

#: The message a new sheet is announced by.
EVENT = "was-segment-preview"

#: Widest a frame is drawn, the longest side a sheet reaches, and its JPEG quality.
CELL_WIDTH = 384
MAX_SHEET = 8192
QUALITY = 80

#: The node property a node's sheets are filed under, where its saved workflow carries one.
OWNER_PROPERTY = "was_preview_key"

#: Frames scaled down at once.
CHUNK = 16

#: Bytes of sheets held across every node.
MAX_BYTES = 384 * 1024 * 1024

_lock = threading.Lock()
_sheets: OrderedDict[tuple[str, int], tuple[bytes, dict]] = OrderedDict()
_held = 0
_version = 0
_registered = False


def owner_of(unique_id, extra_pnginfo) -> str | None:
    """The key one node's sheets are filed under.

    Args:
        unique_id: The node's id in the prompt.
        extra_pnginfo: What ComfyUI hands a run beside the prompt, its ``workflow`` included.

    Returns:
        The node's :data:`OWNER_PROPERTY` where the workflow carries one, else its id, or None.
    """
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    for entry in nodes or ():
        if isinstance(entry, dict) and str(entry.get("id")) == str(unique_id):
            key = (entry.get("properties") or {}).get(OWNER_PROPERTY)
            if key:
                return str(key)
            break
    return None if unique_id is None else str(unique_id)


def layout(count: int, width: int, height: int) -> tuple[int, int, int]:
    """The cell size and column count a sheet of frames is drawn at.

    Args:
        count: Frames on the sheet.
        width: A frame's width in pixels.
        height: A frame's height in pixels.

    Returns:
        ``(cell_width, cell_height, columns)``, every side within :data:`MAX_SHEET`.
    """
    count, width, height = max(1, int(count)), max(1, int(width)), max(1, int(height))
    cell_w = min(CELL_WIDTH, width)
    while True:
        cell_h = max(1, round(cell_w * height / width))
        columns = max(1, min(count, MAX_SHEET // cell_w))
        if math.ceil(count / columns) * cell_h <= MAX_SHEET or cell_w <= 16:
            return cell_w, cell_h, columns
        cell_w = max(16, int(cell_w * 0.85))


def _tiled(frames, cell_w: int, cell_h: int, columns: int):
    """Frames scaled into cells and tiled row by row, as a ``uint8`` array."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    count = int(frames.shape[0])
    rows = math.ceil(count / columns)
    sheet = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, count, CHUNK):
            part = frames[start:start + CHUNK, ..., :3].movedim(-1, 1).float()
            small = F.interpolate(part, size=(cell_h, cell_w), mode="area")
            pixels = (small.clamp(0, 1) * 255).round().to(torch.uint8).movedim(1, -1).cpu().numpy()
            for offset, picture in enumerate(pixels):
                row, column = divmod(start + offset, columns)
                sheet[row * cell_h:(row + 1) * cell_h, column * cell_w:(column + 1) * cell_w] = picture
    return sheet


def publish(owner, segment: int, frames, step: int = 0, steps: int = 0, head: int = 0,
            final: bool = False) -> dict | None:
    """Hold one segment's frames as a sheet and announce it.

    Args:
        owner: The id of the node the segment belongs to.
        segment: Segment number, from 0.
        frames: ``[frames, height, width, channels]`` pictures in ``[0, 1]``.
        step: The sampling step the frames come from, from 1.
        steps: Steps the segment samples for.
        head: Frames at the start of the sheet that come from the segment before.
        final: True for the segment's finished frames.

    Returns:
        The announced fields, or None where nothing was held.
    """
    global _held, _version
    if owner is None or frames is None or int(frames.shape[0]) < 1:
        return None
    from PIL import Image

    count, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    cell_w, cell_h, columns = layout(count, width, height)
    buffer = io.BytesIO()
    Image.fromarray(_tiled(frames, cell_w, cell_h, columns)).save(
        buffer, format="JPEG", quality=QUALITY)
    body = buffer.getvalue()
    key = (str(owner), int(segment))
    with _lock:
        _version += 1
        fields = {
            "node_id": key[0], "segment": key[1], "frames": count, "columns": columns,
            "cell_width": cell_w, "cell_height": cell_h, "width": width, "height": height,
            "head": max(0, int(head)), "step": int(step), "steps": int(steps),
            "final": bool(final), "version": _version, "time": time.time(),
            "prompt_id": executing_prompt_id() or "",
        }
        previous = _sheets.pop(key, None)
        if previous is not None:
            _held -= len(previous[0])
        _sheets[key] = (body, fields)
        _held += len(body)
        while _held > MAX_BYTES and len(_sheets) > 1:
            _, (dropped, _) = _sheets.popitem(last=False)
            _held -= len(dropped)
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(EVENT, fields)
    except Exception as error:
        logger.debug("a segment preview could not be announced (%s)", error)
    return fields


def sheet(owner, segment: int) -> tuple[bytes, dict] | None:
    """The latest sheet held for one segment.

    Args:
        owner: The id of the node the segment belongs to.
        segment: Segment number, from 0.

    Returns:
        ``(jpeg, fields)``, or None.
    """
    with _lock:
        return _sheets.get((str(owner), int(segment)))


def held(owner) -> list[dict]:
    """The fields of every sheet held for one node.

    Args:
        owner: The node's id.

    Returns:
        One entry per segment, in segment order.
    """
    with _lock:
        found = [fields for (node, _), (_, fields) in _sheets.items() if node == str(owner)]
    return sorted(found, key=lambda fields: fields["segment"])


def clear(owner, keep: int | None = None) -> int:
    """Drop one node's sheets.

    Args:
        owner: The node's id.
        keep: Segments below this number are kept, or None to drop every one.

    Returns:
        Sheets dropped.
    """
    global _held
    dropped = 0
    with _lock:
        for key in [key for key in _sheets if key[0] == str(owner)
                    and (keep is None or key[1] >= int(keep))]:
            body, _ = _sheets.pop(key)
            _held -= len(body)
            dropped += 1
    return dropped


def register_routes() -> bool:
    """Register the routes serving segment sheets.

    Returns:
        True when the routes were registered. False when they were registered already, or when
        the server could not be reached.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_segment_preview(request):
            try:
                found = sheet(request.query.get("node_id", ""), int(request.query.get("segment", -1)))
            except (TypeError, ValueError):
                found = None
            if found is None:
                return web.Response(status=204, headers={**NO_STORE, "X-WAS-Refusal": "no sheet held"})
            return web.Response(body=found[0], content_type="image/jpeg", headers=NO_STORE)

        @PromptServer.instance.routes.get(INDEX_ROUTE)
        async def get_segment_preview_index(request):
            return web.json_response(held(request.query.get("node_id", "")), headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so the Prompt Timeline draws no segment previews",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving segment previews", ROUTE)
    return True
