"""Live picture library search for Image Library Search's panel.

``POST /was/interface/api/image_library`` takes the node's settings and prompt and answers its
folder tree with each folder's index state, and the closest pictures, from cached vectors only.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

from .. import log

__all__ = ["MAX_BODY", "ROUTE", "answer", "keep", "register_routes", "settings_key"]

logger = log.get_logger("interface.image_library")

#: The one route the panel asks.
ROUTE = "/was/interface/api/image_library"

#: Largest request body read, in bytes.
MAX_BODY = 256 * 1024

#: Libraries held between requests, and seconds a folder read is reused before it is read again.
MAX_HELD = 6
REREAD_AFTER = 20.0

#: Most folders the tree answers.
MAX_FOLDERS = 2000

#: Most characters of the filter and of the folder selection read.
MAX_FILTER_CHARS = 400
MAX_FOLDERS_CHARS = 64 * 1024

#: Settings key -> ``{"read", "at", "all", "selected"}``. Guarded by ``_lock``.
_held: OrderedDict = OrderedDict()
_lock = threading.Lock()

_registered = False


def keep(key: tuple, found, folders: str, library) -> None:
    """Hold what a run read and built, so the panel's next search starts from it.

    Args:
        key: From :func:`settings_key`.
        found: Everything the run read.
        folders: The selection the run searched.
        library: The library the run built for that selection.
    """
    with _lock:
        entry = _held.pop(key, None) or {}
        entry.update({"read": found, "at": time.monotonic(), "all": None, "selected": {folders: library}})
        _held[key] = entry
        while len(_held) > MAX_HELD:
            _held.popitem(last=False)


def settings_key(root, captions: str, recursive: bool, max_files: int, text_model: str, picture_model: str) -> tuple:
    """The key :func:`keep` files a library under for one node's settings."""
    return (str(root).casefold(), captions, bool(recursive), int(max_files), text_model, picture_model)


def _text(value, limit: int) -> str:
    """A body field as text, cut to ``limit`` characters."""
    return value[:limit] if isinstance(value, str) else ""


def _number(value, default, low, high):
    """A body field as a number held to a range."""
    try:
        number = type(default)(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, number))


def _local(choice: str, empty: str) -> tuple[str, str | None]:
    """A model menu entry in models/LLM or a copy already on disk, as ``(label, problem)``."""
    from ..io import picker
    from ..llm import embeddings
    from ..model import resolve

    label = choice.strip()
    if label == empty:
        return label, None
    if picker.written_path(label) is not None:
        return label, "run the node to read it"
    if embeddings._local_folder(label) is not None:
        return label, None
    try:
        _found, cache_dir = resolve(embeddings.FOLDER, label)
    except Exception:
        return label, "run the node to fetch the model"
    return label, None if cache_dir is None else "run the node to fetch the model"


def answer(body: dict) -> dict:
    """The folder tree and the closest pictures for one panel request.

    Args:
        body: The request: ``folder``, ``folders``, ``captions``, ``recursive``, ``max_files``,
            ``text_model``, ``picture_model``, ``prompt``, ``view``, ``filter``, ``top_k``,
            ``one_per_group``, ``picture_weight`` and ``dimensions``, as the node's widgets hold them.

    Returns:
        ``{"folders", "matches", "searched", "total", "note"}``.
    """
    from ..constants import ALLOWED_EXT
    from ..image import image_library
    from ..io import picker
    from ..util import file_listing

    reply = {"folders": [], "matches": [], "searched": 0, "total": 0, "note": ""}
    label = _text(body.get("folder"), 1000)
    if not label or picker.written_path(label) is not None:
        reply["note"] = "pick a folder"
        return reply
    root = picker.resolve_folder(label)
    if root is None or not root.is_dir():
        reply["note"] = "that folder is gone"
        return reply
    captions = _text(body.get("captions"), 40)
    if captions not in image_library.CAPTION_SOURCES:
        captions = "auto"
    recursive = bool(body.get("recursive", True))
    max_files = _number(body.get("max_files"), 20000, 1, image_library.MAX_FILES)
    text_label, problem = _local(_text(body.get("text_model"), 400), image_library.KEYWORDS)
    picture_label, picture_problem = _local(_text(body.get("picture_model"), 400), image_library.NO_PICTURE_MODEL)
    key = settings_key(root, captions, recursive, max_files, text_label, picture_label)

    with _lock:
        entry = _held.get(key)
        if entry is not None:
            _held.move_to_end(key)
    if entry is None or time.monotonic() - entry["at"] > REREAD_AFTER:
        found = image_library.read(root, captions, recursive, max_files)
        entry = {"read": found, "at": time.monotonic(), "all": None,
                 "selected": dict(entry["selected"]) if entry else {}}
        with _lock:
            _held[key] = entry
            while len(_held) > MAX_HELD:
                _held.popitem(last=False)
    found = entry["read"]
    if problem or picture_problem:
        reply["folders"] = [
            {"label": unit.label, "count": len(unit.entries), "state": image_library.NOT_INDEXED, "source": unit.source}
            for unit in found.units[:MAX_FOLDERS]
        ]
        reply["note"] = problem or picture_problem
        return reply

    text = image_library.text_model(text_label, "Image Library Search")
    picture = image_library.picture_model(picture_label, "Image Library Search")
    if entry["all"] is None:
        entry["all"] = image_library.build(label, found, text, picture, embed=False)
    states = entry["all"].states
    reply["folders"] = [
        {"label": unit.label, "count": len(unit.entries), "state": states.get(unit.label, (image_library.NOT_INDEXED,))[0],
         "source": unit.source}
        for unit in found.units[:MAX_FOLDERS]
    ]

    folders = _text(body.get("folders"), MAX_FOLDERS_CHARS)
    library = entry["selected"].get(folders)
    if library is None:
        library = image_library.build(label, image_library.select(found, folders), text, picture, embed=False)
        entry["selected"][folders] = library
    reply["total"] = len(library.entries)
    prompt = _text(body.get("prompt"), image_library.MAX_PROMPT_CHARS)
    if not prompt.strip():
        reply["note"] = "write a prompt to search"
        return reply
    if not library.entries:
        reply["note"] = "run the node to index"
        return reply
    view = _text(body.get("view"), 20)
    matches, kept = image_library.search(
        library, prompt,
        _number(body.get("top_k"), 8, 1, 64),
        _number(body.get("picture_weight"), 0.35, 0.0, 1.0),
        _text(body.get("filter"), MAX_FILTER_CHARS),
        bool(body.get("one_per_group", True)),
        view if view in image_library.VIEW_CHOICES else "auto",
        _number(body.get("dimensions"), 0, 0, 4096),
    )
    reply["searched"] = kept
    for match in matches:
        try:
            listed = file_listing.typed(match.entry.preview, ALLOWED_EXT, file_listing.ROOTS)
        except Exception:
            listed = None
        reply["matches"].append({
            "key": match.entry.key,
            "label": listed.label if listed is not None else "",
            "name": match.entry.name,
            "caption": match.entry.caption,
            "score": round(match.score, 4),
            "view": match.entry.view,
        })
    return reply


def register_routes() -> bool:
    """Register the route the panel searches through.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case the panel shows the results of the last run.
    """
    global _registered
    if _registered:
        return False
    try:
        import asyncio

        from aiohttp import web
        from server import PromptServer

        from .channel import NO_STORE

        gate = asyncio.Semaphore(1)

        @PromptServer.instance.routes.post(ROUTE)
        async def post_image_library(request):
            if (request.content_length or 0) > MAX_BODY:
                return web.json_response({"error": "request too large"}, status=413, headers=NO_STORE)
            try:
                raw = await request.content.read(MAX_BODY + 1)
                body = json.loads(raw) if len(raw) <= MAX_BODY else None
            except (ValueError, UnicodeDecodeError):
                body = None
            if not isinstance(body, dict):
                return web.json_response({"error": "expected a JSON object"}, status=400, headers=NO_STORE)
            try:
                async with gate:
                    reply = await asyncio.to_thread(answer, body)
            except Exception as error:
                logger.warning("a picture library search failed (%s)", type(error).__name__)
                logger.debug("the picture library search failed", exc_info=True)
                return web.json_response({"error": "the search failed"}, status=500, headers=NO_STORE)
            return web.json_response(reply, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so the picture library panel shows only its last run",
            ROUTE, type(error).__name__, error,
        )
        return False
    _registered = True
    return True
