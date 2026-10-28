"""The system prompt library as JSON a node interface draws, and the folder import it sends.

``GET /was/interface/api/system_prompts`` lists the entries, or with ``key`` answers one with
its text. ``POST`` imports a folder or copies a built-in.
"""

from __future__ import annotations

import asyncio
import json

from .. import log
from ..prompt import system_prompts
from .channel import NO_STORE, REFUSAL_HEADER

__all__ = [
    "COPY_FAILED",
    "IMPORT_FAILED",
    "MAX_BODY_BYTES",
    "MAX_KEY_CHARS",
    "MAX_LISTED_FILES",
    "MAX_TEXT_CHARS",
    "REFUSAL",
    "ROUTE",
    "WORKERS",
    "entry_fields",
    "entry_payload",
    "import_payload",
    "copy_payload",
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

#: Requests walking, reading or writing the library at once.
WORKERS = 2

#: Largest import body read, in bytes: twice the text one import writes, for escaping.
MAX_BODY_BYTES = 32 * 1024 * 1024

#: Files of a skill one entry answer names, at most.
MAX_LISTED_FILES = system_prompts.MAX_SKILL_FILES

#: What an import that failed for a reason outside the request answers.
IMPORT_FAILED = "the folder could not be imported. The server log names the cause"

#: What a copy that failed for a reason outside the request answers.
COPY_FAILED = "the built-in prompt could not be copied. The server log names the cause"

#: The words a refusal carries.
REFUSAL = "not in the system prompt library"

_registered = False


def entry_fields(entry: system_prompts.Entry) -> dict:
    """One entry as the object the route answers with.

    Args:
        entry: An entry of the library.

    Returns:
        ``{"key", "name", "folder", "families", "description", "tags", "size", "modified",
        "skill", "files", "builtin"}``, ``modified`` in seconds since the epoch, ``files`` the
        count of a skill's other files and ``builtin`` true for one of the pack's own prompts.
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
        "skill": bool(entry.skill),
        "files": int(entry.files),
        "builtin": bool(entry.builtin),
    }


def listing_payload() -> dict:
    """Every entry of the library. Never raises.

    Returns:
        ``{"entries", "total", "limit", "builtin_folder", "import"}``, ``limit`` the most
        entries a listing holds, ``builtin_folder`` the folder the pack's own prompts are listed
        under and ``import`` the suffixes, names and sizes a folder import takes.
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
        "builtin_folder": system_prompts.BUILTIN_FOLDER,
        "import": {
            "prompt_extensions": list(system_prompts.IMPORT_EXTENSIONS),
            "skill_extensions": list(system_prompts.SKILL_EXTENSIONS),
            "skipped_folders": sorted(system_prompts.SKIPPED_FOLDERS),
            "max_files": system_prompts.MAX_IMPORT_FILES,
            "max_bytes": system_prompts.MAX_IMPORT_BYTES,
            "max_file_bytes": system_prompts.MAX_SKILL_FILE_BYTES,
            "max_depth": system_prompts.MAX_DEPTH,
        },
    }


def entry_payload(key) -> dict | None:
    """One entry, its text and a skill's files. Never raises.

    Args:
        key: A key of the listing.

    Returns:
        ``{"entry", "text", "truncated", "files"}``, ``files`` a skill's other files as
        ``{"path", "size"}``; or ``None`` when no entry has that key or its file cannot be read.
    """
    wanted = str(key or "")[:MAX_KEY_CHARS]
    try:
        entry = system_prompts.find(wanted)
        if entry is None:
            return None
        document = system_prompts.read(entry)
        files, _skipped = system_prompts.skill_files(entry)
    except Exception as error:
        logger.debug("%s could not read %s (%s)", ROUTE, wanted, error)
        return None
    body = document.body.strip()
    return {
        "entry": entry_fields(entry),
        "text": body[:MAX_TEXT_CHARS],
        "truncated": len(body) > MAX_TEXT_CHARS,
        "files": [{"path": found.relative, "size": int(found.size)} for found in files[:MAX_LISTED_FILES]],
    }


def copy_payload(sent: dict) -> tuple[int, dict]:
    """Copy the built-in prompt a request names into the library. Never raises.

    Args:
        sent: The request body, ``{"copy": key, "replace": bool}``.

    Returns:
        ``(status, payload)``: 200 with ``{"key", "written", "replaced", "unchanged"}``, ``key``
        the copy's; 409 with ``{"error", "exists"}`` when a file at the copy's path holds other
        text and ``replace`` is not true; otherwise 400 with ``{"error"}``.
    """
    key = sent.get("copy")
    if not isinstance(key, str) or not key:
        return 400, {"error": "copy has to name a built-in prompt"}
    try:
        done = system_prompts.copy_builtin(key[:MAX_KEY_CHARS], replace=sent.get("replace") is True)
    except system_prompts.FolderExists as error:
        return 409, {"error": str(error), "exists": True}
    except system_prompts.ImportRefused as error:
        return 400, {"error": str(error)}
    except Exception as error:
        logger.warning("%s could not copy a built-in prompt (%s)", ROUTE, type(error).__name__)
        logger.debug("%s copy failed", ROUTE, exc_info=True)
        return 400, {"error": COPY_FAILED}
    return 200, {"key": done.key, "written": done.written, "replaced": done.replaced, "unchanged": done.unchanged}


def import_payload(body: bytes) -> tuple[int, dict]:
    """Write the folder an import body carries, or the copy of a built-in prompt it names. Never raises.

    Args:
        body: The request body, ``{"folder", "replace", "files": [{"path", "text"}]}``, or
            ``{"copy", "replace"}`` for :func:`copy_payload`.

    Returns:
        ``(status, payload)``: 200 with ``{"folder", "skills", "written", "replaced",
        "unchanged", "skipped"}``; 409 with ``{"error", "exists"}`` when the folder is in the
        library and ``replace`` is not true; otherwise 400 with ``{"error"}``.
    """
    try:
        sent = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return 400, {"error": "the body has to be JSON"}
    if not isinstance(sent, dict):
        return 400, {"error": "the body has to be a JSON object"}
    if "copy" in sent:
        return copy_payload(sent)
    listed = sent.get("files")
    if not isinstance(listed, list):
        return 400, {"error": "files has to be a list"}
    if len(listed) > system_prompts.MAX_IMPORT_FILES:
        return 400, {"error": f"one import takes at most {system_prompts.MAX_IMPORT_FILES} files"}
    pairs = []
    for item in listed:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("text"), str):
            return 400, {"error": "each file has to carry a path and its text"}
        pairs.append((item["path"], item["text"]))
    try:
        done = system_prompts.import_folder(sent.get("folder"), pairs, replace=sent.get("replace") is True)
    except system_prompts.FolderExists as error:
        return 409, {"error": str(error), "exists": True}
    except system_prompts.ImportRefused as error:
        return 400, {"error": str(error)}
    except Exception as error:
        logger.warning("%s could not import a folder (%s)", ROUTE, type(error).__name__)
        logger.debug("%s import failed", ROUTE, exc_info=True)
        return 400, {"error": IMPORT_FAILED}
    return 200, {
        "folder": done.folder,
        "skills": list(done.skills),
        "written": done.written,
        "replaced": done.replaced,
        "unchanged": done.unchanged,
        "skipped": done.skipped,
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

        @PromptServer.instance.routes.post(ROUTE)
        async def post_system_prompts(request):
            # The declared length is checked before any of the body is read.
            length = request.content_length
            if length is None:
                return web.json_response({"error": "the body has to state its length"}, status=411, headers=NO_STORE)
            if length > MAX_BODY_BYTES:
                return web.json_response(
                    {"error": f"the import is {length} bytes, past the {MAX_BODY_BYTES} one import takes"},
                    status=413, headers=NO_STORE,
                )
            body = await request.read()
            async with gate:
                status, payload = await asyncio.to_thread(import_payload, body)
            return web.json_response(payload, status=status, headers=NO_STORE)

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
