"""Files a panel plays or compares, kept in the state directory through a restart.

A kept file is named by its content hash. The folder holds :data:`MAX_BYTES` and
:data:`MAX_FILES` at most, least recently used out first; :data:`ROUTE` serves one.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import shutil
import threading
import time

from .. import log
from ..config import paths
from ..util import sandbox
from .channel import NO_STORE, REFUSAL_HEADER

__all__ = [
    "FOLDER_NAME",
    "KEPT_TYPE",
    "MAX_BYTES",
    "MAX_FILES",
    "MEDIA_TYPES",
    "NAME",
    "ROUTE",
    "digest",
    "folder",
    "keep",
    "keep_bytes",
    "keep_file",
    "keep_ui",
    "path_of",
    "prune",
    "register_routes",
]

logger = log.get_logger("interface.compare_files")

#: The route serving a kept file by name.
ROUTE = "/was/interface/api/compare_file"

#: The folder in the state directory files are kept in.
FOLDER_NAME = "compare"

#: The ``type`` a kept entry carries in a node's ``ui``.
KEPT_TYPE = "was_compare"

#: Bytes the folder holds before the least recently used files are removed.
MAX_BYTES = 2 * 1024 ** 3

#: Files the folder holds before the least recently used are removed.
MAX_FILES = 2000

#: Entries of the folder one prune reads.
MAX_LISTED = MAX_FILES * 4

#: Suffixes kept, each with the media type it is served as.
MEDIA_TYPES = {".mp4": "video/mp4", ".png": "image/png", ".wav": "audio/wav"}

#: The shape of a kept file's name.
NAME = re.compile(r"[0-9a-f]{32}\.(?:mp4|png|wav)")

#: Seconds after which serving a file marks it as used again.
TOUCH_SECONDS = 3600

#: Seconds a leftover ``.part`` file is kept before a prune removes it.
PART_SECONDS = 3600

#: Requests resolving a file at once.
WORKERS = 4

#: Bytes hashed per read.
CHUNK = 1 << 20

#: Headers a kept file is served with.
CACHED = {"Cache-Control": "private, max-age=31536000, immutable"}

#: The words a refusal carries.
REFUSAL = "not a kept compare file"

_lock = threading.Lock()
_registered = False


def folder() -> str:
    """The folder files are kept in, which need not exist yet.

    Returns:
        ``<config directory>/compare`` as an absolute path.
    """
    return os.path.join(os.fspath(paths.config_directory()), FOLDER_NAME)


def digest(path: str) -> str:
    """The first 32 hex digits of a file's SHA-256.

    Args:
        path: The file.

    Returns:
        32 lowercase hex digits.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()[:32]


def _reason(error: Exception) -> str:
    """An error's cause without the paths an OSError carries."""
    if isinstance(error, OSError):
        return error.strerror or "the file system refused it"
    return "the folder could not be written"


def keep_file(source: str) -> str | None:
    """Keep a written file under its content hash, linked where the volume allows, else copied.

    Args:
        source: A file a node wrote, ending in a suffix of :data:`MEDIA_TYPES`.

    Returns:
        The kept name, or None when the suffix is not kept or the file could not be kept.
    """
    suffix = os.path.splitext(source)[1].lower()
    if suffix not in MEDIA_TYPES:
        return None
    try:
        name = digest(source) + suffix
        target = os.fspath(sandbox.resolve_write_file(folder(), name))
        with _lock:
            if os.path.isfile(target):
                os.utime(target)
                return name
            os.makedirs(os.path.dirname(target), exist_ok=True)
            part = f"{target}.part"
            with contextlib.suppress(FileNotFoundError):
                os.remove(part)
            try:
                os.link(source, part)
            except OSError:
                shutil.copyfile(source, part)
            os.replace(part, target)
            os.utime(target)
        return name
    except Exception as error:
        logger.warning(
            "%s was not kept for its compare panel (%s: %s), so the panel shows it only until "
            "ComfyUI restarts",
            os.path.basename(source), type(error).__name__, _reason(error),
        )
        logger.debug("%s could not be kept", source, exc_info=True)
        return None


def keep_bytes(data: bytes, suffix: str) -> str | None:
    """Keep bytes a node encoded under their content hash, writing nothing when already kept.

    Args:
        data: The encoded file.
        suffix: Its suffix, one of :data:`MEDIA_TYPES`.

    Returns:
        The kept name, or None when the suffix is not kept or the file could not be written.
    """
    suffix = str(suffix).lower()
    if suffix not in MEDIA_TYPES:
        return None
    try:
        name = hashlib.sha256(data).hexdigest()[:32] + suffix
        target = os.fspath(sandbox.resolve_write_file(folder(), name))
        with _lock:
            if os.path.isfile(target):
                os.utime(target)
                return name
            os.makedirs(os.path.dirname(target), exist_ok=True)
            part = f"{target}.part"
            with open(part, "wb") as handle:
                handle.write(data)
            os.replace(part, target)
        return name
    except Exception as error:
        logger.warning(
            "a %s file was not kept for its panel (%s: %s), so the panel shows nothing for it",
            suffix, type(error).__name__, _reason(error),
        )
        logger.debug("bytes could not be kept", exc_info=True)
        return None


def keep(entries) -> list[dict]:
    """Keep every temp file a ``ui`` list names, and point its entries at the kept copies.

    Args:
        entries: Entries as a node writes them for a panel: ``filename``, ``subfolder`` and
            ``type`` ``"temp"``, with any other fields.

    Returns:
        The entries in order. A kept one carries its kept name as ``filename``, an empty
        ``subfolder`` and :data:`KEPT_TYPE` as ``type``; one that could not be kept is
        returned unchanged.
    """
    out = []
    for entry in entries or ():
        if not isinstance(entry, dict) or entry.get("type") != "temp":
            out.append(entry)
            continue
        relative = "/".join(
            part for part in (str(entry.get("subfolder") or ""), str(entry.get("filename") or ""))
            if part
        )
        if not relative or sandbox.leaves_folder(relative):
            out.append(entry)
            continue
        try:
            import folder_paths

            placed = os.path.join(folder_paths.get_temp_directory(), relative)
            source = os.fspath(sandbox.resolve_read(placed))
        except Exception as error:
            logger.debug("a compare entry could not be resolved (%s)", error)
            out.append(entry)
            continue
        name = keep_file(source) if os.path.isfile(source) else None
        out.append(entry if name is None else {
            **entry, "filename": name, "subfolder": "", "type": KEPT_TYPE,
        })
    return out


def keep_ui(ui: dict) -> dict:
    """Keep every temp file a node's ``ui`` names, then prune the folder.

    Args:
        ui: The mapping a node returns as ``ui``, each value a list of entries.

    Returns:
        The same keys, each list passed through :func:`keep`. Unchanged while the route is not
        registered.
    """
    if not _registered:
        return dict(ui)
    out = {key: keep(value) if isinstance(value, list) else value for key, value in ui.items()}
    protect = {
        entry.get("filename")
        for value in out.values() if isinstance(value, list)
        for entry in value if isinstance(entry, dict) and entry.get("type") == KEPT_TYPE
    }
    try:
        prune(protect)
    except Exception as error:
        logger.debug("the compare folder could not be pruned (%s)", error)
    return out


def _remove(path: str) -> bool:
    """Remove one file the folder holds.

    Args:
        path: The file.

    Returns:
        True when it was removed. A file open elsewhere, as one being served is on Windows,
        is left.
    """
    try:
        os.remove(path)
        return True
    except OSError as error:
        logger.debug("%s was not removed (%s)", path, error)
        return False


def prune(protect=()) -> int:
    """Remove the least recently used kept files past :data:`MAX_BYTES` or :data:`MAX_FILES`.

    Args:
        protect: Names never removed, such as the files a run has just kept.

    Returns:
        How many files were removed, leftover ``.part`` files included.
    """
    directory = folder()
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    held = []
    removed = 0
    with _lock:
        with os.scandir(directory) as listing:
            for index, item in enumerate(listing):
                if index >= MAX_LISTED:
                    break
                if not item.is_file(follow_symlinks=False):
                    continue
                stats = item.stat(follow_symlinks=False)
                if item.name.endswith(".part"):
                    stale = now - stats.st_mtime > PART_SECONDS
                    if stale and NAME.fullmatch(item.name[:-len(".part")]):
                        removed += _remove(item.path)
                    continue
                if NAME.fullmatch(item.name):
                    held.append((stats.st_mtime, stats.st_size, item.name, item.path))
        held.sort()
        total = sum(size for _, size, _, _ in held)
        count = len(held)
        for _, size, name, path in held:
            if total <= MAX_BYTES and count <= MAX_FILES:
                break
            if name in protect or not _remove(path):
                continue
            removed += 1
            total -= size
            count -= 1
    if removed:
        logger.info(
            "removed %d least recently used compare file(s), %d left holding %.1f MB",
            removed, count, total / 1e6,
        )
    return removed


def path_of(name) -> str | None:
    """The kept file a name refers to, marked as used. Never raises.

    Args:
        name: A kept name, as :data:`NAME` matches.

    Returns:
        Its absolute path, or None when the name has another shape or no such file is kept.
    """
    text = str(name or "")
    if not NAME.fullmatch(text):
        return None
    try:
        placed = os.path.join(folder(), text)
        if os.path.islink(placed):
            return None
        path = os.fspath(sandbox.resolve_read(placed))
        if not os.path.isfile(path):
            return None
        if time.time() - os.path.getmtime(path) > TOUCH_SECONDS:
            os.utime(path)
        return path
    except Exception as error:
        logger.debug("%s could not be resolved (%s)", text, error)
        return None


def register_routes() -> bool:
    """Register the route serving kept files.

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
        async def get_compare_file(request):
            async with gate:
                path = await asyncio.to_thread(path_of, request.query.get("name"))
            if path is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: REFUSAL})
            kind = MEDIA_TYPES[os.path.splitext(path)[1].lower()]
            return web.FileResponse(path, headers={**CACHED, "Content-Type": kind})

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so compare panels show their files only until "
            "ComfyUI restarts",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving kept compare files", ROUTE)
    return True
