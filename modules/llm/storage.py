"""Where the LLM nodes keep what they save: indexes and chat sessions, each under a name.

Everything lives under ``<config dir>/llm``: ``indexes/<name>/`` holds a persisted index and
``sessions/<name>.json`` a conversation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from .. import log
from ..config import paths
from ..util import sandbox

__all__ = [
    "INDEXES",
    "INDEX_FILES",
    "INDEX_MARKER",
    "MAX_COUNTED_BYTES",
    "MAX_LISTED",
    "MAX_SESSION_BYTES",
    "SESSIONS",
    "check_name",
    "clear_index",
    "delete_session",
    "folder",
    "index_dir",
    "index_info",
    "mark_index",
    "names",
    "read_session",
    "rename_session",
    "root",
    "session_file",
    "session_rows",
    "writable",
    "write_json",
    "write_session",
]

logger = log.get_logger("llm.storage")

#: The folder under the pack's config directory.
ROOT_NAME = "llm"

#: Saved indexes, one folder each.
INDEXES = "indexes"

#: Saved conversations, one ``<name>.json`` file each, with a ``<name>.<block>.vectors.json``
#: beside it for each vector memory block.
SESSIONS = "sessions"

#: The file that marks a folder as an index this pack saved.
INDEX_MARKER = "was_index.json"

#: The files a llama-index storage context writes into an index folder, besides the marker.
INDEX_FILES = (
    "docstore.json",
    "index_store.json",
    "graph_store.json",
    "property_graph_store.json",
    "default__vector_store.json",
    "image__vector_store.json",
    "vector_store.json",
    "bm25.json",
)

#: A saved name: letters, digits, spaces, dots, dashes and underscores, up to 64 characters.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")

#: Names Windows opens as a device in every folder.
_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{d}" for d in range(10)), *(f"LPT{d}" for d in range(10))}
)

#: Bytes of a session file read at most.
MAX_SESSION_BYTES = 64 * 1024 * 1024

#: Bytes of an index marker read at most.
MAX_MARKER_BYTES = 64 * 1024

#: Saved conversations one listing answers at most.
MAX_LISTED = 500

#: Bytes of a session file a listing reads to count its messages.
MAX_COUNTED_BYTES = 4 * 1024 * 1024

_lock = threading.Lock()
_listings: dict[str, tuple[float, tuple[str, ...]]] = {}


def root() -> Path:
    """``<config dir>/llm``."""
    return paths.config_directory() / ROOT_NAME


def folder(kind: str) -> Path:
    """The folder holding one kind of saved thing.

    Args:
        kind: :data:`INDEXES` or :data:`SESSIONS`.
    """
    if kind not in (INDEXES, SESSIONS):
        raise ValueError(f"unknown kind {kind!r}")
    return root() / kind


def check_name(name, what: str = "The") -> str:
    """A saved name, trimmed, once it is safe to use as one file or folder name.

    Args:
        name: The name as typed.
        what: What is being named, for the message, such as ``"Index"``.

    Returns:
        The trimmed name.

    Raises:
        ValueError: The name is empty, too long, or holds a character outside the set allowed.
    """
    text = str(name or "").strip()
    stem = text.split(".", 1)[0].strip().upper()
    if NAME.fullmatch(text) and not text.endswith((".", " ")) and stem not in _DEVICES:
        return text
    raise ValueError(
        f"{what} name {text[:80]!r} cannot be used. Use 1 to 64 letters, digits, spaces, dots, "
        f"dashes or underscores, starting with a letter or digit, such as 'project notes'."
    )


def names(kind: str) -> tuple[str, ...]:
    """Every saved name of one kind, sorted, from a listing kept until the folder changes.

    Args:
        kind: :data:`INDEXES` or :data:`SESSIONS`.

    Returns:
        Names, empty when the folder does not exist yet.
    """
    directory = folder(kind)
    try:
        stamp = os.stat(directory).st_mtime
    except OSError:
        return ()
    with _lock:
        held = _listings.get(kind)
        if held is not None and held[0] == stamp:
            return held[1]
    found: list[str] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if _linked(entry.path):
                    continue
                if kind == INDEXES:
                    if entry.is_dir(follow_symlinks=False) and os.path.isfile(
                        os.path.join(entry.path, INDEX_MARKER)
                    ):
                        found.append(entry.name)
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
                    stem = entry.name[: -len(".json")]
                    if not stem.endswith(".vectors"):
                        found.append(stem)
    except OSError as error:
        logger.debug("the %s folder could not be listed (%s)", kind, error)
        return ()
    listing = tuple(sorted(name for name in found if NAME.fullmatch(name)))
    with _lock:
        _listings[kind] = (stamp, listing)
    return listing


def _forget(kind: str) -> None:
    """Drop the held listing of one kind."""
    with _lock:
        _listings.pop(kind, None)


def writable(kind: str, filename: str) -> Path:
    """A file's place inside one kind's folder, contained and with its folder made.

    Args:
        kind: :data:`INDEXES` or :data:`SESSIONS`.
        filename: The file's name inside that folder, which may hold one checked subfolder.

    Returns:
        The resolved path. The file need not exist.

    Raises:
        PathNotAllowed: The place falls outside the pack's config directory.
    """
    target = sandbox.resolve_write_file(folder(kind), filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(kind: str, filename: str, data) -> None:
    """Write JSON beside its destination inside one kind's folder, then move it into place.

    Args:
        kind: :data:`INDEXES` or :data:`SESSIONS`.
        filename: The file's name inside that folder.
        data: Anything ``json`` serialises.
    """
    path = writable(kind, filename)
    draft = sandbox.resolve_write_file(path.parent, f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.part")
    try:
        with open(draft, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
        os.replace(draft, path)
    finally:
        if draft.exists():
            draft.unlink()


def _read_json(path: Path, limit: int):
    """A JSON file's value, or ``None`` when it is absent.

    Raises:
        ValueError: The file is larger than ``limit`` bytes or is not JSON.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size > limit:
        raise ValueError(f"{path.name} is {size} bytes, past the {limit} bytes read at most.")
    with open(path, encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} is not readable JSON ({error.msg}).") from None


def _linked(path) -> bool:
    """Whether a path is a symlink or a Windows junction."""
    target = os.fspath(path)
    junction = getattr(os.path, "isjunction", None)
    return os.path.islink(target) or bool(junction is not None and junction(target))


def index_dir(name: str) -> Path:
    """The folder an index is saved in, by its checked name."""
    return folder(INDEXES) / check_name(name, "Index")


def index_info(name: str) -> dict | None:
    """What an index's marker records, or ``None`` when no index is saved under the name."""
    if _linked(index_dir(name)):
        return None
    data = _read_json(index_dir(name) / INDEX_MARKER, MAX_MARKER_BYTES)
    return data if isinstance(data, dict) else None


def clear_index(name: str) -> Path:
    """Empty a saved index's folder of the files the pack wrote, ready for a new save.

    Args:
        name: The index's name.

    Returns:
        The folder, created when it did not exist.

    Raises:
        ValueError: The folder exists and holds no marker, so the pack did not write it.
    """
    directory = sandbox.resolve_write_file(folder(INDEXES), check_name(name, "Index"))
    if _linked(index_dir(name)):
        raise ValueError(f"The index folder for {name!r} is a link, so nothing was saved into it.")
    if directory.exists():
        if not sandbox.resolve_write_file(directory, INDEX_MARKER).is_file():
            raise ValueError(
                f"A folder named {name!r} already sits among the saved indexes and was not "
                f"written by this pack, so it was left alone. Choose another name."
            )
        for filename in (*INDEX_FILES, INDEX_MARKER):
            target = sandbox.resolve_write_file(directory, filename)
            if target.is_file() and not target.is_symlink():
                target.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    _forget(INDEXES)
    return directory


def mark_index(name: str, info: dict) -> None:
    """Write the marker that records a saved index's kind and models."""
    write_json(INDEXES, f"{check_name(name, 'Index')}/{INDEX_MARKER}", info)
    _forget(INDEXES)


def session_file(name: str, suffix: str = ".json") -> Path:
    """Where a session is saved, by its checked name."""
    return folder(SESSIONS) / f"{check_name(name, 'Session')}{suffix}"


def read_session(name: str) -> dict | None:
    """A saved session, or ``None`` when none is saved under the name."""
    data = _read_json(session_file(name), MAX_SESSION_BYTES)
    return data if isinstance(data, dict) else None


def write_session(name: str, data: dict) -> None:
    """Save a session under its name."""
    write_json(SESSIONS, f"{check_name(name, 'Session')}.json", data)
    _forget(SESSIONS)


def session_rows(limit: int = MAX_LISTED) -> list[dict]:
    """Every saved conversation's name, message count and saved time, newest first.

    Args:
        limit: The most conversations answered.

    Returns:
        ``{"name", "messages", "updated"}`` rows. ``messages`` is ``None`` for a file larger
        than :data:`MAX_COUNTED_BYTES` or one that does not read.
    """
    directory = folder(SESSIONS)
    stamped = []
    for name in names(SESSIONS):
        try:
            held = os.stat(directory / f"{name}.json")
        except OSError:
            continue
        stamped.append((held.st_mtime, held.st_size, name))
    stamped.sort(reverse=True)
    rows = []
    for stamp, size, name in stamped[: max(0, int(limit))]:
        count = None
        if size <= MAX_COUNTED_BYTES:
            try:
                saved = read_session(name) or {}
                count = sum(len(saved[key]) for key in ("archived", "active") if isinstance(saved.get(key), list))
            except (OSError, ValueError):
                count = None
        rows.append({"name": name, "messages": count, "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))})
    return rows


def rename_session(name: str, to: str) -> None:
    """Give a saved conversation a new name, with the vector memory files it records.

    Args:
        name: The conversation's name.
        to: Its new name.

    Raises:
        ValueError: A name cannot be used, nothing is saved under ``name``, or ``to`` is taken.
    """
    name = check_name(name, "Session")
    to = check_name(to, "Session")
    if name == to:
        return
    saved = read_session(name)
    if saved is None:
        raise ValueError(f"No conversation is saved as {name!r}.")
    if session_file(to).exists():
        raise ValueError(f"A conversation is already saved as {to!r}. Choose another name.")
    moves = []
    renamed = []
    for filename in saved.get("vectors", ()) if isinstance(saved.get("vectors"), list) else ():
        text = str(filename)
        if text.startswith(f"{name}.") and text.endswith(".vectors.json") and "/" not in text and "\\" not in text:
            target = f"{to}.{text[len(name) + 1:]}"
            moves.append((text, target))
            renamed.append(target)
    saved["vectors"] = renamed
    for old, new in moves:
        source = sandbox.resolve_write_file(folder(SESSIONS), old)
        if source.is_file() and not source.is_symlink():
            os.replace(source, sandbox.resolve_write_file(folder(SESSIONS), new))
    write_json(SESSIONS, f"{to}.json", saved)
    old_file = sandbox.resolve_write_file(folder(SESSIONS), f"{name}.json")
    if old_file.is_file() and not old_file.is_symlink():
        old_file.unlink()
    _forget(SESSIONS)


def delete_session(name: str) -> bool:
    """Remove a saved session and the vector memory files it records, all written by the pack.

    Returns:
        True when anything was removed.
    """
    name = check_name(name, "Session")
    try:
        saved = read_session(name) or {}
    except ValueError:
        saved = {}
    prefix = f"{name}."
    targets = [session_file(name)]
    for filename in saved.get("vectors", ()) if isinstance(saved.get("vectors"), list) else ():
        text = str(filename)
        if text.startswith(prefix) and text.endswith(".vectors.json") and "/" not in text and "\\" not in text:
            targets.append(folder(SESSIONS) / text)
    removed = False
    for target in targets:
        if target.is_file() and not target.is_symlink():
            target.unlink()
            removed = True
    _forget(SESSIONS)
    return removed
