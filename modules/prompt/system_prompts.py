"""The system prompt library: ``.txt`` and ``.md`` files in ``system_prompts`` under the config directory.

Front matter between ``---`` lines carries ``name``, ``description``, ``model_family`` and
``tags``. A key is a path below the library, ``/`` separated.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from pathlib import Path
from typing import NamedTuple

from .. import log
from ..config import paths
from ..util import sandbox

__all__ = [
    "EXTENSIONS",
    "FILE_TYPES",
    "LIBRARY_NAME",
    "LISTING_TTL",
    "MAX_ENTRIES",
    "MAX_FILE_BYTES",
    "MAX_HEAD_BYTES",
    "NO_PROMPTS",
    "Document",
    "Entry",
    "Saved",
    "compose",
    "find",
    "invalidate",
    "labels",
    "library_directory",
    "library_label",
    "normalise_families",
    "read",
    "save",
    "scan",
    "split_front_matter",
]

logger = log.get_logger("prompt.system_prompts")

#: The library's folder inside the config directory.
LIBRARY_NAME = "system_prompts"

#: The file suffixes the library lists, lower case.
EXTENSIONS = (".txt", ".md")

#: The suffixes a save writes, as the words on the widget.
FILE_TYPES = ["txt", "md"]

#: The menu entry offered while the library holds nothing.
NO_PROMPTS = "No System Prompts"

#: The largest file listed and read, in bytes.
MAX_FILE_BYTES = 256 * 1024

#: Bytes at the top of a file that front matter must close within.
MAX_HEAD_BYTES = 16 * 1024

#: Files listed at most.
MAX_ENTRIES = 2000

#: Folder levels below the library that are walked.
MAX_DEPTH = 8

#: Seconds a listing is reused before the folder is walked again.
LISTING_TTL = 5.0

#: Characters kept of a file name stem, a name, a description, one family and one tag.
MAX_STEM_CHARS = 120
MAX_NAME_CHARS = 200
MAX_DESCRIPTION_CHARS = 500
MAX_FAMILY_CHARS = 40
MAX_TAG_CHARS = 40

#: Families and tags kept per entry.
MAX_FAMILIES = 16
MAX_TAGS = 32

#: Front matter keys read as the model family, first match wins.
FAMILY_KEYS = ("model_family", "model_families", "models", "family")

#: The file name a skill folder holds its instructions in, compared without case.
SKILL_FILE = "skill.md"

#: Characters a file name stem cannot hold on any system the pack runs on.
UNSAFE_STEM = frozenset('<>:"/\\|?*')

#: Lines that open and close front matter.
OPENER = "---"
CLOSERS = ("---", "...")


class Entry(NamedTuple):
    """One file in the library.

    Attributes:
        key: The path below the library, ``/`` separated, as the menu offers it.
        name: The front matter ``name``, or the file stem, or a skill folder's name.
        folder: The folder below the library, ``/`` separated, empty at the top.
        families: The model families it suits, as written.
        description: The front matter ``description``.
        tags: The front matter ``tags``.
        path: The absolute file path.
        size: Bytes on disk.
        modified: Modification time in seconds since the epoch.
        mtime_ns: Modification time in nanoseconds.
    """

    key: str
    name: str
    folder: str
    families: tuple
    description: str
    tags: tuple
    path: str
    size: int
    modified: float
    mtime_ns: int


class Document(NamedTuple):
    """A file's front matter and the text after it.

    Attributes:
        meta: The front matter mapping, empty when the file carries none.
        body: The text after the front matter.
    """

    meta: dict
    body: str


class Saved(NamedTuple):
    """What a save did.

    Attributes:
        entry: The library entry written or found.
        status: ``new``, ``replaced`` or ``unchanged``.
    """

    entry: Entry
    status: str


_lock = threading.RLock()

#: ``(monotonic time it is reused until, entries)``.
_listing: tuple[float, tuple[Entry, ...]] | None = None

#: ``(path, mtime_ns, size)`` -> the metadata read from that file.
_heads: dict[tuple[str, int, int], tuple[str, tuple, str, tuple]] = {}

#: ``(path, size)`` of each oversized file already reported.
_warned: set[tuple[str, int]] = set()


def library_directory() -> str:
    """The library folder, which need not exist yet.

    Returns:
        ``<config directory>/system_prompts`` as an absolute path.
    """
    return os.path.join(os.fspath(paths.config_directory()), LIBRARY_NAME)


def library_label() -> str:
    """The library folder as a short label.

    Returns:
        The path below ComfyUI's own folder with ``/`` separators, such as
        ``ComfyUI/user/was-node-suite/system_prompts``, or the absolute path when it lies elsewhere.
    """
    directory = library_directory()
    try:
        import folder_paths

        relative = os.path.relpath(os.path.realpath(directory), os.path.realpath(folder_paths.base_path))
    except Exception:
        return directory
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return directory
    return "ComfyUI/" + relative.replace(os.sep, "/")


def split_front_matter(text: str) -> tuple[dict | None, str]:
    """Separate front matter from the text after it.

    Args:
        text: A whole file, or its first :data:`MAX_HEAD_BYTES` bytes.

    Returns:
        ``(meta, body)``. ``meta`` is ``None`` and ``body`` the whole text when the text opens
        with no ``---`` line, the block does not close within :data:`MAX_HEAD_BYTES`
        characters, or it is not a YAML mapping.
    """
    import yaml

    source = text[1:] if text.startswith("﻿") else text
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].strip() != OPENER:
        return None, source
    consumed = len(lines[0])
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() in CLOSERS:
            block = "".join(lines[1:index])
            try:
                meta = yaml.safe_load(block) if block.strip() else {}
            except yaml.YAMLError as error:
                logger.debug("front matter could not be read (%s)", error)
                return None, source
            if not isinstance(meta, dict):
                return None, source
            body = "".join(lines[index + 1:])
            return meta, body.lstrip("\r\n")
        consumed += len(line)
        if consumed > MAX_HEAD_BYTES:
            break
    return None, source


def normalise_families(value) -> tuple:
    """Model families from a front matter value or a widget.

    Args:
        value: A string of names separated by commas or semicolons, a list of names, or
            nothing.

    Returns:
        The names, stripped, without repeats regardless of case, at most :data:`MAX_FAMILIES`.
    """
    return _names(value, MAX_FAMILIES, MAX_FAMILY_CHARS)


def compose(meta: dict, body: str) -> str:
    """A file holding front matter and a body.

    Args:
        meta: The front matter. Empty values are left out.
        body: The text after the front matter.

    Returns:
        The file's text, ending in one newline. No front matter block when ``meta`` holds
        nothing.
    """
    import yaml

    kept = {key: value for key, value in meta.items() if value not in ("", None, [], ())}
    text = body.strip("\r\n") + "\n"
    if not kept:
        return text
    block = yaml.safe_dump(kept, sort_keys=False, allow_unicode=True, default_flow_style=None, width=1000)
    return f"{OPENER}\n{block}{OPENER}\n\n{text}"


def scan() -> tuple[Entry, ...]:
    """Every file in the library, reused for :data:`LISTING_TTL` seconds.

    Returns:
        Entries ordered by folder and then name, without regard to case. Empty when the
        folder is missing or cannot be read.
    """
    global _listing
    with _lock:
        now = time.monotonic()
        if _listing is not None and _listing[0] > now:
            return _listing[1]
        found = _walk(library_directory())
        _listing = (time.monotonic() + LISTING_TTL, found)
        return found


def invalidate() -> None:
    """Drop the held listing, so the next call walks the folder again."""
    global _listing
    with _lock:
        _listing = None


def labels() -> list[str]:
    """The keys a menu offers.

    Returns:
        Every key, or ``[NO_PROMPTS]`` while the library is empty.
    """
    keys = [entry.key for entry in scan()]
    return keys or [NO_PROMPTS]


def find(key: str, fresh: bool = False) -> Entry | None:
    """The entry a key names.

    Args:
        key: A key from :func:`labels`.
        fresh: Walk the folder again before giving up on a key the held listing lacks.

    Returns:
        The entry, or ``None`` when no file in the library has that key.
    """
    wanted = str(key or "")
    for entry in scan():
        if entry.key == wanted:
            return entry
    if not fresh:
        return None
    invalidate()
    for entry in scan():
        if entry.key == wanted:
            return entry
    return None


def read(entry: Entry) -> Document:
    """Read one entry's front matter and text.

    Args:
        entry: An entry from :func:`scan`.

    Returns:
        The document.

    Raises:
        PathNotAllowed: The file resolves outside every permitted read root.
        ValueError: The file is gone, larger than :data:`MAX_FILE_BYTES`, or not UTF-8.
    """
    target = sandbox.resolve_read(entry.path)
    try:
        info = os.stat(target)
    except OSError as error:
        raise ValueError(
            f"the system prompt `{entry.key}` could not be read ({error}). It may have been "
            "moved or deleted; pick another, or press R to refresh the menu"
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"the system prompt `{entry.key}` is not a file")
    if info.st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"the system prompt `{entry.key}` is {info.st_size} bytes, over the "
            f"{MAX_FILE_BYTES} byte limit. Shorten it or split it into two files"
        )
    with open(target, "rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"the system prompt `{entry.key}` is not UTF-8 text. Save it again as UTF-8"
        ) from error
    meta, body = split_front_matter(text)
    return Document(meta or {}, body)


def save(
    text: str,
    name: str,
    folder: str = "",
    families="",
    description: str = "",
    file_type: str = "txt",
    overwrite: bool = False,
) -> Saved:
    """Write a system prompt into the library.

    Args:
        text: The system prompt. Front matter it opens with is merged under the other values.
        name: What the entry is called, also its file name.
        folder: A folder below the library, such as ``Captioning/Qwen``. Empty for the top.
        families: Model families, as :func:`normalise_families` takes them.
        description: One line saying what the prompt is for.
        file_type: ``txt`` or ``md``.
        overwrite: Replace a file of the same key holding different text.

    Returns:
        The entry and what was done.

    Raises:
        PathNotAllowed: The folder or name names somewhere outside the library.
        ValueError: The name or text is empty, the type is unknown, or a different file
            already has the key and ``overwrite`` is off.
    """
    wanted = " ".join(str(name or "").split())
    if not wanted:
        raise ValueError(
            "the name is empty, so the system prompt has nothing to be saved under. Type a "
            "name such as Detailed Caption"
        )
    kind = str(file_type or "").strip().lower()
    if kind not in FILE_TYPES:
        raise ValueError(f"file_type `{file_type}` is not one of {', '.join(FILE_TYPES)}")
    given_meta, body = split_front_matter(str(text or ""))
    if not body.strip():
        raise ValueError(
            "the system_prompt box is empty, so there is nothing to save. Type the "
            "instructions, or wire in the text to keep"
        )

    meta = dict(given_meta or {})
    meta["name"] = wanted[:MAX_NAME_CHARS]
    chosen = normalise_families(families)
    if chosen:
        for key in FAMILY_KEYS:
            meta.pop(key, None)
        meta["model_family"] = list(chosen)
    note = " ".join(str(description or "").split())
    if note:
        meta["description"] = note[:MAX_DESCRIPTION_CHARS]
    ordered = {key: meta[key] for key in ("name", "description", "model_family") if key in meta}
    ordered.update({key: value for key, value in meta.items() if key not in ordered})

    below = _folder(folder)
    relative = f"{below}/{_stem(wanted)}.{kind}" if below else f"{_stem(wanted)}.{kind}"
    root = sandbox.resolve_write(library_directory())
    target = sandbox.resolve_write_file(root, relative)
    if not sandbox.contains(root, target):
        raise sandbox.PathNotAllowed(f"refusing to write {target}\n  It is outside the system prompt library {root}.")

    content = compose(ordered, body)
    status = "new"
    if os.path.lexists(target):
        if not os.path.isfile(target):
            raise ValueError(f"`{relative}` in the system prompt library is not a plain file, so it was not replaced")
        with open(target, encoding="utf-8", errors="replace", newline="") as handle:
            held = handle.read(MAX_FILE_BYTES + 1)
        if held == content:
            status = "unchanged"
        elif not overwrite:
            raise ValueError(
                f"`{relative}` is already in the system prompt library with different text. "
                "Turn overwrite on to replace it, or type another name"
            )
        else:
            status = "replaced"
    if status != "unchanged":
        os.makedirs(os.path.dirname(target), exist_ok=True)
        mode = "w" if status == "replaced" else "x"
        with open(target, mode, encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    invalidate()
    entry = find(relative)
    if entry is None:
        info = os.stat(target)
        entry = Entry(
            relative, ordered["name"], below, tuple(ordered.get("model_family") or ()),
            str(ordered.get("description") or ""), (), os.fspath(target), info.st_size,
            info.st_mtime, info.st_mtime_ns,
        )
    return Saved(entry, status)


def _names(value, most: int, longest: int) -> tuple:
    """Distinct short names from a string or a list."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts = [str(part) for part in value if part is not None]
    else:
        parts = str(value).replace(";", ",").split(",")
    kept: list[str] = []
    seen: set[str] = set()
    for part in parts:
        word = " ".join(part.split())[:longest]
        if not word or word.casefold() in seen:
            continue
        seen.add(word.casefold())
        kept.append(word)
        if len(kept) >= most:
            break
    return tuple(kept)


def _stem(name: str) -> str:
    """A file name stem for an entry name.

    Raises:
        ValueError: Nothing usable is left of the name.
    """
    cleaned = "".join("_" if char in UNSAFE_STEM or ord(char) < 32 else char for char in name)
    cleaned = cleaned.strip(" .")[:MAX_STEM_CHARS].rstrip(" .")
    if not cleaned:
        raise ValueError(f"the name `{name}` leaves no usable file name. Use letters or numbers in it")
    return cleaned


def _folder(folder: str) -> str:
    """A folder below the library, ``/`` separated.

    Raises:
        PathNotAllowed: The folder names somewhere outside the library.
    """
    typed = str(folder or "").strip()
    if not typed.replace("\\", "/").strip("/"):
        return ""
    reason = sandbox.leaves_folder(typed)
    if reason is not None:
        raise sandbox.PathNotAllowed(
            f"refusing the folder `{folder}`: it {reason}. Type a folder inside the library, "
            "such as Captioning/Qwen"
        )
    text = typed.replace("\\", "/")
    return "/".join(part.strip() for part in text.split("/") if part.strip())


def _walk(top: str) -> tuple[Entry, ...]:
    """Every listable file below ``top``."""
    try:
        if not os.path.isdir(top):
            return ()
        root = os.path.realpath(top)
    except OSError:
        return ()
    found: list[Entry] = []
    pending: list[tuple[str, str, int]] = [(root, "", 0)]
    while pending and len(found) < MAX_ENTRIES:
        directory, below, depth = pending.pop()
        try:
            with os.scandir(directory) as listing:
                items = sorted(listing, key=lambda item: item.name.casefold())
        except OSError as error:
            logger.debug("%s could not be listed (%s)", directory, error)
            continue
        folders = []
        for item in items:
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            relative = f"{below}/{item.name}" if below else item.name
            try:
                if item.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        folders.append((item.path, relative, depth + 1))
                    continue
                if os.path.splitext(item.name)[1].lower() not in EXTENSIONS:
                    continue
                info = item.stat(follow_symlinks=True)
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_size > MAX_FILE_BYTES:
                if (item.path, info.st_size) not in _warned:
                    if len(_warned) >= MAX_ENTRIES:
                        _warned.clear()
                    _warned.add((item.path, info.st_size))
                    logger.warning(
                        "%s is %d bytes, over the %d byte limit for a system prompt, so it is not "
                        "listed. Shorten it or split it into two files",
                        item.path, info.st_size, MAX_FILE_BYTES,
                    )
                continue
            if not sandbox.contains(Path(root), Path(os.path.realpath(item.path))):
                continue
            name, families, description, tags = _head(item.path, info, below)
            found.append(Entry(
                relative, name, below, families, description, tags, item.path,
                info.st_size, info.st_mtime, info.st_mtime_ns,
            ))
            if len(found) >= MAX_ENTRIES:
                logger.warning(
                    "the system prompt library holds more than %d files, so only the first %d are listed",
                    MAX_ENTRIES, MAX_ENTRIES,
                )
                break
        pending.extend(reversed(folders))
    found.sort(key=lambda entry: (entry.folder.casefold(), entry.name.casefold(), entry.key.casefold()))
    return tuple(found)


def _head(path: str, info: os.stat_result, below: str) -> tuple[str, tuple, str, tuple]:
    """The name, families, description and tags of one file, held per file version."""
    key = (path, info.st_mtime_ns, info.st_size)
    held = _heads.get(key)
    if held is not None:
        return held
    meta: dict = {}
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_HEAD_BYTES)
        found, _body = split_front_matter(raw.decode("utf-8", errors="replace"))
        meta = found or {}
    except OSError as error:
        logger.debug("%s could not be read (%s)", path, error)
    stem, _suffix = os.path.splitext(os.path.basename(path))
    if os.path.basename(path).casefold() == SKILL_FILE and below:
        stem = below.rsplit("/", 1)[-1]
    name = " ".join(str(meta.get("name") or "").split())[:MAX_NAME_CHARS] or stem
    families: tuple = ()
    for family_key in FAMILY_KEYS:
        if meta.get(family_key):
            families = normalise_families(meta[family_key])
            break
    description = " ".join(str(meta.get("description") or "").split())[:MAX_DESCRIPTION_CHARS]
    tags = _names(meta.get("tags"), MAX_TAGS, MAX_TAG_CHARS)
    result = (name, families, description, tags)
    if len(_heads) > MAX_ENTRIES * 2:
        _heads.clear()
    _heads[key] = result
    return result
