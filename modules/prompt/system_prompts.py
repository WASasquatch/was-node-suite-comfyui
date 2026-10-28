"""System prompts: ``.txt`` and ``.md`` files in the config directory's ``system_prompts``, and the pack's own under ``Built-in/``.

Front matter carries ``name``, ``description``, ``model_family`` and ``tags``. A key is a ``/``
separated path. A ``SKILL.md`` folder is one skill.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from .. import log
from ..config import paths
from ..util import file_listing, sandbox

__all__ = [
    "BUILTIN_FOLDER",
    "EXTENSIONS",
    "FILE_TYPES",
    "IMPORT_EXTENSIONS",
    "LIBRARY_NAME",
    "LISTING_TTL",
    "MAX_ENTRIES",
    "MAX_FILE_BYTES",
    "MAX_HEAD_BYTES",
    "MAX_IMPORT_FILES",
    "MAX_SKILL_BYTES",
    "MAX_SKILL_FILES",
    "MAX_SKILL_FILE_BYTES",
    "NO_PROMPTS",
    "SKILL_EXTENSIONS",
    "Copied",
    "CopyExists",
    "Document",
    "Entry",
    "FolderExists",
    "ImportRefused",
    "Imported",
    "Saved",
    "SkillFile",
    "builtin_directory",
    "compose",
    "copy_builtin",
    "find",
    "import_folder",
    "invalidate",
    "labels",
    "library_directory",
    "library_label",
    "normalise_families",
    "read",
    "read_skill_files",
    "save",
    "scan",
    "skill_files",
    "skill_stamp",
    "split_front_matter",
]

logger = log.get_logger("prompt.system_prompts")

#: The library's folder inside the config directory, and the pack's own inside its data folder.
LIBRARY_NAME = "system_prompts"

#: The folder the pack's own prompts are listed under, the first segment of their keys.
BUILTIN_FOLDER = "Built-in"

#: The pack's own prompts listed at most.
MAX_BUILTIN_ENTRIES = 500

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

#: Suffixes of a skill's own files that are read and imported, lower case.
SKILL_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".rst", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ini",
    ".cfg", ".csv", ".tsv", ".xml", ".html", ".htm", ".css", ".py", ".js", ".mjs", ".ts", ".sh",
    ".ps1", ".bat", ".sql", ".lua", ".r",
)

#: Suffixes an imported folder keeps outside a skill: the library's own.
IMPORT_EXTENSIONS = EXTENSIONS

#: Files one skill holds besides ``SKILL.md``, at most.
MAX_SKILL_FILES = 200

#: Bytes of one skill file, at most.
MAX_SKILL_FILE_BYTES = 1024 * 1024

#: Bytes of one skill's files together, at most.
MAX_SKILL_BYTES = 8 * 1024 * 1024

#: Files one import writes, at most.
MAX_IMPORT_FILES = 400

#: Bytes one import writes, at most.
MAX_IMPORT_BYTES = 16 * 1024 * 1024

#: Characters of one imported path, at most.
MAX_IMPORT_PATH_CHARS = 400

#: Directory entries one skill's walk examines before it stops.
MAX_SKILL_EXAMINED = 5000

#: Folder names passed over in a skill and refused in an import.
SKIPPED_FOLDERS = frozenset({"__pycache__", "node_modules"})

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
        skill: True for the ``SKILL.md`` of a skill folder.
        files: The skill's other files, 0 for a plain prompt.
        builtin: True for one of the pack's own read-only prompts.
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
    skill: bool = False
    files: int = 0
    builtin: bool = False


class SkillFile(NamedTuple):
    """One file of a skill besides its ``SKILL.md``.

    Attributes:
        relative: The path below the skill folder, ``/`` separated, such as ``references/layout.md``.
        path: The absolute file path.
        size: Bytes on disk.
        mtime_ns: Modification time in nanoseconds.
    """

    relative: str
    path: str
    size: int
    mtime_ns: int


class Imported(NamedTuple):
    """What an import wrote.

    Attributes:
        folder: The folder below the library it went into, ``/`` separated.
        skills: Keys of the skills it holds, each a ``SKILL.md``.
        written: Files written new.
        replaced: Files written over.
        unchanged: Files already holding the same text.
        skipped: Reason to the count of files passed over.
    """

    folder: str
    skills: tuple
    written: int
    replaced: int
    unchanged: int
    skipped: dict


class ImportRefused(ValueError):
    """An import refused before anything was written. The message names no path on disk."""


class FolderExists(ImportRefused):
    """The folder an import names is already in the library."""


class CopyExists(FolderExists):
    """A copy would write over a file of the library holding different text."""


class Copied(NamedTuple):
    """What a copy of a built-in prompt wrote.

    Attributes:
        key: The copy's key in the library.
        written: Files written new.
        replaced: Files written over.
        unchanged: Files already holding the same text.
    """

    key: str
    written: int
    replaced: int
    unchanged: int


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


def builtin_directory() -> str:
    """The folder holding the pack's own prompts.

    Returns:
        ``system_prompts`` inside the pack's data folder, as an absolute path.
    """
    from ..data import paths as data_paths

    return os.path.join(os.fspath(data_paths.data_directory()), LIBRARY_NAME)


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
        The library's entries ordered by folder and then name, without regard to case, then
        the pack's own under ``Built-in/`` in the same order.
    """
    global _listing
    with _lock:
        now = time.monotonic()
        if _listing is not None and _listing[0] > now:
            return _listing[1]
        found = _walk(library_directory()) + _walk(builtin_directory(), builtin=True)
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
    meta, body = split_front_matter(_text_of(entry))
    return Document(meta or {}, body)


def _readable(entry: Entry, path: str) -> Path:
    """``path`` of ``entry`` resolved for reading: through the sandbox, or inside the pack's own prompts.

    Raises:
        PathNotAllowed: A library file resolves outside every permitted read root.
        ValueError: A built-in file resolves outside the pack's own prompts.
    """
    if not entry.builtin:
        return sandbox.resolve_read(path)
    root = Path(os.path.realpath(builtin_directory()))
    target = Path(os.path.realpath(path))
    if not sandbox.contains(root, target):
        raise ValueError(f"the built-in prompt `{entry.key}` lies outside the pack's own prompts")
    return target


def _text_of(entry: Entry) -> str:
    """The whole text of an entry's file, front matter included.

    Raises:
        PathNotAllowed: The file resolves outside every permitted read root.
        ValueError: The file is gone, larger than :data:`MAX_FILE_BYTES`, or not UTF-8.
    """
    target = _readable(entry, entry.path)
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
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"the system prompt `{entry.key}` is not UTF-8 text. Save it again as UTF-8"
        ) from error


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
    if _reserved(below):
        raise ValueError(
            f"`{BUILTIN_FOLDER}` holds the pack's own prompts, so nothing is saved into it. Type "
            "another folder, such as My Prompts"
        )
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


def skill_files(entry: Entry) -> tuple[tuple[SkillFile, ...], dict]:
    """The files of a skill besides its ``SKILL.md``.

    Args:
        entry: A library entry. A plain prompt has none.

    Returns:
        ``(files, skipped)``: the files ordered by path, and reason to the count passed over.
        Links, junctions, hidden names and files of other kinds are never listed.
    """
    if not entry.skill:
        return (), {}
    return _skill_walk(os.path.dirname(entry.path))


def skill_stamp(entry: Entry) -> str:
    """A stamp that moves whenever the skill's ``SKILL.md`` or any of its files changes.

    Args:
        entry: A library entry.

    Returns:
        A short hex digest.
    """
    digest = hashlib.sha1(f"{entry.key}\x1f{entry.mtime_ns}\x1f{entry.size}".encode("utf-8", "replace"))
    files, _skipped = skill_files(entry)
    for found in files:
        digest.update(f"\x1e{found.relative}\x1f{found.mtime_ns}\x1f{found.size}".encode("utf-8", "replace"))
    return digest.hexdigest()[:16]


def read_skill_files(entry: Entry) -> tuple[list[tuple[SkillFile, str]], dict]:
    """Read the text of a skill's files.

    Args:
        entry: A library entry.

    Returns:
        ``(read, skipped)``: each file with its text, in path order, and reason to the count
        passed over, a file that is not UTF-8 or cannot be read among them.
    """
    files, skipped = skill_files(entry)
    passed = Counter(skipped)
    read_files: list[tuple[SkillFile, str]] = []
    for found in files:
        try:
            target = _readable(entry, found.path)
            with open(target, "rb") as handle:
                raw = handle.read(MAX_SKILL_FILE_BYTES + 1)
        except (OSError, ValueError, sandbox.PathNotAllowed) as error:
            logger.debug("a file of the skill `%s` could not be read (%s)", entry.key, type(error).__name__)
            passed["unreadable"] += 1
            continue
        if len(raw) > MAX_SKILL_FILE_BYTES:
            passed["too large"] += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            passed["not UTF-8"] += 1
            continue
        if text.startswith("﻿"):
            text = text[1:]
        if not text.strip():
            passed["empty"] += 1
            continue
        read_files.append((found, text))
    return read_files, dict(passed)


def import_folder(folder: str, files, replace: bool = False) -> Imported:
    """Write a folder of prompts or skills into the library.

    Args:
        folder: The folder below the library to write into, such as ``moss-image-prompt-enhancer``.
        files: ``(relative path, text)`` pairs, the path ``/`` separated below ``folder``. A
            ``.txt`` or ``.md`` file is kept, and inside a folder holding ``SKILL.md`` any
            suffix of :data:`SKILL_EXTENSIONS`.
        replace: Write over files of the same path when the folder is already in the library.
            Files the import does not name are left in place.

    Returns:
        What was written.

    Raises:
        FolderExists: The folder is in the library and ``replace`` is off.
        ImportRefused: The folder or a path leaves the library, the files break a ceiling,
            nothing is left to write, or a path names something other than a plain file.
    """
    below = _import_folder_name(folder)
    if _reserved(below):
        raise ImportRefused(
            f"`{BUILTIN_FOLDER}` holds the pack's own prompts, so nothing is imported into it. "
            "Rename the folder and import it again"
        )
    if not isinstance(files, (list, tuple)):
        raise ImportRefused("the import carries no file list")
    if len(files) > MAX_IMPORT_FILES:
        raise ImportRefused(
            f"the folder holds {len(files)} files and one import takes at most {MAX_IMPORT_FILES}. "
            "Import a smaller folder"
        )
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in files:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ImportRefused("each imported file needs a path and its text")
        relative, text = item
        if not isinstance(relative, str) or not isinstance(text, str):
            raise ImportRefused("each imported file needs a path and its text")
        cleaned = _import_path(relative)
        folded = cleaned.casefold()
        if folded in seen:
            raise ImportRefused(f"`{cleaned}` appears twice in the import")
        seen.add(folded)
        pairs.append((cleaned, text))

    skill_roots = {
        path.rsplit("/", 1)[0] if "/" in path else ""
        for path, _text in pairs
        if path.rsplit("/", 1)[-1].casefold() == SKILL_FILE
    }
    kept: list[tuple[str, bytes]] = []
    skipped: Counter = Counter()
    total = 0
    for path, text in pairs:
        suffix = os.path.splitext(path)[1].lower()
        in_skill = any(base == "" or path.startswith(base + "/") for base in skill_roots)
        if in_skill and suffix not in SKILL_EXTENSIONS:
            skipped["not a text file"] += 1
            continue
        if not in_skill and suffix not in IMPORT_EXTENSIONS:
            skipped["not .txt or .md outside a skill"] += 1
            continue
        data = text.encode("utf-8", "replace")
        ceiling = MAX_SKILL_FILE_BYTES if in_skill else MAX_FILE_BYTES
        if len(data) > ceiling:
            skipped["too large"] += 1
            continue
        total += len(data)
        if total > MAX_IMPORT_BYTES:
            raise ImportRefused(
                f"the folder's text files come to more than {MAX_IMPORT_BYTES // (1024 * 1024)} MB, "
                "past what one import writes. Import a smaller folder"
            )
        kept.append((path, data))
    if not kept:
        raise ImportRefused(
            "the folder holds no .txt or .md prompt and no skill. A skill is a folder holding SKILL.md"
        )

    with _lock:
        counts = _write_files(below, kept, replace, whole_folder=True)

    skills = tuple(
        f"{below}/{path}" for path, _data in kept if path.rsplit("/", 1)[-1].casefold() == SKILL_FILE
    )
    logger.info(
        "imported `%s` into the system prompt library: %d new, %d replaced, %d unchanged, %d skipped",
        below, counts["new"], counts["replaced"], counts["unchanged"], sum(skipped.values()),
    )
    return Imported(below, skills, counts["new"], counts["replaced"], counts["unchanged"], dict(skipped))


def copy_builtin(key: str, replace: bool = False) -> Copied:
    """Copy one of the pack's own prompts into the library, where it can be edited.

    Args:
        key: A built-in key, such as ``Built-in/FLUX/FLUX.2 Prompt Upsampler.md``. A skill is
            copied with all of its files.
        replace: Write over files at the copy's path that hold different text.

    Returns:
        What was written, and the copy's key: the built-in key without ``Built-in/``.

    Raises:
        CopyExists: A file at the copy's path holds different text and ``replace`` is off.
        ImportRefused: The key names no built-in prompt, or a file cannot be read or written.
    """
    wanted = str(key or "")
    entry = find(wanted, fresh=True)
    if entry is None or not entry.builtin:
        raise ImportRefused(f"`{wanted[:120]}` is not one of the pack's built-in prompts")
    relative = entry.key[len(BUILTIN_FOLDER) + 1:]
    try:
        kept: list[tuple[str, bytes]] = [(os.path.basename(entry.path), _text_of(entry).encode("utf-8"))]
    except (ValueError, sandbox.PathNotAllowed) as error:
        raise ImportRefused(str(error)) from None
    below = relative.rsplit("/", 1)[0] if "/" in relative else ""
    if entry.skill:
        read_files, _skipped = read_skill_files(entry)
        kept.extend((found.relative, text.encode("utf-8")) for found, text in read_files)
    if not below:
        raise ImportRefused(f"`{entry.key}` sits at the top of the built-in prompts and has no folder to copy into")
    with _lock:
        counts = _write_files(below, kept, replace, whole_folder=False)
    logger.info(
        "copied the built-in `%s` into the system prompt library: %d new, %d replaced, %d unchanged",
        entry.key, counts["new"], counts["replaced"], counts["unchanged"],
    )
    return Copied(relative, counts["new"], counts["replaced"], counts["unchanged"])


def _write_files(below: str, kept: list[tuple[str, bytes]], replace: bool, whole_folder: bool) -> Counter:
    """Write files into the library folder ``below``, checking every target before the first write.

    Args:
        below: The folder below the library, ``/`` separated.
        kept: ``(path below the folder, bytes)`` pairs.
        replace: Write over files holding different bytes.
        whole_folder: Refuse a folder already in the library. Off, only a file holding
            different bytes is refused.

    Returns:
        The count of files that were ``new``, ``replaced`` and ``unchanged``.

    Raises:
        FolderExists: ``whole_folder`` is on, the folder exists and ``replace`` is off.
        CopyExists: ``whole_folder`` is off, a file holds different bytes and ``replace`` is off.
        ImportRefused: A target leaves the library or is not a plain file or folder.
    """
    try:
        root = sandbox.resolve_write(library_directory())
        destination = sandbox.resolve_write_file(root, below)
    except sandbox.PathNotAllowed:
        raise ImportRefused(f"the folder `{below}` cannot be written in the library") from None
    if not sandbox.contains(root, destination):
        raise ImportRefused(f"the folder `{below}` lies outside the library")
    exists = os.path.lexists(destination)
    if exists and (os.path.islink(destination) or _junction(destination) or not os.path.isdir(destination)):
        raise ImportRefused(f"`{below}` in the library is not a plain folder, so nothing was written")
    if exists and whole_folder and not replace:
        raise FolderExists(f"`{below}` is already in the system prompt library")

    planned: list[tuple[Path, bytes, str]] = []
    for path, data in kept:
        try:
            target = sandbox.resolve_write_file(root, f"{below}/{path}")
        except sandbox.PathNotAllowed:
            raise ImportRefused(f"`{path}` cannot be written in the library") from None
        if not sandbox.contains(destination, target):
            raise ImportRefused(f"`{path}` lies outside the folder `{below}`")
        status = "new"
        if os.path.lexists(target):
            if os.path.islink(target) or not os.path.isfile(target):
                raise ImportRefused(f"`{below}/{path}` in the library is not a plain file, so nothing was written")
            with open(target, "rb") as handle:
                held = handle.read(len(data) + 1)
            status = "unchanged" if held == data else "replaced"
            if status == "replaced" and not whole_folder and not replace:
                raise CopyExists(f"`{below}/{path}` is already in the system prompt library with different text")
        planned.append((target, data, status))

    counts: Counter = Counter()
    for target, data, status in planned:
        counts[status] += 1
        if status == "unchanged":
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb" if status == "replaced" else "xb") as handle:
            handle.write(data)
    invalidate()
    return counts


def _reserved(below: str) -> bool:
    """Whether a folder below the library lies in the one the pack's own prompts are listed under."""
    first = str(below or "").replace("\\", "/").strip("/").split("/", 1)[0]
    return first.strip().casefold() == BUILTIN_FOLDER.casefold()


def _import_folder_name(folder) -> str:
    """The folder an import writes into, ``/`` separated.

    Raises:
        ImportRefused: It is empty, leaves the library, or holds a hidden or skipped name.
    """
    if not isinstance(folder, str):
        raise ImportRefused("the import names no folder")
    text = folder.strip().replace("\\", "/").strip("/")
    if not text:
        raise ImportRefused("the import names no folder")
    return _import_segments(text, "folder")


def _import_path(relative: str) -> str:
    """One imported file's path below its folder, ``/`` separated.

    Raises:
        ImportRefused: It leaves the folder, is too long or deep, or holds a hidden or skipped name.
    """
    text = relative.strip().replace("\\", "/").strip("/")
    if not text or len(text) > MAX_IMPORT_PATH_CHARS:
        raise ImportRefused("an imported file has an empty or overlong path")
    return _import_segments(text, "file")


def _import_segments(text: str, what: str) -> str:
    """``text`` checked segment by segment.

    Raises:
        ImportRefused: A segment leaves its folder, is hidden or skipped, or there are too many.
    """
    shown = text[:80]
    if sandbox.leaves_folder(text) is not None:
        raise ImportRefused(f"the {what} `{shown}` names somewhere outside the library")
    parts = [part.strip() for part in text.split("/")]
    if len(parts) > MAX_DEPTH + 1:
        raise ImportRefused(f"the {what} `{shown}` is more than {MAX_DEPTH} folders deep")
    for part in parts:
        if not part or part.startswith(".") or part in SKIPPED_FOLDERS:
            raise ImportRefused(f"the {what} `{shown}` holds a hidden or empty name")
        if any(char in UNSAFE_STEM or ord(char) < 32 for char in part) or part != part.rstrip(" ."):
            raise ImportRefused(f"the {what} `{shown}` holds a name a file system cannot hold")
    return "/".join(parts)


def _junction(path) -> bool:
    """Whether a path is a Windows junction."""
    check = getattr(os.path, "isjunction", None)
    try:
        return bool(check(path)) if check is not None else False
    except OSError:
        return True


def _skill_walk(top: str) -> tuple[tuple[SkillFile, ...], dict]:
    """Every listable file of the skill folder ``top``, its ``SKILL.md`` aside."""
    skipped: Counter = Counter()
    try:
        root = os.path.realpath(top)
    except OSError:
        return (), {}
    found: list[SkillFile] = []
    total = 0
    examined = 0
    pending: list[tuple[str, str, int]] = [(root, "", 0)]
    while pending and examined < MAX_SKILL_EXAMINED:
        directory, below, depth = pending.pop()
        try:
            with os.scandir(directory) as listing:
                items = []
                for item in listing:
                    items.append(item)
                    examined += 1
                    if examined >= MAX_SKILL_EXAMINED:
                        skipped["past the skill's limit"] += 1
                        break
                items.sort(key=lambda item: item.name.casefold())
        except OSError as error:
            logger.debug("a skill folder could not be listed (%s)", type(error).__name__)
            continue
        folders = []
        for item in items:
            if item.name.startswith(".") or item.name in SKIPPED_FOLDERS:
                continue
            if file_listing.linked(item):
                skipped["link"] += 1
                continue
            relative = f"{below}/{item.name}" if below else item.name
            try:
                if item.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        folders.append((item.path, relative, depth + 1))
                    continue
                if not below and item.name.casefold() == SKILL_FILE:
                    continue
                if os.path.splitext(item.name)[1].lower() not in SKILL_EXTENSIONS:
                    skipped["not a text file"] += 1
                    continue
                info = item.stat(follow_symlinks=False)
            except OSError:
                skipped["unreadable"] += 1
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_size > MAX_SKILL_FILE_BYTES:
                skipped["too large"] += 1
                continue
            if len(found) >= MAX_SKILL_FILES or total + info.st_size > MAX_SKILL_BYTES:
                skipped["past the skill's limit"] += 1
                continue
            if not sandbox.contains(Path(root), Path(os.path.realpath(item.path))):
                continue
            total += info.st_size
            found.append(SkillFile(relative, item.path, info.st_size, info.st_mtime_ns))
        pending.extend(reversed(folders))
    found.sort(key=lambda each: each.relative.casefold())
    return tuple(found), dict(skipped)


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


def _walk(top: str, builtin: bool = False) -> tuple[Entry, ...]:
    """Every listable file below ``top``.

    Args:
        top: The folder to walk.
        builtin: Walk the pack's own prompts: keys gain ``Built-in/``, and files at the top
            of the folder, its notices and licences, are not listed.
    """
    try:
        if not os.path.isdir(top):
            return ()
        root = os.path.realpath(top)
    except OSError:
        return ()
    most = MAX_BUILTIN_ENTRIES if builtin else MAX_ENTRIES
    prefix = f"{BUILTIN_FOLDER}/" if builtin else ""
    found: list[Entry] = []
    pending: list[tuple[str, str, int]] = [(root, "", 0)]
    while pending and len(found) < most:
        directory, below, depth = pending.pop()
        try:
            with os.scandir(directory) as listing:
                items = sorted(listing, key=lambda item: item.name.casefold())
        except OSError as error:
            logger.debug("%s could not be listed (%s)", directory, error)
            continue
        # A folder holding SKILL.md is one skill: that file is listed and nothing else in it.
        if below:
            marker = [item for item in items if item.name.casefold() == SKILL_FILE]
            if marker:
                items = marker
        folders = []
        for item in items:
            if item.name.startswith(".") or item.name in SKIPPED_FOLDERS:
                continue
            relative = f"{below}/{item.name}" if below else item.name
            try:
                if item.is_dir(follow_symlinks=False):
                    if not below and not builtin and _reserved(item.name):
                        _warn_reserved(item.path)
                        continue
                    if depth < MAX_DEPTH and not file_listing.linked(item):
                        folders.append((item.path, relative, depth + 1))
                    continue
                if builtin and not below:
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
            is_skill = bool(below) and item.name.casefold() == SKILL_FILE
            others = len(_skill_walk(directory)[0]) if is_skill else 0
            found.append(Entry(
                prefix + relative, name, prefix + below if below else "", families, description, tags,
                item.path, info.st_size, info.st_mtime, info.st_mtime_ns, is_skill, others, builtin,
            ))
            if len(found) >= most:
                logger.warning(
                    "%s holds more than %d files, so only the first %d are listed",
                    "the pack's built-in prompts" if builtin else "the system prompt library", most, most,
                )
                break
        pending.extend(reversed(folders))
    found.sort(key=lambda entry: (entry.folder.casefold(), entry.name.casefold(), entry.key.casefold()))
    return tuple(found)


def _warn_reserved(path: str) -> None:
    """Report, once per folder, a library folder passed over for holding the built-in prompts' name."""
    if (path, -1) in _warned:
        return
    if len(_warned) >= MAX_ENTRIES:
        _warned.clear()
    _warned.add((path, -1))
    logger.warning(
        "%s is not listed: `%s` is where the pack's own prompts are listed. Rename the folder to "
        "list what it holds",
        path, BUILTIN_FOLDER,
    )


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
