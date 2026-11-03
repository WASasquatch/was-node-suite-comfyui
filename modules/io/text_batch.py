"""Text files under one folder, read by pattern or by name, and texts written back by name.

A name is the path below the folder without its extension, ``/`` between folders: ``cat``,
``set_b/cat``.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import NamedTuple

from .. import log
from ..util import sandbox, text_files
from . import naming

__all__ = [
    "MAX_FILES", "MAX_FILE_BYTES", "MAX_NOTES", "MAX_TOTAL_BYTES", "Batch", "Saved",
    "fingerprint", "matches", "name_list", "next_free", "read_folder", "read_named",
    "same_text", "snippet", "stem", "write_text",
]

logger = log.get_logger("io.text_batch")

#: The most files one read hands on.
MAX_FILES = 16384

#: The largest file read, in bytes.
MAX_FILE_BYTES = 4 * 1024 * 1024

#: The most bytes one read takes in across every file.
MAX_TOTAL_BYTES = 128 * 1024 * 1024

#: How many files a run names in the log as left out, before counting the rest.
MAX_NOTES = 20

#: Characters a report snippet keeps.
SNIPPET_CHARS = 80


class Batch(NamedTuple):
    """What one read found.

    Attributes:
        texts: One text per entry.
        names: The name each text belongs to, in the same order.
        missing: Names with no readable file, 0 for a pattern read.
        skipped: Files found and not read: not UTF-8, unreadable or too large.
        left_out: Matches past :data:`MAX_FILES`.
        matched: Files the pattern found.
    """

    texts: list
    names: list
    missing: int
    skipped: int
    left_out: int
    matched: int


class Saved(NamedTuple):
    """What one save did.

    Attributes:
        paths: Every file written, new or replaced, in order.
        written: Files that did not exist before.
        replaced: Files written over.
        unchanged: Files already holding their text, left as they were.
        left_out: Entries written nowhere: a blank text or a blank name.
    """

    paths: list
    written: int
    replaced: int
    unchanged: int
    left_out: int


def stem(name, extension: str = "") -> str:
    """A wired name as a name below a folder: slashes forward, an image or ``extension`` suffix dropped.

    Args:
        name: The name as it arrived, such as ``cat``, ``set_b/cat`` or ``cat.png``.
        extension: The text file extension, dropped where the name already ends with it.

    Returns:
        The name without surrounding space, empty where it is blank.
    """
    text = str(name or "").strip().replace("\\", "/").rstrip("/")
    folder, slash, leaf = text.rpartition("/")
    root, suffix = os.path.splitext(leaf)
    wanted = extension.lower()
    if suffix and (suffix.lower() in naming.IMAGE_SUFFIXES or (wanted and suffix.lower() == wanted)):
        leaf = root
    return f"{folder}{slash}{leaf}".strip()


def name_list(value) -> list[str] | None:
    """Every name out of a ``filenames`` input: a STRING list, a LIST, or lines of one string.

    Args:
        value: The input as an ``is_input_list`` node receives it, or None.

    Returns:
        One entry per name, a missing one kept as ``""``, or None where nothing arrived.
    """
    if value is None:
        return None
    found, arrived = [], False
    for item in value if isinstance(value, list) else [value]:
        if item is None:
            found.append("")
            continue
        arrived = True
        for entry in item if isinstance(item, (list, tuple)) else [item]:
            text = "" if entry is None else str(entry)
            lines = [line for line in text.splitlines() if line.strip()]
            found.extend(lines if len(lines) > 1 else [text])
    return found if arrived else None


def snippet(text: str) -> str:
    """The start of a text on one line, cut to :data:`SNIPPET_CHARS` characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= SNIPPET_CHARS else f"{flat[:SNIPPET_CHARS].rstrip()}..."


def matches(base: Path, pattern: str) -> list[str]:
    """Every file under a folder that a pattern matches, inside the folder once links are followed.

    Args:
        base: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern below it; ``**`` crosses folders.

    Returns:
        The matches as the glob spelled them, unsorted, folders and anything a link leads out
        of ``base`` to left out.

    Raises:
        PathNotAllowed: The pattern is empty, absolute, names a drive or climbs out with ``..``.
    """
    found = []
    for name in sandbox.glob_read(base, pattern, recursive=True):
        if not sandbox.contains(base, Path(os.path.realpath(name))):
            logger.debug("a match for `%s` leads out of the folder and is left out", pattern)
            continue
        if os.path.isfile(name):
            found.append(name)
    return found


def _relative(base: Path, match: str) -> str:
    """A match's path below ``base``, ``/`` between folders, extension dropped."""
    below = os.path.relpath(os.path.normpath(match), str(base)).replace("\\", "/")
    return os.path.splitext(below)[0]


def _read(path: Path, budget: int) -> tuple[str | None, str, int]:
    """One file's text, or why it was not read.

    Args:
        path: The file, already resolved through the containment layer.
        budget: Bytes the run may still take in.

    Returns:
        ``(text, "", bytes read)`` with the text decoded and trimmed, or
        ``(None, reason, 0)``.
    """
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            return None, f"is {size} bytes, over the {MAX_FILE_BYTES} byte limit on one file", 0
        if size > budget:
            return None, f"would pass the {MAX_TOTAL_BYTES} byte limit on one read", 0
        with open(path, "rb") as handle:
            data = handle.read(MAX_FILE_BYTES + 1)
    except OSError as error:
        return None, f"could not be read ({error.strerror or 'unreadable'})", 0
    try:
        return text_files.decode(data).strip(), "", len(data)
    except UnicodeDecodeError as error:
        return None, f"is not UTF-8 text at byte {error.start}; save it as UTF-8", 0


def _note(notes: list[str], name: str, reason: str) -> None:
    """Keep one left-out file for the log, up to :data:`MAX_NOTES` of them."""
    if len(notes) < MAX_NOTES:
        notes.append(f"`{name}` {reason}")
    else:
        notes.append("")


def _log(notes: list[str], label: str) -> None:
    """Log the files a read left out, naming the first :data:`MAX_NOTES`."""
    named = [note for note in notes if note]
    for note in named:
        logger.warning("Load Text Batch in %s: %s, so it was left out", label, note)
    if len(notes) > len(named):
        logger.warning(
            "Load Text Batch in %s: %d more file(s) were left out for the same reasons",
            label, len(notes) - len(named),
        )


def read_folder(base: Path, pattern: str, label: str) -> Batch:
    """Read every file a pattern matches under a folder, sorted by path.

    Args:
        base: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern below it.
        label: The folder's menu label, for the log.

    Returns:
        The texts that were read and their names.

    Raises:
        PathNotAllowed: The pattern is empty, absolute, names a drive or climbs out with ``..``.
    """
    held = []
    for match in matches(base, pattern):
        try:
            resolved = sandbox.resolve_read(match)
        except sandbox.PathNotAllowed:
            logger.debug("a match for `%s` in %s is not a file this pack may read", pattern, label)
            continue
        held.append((str(resolved), resolved, _relative(base, match)))
    held.sort(key=lambda entry: entry[0])
    kept = held[:MAX_FILES]
    left_out = len(held) - len(kept)
    if left_out:
        logger.warning(
            "Load Text Batch read the first %d of %d matches for `%s` in %s; the rest are past "
            "the %d file limit on one read. Narrow pattern or split the folder",
            len(kept), len(held), pattern, label, MAX_FILES,
        )

    texts, names, notes, budget = [], [], [], MAX_TOTAL_BYTES
    for _key, resolved, name in kept:
        text, reason, size = _read(resolved, budget)
        if text is None:
            _note(notes, name, reason)
            continue
        budget -= size
        texts.append(text)
        names.append(name)
    _log(notes, label)
    return Batch(texts, names, 0, len(notes), left_out, len(held))


def read_named(base: Path, names: list[str], extension: str, label: str) -> Batch:
    """Read ``<name><extension>`` under a folder for every name, a blank text where there is none.

    Args:
        base: The folder, already resolved inside a permitted read root.
        names: The names, in order; each may hold folders below ``base``.
        extension: Appended to each name, with its leading dot.
        label: The folder's menu label, for the log.

    Returns:
        One text per name, in the names' order.

    Raises:
        PathNotAllowed: A name names another machine, a drive or a root, climbs out with
            ``..``, or resolves outside every permitted read root.
        ValueError: More names arrived than :data:`MAX_FILES`.
    """
    if len(names) > MAX_FILES:
        raise ValueError(
            f"{len(names)} filenames arrived and Load Text Batch reads at most {MAX_FILES} at "
            f"once. Split the names into smaller lists"
        )
    texts, kept, notes, missing, budget = [], [], [], 0, MAX_TOTAL_BYTES
    for given in names:
        name = stem(given, extension)
        kept.append(name)
        if not name:
            texts.append("")
            missing += 1
            continue
        relative = f"{name}{extension}"
        reason = sandbox.leaves_folder(relative)
        if reason is not None:
            raise sandbox.PathNotAllowed(
                f"the name `{sandbox._shown(given)}` {reason}, so it would be read outside "
                f"{label}. Wire names below the folder, such as `cat` or `set_b/cat`"
            )
        candidate = os.path.join(str(base), *relative.split("/"))
        path = sandbox.resolve_read(candidate)
        if not sandbox.contains(base, path) or not path.is_file():
            texts.append("")
            missing += 1
            continue
        text, why, size = _read(path, budget)
        if text is None:
            _note(notes, relative, why)
            texts.append("")
            missing += 1
            continue
        budget -= size
        texts.append(text)
    _log(notes, label)
    if missing:
        logger.info(
            "Load Text Batch found no readable %s file for %d of %d name(s) in %s; each "
            "is an empty text in its place",
            extension or "extensionless", missing, len(names), label,
        )
    return Batch(texts, kept, missing, len(notes), 0, len(names))


def fingerprint(base: Path | None, pattern: str, extension: str):
    """Every file a read could open under a folder, with its time and size, as one value.

    Args:
        base: The folder, already resolved, or None where it is not known yet.
        pattern: The pattern a read without names uses.
        extension: The extension a read by name uses; every file carrying it is covered.

    Returns:
        A string that changes when any of those files is added, removed or changed, or NaN
        where the folder is not known.
    """
    if base is None or not base.is_dir():
        return float("NaN")
    try:
        found = set(matches(base, pattern))
        found.update(matches(base, f"**/*{glob.escape(extension)}"))
    except (sandbox.PathNotAllowed, OSError):
        return float("NaN")
    stats = []
    for name in sorted(found):
        try:
            info = os.stat(name)
        except OSError:
            continue
        stats.append((name, info.st_mtime_ns, info.st_size))
    return repr(stats)


def next_free(folder: Path, prefix: str, delimiter: str, padding: int, suffix: str,
              extension: str) -> str:
    """The next unused ``<prefix><delimiter><number><suffix><extension>`` in a folder.

    Args:
        folder: The folder, already resolved inside a save root.
        prefix: Name part before the number.
        delimiter: Between the name and the number. May be empty.
        padding: Digits the number is padded to. ``0`` answers ``<prefix><suffix><extension>``.
        suffix: Name part after the number.
        extension: File extension, leading dot included.

    Returns:
        A file name, not a path.

    Raises:
        PathNotAllowed: The name built leaves the folder.
    """
    if padding == 0:
        return f"{prefix}{suffix}{extension}"
    head, tail = f"{prefix}{delimiter}", f"{suffix}{extension}"
    seen = []
    try:
        entries = os.listdir(folder)
    except OSError:
        entries = []
    for entry in entries:
        if not (entry.startswith(head) and entry.endswith(tail)):
            continue
        number = entry[len(head): len(entry) - len(tail)]
        if len(number) == padding and number.isdigit():
            seen.append(int(number))
    counter = max(seen) + 1 if seen else 1
    while True:
        name = f"{head}{counter:0{padding}}{tail}"
        if not sandbox.resolve_save_file(folder, name).exists():
            return name
        counter += 1


def same_text(path: Path, text: str, encoding: str) -> bool:
    """Whether a file already holds a text, line endings and surrounding blank space aside.

    Args:
        path: The file, already resolved through the containment layer.
        text: The text about to be written.
        encoding: The encoding the file is read in.

    Returns:
        False where the file is missing, larger than :data:`MAX_FILE_BYTES`, unreadable or
        not in that encoding.
    """
    try:
        if not path.is_file() or os.path.getsize(path) > MAX_FILE_BYTES:
            return False
        with open(path, "rb") as handle:
            held = handle.read(MAX_FILE_BYTES + 1).decode(encoding)
    except (OSError, UnicodeDecodeError, LookupError):
        return False
    held = text_files.normalize_newlines(held.lstrip("﻿")).strip()
    return held == text_files.normalize_newlines(text).strip()


def write_text(folder: Path, name: str, text: str, encoding: str) -> tuple[Path, bool]:
    """Write a text to a file placed in a folder, making the folders between.

    Args:
        folder: The folder, already resolved inside a save root.
        name: The file name; it may hold folders below ``folder``.
        text: Written exactly as given.
        encoding: The file's character encoding.

    Returns:
        ``(path, replaced)``, ``replaced`` True where a file was there before.

    Raises:
        PathNotAllowed: The name leaves the folder.
        OSError: The folder could not be made or the file written.
    """
    path = sandbox.resolve_save_file(folder, name)
    existed = path.exists()
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="\n") as handle:
        handle.write(text)
    return path, existed
