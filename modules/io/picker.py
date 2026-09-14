"""The files and folders a widget offers, and the one a chosen label names.

An input file is listed bare, everything else carries its root as ``[output]`` or
``[temp]``. A folder menu names each root on its own.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import log
from ..util import file_listing, sandbox

__all__ = [
    "INPUT_TAG", "MAX_FOLDERS", "ROOTS", "folders", "labels", "resolve", "resolve_folder",
]

logger = log.get_logger("io.picker")

#: Every root a picker offers: ComfyUI's own three, then whatever the config adds.
ROOTS = file_listing.ROOTS

#: What the listing puts after a name in the input folder, taken off again so an uploaded
#: file is a value in the menu.
INPUT_TAG = f" [{file_listing.INPUT}]"

#: How many folders a menu offers in all, divided between the roots so a dataset tree costs
#: a bounded amount rather than the whole of itself.
MAX_FOLDERS = 2000

#: Serializes the folder walk and its cache, which is read from ComfyUI's server thread for
#: an ``/object_info`` request and from the prompt thread when a node builds its schema.
_lock = threading.RLock()

#: ``(stamp, extra key, labels)`` from the last walk, reused for
#: :data:`modules.util.file_listing.LISTING_TTL` seconds.
_cache: tuple[float, tuple, tuple[str, ...]] = (0.0, (), ())


def labels(extensions, extra=()) -> list[str]:
    """Every file of these kinds under the folders a picker offers.

    Args:
        extensions: Suffixes to keep, such as ``(".zip",)``.
        extra: Folders of this node's own, as ``[(tag, path)]``, listed after the rest. A
            node reading a folder ComfyUI knows nothing about names it here.

    Returns:
        A bare ``<relative path>`` for a file in the input folder and
        ``<relative path> [tag]`` for the rest, newest first within the listing's own order.
        Empty outside ComfyUI and where no root can be read.
    """
    try:
        found = file_listing.labels(extensions, tags=ROOTS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        found = []
    offered = [
        label[: -len(INPUT_TAG)] if label.endswith(INPUT_TAG) else label for label in found
    ]
    return offered + [label for label, _path in _own(extensions, extra)]


def folders(extra=()) -> list[str]:
    """Every folder a picker offers.

    Args:
        extra: Folders of this node's own, as ``[(tag, path)]``, listed after the rest.

    Returns:
        A root's own name for the root itself, and ``<relative path> [tag]`` for each folder
        below it, to :data:`modules.util.file_listing.MAX_DEPTH` levels and
        :data:`MAX_FOLDERS` entries in all, each root taking an equal share. A folder
        holding no listable file is offered too. Empty outside ComfyUI and where no root
        can be read.
    """
    global _cache

    key = tuple((tag, str(directory)) for tag, directory in (extra or ()))
    with _lock:
        stamp, cached_key, labels = _cache
        now = time.monotonic()
        if labels and cached_key == key and now - stamp < file_listing.LISTING_TTL:
            return list(labels)

        # The folders under each root, read straight off disk. A directory-only pass costs
        # no per-file stat, and it offers an empty folder as readily as a full one.
        held: list[tuple[str, list[str]]] = []
        for tag, root in _rooted(extra):
            if not root.is_dir():
                continue
            found, left = _below(root, tag, MAX_FOLDERS)
            held.append((tag, found))
            if left <= 0:
                logger.debug("the folders under %s stopped at %d entries", tag, MAX_FOLDERS)

        offered = _offered(held, MAX_FOLDERS)
        _cache = (now, key, tuple(offered))
        return offered


def resolve_folder(label: str, extra=()) -> Path | None:
    """The folder one label names, resolved inside a permitted read root.

    Args:
        label: The widget's value, as :func:`folders` offered it.
        extra: The same folders :func:`folders` was given.

    Returns:
        The absolute folder, or None when the label names no root that is there. The folder
        itself need not still exist.

    Raises:
        PathNotAllowed: The folder resolved outside every permitted read root.
    """
    chosen = (label or "").strip()
    if not chosen:
        return None
    if chosen.endswith("]") and " [" in chosen:
        relative, _, tag = chosen[:-1].rpartition(" [")
    else:
        relative, tag = "", chosen

    for name, root in _rooted(extra):
        if name != tag:
            continue
        target = root if not relative else Path(os.path.normpath(root / relative))
        if not sandbox.contains(root, target):
            return None
        return Path(sandbox.resolve_read(target))
    return None


def _rooted(extra) -> list[tuple[str, Path]]:
    """Every root a folder menu walks: ComfyUI's own, the configured ones, then a node's."""
    try:
        found = list(file_listing.roots(ROOTS))
    except Exception as error:
        logger.debug("the roots could not be read: %s", error)
        found = []
    return found + [(tag, Path(directory)) for tag, directory in (extra or ())]


def _offered(held: list[tuple[str, list[str]]], limit: int) -> list[str]:
    """One menu out of what each root holds, capped at ``limit`` folders in all.

    Args:
        held: ``[(tag, folders)]`` in the order the menu names them.
        limit: How many folders the menu offers, over and above the root names.

    Returns:
        Each root's own name followed by its folders, every root taking an equal share of
        ``limit`` and a root holding fewer than its share leaving the rest to the others.
    """
    share = max(1, limit // max(1, len(held)))
    taken = [rows[:share] for _tag, rows in held]
    spare = limit - sum(len(rows) for rows in taken)
    for index, (_tag, rows) in enumerate(held):
        if spare <= 0:
            break
        more = rows[len(taken[index]):][:spare]
        taken[index] = taken[index] + more
        spare -= len(more)

    offered: list[str] = []
    for index, (tag, _rows) in enumerate(held):
        offered.append(tag)
        offered.extend(taken[index])
    return offered


def _below(root: Path, tag: str, budget: int) -> tuple[list[str], int]:
    """The folders under one root, breadth first, and what is left of the budget.

    Args:
        root: The directory walked.
        tag: The tag written into every label from this root.
        budget: Folders that may still be named.

    Returns:
        ``(labels, remaining budget)``, to :data:`modules.util.file_listing.MAX_DEPTH`
        levels. A directory that cannot be read contributes nothing and stops nothing.
    """
    found, level = [], [(root, "", 0)]
    while level and budget > 0:
        directory, relative, depth = level.pop(0)
        if depth >= file_listing.MAX_DEPTH:
            continue
        try:
            # scandir rather than iterdir: the directory read already carries each entry's
            # type, where Path.is_dir stats every name one at a time, and a root holding a
            # hundred thousand files is a hundred thousand stat calls per menu. A dotted
            # name is editor and tool state, and a symlink resolves wherever it likes.
            with os.scandir(directory) as listing:
                entries = sorted(
                    (
                        entry
                        for entry in listing
                        if not entry.name.startswith(".")
                        and entry.is_dir(follow_symlinks=False)
                    ),
                    key=lambda entry: entry.name.casefold(),
                )
        except OSError as error:
            logger.debug("%s could not be listed: %s", directory, error)
            continue
        for entry in entries:
            if budget <= 0:
                break
            below = f"{relative}/{entry.name}" if relative else entry.name
            found.append(f"{below} [{tag}]")
            budget -= 1
            level.append((Path(entry.path), below, depth + 1))
    return found, budget



def _own(extensions, extra):
    """Every file under a node's own folders, as ``[(label, path)]``."""
    wanted = tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in extensions or ()
    )
    out = []
    for tag, directory in extra or ():
        root = Path(directory)
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.rglob("*"))
        except OSError as error:
            logger.debug("%s could not be listed: %s", root, error)
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if wanted and entry.suffix.lower() not in wanted:
                continue
            relative = entry.relative_to(root).as_posix()
            out.append((f"{relative} [{tag}]", entry))
    return out


def resolve(label: str, extensions, extra=()) -> str | None:
    """The file one label names, resolved inside a permitted read root.

    Args:
        label: The widget's value, as :func:`labels` offered it.
        extensions: The suffixes that menu was built from.
        extra: The same folders :func:`labels` was given.

    Returns:
        The absolute path as a string, or None when the label names nothing that is there.

    Raises:
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = (label or "").strip()
    if not chosen:
        return None

    found = None
    # A bare name is a file in the input folder, which is what an upload leaves behind.
    try:
        import folder_paths

        if not sandbox.names_another_host(chosen) and folder_paths.exists_annotated_filepath(chosen):
            found = folder_paths.get_annotated_filepath(chosen)
    except Exception as error:
        logger.debug("`%s` could not be resolved through folder_paths: %s", chosen, error)
    if found is None:
        found = file_listing.resolve(chosen, extensions, tags=ROOTS)
    if found is None:
        for label, path in _own(extensions, extra):
            if label == chosen:
                found = path
                break
    if found is None:
        return None
    return str(sandbox.resolve_read(found))
