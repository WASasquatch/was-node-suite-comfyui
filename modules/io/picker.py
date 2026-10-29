"""The files and folders a menu offers, and the one a label or a wired path names.

Input files are listed bare, the rest as ``name [root]``; a folder menu lists each root too.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import log
from ..util import file_listing, sandbox

__all__ = [
    "FILE_HINT", "FOLDER_HINT", "INPUT_TAG", "MAX_FOLDERS", "ROOTS", "folder_label", "folders",
    "labels", "path_input", "resolve", "resolve_folder",
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

#: What a menu's tooltip says about the path it also takes as wired text.
FILE_HINT = "A wired path, such as `E:\\images\\cat.png`, works too."
FOLDER_HINT = "A wired path, such as `E:\\images`, works too."

#: Serializes the folder walk and its cache, which is read from ComfyUI's server thread for
#: an ``/object_info`` request and from the prompt thread when a node builds its schema.
_lock = threading.RLock()

#: ``(monotonic time it is reused until, extra key, labels)`` from the last walk, from
#: :func:`modules.util.file_listing.reuse_until`.
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
        holding no listable file is offered too. Empty outside ComfyUI, where no root can be
        read, and while file walks are deferred.
    """
    global _cache

    if file_listing.walks_deferred():
        return []
    key = tuple((tag, str(directory)) for tag, directory in (extra or ()))
    with _lock:
        until, cached_key, labels = _cache
        now = time.monotonic()
        if labels and cached_key == key and now < until:
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
        _cache = (file_listing.reuse_until(now, time.monotonic()), key, tuple(offered))
        return offered


def path_input(combo, folder: bool = False):
    """A file or folder menu that also takes a path wired in as text.

    Args:
        combo: The ``io.Combo.Input`` the menu is; its tooltip gains :data:`FILE_HINT` or
            :data:`FOLDER_HINT`.
        folder: True for a folder menu.

    Returns:
        An ``io.MultiType.Input`` drawing the menu and accepting a STRING link.
    """
    from comfy_api.latest import io

    hint = FOLDER_HINT if folder else FILE_HINT
    combo.tooltip = f"{combo.tooltip} {hint}" if combo.tooltip else hint
    return io.MultiType.Input(combo, [io.String])


def resolve_folder(label: str, extra=()) -> Path | None:
    """The folder one label or one wired path names, resolved inside a permitted read root.

    Args:
        label: The widget's value, as :func:`folders` offered it, or a path wired in as text.
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
    written = file_listing.path_text(chosen)
    if written is None:
        return None
    target = Path(sandbox.resolve_read(written))
    if any(sandbox.contains(root, target) for _name, root in _rooted(extra)):
        return target
    return None


def folder_label(value: str, extra=()) -> str:
    """The menu label a wired folder path names, or the value itself when it is a label.

    Args:
        value: A label :func:`folders` offered, or a path wired in as text.
        extra: The same folders :func:`folders` was given.

    Returns:
        ``<relative path> [tag]``, or the root's own name for a root itself, so no absolute
        path is written anywhere the label goes.

    Raises:
        PathNotAllowed: The path lies outside every permitted read root.
    """
    chosen = str(value or "").strip()
    written = file_listing.path_text(chosen)
    if written is None:
        return chosen
    target = Path(sandbox.resolve_read(written))
    for name, root in _rooted(extra):
        if sandbox.contains(root, target):
            relative = os.path.relpath(target, root).replace("\\", "/")
            return name if relative in ("", ".") else f"{relative} [{name}]"
    return chosen


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
            # name is editor and tool state, and a symlink or a junction is skipped.
            with os.scandir(directory) as listing:
                entries = sorted(
                    (
                        entry
                        for entry in listing
                        if not entry.name.startswith(".")
                        and not file_listing.linked(entry)
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
            resolved = Path(os.path.realpath(root))
        except OSError as error:
            logger.debug("%s could not be listed: %s", root, error)
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            if wanted and entry.suffix.lower() not in wanted:
                continue
            # A file a link or a junction leads to outside the folder is left out.
            if not sandbox.contains(resolved, Path(os.path.realpath(entry))):
                continue
            relative = entry.relative_to(root).as_posix()
            out.append((f"{relative} [{tag}]", entry))
    return out


def resolve(label: str, extensions, extra=()) -> str | None:
    """The file one label or one wired path names, resolved inside a permitted read root.

    Args:
        label: The widget's value, as :func:`labels` offered it, or a path wired in as text.
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
        found = sandbox.annotated_path(chosen)
    except Exception as error:
        logger.debug("`%s` could not be resolved through folder_paths: %s", chosen, error)
    if found is None:
        found = file_listing.resolve(chosen, extensions, tags=ROOTS, paths=True)
    if found is None:
        for label, path in _own(extensions, extra):
            if label == chosen:
                found = path
                break
    if found is None:
        return None
    return str(sandbox.resolve_read(found))
