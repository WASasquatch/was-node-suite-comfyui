"""The files under ComfyUI's own directories, as the labels a widget stores.

A label is ``<relative path> [tag]``: ``input``, ``output``, ``temp``, or a directory
``paths.allow_read`` names. :func:`scan` walks each root on its own budget, memoized for
:data:`LISTING_TTL` seconds.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from .. import log

__all__ = [
    "DEFAULT_LIMIT",
    "Entry",
    "INPUT",
    "LISTING_TTL",
    "MAX_DEPTH",
    "MAX_SCAN",
    "OUTPUT",
    "CONFIGURED",
    "ROOTS",
    "TAGS",
    "TEMP",
    "configured",
    "find",
    "labels",
    "listing",
    "resolve",
    "roots",
    "scan",
    "view",
]

logger = log.get_logger("util.file_listing")

#: Asking for this tag adds every directory ``paths.allow_read`` names, each under a tag of
#: its own taken from the folder's name.
CONFIGURED = "configured"

#: The tag each root's labels carry, and the order two roots sort in.
INPUT = "input"
OUTPUT = "output"
TEMP = "temp"

#: The roots walked, in the order they are walked and sorted. ``folder_paths`` is asked for
#: each one, following the same spelling ``annotated_filepath`` uses for an annotated name.
TAGS: tuple[str, ...] = (INPUT, OUTPUT, TEMP)

#: ComfyUI's own roots and every directory ``paths.allow_read`` names, which is what a menu
#: offering the whole of what a user may read asks for.
ROOTS: tuple[str, ...] = (*TAGS, CONFIGURED)

#: The ``folder_paths`` accessor behind each tag.
_GETTERS = {
    INPUT: "get_input_directory",
    OUTPUT: "get_output_directory",
    TEMP: "get_temp_directory",
}

#: How many directories below a root the walk goes. Uploads land flat in ``input/``, but
#: prompt lists, render sets and wildcard sets are kept in folders, and three levels covers
#: those without enumerating a dataset tree.
MAX_DEPTH = 3

#: How many files are examined under one root before that root's walk stops. Each root
#: carries a budget of its own, so a 50000-file output folder costs its own entries and
#: leaves every other root's share of the listing whole.
MAX_SCAN = 5000

#: How many labels a view offers when it names no limit of its own.
DEFAULT_LIMIT = 500

#: Seconds a walk is reused for. Long enough that a burst of ``/object_info`` requests costs
#: one walk, and short enough that a newly saved file is listed before anyone has finished
#: reaching for it.
LISTING_TTL = 5.0

#: Serializes the walk and the view cache. Both are read from ComfyUI's server thread, for a
#: browser panel, and from the prompt thread, for a combo and a node.
_lock = threading.RLock()

#: ``(monotonic stamp, walk id, entries)``. The id is taken from :data:`_walks` and changes
#: on every rebuild, so a cached view knows whether the walk under it is the one it was built
#: from.
_scan_cache: tuple[float, int, tuple["Entry", ...]] = (0.0, 0, ())

#: How many walks have been made. Only ever counts up, so a view built from an earlier walk
#: can never be mistaken for a current one.
_walks = 0

#: ``{view key: (walk id, entries)}``.
_view_cache: dict[tuple, tuple[int, tuple["Entry", ...]]] = {}


class Entry(NamedTuple):
    """One listable file.

    Attributes:
        tag: The root it sits under, ``input``, ``output`` or ``temp``.
        order: The root's position in :data:`TAGS`, so two roots sort in that order.
        relative: Path below the root, separated by ``/``. The first sort key.
        label: The stored value, which is ``relative`` followed by ``[tag]``.
        path: Absolute path on disk.
        mtime: Modification time, which decides what survives a view's limit.
        size: Size in bytes, as the walk saw it.
    """

    tag: str
    order: int
    relative: str
    label: str
    path: str
    mtime: float
    size: int


def roots(tags: Sequence[str] = TAGS) -> list[tuple[str, Path]]:
    """The directories walked, each with the tag its labels carry.

    Args:
        tags: Which roots to answer, in :data:`TAGS` order whatever order they are given in.
            :data:`CONFIGURED` adds every directory ``paths.allow_read`` names, and a
            configured tag spelled outright adds that one.

    Returns:
        ``[(tag, path)]``, ComfyUI's own roots first and the configured ones after,
        dropping any that cannot be reached. Empty outside ComfyUI.
    """
    asked = set(tags)
    wanted = [tag for tag in TAGS if tag in asked]
    try:
        import folder_paths
    except ImportError:
        return []
    found = []
    for tag in wanted:
        getter = getattr(folder_paths, _GETTERS[tag], None)
        if getter is None:
            continue
        try:
            value = getter()
            directory = Path(value).expanduser().resolve() if value else None
        except Exception:
            # A getter that raises costs its root and never the listing: a menu falls back
            # to the other directories, or to its placeholder.
            continue
        if directory is not None and directory.is_dir():
            found.append((tag, directory))
    if CONFIGURED in asked or asked - set(TAGS):
        for tag, directory in configured([path for _tag, path in found]):
            if CONFIGURED in asked or tag in asked:
                found.append((tag, directory))
    return found


def configured(known: Sequence[Path] = ()) -> list[tuple[str, Path]]:
    """The directories ``paths.allow_read`` adds, each with the tag its labels carry.

    Args:
        known: Directories already being walked, which are not listed twice.

    Returns:
        ``[(tag, path)]``, the tag taken from the folder's own name and made unique. Empty
        where the config names none, or names none that can be reached.
    """
    from . import sandbox

    try:
        extra = sandbox.configured_read_roots()
    except Exception as error:
        logger.debug("the configured read roots could not be listed: %s", error)
        return []
    seen = {Path(one).resolve() for one in known}
    found, taken = [], set(TAGS)
    for directory in extra:
        try:
            resolved = Path(directory).resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        base = "".join(ch if ch.isalnum() else "-" for ch in resolved.name.lower()).strip("-")
        tag = base or "elsewhere"
        while tag in taken:
            tag = f"{tag}-{len(taken)}"
        taken.add(tag)
        found.append((tag, resolved))
    return found


def scan() -> tuple[Entry, ...]:
    """Every listable file under every root, memoized for :data:`LISTING_TTL` seconds.

    Returns:
        The entries in the order the roots were walked, unsorted and uncapped past
        :data:`MAX_SCAN`. Empty where nothing was found.
    """
    return _scan()[2]


def view(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int = DEFAULT_LIMIT,
) -> tuple[Entry, ...]:
    """The entries one menu or panel offers, filtered, capped and sorted.

    Args:
        extensions: Lowercased suffixes to keep, such as ``(".txt", ".csv")``. ``None``
            keeps every file, whatever it is called.
        tags: Which roots to read, from :data:`TAGS`, plus :data:`CONFIGURED` for every
            directory ``paths.allow_read`` names or one configured tag by name.
        limit: How many entries the view holds. The newest by modification time are kept, so
            a file that was just written is always offered.

    Returns:
        The entries, ordered by casefolded relative path, then by root, then by path.
    """
    wanted = tuple(sorted(_suffixes(extensions))) if extensions is not None else None
    kept = tuple(sorted(_kept(tags)))
    key = (wanted, kept, max(0, int(limit)))
    with _lock:
        _, walk, entries = _scan()
        cached = _view_cache.get(key)
        if cached is not None and cached[0] == walk:
            return cached[1]
        built = _build(entries, wanted, set(kept), key[2])
        # Only the views built from an earlier walk are dropped. Clearing the whole cache
        # would make three menus with three different filters rebuild each other's view on
        # every call, which is the cost this cache exists to avoid.
        for stale in [name for name, (built_from, _) in _view_cache.items() if built_from != walk]:
            del _view_cache[stale]
        _view_cache[key] = (walk, built)
        return built


def listing(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, str]:
    """``{label: absolute path}`` for one view, in the order it offers them.

    Args:
        extensions: Suffixes to keep, or ``None`` for every file.
        tags: Which roots to read.
        limit: How many entries the view holds.

    Returns:
        A new dictionary. Two files cannot share a label, since a label carries the root's
        tag as well as the relative path.
    """
    return {entry.label: entry.path for entry in view(extensions, tags, limit)}


def labels(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int = DEFAULT_LIMIT,
) -> list[str]:
    """The labels of one view, in the order it offers them."""
    return [entry.label for entry in view(extensions, tags, limit)]


def find(
    label: str,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
) -> Entry | None:
    """The entry one label names, read from the whole walk.

    Args:
        label: The exact label, such as ``prompts/animals.txt [input]``. Surrounding space
            is ignored; nothing else about it is read, so it is a key and never a path.
        extensions: Suffixes the caller lists, or ``None`` for every file.
        tags: Which roots the caller reads.

    Returns:
        The entry, or ``None`` when no walked file carries that label, which covers a file
        that has been deleted, renamed, or invented.
    """
    wanted = tuple(_suffixes(extensions)) if extensions is not None else None
    kept = _kept(tags)
    needle = (label or "").strip()
    if not needle:
        return None
    for entry in scan():
        if entry.label != needle or not _holds(entry.tag, kept):
            continue
        if wanted is not None and not entry.relative.lower().endswith(wanted):
            continue
        return entry
    return None


def resolve(
    label: str,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
) -> str | None:
    """The absolute path one label names, from :func:`find`.

    Args:
        label: The exact label, as the menu offered it.
        extensions: Suffixes the caller lists, or ``None`` for every file.
        tags: Which roots the caller reads.

    Returns:
        The absolute path, or ``None`` when no walked file carries that label.
    """
    entry = find(label, extensions, tags)
    return entry.path if entry is not None else None


# ---------------------------------------------------------------------- internals


def _suffixes(extensions: Iterable[str]) -> set[str]:
    """The extensions a caller listed, lowercased and each with its leading dot."""
    found = set()
    for entry in extensions:
        text = str(entry).strip().lower()
        if text:
            found.add(text if text.startswith(".") else f".{text}")
    return found


def _kept(tags: Sequence[str]) -> set[str]:
    """The tags one view holds, reading :data:`CONFIGURED` as a sentinel.

    Args:
        tags: The tags a caller asked for.

    Returns:
        The names from :data:`TAGS` that were asked for, any configured tag named outright,
        and :data:`CONFIGURED` itself where the sentinel was asked for.
    """
    asked = set(tags)
    kept = {tag for tag in TAGS if tag in asked}
    kept |= asked - set(TAGS) - {CONFIGURED}
    if CONFIGURED in asked:
        kept.add(CONFIGURED)
    return kept


def _holds(tag: str, tags: set[str]) -> bool:
    """Whether one entry's tag belongs to a view.

    Args:
        tag: The entry's tag.
        tags: What :func:`_kept` returned, where :data:`CONFIGURED` stands for every root
            outside :data:`TAGS`, whose names are only known once the walk has run.

    Returns:
        ``True`` where the entry belongs in the view.
    """
    return tag in tags or (CONFIGURED in tags and tag not in TAGS)


def _scan() -> tuple[float, int, tuple[Entry, ...]]:
    """The memoized walk, rebuilt when it is older than :data:`LISTING_TTL`."""
    global _scan_cache, _walks
    with _lock:
        stamp, _, entries = _scan_cache
        now = time.monotonic()
        if entries and now - stamp < LISTING_TTL:
            return _scan_cache
        found: list[Entry] = []
        stopped: list[str] = []
        # ComfyUI's own three first, then whatever the config adds, so a menu offers the
        # familiar folders before the rest. Each root is walked with its own budget.
        for order, (tag, directory) in enumerate(roots((*TAGS, CONFIGURED))):
            rows, left = _walk(directory, tag, order, MAX_SCAN)
            found += rows
            if left <= 0:
                stopped.append(tag)
        if stopped:
            logger.debug(
                "the file listing stopped after examining %d files under %s; a menu offers "
                "the newest of what was found", MAX_SCAN, ", ".join(stopped),
            )
        _walks += 1
        _scan_cache = (now, _walks, tuple(found))
        return _scan_cache


def _build(
    entries: tuple[Entry, ...], extensions: tuple[str, ...] | None, tags: set[str], limit: int
) -> tuple[Entry, ...]:
    """Filter, cap and sort one view out of the walk.

    Args:
        entries: The whole walk.
        extensions: Suffixes to keep, or ``None`` for every file.
        tags: Roots to keep, as :func:`_kept` returns them.
        limit: How many entries survive.

    Returns:
        The view's entries. Each root keeps its own share of the limit, newest by
        modification time first, and they are then sorted for the menu.
    """
    kept = [
        entry
        for entry in entries
        if _holds(entry.tag, tags)
        and (extensions is None or entry.relative.lower().endswith(extensions))
    ]
    if len(kept) > limit:
        kept = _shared(kept, limit)
    kept.sort(key=lambda row: (row.relative.casefold(), row.order, row.path))
    return tuple(kept)


def _shared(kept: list[Entry], limit: int) -> list[Entry]:
    """The limit divided between the roots, so one busy root cannot fill the whole view.

    Args:
        kept: Every entry the view covers.
        limit: How many entries survive in all.

    Returns:
        Up to ``limit`` entries, each root's newest first. A root holding fewer than its
        share leaves the rest to the others.
    """
    held: dict[str, list[Entry]] = {}
    for entry in kept:
        held.setdefault(entry.tag, []).append(entry)
    for entries_ in held.values():
        entries_.sort(key=lambda row: row.mtime, reverse=True)

    share = max(1, limit // max(1, len(held)))
    taken: list[Entry] = []
    # Every root the view holds, configured ones included, each taking its share in the
    # order the walk reached it.
    for tag in sorted(held, key=lambda name: held[name][0].order):
        taken.extend(held[tag][:share])
    # What one root did not need is offered to the rest, newest first.
    if len(taken) < limit:
        already = {id(entry) for entry in taken}
        rest = [entry for entry in kept if id(entry) not in already]
        rest.sort(key=lambda row: row.mtime, reverse=True)
        taken.extend(rest[: limit - len(taken)])
    return taken[:limit]


def _walk(directory: Path, tag: str, order: int, budget: int) -> tuple[list[Entry], int]:
    """Every listable file under one root, and what is left of the scan budget.

    Args:
        directory: The root, already resolved.
        tag: The tag written into every label from this root.
        order: The root's position in :data:`TAGS`, carried onto every entry.
        budget: Files that may still be examined.

    Returns:
        ``(entries, remaining budget)``. A directory that cannot be read contributes nothing
        and stops nothing.
    """
    found: list[Entry] = []
    stack: list[tuple[str, int, str]] = [(str(directory), 0, "")]
    while stack and budget > 0:
        parent, depth, prefix = stack.pop()
        try:
            entries = list(os.scandir(parent))
        except OSError:
            continue
        for entry in entries:
            if budget <= 0:
                break
            # A dotted name is editor and tool state, and a symlink is skipped whether it
            # names a directory or a file: it resolves wherever it likes, and one pointing
            # out of the root becomes a label the containment layer then refuses.
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        stack.append((entry.path, depth + 1, f"{prefix}{entry.name}/"))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                budget -= 1
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            # Built with '/' rather than os.sep or os.path.join: the label is the value
            # written into a saved workflow, and a backslash in it would stop a workflow
            # saved on Windows matching the same file listed on Linux.
            relative = f"{prefix}{entry.name}"
            found.append(
                Entry(
                    tag=tag,
                    order=order,
                    relative=relative,
                    label=f"{relative} [{tag}]",
                    path=entry.path,
                    mtime=stat.st_mtime,
                    size=max(0, int(stat.st_size)),
                )
            )
    return found, budget
