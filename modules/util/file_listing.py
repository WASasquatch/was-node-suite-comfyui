"""The files under ComfyUI's own directories, as the labels a widget stores.

A label is ``<relative path> [tag]``. The listing is saved in the state folder, and a worker
thread rereads only the folders whose modified time changed.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple, Sequence

from .. import log

__all__ = [
    "CACHE_NAME",
    "CACHE_VERSION",
    "DEFAULT_LIMIT",
    "DEFER_LIMIT",
    "Entry",
    "Folder",
    "INPUT",
    "LISTING_TTL",
    "MAX_DEPTH",
    "MAX_DIRS",
    "MAX_FILES",
    "MAX_REUSE",
    "MAX_SAVED_BYTES",
    "LIMIT_KEY",
    "MOST_LIMIT",
    "OUTPUT",
    "CONFIGURED",
    "ROOTS",
    "SETTLE",
    "SLOW_WALK_SHARE",
    "TAGS",
    "TEMP",
    "WALK_WAIT",
    "configured",
    "defer_walks",
    "find",
    "folders",
    "label_of",
    "labels",
    "path_text",
    "linked",
    "listing",
    "menu_limit",
    "release_walks",
    "resolve",
    "reuse_until",
    "roots",
    "scan",
    "suffixes",
    "typed",
    "view",
    "walks_deferred",
    "warm",
    "when_changed",
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

#: Most directories below a root the listing descends.
MAX_DEPTH = 64

#: Most files the listing holds under one root, read breadth first, so the files nearer the
#: top of a folder are held first.
MAX_FILES = 20000

#: Most folders the listing reads under one root, breadth first, whether or not their files
#: are held.
MAX_DIRS = 20000

#: How many labels a menu offers where ``paths.menu_limit`` is unset or not a whole number.
DEFAULT_LIMIT = 1000

#: The most labels ``paths.menu_limit`` can ask a menu for.
MOST_LIMIT = MAX_FILES

#: The key under ``paths`` naming how many labels a menu offers.
LIMIT_KEY = "menu_limit"

#: Seconds a pass is trusted for at least before a request checks the folders again.
LISTING_TTL = 5.0

#: A pass is trusted for this many times as long as it took, where that is longer than
#: :data:`LISTING_TTL`.
SLOW_WALK_SHARE = 10.0

#: Longest a pass is trusted for, in seconds, however long it took.
MAX_REUSE = 120.0

#: Seconds, counted from the start of a pass, a request waits for it before being answered
#: from the last pass that finished.
WALK_WAIT = 0.25

#: Longest :func:`defer_walks` holds requests off the folders, in seconds, where nothing
#: releases them.
DEFER_LIMIT = 120.0

#: File in the pack's state folder the listing is saved to.
CACHE_NAME = "file_menus.json"

#: Layout of :data:`CACHE_NAME`. A saved listing of any other layout is read again from disk.
CACHE_VERSION = 1

#: Largest saved listing read back, in bytes.
MAX_SAVED_BYTES = 64 * 1024 * 1024

#: Seconds a folder's modified time must lie in the past before the folder is reused unread.
SETTLE = 2.0

#: Serializes the listing and the view cache. Both are read from ComfyUI's server thread, for
#: a browser panel, and from the prompt thread, for a combo and a node.
_lock = threading.RLock()

#: Signalled whenever a pass finishes, on :data:`_lock`.
_walk_done = threading.Condition(_lock)

#: The worker thread running a pass, or ``None`` when none is running.
_walker: threading.Thread | None = None

#: ``time.monotonic()`` the running pass started at.
_walk_began = 0.0

#: ``time.monotonic()`` :func:`defer_walks` holds requests off until.
_deferred_until = 0.0

#: ``(monotonic time it is trusted until, listing id, entries)``. The id changes whenever
#: the folders do, so a cached view knows whether the listing under it is the one it was
#: built from.
_scan_cache: tuple[float, int, tuple["Entry", ...]] = (0.0, 0, ())

#: How many listings have been published. Only ever counts up.
_walks = 0

#: ``{(tag, root): {relative folder: Folder}}`` in root order, each tree breadth first.
_trees: dict[tuple[str, str], dict[str, "Folder"]] = {}

#: Whether the saved listing has been read back in this process.
_loaded = False

#: ``{view key: (listing id, entries, how many files the view's filter matched)}``.
_view_cache: dict[tuple, tuple[int, tuple["Entry", ...], int]] = {}

#: ``(listing id, tags, [(tag, folder labels)])`` from the last :func:`folders` call.
_folder_cache: tuple[int, tuple, list] = (-1, (), [])

#: ``(listing id, suffixes)`` from the last :func:`suffixes` call.
_suffix_cache: tuple[int, frozenset] = (-1, frozenset())

#: Functions run on the worker thread after a pass changes the listing.
_listeners: list[Callable[[], object]] = []

#: View keys whose cut has been logged, each once per process.
_cuts_logged: set[tuple] = set()

#: Whether an unusable ``paths.menu_limit`` has been logged.
_limit_warned = False

#: Roots whose files did not all fit, each logged once per process.
_stopped: set[str] = set()


class Entry(NamedTuple):
    """One listable file.

    Attributes:
        tag: The root it sits under, ``input``, ``output`` or ``temp``.
        order: The root's position in :data:`TAGS`, so two roots sort in that order.
        relative: Path below the root, separated by ``/``. The first sort key.
        label: The stored value, which is ``relative`` followed by ``[tag]``.
        path: Absolute path on disk.
        mtime: Modification time, which decides what survives a view's limit.
        size: Size in bytes, as the listing saw it.
    """

    tag: str
    order: int
    relative: str
    label: str
    path: str
    mtime: float
    size: int


class Folder(NamedTuple):
    """One directory as a pass read it.

    Attributes:
        stamp: Its ``st_mtime_ns`` when read, or ``0`` where that time was too recent to trust.
        folders: Names of the directories inside it, sorted casefolded.
        files: ``(name, mtime, size)`` per file held, in the order read, or ``None`` where none
            were held.
        count: How many files it holds in all.
    """

    stamp: int
    folders: tuple[str, ...]
    files: tuple[tuple[str, float, int], ...] | None
    count: int


def roots(tags: Sequence[str] = TAGS) -> list[tuple[str, Path]]:
    """The directories listed, each with the tag its labels carry.

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


def reuse_until(started: float, finished: float) -> float:
    """The monotonic time a listing read between ``started`` and ``finished`` expires.

    Args:
        started: ``time.monotonic()`` before the listing was read.
        finished: ``time.monotonic()`` after it.

    Returns:
        ``finished`` plus :data:`LISTING_TTL`, or plus :data:`SLOW_WALK_SHARE` times the
        read's own length where that is longer, at most :data:`MAX_REUSE` seconds on.
    """
    took = max(0.0, finished - started)
    return finished + min(MAX_REUSE, max(LISTING_TTL, SLOW_WALK_SHARE * took))


def warm() -> None:
    """Start bringing the listing up to date on the worker thread, without waiting for it."""
    with _lock:
        _start()


def when_changed(callback: Callable[[], object]) -> None:
    """Run ``callback`` on the worker thread after every pass that changes the listing.

    Args:
        callback: Called with no arguments. An exception it raises is logged at debug.
    """
    with _lock:
        if callback not in _listeners:
            _listeners.append(callback)


def defer_walks() -> None:
    """Answer requests without waiting on a pass until the running event loop next turns."""
    global _deferred_until
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    with _lock:
        _deferred_until = time.monotonic() + DEFER_LIMIT
    loop.call_soon(release_walks)


def release_walks() -> None:
    """End the hold :func:`defer_walks` placed."""
    global _deferred_until
    with _lock:
        _deferred_until = 0.0


def walks_deferred() -> bool:
    """Whether :func:`defer_walks` is holding requests off the folders."""
    return time.monotonic() < _deferred_until


def scan() -> tuple[Entry, ...]:
    """Every listable file under every root, as the last pass left it.

    Returns:
        The entries in root order, each root breadth first, at most :data:`MAX_FILES` per
        root. A request starts a pass once :data:`LISTING_TTL` has passed and waits up to
        :data:`WALK_WAIT` for it. Empty where nothing was found.
    """
    return _scan()[2]


def suffixes() -> set[str]:
    """Every lowercased suffix the listing holds, such as ``.mp4``, without its empty one."""
    global _suffix_cache
    with _lock:
        _, walk, entries = _scan()
        if _suffix_cache[0] != walk:
            found = frozenset(os.path.splitext(entry.relative)[1].lower() for entry in entries)
            _suffix_cache = (walk, found - {""})
        return set(_suffix_cache[1])


def folders(tags: Sequence[str] = ROOTS) -> list[tuple[str, list[str]]]:
    """Each root's folders, as the labels a folder menu stores.

    Args:
        tags: Which roots to read, as :func:`view` takes them.

    Returns:
        ``[(tag, labels)]`` in root order, every root the last pass reached included. Each
        label is ``<relative path> [tag]``, breadth first and casefolded name order within a
        folder; the root itself is not among them.
    """
    global _folder_cache
    kept = _kept(tags)
    key = tuple(sorted(kept))
    with _lock:
        _, walk, _entries = _scan()
        if _folder_cache[0] == walk and _folder_cache[1] == key:
            return [(tag, list(rows)) for tag, rows in _folder_cache[2]]
        built = [
            (tag, [f"{relative} [{tag}]" for relative in tree if relative])
            for (tag, _root), tree in _trees.items()
            if _holds(tag, kept)
        ]
        _folder_cache = (walk, key, built)
        return [(tag, list(rows)) for tag, rows in built]


def menu_limit(at_least: int = 0) -> int:
    """How many labels a menu offers, from ``paths.menu_limit``.

    Args:
        at_least: The menu's own floor, answered where the setting is lower.

    Returns:
        The setting held to 1 to :data:`MOST_LIMIT`, or :data:`DEFAULT_LIMIT` where it is
        unset or not a whole number; never below ``at_least``.
    """
    global _limit_warned
    try:
        from ..config import load_config

        value = (load_config().get("paths") or {}).get(LIMIT_KEY, DEFAULT_LIMIT)
    except Exception as error:
        logger.debug("paths.%s could not be read (%s)", LIMIT_KEY, error)
        value = DEFAULT_LIMIT
    if value is None:
        value = DEFAULT_LIMIT
    if isinstance(value, bool) or not isinstance(value, int):
        if not _limit_warned:
            _limit_warned = True
            logger.warning(
                "paths.%s in config.yaml should be a whole number, not %r, so file menus "
                "list up to %d files. Write it as %s: 2000",
                LIMIT_KEY, value, DEFAULT_LIMIT, LIMIT_KEY,
            )
        value = DEFAULT_LIMIT
    return max(int(at_least), min(MOST_LIMIT, max(1, value)))


def view(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int | None = None,
    quiet: bool = False,
) -> tuple[Entry, ...]:
    """The entries one menu or panel offers, filtered, capped and sorted.

    Args:
        extensions: Lowercased suffixes to keep, such as ``(".txt", ".csv")``. ``None``
            keeps every file, whatever it is called.
        tags: Which roots to read, from :data:`TAGS`, plus :data:`CONFIGURED` for every
            directory ``paths.allow_read`` names or one configured tag by name.
        limit: How many entries the view holds, or ``None`` for :func:`menu_limit`. The
            newest by modification time are kept, so a file just written is always offered.
        quiet: Leave a cut out of the log, for a caller reporting it itself.

    Returns:
        The entries, ordered by casefolded relative path, then by root, then by path.
    """
    wanted = tuple(sorted(_suffixes(extensions))) if extensions is not None else None
    kept = tuple(sorted(_kept(tags)))
    key = (wanted, kept, max(0, int(menu_limit() if limit is None else limit)))
    with _lock:
        _, walk, entries = _scan()
        cached = _view_cache.get(key)
        if cached is None or cached[0] != walk:
            built, matched = _build(entries, wanted, set(kept), key[2])
            # Only the views built from an earlier listing are dropped.
            for stale in [name for name, held in _view_cache.items() if held[0] != walk]:
                del _view_cache[stale]
            cached = _view_cache[key] = (walk, built, matched)
        if not quiet and cached[2] > len(cached[1]):
            _report_cut(key, cached[2], len(cached[1]))
        return cached[1]


def listing(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int | None = None,
) -> dict[str, str]:
    """``{label: absolute path}`` for one view, in the order it offers them.

    Args:
        extensions: Suffixes to keep, or ``None`` for every file.
        tags: Which roots to read.
        limit: How many entries the view holds, or ``None`` for :func:`menu_limit`.

    Returns:
        A new dictionary. Two files cannot share a label, since a label carries the root's
        tag as well as the relative path.
    """
    return {entry.label: entry.path for entry in view(extensions, tags, limit)}


def labels(
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    limit: int | None = None,
) -> list[str]:
    """The labels of one view, in the order it offers them."""
    return [entry.label for entry in view(extensions, tags, limit)]


def _report_cut(key: tuple, matched: int, offered: int) -> None:
    """Log, once per view, that a menu offers fewer files than its filter matched.

    Args:
        key: The view's cache key, ``(suffixes, tags, limit)``.
        matched: Files the view's filter matched.
        offered: Files the view holds.
    """
    if key in _cuts_logged:
        return
    _cuts_logged.add(key)
    suffixes_ = key[0]
    kinds = "" if suffixes_ is None else " " + ", ".join(suffixes_[:4]) + (
        ", ..." if len(suffixes_) > 4 else ""
    )
    logger.info(
        "a file menu lists the newest %d of %d%s files. Raise paths.%s in config.yaml to "
        "list more, or wire the file's path into the menu's input as text to read any of them",
        offered, matched, kinds, LIMIT_KEY,
    )


def find(
    label: str,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    paths: bool = False,
) -> Entry | None:
    """The entry one label names, read straight off disk under the rules the listing keeps.

    Args:
        label: The exact label, such as ``prompts/animals.txt [input]``. Surrounding space
            is ignored.
        extensions: Suffixes the caller lists, or ``None`` for every file.
        tags: Which roots the caller reads.
        paths: Also read text that is no label as a path, through :func:`typed`.

    Returns:
        The entry, or ``None`` when no file the listing would hold carries that label, which
        covers a file that has been deleted, renamed, or invented, and no path names one.

    Raises:
        PathNotAllowed: With ``paths``, the text is a path outside every permitted read root.
    """
    wanted = tuple(_suffixes(extensions)) if extensions is not None else None
    kept = _kept(tags)
    needle = (label or "").strip()
    if not needle:
        return None
    entry = _labelled(needle, wanted, kept)
    if entry is not None:
        return entry
    return typed(needle, extensions, tags) if paths else None


def resolve(
    label: str,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
    paths: bool = False,
) -> str | None:
    """The absolute path one label names, from :func:`find`.

    Args:
        label: The exact label, as the menu offered it.
        extensions: Suffixes the caller lists, or ``None`` for every file.
        tags: Which roots the caller reads.
        paths: Also read text that is no label as a path, through :func:`typed`.

    Returns:
        The absolute path, or ``None`` when no listed file carries that label.

    Raises:
        PathNotAllowed: With ``paths``, the text is a path outside every permitted read root.
    """
    entry = find(label, extensions, tags, paths)
    return entry.path if entry is not None else None


def label_of(
    value,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
) -> str:
    """The label a wired file path is listed under, or the value itself when it is a label.

    Args:
        value: A label, or a path wired in as text.
        extensions: Suffixes the file must carry, or ``None`` for every file.
        tags: The roots it must lie under.

    Returns:
        ``<relative path> [tag]`` for a path naming a file under those roots, else the value.

    Raises:
        PathNotAllowed: The value is a path outside every permitted read root.
    """
    chosen = str(value or "").strip()
    entry = typed(chosen, extensions, tags)
    return entry.label if entry is not None else chosen


def path_text(value) -> str | None:
    """Text wired into a file or folder menu, when it is a path rather than a label.

    Args:
        value: The text, such as ``E:\\images\\cat.png``, quotes and space around it ignored.

    Returns:
        The path as written, or ``None`` for empty text, a label such as ``cat.png [input]``
        and a bare name with no folder in it.
    """
    text = str(value or "").strip().strip("\"'").strip()
    if not text or (text.endswith("]") and " [" in text):
        return None
    if "/" in text or "\\" in text or (len(text) > 1 and text[1] == ":"):
        return text
    return None


def typed(
    value,
    extensions: Iterable[str] | None = None,
    tags: Sequence[str] = TAGS,
) -> Entry | None:
    """The file a path wired in as text names, labelled as the listing labels it.

    Args:
        value: The text, as :func:`path_text` reads it.
        extensions: Suffixes the file must carry, or ``None`` for every file.
        tags: The roots the menu lists, tried first for the label. A file under any other
            root of :data:`ROOTS` is taken too, labelled with that root.

    Returns:
        The entry, or ``None`` when the text is no path, or names no file of these kinds under
        any of those roots.

    Raises:
        PathNotAllowed: The path lies outside every permitted read root.
    """
    from . import sandbox

    text = path_text(value)
    if text is None:
        return None
    target = Path(sandbox.resolve_read(text))
    if not target.is_file():
        return None
    wanted = tuple(_suffixes(extensions)) if extensions is not None else None
    if wanted is not None and not target.name.lower().endswith(wanted):
        return None
    places = roots(tags)
    places += [place for place in roots(ROOTS) if place not in places]
    for order, (tag, root) in enumerate(places):
        if not sandbox.contains(root, target):
            continue
        relative = target.relative_to(root).as_posix()
        info = target.stat()
        return Entry(tag, order, relative, f"{relative} [{tag}]", str(target), info.st_mtime, info.st_size)
    return None


def linked(entry: os.DirEntry) -> bool:
    """Whether a directory entry is a symlink or a Windows junction.

    Args:
        entry: One entry of an ``os.scandir`` listing.

    Returns:
        True for either, and for an entry that cannot be read. A junction is read off its
        reparse tag where ``DirEntry.is_junction`` is missing.
    """
    try:
        if entry.is_symlink():
            return True
        check = getattr(entry, "is_junction", None)
        if check is not None:
            return bool(check())
        tag = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)
        if tag is None:
            return False
        return getattr(entry.stat(follow_symlinks=False), "st_reparse_tag", 0) == tag
    except OSError:
        return True


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
            outside :data:`TAGS`, whose names are only known once a pass has run.

    Returns:
        ``True`` where the entry belongs in the view.
    """
    return tag in tags or (CONFIGURED in tags and tag not in TAGS)


def _start() -> None:
    """Start a pass on the worker thread unless one is running. Called holding :data:`_lock`."""
    global _walker, _walk_began
    if _walker is not None:
        return
    _walk_began = time.monotonic()
    _walker = threading.Thread(target=_refresh, name="was-file-listing", daemon=True)
    _walker.start()


def _scan() -> tuple[float, int, tuple[Entry, ...]]:
    """The published listing, a pass started once it has expired and waited for up to :data:`WALK_WAIT`."""
    with _lock:
        if walks_deferred():
            return _scan_cache
        if _walker is None and time.monotonic() >= _scan_cache[0]:
            _start()
        deadline = _walk_began + WALK_WAIT
        while _walker is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            _walk_done.wait(left)
        return _scan_cache


def _refresh() -> None:
    """Bring every root's folders up to date and publish them, on the worker thread."""
    global _trees, _loaded, _walker
    began = time.monotonic()
    published = False
    try:
        # ComfyUI's own three roots first, then the configured ones.
        places =[(tag, str(directory)) for tag, directory in roots((*TAGS, CONFIGURED))]
        if not _loaded:
            _loaded = True
            saved = _read_saved()
            held = {place: saved[place] for place in places if place in saved}
            if held:
                _publish(held, 0.0)
                published = True
        trees = {place: _pass(place[1], _trees.get(place, {})) for place in places}
        if list(trees.items()) != list(_trees.items()):
            _publish(trees, reuse_until(began, time.monotonic()))
            published = True
            _save(trees)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
    finally:
        with _lock:
            if _scan_cache[0] < time.monotonic():
                _settle(reuse_until(began, time.monotonic()))
            _walker = None
            _walk_done.notify_all()
    if published:
        for callback in list(_listeners):
            try:
                callback()
            except Exception as error:
                logger.debug("a file listing listener failed: %s", error)


def _settle(until: float) -> None:
    """Trust the published listing until ``until`` without changing it. Called holding :data:`_lock`."""
    global _scan_cache
    _scan_cache = (until, _scan_cache[1], _scan_cache[2])


def _publish(trees: dict[tuple[str, str], dict[str, Folder]], until: float) -> None:
    """Make ``trees`` the listing every view reads.

    Args:
        trees: ``{(tag, root): folders}`` in root order.
        until: Monotonic time the listing is trusted until, ``0`` to check it at the next
            request.
    """
    global _scan_cache, _trees, _walks
    found: list[Entry] = []
    for order, ((tag, root), tree) in enumerate(trees.items()):
        rows, left = _rows(tree, tag, order, root)
        found += rows
        if left and root not in _stopped:
            _stopped.add(root)
            logger.info(
                "the file menus hold the first %d files under %s and leave %d out. Files nearer "
                "the top of the folder are held first, and a path wired into a menu's input "
                "reads any of the rest",
                MAX_FILES, tag, left,
            )
    with _lock:
        _walks += 1
        _trees = trees
        _scan_cache = (until, _walks, tuple(found))


def _rows(tree: dict[str, Folder], tag: str, order: int, root: str) -> tuple[list[Entry], int]:
    """One root's entries, breadth first, within :data:`MAX_FILES`.

    Args:
        tree: The root's folders, breadth first.
        tag: The tag written into every label.
        order: The root's position, carried onto every entry.
        root: The root's absolute path.

    Returns:
        ``(entries, files left out)``.
    """
    rows: list[Entry] = []
    budget, left = MAX_FILES, 0
    for relative, folder in tree.items():
        take = (folder.files or ())[: max(0, budget)]
        budget -= len(take)
        left += max(0, folder.count - len(take))
        if not take:
            continue
        base = os.path.join(root, *relative.split("/")) if relative else root
        prefix = f"{relative}/" if relative else ""
        for name, mtime, size in take:
            # A label joins its folders with '/' on every platform.
            label_path = prefix + name
            rows.append(
                Entry(tag, order, label_path, f"{label_path} [{tag}]", os.path.join(base, name), mtime, size)
            )
    return rows, left


def _pass(root: str, old: dict[str, Folder]) -> dict[str, Folder]:
    """One root's folders, each reused from ``old`` while its modified time is unchanged.

    Args:
        root: The root, already resolved.
        old: The folders the last pass read, ``{relative: Folder}``.

    Returns:
        ``{relative: Folder}`` breadth first, the root itself under ``""``. A folder that
        cannot be read holds nothing and stops nothing.
    """
    tree: dict[str, Folder] = {}
    budget, room = MAX_FILES, MAX_DIRS
    settled = time.time_ns() - int(SETTLE * 1e9)
    queue: deque[tuple[str, int]] = deque([("", 0)])
    while queue:
        relative, depth = queue.popleft()
        path = os.path.join(root, *relative.split("/")) if relative else root
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if _linked_info(info) or not stat.S_ISDIR(info.st_mode):
            continue
        folder = old.get(relative)
        if _current(folder, info.st_mtime_ns, budget):
            if folder.files is not None and len(folder.files) > budget:
                folder = folder._replace(files=folder.files[: max(0, budget)] or None)
        else:
            folder = _read(path, info.st_mtime_ns, budget, settled)
        tree[relative] = folder
        budget -= min(max(0, budget), len(folder.files or ()))
        if depth >= MAX_DEPTH:
            continue
        for name in folder.folders:
            if room <= 0:
                break
            room -= 1
            queue.append((f"{relative}/{name}" if relative else name, depth + 1))
    return tree


def _current(folder: Folder | None, stamp: int, budget: int) -> bool:
    """Whether a folder read earlier still answers for one with modified time ``stamp``.

    Args:
        folder: The folder as the last pass read it, or ``None``.
        stamp: Its ``st_mtime_ns`` now.
        budget: Files the root may still hold.

    Returns:
        True where the time matches a trusted one and the folder holds as many of its files
        as the budget now asks for.
    """
    if folder is None or not folder.stamp or folder.stamp != stamp:
        return False
    if budget <= 0:
        return True
    held = len(folder.files) if folder.files is not None else 0
    return held >= min(budget, folder.count)


def _read(path: str, stamp: int, budget: int, settled: int) -> Folder:
    """One folder read off disk.

    Args:
        path: The folder.
        stamp: Its ``st_mtime_ns``.
        budget: Files the root may still hold.
        settled: ``time.time_ns()`` a trusted stamp must lie before.

    Returns:
        The folder, holding at most ``budget`` files in the order read. Dotted names, links
        and junctions are skipped.
    """
    names: list[str] = []
    files: list[tuple[str, float, int]] = []
    count = 0
    try:
        listing_ = os.scandir(path)
    except OSError:
        return Folder(0, (), None, 0)
    with listing_:
        for entry in _entries(listing_):
            # A dotted name is editor and tool state. A symlink and a junction are skipped,
            # whether they name a folder or a file.
            if entry.name.startswith(".") or linked(entry):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    names.append(entry.name)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                count += 1
                if len(files) >= budget:
                    continue
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            files.append((entry.name, info.st_mtime, max(0, int(info.st_size))))
    names.sort(key=str.casefold)
    trusted = stamp if stamp < settled else 0
    return Folder(trusted, tuple(names), tuple(files) if budget > 0 else None, count)


def _saved_file() -> Path | None:
    """Where the listing is saved, or ``None`` outside ComfyUI."""
    try:
        from ..config import comfyui_user_directory, state_file

        if comfyui_user_directory() is None:
            return None
        return state_file(CACHE_NAME)
    except Exception as error:
        logger.debug("the file listing has nowhere to be saved: %s", error)
        return None


def _save(trees: dict[tuple[str, str], dict[str, Folder]]) -> None:
    """Write ``trees`` to :data:`CACHE_NAME`, replacing the file in one step."""
    target = _saved_file()
    if target is None:
        return
    data = {
        "version": CACHE_VERSION,
        "roots": [
            {
                "tag": tag,
                "root": root,
                "folders": [
                    [
                        relative,
                        folder.stamp,
                        list(folder.folders),
                        None if folder.files is None else [list(row) for row in folder.files],
                        folder.count,
                    ]
                    for relative, folder in tree.items()
                ],
            }
            for (tag, root), tree in trees.items()
        ],
    }
    scratch = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        os.replace(scratch, target)
    except OSError as error:
        logger.debug("the file listing could not be saved: %s", error.strerror)
        try:
            scratch.unlink()
        except OSError:
            pass


def _read_saved() -> dict[tuple[str, str], dict[str, Folder]]:
    """The listing :func:`_save` wrote, ``{}`` where there is none or it will not read."""
    source = _saved_file()
    if source is None:
        return {}
    try:
        if source.stat().st_size > MAX_SAVED_BYTES:
            return {}
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return {}
    trees = {}
    for block in data.get("roots") or ():
        try:
            tree = {}
            for relative, stamp, names, files, count in block["folders"]:
                if relative and not all(_plain(part) for part in str(relative).split("/")):
                    raise ValueError(relative)
                held = None
                if files is not None:
                    held = tuple((str(name), float(mtime), int(size)) for name, mtime, size in files[:MAX_FILES])
                    if not all(_plain(row[0]) for row in held):
                        raise ValueError(relative)
                kids = tuple(str(name) for name in names)
                if not all(_plain(name) for name in kids):
                    raise ValueError(relative)
                tree[str(relative)] = Folder(int(stamp), kids, held, int(count))
            trees[(str(block["tag"]), str(block["root"]))] = tree
        except (KeyError, TypeError, ValueError):
            continue
    return trees


def _plain(part: str) -> bool:
    """Whether one segment of a label's relative path is a name the listing holds."""
    if not part or part.startswith(".") or "\0" in part:
        return False
    if os.sep in part or (os.altsep and os.altsep in part):
        return False
    return not (os.name == "nt" and ":" in part)


def _linked_info(info: os.stat_result) -> bool:
    """Whether an ``os.lstat`` result is a symlink or a Windows junction."""
    if stat.S_ISLNK(info.st_mode):
        return True
    tag = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)
    return tag is not None and getattr(info, "st_reparse_tag", 0) == tag


def _labelled(label: str, wanted: tuple[str, ...] | None, kept: set[str]) -> Entry | None:
    """The file one label names, read off disk segment by segment.

    Args:
        label: A label, such as ``prompts/animals.txt [input]``.
        wanted: Lowercased suffixes the file must carry, or ``None`` for any.
        kept: Tags the caller reads, as :func:`_kept` returns them.

    Returns:
        The entry, or ``None`` where the label is malformed, names a root the caller does not
        read, passes a dotted name, a symlink or a junction, or names no regular file.
    """
    from . import sandbox

    relative, separator, tail = label.rpartition(" [")
    if not separator or not relative or not tail.endswith("]"):
        return None
    tag = tail[:-1]
    if not _holds(tag, kept):
        return None
    parts = relative.split("/")
    if len(parts) > MAX_DEPTH + 1 or not all(_plain(part) for part in parts):
        return None
    if wanted is not None and not relative.lower().endswith(wanted):
        return None
    for order, (root_tag, root) in enumerate(roots((*TAGS, CONFIGURED))):
        if root_tag != tag:
            continue
        path = str(root)
        try:
            for part in parts:
                path = os.path.join(path, part)
                info = os.lstat(path)
                if _linked_info(info):
                    return None
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode) or not sandbox.contains(root, Path(path)):
            return None
        return Entry(tag, order, relative, label, path, info.st_mtime, max(0, int(info.st_size)))
    return None


def _build(
    entries: tuple[Entry, ...], extensions: tuple[str, ...] | None, tags: set[str], limit: int
) -> tuple[tuple[Entry, ...], int]:
    """Filter, cap and sort one view out of the listing.

    Args:
        entries: The whole listing.
        extensions: Suffixes to keep, or ``None`` for every file.
        tags: Roots to keep, as :func:`_kept` returns them.
        limit: How many entries survive.

    Returns:
        ``(entries, matched)``: the view's entries, each root keeping its own share of the
        limit, newest by modification time first, then sorted for the menu; and how many
        files the filter matched before the limit.
    """
    kept = [
        entry
        for entry in entries
        if _holds(entry.tag, tags)
        and (extensions is None or entry.relative.lower().endswith(extensions))
    ]
    matched = len(kept)
    if matched > limit:
        kept = _shared(kept, limit)
    kept.sort(key=lambda row: (row.relative.casefold(), row.order, row.path))
    return tuple(kept), matched


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
    # order the listing reached it.
    for tag in sorted(held, key=lambda name: held[name][0].order):
        taken.extend(held[tag][:share])
    # What one root did not need is offered to the rest, newest first.
    if len(taken) < limit:
        already = {id(entry) for entry in taken}
        rest = [entry for entry in kept if id(entry) not in already]
        rest.sort(key=lambda row: row.mtime, reverse=True)
        taken.extend(rest[: limit - len(taken)])
    return taken[:limit]


def _entries(listing_) -> Iterator[os.DirEntry]:
    """The entries of an open ``os.scandir`` iterator, read one at a time.

    Args:
        listing_: The iterator.

    Yields:
        Each entry, until the directory is exhausted or reading it fails.
    """
    while True:
        try:
            entry = next(listing_)
        except (StopIteration, OSError):
            return
        yield entry
