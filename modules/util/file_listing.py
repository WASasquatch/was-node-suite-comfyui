"""The files under ComfyUI's own directories, as the labels a widget stores.

A label is ``<relative path> [tag]``, the tag ``input``, ``output``, ``temp`` or a
``paths.allow_read`` folder. :func:`scan` walks on a worker thread; :func:`find` reads one
label off disk.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Sequence

from .. import log

__all__ = [
    "DEFAULT_LIMIT",
    "DEFER_LIMIT",
    "Entry",
    "INPUT",
    "LISTING_TTL",
    "MAX_DEPTH",
    "MAX_EXAMINED",
    "MAX_REUSE",
    "OUTPUT",
    "CONFIGURED",
    "ROOTS",
    "SLOW_WALK_SHARE",
    "TAGS",
    "TEMP",
    "WALK_WAIT",
    "configured",
    "defer_walks",
    "find",
    "label_of",
    "labels",
    "path_text",
    "linked",
    "listing",
    "release_walks",
    "resolve",
    "reuse_until",
    "roots",
    "scan",
    "typed",
    "view",
    "walks_deferred",
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

#: Most directories below a root the walk descends. Every subfolder is listed;
#: :data:`MAX_EXAMINED` bounds the work, read breadth first.
MAX_DEPTH = 64

#: How many directory entries are examined under one root, breadth first, before that
#: root's walk stops. Files, folders and skipped names all count. Each root carries a budget
#: of its own, and a menu's own limit counts only the files it lists.
MAX_EXAMINED = 20000

#: How many labels a view offers when it names no limit of its own.
DEFAULT_LIMIT = 500

#: Seconds a walk is reused for at least. Long enough that a burst of ``/object_info``
#: requests costs one walk, and short enough that a newly saved file is listed before anyone
#: has finished reaching for it.
LISTING_TTL = 5.0

#: A walk is reused for this many times as long as it took, where that is longer than
#: :data:`LISTING_TTL`.
SLOW_WALK_SHARE = 10.0

#: Longest a walk is reused for, in seconds, however long it took.
MAX_REUSE = 120.0

#: Seconds, counted from the start of a walk, a caller waits for it before being answered
#: from the last walk that finished.
WALK_WAIT = 1.0

#: Longest :func:`defer_walks` holds walks off, in seconds, where nothing releases them.
DEFER_LIMIT = 120.0

#: Serializes the walk and the view cache. Both are read from ComfyUI's server thread, for a
#: browser panel, and from the prompt thread, for a combo and a node.
_lock = threading.RLock()

#: Signalled whenever a walk finishes, on :data:`_lock`.
_walk_done = threading.Condition(_lock)

#: The worker thread walking the roots, or ``None`` when no walk is running.
_walker: threading.Thread | None = None

#: ``time.monotonic()`` the running walk started at.
_walk_began = 0.0

#: ``time.monotonic()`` :func:`defer_walks` holds walks off until.
_deferred_until = 0.0

#: ``(monotonic time it is reused until, walk id, entries)``. The id is taken from
#: :data:`_walks` and changes on every rebuild, so a cached view knows whether the walk under it is the one it was built
#: from.
_scan_cache: tuple[float, int, tuple["Entry", ...]] = (0.0, 0, ())

#: How many walks have been made. Only ever counts up, so a view built from an earlier walk
#: can never be mistaken for a current one.
_walks = 0

#: ``{view key: (walk id, entries)}``.
_view_cache: dict[tuple, tuple[int, tuple["Entry", ...]]] = {}

#: Roots whose walk has been reported as stopping, each logged once per process.
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


def defer_walks() -> None:
    """Hold walks off until the running event loop next turns, at most :data:`DEFER_LIMIT` seconds."""
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
    """Whether :func:`defer_walks` is holding walks off."""
    return time.monotonic() < _deferred_until


def scan() -> tuple[Entry, ...]:
    """Every listable file under every root, memoized for at least :data:`LISTING_TTL` seconds.

    Returns:
        The entries in the order the roots were walked, unsorted, at most
        :data:`MAX_EXAMINED` per root. The last finished walk where a refresh outlasts
        :data:`WALK_WAIT` or walks are deferred. Empty where nothing was found.
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
    paths: bool = False,
) -> Entry | None:
    """The entry one label names, read straight off disk under the rules the walk keeps.

    Args:
        label: The exact label, such as ``prompts/animals.txt [input]``. Surrounding space
            is ignored.
        extensions: Suffixes the caller lists, or ``None`` for every file.
        tags: Which roots the caller reads.
        paths: Also read text that is no label as a path, through :func:`typed`.

    Returns:
        The entry, or ``None`` when no file the walk would list carries that label, which
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
        The absolute path, or ``None`` when no walked file carries that label.

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
    """The file a path wired in as text names, labelled as the walk labels it.

    Args:
        value: The text, as :func:`path_text` reads it.
        extensions: Suffixes the file must carry, or ``None`` for every file.
        tags: The roots it must lie under.

    Returns:
        The entry, or ``None`` when the text is no path, or names no file of these kinds under
        those roots.

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
    for order, (tag, root) in enumerate(roots(tags)):
        if not sandbox.contains(root, target):
            continue
        relative = target.relative_to(root).as_posix()
        info = target.stat()
        return Entry(tag, order, relative, f"{relative} [{tag}]", str(target), info.st_mtime, info.st_size)
    return None


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
    """The memoized walk, refreshed on a worker thread once expired, waited for up to :data:`WALK_WAIT`."""
    global _walker, _walk_began
    with _lock:
        if (_walks and time.monotonic() < _scan_cache[0]) or walks_deferred():
            return _scan_cache
        if _walker is None:
            _walk_began = time.monotonic()
            _walker = threading.Thread(target=_refresh, name="was-file-listing", daemon=True)
            _walker.start()
        deadline = _walk_began + WALK_WAIT
        while _walker is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            _walk_done.wait(left)
        return _scan_cache


def _refresh() -> None:
    """Walk every root and publish the result as the memoized walk, on the worker thread."""
    global _scan_cache, _walks, _walker
    began = time.monotonic()
    found: list[Entry] = []
    try:
        # ComfyUI's own three first, then whatever the config adds, so a menu offers the
        # familiar folders before the rest. Each root is walked with its own budget.
        for order, (tag, directory) in enumerate(roots((*TAGS, CONFIGURED))):
            rows, stopped = _walk(directory, tag, order, MAX_EXAMINED)
            found += rows
            if stopped and str(directory) not in _stopped:
                _stopped.add(str(directory))
                logger.info(
                    "the file menus stopped reading %s (%s) after examining %d entries, %d of "
                    "them files: files past that point are not listed and cannot be picked. "
                    "Files nearer the top of the folder are read first, and moving files out "
                    "of it brings the rest into the menus.",
                    tag, directory, MAX_EXAMINED, len(rows),
                )
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
    finally:
        with _lock:
            _walks += 1
            _scan_cache = (reuse_until(began, time.monotonic()), _walks, tuple(found))
            _walker = None
            _walk_done.notify_all()


def _plain(part: str) -> bool:
    """Whether one segment of a label's relative path is a name the walk lists."""
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


def _walk(directory: Path, tag: str, order: int, budget: int) -> tuple[list[Entry], bool]:
    """Every listable file under one root, breadth first, within a budget of entries.

    Args:
        directory: The root, already resolved.
        tag: The tag written into every label from this root.
        order: The root's position in :data:`TAGS`, carried onto every entry.
        budget: Directory entries that may be examined, whatever they turn out to be.

    Returns:
        ``(entries, stopped)``, ``stopped`` being True where the budget ran out with entries
        left unread. A directory that cannot be read contributes nothing and stops nothing.
    """
    found: list[Entry] = []
    queue: deque[tuple[str, int, str]] = deque([(str(directory), 0, "")])
    while queue:
        parent, depth, prefix = queue.popleft()
        try:
            listing = os.scandir(parent)
        except OSError:
            continue
        with listing:
            for entry in _entries(listing):
                if budget <= 0:
                    return found, True
                budget -= 1
                # A dotted name is editor and tool state. A symlink and a junction are skipped,
                # whether they name a folder or a file.
                if entry.name.startswith(".") or linked(entry):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth < MAX_DEPTH:
                            queue.append((entry.path, depth + 1, f"{prefix}{entry.name}/"))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    info = entry.stat(follow_symlinks=False)
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
                        mtime=info.st_mtime,
                        size=max(0, int(info.st_size)),
                    )
                )
    return found, False


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


def _entries(listing) -> Iterator[os.DirEntry]:
    """The entries of an open ``os.scandir`` iterator, read one at a time.

    Args:
        listing: The iterator.

    Yields:
        Each entry, until the directory is exhausted or reading it fails.
    """
    while True:
        try:
            entry = next(listing)
        except (StopIteration, OSError):
            return
        yield entry
