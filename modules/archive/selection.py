"""Reading a chosen set of files out of a widget, one entry per line.

A selection is text: one menu label per line, such as ``renders/cat.png [output]``, blank
lines and ``#`` comments ignored, order kept.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from .. import log
from ..util import file_listing, sandbox
from ..util.text_files import is_comment
from .save import Source

__all__ = ["MAX_LINES", "gone", "parse", "sources"]

logger = log.get_logger("archive.selection")

#: How many lines of a selection are read. Above this the rest is reported and ignored, so a
#: widget somebody pasted a whole directory listing into is bounded before any file is opened.
MAX_LINES = 8192

#: Why a chosen file did not reach the archive.
UNLISTED = "no file of that name is in the input, output, temp or allow_read folders"
MISSING = "the file is no longer there"
NOT_A_FILE = "that path is a folder rather than a file"


def parse(text: str) -> tuple[list[str], int]:
    """The entries a selection widget holds, in the order they are written.

    Args:
        text: The widget's value.

    Returns:
        ``(entries, repeats)``. Blank lines and comment lines are dropped, a repeated entry
        is kept once at its first position, and ``repeats`` counts the ones dropped, since a
        file cannot go into one archive twice.
    """
    found: list[str] = []
    seen: set[str] = set()
    repeats = 0
    for number, line in enumerate(str(text or "").splitlines()):
        if number >= MAX_LINES:
            logger.warning(
                "the file selection holds more than %d lines; the rest were not read",
                MAX_LINES,
            )
            break
        entry = line.strip()
        if not entry or is_comment(entry):
            continue
        if entry in seen:
            repeats += 1
            continue
        seen.add(entry)
        found.append(entry)
    return found, repeats


def sources(entries: Iterable[str]) -> tuple[list[Source], dict[str, str]]:
    """The files a selection names, ready to be written into an archive.

    Args:
        entries: Menu labels, absolute paths, or a mix of the two, in order.

    Returns:
        ``(sources, missing)``. ``missing`` maps each entry that could not be used to the
        reason, and covers a label nobody lists, a file that has been deleted since it was
        chosen, and a path naming a folder.

    Raises:
        PathNotAllowed: An entry named a path outside every readable root. The message names
            the path, the roots and the config key that would permit it.
    """
    found: list[Source] = []
    missing: dict[str, str] = {}
    known = _tags()
    for entry in entries:
        listed = file_listing.find(entry, tags=file_listing.ROOTS)
        if listed is not None:
            source = Source(entry, Path(listed.path), listed.relative, listed.tag)
        elif _looks_like_label(entry, known):
            missing[entry] = UNLISTED
            continue
        else:
            source = _from_path(entry)
        resolved = sandbox.resolve_read(source.path)
        if not resolved.is_file():
            missing[entry] = NOT_A_FILE if resolved.is_dir() else MISSING
            continue
        found.append(Source(source.label, resolved, source.relative, source.tag))
    return found, missing


def gone(missing: Sequence[str] | dict[str, str]) -> str:
    """One line naming the chosen files that did not reach the archive.

    Args:
        missing: The ``missing`` mapping from :func:`sources`, or the entries alone.

    Returns:
        The reasons, grouped, or an empty string when nothing was missing.
    """
    if not missing:
        return ""
    if not isinstance(missing, dict):
        return f"{len(missing)} chosen file(s) were not there: {', '.join(missing)}"
    grouped: dict[str, list[str]] = {}
    for entry, reason in missing.items():
        grouped.setdefault(reason, []).append(entry)
    return "; ".join(
        f"{len(names)} skipped because {reason}: {', '.join(names)}"
        for reason, names in grouped.items()
    )


def _tags() -> set[str]:
    """Every tag a menu label can carry, ComfyUI's own three and the configured roots."""
    try:
        walked = {tag for tag, _root in file_listing.roots(file_listing.ROOTS)}
    except Exception as error:
        logger.debug("the roots could not be read: %s", error)
        walked = set()
    return set(file_listing.TAGS) | walked


def _looks_like_label(entry: str, tags: set[str]) -> bool:
    """Whether an entry was written as a menu label rather than as a path.

    Args:
        entry: The line, as the widget holds it.
        tags: The tags a label can carry, from :func:`_tags`.

    Returns:
        ``True`` where the entry ends in one of those tags in brackets.
    """
    return any(entry.endswith(f"[{tag}]") for tag in tags)


def _from_path(entry: str) -> Source:
    """One entry read as a path rather than as a menu label.

    Args:
        entry: The line, which is expected to name a file.

    Returns:
        A source whose name inside the archive is the file's own name, since a path from
        somewhere else has no root to be relative to.
    """
    name = os.path.basename(entry.replace("\\", "/").rstrip("/")) or entry
    return Source(entry, Path(entry), name, "")
