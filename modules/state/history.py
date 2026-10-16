"""The image, output-image and text-file history lists.

Three lists live under the keys ``Images``, ``Output_Images`` and ``TextFiles``, each an
ordered list of absolute paths. Appending one path writes one row.
"""

from __future__ import annotations

import os

from .. import config, log
from ..util.sandbox import names_another_host
from .database import HISTORY_CATEGORY, HistoryDatabase, get_history_db

__all__ = [
    "IMAGES",
    "OUTPUT_IMAGES",
    "TEXT_FILES",
    "display_limit",
    "open_history_db",
    "recent",
    "update_history_images",
    "update_history_output_images",
    "update_history_text_files",
]

logger = log.get_logger("state.history")

#: Every image a loading node has read.
IMAGES = "Images"

#: Every image a saving node has written.
OUTPUT_IMAGES = "Output_Images"

#: Every text file a node has read or written.
TEXT_FILES = "TextFiles"

#: How many paths :func:`recent` reads at a time while looking for ones still on disk.
PAGE = 256

#: How many of the newest paths :func:`recent` looks at before it stops looking.
CANDIDATES = 512


def open_history_db() -> HistoryDatabase:
    """Open the history database.

    Returns:
        The process-wide history database.
    """
    return get_history_db()


def display_limit() -> int:
    """How many history entries a history combo shows, from ``history.display_limit``."""
    return config.load_config()["history"]["display_limit"]


def recent(key: str, limit: int, existing_only: bool = True) -> list[str]:
    """The newest paths recorded under one history key, oldest of them first.

    Args:
        key: :data:`IMAGES`, :data:`OUTPUT_IMAGES` or :data:`TEXT_FILES`.
        limit: How many paths to return at most.
        existing_only: Skip a path whose file is not there now. The path stays recorded,
            so a drive that is offline today lists again once it is back. At most
            :data:`CANDIDATES` of the newest are looked at.

    Returns:
        Up to ``limit`` paths, in the order they were recorded.
    """
    if limit <= 0:
        return []
    database = open_history_db()
    if not existing_only:
        newest = database.newest(key, limit)
        newest.reverse()
        return newest
    found: list[str] = []
    skipped = 0
    ceiling = max(limit, CANDIDATES)
    while len(found) < limit and skipped < ceiling:
        page = database.newest(key, min(PAGE, ceiling - skipped), skipped)
        if not page:
            break
        skipped += len(page)
        for path in page:
            if not names_another_host(path) and os.path.exists(path):
                found.append(path)
                if len(found) >= limit:
                    break
    found.reverse()
    return found


def _as_paths(new_paths: str | list[str], key: str) -> list[str] | None:
    """``new_paths`` as a list, or None when it is not something to record.

    Args:
        new_paths: One path, or a list of them.
        key: History key being updated, named in the error.

    Returns:
        A list of paths, or None when the argument is neither a path nor a list of them,
        in which case nothing should be written and the cause has been logged.
    """
    if isinstance(new_paths, str):
        return [new_paths]
    if isinstance(new_paths, list):
        return list(new_paths)
    logger.error(
        "the %s history takes a path or a list of paths and was given a %s, so nothing "
        "was recorded",
        key,
        type(new_paths).__name__,
    )
    return None


def _append(key: str, new_paths: str | list[str]) -> None:
    """Append to one of the three history lists, moving a path already in it to the end.

    Args:
        key: :data:`IMAGES`, :data:`OUTPUT_IMAGES` or :data:`TEXT_FILES`.
        new_paths: One path, or a list of them. Anything else is reported and ignored.
    """
    paths = _as_paths(new_paths, key)
    if not paths:
        return
    open_history_db().append(key, paths)


def update_history_images(new_paths: str | list[str]) -> None:
    """Append one path or a list of paths to the loaded-image history.

    Args:
        new_paths: One path, or a list of them. Anything else is reported and ignored.
    """
    _append(IMAGES, new_paths)


def update_history_output_images(new_paths: str | list[str]) -> None:
    """Append one path or a list of paths to the saved-image history.

    Args:
        new_paths: One path, or a list of them. Anything else is reported and ignored.
    """
    _append(OUTPUT_IMAGES, new_paths)


def update_history_text_files(new_paths: str | list[str]) -> None:
    """Append one path or a list of paths to the text-file history.

    Args:
        new_paths: One path, or a list of them. Anything else is reported and ignored.
    """
    _append(TEXT_FILES, new_paths)
