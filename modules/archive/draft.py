"""Build a new archive from an existing one plus entries the graph produced."""

from __future__ import annotations

import io
import zipfile
from typing import Iterable, NamedTuple

from ..log import get_logger
from . import container, save


def _carried(source, entry, spent: int) -> tuple[bytes, int]:
    """One entry's bytes, and what the draft has unpacked once it is added.

    Args:
        source: The archive being copied from.
        entry: The entry to read.
        spent: Bytes unpacked into this draft so far.

    Returns:
        ``(bytes, spent)``.

    Raises:
        ValueError: The draft would hold more than :data:`container.MAX_TOTAL_BYTES`.
    """
    body = source.read(entry)
    spent += len(body)
    if spent > container.MAX_TOTAL_BYTES:
        raise ValueError(
            f"the archive unpacks to more than {container.MAX_TOTAL_BYTES} bytes, which is "
            f"more than one draft holds. Take fewer entries, or work on it in parts."
        )
    return body, spent

logger = get_logger("archive.draft")

__all__ = ["Addition", "extended", "kept", "unique_name"]

#: Where an archive built in memory says it came from.
BUILT = "a built archive"


class Addition(NamedTuple):
    """One file going into an archive.

    Attributes:
        name: The entry name, spelled with ``/``.
        data: The bytes the entry holds.
    """

    name: str
    data: bytes


def unique_name(name: str, taken: set[str]) -> str:
    """A name no entry has yet, numbering a clash apart.

    Args:
        name: The wanted entry name.
        taken: Names already in the archive, which this adds to.

    Returns:
        ``name`` where it is free, otherwise ``stem_2.ext``, ``stem_3.ext`` and so on.
    """
    candidate = name
    if candidate not in taken:
        taken.add(candidate)
        return candidate
    head, dot, tail = name.rpartition(".")
    stem, extension = (head, dot + tail) if dot else (name, "")
    counter = 2
    while f"{stem}_{counter}{extension}" in taken:
        counter += 1
    candidate = f"{stem}_{counter}{extension}"
    taken.add(candidate)
    return candidate


def extended(
    source: container.Archive | None,
    additions: Iterable[Addition],
    compression: str = "deflate",
) -> container.Archive:
    """An archive holding everything ``source`` held, plus ``additions``.

    Args:
        source: The archive to start from, or None to start empty.
        additions: The entries to append, in the order they are written.
        compression: A key of :data:`modules.archive.save.COMPRESSIONS`.

    Returns:
        The new archive, held as bytes. ``source`` is not changed and not written to.

    Raises:
        ValueError: An addition's name cannot be an entry name, or the archive would hold
            more than :data:`modules.archive.container.MAX_ENTRIES` entries.
    """
    method = save.COMPRESSIONS.get(compression, zipfile.ZIP_DEFLATED)
    additions = list(additions)
    carried = source.files if source is not None else ()
    if len(carried) + len(additions) > container.MAX_ENTRIES:
        raise ValueError(
            f"an archive may hold {container.MAX_ENTRIES} entries, and this would hold "
            f"{len(carried) + len(additions)}. Save the archive and start another."
        )

    buffer = io.BytesIO()
    taken: set[str] = set()
    with zipfile.ZipFile(buffer, "w", compression=method) as target:
        spent = 0
        for entry in carried:
            name = unique_name(entry.name, taken)
            body, spent = _carried(source, entry, spent)
            target.writestr(name, body)
        for addition in additions:
            name, refusal = container.safe_name(addition.name)
            if refusal:
                raise ValueError(
                    f"{addition.name!r} cannot be an entry name: {refusal}."
                )
            target.writestr(unique_name(name, taken), addition.data)

    logger.debug(
        "built an archive of %d entries (%d carried, %d added), %d byte(s)",
        len(taken), len(carried), len(additions), buffer.tell(),
    )
    return container.Archive.from_bytes(buffer.getvalue(), label=BUILT)


def kept(
    source: container.Archive,
    names: Iterable[str],
    compression: str = "deflate",
) -> container.Archive:
    """An archive holding only the named entries of ``source``, in the archive's own order.

    Args:
        source: The archive to take entries from.
        names: Entry names to keep. A name the archive does not hold is passed over.
        compression: A key of :data:`modules.archive.save.COMPRESSIONS`.

    Returns:
        The new archive, held as bytes. ``source`` is not changed and not written to.
    """
    method = save.COMPRESSIONS.get(compression, zipfile.ZIP_DEFLATED)
    wanted = {str(name).strip() for name in names if str(name).strip()}
    buffer = io.BytesIO()
    written = 0
    spent = 0
    with zipfile.ZipFile(buffer, "w", compression=method) as target:
        for entry in source.files:
            if entry.name not in wanted:
                continue
            body, spent = _carried(source, entry, spent)
            target.writestr(entry.name, body)
            written += 1
    logger.debug(
        "kept %d of %d entr(y/ies), %d byte(s)", written, len(source.files), buffer.tell(),
    )
    return container.Archive.from_bytes(buffer.getvalue(), label=BUILT)
