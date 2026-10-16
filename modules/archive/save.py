"""Writing a zip archive from files already on disk.

:func:`build` streams each source file in :data:`CHUNK` bytes at a time, and
:func:`entry_name` decides what a file is called inside the archive, following
:data:`NAMING`.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import NamedTuple, Sequence

from .. import log

__all__ = [
    "CHUNK",
    "COMPRESSIONS",
    "FLATTEN",
    "MAX_FILES",
    "MAX_TOTAL_BYTES",
    "NAMING",
    "RELATIVE",
    "Source",
    "WriteFailed",
    "Written",
    "build",
    "entry_name",
]

logger = log.get_logger("archive.save")

#: Bytes moved per read while a file is being copied into the archive.
CHUNK = 1024 * 1024

#: How many files one archive may be given. An archive of more than this is a job for a
#: dedicated tool, and every entry costs an index record whatever its size.
MAX_FILES = 4096

#: How many bytes one archive may be given in total. Far above any set of renders, and low
#: enough that a runaway list of paths is refused rather than filling the disk.
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024

#: The names entries can be given inside the archive, in the order the widget offers them.
RELATIVE = "relative path"
TAGGED = "source folder and relative path"
FLATTEN = "file name only"
NAMING: tuple[str, ...] = (RELATIVE, TAGGED, FLATTEN)

#: The compression choices, and the zip method behind each.
COMPRESSIONS: dict[str, int] = {
    "deflate": zipfile.ZIP_DEFLATED,
    "store": zipfile.ZIP_STORED,
}

#: The earliest moment a zip can record, so a file older than this is written with this.
ZIP_FLOOR = (1980, 1, 1, 0, 0, 0)

#: File mode written on every entry: a regular file, readable by anybody.
ENTRY_MODE = 0o100644 << 16

#: Suffix of the file an archive is written into before it replaces the target.
PART_SUFFIX = ".part"


class WriteFailed(ValueError):
    """A source file failed part way into the archive, so the archive was abandoned."""


class Source(NamedTuple):
    """One file going into an archive.

    Attributes:
        label: The listing label it was chosen by, or the path as it was given.
        path: Absolute path on disk, already resolved through the containment layer.
        relative: Path below its root, spelled with ``/``, or the file name alone for a file
            that came from outside the listed roots.
        tag: ``input``, ``output``, ``temp``, or empty for a file named by path.
    """

    label: str
    path: Path
    relative: str
    tag: str


class Written(NamedTuple):
    """What one write put in the archive.

    Attributes:
        names: The entry names written, in the order they were written.
        total: Bytes read out of the source files.
        size: Bytes the finished archive occupies.
        renamed: ``{entry name: label}`` for each source numbered apart from another.
        skipped: ``{label: reason}`` for each source that could not be read.
    """

    names: tuple[str, ...]
    total: int
    size: int
    renamed: dict[str, str]
    skipped: dict[str, str]


def entry_name(source: Source, naming: str) -> str:
    """What one file is called inside the archive.

    Args:
        source: The file going in.
        naming: One of :data:`NAMING`. Anything else is read as :data:`RELATIVE`.

    Returns:
        The entry name, spelled with ``/`` and never starting with one. On :data:`TAGGED` the
        root's tag becomes the first folder, so two files of the same name from two folders
        stay apart; on :data:`FLATTEN` the folders are dropped.
    """
    relative = (source.relative or source.path.name).replace("\\", "/").lstrip("/")
    if naming == FLATTEN:
        return relative.rsplit("/", 1)[-1] or source.path.name
    if naming == TAGGED and source.tag:
        return f"{source.tag}/{relative}"
    return relative


def build(
    target: Path,
    sources: Sequence[Source],
    naming: str = RELATIVE,
    compression: str = "deflate",
) -> Written:
    """Write the archive into ``<target>.part`` beside the target, then move it onto the target.

    Args:
        target: Where the archive goes, already resolved through
            ``modules.util.sandbox.resolve_write_file``. Its folder is expected to exist. An
            archive already there is replaced only once the new one is complete.
        sources: The files going in, in the order they go in.
        naming: One of :data:`NAMING`.
        compression: A key of :data:`COMPRESSIONS`. Anything else falls back to ``deflate``.

    Returns:
        What was written, including the sources that were skipped and why.

    Raises:
        ValueError: More than :data:`MAX_FILES` sources, or they hold more than
            :data:`MAX_TOTAL_BYTES` between them.
        OSError: The archive itself could not be written.
    """
    _require_bounded(sources)
    method = COMPRESSIONS.get(str(compression), zipfile.ZIP_DEFLATED)
    names: list[str] = []
    renamed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    taken = 0
    used: set[str] = set()
    part = target.with_name(target.name + PART_SUFFIX)

    try:
        with _created(part) as handle, zipfile.ZipFile(
            handle, "w", method, allowZip64=True
        ) as archive:
            for source in sources:
                wanted = entry_name(source, naming)
                name = _unique(wanted, used)
                if name != wanted:
                    renamed[name] = source.label
                try:
                    written = _put(archive, source, name, method)
                except OSError as error:
                    skipped[source.label] = str(error)
                    logger.warning(
                        "%s could not be read and is not in the archive (%s)",
                        source.path, error,
                    )
                    continue
                used.add(name.casefold())
                names.append(name)
                taken += written
        os.replace(part, target)
    except BaseException:
        # Only the part-written file is removed; an archive already at the target is kept.
        _discard(part)
        raise

    return Written(tuple(names), taken, _size(target), renamed, skipped)


# ---------------------------------------------------------------------- helpers


def _require_bounded(sources: Sequence[Source]) -> None:
    """Refuse a set of files that is too large to archive in one go.

    Args:
        sources: The files going in.

    Raises:
        ValueError: There are more than :data:`MAX_FILES` of them, or they hold more than
            :data:`MAX_TOTAL_BYTES` between them.
    """
    if len(sources) > MAX_FILES:
        raise ValueError(
            f"an archive is written from at most {MAX_FILES} files and this one was given "
            f"{len(sources)}.\n"
            f"  Split the selection across several archives, or archive a folder with a "
            f"dedicated tool."
        )
    declared = 0
    for source in sources:
        try:
            declared += max(0, os.path.getsize(source.path))
        except OSError:
            # A file that cannot be stat'd is reported when it is read, one line at a time,
            # rather than failing the whole archive here.
            continue
    if declared > MAX_TOTAL_BYTES:
        raise ValueError(
            f"the chosen files hold {_gigabytes(declared)} between them and an archive is "
            f"written from at most {_gigabytes(MAX_TOTAL_BYTES)}.\n"
            f"  Archive fewer files at a time."
        )


def _put(archive: zipfile.ZipFile, source: Source, name: str, method: int) -> int:
    """Stream one file into the archive.

    Args:
        archive: The archive being written.
        source: The file going in.
        name: The entry name it goes in under.
        method: The zip compression method.

    Returns:
        Bytes read out of the file.

    Raises:
        OSError: The file could not be opened or measured, which is before anything about it
            has reached the archive, so the caller may skip it and carry on.
        WriteFailed: The file failed while it was being copied in. An entry that has begun
            cannot be taken back out, so the archive now holds part of a file and the whole
            write is abandoned instead.
    """
    stamp = os.stat(source.path)
    taken = 0
    with open(source.path, "rb") as handle:
        info = zipfile.ZipInfo(name, date_time=_when(stamp.st_mtime))
        info.compress_type = method
        info.external_attr = ENTRY_MODE
        # Set so a reader sees the real figure in the index rather than a zero, and so a
        # zip64 record is written for a file that needs one. zipfile corrects it afterwards
        # from what was actually written.
        info.file_size = stamp.st_size
        try:
            with archive.open(info, "w") as target:
                while True:
                    block = handle.read(CHUNK)
                    if not block:
                        break
                    target.write(block)
                    taken += len(block)
        except OSError as error:
            raise WriteFailed(
                f"{source.path} stopped being readable while it was going into the archive "
                f"({error}).\n"
                f"  An entry that has been started cannot be removed again, so the archive "
                f"would hold part of that file. Nothing was written."
            ) from error
    return taken


def _when(mtime: float) -> tuple[int, int, int, int, int, int]:
    """A file's modification time as a zip timestamp.

    Args:
        mtime: Seconds since the epoch, as ``os.stat`` reports them.

    Returns:
        ``(year, month, day, hour, minute, second)`` in local time. A moment a zip cannot
        record, which is anything before 1980, becomes :data:`ZIP_FLOOR`.
    """
    try:
        parts = time.localtime(mtime)[:6]
    except (OSError, OverflowError, ValueError):
        return ZIP_FLOOR
    if parts[0] < 1980:
        return ZIP_FLOOR
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])


def _unique(name: str, used: set[str]) -> str:
    """A name no earlier entry has taken.

    Args:
        name: The name the naming rule produced.
        used: Casefolded names already written.

    Returns:
        ``name``, or the same name with ``_2``, ``_3`` and so on before its extension.
        Comparison is casefolded, so ``Cat.png`` and ``cat.png`` count as the same name.
    """
    if name.casefold() not in used:
        return name
    stem, dot, extension = name.rpartition(".")
    if not dot:
        stem, extension = name, ""
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{dot}{extension}"
        if candidate.casefold() not in used:
            return candidate
        counter += 1


def _created(part: Path):
    """Open a new, empty file to write an archive into, replacing a stale one of that name.

    Args:
        part: The file to create, beside the archive's target.

    Returns:
        A binary file object open for writing.

    Raises:
        OSError: The file could not be created.
    """
    if os.path.lexists(part):
        os.unlink(part)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(part, flags, 0o666), "wb")


def _discard(target: Path) -> None:
    """Remove a part-written archive, reporting a failure to remove it and nothing else."""
    try:
        if target.exists():
            target.unlink()
    except OSError as error:
        logger.warning(
            "the part-written archive %s could not be removed (%s); delete it by hand, it "
            "holds less than its own index says",
            target, error,
        )


def _size(target: Path) -> int:
    """How many bytes the finished archive occupies, or 0 when that cannot be read."""
    try:
        return max(0, target.stat().st_size)
    except OSError:
        return 0


def _gigabytes(count: int) -> str:
    """A byte count as a rounded number of gigabytes, for a message."""
    return f"{count / (1024 ** 3):.1f} GB"
