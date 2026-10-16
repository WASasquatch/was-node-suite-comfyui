"""Which entries inside an archive a pattern picks, and reading them one by one.

At most :data:`MAX_FILES` files come back, :data:`MAX_BYTES_PER_FILE` from one of them and
:data:`MAX_BYTES_TOTAL` in all.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from ..document import container as document_container
from ..util import sandbox
from . import container, kinds
from .summary import size_text

__all__ = [
    "DUPLICATE",
    "MAX_BYTES_PER_FILE",
    "MAX_BYTES_TOTAL",
    "MAX_EXAMPLES",
    "MAX_FILES",
    "MAX_LEVELS",
    "MAX_NOTES",
    "MAX_WILDCARDS",
    "Member",
    "NOT_AN_IMAGE",
    "NOT_A_DOCUMENT",
    "NOT_UTF8",
    "REASONS",
    "TOO_MANY_PIXELS",
    "Report",
    "UNREADABLE",
    "WRONG_KIND",
    "fingerprint",
    "matches",
    "opened_archive",
    "read_matching",
    "refuse_document_container",
    "resolved_archive",
]

#: How many files one read hands back. A file costs a name, its contents and a result row
#: whatever its size, and a list output runs everything below it once per file, so the bound
#: is on the count and not only on the bytes.
MAX_FILES = 1024

#: How much any one file may unpack to here. Above this it is named and skipped, which covers
#: an entry built to unpack to far more than it occupies.
MAX_BYTES_PER_FILE = 16 * 1024 * 1024

#: How much one read may unpack in total. Reached, the read stops and says so, so a thousand
#: large files cannot exhaust memory one file at a time.
MAX_BYTES_TOTAL = 64 * 1024 * 1024

#: How many skipped entries are named individually. The counts are complete either way.
MAX_NOTES = 20

#: How many of an archive's own names a report offers, for the message a node builds when a
#: pattern picked none of them.
MAX_EXAMPLES = 12

#: Most ``*`` wildcards one pattern may hold.
MAX_WILDCARDS = 32

#: Most folder levels one pattern may name.
MAX_LEVELS = 64

#: How much of a media type entry is read before it is compared. A media type is one short
#: line, so anything longer is not one whatever the entry is called.
MEDIA_TYPE_BYTES = 256

#: Why an entry was left out, beyond the container's own refusals. Every skip carries one
#: reason, and :data:`REASONS` phrases it. The last four are raised by the node rather than
#: here, since only the node knows what the bytes it asked for were meant to be.
DUPLICATE = "duplicate"
WRONG_KIND = "wrong_kind"
UNREADABLE = "unreadable"
NOT_UTF8 = "not_utf8"
NOT_A_DOCUMENT = "not_a_document"
NOT_AN_IMAGE = "not_an_image"
TOO_MANY_PIXELS = "too_many_pixels"

#: How each reason reads after a count: "5 of a kind this node does not read". The
#: container's refusals keep their own wording, so an entry refused for the same reason reads
#: the same way wherever it is reported.
REASONS = {
    DUPLICATE: "replaced by a later entry of the same name",
    WRONG_KIND: "of a kind this node does not read",
    UNREADABLE: "damaged, and could not be unpacked",
    NOT_UTF8: "not UTF-8 text",
    NOT_A_DOCUMENT: "not documents after all",
    NOT_AN_IMAGE: "not pictures after all",
    TOO_MANY_PIXELS: "pictures too large to decode",
    container.ESCAPES: "named outside the folder they would be read into",
    container.NUL_NAME: "carrying a null byte in the name",
    container.SYMLINK: "symbolic links rather than files",
    container.RESERVED: "named after a Windows device",
    container.ENCRYPTED: "encrypted",
    container.OVERSIZE: "larger than one file may unpack to",
}


@dataclasses.dataclass(frozen=True)
class Member:
    """One entry that was read.

    Attributes:
        name: The entry's relative name inside the archive, spelled with ``/``.
        data: Its bytes.
    """

    name: str
    data: bytes


@dataclasses.dataclass
class Report:
    """What one read of an archive did, and what it left out.

    Attributes:
        examined: Entries the archive's index offered.
        matched: Entries the pattern picked, before any of them were skipped.
        chosen: Entries of the wanted kind the pattern picked, which is what a ``start``
            counts through and what a node calls the number of files it found.
        reached: How many entries from ``start`` the read got through, so the next page
            begins at ``start + reached``. Counts the ones skipped as damaged, and not the
            one the byte total stopped at, since that one has still to be read.
        skipped: ``{cause: count}``, keyed on the skip constants.
        notes: One line per skipped entry, at most :data:`MAX_NOTES` of them.
        bounds: One line per bound that stopped the read early.
        examples: Names the archive holds, for the message a node builds when a pattern
            picked none of them.
    """

    examined: int = 0
    matched: int = 0
    chosen: int = 0
    reached: int = 0
    skipped: Counter = dataclasses.field(default_factory=Counter)
    notes: list[str] = dataclasses.field(default_factory=list)
    bounds: list[str] = dataclasses.field(default_factory=list)
    examples: list[str] = dataclasses.field(default_factory=list)

    def skip(self, reason: str, message: str, count: int = 1) -> None:
        """Record entries that were left out.

        Args:
            reason: One of the skip constants, or one of the container's refusals. Decides
                how the count reads in :meth:`summary`.
            message: The line naming the entry, for the log.
            count: How many entries this line accounts for.
        """
        self.skipped[reason] += count
        if len(self.notes) < MAX_NOTES:
            self.notes.append(message)

    def bound(self, message: str) -> None:
        """Record that a limit stopped the read, with the line naming which."""
        self.bounds.append(message)

    @property
    def total(self) -> int:
        """How many entries were left out, on every count."""
        return sum(self.skipped.values())

    def summary(self, read: int) -> str:
        """One line saying what was read and what was not.

        Args:
            read: How many files the node ended up with, which is at most the number of
                members handed back, since a node also skips what it cannot parse.

        Returns:
            The count read, then each reason with its count, then any bound that stopped
            the read.
        """
        parts = [f"{read} file(s) read of {self.matched} the pattern picked"]
        if self.total:
            reasons = ", ".join(
                f"{count} {REASONS.get(reason, reason)}"
                for reason, count in sorted(self.skipped.items())
            )
            parts.append(f"{self.total} skipped ({reasons})")
        parts += self.bounds
        return "; ".join(parts)


def resolved_archive(value: str) -> Path:
    """The archive one widget value names, resolved and confirmed to be a file there.

    Args:
        value: The raw widget value.

    Returns:
        The absolute path, inside a permitted read root.

    Raises:
        NotAnArchive: The value is empty, names a folder, or names nothing that is there.
        PathNotAllowed: It resolved outside every permitted read root.
    """
    text = str(value).strip()
    if not text:
        raise container.NotAnArchive(
            "no archive was given.\n"
            "  Type the path of a zip file, such as 'captions.zip' for one in the folder "
            "ComfyUI was started in, or wire a path in from a node that writes one."
        )
    path = sandbox.resolve_read(text)
    if path.is_dir():
        raise container.NotAnArchive(
            f"{path} is a folder rather than a zip file.\n"
            f"  This node reads the files inside one archive. To read a folder of files as "
            f"they sit on disk, use Load Text File or Load Text Line instead."
        )
    if not path.exists():
        raise container.NotAnArchive(
            f"the archive {path} cannot be found.\n"
            f"  A path with no folders in it is read against the folder ComfyUI was started "
            f"in and nowhere else, so give the whole path, or put the file in ComfyUI's "
            f"input folder and name it 'input/{Path(text).name}'."
        )
    return path


def opened_archive(value) -> container.Archive:
    """The archive one widget value names, opened and indexed.

    Args:
        value: The raw widget value, or an archive a ZIP socket already carries, which is
            answered as it stands.

    Returns:
        The archive, holding its index. The file itself is closed again, and each read
        reopens it.

    Raises:
        NotAnArchive: The value is empty, names a folder, names nothing that is there, or
            names a file that is not a readable zip.
        PathNotAllowed: It resolved outside every permitted read root.
    """
    if container.is_archive(value):
        return value
    return container.Archive.from_path(resolved_archive(value))


def refuse_document_container(archive: container.Archive, advice: str) -> None:
    """Refuse an archive that is one document rather than a set of files.

    Args:
        archive: The opened archive.
        advice: What the node reading it offers instead, added to the message.

    Raises:
        NotAnArchive: The archive holds a document and nothing else.
    """
    if not _is_document_container(archive):
        return
    raise container.NotAnArchive(
        f"{archive.label} is one document rather than an archive of files: it holds "
        f"{document_container.CONTENT_ENTRY} and the {document_container.MIMETYPE_ENTRY} "
        f"entry that every {document_container.SUFFIX} carries inside it.\n  {advice}"
    )


def _is_document_container(archive: container.Archive) -> bool:
    """Whether an archive is a document container and holds nothing besides one.

    Args:
        archive: The opened archive.

    Returns:
        True when the fixed entries are there, nothing else is except embedded files, and
        the media type reads exactly as a document's. An archive holding a document's parts
        alongside other files is a set of files and is read as one.
    """
    names = set(archive.names)
    fixed = {
        document_container.CONTENT_ENTRY,
        document_container.METADATA_ENTRY,
        document_container.MIMETYPE_ENTRY,
    }
    if not {document_container.CONTENT_ENTRY, document_container.MIMETYPE_ENTRY} <= names:
        return False
    if any(
        name not in fixed and not name.startswith(document_container.ASSET_PREFIX)
        for name in names
    ):
        return False
    try:
        declared = archive.read(document_container.MIMETYPE_ENTRY, limit=MEDIA_TYPE_BYTES)
    except container.ArchiveError:
        return False
    return declared.strip() == document_container.MIMETYPE.encode("ascii")


def fingerprint(value: str, *extra: object) -> str | float:
    """A value that changes when the archive on disk, or the selection, changes.

    Args:
        value: The raw path widget value, or an archive a ZIP socket carries.
        extra: The other widget values the selection depends on.

    Returns:
        The path, its modification time and its size, with ``extra`` appended. ``NaN`` where
        the file cannot be stated, which never equals itself and so leaves the node to run
        and report what stopped it.
    """
    if container.is_archive(value):
        # The archive already carries what it was opened from and how it looked then.
        return "|".join([str(value.source or value.label), value.revision, *map(str, extra)])
    if not str(value).strip():
        return float("NaN")
    try:
        stamp = sandbox.resolve_read(value).stat()
    except (OSError, ValueError):
        return float("NaN")
    return "|".join([str(value), str(stamp.st_mtime_ns), str(stamp.st_size), *map(str, extra)])


def matches(name: str, pattern: str) -> bool:
    """Whether a pattern picks one entry.

    Args:
        name: The entry's relative name inside the archive, spelled with ``/``.
        pattern: A glob. ``*`` and ``?`` stand for anything within one folder level,
            ``[abc]`` and ``[!abc]`` for one character listed or not listed, and ``**``
            for any number of levels. A pattern holding no ``/`` is matched against
            the file's own name at any depth, so ``*.txt`` finds one in every folder, while
            ``notes/*.txt`` is anchored at the top of the archive. Empty picks everything,
            and case is ignored.

    Returns:
        True when the entry is picked.

    Raises:
        ValueError: The pattern holds more than :data:`MAX_WILDCARDS` ``*`` wildcards or
            :data:`MAX_LEVELS` folder levels, or a character class that cannot be read.
    """
    text = (pattern or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text:
        return True
    target = name if "/" in text else name.rsplit("/", 1)[-1]
    return _fits(_compiled(text), target.split("/"))


def read_matching(
    archive: container.Archive,
    pattern: str,
    kind: str,
    report: Report | None = None,
    start: int = 0,
    count: int = 0,
) -> tuple[list[Member], Report]:
    """Read the files of one kind that a pattern picks, or one page of them.

    Args:
        archive: The opened archive.
        pattern: The glob, as :func:`matches` reads it.
        kind: One of :data:`modules.archive.kinds.KINDS`.
        report: A report to fill in, or ``None`` for a new one.
        start: How many of the picked entries to pass over, counting through them in name
            order. Entries before it are never opened.
        count: How many to read from ``start``, or 0 for as many as the bounds allow.

    Returns:
        ``(members, report)``, the members sorted by name.
    """
    report = report or Report()
    report.examined = len(archive.entries)
    report.examples = list(archive.names[:MAX_EXAMPLES])
    _refusals(archive, report)
    found = _chosen(archive, pattern, kind, report)
    return _read(archive, _paged(found, start, count, report), report), report


# ---------------------------------------------------------------------- helpers


def _refusals(archive: container.Archive, report: Report) -> None:
    """Count what the container would not offer.

    Args:
        archive: The opened archive.
        report: Filled in with one line per refused entry, with the count of names a later
            entry replaced, and with the bound where the index was capped.
    """
    for entry in archive.refused:
        told = container.REFUSALS.get(entry.refusal, "was refused")
        report.skip(entry.refusal, f"the entry {entry.stored!r} {told}")
    readable = [entry for entry in archive.entries if not entry.refused]
    repeated = len(readable) - len(archive.files)
    if repeated > 0:
        # Two entries under one name are one file to every zip reader, which resolves the
        # name to the last of them, so the earlier one is not a file that was read.
        report.skip(
            DUPLICATE,
            f"{repeated} entry(ies) repeat a name the archive already holds; the last entry "
            f"of a repeated name is the one read, as every zip reader resolves it",
            repeated,
        )
    if archive.truncated:
        report.bound(
            f"the archive holds {archive.held} entries and the first {len(archive.entries)} "
            f"were listed"
        )


def _chosen(
    archive: container.Archive, pattern: str, kind: str, report: Report
) -> list[container.Entry]:
    """Every entry of one kind the pattern picks, sorted by name."""
    found = []
    for entry in archive.files:
        if not matches(entry.name, pattern):
            continue
        report.matched += 1
        if entry.kind != kind:
            report.skip(
                WRONG_KIND,
                f"the entry {entry.name!r} is not one of {kinds.extension_list(kind)}, so it "
                f"was skipped",
            )
            continue
        if entry.size > MAX_BYTES_PER_FILE:
            # The declared size is compared and never allocated from: an index states only
            # what the file claims, and a crafted one claims far more than it holds.
            report.skip(
                container.OVERSIZE,
                f"the entry {entry.name!r} says it unpacks to {size_text(entry.size)}, past "
                f"the {size_text(MAX_BYTES_PER_FILE)} one file may unpack to here, and was "
                f"not read",
            )
            continue
        found.append(entry)
    found.sort(key=lambda entry: (entry.name.casefold(), entry.name))
    report.chosen = len(found)
    return found


def _paged(
    found: Sequence[container.Entry], start: int, count: int, report: Report
) -> list[container.Entry]:
    """The page of the chosen entries a read opens, capped at :data:`MAX_FILES`.

    Args:
        found: What :func:`_chosen` picked, sorted by name.
        start: How many to pass over.
        count: How many to take from there, or 0 for the rest of them.
        report: Filled in with the bound where :data:`MAX_FILES` narrowed the page.

    Returns:
        The entries to read, in the order they are read.
    """
    begin = max(0, int(start))
    wanted = max(0, int(count)) or len(found)
    page = list(found[begin:begin + wanted])
    if len(page) > MAX_FILES:
        report.bound(
            f"{len(page)} file(s) were chosen and the first {MAX_FILES} by name were read, "
            f"which is as many as one node hands on; narrow the pattern to choose which"
        )
        page = page[:MAX_FILES]
    return page


def _read(
    archive: container.Archive, entries: Sequence[container.Entry], report: Report
) -> list[Member]:
    """Read one page of entries, one damaged entry costing only itself.

    Args:
        archive: The opened archive.
        entries: What :func:`_paged` chose, in the order they are read.
        report: Filled in with each entry that could not be read, with how far the read got,
            and with the bound where the byte total stopped it.

    Returns:
        One member per entry that was read.
    """
    members: list[Member] = []
    remaining = MAX_BYTES_TOTAL
    index = 0
    spent = False
    while index < len(entries):
        if remaining <= 0:
            spent = True
            break
        taken = 0
        try:
            for name, data in archive.read_many(
                entries[index:], limit=MAX_BYTES_PER_FILE, total=remaining
            ):
                members.append(Member(name, data))
                remaining -= len(data)
                taken += 1
        except container.BudgetSpent:
            # The entry the total stopped at is intact and unread, so it is not counted as
            # skipped and not passed over: the next page starts on it.
            index += taken
            spent = True
            break
        except container.EntryTooLarge as error:
            report.skip(container.OVERSIZE, str(error))
            taken += 1
        except container.ArchiveError as error:
            # The entry a read stopped on is reported here; the entries after it are read
            # on the next pass.
            report.skip(UNREADABLE, str(error))
            taken += 1
        index += taken
    report.reached = index
    if spent:
        report.bound(
            f"the read stopped at {size_text(MAX_BYTES_TOTAL)}, which is as much as one node "
            f"unpacks in one run, with {len(entries) - index} file(s) left unread; nothing is "
            f"wrong with them, and a narrower pattern reaches the rest"
        )
    return members


@functools.lru_cache(maxsize=64)
def _compiled(pattern: str) -> tuple:
    """One glob as its folder levels, held for reuse across entries.

    Args:
        pattern: The glob, already spelled with ``/``.

    Returns:
        One entry per folder level: None for a ``**`` level, otherwise what :func:`_level`
        answers for it.

    Raises:
        ValueError: The glob holds more than :data:`MAX_WILDCARDS` ``*`` wildcards or
            :data:`MAX_LEVELS` folder levels, or a character class that cannot be read.
    """
    stars = pattern.count("*")
    if stars > MAX_WILDCARDS:
        raise ValueError(
            f"The pattern {pattern!r} holds {stars} * wildcards, and at most {MAX_WILDCARDS} "
            f"are read. Use fewer, such as '*.png' or 'frames/**/*.png'."
        )
    levels = pattern.split("/")
    if len(levels) > MAX_LEVELS:
        raise ValueError(
            f"The pattern {pattern!r} names {len(levels)} folder levels, and at most "
            f"{MAX_LEVELS} are read. Stand for the levels between with '**', as "
            f"'frames/**/*.png'."
        )
    return tuple(None if level == "**" else _level(level, pattern) for level in levels)


def _level(level: str, pattern: str) -> tuple[tuple, bool]:
    """One folder level of a glob as the fixed runs between its ``*`` wildcards.

    Args:
        level: The level, holding no ``/``.
        pattern: The whole glob, for the message.

    Returns:
        ``(runs, starred)``. ``runs`` holds one ``(expression, width)`` per stretch between
        ``*`` wildcards, each expression matching exactly ``width`` characters with case
        ignored. ``starred`` is whether the level holds a ``*``.

    Raises:
        ValueError: A character class cannot be read.
    """
    runs = []
    pieces: list[str] = []
    starred = False
    index = 0
    while index < len(level):
        char = level[index]
        index += 1
        if char == "*":
            runs.append(pieces)
            pieces = []
            starred = True
        elif char == "?":
            pieces.append("[^/]")
        elif char == "[":
            close = level.find("]", index)
            # An unclosed or empty class is a literal bracket.
            if close < 0 or level[index:close] in ("", "!", "^"):
                pieces.append(re.escape(char))
                continue
            body = level[index:close].replace("\\", "\\\\")
            index = close + 1
            pieces.append(f"[{'^' + body[1:] if body.startswith('!') else body}]")
        else:
            pieces.append(re.escape(char))
    runs.append(pieces)
    try:
        compiled = tuple(
            (re.compile("".join(run), re.IGNORECASE), len(run)) for run in runs
        )
    except re.error as error:
        raise ValueError(
            f"The pattern {pattern!r} holds a character class that cannot be read ({error}). "
            f"Write a range low to high, as '[a-z]', or list the characters, as '[abc]' or "
            f"'[!abc]'."
        ) from error
    return compiled, starred


def _level_fits(level: tuple[tuple, bool], text: str) -> bool:
    """Whether one folder level of a glob matches one folder level of a name.

    Args:
        level: What :func:`_level` answered.
        text: The name's folder level, holding no ``/``.

    Returns:
        True on a match.
    """
    runs, starred = level
    if not starred:
        expression, width = runs[0]
        return width == len(text) and expression.fullmatch(text) is not None
    (head, head_width), *middle, (tail, tail_width) = runs
    end = len(text) - tail_width
    if end < head_width:
        return False
    if head_width and head.match(text) is None:
        return False
    if tail_width and tail.fullmatch(text, end) is None:
        return False
    position = head_width
    for expression, width in middle:
        if width:
            found = expression.search(text, position, end)
            if found is None:
                return False
            position = found.end()
    return True


def _fits(levels: tuple, parts: list[str]) -> bool:
    """Whether a glob's folder levels match a name's.

    Args:
        levels: What :func:`_compiled` answered.
        parts: The name cut at every ``/``.

    Returns:
        True where each level matches in turn. A ``**`` level takes any number of non-empty
        folder levels, and as the last level takes everything left.
    """
    if None not in levels and len(levels) != len(parts):
        return False
    final = len(parts) - 1
    # The indices of the parts the next level may start on, in ascending order.
    reach = [0]
    for position, level in enumerate(levels):
        if position == len(levels) - 1:
            if level is None:
                return True
            return reach[-1] == final and _level_fits(level, parts[final])
        following = []
        if level is None:
            covered = -1
            for start in reach:
                if start <= covered:
                    continue
                index = start
                following.append(index)
                while index < final and parts[index] != "":
                    index += 1
                    following.append(index)
                covered = index
        else:
            following = [
                index + 1 for index in reach
                if index < final and _level_fits(level, parts[index])
            ]
        if not following:
            return False
        reach = following
    return False
