"""Text files under ComfyUI's input and output folders and each ``paths.allow_read`` folder,
as combo entries.

A label is ``<relative path> [tag]``, the tag being ``input``, ``output`` or the configured
folder's name, and :func:`listing` maps labels to absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import file_listing

__all__ = [
    "ENCODING",
    "MAX_DEPTH",
    "MAX_EXAMINED",
    "MAX_OPTIONS",
    "NO_FILES",
    "OPTIONS_TTL",
    "TEXT_EXTENSIONS",
    "TEXT_TAGS",
    "decode",
    "is_comment",
    "listing",
    "normalize_newlines",
    "options",
    "read_text",
    "resolve",
    "roots",
    "split_lines",
]

#: The extensions offered, lowercased. One line is one record in every one of them, which
#: is what a line browser and a line loader are for. The list can only ever grow: appending
#: a combo option keeps every saved value valid, while retiring one does not.
TEXT_EXTENSIONS = (".txt", ".csv", ".tsv", ".json", ".jsonl", ".md", ".yaml", ".yml")

#: The roots this view reads, and the tag order its labels sort in.
TEXT_TAGS = (file_listing.INPUT, file_listing.OUTPUT, file_listing.CONFIGURED)

#: How many directories below a root the walk goes, from the shared walk.
MAX_DEPTH = file_listing.MAX_DEPTH

#: How many directory entries that walk examines under one root before it stops.
MAX_EXAMINED = file_listing.MAX_EXAMINED

#: How many entries reach the combo. The newest by modification time are the ones kept, so a
#: file that was just written is always in the menu, and ``/object_info`` carries this list
#: once per node per request.
MAX_OPTIONS = 500

#: Seconds a built listing is reused for, from the shared walk.
OPTIONS_TTL = file_listing.LISTING_TTL

#: Combo entry shown when no listed directory holds a text file, and outside ComfyUI.
NO_FILES = "No Text Files"

#: First non-space character marking a line as a comment.
COMMENT_PREFIX = "#"

#: The codec every one of these files is read with. The ``-sig`` form drops a leading byte
#: order mark, which is otherwise an invisible character at the front of the first line and
#: travels into whatever prompt that line becomes.
ENCODING = "utf-8-sig"

def roots() -> list[tuple[str, Path]]:
    """The directories listed, each with the tag its labels carry.

    Returns:
        ``[(tag, path)]``: ``input``, then ``output``, then each ``paths.allow_read``
        directory under its folder's name, dropping any that cannot be reached. Empty
        outside ComfyUI.
    """
    return file_listing.roots(TEXT_TAGS)


def listing() -> dict[str, str]:
    """``{combo label: absolute path}`` for every listable text file, memoized.

    Returns:
        The entries in the order the combo shows them: casefolded relative path first, then
        input before output before a configured root, then the raw path. At most
        :data:`MAX_OPTIONS` of them, the
        newest by modification time. Empty where nothing was found.
    """
    return file_listing.listing(TEXT_EXTENSIONS, TEXT_TAGS, MAX_OPTIONS)


def options() -> list[str]:
    """The combo's entries, or ``[NO_FILES]`` when there are none."""
    return list(listing()) or [NO_FILES]


def resolve(label: str) -> str | None:
    """The absolute path one combo label names.

    Args:
        label: The exact label as the combo offered it, such as
            ``prompts/animals.txt [input]``. Surrounding space is ignored; nothing else
            about it is interpreted, so it is a key and never a path.

    Returns:
        The absolute path, or ``None`` when no walked text file carries that label, which
        covers one that has been deleted, renamed, or invented. A file the menu's own limit
        left out still resolves.
    """
    return file_listing.resolve(label, TEXT_EXTENSIONS, TEXT_TAGS)


def decode(data: bytes, errors: str = "strict") -> str:
    """Bytes from one of these files as text, every line ending translated to ``\\n``.

    Args:
        data: The file's bytes, or a prefix of them.
        errors: Passed to the codec. ``"strict"`` raises on anything that is not UTF-8;
            ``"replace"`` substitutes the replacement character instead.

    Returns:
        The decoded text.

    Raises:
        UnicodeDecodeError: ``errors`` is ``"strict"`` and the bytes are not UTF-8.
    """
    return normalize_newlines(data.decode(ENCODING, errors))


def normalize_newlines(text: str) -> str:
    """Text with ``\\r\\n`` and a lone ``\\r`` translated to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text(path: str | os.PathLike) -> str:
    """One of these files as text, line endings translated.

    Args:
        path: An absolute path, already resolved through the containment layer.

    Returns:
        The whole file.

    Raises:
        OSError: The file could not be opened or read.
        UnicodeDecodeError: The file is not UTF-8.
    """
    with open(path, "rb") as handle:
        return decode(handle.read())


def split_lines(text: str) -> list[str]:
    """The lines of one of these files, in order.

    Args:
        text: The file's text, already through :func:`decode` or :func:`read_text`.

    Returns:
        One entry per line.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def is_comment(line: str) -> bool:
    """Whether a line is a comment, its first non-space character being ``#``."""
    return line.strip().startswith(COMMENT_PREFIX)
