"""The numbered file name the pack's save nodes write to.

The convention is ``<prefix><delimiter><counter><extension>``, or the counter first when
``number_first`` is set. The counter continues past whatever is already in the folder.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "IMAGE_SUFFIXES", "format_name", "given_names", "next_counter", "next_name", "next_names",
    "stem_of",
]

#: Extensions an image name may arrive carrying, dropped by :func:`stem_of`.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".webp", ".bmp", ".exr")


def _suffix(extension: str) -> str:
    """An extension with its leading dot."""
    return extension if extension.startswith(".") else f".{extension}"


def _numbered(prefix: str, delimiter: str, counter: int, padding: int, number_first: bool) -> str:
    """The name one counter value spells, without an extension."""
    number = f"{counter:0{max(1, int(padding))}}"
    if number_first:
        return f"{number}{delimiter}{prefix}"
    return f"{prefix}{delimiter}{number}"


def format_name(
    prefix: str,
    delimiter: str,
    counter: int,
    padding: int,
    extension: str,
    number_first: bool = False,
) -> str:
    """The file name one counter value spells.

    Args:
        prefix: Text beside the number.
        delimiter: What separates the prefix from the number.
        counter: The number to write.
        padding: Digits the number is padded to.
        extension: File extension, with or without its dot.
        number_first: Put the number before the prefix rather than after it.

    Returns:
        The file name, without a folder.
    """
    return _numbered(prefix, delimiter, counter, padding, number_first) + _suffix(extension)


def next_counter(
    directory: str, prefix: str, delimiter: str, number_first: bool = False
) -> int:
    """The next unused number in a folder, for the pack's naming convention.

    Args:
        directory: Folder the files are written to.
        prefix: Text beside the number.
        delimiter: What separates the prefix from the number.
        number_first: Read the number before the prefix rather than after it.

    Returns:
        One past the highest number already present, and 1 for an empty folder.
    """
    name, number = re.escape(prefix), r"(\d+)"
    order = (number, name) if number_first else (name, number)
    pattern = f"{order[0]}{re.escape(delimiter)}{order[1]}"
    seen = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return 1
    for entry in entries:
        found = re.match(pattern, os.path.basename(entry))
        if found:
            seen.append(int(found.group(1)))
    return max(seen) + 1 if seen else 1


def next_names(
    directory: str,
    prefix: str,
    delimiter: str,
    padding: int,
    extension: str,
    count: int = 1,
    overwrite: bool = False,
    number_first: bool = False,
    beside: tuple[str, ...] = (),
    claimed: set | None = None,
) -> list[str]:
    """The file names to write next, numbered in sequence.

    Args:
        directory: Folder the files are written to.
        prefix: Text beside the number.
        delimiter: What separates the prefix from the number.
        padding: Digits the number is padded to.
        extension: File extension, with or without its dot.
        count: How many names to answer.
        overwrite: Answer ``<prefix><extension>`` for every one of them.
        number_first: Put the number before the prefix rather than after it.
        beside: Extensions of files written under the same name, as ``(".txt",)``. A number
            is passed over where any of them is already a file.
        claimed: Paths the run has named so far, lower case. A number naming one is passed
            over, and every name answered is added to it.

    Returns:
        ``count`` file names, without a folder, each one free of any file already in
        ``directory``.

    Raises:
        PathNotAllowed: The prefix and delimiter make a name that names another machine,
            carries a drive, starts at a root or climbs out with ``..``.
    """
    from ..util import sandbox

    suffix = _suffix(extension)
    used = claimed if claimed is not None else set()
    if overwrite:
        names = [f"{prefix}{suffix}"] * count
        used.update(os.path.join(directory, name).lower() for name in names)
        return names
    counter = next_counter(directory, prefix, delimiter, number_first)
    names = []
    for _ in range(count):
        while True:
            stem = _numbered(prefix, delimiter, counter, padding, number_first)
            name = f"{stem}{suffix}"
            keys = [os.path.join(directory, f"{stem}{_suffix(other)}").lower()
                    for other in (suffix, *beside)]
            taken = _taken(directory, name, sandbox) or any(key in used for key in keys) or any(
                os.path.exists(os.path.join(directory, f"{stem}{_suffix(other)}"))
                for other in beside
            )
            if not taken:
                break
            counter += 1
        used.update(keys)
        names.append(name)
        counter += 1
    return names


def _taken(directory: str, name: str, sandbox) -> bool:
    """Whether ``name`` is already a file in ``directory``, refusing a name that leaves it."""
    reason = sandbox.leaves_folder(name)
    if reason is not None:
        raise sandbox.PathNotAllowed(
            f"the file name `{name}` {reason}, so it would be written outside `{directory}`. "
            f"Use a prefix and delimiter without slashes, drives or '..'"
        )
    return os.path.exists(os.path.join(directory, name))


def next_name(
    directory: str,
    prefix: str,
    delimiter: str,
    padding: int,
    extension: str,
    overwrite: bool = False,
    number_first: bool = False,
) -> str:
    """The file name to write next.

    Args:
        directory: Folder the file is written to.
        prefix: Text beside the number.
        delimiter: What separates the prefix from the number.
        padding: Digits the number is padded to.
        extension: File extension, with or without its dot.
        overwrite: Answer ``<prefix><extension>`` and reuse it every run.
        number_first: Put the number before the prefix rather than after it.

    Returns:
        The file name, without a folder.
    """
    return next_names(
        directory, prefix, delimiter, padding, extension, 1, overwrite, number_first
    )[0]


def stem_of(name) -> str:
    """A name's last segment with any image extension dropped, as ``cat`` from ``photos/cat.png``.

    Args:
        name: A file name or path, with either kind of slash.

    Returns:
        The stem without surrounding spaces, empty where the name is.
    """
    leaf = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    root, suffix = os.path.splitext(leaf)
    return root.strip() if suffix.lower() in IMAGE_SUFFIXES else leaf


def given_names(
    directory: str,
    stems,
    delimiter: str,
    padding: int,
    extension: str,
    overwrite: bool = False,
    number_first: bool = False,
    beside: tuple[str, ...] = (),
    claimed: set | None = None,
) -> list[str]:
    """File names for names handed in, each kept while free and numbered once taken.

    Args:
        directory: Folder the files are written to.
        stems: One name per file, without a folder or extension, as from :func:`stem_of`.
        delimiter: What separates a taken name from its number.
        padding: Digits the number is padded to.
        extension: File extension, with or without its dot.
        overwrite: Keep a name a file in ``directory`` already has, replacing that file.
        number_first: Put the number before the name rather than after it.
        beside: Extensions of files written under the same name, as ``(".txt",)``. A name
            is free only where none of them is taken either.
        claimed: Paths this run has already named, lower case. A name in it is numbered
            whatever ``overwrite`` says, and every name answered is added to it.

    Returns:
        One file name per stem, without a folder.

    Raises:
        PathNotAllowed: A name names another machine, carries a drive, starts at a root,
            climbs out with ``..`` or names a Windows device or stream.
    """
    from ..util import sandbox

    suffixes = (_suffix(extension), *(_suffix(other) for other in beside))
    used = claimed if claimed is not None else set()
    try:
        present = {entry.lower() for entry in os.listdir(directory)}
    except OSError:
        present = set()

    def keys(stem):
        return [os.path.join(directory, f"{stem}{suffix}").lower() for suffix in suffixes]

    def free(stem, replace):
        if any(key in used for key in keys(stem)):
            return False
        return replace or not any(f"{stem}{suffix}".lower() in present for suffix in suffixes)

    names = []
    for stem in stems:
        name = f"{stem}{suffixes[0]}"
        reason = sandbox.leaves_folder(name)
        if reason is not None:
            raise sandbox.PathNotAllowed(
                f"the file name `{name}` {reason}, so it would be written outside "
                f"`{directory}`. Wire names without slashes, drives or '..'"
            )
        if not free(stem, overwrite):
            counter = next_counter(directory, stem, delimiter, number_first)
            while not free(_numbered(stem, delimiter, counter, padding, number_first), False):
                counter += 1
            stem = _numbered(stem, delimiter, counter, padding, number_first)
        used.update(keys(stem))
        names.append(f"{stem}{suffixes[0]}")
    return names
