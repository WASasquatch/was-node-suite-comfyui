"""The numbered file name the pack's save nodes write to.

The convention is ``<prefix><delimiter><counter><extension>``, or the counter first when
``number_first`` is set. The counter continues past whatever is already in the folder.
"""

from __future__ import annotations

import os
import re

__all__ = ["format_name", "next_counter", "next_name", "next_names"]


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
    suffix = extension if extension.startswith(".") else f".{extension}"
    number = f"{counter:0{max(1, int(padding))}}"
    if number_first:
        return f"{number}{delimiter}{prefix}{suffix}"
    return f"{prefix}{delimiter}{number}{suffix}"


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

    Returns:
        ``count`` file names, without a folder, each one free of any file already in
        ``directory``.

    Raises:
        PathNotAllowed: The prefix and delimiter make a name that names another machine,
            carries a drive, starts at a root or climbs out with ``..``.
    """
    from ..util import sandbox

    suffix = extension if extension.startswith(".") else f".{extension}"
    if overwrite:
        return [f"{prefix}{suffix}"] * count
    counter = next_counter(directory, prefix, delimiter, number_first)
    names = []
    for _ in range(count):
        name = format_name(prefix, delimiter, counter, padding, extension, number_first)
        while _taken(directory, name, sandbox):
            counter += 1
            name = format_name(prefix, delimiter, counter, padding, extension, number_first)
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
