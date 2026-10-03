"""List the files and folders in a directory, as lists a loop can walk.

A name is the path below the folder listed, spelled with ``/`` at every depth. Sizes are in
bytes and times are in seconds.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from typing import NamedTuple

from comfy_api.latest import io

from ...modules.io import picker
from ...modules import log
from ...modules.compat.types import LIST
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: What a listing holds.
INCLUDE = ("files", "directories", "both")

#: The orders a listing comes out in.
SORTS = ("name", "natural", "modified", "size")

#: Digit runs, which :func:`natural_key` compares as numbers rather than as text.
DIGITS = re.compile(r"(\d+)")


class Listed(NamedTuple):
    """One file or folder a listing holds.

    Attributes:
        name: Path below the folder listed, spelled with ``/``.
        path: Absolute path on disk.
        mtime: Modification time, in seconds.
        size: Size in bytes, which for a folder is the size of the folder itself.
    """

    name: str
    path: str
    mtime: float
    size: int


def natural_key(name: str) -> tuple:
    """A sort key reading digit runs in ``name`` as numbers.

    Args:
        name: The name to build a key from.

    Returns:
        Alternating text and number parts, so ``frame_2`` sorts before ``frame_10``. Text
        is compared case-insensitively.
    """
    parts = DIGITS.split(name)
    return tuple(
        (1, int(part), "") if index % 2 else (0, 0, part.casefold())
        for index, part in enumerate(parts)
    )


def sort_key(entry: Listed, sort: str):
    """The key one entry sorts on.

    Args:
        entry: The entry.
        sort: One of :data:`SORTS`.

    Returns:
        A key for ``list.sort``. Two entries with the same time or size fall back to
        :func:`natural_key`.
    """
    if sort == "modified":
        return (entry.mtime, natural_key(entry.name))
    if sort == "size":
        return (entry.size, natural_key(entry.name))
    if sort == "natural":
        return (natural_key(entry.name), entry.name.casefold())
    return (entry.name.casefold(), natural_key(entry.name))



def listing_directory(folder: str) -> str:
    """The absolute directory one menu label names.

    Args:
        folder: A label :func:`modules.io.picker.folders` offered.

    Returns:
        The resolved absolute directory, inside a permitted read root. Empty where the label
        names no root that is there.

    Raises:
        PathNotAllowed: The label names another machine, or the folder resolved outside
            every permitted read root.
    """
    if sandbox.names_another_host(folder or ""):
        raise sandbox.PathNotAllowed(
            f"`{folder}` names another machine. Pick a folder from the menu; a share added "
            f"under paths.allow_read in config.yaml is listed there under its own name"
        )
    found = picker.resolve_folder(folder)
    return str(found) if found else ""


def matching(directory: str, pattern: str, recursive: bool) -> list[str]:
    """Every path under ``directory`` that ``pattern`` picks.

    Args:
        directory: The directory, already resolved inside a permitted read root.
        pattern: Glob pattern, matched under ``directory``.
        recursive: Whether the pattern is applied at every depth below ``directory``.

    Returns:
        Paths as glob spelled them, unsorted. Empty where the directory does not exist.

    Raises:
        ValueError: The pattern is empty.
        PathNotAllowed: The pattern names another machine, carries a drive, starts at a
            root or climbs out with ``..``.
    """
    text = str(pattern).strip()
    if not text:
        raise ValueError(
            "no pattern given to Directory Listing. Use `*` to list everything in the "
            "folder, or `*.png` to list one kind of file"
        )
    if recursive and "**" not in text:
        return sandbox.glob_read(directory, os.path.join("**", text), recursive=True)
    return sandbox.glob_read(directory, text, recursive=recursive)


def listed(
    directory: str,
    pattern: str,
    recursive: bool,
    include: str,
    sort: str,
    reverse: bool,
    limit: int,
) -> list[Listed]:
    """The entries one listing holds, filtered, sorted and capped.

    Args:
        directory: The directory, already resolved inside a permitted read root.
        pattern: Glob pattern, matched under ``directory``.
        recursive: Whether the pattern is applied at every depth.
        include: One of :data:`INCLUDE`.
        sort: One of :data:`SORTS`.
        reverse: Whether the order is flipped.
        limit: How many entries survive, or 0 for every one.

    Returns:
        The entries in the chosen order. Empty where nothing matched. An entry resolving
        outside every permitted read root is left out and logged.

    Raises:
        PathNotAllowed: The pattern names somewhere other than inside ``directory``.
        ValueError: The pattern is empty.
    """
    found: list[Listed] = []
    for name in matching(directory, pattern, recursive):
        # The name is taken from the path glob answered rather than the resolved one.
        relative = os.path.relpath(name, directory).replace("\\", "/")
        if relative in (".", ".."):
            continue
        try:
            resolved = str(sandbox.resolve_read(name))
        except sandbox.PathNotAllowed as refused:
            logger.warning("Directory Listing leaves out %s. %s", name, refused)
            continue
        try:
            info = os.stat(resolved)
        except OSError:
            logger.debug("%s could not be read and is left out of the listing", resolved)
            continue
        is_directory = stat.S_ISDIR(info.st_mode)
        if include == "files" and is_directory:
            continue
        if include == "directories" and not is_directory:
            continue
        found.append(
            Listed(
                name=relative,
                path=resolved,
                mtime=info.st_mtime,
                size=max(0, int(info.st_size)),
            )
        )
    found.sort(key=lambda entry: sort_key(entry, sort), reverse=bool(reverse))
    return found[:limit] if limit > 0 else found


class DirectoryListing(io.ComfyNode):
    """List a folder as a LIST of paths, a LIST of names, one line each, and a count."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDirectoryListing",
            display_name="Directory Listing",
            search_aliases=[
                "WASDirectoryListing",
                "Directory Listing",
                "list files",
                "folder contents",
                "list folder",
                "glob",
                "dir",
                "ls",
                "file list",
            ],
            category="WAS Suite/IO",
            description=(
                "List what is in a folder: every full path on one wire, every name on "
                "another, and the count beside them, so a For Loop can take one file per "
                "iteration. Files of any kind, picked with a glob such as '*.png' or "
                "'frame_*', optionally descending into subfolders, ordered by name, by "
                "number, by date or by size. 'natural' order reads digit runs as numbers, "
                "so frame_2 comes before frame_10 rather than after it. 'input', 'output' "
                "and 'temp' name ComfyUI's own folders. A folder that is not there stops "
                "the prompt; a folder holding nothing the pattern picks answers empty "
                "lists and a count of 0."
            ),
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "Which folder to read. A bare 'input', 'output' or 'temp' is that "
                        "folder itself; 'plates/shot_01 [input]' is that folder below it. "
                        "Any folder added under paths.allow_read in config.yaml is listed "
                        "under its own name, and so are the folders inside it."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which entries to list, as a glob. `*` takes everything, `*.png` "
                        "only PNGs, `frame_*.png` one numbered run. A name starting with a "
                        "dot is matched only by a pattern starting with one. Matching is "
                        "inside the folder: a drive, a leading slash or a '..' segment is "
                        "refused."
                    ),
                ),
                io.Boolean.Input(
                    "recursive",
                    default=False,
                    tooltip=(
                        "Whether subfolders are listed too. false lists only what sits "
                        "directly in the folder; true applies the pattern at every depth, so "
                        "`*.png` also finds 'shot_a/frame_0.png', and every name below the "
                        "top carries its subfolder."
                    ),
                ),
                io.Combo.Input(
                    "include",
                    options=list(INCLUDE),
                    default="files",
                    tooltip=(
                        "What is listed. `files` = files alone, which is what a loop over "
                        "images wants; `directories` = folders alone, for walking a set of "
                        "shot folders; `both` = the two together in one order."
                    ),
                ),
                io.Combo.Input(
                    "sort",
                    options=list(SORTS),
                    default="natural",
                    tooltip=(
                        "The order entries come out in. `name` = plain alphabetical, where "
                        "frame_10 lands before frame_2; `natural` = digits read as numbers, "
                        "so frame_2 comes first; `modified` = oldest first; `size` = "
                        "smallest first."
                    ),
                ),
                io.Boolean.Input(
                    "reverse",
                    default=False,
                    tooltip=(
                        "Whether the order is flipped. Off, `modified` gives oldest first "
                        "and `size` smallest first; on, newest first and largest first, "
                        "which is how the most recent render is put at index 0."
                    ),
                ),
                io.Int.Input(
                    "limit",
                    default=0,
                    min=0,
                    max=1000000,
                    step=1,
                    tooltip=(
                        "How many entries to keep once they are sorted. 0 = every one; 1 = "
                        "the first alone; 50 = the first fifty. With sort on `modified` and "
                        "reverse on, 10 keeps the ten newest files."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="paths",
                    tooltip=(
                        "Every entry's full path on one wire, in the chosen order, for Text "
                        "List Get, Text List Slice and Text List Length. Wire it and count "
                        "into a For Loop to open one file per iteration."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The same entries named below the folder, as 'frame_0.png' or "
                        "'shot_a/frame_0.png', in the same order. Entry 3 here belongs to "
                        "path 3, so a saved render can carry the name it was made from."
                    ),
                ),
                io.String.Output(
                    display_name="listing",
                    tooltip=(
                        "The names, one per line, for reading on a text preview or cutting "
                        "up again with Text Split to List. Empty when nothing matched."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many entries the listing holds, which is the length of both "
                        "lists. Feed it to a For Loop's iterations to walk every one."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        folder="",
        pattern="*",
        recursive=False,
        include="files",
        sort="natural",
        reverse=False,
        limit=0,
    ):
        """A digest of the listing as it stands, compared against the last run's.

        Returns:
            The digest, or ``NaN`` when the folder is not there or the pattern is empty.

        Raises:
            PathNotAllowed: The label names another machine, the chosen folder is not one
                this pack may read, or the pattern names somewhere outside it.
        """
        directory = listing_directory(folder)
        if not directory or not os.path.isdir(directory):
            return float("NaN")
        try:
            found = listed(directory, pattern, recursive, include, sort, reverse, limit)
        except sandbox.PathNotAllowed:
            raise
        except ValueError:
            return float("NaN")
        written = "\n".join(f"{row.name}|{row.mtime}|{row.size}" for row in found)
        return hashlib.sha256(written.encode("utf-8")).hexdigest()

    @classmethod
    def execute(
        cls,
        folder="",
        pattern="*",
        recursive=False,
        include="files",
        sort="natural",
        reverse=False,
        limit=0,
    ) -> io.NodeOutput:
        """List the folder and answer the paths, the names, the lines and the count.

        Args:
            folder: A label from the folder menu, such as ``input`` or
                ``plates/shot_01 [input]``.
            pattern: Glob matched inside the folder.
            recursive: Whether the pattern is applied at every depth.
            include: One of :data:`INCLUDE`.
            sort: One of :data:`SORTS`.
            reverse: Whether the order is flipped.
            limit: How many entries to keep, or 0 for every one.

        Returns:
            The full paths, the names below the folder, the names one per line, and how
            many there are.

        Raises:
            PathNotAllowed: The label names another machine, the folder resolved outside
                every permitted read root, or the pattern names somewhere outside it.
            ValueError: The folder is not there, or the pattern is empty.
        """
        directory = listing_directory(folder)
        if not directory or not os.path.isdir(directory):
            raise ValueError(
                f"`{folder}` names no folder Directory Listing can read. Pick another "
                f"from the menu, or add its folder to paths.allow_read in config.yaml"
            )

        found = listed(directory, pattern, recursive, include, sort, reverse, limit)
        if not found:
            logger.info(
                "Directory Listing found no %s in %s matching `%s`", include, directory, pattern
            )
        else:
            logger.debug(
                "Directory Listing found %d entries in %s matching `%s`",
                len(found), directory, pattern,
            )

        names = [row.name for row in found]
        # A list of its own per slot, so editing one does not change the other.
        return io.NodeOutput(
            [row.path for row in found], names, "\n".join(names), len(found)
        )
