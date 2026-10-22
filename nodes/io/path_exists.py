"""Ask whether a path is on disk, and report what the filesystem says about it."""

from __future__ import annotations

import os
import stat
from datetime import datetime

from comfy_api.latest import io

from ...modules import log
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: How the readable modification time is written, in local time.
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

#: The answer for a path nothing was found at, without the seventh element.
NOTHING = (False, False, False, 0, 0.0, "")


def probe(path: str, report: bool = False) -> tuple[bool, bool, bool, int, float, str, str]:
    """Stat a path through the containment layer.

    Args:
        path: The raw widget value.
        report: Log the refusal for a path containment refuses.

    Returns:
        Whether anything is there, whether it is a file, whether it is a directory, its
        size in bytes, its modification time as epoch seconds and as a local
        ``YYYY-MM-DD HH:MM:SS`` string, and the absolute path that was probed. An empty or
        refused path answers :data:`NOTHING` with an empty path, whatever is on disk.
    """
    if not str(path or "").strip():
        return NOTHING + ("",)
    try:
        resolved = sandbox.resolve_read(path)
    except sandbox.PathNotAllowed as refusal:
        if report:
            logger.warning("Path Exists answers not existing. %s", refusal)
        return NOTHING + ("",)

    try:
        status = os.stat(resolved)
    except (FileNotFoundError, NotADirectoryError):
        return NOTHING + (str(resolved),)
    except OSError as error:
        logger.warning(
            "%s could not be looked at (%s), so it is answered as not existing. Check the "
            "permissions on it and on the folders leading to it.",
            resolved, error,
        )
        return NOTHING + (str(resolved),)

    is_directory = stat.S_ISDIR(status.st_mode)
    is_file = stat.S_ISREG(status.st_mode)
    size = 0 if is_directory else int(status.st_size)
    epoch = float(status.st_mtime)
    try:
        moment = datetime.fromtimestamp(epoch).strftime(STAMP_FORMAT)
    except (OSError, OverflowError, ValueError):
        moment = ""
    return True, is_file, is_directory, size, epoch, moment, str(resolved)


class PathExists(io.ComfyNode):
    """Report whether a path is on disk, and what is there."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPathExists",
            display_name="Path Exists",
            search_aliases=[
                "WASPathExists",
                "Path Exists",
                "file exists",
                "folder exists",
                "file size",
                "modified time",
                "resume",
                "skip existing",
            ],
            category="WAS Suite/IO",
            description=(
                "Check whether a file or folder is on disk before anything reads or writes "
                "it, with its kind, size in bytes and last change time. Use it to skip frames "
                "already rendered or to test a cache against its source. A path outside the "
                "folders the pack may read answers not existing, with 'resolved' empty."
            ),
            inputs=[
                io.String.Input(
                    "path",
                    default="",
                    multiline=False,
                    tooltip=(
                        "The file or folder to look for, such as "
                        "'ComfyUI/output/renders/frame_0001.png'. A full path must lie inside "
                        "input, output, temp or a folder paths.allow_read adds. "
                        "Nothing is opened, read or written; the path is only asked about."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="exists",
                    tooltip=(
                        "true when something is at that path. A folder counts, and so does "
                        "a file of 0 bytes. false when nothing is there, and false for a "
                        "path outside the folders this pack may read."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_file",
                    tooltip=(
                        "true only for an ordinary file. false for a folder and false when "
                        "nothing is there, so pair it with exists to tell 'a folder is at "
                        "that path' from 'nothing is'."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_directory",
                    tooltip=(
                        "true only for a folder. false for a file and false when nothing is "
                        "there. Wire it to a switch that picks between loading a whole "
                        "folder and loading one file."
                    ),
                ),
                io.Int.Output(
                    display_name="size_bytes",
                    tooltip=(
                        "Bytes the file holds. 0 for a folder, 0 when nothing is there, and "
                        "0 for an empty file, so 'size_bytes > 0' is the test for a render "
                        "that finished writing rather than one cut off at the start."
                    ),
                ),
                io.Float.Output(
                    display_name="modified_epoch",
                    tooltip=(
                        "When it last changed, in seconds since 1970, as 1755864000.0. 0.0 "
                        "when nothing is there. Feed two of these to Compare to tell which "
                        "of a source and a cache is the newer."
                    ),
                ),
                io.String.Output(
                    display_name="modified",
                    tooltip=(
                        "The same moment in local time, as '2026-08-22 11:57:03'. Empty when "
                        "nothing is there. For reading and for naming a file; compare "
                        "modified_epoch instead of this."
                    ),
                ),
                io.String.Output(
                    display_name="resolved",
                    tooltip=(
                        "The absolute path that was probed. Empty for an empty path and for "
                        "a path outside the folders this pack may read, which is not probed; "
                        "the log then names the permitted folders and the config key that "
                        "would allow it."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, path="") -> str:
        """Read again when what is at the path has changed."""
        exists, _is_file, _is_directory, size, epoch, _moment, resolved = probe(path)
        return f"{exists}|{size}|{epoch}|{resolved}"

    @classmethod
    def execute(cls, path="") -> io.NodeOutput:
        """Report what is at the path.

        Args:
            path: The file or folder to look for.

        Returns:
            Existence, whether it is a file, whether it is a directory, its size, its
            modification time as epoch seconds and as text, and the path that was probed.
        """
        return io.NodeOutput(*probe(path, report=True))
