"""Reload a text file from the pack's recently used text file history."""

from __future__ import annotations

import os
import time
from io import StringIO

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import DICT
from ....modules.state import history
from ....modules.util import sandbox

logger = log.get_logger("nodes.io.history")

#: History key holding the paths every text-file node appends to.
HISTORY_KEY = "TextFiles"

#: Combo entry shown when the history holds nothing.
EMPTY = "No History"

#: Widget value that keeps the dictionary keyed on the file's own name.
FILENAME_KEYWORD = "[filename]"

#: Seconds a combo option list is reused for before the history is read again.
OPTIONS_TTL = 1.0

_options_cache: tuple[float, list[str]] = (0.0, [])


def label(path: str) -> str:
    """The menu entry for a history path: ``...<sep><parent dir><sep><file name>``."""
    parent = os.path.basename(os.path.dirname(path))
    return os.path.join("..." + os.sep + parent, os.path.basename(path))


def labelled_history(limit: int | None = None) -> dict[str, str]:
    """``{menu entry: absolute path}`` for text files in the history, oldest first.

    Args:
        limit: How many of the newest to include, or None for every one recorded.

    Returns:
        One entry per label. Two paths sharing a label leave the newer one.
    """
    database = history.open_history_db()
    if limit is None:
        paths = database.get("History", HISTORY_KEY)
    else:
        paths = database.newest(HISTORY_KEY, limit)
        paths.reverse()
    return {label(path): path for path in paths}


def options() -> list[str]:
    """The combo's entries: the newest ``history.display_limit()`` text files.

    Memoized for :data:`OPTIONS_TTL` seconds.
    """
    global _options_cache
    stamp, cached = _options_cache
    now = time.monotonic()
    if cached and now - stamp < OPTIONS_TTL:
        return cached
    limit = history.display_limit()
    entries = [label(path) for path in history.recent(HISTORY_KEY, limit)]
    entries = entries or [EMPTY]
    _options_cache = (now, entries)
    return entries


class TextFileHistoryLoader(io.ComfyNode):
    """Read one of the text files this pack has loaded or written before, comments dropped."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text File History Loader",
            display_name="Text File History Loader",
            search_aliases=["Text File History Loader", "recent text files", "history"],
            category="WAS Suite/History",
            description=(
                "Reload one of the text files the suite has recently read or written. The "
                "menu holds whatever this pack's text loading and saving nodes have "
                "touched, up to the limit in the pack's config. A file that has since been "
                "deleted gives empty text rather than failing the prompt, and one in a "
                "folder this pack may no longer read stops the prompt with that folder "
                "named."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=options(),
                    tooltip=(
                        "Which recently used text file to reread. Entries are listed newest "
                        "last as '.../<folder>/<file>', and read 'No History' until a load "
                        "or save node has run."
                    ),
                ),
                io.String.Input(
                    "dictionary_name",
                    default="[filename]",
                    multiline=True,
                    tooltip=(
                        "The key the lines are stored under in the dictionary output. Left as "
                        "'[filename]' it is the part of the file's name before the first dot, "
                        "so 'animals.txt' becomes 'animals'; anything else is used as the key "
                        "verbatim."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The whole file as one string, with comment lines, those starting "
                        "with '#', removed and the rest kept in order."
                    ),
                ),
                DICT.Output(
                    tooltip=(
                        "The same lines as a list under a single key, so a node that picks a "
                        "line by index or at random can work through them."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file, dictionary_name) -> float:
        """Always stale: a file listed in the history can be rewritten in place."""
        return float("NaN")

    @classmethod
    def execute(cls, file=None, dictionary_name="[filename]") -> io.NodeOutput:
        """Reread the selected file, or report that it is no longer there.

        Raises:
            PathNotAllowed: The recorded path lies outside every permitted read root.
        """
        entry = (file or "").strip()
        # The newest entries first, the whole history only when the selection is older.
        recent = labelled_history(history.display_limit())
        file_path = recent.get(entry) or labelled_history().get(entry, entry)
        base = os.path.basename(file_path)
        name = base.split(".", 1)[0] if "." in base else base
        if dictionary_name != FILENAME_KEYWORD:
            name = dictionary_name

        # A selection the history does not list maps to itself.
        if file_path == entry:
            logger.error("the path `%s` specified cannot be found.", file_path)
            return io.NodeOutput("", {name: []})

        # A recorded path is checked against the read roots permitted now.
        resolved = sandbox.resolve_read(file_path)
        if not resolved.is_file():
            logger.error("the path `%s` specified cannot be found.", resolved)
            return io.NodeOutput("", {name: []})

        with open(resolved, "r", encoding="utf-8", newline="\n") as handle:
            text = handle.read()

        history.update_history_text_files(str(resolved))

        lines = [
            line.replace("\n", "")
            for line in StringIO(text)
            if not line.strip().startswith("#")
        ]
        return io.NodeOutput("\n".join(lines), {name: lines})
