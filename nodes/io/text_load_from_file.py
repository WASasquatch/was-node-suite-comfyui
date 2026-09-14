"""Read a text file into a string and a dictionary of its lines."""

from __future__ import annotations

import os
from io import StringIO

from comfy_api.latest import io

from ...modules.io import picker
from ...modules.util import text_files
from ...modules import log
from ...modules.compat.types import DICT
from ...modules.state import history
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: What the text menu lists, and what it says when there is nothing to list.
NO_FILES = "no text files found"


def text_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(text_files.TEXT_EXTENSIONS) or [NO_FILES]


def text_path(file: str) -> str:
    """The file one menu entry names, as a path, or an empty string."""
    entry = str(file or "").strip()
    if not entry or entry == NO_FILES:
        return ""
    return picker.resolve(entry, text_files.TEXT_EXTENSIONS) or ""



#: Widget value that keeps the dictionary keyed on the file's own name.
FILENAME_KEYWORD = "[filename]"


class LoadTextFile(io.ComfyNode):
    """Read a UTF-8 text file, dropping comment lines."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Load Text File",
            display_name="Load Text File",
            search_aliases=["Load Text File", "read text", "text file"],
            category="WAS Suite/IO",
            description=(
                "Read a text file, dropping comment lines, as text and as a dictionary. "
                "Nowhere but the given path is searched, so a bare file name only works if "
                "it sits in the folder ComfyUI was started in, and the path has to land "
                "inside ComfyUI's input, output or temp folder, the pack's own folder, or a "
                "folder listed under paths.allow_read in config.yaml. A file that cannot be "
                "read gives empty text rather than failing the prompt."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=text_options(),
                    tooltip=(
                        "Which file to read. The menu lists every text file in ComfyUI's "
                        "input, output and temp folders and in any folder added under "
                        "paths.allow_read. It has to be UTF-8."
                    ),
                ),
                io.String.Input(
                    "dictionary_name",
                    default="[filename]",
                    multiline=False,
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
    def execute(cls, file="", dictionary_name="[filename]") -> io.NodeOutput:
        """Read the file and split it into lines.

        Raises:
            PathNotAllowed: the chosen file resolved outside every permitted read root.
        """
        file_path = text_path(file)
        base = os.path.basename(file_path)
        name = base.split(".", 1)[0] if "." in base else base
        if dictionary_name != FILENAME_KEYWORD:
            name = dictionary_name

        # An empty widget names nothing to contain, and reports the same missing file as a
        # path that does not exist.
        if not file_path.strip():
            logger.error("the path `%s` specified cannot be found.", file_path)
            return io.NodeOutput("", {name: []})

        resolved = sandbox.resolve_read(file_path)
        if not resolved.is_file():
            logger.error("the path `%s` specified cannot be found.", resolved)
            return io.NodeOutput("", {name: []})

        with open(resolved, "r", encoding="utf-8", newline="\n") as handle:
            text = handle.read()

        history.update_history_text_files(str(resolved))

        lines = [
            line.replace("\n", "").replace("\r", "")
            for line in StringIO(text)
            if not line.strip().startswith("#")
        ]
        return io.NodeOutput("\n".join(lines), {name: lines})
