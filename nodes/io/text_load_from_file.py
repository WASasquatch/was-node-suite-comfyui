"""Read a text file into a string and a dictionary of its lines."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.io import picker
from ...modules.util import file_listing, text_files
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


def missing(entry: str) -> str:
    """What to log when the chosen entry names no listed file.

    Args:
        entry: The stored combo value, stripped.

    Returns:
        A message naming the entry and every folder the menu is built from.
    """
    folders = ", ".join(f"{tag} ({path})" for tag, path in file_listing.roots(picker.ROOTS))
    where = folders or "ComfyUI's input, output and temp folders, which could not be found"
    elsewhere = (
        "To list another folder, add it under paths.allow_read in config.yaml, as "
        "'D:/prompts', then restart ComfyUI and reload the page."
    )
    if not entry or entry == NO_FILES:
        return (
            f"Load Text File has no file chosen, so it read nothing. Pick one from its menu, "
            f"which lists the text files in {where}. {elsewhere}"
        )
    return (
        f"Load Text File found no text file named `{entry}`, so it read nothing. It may have "
        f"been deleted or renamed since the menu was built. Pick it again from the menu, "
        f"which lists the text files in {where}. {elsewhere}"
    )


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
                "Read a text file picked from a menu, dropping comment lines, as text and as "
                "a dictionary. The menu lists the text files, subfolders included, in "
                "ComfyUI's input, output and temp folders and in every folder under "
                "paths.allow_read in config.yaml, each tagged with its folder's name. A "
                "folder added there appears after a ComfyUI restart and a page reload. A "
                "file that cannot be read gives empty text and a line in the log."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=text_options(),
                    tooltip=(
                        "The UTF-8 text file to read: 'notes.txt' in input, 'notes.txt "
                        "[output]', 'notes.txt [temp]', or 'notes.txt [prompts]' for a "
                        "prompts folder under paths.allow_read."
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

        if not file_path.strip():
            logger.error("%s", missing(str(file or "").strip()))
            return io.NodeOutput("", {name: []})

        resolved = sandbox.resolve_read(file_path)
        if not resolved.is_file():
            logger.error("the path `%s` specified cannot be found.", resolved)
            return io.NodeOutput("", {name: []})

        try:
            text = text_files.read_text(resolved)
        except UnicodeDecodeError:
            logger.error(
                "`%s` is not UTF-8, so Load Text File read nothing from it. Save the file as "
                "UTF-8 and run the prompt again.",
                resolved,
            )
            return io.NodeOutput("", {name: []})
        except OSError as error:
            logger.error("`%s` could not be read (%s).", resolved, error)
            return io.NodeOutput("", {name: []})

        history.update_history_text_files(str(resolved))

        lines = [line for line in text_files.split_lines(text) if not text_files.is_comment(line)]
        return io.NodeOutput("\n".join(lines), {name: lines})
