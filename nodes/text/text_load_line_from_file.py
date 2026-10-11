"""Read one line at a time out of a text file or a multiline input."""

from __future__ import annotations

import hashlib

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DICT
from ...modules.io import picker
from ...modules.state.database import get_settings_db
from ...modules.state.history import update_history_text_files
from ...modules.util import sandbox, text_files

logger = log.get_logger("text.load_line_from_file")

#: What the line menu lists, and what it says when there is nothing to list.
NO_FILES = "no text files found"


def line_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(text_files.TEXT_EXTENSIONS) or [NO_FILES]


def line_path(file: str) -> str:
    """The file one menu entry names, as a path, or an empty string."""
    entry = str(file or "").strip()
    if not entry or entry == NO_FILES:
        return ""
    return picker.resolve(entry, text_files.TEXT_EXTENSIONS) or ""



#: Database categories the read cursor and the file it belongs to are kept in, keyed on
#: the node's label so several batches can advance independently.
COUNTERS = "TextBatch Counters"
PATHS = "TextBatch Paths"


class TextFileLoader:
    """A text file plus the cursor of how far through it the label has read.

    Attributes:
        WDB: The database the cursor is read from and written back to.
        file_path: The file the lines were read from.
        lines: Every line of the file, stripped of surrounding whitespace.
        index: The line the next :meth:`get_next_line` returns.
        label: The batch name the cursor is stored under.
    """

    def __init__(self, file_path, label):
        self.WDB = get_settings_db()
        self.file_path = file_path
        self.lines: list[str] = []
        self.index = 0
        self.label = label
        self.load_file(file_path)

    def load_file(self, file_path) -> None:
        """Read the file and resume the label's cursor, resetting it on a new path.

        Raises:
            OSError: The file could not be read.
        """
        stored_file_path = self.WDB.get(PATHS, self.label)
        stored_index = self.WDB.get(COUNTERS, self.label)
        if stored_file_path != file_path:
            self.index = 0
            self.WDB.insert(COUNTERS, self.label, 0)
            self.WDB.insert(PATHS, self.label, file_path)
        else:
            self.index = stored_index if isinstance(stored_index, int) else 0
        with open(file_path, "r", encoding="utf-8", newline="\n") as file:
            self.lines = [line.strip() for line in file]

    def get_next_line(self) -> tuple[str, list[str]]:
        """The line at the cursor, advancing it and wrapping at the end of the file."""
        if self.index >= len(self.lines):
            self.index = 0
        line = self.lines[self.index]
        self.index += 1
        if self.index == len(self.lines):
            self.index = 0
        logger.info("TextBatch index: %s", self.index)
        return line, self.lines

    def get_line_by_index(self, index) -> tuple[str | None, list[str]]:
        """The line at ``index``, moving the cursor there.

        Returns:
            ``(line, every line)``, or ``(None, [])`` when the index is out of range.
        """
        if index < 0 or index >= len(self.lines):
            logger.error("Invalid line index `%s`", index)
            return None, []
        self.index = index
        logger.info("TextBatch index: %s", self.index)
        return self.lines[self.index], self.lines

    def store_index(self) -> None:
        """Write the cursor back to the database."""
        self.WDB.insert(COUNTERS, self.label, self.index)


class TextLoadLineFromFile(io.ComfyNode):
    """Emit one line of a text file, either the next in sequence or one by index."""

    # Text in the multiline_text box replaces the file entirely and is read the same two ways,
    # so a short list lives in the workflow rather than in a file beside it.

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Load Line From File",
            display_name="Text Load Line From File",
            search_aliases=["Text Load Line From File", "text batch", "prompt list"],
            category="WAS Suite/Text",
            description=(
                "Read one line of a text file per prompt, or one line by index, to walk a "
                "list of prompts one per generation. The second output holds every line, "
                "keyed by the dictionary name. With no file and nothing connected the node "
                "logs an error and emits an empty line."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=line_options(),
                    tooltip=(
                        "Which file to read, one prompt or phrase per line. The menu lists "
                        "every text file in ComfyUI's input, output and temp folders and in "
                        "any folder added under paths.allow_read. Ignored when multiline_text "
                        "is connected."
                    ),
                ),
                io.String.Input(
                    "dictionary_name",
                    default="[filename]",
                    multiline=False,
                    tooltip=(
                        "The key the list of every line is filed under in the dictionary "
                        "output, so Text Dictionary Get can fetch it again by name. Used "
                        "exactly as typed."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="TextBatch",
                    multiline=False,
                    tooltip=(
                        "Name this batch's read position is remembered under, surviving a "
                        "restart. Nodes sharing a label share one position and take turns; "
                        "different labels advance independently. Pointing a label at a "
                        "different file starts it over at the first line."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["automatic", "index"],
                    tooltip=(
                        "`automatic` hands out the next line on every run and wraps around "
                        "at the end of the file, which is what walks a list of prompts one "
                        "per generation. `index` returns the one line asked for and does not "
                        "advance."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip=(
                        "Which line to return in `index` mode, counting from 0 for the first "
                        "line. Ignored in `automatic` mode. An index past the end of a file "
                        "wraps around, so 12 in a 10-line file is line 2."
                    ),
                ),
                io.String.Input(
                    "multiline_text",
                    multiline=True,
                    optional=True,
                    placeholder="one entry per line",
                    tooltip=(
                        "Lines to read instead of file_path; STRING, one entry per "
                        "line. Anything here overrides file_path."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="line_text",
                    tooltip=(
                        "The one line selected, stripped of surrounding whitespace. Empty "
                        "when the file is missing, empty, or the index was out of range."
                    ),
                ),
                DICT.Output(
                    display_name="dictionary",
                    tooltip=(
                        "Every line of the source, as a list filed under dictionary_name, "
                        "the whole list alongside the single line, for a node that needs all "
                        "of it."
                    ),
                ),
            ],
        )

    # ComfyUI calls this with the whole input dictionary, so every input is named here; a
    # signature missing one of them raises rather than fingerprinting.
    @classmethod
    def fingerprint_inputs(
        cls,
        file="",
        dictionary_name="[filename]",
        label="TextBatch",
        mode="automatic",
        index=0,
        multiline_text=None,
    ):
        """Re-run on every prompt in automatic mode, and on a file change in index mode.

        Raises:
            PathNotAllowed: The chosen file resolves outside every permitted read root.
        """
        if mode != "index":
            return float("NaN")
        if multiline_text:
            return multiline_text
        found = line_path(file)
        if not found:
            return False
        path = sandbox.resolve_read(found)
        if not path.is_file():
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            digest.update(file.read())
        return digest.hexdigest()

    @classmethod
    def execute(
        cls,
        file="",
        dictionary_name="[filename]",
        label="TextBatch",
        mode="automatic",
        index=0,
        multiline_text=None,
    ) -> io.NodeOutput:
        # A widget always sends its value, so an empty box is what "no text here" looks
        # like and is the same as an unconnected socket: fall through to the file.
        if multiline_text:
            return cls.from_text(multiline_text, dictionary_name, label, mode, index)

        file_path = line_path(file)
        if not file_path:
            logger.error("no file was chosen, and nothing is connected to multiline_text.")
            return io.NodeOutput("", {dictionary_name: []})

        path = sandbox.resolve_read(file_path)
        if not path.exists():
            logger.error("The path `%s` specified cannot be found.", path)
            return io.NodeOutput("", {dictionary_name: []})

        file_list = TextFileLoader(str(path), label)
        # An empty file leaves nothing to index into: automatic mode reads lines[0] and
        # index mode takes a modulo by zero, so both raise before reaching the
        # `line is None` guard below.
        if not file_list.lines:
            logger.error("No valid line was found. The file may be empty or all lines have been read.")
            return io.NodeOutput("", {dictionary_name: []})

        line, lines = None, []
        if mode == "automatic":
            line, lines = file_list.get_next_line()
        elif mode == "index":
            if index >= len(file_list.lines):
                index = index % len(file_list.lines)
            line, lines = file_list.get_line_by_index(index)
        if line is None:
            logger.error("No valid line was found. The file may be empty or all lines have been read.")
            return io.NodeOutput("", {dictionary_name: []})
        file_list.store_index()
        update_history_text_files(str(path))

        return io.NodeOutput(line, {dictionary_name: lines})

    @classmethod
    def from_text(cls, multiline_text, dictionary_name, label, mode, index) -> io.NodeOutput:
        """The same two modes over a connected string instead of a file."""
        lines = multiline_text.strip().split("\n")
        if mode == "index":
            if index < 0 or index >= len(lines):
                logger.error("Invalid line index `%s`", index)
                return io.NodeOutput("", {dictionary_name: []})
            return io.NodeOutput(lines[index], {dictionary_name: lines})

        database = get_settings_db()
        line_index = database.get(COUNTERS, label)
        if not isinstance(line_index, int):
            line_index = 0
        line = lines[line_index % len(lines)]
        database.insert(COUNTERS, label, line_index + 1)
        return io.NodeOutput(line, {dictionary_name: lines})
