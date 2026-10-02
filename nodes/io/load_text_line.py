"""Read one line, or every line, of a text file picked from a menu."""

from __future__ import annotations

import random

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LIST
from ...modules.state import history
from ...modules.util import sandbox, text_files

logger = log.get_logger("nodes.io")

#: What ``resolved_index`` carries when no single line was selected: every mode that reads
#: the whole file, and every file that has no line to select.
NO_LINE = -1


class LoadTextLine(io.ComfyNode):
    """Read a text file chosen from a menu, by line."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadTextLine",
            display_name="Load Text Line",
            search_aliases=[
                "WASLoadTextLine", "Load Text Line",
                "text file menu",
                "pick line",
                "line by index",
                "random line from file",
                "prompt list",
            ],
            category="WAS Suite/IO",
            description=(
                "Read a text file picked from a menu: the whole file, the line at an index, "
                "or a line drawn from a seed, with every line also given as a list. The menu "
                "lists the text files in ComfyUI's input and output folders, subfolders "
                "included, and in every folder under paths.allow_read in config.yaml, each "
                "tagged with its folder's name. A folder added there appears after a ComfyUI "
                "restart and a page reload. A file since deleted or renamed gives empty text "
                "and a line in the log."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=text_files.options(),
                    tooltip=(
                        "The .txt, .csv, .tsv, .json, .jsonl, .md, .yaml or .yml file to "
                        "read, tagged with its folder: 'notes.txt [input]', 'notes.txt "
                        "[output]', or 'notes.txt [prompts]' for a prompts folder under "
                        "paths.allow_read."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["file", "index", "random"],
                    tooltip=(
                        "What the 'line' output carries. 'file' gives the whole file, every "
                        "line joined back together, which is what feeds a prompt written "
                        "across several lines. 'index' gives the single line at 'index', for "
                        "stepping through a list with a counter. 'random' gives one line "
                        "drawn by 'seed'. The 'lines' and 'text' outputs are the same in all "
                        "three."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-99999999,
                    max=99999999,
                    step=1,
                    tooltip=(
                        "Which line 'index' mode takes, counting from 0, so 0 is the first "
                        "line. -1 is the last line, -2 the one before it. Read only in "
                        "'index' mode. What happens past either end is 'out_of_range'."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=["wrap", "clamp", "empty", "error"],
                    tooltip=(
                        "What an index past either end does: 'wrap' starts from the other "
                        "end, 'clamp' sticks at the first or last line, 'empty' gives "
                        "nothing, 'error' stops the prompt."
                    ),
                ),
                io.Boolean.Input(
                    "skip_comment_lines",
                    default=True,
                    tooltip=(
                        "Whether lines whose first non-space character is '#' are dropped "
                        "before anything is counted or numbered. On by default, so the "
                        "indexes skip notes."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Which line 'random' mode draws. The same seed and the same file "
                        "always give the same line; change it to draw a different one. The "
                        "seed pins a position rather than a line, so adding or removing a "
                        "line in the file, or changing skip_comment_lines, makes the same "
                        "seed produce different text. Read only in 'random' mode."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="line",
                    tooltip=(
                        "What the chosen mode selected: the whole file on 'file', one line on "
                        "'index' and 'random'. Empty when the file could not be read or holds "
                        "no line to select."
                    ),
                ),
                LIST.Output(
                    display_name="lines",
                    tooltip=(
                        "Every line of the file on one wire, for Text List Slice, Text List "
                        "Get, Text List Length and Text List to Strings. Comment lines are "
                        "absent while skip_comment_lines is on."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The whole file as one string, the same lines joined with line "
                        "breaks, whatever the mode. Wire this into Text Random Line or Text "
                        "Find and Replace to work on the file as a whole."
                    ),
                ),
                io.Int.Output(
                    display_name="line_count",
                    tooltip=(
                        "How many lines the file has, after comment lines are dropped. 0 for "
                        "an empty file and for one that could not be read."
                    ),
                ),
                io.Int.Output(
                    display_name="resolved_index",
                    tooltip=(
                        "Which line number the 'line' output really came from, after wrapping "
                        "or clamping, counting from 0. -1 in 'file' mode and whenever no "
                        "single line was selected. Worth watching when a counter drives the "
                        "index or a seed drives the draw."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, file) -> bool:
        """Accept any ``file``, including one the menu no longer offers.

        Args:
            file: The stored combo value. Not inspected here; ``execute`` reports it.

        Returns:
            True, always.
        """
        # ComfyUI checks a combo value against the option list at queue time and skips that
        # check for an input named in this signature. Without it, a workflow naming a file
        # that has been deleted, renamed or pushed out of the menu by the option cap fails
        # before execute runs, with a message naming neither the file nor where it was
        # looked for. Naming only `file` leaves every other input checked as usual.
        return True

    @classmethod
    def fingerprint_inputs(
        cls, file, mode, index, out_of_range, skip_comment_lines, seed
    ) -> float:
        """Always stale: a file in the menu can be rewritten in place between runs."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        file="",
        mode="file",
        index=0,
        out_of_range="wrap",
        skip_comment_lines=True,
        seed=0,
    ) -> io.NodeOutput:
        """Read the selected file and pick from it.

        Raises:
            PathNotAllowed: The listed path lies outside every permitted read root.
            IndexError: ``mode`` is ``index``, the index is past the end and
                ``out_of_range`` is ``error``.
        """
        entry = (file or "").strip()
        path = None if entry in ("", text_files.NO_FILES) else text_files.resolve(entry)
        if path is None:
            logger.error("%s", cls.missing(entry))
            return io.NodeOutput("", [], "", 0, NO_LINE)

        # Every listed folder is a permitted read root; the widget value passes the gate too.
        resolved = sandbox.resolve_read(path)
        if not resolved.is_file():
            logger.error("the path `%s` specified cannot be found.", resolved)
            return io.NodeOutput("", [], "", 0, NO_LINE)

        try:
            text = text_files.read_text(resolved)
        except UnicodeDecodeError:
            logger.error(
                "`%s` is not UTF-8, so Load Text Line read nothing from it. Save the file as "
                "UTF-8 and run the prompt again.",
                resolved,
            )
            return io.NodeOutput("", [], "", 0, NO_LINE)
        except OSError as error:
            logger.error("`%s` could not be read (%s).", resolved, error)
            return io.NodeOutput("", [], "", 0, NO_LINE)

        history.update_history_text_files(str(resolved))

        lines = text_files.split_lines(text)
        if skip_comment_lines:
            lines = [line for line in lines if not text_files.is_comment(line)]
        joined = "\n".join(lines)

        position = cls.select(lines, mode, index, out_of_range, seed)
        selected = joined if mode == "file" else (lines[position] if position >= 0 else "")
        return io.NodeOutput(selected, lines, joined, len(lines), position)

    @staticmethod
    def missing(entry: str) -> str:
        """What to log when the chosen entry names no listed file.

        Args:
            entry: The stored combo value, stripped.

        Returns:
            A message naming the entry and every folder the menu is built from.
        """
        folders = ", ".join(f"{tag} ({path})" for tag, path in text_files.roots())
        where = folders or "ComfyUI's input and output folders, which could not be found"
        elsewhere = (
            "To list another folder, add it under paths.allow_read in config.yaml, as "
            "'D:/prompts', then restart ComfyUI and reload the page."
        )
        if not entry or entry == text_files.NO_FILES:
            return (
                f"Load Text Line has no file chosen, so it read nothing. Pick one from its "
                f"menu, which lists the text files in {where}. {elsewhere}"
            )
        return (
            f"Load Text Line found no text file named `{entry}`, so it read nothing. It may "
            f"have been deleted, renamed, or moved between folders since the workflow was "
            f"saved. Pick it again from the menu, which lists the text files in {where}. "
            f"{elsewhere}"
        )

    @staticmethod
    def select(lines: list[str], mode: str, index: int, out_of_range: str, seed: int) -> int:
        """Which line the mode picks.

        Args:
            lines: The file's lines, comment lines already dropped where that is on.
            mode: ``file``, ``index`` or ``random``.
            index: The requested position for ``index`` mode. Negative counts back from the
                end.
            out_of_range: ``wrap``, ``clamp``, ``empty`` or ``error``, read by ``index``
                mode alone.
            seed: Seeds the draw for ``random`` mode.

        Returns:
            A position in ``lines``, or :data:`NO_LINE` where no single line is selected:
            ``file`` mode, an empty file, and an index outside an ``empty`` range.

        Raises:
            IndexError: The index is outside the file and ``out_of_range`` is ``error``.
        """
        length = len(lines)
        if mode == "file" or length == 0:
            return NO_LINE
        if mode == "random":
            return random.Random(seed).randrange(length)

        position = index + length if index < 0 else index
        if 0 <= position < length:
            return position
        if out_of_range == "wrap":
            return position % length
        if out_of_range == "clamp":
            return 0 if position < 0 else length - 1
        if out_of_range == "error":
            raise IndexError(
                f"Load Text Line was asked for line {index} of a file holding {length} "
                f"line(s). Set out_of_range to 'wrap', 'clamp' or 'empty' to allow it."
            )
        return NO_LINE
