"""Read a style library out of a JSON or AUTOMATIC1111 CSV file."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST
from ....modules.interface import library_report, run_result
from ....modules.io import picker
from ....modules.prompt import styles
from ....modules.util import sandbox

logger = log.get_logger("text.styles")

#: The two kinds of file a style library is read from.
STYLE_EXTENSIONS = (".json", ".csv")

#: What the menu says when no style file is anywhere the pack may read.
NO_FILES = "no .json or .csv files found"

#: What each mode does, in the order the widget offers them.
ADD = "add to the library"
REPLACE = "replace the library"

MODES = (ADD, REPLACE)


def style_file_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(STYLE_EXTENSIONS) or [NO_FILES]


class PromptStylesImport(io.ComfyNode):
    """Store the styles held in a JSON library or an A1111 CSV."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPromptStylesImport",
            display_name="Prompt Styles Import",
            search_aliases=[
                "WASPromptStylesImport",
                "Prompt Styles Import",
                "style",
                "a1111 styles",
                "load styles",
                "styles.csv",
            ],
            category="WAS Suite/Text/Styles",
            description=(
                "Read a style library into this one, from a .json library or from an "
                "AUTOMATIC1111 styles.csv with name, prompt and negative_prompt columns. "
                "The styles land in Prompt Styles Selector's menu. Importing the same file "
                "again brings it up to date, dropping the styles it no longer names and "
                "keeping every style saved here."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file",
                    options=style_file_options(),
                    tooltip=(
                        "Which style file to read. The menu lists every .json and .csv "
                        "file in ComfyUI's input, output and temp folders and in any "
                        "folder added under paths.allow_read in config.yaml, each labelled "
                        "with the folder it sits in."
                    ),
                )),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    tooltip=(
                        "`add to the library` keeps the styles already saved here and adds "
                        "the file's. `replace the library` leaves the library holding "
                        "exactly what the file holds, dropping every other style."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "Every style name in the library after the import, in library "
                        "order. Text List to Text turns it into one line per name."
                    ),
                ),
                io.Int.Output(
                    display_name="imported",
                    tooltip="How many styles the file held.",
                ),
                io.Int.Output(
                    display_name="total",
                    tooltip="How many styles the library holds after the import.",
                ),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, file="", mode=ADD) -> io.NodeOutput:
        """Read the file and store the styles it holds.

        Raises:
            ValueError: No file was chosen, or no style could be read from it.
            PathNotAllowed: The file resolves outside every permitted read root.
        """
        label = str(file or "").strip()
        if not label or label == NO_FILES:
            raise ValueError(
                "no style file was chosen. Put a styles .json or an A1111 styles.csv in "
                "ComfyUI's input folder, or in a folder listed under paths.allow_read in "
                "config.yaml, then pick it from the file menu"
            )
        found = picker.resolve(label, STYLE_EXTENSIONS)
        if not found:
            raise ValueError(
                f"`{label}` is not there any more, so nothing was read. Pick a file from "
                f"the menu again"
            )
        path = sandbox.resolve_read(found)

        before = styles.library()
        imported = styles.import_styles(path, replace=str(mode) == REPLACE)
        if not imported:
            raise ValueError(
                f"no style could be read from `{path}`. A .json library is a list of names, "
                f"each holding a prompt and a negative_prompt; an A1111 .csv needs name, "
                f"prompt and negative_prompt columns"
            )
        after = styles.library()

        added = [name for name in after if name not in before]
        dropped = [name for name in before if name not in after]
        logger.info(
            "imported %s style(s) from %s, %s new, %s dropped",
            imported,
            path.name,
            len(added),
            len(dropped),
        )
        cls.report(path, mode, imported, len(added), len(dropped), after)
        return io.NodeOutput(list(after), imported, len(after))

    @classmethod
    def report(cls, path, mode, imported, added, dropped, after) -> None:
        """Draw what the file held and what the library holds now on the node."""
        library_report.publish(
            summary=(
                f"{imported} style(s) read from {path.name}, {added} new, "
                f"{len(after)} in the library now"
            ),
            counts={
                "read": imported,
                "new": added,
                "dropped": dropped,
                "in library": len(after),
            },
            facts={"file": str(path), "mode": str(mode)},
            lines=list(after),
            listing="styles",
            total=len(after),
            status=run_result.OK if added or dropped else run_result.WARNING,
        )
