"""Read a Noodle Soup Prompts pantry file into the stored pantry."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.io import picker
from ....modules.prompt import nsp
from ....modules.util import sandbox

logger = log.get_logger("nodes.text.terminology")

#: The one kind of file a pantry is read from.
PANTRY_EXTENSIONS = (".json",)

#: What the menu says when no pantry file is anywhere the pack may read.
NO_FILES = "no .json files found"

#: What each mode does, in the order the widget offers them.
ADD = "add to the pantry"
REPLACE = "replace the pantry"

MODES = (ADD, REPLACE)


def pantry_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(PANTRY_EXTENSIONS) or [NO_FILES]


class NoodleSoupPantryImport(io.ComfyNode):
    """Store the terminology held in a pantry JSON file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNoodleSoupPantryImport",
            display_name="Noodle Soup Pantry Import",
            search_aliases=[
                "WASNoodleSoupPantryImport",
                "Noodle Soup Pantry Import",
                "nsp",
                "noodle soup",
                "pantry",
                "load terminology",
            ],
            category="WAS Suite/Text/Terminology",
            description=(
                "Read a Noodle Soup Prompts pantry file into the stored pantry, so a "
                "terminology list shared as a file can be used here. The file is a JSON "
                "object of terminology names to word lists, as Noodle Soup Pantry Export "
                "writes; everything read in counts as yours and survives a refresh."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file",
                    options=pantry_options(),
                    tooltip=(
                        "Which pantry file to read. The menu lists every .json file in "
                        "ComfyUI's input, output and temp folders and in any folder added "
                        "under paths.allow_read in config.yaml, each labelled with the "
                        "folder it sits in."
                    ),
                )),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    tooltip=(
                        "`add to the pantry` keeps everything already stored and adds the "
                        "words the file has that a terminology does not. `replace the "
                        "pantry` leaves the pantry holding exactly what the file holds, "
                        "dropping every other terminology."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "What was read and what was stored, as one line per figure. Eg: "
                        "'entries added 12'."
                    ),
                ),
                io.Int.Output(
                    display_name="terms",
                    tooltip="How many terminologies the file held.",
                ),
                io.Int.Output(
                    display_name="entries",
                    tooltip="How many words the file held, counting every terminology.",
                ),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, file="", mode=ADD) -> io.NodeOutput:
        """Read the file and store what it holds.

        Raises:
            ValueError: No file was chosen, or the file is not a pantry.
            PathNotAllowed: The file resolves outside every permitted read root.
            OSError: The file could not be read.
        """
        label = str(file or "").strip()
        if not label or label == NO_FILES:
            raise ValueError(
                "no pantry file was chosen. Put a pantry .json in ComfyUI's input folder, "
                "or in a folder listed under paths.allow_read in config.yaml, then pick it "
                "from the file menu"
            )
        found = picker.resolve(label, PANTRY_EXTENSIONS)
        if not found:
            raise ValueError(
                f"`{label}` is not there any more, so nothing was read. Pick a file from "
                f"the menu again"
            )
        path = sandbox.resolve_read(found)

        report = nsp.import_pantry(path, replace=str(mode) == REPLACE)
        lines = "\n".join(
            f"{name.replace('_', ' ')} {value}"
            for name, value in report.items()
            if name != "saved"
        )
        logger.info("pantry imported from %s:\n%s", path, lines)
        cls.report(path, mode, report, lines)
        return io.NodeOutput(lines, report["terms"], report["entries"])

    @classmethod
    def report(cls, path, mode, report, lines) -> None:
        """Draw what the file held and what was stored on the node."""
        if not report["saved"]:
            status = run_result.ERROR
            summary = f"{path.name} could not be stored, so the pantry is as it was"
        elif report["entries_added"]:
            status = run_result.OK
            summary = (
                f"{report['entries_added']} entry(s) added from {path.name}, "
                f"{report['total']} in the pantry now"
            )
        else:
            status = run_result.WARNING
            summary = f"every entry in {path.name} was already stored"
        library_report.publish(
            summary=summary,
            counts={
                "terms": report["terms"],
                "entries": report["entries"],
                "terms +": report["terms_added"],
                "entries +": report["entries_added"],
                "in pantry": report["total"],
            },
            facts={"file": str(path), "mode": str(mode)},
            lines=lines.splitlines(),
            listing="figures",
            status=status,
        )
