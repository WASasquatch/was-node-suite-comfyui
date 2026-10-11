"""Write the style library out as a JSON or AUTOMATIC1111 CSV file."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.io import rooted
from ....modules.prompt import styles
from ....modules.util import sandbox

logger = log.get_logger("text.styles")


class PromptStylesExport(io.ComfyNode):
    """Write the style library out to a file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPromptStylesExport",
            display_name="Prompt Styles Export",
            search_aliases=[
                "WASPromptStylesExport",
                "Prompt Styles Export",
                "style",
                "a1111 styles",
                "save styles",
                "styles.csv",
            ],
            category="WAS Suite/Text/Styles",
            description=(
                "Write the whole style library out to a file, to share it or back it up, "
                "replacing any file already at that name. A name ending in .csv writes "
                "AUTOMATIC1111's name, prompt and negative_prompt columns; any other name "
                "writes JSON. Prompt Styles Import reads either back."
            ),
            inputs=[
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, "
                        "listed by its own name. filename names the part below it."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="styles.json",
                    multiline=False,
                    tooltip=(
                        "Name of the file, and any folder below root to put it in. "
                        "styles.json writes a JSON library, styles.csv writes A1111 "
                        "columns. Eg: styles/[time(%Y-%m-%d)].json"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="path",
                    tooltip="The full path of the file that was written.",
                ),
                io.Int.Output(
                    display_name="styles",
                    tooltip="How many styles were written.",
                ),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, root=rooted.DEFAULT, filename="styles.json") -> io.NodeOutput:
        """Write the library and answer where it landed.

        Raises:
            ValueError: ``filename`` is empty, or the library holds no style.
            PathNotAllowed: The folder resolved outside every permitted write root.
            OSError: The folder could not be made.
        """
        wanted = str(filename or "").strip()
        if not wanted:
            raise ValueError(
                "no file name was given, so there is nowhere to write. Type a name such as "
                "styles.json, or styles.csv for AUTOMATIC1111 columns"
            )
        library = styles.library()
        if not library:
            raise ValueError(
                "the style library holds no style, so there is nothing to write. Save one "
                "with Prompt Style Save, or read one in with Prompt Styles Import, first"
            )

        below, _, leaf = wanted.replace("\\", "/").rpartition("/")
        if not leaf:
            raise ValueError(
                f"`{wanted}` names a folder rather than a file. End it with a file name, "
                f"such as {below}/styles.json"
            )
        directory = rooted.destination(root, below)
        directory.mkdir(parents=True, exist_ok=True)
        target = sandbox.resolve_write_file(directory, leaf)

        written = styles.export_styles(target)
        logger.info("wrote %s style(s) to %s", written, target)
        cls.report(target, written, library)
        return io.NodeOutput(str(target), written)

    @classmethod
    def report(cls, target, written, library) -> None:
        """Draw what was written and where on the node."""
        if not written:
            status = run_result.ERROR
            summary = f"{target.name} could not be written, so no file was left behind"
        else:
            status = run_result.OK
            summary = f"wrote {written} style(s) to {target.name}"
        library_report.publish(
            summary=summary,
            counts={"styles": written},
            facts={
                "file": str(target),
                "format": "a1111 csv" if target.suffix.lower() == ".csv" else "json",
            },
            lines=list(library),
            listing="styles",
            total=len(library),
            status=status,
        )
