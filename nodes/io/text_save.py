"""Write a string to a numbered text file in a directory the pack may write to."""

from __future__ import annotations

import os

from comfy_api.latest import io, ui

from ...modules.io import rooted
from ...modules import log
from ...modules.state import history
from ...modules.util import sandbox
from ...modules.util.filenames import generate_filename, resolve_output_directory

logger = log.get_logger("nodes.io")


class SaveTextFile(io.ComfyNode):
    """Write text to a numbered file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Save Text File",
            display_name="Save Text File",
            search_aliases=["Save Text File", "write text", "text file"],
            category="WAS Suite/IO",
            description=(
                "Write text to a numbered file in the folder picked with root, named from "
                "filename_prefix, the number and an optional suffix. Tokens in the prefix "
                "expand, so a date can name a subfolder or the file. Padding 0 writes the "
                "same file every run."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: the file contents",
                    tooltip=(
                        "File contents; STRING, as `a tabby cat`. Written exactly as given. Empty writes "
                        "an empty file."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename_prefix names the part below it, so "
                        "'[time(%Y-%m-%d)]/notes' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "The name part of each file, before the number. Tokens are expanded "
                        "here too, so a date or a custom token can go in the name rather "
                        "than the folder."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: 'ComfyUI_0001.txt' with "
                        "the default, 'ComfyUI0001.txt' if cleared."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    tooltip=(
                        "How many digits the number is padded to with leading zeros: 4 gives "
                        "'_0001', 1 gives '_1'. Set it to 0 to drop the number and the "
                        "delimiter entirely and write to the same file every run, replacing "
                        "what was there."
                    ),
                ),
                io.String.Input(
                    "file_extension",
                    default=".txt",
                    optional=True,
                    tooltip=(
                        "Ending of the file name, leading dot included: '.txt', '.json', "
                        "'.csv'. It only names the file; the text is written exactly as it "
                        "arrives either way."
                    ),
                ),
                io.String.Input(
                    "encoding",
                    default="utf-8",
                    optional=True,
                    tooltip=(
                        "Character encoding the file is written in. 'utf-8' handles every "
                        "language and is what almost everything reads; change it only for a "
                        "program that insists on something else, such as 'latin-1'."
                    ),
                ),
                io.String.Input(
                    "filename_suffix",
                    default="",
                    optional=True,
                    tooltip=(
                        "Extra text placed after the number and before the extension, so a "
                        "suffix of '_caption' gives 'ComfyUI_0001_caption.txt'. Empty by "
                        "default."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        text,
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        filename_number_padding=4,
        file_extension=".txt",
        encoding="utf-8",
        filename_suffix="",
    ) -> io.NodeOutput:
        """Write the text to the next unused file name in the directory.

        Raises:
            OSError: The directory could not be created, or the file could not be written.
            PathNotAllowed: ``path`` resolved outside every permitted write root, or
                ``filename_prefix`` named a file outside the directory.
        """
        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        filename_prefix = leaf

        if not directory.exists():
            logger.warning("the path `%s` doesn't exist! Creating it...", directory)
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as error:
                logger.error(
                    "the path `%s` could not be created! Is there write access?\n%s",
                    directory, error,
                )
                raise

        if text.strip() == "":
            logger.error("there is no text specified to save! Text is empty.")

        filename = generate_filename(
            directory,
            filename_prefix,
            filename_delimiter,
            int(filename_number_padding),
            file_extension,
            filename_suffix,
        )
        file_path = sandbox.resolve_write_file(directory, filename)
        with open(file_path, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)

        history.update_history_text_files(str(file_path))
        return io.NodeOutput(ui=ui.PreviewText(text))
