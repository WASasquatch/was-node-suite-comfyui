"""Write the files an opened archive holds out into a folder on disk."""

from __future__ import annotations

from pathlib import Path

from comfy_api.latest import io, ui

from ...modules.archive import container, extract
from ...modules.compat.types import LIST, ZIP
from ...modules.io import rooted
from ...modules.log import get_logger
from ...modules.util import filenames

logger = get_logger("nodes.archive")

#: Folder the node writes into unless the widget is changed.
DEFAULT_PATH = "./ComfyUI/output/extracted/[time(%Y-%m-%d)]"


class ZipExtract(io.ComfyNode):
    """Unpack the entries a ZIP holds into a folder, and answer where they landed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASZipExtract",
            display_name="Zip Extract",
            search_aliases=[
                "WASZipExtract",
                "Zip Extract",
                "unzip",
                "extract zip",
                "unpack archive",
                "archive to folder",
                "write files from zip",
            ],
            category="WAS Suite/Archive",
            description=(
                "Unpack files of any kind from an archive opened by Zip Open into a folder, "
                "picked by a glob. Every written file's path comes out as one list and one per "
                "run, so the graph below can load, caption or resave each one."
            ),
            inputs=[
                ZIP.Input(
                    "zip",
                    tooltip=(
                        "The opened archive to unpack, from Zip Open's archive output. The "
                        "file itself is read again here, so an archive rewritten between the "
                        "two nodes stops the run rather than writing a mixture."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which entries to write. `*` takes every readable entry, `*.png` "
                        "every picture at any depth, and `frames/**/*.png` anchors at the top "
                        "of the archive and reaches any depth under it. Case is ignored."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the entries are written into: ComfyUI's own 'output' "
                        "or 'temp', or any folder added under paths.allow_write in "
                        "config.yaml, listed by its own name. folder names the part below it."
                    ),
                ),
                io.String.Input(
                    "folder",
                    default="extracted",
                    multiline=False,
                    tooltip=(
                        "Folder below the root the files land in, created if it is not "
                        "there. Tokens expand, so 'extracted/[time(%Y-%m-%d)]' files each "
                        "day's unpacking under its own dated folder. Empty writes into the "
                        "root itself."
                    ),
                ),
                io.Combo.Input(
                    "entry_paths",
                    options=list(extract.NAMING),
                    default=extract.KEEP,
                    tooltip=(
                        "What the files are called on disk. `keep folders` rebuilds the "
                        "folders inside the archive, so 'frames/cat.png' becomes a 'frames' "
                        "folder; `file name only` drops them and writes 'cat.png' straight "
                        "into the chosen folder."
                    ),
                ),
                io.Combo.Input(
                    "existing",
                    options=list(extract.EXISTING),
                    default=extract.OVERWRITE,
                    tooltip=(
                        "Where the folder already holds that name: `overwrite` = replaces it; "
                        "`skip` = leaves it, named in the report; `number apart` = writes "
                        "'cat_2.png' beside it. Two entries reaching one name in a run are "
                        "always numbered apart."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="paths",
                    tooltip=(
                        "Every written file's full path on one wire, in name order, for Text "
                        "List Get, Text List Slice and Text List Length. Entry 3 here is the "
                        "path of name 3 in the 'names' output."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The archive name each file came from, on one wire and in the same "
                        "order, folders inside the archive included, so a file numbered apart "
                        "on disk can still be traced to the entry it held."
                    ),
                ),
                io.String.Output(
                    display_name="file_path",
                    is_output_list=True,
                    tooltip=(
                        "The same paths one per run, so the nodes below run once per file: "
                        "wire it into Load Image Batch, Load Text File or Path Exists. A run "
                        "that wrote nothing stops only the nodes reading this output."
                    ),
                ),
                io.Int.Output(
                    display_name="file_count",
                    tooltip=(
                        "How many files were written. Lower than the number the pattern "
                        "picked where an entry was passed over, which the report names."
                    ),
                ),
                io.String.Output(
                    display_name="folder",
                    tooltip=(
                        "The full path of the folder the files were written into, tokens "
                        "expanded and created if it was not there. Wire it into Directory "
                        "Listing to read the folder back, or into Text to Console to record "
                        "where a run put its files."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "The report shown on the node, as text: where the files went, a line "
                        "per file with its size, and a line per entry that was passed over. "
                        "Wire it into Save Text File to keep a record of what an archive gave "
                        "up."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, zip, pattern, root, folder, entry_paths, existing) -> float:
        """Always stale: the folder written into can be emptied between runs.

        Returns:
            NaN, which never equals itself, so queueing the prompt again writes the files
            rather than reporting the run before it.
        """
        return float("NaN")

    @classmethod
    def execute(
        cls,
        zip=None,
        pattern="*",
        root=rooted.DEFAULT,
        folder="extracted",
        entry_paths=extract.KEEP,
        existing=extract.OVERWRITE,
    ) -> io.NodeOutput:
        """Write the entries a pattern picks into the destination folder.

        Returns:
            The paths written as a list and one per run, how many there were, the folder,
            and the report.

        Raises:
            NotAnArchive: The ``archive`` input was given something other than an archive.
            PathNotAllowed: The chosen root is not a folder this pack may write to.
            OSError: The folder could not be created, or a file could not be written.
        """
        from ...modules.compat.lists import block_if_empty

        opened = container.require_archive(zip, "zip")
        entries = extract.chosen(opened, pattern)
        folder = rooted.destination(root, folder)
        folder.mkdir(parents=True, exist_ok=True)

        result = extract.run(opened, entries, folder, str(entry_paths), str(existing))
        cls.report(opened, folder, result)
        report = extract.report_text(opened, folder, result)
        return io.NodeOutput(
            list(result.paths),
            list(result.names),
            block_if_empty(list(result.paths), cls.nothing_written(opened, pattern, result)),
            len(result.written),
            str(folder),
            report,
            ui=ui.PreviewText(report),
        )

    @staticmethod
    def report(archive: container.Archive, folder: Path, result: extract.Result) -> None:
        """Log what reached the folder, and every entry that did not.

        Args:
            archive: The archive that was read.
            folder: The destination folder.
            result: What the run did.
        """
        logger.info(
            "wrote %d file(s) out of %s into %s",
            len(result.written), archive.label, folder,
        )
        for name, entry in sorted(result.renamed.items()):
            logger.info(
                "%r went in as '%s', because another file had already taken the name", entry, name
            )
        for name, reason in sorted(result.skipped.items()):
            logger.warning("%r was not written: %s", name, reason)
        for line in result.bounds:
            logger.warning("%s", line)

    @staticmethod
    def nothing_written(
        archive: container.Archive, pattern: str, result: extract.Result
    ) -> str:
        """What the nodes reading ``file_path`` are told when no file was written.

        Args:
            archive: The archive that was read.
            pattern: The glob as it was written.
            result: What the run did.

        Returns:
            A message naming why nothing landed: the pattern picked nothing, or everything
            it picked was passed over.
        """
        if result.skipped:
            return (
                f"Zip Extract wrote no files: all {len(result.skipped)} entry(ies) the "
                f"pattern picked were passed over, and the report on the node names each "
                f"reason."
            )
        listed = ", ".join(archive.names[:8]) or "nothing readable"
        return (
            f"Zip Extract found no entry in {archive.label} matching `{pattern}`, so no file "
            f"was written. It holds {listed}.\n"
            f"  A pattern with no '/' is matched against the file name at any depth, so "
            f"'*.png' finds one in every folder, while 'frames/*.png' reads that one folder."
        )
