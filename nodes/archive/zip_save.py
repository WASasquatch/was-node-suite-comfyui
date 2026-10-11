"""Write chosen files from ComfyUI's own folders into one zip archive."""

from __future__ import annotations

import os
from pathlib import Path

from comfy_api.latest import io, ui

from ...modules.io import rooted
from ...modules.archive import save, selection, summary
from ...modules.compat.types import LIST
from ...modules.log import get_logger
from ...modules.util import file_listing, filenames, sandbox

logger = get_logger("nodes.archive")

#: Directory the node writes into unless the widget is changed, the same default the pack's
#: other save nodes carry.
DEFAULT_PATH = "./ComfyUI/output/[time(%Y-%m-%d)]"

#: The extension the archive is written under.
SUFFIX = ".zip"

#: How many entry lines the note on the node lists before it stops naming them.
NOTE_LINES = 200


class ZipSave(io.ComfyNode):
    """Collect files from the input, output and temp folders into a zip archive."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASZipSave",
            display_name="Zip Save",
            search_aliases=[
                "WASZipSave",
                "Zip Save",
                "zip files",
                "archive files",
                "save zip",
                "collect files",
                "compress",
                "bundle outputs",
            ],
            category="WAS Suite/Archive",
            description=(
                (
                    "Put files from ComfyUI's input, output and temp folders into one zip "
                    "archive, picked in the browser on the node or typed one per line, and "
                    "hand on the path of the archive written."
                )
            ),
            inputs=[
                io.String.Input(
                    "files",
                    default="",
                    multiline=True,
                    tooltip=(
                        "The files going into the archive, one per line, in archive order: a "
                        "menu label such as 'renders/cat.png [output]', or a full path. Blank "
                        "lines are ignored, '#' lines are comments, and a file named twice "
                        "goes in once."
                    ),
                ),
                io.Combo.Input(
                    "entry_paths",
                    options=list(save.NAMING),
                    default=save.RELATIVE,
                    tooltip=(
                        "Names inside the archive: `relative path` = its path below input, "
                        "output or temp; `source folder and relative path` = "
                        "'output/batch/cat.png'; `file name only` = 'cat.png', a clash "
                        "numbered 'cat_2.png'."
                    ),
                ),
                io.Combo.Input(
                    "compression",
                    options=list(save.COMPRESSIONS),
                    default="deflate",
                    tooltip=(
                        "How the files are packed. 'deflate' makes text, JSON and "
                        "documents much smaller; 'store' packs them unchanged, which is "
                        "faster and suits PNGs and JPEGs, compressed already."
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
                        "The name part of the archive, before the number. Tokens are "
                        "expanded here too, so a date or a custom token can go in the name "
                        "rather than the folder."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: 'ComfyUI_0001.zip' with "
                        "the default, 'ComfyUI0001.zip' if cleared."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    tooltip=(
                        "How many digits the number is padded to with leading zeros: 4 "
                        "gives '_0001', 1 gives '_1'. 0 drops the number and rewrites the "
                        "same file every run."
                    ),
                ),
                io.String.Input(
                    "filename_suffix",
                    default="",
                    optional=True,
                    tooltip=(
                        "Extra text placed after the number and before the extension, so a "
                        "suffix of '_renders' gives 'ComfyUI_0001_renders.zip'. Empty by "
                        "default."
                    ),
                ),
                io.MultiType.Input(
                    "paths",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "More files to archive, from a link, added after the typed lines. A "
                        "string holding several lines is read as several files."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="zip_path",
                    tooltip=(
                        "The full path of the archive that was written, numbering and all, so "
                        "a later node can report it, print it, or open it again with Zip "
                        "Open."
                    ),
                ),
                io.Int.Output(
                    display_name="file_count",
                    tooltip=(
                        "How many files went into the archive. Lower than the number of lines "
                        "chosen when one of them had been deleted, which the log names."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        files,
        entry_paths,
        compression,
        filename_prefix,
        filename_delimiter,
        filename_number_padding,
        filename_suffix,
        paths,
        root=rooted.DEFAULT,
    ) -> float:
        """Always stale: any chosen file can be rewritten between runs.

        Returns:
            NaN, which never equals itself, so queueing the prompt again archives the files
            as they are now rather than reporting the archive it wrote last time.
        """
        return float("NaN")

    @classmethod
    def execute(
        cls,
        files="",
        entry_paths=save.RELATIVE,
        compression="deflate",
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        filename_number_padding=4,
        filename_suffix="",
        paths=None,
    ) -> io.NodeOutput:
        """Read the selection, then write the archive.

        Returns:
            The path of the archive written and how many files went into it.

        Raises:
            ValueError: Nothing was chosen, nothing that was chosen is still there, or the
                selection is larger than one archive may hold.
            PathNotAllowed: ``path`` resolved outside every permitted write root, or a chosen
                path outside every readable one.
            OSError: The folder could not be created, or the archive could not be written.
        """
        from ...modules.compat.lists import require_values

        entries, repeats = selection.parse(cls.combined(files, paths))
        require_values(entries, cls.nothing_chosen())
        if repeats:
            logger.info(
                "%d chosen file(s) were named more than once and go into the archive once",
                repeats,
            )

        sources, missing = selection.sources(entries)
        if missing:
            logger.warning("Zip Save skipped %d of them: %s", len(missing), selection.gone(missing))
        require_values(sources, cls.nothing_left(missing))

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

        filename = filenames.generate_filename(
            directory,
            filename_prefix,
            filename_delimiter,
            int(filename_number_padding),
            SUFFIX,
            filename_suffix,
        )
        target = sandbox.resolve_write_file(directory, filename)
        replaced = target.exists()
        written = save.build(target, sources, entry_paths, compression)

        for name, label in sorted(written.renamed.items()):
            logger.info(
                "%s went in as %r, because another chosen file had already taken the name "
                "'%s' gives it", label, name, entry_paths,
            )
        logger.info(
            "%s %d file(s), %s, into %s",
            "replaced" if replaced else "wrote", len(written.names),
            summary.size_text(written.size), target,
        )
        note = _note(target, written, missing, entry_paths, compression, replaced)
        return io.NodeOutput(str(target), len(written.names), ui=ui.PreviewText(note))

    @staticmethod
    def combined(files, paths) -> str:
        """The whole selection: the widget's lines, then whatever arrived on ``paths``.

        Args:
            files: The widget value.
            paths: The linked value, a string, a list of strings, or nothing.

        Returns:
            One block of text for :func:`modules.archive.selection.parse`, which drops the
            blanks, the comments and the repeats.
        """
        from ...modules.compat.lists import as_list

        extra = [str(item).strip() for item in as_list(paths)]
        return "\n".join([str(files or "")] + [item for item in extra if item])

    @staticmethod
    def nothing_chosen() -> str:
        """What to say when the selection is empty."""
        folders = ", ".join(f"{tag} ({directory})" for tag, directory in file_listing.roots())
        where = folders or "ComfyUI's input, output and temp folders, which could not be found"
        return (
            f"Zip Save has no files chosen, so there is nothing to put in an archive and no "
            f"file was written.\n"
            f"  Pick files in the browser on the node, or type them into 'files' one per "
            f"line: a menu label such as 'renders/cat.png [output]', or a full path to a "
            f"file this pack may read.\n"
            f"  The browser lists {where}.\n"
            f"  A save node's own output can feed it instead: wire file_path into the "
            f"'paths' input."
        )

    @staticmethod
    def nothing_left(missing: dict[str, str]) -> str:
        """What to say when every chosen file has gone."""
        named = "\n".join(f"    {entry}: {reason}" for entry, reason in missing.items())
        return (
            f"Zip Save found none of the {len(missing)} chosen file(s), so no archive was "
            f"written:\n{named}\n"
            f"  A workflow saved on another machine, or one whose renders have been cleared, "
            f"arrives here. Pick the files again in the browser on the node."
        )


def _note(
    target: Path,
    written: save.Written,
    missing: dict[str, str],
    naming: str,
    compression: str,
    replaced: bool,
) -> str:
    """What the node shows on itself after a write.

    Args:
        target: The archive that was written.
        written: What went into it.
        missing: The chosen files that were not there.
        naming: The naming rule the entries were given.
        compression: How they were packed.
        replaced: Whether a file of that name was already there.

    Returns:
        A few lines naming the archive, its size and what went in, then one line per entry
        with its size, cut short after :data:`NOTE_LINES`, then whatever was skipped.
    """
    lines = [
        f"{'replaced' if replaced else 'wrote'} {target.name} "
        f"({summary.size_text(written.size)})",
        str(target),
        f"{len(written.names)} file(s), {summary.size_text(written.total)} read, "
        f"{compression}, named by {naming}",
    ]
    for name in written.names[:NOTE_LINES]:
        lines.append(f"  {name}")
    if len(written.names) > NOTE_LINES:
        lines.append(f"  ... and {len(written.names) - NOTE_LINES} more")
    if written.renamed:
        lines.append(
            f"{len(written.renamed)} numbered apart from a file of the same name: "
            f"{', '.join(sorted(written.renamed))}"
        )
    if missing:
        lines.append(f"skipped: {selection.gone(missing)}")
    if written.skipped:
        lines.append(
            f"could not be read: {', '.join(sorted(written.skipped))}"
        )
    return "\n".join(lines)
