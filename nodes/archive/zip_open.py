"""Open a zip archive and report what is inside it."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.archive import container, picks, summary
from ...modules.archive.container import Archive
from ...modules.compat.types import LIST, ZIP
from ...modules.log import get_logger
from ...modules.io import picker
from ...modules.util import file_listing

logger = get_logger("nodes.archive")

#: The extensions the menu lists. One entry: a zip is the archive format the pack reads, and
#: a document container is opened by the document nodes rather than as an archive.
ARCHIVE_EXTENSIONS = (container.SUFFIX,)

#: Menu entry shown when none of the three folders holds an archive, and outside ComfyUI,
#: where none of them can be found. One empty state rather than several.
NO_ARCHIVES = "No Archives"

#: How many entries the menu offers.
MAX_OPTIONS = 500


def options() -> list[str]:
    """The menu's entries, or ``[NO_ARCHIVES]`` when there are none."""
    return list(picker.labels(ARCHIVE_EXTENSIONS)[:MAX_OPTIONS]) or [NO_ARCHIVES]


class ZipOpen(io.ComfyNode):
    """Open an archive and list its entries, the ``archive`` output carrying an index of them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASZipOpen",
            display_name="Open ZIP",
            search_aliases=[
                "WASZipOpen",
                "Zip Open",
                "open zip",
                "read zip",
                "list archive",
                "unzip",
                "extract",
            ],
            category="WAS Suite/Archive",
            description=(
                (
                    "Open a zip archive and report what is in it: a line per file with its "
                    "kind and size, on the node and on the listing output. Nothing is "
                    "unpacked, so a large archive costs what a small one costs. The zip "
                    "output feeds ZIP Add, ZIP Manage, Zip Extract and the Load ... from "
                    "ZIP nodes. The file menu reaches every "
                    "subfolder and picks up a dropped file within about five seconds. The "
                    "temp folder is emptied on restart, so a '[temp]' entry will not be there "
                    "next session. Unsafe entries are named and skipped, the rest still read: "
                    "one landing outside its folder, a symbolic link, a name holding a null "
                    "byte, or one claiming to unpack past a quarter of a gigabyte. A file that "
                    "is not a zip, or is damaged, stops the run saying which."
                )
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=options(),
                    tooltip=(
                        "Which archive to open. The menu lists every .zip in ComfyUI's "
                        "input, output and temp folders and in any folder added under "
                        "paths.allow_read, each tagged with where it sits."
                    ),
                ),
            ],
            outputs=[
                ZIP.Output(
                    display_name="zip",
                    tooltip=(
                        "The opened archive: its entry list, each entry's size and where the "
                        "file sits, for a node that reads files out of one."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "Every readable entry name on one wire, folders inside the archive "
                        "included, for Text List Slice, Text List Get, Text List Length and "
                        "Text List to Strings. Empty for an archive holding nothing readable, "
                        "which is a list the list nodes handle. Refused entries are absent."
                    ),
                ),
                io.String.Output(
                    display_name="entry_names",
                    is_output_list=True,
                    tooltip=(
                        "The same names, one per run, so the nodes after this one run once for "
                        "each entry in the archive. An archive holding nothing readable stops "
                        "the nodes reading this output with a message, and leaves the other "
                        "four outputs working."
                    ),
                ),
                io.Int.Output(
                    display_name="entry_count",
                    tooltip=(
                        "How many readable files the archive holds. Folder entries are not "
                        "counted, and two entries under one name count once, which is how "
                        "many files a reader can get out."
                    ),
                ),
                io.String.Output(
                    display_name="listing",
                    tooltip=(
                        "The report shown on the node, as text: what the archive holds, a line "
                        "per file with its kind and size, and a line per refused entry saying "
                        "why it was refused. Wire it into Text to Console or Save Text File to "
                        "keep a record of what an archive held."
                    ),
                ),
                io.String.Output(
                    display_name="zip_path",
                    tooltip=(
                        "The full path of the archive that was opened, which is what the Load "
                        "Images from ZIP, Load Text Files from ZIP and Load Documents from ZIP "
                        "nodes take. Wire this into their zip_path input to pick an archive "
                        "from the menu here and read it there."
                    ),
                ),
            ],
            is_output_node=True,
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
        # check for an input named in this signature. Without it, a workflow naming an archive
        # that has been deleted, renamed or pushed out of the menu by the option cap fails
        # before execute runs, with a message naming neither the file nor where it was looked
        # for. Naming only `file` leaves every other input checked as usual.
        return True

    @classmethod
    def fingerprint_inputs(cls, file) -> str | float:
        """The archive's own path, size and modification time.

        Returns:
            A value that changes when the file on disk does, so an archive rewritten in place
            is read again, and NaN where nothing can be stated, which leaves the node to run
            and report what stopped it.
        """
        return picks.fingerprint(cls.chosen(file), file)

    @classmethod
    def execute(cls, file="") -> io.NodeOutput:
        """Open the chosen archive and read its index.

        Returns:
            The archive, its entry names as a list and one per run, how many it holds, the
            report, and the path that was opened.

        Raises:
            ValueError: No archive was chosen, or the menu entry names no file.
            NotAnArchive: The file is not there, is a folder, or is not a readable zip.
            PathNotAllowed: The chosen file resolved outside every permitted read root.
        """
        from ...modules.compat.lists import block_if_empty

        wanted = cls.chosen(file)
        if not wanted:
            raise ValueError(cls.missing(str(file or "").strip()))
        # One spelling of resolve, exists, is a file and is a zip, so this node and the Load
        # ... from ZIP nodes refuse the same file with the same words.
        archive = picks.opened_archive(wanted)
        report = summary.listing_text(archive)
        cls.report(archive)
        names = list(archive.names)
        return io.NodeOutput(
            archive,
            names,
            block_if_empty(list(names), cls.nothing_readable(archive)),
            len(names),
            report,
            str(archive.source or wanted),
            ui=ui.PreviewText(report),
        )

    @staticmethod
    def chosen(file) -> str:
        """Which archive the menu names, as a path.

        Args:
            file: The menu entry.

        Returns:
            The path to open, or an empty string where nothing was chosen and where the entry
            names no file the listing holds. Never raises, since ``fingerprint_inputs`` calls
            it too.
        """
        entry = str(file or "").strip()
        if not entry or entry == NO_ARCHIVES:
            return ""
        return picker.resolve(entry, ARCHIVE_EXTENSIONS) or ""

    @staticmethod
    def report(archive: Archive) -> None:
        """Log what the archive holds, and every entry that was refused.

        Args:
            archive: The opened archive.
        """
        logger.info(
            "opened %s: %d readable file(s), %s",
            archive.label, len(archive.files),
            summary.kind_counts_text(summary.counts(archive)),
        )
        if archive.truncated:
            logger.warning(
                "%s holds %d entries and the first %d were listed; the rest are not offered",
                archive.label, archive.held, len(archive.entries),
            )
        for entry in archive.refused:
            logger.warning(
                "%r in %s %s",
                entry.stored, archive.label, container.REFUSALS.get(entry.refusal, entry.refusal),
            )

    @staticmethod
    def nothing_readable(archive: Archive) -> str:
        """What the nodes reading ``entry_names`` are told when there are no entries."""
        if archive.refused:
            return (
                f"{archive.label} holds nothing that can be read out: every one of its "
                f"{len(archive.refused)} entries was refused, and the log names each reason."
            )
        if archive.directories:
            return (
                f"{archive.label} holds only folder entries, which hold no data of their own."
            )
        return f"{archive.label} is an empty archive: there is nothing in it to read."

    @staticmethod
    def missing(entry: str) -> str:
        """What to say when the chosen menu entry names no file.

        Args:
            entry: The stored combo value, stripped.

        Returns:
            A message naming the entry and the three folders the menu is built from, since a
            workflow saved elsewhere is the usual way to arrive here.
        """
        folders = ", ".join(f"{tag} ({path})" for tag, path in file_listing.roots())
        where = folders or "ComfyUI's input, output and temp folders, which could not be found"
        if not entry or entry == NO_ARCHIVES:
            return (
                f"Zip Open has no archive chosen. Pick one from its menu, which lists the "
                f".zip files in {where}. A folder added under paths.allow_read in config.yaml appears "
                f"in the menu under its own name."
            )
        return (
            f"Zip Open found no .zip file named `{entry}`. It may have been deleted, renamed, "
            f"or moved between folders since the workflow was saved, and ComfyUI's temp folder "
            f"is emptied on restart.\n"
            f"  Pick it again from the menu, which lists the .zip files in {where}, or type "
            f"its folder to paths.allow_read in config.yaml."
        )
