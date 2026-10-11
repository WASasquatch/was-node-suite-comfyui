"""Read the documents a zip archive holds, as one list and as a DOC list."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.archive import container
from ...modules.io import picker
from ...modules import log
from ...modules.archive import kinds, picks
from ...modules.compat.types import DOC, LIST, ZIP
from ...modules.document import container

logger = log.get_logger("nodes.archive")

#: What the archive menu lists.
ARCHIVE_EXTENSIONS = (container.SUFFIX,)

#: Entries the archive menu offers, and what it says when there are none.
NO_ARCHIVES = "no .zip files found"


def archive_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(ARCHIVE_EXTENSIONS) or [NO_ARCHIVES]


def archive_path(file: str) -> str:
    """The archive one menu entry names, as a path, or an empty string."""
    entry = str(file or "").strip()
    if not entry or entry == NO_ARCHIVES:
        return ""
    return picker.resolve(entry, ARCHIVE_EXTENSIONS) or ""




class LoadDocumentsFromZip(io.ComfyNode):
    """Read every document in one archive that a pattern picks, as ``doc`` and ``name`` lists."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadDocumentsFromZIP",
            display_name="Load Documents from ZIP",
            search_aliases=[
                "WASLoadDocumentsFromZIP", "Load Documents from ZIP",
                "Load DOCs from ZIP",
                "zip",
                "archive",
                "unzip",
                "wasdoc",
                "document batch",
            ],
            category="WAS Suite/Archive",
            description=(
                "Read the .wasdoc documents inside a zip archive, both as one LIST and as a DOC "
                "list that runs everything downstream once per document, each with the name "
                "it arrived under."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=archive_options(),
                    tooltip=(
                        "Which archive to read. The menu lists every .zip in ComfyUI's "
                        "input, output and temp folders and in any folder added under "
                        "paths.allow_read, each tagged with where it sits. Ignored while the "
                        "zip socket is connected."
                    ),
                ),
                ZIP.Input(
                    "zip",
                    optional=True,
                    tooltip=(
                        "The archive to read, from Open ZIP. Connected, it is used and "
                        "the menu is ignored, so the archive is opened and indexed once "
                        "however many nodes read it."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which entries to read, case ignored: '*' = every document at any "
                        "depth; 'report_*' = documents named so, in any folder; 'drafts/*' = "
                        "that folder only; 'drafts/**/*' = it and everything under it."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="documents",
                    tooltip=(
                        "Every document on one wire, in the order the names sort, for Text "
                        "List Get and Text List Length. Entry 3 here is the document of name "
                        "3 in the 'names' output."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The name each document came from, on one wire and in the same order, "
                        "such as 'drafts/report.wasdoc'. The folders inside the archive are "
                        "kept, so two documents of the same name in different folders stay "
                        "apart."
                    ),
                ),
                DOC.Output(
                    display_name="doc",
                    is_output_list=True,
                    tooltip=(
                        "The same documents as a DOC list, so the graph below runs once per "
                        "document: wire it into View DOC Metadata, Convert DOC to Plaintext "
                        "or Save DOC."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    is_output_list=True,
                    tooltip=(
                        "The file name that goes with each run of the 'doc' output, so a "
                        "converted or exported copy can carry the name it came in under. Wire "
                        "it into Save DOC's filename_prefix beside the matching document."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many documents were read, which is the length of both lists.",
                ),
                io.Int.Output(
                    display_name="skipped",
                    tooltip=(
                        "How many entries were not read: not a .wasdoc, an unsafe name, a "
                        "symbolic link, encrypted, repeated, too large, damaged or not a "
                        "document. The log names every one."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", pattern="*", zip=None):
        """Read again when the archive on disk, or the pattern, has changed."""
        return picks.fingerprint(zip if zip is not None else archive_path(file), pattern)

    @classmethod
    def execute(cls, file="", pattern="*", zip=None) -> io.NodeOutput:
        """Open the archive and read every document the pattern picks.

        Raises:
            NotAnArchive: ``file`` is empty, names a folder, names nothing that is
                there, or names a file that is not a readable zip.
            PathNotAllowed: It resolved outside every permitted read root.
            ValueError: No entry produced a document.
        """
        from ...modules.compat.lists import require_values

        archive = picks.opened_archive(zip if zip is not None else archive_path(file))
        picks.refuse_document_container(
            archive,
            "This node reads a zip archive holding one or more documents, rather than one "
            "document on its own. Put the documents into a zip archive and read that.",
        )
        members, report = picks.read_matching(archive, pattern, kinds.DOCUMENT)

        names: list[str] = []
        documents = []
        for member in members:
            try:
                # A container carries its own bounds on what it unpacks to, so a document
                # inside an archive is read under the same limits as one read from disk.
                documents.append(container.Document.from_bytes(member.data))
            except container.DocumentError as error:
                report.skip(
                    picks.NOT_A_DOCUMENT,
                    f"the entry {member.name!r} could not be read as a document, so it was "
                    f"skipped: {error}",
                )
                continue
            names.append(member.name)

        for note in report.notes:
            logger.warning("%s: %s", archive.label, note)
        logger.info(
            "Load Documents from ZIP read %s: %s", archive.label, report.summary(len(documents))
        )
        require_values(documents, cls.nothing(archive, pattern, report))
        # Each slot gets a list of its own. The same list on two slots would let a node
        # that edits the LIST it was handed change how many times the graph under the
        # doc output runs.
        return io.NodeOutput(
            documents, names, list(documents), list(names), len(documents), report.total
        )

    @staticmethod
    def nothing(archive, pattern: str, report: picks.Report) -> str:
        """The message for an archive that produced no document at all.

        Args:
            archive: The archive that was read.
            pattern: The pattern as the user wrote it.
            report: What the read left out.

        Returns:
            A message naming the archive, then whichever of the three reasons applies: it
            holds no readable file, the pattern picked none of them, or every entry the
            pattern picked was skipped.
        """
        offered = ", ".join(report.examples) or "nothing that can be read out of it"
        left_out = f" {report.summary(0)}." if report.total else ""
        if not report.examined:
            return (
                f"the archive {archive.label} holds no files, so Load Documents from ZIP has "
                f"nothing to hand on and the graph below it cannot be run. It opened "
                f"correctly and is simply empty."
            )
        if not report.matched:
            return (
                f"no entry in {archive.label} is picked by the pattern `{pattern}`, so Load "
                f"Documents from ZIP has nothing to hand on and the graph below it cannot be "
                f"run. It holds {offered}.{left_out} A pattern with no '/' in it is matched "
                f"against the file's own name at any depth, so '*' reads them wherever they "
                f"sit."
            )
        return (
            f"every entry the pattern `{pattern}` picked in {archive.label} was skipped, so "
            f"Load Documents from ZIP has nothing to hand on and the graph below it cannot be "
            f"run: {report.summary(0)}. The log names each one. This node reads "
            f"{kinds.extension_list(kinds.DOCUMENT)} documents, the format Save DOC writes; "
            f"text and pictures in an archive are read by the other archive nodes."
        )
