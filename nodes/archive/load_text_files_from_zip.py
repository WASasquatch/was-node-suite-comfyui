"""Read the text files a zip archive holds, as one list and as a STRING list."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.archive import container
from ...modules.io import picker
from ...modules import log
from ...modules.archive import kinds, picks
from ...modules.compat.types import LIST, ZIP
from ...modules.util import text_files

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




class LoadTextFilesFromZip(io.ComfyNode):
    """Read every text file in one archive that a pattern picks, as ``text`` and ``name`` lists."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadTextFilesFromZIP",
            display_name="Load Text Files from ZIP",
            search_aliases=[
                "WASLoadTextFilesFromZIP", "Load Text Files from ZIP",
                "zip",
                "archive",
                "unzip",
                "captions from zip",
                "prompt list archive",
                "text batch",
            ],
            category="WAS Suite/Archive",
            description=(
                "Read the text files inside a zip archive, both as one LIST and as a STRING "
                "list that runs everything downstream once per file, each with the name it "
                "arrived under. Useful for rendering a folder of captions or prompts."
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
                        "Which entries to read, case ignored: '*' = every text file at any "
                        "depth; '*.txt' = every .txt in any folder; 'captions/*.txt' = that "
                        "folder only; 'captions/**/*.txt' = it and everything under it."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="texts",
                    tooltip=(
                        "Every file's text on one wire, in the order the names sort, for Text "
                        "List Get, Text List Slice and Text List Length. Entry 3 here is the "
                        "text of name 3 in the 'names' output."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The name each text came from, on one wire and in the same order, "
                        "such as 'captions/cat.txt'. The folders inside the archive are kept, "
                        "so two files called cat.txt in different folders stay apart."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    is_output_list=True,
                    tooltip=(
                        "The same texts as a STRING list, so the graph below runs once per "
                        "file: wire it into a sampler's prompt to render every caption in the "
                        "archive."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    is_output_list=True,
                    tooltip=(
                        "The file name that goes with each run of the 'text' output, so a "
                        "saved image can carry the name of the caption that made it. Wire it "
                        "into Image Save's filename_prefix beside the matching text."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many files were read, which is the length of both lists.",
                ),
                io.Int.Output(
                    display_name="skipped",
                    tooltip=(
                        "How many entries were not read: wrong kind, an unsafe name, a symbolic "
                        "link, encrypted, repeated, too large, damaged or not UTF-8 text. Files "
                        "past the unpack total are not counted. The log names every one."
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
        """Open the archive and decode every text file the pattern picks.

        Raises:
            NotAnArchive: ``file`` is empty, names a folder, names nothing that is
                there, or names a file that is not a readable zip.
            PathNotAllowed: It resolved outside every permitted read root.
            ValueError: No entry produced any text.
        """
        from ...modules.compat.lists import require_values

        archive = picks.opened_archive(zip if zip is not None else archive_path(file))
        picks.refuse_document_container(
            archive,
            "This node reads a zip archive holding text files. To read the text of a "
            "document, wire the document into Convert DOC to Plaintext instead.",
        )
        members, report = picks.read_matching(archive, pattern, kinds.TEXT)

        names: list[str] = []
        texts: list[str] = []
        for member in members:
            try:
                # utf-8-sig and one spelling of a line ending, so a byte order mark does not
                # become an invisible character at the front of the first line and a file
                # written on Windows reads as the same lines as one written anywhere else.
                texts.append(text_files.decode(member.data))
            except UnicodeDecodeError as error:
                report.skip(
                    picks.NOT_UTF8,
                    f"the entry {member.name!r} is not UTF-8 text: byte {error.start} of it "
                    f"is not valid UTF-8, so it was skipped rather than read with characters "
                    f"replaced. Save it as UTF-8 and zip it again.",
                )
                continue
            names.append(member.name)

        for note in report.notes:
            logger.warning("%s: %s", archive.label, note)
        logger.info(
            "Load Text Files from ZIP read %s: %s", archive.label, report.summary(len(texts))
        )
        require_values(texts, cls.nothing(archive, pattern, report))
        # Each slot gets a list of its own. The same list on two slots would let a node
        # that edits the LIST it was handed change how many times the graph under the
        # text output runs.
        return io.NodeOutput(
            texts, names, list(texts), list(names), len(texts), report.total
        )

    @staticmethod
    def nothing(archive, pattern: str, report: picks.Report) -> str:
        """The message for an archive that produced no text at all.

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
                f"the archive {archive.label} holds no files, so Load Text Files from ZIP "
                f"has nothing to hand on and the graph below it cannot be run. It opened "
                f"correctly and is simply empty."
            )
        if not report.matched:
            return (
                f"no entry in {archive.label} is picked by the pattern `{pattern}`, so Load "
                f"Text Files from ZIP has nothing to hand on and the graph below it cannot "
                f"be run. It holds {offered}.{left_out} A pattern with no '/' in it is "
                f"matched against the file's own name at any depth, so '*' or '*.txt' reads "
                f"them wherever they sit."
            )
        return (
            f"every entry the pattern `{pattern}` picked in {archive.label} was skipped, so "
            f"Load Text Files from ZIP has nothing to hand on and the graph below it cannot "
            f"be run: {report.summary(0)}. The log names each one. The extensions this node "
            f"reads are {kinds.extension_list(kinds.TEXT)}."
        )
