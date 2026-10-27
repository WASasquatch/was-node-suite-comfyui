"""Read every document in a folder onto an LLM documents wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import documents as loading
from ...modules.llm import wires
from ...modules.util import sandbox

REQUIRES = "llm"

logger = log.get_logger("llm.documents")


class LLMLoadDocuments(io.ComfyNode):
    """Every text, Markdown, HTML, code, PDF and word processor file in a folder, as documents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMLoadDocuments",
            display_name="LLM Load Documents",
            search_aliases=[
                "WASLLMLoadDocuments",
                "LLM Load Documents",
                "SimpleDirectoryReader",
                "load folder",
                "read documents",
                "rag documents",
                "pdf folder",
                "knowledge base",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Read every matching file in a folder, such as notes, PDFs, Word documents, e-books, "
                "slide decks and code, into documents for splitting and indexing. Files that cannot be read are "
                "counted and passed over. The panel shows the formats read and a sample of the "
                "text."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "Which folder to read. `input` = ComfyUI's input folder itself; `notes "
                        "[input]` = the notes folder inside it. A folder added under "
                        "paths.allow_read in config.yaml is listed under its own name."
                    ),
                ), folder=True),
                io.String.Input(
                    "extensions",
                    default=(
                        ".txt, .md, .rst, .html, .htm, .csv, .tsv, .json, .jsonl, .yaml, .yml, "
                        ".xml, .py, .js, .ts, .c, .cpp, .h, .java, .cs, .go, .rs, .sh, .sql, "
                        ".pdf, .docx, .odt, .epub, .pptx"
                    ),
                    tooltip=(
                        "Which files to read, separated by commas: `.md, .txt` = notes only; "
                        "`.pdf` = PDFs only. An extension with no reader of its own is read as "
                        "text, and a file holding binary data is passed over."
                    ),
                ),
                io.Boolean.Input(
                    "recursive",
                    default=True,
                    tooltip=(
                        "`true` reads the folders inside it too, as `week 1/monday.md`; `false` "
                        "reads only the files directly in it. Links and junctions are not followed."
                    ),
                ),
                io.Int.Input(
                    "max_files",
                    default=500,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Most files read, taken in path order. `500` covers a notes folder; the "
                        "files past it are counted as skipped."
                    ),
                ),
                io.Int.Input(
                    "max_file_mb",
                    default=32,
                    min=1,
                    max=512,
                    tooltip=(
                        "Largest file read, in megabytes. A larger file is counted as skipped. "
                        "`32` reads most books and reports."
                    ),
                ),
                io.Boolean.Input(
                    "keep_html",
                    default=False,
                    tooltip=(
                        "`false` reads .html files as text; `true` keeps their markup, for LLM "
                        "Split Documents set to html."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    optional=True,
                    tooltip=(
                        "Documents from another loader, such as LLM Load Web Page. The ones read "
                        "here are added after them."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "One document per file, naming its folder, path, size and date, for LLM "
                        "Split Documents or an LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", extensions="", recursive=True, max_files=500,
                           max_file_mb=32, keep_html=False, documents=None):
        """A digest of the files the folder holds, so a changed folder is read again."""
        try:
            root = loading.folder_root(folder)
            wanted = loading.parse_extensions(extensions)
        except (ValueError, sandbox.PathNotAllowed):
            return float("NaN")
        return loading.fingerprint_folder(root, wanted, bool(recursive), int(max_files))

    @classmethod
    def execute(cls, folder, extensions, recursive, max_files, max_file_mb, keep_html,
                documents=None) -> io.NodeOutput:
        """Walk the folder and read each matching file.

        Args:
            folder: A label from the folder menu.
            extensions: The extensions to read.
            recursive: Read subfolders too.
            max_files: Most files read.
            max_file_mb: Largest file read, in megabytes.
            keep_html: Keep HTML markup.
            documents: Documents to add these after.

        Returns:
            The documents.

        Raises:
            ValueError: The folder names nothing, no file matched, or nothing could be read.
            PathNotAllowed: The folder lies outside every permitted read root.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(
                documents, wires.Documents, "LLM Load Documents", "documents", "an LLM document loader"
            )
        wanted = loading.parse_extensions(extensions)
        root = loading.folder_root(folder)
        if root is None or not root.is_dir():
            raise ValueError(
                f"`{folder}` names no folder LLM Load Documents can read. Pick another from the "
                f"menu, or add its folder to paths.allow_read in config.yaml."
            )
        loaded = loading.load_folder(
            root, folder, wanted, bool(recursive), int(max_files),
            int(max_file_mb) * 1024 * 1024, bool(keep_html),
        )
        if not loaded.matched:
            raise ValueError(
                f"LLM Load Documents found no file in {folder} ending in "
                f"{', '.join(wanted[:8])}{' and more' if len(wanted) > 8 else ''}"
                f"{'' if recursive else ', and recursive is false'}. Pick another folder or add "
                f"the extension the files carry."
            )
        if not loaded.items:
            reasons = ", ".join(f"{reason} {count}" for reason, count in loaded.skipped.most_common())
            raise ValueError(
                f"LLM Load Documents read nothing from {folder}: {loaded.matched} matching "
                f"file(s), passed over as {reasons}. Raise max_file_mb for large files; a PDF "
                f"needs pypdf, a .docx python-docx and an .odt odfdo."
            )
        added = wires.Documents(tuple(loaded.items), f"{loaded.read} files from {folder}")
        result = previous.plus(added) if previous is not None else added
        logger.info(
            "LLM Load Documents read %d of %d file(s) from %s into %d document(s), %d skipped",
            loaded.read, loaded.matched, folder, len(loaded.items), sum(loaded.skipped.values()),
        )
        loading.publish(
            f"read {loaded.read} of {loaded.matched} file(s) from {folder}",
            result,
            loaded.items,
            formats=loaded.formats,
            skipped=loaded.skipped,
            facts={"folder": folder},
            status=run_result.WARNING if loaded.skipped else run_result.OK,
        )
        return io.NodeOutput(result)
