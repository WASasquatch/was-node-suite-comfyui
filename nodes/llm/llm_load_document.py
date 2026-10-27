"""Read one document file onto an LLM documents wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS
from ...modules.llm import documents as loading
from ...modules.llm import wires
from ...modules.io import picker

REQUIRES = "llm"

logger = log.get_logger("llm.documents")


class LLMLoadDocument(io.ComfyNode):
    """One text, Markdown, HTML, code, PDF or word processor file, as documents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMLoadDocument",
            display_name="LLM Load Document",
            search_aliases=[
                "WASLLMLoadDocument",
                "LLM Load Document",
                "SimpleDirectoryReader",
                "PDFReader",
                "DocxReader",
                "load pdf",
                "read file",
                "rag document",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Read one file, such as a PDF, a Word document, an e-book, a slide deck or code, "
                "into documents for splitting, indexing or a prompt. A PDF can become one "
                "document per page. The panel shows what was read and a sample of the text."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file",
                    options=loading.file_options(),
                    tooltip=(
                        "Which file to read: every text, Markdown, HTML, code, .pdf, .docx and "
                        ".odt file in ComfyUI's input, output and temp folders, tagged `[input]`, "
                        "`[output]` or `[temp]`, and in folders added under paths.allow_read."
                    ),
                )),
                io.Boolean.Input(
                    "page_per_document",
                    default=False,
                    tooltip=(
                        "For a PDF: `true` = one document per page, each naming its page number, "
                        "so an answer can cite the page; `false` = the whole file as one document."
                    ),
                ),
                io.Boolean.Input(
                    "keep_html",
                    default=False,
                    tooltip=(
                        "`false` reads an .html file as text; `true` keeps its markup, for LLM "
                        "Split Documents set to html."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    optional=True,
                    tooltip=(
                        "Documents from another loader, such as another LLM Load Document. The "
                        "ones read here are added after them."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "The file as one document, or one per PDF page, naming its folder, path, "
                        "size and date, for LLM Split Documents or an LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", page_per_document=False, keep_html=False, documents=None):
        """The file's size and time, so a changed file is read again."""
        return loading.file_fingerprint(file)

    @classmethod
    def execute(cls, file, page_per_document, keep_html, documents=None) -> io.NodeOutput:
        """Read the picked file.

        Args:
            file: A label from the file menu.
            page_per_document: One document per PDF page.
            keep_html: Keep HTML markup.
            documents: Documents to add these after.

        Returns:
            The documents.

        Raises:
            ValueError: Nothing was picked, the file is gone, or it holds nothing to read.
            DependencyError: The package its format needs is missing.
            PathNotAllowed: The file lies outside every permitted read root.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(
                documents, wires.Documents, "LLM Load Document", "documents", "an LLM document loader"
            )
        found, root = loading.picked_file(file)
        try:
            made = loading.file_documents(
                found, root, loading.MAX_FILE_BYTES, bool(keep_html), bool(page_per_document)
            )
        except loading.Unreadable as error:
            raise ValueError(f"LLM Load Document could not read `{file}`: {error}.") from None
        added = wires.Documents(tuple(made), found.relative)
        result = previous.plus(added) if previous is not None else added
        logger.info(
            "LLM Load Document read %s (%s) into %d document(s)",
            found.relative, root, len(made),
        )
        facts = {"file": found.relative, "folder": root}
        pages = made[0].metadata.get("page_count") if made else None
        if pages:
            facts["pages"] = pages
        loading.publish(
            f"read {found.relative} into {len(made)} document(s)",
            result,
            made,
            formats={found.extension: 1},
            facts=facts,
        )
        return io.NodeOutput(result)
