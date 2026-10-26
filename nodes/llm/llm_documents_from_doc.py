"""Turn a DOC document into documents on an LLM documents wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DOC, LLM_DOCUMENTS
from ...modules.document import container
from ...modules.llm import documents as loading
from ...modules.llm import wires

REQUIRES = "llm"

logger = log.get_logger("llm.documents")


class LLMDocumentsFromDOC(io.ComfyNode):
    """A DOC's text as one document, or one document per heading."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMDocumentsFromDOC",
            display_name="LLM Documents From DOC",
            search_aliases=[
                "WASLLMDocumentsFromDOC",
                "LLM Documents From DOC",
                "Document",
                "doc to documents",
                "wasdoc",
                "rag doc",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Turn a DOC, such as one from Load Document or Text to DOC, into documents for "
                "splitting, indexing or a prompt, carrying its title, author and dates. It can "
                "make one document per heading."
            ),
            inputs=[
                DOC.Input(
                    "doc",
                    tooltip="The document to read, from Load Document, Text to DOC or Load Documents from ZIP.",
                ),
                io.Boolean.Input(
                    "split_sections",
                    default=False,
                    tooltip=(
                        "`false` = the whole DOC as one document; `true` = one document per "
                        "heading, each naming its heading and level, and one for any text before "
                        "the first heading."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    optional=True,
                    tooltip=(
                        "Documents from another loader. The ones made here are added after them."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "The DOC as documents, carrying its title, author, description, language, "
                        "keywords and dates, for LLM Split Documents or an LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, doc, split_sections, documents=None) -> io.NodeOutput:
        """Read the DOC's text into documents.

        Args:
            doc: The DOC.
            split_sections: One document per heading.
            documents: Documents to add these after.

        Returns:
            The documents.

        Raises:
            ValueError: The doc input carries no DOC, or the DOC holds no text.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(
                documents, wires.Documents, "LLM Documents From DOC", "documents", "an LLM document loader"
            )
        source = wires.expect(
            doc, container.Document, "LLM Documents From DOC", "doc", "Load Document or Text to DOC"
        )
        made = loading.doc_documents(source, bool(split_sections))
        if not made:
            raise ValueError(
                "LLM Documents From DOC found no text in the DOC. Connect a document that holds "
                "writing, or check it with Convert DOC to Plaintext."
            )
        added = wires.Documents(tuple(made), source.metadata.title or "DOC")
        result = previous.plus(added) if previous is not None else added
        logger.info("LLM Documents From DOC made %d document(s)", len(made))
        facts = {"title": source.metadata.title or "untitled"}
        if source.metadata.author:
            facts["author"] = source.metadata.author
        facts["mode"] = "one per heading" if split_sections else "whole"
        loading.publish(
            f"made {len(made)} document(s) from {source.metadata.title or 'the DOC'}",
            result,
            made,
            formats={"doc": 1},
            facts=facts,
        )
        return io.NodeOutput(result)
