"""Turn typed or wired text into a document on an LLM documents wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DICT, LLM_DOCUMENTS
from ...modules.llm import documents as loading
from ...modules.llm import wires

REQUIRES = "llm"

logger = log.get_logger("llm.documents")


class LLMTextToDocument(io.ComfyNode):
    """One document made from text, with a title and metadata."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMTextToDocument",
            display_name="LLM Text to Document",
            search_aliases=[
                "WASLLMTextToDocument",
                "LLM Text to Document",
                "Document",
                "text document",
                "notes to document",
                "rag text",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Make a document from text, such as notes, a transcript or another node's "
                "output, with a title and metadata, for splitting, indexing or a prompt. Chain "
                "several to build a set."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip=(
                        "What the document holds, such as `The lighthouse keeper logs the "
                        "weather at dawn.` Wire a STRING here to make a document from another "
                        "node's text."
                    ),
                ),
                io.String.Input(
                    "title",
                    default="",
                    tooltip=(
                        "The document's title, such as `Keeper's log`, shown on answers that "
                        "cite it. Empty = no title."
                    ),
                ),
                DICT.Input(
                    "metadata",
                    optional=True,
                    tooltip=(
                        "Extra fields carried on the document and its chunks, such as "
                        "`{\"author\": \"Ada\", \"year\": 2026}` from a dictionary node, for a "
                        "retriever to filter on."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    optional=True,
                    tooltip=(
                        "Documents from another loader or another LLM Text to Document. This "
                        "one is added after them."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "The incoming documents followed by this one, for LLM Split Documents or "
                        "an LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, title, metadata=None, documents=None) -> io.NodeOutput:
        """Make the document and add it to the chain.

        Args:
            text: The document's text.
            title: Its title.
            metadata: Extra fields from a DICT wire.
            documents: Documents to add this one after.

        Returns:
            The documents.

        Raises:
            ValueError: The text is empty, or a wire carries something else.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(
                documents, wires.Documents, "LLM Text to Document", "documents", "an LLM document loader"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(
                f"LLM Text to Document was handed a {type(metadata).__name__} on its metadata "
                f"input. Connect a node with a DICT output to it."
            )
        made = loading.text_document(text, title, metadata)
        added = wires.Documents((made,), made.metadata.get("title", "") or "text")
        result = previous.plus(added) if previous is not None else added
        logger.info("LLM Text to Document made a document of %d character(s)", len(made.text))
        loading.publish(
            f"made a document of {len(made.text)} character(s)",
            result,
            [made],
            facts={"fields": len(made.metadata)},
        )
        return io.NodeOutput(result)
