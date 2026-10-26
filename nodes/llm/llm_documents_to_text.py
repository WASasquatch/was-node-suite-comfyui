"""Join documents into one string for a prompt."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import documents as loading
from ...modules.llm import wires

REQUIRES = "llm"

logger = log.get_logger("llm.documents")

#: Characters of the joined text a run result carries.
SAMPLE_CHARS = 6000


class LLMDocumentsToText(io.ComfyNode):
    """Documents or chunks joined into one string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMDocumentsToText",
            display_name="LLM Documents to Text",
            search_aliases=[
                "WASLLMDocumentsToText",
                "LLM Documents to Text",
                "documents to string",
                "join documents",
                "stuff prompt",
                "context text",
            ],
            category="WAS Suite/LLM/Documents",
            is_output_node=True,
            description=(
                "Join documents or chunks into one piece of text, to hand whole documents to a "
                "prompt or save them. The panel shows how many were joined and the start of the "
                "text."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="The documents or chunks to join, from an LLM document loader or LLM Split Documents.",
                ),
                io.String.Input(
                    "separator",
                    default="\\n\\n",
                    tooltip=(
                        "Written between two documents. `\\n\\n` = a blank line; `\\n---\\n` = a "
                        "rule; type \\n for a line break and \\t for a tab."
                    ),
                ),
                io.Boolean.Input(
                    "include_metadata",
                    default=False,
                    tooltip=(
                        "`true` writes each document's title, file name and other fields above "
                        "its text, as a model reads them; `false` = the text alone."
                    ),
                ),
                io.Int.Input(
                    "max_chars",
                    default=100000,
                    min=1,
                    max=10000000,
                    tooltip=(
                        "Most characters of text handed on; the rest is cut. `100000` is about "
                        "25000 tokens, inside most models' context."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The documents joined, for a prompt input such as LLM Complete's or a text preview.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many documents the text holds, the last one possibly cut by max_chars.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, separator, include_metadata, max_chars) -> io.NodeOutput:
        """Join the documents.

        Args:
            documents: The documents.
            separator: What goes between two documents, with ``\\n`` and ``\\t`` read as escapes.
            include_metadata: Write metadata above each text.
            max_chars: Most characters handed on.

        Returns:
            The text and how many documents it holds.

        Raises:
            ValueError: The documents input carries something else.
        """
        source = wires.expect(
            documents, wires.Documents, "LLM Documents to Text", "documents", "LLM Load Documents"
        )
        between = str(separator or "").replace("\\n", "\n").replace("\\t", "\t")
        text, count, cut = loading.to_text(source.items, between, bool(include_metadata), int(max_chars))
        logger.info(
            "LLM Documents to Text joined %d of %d document(s) into %d character(s)",
            count, len(source.items), len(text),
        )
        run_result.publish(
            status=run_result.WARNING if cut else run_result.OK,
            summary=f"joined {count} of {len(source.items)} document(s)" + (" (cut at max_chars)" if cut else ""),
            counts={"documents": count, "characters": len(text)},
            facts={"metadata": "included" if include_metadata else "left out"},
            bodies=[run_result.body("text", text[:SAMPLE_CHARS])] if text else [],
        )
        return io.NodeOutput(text, count)
