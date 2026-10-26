"""Summarize each document and keep the summary beside its chunks in a document summary index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_EMBEDDING, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Document Summary Index"


class LLMDocumentSummaryIndex(io.ComfyNode):
    """Write one summary per document, for picking documents before reading chunks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMDocumentSummaryIndex",
            display_name="LLM Document Summary Index",
            search_aliases=[
                "WASLLMDocumentSummaryIndex",
                "LLM Document Summary Index",
                "DocumentSummaryIndex",
                "document summary",
                "summaries",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Write one summary per document and keep it beside the document's chunks, so a "
                "question first picks the documents that fit and then reads their chunks. Connect "
                "an embedding model to pick documents by meaning. The panel shows each summary."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to index, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. Each source document gets one summary."
                    ),
                ),
                LLM.Input(
                    "llm",
                    tooltip=(
                        "The model that writes the summaries, such as LLM From CLIP, LLM GGUF "
                        "Loader or LLM OpenAI. A long document costs several generations."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip=(
                        "A model that embeds each summary, such as from LLM Embedding Model, so "
                        "documents are picked by meaning. Left empty, the language model picks them."
                    ),
                ),
                io.String.Input(
                    "summary_query", default="", multiline=True, optional=True, advanced=True,
                    placeholder="Eg: List the people, places and dates this text mentions.",
                    tooltip=(
                        "What each summary is asked to say, such as `List the people, places and "
                        "dates it mentions.` Empty asks what the document is about and which "
                        "questions it can answer."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = finer matches with less context each. Chunks from LLM "
                        "Split Documents keep their own size."
                    ),
                ),
                io.Int.Input(
                    "chunk_overlap", default=20, min=0, max=2048, optional=True, advanced=True,
                    tooltip=(
                        "Tokens two neighbouring chunks share, so a sentence cut at a border is "
                        "still found whole. 20 = a little; 0 = none. Smaller than chunk_size."
                    ),
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The document summary index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, documents, llm, embedding=None, summary_query="", chunk_size=1024, chunk_overlap=20,
    ) -> io.NodeOutput:
        """Split the documents and summarize each one.

        Args:
            documents: The documents or chunks.
            llm: The language model writing the summaries.
            embedding: The embedding model, or ``None``.
            summary_query: What each summary is asked to say, empty for the default.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The document summary index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI")
        embed = None
        if embedding is not None:
            embed = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        wires.ready(model, embed)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = indexes.build_document_summary(prepared, model, embed, summary_query, NODE)
        wire = wires.Index(
            index=index, kind="document_summary", embedding=embed, llm=model,
            size=indexes.size_of(index, "document_summary"),
        )
        seconds = time.perf_counter() - started
        summaries = len(index.index_struct.doc_id_to_summary_id)
        logger.info(
            "%s wrote %d summaries over %d chunk(s) with %s in %.2fs",
            NODE, summaries, wire.size, model.label, seconds,
        )
        facts = {"summary prompt": "custom" if str(summary_query or "").strip() else "default"}
        indexes.report(wire, seconds, f"summarized {summaries} documents with {model.label}", prepared, facts=facts)
        return io.NodeOutput(wire)
