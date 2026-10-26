"""Keep documents as an ordered list of chunks in a summary index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Summary Index"


class LLMSummaryIndex(io.ComfyNode):
    """Keep documents as an ordered list of chunks, read whole when asked."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSummaryIndex",
            display_name="LLM Summary Index",
            search_aliases=[
                "WASLLMSummaryIndex",
                "LLM Summary Index",
                "SummaryIndex",
                "ListIndex",
                "list index",
                "summarize",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Keep documents as an ordered list of chunks, read in full or filtered when a "
                "question is asked, which suits summarizing a whole document. It needs no model "
                "to build. The panel shows the chunk count and the first chunks."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to index, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. Whole documents are cut into chunks of chunk_size tokens."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = finer pieces with less context each. Chunks from LLM "
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
                    tooltip="The summary index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, chunk_size=1024, chunk_overlap=20) -> io.NodeOutput:
        """Split and list the documents.

        Args:
            documents: The documents or chunks.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The summary index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = indexes.build_summary(prepared)
        wire = wires.Index(index=index, kind="summary", size=indexes.size_of(index, "summary"))
        seconds = time.perf_counter() - started
        logger.info("%s listed %d chunk(s) from %d item(s) in %.2fs", NODE, wire.size, prepared.handed, seconds)
        indexes.report(wire, seconds, f"listed {wire.size} chunks", prepared)
        return io.NodeOutput(wire)
