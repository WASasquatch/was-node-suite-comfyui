"""Summarize documents bottom up into a tree index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Tree Index"


class LLMTreeIndex(io.ComfyNode):
    """Summarize chunks into a tree a question walks down from the top."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMTreeIndex",
            display_name="LLM Tree Index",
            search_aliases=[
                "WASLLMTreeIndex",
                "LLM Tree Index",
                "TreeIndex",
                "tree index",
                "hierarchical summary",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Summarize documents bottom up into a tree, each parent a summary of a few chunks "
                "below it, so a question can walk down from the top. The connected language model "
                "writes every summary. The panel shows how many summaries were written and the "
                "top ones."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to index, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. Whole documents are cut into chunks of chunk_size tokens."
                    ),
                ),
                LLM.Input(
                    "llm",
                    tooltip=(
                        "The model that writes the summaries, such as LLM From CLIP, LLM GGUF "
                        "Loader or LLM OpenAI. Each parent costs one generation."
                    ),
                ),
                io.Int.Input(
                    "num_children", default=10, min=2, max=64,
                    tooltip=(
                        "Chunks or summaries each parent summarizes. 10 = a shallow tree and long "
                        "prompts; 3 = a deeper tree, more summaries, shorter prompts."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = more chunks and more summaries. Chunks from LLM Split "
                        "Documents keep their own size."
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
                    tooltip="The tree index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, llm, num_children, chunk_size=1024, chunk_overlap=20) -> io.NodeOutput:
        """Split the documents and summarize them into a tree.

        Args:
            documents: The documents or chunks.
            llm: The language model writing the summaries.
            num_children: Chunks or summaries per parent.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The tree index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI")
        wires.ready(model)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = indexes.build_tree(prepared, model, num_children)
        wire = wires.Index(index=index, kind="tree", llm=model, size=indexes.size_of(index, "tree"))
        seconds = time.perf_counter() - started
        written = indexes.tree_summaries(wire.size, num_children)
        logger.info(
            "%s summarized %d chunk(s) into %d summaries with %s in %.2fs",
            NODE, wire.size, written, model.label, seconds,
        )
        indexes.report(wire, seconds, f"summarized {wire.size} chunks with {model.label}", prepared)
        return io.NodeOutput(wire)
