"""LLM Auto-Merging Retriever: hands on a larger chunk where enough of its parts are found."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_INDEX, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")


class LLMAutoMergingRetriever(io.ComfyNode):
    """A vector retriever merging sibling chunks into the chunk that holds them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMAutoMergingRetriever",
            display_name="LLM Auto-Merging Retriever",
            search_aliases=[
                "WASLLMAutoMergingRetriever",
                "LLM Auto-Merging Retriever",
                "AutoMergingRetriever",
                "hierarchical",
                "parent chunk",
                "small to big",
                "RAG",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Search a vector index of small chunks and, where enough pieces of one larger chunk "
                "are found, hand on that larger chunk instead, so answers read whole passages. "
                "Build the index from every chunk the hierarchical splitter writes."
            ),
            inputs=[
                LLM_INDEX.Input(
                    "index",
                    tooltip="A vector index built from hierarchical chunks, every level included, so its docstore holds the larger chunks.",
                ),
                io.Int.Input(
                    "top_k", default=12, min=1, max=200,
                    tooltip="Small chunks found before merging: 12 = the usual; more lets more parents merge. The merged result can be shorter.",
                ),
                io.Float.Input(
                    "merge_ratio", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Share of a larger chunk's pieces that must be found before it replaces them: 0.5 = more than half; 0.25 = merge eagerly; 0.9 = rarely.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The retriever, for LLM Retrieve, LLM Fusion Retriever, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, index, top_k, merge_ratio) -> io.NodeOutput:
        """Build the auto-merging retriever.

        Args:
            index: A vector index of hierarchical chunks.
            top_k: Small chunks found before merging.
            merge_ratio: Share of a parent's pieces that triggers the merge.

        Returns:
            The retriever.
        """
        node = "LLM Auto-Merging Retriever"
        index = wires.expect(index, wires.Index, node, "index", "LLM Vector Index built from hierarchical chunks")
        wires.ready(index)
        with runtime.guarded():
            retriever, facts = retrieval.auto_merging_retriever(index, top_k, merge_ratio)
        logger.info("LLM Auto-Merging Retriever built over %d chunk(s)", int(index.size))
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {retriever.label}, merging at {float(merge_ratio):.2f}.",
            counts={"chunks in index": int(index.size), "top_k": int(top_k)},
            facts=facts,
        )
        return io.NodeOutput(retriever)
