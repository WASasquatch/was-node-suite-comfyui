"""LLM Fusion Retriever: merges several retrievers' rankings into one."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: Tooltip on every extra retriever slot.
SLOT_TIP = "Another retriever to merge, such as LLM BM25 Retriever beside a vector one. A new slot appears as each is wired."


class LLMFusionRetriever(io.ComfyNode):
    """A retriever merging the rankings of several retrievers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMFusionRetriever",
            display_name="LLM Fusion Retriever",
            search_aliases=[
                "WASLLMFusionRetriever",
                "LLM Fusion Retriever",
                "QueryFusionRetriever",
                "hybrid search",
                "reciprocal rank fusion",
                "RRF",
                "multi query",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Merge the rankings of several retrievers into one, such as BM25 keywords with "
                "vector similarity. A language model can also rewrite the question into variations "
                "that every retriever searches. Connect the result to LLM Retrieve or a query engine."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="The first retriever to merge, from LLM Index Retriever, LLM BM25 Retriever or another fusion.",
                ),
                io.Combo.Input(
                    "mode", options=list(retrieval.FUSION_MODES), default="reciprocal_rerank",
                    tooltip=(
                        "How rankings merge. reciprocal_rerank = by rank position, steady across score "
                        "scales; relative_score = scores scaled 0 to 1 per retriever, then weighted; "
                        "dist_based_score = scaled by their spread; simple = each chunk's best score."
                    ),
                ),
                io.Int.Input(
                    "top_k", default=5, min=1, max=200,
                    tooltip="Chunks the merged ranking hands on: 5 = a short context; 20 = broad recall for a reranker.",
                ),
                io.Int.Input(
                    "num_queries", default=1, min=1, max=10,
                    tooltip="Questions searched: 1 = only the question as asked; 4 = the question plus three variations the llm writes.",
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip="The language model that writes question variations. Needed only when num_queries is above 1.",
                ),
                io.String.Input(
                    "weights", default="", optional=True,
                    tooltip="One weight per connected retriever in slot order, such as `0.6, 0.4`; blank = equal. Read by relative_score and dist_based_score.",
                ),
                LLM_RETRIEVER.Input("retriever_b", optional=True, tooltip=SLOT_TIP),
                LLM_RETRIEVER.Input("retriever_c", optional=True, tooltip=SLOT_TIP),
                LLM_RETRIEVER.Input("retriever_d", optional=True, tooltip=SLOT_TIP),
                LLM_RETRIEVER.Input("retriever_e", optional=True, tooltip=SLOT_TIP),
                LLM_RETRIEVER.Input("retriever_f", optional=True, tooltip=SLOT_TIP),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The merged retriever, for LLM Retrieve, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, retriever, mode, top_k, num_queries, llm=None, weights="", retriever_b=None,
                retriever_c=None, retriever_d=None, retriever_e=None, retriever_f=None) -> io.NodeOutput:
        """Build the fusion retriever.

        Args:
            retriever: The first retriever.
            mode: How rankings merge.
            top_k: Chunks handed on.
            num_queries: Questions searched.
            llm: The model writing question variations.
            weights: Per-retriever weights.
            retriever_b: Another retriever.
            retriever_c: Another retriever.
            retriever_d: Another retriever.
            retriever_e: Another retriever.
            retriever_f: Another retriever.

        Returns:
            The merged retriever.
        """
        node = "LLM Fusion Retriever"
        slots = {
            "retriever": retriever, "retriever_b": retriever_b, "retriever_c": retriever_c,
            "retriever_d": retriever_d, "retriever_e": retriever_e, "retriever_f": retriever_f,
        }
        wires.expect(retriever, wires.Retriever, node, "retriever", "LLM Index Retriever or LLM BM25 Retriever")
        connected = []
        for name, value in slots.items():
            if value is not None:
                connected.append(wires.expect(value, wires.Retriever, node, name, "an LLM retriever node"))
        if llm is not None:
            wires.expect(llm, wires.LanguageModel, node, "llm", "an LLM model node")
        wires.ready(*connected, llm if int(num_queries) > 1 else None)
        with runtime.guarded():
            fused, facts = retrieval.fusion_retriever(connected, mode, top_k, num_queries, llm, weights)
        if llm is not None and int(num_queries) > 1:
            facts["llm"] = llm.label[:64]
        logger.info("LLM Fusion Retriever merges %d retriever(s) by %s", len(connected), mode)
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {fused.label}.",
            counts={"retrievers": len(connected), "top_k": int(top_k), "questions": int(num_queries)},
            facts=facts,
        )
        return io.NodeOutput(fused)
