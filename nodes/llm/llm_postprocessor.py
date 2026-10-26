"""LLM Postprocessor: one step that filters, reorders or rewrites retrieved chunks."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_EMBEDDING, LLM_INDEX, LLM_POSTPROCESSORS
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

#: Tooltip on every top_n setting.
TOP_N_TIP = "Chunks kept, best first: 5 = a short context; 1 = only the best."


class LLMPostprocessor(io.ComfyNode):
    """Adds one postprocessor step to the chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        kinds = [
            io.DynamicCombo.Option(key="similarity cutoff", inputs=[
                io.Float.Input(
                    "cutoff", default=0.5, min=-1000.0, max=1000.0, step=0.01,
                    tooltip="Chunks scoring below this are dropped. Vector similarity and LLM Rerank score about 0 to 1; BM25 scores run higher, often 2 to 20.",
                ),
            ]),
            io.DynamicCombo.Option(key="keywords", inputs=[
                io.String.Input(
                    "required", default="",
                    tooltip="Words or phrases every kept chunk holds, separated by commas, such as `warranty, refund`. Case is ignored.",
                ),
                io.String.Input(
                    "excluded", default="",
                    tooltip="Words or phrases that drop a chunk holding any of them, separated by commas, such as `draft, internal`.",
                ),
            ]),
            io.DynamicCombo.Option(key="long context reorder", inputs=[]),
            io.DynamicCombo.Option(key="sentence window", inputs=[
                io.String.Input(
                    "metadata_key", default="window",
                    tooltip="The metadata key holding each chunk's surrounding sentences, `window` from the sentence window splitter. Chunks without it keep their text.",
                ),
            ]),
            io.DynamicCombo.Option(key="previous and next", inputs=[
                io.Int.Input(
                    "count", default=1, min=1, max=20,
                    tooltip="Neighbouring chunks added on each side asked for: 1 = the adjacent one; 3 = three in a row.",
                ),
                io.Combo.Input(
                    "direction", options=["next", "previous", "both", "auto"], default="next",
                    tooltip="next = what follows each chunk; previous = what leads into it; both; auto = the llm decides per chunk from the question.",
                ),
            ]),
            io.DynamicCombo.Option(key="recency", inputs=[
                io.String.Input(
                    "date_key", default="last_modified_date",
                    tooltip="The metadata key holding a date such as `2026-03-14`, `last_modified_date` from a file loader. Undated chunks go last.",
                ),
                io.Int.Input("top_n", default=5, min=1, max=200, tooltip=TOP_N_TIP),
            ]),
            io.DynamicCombo.Option(key="rerank with llm", inputs=[
                io.Int.Input("top_n", default=5, min=1, max=200, tooltip=TOP_N_TIP),
                io.Int.Input(
                    "batch_size", default=10, min=1, max=50,
                    tooltip="Chunks shown to the language model in one prompt while it scores them: 10 = the usual; smaller suits a short context window.",
                ),
            ]),
            io.DynamicCombo.Option(key="rerank with embedding", inputs=[
                io.Int.Input("top_n", default=5, min=1, max=200, tooltip=TOP_N_TIP),
            ]),
            io.DynamicCombo.Option(key="sentence optimizer", inputs=[
                io.Float.Input(
                    "percentile", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Share of each chunk's sentences kept, closest to the question first: 0.5 = half; 0 = off, threshold alone decides.",
                ),
                io.Float.Input(
                    "threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Sentences less similar to the question than this are dropped: 0 = off; 0.7 = only close matches.",
                ),
                io.Int.Input(
                    "context_before", default=1, min=0, max=10,
                    tooltip="Sentences kept before each one chosen, so it reads in context: 0 = none.",
                ),
                io.Int.Input(
                    "context_after", default=1, min=0, max=10,
                    tooltip="Sentences kept after each one chosen: 0 = none.",
                ),
            ]),
        ]
        return io.Schema(
            node_id="WASLLMPostprocessor",
            display_name="LLM Postprocessor",
            search_aliases=[
                "WASLLMPostprocessor",
                "LLM Postprocessor",
                "SimilarityPostprocessor",
                "LLMRerank",
                "LongContextReorder",
                "MetadataReplacementPostProcessor",
                "PrevNextNodePostprocessor",
                "SentenceEmbeddingOptimizer",
                "node postprocessor",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Add a step that filters, reorders or rewrites retrieved chunks before they reach a "
                "prompt: a score cutoff, required or banned keywords, newest first, neighbouring "
                "chunks, sentence windows, or a rerank by a language or embedding model. Steps chain "
                "and run in order."
            ),
            inputs=[
                LLM_POSTPROCESSORS.Input(
                    "postprocessors", optional=True,
                    tooltip="Steps that run before this one, from another LLM Postprocessor or LLM Rerank.",
                ),
                io.DynamicCombo.Input(
                    "kind", options=kinds, display_name="Kind",
                    tooltip=(
                        "What the step does. `similarity cutoff`, `keywords`, `long context reorder`, "
                        "`sentence window` and `recency` read the chunks alone; the reranks, `sentence "
                        "optimizer` and `auto` neighbours call a model; `previous and next` reads the "
                        "index or documents the chunks came from."
                    ),
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip="The language model for rerank with llm and for previous and next set to auto.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="The embedding model for rerank with embedding and the sentence optimizer.",
                ),
                LLM_INDEX.Input(
                    "index", optional=True,
                    tooltip="The index the chunks come from: previous and next reads its stored chunks, and its models stand in for llm and embedding.",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="The split chunks the retrieved ones come from, for previous and next when no index is connected.",
                ),
            ],
            outputs=[
                LLM_POSTPROCESSORS.Output(
                    display_name="postprocessors",
                    tooltip="The chain with this step added, for LLM Retrieve, a query engine, a chat engine or the next step.",
                ),
            ],
        )

    @classmethod
    def execute(cls, kind, postprocessors=None, llm=None, embedding=None, index=None,
                documents=None) -> io.NodeOutput:
        """Build the step and add it to the chain.

        Args:
            kind: The kind and its settings.
            postprocessors: Steps before this one.
            llm: A language model.
            embedding: An embedding model.
            index: The index the chunks come from.
            documents: The chunks the retrieved ones come from.

        Returns:
            The chain with this step appended.
        """
        node = "LLM Postprocessor"
        if postprocessors is not None and not (
            isinstance(postprocessors, (tuple, list)) and wires.every(postprocessors, wires.Postprocessor)
        ):
            raise ValueError(f"{node}: postprocessors carries something other than postprocessor steps. Connect LLM Postprocessor or LLM Rerank.")
        if llm is not None:
            wires.expect(llm, wires.LanguageModel, node, "llm", "an LLM model node")
        if embedding is not None:
            wires.expect(embedding, wires.EmbeddingModel, node, "embedding", "LLM Embedding Model")
        if index is not None:
            wires.expect(index, wires.Index, node, "index", "an LLM index node")
        if documents is not None:
            wires.expect(documents, wires.Documents, node, "documents", "an LLM splitter")
        with runtime.guarded():
            step, facts = retrieval.postprocessor(kind, llm=llm, embedding=embedding, index=index, documents=documents)
        wires.ready(postprocessors, step)
        chain = wires.chain(postprocessors, step)
        run_result.publish(
            status=run_result.OK,
            summary=f"Step {len(chain)} of the chain: {step.label}.",
            counts={"steps": len(chain)},
            facts=facts,
        )
        return io.NodeOutput(chain)
