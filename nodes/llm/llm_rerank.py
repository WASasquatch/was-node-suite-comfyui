"""LLM Rerank: reorders retrieved chunks with a cross-encoder run on this machine."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM_POSTPROCESSORS
from ...modules.interface import run_result
from ...modules.llm import rerank, runtime, wires

REQUIRES = "llm"


class LLMRerank(io.ComfyNode):
    """Adds a cross-encoder rerank step to the chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        choices = rerank.choices()
        return io.Schema(
            node_id="WASLLMRerank",
            display_name="LLM Rerank",
            search_aliases=[
                "WASLLMRerank",
                "LLM Rerank",
                "cross-encoder",
                "SentenceTransformerRerank",
                "bge-reranker",
                "reranker",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Reorder retrieved chunks by how well each one answers the question, read by a "
                "cross-encoder reranker on this machine, and keep the best. Retrieve broadly, then "
                "rerank to a few."
            ),
            inputs=[
                LLM_POSTPROCESSORS.Input(
                    "postprocessors", optional=True,
                    tooltip="Steps that run before the rerank, from LLM Postprocessor or another LLM Rerank.",
                ),
                io.Combo.Input(
                    "model", options=choices, default=choices[0],
                    tooltip=(
                        "A reranker folder in ComfyUI/models/LLM holding config.json and "
                        "model.safetensors, or a suggested one, downloaded on first use only with "
                        "features.network on. bge-reranker-v2-m3 reads many languages; "
                        "ms-marco-MiniLM is small and fast, English."
                    ),
                ),
                io.Int.Input(
                    "top_n", default=5, min=1, max=200,
                    tooltip="Chunks kept after reranking, best first: 5 = a short context; 1 = only the best.",
                ),
                io.Int.Input(
                    "batch_size", default=16, min=1, max=256,
                    tooltip="Question and chunk pairs scored at once: 16 = the usual; lower it when memory runs short.",
                ),
                io.Int.Input(
                    "max_length", default=512, min=16, max=8192,
                    tooltip="Most tokens of one question and chunk pair read; the rest is cut. 512 suits most rerankers; bge-reranker-v2-m3 reads up to 8192.",
                ),
            ],
            outputs=[
                LLM_POSTPROCESSORS.Output(
                    display_name="postprocessors",
                    tooltip="The chain with the rerank added, for LLM Retrieve, a query engine, a chat engine or the next step.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, top_n, batch_size, max_length, postprocessors=None) -> io.NodeOutput:
        """Load the reranker and add its step to the chain.

        Args:
            model: The reranker choice.
            top_n: Chunks kept.
            batch_size: Pairs scored at once.
            max_length: Most tokens of one pair.
            postprocessors: Steps before this one.

        Returns:
            The chain with the rerank appended.
        """
        node = "LLM Rerank"
        if postprocessors is not None and not (
            isinstance(postprocessors, (tuple, list)) and wires.every(postprocessors, wires.Postprocessor)
        ):
            raise ValueError(f"{node}: postprocessors carries something other than postprocessor steps. Connect LLM Postprocessor or LLM Rerank.")
        wires.ready(postprocessors)
        loaded = rerank.load(model)
        runtime.check_interrupt()
        processor = rerank.reranker(loaded, top_n, batch_size, max_length)
        label = f"rerank {loaded.label} top {int(top_n)}"
        chain = wires.chain(postprocessors, wires.Postprocessor(processor=processor, label=label, depends=()))
        run_result.publish(
            status=run_result.OK,
            summary=f"Step {len(chain)} of the chain: {label}.",
            counts={"steps": len(chain), "top_n": int(top_n)},
            facts={
                "model": loaded.label[:64],
                "parameters": f"{loaded.parameters / 1e6:.1f}M",
                "reads": f"{min(int(max_length), loaded.max_length)} tokens",
                "batch": int(batch_size),
            },
        )
        return io.NodeOutput(chain)
