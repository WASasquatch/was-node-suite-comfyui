"""Rewrite each question before another retriever searches with it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Transform Retriever"


class LLMTransformRetriever(io.ComfyNode):
    """A retriever that searches with a rewritten question."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        transforms = [
            io.DynamicCombo.Option(key="hyde", inputs=[
                io.Boolean.Input(
                    "include_original", default=True,
                    tooltip="`true` searches with the question and the written answer together; `false` with the answer alone.",
                ),
            ]),
            io.DynamicCombo.Option(key="decompose", inputs=[
                io.String.Input(
                    "summary", multiline=True, default="",
                    tooltip="What the documents hold, read when the question is simplified, such as `Camera manuals and studio rates.`",
                ),
            ]),
        ]
        return io.Schema(
            node_id="WASLLMTransformRetriever",
            display_name="LLM Transform Retriever",
            search_aliases=[
                "WASLLMTransformRetriever",
                "LLM Transform Retriever",
                "TransformRetriever",
                "HyDE",
                "hypothetical document",
                "query rewrite",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Rewrite each question before another retriever searches with it: HyDE has the model "
                "write a likely answer and searches with that, decompose asks a simpler question first. "
                "It helps where questions and documents are worded differently."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="The retriever that searches, such as LLM Index Retriever.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that rewrites, such as LLM From CLIP or LLM OpenAI.",
                ),
                io.DynamicCombo.Input(
                    "transform", options=transforms, display_name="Transform",
                    tooltip="`hyde` = search with a written answer; `decompose` = search with a simpler question.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The retriever, for LLM Retrieve, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, retriever, llm, transform) -> io.NodeOutput:
        """Build the retriever.

        Args:
            retriever: The retriever searching.
            llm: The model rewriting.
            transform: The transform and its settings.

        Returns:
            The retriever.
        """
        source = wires.expect(retriever, wires.Retriever, NODE, "retriever", "LLM Index Retriever")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP or LLM OpenAI")
        choice = transform if isinstance(transform, dict) else {"transform": str(transform)}
        wires.ready(source, model)
        with runtime.guarded():
            wire, facts = retrieval.transform_retriever(
                source, model, str(choice.get("transform", "hyde")),
                bool(choice.get("include_original", True)), str(choice.get("summary", "")),
            )
        logger.info("%s built %s", NODE, wire.label)
        run_result.publish(status=run_result.OK, summary=f"Retriever ready: {wire.label}.", facts=facts)
        return io.NodeOutput(wire)
