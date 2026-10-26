"""Rewrite each question before a query engine answers it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_QUERY_ENGINE
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")


class WASLLMQueryTransform(io.ComfyNode):
    """A query engine that rewrites the question, then asks another engine."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        transforms = [
            io.DynamicCombo.Option(
                key="hyde",
                inputs=[
                    io.Boolean.Input(
                        "include_original", default=True,
                        tooltip="`true` searches with the imagined answer and the question together; `false` with the imagined answer alone.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="decompose",
                inputs=[
                    io.Int.Input(
                        "num_steps", default=3, min=1, max=10,
                        tooltip="The most follow-up questions asked before the final answer. `1` asks once, `3` lets the model work through a question in parts.",
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMQueryTransform",
            display_name="LLM Query Transform",
            search_aliases=[
                "WASLLMQueryTransform",
                "LLM Query Transform",
                "TransformQueryEngine",
                "HyDEQueryTransform",
                "MultiStepQueryEngine",
                "StepDecomposeQueryTransform",
                "HyDE",
                "multi step",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Wrap a query engine so each question is rewritten before it is answered. HyDE "
                "searches with an imagined answer, which finds passages a short question misses; "
                "decompose breaks a hard question into follow-up questions asked one after another."
            ),
            inputs=[
                LLM_QUERY_ENGINE.Input(
                    "query_engine",
                    tooltip="The engine that answers the rewritten question, such as from LLM Query Engine.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that rewrites the question, often the same one the engine answers with.",
                ),
                io.DynamicCombo.Input(
                    "transform", options=transforms, display_name="Transform",
                    tooltip="`hyde` = search with an imagined answer to the question; `decompose` = ask a chain of simpler follow-up questions, then answer from all of them.",
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The wrapped query engine, for LLM Query or LLM Query Engine Tool.",
                ),
            ],
        )

    @classmethod
    def execute(cls, query_engine, llm, transform) -> io.NodeOutput:
        """Wrap the engine.

        Args:
            query_engine: The engine answering the rewritten question.
            llm: The model rewriting it.
            transform: The transform choice and its settings.

        Returns:
            The wrapped query engine.
        """
        node = "LLM Query Transform"
        inner = wires.expect(query_engine, wires.QueryEngine, node, "query_engine", "LLM Query Engine")
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        choice = dict(transform or {})
        kind = choice.get("transform", "hyde")
        if kind not in ("hyde", "decompose"):
            raise ValueError(f"{node}: transform {str(kind)[:40]!r} is not hyde or decompose.")
        wires.ready(inner, model)
        with runtime.guarded():
            try:
                engine, label = query.build_transform(inner, model, choice)
            except Exception as error:
                raise query.failure(node, error) from None
        facts = {"transform": kind, "model": model.label, "engine": inner.label}
        if kind == "hyde":
            facts["original question"] = "searched too" if choice.get("include_original", True) else "left out"
        else:
            facts["steps"] = str(int(choice.get("num_steps", 3)))
        logger.info("LLM Query Transform wrapped an engine with %s", kind)
        run_result.publish(status=run_result.OK, summary=label, facts=facts)
        return io.NodeOutput(wires.QueryEngine(engine=engine, label=label, depends=(inner, model)))
