"""Answer a question by splitting it into sub-questions for several query engines."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_QUERY_ENGINE, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")


class WASLLMSubQuestionQueryEngine(io.ComfyNode):
    """A query engine that asks each of its tools the part of a question it can answer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSubQuestionQueryEngine",
            display_name="LLM Sub-Question Query Engine",
            search_aliases=[
                "WASLLMSubQuestionQueryEngine",
                "LLM Sub-Question Query Engine",
                "SubQuestionQueryEngine",
                "sub question",
                "compare documents",
                "multi document",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Answer a question that spans several sources, such as comparing two reports. "
                "The model splits it into sub-questions, sends each to the query engine tool it "
                "suits, then writes one answer from all the replies."
            ),
            inputs=[
                LLM_TOOLS.Input(
                    "tools",
                    tooltip="Query engine tools to send sub-questions to, chained from LLM Query Engine Tool. Each tool's name and description tell the model what it knows.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the sub-questions and the final answer, such as LLM OpenAI or LLM From CLIP.",
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The sub-question engine, for LLM Query or LLM Query Engine Tool.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tools, llm) -> io.NodeOutput:
        """Build the sub-question engine.

        Args:
            tools: Query engine tools.
            llm: The model writing sub-questions and the answer.

        Returns:
            The query engine.
        """
        node = "LLM Sub-Question Query Engine"
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        items = query.query_tools(tools, node)
        wires.ready(model, tuple(items))
        with runtime.guarded():
            try:
                engine = query.build_sub_question(items, model)
            except Exception as error:
                raise query.failure(node, error) from None
        names = [item.metadata.name or "" for item in items]
        label = f"sub-questions over {len(items)} tool{'s' if len(items) != 1 else ''}"
        logger.info("LLM Sub-Question Query Engine built over %d tool(s)", len(items))
        run_result.publish(
            status=run_result.OK,
            summary=f"Splits questions across {', '.join(names)[:120]}",
            counts={"tools": len(items)},
            facts={"model": model.label},
            items=[f"{item.metadata.name}: {(item.metadata.description or '')[:160]}" for item in items],
        )
        return io.NodeOutput(wires.QueryEngine(engine=engine, label=label, depends=(model, *items)))
