"""Route each question to the query engine, or engines, best suited to it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_QUERY_ENGINE, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")


class WASLLMRouterQueryEngine(io.ComfyNode):
    """A query engine that lets a model pick which of its tools answers a question."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRouterQueryEngine",
            display_name="LLM Router Query Engine",
            search_aliases=[
                "WASLLMRouterQueryEngine",
                "LLM Router Query Engine",
                "RouterQueryEngine",
                "LLMSingleSelector",
                "LLMMultiSelector",
                "router",
                "selector",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Send each question to the query engine tool best suited to it, chosen by the "
                "model from each tool's description, such as a summary engine for overviews and "
                "a vector engine for facts. In multi mode several tools answer and the replies "
                "are combined."
            ),
            inputs=[
                LLM_TOOLS.Input(
                    "tools",
                    tooltip="Query engine tools to choose among, chained from LLM Query Engine Tool. The model picks from each tool's description, so describe what each one knows.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that picks the tool and combines several answers, such as LLM OpenAI or LLM From CLIP.",
                ),
                io.Combo.Input(
                    "selector", options=list(query.SELECTORS), default="single",
                    tooltip="`single` = one tool answers each question; `multi` = every tool that fits answers and the replies are summarised into one.",
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The router engine, for LLM Query or LLM Query Engine Tool.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tools, llm, selector) -> io.NodeOutput:
        """Build the router engine.

        Args:
            tools: Query engine tools.
            llm: The model choosing among them.
            selector: ``single`` or ``multi``.

        Returns:
            The query engine.
        """
        node = "LLM Router Query Engine"
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        items = query.query_tools(tools, node)
        if selector not in query.SELECTORS:
            raise ValueError(f"{node}: selector {str(selector)[:40]!r} is not single or multi.")
        wires.ready(model, tuple(items))
        with runtime.guarded():
            try:
                engine = query.build_router(items, model, selector)
            except Exception as error:
                raise query.failure(node, error) from None
        picker = type(engine._selector).__name__
        label = f"{selector} router over {len(items)} tool{'s' if len(items) != 1 else ''}"
        logger.info("LLM Router Query Engine built a %s router over %d tool(s)", selector, len(items))
        run_result.publish(
            status=run_result.OK,
            summary=label,
            counts={"tools": len(items)},
            facts={
                "model": model.label,
                "selector": selector,
                "picks by": "tool calling" if picker.startswith("Pydantic") else "reading the reply",
            },
            items=[f"{item.metadata.name}: {(item.metadata.description or '')[:160]}" for item in items],
        )
        return io.NodeOutput(wires.QueryEngine(engine=engine, label=label, depends=(model, *items)))
