"""Send each question to the retriever best suited to it, chosen by a language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_RETRIEVER, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import query, retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Router Retriever"


class LLMRouterRetriever(io.ComfyNode):
    """A retriever that routes each question among retriever tools."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRouterRetriever",
            display_name="LLM Router Retriever",
            search_aliases=[
                "WASLLMRouterRetriever",
                "LLM Router Retriever",
                "RouterRetriever",
                "route retrieval",
                "selector",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Search only the retriever suited to each question, chosen by the model from each "
                "tool's description, such as the manuals for how-to questions and the invoices for "
                "billing. In multi mode every fitting retriever searches and their chunks are joined."
            ),
            inputs=[
                LLM_TOOLS.Input(
                    "tools",
                    tooltip="Retriever tools to choose among, chained from LLM Retriever Tool. Describe what each one finds.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that picks, such as LLM From CLIP or LLM OpenAI.",
                ),
                io.Combo.Input(
                    "selector", options=list(query.SELECTORS), default="single",
                    tooltip="`single` = one retriever searches each question; `multi` = every retriever that fits searches.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The router, for LLM Retrieve, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tools, llm, selector) -> io.NodeOutput:
        """Build the router.

        Args:
            tools: Retriever tools.
            llm: The model choosing.
            selector: ``single`` or ``multi``.

        Returns:
            The retriever.
        """
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM OpenAI or LLM From CLIP")
        items = query.retriever_tools(tools, NODE)
        if selector not in query.SELECTORS:
            raise ValueError(f"{NODE}: selector {str(selector)[:40]!r} is not single or multi.")
        wires.ready(model, tuple(items))
        with runtime.guarded():
            wire, facts = retrieval.router_retriever(items, model, selector)
        logger.info("%s routes among %d retriever(s)", NODE, len(items))
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {wire.label}.",
            counts={"retrievers": len(items)},
            facts=facts,
            items=[f"{item.metadata.name}: {(item.metadata.description or '')[:160]}" for item in items],
        )
        return io.NodeOutput(wire)
