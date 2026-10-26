"""Search a root retriever and follow the index nodes it finds to the tools they name."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_RETRIEVER, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Recursive Retriever"


class LLMRecursiveRetriever(io.ComfyNode):
    """A retriever that follows index nodes to the retrievers and query engines they name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRecursiveRetriever",
            display_name="LLM Recursive Retriever",
            search_aliases=[
                "WASLLMRecursiveRetriever",
                "LLM Recursive Retriever",
                "RecursiveRetriever",
                "document agents",
                "small to big",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Search a root retriever, and where it finds an index node, hand the search on to the "
                "retriever or query engine that node names: first pick the right manual by its summary, "
                "then search inside it. Build the root from LLM Index Node."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "root",
                    tooltip="The retriever searched first, over an index of LLM Index Node chunks and any other text.",
                ),
                LLM_TOOLS.Input(
                    "tools",
                    tooltip="The retriever and query engine tools the index nodes name, chained from LLM Retriever Tool and LLM Query Engine Tool.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The recursive retriever, for LLM Retrieve, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, root, tools) -> io.NodeOutput:
        """Build the retriever.

        Args:
            root: The retriever searched first.
            tools: The tools index nodes name.

        Returns:
            The retriever.
        """
        start = wires.expect(root, wires.Retriever, NODE, "root", "LLM Index Retriever")
        items = list(tools) if isinstance(tools, (tuple, list)) else []
        if not items:
            raise ValueError(f"{NODE} needs the tools its index nodes name. Connect LLM Retriever Tool or LLM Query Engine Tool.")
        wires.ready(start, tuple(items))
        with runtime.guarded():
            wire, facts = retrieval.recursive_retriever(start, items)
        logger.info("%s follows index nodes to %s", NODE, facts["targets"])
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {wire.label}.",
            counts={"targets": len(items)},
            facts=facts,
            items=[f"{item.metadata.name}: {(item.metadata.description or '')[:160]}" for item in items],
        )
        return io.NodeOutput(wire)
