"""Make a retriever into a tool an agent calls to look up passages."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_RETRIEVER, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import wires
from ...modules.llm import tools as agent_tools

REQUIRES = "llm"

logger = log.get_logger("llm.tools")


class WASLLMRetrieverTool(io.ComfyNode):
    """A retriever as one named tool, appended to a chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRetrieverTool",
            display_name="LLM Retriever Tool",
            search_aliases=[
                "WASLLMRetrieverTool",
                "LLM Retriever Tool",
                "RetrieverTool",
                "agent tool",
                "search passages",
                "lookup tool",
            ],
            category="WAS Suite/LLM/Agents",
            description=(
                "Turn a retriever into a named tool an agent calls to fetch the passages that "
                "best match its query, and then writes its own answer from them. The "
                "description tells the model what the passages hold."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="The retriever the tool searches with, such as one built over an index of meeting notes by an LLM retriever node.",
                ),
                io.String.Input(
                    "name", default="find_passages",
                    tooltip="The tool's name as the model sees it: a letter, then letters, digits or underscores, such as `meeting_notes`. Each tool on one chain needs its own.",
                ),
                io.String.Input(
                    "description", multiline=True,
                    default="Finds passages in the indexed documents that match a query.",
                    tooltip="What the passages hold, read by the model to decide when to call it, such as `Finds passages in the 2026 meeting notes.`",
                ),
                LLM_TOOLS.Input(
                    "tools", optional=True,
                    tooltip="Tools from earlier tool nodes, kept ahead of this one.",
                ),
            ],
            outputs=[
                LLM_TOOLS.Output(
                    display_name="tools",
                    tooltip="The chain with this tool appended, for the next tool node or LLM Agent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, retriever, name, description, tools=None) -> io.NodeOutput:
        """Wrap the retriever and append it.

        Args:
            retriever: The retriever the tool searches with.
            name: The tool's name.
            description: What the passages hold.
            tools: Earlier tools.

        Returns:
            The chain.
        """
        node = "LLM Retriever Tool"
        found = wires.expect(retriever, wires.Retriever, node, "retriever", "an LLM retriever node")
        chained = agent_tools.chain_of(tools, node)
        made = agent_tools.retriever_tool(found, name, description)
        if made.metadata.name in agent_tools.names(chained):
            raise ValueError(f"{node}: a tool named {made.metadata.name!r} is already on the chain. Give this one another name.")
        result = wires.chain(chained, made)
        logger.info("LLM Retriever Tool added one tool; the chain holds %d", len(result))
        run_result.publish(
            status=run_result.OK,
            summary=f"{made.metadata.name} searches {found.label}"[:200],
            counts={"tools": len(result)},
            facts={"retriever": found.label[:120], "name": made.metadata.name},
            items=agent_tools.describe([made]),
        )
        return io.NodeOutput(result)
