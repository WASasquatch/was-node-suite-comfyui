"""Make a query engine into a tool an agent or a routing engine can call."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_QUERY_ENGINE, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import wires
from ...modules.llm import tools as agent_tools

REQUIRES = "llm"

logger = log.get_logger("llm.tools")


class WASLLMQueryEngineTool(io.ComfyNode):
    """A query engine as one named tool, appended to a chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMQueryEngineTool",
            display_name="LLM Query Engine Tool",
            search_aliases=[
                "WASLLMQueryEngineTool",
                "LLM Query Engine Tool",
                "QueryEngineTool",
                "agent tool",
                "rag tool",
                "knowledge base tool",
            ],
            category="WAS Suite/LLM/Agents",
            description=(
                "Turn a query engine into a named tool, so an agent can ask it questions or a "
                "router or sub-question engine can choose it. The description tells the model "
                "what the engine knows."
            ),
            inputs=[
                LLM_QUERY_ENGINE.Input(
                    "query_engine",
                    tooltip="The engine the tool asks, such as from LLM Query Engine built over the camera manuals.",
                ),
                io.String.Input(
                    "name", default="search_documents",
                    tooltip="The tool's name as the model sees it: a letter, then letters, digits or underscores, such as `camera_manuals`. Each tool on one chain needs its own.",
                ),
                io.String.Input(
                    "description", multiline=True,
                    default="Answers questions about the documents in this index.",
                    tooltip="What the engine knows, read by the model to decide when to call it, such as `Answers questions about the Canon R5 and R6 manuals.`",
                ),
                LLM_TOOLS.Input(
                    "tools", optional=True,
                    tooltip="Tools from earlier tool nodes, kept ahead of this one.",
                ),
            ],
            outputs=[
                LLM_TOOLS.Output(
                    display_name="tools",
                    tooltip="The chain with this tool appended, for LLM Agent, LLM Router Query Engine or LLM Sub-Question Query Engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, query_engine, name, description, tools=None) -> io.NodeOutput:
        """Wrap the engine and append it.

        Args:
            query_engine: The engine the tool asks.
            name: The tool's name.
            description: What it answers.
            tools: Earlier tools.

        Returns:
            The chain.
        """
        node = "LLM Query Engine Tool"
        engine = wires.expect(query_engine, wires.QueryEngine, node, "query_engine", "LLM Query Engine")
        chained = agent_tools.chain_of(tools, node)
        made = agent_tools.query_engine_tool(engine, name, description)
        if made.metadata.name in agent_tools.names(chained):
            raise ValueError(f"{node}: a tool named {made.metadata.name!r} is already on the chain. Give this one another name.")
        result = wires.chain(chained, made)
        logger.info("LLM Query Engine Tool added one tool; the chain holds %d", len(result))
        run_result.publish(
            status=run_result.OK,
            summary=f"{made.metadata.name} asks {engine.label}"[:200],
            counts={"tools": len(result)},
            facts={"engine": engine.label[:120], "name": made.metadata.name},
            items=agent_tools.describe([made]),
        )
        return io.NodeOutput(result)
