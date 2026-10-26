"""Give an agent the tools of an MCP server named in config.yaml."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import mcp_servers, runtime, wires
from ...modules.llm import tools as agent_tools

REQUIRES = "llm"

logger = log.get_logger("llm.tools")

#: The node's name in messages.
NODE = "LLM MCP Tools"

#: Tool names the tools input lists at most.
MAX_WANTED = 128


class LLMMCPTools(io.ComfyNode):
    """The tools an MCP server offers, appended to an agent's tool chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMMCPTools",
            display_name="LLM MCP Tools",
            search_aliases=[
                "WASLLMMCPTools",
                "LLM MCP Tools",
                "MCP",
                "Model Context Protocol",
                "McpToolSpec",
                "agent tools",
            ],
            category="WAS Suite/LLM/Agents",
            description=(
                "Give an agent the tools of an MCP server, such as a file browser, a database or a "
                "service's own tools. Servers are added under llm.mcp_servers in config.yaml and "
                "picked here by name. The panel lists each tool and what it does."
            ),
            inputs=[
                io.Combo.Input(
                    "server", options=mcp_servers.options(),
                    tooltip="Which MCP server, by its name under llm.mcp_servers in config.yaml. Restart ComfyUI after adding one.",
                ),
                io.String.Input(
                    "allowed_tools", default="",
                    tooltip="The server's tools to offer, separated by commas, such as `read_file, list_directory`. Blank = every tool it offers.",
                ),
                io.String.Input(
                    "prefix", default="",
                    tooltip="Placed before every tool name, such as `notes` giving `notes_read_file`, so two servers' tools keep apart. Blank = the server's own names.",
                ),
                io.Int.Input(
                    "timeout_seconds", default=60, min=5, max=1800,
                    tooltip="Seconds one call to the server may take: 60 = the usual; 600 for a slow job such as a build.",
                ),
                LLM_TOOLS.Input(
                    "tools", optional=True,
                    tooltip="Tools from earlier tool nodes, kept ahead of these.",
                ),
            ],
            outputs=[
                LLM_TOOLS.Output(
                    display_name="tools",
                    tooltip="The chain with the server's tools appended, for the next tool node or LLM Agent.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, server="", **kwargs) -> str:
        """The server's configured entry, so an edited config.yaml lists the tools again."""
        found = mcp_servers.servers().get(str(server or ""))
        return repr(found) if found is not None else str(server)[:80]

    @classmethod
    async def execute(cls, server, allowed_tools, prefix, timeout_seconds, tools=None) -> io.NodeOutput:
        """List the server's tools and append them.

        Args:
            server: The server's configured name.
            allowed_tools: Tool names to keep, comma separated, blank for all.
            prefix: Placed before every tool name.
            timeout_seconds: Seconds one call may take.
            tools: Earlier tools.

        Returns:
            The chain.
        """
        chained = agent_tools.chain_of(tools, NODE)
        found = mcp_servers.server(server, NODE)
        wanted = [part.strip() for part in str(allowed_tools or "").split(",") if part.strip()][:MAX_WANTED]
        lead = str(prefix or "").strip()[:32]
        with runtime.guarded():
            made = await mcp_servers.fetch_tools(found, wanted, lead, float(timeout_seconds), NODE)
        if not made:
            raise ValueError(f"{NODE}: the MCP server {found.name} offers no tools.")
        taken = set(agent_tools.names(chained))
        for name in agent_tools.names(made):
            if name in taken:
                raise ValueError(
                    f"{NODE}: a tool named {name!r} is already on the chain. Give this server a prefix, "
                    f"such as `{found.name.replace('-', '_')}`."
                )
        result = wires.chain(chained, *made)
        logger.info("%s added %d tool(s) from %s; the chain holds %d", NODE, len(made), found.name, len(result))
        run_result.publish(
            status=run_result.OK,
            summary=f"{found.name}: {len(made)} tool(s); {len(result)} on the chain",
            counts={"tools": len(result), "added": len(made)},
            facts={
                "server": found.label,
                "runs": "on this machine" if found.local else "off this machine",
                "timeout": f"{int(timeout_seconds)} s",
            },
            items=agent_tools.describe(made),
        )
        return io.NodeOutput(result)
