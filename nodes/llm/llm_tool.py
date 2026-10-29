"""Add a built-in tool to the chain an agent calls."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_TOOLS
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import wires
from ...modules.llm import tools as agent_tools

REQUIRES = "llm"

logger = log.get_logger("llm.tools")


class WASLLMTool(io.ComfyNode):
    """One built-in agent tool, appended to a chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        kinds = [
            io.DynamicCombo.Option(key="calculator", inputs=[]),
            io.DynamicCombo.Option(
                key="date and time",
                inputs=[
                    io.String.Input(
                        "timezone", default="UTC",
                        tooltip="The zone the model's dates and times are in unless it names another: `UTC`, `local` for this machine's clock, or an offset such as `+05:30`.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="wikipedia",
                inputs=[
                    io.String.Input(
                        "language", default="en",
                        tooltip="Which Wikipedia to read, by its language code: `en` English, `de` German, `simple` Simple English.",
                    ),
                    io.Int.Input(
                        "results", default=3, min=1, max=10,
                        tooltip="Pages one search lists. `3` keeps the answer short; `10` gives the model more to choose from.",
                    ),
                    io.Int.Input(
                        "max_characters", default=4000, min=200, max=20000,
                        tooltip="The most text one search or page summary hands the model. `4000` is about 700 words.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="web page",
                inputs=[
                    io.Int.Input(
                        "max_kilobytes", default=1024, min=16, max=8192,
                        tooltip="The most of a page downloaded, in kilobytes. `1024` covers nearly every article; a larger page is read up to this point.",
                    ),
                    io.Int.Input(
                        "max_characters", default=8000, min=500, max=100000,
                        tooltip="The most of a page's text handed the model. `8000` is about 1400 words; more costs more tokens.",
                    ),
                    io.Int.Input(
                        "timeout_seconds", default=20, min=1, max=120,
                        tooltip="Seconds to wait for a page before giving up and telling the model it failed.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="web search (Tavily)",
                inputs=[
                    io.Int.Input(
                        "results", default=5, min=1, max=10,
                        tooltip="Results one search lists, each with its address and a passage.",
                    ),
                    io.Int.Input(
                        "max_characters", default=6000, min=500, max=50000,
                        tooltip="The most text one search hands the model. `6000` is about 1000 words.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="web search (Brave)",
                inputs=[
                    io.Int.Input(
                        "results", default=5, min=1, max=20,
                        tooltip="Results one search lists, each with its address and a short description.",
                    ),
                    io.Int.Input(
                        "max_characters", default=6000, min=500, max=50000,
                        tooltip="The most text one search hands the model. `6000` is about 1000 words.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="document reader",
                inputs=[
                    picker.path_input(io.Combo.Input(
                        "document", options=agent_tools.document_options(),
                        tooltip="The one file the model may read: text, Markdown, HTML, PDF, Word, OpenDocument, EPUB, PowerPoint or code in ComfyUI's input, output or temp folders, tagged `[input]`, `[output]` or `[temp]`.",
                    )),
                    io.String.Input(
                        "name", default="read_document",
                        tooltip="The tool's name as the model sees it, such as `read_manual`. Give each document reader on one agent its own name.",
                    ),
                    io.Int.Input(
                        "max_characters", default=8000, min=500, max=50000,
                        tooltip="Characters in one part. The model reads the document part by part, so `8000` hands it about 1400 words at a time.",
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMTool",
            display_name="LLM Tool",
            search_aliases=[
                "WASLLMTool",
                "LLM Tool",
                "FunctionTool",
                "agent tool",
                "calculator",
                "wikipedia",
                "web search",
                "fetch web page",
                "document reader",
            ],
            category="WAS Suite/LLM/Agents",
            description=(
                "Give an agent a ready-made tool: a calculator, the date and time, Wikipedia, a "
                "web page reader, web search through Tavily or Brave, or a reader for one picked "
                "document. Chain several and connect the last to LLM Agent."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "tool", options=kinds, display_name="Tool",
                    tooltip=(
                        "`calculator` = exact arithmetic; `date and time` = today's date and date sums; "
                        "`wikipedia` = search and page summaries; `web page` = reads one address; "
                        "`web search` = Tavily or Brave, keyed by TAVILY_API_KEY or BRAVE_API_KEY; "
                        "`document reader` = one picked file."
                    ),
                ),
                LLM_TOOLS.Input(
                    "tools", optional=True,
                    tooltip="Tools from earlier LLM Tool, LLM Query Engine Tool or LLM Retriever Tool nodes, kept ahead of this one.",
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
    def fingerprint_inputs(cls, tool, tools=None):
        """Text that changes when the picked document changes on disk."""
        choice = dict(tool or {})
        if choice.get("tool") == "document reader":
            return agent_tools.document_stamp(choice.get("document", ""))
        return ""

    @classmethod
    def execute(cls, tool, tools=None) -> io.NodeOutput:
        """Build the tool and append it.

        Args:
            tool: The tool's kind and its settings.
            tools: Earlier tools.

        Returns:
            The chain.
        """
        node = "LLM Tool"
        choice = dict(tool or {})
        which = choice.get("tool", "calculator")
        chained = agent_tools.chain_of(tools, node)
        facts = {"tool": str(which)}
        if which == "calculator":
            made = agent_tools.calculator_tools()
        elif which == "date and time":
            made = agent_tools.datetime_tools(choice.get("timezone", "UTC"))
            facts["timezone"] = str(choice.get("timezone", "UTC"))[:40]
        elif which == "wikipedia":
            made = agent_tools.wikipedia_tools(
                choice.get("language", "en"), int(choice.get("results", 3)), int(choice.get("max_characters", 4000)),
            )
            facts["language"] = str(choice.get("language", "en"))[:20]
        elif which == "web page":
            made = agent_tools.web_page_tools(
                int(choice.get("max_kilobytes", 1024)), int(choice.get("max_characters", 8000)),
                float(choice.get("timeout_seconds", 20)),
            )
            facts["page limit"] = f"{int(choice.get('max_kilobytes', 1024))} KB"
        elif which == "web search (Tavily)":
            made = agent_tools.tavily_tools(int(choice.get("results", 5)), int(choice.get("max_characters", 6000)))
        elif which == "web search (Brave)":
            made = agent_tools.brave_tools(int(choice.get("results", 5)), int(choice.get("max_characters", 6000)))
        elif which == "document reader":
            made = agent_tools.document_tools(
                choice.get("document", ""), choice.get("name", "read_document"), int(choice.get("max_characters", 8000)),
            )
        else:
            raise ValueError(f"{node}: tool {str(which)[:40]!r} is not one of {', '.join(agent_tools.KINDS)}.")
        taken = set(agent_tools.names(chained))
        for name in agent_tools.names(made):
            if name in taken:
                raise ValueError(
                    f"{node}: a tool named {name!r} is already on the chain. Remove the repeated "
                    f"LLM Tool, or give the document reader another name."
                )
        result = wires.chain(chained, *made)
        network = which in ("wikipedia", "web page", "web search (Tavily)", "web search (Brave)")
        if network:
            from ...modules.model import network_enabled

            facts["network"] = "allowed" if network_enabled() else "off: calls are refused"
        logger.info("LLM Tool added %d %s tool(s); the chain holds %d", len(made), which, len(result))
        run_result.publish(
            status=run_result.WARNING if network and facts.get("network", "").startswith("off") else run_result.OK,
            summary=f"{which}: {', '.join(agent_tools.names(made))}; {len(result)} on the chain",
            counts={"tools": len(result), "added": len(made)},
            facts=facts,
            items=agent_tools.describe(made),
        )
        return io.NodeOutput(result)
