"""Describe one agent: its model, tools, instructions and who it may hand off to."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_AGENTS, LLM_TOOLS
from ...modules.interface import run_result
from ...modules.llm import wires
from ...modules.llm import agents as agent_specs
from ...modules.llm import tools as agent_tools

REQUIRES = "llm"

logger = log.get_logger("llm.agents")


class WASLLMAgent(io.ComfyNode):
    """One agent's settings, appended to a chain of agents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMAgent",
            display_name="LLM Agent",
            search_aliases=[
                "WASLLMAgent",
                "LLM Agent",
                "FunctionAgent",
                "ReActAgent",
                "AgentWorkflow",
                "multi agent",
                "tool calling",
            ],
            category="WAS Suite/LLM/Agents",
            description=(
                "Set up one agent: the model it reasons with, the tools it may call, its "
                "instructions, and which agents it may hand the conversation to. Chain several "
                "for a team and run them with LLM Run Agent."
            ),
            inputs=[
                LLM.Input(
                    "llm",
                    tooltip="The model the agent thinks with. One that calls tools natively, such as Claude or GPT, runs as function calling; others run as ReAct.",
                ),
                io.String.Input(
                    "name", default="assistant",
                    tooltip="The agent's name, which other agents hand off to: a letter, then letters, digits or underscores, such as `researcher`. Different for every agent on a chain.",
                ),
                io.String.Input(
                    "description", multiline=True,
                    default="Answers the user's questions, using its tools where they help.",
                    tooltip="What the agent does, read by the other agents to decide when to hand off to it, such as `Writes the final summary in plain English.`",
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default="",
                    tooltip="The agent's instructions, such as `You are a careful researcher. Cite the page each fact came from.` Empty leaves the model's default behaviour.",
                ),
                io.Combo.Input(
                    "kind", options=list(agent_specs.KINDS), default="auto",
                    tooltip="`auto` = function calling where the model has it, ReAct otherwise; `function calling` = the model's own tool calls; `ReAct` = thought and action written as text, for any model.",
                ),
                io.String.Input(
                    "can_handoff_to", default="",
                    tooltip="Agents this one may pass the conversation to, comma separated, such as `writer, checker`. Empty = any agent on the chain.",
                ),
                LLM_TOOLS.Input(
                    "tools", optional=True,
                    tooltip="The tools this agent may call, chained from LLM Tool, LLM Query Engine Tool or LLM Retriever Tool. Empty for an agent that only talks.",
                ),
                LLM_AGENTS.Input(
                    "agents", optional=True,
                    tooltip="Agents from earlier LLM Agent nodes, kept ahead of this one. The first on the chain starts unless LLM Run Agent names another.",
                ),
            ],
            outputs=[
                LLM_AGENTS.Output(
                    display_name="agents",
                    tooltip="The chain with this agent appended, for the next LLM Agent or LLM Run Agent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, llm, name, description, system_prompt, kind, can_handoff_to, tools=None, agents=None) -> io.NodeOutput:
        """Check the settings and append the agent.

        Args:
            llm: The model it reasons with.
            name: Its name.
            description: What it does.
            system_prompt: Its instructions.
            kind: How it runs.
            can_handoff_to: Agents it may hand off to.
            tools: Tools it may call.
            agents: Earlier agents.

        Returns:
            The chain.
        """
        node = "LLM Agent"
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        held = agent_tools.chain_of(tools, node)
        wires.ready(model, held)
        spec = agent_specs.spec(agents, model, held, name, description, system_prompt, kind, can_handoff_to)
        result = wires.chain(agents, spec)
        how = agent_specs.KIND_WORDS[agent_specs.kind_of(spec)]
        logger.info("LLM Agent added an agent with %d tool(s); the chain holds %d", len(held), len(result))
        facts = {"model": model.label, "runs as": how}
        if spec.can_handoff_to:
            facts["hands off to"] = ", ".join(spec.can_handoff_to)
        if spec.system_prompt:
            facts["instructions"] = f"{len(spec.system_prompt)} characters"
        run_result.publish(
            status=run_result.OK,
            summary=f"{spec.name} on {model.label}, {how}, {len(held)} tool{'s' if len(held) != 1 else ''}"[:200],
            counts={"agents": len(result), "tools": len(held)},
            facts=facts,
            items=agent_tools.names(held),
        )
        return io.NodeOutput(result)
