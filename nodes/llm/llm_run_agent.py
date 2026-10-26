"""Run a message through one agent or a team of agents that hand off to each other."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_AGENTS, LLM_MEMORY, LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import messages, runtime, wires
from ...modules.llm import agents as agent_specs
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.agents")


class WASLLMRunAgent(io.ComfyNode):
    """Runs the agents on a chain and answers with the reply, the trace and the sources."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRunAgent",
            display_name="LLM Run Agent",
            search_aliases=[
                "WASLLMRunAgent",
                "LLM Run Agent",
                "AgentWorkflow",
                "FunctionAgent",
                "ReActAgent",
                "run agent",
                "multi agent",
                "handoff",
            ],
            category="WAS Suite/LLM/Agents",
            is_output_node=True,
            description=(
                "Send a message to an agent, or a team of agents that hand the conversation to "
                "each other, and get the reply. The trace lists every tool call and handoff, and "
                "the panel counts steps, tool calls and seconds."
            ),
            inputs=[
                LLM_AGENTS.Input(
                    "agents",
                    tooltip="The agents to run, chained from LLM Agent. One agent answers alone; several hand the conversation to each other.",
                ),
                io.String.Input(
                    "message", multiline=True, default="",
                    tooltip=(
                        "What to ask, such as `Find the R5's battery life in the manuals and convert "
                        "it to hours.` With a saved memory session, running the same message again "
                        "returns the saved reply without calling any tools; change the message to ask again."
                    ),
                ),
                io.String.Input(
                    "root_agent", default="",
                    tooltip="The agent that reads the message first, by name, such as `researcher`. Empty = the first agent on the chain.",
                ),
                io.Int.Input(
                    "max_steps", default=20, min=1, max=200,
                    tooltip="Model turns allowed across every agent before the one speaking is asked for its best answer. `20` suits most questions; `5` keeps a cheap model from wandering.",
                ),
                io.Int.Input(
                    "timeout_seconds", default=600, min=10, max=7200,
                    tooltip="Seconds the whole run may take before it stops with an error. `600` is ten minutes.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Pictures sent with the message, for agents whose model reads images.",
                ),
                LLM_MEMORY.Input(
                    "memory", optional=True,
                    tooltip="Where the conversation is kept between runs, from LLM Memory. Empty = this run only, starting fresh.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="reply",
                    tooltip="The final answer, from whichever agent finished the conversation, thinking removed.",
                ),
                io.String.Output(
                    display_name="trace",
                    tooltip="Every tool call with its arguments and result, and every handoff between agents, in order.",
                ),
                LLM_SOURCES.Output(
                    display_name="sources",
                    tooltip="The passages the agents' query engine and retriever tools found, best first, for LLM Evaluate or LLM Sources to Text.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, agents, message, root_agent, max_steps, timeout_seconds, image=None, memory=None) -> io.NodeOutput:
        """Run the agents on the message.

        Args:
            agents: The agent chain.
            message: What to ask.
            root_agent: The agent that starts.
            max_steps: Model turns allowed.
            timeout_seconds: Seconds the run may take.
            image: Pictures sent with the message.
            memory: Where the conversation is kept.

        Returns:
            The reply, the trace and the sources.
        """
        node = "LLM Run Agent"
        chain = wires.chain(agents)
        if not chain:
            raise ValueError(f"{node} needs at least one agent. Connect LLM Agent to its agents input.")
        if not wires.every(chain, wires.AgentSpec):
            raise ValueError(f"{node} was handed something other than agents. Connect LLM Agent's agents output to it.")
        kept = None if memory is None else wires.expect(memory, memories.MemorySpec, node, "memory", "LLM Memory")
        text = str(message or "")
        if not text.strip() and image is None:
            raise ValueError(f"{node} has nothing to send. Type a message, or connect an image.")
        wires.ready(chain, kept)
        root = agent_specs.root_of(chain, root_agent)
        names = ", ".join(agent.name for agent in chain)

        key = None
        if kept is not None and kept.session:
            key = memories.turn_key(text, image, extra=agent_specs.turn_extra(chain, root))
            last = memories.last_turn(kept)
            if last and last.get("key") == key:
                reply = str(last.get("reply") or "")
                trace = str(last.get("trace") or "")
                logger.info("LLM Run Agent returned the saved reply for a turn its session already answered")
                run_result.publish(
                    status=run_result.OK,
                    summary="This message was already answered in the session; the saved reply is returned and no tool was called.",
                    counts={"agents": len(chain)},
                    facts={"agents": names[:120], "session": kept.session, "run": "saved turn"},
                    bodies=[run_result.body("reply", reply), run_result.body("trace", trace)],
                )
                return io.NodeOutput(reply, trace, wires.Sources(query=text, answer=reply))

        root_model = next(agent.llm for agent in chain if agent.name == root)
        stream = runtime.TextStream(f"{root} on {root_model.label}")
        with runtime.guarded():
            opened = await memories.open_memory(kept, root_model)
            try:
                user = messages.message("user", text, images=image)
                outcome = await agent_specs.run(
                    chain, root, user, opened, int(max_steps), float(timeout_seconds), stream,
                )
            except BaseException:
                await memories.save_memory(memories.ephemeral(), opened)
                raise
            turn = {"key": key, "reply": outcome.reply, "trace": outcome.trace} if key else None
            held = await memories.save_memory(kept, opened, turn=turn)

        kinds = "; ".join(f"{name}: {agent_specs.KIND_WORDS[kind]}" for name, kind in outcome.kinds.items())
        facts = {
            "agents": names[:120],
            "started with": root,
            "answered by": outcome.last_agent,
            "runs as": kinds[:200],
            "memory": f"session {kept.session}" if kept is not None and kept.session else "this run only",
            "messages kept": str(held.get("active", 0)),
        }
        bodies = [run_result.body("reply", outcome.reply), run_result.body("trace", outcome.trace)]
        if outcome.thinking:
            bodies.append(run_result.body("thinking", outcome.thinking))
        summary = (
            f"{outcome.last_agent} answered after {outcome.steps} step{'s' if outcome.steps != 1 else ''}, "
            f"{outcome.tool_calls} tool call{'s' if outcome.tool_calls != 1 else ''} and "
            f"{outcome.handoffs} handoff{'s' if outcome.handoffs != 1 else ''}"
        )
        if outcome.stopped:
            summary += f"; it reached {int(max_steps)} steps and was asked for its best answer"
        run_result.publish(
            status=run_result.WARNING if outcome.stopped or not outcome.reply.strip() else run_result.OK,
            summary=summary[:200],
            counts={
                "steps": outcome.steps,
                "tool calls": outcome.tool_calls,
                "handoffs": outcome.handoffs,
                "seconds": round(outcome.seconds, 2),
                "sources": len(outcome.sources),
            },
            facts=facts,
            bodies=bodies,
        )
        sources = wires.Sources(items=tuple(outcome.sources), query=text, answer=outcome.reply)
        return io.NodeOutput(outcome.reply, outcome.trace, sources)
