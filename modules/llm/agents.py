"""Agents built from the ``WAS_LLM_AGENTS`` chain and run on llama-index's agent workflows.

:func:`spec` checks one agent's settings, :func:`run` builds every agent on a chain, runs the
conversation and returns its reply with a trace of each tool call and handoff.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from .. import log
from . import messages, runtime, tools, wires

__all__ = [
    "KINDS",
    "KIND_VALUES",
    "KIND_WORDS",
    "MAX_TRACE_CHARS",
    "Outcome",
    "kind_of",
    "root_of",
    "run",
    "spec",
    "turn_extra",
]

logger = log.get_logger("llm.agents")

#: The kinds LLM Agent offers, as the words on its menu.
KINDS = ("auto", "function calling", "ReAct")

#: Menu word to the ``AgentSpec.kind`` value.
KIND_VALUES = {"auto": "auto", "function calling": "function", "ReAct": "react"}

#: ``AgentSpec.kind`` value to the words a panel shows.
KIND_WORDS = {"function": "function calling", "react": "ReAct"}

#: Agents one chain holds at most.
MAX_AGENTS = 32

#: Characters of the whole trace kept at most.
MAX_TRACE_CHARS = 200_000

#: Characters of one tool result kept in the trace.
MAX_RESULT_CHARS = 4000

#: Characters of one tool call's arguments kept in the trace.
MAX_ARGUMENT_CHARS = 2000

#: Retrieved chunks gathered from tool results at most.
MAX_SOURCES = 200

#: Characters of a system prompt kept at most.
MAX_SYSTEM_CHARS = 100_000

#: Seconds between two checks for a cancelled queue while an agent waits on its model.
WATCH_INTERVAL = 0.25


@dataclass
class Outcome:
    """What one agent run produced.

    Attributes:
        reply: The final answer, thinking removed.
        thinking: Thinking the final answer carried.
        trace: Every tool call with its arguments and result, and every handoff, as text.
        sources: llama-index ``NodeWithScore`` objects gathered from tool results.
        steps: Model turns taken across every agent.
        tool_calls: Tools called, handoffs left out.
        handoffs: Times the conversation passed to another agent.
        seconds: Wall time of the run.
        last_agent: The agent that answered.
        stopped: True when the step limit was reached and a final answer was asked for.
        kinds: Agent name to ``function`` or ``react``.
    """

    reply: str = ""
    thinking: str = ""
    trace: str = ""
    sources: list = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    handoffs: int = 0
    seconds: float = 0.0
    last_agent: str = ""
    stopped: bool = False
    kinds: dict = field(default_factory=dict)


def _handoff_names(text) -> tuple[str, ...]:
    """The agent names a comma separated list holds, each checked.

    Raises:
        ValueError: A name is malformed.
    """
    found = []
    for part in str(text or "").split(","):
        name = part.strip()
        if name and name not in found:
            found.append(tools.check_name(name, "LLM Agent", "can_handoff_to"))
    return tuple(found)


def spec(previous, llm: wires.LanguageModel, agent_tools, name, description, system_prompt, kind, can_handoff_to) -> wires.AgentSpec:
    """One agent's settings, checked, as an ``AgentSpec``.

    Args:
        previous: The agents chained in before this one, a tuple or ``None``.
        llm: The model the agent reasons with.
        agent_tools: llama-index tools it may call, a tuple or ``None``.
        name: Its name.
        description: What it does, read by the other agents.
        system_prompt: Its instructions.
        kind: A word from :data:`KINDS`.
        can_handoff_to: Comma separated agent names, empty for any.

    Returns:
        The spec.

    Raises:
        ValueError: A setting cannot be used, a name repeats, or the model cannot call tools
            natively where function calling was asked for.
    """
    agent_name = tools.check_name(name, "LLM Agent", "name")
    chained = tuple(previous) if isinstance(previous, (tuple, list)) else ()
    if not wires.every(chained, wires.AgentSpec):
        raise ValueError("LLM Agent was handed something other than agents on its agents input. Connect another LLM Agent's agents output.")
    if len(chained) >= MAX_AGENTS:
        raise ValueError(f"An agent chain holds at most {MAX_AGENTS} agents. Run some of them separately.")
    if any(other.name == agent_name for other in chained):
        raise ValueError(f"Two agents on the chain are both named {agent_name!r}. Give each LLM Agent its own name.")
    words = " ".join(str(description or "").split())
    if not words:
        raise ValueError(
            "LLM Agent needs a description saying what the agent does, such as `Finds facts "
            "in the manuals`. Other agents read it to decide when to hand off."
        )
    held = tuple(agent_tools) if isinstance(agent_tools, (tuple, list)) else ()
    seen: set[str] = set()
    for tool_name in tools.names(held):
        if tool_name in seen:
            raise ValueError(
                f"The agent {agent_name!r} was given two tools named {tool_name!r}. Give one of "
                f"them another name, such as on its LLM Tool or LLM Query Engine Tool node."
            )
        seen.add(tool_name)
    value = KIND_VALUES.get(str(kind), "")
    if not value:
        raise ValueError(f"LLM Agent's kind must be one of {', '.join(KINDS)}.")
    handoffs = _handoff_names(can_handoff_to)
    if agent_name in handoffs:
        raise ValueError(f"The agent {agent_name!r} lists itself in can_handoff_to. Remove its own name.")
    made = wires.AgentSpec(
        name=agent_name,
        description=words[: tools.MAX_DESCRIPTION_CHARS],
        system_prompt=str(system_prompt or "")[:MAX_SYSTEM_CHARS],
        llm=llm,
        tools=held,
        kind=value,
        can_handoff_to=handoffs,
        depends=held,
    )
    kind_of(made)
    return made


def kind_of(agent: wires.AgentSpec) -> str:
    """``function`` or ``react``: how an agent will run.

    Raises:
        ValueError: Function calling was asked for and the model has none.
    """
    runtime.core()
    from llama_index.core.llms.function_calling import FunctionCallingLLM

    model = agent.llm.llm
    metadata = getattr(model, "metadata", None)
    calls = isinstance(model, FunctionCallingLLM) and bool(getattr(metadata, "is_function_calling_model", False))
    if agent.kind == "react":
        return "react"
    if agent.kind == "function":
        if not calls:
            raise ValueError(
                f"The agent {agent.name!r} is set to function calling, and {agent.llm.label} does "
                f"not call tools natively. Set its kind to ReAct or auto."
            )
        return "function"
    return "function" if calls else "react"


def root_of(agents, root: str) -> str:
    """The agent a run starts with.

    Args:
        agents: The chain's specs.
        root: A name, or empty for the first agent.

    Raises:
        ValueError: No agent carries that name.
    """
    chosen = str(root or "").strip()
    if not chosen:
        return agents[0].name
    if chosen not in [agent.name for agent in agents]:
        listed = ", ".join(agent.name for agent in agents)
        raise ValueError(f"No agent on the chain is named {chosen[:64]!r}. The agents are: {listed}. Clear root_agent to start with the first.")
    return chosen


def turn_extra(agents, root: str) -> str:
    """The text beside a message that decides its answer: the agents and the root."""
    return "agents=" + ",".join(agent.name for agent in agents) + f";root={root}"


def _build(agent: wires.AgentSpec, timeout: float):
    """A llama-index workflow agent from a spec."""
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent

    common = {
        "name": agent.name,
        "description": agent.description,
        "system_prompt": agent.system_prompt or None,
        "tools": list(agent.tools),
        "llm": agent.llm.llm,
        "can_handoff_to": list(agent.can_handoff_to) or None,
        "streaming": True,
        "early_stopping_method": "generate",
        "timeout": timeout,
    }
    if kind_of(agent) == "function":
        # A forced tool choice is refused by several hosted models, so none is set.
        return FunctionAgent(initial_tool_choice=None, **common)
    return ReActAgent(**common)


def _check_handoffs(agents) -> None:
    """Refuse a handoff naming an agent the chain does not hold.

    Raises:
        ValueError: A ``can_handoff_to`` name matches no agent.
    """
    known = {agent.name for agent in agents}
    for agent in agents:
        missing = [name for name in agent.can_handoff_to if name not in known]
        if missing:
            raise ValueError(
                f"The agent {agent.name!r} may hand off to {', '.join(repr(n) for n in missing)}, "
                f"and no agent on the chain has that name. The agents are: {', '.join(sorted(known))}."
            )


def _json(value) -> str:
    """Tool arguments as one line of JSON."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class _Trace:
    """The run's trace, held under :data:`MAX_TRACE_CHARS`."""

    def __init__(self):
        self.lines: list[str] = []
        self.size = 0
        self.cut = False

    def add(self, line: str) -> None:
        if self.cut:
            return
        if self.size + len(line) > MAX_TRACE_CHARS:
            self.lines.append("[the trace stops here; it reached its length limit]")
            self.cut = True
            return
        self.lines.append(line)
        self.size += len(line) + 1

    def text(self) -> str:
        return "\n".join(self.lines)


async def run(agents, root: str, message, memory, max_steps: int, timeout: float, stream=None) -> Outcome:
    """Run a conversation turn through a chain of agents.

    Args:
        agents: The ``AgentSpec`` chain.
        root: The agent that starts, from :func:`root_of`.
        message: The llama-index user ``ChatMessage``.
        memory: The llama-index memory the run reads and writes.
        max_steps: Model turns allowed before a final answer is asked for.
        timeout: Seconds the run may take.
        stream: A ``runtime.TextStream`` the reply is drawn on, or ``None``.

    Returns:
        The :class:`Outcome`.

    Raises:
        InterruptProcessingException: The queue was cancelled.
        RuntimeError: A model or the workflow failed, or the run took longer than ``timeout``.
    """
    from llama_index.core.agent.workflow import AgentInput, AgentOutput, AgentStream, AgentWorkflow, ToolCall, ToolCallResult

    _check_handoffs(agents)
    built = [_build(agent, timeout) for agent in agents]
    outcome = Outcome(kinds={agent.name: kind_of(agent) for agent in agents})
    if len(built) == 1:
        workflow = built[0]
    else:
        workflow = AgentWorkflow(agents=built, root_agent=root, timeout=timeout, early_stopping_method="generate")
    labels = {agent.name: agent.llm.label for agent in agents}
    trace = _Trace()
    sources: dict[str, object] = {}
    current = root
    started = time.monotonic()
    interrupted: list[BaseException] = []

    # Allows max_steps full turns; the turn after them asks for the final answer.
    limit = int(max_steps) + 1
    handler = workflow.run(user_msg=message, memory=memory, max_iterations=limit, early_stopping_method="generate")

    async def watch() -> None:
        while not handler.is_done():
            try:
                runtime.check_interrupt()
            except BaseException as error:
                interrupted.append(error)
                await handler.cancel_run()
                return
            await asyncio.sleep(WATCH_INTERVAL)

    def switch(name: str) -> None:
        nonlocal current
        if name and name != current:
            outcome.handoffs += 1
            trace.add(f"[{current}] handed the conversation to {name}")
            current = name
            if stream is not None:
                stream.write(f"\n\n[{name}]\n")
                stream.note(f"{name} on {labels.get(name, '')}")

    if stream is not None:
        stream.note(f"{root} on {labels.get(root, '')}")
    watcher = asyncio.create_task(watch())
    result = None
    try:
        async for event in handler.stream_events():
            if interrupted:
                break
            try:
                runtime.check_interrupt()
            except BaseException as error:
                interrupted.append(error)
                await handler.cancel_run()
                break
            if isinstance(event, AgentStream):
                switch(event.current_agent_name)
                if stream is not None and event.delta:
                    stream.write(event.delta)
            elif isinstance(event, AgentInput):
                switch(event.current_agent_name)
            elif isinstance(event, AgentOutput):
                switch(event.current_agent_name)
                outcome.steps += 1
            elif isinstance(event, ToolCallResult):
                output = event.tool_output
                content = str(getattr(output, "content", "") or "")
                if event.tool_name == "handoff":
                    trace.add(f"[{current}] hands off with {_json(event.tool_kwargs)[:MAX_ARGUMENT_CHARS]}: {content[:MAX_RESULT_CHARS]}")
                    continue
                outcome.tool_calls += 1
                failed = " failed" if getattr(output, "is_error", False) else ""
                trace.add(f"[{current}] {event.tool_name}{failed} returned:\n{content[:MAX_RESULT_CHARS]}")
                if stream is not None:
                    stream.note(f"{current}: {event.tool_name}{failed} answered, {len(content)} characters")
                for item in tools.sources_in(getattr(output, "raw_output", None)):
                    key = item.node.node_id
                    held = sources.get(key)
                    if held is None or (item.score or 0) > (held.score or 0):
                        sources[key] = item
            elif isinstance(event, ToolCall):
                if event.tool_name != "handoff":
                    trace.add(f"[{current}] calls {event.tool_name} with {_json(event.tool_kwargs)[:MAX_ARGUMENT_CHARS]}")
                    if stream is not None:
                        stream.note(f"{current}: calling {event.tool_name}")
        if not interrupted:
            result = await handler
    except Exception as error:
        if interrupted:
            raise interrupted[0] from None
        if type(error).__name__ == "WorkflowTimeoutError":
            raise RuntimeError(
                f"LLM Run Agent stopped after {timeout:.0f} seconds without an answer. Raise "
                f"timeout_seconds, or lower max_steps."
            ) from None
        raise RuntimeError(f"LLM Run Agent failed: {runtime.describe_error(error)}") from None
    finally:
        watcher.cancel()
        if stream is not None:
            stream.finish()
    if interrupted:
        raise interrupted[0]

    response = getattr(result, "response", None)
    reply, inline = messages.split_thinking(messages.text_of(response) if response is not None else str(result or ""))
    thought = messages.thinking_of(response) if response is not None else ""
    outcome.reply = reply
    outcome.thinking = "\n\n".join(part for part in (thought, inline) if part)
    outcome.last_agent = str(getattr(result, "current_agent_name", "") or current)
    outcome.stopped = outcome.steps > limit
    outcome.sources = sorted(sources.values(), key=lambda item: -(item.score or 0.0))[:MAX_SOURCES]
    outcome.seconds = time.monotonic() - started
    if outcome.thinking:
        trace.add(f"[{outcome.last_agent}] thought before answering:\n{outcome.thinking[:MAX_RESULT_CHARS]}")
    if outcome.stopped:
        trace.add(f"[{outcome.last_agent}] reached {int(max_steps)} steps and was asked for its best final answer")
    outcome.trace = trace.text()
    logger.info(
        "LLM Run Agent answered with %d agent(s) in %d step(s), %d tool call(s), %d handoff(s), %.2fs",
        len(agents), outcome.steps, outcome.tool_calls, outcome.handoffs, outcome.seconds,
    )
    return outcome
