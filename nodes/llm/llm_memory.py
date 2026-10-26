"""Name where a conversation is kept between runs and how much of it reaches the model."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_MEMORY, LLM_MEMORY_BLOCKS
from ...modules.interface import run_result
from ...modules.llm import chat, storage, wires
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.memory")

#: Where long-term memory goes, as the widget offers them.
INSERT_CHOICES = ["system", "user"]


class WASLLMMemory(io.ComfyNode):
    """A conversation memory: its session name, its size and its long-term blocks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMMemory",
            display_name="LLM Memory",
            search_aliases=[
                "WASLLMMemory",
                "LLM Memory",
                "Memory",
                "chat history",
                "session",
                "conversation memory",
            ],
            category="WAS Suite/LLM/Memory",
            description=(
                "Keep a conversation between runs under a session name, for LLM Chat Session. "
                "Recent messages are handed to the model up to a token limit, and older ones "
                "move into the long-term memory blocks. The panel shows what the session holds."
            ),
            inputs=[
                io.String.Input(
                    "session", default="",
                    tooltip="The name the conversation is saved under, such as `project notes`. The same name carries it on in later runs; empty keeps it for one run only.",
                ),
                io.Int.Input(
                    "token_limit", default=memories.DEFAULT_TOKEN_LIMIT, min=256, max=1048576, step=256,
                    tooltip="The most tokens of conversation and memory blocks handed to the model each message. `30000` suits most models; lower it for a small context window.",
                ),
                io.Float.Input(
                    "history_ratio", default=0.7, min=0.05, max=1.0, step=0.05,
                    tooltip="Share of token_limit kept for recent messages. `0.7` leaves the rest for memory blocks; `1.0` gives blocks no room of their own.",
                ),
                io.Combo.Input(
                    "insert", options=INSERT_CHOICES, default="system",
                    tooltip="`system` = long-term memory goes in the system prompt; `user` = it is added to the newest message, for models that ignore system prompts.",
                ),
                LLM_MEMORY_BLOCKS.Input(
                    "blocks", optional=True,
                    tooltip="Long-term memory blocks chained from LLM Memory Block: fixed text, facts pulled from older messages, or older messages found by similarity.",
                ),
            ],
            outputs=[
                LLM_MEMORY.Output(
                    display_name="memory",
                    tooltip="The memory, for LLM Chat Session, LLM Memory History or LLM Clear Memory.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, session="", token_limit=memories.DEFAULT_TOKEN_LIMIT, history_ratio=0.7, insert="system", blocks=None):
        """Whether the node has to run again.

        Returns:
            The session name and when its file last changed.
        """
        name = str(session or "").strip()
        if not name:
            return ""
        try:
            return f"{name}|{os.path.getmtime(storage.session_file(name))}"
        except (OSError, ValueError):
            return name

    @classmethod
    def execute(cls, session, token_limit, history_ratio, insert, blocks=None) -> io.NodeOutput:
        """Describe the memory.

        Args:
            session: The name the conversation is saved under.
            token_limit: Tokens handed to the model at most.
            history_ratio: Share kept for recent messages.
            insert: Where block content goes.
            blocks: Long-term memory blocks.

        Returns:
            The memory spec.
        """
        node = "LLM Memory"
        name = str(session or "").strip()
        if name:
            name = storage.check_name(name, "Session")
        if insert not in INSERT_CHOICES:
            raise ValueError(f"{node}: insert {str(insert)[:40]!r} is not system or user.")
        chained = wires.chain(blocks)
        if not wires.every(chained, memories.BlockSpec):
            raise ValueError(f"{node} was handed something other than memory blocks on its blocks input. Connect LLM Memory Block to it.")
        seen: set[str] = set()
        for block in chained:
            if block.name in seen:
                raise ValueError(
                    f"{node} was handed two memory blocks named {block.name!r}. Give each LLM "
                    f"Memory Block its own name."
                )
            seen.add(block.name)
        spec = memories.MemorySpec(
            session=name,
            token_limit=int(token_limit),
            history_ratio=float(history_ratio),
            insert=insert,
            blocks=chained,
        )
        facts = {
            "session": name or "this run only",
            "token limit": str(int(token_limit)),
            "recent share": f"{float(history_ratio):.2f}",
            "insert": insert,
        }
        if chained:
            facts["blocks"] = ", ".join(f"{block.name} ({block.kind})" for block in chained)[:64]
        counts = {"blocks": len(chained)}
        status, line = run_result.OK, "This memory lasts one run"
        if name:
            try:
                held = chat.session_summary(name)
            except ValueError as error:
                held = None
                status, line = run_result.WARNING, f"The saved session cannot be read: {error}"
            if held is None and status == run_result.OK:
                line = f"Session {name!r} is new; the first message saves it"
            elif held is not None:
                counts.update({"messages": held["messages"], "archived": held["archived"], "facts": held["facts"]})
                facts["last updated"] = held["updated"] or "unknown"
                line = f"Session {name!r} holds {held['messages'] + held['archived']} messages"
        logger.info("LLM Memory set up session %s with %d block(s)", name or "(one run)", len(chained))
        run_result.publish(status=status, summary=line, counts=counts, facts=facts)
        return io.NodeOutput(spec)
