"""Read back the conversation and facts a saved session holds."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_MEMORY, LLM_MESSAGES
from ...modules.interface import run_result
from ...modules.llm import messages, storage, wires
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.memory")


class WASLLMMemoryHistory(io.ComfyNode):
    """The saved conversation of a memory, without calling a model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMMemoryHistory",
            display_name="LLM Memory History",
            search_aliases=[
                "WASLLMMemoryHistory",
                "LLM Memory History",
                "chat history",
                "transcript",
                "session history",
            ],
            category="WAS Suite/LLM/Memory",
            is_output_node=True,
            description=(
                "Read a saved conversation back from LLM Memory without calling a model: the "
                "transcript, the messages and the facts its facts blocks have gathered. Useful "
                "for saving a chat to a file or handing it to another model."
            ),
            inputs=[
                LLM_MEMORY.Input(
                    "memory",
                    tooltip="The memory whose saved session to read, from LLM Memory.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="history",
                    tooltip="The conversation as text, one `role: text` paragraph per message, oldest first.",
                ),
                LLM_MESSAGES.Output(
                    display_name="messages",
                    tooltip="The conversation as messages, for LLM Chat, the agents or LLM Message.",
                ),
                io.String.Output(
                    display_name="facts",
                    tooltip="Facts the session's facts blocks have gathered, one `block: fact` line each. Empty without a facts block.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, memory) -> io.NodeOutput:
        """Read the session.

        Args:
            memory: The memory to read.

        Returns:
            The transcript, the messages and the facts.
        """
        node = "LLM Memory History"
        spec = wires.expect(memory, memories.MemorySpec, node, "memory", "LLM Memory")
        found = await memories.history(spec)
        saved = (storage.read_session(spec.session) or {}) if spec.session else {}
        held = saved.get("facts") if isinstance(saved.get("facts"), dict) else {}
        lines = [
            f"{name}: {fact}"
            for name, facts in held.items()
            if isinstance(facts, list)
            for fact in facts
            if isinstance(fact, str)
        ]
        facts_text = "\n".join(lines)
        transcript = messages.transcript(found)
        if not spec.session:
            status, line = run_result.WARNING, "This memory keeps nothing between runs, so there is no history to read"
        elif not saved:
            status, line = run_result.WARNING, f"Nothing is saved under session {spec.session!r} yet"
        else:
            status, line = run_result.OK, f"Session {spec.session!r} holds {len(found)} messages"
        logger.info("LLM Memory History read %d message(s) and %d fact(s)", len(found), len(lines))
        bodies = [run_result.body("transcript", transcript)]
        if facts_text:
            bodies.append(run_result.body("facts", facts_text))
        run_result.publish(
            status=status,
            summary=line,
            counts={"messages": len(found), "facts": len(lines)},
            facts={
                "session": spec.session or "this run only",
                "last updated": str(saved.get("updated") or "never")[:32],
            },
            bodies=bodies,
        )
        return io.NodeOutput(transcript, wires.Messages(tuple(found)), facts_text)
