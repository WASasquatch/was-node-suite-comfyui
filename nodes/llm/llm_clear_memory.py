"""Delete the conversation a memory has saved, and pass the memory on."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_MEMORY
from ...modules.interface import run_result
from ...modules.llm import storage, wires
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.memory")


class WASLLMClearMemory(io.ComfyNode):
    """Delete a saved chat session once confirmed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMClearMemory",
            display_name="LLM Clear Memory",
            search_aliases=[
                "WASLLMClearMemory",
                "LLM Clear Memory",
                "reset chat",
                "delete session",
                "forget",
            ],
            category="WAS Suite/LLM/Memory",
            is_output_node=True,
            description=(
                "Delete the conversation LLM Memory has saved under its session name, with its "
                "recalled-message files, so the next message starts fresh. Nothing is deleted "
                "until confirm is on. The memory passes through for LLM Chat Session."
            ),
            inputs=[
                LLM_MEMORY.Input(
                    "memory",
                    tooltip="The memory whose saved session to delete, from LLM Memory.",
                ),
                io.Boolean.Input(
                    "confirm", default=False,
                    tooltip="`true` deletes the saved session every time the graph runs; `false` deletes nothing.",
                ),
            ],
            outputs=[
                LLM_MEMORY.Output(
                    display_name="memory",
                    tooltip="The same memory, now empty while confirm is `true`, for LLM Chat Session.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, memory=None, confirm=False):
        """Whether the node has to run again.

        Returns:
            ``NaN`` while confirm is on, so every run deletes, and ``False`` otherwise.
        """
        held = confirm[0] if isinstance(confirm, list) else confirm
        return float("NaN") if held else False

    @classmethod
    def execute(cls, memory, confirm) -> io.NodeOutput:
        """Delete the session when confirmed.

        Args:
            memory: The memory whose session to delete.
            confirm: Whether to delete.

        Returns:
            The memory.
        """
        node = "LLM Clear Memory"
        spec = wires.expect(memory, memories.MemorySpec, node, "memory", "LLM Memory")
        if not spec.session:
            run_result.publish(
                status=run_result.WARNING,
                summary="This memory keeps nothing between runs, so there is nothing to delete",
                facts={"session": "this run only"},
            )
            return io.NodeOutput(spec)
        if not confirm:
            run_result.publish(
                status=run_result.WARNING,
                summary=f"Session {spec.session!r} was kept; set confirm to true to delete it",
                facts={"session": spec.session, "confirm": "false"},
            )
            return io.NodeOutput(spec)
        removed = storage.delete_session(spec.session)
        logger.info("LLM Clear Memory %s session %s", "deleted" if removed else "found nothing saved for", spec.session)
        run_result.publish(
            status=run_result.OK if removed else run_result.WARNING,
            summary=(
                f"Session {spec.session!r} was deleted" if removed
                else f"Nothing was saved under session {spec.session!r}"
            ),
            counts={"deleted": int(bool(removed))},
            facts={"session": spec.session, "confirm": "true"},
        )
        return io.NodeOutput(spec)
