"""Write a conversation out as plain text."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_MESSAGES
from ...modules.interface import run_result
from ...modules.llm import runtime, wires
from ...modules.llm import messages as llm_messages

REQUIRES = "llm"

logger = log.get_logger("llm.messages")

#: The node's name in messages.
NODE = "LLM Messages to Text"


class LLMMessagesToText(io.ComfyNode):
    """A conversation as a ``role: text`` transcript."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMMessagesToText",
            display_name="LLM Messages to Text",
            search_aliases=[
                "WASLLMMessagesToText",
                "LLM Messages to Text",
                "transcript",
                "chat history",
                "conversation to text",
            ],
            category="WAS Suite/LLM/Messages",
            is_output_node=True,
            description=(
                "Turn a conversation into plain text, one `role: text` paragraph per message, such "
                "as to save a chat or show it. Pictures and sound are noted in brackets."
            ),
            inputs=[
                LLM_MESSAGES.Input(
                    "messages",
                    tooltip="The conversation, from LLM Message or LLM Chat.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The transcript, such as `user: Hello` then `assistant: Hi there`, with "
                        "thinking left out."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many messages the conversation holds.",
                ),
            ],
        )

    @classmethod
    def execute(cls, messages) -> io.NodeOutput:
        """Write the transcript.

        Args:
            messages: The conversation.

        Returns:
            The transcript and the number of messages.

        Raises:
            ValueError: Nothing, or something other than a conversation, is connected.
        """
        conversation = wires.expect(messages, wires.Messages, NODE, "messages", "LLM Message or LLM Chat")
        with runtime.guarded():
            text = llm_messages.transcript(conversation.items)
        logger.debug("%s wrote %d messages", NODE, len(conversation))
        run_result.publish(
            status=run_result.OK if len(conversation) else run_result.WARNING,
            summary=(
                f"{len(conversation)} message{'s' if len(conversation) != 1 else ''}, "
                f"{len(text):,} characters."
            ),
            counts={"messages": len(conversation), "characters": len(text)},
            bodies=[run_result.body("transcript", text)],
        )
        return io.NodeOutput(text, len(conversation))
