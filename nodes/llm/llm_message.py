"""Add one message, with any pictures and sound, to a conversation."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_MESSAGES
from ...modules.interface import run_result
from ...modules.llm import generate, runtime, wires
from ...modules.llm import messages as llm_messages

REQUIRES = "llm"

logger = log.get_logger("llm.messages")

#: The node's name in messages.
NODE = "LLM Message"


class LLMMessage(io.ComfyNode):
    """One message appended to a conversation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMMessage",
            display_name="LLM Message",
            search_aliases=[
                "WASLLMMessage",
                "LLM Message",
                "ChatMessage",
                "chat message",
                "conversation",
                "system prompt",
                "few-shot example",
            ],
            category="WAS Suite/LLM/Messages",
            description=(
                "Add one message to a conversation: standing instructions, something said to the "
                "model, or an example answer, with pictures and sound if wanted. Chain several "
                "into LLM Chat or an agent. The panel shows the conversation so far."
            ),
            inputs=[
                io.Combo.Input(
                    "role", options=list(llm_messages.ROLES), default="user",
                    tooltip=(
                        "`user` = something said to the model; `system` = standing instructions, "
                        "such as a persona or rules; `assistant` = an answer the model gave, such "
                        "as an example of the style wanted."
                    ),
                ),
                io.String.Input(
                    "text", multiline=True, dynamic_prompts=True, default="",
                    tooltip="The message's words, such as `Describe this picture in one sentence.`",
                ),
                LLM_MESSAGES.Input(
                    "messages", optional=True,
                    tooltip=(
                        "The conversation so far, from another LLM Message. Left empty, this "
                        "message starts a new conversation."
                    ),
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip=(
                        "Pictures sent with the message, for a model that reads them. Every frame "
                        "of a batch becomes one picture, up to 64."
                    ),
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip=(
                        "Sound sent with the message, for a model that hears it. The first clip of "
                        "a batch is sent, up to 10 minutes."
                    ),
                ),
            ],
            outputs=[
                LLM_MESSAGES.Output(
                    display_name="messages",
                    tooltip=(
                        "The conversation with this message added last, for the next LLM Message, "
                        "LLM Chat or an agent."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, role, text, messages=None, image=None, audio=None) -> io.NodeOutput:
        """Append the message.

        Args:
            role: ``user``, ``system`` or ``assistant``.
            text: The message's words.
            messages: The conversation so far, or ``None``.
            image: Pictures to send with it, or ``None``.
            audio: Sound to send with it, or ``None``.

        Returns:
            The longer conversation.

        Raises:
            ValueError: The role is unknown, the message is empty, or the input is not a conversation.
        """
        if role not in llm_messages.ROLES:
            raise ValueError(
                f"{NODE} takes a role of {', '.join(llm_messages.ROLES)}, not {str(role)[:40]!r}."
            )
        conversation = (
            wires.Messages() if messages is None
            else wires.expect(messages, wires.Messages, NODE, "messages", "another LLM Message")
        )
        text = str(text or "")
        if not text.strip() and image is None and not audio:
            raise ValueError(f"{NODE} has nothing to add. Type its text, or connect an image or audio.")
        with runtime.guarded():
            item = llm_messages.message(role, text, images=image, audio=audio)
            result = conversation.plus(item)
            transcript = llm_messages.transcript(result.items)
        pictures = generate.pictures_in([item])
        sound = any(type(block).__name__ == "AudioBlock" for block in item.blocks)
        logger.debug("%s added a %s message; the conversation holds %d", NODE, role, len(result))
        extras = []
        if pictures:
            extras.append(f"{pictures} picture{'s' if pictures != 1 else ''}")
        if sound:
            extras.append("sound")
        run_result.publish(
            status=run_result.OK,
            summary=(
                f"Added {'an' if role[:1] in 'aeiou' else 'a'} {role} message"
                f"{' with ' + ' and '.join(extras) if extras else ''}; the "
                f"conversation holds {len(result)} message{'s' if len(result) != 1 else ''}."
            ),
            counts={"messages": len(result), "characters": len(text), "pictures": pictures},
            facts={"role": role, "sound": "yes" if sound else "no"},
            bodies=[run_result.body("conversation", transcript)],
        )
        return io.NodeOutput(result)
