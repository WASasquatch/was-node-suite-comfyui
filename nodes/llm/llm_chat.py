"""Answer the next turn of a conversation, kept in a memory between runs when one is connected."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_MEMORY, LLM_MESSAGES
from ...modules.interface import run_result
from ...modules.llm import generate, runtime, wires
from ...modules.llm import memory as llm_memory
from ...modules.llm import messages as llm_messages

REQUIRES = "llm"

logger = log.get_logger("llm.chat")

#: The node's name in messages.
NODE = "LLM Chat"

#: The nodes that make a language model, for messages.
LLM_HINT = "LLM From CLIP, LLM GGUF Loader or LLM OpenAI"


class LLMChat(io.ComfyNode):
    """One chat turn answered by a language model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMChat",
            display_name="LLM Chat",
            search_aliases=[
                "WASLLMChat",
                "LLM Chat",
                "chat",
                "astream_chat",
                "conversation",
                "chatbot",
                "Memory",
            ],
            category="WAS Suite/LLM/Generate",
            is_output_node=True,
            description=(
                "Talk with a language model one turn at a time. Connect a memory to keep the "
                "conversation between runs, so each queue adds a turn. The reply streams onto the "
                "node, and the conversation comes out for the next node."
            ),
            inputs=[
                LLM.Input(
                    "llm",
                    tooltip=(
                        "The model that answers, from LLM From CLIP, LLM GGUF Loader, LLM OpenAI "
                        "or another LLM model node."
                    ),
                ),
                io.String.Input(
                    "message", multiline=True, default="", optional=True,
                    tooltip=(
                        "What to say next, such as `And in winter?`. With a saved memory the same "
                        "message twice in a row is answered once, so change it to ask again; left "
                        "empty, nothing is asked and the newest reply comes back."
                    ),
                ),
                LLM_MESSAGES.Input(
                    "messages", optional=True,
                    tooltip=(
                        "The conversation before this turn, from LLM Message, such as a system "
                        "message setting a persona. With a memory, messages it already holds are "
                        "not added again."
                    ),
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip=(
                        "Pictures sent with the message, for a model that reads them. Every frame "
                        "of a batch is sent, up to 64."
                    ),
                ),
                LLM_MEMORY.Input(
                    "memory", optional=True,
                    tooltip=(
                        "Where the conversation is kept, from LLM Memory. A session name keeps it "
                        "between runs; left empty, the chat starts fresh every run."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="reply",
                    tooltip="The model's answer to this turn, thinking removed.",
                ),
                io.String.Output(
                    display_name="thinking",
                    tooltip=(
                        "What a reasoning model thought before answering. Empty for a model that "
                        "does not think aloud."
                    ),
                ),
                LLM_MESSAGES.Output(
                    display_name="messages",
                    tooltip=(
                        "The whole conversation with the reply last, for LLM Messages to Text or "
                        "another LLM Chat."
                    ),
                ),
            ],
        )

    @classmethod
    async def execute(cls, llm, message="", messages=None, image=None, memory=None) -> io.NodeOutput:
        """Answer the turn, through the memory when one is connected.

        Args:
            llm: The ``LanguageModel`` that answers.
            message: What to say next, ``""`` for nothing.
            messages: The conversation before this turn, or ``None``.
            image: Pictures sent with the message, or ``None``.
            memory: A ``MemorySpec``, or ``None``.

        Returns:
            The reply, the thinking before it and the conversation.

        Raises:
            ValueError: Nothing to send, a wrong wire, or pictures for a model that does not read them.
            RuntimeError: The model failed.
        """
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", LLM_HINT)
        given = (
            wires.Messages() if messages is None
            else wires.expect(messages, wires.Messages, NODE, "messages", "LLM Message")
        )
        spec = (
            None if memory is None
            else wires.expect(memory, llm_memory.MemorySpec, NODE, "memory", "LLM Memory")
        )
        wires.ready(model, spec)
        text = str(message or "")
        pictures = (int(image.shape[0]) if image.ndim == 4 else 1) if image is not None else 0
        with runtime.guarded():
            generate.require_vision(model, NODE, pictures + generate.pictures_in(given.items))
            turn = await generate.converse(model, given.items, text, image, spec, NODE)
            transcript = llm_messages.transcript(turn.conversation)
        conversation = wires.Messages(tuple(turn.conversation))
        logger.info(
            "%s: %s, %d characters in %.2fs; the conversation holds %d messages",
            NODE, turn.source, len(turn.reply), turn.seconds, len(conversation),
        )
        if turn.source == "stored" and text.strip():
            summary = "The same message was answered on the last run, so its stored reply came back."
        elif turn.source == "stored":
            summary = "Nothing was asked, so the newest stored reply came back."
        elif turn.source == "none":
            summary = "Nothing was asked and the session holds no reply yet. Type a message."
        elif not turn.reply.strip():
            summary = "The model answered nothing."
        else:
            summary = f"Answered in {turn.seconds:.1f}s; the conversation holds {len(conversation)} messages."
        facts = {"model": model.label, "memory": "none"}
        if spec is not None:
            facts["memory"] = spec.session or "this run only"
        counts = {"messages": len(conversation), "characters": len(turn.reply), "seconds": round(turn.seconds, 2)}
        for name in ("active", "archived", "facts"):
            if name in turn.stored:
                counts[name] = turn.stored[name]
        bodies = [run_result.body("reply", turn.reply)]
        if turn.thinking:
            bodies.append(run_result.body("thinking", turn.thinking))
        bodies.append(run_result.body("conversation", transcript))
        run_result.publish(
            status=run_result.WARNING if turn.source == "none" or not turn.reply.strip() else run_result.OK,
            summary=summary,
            counts=counts,
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(turn.reply, turn.thinking, conversation)
