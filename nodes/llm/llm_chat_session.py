"""Hold a conversation with a chat engine, one message per run, kept between runs by a memory."""

from __future__ import annotations

import dataclasses
import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_CHAT_ENGINE, LLM_MEMORY, LLM_MESSAGES, LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import chat, messages, query, runtime, storage, wires
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.chat")


class WASLLMChatSession(io.ComfyNode):
    """Send one message to a chat engine and keep the conversation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMChatSession",
            display_name="LLM Chat Session",
            search_aliases=[
                "WASLLMChatSession",
                "LLM Chat Session",
                "chat",
                "conversation",
                "chatbot",
                "astream_chat",
            ],
            category="WAS Suite/LLM/Chat",
            is_output_node=True,
            description=(
                "Chat with a model one message per run, the reply streaming into the panel. "
                "Each conversation is saved under its name and carries on in the next run. "
                "The panel's conversation menu starts, opens, renames and deletes them."
            ),
            inputs=[
                LLM_CHAT_ENGINE.Input(
                    "chat_engine",
                    tooltip="How each message is answered, from LLM Chat Engine.",
                ),
                io.String.Input(
                    "message", multiline=True, default="",
                    tooltip="What to say next, such as `And what about the second chapter?` Empty sends nothing and shows the saved conversation.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Pictures sent with the message, for a model that reads them. Only the simple chat mode passes pictures on.",
                ),
                LLM_MEMORY.Input(
                    "memory", optional=True,
                    tooltip="Where the conversation is kept, from LLM Memory. Without it the conversation lasts this run only.",
                ),
                io.Boolean.Input(
                    "reset", default=False, optional=True,
                    tooltip="`true` deletes the saved conversation before this message, starting it over; `false` carries on.",
                ),
                io.String.Input(
                    "conversation", default="", optional=True,
                    tooltip=(
                        "The saved conversation this message joins, such as `project notes`; a new name starts one. "
                        "Empty uses the session of LLM Memory, or keeps the conversation for this run only."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="reply",
                    tooltip="The model's answer to this message, its thinking removed.",
                ),
                io.String.Output(
                    display_name="thinking",
                    tooltip="What a reasoning model wrote between think tags before answering. Empty otherwise.",
                ),
                io.String.Output(
                    display_name="history",
                    tooltip="The whole conversation as text, one `role: text` paragraph per message, oldest first.",
                ),
                LLM_MESSAGES.Output(
                    display_name="messages",
                    tooltip="The whole conversation as messages, for LLM Chat, the agents or another LLM Message.",
                ),
                LLM_SOURCES.Output(
                    display_name="sources",
                    tooltip="The chunks this reply was written from, for the evaluators and LLM Sources to Text. Empty in the simple mode.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls, chat_engine, message, image=None, memory=None, reset=False, conversation="",
    ) -> io.NodeOutput:
        """Answer one message and keep the conversation.

        Args:
            chat_engine: How the message is answered.
            message: What to say.
            image: Pictures sent with it.
            memory: Where the conversation is kept.
            reset: Whether the saved session is deleted first.
            conversation: The saved conversation's name, ahead of the memory's session.

        Returns:
            The reply, its thinking, the transcript, the messages and the sources.
        """
        node = "LLM Chat Session"
        engine = wires.expect(chat_engine, wires.ChatEngine, node, "chat_engine", "LLM Chat Engine")
        if memory is None:
            spec = memories.ephemeral()
        else:
            spec = wires.expect(memory, memories.MemorySpec, node, "memory", "LLM Memory")
        if str(conversation or "").strip():
            spec = dataclasses.replace(spec, session=storage.check_name(conversation, "Conversation"))
        text = str(message or "")
        model = chat.engine_model(engine)
        key = memories.turn_key(text, image, extra=f"{engine.mode}|{engine.label}")
        held = memories.last_turn(spec) if spec.session else None
        if held is not None and held.get("key") == key:
            return await cls._replay(spec, engine, model, held, text, "replayed")
        cleared = False
        if reset and spec.session:
            cleared = storage.delete_session(spec.session)
            logger.info("LLM Chat Session cleared session %s", spec.session)
            held = None
        if not text.strip() and image is None:
            return await cls._replay(spec, engine, model, held, text, "cleared" if cleared else "shown")
        wires.ready(engine, spec)
        payload = chat.turn_message(text, image, engine.mode, node)
        started = time.monotonic()
        with runtime.guarded():
            opened = await chat.open_session(spec, model)
            try:
                built = engine.build(opened)
                result = await chat.chat_turn(built, payload, runtime.TextStream(""))
                await chat.drop_thinking(opened, result["raw"], result["reply"])
                everything = await chat.all_messages(opened)
            except BaseException as error:
                await memories.close_memory(opened)
                if isinstance(error, Exception):
                    raise query.failure(node, error) from None
                raise
            turn = {
                "key": key,
                "reply": result["reply"],
                "thinking": result["thinking"],
                "sources": chat.stored_sources(result["nodes"]),
                "mode": engine.mode,
            }
            counts = await memories.save_memory(spec, opened, turn=turn)
        seconds = time.monotonic() - started
        logger.info(
            "LLM Chat Session answered in %.2fs with %d chunk(s); %d message(s) active, %d archived%s",
            seconds, len(result["nodes"]), counts["active"], counts["archived"],
            f" in session {spec.session}" if spec.session else "",
        )
        line, status = f"Answered in {seconds:.1f}s", run_result.OK
        if image is not None and model is not None and not model.vision:
            line = f"{line}; {model.label} is not marked as reading pictures, so it may not have seen them"
            status = run_result.WARNING
        return cls._answer(
            spec, engine, model, result["reply"], result["thinking"], everything, result["nodes"],
            text, counts, line, status, seconds,
        )

    @classmethod
    async def _replay(cls, spec, engine, model, held, text, what) -> io.NodeOutput:
        """The newest saved turn and the conversation, without calling the model."""
        everything = await memories.history(spec)
        held = held or {}
        reply = str(held.get("reply") or "")
        thinking = str(held.get("thinking") or "")
        nodes = chat.restored_sources(held.get("sources"))
        summary = chat.session_summary(spec.session) if spec.session else None
        counts = {
            "active": (summary or {}).get("messages", 0),
            "archived": (summary or {}).get("archived", 0),
            "facts": (summary or {}).get("facts", 0),
        }
        if what == "replayed":
            line, status = "This message was answered already; showing the saved reply", run_result.OK
        elif what == "cleared":
            line, status = "The session was cleared and no message was sent", run_result.OK
        elif spec.session:
            line, status = "No message was sent; showing the saved conversation", run_result.OK
        else:
            line, status = "No message was sent and no conversation is named, so there is nothing to show", run_result.WARNING
        return cls._answer(spec, engine, model, reply, thinking, everything, nodes, text, counts, line, status, None)

    @classmethod
    def _answer(cls, spec, engine, model, reply, thinking, everything, nodes, text, counts, line, status, seconds):
        """Publish the run result and build the outputs."""
        transcript = chat.transcript_lines(everything)
        start = run_result.window(len(transcript), len(transcript))[0]
        if start:
            start = transcript.find("\n", start - 1) + 1 or start
        bodies = [run_result.body("reply", reply)]
        if thinking:
            bodies.append(run_result.body("thinking", thinking))
        bodies.append(run_result.body("transcript", transcript, start=start))
        figures = {
            "messages saved": counts.get("active", 0),
            "archived": counts.get("archived", 0),
            "facts": counts.get("facts", 0),
            "sources": len(nodes),
        }
        if seconds is not None:
            figures["seconds"] = round(seconds, 2)
        run_result.publish(
            status=status,
            summary=line,
            counts=figures,
            facts={
                "session": spec.session or "this run only",
                "model": model.label if model is not None else "",
                "mode": engine.mode,
            },
            items=query.source_rows(nodes),
            items_total=len(nodes),
            bodies=bodies,
        )
        sources = wires.Sources(items=tuple(nodes), query=text, answer=reply)
        return io.NodeOutput(
            reply, thinking, messages.transcript(everything), wires.Messages(tuple(everything)), sources,
        )
