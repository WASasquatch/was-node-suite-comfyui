"""Chat engines built around a memory, one turn of a chat session, and what a session holds.

:func:`make_builder` returns the callable a ``ChatEngine`` wire carries; :func:`open_session`
and :func:`chat_turn` are coroutines run inside one run's event loop.
"""

from __future__ import annotations

import json

from .. import log
from . import memory as memories
from . import messages, runtime, storage, wires
from .query import quiet, source_name

__all__ = [
    "CHAT_MODES",
    "CONTEXT_MODES",
    "TRANSCRIPT_MESSAGES",
    "all_messages",
    "chat_turn",
    "drop_thinking",
    "engine_model",
    "make_builder",
    "open_session",
    "restored_sources",
    "session_summary",
    "stored_sources",
    "transcript_lines",
    "turn_message",
]

logger = log.get_logger("llm.chat")

#: How a chat engine answers, as the widget offers them.
CHAT_MODES = ("simple", "context", "condense_question", "condense_plus_context")

#: The modes that retrieve chunks for every message.
CONTEXT_MODES = ("context", "condense_question", "condense_plus_context")

#: Newest messages a published transcript carries.
TRANSCRIPT_MESSAGES = 200

#: Characters of one message's text in a published transcript.
TRANSCRIPT_TEXT_CHARS = 4000

#: Characters of a whole published transcript.
TRANSCRIPT_TOTAL_CHARS = 480000

#: Retrieved chunks kept with a saved turn.
STORED_SOURCES = 10

#: Characters of one retrieved chunk kept with a saved turn.
STORED_SOURCE_CHARS = 2000

#: Roles a transcript names, from llama-index's roles.
_ROLES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
    "chatbot": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
}


def make_builder(
    mode: str,
    model: wires.LanguageModel,
    retriever: wires.Retriever | None = None,
    system_prompt: str = "",
    postprocessors: tuple = (),
):
    """The callable that builds a chat engine around a memory.

    Args:
        mode: One of :data:`CHAT_MODES`.
        model: The model answering.
        retriever: Where a context mode finds chunks; required by :data:`CONTEXT_MODES`.
        system_prompt: Instructions for every answer, empty for none; not read by
            ``condense_question``.
        postprocessors: Steps run over the chunks a context mode retrieves.

    Returns:
        A function taking a llama-index memory and returning the llama-index chat engine.
    """
    processors = [step.processor for step in postprocessors]
    prompt = str(system_prompt or "").strip() or None

    def build(memory):
        """The chat engine around ``memory``."""
        runtime.core()
        quiet()
        from llama_index.core.chat_engine import (
            CondensePlusContextChatEngine,
            CondenseQuestionChatEngine,
            ContextChatEngine,
            SimpleChatEngine,
        )

        if mode == "simple":
            return SimpleChatEngine.from_defaults(llm=model.llm, memory=memory, system_prompt=prompt)
        if mode == "context":
            return ContextChatEngine.from_defaults(
                retriever=retriever.retriever, llm=model.llm, memory=memory,
                system_prompt=prompt, node_postprocessors=processors,
            )
        if mode == "condense_question":
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.response_synthesizers import get_response_synthesizer

            engine = RetrieverQueryEngine(
                retriever=retriever.retriever,
                response_synthesizer=get_response_synthesizer(llm=model.llm),
                node_postprocessors=processors,
            )
            return CondenseQuestionChatEngine.from_defaults(
                query_engine=engine, memory=memory, llm=model.llm,
            )
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever.retriever, llm=model.llm, memory=memory,
            system_prompt=prompt, node_postprocessors=processors,
        )

    return build


def engine_model(engine: wires.ChatEngine) -> wires.LanguageModel | None:
    """The first language model a chat engine wire calls."""
    for found in wires.models_in(engine):
        if isinstance(found, wires.LanguageModel):
            return found
    return None


async def open_session(spec: memories.MemorySpec, llm: wires.LanguageModel | None = None):
    """A llama-index memory holding a session, its archived messages and vector blocks included.

    Args:
        spec: The memory to open.
        llm: The conversation's model, for a ``facts`` block that names none.

    Returns:
        The memory.

    Raises:
        ValueError: The session name cannot be used, its file cannot be read, or a block lacks
            its model.
    """
    return await memories.open_memory(spec, llm)


def turn_message(text: str, image, mode: str, node: str):
    """What a chat engine is handed for one message.

    Args:
        text: The message's words.
        image: An ``IMAGE`` batch sent with it, or ``None``.
        mode: The chat engine's mode.
        node: The node's display name, for the message.

    Returns:
        The text alone, or for pictures a list of llama-index content blocks.

    Raises:
        ValueError: Pictures were connected and the mode sends text only.
    """
    if image is None:
        return text
    if mode != "simple":
        raise ValueError(
            f"{node}: the {mode} chat mode hands the model only the text of a message, so the "
            f"picture would never be seen. Set LLM Chat Engine's mode to simple to talk about "
            f"pictures, or disconnect the image input."
        )
    from llama_index.core.llms import TextBlock

    blocks = [TextBlock(text=text)] if text else []
    return blocks + messages.image_blocks(image)


async def chat_turn(engine, payload, stream: runtime.TextStream) -> dict:
    """Send one message to a chat engine and stream the reply.

    Args:
        engine: A llama-index chat engine around an open memory.
        payload: What :func:`turn_message` returned.
        stream: Where the reply is drawn as it is written.

    Returns:
        ``{"raw", "reply", "thinking", "nodes"}``: the whole reply, the reply without inline
        thinking, that thinking, and the ``NodeWithScore`` chunks it answered from.
    """
    import asyncio

    try:
        response = await engine.astream_chat(payload)
    except ValueError as error:
        if "Streaming is not enabled" not in str(error):
            raise
        response = await engine.achat(payload)
    text = ""
    if hasattr(response, "async_response_gen") and getattr(response, "achat_stream", None) is not None:
        generator = response.async_response_gen()
        finished = False
        try:
            async for delta in generator:
                runtime.check_interrupt()
                text += delta or ""
                stream.write(delta or "")
            finished = True
        finally:
            if not finished:
                task = response.awrite_response_to_history_task
                if task is not None and not task.done():
                    task.cancel()
                try:
                    await generator.aclose()
                except (asyncio.CancelledError, Exception) as error:
                    logger.debug("a cancelled reply stream closed with %s", type(error).__name__)
    else:
        text = str(getattr(response, "response", "") or "")
        stream.write(text)
    stream.finish()
    raw = text.strip()
    reply, thinking = messages.split_thinking(raw, opened=messages.closes_only(raw))
    return {"raw": raw, "reply": reply, "thinking": thinking, "nodes": _nodes_of(response)}


def _nodes_of(response) -> tuple:
    """The retrieved chunks a chat response answered from."""
    nodes = list(getattr(response, "source_nodes", None) or ())
    if not nodes:
        for output in getattr(response, "sources", None) or ():
            raw = getattr(output, "raw_output", None)
            found = getattr(raw, "source_nodes", None)
            if found is None and isinstance(raw, list):
                found = raw
            nodes.extend(found or ())
    return tuple(item for item in nodes if hasattr(item, "node"))


async def drop_thinking(memory, raw: str, reply: str) -> None:
    """Replace the newest assistant message's text with the reply alone, keeping archived ones.

    Args:
        memory: The open memory.
        raw: The reply as written, thinking included.
        reply: The reply without its thinking.
    """
    if raw == reply:
        return
    from llama_index.core.llms import TextBlock
    from llama_index.core.storage.chat_store.sql import MessageStatus

    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    if not active or getattr(active[-1].role, "value", "") != "assistant":
        return
    if messages.text_of(active[-1]).strip() != raw:
        return
    active[-1] = active[-1].model_copy(update={"blocks": [TextBlock(text=reply)]})
    archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
    await memory.sql_store.set_messages(memory.session_id, active, status=MessageStatus.ACTIVE)
    if archived:
        await memory.sql_store.add_messages(memory.session_id, archived, status=MessageStatus.ARCHIVED)


async def all_messages(memory) -> list:
    """Every message an open memory holds, archived ones first, oldest first."""
    from llama_index.core.storage.chat_store.sql import MessageStatus

    archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    return list(archived) + list(active)


def transcript_lines(items) -> str:
    """The newest messages as JSON Lines, oldest first, for a chat panel.

    Args:
        items: llama-index ``ChatMessage`` objects, oldest first.

    Returns:
        One ``{"role", "text", "pictures"}`` object per line, at most
        :data:`TRANSCRIPT_MESSAGES` lines and :data:`TRANSCRIPT_TOTAL_CHARS` characters.
    """
    from llama_index.core.llms import ImageBlock

    lines: list[str] = []
    total = 0
    for item in reversed(list(items or ())[-TRANSCRIPT_MESSAGES:]):
        role = getattr(item.role, "value", str(item.role))
        text = messages.text_of(item).strip()
        if len(text) > TRANSCRIPT_TEXT_CHARS:
            text = text[: TRANSCRIPT_TEXT_CHARS - 3] + "..."
        pictures = sum(isinstance(block, ImageBlock) for block in item.blocks)
        line = json.dumps(
            {"role": _ROLES.get(role, "assistant"), "text": text, "pictures": pictures},
            ensure_ascii=False,
        )
        if total + len(line) + 1 > TRANSCRIPT_TOTAL_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(reversed(lines))


def stored_sources(nodes) -> list[dict]:
    """Retrieved chunks as JSON-ready rows kept with a saved turn."""
    rows = []
    for item in list(nodes or ())[:STORED_SOURCES]:
        try:
            text = item.node.get_content()
        except Exception:
            text = ""
        score = getattr(item, "score", None)
        rows.append(
            {
                "text": text[:STORED_SOURCE_CHARS],
                "score": float(score) if isinstance(score, (int, float)) else None,
                "name": source_name(item),
            }
        )
    return rows


def restored_sources(rows) -> tuple:
    """``NodeWithScore`` objects read back from :func:`stored_sources` rows."""
    from llama_index.core.schema import NodeWithScore, TextNode

    found = []
    for row in rows or ():
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            continue
        name = row.get("name") if isinstance(row.get("name"), str) else ""
        score = row.get("score") if isinstance(row.get("score"), (int, float)) else None
        node = TextNode(text=row["text"][:STORED_SOURCE_CHARS], metadata={"file_name": name} if name else {})
        found.append(NodeWithScore(node=node, score=score))
    return tuple(found)


def session_summary(name: str) -> dict | None:
    """What a saved session holds, read from its file.

    Args:
        name: The session's name.

    Returns:
        ``{"messages", "archived", "facts", "updated", "vectors"}``, or ``None`` when nothing
        is saved under the name.

    Raises:
        ValueError: The name cannot be used, or the file is too large or not JSON.
    """
    saved = storage.read_session(name)
    if saved is None:
        return None
    facts = saved.get("facts") if isinstance(saved.get("facts"), dict) else {}
    return {
        "messages": len(saved.get("active") or ()) if isinstance(saved.get("active"), list) else 0,
        "archived": len(saved.get("archived") or ()) if isinstance(saved.get("archived"), list) else 0,
        "facts": sum(len(values) for values in facts.values() if isinstance(values, list)),
        "updated": str(saved.get("updated") or "")[:32],
        "vectors": len(saved.get("vectors") or ()) if isinstance(saved.get("vectors"), list) else 0,
    }
