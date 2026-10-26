"""Conversation memory kept between runs under a session name.

:func:`open_memory` builds a llama-index ``Memory`` holding a saved session, and
:func:`save_memory` writes it back. Both are coroutines: a memory lives inside one run's event
loop.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import messages, runtime, storage
from .wires import EmbeddingModel, LanguageModel, models_in

__all__ = [
    "BLOCK_KINDS",
    "BlockSpec",
    "DEFAULT_TOKEN_LIMIT",
    "MAX_ARCHIVED",
    "MemorySpec",
    "close_memory",
    "ephemeral",
    "history",
    "last_turn",
    "open_memory",
    "save_memory",
    "text_vector_store_class",
    "turn_key",
    "vector_file",
]

#: Tokens of conversation and memory blocks handed to the model at most, by default.
DEFAULT_TOKEN_LIMIT = 30000

#: Messages that have left the model's view kept in a session file for the transcript.
MAX_ARCHIVED = 2000

#: The kinds of long-term memory block.
BLOCK_KINDS = ("static", "facts", "vector")

#: The version written into a session file.
SESSION_VERSION = 1


@dataclass(frozen=True, eq=False)
class BlockSpec:
    """One long-term memory block.

    Attributes:
        kind: ``static`` always shows ``text``; ``facts`` keeps facts an LLM pulls out of
            older messages; ``vector`` finds older messages similar to the newest one.
        name: The block's name, unique within a memory.
        priority: 0 is never cut when the memory is over its limit; higher numbers are cut first.
        text: What a ``static`` block shows.
        llm: The model a ``facts`` block extracts with, or ``None`` for the conversation's own.
        max_facts: Facts a ``facts`` block keeps before condensing them.
        embedding: The model a ``vector`` block embeds with.
        top_k: Older messages a ``vector`` block brings back.
    """

    kind: str
    name: str
    priority: int = 0
    text: str = ""
    llm: LanguageModel | None = None
    max_facts: int = 50
    embedding: EmbeddingModel | None = None
    top_k: int = 2

    def models(self) -> tuple:
        return models_in(self.llm, self.embedding)


@dataclass(frozen=True, eq=False)
class MemorySpec:
    """Where a conversation is kept and how much of it reaches the model.

    Attributes:
        session: The name it is saved under, ``""`` to keep it for one run only.
        token_limit: Tokens of conversation and blocks handed to the model at most.
        history_ratio: Share of ``token_limit`` kept for recent messages; older ones move to
            the blocks.
        insert: Where block content goes: ``system`` into the system prompt, ``user`` into the
            newest message.
        blocks: Long-term memory blocks.
    """

    session: str = ""
    token_limit: int = DEFAULT_TOKEN_LIMIT
    history_ratio: float = 0.7
    insert: str = "system"
    blocks: tuple = field(default_factory=tuple)

    def models(self) -> tuple:
        return models_in(*self.blocks)


def ephemeral() -> MemorySpec:
    """A memory kept for one run, used where no memory is connected."""
    return MemorySpec()


async def open_memory(spec: MemorySpec | None, llm: LanguageModel | None = None):
    """A llama-index ``Memory`` holding the session a spec names.

    Args:
        spec: The memory to open, ``None`` for :func:`ephemeral`.
        llm: The conversation's model, used by a ``facts`` block that names none.

    Returns:
        The memory, empty when the session has not been saved yet.

    Raises:
        ValueError: The session name cannot be used, or its file cannot be read.
    """
    spec = spec or ephemeral()
    runtime.core()
    from llama_index.core.memory import Memory
    from llama_index.core.memory.memory import InsertMethod
    from llama_index.core.storage.chat_store.sql import MessageStatus

    saved = storage.read_session(spec.session) if spec.session else None
    saved = saved or {}
    memory = Memory.from_defaults(
        session_id=spec.session or "this run",
        token_limit=int(spec.token_limit),
        chat_history_token_ratio=float(spec.history_ratio),
        memory_blocks=_blocks(spec, saved, llm),
        insert_method=InsertMethod.USER if spec.insert == "user" else InsertMethod.SYSTEM,
    )
    archived = messages.from_dicts(saved.get("archived"))
    active = messages.from_dicts(saved.get("active"))
    # The active messages are set first, then the archived ones are added beside them.
    if active:
        await memory.aset(active)
    if archived:
        await memory.sql_store.add_messages(memory.session_id, archived, status=MessageStatus.ARCHIVED)
    return memory


def _blocks(spec: MemorySpec, saved: dict, llm: LanguageModel | None) -> list:
    """The llama-index memory blocks a spec asks for, holding what the session saved."""
    from llama_index.core.memory import FactExtractionMemoryBlock, StaticMemoryBlock, VectorMemoryBlock

    facts = saved.get("facts") if isinstance(saved.get("facts"), dict) else {}
    built = []
    for block in spec.blocks:
        if block.kind == "static":
            built.append(
                StaticMemoryBlock(name=block.name, priority=block.priority, static_content=block.text)
            )
        elif block.kind == "facts":
            model = block.llm or llm
            if model is None:
                raise ValueError(
                    f"The facts memory block {block.name!r} needs a language model. Connect one to "
                    f"its llm input, or to the node the memory is used on."
                )
            kept = [str(fact) for fact in facts.get(block.name, []) if isinstance(fact, str)]
            built.append(
                FactExtractionMemoryBlock(
                    name=block.name, priority=block.priority, llm=model.ready().llm,
                    facts=kept, max_facts=int(block.max_facts),
                )
            )
        elif block.kind == "vector":
            if block.embedding is None:
                raise ValueError(
                    f"The vector memory block {block.name!r} needs an embedding model. Connect "
                    f"LLM Embedding Model to its embedding input."
                )
            built.append(
                VectorMemoryBlock(
                    name=block.name, priority=block.priority,
                    vector_store=_vector_store(spec.session, block.name),
                    embed_model=block.embedding.ready().model,
                    similarity_top_k=int(block.top_k),
                )
            )
    return built


_text_store = None


def text_vector_store_class():
    """A ``SimpleVectorStore`` that keeps each node's text beside its embedding.

    Returns:
        The class, built once. A vector memory block takes only a store that keeps text.
    """
    global _text_store
    if _text_store is not None:
        return _text_store
    runtime.core()
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.vector_stores.types import VectorStoreQueryResult
    from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

    class TextVectorStore(SimpleVectorStore):
        """A simple vector store answering queries with the nodes it holds."""

        stores_text: bool = True

        def add(self, nodes, **add_kwargs):
            ids = super().add(nodes, **add_kwargs)
            for node in nodes:
                self.data.metadata_dict[node.node_id] = node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=False
                )
            return ids

        def query(self, query, **kwargs):
            found = super().query(query, **kwargs)
            nodes, scores, ids = [], [], []
            for node_id, score in zip(found.ids or (), found.similarities or ()):
                held = self.data.metadata_dict.get(node_id)
                if not isinstance(held, dict) or "_node_content" not in held:
                    continue
                nodes.append(metadata_dict_to_node(held))
                scores.append(score)
                ids.append(node_id)
            return VectorStoreQueryResult(nodes=nodes, similarities=scores, ids=ids)

    _text_store = TextVectorStore
    return _text_store


def vector_file(session: str, block: str) -> str:
    """The file name a session's ``vector`` block is saved under."""
    return f"{storage.check_name(session, 'Session')}.{_safe(block)}.vectors.json"


def _vector_store(session: str, name: str):
    """The vector store a ``vector`` block keeps, read back from the session's vector file."""
    kind = text_vector_store_class()
    if session:
        path = storage.folder(storage.SESSIONS) / vector_file(session, name)
        if path.is_file() and os.path.getsize(path) <= storage.MAX_SESSION_BYTES:
            try:
                return kind.from_persist_path(str(path))
            except (ValueError, KeyError, TypeError) as error:
                runtime.logger.debug("a vector memory file was not readable (%s)", error)
    return kind()


def _safe(name: str) -> str:
    """A block name as one part of a file name."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:48] or "block"


def turn_key(text: str = "", images=None, extra: str = "") -> str:
    """A digest naming one new turn: its words, its pictures and anything else that shapes it.

    Args:
        text: The new message.
        images: An ``IMAGE`` batch sent with it, or ``None``.
        extra: Further text the answer depends on, such as the agent names.

    Returns:
        A hex digest, the same for the same turn.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(text or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(extra or "").encode("utf-8"))
    if images is not None:
        digest.update(b"\0")
        digest.update(str(tuple(images.shape)).encode("ascii"))
        digest.update(images.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def last_turn(spec: MemorySpec | None) -> dict | None:
    """The newest turn a saved session answered, or ``None``.

    Returns:
        ``{"key", "reply", "thinking"}`` and any other fields saved with it; ``None`` for an
        unsaved or one-run session.
    """
    spec = spec or ephemeral()
    if not spec.session:
        return None
    saved = storage.read_session(spec.session) or {}
    turn = saved.get("turn")
    return turn if isinstance(turn, dict) and isinstance(turn.get("key"), str) else None


async def save_memory(spec: MemorySpec | None, memory, turn: dict | None = None) -> dict:
    """Write a memory back under its session name, then close it.

    Args:
        spec: The spec the memory was opened from.
        memory: What :func:`open_memory` returned.
        turn: The turn just answered, ``{"key": turn_key(...), "reply": ..., "thinking": ...}``
            plus any JSON-ready fields, read back by :func:`last_turn`. Left out, the session
            keeps the turn it held.

    Returns:
        Counts for a panel: ``active``, ``archived`` and ``facts``.
    """
    spec = spec or ephemeral()
    from llama_index.core.memory import FactExtractionMemoryBlock, VectorMemoryBlock
    from llama_index.core.storage.chat_store.sql import MessageStatus

    try:
        active = await memory.aget_all(status=MessageStatus.ACTIVE)
        archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
        facts = {
            block.name: list(block.facts)
            for block in memory.memory_blocks
            if isinstance(block, FactExtractionMemoryBlock)
        }
        counts = {
            "active": len(active),
            "archived": len(archived),
            "facts": sum(len(values) for values in facts.values()),
        }
        if spec.session:
            if turn is None:
                previous = storage.read_session(spec.session) or {}
                turn = previous.get("turn") if isinstance(previous.get("turn"), dict) else None
            vectors = []
            for block in memory.memory_blocks:
                if isinstance(block, VectorMemoryBlock):
                    path = storage.writable(storage.SESSIONS, vector_file(spec.session, block.name))
                    block.vector_store.persist(persist_path=str(path))
                    vectors.append(path.name)
            storage.write_session(
                spec.session,
                {
                    "version": SESSION_VERSION,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "active": messages.to_dicts(messages.storable(active)),
                    "archived": messages.to_dicts(messages.storable(archived[-MAX_ARCHIVED:])),
                    "facts": facts,
                    "vectors": vectors,
                    "turn": turn,
                },
            )
        return counts
    finally:
        await close_memory(memory)


async def history(spec: MemorySpec | None) -> list:
    """Every message a saved session holds, oldest first, without opening a model.

    Returns:
        llama-index ``ChatMessage`` objects; empty for an unsaved or one-run session.
    """
    spec = spec or ephemeral()
    if not spec.session:
        return []
    saved = storage.read_session(spec.session) or {}
    return messages.from_dicts(saved.get("archived")) + messages.from_dicts(saved.get("active"))


async def close_memory(memory) -> None:
    """Close a memory without saving it, releasing the database engine it opened."""
    store = getattr(memory, "sql_store", None)
    engine = getattr(store, "_async_engine", None)
    if engine is not None:
        try:
            await engine.dispose()
        except Exception as error:
            runtime.logger.debug("a memory's database did not close cleanly (%s)", error)
