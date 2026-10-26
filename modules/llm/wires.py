"""What the LLM sockets carry.

Each class wraps a llama-index object with what a node reports about it. :func:`expect` checks
a wire's value before a node uses it, and :func:`ready` gates every hosted model it reaches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from . import runtime

__all__ = [
    "AgentSpec",
    "ChatEngine",
    "Documents",
    "EmbeddingModel",
    "Index",
    "LanguageModel",
    "Messages",
    "Postprocessor",
    "QueryEngine",
    "Retriever",
    "Sources",
    "chain",
    "every",
    "expect",
    "models_in",
    "ready",
]


@dataclass(frozen=True, eq=False)
class LanguageModel:
    """A language model on the ``WAS_LLM`` wire.

    Attributes:
        llm: The llama-index LLM.
        label: What panels and errors call it, such as ``"claude-opus-5-5 on Anthropic"``.
        provider: The family that built it: ``openai``, ``anthropic``, ``gemini``, ``provider``,
            ``ollama``, ``clip``, ``gguf`` or ``transformers``.
        local: True when it runs on this machine, False when calling it leaves the machine.
        function_calling: True when it calls tools natively.
        vision: True when it reads pictures.
        context_window: The most tokens it reads at once.
        release: Frees what the model holds in memory, or ``None`` when it holds nothing.
    """

    llm: Any
    label: str
    provider: str
    local: bool
    function_calling: bool = False
    vision: bool = False
    context_window: int = 4096
    release: Callable[[], None] | None = None

    def models(self) -> tuple:
        """This model, as :func:`models_in` collects them."""
        return (self,)

    def ready(self) -> "LanguageModel":
        """This model, once a hosted one is allowed to reach the network.

        Raises:
            RuntimeError: The model is hosted and ``features.network`` is off.
        """
        if not self.local:
            runtime.require_network(self.label)
        return self

    def with_llm(self, llm) -> "LanguageModel":
        """The same model description around another llama-index LLM."""
        return replace(self, llm=llm)


@dataclass(frozen=True, eq=False)
class EmbeddingModel:
    """An embedding model on the ``WAS_LLM_EMBEDDING`` wire.

    Attributes:
        model: The llama-index embedding model.
        label: What panels and errors call it.
        provider: The family that built it: ``transformers``, ``gguf``, ``openai``, ``gemini``,
            ``ollama`` or ``provider``.
        local: True when it runs on this machine.
        dimensions: Length of one vector, 0 while unknown.
        release: Frees what the model holds in memory, or ``None``.
        pictures: True when it embeds pictures as well as text, into the same space.
    """

    model: Any
    label: str
    provider: str
    local: bool
    dimensions: int = 0
    release: Callable[[], None] | None = None
    pictures: bool = False

    def models(self) -> tuple:
        """This model, as :func:`models_in` collects them."""
        return (self,)

    def ready(self) -> "EmbeddingModel":
        """This model, once a hosted one is allowed to reach the network.

        Raises:
            RuntimeError: The model is hosted and ``features.network`` is off.
        """
        if not self.local:
            runtime.require_network(self.label)
        return self


@dataclass(frozen=True, eq=False)
class Messages:
    """A conversation on the ``WAS_LLM_MESSAGES`` wire.

    Attributes:
        items: llama-index ``ChatMessage`` objects, oldest first.
    """

    items: tuple = ()

    def __len__(self) -> int:
        return len(self.items)

    def plus(self, *messages) -> "Messages":
        """This conversation with messages appended."""
        return Messages(self.items + tuple(m for m in messages if m is not None))

    def models(self) -> tuple:
        return ()


@dataclass(frozen=True, eq=False)
class Documents:
    """Documents or chunks on the ``WAS_LLM_DOCUMENTS`` wire.

    Attributes:
        items: llama-index nodes: whole ``Document`` objects, or the chunks split from them.
        origin: What produced them, for panels, such as ``"12 files from docs [input]"``.
    """

    items: tuple = ()
    origin: str = ""

    def __len__(self) -> int:
        return len(self.items)

    @property
    def chunked(self) -> bool:
        """True when any item is a chunk rather than a whole document."""
        runtime.core()
        from llama_index.core import Document

        return any(not isinstance(item, Document) for item in self.items)

    @property
    def characters(self) -> int:
        """Characters of text across every item."""
        return sum(len(getattr(item, "text", "") or "") for item in self.items)

    def plus(self, other: "Documents") -> "Documents":
        """These items followed by another set's."""
        origin = " + ".join(part for part in (self.origin, other.origin) if part)
        return Documents(self.items + other.items, origin)

    def models(self) -> tuple:
        return ()


@dataclass(frozen=True, eq=False)
class Index:
    """A built index on the ``WAS_LLM_INDEX`` wire.

    Attributes:
        index: The llama-index index.
        kind: ``vector``, ``summary``, ``tree``, ``keyword``, ``graph``, ``document_summary``
            or ``image``.
        embedding: The embedding model it was built with, or ``None``.
        llm: The language model it was built with, or ``None``.
        name: The name it is saved under, ``""`` while unsaved.
        size: Chunks it holds.
        pictures: The model its pictures were embedded with, for an ``image`` index.
        store: The vector store collection a ``vector`` index lives in, ``None`` for its own files.
    """

    index: Any
    kind: str
    embedding: EmbeddingModel | None = None
    llm: LanguageModel | None = None
    name: str = ""
    size: int = 0
    pictures: EmbeddingModel | None = None
    store: Any = None

    def models(self) -> tuple:
        return models_in(self.embedding, self.llm, self.pictures)


@dataclass(frozen=True, eq=False)
class Retriever:
    """A retriever on the ``WAS_LLM_RETRIEVER`` wire.

    Attributes:
        retriever: The llama-index retriever.
        label: What panels call it, such as ``"vector top 4"``.
        depends: The models and indexes it calls when it retrieves.
        pictures: True when it can find pictures, and search by one.
    """

    retriever: Any
    label: str
    depends: tuple = ()
    pictures: bool = False

    def models(self) -> tuple:
        return models_in(*self.depends)


@dataclass(frozen=True, eq=False)
class Postprocessor:
    """One step of the ``WAS_LLM_POSTPROCESSORS`` chain.

    Attributes:
        processor: The llama-index node postprocessor.
        label: What panels call it, such as ``"rerank top 3"``.
        depends: The models it calls.
    """

    processor: Any
    label: str
    depends: tuple = ()

    def models(self) -> tuple:
        return models_in(*self.depends)


@dataclass(frozen=True, eq=False)
class QueryEngine:
    """A query engine on the ``WAS_LLM_QUERY_ENGINE`` wire.

    Attributes:
        engine: The llama-index query engine.
        label: What panels call it.
        depends: The models, retrievers and engines it calls when queried.
    """

    engine: Any
    label: str
    depends: tuple = ()

    def models(self) -> tuple:
        return models_in(*self.depends)


@dataclass(frozen=True, eq=False)
class ChatEngine:
    """How a chat engine is built around a memory, on the ``WAS_LLM_CHAT_ENGINE`` wire.

    Attributes:
        build: Called with a llama-index memory, returns the llama-index chat engine.
        label: What panels call it, such as ``"context chat on claude-opus-5-5"``.
        mode: The llama-index chat mode, such as ``condense_plus_context``.
        depends: The models and retrievers the engine calls.
    """

    build: Callable[[Any], Any]
    label: str
    mode: str
    depends: tuple = ()

    def models(self) -> tuple:
        return models_in(*self.depends)


@dataclass(frozen=True, eq=False)
class AgentSpec:
    """One agent on the ``WAS_LLM_AGENTS`` chain.

    Attributes:
        name: The agent's name, which other agents hand off to.
        description: What the agent does, read by the agents that may hand off to it.
        system_prompt: The agent's instructions.
        llm: The model it reasons with.
        tools: llama-index tools it may call.
        kind: ``auto``, ``function`` or ``react``.
        can_handoff_to: Names of the agents it may pass the conversation to, empty for any.
        depends: The models, engines and retrievers its tools call.
    """

    name: str
    description: str
    system_prompt: str
    llm: LanguageModel
    tools: tuple = ()
    kind: str = "auto"
    can_handoff_to: tuple = ()
    depends: tuple = ()

    def models(self) -> tuple:
        return models_in(self.llm, *self.depends)


@dataclass(frozen=True, eq=False)
class Sources:
    """Retrieved chunks with their scores on the ``WAS_LLM_SOURCES`` wire.

    Attributes:
        items: llama-index ``NodeWithScore`` objects, best first.
        query: The question they were retrieved for.
        answer: The answer written from them, ``""`` for a retrieval alone.
    """

    items: tuple = ()
    query: str = ""
    answer: str = ""

    def __len__(self) -> int:
        return len(self.items)

    @property
    def contexts(self) -> list[str]:
        """The text of every chunk, best first."""
        return [item.node.get_content() for item in self.items]

    def models(self) -> tuple:
        return ()


def chain(previous, *added) -> tuple:
    """A chain wire's tuple with items appended.

    Args:
        previous: The chain arriving on the node's input, ``None`` or a tuple.
        added: Items to append, ``None`` entries dropped.

    Returns:
        A new tuple, the incoming items first.
    """
    head = tuple(previous) if isinstance(previous, (tuple, list)) else ()
    return head + tuple(item for item in added if item is not None)


def models_in(*values) -> tuple:
    """Every language and embedding model reachable from wire values, each once.

    Args:
        values: Wire values, tuples of them, or ``None``.

    Returns:
        :class:`LanguageModel` and :class:`EmbeddingModel` objects in the order first reached.
    """
    found: list = []
    seen: set[int] = set()

    def visit(value, depth: int) -> None:
        if value is None or depth > 32:
            return
        if isinstance(value, (LanguageModel, EmbeddingModel)):
            if id(value) not in seen:
                seen.add(id(value))
                found.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item, depth + 1)
            return
        models = getattr(value, "models", None)
        if callable(models):
            for item in models():
                visit(item, depth + 1)

    for value in values:
        visit(value, 0)
    return tuple(found)


def ready(*values) -> None:
    """Gate every hosted model reachable from wire values on ``features.network``.

    Raises:
        RuntimeError: A hosted model is reachable and ``features.network`` is off.
    """
    for model in models_in(*values):
        model.ready()


def expect(value, kind: type | tuple, node: str, socket: str, hint: str):
    """A wire's value, checked to be what the node takes.

    Args:
        value: What arrived on the input.
        kind: The class, or classes, accepted.
        node: The node's display name, for the message.
        socket: The input's name, for the message.
        hint: The node to connect, such as ``"LLM Load Documents"``.

    Returns:
        ``value``.

    Raises:
        ValueError: Nothing arrived, or something of another kind did.
    """
    if isinstance(value, kind):
        return value
    if value is None:
        raise ValueError(f"{node} needs its {socket} input connected. Connect {hint} to it.")
    raise ValueError(
        f"{node} was handed a {type(value).__name__} on its {socket} input, which it cannot use. "
        f"Connect {hint} to it."
    )


def every(items: Iterable, kind: type | tuple) -> bool:
    """Whether every item is of a kind."""
    return all(isinstance(item, kind) for item in items)
