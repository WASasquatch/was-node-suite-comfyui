"""Building, inserting into, copying, saving and loading the indexes on the ``WAS_LLM_INDEX`` wire.

Each built index keeps its build settings beside it, written into the saved marker and read
back on load. Chunk sizes are in tokens.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import os
import time
import weakref
from dataclasses import dataclass, field, replace

from .. import log
from ..interface import progress, run_result
from ..util import sandbox
from . import ingest, runtime, storage, wires

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DEFAULT_SUMMARY_QUERY",
    "EXTRACTORS",
    "INSERT_MODES",
    "KINDS",
    "NO_INDEXES",
    "Prepared",
    "Refusal",
    "build_document_summary",
    "build_graph",
    "build_image",
    "build_keyword",
    "build_summary",
    "build_tree",
    "build_vector",
    "open_store_index",
    "check_chunking",
    "dimensions_of",
    "duplicate",
    "failing",
    "figures",
    "graph_settings",
    "insert",
    "load",
    "prepare",
    "hushed",
    "persist",
    "picture_count",
    "report",
    "saved_names",
    "settings_of",
    "size_of",
    "stamp",
    "tree_summaries",
    "watched",
]

logger = log.get_logger("llm.index")

#: What each kind is called on a panel and in a message.
KINDS = {
    "vector": "vector",
    "summary": "summary",
    "tree": "tree",
    "keyword": "keyword",
    "graph": "knowledge graph",
    "document_summary": "document summary",
    "image": "image",
}

#: Tokens in one chunk when whole documents are split.
CHUNK_SIZE = 1024

#: Tokens two neighbouring chunks share when whole documents are split.
CHUNK_OVERLAP = 20

#: The ways a knowledge graph finds its paths.
EXTRACTORS = ("simple", "implicit", "dynamic")

#: The entry the saved index menu shows while nothing is saved, never a valid name.
NO_INDEXES = "(no saved indexes)"

#: Chunks one language model step handles between two checks for a cancelled queue.
LLM_BATCH = 16

#: Documents split between two checks for a cancelled queue.
SPLIT_BATCH = 32

#: Calls a graph extractor makes at once.
WORKERS = 4

#: Rows of a sample on a panel.
SAMPLES = 8

#: Characters of a chunk or summary in one sample row.
SAMPLE_CHARS = 200

#: Lines of the keyword or triplet listing a panel body carries.
LISTING_LINES = 2000

#: Entity or relation types a dynamic graph is seeded with, at most.
MAX_TYPES = 64

#: Characters of one entity or relation type.
MAX_TYPE_CHARS = 64

#: Characters of a document summary prompt.
MAX_QUERY_CHARS = 4000

#: The marker layout written by :func:`persist`.
MARKER_FORMAT = 1

#: What a document summary is asked to say when no prompt is given.
DEFAULT_SUMMARY_QUERY = (
    "Describe what the provided text is about. Also describe some of the questions that this "
    "text can answer."
)

#: The range each recorded setting is held to when a marker is read back.
LIMITS = {
    "chunk_size": (32, 8192, CHUNK_SIZE),
    "chunk_overlap": (0, 2048, CHUNK_OVERLAP),
    "batch_size": (1, 4096, 64),
    "num_children": (2, 64, 10),
    "max_keywords": (1, 100, 10),
    "max_paths": (1, 100, 10),
}

#: Build settings of every live index, keyed by the llama-index object.
_SETTINGS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

_watcher = None
_holder = None


class Refusal(ValueError):
    """A request an index node refuses, with the cause and the fix in its text."""


@dataclass
class Prepared:
    """Chunks ready to index.

    Attributes:
        leaves: The chunks an index reads.
        others: Every other level of a hierarchy, kept in the store beside the leaves.
        handed: How many items arrived.
        split: How many whole documents were split here.
        chunk_size: Tokens in one chunk for whole documents split here.
        chunk_overlap: Tokens neighbouring chunks share.
        hashes: ``(document id, hash)`` of every whole document split here.
    """

    leaves: list
    others: list = field(default_factory=list)
    handed: int = 0
    split: int = 0
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    hashes: list = field(default_factory=list)


def saved_names() -> list[str]:
    """Every saved index name, or :data:`NO_INDEXES` alone while none is saved."""
    return list(storage.names(storage.INDEXES)) or [NO_INDEXES]


def stamp(name) -> str:
    """A value that moves whenever the index saved under a name is saved again.

    Args:
        name: The saved name as the widget holds it.

    Returns:
        The name with its marker's modification time, or the name alone when nothing is saved.
    """
    try:
        marker = storage.index_dir(name) / storage.INDEX_MARKER
        return f"{name}:{os.stat(marker).st_mtime_ns}"
    except (OSError, ValueError):
        return str(name)[:80]


def settings_of(index) -> dict:
    """The build settings recorded for a llama-index index, empty when none are."""
    return dict(_SETTINGS.get(index, {}))


def _remember(index, kind: str, prepared: Prepared | None = None, **values) -> None:
    """Record an index's build settings.

    Args:
        index: The llama-index index.
        kind: One of :data:`KINDS`.
        prepared: The chunks it was built from, for the chunk sizes.
        values: Further settings of its kind.
    """
    held = {"kind": kind}
    # Chunk sizes are recorded only when this build split whole documents itself.
    if prepared is not None and prepared.split:
        held["chunk_size"] = prepared.chunk_size
        held["chunk_overlap"] = prepared.chunk_overlap
    held.update(values)
    _SETTINGS[index] = held


def _batches(items, size: int):
    """Consecutive slices of a list, ``size`` items each."""
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _splitter(chunk_size: int, chunk_overlap: int):
    """A sentence splitter cutting documents into chunks of a set size."""
    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))


def check_chunking(node: str, chunk_size: int, chunk_overlap: int) -> None:
    """Refuse an overlap that is not smaller than the chunk.

    Raises:
        Refusal: ``chunk_overlap`` is ``chunk_size`` or more.
    """
    if int(chunk_overlap) >= int(chunk_size):
        raise Refusal(
            f"{node} has chunk_overlap {chunk_overlap}, which is not smaller than chunk_size "
            f"{chunk_size}. Lower chunk_overlap, such as to 20, or raise chunk_size."
        )


def prepare(documents, node: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
            pictures: bool = False) -> Prepared:
    """Split whole documents into chunks and part a hierarchy's leaves from its upper levels.

    Args:
        documents: The :class:`wires.Documents` that arrived.
        node: The node's display name, for messages.
        chunk_size: Tokens in one chunk for whole documents.
        chunk_overlap: Tokens neighbouring chunks share.
        pictures: Keep picture documents that hold no text, for an image index. Otherwise a
            picture is indexed by its text alone.

    Returns:
        The chunks to index.

    Raises:
        Refusal: Nothing arrived, nothing holds text, or a document cannot be split.
    """
    runtime.core()
    from llama_index.core import Document
    from llama_index.core.schema import ImageNode, MetadataMode, NodeRelationship

    items = list(documents.items)
    if not items:
        raise Refusal(
            f"{node} was handed no documents. Check that LLM Load Documents found files in "
            f"the folder it reads."
        )
    check_chunking(node, chunk_size, chunk_overlap)
    whole = [item for item in items if isinstance(item, Document)]
    chunks = [item for item in items if not isinstance(item, Document)]
    if whole:
        splitter = _splitter(chunk_size, chunk_overlap)
        for batch in _batches(whole, SPLIT_BATCH):
            runtime.check_interrupt()
            try:
                chunks.extend(splitter.get_nodes_from_documents(batch))
            except ValueError as error:
                raise Refusal(
                    f"{node} could not cut its documents into chunks of {chunk_size} tokens "
                    f"({runtime.describe_error(error)}). Raise chunk_size, or split them with "
                    f"LLM Split Documents."
                ) from None
    leaves, others = [], []
    for chunk in chunks:
        upper = NodeRelationship.CHILD in chunk.relationships or chunk.metadata.get(ingest.LEAF_KEY) is False
        (others if upper else leaves).append(chunk)
    leaves = [
        chunk for chunk in leaves
        if (pictures and isinstance(chunk, ImageNode)) or chunk.get_content(metadata_mode=MetadataMode.NONE).strip()
    ]
    if not leaves:
        if any(isinstance(item, ImageNode) for item in items):
            raise Refusal(
                f"The pictures handed to {node} hold no text, and it indexes text only. Index "
                f"them with LLM Image Index, or describe them with a vision model in LLM Load Images."
            )
        raise Refusal(
            f"None of the {len(items)} documents handed to {node} holds any text to index. "
            f"Check what LLM Load Documents read."
        )
    return Prepared(
        leaves=leaves,
        others=others,
        handed=len(items),
        split=len(whole),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        hashes=[(doc.id_, doc.hash) for doc in whole],
    )


def _context():
    """A new in-memory storage context."""
    from llama_index.core import StorageContext

    return StorageContext.from_defaults()


def _store_extras(index, prepared: Prepared) -> None:
    """Keep a hierarchy's upper levels and the split documents' hashes in an index's store."""
    if prepared.others:
        index.docstore.add_documents(prepared.others, allow_update=True)
    for doc_id, digest in prepared.hashes:
        index.docstore.set_document_hash(doc_id, digest)


def _watcher_class():
    """The callback handler that stops a build on a cancelled queue and counts model calls."""
    global _watcher
    if _watcher is not None:
        return _watcher
    from llama_index.core.callbacks import CBEventType
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler

    class Watcher(BaseCallbackHandler):
        """Stops a build when the queue is cancelled, and counts finished model calls."""

        def __init__(self, tick=None):
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self.tick = tick
            self.depth = 0

        def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
            if event_type in (CBEventType.LLM, CBEventType.EMBEDDING):
                runtime.check_interrupt()
            if event_type == CBEventType.LLM:
                self.depth += 1
            return event_id

        def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
            if event_type != CBEventType.LLM:
                return
            self.depth = max(0, self.depth - 1)
            if self.depth == 0 and self.tick is not None:
                self.tick()

        def start_trace(self, trace_id=None):
            return None

        def end_trace(self, trace_id=None, trace_map=None):
            return None

    _watcher = Watcher
    return _watcher


@contextlib.contextmanager
def watched(*models, tick=None):
    """Check for a cancelled queue before every call the models make while the block runs.

    Args:
        models: llama-index language and embedding models, ``None`` entries skipped.
        tick: Called after each finished language model call, such as a progress bar's step.
    """
    handler = _watcher_class()(tick)
    held = []
    for model in models:
        manager = getattr(model, "callback_manager", None)
        if manager is None or any(manager is seen for seen, _ in held):
            continue
        before = getattr(manager, "handlers", None)
        held.append((manager, before))
        manager.handlers = [*(before or []), handler]
    try:
        yield handler
    finally:
        for manager, before in reversed(held):
            manager.handlers = before if before is not None else []


@contextlib.contextmanager
def hushed():
    """Send what llama-index prints to the console while the block runs nowhere."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def failing(node: str):
    """Re-raise a library or provider error as a message naming the node, keys hidden.

    A cancelled queue derives from ``BaseException`` and passes through unchanged.

    Args:
        node: The node's display name.

    Raises:
        RuntimeError: Anything but a :class:`Refusal` or a refused path went wrong.
    """
    try:
        yield
    except (Refusal, sandbox.PathNotAllowed):
        raise
    except Exception as error:
        text = runtime.describe_error(error)
        logger.debug("%s failed with %s: %s", node, type(error).__name__, text)
        raise RuntimeError(f"{node} failed: {text}") from None


def build_vector(prepared: Prepared, embedding: wires.EmbeddingModel, batch_size: int, store=None):
    """Embed the leaves into a vector index, in memory or in a vector store's collection.

    Args:
        prepared: The chunks.
        embedding: The embedding model.
        batch_size: Chunks embedded between two checks for a cancelled queue.
        store: A :class:`~modules.llm.vector_stores.VectorStore`, or ``None`` for memory. Chunks of a
            document the collection already holds replace that document's chunks.

    Returns:
        The llama-index ``VectorStoreIndex``.
    """
    from llama_index.core import VectorStoreIndex

    batch_size = max(1, int(batch_size))
    if store is not None:
        index = store.open_index(embedding, create=True)
        index._insert_batch_size = batch_size
        _drop_documents(index, store, prepared.leaves)
    else:
        index = VectorStoreIndex(
            nodes=[],
            embed_model=embedding.model,
            storage_context=_context(),
            insert_batch_size=batch_size,
            transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
        )
    _store_extras(index, prepared)
    bar = progress.progress_bar(math.ceil(len(prepared.leaves) / batch_size))
    with watched(embedding.model):
        for batch in _batches(prepared.leaves, batch_size):
            runtime.check_interrupt()
            index.insert_nodes(batch)
            bar.update(1)
    if store is not None:
        store.written(index)
    _remember(index, "vector", prepared, batch_size=batch_size)
    return index


def _drop_documents(index, store, chunks) -> None:
    """Remove from a collection every chunk of the documents new chunks come from."""
    from . import vector_stores

    if not vector_stores.count(store):
        return
    for document in sorted({chunk.ref_doc_id for chunk in chunks if chunk.ref_doc_id}):
        runtime.check_interrupt()
        store.delete_document(index, document)


def open_store_index(store, embedding: wires.EmbeddingModel):
    """A vector index over what a vector store's collection already holds.

    Returns:
        The llama-index ``VectorStoreIndex``.

    Raises:
        Refusal: The collection holds nothing yet.
    """
    index = store.open_index(embedding)
    if index is None:
        raise Refusal(f"The collection {store.label} holds nothing yet. Fill it with LLM Vector Index.")
    _remember(index, "vector")
    return index


def build_image(prepared: Prepared, pictures: wires.EmbeddingModel, embedding: wires.EmbeddingModel | None,
                batch_size: int):
    """Embed pictures and text into an in-memory image index.

    Args:
        prepared: The chunks and picture documents.
        pictures: The model pictures are embedded with, and text when ``embedding`` is ``None``.
        embedding: The model text is embedded with, or ``None``.
        batch_size: Items embedded between two checks for a cancelled queue.

    Returns:
        The llama-index ``MultiModalVectorStoreIndex``.
    """
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    batch_size = max(1, int(batch_size))
    text_model = embedding if embedding is not None else pictures
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=text_model.model,
        image_embed_model=pictures.model,
        storage_context=_context(),
        insert_batch_size=batch_size,
        transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
    )
    _store_extras(index, prepared)
    bar = progress.progress_bar(math.ceil(len(prepared.leaves) / batch_size))
    with watched(text_model.model, pictures.model):
        for batch in _batches(prepared.leaves, batch_size):
            runtime.check_interrupt()
            index.insert_nodes(batch)
            bar.update(1)
    _remember(index, "image", prepared, batch_size=batch_size)
    return index


def picture_count(index) -> int:
    """How many pictures an image index holds embeddings for."""
    try:
        return len(index.storage_context.vector_stores["image"].data.embedding_dict)
    except (AttributeError, KeyError):
        return 0


def build_summary(prepared: Prepared):
    """Keep the leaves in order in a summary index, which needs no model to build.

    Returns:
        The llama-index ``SummaryIndex``.
    """
    from llama_index.core import SummaryIndex

    runtime.check_interrupt()
    index = SummaryIndex(
        nodes=prepared.leaves,
        storage_context=_context(),
        transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
    )
    _store_extras(index, prepared)
    _remember(index, "summary", prepared)
    return index


def tree_summaries(leaves: int, num_children: int) -> int:
    """How many summaries a tree of ``leaves`` chunks writes with ``num_children`` per parent."""
    total, level = 0, int(leaves)
    children = max(2, int(num_children))
    while level > children:
        level = math.ceil(level / children)
        total += level
    return total


def build_tree(prepared: Prepared, model: wires.LanguageModel, num_children: int):
    """Summarize the leaves bottom up into a tree, each parent covering ``num_children``.

    Returns:
        The llama-index ``TreeIndex``.
    """
    from llama_index.core import TreeIndex

    num_children = max(2, int(num_children))
    bar = progress.progress_bar(max(1, tree_summaries(len(prepared.leaves), num_children)))
    runtime.check_interrupt()
    with watched(model.llm, tick=lambda: bar.update(1)):
        index = TreeIndex(
            nodes=prepared.leaves,
            llm=model.llm,
            num_children=num_children,
            storage_context=_context(),
            transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
        )
    _store_extras(index, prepared)
    _remember(index, "tree", prepared, num_children=num_children)
    return index


def _text_only(nodes) -> list:
    """Copies of chunks whose keywords are read from their text alone, their metadata left out."""
    return [node.model_copy(update={"excluded_llm_metadata_keys": list(node.metadata)}) for node in nodes]


def build_keyword(prepared: Prepared, model: wires.LanguageModel | None, max_keywords: int):
    """Map keywords to the chunks holding them, asked of a model or found by a word count.

    Args:
        prepared: The chunks.
        model: The language model that names each chunk's keywords, or ``None`` for the word count.
        max_keywords: Keywords kept per chunk.

    Returns:
        The llama-index ``KeywordTableIndex`` or ``SimpleKeywordTableIndex``.
    """
    from llama_index.core import KeywordTableIndex, SimpleKeywordTableIndex

    max_keywords = max(1, int(max_keywords))
    transformations = [_splitter(prepared.chunk_size, prepared.chunk_overlap)]
    if model is None:
        runtime.check_interrupt()
        index = SimpleKeywordTableIndex(
            nodes=_text_only(prepared.leaves),
            max_keywords_per_chunk=max_keywords,
            storage_context=_context(),
            transformations=transformations,
        )
        index.docstore.add_documents(prepared.leaves, allow_update=True)
    else:
        index = KeywordTableIndex(
            nodes=[],
            llm=model.llm,
            max_keywords_per_chunk=max_keywords,
            storage_context=_context(),
            transformations=transformations,
        )
        bar = progress.progress_bar(len(prepared.leaves))
        with watched(model.llm, tick=lambda: bar.update(1)):
            for batch in _batches(prepared.leaves, LLM_BATCH):
                runtime.check_interrupt()
                index.insert_nodes(_text_only(batch))
                index.docstore.add_documents(batch, allow_update=True)
    _store_extras(index, prepared)
    _remember(
        index, "keyword", prepared,
        keywords="simple" if model is None else "llm", max_keywords=max_keywords,
    )
    return index


def _holder_class():
    """The graph step that hands each chunk the paths found for it before insertion."""
    global _holder
    if _holder is not None:
        return _holder
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
    from llama_index.core.schema import TransformComponent

    class FoundPaths(TransformComponent):
        """Hands each chunk the entities and relations found for it."""

        _found: dict = PrivateAttr(default_factory=dict)

        def hold(self, found: dict) -> None:
            """Keep the paths found per chunk id for the next insertion."""
            self._found = found

        def __call__(self, nodes, **kwargs):
            for node in nodes:
                entities, relations = self._found.get(node.id_, ((), ()))
                node.metadata.update({KG_NODES_KEY: list(entities), KG_RELATIONS_KEY: list(relations)})
            self._found = {}
            return nodes

    _holder = FoundPaths
    return _holder


def _types(text) -> list[str]:
    """Comma separated entity or relation types as a list, each trimmed and bounded."""
    found = []
    for part in str(text or "").split(","):
        name = part.strip()[:MAX_TYPE_CHARS]
        if name and name not in found:
            found.append(name)
        if len(found) >= MAX_TYPES:
            break
    return found


def graph_settings(choice: dict | None) -> dict:
    """A knowledge graph's extractor settings from its extractor input's value.

    Args:
        choice: The extractor input's value, holding ``extractor`` and that option's inputs.

    Returns:
        ``extractor``, ``max_paths``, ``entity_types`` and ``relation_types``.

    Raises:
        Refusal: The extractor is not one of :data:`EXTRACTORS`.
    """
    choice = choice if isinstance(choice, dict) else {"extractor": choice}
    extractor = str(choice.get("extractor", "simple"))
    if extractor not in EXTRACTORS:
        raise Refusal(f"The extractor {extractor[:40]!r} is not one of {', '.join(EXTRACTORS)}. Pick one of those.")
    paths = choice.get("max_paths_per_chunk", choice.get("max_triplets_per_chunk", LIMITS["max_paths"][2]))
    low, high, _ = LIMITS["max_paths"]
    return {
        "extractor": extractor,
        "max_paths": min(high, max(low, int(paths))),
        "entity_types": _types(choice.get("entity_types", "")) if extractor == "dynamic" else [],
        "relation_types": _types(choice.get("relation_types", "")) if extractor == "dynamic" else [],
    }


def _extractor(settings: dict, model: wires.LanguageModel | None, node: str):
    """The llama-index path extractor a graph's settings name.

    Raises:
        Refusal: The extractor asks a model and none is connected.
    """
    from llama_index.core.indices.property_graph import (
        DynamicLLMPathExtractor,
        ImplicitPathExtractor,
        SimpleLLMPathExtractor,
    )

    choice = settings.get("extractor", "simple")
    if choice == "implicit":
        return ImplicitPathExtractor()
    if model is None:
        raise Refusal(
            f"{node} uses the {choice} extractor, which asks a language model for the paths, "
            f"and none is connected. Connect an LLM to its llm input, or pick the implicit "
            f"extractor."
        )
    if choice == "dynamic":
        return DynamicLLMPathExtractor(
            llm=model.llm,
            max_triplets_per_chunk=settings.get("max_paths", 10),
            num_workers=WORKERS,
            allowed_entity_types=list(settings.get("entity_types") or []) or None,
            allowed_relation_types=list(settings.get("relation_types") or []) or None,
        )
    return SimpleLLMPathExtractor(
        llm=model.llm, max_paths_per_chunk=settings.get("max_paths", 10), num_workers=WORKERS,
    )


async def _graph_insert(index, chunks: list, model, node: str) -> None:
    """Find the paths in chunks and insert them into a knowledge graph, a batch at a time.

    Args:
        index: The llama-index ``PropertyGraphIndex``.
        chunks: The chunks to read paths from, left unchanged.
        model: The :class:`wires.LanguageModel` the extractor asks, or ``None``.
        node: The node's display name, for messages.
    """
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY

    held = _SETTINGS.get(index, {})
    extractor = _extractor(held, model, node)
    holder = held.get("holder")
    bar = progress.progress_bar(math.ceil(len(chunks) / LLM_BATCH))
    with watched(model.llm if model is not None else None):
        for batch in _batches(chunks, LLM_BATCH):
            runtime.check_interrupt()
            copies = [chunk.model_copy(deep=True) for chunk in batch]
            found_in = await extractor.acall(copies, show_progress=False)
            found = {}
            for chunk in found_in:
                found[chunk.id_] = (
                    chunk.metadata.pop(KG_NODES_KEY, []),
                    chunk.metadata.pop(KG_RELATIONS_KEY, []),
                )
            holder.hold(found)
            index.insert_nodes(copies)
            bar.update(1)


async def build_graph(
    prepared: Prepared,
    model: wires.LanguageModel | None,
    embedding: wires.EmbeddingModel | None,
    settings: dict,
    node: str,
):
    """Read entities and relations out of the leaves into a property graph.

    Args:
        prepared: The chunks.
        model: The language model the extractor asks, or ``None`` for the implicit extractor.
        embedding: The model that embeds the graph for vector search, or ``None``.
        settings: What :func:`graph_settings` answered.
        node: The node's display name, for messages.

    Returns:
        The llama-index ``PropertyGraphIndex``.
    """
    from llama_index.core import PropertyGraphIndex
    from llama_index.core.graph_stores import SimplePropertyGraphStore

    _extractor(settings, model, node)
    holder = _holder_class()()
    index = PropertyGraphIndex(
        nodes=[],
        llm=model.llm if model is not None else None,
        kg_extractors=[holder],
        property_graph_store=SimplePropertyGraphStore(),
        use_async=False,
        embed_model=embedding.model if embedding is not None else None,
        embed_kg_nodes=embedding is not None,
        storage_context=_context(),
        transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
    )
    _remember(index, "graph", prepared, holder=holder, **settings)
    _store_extras(index, prepared)
    with watched(embedding.model if embedding is not None else None):
        await _graph_insert(index, prepared.leaves, model, node)
    return index


def build_document_summary(
    prepared: Prepared,
    model: wires.LanguageModel,
    embedding: wires.EmbeddingModel | None,
    summary_query: str,
    node: str,
):
    """Write one summary per document from its chunks, embedded when a model is given.

    Args:
        prepared: The chunks.
        model: The language model that writes the summaries.
        embedding: The model that embeds the summaries, or ``None``.
        summary_query: What each summary is asked to say.
        node: The node's display name, for messages.

    Returns:
        The llama-index ``DocumentSummaryIndex``.

    Raises:
        Refusal: A chunk does not say which document it came from.
    """
    from llama_index.core import DocumentSummaryIndex

    groups: dict = {}
    for chunk in prepared.leaves:
        if chunk.ref_doc_id is None:
            raise Refusal(
                f"{node} needs chunks that know which document they came from. Connect LLM "
                f"Load Documents to it directly, or split the documents with LLM Split Documents."
            )
        groups.setdefault(chunk.ref_doc_id, []).append(chunk)
    query = str(summary_query or "").strip()[:MAX_QUERY_CHARS] or DEFAULT_SUMMARY_QUERY
    index = DocumentSummaryIndex(
        nodes=[],
        llm=model.llm,
        embed_model=embedding.model if embedding is not None else None,
        storage_context=_context(),
        summary_query=query,
        embed_summaries=embedding is not None,
        transformations=[_splitter(prepared.chunk_size, prepared.chunk_overlap)],
    )
    bar = progress.progress_bar(len(groups))
    with watched(model.llm, embedding.model if embedding is not None else None), hushed():
        for chunks in groups.values():
            runtime.check_interrupt()
            index.insert_nodes(chunks)
            bar.update(1)
    _store_extras(index, prepared)
    _remember(index, "document_summary", prepared, summary_query=query, embedded=embedding is not None)
    return index


def size_of(index, kind: str, store=None) -> int:
    """How many chunks an index reads, counted in its vector store's collection when it has one."""
    if store is not None:
        from . import vector_stores

        return vector_stores.count(store)
    struct = index.index_struct
    if kind in ("vector", "image"):
        return len(struct.nodes_dict)
    if kind == "summary":
        return len(struct.nodes)
    if kind == "tree":
        parents = sum(1 for children in struct.node_id_to_children_ids.values() if children)
        return max(0, len(struct.all_nodes) - parents)
    if kind == "keyword":
        return len(set().union(*struct.table.values())) if struct.table else 0
    if kind == "graph":
        from llama_index.core.graph_stores.types import ChunkNode

        return sum(1 for item in index.property_graph_store.graph.nodes.values() if isinstance(item, ChunkNode))
    if kind == "document_summary":
        return len(struct.node_id_to_summary_id)
    return 0


def _vectors(index) -> dict:
    """The embeddings held in an index's default vector store, by id."""
    try:
        return index.storage_context.vector_store.data.embedding_dict
    except (AttributeError, KeyError):
        return {}


def dimensions_of(index) -> int:
    """The length of the vectors an index holds, 0 when it holds none."""
    for vector in _vectors(index).values():
        return len(vector or ())
    return 0


def _text(node, chars: int = SAMPLE_CHARS) -> str:
    """A node's text on one line, cut to ``chars``."""
    from llama_index.core.schema import MetadataMode

    text = " ".join(node.get_content(metadata_mode=MetadataMode.NONE).split())
    return text if len(text) <= chars else text[: chars - 3] + "..."


def _nodes(index, ids) -> list:
    """The stored nodes behind ids, skipping any the store does not hold."""
    found = []
    for node_id in ids:
        node = index.docstore.get_node(node_id, raise_error=False)
        if node is not None:
            found.append(node)
    return found


def _entity(item) -> str:
    """What a graph node is called in a triplet row."""
    name = getattr(item, "name", None)
    if name is not None:
        return str(name)
    text = " ".join(str(getattr(item, "text", "") or "").split())
    return f"[{text[:40]}...]" if len(text) > 40 else f"[{text}]"


def figures(wire: wires.Index) -> tuple[dict, dict, list, list]:
    """What a panel shows about an index.

    Args:
        wire: The index.

    Returns:
        ``(counts, facts, items, bodies)`` for the run result.
    """
    from llama_index.core.graph_stores.types import EntityNode

    index, kind = wire.index, wire.kind
    struct = index.index_struct
    counts: dict = {"chunks": wire.size}
    facts: dict = {"kind": KINDS.get(kind, kind)}
    if wire.embedding is not None:
        facts["embedding"] = wire.embedding.label
    if wire.pictures is not None:
        facts["pictures by"] = wire.pictures.label
    if wire.llm is not None:
        facts["llm"] = wire.llm.label
    held = settings_of(index)
    items: list = []
    bodies: list = []
    vectors = _vectors(index)
    if kind == "vector" and wire.store is not None:
        counts["embeddings"] = wire.size
        facts["vector store"] = wire.store.label
    elif kind == "vector":
        counts["embeddings"] = len(vectors)
        counts["dimensions"] = dimensions_of(index)
        items = [_text(node) for node in _nodes(index, list(struct.nodes_dict.values())[:SAMPLES])]
    elif kind == "image":
        counts["pictures"] = picture_count(index)
        counts["text embeddings"] = len(vectors)
        items = [
            _text(node) or f"picture {node.metadata.get('file_name', '')}".strip()
            for node in _nodes(index, list(struct.nodes_dict.values())[:SAMPLES])
        ]
    elif kind == "summary":
        items = [_text(node) for node in _nodes(index, struct.nodes[:SAMPLES])]
    elif kind == "tree":
        parents = [node_id for node_id, children in struct.node_id_to_children_ids.items() if children]
        counts["summaries"] = len(parents)
        counts["roots"] = len(struct.root_nodes)
        facts["children per parent"] = str(held.get("num_children", ""))
        roots = _nodes(index, list(struct.root_nodes.values()))
        items = [_text(node) for node in roots[:SAMPLES]]
        if parents:
            bodies.append(run_result.body("root summaries", "\n\n".join(_text(node, 4000) for node in roots)))
    elif kind == "keyword":
        ranked = sorted(struct.table.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        counts["keywords"] = len(ranked)
        facts["keywords from"] = "word count" if held.get("keywords") == "simple" else "language model"
        items = [f"{word} ({len(ids)} chunk{'s' if len(ids) != 1 else ''})" for word, ids in ranked[:SAMPLES]]
        if ranked:
            lines = [f"{word}: {len(ids)}" for word, ids in ranked[:LISTING_LINES]]
            bodies.append(run_result.body("keywords", "\n".join(lines)))
    elif kind == "graph":
        graph = index.property_graph_store.graph
        triplets = sorted(graph.triplets)
        counts["triplets"] = len(triplets)
        counts["entities"] = sum(1 for item in graph.nodes.values() if isinstance(item, EntityNode))
        if vectors:
            counts["embeddings"] = len(vectors)
        facts["extractor"] = str(held.get("extractor", ""))
        rows = []
        for subject, relation, target in triplets[:LISTING_LINES]:
            left, right = graph.nodes.get(subject), graph.nodes.get(target)
            rows.append(
                f"{_entity(left) if left is not None else subject} -> {relation} -> "
                f"{_entity(right) if right is not None else target}"
            )
        items = rows[:SAMPLES]
        if rows:
            bodies.append(run_result.body("triplets", "\n".join(rows)))
    elif kind == "document_summary":
        summaries = _nodes(index, list(struct.doc_id_to_summary_id.values()))
        counts["summaries"] = len(summaries)
        if vectors:
            counts["embeddings"] = len(vectors)
        items = [_text(node) for node in summaries[:SAMPLES]]
        if summaries:
            bodies.append(run_result.body("summaries", "\n\n".join(_text(node, 4000) for node in summaries)))
    if "chunk_size" in held:
        facts["chunking"] = f"{held['chunk_size']} tokens, {held.get('chunk_overlap', 0)} overlap"
    return counts, facts, items, bodies


def report(
    wire: wires.Index,
    seconds: float,
    summary: str,
    prepared: Prepared | None = None,
    warnings=(),
    counts: dict | None = None,
    facts: dict | None = None,
) -> None:
    """Publish what an index node did for its panel.

    Args:
        wire: The index the node hands on.
        seconds: How long the node took.
        summary: One line saying what it did.
        prepared: The chunks it read, or ``None``.
        warnings: Short notes that turn the result into a warning.
        counts: Further named numbers, drawn after the index's own.
        facts: Further named strings, drawn after the index's own.
    """
    shown, held, items, bodies = figures(wire)
    numbers: dict = {}
    if prepared is not None:
        numbers["items in"] = prepared.handed
    numbers.update(shown)
    if prepared is not None and prepared.others:
        numbers["upper levels"] = len(prepared.others)
    numbers.update(counts or {})
    numbers["seconds"] = round(float(seconds), 2)
    named = dict(held)
    if prepared is not None and prepared.split:
        named["split here"] = f"{prepared.split} documents"
    elif prepared is not None:
        named.pop("chunking", None)
        named["chunks"] = "as split before"
    named.update(facts or {})
    if warnings:
        named["warning"] = "; ".join(warnings)
    run_result.publish(
        status=run_result.WARNING if warnings else run_result.OK,
        summary=summary,
        counts=numbers,
        facts=named,
        items=items,
        bodies=bodies,
    )


def _needs(wire: wires.Index) -> tuple[bool, bool]:
    """Whether inserting into an index asks a language model and an embedding model."""
    held = settings_of(wire.index)
    kind = wire.kind
    llm = kind in ("tree", "document_summary") or (kind == "keyword" and held.get("keywords") != "simple")
    if kind == "graph":
        llm = held.get("extractor") != "implicit"
    embed = kind in ("vector", "image") or (kind == "document_summary" and held.get("embedded")) or (
        kind == "graph" and wire.embedding is not None
    )
    return llm, bool(embed)


def duplicate(wire: wires.Index):
    """A copy of an index's llama-index object, sharing nothing it can change.

    Returns:
        The copy, its build settings recorded.
    """
    from llama_index.core import StorageContext
    from llama_index.core.constants import VECTOR_STORE_KEY

    held = settings_of(wire.index)
    data = wire.index.storage_context.to_dict()
    # Every store but the vector stores, which serialise into new containers, is deep copied.
    vectors = data.pop(VECTOR_STORE_KEY)
    copied = copy.deepcopy(data)
    copied[VECTOR_STORE_KEY] = vectors
    context = StorageContext.from_dict(copied)
    return _open(wire.kind, context, held, wire.llm, wire.embedding, wire.index.index_id, wire.pictures)


#: How LLM Index Insert treats a document the index already holds.
INSERT_MODES = ("update", "add")


def _changes(index, prepared: Prepared) -> tuple[set, set]:
    """Which incoming documents an index holds unchanged, and which it holds in another version.

    Returns:
        ``(unchanged, replaced)`` document ids. An unchanged whole document has the hash the
        index recorded for it; a replaced one is any other incoming document the index holds.
    """
    unchanged = {
        doc_id for doc_id, digest in prepared.hashes
        if index.docstore.get_document_hash(doc_id) == digest
    }
    incoming = {chunk.ref_doc_id for chunk in prepared.leaves if chunk.ref_doc_id} - unchanged
    replaced = {ref for ref in incoming if index.docstore.get_ref_doc_info(ref) is not None}
    return unchanged, replaced


def _without(prepared: Prepared, unchanged: set) -> Prepared:
    """The chunks and hashes of every document but the unchanged ones."""
    return replace(
        prepared,
        leaves=[chunk for chunk in prepared.leaves if chunk.ref_doc_id not in unchanged],
        hashes=[pair for pair in prepared.hashes if pair[0] not in unchanged],
    )


def _regrown_tree(wire: wires.Index, prepared: Prepared, dropped: set = frozenset()):
    """A tree summarized again from an index's own chunks followed by new ones.

    Args:
        wire: The tree.
        prepared: The new chunks.
        dropped: Document ids whose old chunks are left out.

    Returns:
        The new llama-index ``TreeIndex``, holding the old store's other levels and hashes.
    """
    old = wire.index
    struct = old.index_struct
    order = [struct.all_nodes[place] for place in sorted(struct.all_nodes)]
    leaves = _nodes(old, [node_id for node_id in order if not struct.node_id_to_children_ids.get(node_id)])
    leaves = [leaf for leaf in leaves if leaf.ref_doc_id not in dropped]
    placed = set(order)
    others = [item for node_id, item in old.docstore.docs.items() if node_id not in placed]
    hashes = [(doc_id, digest) for digest, doc_id in old.docstore.get_all_document_hashes().items()]
    held = settings_of(old)
    merged = Prepared(
        leaves=leaves + list(prepared.leaves),
        others=others + list(prepared.others),
        handed=prepared.handed,
        split=prepared.split,
        chunk_size=held.get("chunk_size", prepared.chunk_size),
        chunk_overlap=held.get("chunk_overlap", prepared.chunk_overlap),
        hashes=hashes + list(prepared.hashes),
    )
    return build_tree(merged, wire.llm, held.get("num_children", LIMITS["num_children"][2]))


async def insert(wire: wires.Index, prepared: Prepared, node: str, mode: str = "add") -> tuple[wires.Index, dict]:
    """A copy of an index with chunks added, each kind reading them as it reads its own.

    Args:
        wire: The index to add to, left unchanged.
        prepared: The chunks to add.
        node: The node's display name, for messages.
        mode: ``update`` skips documents the index holds unchanged and replaces the chunks of
            documents it holds in another version; ``add`` adds everything.

    Returns:
        A new :class:`wires.Index` around the copy, unsaved, and ``{"skipped", "replaced",
        "added"}`` counted in documents, documents and chunks.

    Raises:
        Refusal: The index needs a model to read chunks and was loaded without one.
    """
    wants_llm, wants_embedding = _needs(wire)
    label = KINDS.get(wire.kind, wire.kind)
    if wants_llm and wire.llm is None:
        raise Refusal(
            f"This {label} index has no language model to read new chunks with. Connect an LLM "
            f"to the llm input of LLM Load Index, then insert."
        )
    if wire.kind == "image" and wire.pictures is None:
        raise Refusal(
            "This image index has no picture model to embed new pictures with. Connect LLM Image "
            "Embedding Model to the image_embedding input of LLM Load Index, then insert."
        )
    if wants_embedding and wire.embedding is None and wire.kind != "image":
        raise Refusal(
            f"This {label} index has no embedding model to embed new chunks with. Connect LLM "
            f"Embedding Model to the embedding input of LLM Load Index, then insert."
        )
    stats = {"skipped": 0, "replaced": 0, "added": len(prepared.leaves)}
    if wire.kind == "tree":
        dropped: set = set()
        if mode == "update":
            unchanged, dropped = _changes(wire.index, prepared)
            prepared = _without(prepared, unchanged)
            stats.update(skipped=len(unchanged), replaced=len(dropped), added=len(prepared.leaves))
        index = _regrown_tree(wire, prepared, dropped)
        return replace(wire, index=index, name="", size=size_of(index, "tree")), stats
    if wire.store is not None:
        _drop_documents(wire.index, wire.store, prepared.leaves)
        held = _SETTINGS.get(wire.index, {})
        bar = progress.progress_bar(math.ceil(len(prepared.leaves) / max(1, held.get("batch_size", LLM_BATCH))))
        with watched(wire.embedding.model):
            for batch in _batches(prepared.leaves, held.get("batch_size", LLM_BATCH)):
                runtime.check_interrupt()
                wire.index.insert_nodes(batch)
                bar.update(1)
        wire.store.written(wire.index)
        return replace(wire, name="", size=size_of(wire.index, "vector", wire.store)), stats
    index = duplicate(wire)
    held = _SETTINGS.get(index, {})
    if mode == "update":
        unchanged, replaced = _changes(index, prepared)
        for document in sorted(replaced):
            runtime.check_interrupt()
            index.delete_ref_doc(document, delete_from_docstore=True)
        prepared = _without(prepared, unchanged)
        stats.update(skipped=len(unchanged), replaced=len(replaced), added=len(prepared.leaves))
    _store_extras(index, prepared)
    if wire.kind == "graph":
        with watched(wire.embedding.model if wire.embedding is not None else None):
            await _graph_insert(index, prepared.leaves, wire.llm, node)
    else:
        models = [
            wire.llm.llm if wire.llm is not None else None,
            wire.embedding.model if wire.embedding is not None else None,
            wire.pictures.model if wire.pictures is not None else None,
        ]
        if wire.kind == "document_summary":
            groups: dict = {}
            for chunk in prepared.leaves:
                if chunk.ref_doc_id is None:
                    raise Refusal(
                        f"{node} needs chunks that know which document they came from to add "
                        f"them to a document summary index. Split the documents with LLM Split "
                        f"Documents."
                    )
                groups.setdefault(chunk.ref_doc_id, []).append(chunk)
            batches = list(groups.values())
        else:
            size = held.get("batch_size", LLM_BATCH) if wire.kind in ("vector", "image") else LLM_BATCH
            batches = list(_batches(prepared.leaves, size))
        bar = progress.progress_bar(len(batches))
        with watched(*models), hushed():
            for batch in batches:
                runtime.check_interrupt()
                if wire.kind == "keyword":
                    index.insert_nodes(_text_only(batch))
                    index.docstore.add_documents(batch, allow_update=True)
                else:
                    index.insert_nodes(batch)
                bar.update(1)
    grown = wires.Index(
        index=index,
        kind=wire.kind,
        embedding=wire.embedding,
        llm=wire.llm,
        name="",
        size=size_of(index, wire.kind),
        pictures=wire.pictures,
    )
    return grown, stats


def _clean(kind: str, raw) -> dict:
    """A marker's recorded settings, each held to its range."""
    raw = raw if isinstance(raw, dict) else {}
    held: dict = {"kind": kind}
    for key, (low, high, default) in LIMITS.items():
        # Chunk sizes stay unrecorded for an index built from chunks split before it.
        if key in ("chunk_size", "chunk_overlap") and "chunk_size" not in raw:
            continue
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError):
            value = default
        held[key] = min(high, max(low, value))
    if "chunk_size" in held and held["chunk_overlap"] >= held["chunk_size"]:
        held["chunk_overlap"] = min(CHUNK_OVERLAP, held["chunk_size"] - 1)
    held["keywords"] = "simple" if raw.get("keywords") == "simple" else "llm"
    held.update(graph_settings({
        "extractor": raw.get("extractor") if raw.get("extractor") in EXTRACTORS else "simple",
        "max_paths_per_chunk": held["max_paths"],
        "entity_types": ", ".join(str(item) for item in raw.get("entity_types") or [] if isinstance(item, str)),
        "relation_types": ", ".join(str(item) for item in raw.get("relation_types") or [] if isinstance(item, str)),
    }))
    held["summary_query"] = str(raw.get("summary_query") or DEFAULT_SUMMARY_QUERY)[:MAX_QUERY_CHARS]
    held["embedded"] = bool(raw.get("embedded", True))
    return held


def _open(kind: str, context, held: dict, model, embedding, index_id: str | None = None, pictures=None):
    """A llama-index index of one kind around a storage context, its models passed explicitly.

    Args:
        kind: One of :data:`KINDS`.
        context: The storage context holding it.
        held: Its build settings.
        model: The :class:`wires.LanguageModel` it asks, or ``None``.
        embedding: The :class:`wires.EmbeddingModel` it embeds with, or ``None``.
        index_id: Which index in the store, or ``None`` for the only one.
        pictures: The :class:`wires.EmbeddingModel` pictures are embedded with, for an image index.

    Returns:
        The index, its settings recorded.

    Raises:
        Refusal: The store holds no index, or several and none named.
    """
    from llama_index.core import (
        DocumentSummaryIndex,
        KeywordTableIndex,
        PropertyGraphIndex,
        SimpleKeywordTableIndex,
        SummaryIndex,
        TreeIndex,
        VectorStoreIndex,
    )

    if index_id is not None:
        struct = context.index_store.get_index_struct(index_id)
    else:
        structs = context.index_store.index_structs()
        struct = structs[0] if len(structs) == 1 else None
    if struct is None:
        raise Refusal("The saved files hold no single index to open. Save the index again with LLM Save Index.")
    llm = model.llm if model is not None else None
    embed = embedding.model if embedding is not None else None
    transformations = [_splitter(held.get("chunk_size", CHUNK_SIZE), held.get("chunk_overlap", CHUNK_OVERLAP))]
    common = {"index_struct": struct, "storage_context": context, "transformations": transformations}
    held = dict(held)
    if kind == "vector":
        index = VectorStoreIndex(embed_model=embed, **common)
    elif kind == "image":
        from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

        if pictures is None:
            raise Refusal("An image index opens only with a picture model. Connect LLM Image Embedding Model.")
        index = MultiModalVectorStoreIndex(
            embed_model=embed if embed is not None else pictures.model,
            image_embed_model=pictures.model,
            **common,
        )
    elif kind == "summary":
        index = SummaryIndex(**common)
    elif kind == "tree":
        index = TreeIndex(llm=llm, num_children=held.get("num_children", 10), **common)
    elif kind == "keyword":
        maker = SimpleKeywordTableIndex if held.get("keywords") == "simple" else KeywordTableIndex
        extra = {} if held.get("keywords") == "simple" else {"llm": llm}
        index = maker(max_keywords_per_chunk=held.get("max_keywords", 10), **extra, **common)
    elif kind == "graph":
        holder = _holder_class()()
        held["holder"] = holder
        index = PropertyGraphIndex(
            llm=llm,
            kg_extractors=[holder],
            use_async=False,
            embed_model=embed,
            embed_kg_nodes=embed is not None,
            **common,
        )
    elif kind == "document_summary":
        embedded = bool(held.get("embedded")) and embed is not None
        index = DocumentSummaryIndex(
            llm=llm,
            embed_model=embed,
            summary_query=held.get("summary_query", DEFAULT_SUMMARY_QUERY),
            embed_summaries=embedded,
            **common,
        )
    else:
        raise Refusal(f"The saved index is of kind {kind[:40]!r}, which this pack does not open.")
    held.pop("kind", None)
    _remember(index, kind, **held)
    return index


def persist(wire: wires.Index, name) -> dict:
    """Write an index under a name in the pack's saved indexes, replacing what the pack saved there.

    Args:
        wire: The index.
        name: The name to save it under.

    Returns:
        What the marker records.

    Raises:
        Refusal: The name cannot be used, or the folder was not written by the pack.
        PathNotAllowed: The saved indexes folder is outside every folder the pack may write.
    """
    if wire.store is not None:
        raise Refusal(
            f"This index lives in the vector store {wire.store.label}, which keeps it already. Open it "
            f"again with LLM Vector Store and LLM Vector Store Index."
        )
    try:
        name = storage.check_name(name, "Index")
        directory = storage.clear_index(name)
    except sandbox.PathNotAllowed:
        raise
    except ValueError as error:
        raise Refusal(str(error)) from None
    held = {key: value for key, value in settings_of(wire.index).items() if key not in ("holder", "kind")}
    dimensions = (wire.embedding.dimensions if wire.embedding is not None else 0) or dimensions_of(wire.index)
    info = {
        "format": MARKER_FORMAT,
        "kind": wire.kind,
        "embedding": wire.embedding.label if wire.embedding is not None else None,
        "dimensions": int(dimensions),
        "llm": wire.llm.label if wire.llm is not None else None,
        "pictures": wire.pictures.label if wire.pictures is not None else None,
        "picture_dimensions": int(wire.pictures.dimensions) if wire.pictures is not None else 0,
        "chunks": int(wire.size),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": held,
    }
    storage.mark_index(name, {**info, "complete": False})
    wire.index.storage_context.persist(persist_dir=str(directory))
    storage.mark_index(name, {**info, "complete": True})
    return {**info, "complete": True}


def load(name, model, embedding, node: str, pictures=None) -> tuple[wires.Index, list[str]]:
    """Open an index saved under a name with the models it is given.

    Args:
        name: The saved name.
        model: The :class:`wires.LanguageModel` to hand it, or ``None``.
        embedding: The :class:`wires.EmbeddingModel` to hand it, or ``None``.
        node: The node's display name, for messages.
        pictures: The picture :class:`wires.EmbeddingModel` an image index needs, or ``None``.

    Returns:
        The index and any warnings about the models handed to it.

    Raises:
        Refusal: Nothing is saved under the name, the save did not finish, or a needed
            embedding model is missing or makes vectors of another length.
    """
    from llama_index.core import StorageContext

    if str(name or "") == NO_INDEXES:
        raise Refusal("No index has been saved yet. Build one and save it with LLM Save Index, then pick its name here.")
    try:
        name = storage.check_name(name, "Index")
    except ValueError as error:
        raise Refusal(str(error)) from None
    try:
        info = storage.index_info(name)
    except ValueError as error:
        raise Refusal(f"The marker of the saved index {name!r} cannot be read ({error}). Save the index again with LLM Save Index.") from None
    if info is None:
        raise Refusal(f"No index is saved under {name!r}. Save one with LLM Save Index, or pick a name from the menu.")
    if info.get("complete") is False:
        raise Refusal(f"Saving the index {name!r} did not finish. Save it again with LLM Save Index.")
    kind = str(info.get("kind", ""))
    if kind not in KINDS:
        raise Refusal(f"The index saved under {name!r} is of kind {kind[:40]!r}, which this pack does not open.")
    held = _clean(kind, info.get("settings"))
    recorded = info.get("embedding") if isinstance(info.get("embedding"), str) else None
    painted = info.get("pictures") if isinstance(info.get("pictures"), str) else None
    if kind == "image":
        if pictures is None:
            raise Refusal(
                f"The image index {name!r} was built with the picture model {painted or 'unknown'} and "
                f"needs it to search. Connect LLM Image Embedding Model to {node}'s image_embedding input."
            )
        picture_dims = info.get("picture_dimensions") if isinstance(info.get("picture_dimensions"), int) else 0
        if pictures.dimensions and picture_dims and pictures.dimensions != picture_dims:
            raise Refusal(
                f"The image index {name!r} holds picture vectors of {picture_dims} numbers from "
                f"{painted}, and {pictures.label} makes {pictures.dimensions}. Connect {painted}, or "
                f"build the index again with this model."
            )
        # Text embedded with the picture model reads back with it.
        if recorded is not None and recorded == painted and embedding is None:
            embedding = pictures
    needs_embedding = kind == "vector" or (kind in ("graph", "document_summary", "image") and recorded is not None)
    if kind == "document_summary":
        held["embedded"] = recorded is not None
    warnings: list[str] = []
    label = KINDS[kind]
    if needs_embedding and embedding is None:
        raise Refusal(
            f"The {label} index {name!r} was built with the embedding model {recorded or 'unknown'} "
            f"and needs it to search. Connect LLM Embedding Model to {node}'s embedding input."
        )
    if embedding is not None and not needs_embedding and kind in ("graph", "document_summary"):
        warnings.append(f"built without an embedding model, so {embedding.label} is not used")
        embedding = None
    if embedding is not None and recorded and embedding.label != recorded:
        warnings.append(f"built with {recorded}, opened with {embedding.label}")
    saved_dims = int(info.get("dimensions") or 0) if isinstance(info.get("dimensions"), int) else 0
    if needs_embedding and embedding is not None and embedding.dimensions and saved_dims and embedding.dimensions != saved_dims:
        raise Refusal(
            f"The index {name!r} holds vectors of {saved_dims} numbers from {recorded}, and "
            f"{embedding.label} makes {embedding.dimensions}. Connect {recorded}, or build the "
            f"index again with this model."
        )
    directory = storage.index_dir(name)
    try:
        context = StorageContext.from_defaults(persist_dir=str(directory))
    except FileNotFoundError:
        logger.debug("the saved index %s is missing a file", name, exc_info=True)
        raise Refusal(f"The saved index {name!r} is missing some of its files. Save it again with LLM Save Index.") from None
    except (OSError, ValueError, TypeError, KeyError) as error:
        logger.debug("the saved index %s could not be read", name, exc_info=True)
        raise Refusal(
            f"The saved index {name!r} could not be read ({type(error).__name__}). Save it again "
            f"with LLM Save Index."
        ) from None
    index = _open(kind, context, held, model, embedding, pictures=pictures)
    dims = dimensions_of(index)
    if needs_embedding and embedding is not None and embedding.dimensions and dims and embedding.dimensions != dims:
        raise Refusal(
            f"The index {name!r} holds vectors of {dims} numbers, and {embedding.label} makes "
            f"{embedding.dimensions}. Connect the model it was built with, or build it again."
        )
    wire = wires.Index(
        index=index,
        kind=kind,
        embedding=embedding,
        llm=model,
        name=name,
        size=size_of(index, kind),
        pictures=pictures if kind == "image" else None,
    )
    return wire, warnings
