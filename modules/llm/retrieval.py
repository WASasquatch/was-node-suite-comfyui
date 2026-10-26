"""Retrievers over built indexes, fusion and auto-merging retrievers, the postprocessors that
filter and rerank retrieved chunks, and the text a retrieval hands to a prompt.

Every llama-index class is built inside a function, on first use.
"""

from __future__ import annotations

import io
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any

from .. import deps
from . import FEATURE, runtime, wires

__all__ = [
    "FUSION_MODES",
    "RETRIEVER_TRANSFORMS",
    "INDEX_KINDS",
    "MODE_KINDS",
    "POSTPROCESSOR_KINDS",
    "SOURCE_FIELDS",
    "auto_merging_retriever",
    "auto_retriever",
    "file_label",
    "fusion_retriever",
    "index_node",
    "index_retriever",
    "metadata_fields",
    "page_label",
    "parse_filters",
    "picture_label",
    "postprocessor",
    "prompt_text",
    "quiet_library_logs",
    "recursive_retriever",
    "retrieve",
    "router_retriever",
    "rows",
    "searches_by_picture",
    "sources_text",
    "transform_retriever",
]

#: The index kinds an ``LLM_INDEX`` wire carries.
INDEX_KINDS = ("vector", "summary", "tree", "keyword", "graph", "document_summary", "image")

#: Each retriever mode, and the index kinds it reads.
MODE_KINDS = {
    "default": INDEX_KINDS,
    "mmr": ("vector",),
    "embedding": ("summary", "document_summary"),
    "llm": ("summary", "document_summary"),
    "simple": ("keyword",),
    "rake": ("keyword",),
    "select_leaf": ("tree",),
    "select_leaf_embedding": ("tree",),
    "root": ("tree",),
    "all_leaf": ("tree",),
    "graph_synonyms": ("graph",),
    "graph_vector": ("graph",),
    "pictures": ("image",),
    "text": ("image",),
}

#: How a fusion retriever merges its retrievers' rankings.
FUSION_MODES = ("reciprocal_rerank", "relative_score", "dist_based_score", "simple")

#: The steps the postprocessor node builds.
POSTPROCESSOR_KINDS = (
    "similarity cutoff",
    "keywords",
    "long context reorder",
    "sentence window",
    "previous and next",
    "recency",
    "rerank with llm",
    "rerank with embedding",
    "sentence optimizer",
)

#: Fields every source has in a template, beside its metadata keys.
SOURCE_FIELDS = ("rank", "score", "text", "file_name", "page")

#: llama-index loggers held at WARNING while retrieving.
LIBRARY_LOGGERS = (
    "llama_index.core.indices",
    "llama_index.core.retrievers",
    "llama_index.core.postprocessor",
)

#: Lines one metadata filter text may hold.
MAX_FILTER_LINES = 32

#: Characters one metadata filter line may hold.
MAX_FILTER_CHARS = 1000

#: Chunks read to learn which metadata keys hold numbers.
METADATA_SAMPLE = 2000

#: Entries a comma separated list may hold: keywords, ``in`` values, weights.
MAX_LIST = 64

#: Characters of one keyword.
MAX_KEYWORD_CHARS = 200

#: Characters of a template for sources.
MAX_TEMPLATE_CHARS = 4000

#: Fields one template for sources may name.
MAX_TEMPLATE_FIELDS = 64

#: Characters one field's value adds to the text.
MAX_FIELD_CHARS = 100_000

#: Characters :func:`sources_text` writes at most, whatever ``max_chars`` says.
MAX_TEXT_CHARS = 2_000_000

#: Characters of a file name or page label in headers and rows.
MAX_LABEL_CHARS = 120

#: Characters of a chunk excerpt in a published row.
EXCERPT_CHARS = 200

#: One metadata filter line: a key, an operator, a value.
FILTER_LINE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][\w.\-]{0,127})"
    r"(?:\s*(?P<symbol>==|!=|>=|<=|=|>|<)\s*|\s+(?P<word>not\s+in|in|contains)\s+)"
    r"(?P<value>.*?)\s*$",
    re.IGNORECASE,
)

#: A metadata key named in a widget.
METADATA_KEY = re.compile(r"[A-Za-z_][\w.\-]{0,127}")

#: A field in a template for sources.
TEMPLATE_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]{0,63})\}")

#: A number in a ranking answer.
NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: One word, for keyword matching.
WORD = re.compile(r"\w+")

#: Whitespace runs, collapsed in excerpts.
SPACES = re.compile(r"\s+")

#: Widget operator to llama-index filter operator.
OPERATORS = {
    "=": "==", "==": "==", "!=": "!=", ">": ">", ">=": ">=", "<": "<", "<=": "<=",
    "in": "in", "not in": "nin", "contains": "contains",
}

_classes: dict[str, type] = {}


def quiet_library_logs() -> None:
    """Hold llama-index's index, retriever and postprocessor loggers at WARNING."""
    for name in LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        if library.level < logging.WARNING:
            library.setLevel(logging.WARNING)


def _interrupt_type(error: BaseException) -> bool:
    """Whether an exception is ComfyUI's interrupt."""
    return type(error).__name__ == "InterruptProcessingException"


def _relabel(error: Exception, node: str) -> Exception:
    """The exception a node raises for a failure inside llama-index or a provider."""
    if isinstance(error, deps.DependencyError):
        return error
    if isinstance(error, ValueError):
        return ValueError(runtime.describe_error(error))
    return RuntimeError(f"{node} failed: {runtime.describe_error(error)}")


# Metadata filters


def parse_filters(text: str) -> list[tuple[str, str, str, bool]]:
    """Read metadata filter lines such as ``file_type = pdf`` or ``page_label > 3``.

    Args:
        text: One filter per line. Blank lines and lines starting with ``#`` are skipped.

    Returns:
        ``(key, operator, value, quoted)`` per line, the operator in llama-index's spelling.

    Raises:
        ValueError: A line does not read as a filter, or there are too many.
    """
    found = []
    for number, line in enumerate(str(text or "").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(found) >= MAX_FILTER_LINES:
            raise ValueError(f"metadata_filters holds more than {MAX_FILTER_LINES} filters. Keep the ones that matter.")
        if len(stripped) > MAX_FILTER_CHARS:
            raise ValueError(f"metadata_filters line {number} is longer than {MAX_FILTER_CHARS} characters.")
        match = FILTER_LINE.match(stripped)
        if match is None or not match.group("value"):
            raise ValueError(
                f"metadata_filters line {number} does not read as a filter. Write one per line as "
                f"`key = value`, `key != value`, `key > 3`, `key in a, b` or `key contains word`."
            )
        operator = match.group("symbol") or " ".join(match.group("word").lower().split())
        value = match.group("value")
        quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
        found.append((match.group("key"), OPERATORS[operator], value[1:-1] if quoted else value, quoted))
    return found


def _metadata_kinds(index) -> dict[str, set]:
    """Which python types each metadata key holds across a sample of an index's chunks."""
    kinds: dict[str, set] = {}
    store = getattr(getattr(index, "vector_store", None), "data", None)
    held = getattr(store, "metadata_dict", None)
    if isinstance(held, dict) and held:
        sample = list(held.values())[:METADATA_SAMPLE]
    else:
        docstore = getattr(index, "docstore", None)
        sample = [node.metadata for node in list((getattr(docstore, "docs", None) or {}).values())[:METADATA_SAMPLE]]
    for metadata in sample:
        for key, value in (metadata or {}).items():
            kinds.setdefault(key, set()).add(type(value))
    return kinds


def _typed(key: str, raw: str, quoted: bool, kinds: set):
    """A filter value in the type the metadata key holds."""
    if quoted:
        return raw
    numeric = {int, float}
    if kinds and kinds <= numeric:
        try:
            number = float(raw)
        except ValueError:
            raise ValueError(f"metadata_filters: `{key}` holds numbers, and `{raw}` is not one.") from None
        if not math.isfinite(number):
            raise ValueError(f"metadata_filters: `{raw}` is not a finite number.")
        return int(number) if number.is_integer() and kinds == {int} else number
    if not kinds:
        try:
            number = float(raw)
        except ValueError:
            return raw
        if math.isfinite(number):
            return int(number) if number.is_integer() and "." not in raw else number
    return raw


def _value(key: str, operator: str, raw: str, quoted: bool, kinds: set):
    """A filter's value, split into a list for ``in`` and ``nin``."""
    if operator not in ("in", "nin"):
        return _typed(key, raw, quoted, kinds)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts or len(parts) > MAX_LIST:
        raise ValueError(f"metadata_filters: `{key} in` takes 1 to {MAX_LIST} values separated by commas.")
    values = []
    for part in parts:
        inner = len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'"
        values.append(_typed(key, part[1:-1] if inner else part, quoted or inner, kinds))
    if any(isinstance(item, str) for item in values):
        return [str(item) for item in values]
    if any(isinstance(item, float) for item in values):
        return [float(item) for item in values]
    return values


def metadata_filters(parsed, index=None, match: str = "all"):
    """llama-index ``MetadataFilters`` from parsed lines, values typed after the index's metadata.

    Args:
        parsed: What :func:`parse_filters` returned.
        index: The llama-index index whose chunks show each key's type, or ``None``.
        match: ``all`` keeps a chunk passing every filter, ``any`` one passing at least one.

    Returns:
        The filters, or ``None`` when there are none.
    """
    if not parsed:
        return None
    runtime.core()
    from llama_index.core.vector_stores.types import FilterCondition, MetadataFilter, MetadataFilters

    kinds = _metadata_kinds(index) if index is not None else {}
    filters = [
        MetadataFilter(key=key, value=_value(key, operator, raw, quoted, kinds.get(key, set())), operator=operator)
        for key, operator, raw, quoted in parsed
    ]
    condition = FilterCondition.OR if match == "any" else FilterCondition.AND
    return MetadataFilters(filters=filters, condition=condition)


def _passes(filters, metadata) -> bool:
    """Whether one chunk's metadata passes the filters."""
    from llama_index.core.vector_stores.utils import build_metadata_filter_fn

    try:
        return bool(build_metadata_filter_fn(lambda _: metadata or {}, filters)(""))
    except (TypeError, ValueError):
        return False


# Retrievers


def _bounded_class():
    """The retriever wrapping another to filter, rank and cap its results, defined on first use."""
    if "bounded" in _classes:
        return _classes["bounded"]
    runtime.core()
    from llama_index.core.base.base_retriever import BaseRetriever

    class BoundedRetriever(BaseRetriever):
        """Another retriever's results, filtered by metadata, ranked and capped."""

        def __init__(self, inner, cap=None, filters=None, rank=False, zero=False, native=None):
            super().__init__()
            self.inner = inner
            self._cap = cap
            self._filters = filters
            self._rank = rank
            self._zero = zero
            self._native = native

        def _finish(self, found):
            if self._filters is not None:
                found = [item for item in found if _passes(self._filters, item.node.metadata)]
            if self._zero:
                for item in found:
                    if item.score is None:
                        item.score = 0.0
            if self._rank:
                found = sorted(found, key=lambda item: -math.inf if item.score is None else item.score, reverse=True)
            return found[: self._cap] if self._cap else found

        def _retrieve(self, query_bundle):
            try:
                found = self.inner.retrieve(query_bundle)
            except TypeError as error:
                if self._native is None:
                    raise
                raise ValueError(
                    f"A metadata filter compares values of different kinds ({runtime.describe_error(error)}). "
                    f"Quote the value to compare it as text, or filter on a key holding numbers."
                ) from None
            return self._finish(found)

        async def _aretrieve(self, query_bundle):
            return self._finish(await self.inner.aretrieve(query_bundle))

    _classes["bounded"] = BoundedRetriever
    return BoundedRetriever


def _picture_class():
    """The retriever over an image index that finds text, pictures or both, defined on first use."""
    if "picture" in _classes:
        return _classes["picture"]
    runtime.core()
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import QueryBundle

    class PictureRetriever(BaseRetriever):
        """An image index's text and picture searches, or a search by a picture."""

        def __init__(self, inner, what: str = "default"):
            super().__init__()
            self.inner = inner
            self._what = what

        def _retrieve(self, query_bundle):
            if query_bundle.image_path is not None:
                fresh = QueryBundle(query_str="", image_path=query_bundle.image_path)
                return self.inner.image_to_image_retrieve(fresh)
            found = []
            if self._what in ("default", "text"):
                found.extend(self.inner.text_retrieve(QueryBundle(query_str=query_bundle.query_str)))
            if self._what in ("default", "pictures"):
                found.extend(self.inner.text_to_image_retrieve(QueryBundle(query_str=query_bundle.query_str)))
            return found

        async def _aretrieve(self, query_bundle):
            return self._retrieve(query_bundle)

    _classes["picture"] = PictureRetriever
    return PictureRetriever


def searches_by_picture(wire) -> bool:
    """Whether a retriever wire can search by a picture: an image index's retriever, directly."""
    held = wire.retriever
    for _depth in range(8):
        if type(held).__name__ == "PictureRetriever":
            return True
        held = getattr(held, "inner", None)
        if held is None:
            return False
    return False


def _need(model, what: str, mode: str, kind: str, other: str = ""):
    """A model a mode calls, or a ValueError naming what to connect."""
    if model is None:
        language = what == "a language model"
        hint = "an LLM model node to the llm input" if language else "LLM Embedding Model to the embedding input"
        raise ValueError(
            f"LLM Index Retriever: the {mode} mode on a {kind} index calls {what}, and none is "
            f"connected. Connect {hint}, or build the index with one{other}."
        )
    return model


def index_retriever(index, mode: dict | str, top_k: int, filters_text: str = "", match: str = "all",
                    llm=None, embedding=None):
    """A retriever over a built index in one of its retriever modes.

    Args:
        index: The :class:`~modules.llm.wires.Index`.
        mode: The mode combo's value: a dict holding ``mode`` and that mode's settings.
        top_k: The most chunks one retrieval returns.
        filters_text: Metadata filter lines, see :func:`parse_filters`.
        match: ``all`` or ``any`` of the filters.
        llm: A :class:`~modules.llm.wires.LanguageModel` used in place of the index's own.
        embedding: An :class:`~modules.llm.wires.EmbeddingModel` used in place of the index's own.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: The mode does not suit the index, or a model it calls is missing.
    """
    runtime.core()
    quiet_library_logs()
    choice = mode if isinstance(mode, dict) else {"mode": str(mode or "default")}
    name = str(choice.get("mode", "default"))
    kind = index.kind
    if kind not in INDEX_KINDS:
        raise ValueError(f"LLM Index Retriever cannot read a `{kind}` index. Connect one of the LLM index nodes.")
    if name not in MODE_KINDS:
        raise ValueError(f"`{name}` is not a retriever mode. Pick one of: {', '.join(MODE_KINDS)}.")
    if kind not in MODE_KINDS[name]:
        fitting = [key for key, kinds in MODE_KINDS.items() if kind in kinds]
        raise ValueError(
            f"The {name} mode does not read a {kind} index. For a {kind} index pick one of: {', '.join(fitting)}."
        )
    llm_wire = llm or index.llm
    embed_wire = embedding or index.embedding
    parsed = parse_filters(filters_text)
    built = index.index
    builder = {
        "vector": _vector, "summary": _summary, "keyword": _keyword, "tree": _tree,
        "document_summary": _document_summary, "graph": _graph, "image": _image,
    }[kind]
    filters = metadata_filters(parsed, built, match)
    inner, cap, native, ranked = builder(built, name, choice, int(top_k), llm_wire, embed_wire, filters)
    bounded = _bounded_class()(
        inner, cap=cap, filters=None if native else filters, rank=ranked, native=filters if native else None
    )
    label = f"{kind} {name} top {int(top_k)}"
    depends = tuple(item for item in (index, llm, embedding) if item is not None)
    facts = {"index": kind, "mode": name, "top_k": int(top_k)}
    if parsed:
        facts["filters"] = f"{len(parsed)} ({'all' if match != 'any' else 'any'})"
        facts["filtered by"] = "the index" if native else "after retrieval"
    if kind == "image":
        facts["finds"] = {"default": "text and pictures", "pictures": "pictures", "text": "text"}[name]
    wire = wires.Retriever(retriever=bounded, label=label, depends=depends, pictures=kind == "image")
    return wire, facts


def _image(built, name, choice, top_k, llm, embedding, filters):
    """An image index's search for text, pictures or both, each to ``top_k``, text first."""
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    _kind_check(built, MultiModalVectorStoreIndex, "image")
    inner = built.as_retriever(similarity_top_k=top_k, image_similarity_top_k=top_k, filters=filters)
    return _picture_class()(inner, name), None, True, False


def _vector(built, name, choice, top_k, llm, embedding, filters):
    """A vector index's similarity or MMR retriever."""
    from llama_index.core.indices.vector_store.base import VectorStoreIndex

    _kind_check(built, VectorStoreIndex, "vector")
    embed = _need(embedding, "an embedding model", name, "vector")
    options = {"similarity_top_k": top_k, "embed_model": embed.model, "filters": filters}
    if name == "mmr":
        options["vector_store_query_mode"] = "mmr"
        options["vector_store_kwargs"] = {"mmr_threshold": float(choice.get("mmr_threshold", 0.5))}
    return built.as_retriever(**options), None, True, False


def _summary(built, name, choice, top_k, llm, embedding, filters):
    """A summary index's retriever: every chunk, ranked by embedding, or chosen by a language model."""
    from llama_index.core.indices.list.base import SummaryIndex
    from llama_index.core.indices.list.retrievers import (
        SummaryIndexEmbeddingRetriever,
        SummaryIndexLLMRetriever,
        SummaryIndexRetriever,
    )

    _kind_check(built, SummaryIndex, "summary")
    if name == "embedding":
        embed = _need(embedding, "an embedding model", name, "summary")
        return SummaryIndexEmbeddingRetriever(built, embed_model=embed.model, similarity_top_k=top_k), None, False, False
    if name == "llm":
        model = _need(llm, "a language model", name, "summary")
        batch = int(choice.get("choice_batch_size", 10))
        inner = SummaryIndexLLMRetriever(
            built, llm=model.llm, choice_batch_size=batch, parse_choice_select_answer_fn=_parse_choices
        )
        return inner, top_k, False, True
    return SummaryIndexRetriever(built), top_k, False, False


def _keyword(built, name, choice, top_k, llm, embedding, filters):
    """A keyword index's retriever, finding keywords with a language model, by counting or with RAKE."""
    from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
    from llama_index.core.indices.keyword_table.retrievers import (
        KeywordTableGPTRetriever,
        KeywordTableRAKERetriever,
        KeywordTableSimpleRetriever,
    )

    _kind_check(built, BaseKeywordTableIndex, "keyword")
    keywords = int(choice.get("max_keywords", 10))
    options = {"num_chunks_per_query": top_k, "max_keywords_per_query": keywords}
    if name == "simple":
        return KeywordTableSimpleRetriever(built, **options), None, False, False
    if name == "rake":
        deps.require("rake_nltk", feature=FEATURE)
        return KeywordTableRAKERetriever(built, **options), None, False, False
    model = _need(llm, "a language model", "default", "keyword", ", or pick simple or rake")
    return KeywordTableGPTRetriever(built, llm=model.llm, **options), None, False, False


def _tree(built, name, choice, top_k, llm, embedding, filters):
    """A tree index's retriever: walking down by a model's choice, the root summaries or every leaf."""
    from llama_index.core.indices.prompt_helper import PromptHelper
    from llama_index.core.indices.tree.all_leaf_retriever import TreeAllLeafRetriever
    from llama_index.core.indices.tree.base import TreeIndex, TreeRetrieverMode
    from llama_index.core.indices.tree.select_leaf_embedding_retriever import TreeSelectLeafEmbeddingRetriever
    from llama_index.core.indices.tree.select_leaf_retriever import TreeSelectLeafRetriever
    from llama_index.core.indices.tree.tree_root_retriever import TreeRootRetriever

    _kind_check(built, TreeIndex, "tree")
    walk = "select_leaf" if name == "default" else name
    try:
        built._validate_build_tree_required(TreeRetrieverMode(walk))
    except ValueError:
        raise ValueError(
            f"The {walk} mode walks the tree's summaries, and this tree index was built without them. "
            f"Pick all_leaf, or rebuild the index with its tree built."
        ) from None
    # The root and all_leaf modes hand on every summary or leaf they reach, uncut by top_k.
    if walk == "root":
        return TreeRootRetriever(built), None, False, False
    if walk == "all_leaf":
        return TreeAllLeafRetriever(built), None, False, False
    branches = int(choice.get("child_branch_factor", 1))
    if walk == "select_leaf_embedding":
        embed = _need(embedding, "an embedding model", walk, "tree")
        inner = TreeSelectLeafEmbeddingRetriever(built, embed_model=embed.model, child_branch_factor=branches)
    else:
        _need(llm, "a language model", walk, "tree")
        inner = TreeSelectLeafRetriever(built, child_branch_factor=branches)
    if llm is not None:
        inner._llm = llm.llm
        inner._prompt_helper = PromptHelper.from_llm_metadata(llm.llm.metadata)
    return inner, top_k, False, False


def _document_summary(built, name, choice, top_k, llm, embedding, filters):
    """A document summary index's retriever, picking documents by their summaries."""
    from llama_index.core.indices.document_summary.base import DocumentSummaryIndex
    from llama_index.core.indices.document_summary.retrievers import (
        DocumentSummaryIndexEmbeddingRetriever,
        DocumentSummaryIndexLLMRetriever,
    )

    _kind_check(built, DocumentSummaryIndex, "document summary")
    if name == "llm":
        model = _need(llm, "a language model", name, "document summary")
        batch = int(choice.get("choice_batch_size", 10))
        inner = DocumentSummaryIndexLLMRetriever(
            built, llm=model.llm, choice_top_k=top_k, choice_batch_size=batch,
            parse_choice_select_answer_fn=_parse_choices,
        )
        return inner, None, False, False
    if not getattr(built, "_embed_summaries", True):
        raise ValueError(
            "This document summary index was built without embedding its summaries, so its default "
            "and embedding modes cannot rank them. Pick llm, or rebuild the index with summaries embedded."
        )
    embed = _need(embedding, "an embedding model", name, "document summary")
    return DocumentSummaryIndexEmbeddingRetriever(built, similarity_top_k=top_k, embed_model=embed.model), None, False, False


def _graph(built, name, choice, top_k, llm, embedding, filters):
    """A property graph's retriever, walking out from entities found by synonyms, vectors or both."""
    from llama_index.core.indices.property_graph.base import PropertyGraphIndex
    from llama_index.core.indices.property_graph.retriever import PGRetriever
    from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import LLMSynonymRetriever
    from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever

    if not isinstance(built, PropertyGraphIndex):
        if name != "default":
            raise ValueError("This graph index is not a property graph, so it has only the default mode.")
        return built.as_retriever(similarity_top_k=top_k), top_k, False, False
    store = built.property_graph_store
    depth = int(choice.get("path_depth", 1))
    vectors = bool(getattr(store, "supports_vector_queries", False) or built.vector_store is not None)
    synonyms = name == "graph_synonyms" or (name == "default" and llm is not None)
    searched = name == "graph_vector" or (name == "default" and embedding is not None and vectors)
    subs = []
    if synonyms:
        model = _need(llm, "a language model", name, "graph")
        subs.append(LLMSynonymRetriever(
            graph_store=store, llm=model.llm, include_text=True,
            max_keywords=int(choice.get("max_keywords", 10)), path_depth=depth,
        ))
    if searched:
        embed = _need(embedding, "an embedding model", name, "graph")
        if not vectors:
            raise ValueError(
                "This graph index holds no embeddings of its entities, so graph_vector cannot search "
                "them. Pick graph_synonyms, or rebuild the graph with an embedding model."
            )
        subs.append(VectorContextRetriever(
            graph_store=store, vector_store=built.vector_store, embed_model=embed.model,
            similarity_top_k=top_k, path_depth=depth, include_text=True,
        ))
    if not subs:
        raise ValueError(
            "LLM Index Retriever: the default mode on a graph index calls a language model, an "
            "embedding model or both, and neither is connected. Connect one to the llm or embedding input."
        )
    return PGRetriever(subs, use_async=False), top_k, False, False


def _kind_check(built, expected, kind: str) -> None:
    """Refuse an index whose object is not the kind its wire names."""
    if not isinstance(built, expected):
        raise ValueError(
            f"The index on this wire says it is a {kind} index and holds a {type(built).__name__}. "
            f"Rebuild it with the matching LLM index node."
        )


def parse_weights(text: str, count: int) -> list[float] | None:
    """Retriever weights from ``0.6, 0.4``.

    Args:
        text: Numbers separated by commas, or blank for equal weights.
        count: Retrievers connected.

    Returns:
        One weight per retriever, or ``None`` when the text is blank.

    Raises:
        ValueError: A weight is not a number, is negative, the count differs, or all are zero.
    """
    parts = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) != count:
        raise ValueError(
            f"weights names {len(parts)} number(s) for {count} connected retriever(s). Give one per "
            f"retriever, in slot order, such as `0.6, 0.4`, or leave it blank for equal weights."
        )
    weights = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"weights: `{part}` is not a number.") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"weights: `{part}` is not a number of 0 or more.")
        weights.append(value)
    if sum(weights) <= 0:
        raise ValueError("weights adds up to 0. Give at least one retriever a weight above 0.")
    return weights


def fusion_retriever(retrievers, mode: str, top_k: int, num_queries: int = 1, llm=None, weights: str = ""):
    """A retriever merging several retrievers' rankings, optionally over rewritten questions.

    Args:
        retrievers: :class:`~modules.llm.wires.Retriever` wires, in slot order.
        mode: One of :data:`FUSION_MODES`.
        top_k: The most chunks one retrieval returns.
        num_queries: Questions searched, the one asked included. Above 1 a language model writes the rest.
        llm: The :class:`~modules.llm.wires.LanguageModel` writing them, or ``None``.
        weights: Per-retriever weights for the score fusions, see :func:`parse_weights`.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: A setting is out of reach or a model it calls is missing.
    """
    runtime.core()
    from llama_index.core.retrievers import QueryFusionRetriever

    if mode not in FUSION_MODES:
        raise ValueError(f"`{mode}` is not a fusion mode. Pick one of: {', '.join(FUSION_MODES)}.")
    if int(num_queries) > 1 and llm is None:
        raise ValueError(
            "LLM Fusion Retriever needs a language model to write the extra questions num_queries "
            "asks for. Connect an LLM model node to llm, or set num_queries to 1."
        )
    parsed = parse_weights(weights, len(retrievers))
    subs = [_bounded_class()(wire.retriever, zero=True) for wire in retrievers]
    options = {}
    if llm is not None:
        options["llm"] = llm.llm
    fused = QueryFusionRetriever(
        subs, mode=mode, similarity_top_k=int(top_k), num_queries=int(num_queries), use_async=False,
        retriever_weights=parsed, verbose=False, **options,
    )
    label = f"fusion of {len(retrievers)} ({mode}) top {int(top_k)}"
    depends = tuple(retrievers) + ((llm,) if llm is not None and int(num_queries) > 1 else ())
    facts = {
        "retrievers": ", ".join(wire.label for wire in retrievers)[:200],
        "mode": mode,
        "questions": int(num_queries),
    }
    if parsed:
        facts["weights"] = ", ".join(f"{value / sum(parsed):.2f}" for value in parsed)
    pictures = any(wire.pictures for wire in retrievers)
    return wires.Retriever(retriever=fused, label=label, depends=depends, pictures=pictures), facts


def router_retriever(tools, model, selector: str = "single"):
    """A retriever handing each question to the retriever, or retrievers, best suited.

    Args:
        tools: Retriever tools, each describing what its retriever finds.
        model: The :class:`~modules.llm.wires.LanguageModel` choosing.
        selector: ``single`` or ``multi``.

    Returns:
        ``(retriever wire, facts)``.
    """
    runtime.core()
    from llama_index.core.retrievers import RouterRetriever

    from .query import selector_for

    routed = RouterRetriever(selector=selector_for(model, selector), retriever_tools=list(tools), llm=model.llm)
    depends = (model, *tools)
    label = f"{selector} router over {len(tools)} retriever(s)"
    facts = {"model": model.label, "selector": selector, "retrievers": ", ".join(tool.metadata.name for tool in tools)[:200]}
    return wires.Retriever(retriever=routed, label=label, depends=depends), facts


#: Metadata keys an auto retriever never offers: where a file sits and when it changed.
_AUTO_HIDDEN = frozenset({
    "folder", "relative_path", "size", "modified", "source", "extension", "width", "height",
    "doc_id", "document_id", "ref_doc_id",
})


def metadata_fields(index, written: str = "") -> list[tuple[str, str, str]]:
    """The metadata fields an auto retriever may filter on: typed lines, or what the index holds.

    Args:
        index: The llama-index index.
        written: Lines of ``name (type): description``; blank reads the keys off the index.

    Returns:
        ``(name, type, description)`` triples.

    Raises:
        ValueError: A written line is not in that form.
    """
    fields = []
    for line in str(written or "").splitlines():
        if not line.strip():
            continue
        found = re.fullmatch(r"\s*([A-Za-z_][\w.\-]{0,63})\s*(?:\((str|int|float|bool)\))?\s*:\s*(.*)", line)
        if found is None:
            raise ValueError(
                f"The metadata line {line.strip()[:80]!r} is not `name (type): description`, such as "
                f"`file_name (str): the file a chunk came from`."
            )
        fields.append((found.group(1), found.group(2) or "str", found.group(3).strip()[:300]))
    if fields:
        return fields[:MAX_LIST]
    kinds = _metadata_kinds(index)
    for name in sorted(kinds)[:MAX_LIST]:
        if name in _AUTO_HIDDEN or name.startswith("_"):
            continue
        seen = kinds[name]
        if seen == {bool}:
            kind = "bool"
        elif float in seen and seen <= {int, float}:
            kind = "float"
        elif seen <= {int}:
            kind = "int"
        else:
            kind = "str"
        fields.append((name, kind, f"the {name.replace('_', ' ')} of a chunk"))
    return fields


def auto_retriever(index, model, content: str, written: str, top_k: int):
    """A retriever whose model reads each question and sets metadata filters and a search text.

    Args:
        index: The vector :class:`~modules.llm.wires.Index`.
        model: The :class:`~modules.llm.wires.LanguageModel` writing the filters.
        content: What the chunks are, such as ``studio notes on rendering and billing``.
        written: Metadata lines, see :func:`metadata_fields`.
        top_k: The most chunks one retrieval returns.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: The index is not a vector index, or holds no metadata to filter on.
    """
    runtime.core()
    from llama_index.core.retrievers import VectorIndexAutoRetriever
    from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

    if index.kind != "vector":
        raise ValueError(f"LLM Auto Retriever reads a vector index, and was handed a {index.kind} index.")
    fields = metadata_fields(index.index, written)
    if not fields:
        raise ValueError(
            "LLM Auto Retriever found no metadata to filter on. Type the fields into metadata_fields, or "
            "index documents that carry metadata, such as from LLM Load Documents."
        )
    info = VectorStoreInfo(
        content_info=str(content or "").strip()[:MAX_TEMPLATE_CHARS] or "chunks of documents",
        metadata_info=[MetadataInfo(name=name, type=kind, description=text) for name, kind, text in fields],
    )
    retriever = VectorIndexAutoRetriever(
        index.index, vector_store_info=info, llm=model.llm, similarity_top_k=int(top_k),
        max_top_k=max(int(top_k), 50), verbose=False,
    )
    label = f"auto top {int(top_k)}"
    facts = {"model": model.label, "filters on": ", ".join(name for name, _kind, _text in fields)[:200]}
    return wires.Retriever(retriever=retriever, label=label, depends=(index, model)), facts


#: Transforms a transform retriever applies to each question.
RETRIEVER_TRANSFORMS = ("hyde", "decompose")


def index_node(text: str, target: str):
    """A chunk that hands retrieval on to the retriever or query engine a recursive retriever names.

    Args:
        text: What is behind it, read when it is searched, such as a summary.
        target: The name of the tool it hands on to.

    Returns:
        A llama-index ``IndexNode``.

    Raises:
        ValueError: The text is empty or the target is not a tool name.
    """
    runtime.core()
    from llama_index.core.schema import IndexNode

    from .tools import check_name

    body = str(text or "").strip()
    if not body:
        raise ValueError("LLM Index Node needs text saying what is behind it, such as a summary of the documents.")
    name = check_name(target, "LLM Index Node", "target")
    return IndexNode(text=body[:MAX_TEXT_CHARS], index_id=name, id_=f"index-node:{name}", metadata={"target": name})


def recursive_retriever(root, tools):
    """A retriever whose found index nodes hand the question on to the tools they name.

    Args:
        root: The :class:`~modules.llm.wires.Retriever` searched first.
        tools: Retriever and query engine tools, each reached by its name.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: A tool is neither a retriever tool nor a query engine tool, or is named ``root``.
    """
    runtime.core()
    from llama_index.core.retrievers import RecursiveRetriever
    from llama_index.core.tools import QueryEngineTool, RetrieverTool

    retrievers = {"root": root.retriever}
    engines = {}
    for tool in tools:
        name = tool.metadata.name
        if name == "root":
            raise ValueError("LLM Recursive Retriever: no tool may be named `root`. Rename it.")
        if isinstance(tool, RetrieverTool):
            retrievers[name] = tool.retriever
        elif isinstance(tool, QueryEngineTool):
            engines[name] = tool.query_engine
        else:
            raise ValueError(
                f"LLM Recursive Retriever takes retriever and query engine tools, and was handed a "
                f"{type(tool).__name__}. Use LLM Retriever Tool or LLM Query Engine Tool."
            )
    built = RecursiveRetriever("root", retriever_dict=retrievers, query_engine_dict=engines, verbose=False)
    label = f"recursive over {len(tools)} target(s)"
    facts = {"root": root.label, "targets": ", ".join(tool.metadata.name for tool in tools)[:200]}
    depends = (root, *tools)
    return wires.Retriever(retriever=built, label=label, depends=depends, pictures=root.pictures), facts


def transform_retriever(retriever, model, transform: str, include_original: bool = True, summary: str = ""):
    """A retriever that rewrites each question before another retriever searches with it.

    Args:
        retriever: The :class:`~modules.llm.wires.Retriever` searching.
        model: The :class:`~modules.llm.wires.LanguageModel` rewriting.
        transform: ``hyde`` writes an answer and searches with it; ``decompose`` asks a simpler question.
        include_original: For ``hyde``, search with the question as well as the written answer.
        summary: For ``decompose``, what the documents hold, read when it simplifies.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: The transform is unknown.
    """
    runtime.core()
    from llama_index.core.indices.query.query_transform import DecomposeQueryTransform, HyDEQueryTransform
    from llama_index.core.retrievers import TransformRetriever

    if transform == "hyde":
        rewriter = HyDEQueryTransform(llm=model.llm, include_original=bool(include_original))
        metadata = None
    elif transform == "decompose":
        rewriter = DecomposeQueryTransform(llm=model.llm)
        metadata = {"index_summary": str(summary or "").strip()[:MAX_TEMPLATE_CHARS] or "The documents."}
    else:
        raise ValueError(f"`{transform}` is not a transform. Pick one of {', '.join(RETRIEVER_TRANSFORMS)}.")
    built = TransformRetriever(retriever.retriever, query_transform=rewriter, transform_metadata=metadata)
    facts = {"transform": transform, "model": model.label, "retriever": retriever.label}
    label = f"{transform} over {retriever.label}"
    return wires.Retriever(retriever=built, label=label, depends=(retriever, model), pictures=retriever.pictures), facts


def _hierarchy(built) -> tuple[int, int, int]:
    """``(chunks read, chunks with a parent, parents in the docstore)`` over a sample of an index."""
    docstore = built.docstore
    ids = list(getattr(built.index_struct, "nodes_dict", {}).values())[:64]
    read = with_parent = present = 0
    for node_id in ids:
        node = docstore.get_node(node_id, raise_error=False)
        if node is None:
            continue
        read += 1
        parent = node.parent_node
        if parent is None:
            continue
        with_parent += 1
        if docstore.document_exists(parent.node_id):
            present += 1
    return read, with_parent, present


def auto_merging_retriever(index, top_k: int, merge_ratio: float):
    """A vector retriever that replaces sibling chunks with the larger chunk holding them.

    Args:
        index: A vector :class:`~modules.llm.wires.Index` built from hierarchical chunks.
        top_k: Small chunks searched before merging.
        merge_ratio: Share of a parent's chunks that must be found before the parent replaces them.

    Returns:
        ``(retriever wire, facts)``.

    Raises:
        ValueError: The index is not a vector index, or its docstore holds no parent chunks.
    """
    runtime.core()
    from llama_index.core.indices.vector_store.base import VectorStoreIndex
    from llama_index.core.retrievers import AutoMergingRetriever

    quiet_library_logs()
    built = index.index
    if index.kind != "vector" or not isinstance(built, VectorStoreIndex):
        raise ValueError(
            f"LLM Auto-Merging Retriever reads a vector index, and this is a {index.kind} index. Build "
            f"a vector index from hierarchical chunks and connect it."
        )
    read, with_parent, present = _hierarchy(built)
    if not with_parent:
        raise ValueError(
            "LLM Auto-Merging Retriever found no parent chunks: none of this index's chunks belongs to "
            "a larger one. Split the documents with the hierarchical splitter, build the vector index "
            "from every chunk it writes, and connect that index."
        )
    if not present:
        raise ValueError(
            "LLM Auto-Merging Retriever: this index's chunks name parent chunks its docstore does not "
            "hold. Build the vector index from every chunk the hierarchical splitter writes, the "
            "larger ones included, so the docstore keeps them."
        )
    embed = index.embedding.model if index.embedding is not None else None
    options = {"similarity_top_k": int(top_k)}
    if embed is not None:
        options["embed_model"] = embed
    base = built.as_retriever(**options)
    merging = AutoMergingRetriever(base, built.storage_context, simple_ratio_thresh=float(merge_ratio), verbose=False)
    label = f"auto-merging top {int(top_k)}"
    facts = {"top_k": int(top_k), "merge ratio": f"{float(merge_ratio):.2f}", "chunks checked": f"{present} of {read} with parents"}
    return wires.Retriever(retriever=merging, label=label, depends=(index,)), facts


# Retrieving


def _copied(item):
    """A scored node whose node and metadata may be changed without touching the original."""
    from llama_index.core.schema import NodeWithScore

    node = item.node.model_copy(update={"metadata": dict(item.node.metadata)})
    return NodeWithScore(node=node, score=item.score)


def retrieve(retriever, query: str, postprocessors=(), node: str = "LLM Retrieve", picture=None):
    """Retrieve chunks for a question, or pictures like a picture, and run them through a chain.

    Args:
        retriever: The :class:`~modules.llm.wires.Retriever`.
        query: The question.
        postprocessors: :class:`~modules.llm.wires.Postprocessor` steps in chain order.
        node: The node's name, for messages.
        picture: PNG bytes of a picture to search by in place of the question, or ``None``.

    Returns:
        ``(chunks, retrieved, seconds)``: the scored chunks after every step, how many the
        retriever found, and the seconds it all took.

    Raises:
        RuntimeError: The retriever or a step failed.
        ValueError: A picture was given to a retriever that cannot search by one.
    """
    core = runtime.core()
    quiet_library_logs()
    started = time.perf_counter()
    if picture is not None and not searches_by_picture(retriever):
        raise ValueError(
            f"{node} was given a picture to search by, and {retriever.label} searches text only. "
            f"Connect LLM Index Retriever over an LLM Image Index, or disconnect the image."
        )
    if picture is not None:
        bundle = core.QueryBundle(query_str=str(query or ""), image_path=io.BytesIO(picture))
    else:
        bundle = core.QueryBundle(query_str=str(query))
    try:
        found = [_copied(item) for item in retriever.retriever.retrieve(bundle)]
        retrieved = len(found)
        for step in postprocessors or ():
            runtime.check_interrupt()
            found = list(step.processor.postprocess_nodes(found, query_bundle=bundle))
    except Exception as error:
        if _interrupt_type(error):
            raise
        raise _relabel(error, node) from None
    return found, retrieved, time.perf_counter() - started


def _cut(text: str, limit: int) -> str:
    """Text cut to a length, ``...`` marking the cut."""
    text = str(text)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def picture_label(node) -> str:
    """``picture <file name>`` for a picture document, ``""`` for anything else."""
    from llama_index.core.schema import ImageNode

    if not isinstance(node, ImageNode):
        return ""
    return f"picture {file_label(node)}".strip()


def file_label(node) -> str:
    """The file a chunk came from, by name alone, or ``""``."""
    metadata = getattr(node, "metadata", None) or {}
    for key in ("file_name", "file_path", "source", "filename"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return _cut(name, MAX_LABEL_CHARS)
    return ""


def page_label(node) -> str:
    """The page a chunk came from, or ``""``."""
    metadata = getattr(node, "metadata", None) or {}
    for key in ("page_label", "page", "page_number"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _cut(str(value).strip(), 32)
    return ""


def _content(node) -> str:
    """A chunk's own text, without its metadata."""
    from llama_index.core.schema import MetadataMode

    return node.get_content(metadata_mode=MetadataMode.NONE)


def prompt_text(items) -> str:
    """Retrieved chunks as numbered blocks for a prompt, each headed by its file and page.

    Args:
        items: Scored chunks, best first.

    Returns:
        ``[1] report.pdf, page 3`` then the chunk's text, blocks separated by a blank line.
    """
    blocks = []
    for rank, item in enumerate(items, 1):
        head = f"[{rank}]"
        name, page = file_label(item.node), page_label(item.node)
        if name:
            head += f" {name}"
        if page:
            head += f", page {page}"
        blocks.append(f"{head}\n{_content(item.node).strip() or picture_label(item.node)}")
    return "\n\n".join(blocks)


def _score(value) -> str:
    """A score in three decimals, ``""`` for none."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.3f}" if math.isfinite(number) else ""


def rows(items, limit: int = 8) -> list[dict]:
    """Published rows for ranked chunks: an excerpt, noted with rank, score and file.

    Args:
        items: Scored chunks, best first.
        limit: Rows to build.

    Returns:
        Mappings with ``text``, ``note`` and ``clipped``.
    """
    built = []
    for rank, item in enumerate(items[:limit], 1):
        text = SPACES.sub(" ", _content(item.node)).strip()
        excerpt = _cut(text, EXCERPT_CHARS)
        page = page_label(item.node)
        parts = (f"{rank}.", _score(item.score), file_label(item.node), f"p{page}" if page else "")
        note = " ".join(part for part in parts if part)
        shown = excerpt or picture_label(item.node) or "(empty chunk)"
        built.append({"text": shown, "note": note, "clipped": len(excerpt) < len(text)})
    return built


def _unescaped(text: str) -> str:
    """A separator with ``\\n`` and ``\\t`` turned into a new line and a tab."""
    return str(text or "").replace("\\n", "\n").replace("\\t", "\t")


def sources_text(items, template: str, separator: str = "\\n\\n", max_chars: int = 0) -> tuple[str, int, bool]:
    """Sources written through a template, one block per chunk.

    Args:
        items: Scored chunks, best first.
        template: Text with ``{rank}``, ``{score}``, ``{text}``, ``{file_name}``, ``{page}`` or
            any metadata key in braces. Any other ``{name}`` stays as written.
        separator: Text between blocks. ``\\n`` is a new line, ``\\t`` a tab.
        max_chars: The most characters written, 0 for :data:`MAX_TEXT_CHARS`.

    Returns:
        ``(text, blocks written, cut)``, cut True when the limit stopped it.

    Raises:
        ValueError: The template is too long or names too many fields.
    """
    template = str(template or "")
    if len(template) > MAX_TEMPLATE_CHARS:
        raise ValueError(f"template is longer than {MAX_TEMPLATE_CHARS} characters. Shorten it.")
    if len(TEMPLATE_FIELD.findall(template)) > MAX_TEMPLATE_FIELDS:
        raise ValueError(f"template names more than {MAX_TEMPLATE_FIELDS} fields. Name fewer.")
    gap = _unescaped(separator)[:200]
    limit = min(int(max_chars), MAX_TEXT_CHARS) if int(max_chars) > 0 else MAX_TEXT_CHARS
    known = set(SOURCE_FIELDS)
    for item in items:
        known.update(str(key) for key in (item.node.metadata or {}))
    parts: list[str] = []
    total, cut = 0, False
    for rank, item in enumerate(items, 1):
        values = {str(key): value for key, value in (item.node.metadata or {}).items()}
        values.update({
            "rank": rank, "score": _score(item.score), "text": _content(item.node),
            "file_name": file_label(item.node), "page": page_label(item.node),
        })

        def field(match, values=values):
            name = match.group(1)
            if name not in known:
                return match.group(0)
            value = values.get(name)
            return "" if value is None else str(value)[:MAX_FIELD_CHARS]

        block = TEMPLATE_FIELD.sub(field, template)
        extra = len(block) + (len(gap) if parts else 0)
        if total + extra > limit:
            cut = True
            if not parts:
                parts.append(block[:limit])
            break
        parts.append(block)
        total += extra
    return gap.join(parts), len(parts), cut


# Postprocessors


def _keyword_list(text: str, name: str) -> list[str]:
    """Keywords from a comma separated widget, lowercased and split into words."""
    found = []
    for part in str(text or "").split(","):
        words = WORD.findall(part.lower()[:MAX_KEYWORD_CHARS])
        if words:
            found.append(" ".join(words))
    if len(found) > MAX_LIST:
        raise ValueError(f"{name} holds more than {MAX_LIST} keywords. Keep the ones that matter.")
    return found


def _classes_for_postprocessors() -> dict:
    """The postprocessor classes written for this pack, defined on first use."""
    if "keywords" in _classes:
        return _classes
    runtime.core()
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.postprocessor import MetadataReplacementPostProcessor
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore

    class KeywordFilter(BaseNodePostprocessor):
        """Keeps chunks holding every required keyword and none of the excluded ones."""

        required: list = Field(default_factory=list)
        excluded: list = Field(default_factory=list)

        @classmethod
        def class_name(cls) -> str:
            return "WASKeywordFilter"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            kept = []
            for item in nodes:
                words = " " + " ".join(WORD.findall(_content(item.node).lower())) + " "
                if any(f" {word} " not in words for word in self.required):
                    continue
                if any(f" {word} " in words for word in self.excluded):
                    continue
                kept.append(item)
            return kept

    class WindowReplacement(MetadataReplacementPostProcessor):
        """Replaces each chunk's text with a metadata value, on a copy of the chunk."""

        @classmethod
        def class_name(cls) -> str:
            return "WASWindowReplacement"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            replaced = []
            for item in nodes:
                copy = _copied(item)
                value = copy.node.metadata.get(self.target_metadata_key)
                if value is not None:
                    copy.node.set_content(str(value))
                replaced.append(copy)
            return replaced

    class Neighbours(BaseNodePostprocessor):
        """Adds the chunks before or after each retrieved chunk, read from a docstore."""

        docstore: Any = None
        num_nodes: int = 1
        mode: str = "next"
        llm: Any = None

        @classmethod
        def class_name(cls) -> str:
            return "WASNeighbours"

        def _walk(self, node, forward: bool) -> list:
            found = []
            current = node
            for _ in range(self.num_nodes):
                link = current.next_node if forward else current.prev_node
                if link is None:
                    break
                neighbour = self.docstore.get_node(link.node_id, raise_error=False)
                if neighbour is None:
                    break
                found.append(neighbour)
                current = neighbour
            return found

        def _direction(self, item, query_bundle) -> str:
            if self.mode != "auto":
                return self.mode
            from llama_index.core.postprocessor.node import DEFAULT_INFER_PREV_NEXT_TMPL
            from llama_index.core.prompts import PromptTemplate

            answer = self.llm.predict(
                PromptTemplate(DEFAULT_INFER_PREV_NEXT_TMPL),
                context_str=_content(item.node), query_str=query_bundle.query_str if query_bundle else "",
            ).strip().lower()
            if "previous" in answer:
                return "previous"
            if "next" in answer:
                return "next"
            return "none"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            seen, ordered = set(), []
            for item in nodes:
                runtime.check_interrupt()
                direction = self._direction(item, query_bundle)
                before = self._walk(item.node, False)[::-1] if direction in ("previous", "both") else []
                after = self._walk(item.node, True) if direction in ("next", "both") else []
                for neighbour in before:
                    if neighbour.node_id not in seen:
                        seen.add(neighbour.node_id)
                        ordered.append(NodeWithScore(node=neighbour, score=item.score))
                if item.node.node_id not in seen:
                    seen.add(item.node.node_id)
                    ordered.append(item)
                for neighbour in after:
                    if neighbour.node_id not in seen:
                        seen.add(neighbour.node_id)
                        ordered.append(NodeWithScore(node=neighbour, score=item.score))
            return ordered

    class Recency(BaseNodePostprocessor):
        """Orders chunks newest first by a date in their metadata and keeps the newest."""

        date_key: str = "last_modified_date"
        top_n: int = 5

        @classmethod
        def class_name(cls) -> str:
            return "WASRecency"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            dated, undated = [], []
            for order, item in enumerate(nodes):
                when = _when(item.node.metadata.get(self.date_key))
                (dated if when is not None else undated).append((when, order, item))
            dated.sort(key=lambda entry: (-entry[0], entry[1]))
            return [entry[2] for entry in dated + undated][: self.top_n]

    class EmbeddingRerank(BaseNodePostprocessor):
        """Reorders chunks by how close their embedding is to the question's."""

        embed_model: Any = None
        top_n: int = 5

        @classmethod
        def class_name(cls) -> str:
            return "WASEmbeddingRerank"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            import numpy as np

            if not nodes or query_bundle is None:
                return list(nodes)[: self.top_n]
            query = np.asarray(self.embed_model.get_query_embedding(query_bundle.query_str), dtype=np.float64)
            texts = [_content(item.node) for item in nodes]
            vectors = np.asarray(self.embed_model.get_text_embedding_batch(texts), dtype=np.float64)
            norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(query) or 1.0)
            scores = vectors @ query / np.where(norms > 0, norms, 1.0)
            order = np.argsort(-scores, kind="stable")[: self.top_n]
            return [NodeWithScore(node=nodes[int(row)].node, score=float(scores[row])) for row in order]

    _classes.update({
        "keywords": KeywordFilter, "window": WindowReplacement, "neighbours": Neighbours,
        "recency": Recency, "embedding": EmbeddingRerank,
    })
    return _classes


def _when(value) -> float | None:
    """Seconds since 1970 for a date in metadata, or ``None`` when it does not read as one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip()[:64]
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        moment = None
        for shape in ("%Y/%m/%d", "%Y/%m/%d %H:%M:%S", "%d %B %Y", "%B %d, %Y"):
            try:
                moment = datetime.strptime(text, shape)
                break
            except ValueError:
                continue
        if moment is None:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _parse_choices(answer: str, count: int) -> tuple[list[int], list[float]]:
    """Chunk numbers and relevances from a ranking answer such as ``Doc: 2, Relevance: 8``."""
    choices, relevances = [], []
    for line in str(answer or "").splitlines()[: max(8, count * 4)]:
        numbers = NUMBER.findall(line)
        if len(numbers) < 2:
            continue
        choice = int(float(numbers[0]))
        if 1 <= choice <= count and choice not in choices:
            choices.append(choice)
            relevances.append(float(numbers[1]))
    return choices, relevances


def _docstore(index, documents):
    """The docstore neighbouring chunks are read from: the index's, or one built from documents."""
    if index is not None:
        return index.index.docstore
    if documents is not None and len(documents):
        from llama_index.core.storage.docstore import SimpleDocumentStore

        store = SimpleDocumentStore()
        store.add_documents(list(documents.items), allow_update=True)
        return store
    raise ValueError(
        "LLM Postprocessor: previous and next reads neighbouring chunks from a docstore. Connect the "
        "index the chunks were retrieved from to index, or the chunks themselves to documents."
    )


def postprocessor(choice: dict | str, llm=None, embedding=None, index=None, documents=None):
    """One postprocessor step from the kind combo's value.

    Args:
        choice: The combo's value: a dict holding ``kind`` and that kind's settings.
        llm: A :class:`~modules.llm.wires.LanguageModel`, for the steps that call one.
        embedding: An :class:`~modules.llm.wires.EmbeddingModel`, for the steps that call one.
        index: An :class:`~modules.llm.wires.Index`, whose docstore and models stand in.
        documents: A :class:`~modules.llm.wires.Documents`, read as a docstore.

    Returns:
        ``(postprocessor wire, facts)``.

    Raises:
        ValueError: A setting is unusable or a model the step calls is missing.
    """
    runtime.core()
    from llama_index.core.postprocessor import (
        LLMRerank,
        LongContextReorder,
        SentenceEmbeddingOptimizer,
        SimilarityPostprocessor,
    )

    made = _classes_for_postprocessors()
    choice = choice if isinstance(choice, dict) else {"kind": str(choice)}
    kind = str(choice.get("kind", ""))
    llm = llm or (index.llm if index is not None else None)
    embedding = embedding or (index.embedding if index is not None else None)
    depends: tuple = ()
    facts: dict = {"step": kind}
    if kind == "similarity cutoff":
        cutoff = float(choice.get("cutoff", 0.5))
        processor, label = SimilarityPostprocessor(similarity_cutoff=cutoff), f"score at least {cutoff:g}"
        facts["cutoff"] = f"{cutoff:g}"
    elif kind == "keywords":
        required = _keyword_list(choice.get("required", ""), "required")
        excluded = _keyword_list(choice.get("excluded", ""), "excluded")
        if not required and not excluded:
            raise ValueError("LLM Postprocessor: keywords needs at least one required or excluded keyword, such as `invoice, total`.")
        processor = made["keywords"](required=required, excluded=excluded)
        label = f"keywords {len(required)} required, {len(excluded)} excluded"
        facts.update({"required": len(required), "excluded": len(excluded)})
    elif kind == "long context reorder":
        processor, label = LongContextReorder(), "long context reorder"
    elif kind == "sentence window":
        key = str(choice.get("metadata_key", "window")).strip()
        if not METADATA_KEY.fullmatch(key):
            raise ValueError(f"metadata_key `{key[:64]}` is not a metadata key name, such as `window`.")
        processor, label = made["window"](target_metadata_key=key), f"sentence window from {key}"
        facts["metadata key"] = key
    elif kind == "previous and next":
        direction = str(choice.get("direction", "next"))
        if direction not in ("next", "previous", "both", "auto"):
            raise ValueError(f"direction `{direction[:32]}` is not one of next, previous, both or auto.")
        count = int(choice.get("count", 1))
        model = None
        if direction == "auto":
            model = _need_step(llm, "previous and next set to auto", "a language model", "llm")
            depends = (model,)
        processor = made["neighbours"](
            docstore=_docstore(index, documents), num_nodes=count, mode=direction,
            llm=model.llm if model is not None else None,
        )
        label = f"{count} {direction} neighbour(s)"
        facts.update({"direction": direction, "count": count})
    elif kind == "recency":
        key = str(choice.get("date_key", "last_modified_date")).strip()
        if not METADATA_KEY.fullmatch(key):
            raise ValueError(f"date_key `{key[:64]}` is not a metadata key name, such as `last_modified_date`.")
        top_n = int(choice.get("top_n", 5))
        processor, label = made["recency"](date_key=key, top_n=top_n), f"newest {top_n} by {key}"
        facts.update({"date key": key, "top_n": top_n})
    elif kind == "rerank with llm":
        model = _need_step(llm, "rerank with llm", "a language model", "llm")
        top_n, batch = int(choice.get("top_n", 5)), int(choice.get("batch_size", 10))
        processor = LLMRerank(llm=model.llm, top_n=top_n, choice_batch_size=batch, parse_choice_select_answer_fn=_parse_choices)
        label, depends = f"llm rerank top {top_n}", (model,)
        facts.update({"model": model.label[:64], "top_n": top_n})
    elif kind == "rerank with embedding":
        model = _need_step(embedding, "rerank with embedding", "an embedding model", "embedding")
        top_n = int(choice.get("top_n", 5))
        processor = made["embedding"](embed_model=model.model, top_n=top_n)
        label, depends = f"embedding rerank top {top_n}", (model,)
        facts.update({"model": model.label[:64], "top_n": top_n})
    elif kind == "sentence optimizer":
        model = _need_step(embedding, "sentence optimizer", "an embedding model", "embedding")
        share = float(choice.get("percentile", 0.5))
        threshold = float(choice.get("threshold", 0.0))
        if share <= 0 and threshold <= 0:
            raise ValueError("LLM Postprocessor: sentence optimizer needs percentile or threshold above 0, or it keeps every sentence.")
        deps.require("nltk", feature=FEATURE)
        from nltk.tokenize import PunktSentenceTokenizer

        processor = SentenceEmbeddingOptimizer(
            embed_model=model.model,
            percentile_cutoff=share if share > 0 else None,
            threshold_cutoff=threshold if threshold > 0 else None,
            tokenizer_fn=PunktSentenceTokenizer().tokenize,
            context_before=int(choice.get("context_before", 1)),
            context_after=int(choice.get("context_after", 1)),
        )
        label, depends = "sentence optimizer", (model,)
        facts.update({"model": model.label[:64], "percentile": f"{share:g}", "threshold": f"{threshold:g}"})
    else:
        raise ValueError(f"`{kind[:64]}` is not a postprocessor kind. Pick one of: {', '.join(POSTPROCESSOR_KINDS)}.")
    return wires.Postprocessor(processor=processor, label=label, depends=depends), facts


def _need_step(model, step: str, what: str, socket: str):
    """A model a postprocessor step calls, or a ValueError naming what to connect."""
    if model is None:
        hint = "an LLM model node" if socket == "llm" else "LLM Embedding Model"
        raise ValueError(
            f"LLM Postprocessor: {step} calls {what}, and none is connected. Connect {hint} to the "
            f"{socket} input, or an index built with one to index."
        )
    return model
