"""Splitting documents into chunks, and writing titles, summaries, questions and keywords onto them.

A hierarchical split marks each chunk with :data:`LEVEL_KEY` (0 is the largest) and
:data:`LEAF_KEY`; a sentence window split keeps the window under :data:`WINDOW_KEY`.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .. import deps, log
from . import FEATURE, runtime, wires

__all__ = [
    "DEFAULT_HTML_TAGS",
    "LEAF_KEY",
    "LEVEL_KEY",
    "MAX_CHUNKS",
    "ORIGINAL_KEY",
    "SPLITTERS",
    "SUMMARY_CHOICES",
    "Split",
    "WINDOW_KEY",
    "chunk_figures",
    "extract",
    "parse_sizes",
    "parse_tags",
    "split",
]

logger = log.get_logger("llm.ingest")

#: The splitters offered, default first.
SPLITTERS = (
    "sentence", "token", "semantic", "markdown", "html", "json", "hierarchical", "sentence window",
)

#: Metadata key holding a hierarchical chunk's level, 0 for the largest chunks.
LEVEL_KEY = "hierarchy_level"

#: Metadata key that is True on a hierarchical chunk with no smaller chunk below it.
LEAF_KEY = "hierarchy_leaf"

#: Metadata keys a sentence window split writes: the sentences around a chunk, and the chunk.
WINDOW_KEY = "window"
ORIGINAL_KEY = "original_text"

#: Most chunks one split hands on.
MAX_CHUNKS = 100000

#: Characters of one document a json split reads, and of the lines it writes from it, at most.
MAX_JSON_SPLIT_INPUT = 16 * 1024 * 1024
MAX_JSON_SPLIT_OUTPUT = 64 * 1024 * 1024

#: Most levels a hierarchical split builds.
MAX_LEVELS = 5

#: Smallest and largest chunk size, in tokens.
MIN_CHUNK = 32
MAX_CHUNK = 32768

#: The HTML elements an html split reads when none are typed.
DEFAULT_HTML_TAGS = "p, h1, h2, h3, h4, h5, h6, li, pre, blockquote, td, th, section"

#: Most tags an html split reads.
MAX_TAGS = 32

#: One typed tag name.
_TAG = re.compile(r"^[a-z][a-z0-9]{0,15}$")

#: The summary choices, default first.
SUMMARY_CHOICES = ("none", "self", "self and neighbours")

#: Chunks given to one questions or keywords pass, between two interrupt checks.
EXTRACT_BATCH = 16

#: The label a keyword answer follows, split on to drop a prompt the model repeated before it.
_KEYWORDS_LABEL = re.compile(r"(?i)\bkeywords\s*:\s*")

#: Parallel requests to a hosted model while extracting.
HOSTED_WORKERS = 4

#: Metadata keys each extractor writes.
EXTRACTED_KEYS = (
    "document_title", "section_summary", "prev_section_summary", "next_section_summary",
    "questions_this_excerpt_can_answer", "excerpt_keywords", "entities",
)

#: What a model is asked to name in a chunk.
ENTITY_PROMPT = (
    "List the people, places, organisations, products and works named in the text below, as one "
    "comma separated line and nothing else. Write `none` if it names none.\n\nText:\n{text}\n\nNames:"
)

#: Characters of a chunk a model reads to name its entities.
ENTITY_TEXT_CHARS = 12_000

#: Names kept per chunk.
MAX_ENTITIES = 40

#: Characters of an extracted value drawn under the running node.
STREAM_VALUE_CHARS = 200


@dataclass
class Split:
    """The chunks one split made.

    Attributes:
        items: llama-index nodes, in document order.
        label: What the run result calls the split, such as ``"sentence 1024 / 200"``.
        levels: Hierarchy level to chunks, for a hierarchical split.
        leaves: Chunks with no smaller chunk below them, for a hierarchical split.
        empty: Documents the splitter made no chunk from.
    """

    items: list = field(default_factory=list)
    label: str = ""
    levels: Counter = field(default_factory=Counter)
    leaves: int = 0
    empty: int = 0


def parse_sizes(text: str) -> list[int]:
    """The chunk sizes of a hierarchical split, largest first.

    Args:
        text: Sizes separated by commas, such as ``2048,512,128``.

    Returns:
        The sizes as typed.

    Raises:
        ValueError: A size is not a whole number from :data:`MIN_CHUNK` to :data:`MAX_CHUNK`,
            there are none or more than :data:`MAX_LEVELS`, or they do not get smaller.
    """
    sizes: list[int] = []
    for token in re.split(r"[\s,;]+", str(text or "")):
        if not token:
            continue
        if not token.isdigit() or not MIN_CHUNK <= int(token) <= MAX_CHUNK:
            raise ValueError(
                f"`{token[:20]}` is not a chunk size. Each size is a whole number of tokens from "
                f"{MIN_CHUNK} to {MAX_CHUNK}, such as `2048,512,128`."
            )
        sizes.append(int(token))
    if not sizes or len(sizes) > MAX_LEVELS:
        raise ValueError(
            f"chunk_sizes takes 1 to {MAX_LEVELS} sizes, largest first, such as `2048,512,128`."
        )
    if any(later >= earlier for earlier, later in zip(sizes, sizes[1:])):
        raise ValueError(
            "chunk_sizes must get smaller from left to right, such as `2048,512,128`, since each "
            "level is split out of the one before it."
        )
    return sizes


def parse_tags(text: str) -> list[str]:
    """The HTML element names an html split reads.

    Args:
        text: Names separated by commas or spaces, such as ``p, h1, li``.

    Returns:
        The lowercased names, each once.

    Raises:
        ValueError: A name is not an element name, or there are none or more than :data:`MAX_TAGS`.
    """
    tags: list[str] = []
    for token in re.split(r"[\s,;<>/]+", str(text or "")):
        word = token.strip().lower()
        if not word:
            continue
        if not _TAG.match(word):
            raise ValueError(f"`{token[:20]}` is not an HTML element name, such as p, h1 or li.")
        if word not in tags:
            tags.append(word)
    if not tags or len(tags) > MAX_TAGS:
        raise ValueError(f"tags takes 1 to {MAX_TAGS} element names, such as `p, h1, h2, li`.")
    return tags


def split(documents: wires.Documents, choice: dict, embedding: wires.EmbeddingModel | None) -> Split:
    """Split every document with the chosen splitter.

    Args:
        documents: What arrived on the documents input.
        choice: The splitter combo's value: its own key and that option's settings.
        embedding: The embedding model, used by the semantic splitter alone.

    Returns:
        The chunks.

    Raises:
        ValueError: A setting is out of range, the semantic splitter has no embedding model, or the
            split makes more than :data:`MAX_CHUNKS` chunks.
        RuntimeError: The splitter failed.
    """
    mode = str(choice.get("splitter", SPLITTERS[0]))
    parser, label = _parser(mode, choice, embedding)
    made = Split(label=label)
    for item in documents.items:
        runtime.check_interrupt()
        if mode == "json":
            _check_json_split(item)
        try:
            nodes = parser.get_nodes_from_documents([item], show_progress=False)
        except Exception as error:
            if _interrupted(error):
                raise
            raise RuntimeError(f"LLM Split Documents failed: {runtime.describe_error(error)}") from None
        if not nodes:
            made.empty += 1
        if mode == "hierarchical":
            _mark_levels(nodes, made)
        made.items.extend(nodes)
        if len(made.items) > MAX_CHUNKS:
            raise ValueError(
                f"LLM Split Documents made more than {MAX_CHUNKS} chunks. Raise the chunk size, "
                f"or split fewer documents at once."
            )
    return made


def _check_json_split(item) -> None:
    """Refuse a JSON document whose split would read or write past its ceilings.

    Raises:
        ValueError: The document is past :data:`MAX_JSON_SPLIT_INPUT`, or its key paths would
            write past :data:`MAX_JSON_SPLIT_OUTPUT`.
    """
    text = getattr(item, "text", "") or ""
    if len(text) > MAX_JSON_SPLIT_INPUT:
        raise ValueError(
            f"LLM Split Documents reads a JSON document of at most {MAX_JSON_SPLIT_INPUT // (1024 * 1024)} "
            f"million characters in json mode. Split it with another splitter."
        )
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return
    written = 0
    pending = [(data, 0)]
    while pending:
        value, carried = pending.pop()
        if isinstance(value, dict):
            pending.extend((inner, carried + len(str(key)) + 1) for key, inner in value.items())
        elif isinstance(value, list):
            pending.extend((inner, carried) for inner in value)
        else:
            written += carried + len(str(value)) + 1
            if written > MAX_JSON_SPLIT_OUTPUT:
                raise ValueError(
                    "LLM Split Documents would write more than "
                    f"{MAX_JSON_SPLIT_OUTPUT // (1024 * 1024)} million characters from one JSON document, "
                    "its key paths repeated on every value. Split it with another splitter."
                )


def chunk_figures(items: list) -> dict[str, float]:
    """The count, total characters and the mean, shortest and longest chunk length in characters."""
    lengths = [len(getattr(item, "text", "") or "") for item in items]
    if not lengths:
        return {"chunks": 0}
    return {
        "chunks": len(lengths),
        "characters": sum(lengths),
        "mean chars": round(sum(lengths) / len(lengths), 1),
        "min chars": min(lengths),
        "max chars": max(lengths),
    }


def extract(items: list, model: wires.LanguageModel, title: int, summaries: str, questions: int,
            keywords: int, stream: runtime.TextStream | None = None, entities: bool = False) -> tuple[list, Counter]:
    """Copies of some chunks with titles, summaries, questions and keywords written by a model.

    Args:
        items: llama-index nodes, left unchanged.
        model: The language model that writes the metadata.
        title: Chunks a document's title is written from, 0 for no title.
        summaries: One of :data:`SUMMARY_CHOICES`.
        questions: Questions written per chunk, 0 for none.
        keywords: Keywords written per chunk, 0 for none.
        stream: Where progress is drawn, or ``None``.
        entities: Write the names of people, places, organisations and products each chunk holds.

    Returns:
        ``(the copies, metadata key to chunks it was written on)``.

    Raises:
        ValueError: Every extractor is off.
        RuntimeError: The model failed.
    """
    extractors = _extractors(model, title, summaries, questions, keywords, entities)
    if not extractors:
        raise ValueError(
            "LLM Extract Metadata has nothing to write: title and questions and keywords are 0, "
            "summaries is none and entities is off. Turn on at least one of them."
        )
    copies = [copy.deepcopy(item) for item in items]
    views = [_text_view(item) for item in copies]
    groups = _grouped(views)
    done = 0
    for group in groups:
        for name, extractor, whole in extractors:
            batches = [group] if whole else [
                group[at:at + EXTRACT_BATCH] for at in range(0, len(group), EXTRACT_BATCH)
            ]
            for batch in batches:
                runtime.check_interrupt()
                if stream is not None:
                    stream.note(f"{model.label}: {name}, chunk {done + 1} of {len(copies)}")
                try:
                    extractor.process_nodes(batch)
                except Exception as error:
                    if _interrupted(error):
                        raise
                    raise RuntimeError(
                        f"LLM Extract Metadata failed while writing {name}: "
                        f"{runtime.describe_error(error)}"
                    ) from None
        done += len(group)
        if stream is not None:
            stream.write(_progress_lines(group[0]))
    for item, view in zip(copies, views):
        if view is not item:
            item.metadata.update({key: value for key, value in view.metadata.items() if key in EXTRACTED_KEYS})
    written = Counter()
    for item in copies:
        raw = item.metadata.get("excerpt_keywords")
        if isinstance(raw, str):
            item.metadata["excerpt_keywords"] = _keyword_line(raw)
        for key in EXTRACTED_KEYS:
            if item.metadata.get(key):
                written[key] += 1
    return copies, written


def _keyword_line(text: str) -> str:
    """The keywords a model wrote, without any of the prompt it repeated before them.

    Args:
        text: What the keyword extractor recorded.

    Returns:
        The first line after the last ``Keywords:``, or the first line when there is none.
    """
    tail = _KEYWORDS_LABEL.split(text)[-1]
    for line in tail.splitlines():
        line = line.strip().rstrip(".").strip()
        if line:
            return line
    return ""


def _parser(mode: str, choice: dict, embedding):
    """The llama-index node parser one splitter choice builds, and its label.

    Raises:
        ValueError: A setting is out of range, or the semantic splitter has no embedding model.
    """
    runtime.core()
    from llama_index.core import node_parser

    if mode == "sentence":
        size, overlap = _sized(choice, "chunk_size", "chunk_overlap", 1024, 200)
        return node_parser.SentenceSplitter(chunk_size=size, chunk_overlap=overlap), f"sentence {size} / {overlap}"
    if mode == "token":
        size, overlap = _sized(choice, "token_size", "token_overlap", 1024, 20)
        return node_parser.TokenTextSplitter(chunk_size=size, chunk_overlap=overlap), f"token {size} / {overlap}"
    if mode == "semantic":
        if embedding is None:
            raise ValueError(
                "The semantic splitter compares sentences by meaning and needs an embedding model. "
                "Connect LLM Embedding Model to the embedding input, or pick another splitter."
            )
        percentile = int(choice.get("breakpoint_percentile", 95))
        buffer = int(choice.get("buffer_size", 1))
        parser = node_parser.SemanticSplitterNodeParser(
            buffer_size=buffer,
            breakpoint_percentile_threshold=percentile,
            embed_model=embedding.model,
        )
        return parser, f"semantic {percentile}th percentile on {embedding.label}"
    if mode == "markdown":
        return node_parser.MarkdownNodeParser(), "markdown headings"
    if mode == "html":
        deps.require("bs4", feature=FEATURE)
        tags = parse_tags(choice.get("tags", DEFAULT_HTML_TAGS))
        return node_parser.HTMLNodeParser(tags=tags), f"html {', '.join(tags[:6])}"
    if mode == "json":
        return node_parser.JSONNodeParser(), "json"
    if mode == "hierarchical":
        sizes = parse_sizes(choice.get("chunk_sizes", "2048,512,128"))
        overlap = int(choice.get("level_overlap", 20))
        if overlap >= sizes[-1]:
            raise ValueError(
                f"level_overlap ({overlap}) must be smaller than the smallest chunk size "
                f"({sizes[-1]}). Lower it, or raise the last size."
            )
        parser = node_parser.HierarchicalNodeParser.from_defaults(chunk_sizes=sizes, chunk_overlap=overlap)
        return parser, "hierarchical " + "/".join(str(size) for size in sizes)
    if mode == "sentence window":
        window = int(choice.get("window_size", 3))
        parser = node_parser.SentenceWindowNodeParser.from_defaults(
            window_size=window, window_metadata_key=WINDOW_KEY, original_text_metadata_key=ORIGINAL_KEY,
        )
        return parser, f"sentence window {window}"
    raise ValueError(f"`{mode}` is not a splitter. Pick one of: {', '.join(SPLITTERS)}.")


def _sized(choice: dict, size_key: str, overlap_key: str, size_default: int, overlap_default: int) -> tuple[int, int]:
    """A chunk size and an overlap smaller than it.

    Raises:
        ValueError: The overlap is not smaller than the size.
    """
    size = int(choice.get(size_key, size_default))
    overlap = int(choice.get(overlap_key, overlap_default))
    if overlap >= size:
        raise ValueError(
            f"{overlap_key} ({overlap}) must be smaller than {size_key} ({size}). Lower the "
            f"overlap, such as to a fifth of the size."
        )
    return size, overlap


def _mark_levels(nodes: list, made: Split) -> None:
    """Write each hierarchical chunk's level and whether it is a leaf into its metadata."""
    from llama_index.core.schema import NodeRelationship

    by_id = {node.node_id: node for node in nodes}
    levels: dict[str, int] = {}

    def level(node) -> int:
        chain = []
        current = node
        while current is not None and current.node_id not in levels and len(chain) <= MAX_LEVELS:
            chain.append(current)
            parent = current.relationships.get(NodeRelationship.PARENT)
            current = by_id.get(parent.node_id) if parent is not None else None
        base = levels.get(current.node_id, -1) if current is not None else -1
        for depth, item in enumerate(reversed(chain), 1):
            levels[item.node_id] = base + depth
        return levels[node.node_id]

    for node in nodes:
        depth = level(node)
        leaf = NodeRelationship.CHILD not in node.relationships
        node.metadata[LEVEL_KEY] = depth
        node.metadata[LEAF_KEY] = leaf
        for keys in (node.excluded_embed_metadata_keys, node.excluded_llm_metadata_keys):
            for key in (LEVEL_KEY, LEAF_KEY):
                if key not in keys:
                    keys.append(key)
        made.levels[depth] += 1
        made.leaves += int(leaf)


def _entity_line(text: str) -> str:
    """The names a model listed, one comma separated line, ``""`` for none."""
    from .messages import split_thinking

    answer = split_thinking(str(text or ""))[0].strip().splitlines()
    line = next((row for row in answer if row.strip()), "")
    names = []
    for part in line.split(","):
        name = part.strip().strip(".`*\"'").strip()
        if name and name.lower() not in ("none", "n/a") and name.lower() not in {kept.lower() for kept in names}:
            names.append(name[:80])
    return ", ".join(names[:MAX_ENTITIES])


def _entity_class():
    """The extractor that asks a model for the names a chunk holds, defined on first use."""
    if "entity" in _ENTITY_CLASS:
        return _ENTITY_CLASS["entity"]
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.extractors import BaseExtractor
    from llama_index.core.schema import MetadataMode

    class EntityNamesExtractor(BaseExtractor):
        """The people, places, organisations and products a chunk names, written by a model."""

        llm: Any = Field(default=None, description="The llama-index LLM asked.")

        @classmethod
        def class_name(cls) -> str:
            return "WAS_EntityNamesExtractor"

        async def aextract(self, nodes) -> list[dict]:
            found = []
            for node in nodes:
                text = node.get_content(metadata_mode=MetadataMode.NONE)[:ENTITY_TEXT_CHARS]
                answer = await self.llm.acomplete(ENTITY_PROMPT.format(text=text))
                names = _entity_line(getattr(answer, "text", answer))
                found.append({"entities": names} if names else {})
            return found

    _ENTITY_CLASS["entity"] = EntityNamesExtractor
    return EntityNamesExtractor


_ENTITY_CLASS: dict = {}


def _extractors(model: wires.LanguageModel, title: int, summaries: str, questions: int,
                keywords: int, entities: bool = False) -> list[tuple[str, Any, bool]]:
    """``(name, extractor, whole document at once)`` for every extractor turned on."""
    runtime.core()
    from llama_index.core import extractors

    workers = 1 if model.local else HOSTED_WORKERS
    shared = {"llm": model.llm, "num_workers": workers, "show_progress": False}
    chosen: list[tuple[str, Any, bool]] = []
    if title > 0:
        chosen.append(("titles", extractors.TitleExtractor(nodes=int(title), **shared), True))
    if summaries == "self":
        chosen.append(("summaries", extractors.SummaryExtractor(summaries=["self"], **shared), True))
    elif summaries == "self and neighbours":
        chosen.append((
            "summaries",
            extractors.SummaryExtractor(summaries=["self", "prev", "next"], **shared),
            True,
        ))
    if questions > 0:
        chosen.append((
            "questions",
            extractors.QuestionsAnsweredExtractor(questions=int(questions), **shared),
            False,
        ))
    if keywords > 0:
        chosen.append(("keywords", extractors.KeywordExtractor(keywords=int(keywords), **shared), False))
    if entities:
        chosen.append(("entities", _entity_class()(llm=model.llm, show_progress=False), False))
    return chosen


def _text_view(item):
    """A chunk itself, or a ``TextNode`` holding a whole document's text and metadata."""
    from llama_index.core.schema import TextNode

    if isinstance(item, TextNode):
        return item
    return TextNode(
        text=getattr(item, "text", "") or "",
        id_=item.node_id,
        metadata=dict(item.metadata),
        excluded_embed_metadata_keys=list(item.excluded_embed_metadata_keys),
        excluded_llm_metadata_keys=list(item.excluded_llm_metadata_keys),
    )


def _grouped(items: list) -> list[list]:
    """Chunks grouped by the document they came from, in the order they arrived."""
    groups: dict[str, list] = {}
    for item in items:
        key = getattr(item, "ref_doc_id", None) or item.node_id
        groups.setdefault(key, []).append(item)
    return list(groups.values())


def _progress_lines(item) -> str:
    """The values written on one chunk, one line each, for the text drawn under the node."""
    lines = []
    for key in EXTRACTED_KEYS:
        value = item.metadata.get(key)
        if value:
            lines.append(f"{key}: {' '.join(str(value).split())[:STREAM_VALUE_CHARS]}")
    return "\n".join(lines) + "\n\n" if lines else ""


def _interrupted(error: BaseException) -> bool:
    """Whether an error is ComfyUI stopping the run."""
    return type(error).__name__ == "InterruptProcessingException"
