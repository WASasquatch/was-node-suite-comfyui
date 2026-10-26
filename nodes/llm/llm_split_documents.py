"""Split documents into chunks for an index."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_EMBEDDING
from ...modules.interface import run_result
from ...modules.llm import ingest, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.ingest")

#: Chunks a run result shows.
SAMPLE_CHUNKS = 3

#: Characters of each sampled chunk a run result carries.
SAMPLE_CHARS = 2000


class LLMSplitDocuments(io.ComfyNode):
    """Documents split into chunks by sentences, tokens, meaning, headings, tags or levels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        splitters = [
            io.DynamicCombo.Option(
                key="sentence",
                inputs=[
                    io.Int.Input(
                        "chunk_size", default=1024, min=32, max=32768,
                        tooltip="Tokens per chunk, about three quarters of a word each. `1024` suits most indexes; `256` gives sharper matches.",
                    ),
                    io.Int.Input(
                        "chunk_overlap", default=200, min=0, max=8192,
                        tooltip="Tokens shared by neighbouring chunks, so a sentence cut at an edge is in both. `200` with a 1024 chunk; `0` = none.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="token",
                inputs=[
                    io.Int.Input(
                        "token_size", default=1024, min=32, max=32768,
                        tooltip="Tokens per chunk, cut on word boundaries whatever the sentences. `512` for short notes; `1024` for prose.",
                    ),
                    io.Int.Input(
                        "token_overlap", default=20, min=0, max=8192,
                        tooltip="Tokens shared by neighbouring chunks. `20` keeps a phrase cut at the edge readable in both; `0` = none.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="semantic",
                inputs=[
                    io.Int.Input(
                        "breakpoint_percentile", default=95, min=50, max=99,
                        tooltip="How large a change of meaning between sentences starts a new chunk. `95` = only sharp changes, long chunks; `80` = more, shorter chunks.",
                    ),
                    io.Int.Input(
                        "buffer_size", default=1, min=1, max=10,
                        tooltip="Sentences compared together on each side of a possible cut. `1` = each sentence alone; `3` = smoother, fewer cuts.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(key="markdown", inputs=[]),
            io.DynamicCombo.Option(
                key="html",
                inputs=[
                    io.String.Input(
                        "tags", default="p, h1, h2, h3, h4, h5, h6, li, pre, blockquote, td, th, section",
                        tooltip="The elements that become chunks, separated by commas, such as `p, h1, h2, li`. Reads documents that hold markup, from keep_html on a loader.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(key="json", inputs=[]),
            io.DynamicCombo.Option(
                key="hierarchical",
                inputs=[
                    io.String.Input(
                        "chunk_sizes", default="2048,512,128",
                        tooltip="Tokens per chunk at each level, largest first, such as `2048,512,128`. Every level is handed on; the smallest are marked as leaves for auto-merging retrieval.",
                    ),
                    io.Int.Input(
                        "level_overlap", default=20, min=0, max=1024,
                        tooltip="Tokens shared by neighbouring chunks within a level. `20` is usual; it must stay below the smallest size.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="sentence window",
                inputs=[
                    io.Int.Input(
                        "window_size", default=3, min=1, max=10,
                        tooltip="Sentences kept on each side of every one-sentence chunk, for a window postprocessor to show the model. `3` = seven sentences of context.",
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMSplitDocuments",
            display_name="LLM Split Documents",
            search_aliases=[
                "WASLLMSplitDocuments",
                "LLM Split Documents",
                "SentenceSplitter",
                "TokenTextSplitter",
                "SemanticSplitterNodeParser",
                "MarkdownNodeParser",
                "HTMLNodeParser",
                "JSONNodeParser",
                "HierarchicalNodeParser",
                "SentenceWindowNodeParser",
                "chunk",
                "node parser",
                "text splitter",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Split documents into chunks an index can search, by sentences, tokens, changes "
                "of meaning, Markdown headings, HTML elements, JSON entries, nested levels or "
                "sentence windows. The panel shows how many chunks were made, their lengths and "
                "a few of them."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="The documents to split, from an LLM document loader such as LLM Load Documents.",
                ),
                io.DynamicCombo.Input(
                    "splitter",
                    options=splitters,
                    display_name="Splitter",
                    tooltip=(
                        "`sentence` = chunks of whole sentences; `token` = fixed token counts; "
                        "`semantic` = cut where the meaning changes, needs embedding; `markdown` = "
                        "by headings; `html` = by elements; `json` = by entries; `hierarchical` = "
                        "nested sizes; `sentence window` = one sentence with its neighbours."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "embedding",
                    optional=True,
                    tooltip=(
                        "The embedding model the semantic splitter compares sentences with, from "
                        "LLM Embedding Model. Read by semantic alone."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "The chunks, each carrying its document's metadata, for an LLM index or "
                        "LLM Extract Metadata."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, splitter, embedding=None) -> io.NodeOutput:
        """Split every document with the chosen splitter.

        Args:
            documents: The documents.
            splitter: The splitter choice and its settings.
            embedding: The embedding model for the semantic splitter.

        Returns:
            The chunks.

        Raises:
            ValueError: A wire carries something else, a setting is out of range, or the split
                makes nothing.
            RuntimeError: A hosted embedding model with features.network off, or the split failed.
        """
        source = wires.expect(
            documents, wires.Documents, "LLM Split Documents", "documents", "LLM Load Documents"
        )
        choice = splitter if isinstance(splitter, dict) else {"splitter": str(splitter)}
        model = None
        if choice.get("splitter") == "semantic" and embedding is not None:
            model = wires.expect(
                embedding, wires.EmbeddingModel, "LLM Split Documents", "embedding", "LLM Embedding Model"
            )
            wires.ready(model)
        if not source.items:
            raise ValueError("LLM Split Documents was handed no documents. Connect a loader that read something.")
        with runtime.guarded():
            made = ingest.split(source, choice, model)
        if not made.items:
            raise ValueError(
                f"LLM Split Documents made no chunks with {made.label}. The json splitter reads "
                f"documents that are JSON, and html reads documents that hold markup; pick "
                f"sentence for plain text."
            )
        result = wires.Documents(tuple(made.items), f"{len(made.items)} chunks of {source.origin or 'documents'}")
        figures = ingest.chunk_figures(made.items)
        logger.info(
            "LLM Split Documents made %d chunk(s) from %d document(s) with %s",
            len(made.items), len(source.items), made.label,
        )
        facts = {"splitter": made.label, "documents in": len(source.items)}
        counts = dict(figures)
        if made.empty:
            counts["documents left out"] = made.empty
        if made.levels:
            counts["leaves"] = made.leaves
            facts["levels"] = ", ".join(f"{level}: {count}" for level, count in sorted(made.levels.items()))
        samples = [item for item in made.items if item.metadata.get(ingest.LEAF_KEY, True)][:SAMPLE_CHUNKS]
        bodies = [
            run_result.body(f"chunk {index + 1}", (item.text or "")[:SAMPLE_CHARS])
            for index, item in enumerate(samples)
        ]
        if made.empty:
            logger.info(
                "LLM Split Documents made no chunk from %d document(s) with %s", made.empty, made.label
            )
        run_result.publish(
            status=run_result.WARNING if made.empty else run_result.OK,
            summary=(
                f"split {len(source.items)} document(s) into {len(made.items)} chunk(s)"
                + (f", {made.empty} gave none" if made.empty else "")
            ),
            counts=counts,
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(result)
