"""Embed documents into an in-memory vector index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_EMBEDDING, LLM_INDEX, LLM_VECTOR_STORE
from ...modules.llm import indexes, runtime, vector_stores, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Vector Index"


class LLMVectorIndex(io.ComfyNode):
    """Embed documents into a vector index searched by meaning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMVectorIndex",
            display_name="LLM Vector Index",
            search_aliases=[
                "WASLLMVectorIndex",
                "LLM Vector Index",
                "VectorStoreIndex",
                "vector index",
                "vector store",
                "RAG",
                "embeddings",
                "semantic search",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Embed documents into a vector index, for finding the passages closest in meaning "
                "to a question. Whole documents are cut into chunks first; chunks from LLM Split "
                "Documents are indexed as they are. The panel shows the chunk and vector counts "
                "and a few chunks."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to index, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. From a hierarchical split only the smallest chunks are embedded "
                        "and every level is kept, for LLM Auto-Merging Retriever."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "embedding",
                    tooltip=(
                        "The model that turns each chunk into a vector, such as from LLM Embedding "
                        "Model. Questions are embedded with the same model later, so save or load "
                        "the index with it."
                    ),
                ),
                io.Int.Input(
                    "batch_size", default=64, min=1, max=4096,
                    tooltip=(
                        "Chunks embedded per step. 64 suits most models; 16 for a hosted model that "
                        "refuses large requests. Cancel and the progress bar are checked between steps."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = finer matches with less context each. Chunks from LLM "
                        "Split Documents keep their own size."
                    ),
                ),
                io.Int.Input(
                    "chunk_overlap", default=20, min=0, max=2048, optional=True, advanced=True,
                    tooltip=(
                        "Tokens two neighbouring chunks share, so a sentence cut at a border is "
                        "still found whole. 20 = a little; 0 = none. Smaller than chunk_size."
                    ),
                ),
                LLM_VECTOR_STORE.Input(
                    "vector_store", optional=True,
                    tooltip=(
                        "A collection from LLM Vector Store to keep the vectors in, in place of memory. A "
                        "document it already holds has its chunks replaced. Unconnected = memory, saved with LLM Save Index."
                    ),
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The vector index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, embedding, batch_size, chunk_size=1024, chunk_overlap=20,
                vector_store=None) -> io.NodeOutput:
        """Split, embed and index the documents.

        Args:
            documents: The documents or chunks.
            embedding: The embedding model.
            batch_size: Chunks embedded per step.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.
            vector_store: The collection the vectors are kept in, or ``None`` for memory.

        Returns:
            The vector index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        model = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        store = None
        if vector_store is not None:
            store = wires.expect(vector_store, vector_stores.VectorStore, NODE, "vector_store", "LLM Vector Store")
        wires.ready(model)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = indexes.build_vector(prepared, model, batch_size, store=store)
        wire = wires.Index(
            index=index, kind="vector", embedding=model, size=indexes.size_of(index, "vector", store), store=store,
        )
        seconds = time.perf_counter() - started
        logger.info(
            "%s embedded %d chunk(s) from %d item(s) with %s in %.2fs",
            NODE, wire.size, prepared.handed, model.label, seconds,
        )
        indexes.report(
            wire, seconds, f"embedded {wire.size} chunks with {model.label}", prepared,
            facts={"batch size": str(int(batch_size))},
        )
        return io.NodeOutput(wire)
