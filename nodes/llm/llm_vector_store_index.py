"""Open the vector index a vector store collection already holds."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_EMBEDDING, LLM_INDEX, LLM_VECTOR_STORE
from ...modules.llm import indexes, runtime, vector_stores, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Vector Store Index"


class LLMVectorStoreIndex(io.ComfyNode):
    """A vector index over a collection's existing vectors."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMVectorStoreIndex",
            display_name="LLM Vector Store Index",
            search_aliases=[
                "WASLLMVectorStoreIndex",
                "LLM Vector Store Index",
                "from_vector_store",
                "open vector store",
                "Qdrant index",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Open the index a vector store collection already holds, without embedding anything "
                "again, for a retriever, a query engine or LLM Index Insert. Use the embedding model "
                "the collection was filled with. The panel shows its vector count."
            ),
            inputs=[
                LLM_VECTOR_STORE.Input(
                    "vector_store",
                    tooltip="The collection, from LLM Vector Store.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding",
                    tooltip="The embedding model the collection was filled with, from LLM Embedding Model. Questions are embedded with it.",
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The vector index over the collection, for LLM Index Retriever, a query engine or LLM Index Insert.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """Always run, so the count follows the collection."""
        return float("NaN")

    @classmethod
    def execute(cls, vector_store, embedding) -> io.NodeOutput:
        """Open the index.

        Args:
            vector_store: The collection.
            embedding: The embedding model.

        Returns:
            The vector index.
        """
        store = wires.expect(vector_store, vector_stores.VectorStore, NODE, "vector_store", "LLM Vector Store")
        model = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        wires.ready(model)
        held = vector_stores.count(store)
        if not held:
            raise ValueError(
                f"{NODE}: the collection {store.label} holds no vectors yet. Fill it with LLM Vector Index, "
                f"its vector_store input connected."
            )
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            index = indexes.open_store_index(store, model)
        wire = wires.Index(index=index, kind="vector", embedding=model, size=held, store=store)
        seconds = time.perf_counter() - started
        logger.info("%s opened %s: %d vector(s)", NODE, store.label, held)
        indexes.report(wire, seconds, f"opened {store.label} with {held} vectors")
        return io.NodeOutput(wire)
