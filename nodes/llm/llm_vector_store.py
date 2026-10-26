"""Pick a vector store collection an index keeps its vectors in."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_VECTOR_STORE
from ...modules.interface import run_result
from ...modules.llm import runtime, vector_stores

REQUIRES = "llm"

logger = log.get_logger("llm.vector_stores")

#: The node's name in messages.
NODE = "LLM Vector Store"


class LLMVectorStore(io.ComfyNode):
    """A Qdrant collection on this machine or on a configured server."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMVectorStore",
            display_name="LLM Vector Store",
            search_aliases=[
                "WASLLMVectorStore",
                "LLM Vector Store",
                "QdrantVectorStore",
                "ChromaVectorStore",
                "LanceDBVectorStore",
                "FaissVectorStore",
                "PGVectorStore",
                "Qdrant",
                "Chroma",
                "LanceDB",
                "FAISS",
                "pgvector",
                "vector database",
                "persistent index",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Keep an index's vectors in a Qdrant, Chroma, LanceDB or FAISS collection on this machine, "
                "or on a server named in config.yaml, so a large index is written once and opened at once. "
                "Connect it to LLM Vector Index or LLM Vector Store Index."
            ),
            inputs=[
                io.Combo.Input(
                    "store", options=vector_stores.options(),
                    tooltip=(
                        "`qdrant`, `chroma`, `lancedb`, `faiss` = a folder in ComfyUI/user/was-node-suite/llm/vector_stores, all but "
                        "qdrant with features.vector_stores on; `server: <name>` = a Qdrant, Chroma or Postgres server under llm.vector_stores."
                    ),
                ),
                io.String.Input(
                    "collection", default="documents",
                    tooltip="The collection's name, such as `studio_notes`: letters, digits, dashes and underscores; lower case and no dashes on Postgres. A new name starts an empty one.",
                ),
            ],
            outputs=[
                LLM_VECTOR_STORE.Output(
                    display_name="vector_store",
                    tooltip="The collection, for LLM Vector Index or LLM Vector Store Index.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """Always run, so the count on the panel follows the collection."""
        return float("NaN")

    @classmethod
    def execute(cls, store, collection) -> io.NodeOutput:
        """Open the collection.

        Args:
            store: ``local`` or a configured server.
            collection: The collection's name.

        Returns:
            The vector store.
        """
        with runtime.guarded():
            opened = vector_stores.open_store(store, collection, NODE)
        held = vector_stores.count(opened)
        logger.info("%s opened %s holding %d vector(s)", NODE, opened.label, held)
        run_result.publish(
            status=run_result.OK,
            summary=f"{opened.label} holds {held} vector(s)" if held else f"{opened.label} is empty; an index fills it",
            counts={"vectors": held},
            facts={
                "collection": opened.collection,
                "backend": opened.kind,
                "where": "this machine" if opened.local else "a server off this machine",
            },
        )
        return io.NodeOutput(opened)
