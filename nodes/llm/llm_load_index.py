"""Load an index saved with LLM Save Index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_EMBEDDING, LLM_INDEX
from ...modules.llm import indexes, runtime, storage, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Load Index"


class LLMLoadIndex(io.ComfyNode):
    """Load a saved index with the models it needs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMLoadIndex",
            display_name="LLM Load Index",
            search_aliases=[
                "WASLLMLoadIndex",
                "LLM Load Index",
                "load_index_from_storage",
                "load index",
                "persist",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Load an index saved with LLM Save Index, ready for a retriever or query engine "
                "without building it again. Vector, graph and image indexes need the models they "
                "were built with. The panel shows what the index holds and those models."
            ),
            inputs=[
                io.Combo.Input(
                    "name",
                    options=indexes.saved_names(),
                    tooltip=(
                        "Which saved index to load, by the name LLM Save Index gave it. The indexes "
                        "are in ComfyUI/user/was-node-suite/llm/indexes; press R after saving a new one."
                    ),
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip=(
                        "The model the index asks when searched or added to, such as for a tree, "
                        "keyword or document summary index built with one. Any model works; it "
                        "need not be the one it was built with."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip=(
                        "The embedding model the index was built with, from LLM Embedding Model. "
                        "Needed for a vector index, and for a graph, document summary or image index "
                        "built with one."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "image_embedding", optional=True,
                    tooltip="The picture model an image index was built with, from LLM Image Embedding Model. Needed for an image index.",
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The loaded index, for LLM Index Retriever, a query engine or LLM Index Insert.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, name) -> bool | str:
        """Accept any well-formed name, so an index saved earlier in the same run is found."""
        if str(name or "") == indexes.NO_INDEXES:
            return "No index has been saved yet. Build one, save it with LLM Save Index, then pick its name here."
        try:
            storage.check_name(name, "Index")
        except ValueError as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, name="", **kwargs) -> str:
        """A stamp that moves whenever the index is saved again under the name."""
        return indexes.stamp(name)

    @classmethod
    def execute(cls, name, llm=None, embedding=None, image_embedding=None) -> io.NodeOutput:
        """Read the saved files and open the index with the models given.

        Args:
            name: The saved name.
            llm: The language model, or ``None``.
            embedding: The embedding model, or ``None``.
            image_embedding: The picture model, or ``None``.

        Returns:
            The index.
        """
        model = None
        if llm is not None:
            model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI")
        embed = None
        if embedding is not None:
            embed = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        painter = None
        if image_embedding is not None:
            painter = wires.expect(image_embedding, wires.EmbeddingModel, NODE, "image_embedding", "LLM Image Embedding Model")
        wires.ready(model, embed, painter)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            wire, warnings = indexes.load(name, model, embed, NODE, pictures=painter)
        info = storage.index_info(wire.name) or {}
        seconds = time.perf_counter() - started
        logger.info("%s opened the %s index %r: %d chunk(s) in %.2fs", NODE, wire.kind, wire.name, wire.size, seconds)
        facts = {"name": wire.name, "saved": str(info.get("saved", ""))}
        if isinstance(info.get("embedding"), str):
            facts["built with"] = info["embedding"]
        if isinstance(info.get("llm"), str):
            facts["built by"] = info["llm"]
        if isinstance(info.get("pictures"), str):
            facts["pictures by"] = info["pictures"]
        indexes.report(
            wire, seconds, f"loaded the {indexes.KINDS.get(wire.kind, wire.kind)} index {wire.name!r}",
            warnings=warnings, facts=facts,
        )
        return io.NodeOutput(wire)
