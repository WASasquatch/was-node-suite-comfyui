"""A retriever whose model turns each question into metadata filters and a search."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_INDEX, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Auto Retriever"


class LLMAutoRetriever(io.ComfyNode):
    """A vector retriever that writes its own metadata filters for each question."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMAutoRetriever",
            display_name="LLM Auto Retriever",
            search_aliases=[
                "WASLLMAutoRetriever",
                "LLM Auto Retriever",
                "VectorIndexAutoRetriever",
                "auto retrieval",
                "self query",
                "metadata filters",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Search a vector index with filters the model writes from each question, so `notes from "
                "the billing file` searches only that file's chunks. It filters on the metadata listed, "
                "or on what the index holds. The panel shows the fields."
            ),
            inputs=[
                LLM_INDEX.Input(
                    "index",
                    tooltip="A vector index, from LLM Vector Index, LLM Vector Store Index or LLM Load Index.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the filters, such as LLM From CLIP or LLM OpenAI.",
                ),
                io.String.Input(
                    "content", default="",
                    tooltip="What the chunks are, read by the model, such as `studio notes on rendering, delivery and billing`. Blank = chunks of documents.",
                ),
                io.String.Input(
                    "metadata_fields", multiline=True, default="",
                    tooltip="One field per line, `name (type): description`, such as `file_name (str): the note a chunk came from`. Blank = every key the index holds.",
                ),
                io.Int.Input(
                    "top_k", default=4, min=1, max=50,
                    tooltip="Chunks one search hands on: 4 = a short context.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The retriever, for LLM Retrieve, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, index, llm, content, metadata_fields, top_k) -> io.NodeOutput:
        """Build the retriever.

        Args:
            index: A vector index.
            llm: The model writing the filters.
            content: What the chunks are.
            metadata_fields: Field lines, or blank.
            top_k: Chunks one search hands on.

        Returns:
            The retriever.
        """
        wire = wires.expect(index, wires.Index, NODE, "index", "LLM Vector Index")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM OpenAI or LLM From CLIP")
        wires.ready(wire, model)
        with runtime.guarded():
            built, facts = retrieval.auto_retriever(wire, model, content, metadata_fields, int(top_k))
        logger.info("%s filters on %s", NODE, facts.get("filters on", ""))
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {built.label}.",
            counts={"top_k": int(top_k)},
            facts=facts,
            items=[
                f"{name} ({kind}): {text}"
                for name, kind, text in retrieval.metadata_fields(wire.index, metadata_fields)
            ],
        )
        return io.NodeOutput(built)
