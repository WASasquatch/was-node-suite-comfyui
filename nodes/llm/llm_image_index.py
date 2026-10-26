"""Embed pictures and text into an in-memory index searched by words or by a picture."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_EMBEDDING, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Image Index"


class LLMImageIndex(io.ComfyNode):
    """Embed pictures and text into an image index."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMImageIndex",
            display_name="LLM Image Index",
            search_aliases=[
                "WASLLMImageIndex",
                "LLM Image Index",
                "MultiModalVectorStoreIndex",
                "multimodal index",
                "image search",
                "multimodal RAG",
                "picture index",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Index pictures and text together, so a question finds the matching pictures, their "
                "prompts and passages, and a picture finds others like it. The panel shows the "
                "picture and text counts and a few entries."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="Pictures from LLM Load Images or LLM Images to Documents, and any text documents to search beside them.",
                ),
                LLM_EMBEDDING.Input(
                    "image_embedding",
                    tooltip="The picture model from LLM Image Embedding Model. Save or load the index with it.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="A text model for prompts, captions and passages, from LLM Embedding Model. Unconnected = the picture model reads the text, which suits short prompts.",
                ),
                io.Int.Input(
                    "batch_size", default=32, min=1, max=1024,
                    tooltip="Pictures and chunks embedded per step: 32 = the usual; lower it if VRAM runs out.",
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip="Tokens in each chunk when whole text documents are cut here: `1024` = a few paragraphs; `256` = finer matches. Pictures are never cut.",
                ),
                io.Int.Input(
                    "chunk_overlap", default=20, min=0, max=2048, optional=True, advanced=True,
                    tooltip="Tokens two neighbouring text chunks share: 20 = a little; 0 = none. Smaller than chunk_size.",
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The image index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, image_embedding, batch_size, embedding=None, chunk_size=1024,
                chunk_overlap=20) -> io.NodeOutput:
        """Embed and index the pictures and text.

        Args:
            documents: Picture documents and any text.
            image_embedding: The picture model.
            batch_size: Items embedded per step.
            embedding: A text model, or ``None`` for the picture model.
            chunk_size: Tokens per chunk for whole text documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The image index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Images")
        painter = wires.expect(image_embedding, wires.EmbeddingModel, NODE, "image_embedding", "LLM Image Embedding Model")
        if not painter.pictures:
            raise ValueError(
                f"{NODE}: {painter.label} embeds text only, so it cannot place pictures. Connect LLM "
                f"Image Embedding Model to image_embedding."
            )
        text_model = None
        if embedding is not None:
            text_model = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        wires.ready(painter, text_model)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap, pictures=True)
            index = indexes.build_image(prepared, painter, text_model, batch_size)
        wire = wires.Index(
            index=index, kind="image", embedding=text_model or painter, pictures=painter,
            size=indexes.size_of(index, "image"),
        )
        seconds = time.perf_counter() - started
        shown = indexes.picture_count(index)
        logger.info("%s embedded %d picture(s) and %d item(s) in %.2fs", NODE, shown, wire.size, seconds)
        indexes.report(
            wire, seconds, f"embedded {shown} pictures with {painter.label}", prepared,
            facts={"batch size": str(int(batch_size))},
        )
        return io.NodeOutput(wire)
