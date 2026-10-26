"""Add documents to a copy of an index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Index Insert"


class LLMIndexInsert(io.ComfyNode):
    """Add documents to an index without building it again."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMIndexInsert",
            display_name="LLM Index Insert",
            search_aliases=[
                "WASLLMIndexInsert",
                "LLM Index Insert",
                "insert_nodes",
                "insert documents",
                "add documents",
                "update index",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Add documents to an index without building it again, skipping the ones it holds "
                "unchanged and replacing the ones that changed. The index on the input is left as it "
                "was and the copy passes on. The panel shows what was added, skipped and replaced."
            ),
            inputs=[
                LLM_INDEX.Input(
                    "index",
                    tooltip=(
                        "The index to add to, from an LLM index node or LLM Load Index. A vector "
                        "index embeds the new chunks with its own embedding model; a tree index is "
                        "summarized again from all its chunks."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "The documents to add, such as from LLM Load Documents. Whole documents are "
                        "cut with the chunk size the index was built with."
                    ),
                ),
                io.Combo.Input(
                    "mode", options=list(indexes.INSERT_MODES), default="update", optional=True,
                    tooltip=(
                        "`update` = a document the index holds unchanged is skipped, and one that changed "
                        "has its old chunks replaced; `add` = everything is added beside what is there."
                    ),
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The index with the documents added, unsaved until it reaches LLM Save Index.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, index, documents, mode="update") -> io.NodeOutput:
        """Split the documents with the index's own settings and insert them into a copy.

        Args:
            index: The index.
            documents: The documents or chunks to add.
            mode: ``update`` or ``add``.

        Returns:
            A new index holding the documents.
        """
        wire = wires.expect(index, wires.Index, NODE, "index", "an LLM index node or LLM Load Index")
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        wires.ready(wire)
        held = indexes.settings_of(wire.index)
        chunk_size = held.get("chunk_size", indexes.CHUNK_SIZE)
        chunk_overlap = held.get("chunk_overlap", indexes.CHUNK_OVERLAP)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap, pictures=wire.kind == "image")
            if mode not in indexes.INSERT_MODES:
                raise indexes.Refusal(f"{NODE}: mode {str(mode)[:40]!r} is not one of {', '.join(indexes.INSERT_MODES)}.")
            grown, stats = await indexes.insert(wire, prepared, NODE, mode)
        seconds = time.perf_counter() - started
        added = stats["added"]
        logger.info(
            "%s added %d chunk(s) to the %s index, %d document(s) unchanged, %d replaced, %d before and %d after, in %.2fs",
            NODE, added, wire.kind, stats["skipped"], stats["replaced"], wire.size, grown.size, seconds,
        )
        kind = indexes.KINDS.get(wire.kind, wire.kind)
        summary = f"added {added} chunks to the {kind} index"
        if stats["skipped"] or stats["replaced"]:
            summary += f"; {stats['skipped']} document(s) unchanged, {stats['replaced']} replaced"
        indexes.report(
            grown, seconds, summary, prepared,
            counts={"added": added, "unchanged": stats["skipped"], "replaced": stats["replaced"]},
            facts={"mode": str(mode)},
        )
        return io.NodeOutput(grown)
