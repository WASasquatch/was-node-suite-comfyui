"""Save an index under a name for LLM Load Index."""

from __future__ import annotations

import os
import time
from dataclasses import replace

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_INDEX
from ...modules.llm import indexes, runtime, storage, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Save Index"


class LLMSaveIndex(io.ComfyNode):
    """Save an index under a name and pass it on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSaveIndex",
            display_name="LLM Save Index",
            search_aliases=[
                "WASLLMSaveIndex",
                "LLM Save Index",
                "persist",
                "StorageContext",
                "save index",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Save an index under a name, so a later run or workflow loads it with LLM Load "
                "Index instead of building it again. Saving under a name already used replaces "
                "that index. The index passes on unchanged."
            ),
            is_output_node=True,
            inputs=[
                LLM_INDEX.Input(
                    "index",
                    tooltip="The index to save, from any LLM index node, LLM Index Insert or LLM Load Index.",
                ),
                io.String.Input(
                    "name", default="",
                    placeholder="Eg: project notes",
                    tooltip=(
                        "What the index is saved as, such as `project notes`: letters, digits, "
                        "spaces, dots, dashes and underscores. It lands in "
                        "ComfyUI/user/was-node-suite/llm/indexes."
                    ),
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The same index, now carrying the name it was saved under.",
                ),
            ],
        )

    @classmethod
    def execute(cls, index, name) -> io.NodeOutput:
        """Write the index's files and its marker.

        Args:
            index: The index.
            name: The name to save it under.

        Returns:
            The index, carrying the name.

        Raises:
            ValueError: The name cannot be used, or another folder already holds it.
        """
        wire = wires.expect(index, wires.Index, NODE, "index", "an LLM index node such as LLM Vector Index")
        if not str(name or "").strip():
            raise ValueError(f"{NODE} needs a name to save the index under. Type one, such as `project notes`.")
        name = storage.check_name(name, "Index")
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            try:
                info = indexes.persist(wire, name)
            except OSError as error:
                raise indexes.Refusal(
                    f"{NODE} could not write the index {name!r} ({error.strerror or type(error).__name__}). "
                    f"Check the free space in ComfyUI/user/was-node-suite."
                ) from None
        directory = storage.index_dir(name)
        written = [part for part in (*storage.INDEX_FILES, storage.INDEX_MARKER) if (directory / part).is_file()]
        size = sum(os.path.getsize(directory / part) for part in written)
        seconds = time.perf_counter() - started
        saved = replace(wire, name=name)
        logger.info(
            "%s saved the %s index %r: %d chunk(s), %d file(s), %d bytes in %.2fs",
            NODE, wire.kind, name, wire.size, len(written), size, seconds,
        )
        indexes.report(
            saved, seconds, f"saved the {indexes.KINDS.get(wire.kind, wire.kind)} index {name!r}",
            counts={"files": len(written), "kilobytes": round(size / 1024, 1)},
            facts={"name": name, "folder": f"llm/indexes/{name}", "saved": str(info.get("saved", ""))},
        )
        return io.NodeOutput(saved)
