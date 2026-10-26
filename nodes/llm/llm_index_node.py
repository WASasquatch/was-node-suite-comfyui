"""A chunk that hands a search on to a named retriever or query engine."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import retrieval, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Index Node"


class LLMIndexNode(io.ComfyNode):
    """An index node: text that, once found, hands the search on to a named tool."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMIndexNode",
            display_name="LLM Index Node",
            search_aliases=[
                "WASLLMIndexNode",
                "LLM Index Node",
                "IndexNode",
                "recursive retrieval",
                "document summary link",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Write a chunk that, once a search finds it, hands the search on to the retriever or "
                "query engine it names, such as a summary of each manual pointing at that manual's own "
                "index. Index the chain and search it with LLM Recursive Retriever."
            ),
            inputs=[
                io.String.Input(
                    "text", multiline=True, default="",
                    tooltip="What is behind it, searched like any chunk, such as `The R5 camera manual: batteries, menus, video modes.`",
                ),
                io.String.Input(
                    "target", default="",
                    tooltip="The name of the LLM Retriever Tool or LLM Query Engine Tool it hands on to, such as `camera_manual`.",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="Earlier index nodes or documents, kept ahead of this one.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip="The chain with this index node appended, for an LLM index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text, target, documents=None) -> io.NodeOutput:
        """Append the index node.

        Args:
            text: What is behind it.
            target: The tool it hands on to.
            documents: Earlier items.

        Returns:
            The documents.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Index Node")
        made = retrieval.index_node(text, target)
        added = wires.Documents((made,), f"index node to {made.index_id}")
        result = previous.plus(added) if previous is not None else added
        run_result.publish(
            status=run_result.OK,
            summary=f"hands on to {made.index_id}; {len(result)} in the chain",
            counts={"in the chain": len(result)},
            facts={"target": made.index_id},
            bodies=[run_result.body("text", made.text[:2000])],
        )
        return io.NodeOutput(result)
