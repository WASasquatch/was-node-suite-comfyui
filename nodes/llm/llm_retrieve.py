"""LLM Retrieve: finds the chunks that answer a question and hands them on as text and sources."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_POSTPROCESSORS, LLM_RETRIEVER, LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import pictures, retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")


class LLMRetrieve(io.ComfyNode):
    """Retrieves chunks for a question, through an optional postprocessor chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRetrieve",
            display_name="LLM Retrieve",
            search_aliases=[
                "WASLLMRetrieve",
                "LLM Retrieve",
                "retrieve",
                "search documents",
                "context for prompt",
                "RAG",
            ],
            category="WAS Suite/LLM/Retrieve",
            is_output_node=True,
            description=(
                "Find the chunks that answer a question, or the pictures most like a picture, and "
                "hand them on as numbered text for a prompt and as sources with their scores. The "
                "panel lists each find with its score and the file it came from."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="What searches, from LLM Index Retriever, LLM BM25 Retriever, LLM Fusion Retriever or LLM Auto-Merging Retriever.",
                ),
                io.String.Input(
                    "query", multiline=True, default="",
                    tooltip="The question to find chunks for, such as `What does the warranty cover?`",
                ),
                LLM_POSTPROCESSORS.Input(
                    "postprocessors", optional=True,
                    tooltip="Steps run over the found chunks in order, from LLM Postprocessor or LLM Rerank: a cutoff, a rerank, neighbouring chunks.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="A picture to search by in place of the question; the first frame is used. Needs LLM Index Retriever over an LLM Image Index.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The chunks as `[1] file, page` blocks with their text, separated by blank lines, ready to place in a prompt.",
                ),
                LLM_SOURCES.Output(
                    display_name="sources",
                    tooltip="The chunks with their scores, best first, for LLM Sources to Text or an evaluator.",
                ),
            ],
        )

    @classmethod
    def execute(cls, retriever, query, postprocessors=None, image=None) -> io.NodeOutput:
        """Retrieve and postprocess.

        Args:
            retriever: The retriever wire.
            query: The question.
            postprocessors: Postprocessor steps in chain order.
            image: A picture to search by, or ``None``.

        Returns:
            The chunks as prompt text, and as sources.
        """
        node = "LLM Retrieve"
        retriever = wires.expect(retriever, wires.Retriever, node, "retriever", "an LLM retriever node")
        steps = tuple(postprocessors) if isinstance(postprocessors, (tuple, list)) else ()
        if postprocessors is not None and not isinstance(postprocessors, (tuple, list)):
            raise ValueError(f"{node} was handed a {type(postprocessors).__name__} on postprocessors. Connect LLM Postprocessor or LLM Rerank.")
        if not wires.every(steps, wires.Postprocessor):
            raise ValueError(f"{node}: postprocessors carries something other than postprocessor steps. Connect LLM Postprocessor or LLM Rerank.")
        question = str(query or "").strip()
        picture = pictures.png_bytes(image) if image is not None else None
        if not question and picture is None:
            raise ValueError(f"{node} needs a question. Type one into query, such as `What does the warranty cover?`")
        wires.ready(retriever, steps)
        with runtime.guarded():
            found, retrieved, seconds = retrieval.retrieve(retriever, question, steps, node, picture=picture)
        text = retrieval.prompt_text(found)
        sources = wires.Sources(items=tuple(found), query=question)
        logger.info(
            "LLM Retrieve found %d chunk(s), kept %d after %d step(s), in %.2fs",
            retrieved, len(found), len(steps), seconds,
        )
        facts = {"retriever": retriever.label[:64], "seconds": f"{seconds:.2f}"}
        if picture is not None:
            facts["searched by"] = "a picture"
        if steps:
            facts["steps"] = ", ".join(step.label for step in steps)[:64]
        run_result.publish(
            status=run_result.OK if found else run_result.WARNING,
            summary=(
                f"{len(found)} chunk(s) for the question, from {retrieved} found."
                if found else "Nothing matched the question. Widen top_k or loosen the filters."
            ),
            counts={"found": retrieved, "kept": len(found), "characters": len(text)},
            facts=facts,
            items=retrieval.rows(found, run_result.MAX_ITEMS),
            items_total=len(found),
            bodies=[run_result.body("context", text)] if text else None,
        )
        return io.NodeOutput(text, sources)
