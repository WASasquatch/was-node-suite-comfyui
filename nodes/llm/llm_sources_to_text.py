"""LLM Sources to Text: writes retrieved chunks out through a template."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"


class LLMSourcesToText(io.ComfyNode):
    """Formats sources as text through a template."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSourcesToText",
            display_name="LLM Sources to Text",
            search_aliases=[
                "WASLLMSourcesToText",
                "LLM Sources to Text",
                "citations",
                "references",
                "format sources",
                "source nodes",
            ],
            category="WAS Suite/LLM/Retrieve",
            is_output_node=True,
            description=(
                "Write the chunks a retrieval or a query answered from as text, one block per chunk "
                "through a template, such as a numbered reference list or context for another prompt."
            ),
            inputs=[
                LLM_SOURCES.Input(
                    "sources",
                    tooltip="The chunks and scores, from LLM Retrieve or a query node.",
                ),
                io.String.Input(
                    "template", multiline=True, default="[{rank}] {file_name} ({score})\n{text}",
                    tooltip=(
                        "One block per chunk. Fields: {rank}, {score}, {text}, {file_name}, {page} and "
                        "any metadata key, such as {last_modified_date}. Any other {name} is written as typed."
                    ),
                ),
                io.String.Input(
                    "separator", default="\\n\\n",
                    tooltip="Text between blocks: `\\n` = a new line, `\\t` = a tab, so `\\n\\n` = a blank line.",
                ),
                io.Int.Input(
                    "max_chars", default=0, min=0, max=retrieval.MAX_TEXT_CHARS,
                    tooltip="Most characters written, in whole blocks unless the first alone is longer: 0 = no limit of its own; 12000 = about 3000 tokens of context.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The blocks joined by the separator.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many blocks were written, fewer than the sources when max_chars stopped it.",
                ),
            ],
        )

    @classmethod
    def execute(cls, sources, template, separator, max_chars) -> io.NodeOutput:
        """Write the sources out.

        Args:
            sources: The sources wire.
            template: One block's template.
            separator: Text between blocks.
            max_chars: Most characters written.

        Returns:
            The text and how many blocks it holds.
        """
        node = "LLM Sources to Text"
        sources = wires.expect(sources, wires.Sources, node, "sources", "LLM Retrieve")
        runtime.core()
        text, count, cut = retrieval.sources_text(sources.items, template, separator, max_chars)
        run_result.publish(
            status=run_result.OK if count or not len(sources) else run_result.WARNING,
            summary=(
                f"{count} of {len(sources)} source(s) written" + (", stopped by max_chars." if cut else ".")
            ),
            counts={"sources": len(sources), "written": count, "characters": len(text)},
            bodies=[run_result.body("text", text)] if text else None,
        )
        return io.NodeOutput(text, count)
