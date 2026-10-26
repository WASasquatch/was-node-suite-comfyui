"""Turn the pictures among found sources into an IMAGE batch."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.compat.types import LLM_SOURCES
from ...modules.image import sizing
from ...modules.llm import pictures, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: The node's name in messages.
NODE = "LLM Sources to Images"

#: Largest picture file read back, in bytes.
MAX_FILE_BYTES = 256 * 1024 * 1024


class LLMSourcesToImages(io.ComfyNode):
    """The pictures among found sources, as one IMAGE batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSourcesToImages",
            display_name="LLM Sources to Images",
            search_aliases=[
                "WASLLMSourcesToImages",
                "LLM Sources to Images",
                "retrieved images",
                "image search results",
                "multimodal RAG",
            ],
            category="WAS Suite/LLM/Retrieve",
            is_output_node=True,
            description=(
                "Hand on the pictures a search found as one image batch, best first, at their full "
                "size where the files are still there. Use them as references or compare them; the "
                "node previews them."
            ),
            inputs=[
                LLM_SOURCES.Input(
                    "sources",
                    tooltip="What was found, from LLM Retrieve or a query engine over an LLM Image Index. Text chunks are passed over.",
                ),
                io.Boolean.Input(
                    "full_size", default=True,
                    tooltip="`true` reads each picture's file again at its own size where it is still there; `false` uses the 384 pixel copy the index stores.",
                ),
                io.Int.Input(
                    "width", default=0, min=0, max=sizing.MAX_CANVAS_SIDE, step=8,
                    tooltip="Width every picture is brought to, in pixels: 0 = the first picture's width.",
                ),
                io.Int.Input(
                    "height", default=0, min=0, max=sizing.MAX_CANVAS_SIDE, step=8,
                    tooltip="Height every picture is brought to, in pixels: 0 = the first picture's height.",
                ),
                io.Combo.Input(
                    "fit", options=list(sizing.MODES), default=sizing.FIT_AND_PAD,
                    tooltip="How a picture of another shape is brought to the size: `fit and pad` = whole picture, bars added; `fill and crop` = fills the frame, edges cut.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The pictures found, best first, all one size.",
                ),
                io.String.Output(
                    display_name="names",
                    tooltip="One line per picture, in batch order: its file name and score, such as `ComfyUI_00042_.png 0.312`.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many pictures the batch holds.",
                ),
            ],
        )

    @classmethod
    def execute(cls, sources, full_size, width, height, fit) -> io.NodeOutput:
        """Collect the pictures and bring them to one size.

        Args:
            sources: The found sources.
            full_size: Read each file again at its own size.
            width: Target width, 0 for the first picture's.
            height: Target height, 0 for the first picture's.
            fit: How a picture of another shape is placed.

        Returns:
            The batch, the names and the count.
        """
        found = wires.expect(sources, wires.Sources, NODE, "sources", "LLM Retrieve")
        if fit not in sizing.MODES:
            raise ValueError(f"{NODE} does not know the fit `{str(fit)[:40]}`. Pick one of {', '.join(sizing.MODES)}.")
        batch, rows = pictures.to_batch(found.items, bool(full_size), (int(width), int(height)), fit, MAX_FILE_BYTES, NODE)
        if batch is None:
            raise ValueError(
                f"{NODE} was handed {len(found.items)} source(s) and none is a picture. Search an LLM "
                f"Image Index with LLM Index Retriever in its default or pictures mode."
            )
        names = "\n".join(f"{name} {score:.3f}" if score is not None else name for name, score, _ in rows)
        logger.info(
            "%s handed on %d picture(s), %d read from their files",
            NODE, len(rows), sum(1 for row in rows if row[2]),
        )
        return io.NodeOutput(batch, names, len(rows), ui=ui.PreviewImage(batch, cls=cls))
