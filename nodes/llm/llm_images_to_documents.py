"""Turn an IMAGE batch into picture documents, with a caption per frame."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import documents as loading
from ...modules.llm import pictures, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.documents")

#: The node's name in messages.
NODE = "LLM Images to Documents"

#: Frames one run turns into documents at most.
MAX_FRAMES = 1024

#: What a vision model is asked about each frame, unless the node says otherwise.
DESCRIBE_PROMPT = (
    "Describe this picture in one short paragraph for a search index: the subject, setting, "
    "style, colours and composition. Write only the description."
)


class LLMImagesToDocuments(io.ComfyNode):
    """Picture documents from an IMAGE batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMImagesToDocuments",
            display_name="LLM Images to Documents",
            search_aliases=[
                "WASLLMImagesToDocuments",
                "LLM Images to Documents",
                "ImageNode",
                "image to document",
                "multimodal RAG",
                "picture documents",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Turn pictures from the graph into documents for LLM Image Index, one per frame, each "
                "with a caption from a line of text or written by a vision model. The panel shows "
                "the count and a sample of the captions."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The pictures, one document per frame.",
                ),
                io.String.Input(
                    "captions", multiline=True, default="",
                    tooltip="One caption per line, in frame order, such as `red car at night`. Frames past the last line get none; blank = no captions.",
                ),
                io.String.Input(
                    "name", default="picture",
                    tooltip="What the documents are called, numbered per frame: `reference` = reference 1, reference 2.",
                ),
                LLM.Input(
                    "describe", optional=True,
                    tooltip="A vision model that writes a description of each frame into its text, after any caption. Slow: one answer per frame.",
                ),
                io.String.Input(
                    "describe_prompt", multiline=True, default=DESCRIBE_PROMPT, optional=True, advanced=True,
                    tooltip="What the vision model is asked about each frame, such as `Name the art style of this picture.`",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="Documents from another loader. The frames' documents are added after them.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip="One picture document per frame, for LLM Image Index or LLM Index Insert.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, captions, name, describe=None, describe_prompt=DESCRIBE_PROMPT,
                documents=None) -> io.NodeOutput:
        """Store each frame as a picture document.

        Args:
            image: The pictures.
            captions: One caption per line.
            name: What the documents are called.
            describe: A vision model describing each frame, or ``None``.
            describe_prompt: What it is asked.
            documents: Documents to add these after.

        Returns:
            The documents.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(documents, wires.Documents, NODE, "documents", "an LLM document loader")
        model = None
        if describe is not None:
            model = wires.expect(describe, wires.LanguageModel, NODE, "describe", "LLM From CLIP with a vision model")
        frames = image if image.ndim == 4 else image[None]
        if frames.shape[0] > MAX_FRAMES:
            raise ValueError(
                f"{NODE} was given {frames.shape[0]} frames and turns {MAX_FRAMES} into documents at "
                f"most. Pick fewer frames, such as with Image Batch Get."
            )
        lines = [line for line in str(captions or "").splitlines()]
        label = str(name or "").strip()[:80] or "picture"
        wires.ready(model)
        made = pictures.tensor_nodes(frames, lines, label, loading.clean_metadata({"title": label}))
        described = 0
        if model is not None:
            asked = str(describe_prompt or "").strip() or DESCRIBE_PROMPT
            with runtime.guarded():
                described = pictures.describe(made, model, asked, NODE)
        added = wires.Documents(tuple(made), f"{len(made)} pictures from the graph")
        result = previous.plus(added) if previous is not None else added
        logger.info("%s stored %d frame(s), %d described", NODE, len(made), described)
        counts = {"pictures": len(made), "captioned": sum(1 for item in made if item.text)}
        if model is not None:
            counts["described"] = described
        if len(result) != len(added):
            counts["with chain"] = len(result)
        run_result.publish(
            status=run_result.OK,
            summary=f"stored {len(made)} picture(s)",
            counts=counts,
            facts={"named": label, **({"described by": model.label} if model is not None else {})},
            bodies=[
                run_result.body(str(item.metadata.get("file_name", "")), item.text[:2000])
                for item in [node for node in made if node.text][:3]
            ],
        )
        return io.NodeOutput(result)
