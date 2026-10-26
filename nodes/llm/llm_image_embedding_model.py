"""Pick the model that embeds pictures and text into one space, for picture search."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM_EMBEDDING
from ...modules.interface import run_result
from ...modules.llm import pictures, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM Image Embedding Model"


class LLMImageEmbeddingModel(io.ComfyNode):
    """A CLIP or SigLIP model that embeds pictures and text into one space."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMImageEmbeddingModel",
            display_name="LLM Image Embedding Model",
            search_aliases=[
                "WASLLMImageEmbeddingModel",
                "LLM Image Embedding Model",
                "ClipEmbedding",
                "MultiModalEmbedding",
                "SigLIP",
                "CLIP",
                "image search",
                "multimodal embedding",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose a CLIP or SigLIP model that places pictures and text in the same space, "
                "so words find pictures and a picture finds others like it. Use it with LLM Image "
                "Index. The panel shows the vector length and device."
            ),
            inputs=[
                io.Combo.Input(
                    "model", options=pictures.choices(),
                    tooltip=(
                        "A CLIP or SigLIP folder in models/LLM, or a suggested model downloaded there on "
                        f"first use with features.network on. `{pictures.SUGGESTED[0]}` = small and quick; "
                        f"`{pictures.SUGGESTED[1]}` = more exact."
                    ),
                ),
                io.Int.Input(
                    "batch_size", default=16, min=1, max=256,
                    tooltip="Pictures or texts embedded together: 16 = the usual; lower it if VRAM runs out.",
                ),
            ],
            outputs=[
                LLM_EMBEDDING.Output(
                    display_name="image_embedding",
                    tooltip="The picture model, for LLM Image Index or the image_embedding input of LLM Load Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, batch_size) -> io.NodeOutput:
        """Load the model and measure its vectors.

        Args:
            model: A menu entry.
            batch_size: Pictures or texts embedded together.

        Returns:
            The picture embedding model.
        """
        with runtime.guarded():
            built, details = pictures.picture_embedding(model, int(batch_size), NODE)
        run_result.publish(
            status=run_result.OK,
            summary=f"{built.label} makes vectors of {built.dimensions} numbers on {details['device']}.",
            counts={"dimensions": built.dimensions},
            facts={
                "model": built.label,
                "family": details["family"],
                "device": details["device"],
                "pictures go": "nowhere, they stay on this machine",
            },
        )
        return io.NodeOutput(built)
