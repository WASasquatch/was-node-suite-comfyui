"""Segment Anything model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import SAM_MODEL
from ...modules.model import sam

REQUIRES = "sam"


class SamModelLoader(io.ComfyNode):
    """Load the Segment Anything model for `SAM Image Mask`."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM Model Loader",
            display_name="SAM Model Loader",
            search_aliases=["SAM Model Loader", "segment anything", "sam", "vit"],
            category="WAS Suite/Loaders",
            description=(
                "Load a Segment Anything model for SAM Image Mask, which turns clicked "
                "points into a mask. Enable features.sam to load this node."
            ),
            inputs=[
                io.Combo.Input(
                    "model_size",
                    options=["ViT-H", "ViT-L", "ViT-B"],
                    tooltip=(
                        "Which size of Segment Anything to load: `ViT-B` = about 375 MB, "
                        "fastest; `ViT-L` = middle size; `ViT-H` = about 2.4 GB, most "
                        "accurate. All three take the same points."
                    ),
                ),
            ],
            outputs=[
                SAM_MODEL.Output(
                    tooltip=(
                        "The loaded model, for the sam_model input of SAM Image Mask."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model_size) -> io.NodeOutput:
        # Loading here rather than inside the masking node keeps one copy resident no
        # matter how many nodes read it.
        return io.NodeOutput(sam.load(model_size))
