"""CLIPSeg segmentation model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import CLIPSEG_MODEL
from ...modules.model import clipseg

REQUIRES = "clipseg"


class ClipsegModelLoader(io.ComfyNode):
    """Load the CLIPSeg processor and model the CLIPSeg masking nodes run on.

    The CLIPSEG_MODEL socket carries ``(processor, model)``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPSeg Model Loader",
            display_name="CLIPSeg Model Loader",
            search_aliases=["CLIPSeg Model Loader", "clipseg", "segmentation", "text mask"],
            category="WAS Suite/Loaders",
            description=(
                "Load a CLIPSeg model for the CLIPSeg masking nodes, which turn a text "
                "description into a mask. Enable features.clipseg to load this node."
            ),
            inputs=[
                io.String.Input(
                    "model",
                    default="CIDAS/clipseg-rd64-refined",
                    multiline=False,
                    tooltip=(
                        "Hugging Face repository of the CLIPSeg model. "
                        "'CIDAS/clipseg-rd64-refined' = the general-use default; "
                        "'CIDAS/clipseg-rd16' = smaller and coarser. Repository names only: a "
                        "folder path is refused, and a local checkpoint in ComfyUI's models "
                        "directory is picked up."
                    ),
                ),
            ],
            outputs=[
                CLIPSEG_MODEL.Output(
                    display_name="clipseg_model",
                    tooltip=(
                        "The loaded model, for the clipseg_model input of CLIPSeg Masking "
                        "and CLIPSeg Batch Masking."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(clipseg.load(model))
