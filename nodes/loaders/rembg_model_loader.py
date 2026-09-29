"""Cutout model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import REMBG_MODEL
from ...modules.model import cutout

REQUIRES = "preprocessors"

#: Shown in the model list when neither folder holds a usable checkpoint, so the widget has
#: something to draw.
NO_CHECKPOINT = "put a checkpoint in models/birefnet or models/ben2"


class RembgModelLoader(io.ComfyNode):
    """Build the cutout network Image Remove Background runs on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        found = cutout.offered()
        return io.Schema(
            node_id="WASRembgModelLoader",
            display_name="Image Remove Background Model Loader",
            search_aliases=[
                "WASRembgModelLoader",
                "Image Remove Background Model Loader",
                "Cutout Model Loader",
                "Rembg Model Loader",
                "rembg",
                "remove background",
                "cutout",
                "birefnet",
                "ben2",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Build a cutout network for Image Remove Background from a checkpoint in "
                "ComfyUI/models/birefnet or ComfyUI/models/ben2. The network is kept for the "
                "life of the process, so one loader can feed several nodes. With "
                "features.network on, the published checkpoints are listed before they are "
                "downloaded and fetched on first use."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=found or [NO_CHECKPOINT],
                    default=cutout.DEFAULT if cutout.DEFAULT in found else None,
                    tooltip=(
                        "Checkpoint to build, as folder/file. `birefnet/General.safetensors` "
                        "suits most pictures, `Portrait` people, `Matting-HR` hair and fine "
                        "edges; `ben2/ben2-base.safetensors` is a second opinion. Lists every "
                        "full size BiRefNet or BEN2 .safetensors in those folders; Lite files "
                        "are left out."
                    ),
                ),
            ],
            outputs=[
                REMBG_MODEL.Output(
                    display_name="rembg_model",
                    tooltip=(
                        "The built network, for the rembg_model input of Image Remove "
                        "Background."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, model) -> bool | str:
        """Accept a listed checkpoint or a name an earlier menu offered.

        Args:
            model: The stored combo value.

        Returns:
            True, or the message naming what to pick.
        """
        if model in cutout.LEGACY or model in cutout.offered():
            return True
        if model == NO_CHECKPOINT:
            return (
                "no cutout checkpoint was found. Put a BiRefNet or BEN2 .safetensors in "
                "ComfyUI/models/birefnet or ComfyUI/models/ben2, or set features.network: "
                "true in config.yaml, then press R to refresh the list"
            )
        return (
            f"{model} is not in models/birefnet or models/ben2 any more, or is not a full "
            "size BiRefNet or BEN2 checkpoint. Pick one from the model list"
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(cutout.load(model))
