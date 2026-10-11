"""BLIP captioning and question-answering model loading."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import BLIP_MODEL
from ...modules.model import blip

REQUIRES = "blip"

#: Values the ``blip_model`` widget held in early WAS Node Suite 2 releases, when it named a
#: mode rather than a repository. Both stood for the default captioning model.
LEGACY_MODE_NAMES = ("caption", "interrogate")


class BlipModelLoader(io.ComfyNode):
    """Load the two BLIP models `BLIP Analyze Image` runs on.

    The BLIP_MODEL socket carries ``{"caption": backend, "question": backend}``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BLIP Model Loader",
            display_name="BLIP Model Loader",
            search_aliases=["BLIP Model Loader", "blip", "caption model", "interrogate"],
            category="WAS Suite/Loaders",
            description=(
                "Load the BLIP captioning and visual question answering models for BLIP "
                "Analyze Image. Enable features.blip to load this node."
            ),
            inputs=[
                io.String.Input(
                    "blip_model",
                    default="Salesforce/blip-image-captioning-base",
                    tooltip=(
                        "Hugging Face repository of the captioning model for caption mode. "
                        "'Salesforce/blip-image-captioning-large' = heavier and more detailed "
                        "than the default. Repository names only: a folder path is refused, "
                        "and a local checkpoint in ComfyUI's models directory is picked up."
                    ),
                ),
                io.String.Input(
                    "vqa_model_id",
                    default="Salesforce/blip-vqa-base",
                    tooltip=(
                        "Hugging Face repository of the question answering model for "
                        "interrogate mode; it must be a BLIP VQA model. Repository names "
                        "only: a folder path is refused, and a local checkpoint in ComfyUI's "
                        "models directory is picked up."
                    ),
                ),
                io.Combo.Input(
                    "device",
                    options=["cuda", "cpu"],
                    tooltip=(
                        "Where the models are held. `cuda` is faster and costs VRAM for as "
                        "long as they stay loaded; `cpu` keeps the GPU free. `cuda` on a "
                        "machine with no CUDA device falls back to the CPU."
                    ),
                ),
            ],
            outputs=[
                BLIP_MODEL.Output(
                    tooltip=(
                        "Both loaded models, for the blip_model input of BLIP Analyze Image."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, blip_model, vqa_model_id, device) -> io.NodeOutput:
        # An empty repository id selects the task's default, which is the captioning model
        # these two mode names stood for.
        if blip_model in LEGACY_MODE_NAMES:
            blip_model = ""
        # Captioning and question answering are separate repositories with separate model
        # classes, so both are loaded here and handed on together.
        return io.NodeOutput(
            {
                "caption": blip.load(blip_model, task="caption", device=device),
                "question": blip.load(vqa_model_id, task="question", device=device),
            }
        )
