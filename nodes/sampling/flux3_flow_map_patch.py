"""A FLUX 3 Image Klein base model given the distilled checkpoint's target-time branch."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model import flux3_flow_map

#: The distilled checkpoint the menu starts on, where the folder holds it.
DISTILLED = "flux3_image_klein_sd_bf16.safetensors"


class Flux3FlowMapPatch(io.ComfyNode):
    """Attach a distilled FLUX 3 Image Klein checkpoint's target-time branch to a model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = flux3_flow_map.checkpoints()
        default = next((name for name in options if name.replace("\\", "/").endswith(DISTILLED)),
                       next((name for name in options if "_sd_" in name), None))
        return io.Schema(
            node_id="WASFlux3FlowMapPatch",
            display_name="Flux 3 Flow Map Patch",
            search_aliases=[
                "WASFlux3FlowMapPatch",
                "Flux 3 Flow Map Patch",
                "flux 3 klein",
                "ref_time_in",
                "distill lora",
                "flow map",
                "target time",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Give a FLUX 3 Image Klein base model the distilled checkpoint's target-time "
                "branch, so the base with a distillation LoRA samples the way the distilled "
                "model does, from 16 steps. CFG 1 matches the distilled picture from the same "
                "seed; CFG 2.5 to 3.5 adds contrast and colour. Only the branch's two weights are "
                "read."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "The FLUX 3 Image Klein base model, before or after the distillation LoRA: "
                        "Load Diffusion Model on flux3_image_klein_bf16, then Load LoRA on "
                        "flux3_image_klein_distill_r256, r64, r32 or r16."
                    ),
                ),
                io.Combo.Input(
                    "checkpoint",
                    options=options,
                    default=default,
                    tooltip=(
                        "The distilled checkpoint the branch is read from: "
                        "flux3_image_klein_sd_bf16 or flux3_image_klein_sd_int8_convrot. Both hold "
                        "the same bf16 branch. A file with no _sd_ holds none."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "The model with the branch, for a KSampler at 16 steps: CFG 1 for the "
                        "distilled picture, 2.5 to 3.5 for more contrast and colour."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, checkpoint) -> io.NodeOutput:
        """Attach the branch read from ``checkpoint`` to a clone of ``model``."""
        return io.NodeOutput(flux3_flow_map.attach(model, checkpoint))
