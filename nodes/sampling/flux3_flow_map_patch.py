"""A FLUX 3 Image Klein base model given the distilled checkpoint's target-time branch."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.io import picker
from ...modules.model import flux3_flow_map

#: Distilled checkpoints the menu starts on, the first the folder holds: BFL's published names, then older ones.
DISTILLED = (
    "flux3-image-klein-bf16.safetensors",
    "flux3-image-klein-int8-convrot.safetensors",
    "flux3_image_klein_sd_bf16.safetensors",
    "flux3_image_klein_sd_int8_convrot.safetensors",
)


class Flux3FlowMapPatch(io.ComfyNode):
    """Attach a distilled FLUX 3 Image Klein checkpoint's target-time branch to a model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = flux3_flow_map.checkpoints()
        plain = {name.replace("\\", "/").rsplit("/", 1)[-1]: name for name in options}
        default = next((plain[name] for name in DISTILLED if name in plain), None)
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
                        "Load Diffusion Model on flux3-image-klein-base-bf16, then Load LoRA on "
                        "flux3_image_klein_distill_r256, r64, r32 or r16."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "checkpoint",
                    options=options,
                    default=default,
                    tooltip=(
                        "The distilled checkpoint the branch is read from: "
                        "flux3-image-klein-bf16 or flux3-image-klein-int8-convrot. Both hold "
                        "the same bf16 branch. A -base file holds none."
                    ),
                ), model=True),
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
        """Attach the branch read from ``checkpoint`` to a clone of ``model``.

        Raises:
            PathNotAllowed: A wired path lies outside the diffusion_models folders.
        """
        checkpoint = picker.model_name(checkpoint, (flux3_flow_map.FOLDER,))
        return io.NodeOutput(flux3_flow_map.attach(model, checkpoint))
