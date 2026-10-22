"""Apply a PTD step-adapter file to MiniMax H3."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.model import ptd

logger = log.get_logger("nodes.h3_ptd_patch")

NODE_NAME = "H3 PTD Patch"


class H3PTDPatch(io.ComfyNode):
    """Give MiniMax H3 a PTD file's per-step adapters and heads."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3PTDPatch",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3PTDPatch", NODE_NAME,
                "ptd",
                "parametric trajectory distillation",
                "few step",
                "4 step",
                "8 step",
                "minimax h3",
                "ref2va",
                "fl2va",
                "model patch",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make MiniMax H3 generate video and sound in 4 to 8 steps from a PTD file loaded "
                "with Load Model Patch, on the Ref2VA or fl2va transformer the file was trained "
                "on. Sample with euler, scheduler simple, cfg 1 and the steps shown, or with this "
                "node's sigmas. LoRAs still apply."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "A plain MiniMax H3 transformer of the kind the file was trained on, "
                        "ref2va or fl2va, pruned and quantized ones included, after any LoRA."
                    ),
                ),
                io.ModelPatch.Input(
                    "model_patch",
                    tooltip=(
                        "A PTD file from Load Model Patch, kept in ComfyUI/models/model_patches "
                        "or ComfyUI/models/ptd: bf16 or int8, Ref2VA or fl2va."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with the file's step adapters, for a sampler at the steps below.",
                ),
                io.Sigmas.Output(
                    display_name="sigmas",
                    tooltip=(
                        "The trained step times, for SamplerCustomAdvanced: `1.0, 0.988, ...` "
                        "for Ref2VA, `0.999, 0.947, 0.857, 0.667, 0` for fl2va 4 step. A sampler "
                        "given any other sigmas stops before its first step."
                    ),
                ),
                io.Int.Output(
                    display_name="steps",
                    tooltip="Steps the file was trained for: `4`, `6` or `8`, for KSampler's steps.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, model_patch) -> io.NodeOutput:
        """Patch the model.

        Raises:
            ValueError: The patch is not a PTD file, or the model cannot take it.
        """
        bank = getattr(model_patch, "model", None)
        if not isinstance(bank, ptd.Bank):
            raise ValueError(
                "H3 PTD Patch takes a PTD file from Load Model Patch. The model patch connected "
                "is another kind; pick a PTD file in Load Model Patch.")
        patched, notes = ptd.apply(model, bank)
        facts = ptd.describe(bank)
        facts["base"] = "; ".join(notes)
        run_result.publish(
            summary=f"{bank.hops} steps: KSampler euler, simple, cfg 1, or these sigmas",
            facts=facts,
        )
        logger.info("%s applied %s, %d steps", NODE_NAME, bank.name, bank.hops)
        return io.NodeOutput(patched, ptd.sigmas(bank), bank.hops)
