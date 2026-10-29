"""Read the trained step times of the PTD file a MiniMax H3 model carries."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model import ptd

NODE_NAME = "H3 PTD Sigmas"


class H3PTDSigmas(io.ComfyNode):
    """The trained step times of the PTD file a MiniMax H3 model carries."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3PTDSigmas",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3PTDSigmas", NODE_NAME,
                "ptd sigmas",
                "ptd schedule",
                "parametric trajectory distillation",
                "minimax h3",
                "extend loop",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Read the trained step times of the PTD file a MiniMax H3 model carries, for "
                "SamplerCustomAdvanced. Wired to H3 Extend Window's model, every segment of an "
                "extend loop samples on its own file's steps, fl2va or Ref2VA."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "A MiniMax H3 model from H3 PTD Patch, or any LoRA, attention or memory "
                        "node after it, as H3 Extend Window's model output."
                    ),
                ),
            ],
            outputs=[
                io.Sigmas.Output(
                    display_name="sigmas",
                    tooltip=(
                        "The file's trained step times: `0.999, 0.947, 0.857, 0.667, 0` for "
                        "fl2va 4 step, `1.0, 0.973, 0.923, 0.8, 0` for Ref2VA 4 step."
                    ),
                ),
                io.Int.Output(
                    display_name="steps",
                    tooltip="Steps the file was trained for: `4`, `6` or `8`.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        """Read the step times off the model.

        Raises:
            ValueError: The model carries no PTD file.
        """
        bank = ptd.bank_of(model)
        if bank is None:
            raise ValueError(
                "H3 PTD Sigmas takes a model that has been through H3 PTD Patch, and this one "
                "carries no PTD file. Wire the model from H3 PTD Patch, or from a node after it")
        facts = ptd.describe(bank)
        run_result.publish(summary=f"{bank.hops} steps: {facts['sigmas']}", facts=facts)
        return io.NodeOutput(ptd.sigmas(bank), bank.hops)
