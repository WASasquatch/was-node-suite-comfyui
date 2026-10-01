"""A MiniMax H3 model set up to sample long clips in less VRAM."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model import h3_low_vram


class H3LowVRAM(io.ComfyNode):
    """Run a MiniMax H3 model's blocks in token slices with its resident weights budgeted."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3LowVRAM",
            display_name="H3 Low VRAM",
            search_aliases=[
                "WASH3LowVRAM",
                "H3 Low VRAM",
                "low vram",
                "streamed blocks",
                "minimax h3",
                "out of memory",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Set a MiniMax H3 model up to sample long clips in less VRAM, with the same "
                "result. Each block runs over the clip in slices of 8192 tokens, and only the "
                "weights that fit beside the clip stay on the card while the rest stream in as "
                "each block runs. Place it after the model loaders and any LoRA so every H3 "
                "sampler in the graph uses it."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model for every H3 sampler in the graph.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        """Wrap the model, leaving one that already carries the setup as it is."""
        if h3_low_vram.is_low_vram(model):
            return io.NodeOutput(model)
        return io.NodeOutput(h3_low_vram.low_vram_model(model))
