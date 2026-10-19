"""A model patch putting a range of transformer blocks back to another checkpoint."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model import block_restore

#: The source checkpoint the menu starts on, where the folder holds one.
BASE = "moss_image_base"


class ModelBlockRestore(io.ComfyNode):
    """Restore a range of a model's transformer blocks from another checkpoint of it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        options = block_restore.checkpoints()
        default = next((name for name in options if BASE in name.replace("\\", "/").lower()), None)
        return io.Schema(
            node_id="WASModelBlockRestore",
            display_name="Model Block Restore",
            search_aliases=[
                "WASModelBlockRestore",
                "Model Block Restore",
                "banding fix",
                "deband",
                "moss image",
                "block merge",
                "block swap",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Put a range of a model's transformer blocks back to another checkpoint of the same "
                "model, such as the Base weights under a fine-tune. Blocks 32 to 41 from MOSS-Image "
                "Base remove the vertical banding MOSS-Image SFT draws across skies and smooth "
                "backgrounds. LoRAs on the model stay applied."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to restore, such as MOSS-Image SFT with its LoRAs; MODEL.",
                ),
                io.Combo.Input(
                    "checkpoint",
                    options=options,
                    default=default,
                    tooltip="The checkpoint the blocks are read from, the same model before "
                    "fine-tuning: moss_image_base_int8_convrot or moss_image_base_bf16 for "
                    "MOSS-Image SFT. Only the chosen blocks are read.",
                ),
                io.Int.Input(
                    "first_block",
                    default=32,
                    min=0,
                    max=1023,
                    tooltip="First block restored, from 0. 32 = MOSS-Image SFT's banding blocks, "
                    "32 to 41; 37 = the top five only, which leaves part of the banding.",
                ),
                io.Int.Input(
                    "last_block",
                    default=41,
                    min=0,
                    max=1023,
                    tooltip="Last block restored. 41 = MOSS-Image's last block.",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Share of the checkpoint's weights. 1 = its blocks exactly; 0.5 = "
                    "halfway; 0 = no change.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with those blocks restored and its LoRAs still applied; MODEL.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, checkpoint, first_block=32, last_block=41,
                strength=1.0) -> io.NodeOutput:
        """Restore the blocks on a clone of ``model``.

        Raises:
            ValueError: The blocks are out of range, or the checkpoint does not match the model.
        """
        if strength <= 0.0:
            return io.NodeOutput(model)
        patched, _ = block_restore.restore(model, checkpoint, int(first_block), int(last_block),
                                           float(strength))
        return io.NodeOutput(patched)
