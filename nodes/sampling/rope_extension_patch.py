"""A model patch scaling RoPE on every call larger than the model's native side."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.sampling import rope_extension

logger = log.get_logger("nodes.sampling.rope_extension_patch")


class RoPEExtensionPatch(io.ComfyNode):
    """Keep a model's RoPE positions in range on frames longer than its native side."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASRoPEExtensionPatch",
            display_name="RoPE Extension Patch",
            search_aliases=[
                "WASRoPEExtensionPatch",
                "RoPE Extension Patch",
                "rope scale",
                "ntk",
                "high resolution",
                "moss image",
                "scale rope",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Scale a model's RoPE on every call longer than its native side, so a large frame "
                "stays in the range of positions it learned. MOSS-Image widens its RoPE "
                "frequencies and FLUX compresses its positions. Tiles at native size from Tiled "
                "Diffusion Model Patch are left unscaled, on either side of it."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="A MOSS-Image or FLUX 1 or 2 model; MODEL.",
                ),
                io.Int.Input(
                    "native_side",
                    default=rope_extension.DEFAULT_SIDE,
                    min=256,
                    max=16384,
                    step=16,
                    tooltip="Longest side in pixels sampled unscaled. 2160 = MOSS-Image's widest "
                    "2 MP size, 21:9; a side past it is scaled, as 0.94 at 2304.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with RoPE scaled on large calls, for any sampler; MODEL.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, native_side) -> io.NodeOutput:
        """Wrap a clone of ``model`` so large calls scale their RoPE.

        Raises:
            ValueError: The model's transformer reads no RoPE scale.
        """
        patched, kind = rope_extension.patch(model, int(native_side))
        logger.info("RoPE Extension Patch: sides past %d px scale the RoPE %s", native_side, kind)
        return io.NodeOutput(patched)
