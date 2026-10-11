"""Two Three.js materials blended through a mask."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL
from ...modules.threejs import layers
from ...modules.threejs.spec import require_spec

REQUIRES = "threejs"


class ThreeMaterialMix(io.ComfyNode):
    """Blend one material into another through a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeMaterialMix",
            display_name="Three Material Mix",
            search_aliases=[
                "WASThreeMaterialMix",
                "Three Material Mix",
                "layer material",
                "mask material",
                "blend material",
            ],
            category="WAS Suite/Three",
            description=(
                "Paint one material over another through a mask, the way a texturing tool "
                "stacks layers: black keeps the base, white shows the top, grey mixes them. "
                "Every channel follows the same mask, and the result is a normal material, so "
                "feeding it back in as the base stacks another layer."
            ),
            inputs=[
                THREE_MATERIAL.Input(
                    "base",
                    tooltip="The material underneath, showing wherever the mask is black.",
                ),
                THREE_MATERIAL.Input(
                    "top",
                    tooltip="The material painted over it, showing wherever the mask is white.",
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Where the top material shows, in UV space. White is all top, black "
                        "all base, 0.5 an even mix."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the top layer comes through overall. 1.0 uses the mask "
                        "as it is, 0.35 fades the whole layer back."
                    ),
                ),
                io.Boolean.Input(
                    "invert_mask",
                    default=False,
                    tooltip="`true` swaps which side the mask shows, so black becomes the top material.",
                ),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The blended surface, for Three Mesh or as the base of another mix.",
                ),
            ],
        )

    @classmethod
    def execute(cls, base, top, mask, opacity, invert_mask) -> io.NodeOutput:
        """Blend the two materials.

        Raises:
            ValueError: An input is not a material descriptor.
        """
        import numpy as np

        require_spec(base, "material")
        require_spec(top, "material")

        weights = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        while weights.ndim > 2:
            weights = weights[0]
        weights = np.clip(weights.astype(np.float32), 0.0, 1.0)
        if invert_mask:
            weights = 1.0 - weights
        weights = weights * float(opacity)

        height, width = weights.shape[:2]
        return io.NodeOutput(layers.mixed(base, top, weights, (width, height)))
