"""Erode the set area of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import morph_masks


class MaskErodeRegion(io.ComfyNode):
    """Shrink the white area of every mask in the batch by binary erosion."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Erode Region",
            display_name="Mask Erode Region",
            search_aliases=["Mask Erode Region", "erode", "shrink mask", "morphology"],
            category="WAS Suite/Image/Masking",
            description="Shrink the set area of a mask by binary erosion. Any non-zero pixel "
            "counts as set, so grey levels are lost.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to shrink. A batch is handled one mask at a time, all by the "
                        "same amount. A set area running off the edge of the frame keeps its "
                        "border there."
                    ),
                ),
                io.Int.Input(
                    "iterations",
                    default=5,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "Passes of shrinkage, each about one pixel on every side. 0 on a wire = "
                        "shrinks until nothing changes, erasing the mask; 5 = about 5 pixels "
                        "narrower, erasing anything thinner than about 10 pixels."
                    ),
                ),
                io.Float.Input(
                    "blur",
                    default=0.0,
                    min=0.0,
                    max=128.0,
                    step=0.5,
                    optional=True,
                    tooltip=(
                        "Soften the edge after the shape is settled, as a blur radius in "
                        "pixels; fractions are dropped. 0 or 0.5 = the hard binary edge; a few "
                        "pixels hide the seam of an inpaint or a composite."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The shrunken mask. Any pixel that was not fully black counted as set, "
                        "so it arrives hard black and white unless blur was above 0."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, iterations, blur=0.0) -> io.NodeOutput:
        return io.NodeOutput(morph_masks(masks, iterations, False, blur))
