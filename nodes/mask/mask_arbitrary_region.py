"""Subtract the smallest connected region of a mask that clears a size floor."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import arbitrary_region


class MaskArbitraryRegion(io.ComfyNode):
    """Clear one connected region per mask, chosen by area."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Arbitrary Region",
            display_name="Mask Arbitrary Region",
            search_aliases=["Mask Arbitrary Region", "region by size", "select blob"],
            category="WAS Suite/Image/Masking",
            description="Clear the smallest connected region that is still at least size big and "
            "set everything else. size is relative: it is scaled by image area / 10000, so the "
            "same value picks comparable regions at any resolution. When no region reaches it the "
            "mask comes back inverted.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to subtract a region from. A batch is handled one mask at a "
                        "time, each losing its own region."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=256,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Smallest area a region may have to qualify, in ten-thousandths of the "
                        "frame, so it holds at any resolution. 100 = 1% of the image; 256 = "
                        "about 2.6%; 10000 = the whole frame."
                    ),
                ),
                io.Int.Input(
                    "threshold",
                    default=128,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Brightness cut-off, on a 0-255 scale, above which a pixel counts as part "
                        "of a region. 128 splits at mid-grey; lower it to take in faint areas, "
                        "raise it to keep only near-white ones."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "Everything except the qualifying region, as hard black and white: that "
                        "one region is the only area left unset. If no region reached size, the "
                        "input mask comes back inverted instead."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, size, threshold) -> io.NodeOutput:
        regions = [
            pil2mask(arbitrary_region(image, size, threshold)) for image in mask_images(masks)
        ]
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
