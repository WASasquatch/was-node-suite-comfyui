"""Round off the edges of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import smooth_region


class MaskSmoothRegion(io.ComfyNode):
    """Blur and re-threshold every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Smooth Region",
            display_name="Mask Smooth Region",
            search_aliases=["Mask Smooth Region", "smooth", "round edges"],
            category="WAS Suite/Image/Masking",
            description="Blur a mask and re-threshold it at half the blurred maximum, which "
            "rounds off its edges and leaves no intermediate grey levels.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to tidy up, such as one with jagged or stair-stepped edges. A "
                        "batch is handled one mask at a time."
                    ),
                ),
                io.Float.Input(
                    "sigma",
                    default=5.0,
                    min=0.0,
                    max=128.0,
                    step=0.1,
                    tooltip=(
                        "How far the mask is blurred, in pixels, before it is turned back into "
                        "hard black and white. 0 = edges hardened only; 20 or more = visibly "
                        "reshapes the mask. Larger values round corners and swallow small specks."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The rounded-off mask, hard black and white with no grey levels left. It "
                        "is split at half of the blurred mask's own brightest level, so a faint "
                        "mask is smoothed as readily as a solid white one."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, sigma) -> io.NodeOutput:
        regions = [pil2mask(smooth_region(image, sigma)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
