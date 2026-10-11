"""Clip the extremes of a mask to black and white."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report, preview
from ...modules.mask import drawn
from ...modules.mask.regions import threshold_region


class MaskThresholdRegion(io.ComfyNode):
    """Push the darkest and brightest levels of every mask in the batch to the extremes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Threshold Region",
            display_name="Mask Threshold Region",
            search_aliases=["Mask Threshold Region", "threshold", "clip levels"],
            category="WAS Suite/Image/Masking",
            description="Send every level below black_threshold to black and every level above "
            "white_threshold to white. Levels in between are passed through untouched. A mask "
            "painted on the node joins the result in whichever way drawn_combine names.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to clean up, such as a soft or noisy one that should be solid "
                        "at both ends. A batch is handled one mask at a time."
                    ),
                ),
                io.Int.Input(
                    "black_threshold",
                    default=75,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Anything dimmer than this, on a 0-255 brightness scale, is forced to "
                        "pure black. 75 wipes out faint haze; 0 clips nothing at the dark end."
                    ),
                ),
                io.Int.Input(
                    "white_threshold",
                    default=175,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Anything brighter than this, on a 0-255 brightness scale, is forced to "
                        "pure white. 175 makes nearly-white areas solid; 255 clips nothing at the "
                        "bright end. Set it below black_threshold and the mask becomes purely "
                        "black and white, with no grey band left in between."
                    ),
                ),
                io.String.Input(
                    "drawn_mask",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "The mask painted on the node, saved with the workflow. Empty = nothing "
                        "painted, so the node gives the thresholded mask alone. Run the node once "
                        "to get a mask to paint on; clear the field to remove the painting."
                    ),
                ),
                io.Combo.Input(
                    "drawn_combine",
                    options=list(drawn.COMBINE_MODES),
                    default=drawn.DEFAULT_COMBINE,
                    optional=True,
                    tooltip=(
                        "How the painting joins the mask this node produced, resized to it if "
                        "they differ. `union` = the brighter of the two at each pixel; "
                        "`subtract` = the painting taken away; `intersect` = the darker; `off` "
                        "= ignored, not deleted."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The mask with its dark levels crushed to black and its bright levels "
                        "lifted to white. Levels between the two thresholds pass through at their "
                        "original strength, and anything painted on the node is joined in."
                    ),
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        masks,
        black_threshold,
        white_threshold,
        drawn_mask="",
        drawn_combine=drawn.DEFAULT_COMBINE,
    ) -> io.NodeOutput:
        regions = [
            pil2mask(threshold_region(image, black_threshold, white_threshold))
            for image in mask_images(masks)
        ]
        stacked = stack_masks(regions)
        # The mask before the painting joins it, which is what the brush is drawn over. The
        # joined mask would put the painting back into its own backdrop.
        preview.publish_mask_output(stacked)
        painted = drawn.apply(stacked, drawn_mask, drawn_combine)
        mask_report.publish(masks, painted)
        return io.NodeOutput(painted)
