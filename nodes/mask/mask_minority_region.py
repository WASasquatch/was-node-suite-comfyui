"""Subtract the smallest connected region of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report, preview
from ...modules.mask import drawn
from ...modules.mask.regions import minority_region


class MaskMinorityRegion(io.ComfyNode):
    """Clear the smallest connected region of every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Minority Region",
            display_name="Mask Minority Region",
            search_aliases=["Mask Minority Region", "smallest region", "smallest blob"],
            category="WAS Suite/Image/Masking",
            description="Clear the smallest connected region of a mask and set everything else. "
            "The regions are labelled on the mask as it arrives and the result is inverted on the "
            "way out, so the smallest region is subtracted rather than kept. A mask painted on "
            "the node joins the result in whichever way drawn_combine names.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to subtract a region from. A batch is handled one mask at a "
                        "time, each losing its own smallest region."
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
                io.String.Input(
                    "drawn_mask",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "The mask painted on the node, saved with the workflow. Empty = nothing "
                        "painted, so the node gives the region mask alone. Run the node once "
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
                        "Everything except the smallest connected region, as hard black and white "
                        "with no grey levels left: the smallest region is the one area that comes "
                        "back unset. Anything painted on the node is joined in."
                    ),
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        masks,
        threshold,
        drawn_mask="",
        drawn_combine=drawn.DEFAULT_COMBINE,
    ) -> io.NodeOutput:
        regions = [pil2mask(minority_region(image, threshold)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        # The mask before the painting joins it, which is what the brush is drawn over. The
        # joined mask would put the painting back into its own backdrop.
        preview.publish_mask_output(stacked)
        painted = drawn.apply(stacked, drawn_mask, drawn_combine)
        mask_report.publish(masks, painted)
        return io.NodeOutput(painted)
