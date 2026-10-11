"""Subtract the largest connected region of a mask's unset area."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report, preview
from ...modules.mask import drawn
from ...modules.mask.regions import dominant_region


class MaskDominantRegion(io.ComfyNode):
    """Clear the largest connected unset region of every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Dominant Region",
            display_name="Mask Dominant Region",
            search_aliases=["Mask Dominant Region", "largest region", "biggest blob"],
            category="WAS Suite/Image/Masking",
            description="Clear the largest connected region of a mask's unset area and set "
            "everything else, so smaller unset pieces are filled. The result is hard black and "
            "white, and a mask painted on the node joins it.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to subtract a region from. A batch is handled one mask at a "
                        "time, each losing its own largest unset region."
                    ),
                ),
                io.Int.Input(
                    "threshold",
                    default=128,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Brightness cut-off, 0 to 255: a pixel joins the search when its level "
                        "is below 255 minus this. 128 = everything dimmer than 127; higher takes "
                        "in less of the unset area."
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
                        "Everything outside the largest connected unset region is set, as hard "
                        "black and white: an unset area in one piece comes back as it was, a "
                        "broken one keeps only its largest piece. Anything painted on the node "
                        "is joined in."
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
        regions = [pil2mask(dominant_region(image, threshold)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        # The mask before the painting joins it, which is what the brush is drawn over. The
        # joined mask would put the painting back into its own backdrop.
        preview.publish_mask_output(stacked)
        painted = drawn.apply(stacked, drawn_mask, drawn_combine)
        mask_report.publish(masks, painted)
        return io.NodeOutput(painted)
