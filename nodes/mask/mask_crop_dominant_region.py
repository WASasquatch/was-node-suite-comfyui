"""Crop a mask to its largest connected region."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import crop_dominant_region

#: Level the region search thresholds at. A mask with nothing above it holds no region,
#: which leaves the search choosing from an empty range of labels.
REGION_THRESHOLD = 128


class MaskCropDominantRegion(io.ComfyNode):
    """Crop every mask in the batch around its largest connected region."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Crop Dominant Region",
            display_name="Mask Crop Dominant Region",
            search_aliases=["Mask Crop Dominant Region", "crop largest region"],
            category="WAS Suite/Image/Masking",
            description="Crop the largest connected white region of a mask and centre it on a "
            "square canvas whose side is the longer crop edge plus twice the padding. Grey "
            "levels survive: the region locates the crop and nothing else. A mask with "
            "nothing above mid-grey holds no region and comes back uncropped.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to crop. Its largest connected white area decides where the "
                        "crop is taken; a batch is cropped one mask at a time, each around its "
                        "own region."
                    ),
                ),
                io.Int.Input(
                    "padding",
                    default=24,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Margin left around the region, in pixels on each side. 0 crops tight to "
                        "the region, 24 leaves a 24-pixel border all round and so widens the "
                        "square output by 48 pixels."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The cropped region on a square canvas whose side is the crop's longer "
                        "edge plus twice padding. Where masks in a batch crop to different "
                        "sizes, each is centred on the largest."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, padding) -> io.NodeOutput:
        from PIL import ImageOps

        regions = []
        for image in mask_images(masks):
            if image.getextrema()[1] > REGION_THRESHOLD:
                regions.append(pil2mask(crop_dominant_region(image, padding)))
            else:
                regions.append(pil2mask(ImageOps.invert(image)))
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
