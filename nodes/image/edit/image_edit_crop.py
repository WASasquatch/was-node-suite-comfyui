"""Cut the regions to edit out of a picture, sized for an edit model."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import EDIT_CROP, EDIT_JOBS
from ....modules.image import boxes as box_values
from ....modules.image import edit_crop, edit_regions
from ....modules.interface import preview, run_result

NODE = "Image Edit Crop"

logger = log.get_logger("nodes.image.edit")


class ImageEditCrop(io.ComfyNode):
    """Cut the regions an edit changes out of a picture, one crop per group of nearby regions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageEditCrop",
            display_name="Image Edit Crop",
            search_aliases=[
                "WASImageEditCrop",
                "Image Edit Crop",
                "crop and stitch",
                "inpaint crop",
                "regional edit",
                "edit region",
                "wevisone",
            ],
            category="WAS Suite/Image/Edit",
            description=(
                "Cut the areas to edit out of a picture, such as WeVisOne-VL Detect's masks or the edits "
                "picked on Image Edit Regions, into crops sized for an edit model like MiniMax H3 Image. "
                "Nearby areas share a crop and are edited together; Image Edit Restore puts each edit "
                "back through its own area alone."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The picture to edit. A batch is read from its first picture.",
                ),
                io.Mask.Input(
                    "masks", optional=True,
                    tooltip=(
                        "The areas to edit, such as WeVisOne-VL Detect's masks. Every separate area on "
                        "every mask is one region. Unwired or empty = the bboxes are used."
                    ),
                ),
                io.BoundingBox.Input(
                    "bboxes", optional=True, force_input=True,
                    tooltip=(
                        "Boxes to edit where no mask marks anything, such as a detector's boxes. "
                        "Each box is one region, edited whole."
                    ),
                ),
                io.Float.Input(
                    "context", default=0.3, min=0.0, max=3.0, step=0.05,
                    tooltip=(
                        "Room the model sees around the regions, as a share of their longer side: "
                        "0 = tight, 0.3 = some surroundings, 1 = a wide view."
                    ),
                ),
                io.Combo.Input(
                    "grouping", options=list(edit_crop.GROUPINGS), default="auto",
                    tooltip=(
                        "`auto` = nearby regions share a crop while it stays at full resolution, far ones "
                        "get their own; `one crop` = every region in one; `each region` = one crop each."
                    ),
                ),
                io.Int.Input(
                    "max_resolution", default=2048, min=256, max=8192, step=32,
                    tooltip=(
                        "Longest side a crop is handed to the model at: 1024 = older edit models such as "
                        "FLUX.1 Kontext; 2048 = MiniMax H3 Image; 4096 = models that take larger pictures. "
                        "A larger crop is shrunk to it."
                    ),
                ),
                io.Int.Input(
                    "min_resolution", default=1024, min=64, max=8192, step=8, advanced=True,
                    tooltip=(
                        "Shortest side a small crop is enlarged toward, never past max_resolution or 4 times "
                        "its size: 1024 = MiniMax H3 Image; 512 = older models."
                    ),
                ),
                io.Int.Input(
                    "multiple", default=32, min=1, max=256, step=1, advanced=True,
                    tooltip="Step both sides of a crop are held to: 32 = MiniMax H3 Image, 16 = most other models.",
                ),
                EDIT_JOBS.Input(
                    "jobs", optional=True,
                    tooltip=(
                        "Edits picked on Image Edit Regions, each with its own mask and prompt. Wired, they "
                        "replace masks and bboxes, and each crop gets a prompt naming every edit inside it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="crops", is_output_list=True,
                    tooltip="One crop per group of regions, for the edit model's picture input.",
                ),
                io.Mask.Output(
                    display_name="crop_masks", is_output_list=True,
                    tooltip="The regions inside each crop, at the crop's size, for a model that takes a mask.",
                ),
                EDIT_CROP.Output(
                    display_name="edits", is_output_list=True,
                    tooltip="Where each crop came from and what it holds, for Image Edit Restore.",
                ),
                io.Image.Output(
                    display_name="preview",
                    tooltip="The picture with each crop's window in blue and each region's box in amber.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many crops the regions made.",
                ),
                io.String.Output(
                    display_name="prompts", is_output_list=True,
                    tooltip=(
                        "One prompt per crop naming every job inside it, for the edit model's prompt. "
                        "Empty without jobs."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, image, context, grouping, max_resolution, min_resolution, multiple, masks=None, bboxes=None, jobs=None,
    ) -> io.NodeOutput:
        """Find the regions, join them into crops and cut the crops.

        Args:
            image: The picture, ``(batch, height, width, channels)``.
            context: Room around the regions, as a share of their longer side.
            grouping: One of ``GROUPINGS``.
            max_resolution: Longest side a crop is handed over at.
            min_resolution: Shortest side a small crop is enlarged toward.
            multiple: Step both sides of a crop are held to.
            masks: Mattes of the areas to edit, or ``None``.
            bboxes: Boxes used where no mask marks anything, or ``None``.
            jobs: Picked edits with their masks and prompts, or ``None``.

        Returns:
            The crops, their masks and records as lists, the preview, the count and one prompt per crop.

        Raises:
            ValueError: Nothing marks an area to edit.
        """
        import torch
        import torch.nn.functional as F

        if grouping not in edit_crop.GROUPINGS:
            raise ValueError(f"{NODE} does not know the grouping `{str(grouping)[:40]}`. Pick one from the menu.")
        picture = image[0]
        height, width = picture.shape[:2]
        planes = None
        if masks is not None:
            planes = masks if masks.dim() == 3 else masks.reshape(-1, *masks.shape[-2:])
            if tuple(planes.shape[-2:]) != (height, width):
                planes = F.interpolate(planes[:, None].float(), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
        boxes = box_values.normalise(bboxes) if bboxes is not None else []
        picked = list(jobs.jobs) if jobs is not None else []
        if picked:
            planes = torch.stack([job.mask.float() for job in picked])
            if tuple(planes.shape[-2:]) != (height, width):
                planes = F.interpolate(planes[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
            boxes = []
        crops = edit_crop.plan(
            planes, boxes, width, height, float(context), grouping, int(min_resolution), int(max_resolution), int(multiple),
        )
        if not crops:
            raise ValueError(
                f"{NODE} found nothing to edit: the masks mark no area and no box was wired. Check that "
                "the detector found what its query names, or wire a mask painted over the area."
            )
        prompts = []
        for crop in crops:
            if not picked:
                prompts.append("")
                continue
            for region in crop.regions:
                region.whole = picked[region.plane].whole
            inside = sorted({region.plane for region in crop.regions})
            prompts.append(edit_regions.compose([picked[plane].prompt for plane in inside], jobs.reference))
        pictures = [edit_crop.crop_picture(picture, crop)[None] for crop in crops]
        crop_masks = []
        for crop in crops:
            union = torch.stack([region.matte for region in crop.regions]).amax(dim=0)
            union = F.interpolate(union[None, None], size=(crop.size[1], crop.size[0]), mode="bilinear", align_corners=False)
            crop_masks.append(union[0].clamp(0, 1))
        shown = edit_crop.outline(picture, crops)[None]
        preview.publish_output(shown, slot="image")
        regions = sum(len(crop.regions) for crop in crops)
        logger.info("%s cut %d crop(s) holding %d region(s)", NODE, len(crops), regions)
        try:
            run_result.publish(
                status=run_result.OK,
                summary=f"{len(crops)} crop(s), {regions} region(s)",
                counts={"crops": len(crops), "regions": regions},
                facts={
                    f"crop {crop.index + 1}": (
                        f"{crop.window[2] - crop.window[0]}x{crop.window[3] - crop.window[1]} at "
                        f"({crop.window[0]}, {crop.window[1]}) -> {crop.size[0]}x{crop.size[1]}, "
                        f"{len(crop.regions)} region(s), scale {crop.scale:.2f}"
                    )
                    for crop in crops
                },
            )
        except Exception as error:
            logger.debug("%s published no report (%s)", NODE, error)
        return io.NodeOutput(pictures, crop_masks, crops, shown, len(crops), prompts)
