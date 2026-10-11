"""Crop images to the padded bounding box of a mask."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.interface import size_report
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


class BoundedImageCropWithMask(io.ComfyNode):
    """Crop each image to its mask's bounding box and emit the bounds used."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Bounded Image Crop with Mask",
            display_name="Bounded Image Crop with Mask",
            search_aliases=[
                "Bounded Image Crop with Mask",
                "crop to mask",
                "bounding box",
                "auto crop",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Find the smallest box holding everything the mask marks, grow it by the "
                "padding given, and crop each image to it. The bounds come out alongside the "
                "crop so Bounded Image Blend with Mask can put the result back where it came "
                "from."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to cut down. The box is measured on the mask and then "
                        "sliced straight out of the image, so an image that is not the same "
                        "size as its mask crops in the wrong place."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Marks the area to keep; the box is the tightest rectangle around "
                        "everything non-black. An all-black mask keeps that frame whole, with a "
                        "console note. One mask per image crops each separately; any other "
                        "count uses the first mask for every image."
                    ),
                ),
                io.Int.Input(
                    "padding_left",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Extra pixels kept to the left of the marked area, stopping at the "
                        "edge. 0 crops tight against the mask."
                    ),
                ),
                io.Int.Input(
                    "padding_right",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Extra pixels kept to the right of the marked area.",
                ),
                io.Int.Input(
                    "padding_top",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Extra pixels kept above the marked area.",
                ),
                io.Int.Input(
                    "padding_bottom",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Extra pixels kept below the marked area. Padding gives an inpainting "
                        "pass some of the surroundings to match against."
                    ),
                ),
                io.Boolean.Input(
                    "return_list",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Return the crops as a list of separate images rather than one batch. "
                        "Needed when the masks differ per image, because crops of different "
                        "sizes cannot be stacked into a batch, and only nodes that accept a "
                        "list can read the result."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The cropped regions, as a batch or as a list depending on "
                        "return_list."
                    ),
                ),
                IMAGE_BOUNDS.Output(
                    tooltip=(
                        "One row per cropped image giving the box it was taken from, to wire "
                        "into Bounded Image Blend with Mask so the result goes back in the "
                        "same place."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, mask, padding_left, padding_right, padding_top, padding_bottom,
                return_list=False) -> io.NodeOutput:
        image = image.unsqueeze(0) if image.dim() == 3 else image
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask

        # A mask count that does not match the batch means one mask for every image, so the
        # box is measured once and reused.
        mask_len = 1 if len(image) != len(mask) else len(image)

        cropped_images = []
        all_bounds = []
        for i in range(len(image)):
            if (mask_len == 1 and i == 0) or mask_len > 1:
                rows = torch.where(torch.any(mask[i], dim=1))[0]
                cols = torch.where(torch.any(mask[i], dim=0))[0]
                if len(rows) == 0:
                    # A blank mask has no box. The whole picture stands in, so a detector
                    # returning nothing costs this frame its crop and not the run.
                    logger.warning(
                        "mask %s is blank, so its frame is cropped to the whole image", i
                    )
                    rmin, rmax = 0, mask[i].shape[0] - 1
                    cmin, cmax = 0, mask[i].shape[1] - 1
                else:
                    rmin, rmax = rows[0], rows[-1]
                    cmin, cmax = cols[0], cols[-1]

                    rmin = max(rmin - padding_top, 0)
                    rmax = min(rmax + padding_bottom, mask[i].shape[0] - 1)
                    cmin = max(cmin - padding_left, 0)
                    cmax = min(cmax + padding_right, mask[i].shape[1] - 1)

            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin:rmax + 1, cmin:cmax + 1, :])

        # Every frame is cut to its own mask, so the first crop is the one measured and
        # the row appears only where there are others that can differ from it.
        if cropped_images:
            size_report.publish(
                image,
                cropped_images[0],
                action="cropped",
                facts=(
                    {"frames": f"{len(cropped_images)}, the first measured"}
                    if len(cropped_images) > 1 else None
                ),
            )

        if return_list:
            return io.NodeOutput(cropped_images, all_bounds)

        return io.NodeOutput(torch.stack(cropped_images), all_bounds)
