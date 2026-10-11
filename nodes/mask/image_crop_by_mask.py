"""Crop a batch of images to the bounding box of a mask.

``crop_data`` is ``(size, (left, top, right, bottom))`` with exclusive right and bottom
edges. A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import mask_images, mask_planes, stack_masks
from ...modules.image import dynamic
from ...modules.compat import limits
from ...modules.compat.types import CROP_DATA, IMAGE_BOUNDS
from ...modules.convert.tensors import image_planes, pil2mask, stack_images, tensor2pil
from ...modules.interface import mask_report
from ...modules.log import get_logger

logger = get_logger("nodes.mask")

#: Step both sides of a crop are put on unless divisible_by asks for another, the step most
#: samplers work in. Rounding is down, and never below one whole step.
DEFAULT_DIVISIBLE_BY = 8

#: Largest step divisible_by accepts. It covers the widest stride a diffusion model asks a
#: side to sit on, and a side shorter than the step comes back at one whole step, so a
#: wider bound would resize a small crop up by an arbitrary factor.
MAX_DIVISIBLE_BY = 64

#: Shortest side a crop and its mask are emitted at, whatever the measured rectangle. The
#: tensor conversions squeeze every length-1 axis away, so a crop one pixel on a side would
#: reach the next node as a transposed greyscale image rather than as a picture.
MIN_CROP_SIDE = 2


class ImageCropByMask(io.ComfyNode):
    """Crop every image of a batch to the area its mask marks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCropByMask",
            display_name="Image Crop by Mask",
            search_aliases=[
                "WASImageCropByMask",
                "Image Crop by Mask",
                "crop to mask",
                "bounding box",
                "auto crop",
                "inpaint crop",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Crop each image to the area its mask marks, padded and clamped to the "
                "picture, and pass on the crop window so the result can be pasted back at "
                "full size. Every crop comes out the same size, so the batch travels on as one."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to cut down. The box is measured on the mask and then "
                        "sliced straight out of the image, so a mask that is not the same "
                        "size as its image crops in the wrong place. A batch comes back the "
                        "same length."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Marks the area to keep: the tightest box around everything above "
                        "threshold sets the crop, and a blank mask crops that frame whole. With "
                        "a different count from the images, the first mask is used for all."
                    ),
                ),
                io.Combo.Input(
                    "bbox_mode",
                    options=["union", "per_frame"],
                    tooltip=(
                        "Which box a batch of masks is cropped to. 'union' uses one "
                        "rectangle covering every mask; 'per_frame' gives each image its "
                        "own, following a subject that moves, resized to the widest box's "
                        "width by the tallest box's height."
                    ),
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Extra pixels kept on all four sides of the marked area, trimmed "
                        "where the box would run off the picture. 0 crops tight against the "
                        "mask; 64 leaves an inpainting pass some of the surroundings to "
                        "match against."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How bright a mask pixel must be, from 0.0 to 1.0, to count as "
                        "marked. 0.5 boxes a mask's solid core; 0.0 takes its whole "
                        "feathered edge in; 1.0 marks nothing, so every frame is cropped whole."
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    default=DEFAULT_DIVISIBLE_BY,
                    min=1,
                    max=MAX_DIVISIBLE_BY,
                    step=1,
                    tooltip=(
                        "Rounds both sides of the crop down to a multiple of this and "
                        "resizes to match. 1 = the crop as cut; 8 = suits most models. The "
                        "window and bounds are measured before rounding."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="cropped_image",
                    tooltip=(
                        "The cropped regions, one per input image and all one size, so "
                        "they travel on as a single batch. That size is a multiple of "
                        "divisible_by."
                    ),
                ),
                CROP_DATA.Output(
                    display_name="crop_data",
                    tooltip=(
                        "The crop window, for Image Paste Crop to put the finished region "
                        "back in the right place at its original size. In per_frame mode it "
                        "holds the union rectangle, so a paste stretches every crop across it."
                    ),
                ),
                IMAGE_BOUNDS.Output(
                    display_name="bounds",
                    tooltip=(
                        "One row per image giving the rectangle it was cropped from, as "
                        "top, bottom, left and right pixel positions, to wire into Draw "
                        "Image Bounds."
                    ),
                ),
                io.Mask.Output(
                    display_name="cropped_mask",
                    tooltip=(
                        "The mask cut to the same rectangle as the crop and carried at the "
                        "crop's own size, so an inpainting pass knows which pixels were "
                        "marked."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        mask,
        bbox_mode="union",
        padding=0,
        threshold=0.5,
        divisible_by=DEFAULT_DIVISIBLE_BY,
    ) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        from PIL import ImageOps

        images = image_planes(image)
        planes = mask_planes(mask)
        regions = mask_images(mask)
        if not images:
            raise ValueError(
                "At least one image must be provided. The batch reaching this node is empty, "
                "so there is nothing for the mask to be measured against."
            )

        # A mask count that does not match the batch means one mask for every image, so the
        # box is measured on the first and reused.
        mask_len = 1 if len(images) != len(planes) else len(images)
        sources = [index if mask_len > 1 else 0 for index in range(len(images))]

        # A frame whose mask marks nothing has no box. Its whole picture stands in, so a
        # detector that returns a blank mask costs that frame its crop rather than costing
        # the run its result.
        whole = [(0, 0, plane.shape[1], plane.shape[0]) for plane in images]
        boxes = [
            cls.mask_box(planes[source], plane.shape[1], plane.shape[0], threshold, padding, source)
            for plane, source in zip(images, sources)
        ]

        # The union is measured over the frames that marked something. An unmarked frame
        # contributes nothing to it, so one blank mask in a batch cannot widen the window
        # every other frame is cropped to. With nothing marked anywhere, the window is the
        # whole picture.
        measured = [box for box in boxes if box is not None]
        if measured:
            union = (
                min(box[0] for box in measured),
                min(box[1] for box in measured),
                max(box[2] for box in measured),
                max(box[3] for box in measured),
            )
        else:
            union = whole[0]
        crop_data = ((union[2] - union[0], union[3] - union[1]), union)

        if bbox_mode == "per_frame":
            boxes = [box if box is not None else full for box, full in zip(boxes, whole)]
            size = (
                max(box[2] - box[0] for box in boxes),
                max(box[3] - box[1] for box in boxes),
            )
        else:
            boxes = [union for _ in boxes]
            size = crop_data[0]
        size = (cls.rounded(size[0], divisible_by), cls.rounded(size[1], divisible_by))
        # A crop one pixel on a side is squeezed to a transposed greyscale image by the
        # tensor conversions, so both sides are widened to a shape that survives them.
        # crop_data holds the measured rectangle, so a paste scales the widening back out.
        size = (max(size[0], MIN_CROP_SIDE), max(size[1], MIN_CROP_SIDE))

        crops = []
        cropped_masks = []
        # A crop already at the size is carried on rather than resized, so a divisible_by of
        # 1 in union mode reaches the output with the pixels it was cut with.
        for plane, source, box in zip(images, sources, boxes):
            crop = tensor2pil(plane).crop(box)
            region = regions[source].crop(box)
            crops.append(crop if crop.size == size else crop.resize(size))
            region = region if region.size == size else region.resize(size)
            cropped_masks.append(pil2mask(ImageOps.invert(region)))

        bounds = [(box[1], box[3] - 1, box[0], box[2] - 1) for box in boxes]
        cropped = stack_masks(cropped_masks)
        mask_report.publish(mask, cropped, source="mask")

        return io.NodeOutput(
            dynamic.unfold(stack_images(crops), folded), crop_data, bounds, cropped
        )

    @classmethod
    def mask_box(cls, mask, width, height, threshold, padding, index):
        """Measure one mask's bounding box, padded and clamped to the image.

        Args:
            mask: One ``(height, width)`` mask plane holding values in ``[0, 1]``.
            width: Width of the image the box is cropped out of, in pixels. Every edge is
                clamped to it, since the rectangle is sliced out of the image.
            height: Height of the image the box is cropped out of, in pixels.
            threshold: Level a sample must exceed to count as marked.
            padding: Pixels added to every side before the box is clamped.
            index: Position of the mask in its batch, for the error message.

        Returns:
            ``(left, top, right, bottom)``, right and bottom exclusive, with every edge
            inside the image and both sides at least one pixel. ``None`` when the mask marks
            nothing inside the image, which the caller reads as no box for that frame.
        """
        limit_x = int(width)
        limit_y = int(height)

        marked = mask > threshold
        rows = torch.where(torch.any(marked, dim=1))[0]
        cols = torch.where(torch.any(marked, dim=0))[0]
        if len(rows) == 0:
            logger.warning(
                "mask %s marks nothing above threshold %s, so its frame is cropped to the "
                "whole image", index, threshold,
            )
            return None
        # A mask larger than its image can mark an area that is entirely off the picture,
        # which the clamps below would otherwise collapse into a one-pixel corner.
        if int(cols[0]) >= limit_x or int(rows[0]) >= limit_y:
            logger.warning(
                "mask %s is %sx%s against a %sx%s image and everything it marks lies past the "
                "right or bottom edge, so its frame is cropped to the whole image",
                index, int(mask.shape[-1]), int(mask.shape[0]), limit_x, limit_y,
            )
            return None

        # Both ends of every edge are clamped. A padded edge running past the picture would
        # put a negative index in the bounds row, which reads from the far edge of the image
        # where a bounds node slices with it, and would shift where crop_data pastes back.
        left = min(max(int(cols[0]) - padding, 0), limit_x - 1)
        top = min(max(int(rows[0]) - padding, 0), limit_y - 1)
        right = max(min(int(cols[-1]) + 1 + padding, limit_x), left + 1)
        bottom = max(min(int(rows[-1]) + 1 + padding, limit_y), top + 1)

        return left, top, right, bottom

    @staticmethod
    def rounded(length, step):
        """Round one side of a crop down to a multiple of a step.

        Args:
            length: Side length in pixels.
            step: Step the side is put on, from 1 to :data:`MAX_DIVISIBLE_BY`. A step of 1
                returns ``length`` unchanged, so nothing is rounded and nothing is resized.

        Returns:
            The largest multiple of ``step`` that is no longer than ``length``, or one whole
            step for a side shorter than that, since a side of zero pixels is not an image.
        """
        return max(step, (length // step) * step)
