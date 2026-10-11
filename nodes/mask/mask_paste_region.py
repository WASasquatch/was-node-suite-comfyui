"""Paste a cropped mask back into the mask it came from."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import mask_images, mask_planes, stack_masks
from ...modules.compat.types import CROP_DATA
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report


class MaskPasteRegion(io.ComfyNode):
    """Composite a cropped mask back onto the full-size mask through a feathered edge."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Paste Region",
            display_name="Mask Paste Region",
            search_aliases=["Mask Paste Region", "paste mask", "uncrop"],
            category="WAS Suite/Image/Masking",
            description="Paste crop_mask back into mask at the window crop_data records. The "
            "seam is feathered by a linear gradient on every edge that is not against the "
            "image border. A batch is pasted one mask at a time, and the shorter of the two "
            "batches repeats its last mask.",
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The full-size mask to paste into, normally the same one the crop was "
                        "taken from. A batch is pasted into one mask at a time."
                    ),
                ),
                io.Mask.Input(
                    "crop_mask",
                    tooltip=(
                        "The cropped mask to paste back. It is resized to the size recorded in "
                        "crop_data first, so it may return at a different resolution than it left "
                        "at. A single crop is pasted into every mask of a batch."
                    ),
                ),
                CROP_DATA.Input(
                    "crop_data",
                    tooltip=(
                        "The crop window from Mask Crop Region, which says where in mask the crop "
                        "belongs. Nodes such as Image Crop Face emit False when they found "
                        "nothing to crop, and that yields a black mask the size of mask."
                    ),
                ),
                io.Float.Input(
                    "crop_blending",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Width of the soft band along each pasted edge, as a fraction of the "
                        "crop. 0.0 = a hard seam; 0.25 = the outer quarter of each edge; 1.0 = "
                        "right across the crop. Edges flush with the image border are never "
                        "feathered."
                    ),
                ),
                io.Int.Input(
                    "crop_sharpening",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    tooltip=(
                        "How many sharpening passes to run on the crop before pasting it, to undo "
                        "softness introduced by resizing. 0 pastes it as it is; 3 is the maximum "
                        "and tends to leave visible edge artefacts."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="RESULT_MASK",
                    tooltip="The full-size mask with the crop composited back into it.",
                ),
                io.Mask.Output(
                    display_name="CROP_MASK",
                    tooltip=(
                        "The blend mask the paste used, full size and black outside the window. "
                        "It shows where the crop landed and how far the seam was faded, which is "
                        "useful for checking crop_blending."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, mask, crop_mask, crop_data, crop_blending, crop_sharpening) -> io.NodeOutput:
        if crop_data is False:
            blank = stack_masks([
                torch.zeros(plane.shape[:2], dtype=torch.float32) for plane in mask_planes(mask)
            ])
            mask_report.publish(crop_mask, blank, source="crop_mask")
            return io.NodeOutput(blank, blank)

        images = mask_images(mask)
        crops = mask_images(crop_mask)
        pasted = []
        blends = []
        for index in range(max(len(images), len(crops))):
            result_mask, result_crop_mask = cls.paste_image(
                images[min(index, len(images) - 1)],
                crops[min(index, len(crops) - 1)],
                crop_data,
                crop_blending,
                crop_sharpening,
            )
            pasted.append(pil2mask(result_mask))
            blends.append(pil2mask(result_crop_mask))

        composited = stack_masks(pasted)
        mask_report.publish(crop_mask, composited, source="crop_mask")
        return io.NodeOutput(composited, stack_masks(blends))

    @staticmethod
    def lingrad(size, direction: str, white_ratio: float):
        """Build a linear black-to-white ramp.

        Args:
            size: ``(width, height)`` of the ramp.
            direction: ``"vertical"`` or ``"horizontal"``.
            white_ratio: Fraction of the axis the ramp occupies. The remainder, measured
                from the low edge, stays solid black.

        Returns:
            A PIL image in mode ``L``.
        """
        from PIL import Image, ImageDraw

        image = Image.new("RGB", size)
        draw = ImageDraw.Draw(image)
        if direction == "vertical":
            black_end = int(size[1] * (1 - white_ratio))
            for y in range(0, size[1]):
                if y <= black_end:
                    color = (0, 0, 0)
                else:
                    color_value = int(((y - black_end) / (size[1] - black_end)) * 255)
                    color = (color_value, color_value, color_value)
                draw.line([(0, y), (size[0], y)], fill=color)
        elif direction == "horizontal":
            black_end = int(size[0] * (1 - white_ratio))
            for x in range(0, size[0]):
                if x <= black_end:
                    color = (0, 0, 0)
                else:
                    color_value = int(((x - black_end) / (size[0] - black_end)) * 255)
                    color = (color_value, color_value, color_value)
                draw.line([(x, 0), (x, size[1])], fill=color)

        return image.convert("L")

    @classmethod
    def paste_image(cls, image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1):
        """Composite a crop back onto its source through a feathered edge.

        Args:
            image: Full-size mask, mode ``L``.
            crop_image: Cropped mask, resized to the size recorded in ``crop_data``.
            crop_data: ``(size, (left, top, right, bottom))`` from a crop node.
            blend_amount: Width of the feathered band on each edge, as a fraction of the
                crop.
            sharpen_amount: Number of sharpen passes applied to the crop first.

        Returns:
            ``(pasted mask, blend mask)``, both inverted and in mode ``L``.
        """
        from PIL import Image, ImageChops, ImageFilter, ImageOps

        crop_size, (left, top, right, bottom) = crop_data
        crop_image = crop_image.resize(crop_size)

        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image = crop_image.filter(ImageFilter.SHARPEN)

        blended_image = Image.new("RGBA", image.size, (0, 0, 0, 255))
        blended_mask = Image.new("L", image.size, 0)
        crop_padded = Image.new("RGBA", image.size, (0, 0, 0, 0))
        blended_image.paste(image, (0, 0))
        crop_padded.paste(crop_image, (left, top))
        crop_mask = Image.new("L", crop_image.size, 0)

        if top > 0:
            gradient_image = ImageOps.flip(cls.lingrad(crop_image.size, "vertical", blend_amount))
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if left > 0:
            gradient_image = ImageOps.mirror(
                cls.lingrad(crop_image.size, "horizontal", blend_amount)
            )
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if right < image.width:
            gradient_image = cls.lingrad(crop_image.size, "horizontal", blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        if bottom < image.height:
            gradient_image = cls.lingrad(crop_image.size, "vertical", blend_amount)
            crop_mask = ImageChops.screen(crop_mask, gradient_image)

        crop_mask = ImageOps.invert(crop_mask)
        blended_mask.paste(crop_mask, (left, top))
        blended_mask = blended_mask.convert("L")
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (
            ImageOps.invert(blended_image.convert("RGB")).convert("L"),
            ImageOps.invert(blended_mask.convert("RGB")).convert("L"),
        )
