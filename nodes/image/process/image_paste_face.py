"""Paste a cropped face back into the picture it came from."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import CROP_DATA
from ....modules.interface import preview, size_report
from ....modules.convert.tensors import (
    broadcast_image_planes,
    image_planes,
    pil2tensor,
    tensor2pil,
)

logger = log.get_logger("nodes.image.process")


class ImagePasteFace(io.ComfyNode):
    """Composite a face crop back onto its source through a feathered seam."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Paste Face",
            display_name="Image Paste Face",
            search_aliases=["Image Paste Face", "paste face", "face uncrop", "restore face"],
            category="WAS Suite/Image/Process",
            description=(
                "Paste a face crop back into the picture Image Crop Face took it from, "
                "with a soft seam around it."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The full-size image to paste into, normally the same one the face was "
                        "cropped from."
                    ),
                ),
                io.Image.Input(
                    "crop_image",
                    tooltip=(
                        "The reworked face to paste back. It is resized to the size recorded in "
                        "crop_data, so it may be scaled up or down on the way in."
                    ),
                ),
                CROP_DATA.Input(
                    "crop_data",
                    tooltip=(
                        "The crop window from Image Crop Face, which says where the face "
                        "belongs. That node passes False here when it found no face, and this "
                        "returns the image untouched with a black mask."
                    ),
                ),
                io.Float.Input(
                    "crop_blending",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of each pasted edge is faded out, as a fraction of the crop. "
                        "0.0 gives a hard visible seam, 0.25 fades the outer quarter of each "
                        "edge, and 1.0 fades right across the crop so only its centre is fully "
                        "opaque."
                    ),
                ),
                io.Int.Input(
                    "crop_sharpening",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    tooltip=(
                        "How many sharpening passes to run on the face before pasting, to "
                        "recover detail lost to resizing. 0 pastes it as it is; 3 is the "
                        "strongest and tends to leave halos around the eyes and hair."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip="The full-size image with the face composited back into it.",
                ),
                io.Image.Output(
                    display_name="MASK_IMAGE",
                    tooltip=(
                        "The blend mask the paste used, full size and black outside the window. "
                        "White is where the face fully replaced the image, so it shows how far "
                        "the seam was faded."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, image, crop_image, crop_data=None, crop_blending=0.25, crop_sharpening=0
    ) -> io.NodeOutput:
        from ....modules.image import scratch

        advice = "Passing fewer or smaller frames also fits it."
        if crop_data is False:
            logger.error("no valid crop data found!")
            planes = image_planes(image)
            width, height = tensor2pil(planes[0]).size
            blank = scratch.allocate(
                (len(planes), height, width, 3), node="Image Paste Face", advice=advice
            )
            return io.NodeOutput(image, blank.zero_())

        # A socket holding a single image cycles, contributing it to every result.
        pairs = broadcast_image_planes(image, crop_image)
        answered = blends = None
        for index, (plane, crop) in enumerate(pairs):
            result_image, result_mask = cls.paste_image(
                tensor2pil(plane), tensor2pil(crop), crop_data, crop_blending, crop_sharpening
            )
            if answered is None:
                answered = scratch.allocate(
                    (len(pairs),) + tuple(result_image.shape[1:]),
                    node="Image Paste Face",
                    advice=advice,
                )
                blends = scratch.allocate(
                    (len(pairs),) + tuple(result_mask.shape[1:]),
                    node="Image Paste Face",
                    advice=advice,
                )
            answered[index] = result_image[0]
            blends[index] = result_mask[0]

        # The crop is reported against the window crop_data recorded.
        size_report.publish(
            crop_image,
            crop_data[0],
            action="pasted",
            resampled=True,
            facts={"canvas": size_report.spell(image)},
        )
        preview.publish_output(answered)
        return io.NodeOutput(answered, blends)

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
        """Composite a crop back onto its source through a feathered seam.

        Args:
            image: The full-size image.
            crop_image: The crop, resized to the size recorded in ``crop_data``.
            crop_data: ``(size, (left, top, right, bottom))`` from Image Crop Face.
            blend_amount: Fraction of each edge that is faded out.
            sharpen_amount: Number of sharpen passes applied to the crop first.

        Returns:
            ``(pasted image tensor, blend mask tensor)``, both ``RGB``.
        """
        from PIL import Image, ImageChops, ImageFilter, ImageOps

        # The crop nodes write the window as (left, top, right, bottom); the first two are
        # bound as top, left, which makes the paste position (top, left) the (x, y) PIL takes.
        crop_size, (top, left, right, bottom) = crop_data
        crop_image = crop_image.resize(crop_size)

        if sharpen_amount > 0:
            for _ in range(int(sharpen_amount)):
                crop_image = crop_image.filter(ImageFilter.SHARPEN)

        blended_image = Image.new("RGBA", image.size, (0, 0, 0, 255))
        blended_mask = Image.new("L", image.size, 0)
        crop_padded = Image.new("RGBA", image.size, (0, 0, 0, 0))
        blended_image.paste(image, (0, 0))
        crop_padded.paste(crop_image, (top, left))
        crop_mask = Image.new("L", crop_image.size, 0)

        # These two edge tests are paired the other way round, so the top edge is feathered
        # on the strength of the horizontal coordinate and the left edge on the vertical
        # one. It shows only when the window is flush against exactly one of those borders.
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
        blended_mask.paste(crop_mask, (top, left))
        blended_mask = blended_mask.convert("L")
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (
            pil2tensor(blended_image.convert("RGB")),
            pil2tensor(blended_mask.convert("RGB")),
        )
