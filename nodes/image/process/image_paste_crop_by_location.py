"""Paste an image into a rectangle given by its four edges."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import broadcast_image_planes, pil2tensor, tensor2pil
from ....modules.interface import size_report


class ImagePasteCropByLocation(io.ComfyNode):
    """Paste an image into a rectangle of another, blurring the edge."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Paste Crop by Location",
            display_name="Image Paste Crop by Location",
            search_aliases=[
                "Image Paste Crop by Location",
                "paste at position",
                "composite rectangle",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Paste one image into a rectangle of another, stretching it to fit and "
                "softening the edge. A crop_blending of 1.0 blurs so far that little of the "
                "pasted image stays fully opaque. The fade is always measured from the "
                "rectangle's longer side, so on a long thin rectangle a high value leaves the "
                "whole paste faint rather than sharply cut. Lower it until the paste reads at "
                "the strength wanted."
            ),
            inputs=[
                io.Image.Input("image", tooltip="The image to paste into, which sets the canvas size."),
                io.Image.Input(
                    "crop_image",
                    tooltip=(
                        "The image to paste in. It is stretched to the rectangle, so its own "
                        "aspect ratio is not preserved."
                    ),
                ),
                io.Int.Input(
                    "top",
                    default=0,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip="Distance in pixels from the top of the image to the top of the paste.",
                ),
                io.Int.Input(
                    "left",
                    default=0,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip="Distance in pixels from the left of the image to the left of the paste.",
                ),
                io.Int.Input(
                    "right",
                    default=256,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip=(
                        "Position of the paste's right edge, in pixels from the left of the "
                        "image. left 0 with right 256 pastes into a 256-pixel-wide area. Values "
                        "past the image edge are trimmed to it."
                    ),
                ),
                io.Int.Input(
                    "bottom",
                    default=256,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip=(
                        "Position of the paste's bottom edge, in pixels from the top of the "
                        "image. Values past the image edge are trimmed to it."
                    ),
                ),
                io.Float.Input(
                    "crop_blending",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How soft the edge of the paste is. 0.0 gives a hard rectangle, and "
                        "0.25 blurs the edge over roughly an eighth of the rectangle's longer "
                        "side."
                    ),
                ),
                io.Int.Input(
                    "crop_sharpening",
                    default=0,
                    min=0,
                    max=3,
                    step=1,
                    tooltip=(
                        "How many sharpening passes to run on the pasted image after it is "
                        "stretched, to recover detail lost to the resize. 0 pastes it as it is; "
                        "3 is the strongest and tends to leave halos."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip="The image with the second one composited into the rectangle.",
                ),
                io.Image.Output(
                    display_name="MASK",
                    tooltip=(
                        "The blend mask the paste used, white where the pasted image fully "
                        "covers and fading to black across the softened edge. A rectangle too "
                        "thin to hold the whole fade never reaches white, which is the sign to "
                        "lower crop_blending."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        crop_image,
        top=0,
        left=0,
        right=256,
        bottom=256,
        crop_blending=0.25,
        crop_sharpening=0,
    ) -> io.NodeOutput:
        from ....modules.image import scratch

        folded = dynamic.fold(image)
        if folded.scale != 1.0 and image.shape[-1] == 4:
            # Alpha keeps its own 0 to 1 scale while the colour is folded.
            folded.images[..., 3] = image[..., 3].clamp(0.0, 1.0)
        image = folded.images
        pairs = broadcast_image_planes(image, crop_image)
        pasted = blends = None
        for index, (plane, crop) in enumerate(pairs):
            result_image, result_mask = cls.paste_image(
                tensor2pil(plane),
                tensor2pil(crop),
                top,
                left,
                right,
                bottom,
                crop_blending,
                crop_sharpening,
            )
            if pasted is None:
                advice = "Passing fewer or smaller frames also fits it."
                pasted = scratch.allocate(
                    (len(pairs),) + tuple(result_image.shape[1:]),
                    node="Image Paste Crop by Location",
                    advice=advice,
                )
                blends = scratch.allocate(
                    (len(pairs),) + tuple(result_mask.shape[1:]),
                    node="Image Paste Crop by Location",
                    advice=advice,
                )
            pasted[index, ..., :3] = dynamic.unfold(result_image[0, ..., :3], folded)
            pasted[index, ..., 3:] = result_image[0, ..., 3:]
            blends[index] = result_mask[0]

        # The crop is reported against the rectangle it is resampled into.
        size_report.publish(
            crop_image,
            (right - left, bottom - top),
            action="pasted",
            resampled=True,
            facts={"canvas": size_report.spell(image)},
        )
        return io.NodeOutput(pasted, blends)

    @staticmethod
    def inset_border(image, border_width=20, border_color=(0)):
        """Draw a rectangle of ``border_color`` just inside an image's own edge.

        Args:
            image: Image to draw on. A copy is returned; the input is untouched.
            border_width: Thickness of the drawn rectangle in pixels.
            border_color: Fill for the rectangle, in whatever form the image's mode takes.

        Returns:
            A new image the same size and mode.
        """
        from PIL import Image, ImageDraw

        width, height = image.size
        bordered_image = Image.new(image.mode, (width, height), border_color)
        bordered_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(bordered_image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
        return bordered_image

    @staticmethod
    def feather_inset(crop_size, blend_ratio):
        """Inset width that keeps a feather inside a rectangle without erasing it.

        Args:
            crop_size: ``(width, height)`` of the rectangle being pasted into, in pixels.
            blend_ratio: Blend radius in pixels, half the longer side times the blend amount.

        Returns:
            Inset width in pixels, never negative and never half the shorter side or more.
        """
        return max(0, min(int(blend_ratio / 2), (min(crop_size) - 1) // 2))

    @classmethod
    def paste_image(
        cls,
        image,
        crop_image,
        top=0,
        left=0,
        right=256,
        bottom=256,
        blend_amount=0.25,
        sharpen_amount=1,
    ):
        """Composite one image into a rectangle of another through a blurred mask.

        Args:
            image: The image pasted into.
            crop_image: The image pasted in, stretched to the rectangle.
            top: Top edge of the rectangle, clamped to the image.
            left: Left edge of the rectangle, clamped to the image.
            right: Right edge of the rectangle, clamped to the image.
            bottom: Bottom edge of the rectangle, clamped to the image.
            blend_amount: Softness of the edge, 0.0 to 1.0. Values outside that range are
                clamped into it.
            sharpen_amount: Number of sharpen passes applied to the stretched image.

        Returns:
            ``(composited image tensor, blend mask tensor)``.

        Raises:
            ValueError: The rectangle has no area once clamped, with an edge on or past its
                opposite edge, which leaves nothing to resize the pasted image to.
        """
        from PIL import Image, ImageFilter

        image = image.convert("RGBA")
        crop_image = crop_image.convert("RGBA")

        img_width, img_height = image.size

        top = min(max(top, 0), img_height)
        left = min(max(left, 0), img_width)
        bottom = min(max(bottom, 0), img_height)
        right = min(max(right, 0), img_width)

        crop_size = (right - left, bottom - top)
        if crop_size[0] < 1 or crop_size[1] < 1:
            raise ValueError(
                f"The paste rectangle has no area. Clamped to the {img_width} by "
                f"{img_height} image, left {left} to right {right} is {crop_size[0]} pixels "
                f"wide and top {top} to bottom {bottom} is {crop_size[1]} pixels tall. Set "
                f"right above left and bottom above top, both inside the image."
            )

        crop_img = crop_image.resize(crop_size)
        crop_img = crop_img.convert("RGBA")

        if sharpen_amount > 0:
            for _ in range(sharpen_amount):
                crop_img = crop_img.filter(ImageFilter.SHARPEN)

        if blend_amount > 1.0:
            blend_amount = 1.0
        elif blend_amount < 0.0:
            blend_amount = 0.0
        blend_ratio = (max(crop_size) / 2) * float(blend_amount)

        blend = image.copy()
        mask = Image.new("L", image.size, 0)

        mask_block = Image.new("L", crop_size, 255)
        mask_block = cls.inset_border(mask_block, cls.feather_inset(crop_size, blend_ratio), (0))

        Image.Image.paste(mask, mask_block, (left, top))
        blend.paste(crop_img, (left, top), crop_img)

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.putalpha(mask)
        image = Image.alpha_composite(image, blend)

        return (pil2tensor(image), pil2tensor(mask.convert("RGB")))
