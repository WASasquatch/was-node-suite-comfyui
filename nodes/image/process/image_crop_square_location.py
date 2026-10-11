"""Crop a square region centred on a point."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.compat.types import CROP_DATA
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil


class ImageCropSquareLocation(io.ComfyNode):
    """Cut a square out of an image, centred on ``(x, y)``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Crop Square Location",
            display_name="Image Crop Square Location",
            search_aliases=["Image Crop Square Location", "crop square", "centre crop"],
            category="WAS Suite/Image/Process",
            description=(
                "Crop a square region centred on a point, and pass on the crop window so "
                "the result can be pasted back later. The size can be rounded to a multiple "
                "a sampler accepts."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to crop. A batch is cropped frame by frame to the same "
                        "square and comes back the same length."
                    ),
                ),
                io.Int.Input(
                    "x",
                    default=0,
                    max=24576,
                    min=0,
                    step=1,
                    tooltip=(
                        "Horizontal centre of the square, in pixels from the left of the image."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    max=24576,
                    min=0,
                    step=1,
                    tooltip=(
                        "Vertical centre of the square, in pixels from the top of the image. "
                        "With x, 0/0 asks for a square centred on the top-left corner, which "
                        "slides down and right until it fits."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=256,
                    max=4096,
                    min=5,
                    step=1,
                    tooltip=(
                        "Length of each side of the square, in pixels. A size larger than the "
                        "image gives the whole image instead."
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    default=8,
                    max=64,
                    min=1,
                    step=1,
                    tooltip=(
                        "Rounds the crop down to a multiple of this on both axes, which "
                        "resamples it. 1 = the exact square, untouched; 8 = most latent "
                        "models; 16, 32 or 64 = models asking for a coarser step."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The cropped square, its width and height rounded down to a multiple "
                        "of divisible_by, with a side shorter than that taken up to one whole "
                        "step instead. At a divisible_by of 1 it is the exact region asked "
                        "for, carried through without resampling."
                    ),
                ),
                CROP_DATA.Output(
                    tooltip=(
                        "The crop window, for Image Paste Crop to put the result back in the "
                        "right place at the right size."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, x=256, y=256, size=512, divisible_by=8) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        cropped = [
            cls.crop_square_location(tensor2pil(plane), x, y, size, divisible_by)
            for plane in image_planes(image)
        ]
        # The square is set by the widgets and the image size, and one batch carries one
        # size, so every image is cropped alike and one crop_data describes them all.
        return io.NodeOutput(dynamic.unfold(stack_images([crop for crop, _ in cropped]), folded), cropped[0][1])

    @classmethod
    def crop_square_location(cls, image, x=256, y=256, size=512, divisible_by=8):
        """Cut one square out of one image.

        Args:
            image: Source image.
            x: Horizontal centre of the square, in pixels from the left.
            y: Vertical centre of the square, in pixels from the top.
            size: Length of each side, in pixels, before any trimming to the image.
            divisible_by: Step each side of the crop is put on a multiple of. 1 rounds
                nothing and returns the region unresampled.

        Returns:
            ``(crop, crop_data)``, where ``crop`` is the square with each side rounded down
            to a multiple of ``divisible_by`` and ``crop_data`` is
            ``(size, (left, top, right, bottom))`` before that rounding.
        """
        img_width, img_height = image.size
        exp_size = size // 2
        left = max(x - exp_size, 0)
        top = max(y - exp_size, 0)
        right = min(x + exp_size, img_width)
        bottom = min(y + exp_size, img_height)

        if right - left < size:
            if right < img_width:
                right = min(right + size - (right - left), img_width)
            elif left > 0:
                left = max(left - (size - (right - left)), 0)
        if bottom - top < size:
            if bottom < img_height:
                bottom = min(bottom + size - (bottom - top), img_height)
            elif top > 0:
                top = max(top - (size - (bottom - top)), 0)

        crop = image.crop((left, top, right, bottom))
        # crop_data holds the rectangle as it was clamped, before the rounding below, so a
        # paste node puts the crop back at the size it was taken from.
        crop_data = (crop.size, (left, top, right, bottom))
        target = (
            cls.rounded(crop.size[0], divisible_by),
            cls.rounded(crop.size[1], divisible_by),
        )
        # A divisible_by of 1 leaves every side already on a multiple, so the exact pixels
        # travel on rather than through a resampler.
        if target != crop.size:
            crop = crop.resize(target)

        return crop, crop_data

    @staticmethod
    def rounded(length, divisible_by):
        """Round one side of a crop down to a multiple of a step.

        Args:
            length: Side length in pixels.
            divisible_by: Step the side is put on a multiple of.

        Returns:
            The largest multiple of ``divisible_by`` that is no longer than ``length``, or
            one whole step for a side shorter than that, since a side of zero pixels is not
            an image.
        """
        return max(divisible_by, (length // divisible_by) * divisible_by)
