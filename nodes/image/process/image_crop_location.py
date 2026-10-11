"""Crop a rectangle out of an image by edge position."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.compat.types import CROP_DATA
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.interface import preview


class ImageCropLocation(io.ComfyNode):
    """Cut a rectangle out of an image and record its window."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Crop Location",
            display_name="Image Crop Location",
            search_aliases=["Image Crop Location", "crop rectangle", "cut out region"],
            category="WAS Suite/Image/Process",
            description=(
                "Crop a rectangle given by its four edges, and pass on the crop window so "
                "the result can be pasted back later. The size can be rounded to a multiple "
                "a sampler accepts."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to crop. A batch is cropped frame by frame to the same "
                        "rectangle and comes back the same length."
                    ),
                ),
                io.Int.Input(
                    "top",
                    default=0,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip="Distance in pixels from the top of the image to the top of the crop.",
                ),
                io.Int.Input(
                    "left",
                    default=0,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip="Distance in pixels from the left of the image to the left of the crop.",
                ),
                io.Int.Input(
                    "right",
                    default=256,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip=(
                        "Position of the crop's right edge, in pixels from the left of the "
                        "image. left 0 with right 256 gives a crop 256 pixels wide. A value past "
                        "the image edge is trimmed to it."
                    ),
                ),
                io.Int.Input(
                    "bottom",
                    default=256,
                    max=10000000,
                    min=0,
                    step=1,
                    tooltip=(
                        "Position of the crop's bottom edge, in pixels from the top of the "
                        "image. A value past the image edge is trimmed to it."
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
                        "resamples it. 1 = the exact rectangle, untouched; 8 = most latent "
                        "models; 16, 32 or 64 = models asking for a coarser step."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The cropped region, its width and height rounded down to a multiple "
                        "of divisible_by, with a side shorter than that taken up to one whole "
                        "step instead. At a divisible_by of 1 it is the exact rectangle asked "
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
    def execute(cls, image, top=0, left=0, right=256, bottom=256, divisible_by=8) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        # The four edges are positions in the image published here, which is what an
        # overlay places them on. Publishing changes nothing this returns, and does nothing
        # at all while no browser is connected.
        preview.publish(image)
        # The rectangle follows from the widgets and the image size, and one batch carries
        # one size, so the first crop_data describes every image in it.
        cropped = [
            cls.crop_location(tensor2pil(plane), top, left, right, bottom, divisible_by)
            for plane in image_planes(image)
        ]
        return io.NodeOutput(dynamic.unfold(stack_images([crop for crop, _ in cropped]), folded), cropped[0][1])

    @classmethod
    def crop_location(cls, image, top=0, left=0, right=256, bottom=256, divisible_by=8):
        """Cut one rectangle out of one image.

        Args:
            image: Source image.
            top: Distance from the top of the image to the top of the crop, in pixels.
            left: Distance from the left of the image to the left of the crop, in pixels.
            right: Position of the crop's right edge, in pixels from the left.
            bottom: Position of the crop's bottom edge, in pixels from the top.
            divisible_by: Step each side of the crop is put on a multiple of. 1 rounds
                nothing and returns the rectangle unresampled.

        Returns:
            ``(crop, crop_data)``, where ``crop`` is the rectangle with each side rounded
            down to a multiple of ``divisible_by`` and ``crop_data`` is
            ``(size, (left, top, right, bottom))`` before that rounding.

        Raises:
            ValueError: The clamped rectangle has no width or no height.
        """
        img_width, img_height = image.size

        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(
                "Invalid crop dimensions. Please check the values for top, left, right, and bottom."
            )

        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        # crop_data holds the rectangle as it was clamped, before the rounding below, so a
        # paste node puts the crop back at the size it was taken from.
        crop_data = (crop.size, (crop_left, crop_top, crop_right, crop_bottom))
        size = (
            cls.rounded(crop.size[0], divisible_by),
            cls.rounded(crop.size[1], divisible_by),
        )
        # A divisible_by of 1 leaves every side already on a multiple, so the exact pixels
        # travel on rather than through a resampler.
        if size != crop.size:
            crop = crop.resize(size)

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
