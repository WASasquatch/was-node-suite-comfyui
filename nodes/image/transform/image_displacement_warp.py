"""Warp images by a displacement map."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import pil2tensor, tensor2pil


def resize_and_crop(image, target_size):
    """Scale an image to cover a target size and centre-crop it to exactly that size.

    Args:
        image: Source image.
        target_size: ``(width, height)`` in pixels.

    Returns:
        An image of exactly ``target_size``.
    """
    width, height = image.size
    target_width, target_height = target_size
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    image = image.resize((new_width, new_height))
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))


class ImageDisplacementWarp(io.ComfyNode):
    """Push each pixel of an image along the brightness of a displacement map."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Displacement Warp",
            display_name="Image Displacement Warp",
            search_aliases=["Image Displacement Warp", "displace", "distort", "warp"],
            category="WAS Suite/Image/Transform",
            description=(
                "Bend an image by a second, greyscale image. Bright areas of the map pull "
                "pixels diagonally down and right, dark areas leave them where they are, "
                "which turns any texture into a ripple, smear or melt. Each pixel is read "
                "one at a time, so a large image takes a while. A displacement map is scaled "
                "to cover the image and centre-cropped, and where the map batch is shorter "
                "than the image batch the last map is reused for the rest. Smooth maps such "
                "as clouds or a blurred gradient give flowing results, and sharp ones tear "
                "the image."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to bend. A batch is warped one image at a time and comes "
                        "back the same length, and every image keeps the width and height it "
                        "went in at."
                    ),
                ),
                io.Image.Input(
                    "displacement_maps",
                    tooltip=(
                        "The map that says how far to push each pixel, read as brightness: "
                        "black does not move, white moves by the full amplitude. Any size is "
                        "accepted."
                    ),
                ),
                io.Float.Input(
                    "amplitude",
                    default=25.0,
                    min=-4096,
                    max=4096,
                    step=0.1,
                    tooltip=(
                        "How far a fully white area of the map moves a pixel, in pixels, on "
                        "both axes at once. 25 gives a gentle ripple, 200 a heavy smear, 0 "
                        "leaves the image alone, and a negative value pushes up and left. "
                        "Values past the image's own width or height read outside it and "
                        "raise an error."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The warped images, in the order they arrived, as RGB.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, displacement_maps, amplitude) -> io.NodeOutput:
        """Warp every image of the batch by its displacement map.

        Raises:
            ValueError: The batch holds no images.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        scale = dynamic.peak(images) if dynamic.carries(images) else 1.0
        import comfy.utils

        from ....modules.image import scratch
        from ....modules.image.warp import displace_image

        progress = comfy.utils.ProgressBar(len(images))

        displaced = None
        for i in range(len(images)):
            folded = dynamic.fold(images[i], scale)
            img = tensor2pil(folded.images)
            if i < len(displacement_maps):
                disp = tensor2pil(displacement_maps[i])
            else:
                disp = tensor2pil(displacement_maps[-1])
            disp = resize_and_crop(disp, img.size)
            frame = pil2tensor(displace_image(img, disp, amplitude))
            if displaced is None:
                displaced = scratch.allocate(
                    (len(images),) + tuple(frame.shape[1:]),
                    node="Image Displacement Warp",
                    advice="Passing fewer frames also fits it.",
                )
            displaced[i] = frame[0]
            if folded.scale != 1.0:
                displaced[i].mul_(folded.scale)
            progress.update(1)

        if displaced is None:
            raise ValueError(
                "Image Displacement Warp was given no images. Connect an image or a batch of "
                "them."
            )
        return io.NodeOutput(displaced)
