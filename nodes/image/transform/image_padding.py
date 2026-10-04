"""Pad an image with transparency and emit a matching outpainting mask."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules import log
from ....modules.convert.tensors import image_planes, pil2tensor, tensor2pil
from ....modules.interface import size_report

logger = log.get_logger("nodes.image.transform")

#: Mask level for each alpha level of the padded canvas: 255 where it is fully transparent.
UNCOVERED = [int(255 * (1 - alpha / 255)) if alpha != 0 else 255 for alpha in range(256)]


def _faded(size, masks):
    """Build the alpha a run of opaque pastes through ``masks`` leaves behind.

    Args:
        size: ``(width, height)`` of the plane.
        masks: Mode ``L`` masks, applied in the order given.

    Returns:
        A mode ``L`` image.
    """
    from PIL import Image

    opaque = Image.new('L', size, 255)
    faded = Image.new('L', size, 0)
    for mask in masks:
        faded = Image.composite(opaque, faded, mask)
    return faded


def apply_image_padding(image, left_pad=100, right_pad=100, top_pad=100, bottom_pad=100,
                        feather_radius=50, second_pass=True):
    """Fade an image's edges to transparent and place it on a larger transparent canvas.

    Args:
        image: Source image.
        left_pad: Transparent margin added on the left, in pixels.
        right_pad: Transparent margin added on the right, in pixels.
        top_pad: Transparent margin added above, in pixels.
        bottom_pad: Transparent margin added below, in pixels.
        feather_radius: Half-width in pixels of the band the edges fade over. The band
            drawn is twice this and is then blurred by the same radius, so the fade
            reaches further in than the number suggests. Four times this value covers an
            image entirely, in which case nothing is pasted and the warning below is
            logged.
        second_pass: Run a second, quarter-radius fade over the first and paste each
            through twice, which pulls the fade back towards the edge and leaves the
            middle of the image at full opacity.

    Returns:
        ``(padded, mask)``. ``padded`` is the faded image on the larger canvas in mode
        ``RGBA``; ``mask`` is the same size in mode ``RGB``, black where the image is
        opaque and white where it is fully transparent.
    """
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new('L', image.size, 255)
    draw = ImageDraw.Draw(mask)

    draw.rectangle((0, 0, feather_radius * 2, image.height), fill=0)
    draw.rectangle((image.width - feather_radius * 2, 0, image.width, image.height), fill=0)
    draw.rectangle((0, 0, image.width, feather_radius * 2), fill=0)
    draw.rectangle((0, image.height - feather_radius * 2, image.width, image.height), fill=0)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    if second_pass:

        mask2 = Image.new('L', image.size, 255)
        draw2 = ImageDraw.Draw(mask2)

        feather_radius2 = int(feather_radius / 4)
        draw2.rectangle((0, 0, feather_radius2 * 2, image.height), fill=0)
        draw2.rectangle((image.width - feather_radius2 * 2, 0, image.width, image.height), fill=0)
        draw2.rectangle((0, 0, image.width, feather_radius2 * 2), fill=0)
        draw2.rectangle(
            (0, image.height - feather_radius2 * 2, image.width, image.height), fill=0
        )

        mask2 = mask2.filter(ImageFilter.GaussianBlur(radius=feather_radius2))

        # Each mask is applied twice, which carries the band back towards opaque.
        faded = _faded(image.size, (mask, mask, mask2, mask2))

    else:

        faded = _faded(image.size, (mask,))

    feathered_im = image.convert('RGBA')
    feathered_im.putalpha(faded)

    new_size = (
        feathered_im.width + left_pad + right_pad,
        feathered_im.height + top_pad + bottom_pad,
    )

    new_im = Image.new('RGBA', new_size, (0, 0, 0, 0))
    new_im.paste(feathered_im, (left_pad, top_pad))

    # Warns when the fade has left nothing of the image on the canvas.
    if new_im.getbbox() is None:
        logger.warning(
            "feathering of %s pixels fades in over %s pixels from every edge, which covers "
            "the whole %sx%s image: the padded canvas is empty and the mask asks for all of "
            "it to be filled. Feathering below a quarter of the shorter side, %s here, "
            "leaves some of the image standing.",
            feather_radius, feather_radius * 2, image.width, image.height,
            min(image.width, image.height) // 4,
        )

    padding_mask = new_im.getchannel('A').point(UNCOVERED)

    return (new_im, padding_mask.convert('RGB'))


class ImagePadding(io.ComfyNode):
    """Pad an image out to a larger canvas for outpainting."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Padding",
            display_name="Image Padding",
            search_aliases=["Image Padding", "outpaint", "expand canvas", "border"],
            category="WAS Suite/Image/Transform",
            description=(
                "Put an image on a larger empty canvas and fade its edges out, then return "
                "the canvas and a mask of everything that was added. Feed both to an "
                "inpainting pass to fill the new space. A larger feathering gives that fill "
                "more of a run-up and eats further into the original: 120 fades most of a 512 "
                "pixel image and wipes out anything smaller than about 480 pixels altogether, "
                "so keep it under a quarter of the image's shorter side."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to place on the larger canvas.",
                ),
                io.Int.Input(
                    "feathering",
                    default=120,
                    min=0,
                    max=2048,
                    step=1,
                    tooltip=(
                        "How far the image's own edges fade out, in pixels. 0 leaves a hard "
                        "edge, and the fade reaches roughly three times this far in."
                    ),
                ),
                io.Boolean.Input(
                    "feather_second_pass",
                    default=True,
                    tooltip=(
                        "`on` runs a second, narrower fade that restores the middle of "
                        "the image to full opacity and keeps the softening near the edge. "
                        "`off` applies the wide fade alone, which leaves the whole image "
                        "noticeably lighter."
                    ),
                ),
                io.Int.Input(
                    "left_padding",
                    default=512,
                    min=8,
                    max=48000,
                    step=1,
                    tooltip="Empty space added on the left, in pixels.",
                ),
                io.Int.Input(
                    "right_padding",
                    default=512,
                    min=8,
                    max=48000,
                    step=1,
                    tooltip="Empty space added on the right, in pixels.",
                ),
                io.Int.Input(
                    "top_padding",
                    default=512,
                    min=8,
                    max=48000,
                    step=1,
                    tooltip="Empty space added above, in pixels.",
                ),
                io.Int.Input(
                    "bottom_padding",
                    default=512,
                    min=8,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Empty space added below, in pixels. The four margins are "
                        "independent, so 8 on three sides and 512 on one extends the image "
                        "in a single direction."
                    ),
                ),
                io.Int.Input(
                    "target_width",
                    default=0,
                    min=0,
                    max=48000,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Pad out to this width instead of using the four side amounts, which "
                        "is what outpainting wants. 0 leaves the sides in charge. The picture "
                        "is centred and a target narrower than the picture pads nothing."
                    ),
                ),
                io.Int.Input(
                    "target_height",
                    default=0,
                    min=0,
                    max=48000,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Pad out to this height instead of using the four side amounts. 0 "
                        "leaves the sides in charge."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip=(
                        "The padded canvas, with the faded image on it and transparency "
                        "everywhere else."
                    ),
                ),
                io.Image.Output(
                    display_name="MASK",
                    tooltip=(
                        "The area to fill, as an image: white where the canvas is empty, "
                        "black where the image is solid, grey across the fade. Convert it "
                        "with Image to Mask to wire it into an inpainting node."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, feathering, feather_second_pass, left_padding, right_padding,
                top_padding, bottom_padding, target_width=0, target_height=0) -> io.NodeOutput:
        """Pad every frame and assemble the canvases and masks one frame at a time.

        Raises:
            ValueError: The batch holds no frame.
            MemoryError: Neither free memory nor a scratch drive can hold the canvases.
        """
        from ....modules.image import scratch

        scale = dynamic.peak(image) if dynamic.carries(image) else 1.0
        planes = image_planes(image)
        if not planes:
            raise ValueError(
                "Image Padding was given an empty batch. Connect an image with at least one "
                "frame."
            )
        if target_width or target_height:
            width, height = tensor2pil(planes[0]).size
            # Centred, and nothing added on an axis the target is smaller than.
            spare_x = max(0, int(target_width) - width)
            spare_y = max(0, int(target_height) - height)
            left_padding = spare_x // 2
            right_padding = spare_x - left_padding
            top_padding = spare_y // 2
            bottom_padding = spare_y - top_padding

        advice = "Smaller paddings, a smaller target size or fewer frames also fit it."
        canvases = masks = None
        for index, plane in enumerate(planes):
            folded = dynamic.fold(plane, scale)
            canvas, mask = apply_image_padding(
                tensor2pil(folded.images),
                left_padding,
                right_padding,
                top_padding,
                bottom_padding,
                feathering,
                second_pass=feather_second_pass,
            )
            canvas, mask = pil2tensor(canvas)[0], pil2tensor(mask)[0]
            if canvases is None:
                canvases = scratch.allocate(
                    (len(planes),) + tuple(canvas.shape), node="Image Padding", advice=advice
                )
                masks = scratch.allocate(
                    (len(planes),) + tuple(mask.shape), node="Image Padding", advice=advice
                )
            canvases[index].copy_(canvas)
            if folded.scale != 1.0:
                canvases[index].mul_(folded.scale)
            masks[index].copy_(mask)

        size_report.publish(image, canvases, action="padded")
        return io.NodeOutput(canvases, masks)
