"""Brightness, contrast, saturation, sharpness, blur and edge adjustments in one node."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.image import dynamic
from ....modules.interface import preview


def filter_adjustments(img, brightness: float, contrast: float, saturation: float,
                       sharpness: float, blur: int, gaussian_blur: float,
                       edge_enhance: float, detail_enhance: str):
    """Apply the eight adjustments to one image tensor, in widget order.

    Args:
        img: Image tensor scaled to ``[0, 1]``.
        brightness: Amount added to every sample. 0.0 skips the step.
        contrast: Multiplier applied to every sample. 1.0 skips the step.
        saturation: PIL colour enhancement factor. 1.0 skips the step.
        sharpness: PIL sharpness enhancement factor. 1.0 skips the step.
        blur: Number of 3x3 box-blur passes. 0 skips the step.
        gaussian_blur: Gaussian blur radius in pixels. 0.0 skips the step.
        edge_enhance: Blend weight of an edge-enhanced copy, 0.0 to 1.0. 0.0 skips the step.
        detail_enhance: Apply PIL's detail filter.

    Returns:
        ``(adjusted, img)``: the adjusted PIL image, or ``None`` when no step past
        brightness and contrast ran, alongside the tensor those two produced.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    pil_image = None

    if brightness > 0.0 or brightness < 0.0:
        img = np.clip(img + brightness, 0.0, 1.0)

    if contrast > 1.0 or contrast < 1.0:
        img = np.clip(img * contrast, 0.0, 1.0)

    if saturation > 1.0 or saturation < 1.0:
        pil_image = tensor2pil(img)
        pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

    if sharpness > 1.0 or sharpness < 1.0:
        pil_image = pil_image if pil_image is not None else tensor2pil(img)
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

    if blur > 0:
        pil_image = pil_image if pil_image is not None else tensor2pil(img)
        for _ in range(blur):
            pil_image = pil_image.filter(ImageFilter.BLUR)

    if gaussian_blur > 0.0:
        pil_image = pil_image if pil_image is not None else tensor2pil(img)
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

    if edge_enhance > 0.0:
        pil_image = pil_image if pil_image is not None else tensor2pil(img)
        edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        blend_mask = Image.new(mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
        pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)

    if detail_enhance:
        pil_image = pil_image if pil_image is not None else tensor2pil(img)
        pil_image = pil_image.filter(ImageFilter.DETAIL)

    return pil_image, img


class ImageFilterAdjustments(io.ComfyNode):
    """Apply the common tonal and softening adjustments to an image in one pass."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Filter Adjustments",
            display_name="Image Filter Adjustments",
            search_aliases=[
                "Image Filter Adjustments",
                "brightness",
                "contrast",
                "saturation",
                "sharpness",
                "blur",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "The everyday image controls in one node: brightness, contrast, "
                "saturation, sharpness, two kinds of blur, and edge or detail enhancement. "
                "Each one is skipped at its neutral setting."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to adjust. A batch is handled one image at a time.",
                ),
                io.Float.Input(
                    "brightness",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Added to every pixel, where 1.0 is the whole black-to-white range. 0.0 "
                        "leaves the image alone, 0.1 lifts it slightly, -0.25 darkens it "
                        "noticeably."
                    ),
                ),
                io.Float.Input(
                    "contrast",
                    default=1.0,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Every pixel is multiplied by this, so it brightens as it separates. "
                        "1.0 leaves the image alone, 1.3 is a firm push, 0.7 flattens it, and "
                        "negative values invert and clip it to mostly black."
                    ),
                ),
                io.Float.Input(
                    "saturation",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Colour strength. 1.0 leaves the image alone, 0.0 gives black and "
                        "white, 2.0 doubles the colour, 5.0 is poster-like."
                    ),
                ),
                io.Float.Input(
                    "sharpness",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Edge crispness. 1.0 leaves the image alone, 2.0 sharpens, 0.0 softens, "
                        "and negative values overshoot into an embossed outline."
                    ),
                ),
                io.Int.Input(
                    "blur",
                    default=0,
                    min=0,
                    max=16,
                    step=1,
                    tooltip=(
                        "How many passes of a small fixed blur to run. 0 skips it, 1 is a slight "
                        "softening, 16 is heavy. For a specific radius use gaussian_blur instead."
                    ),
                ),
                io.Float.Input(
                    "gaussian_blur",
                    default=0.0,
                    min=0.0,
                    max=1024.0,
                    step=0.1,
                    tooltip=(
                        "Blur radius in pixels. 0.0 skips it, 2 is a gentle soften, 25 removes "
                        "all detail and leaves colour shapes."
                    ),
                ),
                io.Float.Input(
                    "edge_enhance",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of an edge-enhanced copy is mixed in. 0.0 skips it, 0.3 picks "
                        "the outlines out gently, 1.0 uses the enhanced copy outright and looks "
                        "harsh."
                    ),
                ),
                io.Boolean.Input(
                    "detail_enhance",
                    default=False,
                    tooltip=(
                        "Run a fixed detail filter at the end, which is a mild local-contrast "
                        "boost with no strength setting. `off` skips it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The adjusted image, or the input unchanged."),
            ],
        )

    @classmethod
    def execute(cls, image, brightness, contrast, saturation, sharpness, blur, gaussian_blur,
                edge_enhance, detail_enhance) -> io.NodeOutput:
        """Adjust every image of the batch.

        Raises:
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch

        settings = (brightness, contrast, saturation, sharpness, blur, gaussian_blur,
                    edge_enhance, detail_enhance)

        # Publishes the input for a preview panel to apply the settings to.
        preview.publish(image)

        scale = dynamic.peak(image) if dynamic.carries(image) else 1.0
        if len(image) > 1:
            adjusted_batch = None
            for index, plane in enumerate(image):
                folded = dynamic.fold(plane, scale)
                adjusted, img = filter_adjustments(folded.images, *settings)
                frame = pil2tensor(adjusted)[0] if adjusted is not None else img
                if adjusted_batch is None:
                    shape = (len(image),) + tuple(frame.shape)
                    adjusted_batch = (
                        scratch.allocate(
                            shape,
                            frame.dtype,
                            node="Image Filter Adjustments",
                            advice="Passing fewer frames also fits it.",
                        )
                        if frame.device.type == "cpu"
                        else torch.empty(shape, dtype=frame.dtype, device=frame.device)
                    )
                adjusted_batch[index] = frame
                if folded.scale != 1.0:
                    adjusted_batch[index].mul_(folded.scale)
        else:
            folded = dynamic.fold(image, scale)
            adjusted, img = filter_adjustments(folded.images, *settings)
            adjusted_batch = pil2tensor(adjusted) if adjusted is not None else img
            adjusted_batch = dynamic.unfold(adjusted_batch, folded)

        return io.NodeOutput(adjusted_batch)
