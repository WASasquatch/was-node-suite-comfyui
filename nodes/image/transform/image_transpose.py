"""Composite one image onto another with a transform and a feathered border."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import broadcast_image_planes, pil2tensor, tensor2pil
from ....modules.interface import size_report


def rotated_silhouette(source_size, canvas_size, target_size, rotate):
    """Corners of an element's own rectangle inside the canvas a rotation expands it into.

    Args:
        source_size: ``(width, height)`` of the element before rotation.
        canvas_size: ``(width, height)`` of the canvas ``Image.rotate(expand=True)``
            returns, which holds the turned rectangle and the empty corners around it.
        target_size: ``(width, height)`` that canvas is scaled to after the rotation.
        rotate: Rotation in degrees counter-clockwise, as ``Image.rotate`` reads it.

    Returns:
        Four ``(x, y)`` corners in the scaled canvas, starting at the turned top left and
        following the rectangle's edges. Pixel centres sit on integer coordinates, so the
        far edges land one pixel past the last column and the last row.
    """
    turn = math.radians(rotate % 360)
    # Image.rotate rounds its own matrix to fifteen places, which puts a quarter turn
    # exactly on the axis rather than a float epsilon away from it.
    cosine = round(math.cos(turn), 15)
    sine = round(math.sin(turn), 15)
    source_width, source_height = source_size
    canvas_width, canvas_height = canvas_size
    corners = []
    for offset_x, offset_y in (
        (-source_width / 2, -source_height / 2),
        (source_width / 2, -source_height / 2),
        (source_width / 2, source_height / 2),
        (-source_width / 2, source_height / 2),
    ):
        turned_x = canvas_width / 2 + offset_x * cosine + offset_y * sine
        turned_y = canvas_height / 2 + offset_y * cosine - offset_x * sine
        corners.append(
            (
                turned_x * target_size[0] / canvas_width,
                turned_y * target_size[1] / canvas_height,
            )
        )
    return corners


def feather_mask(size, corners, feathering):
    """Ramp that fades an element's alpha inward from each of its own edges.

    Args:
        size: ``(width, height)`` of the mask, which is the element's size after rotation
            and scaling.
        corners: Four ``(x, y)`` corners of the element's rectangle inside that size, as
            :func:`rotated_silhouette` returns them, so the fade follows a turned edge
            instead of the box around it.
        feathering: Width of the fade in pixels. A pixel on an edge keeps
            ``1 / feathering`` of its alpha and one ``feathering - 1`` pixels in keeps all
            of it.

    Returns:
        A PIL image in mode ``L``, 255 across the interior, ramping down towards each edge
        and 0 outside the rectangle.
    """
    import numpy as np
    from PIL import Image

    columns = np.arange(size[0], dtype=np.float64).reshape(1, size[0])
    rows = np.arange(size[1], dtype=np.float64).reshape(size[1], 1)
    inward = np.full((size[1], size[0]), np.inf)
    for (start_x, start_y), (end_x, end_y) in zip(corners, corners[1:] + corners[:1]):
        run_x, run_y = end_x - start_x, end_y - start_y
        # Distance from the edge's line, positive on the interior side, with the corners
        # running in one direction around the rectangle.
        distance = (run_x * (rows - start_y) - run_y * (columns - start_x)) / math.hypot(
            run_x, run_y
        )
        np.minimum(inward, distance, out=inward)
    ramp = np.clip(255.0 * (inward + 1.0) / feathering, 0, 255).astype(np.uint8)
    return Image.fromarray(ramp, mode='L')


def apply_transpose_image(image_bg, image_element, size, loc, rotate=0, feathering=0):
    """Rotate, scale and paste one image onto another.

    Args:
        image_bg: Background image. Sets the size of the result.
        image_element: Image pasted on top. It carries an alpha channel through the
            rotation, so the corners the rotation adds around it stay transparent.
        size: ``(width, height)`` the element is scaled to, after rotation. Both must be
            positive.
        loc: ``(x, y)`` of the element's top left corner in the background, in pixels.
            Negative values and values past the edge place it partly outside.
        rotate: Rotation in degrees counter-clockwise, applied before scaling. The element
            grows to hold the rotated result, so the scale that follows sees the larger
            box.
        feathering: Width in pixels of the fade at the element's own edges, which follows
            them around a rotation. 0 pastes a hard edge.

    Returns:
        The background in mode ``RGBA``, with the element composited onto it.

    Raises:
        ValueError: ``size`` holds a value below 1, which scales the element to nothing.
    """
    from PIL import Image

    if size[0] < 1 or size[1] < 1:
        raise ValueError(
            "The overlay is scaled to width by height, so both need to be 1 pixel or "
            f"more, and this is {size[0]} by {size[1]}."
        )

    element = image_element.convert('RGBA')
    rotated = element.rotate(rotate, expand=True)
    scaled = rotated.resize(size)

    if feathering > 0:
        corners = rotated_silhouette(element.size, rotated.size, size, rotate)
        ramp = feather_mask(size, corners, feathering)
        # Fading the channel towards nothing through a composite rounds its product with
        # the ramp, where a channel multiply truncates that product.
        clear = Image.new('L', size, 0)
        scaled.putalpha(Image.composite(scaled.getchannel('A'), clear, ramp))

    new_image = Image.new('RGBA', image_bg.size, (0, 0, 0, 0))
    new_image.paste(scaled, loc)

    image_bg = image_bg.convert('RGBA')
    image_bg.paste(new_image, (0, 0), new_image)

    return image_bg


class ImageTranspose(io.ComfyNode):
    """Place a second image over the first at a given size, position and rotation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Transpose",
            display_name="Image Transpose",
            search_aliases=["Image Transpose", "overlay", "paste image", "composite"],
            category="WAS Suite/Image/Transform",
            description=(
                "Scale, rotate and paste image_overlay onto image, with an optional soft "
                "edge. The result is the size of image and carries an alpha channel."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The background. The result is this image's size, whatever the "
                        "overlay's."
                    ),
                ),
                io.Image.Input(
                    "image_overlay",
                    tooltip="The image placed on top of the background.",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=1,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Width the overlay is scaled to, in pixels, ignoring its own aspect "
                        "ratio. Must be above 0."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=1,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Height the overlay is scaled to, in pixels, ignoring its own aspect "
                        "ratio. Must be above 0."
                    ),
                ),
                io.Int.Input(
                    "X",
                    default=0,
                    min=-48000,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Horizontal position of the overlay's left edge, in pixels from the "
                        "left of the background. 0 is flush left; negative values push it "
                        "off the left edge."
                    ),
                ),
                io.Int.Input(
                    "Y",
                    default=0,
                    min=-48000,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Vertical position of the overlay's top edge, in pixels down from "
                        "the top of the background. 0 is flush with the top; negative values "
                        "push it off the top edge."
                    ),
                ),
                io.Int.Input(
                    "rotation",
                    default=0,
                    min=-360,
                    max=360,
                    step=1,
                    tooltip=(
                        "How far to turn the overlay before scaling, in degrees "
                        "counter-clockwise. Negative values turn it clockwise. The overlay "
                        "is squeezed back into width by height afterwards, so a rotated "
                        "overlay comes out narrower than an unrotated one."
                    ),
                ),
                io.Int.Input(
                    "feathering",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Width of the fade around the overlay's edge, in pixels. 0 gives a "
                        "hard edge; 32 fades the outer 32 pixels to transparent, which "
                        "blends the overlay into the background."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The background with the overlay composited onto it, in RGBA. Nodes "
                        "that need three channels want Images to RGB after this."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, image_overlay, width, height, X, Y, rotation,
                feathering) -> io.NodeOutput:
        """Composite the overlay onto every background frame.

        Raises:
            ValueError: An input holds no images, or width or height is below 1.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch

        scale = dynamic.peak(image) if dynamic.carries(image) else 1.0
        pairs = broadcast_image_planes(image, image_overlay)
        composited = None
        for index, (background, overlay) in enumerate(pairs):
            folded = dynamic.fold(background, scale)
            frame = pil2tensor(
                apply_transpose_image(
                    tensor2pil(folded.images),
                    tensor2pil(overlay),
                    (width, height),
                    (X, Y),
                    rotation,
                    feathering,
                )
            )
            if composited is None:
                composited = scratch.allocate(
                    (len(pairs),) + tuple(frame.shape[1:]),
                    node="Image Transpose",
                    advice="Passing fewer frames or a smaller image also fits it.",
                )
            composited[index] = frame[0]
            if folded.scale != 1.0:
                composited[index].mul_(folded.scale)

        # The canvas keeps the background's size, so the pair worth reporting is the
        # overlay against the box the widgets scale it into.
        size_report.publish(
            image_overlay,
            (width, height),
            action="placed",
            resampled=True,
            facts={"canvas": size_report.spell(image)},
        )
        return io.NodeOutput(composited)
