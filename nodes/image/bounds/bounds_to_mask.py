"""Paint image bounds as white rectangles on a black mask.

A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.compat.sockets import require_input
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import image_planes, mask_planes
from ....modules.image import bounds
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


def frame_size(image, mask) -> tuple[int, int] | None:
    """Read a frame size off whichever of an image and a mask is connected.

    Args:
        image: ``IMAGE`` tensor, or None.
        mask: ``MASK`` tensor, or None.

    Returns:
        ``(width, height)`` in pixels, or None when neither carries a frame.

    Raises:
        ValueError: Both are connected and they cover different areas.
    """
    sizes = {}
    if image is not None:
        planes = image_planes(image)
        if planes:
            sizes["image"] = (int(planes[0].shape[1]), int(planes[0].shape[0]))
    if mask is not None:
        planes = mask_planes(mask)
        if planes:
            sizes["mask"] = (int(planes[0].shape[1]), int(planes[0].shape[0]))

    if len(set(sizes.values())) > 1:
        listed = ", ".join(f"{name} is {w}x{h}" for name, (w, h) in sizes.items())
        raise ValueError(
            f"Bounds to Mask was handed two frames of different sizes: {listed}. The bounds "
            f"were measured on one picture, so connect that one and disconnect the other."
        )
    return next(iter(sizes.values()), None)


class BoundsToMask(io.ComfyNode):
    """Paint each row of an ``IMAGE_BOUNDS`` value as a filled rectangle on a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundsToMask",
            display_name="Bounds to Mask",
            search_aliases=[
                "WASBoundsToMask",
                "Bounds to Mask",
                "bounds to mask",
                "rectangle mask",
                "region mask",
                "box mask",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Paint every bounds rectangle as a white block on a black mask, which is "
                "how a measured region reaches the masking, compositing and inpainting "
                "nodes. The frame comes from the picture connected to image or mask, so "
                "the rectangles land on the pixels the bounds name. With neither "
                "connected, width and height give the frame instead."
            ),
            inputs=[
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "The rectangles to paint, from Image Bounds, Inset Image Bounds, "
                        "Mask to Bounds or Image Crop by Mask. Each row becomes its own "
                        "mask, so a bounds holding a row per image answers a mask per image."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width of the mask in pixels, read only when neither image nor mask "
                        "is connected. 0 = take it from the connected picture; 1872 = an "
                        "1872px frame. Use the width the bounds were measured on."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Height of the mask in pixels, read only when neither image nor "
                        "mask is connected. 0 = take it from the connected picture; 2272 = "
                        "a 2272px frame."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "The picture the bounds were measured on. Its size becomes the mask "
                        "size and width and height are then ignored. Wire the same image "
                        "that fed Image Bounds or Image Crop by Mask."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "A mask to take the frame from instead, for bounds measured by Mask "
                        "to Bounds. Read when image is empty, and it has to cover the same "
                        "area as the picture the bounds were measured on."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    tooltip=(
                        "One mask per bounds row, white inside the rectangle and black "
                        "around it, ready for a masked composite or as an inpainting region."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image_bounds, width=0, height=0, image=None, mask=None) -> io.NodeOutput:
        """Paint each bounds row onto its own mask.

        Args:
            image_bounds: Rows of ``(rmin, rmax, cmin, cmax)``, or one bare row.
            width: Width of every mask in pixels, read when no picture is connected.
            height: Height of every mask in pixels, read when no picture is connected.
            image: Picture whose size every mask takes.
            mask: Mask whose size every mask takes, read when image is empty.

        Returns:
            A ``(rows, height, width)`` mask holding 1.0 inside each rectangle.

        Raises:
            ValueError: Nothing is connected to image_bounds, the value holds no rows, no
                frame is available, or a row is inside out or outside the frame.
        """
        require_input(
            image_bounds,
            "Bounds to Mask",
            "image_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Mask to Bounds",
            "IMAGE_BOUNDS",
        )

        rows = bounds.rows(image_bounds)
        if not rows:
            raise ValueError(
                "Bounds to Mask was given a bounds value holding no rectangles. Check the "
                "node feeding image_bounds: a region search that matched nothing produces "
                "this."
            )

        width, height = int(width), int(height)
        size = frame_size(image, mask)
        if size is None:
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Bounds to Mask has no frame to paint into. Nothing is connected to "
                    f"its image or mask input, and width is {width} with height {height}, "
                    f"so there is no size to use. Connect the picture the bounds were "
                    f"measured on to image, or the mask they were measured on to mask, or "
                    f"type that picture's size into width and height, such as 1872 by 2272."
                )
            size = (width, height)
        elif width > 0 and height > 0 and (width, height) != size:
            logger.info(
                "the connected picture is %sx%s, so the mask is built at that size rather "
                "than the %sx%s width and height hold", size[0], size[1], width, height,
            )

        frame_width, frame_height = size
        painted = torch.zeros((len(rows), frame_height, frame_width), dtype=torch.float32)
        for index, (rmin, rmax, cmin, cmax) in enumerate(rows):
            if rmin > rmax or cmin > cmax:
                raise ValueError(
                    f"Bounds row {index} is inside out: it reads (rmin={rmin}, rmax={rmax}, "
                    f"cmin={cmin}, cmax={cmax}), and rmin must not exceed rmax, nor cmin "
                    f"exceed cmax."
                )

            # Clamped rather than sliced as given: a negative index counts from the far
            # edge of the mask and would paint the rectangle on the wrong side of it.
            top, bottom = max(rmin, 0), min(rmax, frame_height - 1)
            left, right = max(cmin, 0), min(cmax, frame_width - 1)
            if top > bottom or left > right:
                raise ValueError(
                    f"Bounds row {index} reads (rmin={rmin}, rmax={rmax}, cmin={cmin}, "
                    f"cmax={cmax}), which is wholly outside the {frame_width}x"
                    f"{frame_height} frame, so there is nothing to paint. The bounds belong "
                    f"to a different picture: connect the one they were measured on to "
                    f"image or mask, or set width and height to its size."
                )
            if (top, bottom, left, right) != (rmin, rmax, cmin, cmax):
                logger.warning(
                    "bounds row %s reads (%s, %s, %s, %s) and was trimmed to (%s, %s, %s, "
                    "%s) to fit a %sx%s frame", index, rmin, rmax, cmin, cmax, top, bottom,
                    left, right, frame_width, frame_height,
                )

            painted[index, top:bottom + 1, left:right + 1] = 1.0

        return io.NodeOutput(painted)
