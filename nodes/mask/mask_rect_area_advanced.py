"""Build a rectangular mask from pixel coordinates."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.image.convolve import gaussian_blur
from ...modules.mask import drawn


class MaskRectAreaAdvanced(io.ComfyNode):
    """Draw a white rectangle on a mask of a given pixel size."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Rect Area (Advanced)",
            display_name="Mask Rect Area (Advanced)",
            search_aliases=[
                "Mask Rect Area (Advanced)",
                "rectangle mask",
                "solid mask",
                "box mask",
            ],
            category="WAS Suite/Image/Masking",
            description="Draw a rectangle on a mask of image_width by image_height. Every "
            "coordinate is in pixels, and a rectangle running past the canvas is clipped. A "
            "mask painted on the node joins the rectangle in whichever way drawn_combine names.",
            inputs=[
                io.Int.Input(
                    "x",
                    default=0,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Left edge of the rectangle, in pixels from the left of the mask. 0 "
                        "starts flush against the edge."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Top edge of the rectangle, in pixels from the top of the mask. 0 starts "
                        "flush against the edge."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=256,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "How wide the rectangle is, in pixels, measured rightwards from x. 0 "
                        "produces nothing, and anything past image_width is cut off."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=256,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "How tall the rectangle is, in pixels, measured downwards from y. 0 "
                        "produces nothing, and anything past image_height is cut off."
                    ),
                ),
                io.Int.Input(
                    "image_width",
                    default=512,
                    min=64,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Width of the mask itself, in pixels. Match it to the image the mask will "
                        "be applied to, or the two will not line up."
                    ),
                ),
                io.Int.Input(
                    "image_height",
                    default=512,
                    min=64,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Height of the mask itself, in pixels. Match it to the image the mask "
                        "will be applied to, or the two will not line up."
                    ),
                ),
                io.Int.Input(
                    "blur_radius",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Softens the rectangle's edges, in pixels. 0 keeps them hard; 32 gives a "
                        "wide fade that also pulls the corners in. Large values are slow, because "
                        "the blur window is twice this plus one across."
                    ),
                ),
                io.String.Input(
                    "drawn_mask",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "The mask painted on the node, written by the interface and saved with "
                        "the workflow. Empty means nothing was painted, and the node then "
                        "produces the rectangle alone. Clear the field to remove the painting."
                    ),
                ),
                io.Combo.Input(
                    "drawn_combine",
                    options=list(drawn.COMBINE_MODES),
                    default=drawn.DEFAULT_COMBINE,
                    optional=True,
                    tooltip=(
                        "How the painting joins the rectangle, resized to it if they differ. "
                        "`union` = the brighter of the two at each pixel; `subtract` = the "
                        "painting taken away; `intersect` = the darker; `off` = ignored, not "
                        "deleted."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "A mask of image_width by image_height, white inside the rectangle and "
                        "black outside."
                    ),
                )
            ],
            hidden=[io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        x,
        y,
        width,
        height,
        image_width,
        image_height,
        blur_radius,
        drawn_mask="",
        drawn_combine=drawn.DEFAULT_COMBINE,
    ) -> io.NodeOutput:
        max_x = x + width
        max_y = y + height

        mask = torch.zeros((image_height, image_width))
        mask[int(y):int(max_y), int(x):int(max_x)] = 1

        if blur_radius > 0:
            mask = gaussian_blur(mask[None, None], size=blur_radius * 2 + 1)[0, 0]

        return io.NodeOutput(drawn.apply(mask.unsqueeze(0), drawn_mask, drawn_combine))
