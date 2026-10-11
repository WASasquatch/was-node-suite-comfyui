"""Build a rectangular mask from percentages of a fixed canvas."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.image.convolve import gaussian_blur
from ...modules.mask import drawn

#: Side of the canvas the percentages are measured against, in pixels.
RESOLUTION = 512


class MaskRectArea(io.ComfyNode):
    """Draw a white rectangle on a 512x512 mask, positioned in percent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Rect Area",
            display_name="Mask Rect Area",
            search_aliases=["Mask Rect Area", "rectangle mask", "solid mask", "box mask"],
            category="WAS Suite/Image/Masking",
            description="Draw a rectangle on a 512x512 mask. Every coordinate is a percentage "
            "of that canvas, so the mask scales with whatever it is applied to. A mask painted "
            "on the node joins the rectangle in whichever way drawn_combine names.",
            inputs=[
                io.Int.Input(
                    "x",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "Left edge of the rectangle, as a percentage across the canvas. 0 is the "
                        "far left, 50 is the centre line."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "Top edge of the rectangle, as a percentage down the canvas. 0 is the "
                        "top, 50 is halfway down."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=50,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "How wide the rectangle is, as a percentage of the canvas. 100 spans the "
                        "full width, 50 covers half of it, 0 produces nothing. Anything running "
                        "past the right edge is cut off."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=50,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "How tall the rectangle is, as a percentage of the canvas. 100 spans the "
                        "full height, 50 covers half of it, 0 produces nothing. Anything running "
                        "past the bottom edge is cut off."
                    ),
                ),
                io.Int.Input(
                    "blur_radius",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Softens the rectangle's edges, in pixels of the 512-pixel canvas. 0 "
                        "keeps them hard; 32 gives a wide fade that also pulls the corners in. "
                        "Large values are slow, because the blur window is twice this plus one "
                        "across."
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
                        "A 512x512 mask, white inside the rectangle and black outside. Scale it "
                        "or let a downstream node resize it to match the image it is used with."
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
        blur_radius,
        drawn_mask="",
        drawn_combine=drawn.DEFAULT_COMBINE,
    ) -> io.NodeOutput:
        min_x = x / 100
        min_y = y / 100
        width = width / 100
        height = height / 100

        mask = torch.zeros((RESOLUTION, RESOLUTION))

        min_x_px = int(min_x * RESOLUTION)
        min_y_px = int(min_y * RESOLUTION)
        max_x_px = int((min_x + width) * RESOLUTION)
        max_y_px = int((min_y + height) * RESOLUTION)

        mask[min_y_px:max_y_px, min_x_px:max_x_px] = 1

        if blur_radius > 0:
            mask = gaussian_blur(mask[None, None], size=blur_radius * 2 + 1)[0, 0]

        return io.NodeOutput(drawn.apply(mask.unsqueeze(0), drawn_mask, drawn_combine))
