"""Cut a width by height rectangle out of a picture from a corner position."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.interface import size_report
from ....modules.log import get_logger

logger = get_logger("nodes.image.core")


class ImageCropRegion(io.ComfyNode):
    """Take one rectangle out of every frame in a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCropRegion",
            display_name="Image Crop Region",
            search_aliases=[
                "WASImageCropRegion",
                "Image Crop Region",
                "ImageCrop",
                "crop",
                "cut out",
                "trim",
                "region",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Cut a rectangle out of a picture, given its width, its height and the "
                "corner it starts at. The node draws the crop inside the original frame with "
                "both sizes, so a rectangle that ran off an edge and came back smaller shows "
                "at once."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The picture to crop. A batch is cut to the same rectangle frame by "
                        "frame and comes back the same length."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the rectangle spans across; INT. 512 is a square crop's side, "
                        "1 is a single column. A rectangle reaching past the right edge stops "
                        "there, so the answer comes back narrower than this."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the rectangle spans down; INT. 512 is a square crop's side, 1 "
                        "is a single row. A rectangle reaching past the bottom edge stops "
                        "there, so the answer comes back shorter than this."
                    ),
                ),
                io.Int.Input(
                    "x",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels from the left of the frame to the rectangle's left edge; INT. "
                        "0 starts at the edge, 256 skips the first 256 columns. A value past "
                        "the last column is pulled back onto it."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels from the top of the frame to the rectangle's top edge; INT. 0 "
                        "starts at the edge, 256 skips the first 256 rows. A value past the "
                        "last row is pulled back onto it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The rectangle that was cut out, at most width by height and smaller "
                        "where it ran off an edge. Same batch length and channel count as the "
                        "picture that went in."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, width=512, height=512, x=0, y=0) -> io.NodeOutput:
        left = min(x, int(image.shape[2]) - 1)
        top = min(y, int(image.shape[1]) - 1)
        cropped = image[:, top:top + height, left:left + width, :]
        cls.report(image, cropped, width, height, left, top)
        return io.NodeOutput(cropped)

    @classmethod
    def report(cls, image, cropped, width, height, left, top) -> None:
        """Draw both frames on the node and log a rectangle that hit an edge.

        Args:
            image: The picture the rectangle was cut from.
            cropped: The rectangle that came out.
            width: Width asked for, in pixels.
            height: Height asked for, in pixels.
            left: Column the rectangle starts at, after clamping.
            top: Row the rectangle starts at, after clamping.
        """
        taken = (int(cropped.shape[2]), int(cropped.shape[1]))
        if taken != (width, height):
            logger.warning(
                "Image Crop Region was asked for %dx%d at %d,%d and the frame is %dx%d, so "
                "the rectangle stops at the edge and %dx%d came out. Lower x and y, or lower "
                "width and height, to take the whole rectangle.",
                width, height, left, top,
                int(image.shape[2]), int(image.shape[1]),
                taken[0], taken[1],
            )
        size_report.publish(
            image,
            cropped,
            action="cropped",
            requested=(width, height),
            facts={"window": f"{left},{top} to {left + taken[0]},{top + taken[1]}"},
        )
