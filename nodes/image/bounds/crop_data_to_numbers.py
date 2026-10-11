"""Read a crop window's measurements as separate numbers.

``crop_data`` is ``(size, (left, top, right, bottom))`` with exclusive right and bottom
edges, so the window runs ``right - left`` across and ``bottom - top`` down.
"""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import CROP_DATA
from ....modules.image import bounds


class CropDataToNumbers(io.ComfyNode):
    """Split a ``CROP_DATA`` crop window into plain integers and a line of text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCropDataToNumbers",
            display_name="Crop Data to Numbers",
            search_aliases=[
                "WASCropDataToNumbers",
                "Crop Data to Numbers",
                "crop data to numbers",
                "crop position",
                "crop size",
                "crop x y width height",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Open the crop window a crop node recorded into plain numbers: the size the "
                "crop came out at, where it starts in the source image, and how far it runs. "
                "Use it to drive a resize, place a paste or label at the same spot, or write "
                "the window out as text."
            ),
            inputs=[
                CROP_DATA.Input(
                    "crop_data",
                    tooltip=(
                        "The crop window to read, from any node with a crop_data output. "
                        "Image Crop Face and Image Crop Face (YuNet) pass False here when "
                        "they found no face, and that raises, since there are no numbers "
                        "to read."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="crop_width",
                    tooltip=(
                        "Width of the picture the crop node emitted, in pixels. Matches width "
                        "except for Image Crop Face and Image Crop Face (YuNet), which square "
                        "their window: 480x460 comes out 480x480. Wire it wherever a resize "
                        "has to match the cut."
                    ),
                ),
                io.Int.Output(
                    display_name="crop_height",
                    tooltip=(
                        "Height of the picture the crop node emitted, in pixels. It parts "
                        "from height only for the two face crops, for the same reason "
                        "crop_width does. Pair the two to resize a reworked region back "
                        "before Image Paste Crop puts it home."
                    ),
                ),
                io.Int.Output(
                    display_name="x",
                    tooltip=(
                        "Left edge of the window in the source image, counting from 0. "
                        "100 = the window starts 100 pixels in from the left. Feed it to "
                        "Image Crop Location's left, or to a draw node marking the same spot."
                    ),
                ),
                io.Int.Output(
                    display_name="y",
                    tooltip=(
                        "Top edge of the window in the source image, counting from 0. "
                        "40 = the window starts 40 pixels down. Pair it with x to place "
                        "text, a paste or a mask exactly where the crop was taken from."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "How far the window runs across, in pixels: its right edge minus x. "
                        "Add it to x for the right edge Image Crop Location asks for. This "
                        "is the rectangle cut from the source, which crop_width matches "
                        "unless a face crop squared it up."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "How far the window runs down, in pixels: its bottom edge minus y. "
                        "Add it to y for the bottom edge. Compare it with crop_height to see "
                        "how much a face crop padded the window on the way out."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "The whole window on one line, as `crop 480x480 at (100, 40) to "
                        "(580, 500), 480x460`, for a filename, a note or Text to Console."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, crop_data=None) -> io.NodeOutput:
        """Read a crop window's size, origin and extent.

        Args:
            crop_data: ``(size, (left, top, right, bottom))`` from a crop node.

        Returns:
            The recorded width and height, the window's left and top, its width and height,
            and all of it written out as one line.

        Raises:
            ValueError: Nothing is connected to crop_data, the value is False, it is not a
                size and a rectangle, or the rectangle is empty or inside out.
        """
        require_input(
            crop_data,
            "Crop Data to Numbers",
            "crop_data",
            "crop window",
            "Image Crop Location, Image Crop by Mask or Mask Crop Region",
            "crop_data",
        )

        if crop_data is False:
            raise ValueError(
                "Crop Data to Numbers was given False on its crop_data input, which Image "
                "Crop Face and Image Crop Face (YuNet) pass on when they found no face. "
                "There is no rectangle to measure. Feed it a crop that found something, or "
                "take the numbers from the mask instead with Mask Crop Region."
            )

        (crop_width, crop_height), (left, top, right, bottom) = bounds.crop_window(
            crop_data, "Crop Data to Numbers"
        )
        if right < left or bottom < top:
            raise ValueError(
                f"Crop Data to Numbers was given the inside out window ({left}, {top}, "
                f"{right}, {bottom}). A crop window's right edge cannot sit left of its "
                f"left edge, nor its bottom above its top, so its width and height would "
                f"come out negative. Whatever produced this crop_data measured the "
                f"rectangle backwards."
            )
        if right == left or bottom == top:
            raise ValueError(
                f"Crop Data to Numbers was given the empty window ({left}, {top}, {right}, "
                f"{bottom}), which Mask Crop Region emits when its mask marks nothing. Its "
                f"width or height would come out as 0 and stop a resize further on. Check "
                f"the mask feeding that node."
            )

        # A crop window ends one past its last pixel row and column, so the extent is the
        # difference between the two edges.
        width = right - left
        height = bottom - top
        summary = (
            f"crop {crop_width}x{crop_height} at ({left}, {top}) to ({right}, {bottom}), "
            f"{width}x{height}"
        )
        return io.NodeOutput(crop_width, crop_height, left, top, width, height, summary)
