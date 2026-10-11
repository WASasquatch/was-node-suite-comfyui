"""Read one image bounds row as separate whole numbers.

A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive. ``right`` and
``bottom`` are one past the last column and row the rectangle covers.
"""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.compat.sockets import require_input
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.image import bounds


class BoundsToNumbers(io.ComfyNode):
    """Emit one row of an ``IMAGE_BOUNDS`` value as x, y, width, height and far edges."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundsToNumbers",
            display_name="Bounds to Numbers",
            search_aliases=[
                "WASBoundsToNumbers",
                "Bounds to Numbers",
                "bounds to numbers",
                "bounds x y",
                "region position",
                "region size",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Open one bounds rectangle into numbers a graph can wire: the top left "
                "corner, the width and height, the far edges, and the rectangle as text. Use "
                "it to feed a measured region into an x and y input or the four edges of a "
                "crop."
            ),
            inputs=[
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "The rectangle to read, from Image Bounds, Inset Image Bounds, "
                        "Mask to Bounds or Image Crop by Mask. A bounds measured over a "
                        "batch holds one row per image, which row_count reports."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which row to read. 0 = the first row; -1 = the last, counting "
                        "back from the end; -2 = the one before it. A row that is not "
                        "there is settled by out_of_range."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=["wrap", "clamp", "error"],
                    default="wrap",
                    tooltip=(
                        "Index outside 0..row_count-1. With 3 rows and index 4: `wrap` = "
                        "row 1, `clamp` = row 2, `error` stops the prompt. A bounds value "
                        "holding no rows stops the prompt whichever is chosen."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="x",
                    tooltip=(
                        "First pixel column the rectangle covers, counting from the left "
                        "of the image. Wire it to an x or left input, such as Mask Rect "
                        "Area's x or Image Crop Location's left."
                    ),
                ),
                io.Int.Output(
                    display_name="y",
                    tooltip=(
                        "First pixel row the rectangle covers, counting from the top of "
                        "the image. With x it gives the rectangle's top left corner."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Pixel columns the rectangle covers, so a row running from column "
                        "10 to column 137 is 128 wide. Feed it to an empty latent or image "
                        "to work at the region's own size."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "Pixel rows the rectangle covers. Divide width by height for the "
                        "aspect the region wants an upscale or a generation to keep."
                    ),
                ),
                io.Int.Output(
                    display_name="right",
                    tooltip=(
                        "One column past the last the rectangle covers, which is x plus "
                        "width. That is what Image Crop Location's right input reads: left "
                        "10 with right 138 crops 128 columns."
                    ),
                ),
                io.Int.Output(
                    display_name="bottom",
                    tooltip=(
                        "One row past the last the rectangle covers, which is y plus "
                        "height. Image Crop Location's bottom input reads the same way."
                    ),
                ),
                io.Int.Output(
                    display_name="row_count",
                    tooltip=(
                        "Rectangles the bounds value holds, which is 1 for a single mask "
                        "and one per image for a batch. index runs from 0 to row_count-1."
                    ),
                ),
                io.String.Output(
                    display_name="rectangle",
                    tooltip=(
                        "The rectangle as `x,y,width,height`: a region at column 10, row "
                        "20, 128 wide and 96 high reads `10,20,128,96`. Useful in a "
                        "filename, a caption or a note beside a render."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image_bounds, index=0, out_of_range="wrap") -> io.NodeOutput:
        """Read one bounds row as separate numbers.

        Args:
            image_bounds: Rows of ``(rmin, rmax, cmin, cmax)``, or one bare row.
            index: Which row to read, negatives counting from the end.
            out_of_range: ``wrap``, ``clamp`` or ``error``.

        Returns:
            ``x``, ``y``, ``width``, ``height``, ``right``, ``bottom``, ``row_count`` and
            the rectangle as text.

        Raises:
            ValueError: Nothing is connected to image_bounds, the value holds no rows, the
                chosen row is malformed or inside out, or the index is outside and
                ``out_of_range`` is ``error``.
        """
        require_input(
            image_bounds,
            "Bounds to Numbers",
            "image_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Mask to Bounds",
            "IMAGE_BOUNDS",
        )

        rows = bounds.rows(image_bounds)
        if not rows:
            raise ValueError(
                "Bounds to Numbers was given a bounds value holding no rectangles. Check "
                "the node feeding image_bounds: a region search that matched nothing "
                "produces this."
            )

        position = cls.resolve(int(index), len(rows), out_of_range)
        row = rows[position]
        if len(row) != 4:
            raise ValueError(
                f"Bounds row {position} holds {len(row)} number(s) rather than four. A "
                f"bounds row reads (rmin, rmax, cmin, cmax), so check the node feeding "
                f"image_bounds."
            )

        rmin, rmax, cmin, cmax = row
        if rmin > rmax or cmin > cmax:
            raise ValueError(
                f"Bounds row {position} is inside out: it reads (rmin={rmin}, rmax={rmax}, "
                f"cmin={cmin}, cmax={cmax}), and rmin must not exceed rmax, nor cmin exceed "
                f"cmax."
            )

        # A bounds row names its last pixel row and column, so width and height count that
        # last one in and the far edges land one past it.
        width, height = cmax + 1 - cmin, rmax + 1 - rmin
        return io.NodeOutput(
            cmin,
            rmin,
            width,
            height,
            cmax + 1,
            rmax + 1,
            len(rows),
            f"{cmin},{rmin},{width},{height}",
        )

    @staticmethod
    def resolve(index: int, row_count: int, out_of_range: str) -> int:
        """Turn a requested row number into one the bounds value holds.

        Args:
            index: The requested row, counting from 0. Negative counts back from the end.
            row_count: Rows the bounds value holds, which is one or more.
            out_of_range: ``wrap``, ``clamp`` or ``error``.

        Returns:
            A row number from 0 to ``row_count - 1``.

        Raises:
            ValueError: The row is outside and ``out_of_range`` is ``error``.
        """
        position = index + row_count if index < 0 else index
        if 0 <= position < row_count:
            return position
        if out_of_range == "wrap":
            return position % row_count
        if out_of_range == "clamp":
            return 0 if position < 0 else row_count - 1
        plural = "" if row_count == 1 else "s"
        raise ValueError(
            f"Bounds to Numbers was asked for row {index} of a bounds value holding "
            f"{row_count} row{plural}, numbered 0 to {row_count - 1}. Rows count from 0, "
            f"and -1 is the last one. Set out_of_range to clamp for the nearest row, or "
            f"to wrap to count round again."
        )
