"""Extract an image's dominant colours as a swatch chart and a list of hex codes."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.interface import run_result

logger = log.get_logger("nodes.image.analyze")

#: Swatch size in pixels, and the gap left above and below a swatch's label.
CELL_SIZE = 128
LABEL_PADDING = 10

#: Point size of the swatch labels.
FONT_SIZE = 15

#: Characters one ``#rrggbb`` code costs in the report body, the newline after it included.
_CODE_CHARS = len("#000000\n")

#: Whole codes that fit the body a run result carries. 125 of them are 999 characters.
_CODES_IN_BODY = (run_result.MAX_BODY_CHARS + 1) // _CODE_CHARS


def _publish_report(palette: str, colors: int, images: int) -> None:
    """Report the palette that was found to the node's own interface.

    Never raises, and never changes what the node returns.

    Args:
        palette: The first image's ``#rrggbb`` codes, newline separated, as
            ``generate_palette`` wrote them.
        colors: How many swatches were asked for, which is also how many were drawn.
        images: Frames in the batch, of which the first is the one reported.
    """
    try:
        if not run_result.watching():
            return
        codes = [code for code in palette.split("\n") if code]
        distinct = len(set(codes))
        carried = codes[:_CODES_IN_BODY]
        # The clusters are averaged and then rounded to 8 bits, so two of them can land on
        # one code. That is the reading that says the count asked for is above what the
        # picture holds, which is the one thing a person changes the count in response to.
        short = distinct < len(codes)
        summary = f"{len(codes)} swatches, {distinct} of them distinct"
        if images > 1:
            summary = f"{summary}, from image 1 of {images}"
        run_result.publish(
            status=run_result.WARNING if short else run_result.OK,
            summary=summary,
            counts={"colours": colors, "distinct": distinct, "images": images},
            facts={"drawn": f"{len(carried)} of {len(codes)} carried to the panel"},
            bodies=[run_result.body("palette", "\n".join(carried))],
        )
    except Exception as error:
        logger.debug("no palette report was published (%s)", error)


class ImageColorPalette(io.ComfyNode):
    """Show an image's dominant colours as labelled swatches."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Color Palette",
            display_name="Image Color Palette",
            search_aliases=["Image Color Palette", "dominant colours", "swatches", "palette"],
            category="WAS Suite/Image/Analyze",
            description=(
                "Find an image's dominant colours and return them both as a chart of "
                "swatches and as a list of hex codes."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to take the colours from. A batch gives one palette per "
                        "image."
                    ),
                ),
                io.Int.Input(
                    "colors",
                    default=16,
                    min=8,
                    max=256,
                    step=1,
                    tooltip=(
                        "How many colours to pick out. 8 gives the broad strokes, 16 a usable "
                        "working palette, and 256 something close to the full range of the "
                        "image. Larger values take noticeably longer."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["Chart", "back_to_back"],
                    tooltip=(
                        "How the swatches are laid out. `Chart` arranges them in a grid and "
                        "writes each colour's RGB values under it. 'back_to_back' draws one "
                        "unlabelled row, which is what to use as a palette strip."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The palette drawn as swatches, one image per image in.",
                ),
                LIST.Output(
                    display_name="color_palettes",
                    tooltip=(
                        "One entry per image, each holding the palette's '#rrggbb' codes as "
                        "newline-separated text in the order they were drawn. Feed it to Image "
                        "Pixelate to repaint an image in these colours."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, colors=16, mode="Chart") -> io.NodeOutput:
        from ....modules.image.palette import default_font_path, generate_palette

        font = default_font_path()
        if font is None:
            logger.debug("no label font is available, so the swatch labels use PIL's built-in font")
        elif mode == "Chart":
            logger.debug("found font at `%s`", font)

        if len(image) > 1:
            palette_strings = []
            palette_images = []
            for img in image:
                palette_image, palette = generate_palette(
                    tensor2pil(img), colors, CELL_SIZE, LABEL_PADDING, font, FONT_SIZE, mode.lower()
                )
                palette_images.append(pil2tensor(palette_image))
                palette_strings.append(palette)
            _publish_report(palette_strings[0], colors, len(palette_strings))
            return io.NodeOutput(torch.cat(palette_images, dim=0), palette_strings)

        palette_image, palette = generate_palette(
            tensor2pil(image), colors, CELL_SIZE, LABEL_PADDING, font, FONT_SIZE, mode.lower()
        )
        _publish_report(palette, colors, 1)
        return io.NodeOutput(pil2tensor(palette_image), [palette])
