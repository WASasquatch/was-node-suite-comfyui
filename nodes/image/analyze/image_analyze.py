"""Plot an image's tonal distribution as a chart."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import image_planes, pil2tensor, tensor2pil
from ....modules.image.histogram import bins
from ....modules.image.levels import black_white_levels, black_white_points, channel_frequency
from ....modules.interface import preview, run_result

logger = log.get_logger("nodes.image.analyze")

#: The charts the node draws, keyed on the value the ``mode`` combo carries.
CHARTS = {
    "Black White Levels": black_white_levels,
    "RGB Levels": channel_frequency,
}


def _publish_report(source, mode: str, images: int) -> None:
    """Report the tones that were measured to the node's own interface.

    Never raises, and never changes what the node returns.

    Args:
        source: The PIL image the chart was drawn from.
        mode: Which chart was drawn, as the combo spells it.
        images: Frames in the batch, of which the first is the one charted.
    """
    try:
        if not run_result.watching():
            return
        black, white = black_white_points(source)
        run_result.publish(
            # The node charts one image whatever it is handed, so a batch loses everything
            # after the first frame and nothing else on the canvas says so.
            status=run_result.WARNING if images > 1 else run_result.OK,
            summary=(
                f"{mode} of image 1 of {images}, luminance {black} to {white}"
                if images > 1
                else f"{mode}, luminance {black} to {white}"
            ),
            counts={"black point": black, "white point": white, "images": images},
            facts={
                "chart": mode,
                # PIL's own L conversion, which is fixed point ITU-R 601-2 with rounding and
                # is not what a browser's luma coefficients answer.
                "measured on": "the first image's luminance",
            },
        )
    except Exception as error:
        logger.debug("no tonal report was published (%s)", error)


class ImageAnalyze(io.ComfyNode):
    """Chart how an image's tones are distributed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Analyze",
            display_name="Image Histogram Chart",
            search_aliases=[
                "Image Analyze", "Image Histogram Chart", "histogram", "levels",
                "channel plot",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Render a histogram of an image's tones as a chart image, either overall "
                "brightness or the three colour channels side by side. The same counts come "
                "out on five histogram sockets, which a Curve Editor draws a curve against."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to measure. Only the first image of a batch is charted."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["Black White Levels", "RGB Levels"],
                    tooltip=(
                        "`Black White Levels` = one brightness histogram with the darkest and "
                        "lightest tones marked in red, for spotting clipping; `RGB Levels` = "
                        "red, green and blue as three graphs side by side, for spotting a "
                        "colour cast."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The chart, as an image. It is a picture of the graph, not of the "
                        "image that was measured."
                    ),
                ),
                io.Histogram.Output(
                    display_name="rgb",
                    tooltip=(
                        "The three colour channels averaged, as 256 counts. Wire it into a "
                        "Curve Editor to draw a curve against the tones it acts on."
                    ),
                ),
                io.Histogram.Output(
                    display_name="luminance",
                    tooltip="Brightness as the eye weighs it, as 256 counts.",
                ),
                io.Histogram.Output(
                    display_name="red",
                    tooltip="The red channel, as 256 counts.",
                ),
                io.Histogram.Output(
                    display_name="green",
                    tooltip="The green channel, as 256 counts.",
                ),
                io.Histogram.Output(
                    display_name="blue",
                    tooltip="The blue channel, as 256 counts.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, mode="Black White Levels") -> io.NodeOutput:
        chart = CHARTS.get(mode)
        if chart is None:
            raise ValueError(
                f"Image Histogram Chart has no chart called {mode!r}. Set mode to one of: "
                f"{', '.join(CHARTS)}."
            )

        planes = image_planes(image)
        source = tensor2pil(planes[0])

        drawn = chart(source)

        counted = bins(planes[0])

        preview.publish(image)
        _publish_report(source, mode, len(planes))
        return io.NodeOutput(pil2tensor(drawn), *counted)
