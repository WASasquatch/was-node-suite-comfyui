"""Measure an image and emit the numbers."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat.types import DICT
from ....modules.convert.tensors import image_planes


class ImageStatistics(io.ComfyNode):
    """Measure every image in a batch, one value per image on each socket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageStatistics",
            display_name="Image Statistics",
            search_aliases=[
                "WASImageStatistics", "Image Statistics",
                "measure",
                "brightness",
                "contrast",
                "sharpness",
                "blur detection",
                "exposure",
                "histogram",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Measure brightness, contrast, sharpness, saturation, clipping and entropy "
                "for every image in a batch, as numbers a condition node can act on."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to measure. Every image is measured on its own and "
                        "produces its own set of numbers."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="mean",
                    is_output_list=True,
                    tooltip=(
                        "Average brightness, 0.0 for black and 1.0 for white. A photograph "
                        "normally lands between 0.35 and 0.6; well below that is an "
                        "underexposed render."
                    ),
                ),
                io.Float.Output(
                    display_name="median",
                    is_output_list=True,
                    tooltip=(
                        "The middle brightness, 0.0 to 1.0. Far below the mean means the "
                        "picture is mostly dark with a few bright areas pulling the average "
                        "up, which a mean on its own cannot tell apart from an even "
                        "mid-tone."
                    ),
                ),
                io.Float.Output(
                    display_name="minimum",
                    is_output_list=True,
                    tooltip="Brightness of the darkest pixel, 0.0 to 1.0.",
                ),
                io.Float.Output(
                    display_name="maximum",
                    is_output_list=True,
                    tooltip="Brightness of the brightest pixel, 0.0 to 1.0.",
                ),
                io.Float.Output(
                    display_name="contrast",
                    is_output_list=True,
                    tooltip=(
                        "Spread of brightness around the mean, 0.0 to about 0.5. Below "
                        "roughly 0.1 is a flat, hazy picture; this is the number to test "
                        "when deciding whether a frame needs a levels pass."
                    ),
                ),
                io.Float.Output(
                    display_name="sharpness",
                    is_output_list=True,
                    tooltip=(
                        "How much fine detail the picture holds, from the spread of its edge "
                        "response. No fixed scale: it rises with resolution and texture, so "
                        "compare frames of one sequence to find the soft ones."
                    ),
                ),
                io.Float.Output(
                    display_name="saturation",
                    is_output_list=True,
                    tooltip=(
                        "Average colourfulness, 0.0 for greyscale and 1.0 for fully "
                        "saturated. Useful for catching a render that has drifted grey, and "
                        "for telling a black and white frame from a colour one."
                    ),
                ),
                io.Float.Output(
                    display_name="clipped_shadows",
                    is_output_list=True,
                    tooltip=(
                        "The fraction of pixels at pure black, 0.0 to 1.0. Detail there is "
                        "gone rather than dark, so no amount of lifting brings it back."
                    ),
                ),
                io.Float.Output(
                    display_name="clipped_highlights",
                    is_output_list=True,
                    tooltip=(
                        "The fraction of pixels at pure white, 0.0 to 1.0. A few percent is "
                        "normal for a picture with a light source in it; much more than that "
                        "is an overexposed render."
                    ),
                ),
                io.Float.Output(
                    display_name="entropy",
                    is_output_list=True,
                    tooltip=(
                        "How much of the tonal range is in use, in bits, 0.0 to 8.0. A "
                        "detailed photograph sits around 7; a flat or nearly empty frame "
                        "sits far below it, which makes this a good test for a render that "
                        "collapsed."
                    ),
                ),
                DICT.Output(
                    display_name="stats",
                    is_output_list=True,
                    tooltip=(
                        "Every measurement for that image in one dictionary, for Dictionary "
                        "to Console, Text Dictionary Get or writing out beside the image."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    is_output_list=True,
                    tooltip=(
                        "The measurements as one line of text per image, for a log or a "
                        "caption burnt in with Image Draw Text."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        from ....modules.image.statistics import FIELDS, measure

        readings = [measure(plane) for plane in image_planes(images)]
        if not readings:
            raise ValueError(
                "Image Statistics was given an image batch holding no frames, so there is "
                "nothing to measure and the graph below it cannot be run. Check the node "
                "feeding images."
            )
        summaries = [cls.summarise(reading) for reading in readings]

        # FIELDS is the declared order of the ten numeric outputs. A socket carrying a
        # different measurement than its label says would stay connected and quietly
        # report the wrong number, so the two are read from one list rather than kept in
        # step by hand.
        columns = [[reading[field] for reading in readings] for field in FIELDS]
        return io.NodeOutput(
            *columns,
            readings,
            summaries,
            ui=ui.PreviewText("\n".join(summaries)),
        )

    @staticmethod
    def summarise(reading: dict[str, float]) -> str:
        """One line of text describing one image's measurements.

        Args:
            reading: A mapping from :func:`modules.image.statistics.measure`.

        Returns:
            The measurements as ``name=value`` pairs. Sharpness is given more decimal
            places than the rest.
        """
        parts = [
            f"mean={reading['mean']:.4f}",
            f"median={reading['median']:.4f}",
            f"min={reading['minimum']:.4f}",
            f"max={reading['maximum']:.4f}",
            f"contrast={reading['contrast']:.4f}",
            f"sharpness={reading['sharpness']:.6f}",
            f"saturation={reading['saturation']:.4f}",
            f"clipped_shadows={reading['clipped_shadows']:.4f}",
            f"clipped_highlights={reading['clipped_highlights']:.4f}",
            f"entropy={reading['entropy']:.4f}",
        ]
        return "  ".join(parts)
