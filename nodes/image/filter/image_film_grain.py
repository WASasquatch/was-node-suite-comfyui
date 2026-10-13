"""Photographic film grain."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image.grain import film_grain


class ImageFilmGrain(io.ComfyNode):
    """Add photographic film grain to an image batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Film Grain",
            display_name="Image Film Grain",
            search_aliases=["Image Film Grain", "film grain", "grain", "noise", "analog", "35mm"],
            category="WAS Suite/Image/Filter",
            description=(
                "Add photographic film grain to an image or a video batch. The grain is "
                "strongest in the midtones and fades toward black and white, leaving colour "
                "and brightness as they were. Each frame gets its own grain pattern from the seed."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image or batch to add grain to. Values above white are kept, and an "
                        "alpha channel is left as it was."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=0.03,
                    min=0.0,
                    max=0.5,
                    step=0.005,
                    tooltip=(
                        "0 = no grain, image unchanged; 0.03 = fine film grain; 0.06 = coarse, "
                        "pushed film; 0.15 = heavy. Grain shows most on smooth areas such as "
                        "skin and sky."
                    ),
                ),
                io.Float.Input(
                    "grain_size",
                    default=1.5,
                    min=0.5,
                    max=8.0,
                    step=0.1,
                    tooltip=(
                        "Width of one grain clump in pixels. 1 = fine and crisp; 1.5 = soft film "
                        "grain; 3 = coarse. Raise it with the frame size: 2 to 3 suits a 4K frame."
                    ),
                ),
                io.Float.Input(
                    "color",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "0 = the same grain on red, green and blue, like black-and-white film; "
                        "0.25 = a hint of colour, like colour negative; 1 = separate grain per "
                        "channel, which reads as digital noise."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip=(
                        "Which grain pattern: `0` and `1` give two different ones, and one seed "
                        "always gives the same. Each frame of a batch draws its own, so grain "
                        "moves on video like it does on film."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with grain added."),
            ],
        )

    @classmethod
    def execute(cls, image, strength, grain_size, color, seed) -> io.NodeOutput:
        return io.NodeOutput(film_grain(image, strength, grain_size, color, seed))
