"""Push colour into the shadows, the midtones and the highlights separately."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, grade

logger = log.get_logger("nodes.image.adjust")


class ImageColorBalance(io.ComfyNode):
    """Grade a batch with a three-way colour balance."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageColorBalance",
            display_name="Image Color Balance",
            search_aliases=[
                "WASImageColorBalance",
                "Image Color Balance",
                "Image Colour Balance",
                "three way",
                "lift gamma gain",
                "colour grade",
                "colour cast",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Move colour in the shadows, midtones and highlights of a frame separately, "
                "the way a colourist's three wheels do. Use it for a graded look, such as cool "
                "shadows against warm highlights, or to correct a cast by pushing the opposite "
                "way in whichever range carries it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to grade. Each one gets the same balance and comes back "
                        "at the size it went in at."
                    ),
                ),
                io.Float.Input(
                    "shadow_red",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Red pushed into the dark parts. 0.0 = none, 0.05 = a warm shadow, "
                        "-0.05 = a cyan one."
                    ),
                ),
                io.Float.Input(
                    "shadow_green",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Green pushed into the dark parts. 0.0 = none, 0.05 = greener, -0.05 "
                        "= a magenta shadow."
                    ),
                ),
                io.Float.Input(
                    "shadow_blue",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Blue pushed into the dark parts. 0.0 = none, 0.06 = the cool shadow "
                        "of a teal and orange grade, -0.05 = a yellow one."
                    ),
                ),
                io.Float.Input(
                    "midtone_red",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Red pushed through the middle of the range, where skin sits. 0.0 = "
                        "none, 0.03 = a warmer face, -0.03 = a cooler one."
                    ),
                ),
                io.Float.Input(
                    "midtone_green",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Green pushed through the middle of the range. 0.0 = none, 0.03 = "
                        "greener, -0.03 = a magenta cast lifted off it."
                    ),
                ),
                io.Float.Input(
                    "midtone_blue",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Blue pushed through the middle of the range. 0.0 = none, 0.03 = "
                        "cooler, -0.03 = warmer."
                    ),
                ),
                io.Float.Input(
                    "highlight_red",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Red pushed into the bright parts. 0.0 = none, 0.06 = warm "
                        "highlights, -0.06 = cyan ones."
                    ),
                ),
                io.Float.Input(
                    "highlight_green",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Green pushed into the bright parts. 0.0 = none, 0.03 = greener "
                        "highlights, -0.03 = magenta ones."
                    ),
                ),
                io.Float.Input(
                    "highlight_blue",
                    default=0.0, min=-1.0, max=1.0, step=0.005,
                    tooltip=(
                        "Blue pushed into the bright parts. 0.0 = none, -0.04 = the warm "
                        "highlight of a teal and orange grade, 0.04 = a cooler one."
                    ),
                ),
                io.Boolean.Input(
                    "preserve_luminosity",
                    default=True,
                    tooltip=(
                        "`on` puts every pixel back to the brightness it had, so the grade "
                        "moves colour and never exposure; `off` lets a push brighten or "
                        "darken, which is what a lift and gain do."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The graded frames.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, shadow_red=0.0, shadow_green=0.0, shadow_blue=0.0,
        midtone_red=0.0, midtone_green=0.0, midtone_blue=0.0,
        highlight_red=0.0, highlight_green=0.0, highlight_blue=0.0,
        preserve_luminosity=True,
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        graded = grade.balanced(
            folded.images,
            (shadow_red, shadow_green, shadow_blue),
            (midtone_red, midtone_green, midtone_blue),
            (highlight_red, highlight_green, highlight_blue),
            bool(preserve_luminosity),
        )
        logger.info("Image Color Balance graded %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(dynamic.unfold(graded, folded))
