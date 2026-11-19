"""Calm a generated render's overcooked tone, colour and crunch toward a photograph."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import develop

logger = log.get_logger("nodes.image.develop")


class ImageNaturalFinish(io.ComfyNode):
    """Pull a render's excess contrast, saturation and sharpness back toward a photograph."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageNaturalFinish",
            display_name="Image Natural Finish",
            search_aliases=[
                "WASImageNaturalFinish", "Image Natural Finish", "anti ai", "de-ai", "natural",
                "tame render", "overcooked", "photographic", "realism",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Take the generated look off a render: crushed blacks and hard highlights eased, "
                "neon and orange-heavy colour calmed, and over-sharpened, crunchy texture "
                "softened. A first step before developing or adding a film look."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The renders to calm. An alpha channel passes through.",
                ),
                io.Float.Input(
                    "amount", default=0.5, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Tone and colour. 0 = unchanged; 0.5 = calmer contrast and colour, skin less orange; 1 = strongly muted, like an unedited photograph.",
                ),
                io.Float.Input(
                    "detail", default=0.3, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Over-sharpened texture and local contrast. 0 = unchanged; 0.3 = the crunch taken off; 1 = soft, as from a lens rather than a sharpening pass.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The calmer pictures."),
            ],
        )

    @classmethod
    def execute(cls, images, amount=0.5, detail=0.3) -> io.NodeOutput:
        result = develop.natural_finish(images, amount, detail)
        logger.info("Image Natural Finish finished %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(result)
