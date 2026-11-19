"""Light leaking into a camera, burned into the frame."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film

logger = log.get_logger("nodes.image.film")


class ImageLightLeak(io.ComfyNode):
    """Burn a light leak into a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageLightLeak",
            display_name="Image Light Leak",
            search_aliases=[
                "WASImageLightLeak", "Image Light Leak", "light leak", "film burn", "end of roll",
                "lomo", "flare", "prism", "toy camera",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Burn light leaking into the camera into a picture: a glow along an edge, a "
                "corner flare, the burned end of a roll, a streak or a prism rainbow. Each seed "
                "draws a different leak."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to leak light into. An alpha channel passes through.",
                ),
                io.Combo.Input(
                    "style",
                    options=list(film.LEAK_STYLES),
                    tooltip="`edge glow` = a soft blotchy glow along an edge; `corner flare` = a round flare from a corner; `end of roll` = a burned-out region; `streak` = a bar of light; `prism` = a rainbow arc.",
                ),
                io.Combo.Input(
                    "position",
                    options=list(film.LEAK_POSITIONS),
                    tooltip="Where the light comes in. `random` = an edge or corner chosen by the seed.",
                ),
                io.String.Input(
                    "color", default="#ff6a2b",
                    tooltip="Colour of the leak as hexadecimal; its core burns toward warm white. #ff6a2b = orange; #ff2a3c = red; #ff3fb0 = magenta; #ffc24a = gold. `prism` ignores it.",
                ),
                io.Float.Input(
                    "amount", default=0.6, min=0.0, max=2.0, step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="0 = none; 0.6 = a clear leak; 1 = bright; 2 = burned out to white where it enters.",
                ),
                io.Float.Input(
                    "size", default=0.5, min=0.05, max=2.0, step=0.05,
                    tooltip="How far it reaches in, as a share of the short side. 0.2 = a thin edge; 0.5 = a third of the frame; 1.5 = across it.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Which leak. Each seed gives a different shape, and for `random` a different edge.",
                ),
                io.Boolean.Input(
                    "each_frame", default=False,
                    tooltip="`false` = every frame of a batch shares one leak; `true` = each frame draws its own, so the leak flickers on video.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The pictures with the leak."),
            ],
        )

    @classmethod
    def execute(cls, images, style="edge glow", position="random", color="#ff6a2b", amount=0.6,
                size=0.5, seed=0, each_frame=False) -> io.NodeOutput:
        try:
            colour = film.parse_hex(color)
        except ValueError as error:
            raise ValueError(f"Image Light Leak: {error}. Type the colour as #rrggbb.") from error
        result = film.light_leak(images, style, position, colour, amount, size, seed, each_frame)
        logger.info("Image Light Leak burned a %s into %d frame(s)", style, int(images.shape[0]))
        return io.NodeOutput(result)
