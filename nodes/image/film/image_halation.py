"""The red glow film forms around bright light."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film

logger = log.get_logger("nodes.image.film")


class ImageHalation(io.ComfyNode):
    """Add halation around the highlights of a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageHalation",
            display_name="Image Halation",
            search_aliases=[
                "WASImageHalation", "Image Halation", "halation", "cinestill", "red glow",
                "highlight glow", "film glow",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Add the warm red glow that bleeds round lamps, neon and bright windows on film "
                "shot without an anti-halation layer. Only light above the threshold spreads, "
                "so the rest of the picture keeps its contrast."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to add halation to. Light above white glows further, and an alpha channel passes through.",
                ),
                io.Float.Input(
                    "amount", default=0.5, min=0.0, max=3.0, step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="0 = none; 0.5 = a visible fringe round lights; 1 = strong, as on CineStill at night; 3 = heavy.",
                ),
                io.Float.Input(
                    "radius", default=1.0, min=0.1, max=10.0, step=0.1,
                    tooltip="How far it spreads, in hundredths of the long edge. 0.5 = a tight fringe; 1 = a halo; 3 = a wide bloom.",
                ),
                io.Float.Input(
                    "threshold", default=0.75, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Brightness the glow starts at. 0.75 = lamps, sky and specular light; 0.9 = only the brightest; 0.5 = bright skin and walls too.",
                ),
                io.String.Input(
                    "color", default="#ff4b1f",
                    tooltip="Colour of the glow as hexadecimal. #ff4b1f = film's red orange; #ff8c40 = warmer amber; #ffffff = a neutral glow for black and white.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The pictures with halation."),
            ],
        )

    @classmethod
    def execute(cls, images, amount=0.5, radius=1.0, threshold=0.75, color="#ff4b1f") -> io.NodeOutput:
        try:
            colour = film.parse_hex(color)
        except ValueError as error:
            raise ValueError(f"Image Halation: {error}. Type the colour as #rrggbb.") from error
        result = film.halation(images, amount, radius, threshold, colour)
        logger.info("Image Halation glowed %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(result)
