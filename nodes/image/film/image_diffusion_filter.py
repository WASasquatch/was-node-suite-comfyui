"""Diffusion filters and soft lenses: mist, glimmer, soft focus, Orton, and soft corners."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film

logger = log.get_logger("nodes.image.film")


class ImageDiffusionFilter(io.ComfyNode):
    """Soften a batch as a diffusion filter or a soft lens would."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDiffusionFilter",
            display_name="Image Diffusion Filter",
            search_aliases=[
                "WASImageDiffusionFilter", "Image Diffusion Filter", "pro mist", "black mist",
                "soft focus", "glimmer glass", "orton", "dreamy", "lens softness",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Soften a picture the way a filter in front of the lens does: black mist halos "
                "the highlights, white mist veils the whole frame, glimmer sparkles, soft focus "
                "and Orton glow. Soft corners blur the edges like an old or cheap lens."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to soften. An alpha channel passes through.",
                ),
                io.Combo.Input(
                    "style",
                    options=list(film.DIFFUSION_STYLES),
                    tooltip="`black mist` = glow round highlights, blacks kept; `white mist` = a veil that lifts shadows; `glimmer` = small sparkling halos; `soft focus` = highlights bleed over detail; `orton` = a bright dreamy glow.",
                ),
                io.Float.Input(
                    "strength", default=0.25, min=0.0, max=2.0, step=0.025,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Matches the grades sold: 0.125 = 1/8, subtle; 0.25 = 1/4; 0.5 = 1/2, obvious; 1 = full. 0 leaves only soft corners.",
                ),
                io.Float.Input(
                    "size", default=1.0, min=0.1, max=10.0, step=0.1,
                    tooltip="How far the glow spreads, in hundredths of the long edge. 0.5 = tight; 1 = a lens filter; 3 = a wide bloom.",
                ),
                io.Float.Input(
                    "corners", default=0.0, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Blur toward the corners, centre left sharp. 0 = none; 0.4 = a vintage lens; 1 = toy camera softness.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The softened pictures."),
            ],
        )

    @classmethod
    def execute(cls, images, style="black mist", strength=0.25, size=1.0, corners=0.0) -> io.NodeOutput:
        result = film.diffuse(images, style, strength, size, corners)
        logger.info("Image Diffusion Filter softened %d frame(s) with %s", int(images.shape[0]), style)
        return io.NodeOutput(result)
