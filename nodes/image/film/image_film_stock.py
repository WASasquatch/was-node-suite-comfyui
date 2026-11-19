"""Develop a picture as a film stock: its tone, colour, latitude, halation and grain."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film, film_stocks

logger = log.get_logger("nodes.image.film")


class ImageFilmStock(io.ComfyNode):
    """Give a batch the look of a film stock."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageFilmStock",
            display_name="Image Film Stock",
            search_aliases=[
                "WASImageFilmStock", "Image Film Stock", "film emulation", "film simulation",
                "portra", "velvia", "kodachrome", "cinestill", "tri-x", "polaroid", "analog",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Develop a picture as one of 70 film stocks: colour negative, slide, cinema, "
                "instant, black and white, lab processes and camera film simulations. Picking a "
                "stock brings its tone curve, colour, highlight latitude, halation and grain together."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to develop. An alpha channel passes through.",
                ),
                io.Combo.Input(
                    "stock",
                    options=list(film_stocks.NAMES),
                    default="Color Negative: Kodak Portra 400",
                    tooltip="The film. Looks are authored after each stock's character, not measured from it. `None` = no look, so exposure, grain and halation can be used alone.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=2.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How far the stock's tone and colour apply. 0 = none; 0.5 = halfway; 1 = the stock; 2 = exaggerated. Grain and halation have their own.",
                ),
                io.Float.Input(
                    "exposure", default=0.0, min=-3.0, max=3.0, step=0.1,
                    tooltip="Stops over or under. 1 = overexposed a stop: brighter, pastel on a negative, washed out on a slide. -1 = underexposed: darker and denser.",
                ),
                io.Float.Input(
                    "grain", default=1.0, min=0.0, max=3.0, step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="0 = none; 1 = the stock's own grain, coarser for faster film and sized to the frame; 2 = twice as strong.",
                ),
                io.Float.Input(
                    "halation", default=1.0, min=0.0, max=3.0, step=0.05,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Red glow round bright lights. 0 = none; 1 = the stock's own, strong on CineStill, faint elsewhere; 3 = heavy.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Which grain pattern: `0` and `1` give two different ones, and one seed always gives the same. Each frame of a batch draws its own.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The pictures as the stock."),
            ],
        )

    @classmethod
    def execute(cls, images, stock="Color Negative: Kodak Portra 400", strength=1.0, exposure=0.0,
                grain=1.0, halation=1.0, seed=0) -> io.NodeOutput:
        chosen = film_stocks.find(stock)
        result = film.apply_stock(images, chosen, strength, exposure, grain, halation, seed)
        logger.info("Image Film Stock developed %d frame(s) as %s", int(images.shape[0]), stock)
        return io.NodeOutput(result)
