"""A black and white darkroom print: a lens filter, a paper grade, toning and solarisation."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film

logger = log.get_logger("nodes.image.film")


class ImageDarkroomPrint(io.ComfyNode):
    """Print a batch in black and white, then tone it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDarkroomPrint",
            display_name="Image Darkroom Print",
            search_aliases=[
                "WASImageDarkroomPrint", "Image Darkroom Print", "black and white", "monochrome",
                "sepia", "selenium", "cyanotype", "split tone", "sabattier", "solarize", "paper grade",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Print a picture in black and white as a darkroom would: a coloured lens filter "
                "for the grey mix, a paper grade for contrast, then sepia, selenium, gold, "
                "platinum or cyanotype toning on a paper base."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to print, in colour or already black and white. An alpha channel passes through.",
                ),
                io.Combo.Input(
                    "lens_filter",
                    options=list(film.LENS_FILTERS),
                    tooltip="Coloured filter the scene is read through. `yellow` = skies a little darker; `orange` = darker, skin smoother; `red` = dramatic black skies; `green` = foliage lighter; `blue` = skies pale, skin darker. No effect on a picture already black and white.",
                ),
                io.Float.Input(
                    "paper_grade", default=2.0, min=0.0, max=5.0, step=0.5,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Contrast of the paper. 0 = very soft; 2 = normal; 3.5 = punchy; 5 = hard, near graphic.",
                ),
                io.Float.Input(
                    "exposure", default=0.0, min=-2.0, max=2.0, step=0.1,
                    tooltip="Stops of print exposure. 0.5 = a darker, heavier print; -0.5 = lighter. 0 = as the picture is.",
                ),
                io.Combo.Input(
                    "toner",
                    options=list(film.TONERS),
                    tooltip="`sepia` = warm brown highlights; `selenium` = cool purple-brown shadows; `split` = both; `gold` = blue-black; `platinum` = soft warm neutral; `cyanotype` = Prussian blue; `copper` = red brown; `blue` = cool blue.",
                ),
                io.Float.Input(
                    "toner_amount", default=0.6, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How deeply the print is toned. 0 = untoned; 0.6 = clearly toned; 1 = fully. Ignored with toner `none`.",
                ),
                io.Float.Input(
                    "sabattier", default=0.0, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Solarisation, as from fogging the print mid-development. 0 = none; 0.4 = highlights turned silvery grey; 1 = the brightest tones reversed toward dark.",
                ),
                io.Combo.Input(
                    "paper",
                    options=list(film.PAPERS),
                    tooltip="The paper's white. `neutral white` = modern resin paper; `warm fibre` = classic fibre paper; `cream` = old stock; `cool white` = a cold-tone paper.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The prints, in three colour channels."),
            ],
        )

    @classmethod
    def execute(cls, images, lens_filter="none", paper_grade=2.0, exposure=0.0, toner="none",
                toner_amount=0.6, sabattier=0.0, paper="neutral white") -> io.NodeOutput:
        result = film.darkroom_print(
            images, lens_filter, paper_grade, exposure, toner, toner_amount, sabattier, paper
        )
        logger.info("Image Darkroom Print printed %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(result)
