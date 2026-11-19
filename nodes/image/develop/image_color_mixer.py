"""Hue, saturation and luminance for eight colour bands, or a black and white mix through them."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import develop

logger = log.get_logger("nodes.image.develop")

#: The two ways the mixer renders.
MODES = ("color", "black and white")

RED_HUE = [
    {"offset": 0.0, "color": [220, 55, 190]},
    {"offset": 0.5, "color": [230, 45, 45]},
    {"offset": 1.0, "color": [240, 140, 35]},
]
RED_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [230, 45, 45]},
]
RED_LUMINANCE = [
    {"offset": 0.0, "color": [57, 11, 11]},
    {"offset": 0.5, "color": [230, 45, 45]},
    {"offset": 1.0, "color": [246, 185, 185]},
]
ORANGE_HUE = [
    {"offset": 0.0, "color": [230, 45, 45]},
    {"offset": 0.5, "color": [240, 140, 35]},
    {"offset": 1.0, "color": [235, 215, 45]},
]
ORANGE_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [240, 140, 35]},
]
ORANGE_LUMINANCE = [
    {"offset": 0.0, "color": [60, 35, 8]},
    {"offset": 0.5, "color": [240, 140, 35]},
    {"offset": 1.0, "color": [250, 216, 181]},
]
YELLOW_HUE = [
    {"offset": 0.0, "color": [240, 140, 35]},
    {"offset": 0.5, "color": [235, 215, 45]},
    {"offset": 1.0, "color": [60, 185, 70]},
]
YELLOW_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [235, 215, 45]},
]
YELLOW_LUMINANCE = [
    {"offset": 0.0, "color": [58, 53, 11]},
    {"offset": 0.5, "color": [235, 215, 45]},
    {"offset": 1.0, "color": [248, 241, 185]},
]
GREEN_HUE = [
    {"offset": 0.0, "color": [235, 215, 45]},
    {"offset": 0.5, "color": [60, 185, 70]},
    {"offset": 1.0, "color": [40, 195, 205]},
]
GREEN_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [60, 185, 70]},
]
GREEN_LUMINANCE = [
    {"offset": 0.0, "color": [15, 46, 17]},
    {"offset": 0.5, "color": [60, 185, 70]},
    {"offset": 1.0, "color": [190, 231, 193]},
]
AQUA_HUE = [
    {"offset": 0.0, "color": [60, 185, 70]},
    {"offset": 0.5, "color": [40, 195, 205]},
    {"offset": 1.0, "color": [55, 95, 230]},
]
AQUA_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [40, 195, 205]},
]
AQUA_LUMINANCE = [
    {"offset": 0.0, "color": [10, 48, 51]},
    {"offset": 0.5, "color": [40, 195, 205]},
    {"offset": 1.0, "color": [183, 235, 238]},
]
BLUE_HUE = [
    {"offset": 0.0, "color": [40, 195, 205]},
    {"offset": 0.5, "color": [55, 95, 230]},
    {"offset": 1.0, "color": [140, 65, 220]},
]
BLUE_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [55, 95, 230]},
]
BLUE_LUMINANCE = [
    {"offset": 0.0, "color": [13, 23, 57]},
    {"offset": 0.5, "color": [55, 95, 230]},
    {"offset": 1.0, "color": [188, 201, 246]},
]
PURPLE_HUE = [
    {"offset": 0.0, "color": [55, 95, 230]},
    {"offset": 0.5, "color": [140, 65, 220]},
    {"offset": 1.0, "color": [220, 55, 190]},
]
PURPLE_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [140, 65, 220]},
]
PURPLE_LUMINANCE = [
    {"offset": 0.0, "color": [35, 16, 55]},
    {"offset": 0.5, "color": [140, 65, 220]},
    {"offset": 1.0, "color": [216, 191, 243]},
]
MAGENTA_HUE = [
    {"offset": 0.0, "color": [140, 65, 220]},
    {"offset": 0.5, "color": [220, 55, 190]},
    {"offset": 1.0, "color": [230, 45, 45]},
]
MAGENTA_SATURATION = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [220, 55, 190]},
]
MAGENTA_LUMINANCE = [
    {"offset": 0.0, "color": [55, 13, 47]},
    {"offset": 0.5, "color": [220, 55, 190]},
    {"offset": 1.0, "color": [243, 188, 233]},
]


class ImageColorMixer(io.ComfyNode):
    """Adjust eight colour bands of a batch, or mix it to black and white through them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageColorMixer",
            display_name="Image Color Mixer",
            search_aliases=[
                "WASImageColorMixer", "Image Color Mixer", "hsl", "color mixer", "hue saturation luminance",
                "black and white mix", "b&w mixer", "lightroom", "selective color",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Change the hue, saturation and brightness of eight colour ranges, red to "
                "magenta, without touching the rest: darken a blue sky, mute green foliage, warm "
                "skin. In black and white the luminance sliders set how light each colour prints."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to mix. Greys and near-greys are left alone, and an alpha channel passes through.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    tooltip="`color` = adjust each band; `black and white` = render grey, each band as light or dark as its luminance slider says. Hue and saturation do nothing in black and white.",
                ),
                io.Float.Input(
                    "red_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=RED_HUE,
                    tooltip="Turns reds toward magenta at -100 or toward orange at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "orange_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=ORANGE_HUE,
                    tooltip="Turns oranges toward red at -100 or toward yellow at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "yellow_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=YELLOW_HUE,
                    tooltip="Turns yellows toward orange at -100 or toward green at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "green_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=GREEN_HUE,
                    tooltip="Turns greens toward yellow at -100 or toward aqua at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "aqua_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=AQUA_HUE,
                    tooltip="Turns aquas toward green at -100 or toward blue at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "blue_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=BLUE_HUE,
                    tooltip="Turns blues toward aqua at -100 or toward purple at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "purple_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=PURPLE_HUE,
                    tooltip="Turns purples toward blue at -100 or toward magenta at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "magenta_hue", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=MAGENTA_HUE,
                    tooltip="Turns magentas toward purple at -100 or toward red at 100. 0 = unchanged.",
                ),
                io.Float.Input(
                    "red_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=RED_SATURATION,
                    tooltip="How strong reds are. -100 = grey; 0 = unchanged; 100 = twice as strong.",
                ),
                io.Float.Input(
                    "orange_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=ORANGE_SATURATION,
                    tooltip="How strong oranges are. -100 = grey; 0 = unchanged; 100 = twice as strong. -15 calms skin that reads too orange.",
                ),
                io.Float.Input(
                    "yellow_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=YELLOW_SATURATION,
                    tooltip="How strong yellows are. -100 = grey; 0 = unchanged; 100 = twice as strong.",
                ),
                io.Float.Input(
                    "green_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=GREEN_SATURATION,
                    tooltip="How strong greens are. -100 = grey; 0 = unchanged; 100 = twice as strong. -40 mutes foliage.",
                ),
                io.Float.Input(
                    "aqua_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=AQUA_SATURATION,
                    tooltip="How strong aquas are. -100 = grey; 0 = unchanged; 100 = twice as strong.",
                ),
                io.Float.Input(
                    "blue_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=BLUE_SATURATION,
                    tooltip="How strong blues are. -100 = grey; 0 = unchanged; 100 = twice as strong. -30 calms a loud sky.",
                ),
                io.Float.Input(
                    "purple_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=PURPLE_SATURATION,
                    tooltip="How strong purples are. -100 = grey; 0 = unchanged; 100 = twice as strong.",
                ),
                io.Float.Input(
                    "magenta_saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=MAGENTA_SATURATION,
                    tooltip="How strong magentas are. -100 = grey; 0 = unchanged; 100 = twice as strong.",
                ),
                io.Float.Input(
                    "red_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=RED_LUMINANCE,
                    tooltip="How light reds are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey.",
                ),
                io.Float.Input(
                    "orange_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=ORANGE_LUMINANCE,
                    tooltip="How light oranges are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey. 15 brightens skin.",
                ),
                io.Float.Input(
                    "yellow_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=YELLOW_LUMINANCE,
                    tooltip="How light yellows are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey.",
                ),
                io.Float.Input(
                    "green_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=GREEN_LUMINANCE,
                    tooltip="How light greens are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey. 20 lifts foliage.",
                ),
                io.Float.Input(
                    "aqua_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=AQUA_LUMINANCE,
                    tooltip="How light aquas are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey.",
                ),
                io.Float.Input(
                    "blue_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=BLUE_LUMINANCE,
                    tooltip="How light blues are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey. -40 darkens a blue sky.",
                ),
                io.Float.Input(
                    "purple_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=PURPLE_LUMINANCE,
                    tooltip="How light purples are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey.",
                ),
                io.Float.Input(
                    "magenta_luminance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=MAGENTA_LUMINANCE,
                    tooltip="How light magentas are. -100 = about a stop darker; 100 = a stop lighter. In black and white it sets their grey.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="Where the mix applies: white = fully, black = untouched, grey = partly. One mask per frame or one for all; unwired = the whole picture.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The mixed pictures."),
            ],
        )

    @classmethod
    def execute(
        cls, images, mode="color",
        red_hue=0.0, orange_hue=0.0, yellow_hue=0.0, green_hue=0.0, aqua_hue=0.0, blue_hue=0.0, purple_hue=0.0, magenta_hue=0.0,
        red_saturation=0.0, orange_saturation=0.0, yellow_saturation=0.0, green_saturation=0.0, aqua_saturation=0.0, blue_saturation=0.0, purple_saturation=0.0, magenta_saturation=0.0,
        red_luminance=0.0, orange_luminance=0.0, yellow_luminance=0.0, green_luminance=0.0, aqua_luminance=0.0, blue_luminance=0.0, purple_luminance=0.0, magenta_luminance=0.0, mask=None) -> io.NodeOutput:
        hues = [red_hue, orange_hue, yellow_hue, green_hue, aqua_hue, blue_hue, purple_hue, magenta_hue]
        saturations = [red_saturation, orange_saturation, yellow_saturation, green_saturation, aqua_saturation, blue_saturation, purple_saturation, magenta_saturation]
        luminances = [red_luminance, orange_luminance, yellow_luminance, green_luminance, aqua_luminance, blue_luminance, purple_luminance, magenta_luminance]
        mono = mode == "black and white"
        result = develop.mix(images, hues, saturations, luminances, mono)
        logger.info("Image Color Mixer mixed %d frame(s) in %s", int(images.shape[0]), mode)
        return io.NodeOutput(develop.masked(images, result, mask))
