"""Shadow, midtone, highlight and global colour wheels, as a raw editor's colour grading panel."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import develop

logger = log.get_logger("nodes.image.develop")

#: A hue slider's colours, red round the wheel to red.
HUE_STOPS = [
    {"offset": 0.0, "color": [255, 0, 0]},
    {"offset": 1 / 6, "color": [255, 255, 0]},
    {"offset": 2 / 6, "color": [0, 255, 0]},
    {"offset": 3 / 6, "color": [0, 255, 255]},
    {"offset": 4 / 6, "color": [0, 0, 255]},
    {"offset": 5 / 6, "color": [255, 0, 255]},
    {"offset": 1.0, "color": [255, 0, 0]},
]
LIGHT_STOPS = [
    {"offset": 0.0, "color": [0, 0, 0]},
    {"offset": 1.0, "color": [255, 255, 255]},
]


class ImageColorGrade(io.ComfyNode):
    """Tint the tonal ranges of a batch with colour grading wheels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageColorGrade",
            display_name="Image Color Grading",
            search_aliases=[
                "WASImageColorGrade", "Image Color Grading", "color grading", "color wheels",
                "split toning", "teal and orange", "lift gamma gain", "lightroom",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Tint the shadows, midtones and highlights each their own colour, plus the whole "
                "picture, by dragging four colour wheels on the node. Teal shadows under warm "
                "highlights, a cool night or a golden glow take a drag each."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to grade. An alpha channel passes through.",
                ),
                io.Float.Input(
                    "shadows_hue", default=0.0, advanced=True, min=0.0, max=360.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=HUE_STOPS,
                    tooltip="Colour the shadows lean to, in degrees. 0 = red, 30 = orange, 60 = yellow, 120 = green, 190 = teal, 240 = blue, 300 = magenta.",
                ),
                io.Float.Input(
                    "shadows_saturation", default=0.0, advanced=True, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How strongly the shadows take that colour. 0 = untinted; 15 = subtle; 40 = strong.",
                ),
                io.Float.Input(
                    "shadows_luminance", default=0.0, advanced=True, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="-100 = shadows deeper; 0 = unchanged; 100 = shadows lifted.",
                ),
                io.Float.Input(
                    "midtones_hue", default=0.0, advanced=True, min=0.0, max=360.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=HUE_STOPS,
                    tooltip="Colour the midtones lean to, in degrees. 0 = red, 30 = orange, 60 = yellow, 120 = green, 190 = teal, 240 = blue, 300 = magenta.",
                ),
                io.Float.Input(
                    "midtones_saturation", default=0.0, advanced=True, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How strongly the midtones take that colour. 0 = untinted; 10 = subtle; 30 = strong.",
                ),
                io.Float.Input(
                    "midtones_luminance", default=0.0, advanced=True, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="-100 = midtones darker; 0 = unchanged; 100 = midtones brighter.",
                ),
                io.Float.Input(
                    "highlights_hue", default=0.0, advanced=True, min=0.0, max=360.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=HUE_STOPS,
                    tooltip="Colour the highlights lean to, in degrees. 0 = red, 30 = orange, 45 = gold, 60 = yellow, 190 = teal, 240 = blue.",
                ),
                io.Float.Input(
                    "highlights_saturation", default=0.0, advanced=True, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How strongly the highlights take that colour. 0 = untinted; 15 = subtle; 40 = strong.",
                ),
                io.Float.Input(
                    "highlights_luminance", default=0.0, advanced=True, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="-100 = highlights darker; 0 = unchanged; 100 = highlights brighter.",
                ),
                io.Float.Input(
                    "global_hue", default=0.0, advanced=True, min=0.0, max=360.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=HUE_STOPS,
                    tooltip="Colour the whole picture leans to, in degrees. 0 = red, 30 = orange, 60 = yellow, 190 = teal, 240 = blue.",
                ),
                io.Float.Input(
                    "global_saturation", default=0.0, advanced=True, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How strongly the whole picture takes that colour. 0 = untinted; 10 = a wash; 30 = a strong cast.",
                ),
                io.Float.Input(
                    "global_luminance", default=0.0, advanced=True, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="-100 = the whole picture darker; 0 = unchanged; 100 = brighter.",
                ),
                io.Float.Input(
                    "blending", default=50.0, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How far the three ranges overlap. 0 = they meet sharply; 50 = a smooth handover; 100 = each tint reaches across most of the picture.",
                ),
                io.Float.Input(
                    "balance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Where shadows end and highlights begin. -100 = most of the picture counts as shadow; 0 = even; 100 = most counts as highlight.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="Where the grade applies: white = fully, black = untouched, grey = partly. One mask per frame or one for all; unwired = the whole picture.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The graded pictures."),
            ],
        )

    @classmethod
    def execute(
        cls, images, shadows_hue=0.0, shadows_saturation=0.0, shadows_luminance=0.0,
        midtones_hue=0.0, midtones_saturation=0.0, midtones_luminance=0.0,
        highlights_hue=0.0, highlights_saturation=0.0, highlights_luminance=0.0,
        global_hue=0.0, global_saturation=0.0, global_luminance=0.0,
        blending=50.0, balance=0.0, mask=None) -> io.NodeOutput:
        wheels = {
            "shadows": (shadows_hue, shadows_saturation, shadows_luminance),
            "midtones": (midtones_hue, midtones_saturation, midtones_luminance),
            "highlights": (highlights_hue, highlights_saturation, highlights_luminance),
            "global": (global_hue, global_saturation, global_luminance),
        }
        result = develop.grade(images, wheels, blending, balance)
        logger.info("Image Color Grading graded %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(develop.masked(images, result, mask))
