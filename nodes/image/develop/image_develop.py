"""White balance, tone and presence, as the basic panel of a raw editor."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import develop

logger = log.get_logger("nodes.image.develop")

#: Slider colours: cool to warm, green to magenta, dark to light, grey to colour.
TEMPERATURE_STOPS = [
    {"offset": 0.0, "color": [70, 125, 255]},
    {"offset": 0.5, "color": [190, 190, 190]},
    {"offset": 1.0, "color": [255, 185, 55]},
]
TINT_STOPS = [
    {"offset": 0.0, "color": [70, 200, 70]},
    {"offset": 0.5, "color": [190, 190, 190]},
    {"offset": 1.0, "color": [225, 70, 220]},
]
LIGHT_STOPS = [
    {"offset": 0.0, "color": [0, 0, 0]},
    {"offset": 1.0, "color": [255, 255, 255]},
]
COLOUR_STOPS = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [255, 40, 40]},
]


class ImageDevelop(io.ComfyNode):
    """Develop a batch with the basic and presence sliders of a raw editor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDevelop",
            display_name="Image Develop",
            search_aliases=[
                "WASImageDevelop", "Image Develop", "develop", "lightroom", "basic panel",
                "exposure", "highlights", "shadows", "clarity", "dehaze", "vibrance", "raw editor",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Develop a picture the way a raw editor's basic panel does: white balance, "
                "exposure, contrast, highlights, shadows, whites, blacks, texture, clarity, "
                "dehaze, vibrance and saturation. Every slider at 0 hands the picture on unchanged."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to develop. Light above white is kept, and an alpha channel passes through.",
                ),
                io.Float.Input(
                    "temperature", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=TEMPERATURE_STOPS,
                    tooltip="-100 = much cooler and bluer; 0 = unchanged; 100 = much warmer and more amber. 10 to 20 warms a cold render.",
                ),
                io.Float.Input(
                    "tint", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=TINT_STOPS,
                    tooltip="-100 = toward green; 0 = unchanged; 100 = toward magenta. Corrects a green cast from fluorescent light.",
                ),
                io.Float.Input(
                    "exposure", default=0.0, min=-5.0, max=5.0, step=0.05,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="Stops of light. 1 = twice as bright; -1 = half; 0 = unchanged. Light pushed past white clips unless the picture carries more.",
                ),
                io.Float.Input(
                    "contrast", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="-100 = flat; 0 = unchanged; 100 = twice as steep through the midtones. Black and white stay put.",
                ),
                io.Float.Input(
                    "highlights", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="-100 = bright areas pulled down to recover a sky or a face in sun; 100 = bright areas lifted. Detail is held, so edges keep no halo.",
                ),
                io.Float.Input(
                    "shadows", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="100 = dark areas opened up to show what is in them; -100 = deepened. Pure black stays black; that is the blacks slider.",
                ),
                io.Float.Input(
                    "whites", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Sets the white point. 100 = brightest tones pushed to white; -100 = white pulled down to light grey.",
                ),
                io.Float.Input(
                    "blacks", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Sets the black point. -100 = deep tones crushed to black; 100 = black lifted to a matte grey, as on faded film.",
                ),
                io.Float.Input(
                    "texture", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Fine detail such as pores, fabric and foliage. -50 = smoother skin; 50 = crisper surfaces. Leaves larger shapes alone.",
                ),
                io.Float.Input(
                    "clarity", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Midtone local contrast. 30 = punchier, more defined; -30 = a soft, dreamy glow. Edges are followed, so it adds no halo.",
                ),
                io.Float.Input(
                    "dehaze", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="50 = haze, fog and flare cut out, colour and depth back; -50 = a soft veil of haze laid over the frame.",
                ),
                io.Float.Input(
                    "vibrance", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=COLOUR_STOPS,
                    tooltip="Raises muted colours most and skin tones least. 30 = livelier without orange faces; -30 = quieter colour.",
                ),
                io.Float.Input(
                    "saturation", default=0.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=COLOUR_STOPS,
                    tooltip="Every colour alike. -100 = black and white; 0 = unchanged; 100 = twice the colour.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="Where the develop applies: white = fully, black = untouched, grey = partly. One mask per frame or one for all; unwired = the whole picture.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The developed pictures."),
            ],
        )

    @classmethod
    def execute(
        cls, images, temperature=0.0, tint=0.0, exposure=0.0, contrast=0.0, highlights=0.0,
        shadows=0.0, whites=0.0, blacks=0.0, texture=0.0, clarity=0.0, dehaze=0.0,
        vibrance=0.0, saturation=0.0, mask=None) -> io.NodeOutput:
        result = develop.develop(
            images, temperature=temperature, tint=tint, exposure=exposure, contrast=contrast,
            highlights=highlights, shadows=shadows, whites=whites, blacks=blacks,
            texture=texture, clarity=clarity, dehaze=dehaze, vibrance=vibrance,
            saturation=saturation,
        )
        logger.info("Image Develop developed %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(develop.masked(images, result, mask))
