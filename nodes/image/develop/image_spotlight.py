"""Light a subject and hold back its surroundings, frame by frame, from a subject matte."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import develop

logger = log.get_logger("nodes.image.develop")

LIGHT_STOPS = [
    {"offset": 0.0, "color": [0, 0, 0]},
    {"offset": 1.0, "color": [255, 255, 255]},
]
COLOUR_STOPS = [
    {"offset": 0.0, "color": [128, 128, 128]},
    {"offset": 1.0, "color": [255, 40, 40]},
]


class ImageSpotlight(io.ComfyNode):
    """Draw the eye to a subject by lighting it and holding back the rest."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageSpotlight",
            display_name="Image Spotlight",
            search_aliases=[
                "WASImageSpotlight", "Image Spotlight", "spotlight", "relight subject", "dodge and burn",
                "subject exposure", "draw attention", "local exposure",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Draw the eye to a subject: brighten and open it up, dim and quiet everything around it, "
                "and if wanted let its light spill past its edge. Takes a matte per frame, so a subject "
                "followed by Image Track Subject stays lit through a whole shot."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures or video frames to light. An alpha channel passes through.",
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="The subject: one matte per frame, or one for every frame. White = subject.",
                ),
                io.Float.Input(
                    "subject_exposure", default=0.6, min=-3.0, max=3.0, step=0.05,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="Stops of light on the subject. 0.6 = clearly lifted; 1 = twice as bright; 0 = unchanged.",
                ),
                io.Float.Input(
                    "subject_shadows", default=30.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Opens the subject's dark areas. 30 = faces and clothing read in a dim scene; 0 = unchanged.",
                ),
                io.Float.Input(
                    "subject_clarity", default=15.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Local contrast on the subject. 15 = it separates from murk or haze; -20 = softer.",
                ),
                io.Float.Input(
                    "surround_exposure", default=-0.5, min=-3.0, max=3.0, step=0.05,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=LIGHT_STOPS,
                    tooltip="Stops of light on everything else. -0.5 = held back; -1 = half as bright; 0 = unchanged.",
                ),
                io.Float.Input(
                    "surround_saturation", default=-20.0, min=-100.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.gradient_slider, gradient_stops=COLOUR_STOPS,
                    tooltip="Colour of everything else. -20 = quieter, so the subject's colour stands out; -100 = grey.",
                ),
                io.Float.Input(
                    "spill", default=0.0, min=0.0, max=100.0, step=1.0,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="How far the subject's light reaches past it, in hundredths of the short side. 0 = it stops at the subject's edge; 10 = a soft pool; 40 = a broad glow.",
                ),
                io.Float.Input(
                    "softness", default=0.3, min=0.0, max=10.0, step=0.1,
                    tooltip="Softens the subject's own edge, in hundredths of the short side. 0 = the matte as it is; 0.3 = just off crisp; 2 = very soft.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The lit pictures."),
                io.Mask.Output(
                    display_name="light",
                    tooltip="Where the subject's light fell, 1 on the subject fading to 0. Reuse it for a matching grade.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mask, subject_exposure=0.6, subject_shadows=30.0, subject_clarity=15.0,
                surround_exposure=-0.5, surround_saturation=-20.0, spill=0.0, softness=0.3) -> io.NodeOutput:
        lit, light = develop.spotlight(
            images, mask,
            {"exposure": subject_exposure, "shadows": subject_shadows, "clarity": subject_clarity},
            {"exposure": surround_exposure, "saturation": surround_saturation},
            spill, softness,
        )
        logger.info("Image Spotlight lit %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(lit, light)
