"""Generate a linear colour gradient image from a list of stops."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import pil2tensor


class ImageGenerateGradient(io.ComfyNode):
    """Draw a linear gradient through a list of ``position:r,g,b`` colour stops."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Generate Gradient",
            display_name="Image Generate Gradient",
            search_aliases=[
                "Image Generate Gradient",
                "gradient",
                "ramp",
                "colour ramp",
                "background",
            ],
            category="WAS Suite/Image/Generate",
            description=(
                "Generate a horizontal or vertical colour gradient from a list of colour "
                "stops, for backgrounds, sky ramps and gradient maps. Raise tolerance for a "
                "banded, posterised ramp."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip="Width of the generated image, in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip="Height of the generated image, in pixels.",
                ),
                io.Combo.Input(
                    "direction",
                    options=["horizontal", "vertical"],
                    tooltip=(
                        "Which way the colours run. `horizontal` puts the first stop at the "
                        "left edge and the last at the right; `vertical` runs top to bottom."
                    ),
                ),
                io.Int.Input(
                    "tolerance",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Rounds every colour channel to a multiple of this number, which "
                        "turns the smooth ramp into visible bands. 0 leaves it smooth, 32 "
                        "gives eight steps per channel, 64 gives four. A blur is applied "
                        "afterwards, so the edges of the bands stay soft."
                    ),
                ),
                io.String.Input(
                    "gradient_stops",
                    default="0:255,0,0\n25:255,255,255\n50:0,255,0\n75:0,0,255",
                    multiline=True,
                    tooltip=(
                        "One stop per line as position:red,green,blue, so '0:255,0,0' puts "
                        "pure red at the start. Positions run 0 to 100, channels 0 to 255. "
                        "Colour holds flat beyond the end stops; an unreadable line is skipped."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The gradient, as an RGB image."),
            ],
        )

    @classmethod
    def execute(cls, width, height, direction, tolerance, gradient_stops) -> io.NodeOutput:
        from ....modules.image.gradient import gradient, parse_gradient_stops

        image = gradient(
            (width, height), direction, parse_gradient_stops(gradient_stops), tolerance
        )

        return io.NodeOutput(pil2tensor(image))
