"""Recolour an image by brightness, looking each pixel up in a gradient."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import (
    broadcast_image_planes,
    image_planes,
    stack_images,
    tensor2pil,
)
from ....modules.image import dynamic

#: Stops the widget opens with, a plain black to white ramp.
DEFAULT_STOPS = "0:0,0,0\n100:255,255,255"


class ImageGradientMapNative(io.ComfyNode):
    """Map luminance to a gradient built from stops, or sampled from a picture."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageGradientMapNative",
            display_name="Image Gradient Map",
            search_aliases=[
                'WASImageGradientMapNative',
                "Image Gradient Map Native",
                "gradient map",
                "duotone",
                "colour ramp",
                "recolor",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Recolour an image by brightness: the darkest pixels take the first colour "
                "of a gradient, the brightest take the last, and everything else lands in "
                "between. The gradient comes from the stops typed on the node, or from a "
                "picture connected to gradient_image. Needs nothing installed."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to recolour. Only its brightness is used, so its own "
                        "colours are discarded. A batch is handled one image at a time."
                    ),
                ),
                io.Boolean.Input(
                    "flip_left_right",
                    default=False,
                    tooltip=(
                        "`true` sends the ramp's last colour to the shadows and its first to "
                        "the highlights. `false` reads the ramp from its first colour to its last."
                    ),
                ),
                io.String.Input(
                    "gradient_stops",
                    default="0:0,0,0\n100:255,255,255",
                    multiline=True,
                    tooltip=(
                        "One stop per line as position:red,green,blue, so '0:0,0,0' puts "
                        "black in the shadows. Positions run 0 to 100 from shadows to "
                        "highlights and channels 0 to 255. The editor below writes here, and "
                        "this is what a run reads. Not used while gradient_image is connected."
                    ),
                ),
                io.Image.Input(
                    "gradient_image",
                    optional=True,
                    tooltip=(
                        "A gradient picture to read instead of the stops. The axis along which "
                        "colour changes most is read, so top to bottom works as well as left to "
                        "right. A batch pairs with the image batch frame by frame."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The recoloured image, the same size as the source."),
            ],
        )

    @classmethod
    def execute(
        cls, image, flip_left_right=False, gradient_stops=DEFAULT_STOPS,
        gradient_image=None,
    ) -> io.NodeOutput:
        """Recolour every frame by brightness.

        Raises:
            ValueError: No gradient was connected and no stop could be read.
        """
        from ....modules.image.gradient import gradient_map, parse_gradient_stops

        turned = str(flip_left_right).lower() == "true"
        image = dynamic.fold(image).images
        if gradient_image is not None:
            return io.NodeOutput(stack_images([
                gradient_map(tensor2pil(plane), tensor2pil(ramp), turned)
                for plane, ramp in broadcast_image_planes(image, gradient_image)
            ]))

        stops = parse_gradient_stops(gradient_stops)
        return io.NodeOutput(stack_images([
            gradient_map(tensor2pil(plane), reverse=turned, stops=stops)
            for plane in image_planes(image)
        ]))
