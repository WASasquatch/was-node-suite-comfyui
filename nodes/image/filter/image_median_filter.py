"""Edge-preserving smoothing."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


class ImageMedianFilter(io.ComfyNode):
    """Smooth an image without softening the boundaries between areas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Median Filter",
            display_name="Image Median Filter",
            search_aliases=[
                "Image Median Filter",
                "bilateral",
                "smooth",
                "denoise",
                "skin softening",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Even out flat areas of an image while keeping its outlines crisp. Good for "
                "smoothing skin or removing noise without turning the whole picture soft."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to smooth. A batch is handled one image at a time.",
                ),
                io.Int.Input(
                    "diameter",
                    default=2.0,
                    min=0.1,
                    max=255,
                    step=1,
                    tooltip=(
                        "How wide an area each output pixel is averaged over, in pixels. 2 is a "
                        "gentle clean-up, 15 visibly flattens texture, and large values are "
                        "very slow because the cost grows with the square of this."
                    ),
                ),
                io.Float.Input(
                    "sigma_color",
                    default=10.0,
                    min=-255.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "How different in colour two pixels may be and still be mixed, on a "
                        "0-255 scale. Small values such as 10 mix only near-identical colours "
                        "and so preserve every edge; 150 mixes across most colours and blurs "
                        "the picture like an ordinary blur."
                    ),
                ),
                io.Float.Input(
                    "sigma_space",
                    default=10.0,
                    min=-255.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "How far away in pixels a neighbour may be and still count. Larger "
                        "values pull in more distant pixels, up to the limit set by diameter."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The smoothed image, still in RGB."),
            ],
        )

    @classmethod
    def execute(cls, image, diameter, sigma_color, sigma_space) -> io.NodeOutput:
        from ....modules.image.basic import medianFilter

        return io.NodeOutput(filtered_planes(
            image,
            lambda plane: medianFilter(plane, diameter, sigma_color, sigma_space),
            parallel=False,
        ))
