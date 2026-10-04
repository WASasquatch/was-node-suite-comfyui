"""Reduce an image to two tones at a brightness cut-off."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import pil2tensor, tensor2pil


class ImageThreshold(io.ComfyNode):
    """Turn every image in the batch into single-channel pure black and white."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Threshold",
            display_name="Image Threshold",
            search_aliases=["Image Threshold", "binarize", "black and white", "cutoff"],
            category="WAS Suite/Image/Process",
            description=(
                "Turn an image into flat black and white: anything brighter than the "
                "threshold becomes white and everything else becomes black."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to reduce. Colour is discarded; only brightness is read.",
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Brightness a pixel must reach to come out white, where 0.0 is black "
                        "and 1.0 is white. 0.5 splits the image at mid grey; lower it to keep "
                        "more of the image white, raise it to keep only the brightest "
                        "highlights."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The two-tone result, one image per image in. Every pixel is either "
                        "fully black or fully white."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, threshold=0.5) -> io.NodeOutput:
        from ....modules.image import scratch

        batch = None
        for index, plane in enumerate(image):
            frame = pil2tensor(cls.apply_threshold(tensor2pil(plane), threshold))
            if batch is None:
                batch = scratch.allocate(
                    (len(image),) + tuple(frame.shape[1:]),
                    node="Image Threshold",
                    advice="Passing fewer or smaller frames also fits it.",
                )
            batch[index] = frame[0]
        if batch is None:
            raise ValueError(
                "Image Threshold was handed an empty image batch. Wire in at least one image."
            )
        return io.NodeOutput(batch)

    @classmethod
    def apply_threshold(cls, input_image, threshold=0.5):
        """Split one image into black and white.

        Args:
            input_image: Source image, converted to ``L`` internally.
            threshold: Cut-off in the range 0.0 to 1.0, scaled to 0-255 and truncated, so
                a sample is white when it is greater than or equal to
                ``int(threshold * 255)``.

        Returns:
            A PIL image in mode ``L`` holding only 0 and 255.
        """
        grayscale_image = input_image.convert("L")
        threshold_value = int(threshold * 255)
        return grayscale_image.point(lambda x: 255 if x >= threshold_value else 0, mode="L")
