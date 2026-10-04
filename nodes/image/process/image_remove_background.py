"""Cut a flat background or foreground out with a brightness threshold."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import pil2tensor, tensor2pil


class ImageRemoveBackground(io.ComfyNode):
    """Make the light or dark part of each image transparent, returning RGBA."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Remove Background (Alpha)",
            display_name="Image Remove Background (Threshold)",
            search_aliases=[
                "Image Remove Background (Threshold)",
                "Image Remove Background (Alpha)",
                "cut out",
                "transparency",
                "alpha",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Make the brightest or darkest part of an image transparent, judged on "
                "brightness alone. Best on a flat backdrop, such as a white studio sweep."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The images to cut out. Each is handled on its own."),
                io.Combo.Input(
                    "mode",
                    options=["background", "foreground"],
                    tooltip=(
                        "Which part of the image to clear, decided on brightness alone. "
                        "`background` clears every pixel at or below the threshold, so a dark "
                        "backdrop goes and the light part is kept. `foreground` clears every "
                        "pixel at or above it, so a white studio sweep goes instead."
                    ),
                ),
                io.Int.Input(
                    "threshold",
                    default=127,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Brightness the cut is made at, 0 for black and 255 for white; 127 cuts "
                        "at mid grey. In `background` mode everything this dark or darker is "
                        "cleared, so raising it clears more of the image; in `foreground` mode "
                        "everything this bright or brighter is cleared, so raising it clears less."
                    ),
                ),
                io.Int.Input(
                    "threshold_tolerance",
                    default=2,
                    min=1,
                    max=24,
                    step=1,
                    tooltip=(
                        "How much the brightness is blurred before the cut, in pixels. Small "
                        "values follow the edge closely but keep speckles; larger values give a "
                        "smoother outline that creeps in over fine detail such as hair."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The images with an alpha channel, transparent wherever the threshold "
                        "cut them away."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode="background", threshold=127, threshold_tolerance=2) -> io.NodeOutput:
        return io.NodeOutput(cls.remove_background(images, mode, threshold, threshold_tolerance))

    @classmethod
    def remove_background(cls, image, mode, threshold, threshold_tolerance):
        """Give every image in a batch an alpha channel cut at a brightness threshold.

        Args:
            image: Batch of image tensors.
            mode: ``"background"`` clears every pixel at or below ``threshold``, anything
                else clears every pixel at or above it.
            threshold: Brightness the cut is made at, 0-255.
            threshold_tolerance: Gaussian blur radius applied to the luminance before the
                cut, in pixels.

        Returns:
            A batch tensor of ``RGBA`` images.

        Raises:
            ValueError: The batch holds no image.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from PIL import ImageFilter, ImageOps

        from ....modules.image import scratch

        # One cut-off serves every image of the batch.
        cutoff = 255 - threshold if mode == "background" else threshold

        images = None
        for index, plane in enumerate(image):
            img = tensor2pil(plane)
            grayscale_image = img.convert("L")
            if mode == "background":
                grayscale_image = ImageOps.invert(grayscale_image)
            blurred_image = grayscale_image.filter(
                ImageFilter.GaussianBlur(radius=threshold_tolerance)
            )
            binary_image = blurred_image.point(lambda x: 0 if x < cutoff else 255, "1")
            mask = binary_image.convert("L")
            inverted_mask = ImageOps.invert(mask)
            transparent_image = img.copy()
            transparent_image.putalpha(inverted_mask)
            frame = pil2tensor(transparent_image)
            if images is None:
                images = scratch.allocate(
                    (len(image),) + tuple(frame.shape[1:]),
                    node="Image Remove Background (Threshold)",
                    advice="Passing fewer or smaller frames also fits it.",
                )
            images[index] = frame[0]

        if images is None:
            raise ValueError(
                "Image Remove Background (Threshold) was handed an empty image batch. Wire in "
                "at least one image."
            )
        return images
