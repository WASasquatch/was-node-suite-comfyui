"""Brightness, contrast, saturation, sharpness, blur and edge adjustments in one node."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.image.batching import as_batch
from ....modules.interface import preview

#: Fewest runs a batch is split into, each advancing the progress bar.
PROGRESS_RUNS = 8


class ImageFilterAdjustments(io.ComfyNode):
    """Apply the common tonal and softening adjustments to an image in one pass."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Filter Adjustments",
            display_name="Image Filter Adjustments",
            search_aliases=[
                "Image Filter Adjustments",
                "brightness",
                "contrast",
                "saturation",
                "sharpness",
                "blur",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "The everyday image controls in one node: brightness, contrast, "
                "saturation, sharpness, two kinds of blur, and edge or detail enhancement. "
                "Each one is skipped at its neutral setting."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image or batch to adjust. A batch is adjusted together on the GPU "
                        "where there is one."
                    ),
                ),
                io.Float.Input(
                    "brightness",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Added to every pixel, where 1.0 is the whole black-to-white range. 0.0 "
                        "leaves the image alone, 0.1 lifts it slightly, -0.25 darkens it "
                        "noticeably."
                    ),
                ),
                io.Float.Input(
                    "contrast",
                    default=1.0,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Every pixel is multiplied by this, so it brightens as it separates. "
                        "1.0 leaves the image alone, 1.3 is a firm push, 0.7 flattens it, and "
                        "negative values invert and clip it to mostly black."
                    ),
                ),
                io.Float.Input(
                    "saturation",
                    default=1.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Colour strength. 1.0 leaves the image alone, 0.0 gives black and "
                        "white, 2.0 doubles the colour, 5.0 is poster-like."
                    ),
                ),
                io.Float.Input(
                    "sharpness",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Edge crispness. 1.0 leaves the image alone, 2.0 sharpens, 0.0 softens, "
                        "and negative values overshoot into an embossed outline."
                    ),
                ),
                io.Int.Input(
                    "blur",
                    default=0,
                    min=0,
                    max=16,
                    step=1,
                    tooltip=(
                        "How many passes of a small fixed blur to run. 0 skips it, 1 is a slight "
                        "softening, 16 is heavy. For a specific radius use gaussian_blur instead."
                    ),
                ),
                io.Float.Input(
                    "gaussian_blur",
                    default=0.0,
                    min=0.0,
                    max=1024.0,
                    step=0.1,
                    tooltip=(
                        "Blur radius in pixels. 0.0 skips it, 2 is a gentle soften, 25 removes "
                        "all detail and leaves colour shapes."
                    ),
                ),
                io.Float.Input(
                    "edge_enhance",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of an edge-enhanced copy is mixed in. 0.0 skips it, 0.3 picks "
                        "the outlines out gently, 1.0 uses the enhanced copy outright and looks "
                        "harsh."
                    ),
                ),
                io.Boolean.Input(
                    "detail_enhance",
                    default=False,
                    tooltip=(
                        "Run a fixed detail filter at the end, which is a mild local-contrast "
                        "boost with no strength setting. `off` skips it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The adjusted image, or the input unchanged."),
            ],
        )

    @classmethod
    def execute(cls, image, brightness, contrast, saturation, sharpness, blur, gaussian_blur,
                edge_enhance, detail_enhance) -> io.NodeOutput:
        """Adjust the whole batch on ComfyUI's compute device, a run of frames at a time.

        Raises:
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import accelerate, pillow_exact

        settings = pillow_exact.Adjustments(
            brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance,
            detail_enhance,
        )

        # Publishes the input for a preview panel to apply the settings to.
        preview.publish(image)

        frames = as_batch(image)
        scale = dynamic.factor(frames)
        return io.NodeOutput(accelerate.run_frames(
            frames,
            lambda run: pillow_exact.adjust(run, settings, scale),
            factor=pillow_exact.working_copies(frames, float(gaussian_blur)),
            node="Image Filter Adjustments",
            advice="Passing fewer frames also fits it.",
            runs=PROGRESS_RUNS,
        ))
