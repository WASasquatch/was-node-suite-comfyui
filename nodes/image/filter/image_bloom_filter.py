"""Bloom glow around bright areas."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.image.batching import as_batch

#: Fewest runs a batch is split into, each advancing the progress bar.
PROGRESS_RUNS = 8


class ImageBloomFilter(io.ComfyNode):
    """Add a soft glow that spills out of an image's bright areas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Bloom Filter",
            display_name="Image Bloom Filter",
            search_aliases=["Image Bloom Filter", "bloom", "glow", "halation", "light bleed"],
            category="WAS Suite/Image/Filter",
            description=(
                "Add a soft halo of light around the bright, detailed parts of an image, "
                "the way a camera lens blooms when it points at a light source."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image or batch to add the glow to. A batch is processed together "
                        "on the GPU where there is one, and a fourth channel passes through."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    default=10,
                    min=0.0,
                    max=1024,
                    step=0.1,
                    tooltip=(
                        "How far the glow spreads, in pixels. 2 gives a tight sheen on edges, "
                        "10 a visible halo, 50 a broad wash of light over the whole frame. 0 "
                        "leaves the image unchanged."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=1,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip=(
                        "How bright the glow is. 0.0 adds nothing, 0.3 is a subtle lift, 1.0 is "
                        "the full effect."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with the glow screened over it."),
            ],
        )

    @classmethod
    def execute(cls, image, radius, intensity) -> io.NodeOutput:
        """Bloom the whole batch on ComfyUI's compute device, a run of frames at a time.

        Raises:
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import accelerate, pillow_exact

        frames = as_batch(image)
        scale = dynamic.factor(frames)
        return io.NodeOutput(accelerate.run_frames(
            frames,
            lambda run: pillow_exact.bloom(run, float(radius), float(intensity), scale),
            factor=pillow_exact.working_copies(frames, float(radius) * 2.0),
            node="Image Bloom Filter",
            advice="Passing fewer frames also fits it.",
            runs=PROGRESS_RUNS,
        ))
