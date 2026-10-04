"""Erode, dilate, open, close and their differences on a batch of images."""

from __future__ import annotations

import comfy.model_management
import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import morphology

logger = log.get_logger("nodes.image.filter")

#: Frames worked on at once, which bounds the padded copy each pass makes.
CHUNK_FRAMES = 16


class ImageMorphology(io.ComfyNode):
    """Grey-level morphology with a square kernel, run as separable passes on the GPU."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageMorphology",
            display_name="Image Morphology",
            search_aliases=[
                "WASImageMorphology",
                "Image Morphology",
                "Apply Morphology",
                "erode",
                "dilate",
                "open",
                "close",
                "top hat",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Erode, dilate, open, close, or take the gradient, top hat or bottom hat of a "
                "batch of images with a square kernel. Gives the same pixels as core Apply "
                "Morphology, in milliseconds and without running out of memory on large "
                "kernels or long batches."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The frames to reshape. Every channel is worked on separately.",
                ),
                io.Combo.Input(
                    "operation",
                    options=list(morphology.OPERATIONS),
                    tooltip=(
                        "`erode` shrinks bright areas, `dilate` grows them, `open` removes "
                        "bright specks, `close` fills dark gaps, `gradient` keeps edges, "
                        "`top_hat` keeps bright detail smaller than the kernel, `bottom_hat` "
                        "keeps dark detail smaller than it."
                    ),
                ),
                io.Int.Input(
                    "kernel_size",
                    default=3,
                    min=3,
                    max=999,
                    step=1,
                    tooltip="Side of the square kernel in pixels. `3` touches single pixels, `15` reaches across small shapes.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip="The frames after the operation, the size they went in at.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, operation, kernel_size) -> io.NodeOutput:
        """Run the operation over the batch in chunks.

        Args:
            image: ``(batch, height, width, channels)`` images.
            operation: One of the morphology operations.
            kernel_size: Kernel side in pixels.

        Returns:
            The reshaped images on the intermediate device.

        Raises:
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch

        device = comfy.model_management.get_torch_device()
        keep = torch.device(comfy.model_management.intermediate_device())
        if keep.type == "cpu":
            reshaped = scratch.allocate(
                tuple(image.shape),
                image.dtype,
                node="Image Morphology",
                advice="Passing fewer frames also fits it.",
            )
        else:
            reshaped = torch.empty(tuple(image.shape), dtype=image.dtype, device=keep)
        for start in range(0, int(image.shape[0]), CHUNK_FRAMES):
            planes = image[start:start + CHUNK_FRAMES].to(device).movedim(-1, 1)
            shaped = morphology.morph(planes, operation, int(kernel_size))
            reshaped[start:start + CHUNK_FRAMES] = shaped.movedim(1, -1)
        return io.NodeOutput(reshaped)
