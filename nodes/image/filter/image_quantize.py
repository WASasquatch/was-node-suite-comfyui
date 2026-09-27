"""Reduce each image in a batch to a palette of its own colours."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules.image import quantize


class ImageQuantize(io.ComfyNode):
    """Quantize Image with the frames of a batch reduced side by side."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageQuantize",
            display_name="Image Quantize",
            search_aliases=[
                "WASImageQuantize",
                "Image Quantize",
                "Quantize Image",
                "palette",
                "reduce colors",
                "dither",
                "posterize",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Reduce every image to a palette of its own most used colours, with or without "
                "dithering. Gives the same pixels as core Quantize Image, with the frames of a "
                "batch worked on at the same time."
            ),
            inputs=[
                io.Image.Input("image", tooltip="The frames to reduce. Each gets its own palette."),
                io.Int.Input(
                    "colors", default=256, min=1, max=256, step=1,
                    tooltip="Palette size. `256` is nearly invisible, `16` is posterised, `2` is two tones.",
                ),
                io.Combo.Input(
                    "dither", options=list(quantize.DITHERS),
                    tooltip=(
                        "`none` gives flat bands, `floyd-steinberg` scatters noise to hide "
                        "them, `bayer-2` to `bayer-16` lay a regular pattern, coarser as the "
                        "number rises."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE", tooltip="The reduced frames, alpha kept as it was."),
            ],
        )

    @classmethod
    def execute(cls, image, colors=256, dither="none") -> io.NodeOutput:
        """Quantize each frame.

        Args:
            image: ``(batch, height, width, channels)`` images.
            colors: Palette size.
            dither: A dither choice.

        Returns:
            The reduced batch.
        """
        codes = (image[..., :3] * 255).to(torch.uint8).cpu().numpy()
        reduced = quantize.frames(list(codes), int(colors), dither)
        result = torch.from_numpy(np.stack(reduced)).float() / 255
        result = result.to(image.device)
        if image.shape[-1] == 4:
            result = torch.cat((result, image[..., 3:]), dim=-1)
        return io.NodeOutput(result)
