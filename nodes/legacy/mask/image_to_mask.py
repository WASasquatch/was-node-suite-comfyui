"""Convert one channel of an image into a mask."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules.convert.tensors import image_planes, tensor2pil
from ....modules.interface import mask_report

REQUIRES = "core_dupes"


class ImageToLatentMask(io.ComfyNode):
    """Pull a single channel out of every image in a batch as one ``(height, width)`` mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image to Latent Mask",
            display_name="Image to Latent Mask",
            search_aliases=["Image to Latent Mask", "image to mask", "channel to mask"],
            category="WAS Suite/Image/Masking",
            description="Deprecated: use core ImageToMask instead, or ImageColorToMask to "
            "key on a colour. Takes one channel of each image in a batch as a mask: "
            "`alpha` for transparency (solid white for an opaque image), or `red`, `green` "
            "or `blue` for a mask painted in one colour.",
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to read a channel out of. Each image in the batch produces "
                        "one mask."
                    ),
                ),
                io.Combo.Input(
                    "channel",
                    options=["alpha", "red", "green", "blue"],
                    tooltip=(
                        "Which channel supplies the mask, taken at its own brightness so a "
                        "half-lit channel gives a half-strength mask."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "One mask per input image, taken from the chosen channel, as a batch "
                        "the same length as the images that came in."
                    ),
                )
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, images, channel) -> io.NodeOutput:
        masks = []
        for image in image_planes(images):
            r, g, b, a = tensor2pil(image).convert("RGBA").split()
            channel_image = {"red": r, "green": g, "blue": b, "alpha": a}[channel]
            mask = torch.from_numpy(
                np.array(channel_image.convert("L")).astype(np.float32) / 255.0
            )
            masks.append(mask)

        # A single image keeps the unbatched 2D mask it has always returned; more than one
        # takes a batch axis, so the masks stay one per image instead of being joined into
        # one mask of stacked rows.
        stacked = masks[0] if len(masks) == 1 else torch.stack(masks, dim=0)
        mask_report.publish(None, stacked)
        return io.NodeOutput(stacked)
