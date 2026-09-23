"""Encode an image into an inpainting latent, growing or shrinking the mask first."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "extras"


def modify_mask(mask: torch.Tensor, modify_by: int) -> torch.Tensor:
    """Grow or shrink the white area of a mask.

    Args:
        mask: A ``[B, 1, H, W]`` mask holding values between 0.0 and 1.0.
        modify_by: Distance in mask positions. Positive grows, negative shrinks, 0 returns
            the mask untouched.

    Returns:
        The adjusted mask, clamped to 0.0-1.0. Its edges are hard: the mask is rounded to
        0 or 1 before the convolution.
    """
    if modify_by == 0:
        return mask
    if modify_by > 0:
        kernel_size = 2 * modify_by + 1
        kernel_tensor = torch.ones((1, 1, kernel_size, kernel_size))
        padding = modify_by
        modified_mask = torch.clamp(
            torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1
        )
    else:
        kernel_size = 2 * abs(modify_by) + 1
        kernel_tensor = torch.ones((1, 1, kernel_size, kernel_size))
        padding = abs(modify_by)
        eroded_mask = torch.nn.functional.conv2d(1 - mask.round(), kernel_tensor, padding=padding)
        modified_mask = torch.clamp(1 - eroded_mask, 0, 1)
    return modified_mask


class VAEEncodeForInpaintWAS(io.ComfyNode):
    """Encode pixels and a mask into a latent carrying a noise mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVAEEncodeForInpaint",
            display_name="Inpainting VAE Encode",
            search_aliases=[
                "WASVAEEncodeForInpaint",
                "Inpainting VAE Encode (WAS)",
                "inpaint encode",
                "noise mask",
                "mask offset",
            ],
            category="WAS Suite/Latent",
            description=(
                "Encode an image into a latent for inpainting, with control over how far "
                "the mask grows or shrinks first. The masked pixels are flattened to mid "
                "grey before encoding so the sampler is not led by what was there, and the "
                "adjusted mask travels with the latent as its noise mask, which is what "
                "tells a KSampler which part to repaint. A positive mask_offset grows the "
                "painted area, so 6 reaches six pixels past what was drawn and hides the seam "
                "where new and old meet, which suits removing an object. A negative offset "
                "shrinks it, keeping more of the original, which suits touching up the middle "
                "of a region without disturbing its outline."
            ),
            inputs=[
                io.Image.Input(
                    "pixels",
                    tooltip=(
                        "The image to inpaint. Its width and height are cropped to the "
                        "nearest multiple of 8, taking the trim evenly from both sides. A "
                        "latent addresses the image 8 pixels at a time."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The VAE that turns the prepared image into a latent. Use the one "
                        "that belongs to the checkpoint the sampler runs, or the colours "
                        "shift."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "Which part is repainted. White is repainted, black is kept, and "
                        "grey is rounded to one or the other. It is stretched to the "
                        "image's size first, so a mask drawn at another resolution still "
                        "lines up."
                    ),
                ),
                io.Int.Input(
                    "mask_offset",
                    default=6,
                    min=-128,
                    max=128,
                    step=1,
                    tooltip=(
                        "How far the painted area grows or shrinks before encoding, in "
                        "pixels of the input image. 0 uses the mask exactly as drawn."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip=(
                        "The encoded latent with the adjusted mask attached as its noise "
                        "mask. Feed it to a KSampler, which will only replace the masked "
                        "part."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, pixels, vae, mask, mask_offset) -> io.NodeOutput:
        """Encode the masked image.

        Raises:
            ValueError: Nothing is connected to the vae input.
        """
        require_input(
            vae,
            "Inpainting VAE Encode (WAS)",
            "vae",
            "VAE",
            "Load VAE or a checkpoint loader",
            "VAE",
        )

        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(pixels.shape[1], pixels.shape[2]),
            mode="bilinear",
        )

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        mask_erosion = modify_mask(mask, mask_offset)

        # Pull the masked pixels to 0.5 about the mid point rather than to black: an
        # encoder reads a black patch as content and a flat mid grey as absence.
        m = (1.0 - mask_erosion.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        t = vae.encode(pixels)

        return io.NodeOutput(
            {"samples": t, "noise_mask": mask_erosion[:, :, :x, :y].round()}
        )
