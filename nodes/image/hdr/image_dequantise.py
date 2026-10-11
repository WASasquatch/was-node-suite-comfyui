"""Rebuilding the levels a quantiser threw away."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic, hdr

#: Codes a 16-bit source is stored with, less one.
MAX_LEVELS = 65535

#: Passes the reconstruction may be asked for.
MAX_ROUNDS = 16


class ImageDequantise(io.ComfyNode):
    """Turn the flat steps of a banded gradient back into a gradient."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDequantise",
            display_name="Image Dequantise",
            search_aliases=[
                "WASImageDequantise", "Image Dequantise",
                "deband",
                "debanding",
                "banding",
                "posterisation",
                "bit depth",
                "gradient repair",
                "8-bit",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Rebuild the levels an 8-bit file threw away, so a band stored as one flat "
                "value comes back as the ramp it was cut from. Use it where a sky, soft shadow "
                "or vignette has banded, and before grading, tone mapping or an HDR pass on a "
                "PNG or JPEG."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The picture to rebuild; IMAGE. A batch is read as frames and each "
                        "one is rebuilt on its own."
                    ),
                ),
                io.Int.Input(
                    "levels",
                    default=hdr.EIGHT_BIT,
                    min=1,
                    max=MAX_LEVELS,
                    tooltip=(
                        "Codes the source was stored with, less one; INT. 255 = 8-bit, "
                        "1023 = 10-bit, 4095 = 12-bit. A figure above the source's own "
                        "leaves some of the banding in place."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    default=8.0,
                    min=hdr.SMALLEST_RADIUS,
                    max=hdr.LARGEST_RADIUS,
                    step=0.5,
                    tooltip=(
                        "Pixels the smoothing spans; FLOAT, 1.0 to 24.0. 2.0 = fine dither "
                        "noise; 8.0 = an ordinary banded sky; 24.0 = the widest bands. "
                        "Raise it until the steps stop showing."
                    ),
                ),
                io.Int.Input(
                    "rounds",
                    default=hdr.ROUNDS,
                    min=1,
                    max=MAX_ROUNDS,
                    tooltip=(
                        "Passes of smoothing; INT, 1 to 16. 1 = a quick pass that leaves the "
                        "widest bands; 6 = a smooth ramp; 16 = the most it recovers. Each "
                        "pass costs another blur."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The rebuilt picture; IMAGE, the same size, length and channel "
                        "count as it went in, on the same 0 to 1 scale."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, levels=hdr.EIGHT_BIT, radius=8.0, rounds=hdr.ROUNDS
    ) -> io.NodeOutput:
        """Rebuild the levels in every frame of the batch.

        Raises:
            ValueError: The input is not a batch of images.
        """
        folded = dynamic.fold(images)
        images = folded.images
        if getattr(images, "ndim", 0) != 4 or int(images.shape[0]) < 1:
            raise ValueError(
                "Image Dequantise needs an image to work on. Connect an image or a batch "
                "of frames."
            )
        channels = min(int(images.shape[-1]), 3)
        colour = images[..., :channels]
        if channels < 3:
            # A greyscale frame is filled out with copies of its last channel.
            colour = torch.cat([colour] + [colour[..., -1:]] * (3 - channels), dim=-1)
        answer = hdr.dequantise(
            colour, levels=int(levels), radius=float(radius), rounds=int(rounds)
        )[..., :channels]
        if int(images.shape[-1]) > 3:
            # Alpha and anything past it are carried through untouched.
            answer = torch.cat([answer, images[..., 3:]], dim=-1)
        return io.NodeOutput(dynamic.unfold(answer, folded))
