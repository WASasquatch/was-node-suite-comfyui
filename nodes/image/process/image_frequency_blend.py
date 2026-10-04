"""Taking detail from one picture and structure from another, split by frequency."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.convert.tensors import image_planes
from ....modules.image import dynamic

#: Rows of the RGB to YUV matrix, and its inverse. The luma row is the usual Rec.601 weighting;
#: only luma is fused, so chroma travels from the structural picture untouched and no colour
#: shift can be introduced by the blend.
TO_YUV = (
    (0.299, 0.587, 0.114),
    (-0.14713, -0.28886, 0.436),
    (0.615, -0.51499, -0.10001),
)
FROM_YUV = (
    (1.0, 0.0, 1.13983),
    (1.0, -0.39465, -0.5806),
    (1.0, 2.03211, 0.0),
)


class ImageFrequencyBlend(io.ComfyNode):
    """Keep one picture's structure and take its detail from another."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageFrequencyBlend",
            display_name="Image Frequency Blend",
            search_aliases=[
                "WASImageFrequencyBlend", "Image Frequency Blend",
                "frequency", "detail transfer", "butterworth", "high pass merge",
                "fuse sharp and consistent",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Split two pictures into low and high frequencies and keep the low ones from "
                "the first while taking the high ones from whichever has more detail there. "
                "For a steady pass and a sharper pass of the same shot, this keeps the "
                "steadiness and gains the detail. Colour comes from the first picture only."
            ),
            inputs=[
                io.Image.Input(
                    "consistent",
                    tooltip=(
                        "The picture to keep. Its low frequencies and all of its colour are "
                        "carried through, so this is the one that decides structure and tone."
                    ),
                ),
                io.Image.Input(
                    "sharp",
                    tooltip=(
                        "The picture to take detail from. Only its luma high frequencies are "
                        "used, and only where it has more of them than the first picture. Must "
                        "be the same size and hold the same number of frames."
                    ),
                ),
                io.Float.Input(
                    "cutoff",
                    default=0.20,
                    min=0.01,
                    max=0.49,
                    step=0.01,
                    tooltip=(
                        "Where low stops and high starts, as a fraction of the picture's own "
                        "frequency range rather than in pixels. Lower keeps more of the first "
                        "picture; higher hands more of the image over to the second."
                    ),
                ),
                io.Int.Input(
                    "order",
                    default=2,
                    min=1,
                    max=8,
                    step=1,
                    tooltip=(
                        "How abruptly the split happens. 1 is a gentle roll-off, high values "
                        "approach a hard cut and can ring around strong edges."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip=(
                        "How much of the blended detail to add back. 0 returns the first "
                        "picture blurred to the cutoff, 1 is the intended amount, and above 1 "
                        "exaggerates."
                    ),
                ),
                io.Int.Input(
                    "border",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Pixels around the edge left as the first picture. A transform of a "
                        "whole frame rings slightly at its boundary, and this hides it. 0 "
                        "blends right to the edge."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The first picture's structure carrying the second one's detail.",
                ),
            ],
        )

    @classmethod
    def execute(cls, consistent, sharp, cutoff, order, strength, border) -> io.NodeOutput:
        """Blend every frame and answer them as one batch.

        Raises:
            ValueError: The two inputs are different sizes, or one is empty.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch

        steady = image_planes(consistent)
        detailed = image_planes(sharp)
        if not steady or not detailed:
            raise ValueError(
                "Image Frequency Blend needs a picture on both inputs; one of them is empty."
            )
        if steady[0].shape[:2] != detailed[0].shape[:2]:
            first, second = steady[0].shape, detailed[0].shape
            raise ValueError(
                f"Image Frequency Blend fuses two pictures pixel for pixel, so they must be the "
                f"same size. consistent is {first[1]}x{first[0]} and sharp is "
                f"{second[1]}x{second[0]}. Resize one to match the other."
            )

        # A shorter input repeats its last frame.
        frames = max(len(steady), len(detailed))
        blended = None
        for index in range(frames):
            frame = cls.blend_one(
                steady[min(index, len(steady) - 1)],
                detailed[min(index, len(detailed) - 1)],
                cutoff, order, strength, border,
            )
            if blended is None:
                shape = (frames,) + tuple(frame.shape)
                blended = (
                    scratch.allocate(
                        shape,
                        frame.dtype,
                        node="Image Frequency Blend",
                        advice="Passing fewer frames also fits it.",
                    )
                    if frame.device.type == "cpu"
                    else torch.empty(shape, dtype=frame.dtype, device=frame.device)
                )
            blended[index] = frame
        return io.NodeOutput(blended)

    @classmethod
    def blend_one(cls, steady, detailed, cutoff, order, strength, border):
        """Blend one frame.

        Args:
            steady: The frame to keep, ``(height, width, channels)`` in 0 to 1.
            detailed: The frame to take detail from, the same shape.
            cutoff: Fraction of the frequency range where low becomes high.
            order: Steepness of the split.
            strength: How much blended detail to add back.
            border: Pixels at the edge left as ``steady``.

        Returns:
            The blended frame, same shape, held to 0 to 1 where both inputs were.
        """
        work = torch.promote_types(steady.dtype, torch.float32)
        a = steady.to(work)
        b = detailed.to(work)

        luma_a, chroma_a = cls.to_yuv(a)
        luma_b, _ = cls.to_yuv(b)

        low_pass = cls.butterworth(a.shape[0], a.shape[1], cutoff, order, a.device, work)
        low_a, high_a = cls.split(luma_a, low_pass)
        _, high_b = cls.split(luma_b, low_pass)

        # Per pixel, whichever frame carries more detail here contributes more of it. Where both
        # are flat the weight is meaningless, and the epsilon keeps it from dividing by zero.
        share = high_b.abs() / (high_b.abs() + high_a.abs() + 1e-6)
        fused = low_a + strength * (share * high_b + (1.0 - share) * high_a)

        if border > 0 and 2 * border < min(a.shape[0], a.shape[1]):
            keep = torch.ones_like(fused)
            keep[border:-border, border:-border] = 0.0
            fused = torch.where(keep > 0, luma_a, fused)

        blended = dynamic.hold(cls.from_yuv(fused, chroma_a), steady, detailed)
        return blended.to(steady.dtype)

    @staticmethod
    def to_yuv(frame):
        """Split a frame into its luma and its two chroma channels."""
        rgb = frame[..., :3]
        matrix = torch.tensor(TO_YUV, device=frame.device, dtype=frame.dtype)
        yuv = rgb @ matrix.T
        return yuv[..., 0], yuv[..., 1:]

    @staticmethod
    def from_yuv(luma, chroma):
        """Put a luma and a chroma pair back together as RGB."""
        yuv = torch.cat([luma.unsqueeze(-1), chroma], dim=-1)
        matrix = torch.tensor(FROM_YUV, device=luma.device, dtype=luma.dtype)
        return yuv @ matrix.T

    @staticmethod
    def butterworth(height, width, cutoff, order, device, dtype):
        """A radial Butterworth low-pass over a centred frequency grid."""
        columns = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype).view(1, width)
        rows = torch.linspace(-0.5, 0.5, height, device=device, dtype=dtype).view(height, 1)
        radius = torch.sqrt(columns**2 + rows**2)
        return 1.0 / (1.0 + (radius / cutoff) ** (2 * order))

    @staticmethod
    def split(luma, low_pass):
        """Separate one luma plane into what the filter passes and what it holds back."""
        spectrum = torch.fft.fftshift(torch.fft.fft2(luma, norm="ortho"))
        low = torch.fft.ifft2(torch.fft.ifftshift(spectrum * low_pass), norm="ortho").real
        return low, luma - low
