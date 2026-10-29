"""Richardson-Lucy deconvolution sharpening."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import CHANNEL_COUNTS
from ....modules.image import dynamic
from ....modules.image.accelerate import run_frames

logger = log.get_logger("image.lucy_sharpen")

#: Added to the blurred estimate before the ratio against it is taken.
EPSILON = 1e-6

#: Float32 copies of a frame the deconvolution holds at its peak.
WORKING_COPIES = 10


def lucy_sharpen(images, iterations: int = 10, kernel_size: int = 3):
    """Sharpen an image batch by Richardson-Lucy deconvolution.

    Args:
        images: Image tensor in the layout an ``IMAGE`` socket carries. Every channel is
            deconvolved, alpha included.
        iterations: How many deconvolution passes to run.
        kernel_size: Side of the square box kernel, in pixels. Held to 1 at the least.

    Returns:
        A float32 tensor shaped like ``images``, on the scale it arrived on and on the
        device it arrived on.
    """
    folded = dynamic.fold(images)
    passes, kernel = int(iterations), max(1, int(kernel_size))

    def work(batch: torch.Tensor) -> torch.Tensor:
        planes, restore = _as_planes(batch.to(torch.float32).clamp(0.0, 1.0))
        return restore(_iterate(planes, passes, kernel).clamp(0.0, 1.0))

    sharpened = run_frames(folded.images, work, factor=WORKING_COPIES, node="Image Lucy Sharpen")
    return dynamic.unfold(sharpened, folded, in_place=True)


def _as_planes(images: torch.Tensor):
    """The batch as ``(batch, channels, height, width)``, with the call that undoes it.

    Args:
        images: Image tensor in the layout an ``IMAGE`` socket carries. Four axes or more
            are read as a batch of ``(height, width, channels)``, three as one image where
            the last axis is one of :data:`modules.convert.tensors.CHANNEL_COUNTS` and as a
            batch of single-channel frames otherwise, and fewer as one single-channel image.

    Returns:
        ``(planes, restore)``. ``restore`` takes a tensor shaped like ``planes`` and answers
        one shaped like ``images``.
    """
    shape = images.shape
    if images.ndim >= 4:
        flat = images.reshape(-1, *shape[-3:]).movedim(-1, 1)
        return flat, lambda planes: planes.movedim(1, -1).reshape(shape)
    if images.ndim == 3 and int(shape[-1]) in CHANNEL_COUNTS:
        return (
            images.movedim(-1, 0).unsqueeze(0),
            lambda planes: planes.squeeze(0).movedim(0, -1),
        )
    if images.ndim == 3:
        return images.unsqueeze(1), lambda planes: planes.squeeze(1)
    return (
        images.reshape(1, 1, 1, -1) if images.ndim < 2 else images.reshape(1, 1, *shape),
        lambda planes: planes.reshape(shape),
    )


def _iterate(planes: torch.Tensor, iterations: int, kernel_size: int) -> torch.Tensor:
    """Run the Richardson-Lucy loop over an edge-padded copy of the batch.

    Args:
        planes: ``(batch, channels, height, width)`` float tensor in 0 to 1.
        iterations: How many deconvolution passes to run.
        kernel_size: Side of the square box kernel, in pixels.

    Returns:
        A tensor the same shape as ``planes``, the padding cropped back off.
    """
    working = F.pad(planes, (kernel_size,) * 4, mode="replicate")
    weight = working.new_full((working.shape[1], 1, 1, kernel_size), 1.0 / kernel_size)
    for _ in range(iterations):
        ratio = working / (_box(working, weight) + EPSILON)
        working = working * _box(ratio, weight)
    return working[..., kernel_size:-kernel_size, kernel_size:-kernel_size]


def _box(planes: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Blur with a normalised box kernel, samples off the edge counted as zero.

    Args:
        planes: ``(batch, channels, height, width)`` tensor.
        weight: ``(channels, 1, 1, kernel_size)`` row of ``1 / kernel_size``.

    Returns:
        A tensor the same shape as ``planes``. A box of an even width sits one sample left
        of centre and one sample above it.
    """
    size = int(weight.shape[-1])
    before = size // 2
    after = size - 1 - before
    groups = planes.shape[1]
    rows = F.conv2d(F.pad(planes, (before, after, 0, 0)), weight, groups=groups)
    return F.conv2d(
        F.pad(rows, (0, 0, before, after)), weight.transpose(-1, -2), groups=groups
    )


class ImageLucySharpen(io.ComfyNode):
    """Recover detail lost to blur by iterative deconvolution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Lucy Sharpen",
            display_name="Image Lucy Sharpen",
            search_aliases=[
                "Image Lucy Sharpen",
                "lucy",
                "deconvolution",
                "sharpen",
                "unblur",
                "richardson",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Sharpen an image by working out what it looked like before it was blurred, "
                "rather than by boosting edges. Recovers real detail from a soft photo, and "
                "amplifies noise and compression artefacts along with it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The image to sharpen. Every image in a batch is sharpened.",
                ),
                io.Int.Input(
                    "iterations",
                    default=2,
                    min=1,
                    max=12,
                    step=1,
                    tooltip=(
                        "How many refinement passes to run. 2 is a gentle recovery, 6 is "
                        "aggressive, and by 12 the noise and ringing usually outweigh the detail "
                        "gained. Cost is proportional to this."
                    ),
                ),
                io.Int.Input(
                    "kernel_size",
                    default=3,
                    min=1,
                    max=16,
                    step=1,
                    tooltip=(
                        "How wide the blur being undone is assumed to be, in pixels. 3 suits a "
                        "slightly soft image; larger values target a heavier blur but spread "
                        "ringing further. 1 assumes no blur and leaves the image nearly as it is."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The sharpened image."),
            ],
        )

    @classmethod
    def execute(cls, images, iterations, kernel_size) -> io.NodeOutput:
        return io.NodeOutput(lucy_sharpen(images, iterations, kernel_size))
