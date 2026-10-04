"""Image nodes, and the batch handling they share.

An ``IMAGE`` reaching one of these nodes is ``(batch, height, width, channels)``.
"""

from __future__ import annotations

import torch
from PIL import Image

from ...modules.convert import tensors

__all__ = ["image_planes", "quantises_exactly", "stack_images"]


def image_planes(images: torch.Tensor) -> list[torch.Tensor]:
    """Split an image tensor into one image per item of its batch.

    Args:
        images: Image tensor. Four axes or more are read as a batch and iterated; three are
            read as a single unbatched image; fewer are read as one image of one row.

    Returns:
        One ``(height, width, channels)`` view per image, in batch order. Never empty.
    """
    if images.ndim >= 4:
        return list(images)
    if images.ndim == 3:
        return [images]
    if images.ndim == 2:
        return [images.unsqueeze(-1)]
    return [images.reshape(1, -1, 1)]


def stack_images(images: list[Image.Image]) -> torch.Tensor:
    """Assemble PIL images into the ``(batch, height, width, channels)`` tensor a node emits.

    Args:
        images: One PIL image per item of the batch, in batch order, all in the same mode.

    Returns:
        A tensor shaped ``(len(images), height, width, channels)``.

    Raises:
        ValueError: No image was given, or the images do not share a channel count.
        MemoryError: Neither free memory nor any scratch drive has room for the batch.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    shapes = [tensors.array_shape(image) for image in images]
    channels = {shape[2] if len(shape) > 2 else 1 for shape in shapes}
    if len(channels) > 1:
        raise ValueError(f"All images must share a channel count, got {sorted(channels)}.")
    return tensors.stack_images(images, advice="Fewer or smaller frames also fit it.")


def quantises_exactly(plane: torch.Tensor) -> bool:
    """Whether a PIL round trip on this image is nothing but an 8-bit quantisation.

    Args:
        plane: One ``(height, width, channels)`` image of a batch.

    Returns:
        Whether tensor arithmetic on this image is equivalent to the PIL round trip.
    """
    return plane.ndim == 3 and plane.shape[0] > 1 and plane.shape[1] > 1 and plane.shape[2] in (3, 4)
