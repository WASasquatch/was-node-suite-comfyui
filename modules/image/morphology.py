"""Grey-level morphology with a flat square kernel, as separable max and min passes.

Tensors are ``(batch, channels, height, width)``. The anchor is ``kernel // 2`` on each axis
and pixels outside the frame are ignored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Every operation :func:`morph` takes, in the order a menu lists them.
OPERATIONS = ("erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat")


def _padding(kernel: int) -> tuple[int, int, int, int]:
    """The padding around a frame for a kernel anchored at ``kernel // 2``.

    Args:
        kernel: Kernel side in pixels.

    Returns:
        ``(left, right, top, bottom)`` in pixels.
    """
    before = kernel // 2
    after = kernel - before - 1
    return before, after, before, after


def dilate(planes: torch.Tensor, kernel: int) -> torch.Tensor:
    """The largest value under the kernel at every pixel.

    Args:
        planes: ``(batch, channels, height, width)`` float tensor.
        kernel: Kernel side in pixels.

    Returns:
        A tensor the same shape.
    """
    if kernel <= 1:
        return planes.clone()
    padded = F.pad(planes, _padding(kernel), value=float("-inf"))
    rows = F.max_pool2d(padded, kernel_size=(1, kernel), stride=1)
    return F.max_pool2d(rows, kernel_size=(kernel, 1), stride=1)


def erode(planes: torch.Tensor, kernel: int) -> torch.Tensor:
    """The smallest value under the kernel at every pixel.

    Args:
        planes: ``(batch, channels, height, width)`` float tensor.
        kernel: Kernel side in pixels.

    Returns:
        A tensor the same shape.
    """
    return -dilate(-planes, kernel)


def morph(planes: torch.Tensor, operation: str, kernel: int) -> torch.Tensor:
    """Apply one morphology operation.

    Args:
        planes: ``(batch, channels, height, width)`` float tensor.
        operation: One of :data:`OPERATIONS`.
        kernel: Kernel side in pixels.

    Returns:
        A tensor the same shape.

    Raises:
        ValueError: The operation is not one of :data:`OPERATIONS`.
    """
    if operation == "erode":
        return erode(planes, kernel)
    if operation == "dilate":
        return dilate(planes, kernel)
    if operation == "open":
        return dilate(erode(planes, kernel), kernel)
    if operation == "close":
        return erode(dilate(planes, kernel), kernel)
    if operation == "gradient":
        return dilate(planes, kernel) - erode(planes, kernel)
    if operation == "top_hat":
        return planes - dilate(erode(planes, kernel), kernel)
    if operation == "bottom_hat":
        return erode(dilate(planes, kernel), kernel) - planes
    raise ValueError(
        f"Unknown morphology operation {operation!r}. Pick one of {', '.join(OPERATIONS)}."
    )
