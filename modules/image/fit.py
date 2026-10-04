"""Bringing images of different sizes to one size, so they can share a batch.

Images are ``(batch, height, width, channels)`` in ``[0, 1]``. Every method writes frames of the
requested size into a batch the caller holds.
"""

from __future__ import annotations

import torch

__all__ = ["METHODS", "PAD_LEVEL", "RGBA", "check_method", "fit_into", "pad_to", "target_size"]

# `resize` scales to the target and ignores the shape it had, so nothing is lost or added but a
# picture of another shape is stretched. `crop` scales until the target is covered, keeping the
# shape, then takes the middle, so what falls outside the frame is gone. `pad` scales until the
# target contains it and centres it on a flat field, keeping the whole picture.

#: The methods, in the order a node offers them.
METHODS = ("resize", "crop", "pad")

#: What ``pad`` fills the rest of the frame with, as a level in ``[0, 1]``.
PAD_LEVEL = 0.0

#: Channels a frame carrying alpha holds.
RGBA = 4


def target_size(tensor) -> tuple[int, int]:
    """The height and width of an image batch.

    Args:
        tensor: An ``IMAGE`` tensor.

    Returns:
        ``(height, width)``.
    """
    return int(tensor.shape[1]), int(tensor.shape[2])


def _scaled(tensor, height: int, width: int):
    """``tensor`` resampled to exactly ``height`` by ``width`` and clamped to ``[0, 1]``."""
    # Channels move next to the batch for interpolation and back after it.
    planes = tensor.permute(0, 3, 1, 2)
    resampled = torch.nn.functional.interpolate(
        planes, size=(height, width), mode="bilinear", align_corners=False, antialias=True,
    )
    return resampled.clamp_(0.0, 1.0).permute(0, 2, 3, 1)


def _centre_crop(tensor, height: int, width: int):
    """The middle ``height`` by ``width`` of ``tensor``, which must be at least that large."""
    top = max(0, (int(tensor.shape[1]) - height) // 2)
    left = max(0, (int(tensor.shape[2]) - width) // 2)
    return tensor[:, top:top + height, left:left + width, :]


def pad_to(
    tensor, height: int, width: int, level: float = PAD_LEVEL, transparent: bool = False,
    out=None,
):
    """One image batch centred on a larger field, with nothing resampled.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)``, no larger than the
            field on either axis.
        height: Height of the field.
        width: Width of the field.
        level: What the field around the frame holds, as a level in ``[0, 1]``.
        transparent: Answer 4 channels, the field fully transparent and the frame opaque. A
            frame that already carries alpha keeps its own.
        out: A tensor shaped as the answer to write the field into, or None for a new one.

    Returns:
        ``out``, or a new tensor, of exactly that height and width, at the source channel
        count, or at :data:`RGBA` channels where ``transparent`` is set.
    """
    height, width = int(height), int(width)
    batch, current_height, current_width, channels = (int(axis) for axis in tensor.shape)
    depth = RGBA if transparent else channels
    if out is None:
        field = torch.full(
            (batch, height, width, depth), level,
            dtype=tensor.dtype, device=tensor.device,
        )
    else:
        field = out.fill_(level)
    if transparent:
        field[..., 3] = 0.0
    top = max(0, (height - current_height) // 2)
    left = max(0, (width - current_width) // 2)
    carried = min(channels, depth)
    rows = slice(top, top + current_height)
    columns = slice(left, left + current_width)
    field[:, rows, columns, :carried] = tensor[..., :carried]
    if transparent and channels < RGBA:
        field[:, rows, columns, 3] = 1.0
    return field


def check_method(method: str) -> None:
    """Refuse a fitting method that is not one of :data:`METHODS`.

    Args:
        method: The method named on the node.

    Raises:
        ValueError: ``method`` is not one of :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(
            f"resize_method is {method!r}, which is not one of {', '.join(METHODS)}."
        )


def fit_into(tensor, out, method: str = "resize"):
    """Write one image batch into ``out``, brought to its height and width a frame at a time.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)``.
        out: A tensor of the same batch and channels at the target height and width, which
            is overwritten.
        method: One of :data:`METHODS`.

    Returns:
        ``out``.

    Raises:
        ValueError: ``method`` is not one of :data:`METHODS`.
    """
    check_method(method)
    height, width = target_size(out)
    current_height, current_width = target_size(tensor)
    if (current_height, current_width) == (height, width):
        return out.copy_(tensor)

    if method == "resize":
        scaled_height, scaled_width = height, width
    else:
        # Crop scales until the target is covered, pad until the target contains the frame.
        if method == "crop":
            factor = max(height / current_height, width / current_width)
        else:
            factor = min(height / current_height, width / current_width)
        scaled_height = max(1, round(current_height * factor))
        scaled_width = max(1, round(current_width * factor))

    for index in range(int(tensor.shape[0])):
        scaled = _scaled(tensor[index:index + 1], scaled_height, scaled_width)
        frame = out[index:index + 1]
        if method == "pad":
            pad_to(scaled, height, width, out=frame)
        else:
            frame.copy_(_centre_crop(scaled, height, width))
        del scaled
    return out
