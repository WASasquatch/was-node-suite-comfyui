"""Whether a batch carries light outside 0 to 1, and how a filter keeps it.

:func:`hold` suits a filter working in floats, :func:`fold` and :func:`unfold` one working
in picture codes.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = ["TOLERANCE", "Folded", "carries", "factor", "fold", "hold", "peak", "unfold"]

#: How far outside 0 to 1 a value reaches before the batch counts as carrying more than a
#: picture. Ordinary float arithmetic lands a few parts in a million out.
TOLERANCE = 1e-3


class Folded(NamedTuple):
    """A batch brought inside 0 to 1, and what it took to get there.

    Attributes:
        images: The batch, every value in 0 to 1.
        scale: What the batch was divided by. 1.0 where it already fitted.
    """

    images: torch.Tensor
    scale: float


def carries(images) -> bool:
    """Whether a batch holds values a 0 to 1 picture cannot.

    Args:
        images: Any float tensor.

    Returns:
        True where a value reaches past 1.0 or under 0.0 by more than :data:`TOLERANCE`.
    """
    if images is None or not getattr(images, "numel", lambda: 0)():
        return False
    return float(images.amax()) > 1.0 + TOLERANCE or float(images.amin()) < -TOLERANCE


def hold(result, *sources):
    """The result held inside 0 to 1, or left as it is where a source carried more.

    Args:
        result: What the node produced.
        *sources: The batches it was produced from.

    Returns:
        ``result`` clamped to 0 to 1, or ``result`` unchanged.
    """
    return result if any(carries(source) for source in sources) else result.clamp(0.0, 1.0)


def peak(*batches) -> float:
    """The one scale that brings every batch given inside 0 to 1.

    Args:
        *batches: Any number of float tensors, None among them ignored.

    Returns:
        The highest value any of them holds, never under 1.0.
    """
    highest = 1.0
    for batch in batches:
        if batch is not None and getattr(batch, "numel", lambda: 0)():
            highest = max(highest, float(batch.amax()))
    return highest


def factor(images, scale=None) -> float:
    """What :func:`fold` divides a batch by, read without dividing it.

    Args:
        images: The batch on its own scale.
        scale: What to divide by, from :func:`peak`, where several batches share one
            scale. Left out, the batch's own peak.

    Returns:
        The scale, or 1.0 where the batch already fits inside 0 to 1.
    """
    if scale is None:
        if not carries(images):
            return 1.0
        scale = peak(images)
    return 1.0 if scale <= 1.0 + TOLERANCE else float(scale)


def fold(images, scale=None) -> Folded:
    """A batch a 0 to 1 filter can take, with the scale that puts it back.

    Args:
        images: The batch on its own scale.
        scale: What to divide by, from :func:`peak`, where several batches share one
            scale. Left out, the batch's own peak.

    Returns:
        A :class:`Folded`. A batch already inside 0 to 1 is answered untouched, on a scale
        of 1.0.
    """
    scale = factor(images, scale)
    if scale == 1.0:
        return Folded(images, 1.0)
    return Folded((images / scale).clamp_(0.0, 1.0), scale)


def unfold(result, folded, in_place: bool = False) -> torch.Tensor:
    """The filtered batch back on the scale it arrived on.

    Args:
        result: What the filter answered, in 0 to 1.
        folded: What :func:`fold` answered for the batch it was given.
        in_place: Multiply ``result`` itself rather than a copy of it.

    Returns:
        ``result`` multiplied by the scale, or ``result`` where the scale is 1.0.
    """
    if folded.scale == 1.0:
        return result
    return result.mul_(folded.scale) if in_place else result * folded.scale
