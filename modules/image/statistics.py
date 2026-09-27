"""Per-image measurements, computed with torch.

:func:`measure` returns one value per name in :data:`FIELDS` for a single image. Tonal
values are 0.0 to 1.0 rather than 0 to 255. Luminance uses the Rec. 709 weights in
:data:`LUMA_WEIGHTS`.
"""

from __future__ import annotations

import torch

__all__ = ["FIELDS", "LUMA_WEIGHTS", "measure"]

#: Rec. 709 luminance weights, the coefficients a display applies when it turns RGB into
#: perceived brightness. Used rather than a channel mean, which reads a saturated blue as
#: being as bright as the same-valued green.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: The measurement names :func:`measure` returns, in the order the node emits them.
FIELDS = (
    "mean",
    "median",
    "minimum",
    "maximum",
    "contrast",
    "sharpness",
    "saturation",
    "clipped_shadows",
    "clipped_highlights",
    "entropy",
)

#: Kernel of the discrete Laplacian, whose response to an image is large at an edge and
#: near zero across a flat area. Its variance over the whole image is the standard
#: focus measure: a blurred picture has no strong edges left and so very little spread.
_LAPLACIAN = (
    (0.0, 1.0, 0.0),
    (1.0, -4.0, 1.0),
    (0.0, 1.0, 0.0),
)

#: Bins used for the entropy estimate. 256 matches an 8-bit histogram, so the number is
#: comparable with entropy reported by an image editor.
_BINS = 256

#: Values within this distance of pure black or pure white count as clipped. A tenth of a
#: percent either way is below what an 8-bit file can represent as a distinct step.
_CLIP_EPSILON = 1.0 / 255.0


def _luminance(image: torch.Tensor) -> torch.Tensor:
    """The perceived-brightness plane of one image.

    Args:
        image: One image, ``(height, width, channels)``. A single-channel image is used as
            its own luminance; an image carrying alpha has it ignored, since a transparent
            pixel still has a colour and dropping it would change the measurement based on
            a channel nothing else here reads.

    Returns:
        A ``(height, width)`` float tensor.
    """
    if image.ndim == 2:
        return image.float()
    channels = image.shape[-1]
    if channels == 1:
        return image[..., 0].float()
    weights = torch.tensor(LUMA_WEIGHTS, dtype=torch.float32, device=image.device)
    return (image[..., :3].float() * weights).sum(dim=-1)


def _saturation(image: torch.Tensor) -> torch.Tensor:
    """The HSV saturation plane of one image, or zeros for a greyscale one."""
    if image.ndim == 2 or image.shape[-1] == 1:
        return torch.zeros(image.shape[:2], dtype=torch.float32, device=image.device)
    rgb = image[..., :3].float()
    largest = rgb.max(dim=-1).values
    smallest = rgb.min(dim=-1).values
    # A black pixel has no hue and no saturation; dividing by its zero maximum would give
    # a NaN that then poisons the mean.
    return torch.where(largest > 0, (largest - smallest) / largest.clamp(min=1e-12), largest)


def _sharpness(luma: torch.Tensor) -> float:
    """Variance of the Laplacian response, the standard focus measure.

    Args:
        luma: A ``(height, width)`` luminance plane.

    Returns:
        The variance. Larger is sharper; the absolute value depends on image size and
        content, so it is compared between frames of one sequence rather than read alone.
        An image smaller than the kernel returns 0.0, having no interior to measure.
    """
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    kernel = torch.tensor(_LAPLACIAN, dtype=torch.float32, device=luma.device).view(1, 1, 3, 3)
    response = torch.nn.functional.conv2d(luma.view(1, 1, *luma.shape), kernel)
    return float(response.var(unbiased=False))


def _entropy(luma: torch.Tensor, bins: int = _BINS) -> float:
    """Shannon entropy of the luminance histogram, in bits.

    Args:
        luma: A ``(height, width)`` luminance plane.
        bins: Histogram bins, 256 for 8 bits and 65536 for 16.

    Returns:
        Between 0.0 for a single-valued image and ``log2(bins)`` for one spread evenly
        across every bin.
    """
    counts = torch.histc(luma.flatten().float().clamp(0.0, 1.0), bins=bins, min=0.0, max=1.0)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probability = counts / total
    occupied = probability[probability > 0]
    return float(-(occupied * occupied.log2()).sum())


def measure(image: torch.Tensor, bins: int = _BINS) -> dict[str, float]:
    """Measure one image.

    Args:
        image: One image shaped ``(height, width, channels)``, float in ``[0, 1]``. A
            ``(height, width)`` plane is read as greyscale.
        bins: Histogram bins the entropy is counted over.

    Returns:
        A mapping with one entry per name in :data:`FIELDS`:

        - ``mean``, ``median``, ``minimum``, ``maximum``: luminance, 0.0 to 1.0.
        - ``contrast``: standard deviation of luminance, 0.0 to 0.5.
        - ``sharpness``: variance of the Laplacian. Unbounded above.
        - ``saturation``: mean HSV saturation, 0.0 to 1.0.
        - ``clipped_shadows``, ``clipped_highlights``: fraction of pixels at the bottom
          or top of the range, 0.0 to 1.0.
        - ``entropy``: bits, 0.0 to ``log2(bins)``.
    """
    plane = image.detach()
    if plane.ndim == 4:
        plane = plane[0]
    luma = _luminance(plane)
    flat = luma.flatten().float()

    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "contrast": float(flat.std(unbiased=False)),
        "sharpness": _sharpness(luma),
        "saturation": float(_saturation(plane).mean()),
        "clipped_shadows": float((flat <= _CLIP_EPSILON).float().mean()),
        "clipped_highlights": float((flat >= 1.0 - _CLIP_EPSILON).float().mean()),
        "entropy": _entropy(luma, bins),
    }
