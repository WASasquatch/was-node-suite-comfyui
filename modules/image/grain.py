"""Photographic film grain added to an image batch.

Images are ``[batch, height, width, channels]`` float tensors. Grain is zero-mean, follows the
luma of each pixel and leaves an alpha channel untouched.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import dynamic
from ..interface.progress import progress_bar

__all__ = ["film_grain"]

#: Rec. 709 weights of the luma the grain follows.
LUMA = (0.2126, 0.7152, 0.0722)

#: Gaussian blur sigma per pixel of grain size.
SIGMA_PER_SIZE = 0.5

#: Blur sigma below which grain is left as one value per pixel.
MIN_SIGMA = 0.2


def film_grain(images: torch.Tensor, strength: float = 0.03, size: float = 1.5,
               color: float = 0.0, seed: int = 0) -> torch.Tensor:
    """Add film grain to every frame of a batch.

    Args:
        images: ``[batch, height, width, channels]`` float tensor.
        strength: Standard deviation of the grain at mid-grey, on the 0 to 1 scale.
        size: Width of a grain clump, in pixels.
        color: 0.0 = one grain pattern on every channel; 1.0 = a separate pattern per channel.
        seed: Seed of the grain. Frame ``k`` of a batch takes the ``k``-th pattern drawn.

    Returns:
        A new tensor on the batch's device and dtype, or ``images`` itself when ``strength`` is
        0. Held inside 0 to 1 unless the batch carried values outside it.
    """
    if strength <= 0.0 or images.numel() == 0:
        return images

    batch, height, width, channels = images.shape
    colours = min(channels, 3)
    kernel = _kernel(size * SIGMA_PER_SIZE)
    # A blurred field is divided by its own spread to bring it back to unit deviation.
    spread = float((kernel ** 2).sum()) if kernel is not None else 1.0
    mix = min(max(float(color), 0.0), 1.0)
    norm = math.hypot(1.0 - mix, mix)
    weights = torch.tensor(LUMA[:colours], device=images.device, dtype=torch.float32)
    weights = weights / weights.sum()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (1 << 64))
    result = images.clone()
    bar = progress_bar(batch)
    for index in range(batch):
        frame = images[index, :, :, :colours].float()
        fields = torch.randn((1 + colours, 1, height, width), generator=generator)
        fields = _blurred(fields.to(images.device), kernel) / spread
        grain = ((1.0 - mix) * fields[0] + mix * fields[1:]) / norm
        luma = (frame * weights).sum(dim=-1).clamp(0.0, 1.0)
        response = 2.0 * torch.sqrt(luma * (1.0 - luma))
        frame = frame + strength * response.unsqueeze(-1) * grain.squeeze(1).permute(1, 2, 0)
        result[index, :, :, :colours] = frame.to(result.dtype)
        bar.update()
    return dynamic.hold(result, images)


def _kernel(sigma: float) -> torch.Tensor | None:
    """A normalised one-dimensional Gaussian.

    Args:
        sigma: Standard deviation in pixels.

    Returns:
        A float32 tensor of odd length summing to 1, or ``None`` below :data:`MIN_SIGMA`.
    """
    if sigma < MIN_SIGMA:
        return None
    radius = max(1, math.ceil(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _blurred(fields: torch.Tensor, kernel: torch.Tensor | None) -> torch.Tensor:
    """Blur ``[n, 1, height, width]`` fields with a separable Gaussian.

    Args:
        fields: The fields to blur.
        kernel: A one-dimensional kernel from :func:`_kernel`, or ``None`` for no blur.

    Returns:
        The blurred fields, the same shape as ``fields``.
    """
    if kernel is None:
        return fields
    radius = kernel.numel() // 2
    kernel = kernel.to(fields.device)
    across = F.pad(fields, (radius, radius, 0, 0), mode="replicate")
    across = F.conv2d(across, kernel.view(1, 1, 1, -1))
    down = F.pad(across, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(down, kernel.view(1, 1, -1, 1))
