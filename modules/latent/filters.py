"""Spatial filters the latent detail and upscale nodes are built from.

Everything here works on a 4D ``[B, C, H, W]`` tensor and keeps that shape. Kernels are
built on the tensor's own device and dtype.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["clamp01", "gaussian_blur_depthwise", "mask_preview", "sobel_grad_mag"]


def clamp01(x: torch.Tensor) -> torch.Tensor:
    """Hold a tensor inside 0.0-1.0.

    Args:
        x: Any tensor.

    Returns:
        The tensor with everything below 0.0 raised to 0.0 and everything above 1.0
        lowered to 1.0.
    """
    return torch.clamp(x, 0.0, 1.0)


def gaussian_blur_depthwise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur every channel of a tensor independently with a Gaussian.

    Args:
        x: A ``[B, C, H, W]`` tensor.
        sigma: Standard deviation in tensor units, for a latent, one unit is one latent
            block rather than one pixel. The kernel reaches out three sigma, so 1.0 spans
            seven units.

    Returns:
        The blurred tensor, same shape, dtype and device. A sigma of 0.0 or less is not a
        blur and returns the input unchanged.
    """
    if sigma <= 0.0:
        return x

    radius = max(1, int(math.ceil(3.0 * sigma)))
    ksize = 2 * radius + 1

    device = x.device
    dtype = x.dtype

    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, ksize)
    kernel_y = kernel_1d.view(1, 1, ksize, 1)

    c = x.shape[1]
    # Reflecting needs a row or column to fold back on, which a map only one or two samples
    # tall does not have.
    mode = "reflect" if radius < min(x.shape[-2], x.shape[-1]) else "replicate"
    # The 2D Gaussian is separable, so two 1D passes cost the radius rather than its square.
    x_pad = F.pad(x, (radius, radius, radius, radius), mode=mode)
    x_blur = F.conv2d(x_pad, kernel_x.expand(c, 1, 1, ksize), groups=c)
    x_blur = F.conv2d(x_blur, kernel_y.expand(c, 1, ksize, 1), groups=c)
    return x_blur


def sobel_grad_mag(x_bt1hw: torch.Tensor) -> torch.Tensor:
    """Measure how fast a single-channel map changes at each position.

    Args:
        x_bt1hw: A ``[B, 1, H, W]`` map, typically the per-position energy of a latent.

    Returns:
        The gradient magnitude, same shape. Zero padding means the outermost row and column
        read as a boundary against black.
    """
    device = x_bt1hw.device
    dtype = x_bt1hw.dtype

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x_bt1hw, kx, padding=1)
    gy = F.conv2d(x_bt1hw, ky, padding=1)
    # The constant under the root keeps the gradient differentiable and finite where both
    # slopes are exactly zero.
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def mask_preview(mask_bt1hw: torch.Tensor, pixel_scale: int) -> torch.Tensor:
    """Turn a latent-resolution mask into an IMAGE that can be viewed.

    Args:
        mask_bt1hw: A ``[B, 1, H, W]`` mask. Extra channels are ignored; the first is used.
        pixel_scale: How many output pixels each latent block becomes. 1 leaves the mask at
            latent resolution, 8 matches the size the latent decodes to on most VAEs.
            Values below 1 are treated as 1.

    Returns:
        A ``[B, H*scale, W*scale, 3]`` image tensor clamped to 0.0-1.0, black where the mask
        is 0.0 and white where it is 1.0.
    """
    ps = max(1, int(pixel_scale))
    if ps == 1:
        m = mask_bt1hw
    else:
        h, w = mask_bt1hw.shape[-2], mask_bt1hw.shape[-1]
        m = F.interpolate(mask_bt1hw, size=(h * ps, w * ps), mode="nearest-exact")
    img = m[:, 0:1].repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
    return clamp01(img)
