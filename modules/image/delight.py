"""Taking the lighting out of a picture with an albedo estimated at a smaller size.

Pictures are ``(H, W, C)`` float sRGB codes in ``[0, 1]``; shading is found in linear light.
"""

from __future__ import annotations

__all__ = ["COLOURS", "MAX_DETAIL", "delight"]

import torch
import torch.nn.functional as F

from .convolve import gaussian_blur

#: Where the delit colour comes from: per-channel shading takes the albedo's colour, grey
#: shading keeps the picture's own.
COLOURS = ("albedo", "image")

#: Widest smoothing radius, in pixels of the picture.
MAX_DETAIL = 256.0

#: Smallest and largest shading a picture is divided by.
SHADING_FLOOR = 1.0 / 64.0
SHADING_CEILING = 64.0

#: Albedo level below which the shading is not trusted.
DARK = 1e-3

#: Share of the mask's smoothed coverage below which a pixel takes no shading.
MASK_FLOOR = 1e-3

#: Smoothing radius, in working pixels, above which the working grid is shrunk first.
SHRINK_RADIUS = 4.0

#: Power the shading output is raised to for showing.
SHOWN = 1.0 / 2.2

#: Rec. 709 luminance weights.
LUMA = (0.2126, 0.7152, 0.0722)


def _to_linear(codes: torch.Tensor) -> torch.Tensor:
    """sRGB codes as linear light."""
    codes = codes.clamp(0.0, 1.0)
    return torch.where(codes <= 0.04045, codes / 12.92, ((codes + 0.055) / 1.055) ** 2.4)


def _to_codes(values: torch.Tensor) -> torch.Tensor:
    """Linear light as sRGB codes."""
    values = values.clamp(0.0, 1.0)
    return torch.where(values <= 0.0031308, values * 12.92, 1.055 * values.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def _planes(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """An ``(H, W, C)`` image as ``(1, C, h, w)`` planes, area-averaged or bilinearly scaled to ``size``."""
    planes = image.permute(2, 0, 1)[None].float()
    if tuple(planes.shape[-2:]) == tuple(size):
        return planes
    shrinking = size[0] < planes.shape[-2] or size[1] < planes.shape[-1]
    return F.interpolate(planes, size=size, mode="bilinear", align_corners=False, antialias=shrinking)


def _luma(planes: torch.Tensor) -> torch.Tensor:
    """``(1, 3, h, w)`` linear planes reduced to ``(1, 1, h, w)`` luminance."""
    weights = torch.tensor(LUMA, dtype=planes.dtype, device=planes.device).view(1, 3, 1, 1)
    return (planes * weights).sum(dim=1, keepdim=True)


def _shading(image: torch.Tensor, albedo: torch.Tensor, mask: torch.Tensor | None, radius: float,
             colour: str) -> torch.Tensor:
    """Shading of one picture on the albedo's grid, or a coarser one for a wide radius.

    Args:
        image: ``(1, 3, h, w)`` linear picture at the albedo's size.
        albedo: ``(1, 3, h, w)`` linear albedo.
        mask: ``(1, 1, h, w)`` share of each pixel that is the subject, or None for all of it.
        radius: Smoothing radius in pixels of this grid.
        colour: One of `COLOURS`.

    Returns:
        ``(1, C, h', w')`` shading, 3 channels for ``"albedo"`` and 1 for ``"image"``.
    """
    if colour == "image":
        image, albedo = _luma(image), _luma(albedo)
    weight = mask if mask is not None else torch.ones_like(image[:, :1])
    shrink = int(radius // (SHRINK_RADIUS / 2)) if radius > SHRINK_RADIUS else 1
    if shrink > 1:
        height = max(1, image.shape[-2] // shrink)
        width = max(1, image.shape[-1] // shrink)
        image = F.interpolate(image * weight, size=(height, width), mode="area")
        albedo = F.interpolate(albedo * weight, size=(height, width), mode="area")
        weight = F.interpolate(weight, size=(height, width), mode="area")
        radius = radius / shrink
    else:
        image, albedo = image * weight, albedo * weight
    if radius >= 0.5:
        image = gaussian_blur(image, sigma=radius)
        albedo = gaussian_blur(albedo, sigma=radius)
        weight = gaussian_blur(weight, sigma=radius)
    level = weight.clamp_min(MASK_FLOOR)
    trusted = (albedo / level) > DARK
    shading = (image / albedo.clamp_min(DARK * MASK_FLOOR)).clamp(SHADING_FLOOR, SHADING_CEILING)
    shading = torch.where(trusted, shading, torch.ones_like(shading))
    return torch.where(weight > MASK_FLOOR, shading, torch.ones_like(shading))


def delight(image: torch.Tensor, albedo: torch.Tensor, detail: float = 16.0, strength: float = 1.0,
            colour: str = "albedo", mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide a picture's lighting out at its own size, using an albedo estimated at any size.

    Args:
        image: ``(H, W, C)`` sRGB codes, ``C`` 3 or 4.
        albedo: ``(h, w, C)`` sRGB codes of the same picture's albedo, any size of its shape.
        detail: Smoothing radius of the shading, in pixels of ``image``; 0 smooths nothing.
        strength: Power the shading is divided out at: 0 = none, 1 = all.
        colour: One of `COLOURS`.
        mask: ``(H', W')`` share of each pixel that is the subject, any size, or None.

    Returns:
        ``(albedo, shading)``: ``(H, W, C)`` sRGB codes with any alpha kept, and ``(H, W, 3)``
        shading raised to 1/2.2, 1.0 where the picture is as bright as its albedo.

    Raises:
        ValueError: ``colour`` is not one of `COLOURS`, or a picture has no colour channels.
    """
    if colour not in COLOURS:
        raise ValueError(f"colour must be one of {', '.join(COLOURS)}, not {colour!r}")
    if image.shape[-1] < 3 or albedo.shape[-1] < 3:
        raise ValueError("image and albedo need three colour channels; wire RGB pictures")
    height, width = int(image.shape[0]), int(image.shape[1])
    power = min(max(float(strength), 0.0), 1.0)
    if power <= 0.0:
        return image.float().clone(), torch.ones(height, width, 3)
    small = (int(albedo.shape[0]), int(albedo.shape[1]))
    linear = _to_linear(image[..., :3].float())
    estimate = _to_linear(albedo[..., :3].float()).permute(2, 0, 1)[None]
    reduced = _planes(linear, small)
    weight = None
    if mask is not None:
        weight = _planes(mask.float().clamp(0.0, 1.0)[..., None], small)
    radius = max(float(detail), 0.0) * small[1] / max(width, 1)
    shading = _shading(reduced, estimate, weight, radius, colour)
    shading = F.interpolate(shading, size=(height, width), mode="bilinear", align_corners=False)
    shading = shading.clamp(SHADING_FLOOR, SHADING_CEILING) ** power
    if mask is not None:
        full = _planes(mask.float().clamp(0.0, 1.0)[..., None], (height, width))
        shading = full * shading + (1.0 - full)
    shading = shading[0].permute(1, 2, 0)
    delit = _to_codes(linear / shading)
    if image.shape[-1] > 3:
        delit = torch.cat([delit, image[..., 3:].float()], dim=-1)
    shown = shading.expand(height, width, 3).clamp(0.0, 1.0) ** SHOWN
    return delit.contiguous(), shown.contiguous()
