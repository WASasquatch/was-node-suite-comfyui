"""CIEDE2000 colour difference between sRGB picture codes.

Codes are converted to CIE L*a*b* under D65 and compared with the CIEDE2000 formula, with
``kL = kC = kH = 1``. Every function works in float64 on ``(..., 3)`` tensors.
"""

from __future__ import annotations

import torch

__all__ = ["ciede2000", "srgb_to_lab", "srgb_delta_e"]

#: D65 reference white in XYZ, Y normalised to 1.
WHITE_D65 = (0.95047, 1.0, 1.08883)

#: Linear sRGB to XYZ, IEC 61966-2-1.
SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)


def _linear(codes: torch.Tensor) -> torch.Tensor:
    """The sRGB transfer undone, mirrored below 0 and extended as a power law above 1."""
    magnitude = codes.abs()
    straight = magnitude / 12.92
    curved = ((magnitude + 0.055) / 1.055) ** 2.4
    return torch.sign(codes) * torch.where(magnitude <= 0.04045, straight, curved)


def srgb_to_lab(codes: torch.Tensor) -> torch.Tensor:
    """Convert sRGB picture codes to CIE L*a*b* under D65.

    Args:
        codes: ``(..., 3)`` codes, 0 to 1 for a picture.

    Returns:
        ``(..., 3)`` float64 holding L*, a* and b*.
    """
    light = _linear(codes.to(torch.float64))
    matrix = torch.tensor(SRGB_TO_XYZ, dtype=torch.float64, device=light.device)
    xyz = light @ matrix.T
    white = torch.tensor(WHITE_D65, dtype=torch.float64, device=light.device)
    ratio = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = torch.where(ratio > epsilon, ratio.clamp(min=0.0) ** (1.0 / 3.0), (kappa * ratio + 16.0) / 116.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([lightness, a, b], dim=-1)


def ciede2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 difference between two sets of L*a*b* colours.

    Args:
        lab1: ``(..., 3)`` L*a*b*.
        lab2: ``(..., 3)`` L*a*b*, broadcastable against ``lab1``.

    Returns:
        ``(...)`` float64 differences.
    """
    lab1 = lab1.to(torch.float64)
    lab2 = lab2.to(torch.float64)
    l1, a1, b1 = lab1.unbind(-1)
    l2, a2, b2 = lab2.unbind(-1)

    c1 = torch.sqrt(a1 * a1 + b1 * b1)
    c2 = torch.sqrt(a2 * a2 + b2 * b2)
    c_bar7 = ((c1 + c2) / 2.0) ** 7
    g = 0.5 * (1.0 - torch.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = torch.sqrt(a1p * a1p + b1 * b1)
    c2p = torch.sqrt(a2p * a2p + b2 * b2)

    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0
    h1p = torch.where((a1p == 0) & (b1 == 0), torch.zeros_like(h1p), h1p)
    h2p = torch.where((a2p == 0) & (b2 == 0), torch.zeros_like(h2p), h2p)

    chroma_zero = (c1p * c2p) == 0
    spread = h2p - h1p
    dhp = torch.where(spread > 180.0, spread - 360.0, torch.where(spread < -180.0, spread + 360.0, spread))
    dhp = torch.where(chroma_zero, torch.zeros_like(dhp), dhp)

    dlp = l2 - l1
    dcp = c2p - c1p
    dhp_big = 2.0 * torch.sqrt(c1p * c2p) * torch.sin(torch.deg2rad(dhp) / 2.0)

    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    total = h1p + h2p
    h_bar = torch.where(
        (h1p - h2p).abs() <= 180.0,
        total / 2.0,
        torch.where(total < 360.0, (total + 360.0) / 2.0, (total - 360.0) / 2.0),
    )
    h_bar = torch.where(chroma_zero, total, h_bar)

    t = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(h_bar - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * h_bar))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * h_bar + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * h_bar - 63.0))
    )
    d_theta = 30.0 * torch.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p ** 7
    rc = 2.0 * torch.sqrt(c_bar_p7 / (c_bar_p7 + 25.0 ** 7))
    offset = (l_bar - 50.0) ** 2
    sl = 1.0 + 0.015 * offset / torch.sqrt(20.0 + offset)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -torch.sin(torch.deg2rad(2.0 * d_theta)) * rc

    lightness = dlp / sl
    chroma = dcp / sc
    hue = dhp_big / sh
    squared = lightness * lightness + chroma * chroma + hue * hue + rt * chroma * hue
    return torch.sqrt(squared.clamp(min=0.0))


def srgb_delta_e(codes1: torch.Tensor, codes2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 difference between two sets of sRGB picture codes.

    Args:
        codes1: ``(..., 3)`` codes.
        codes2: ``(..., 3)`` codes, broadcastable against ``codes1``.

    Returns:
        ``(...)`` float64 differences.
    """
    return ciede2000(srgb_to_lab(codes1), srgb_to_lab(codes2))
