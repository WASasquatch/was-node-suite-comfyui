"""BluePixel's sampling schedule: evenly spaced times, shifted for the image's token count."""

from __future__ import annotations

import math

from .model import COMPRESSION


def mu(tokens: int, steps: int) -> float:
    """The shift FLUX.2 Klein's empirical rule gives ``tokens`` image tokens over ``steps`` steps."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if tokens > 4300:
        return float(a2 * tokens + b2)
    m_200 = a2 * tokens + b2
    m_10 = a1 * tokens + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * steps + b)


def sigmas(steps: int, tokens: int) -> list[float]:
    """``steps + 1`` sigmas from 1 to 0: ``linspace(1, 1 / steps)`` shifted by :func:`mu`, then 0.

    Args:
        steps: Denoising steps.
        tokens: Image tokens, ``(height / 64) * (width / 64)``.
    """
    import numpy as np

    shift = math.exp(mu(tokens, steps))
    times = np.linspace(1.0, 1 / steps, steps).astype(np.float32)
    shifted = shift / (shift + (1 / times - 1) ** 1.0)
    return [float(value) for value in shifted.astype(np.float32)] + [0.0]


def tokens_of(latent: dict) -> int:
    """Image tokens a LATENT samples as, read through the ratio it was made at when it carries one."""
    samples = latent["samples"]
    height, width = int(samples.shape[-2]), int(samples.shape[-1])
    ratio = latent.get("downscale_ratio_spacial")
    if ratio is not None and int(ratio) != COMPRESSION:
        height = round(height * int(ratio) / COMPRESSION)
        width = round(width * int(ratio) / COMPRESSION)
    return height * width
