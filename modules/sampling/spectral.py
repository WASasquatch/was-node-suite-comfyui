"""Growing and shrinking a latent in a spectral basis.

:func:`downscale` truncates a latent to a smaller grid, :func:`expand` grows one back, and
:func:`kappa` rescales a grown latent.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "TRANSFORMS",
    "dct_2d",
    "idct_2d",
    "downscale",
    "expand",
    "kappa",
]

#: The bases a transition can grow the latent in.
TRANSFORMS = ("dct", "dwt", "fft")


def _dct_1d(x: torch.Tensor, axis: int) -> torch.Tensor:
    """DCT-II along one axis, orthonormal, matching ``scipy.fft.dct(type=2, norm="ortho")``."""
    x = x.transpose(axis, -1)
    shape, n = x.shape, x.shape[-1]
    # The even samples followed by the odd ones reversed turns a DCT-II into a single FFT.
    folded = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
    k = torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2 * n))
    twiddled = 2 * (torch.fft.fft(folded, dim=-1) * torch.polar(torch.ones_like(k), -k)).real
    scale = torch.full((n,), math.sqrt(2 * n), device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(4 * n)
    return (twiddled / scale).reshape(shape).transpose(axis, -1)


def _idct_1d(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Inverse of :func:`_dct_1d`, matching ``scipy.fft.idct(type=2, norm="ortho")``."""
    x = x.transpose(axis, -1)
    shape, n = x.shape, x.shape[-1]
    scale = torch.full((n,), math.sqrt(2 * n), device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(4 * n)
    weighted = (x / 2) * scale
    k = torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2 * n))
    # The imaginary half a real inverse needs, which is the spectrum reversed and negated.
    imaginary = torch.cat(
        [torch.zeros_like(weighted[..., :1]), -weighted.flip(-1)[..., :-1]], dim=-1
    )
    spectrum = torch.complex(weighted, imaginary) * torch.polar(torch.ones_like(k), k)
    folded = torch.fft.irfft(spectrum, n=n, dim=-1)
    out = torch.empty_like(folded)
    out[..., 0::2] = folded[..., : n - n // 2]
    out[..., 1::2] = folded.flip(-1)[..., : n // 2]
    return out.reshape(shape).transpose(axis, -1)


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """The orthonormal DCT-II of the trailing two axes.

    Args:
        x: Any real tensor whose last two axes are the grid to transform.

    Returns:
        The coefficients, same shape and dtype.
    """
    return _dct_1d(_dct_1d(x, -1), -2)


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`dct_2d`.

    Args:
        x: Coefficients as :func:`dct_2d` returns them.

    Returns:
        The reconstructed grid, same shape and dtype.
    """
    return _idct_1d(_idct_1d(x, -2), -1)


def kappa(t: float, r: float) -> float:
    """The factor a latent is rescaled by once it has been grown.

    Args:
        t: Flow-matching time at the transition.
        r: Ratio of the new scale to the old one.

    Returns:
        The rescaling factor.
    """
    return r / (1.0 + (r - 1.0) * t)


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype a transform runs in: at least single precision, never below the input's."""
    return torch.promote_types(dtype, torch.float32)


def _noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """Standard normal noise for one transition, drawn on the CPU, where a seed travels."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    drawn = torch.randn(shape, generator=generator, dtype=torch.float32)
    return drawn.to(device=device, dtype=dtype)


def downscale(x: torch.Tensor, scale: float, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Truncate a latent to a fraction of its size by dropping its high DCT coefficients.

    Args:
        x: Latent whose trailing two axes are spatial.
        scale: Fraction of the current size to keep. A value of 1.0 or above is a no-op.
        size: Exact ``(height, width)`` to keep, in place of ``scale``.

    Returns:
        The truncated latent, or ``x`` unchanged when nothing is dropped.
    """
    height, width = x.shape[-2], x.shape[-1]
    if size is None:
        if scale >= 1.0:
            return x
        kept_h, kept_w = round(height * scale), round(width * scale)
    else:
        kept_h, kept_w = min(int(size[0]), height), min(int(size[1]), width)
        if (kept_h, kept_w) == (height, width):
            return x
    coefficients = dct_2d(x.to(_working_dtype(x.dtype)))
    return idct_2d(coefficients[..., :kept_h, :kept_w].contiguous()).to(x.dtype)


def expand(
    x: torch.Tensor,
    target: tuple[int, int],
    t: float,
    transform: str,
    seed: int,
) -> torch.Tensor:
    """Grow a latent to ``target``, filling the frequencies it never carried with noise.

    Args:
        x: Latent whose trailing two axes are spatial.
        target: The ``(height, width)`` to grow to.
        t: Flow-matching time at the transition, which doubles as the noise amplitude: early in
            a trajectory the new detail is mostly noise, late it is mostly nothing.
        transform: One of :data:`TRANSFORMS`.
        seed: Seed for the noise filling the new frequencies.

    Returns:
        The grown latent, same dtype, trailing axes ``target``.

    Raises:
        ValueError: ``transform`` is not a known basis, ``target`` is smaller than ``x``, or the
            basis is ``dwt`` and the growth is not exactly a doubling.
    """
    if transform not in TRANSFORMS:
        raise ValueError(
            "transform must be one of " + ", ".join(TRANSFORMS) + f", not {transform!r}."
        )
    height, width = x.shape[-2], x.shape[-1]
    target_h, target_w = target
    if target_h < height or target_w < width:
        raise ValueError(
            f"expand grows a latent, so the target {target_h}x{target_w} cannot be smaller than "
            f"the source {height}x{width}."
        )

    work = x.to(_working_dtype(x.dtype))
    if transform == "dct":
        grown = _expand_dct(work, (target_h, target_w), t, seed)
    elif transform == "fft":
        grown = _expand_fft(work, (target_h, target_w), t, seed)
    else:
        if target_h != height * 2 or target_w != width * 2:
            raise ValueError(
                f"the dwt basis doubles a latent, so it cannot grow {height}x{width} to "
                f"{target_h}x{target_w}. Use dct or fft for any other ratio."
            )
        grown = _expand_haar(work, t, seed)
    return grown.to(x.dtype)


def _expand_dct(x: torch.Tensor, target: tuple[int, int], t: float, seed: int) -> torch.Tensor:
    """Grow in the cosine basis, the source coefficients in the corner of a larger noise field."""
    height, width = x.shape[-2], x.shape[-1]
    coefficients = _noise((*x.shape[:-2], *target), seed, x.device, x.dtype) * t
    coefficients[..., :height, :width] = dct_2d(x)
    return idct_2d(coefficients)


def _expand_fft(x: torch.Tensor, target: tuple[int, int], t: float, seed: int) -> torch.Tensor:
    """Grow in the Fourier basis, the source spectrum centred inside a larger noise spectrum."""
    height, width = x.shape[-2], x.shape[-1]
    target_h, target_w = target
    top, left = (target_h - height) // 2, (target_w - width) // 2

    source = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
    real = _noise((*x.shape[:-2], target_h, target_w), seed, x.device, x.dtype)
    imaginary = _noise((*x.shape[:-2], target_h, target_w), seed + 1, x.device, x.dtype)
    spectrum = torch.fft.fftshift(torch.complex(real, imaginary) * t, dim=(-2, -1))
    spectrum[..., top : top + height, left : left + width] = source
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), norm="ortho").real


def _expand_haar(x: torch.Tensor, t: float, seed: int) -> torch.Tensor:
    """Grow by doubling, the latent taken as the average band and the details drawn as noise."""
    horizontal, vertical, diagonal = (
        _noise(x.shape, seed + offset, x.device, x.dtype) * t for offset in (1, 2, 3)
    )
    # Orthonormal Haar synthesis of each 2x2 output block from the four bands.
    top_left = (x + horizontal + vertical + diagonal) * 0.5
    top_right = (x + horizontal - vertical - diagonal) * 0.5
    bottom_left = (x - horizontal + vertical - diagonal) * 0.5
    bottom_right = (x - horizontal - vertical + diagonal) * 0.5

    out = torch.empty(
        (*x.shape[:-2], x.shape[-2] * 2, x.shape[-1] * 2), device=x.device, dtype=x.dtype
    )
    out[..., 0::2, 0::2] = top_left
    out[..., 0::2, 1::2] = top_right
    out[..., 1::2, 0::2] = bottom_left
    out[..., 1::2, 1::2] = bottom_right
    return out
