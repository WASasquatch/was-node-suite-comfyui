"""Metering and softening the decoded frames a MiniMax H3 segment carries forward.

Texture is measured against contrast, so a brighter clip does not read as a sharper one.
Frames are shaped ``(frames, height, width, channels)`` in ``[0, 1]``.
"""

from __future__ import annotations

__all__ = [
    "DEAD_BAND",
    "GAIN_LIMITS",
    "MAX_SIGMA",
    "SIGMA_STEP",
    "excess",
    "levelled",
    "softened",
    "texture",
]

#: Largest blur applied to any frame, and the step the search walks.
MAX_SIGMA = 1.6
SIGMA_STEP = 0.05

#: How far above target a block may sit before it is softened at all.
DEAD_BAND = 1.02

#: Smallest and largest per-channel gain a levelling may apply.
GAIN_LIMITS = (0.7, 1.4)

#: Exponent taking display values to linear light and back.
GAMMA = 2.2

#: Weights luma takes from red, green and blue.
LUMA = (0.299, 0.587, 0.114)

#: Level a row or column must reach to count as carrying picture.
SIGNAL = 0.02


def _luma(frames):
    """The luma of a batch of frames, cropped to the rows and columns carrying picture.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor.

    Returns:
        A ``(frames, 1, height, width)`` tensor.
    """
    weights = frames.new_tensor(LUMA)
    grey = (frames[..., :3] * weights).sum(dim=-1).unsqueeze(1)
    if grey.shape[-1] > 8 and grey.shape[-2] > 8:
        rows = grey.amax(dim=(0, 1, 3)) > SIGNAL
        columns = grey.amax(dim=(0, 1, 2)) > SIGNAL
        if bool(rows.any()) and bool(columns.any()):
            grey = grey[:, :, rows][:, :, :, columns]
    return grey


def texture(frames) -> float:
    """How much fine detail a batch of frames carries for the contrast it has.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor in ``[0, 1]``.

    Returns:
        Laplacian variance over luma variance, and 0.0 for a flat batch.
    """
    import torch
    import torch.nn.functional as functional

    grey = _luma(frames.float())
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=grey.dtype, device=grey.device,
    ).view(1, 1, 3, 3)
    detail = float(functional.conv2d(grey, kernel, padding=1).var())
    contrast = float(grey.var())
    return detail / max(contrast, 1e-9)


def levelled(frames, anchor):
    """Frames brought to the colour of an anchor by a per-channel gain in linear light.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor in ``[0, 1]``.
        anchor: Frames whose colour is matched, the same shape.

    Returns:
        ``(levelled, gains)``, the corrected frames and the three gains applied.
    """
    linear = frames.float().clamp(0.0, 1.0) ** GAMMA
    wanted = anchor.float().clamp(0.0, 1.0) ** GAMMA
    low, high = GAIN_LIMITS
    gains = []
    corrected = linear.clone()
    for channel in range(min(3, linear.shape[-1])):
        source = linear[..., channel].median()
        target = wanted[..., channel].median()
        gain = 1.0
        if float(source) > 1e-6:
            gain = max(low, min(high, float(target / source)))
        corrected[..., channel] = linear[..., channel] * gain
        gains.append(gain)
    return (corrected.clamp(0.0, 1.0) ** (1.0 / GAMMA)).to(frames.dtype), tuple(gains)


def _blurred(frames, sigma: float):
    """One block of frames blurred by a separable Gaussian.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor.
        sigma: Standard deviation in pixels.

    Returns:
        A tensor of the same shape, clamped to ``[0, 1]``.
    """
    from ..image.convolve import gaussian_blur

    planes = frames.movedim(-1, 1)
    return gaussian_blur(planes, sigma=sigma).movedim(1, -1).clamp(0.0, 1.0)


def excess(frames, anchor, radius: int = 4):
    """Where a batch carries more fine detail than an anchor does, pixel by pixel.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor in ``[0, 1]``.
        anchor: Frames the detail is measured against, the same width and height.
        radius: Half the window local detail is averaged over.

    Returns:
        A ``(frames, height, width, 1)`` tensor in ``[0, 1]``, 0 where detail is at or
        below the anchor and 1 where it is at least twice the anchor.
    """
    import torch
    import torch.nn.functional as functional

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=frames.device,
    ).view(1, 1, 3, 3)
    size = radius * 2 + 1

    def detail(batch):
        weights = batch.new_tensor(LUMA)
        grey = (batch[..., :3].float() * weights).sum(dim=-1).unsqueeze(1)
        edges = functional.conv2d(grey, kernel, padding=1).abs()
        return functional.avg_pool2d(edges, size, stride=1, padding=radius,
                                     count_include_pad=False)

    here = detail(frames)
    there = detail(anchor).mean(dim=0, keepdim=True).mean()
    ratio = here / there.clamp(min=1e-6)
    return (ratio - 1.0).clamp(0.0, 1.0).movedim(1, -1)


def _sigma_for(frames, target: float) -> float:
    """The smallest blur that brings a block's texture down to a target.

    Args:
        frames: One block of frames.
        target: The texture reading to reach.

    Returns:
        A blur in pixels, 0.0 where none is needed and :data:`MAX_SIGMA` where the
        target is out of reach.
    """
    steps = int(round(MAX_SIGMA / SIGMA_STEP))
    for step in range(1, steps + 1):
        sigma = step * SIGMA_STEP
        if texture(_blurred(frames, sigma)) <= target:
            return sigma
    return MAX_SIGMA


def softened(frames, anchor, target: float):
    """Frames softened only where they carry more fine detail than an anchor.

    Args:
        frames: A ``(frames, height, width, channels)`` tensor in ``[0, 1]``.
        anchor: Frames the detail is measured against.
        target: The texture reading to aim for. 0.0 or less returns the frames unchanged.

    Returns:
        ``(frames, sigma, covered)``, the softened frames, the blur used and the share of
        the picture it reached.
    """
    if target <= 0.0 or frames.shape[0] == 0:
        return frames, 0.0, 0.0
    if texture(frames) <= target * DEAD_BAND:
        return frames, 0.0, 0.0

    sigma = _sigma_for(frames[:1], target)
    weight = excess(frames, anchor)
    blurred = _blurred(frames, sigma)
    mixed = frames * (1.0 - weight) + blurred * weight
    return mixed.to(frames.dtype), sigma, float(weight.mean())
