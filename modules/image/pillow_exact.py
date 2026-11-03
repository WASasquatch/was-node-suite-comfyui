"""Pillow's 8-bit filters over whole batches of tensors, matching Pillow to the byte.

Frames are float ``(batch, height, width, channels)`` from 0 to 1. The filters work on int32
codes from 0 to 255.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

__all__ = ["Adjustments", "adjust", "bloom", "working_copies"]

#: Box passes Pillow spreads one gaussian radius over, per axis.
GAUSSIAN_PASSES = 3

#: Weights Pillow's 8-bit greyscale sums red, green and blue with, over 65536.
LUMA = (19595, 38470, 7471)

#: The fixed point one box pass accumulates in.
BOX_ONE = 1 << 24

#: Side and centre taps of Pillow's ``DETAIL`` kernel, as the float32 values it convolves with.
DETAIL_SIDE = float(np.float32(-1.0) / np.float32(6.0))
DETAIL_CENTRE = float(np.float32(10.0) / np.float32(6.0))

#: Float32 copies of a frame a pipeline here holds at its peak, before a gaussian blur's margin.
BASE_COPIES = 12.0


def _moved(value: float, neutral: float) -> bool:
    """Whether a setting differs from its neutral value. NaN counts as neutral."""
    return value > neutral or value < neutral


class Adjustments(NamedTuple):
    """The eight settings of Image Filter Adjustments, in the order they apply.

    Attributes:
        brightness: Added to every sample. 0.0 skips the step.
        contrast: Every sample is multiplied by this. 1.0 skips the step.
        saturation: Colour enhancement factor. 1.0 skips the step.
        sharpness: Sharpness enhancement factor. 1.0 skips the step.
        blur: Passes of Pillow's 5x5 ``BLUR`` kernel. 0 skips the step.
        gaussian_blur: Gaussian blur radius in pixels. 0.0 skips the step.
        edge_enhance: Share of an ``EDGE_ENHANCE_MORE`` copy mixed in, 0.0 to 1.0.
        detail_enhance: Run Pillow's ``DETAIL`` kernel last.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    blur: int = 0
    gaussian_blur: float = 0.0
    edge_enhance: float = 0.0
    detail_enhance: bool = False

    def filters(self) -> bool:
        """Whether any step past brightness and contrast runs."""
        return (
            _moved(self.saturation, 1.0)
            or _moved(self.sharpness, 1.0)
            or self.blur > 0
            or self.gaussian_blur > 0.0
            or self.edge_enhance > 0.0
            or bool(self.detail_enhance)
        )


def _folded(frames: torch.Tensor, scale: float) -> torch.Tensor:
    """A float32 batch divided by its scale and held inside 0 to 1, or as it is at 1.0."""
    frames = frames.float()
    if scale == 1.0:
        return frames
    return (frames / frames.new_tensor(scale)).clamp_(0.0, 1.0)


def _unfolded(frames: torch.Tensor, scale: float) -> torch.Tensor:
    """A filtered batch multiplied back onto the scale :func:`_folded` divided it by."""
    return frames if scale == 1.0 else frames * scale


def _codes(frames: torch.Tensor) -> torch.Tensor:
    """The 8-bit codes a float batch converts to: 255 times each sample, clipped and truncated.

    Args:
        frames: Float32 batch on a 0 to 1 scale.

    Returns:
        An int32 tensor of the same shape, 0 to 255.
    """
    scaled = torch.nan_to_num(frames * 255.0, nan=0.0)
    return scaled.clamp_(0.0, 255.0).trunc_().to(torch.int32)


def _from_codes(codes: torch.Tensor) -> torch.Tensor:
    """Int32 codes as float32 samples, each divided by 255."""
    return codes.float() / torch.tensor(255.0, device=codes.device)


def _blend(degenerate: torch.Tensor, image: torch.Tensor, factor: float) -> torch.Tensor:
    """``Image.blend(degenerate, image, factor)`` on codes, truncating as Pillow truncates.

    Args:
        degenerate: Int32 codes the factor measures from.
        image: Int32 codes being enhanced.
        factor: 0 answers ``degenerate``, 1 answers ``image``, beyond either extrapolates.

    Returns:
        Int32 codes, 0 to 255.
    """
    low = degenerate.float()
    value = low + (image.float() - low) * float(np.float32(factor))
    return value.clamp_(0.0, 255.0).trunc_().to(torch.int32)


def _alpha_index(channels: int) -> int | None:
    """The channel holding alpha in a frame of this many channels, or None."""
    return channels - 1 if channels in (2, 4) else None


def _box3(codes: torch.Tensor) -> torch.Tensor:
    """Sum of every 3x3 neighbourhood, for the pixels one in from the edge.

    Args:
        codes: Int32 ``(batch, height, width, channels)``, at least 3 on each side.

    Returns:
        Int32 ``(batch, height - 2, width - 2, channels)``.
    """
    rows = codes[:, :, :-2] + codes[:, :, 1:-1] + codes[:, :, 2:]
    return rows[:, :-2] + rows[:, 1:-1] + rows[:, 2:]


def _saturated(codes: torch.Tensor, factor: float) -> torch.Tensor:
    """``ImageEnhance.Color``: blend against the frame's own 8-bit greyscale.

    Args:
        codes: Int32 frames. Fewer than three channels come back unchanged.
        factor: Enhancement factor.

    Returns:
        Int32 codes.
    """
    if codes.shape[-1] < 3:
        return codes
    red, green, blue = LUMA
    grey = (codes[..., 0] * red + codes[..., 1] * green + codes[..., 2] * blue + 32768) >> 16
    degenerate = codes.clone()
    degenerate[..., :3] = grey.unsqueeze(-1)
    return _blend(degenerate, codes, factor)


def _smoothed(codes: torch.Tensor) -> torch.Tensor:
    """Pillow's ``SMOOTH`` kernel, its one-pixel border copied from the source.

    Args:
        codes: Int32 frames.

    Returns:
        Int32 codes. A frame under 3 pixels on a side comes back unchanged.
    """
    height, width = codes.shape[1], codes.shape[2]
    if height < 3 or width < 3:
        return codes.clone()
    total = _box3(codes) + codes[:, 1:-1, 1:-1] * 4
    smoothed = codes.clone()
    smoothed[:, 1:-1, 1:-1] = (total * 2 + 13) // 26
    return smoothed


def _sharpened(codes: torch.Tensor, factor: float) -> torch.Tensor:
    """``ImageEnhance.Sharpness``: blend against a ``SMOOTH`` copy that keeps the alpha.

    Args:
        codes: Int32 frames.
        factor: Enhancement factor.

    Returns:
        Int32 codes.
    """
    degenerate = _smoothed(codes)
    alpha = _alpha_index(codes.shape[-1])
    if alpha is not None:
        degenerate[..., alpha] = codes[..., alpha]
    return _blend(degenerate, codes, factor)


def _box_blurred(codes: torch.Tensor) -> torch.Tensor:
    """One pass of Pillow's 5x5 ``BLUR`` kernel, its two-pixel border copied from the source.

    Args:
        codes: Int32 frames.

    Returns:
        Int32 codes. A frame under 5 pixels on a side comes back unchanged.
    """
    height, width = codes.shape[1], codes.shape[2]
    if height < 5 or width < 5:
        return codes
    inner = codes[:, :, 1:-3] + codes[:, :, 2:-2] + codes[:, :, 3:-1]
    outer = inner + codes[:, :, :-4] + codes[:, :, 4:]
    ring = (
        outer[:, :-4] + outer[:, 1:-3] + outer[:, 2:-2] + outer[:, 3:-1] + outer[:, 4:]
        - inner[:, 1:-3] - inner[:, 2:-2] - inner[:, 3:-1]
    )
    blurred = codes.clone()
    blurred[:, 2:-2, 2:-2] = (ring + 8) >> 4
    return blurred


def _box_radius(radius: float) -> np.float32:
    """The radius each of Pillow's three box passes takes for one gaussian radius.

    Args:
        radius: Gaussian radius in pixels, above 0.

    Returns:
        The box radius as a float32: the whole part is the span, the fraction weights the
        two outermost taps.
    """
    one = np.float32(1.0)
    radius = np.float32(radius)
    sigma2 = (radius * radius) / np.float32(GAUSSIAN_PASSES)
    length = np.float32(math.sqrt(12.0 * float(sigma2) + 1.0))
    whole = np.float32(math.floor((float(length) - 1.0) / 2.0))
    part = (np.float32(2.0) * whole + one) * (whole * (whole + one) - np.float32(3.0) * sigma2)
    part = part / (np.float32(6.0) * (sigma2 - (whole + one) * (whole + one)))
    return np.float32(whole + part)


def _box_pass(codes: torch.Tensor, radius: np.float32, axis: int) -> torch.Tensor:
    """One of Pillow's box passes along an axis, in its 24-bit fixed point.

    Args:
        codes: Int32 frames.
        radius: Box radius from :func:`_box_radius`.
        axis: 2 for along each row, 1 for down each column.

    Returns:
        Int32 codes. The edge pixel stands in for everything past either end.
    """
    span = int(radius)
    weight = int(np.float32(BOX_ONE) / (radius * np.float32(2.0) + np.float32(1.0)))
    edge = (BOX_ONE - (span * 2 + 1) * weight) // 2
    count = codes.shape[axis]
    reach = span + 1
    index = torch.arange(-reach, count + reach, device=codes.device).clamp_(0, count - 1)
    padded = codes.index_select(axis, index)
    running = padded.cumsum(axis, dtype=torch.int32)
    window = running.narrow(axis, span * 2 + 1, count) - running.narrow(axis, 0, count)
    del running
    outer = padded.narrow(axis, 0, count) + padded.narrow(axis, span * 2 + 2, count)
    del padded
    bulk = window.to(torch.int64) * weight + outer.to(torch.int64) * edge
    return ((bulk + (1 << 23)) >> 24).to(torch.int32)


def _gaussian_blurred(codes: torch.Tensor, radius: float) -> torch.Tensor:
    """``ImageFilter.GaussianBlur``: three box passes along the rows, then three down the columns.

    Args:
        codes: Int32 frames.
        radius: Gaussian radius in pixels, above 0.

    Returns:
        Int32 codes.
    """
    box = _box_radius(radius)
    for axis in (2, 1):
        for _ in range(GAUSSIAN_PASSES):
            codes = _box_pass(codes, box, axis)
    return codes


def _edge_enhanced(codes: torch.Tensor, amount: float) -> torch.Tensor:
    """Composite an ``EDGE_ENHANCE_MORE`` copy over the frames through a constant mask.

    Args:
        codes: Int32 frames.
        amount: Share of the enhanced copy, 0.0 to 1.0, rounded half to even onto 0 to 255.

    Returns:
        Int32 codes.
    """
    level = min(255, max(0, round(amount * 255)))
    height, width = codes.shape[1], codes.shape[2]
    enhanced = codes
    if height >= 3 and width >= 3:
        enhanced = codes.clone()
        enhanced[:, 1:-1, 1:-1] = (codes[:, 1:-1, 1:-1] * 10 - _box3(codes)).clamp_(0, 255)
    mixed = codes * (255 - level) + enhanced * level + 128
    return (mixed + (mixed >> 8)) >> 8


def _detailed(codes: torch.Tensor) -> torch.Tensor:
    """Pillow's ``DETAIL`` kernel, summed in float32 in the order Pillow sums it.

    Args:
        codes: Int32 frames.

    Returns:
        Int32 codes. A frame under 3 pixels on a side comes back unchanged.
    """
    height, width = codes.shape[1], codes.shape[2]
    if height < 3 or width < 3:
        return codes
    plane = codes.float()
    total = plane[:, 2:, 1:-1] * DETAIL_SIDE + 0.5
    row = plane[:, 1:-1, :-2] * DETAIL_SIDE + plane[:, 1:-1, 1:-1] * DETAIL_CENTRE
    total = total + (row + plane[:, 1:-1, 2:] * DETAIL_SIDE)
    del row
    total = total + plane[:, :-2, 1:-1] * DETAIL_SIDE
    detailed = codes.clone()
    detailed[:, 1:-1, 1:-1] = total.clamp_(0.0, 255.0).trunc_().to(torch.int32)
    return detailed


def working_copies(frames: torch.Tensor, radius: float = 0.0) -> float:
    """Float32 copies of one frame :func:`adjust` or :func:`bloom` holds at its peak.

    Args:
        frames: ``(batch, height, width, channels)`` batch the pipeline runs over.
        radius: The widest gaussian radius the pipeline blurs by, or 0 for none.

    Returns:
        The count, for :func:`modules.image.accelerate.run_frames`.
    """
    if radius > 0.0 and frames.ndim == 4:
        span = int(_box_radius(radius)) * 2 + 2
        shortest = max(1, min(int(frames.shape[1]), int(frames.shape[2])))
        return BASE_COPIES + 2.0 * span / shortest
    return BASE_COPIES


def adjust(frames: torch.Tensor, settings: Adjustments, scale: float = 1.0) -> torch.Tensor:
    """Apply the settings to a run of frames, in the order of :class:`Adjustments`.

    Args:
        frames: ``(batch, height, width, channels)`` float tensor on its own scale.
        settings: The settings.
        scale: What the frames are divided by to bring them inside 0 to 1, and multiplied
            by again at the end. 1.0 for frames already inside.

    Returns:
        A float32 tensor the shape of ``frames`` on its device. Where only brightness and
        contrast run, the result is not rounded to 8-bit codes.
    """
    frames = _folded(frames, scale)
    if _moved(settings.brightness, 0.0):
        frames = (frames + settings.brightness).clamp(0.0, 1.0)
    if _moved(settings.contrast, 1.0):
        frames = (frames * settings.contrast).clamp(0.0, 1.0)

    if settings.filters():
        codes = _codes(frames)
        if _moved(settings.saturation, 1.0):
            codes = _saturated(codes, settings.saturation)
        if _moved(settings.sharpness, 1.0):
            codes = _sharpened(codes, settings.sharpness)
        for _ in range(max(0, int(settings.blur))):
            codes = _box_blurred(codes)
        if settings.gaussian_blur > 0.0:
            codes = _gaussian_blurred(codes, float(settings.gaussian_blur))
        if settings.edge_enhance > 0.0:
            codes = _edge_enhanced(codes, float(settings.edge_enhance))
        if settings.detail_enhance:
            codes = _detailed(codes)
        frames = _from_codes(codes)
    return _unfolded(frames, scale)


def bloom(frames: torch.Tensor, radius: float, intensity: float,
          scale: float = 1.0) -> torch.Tensor:
    """Screen a glow drawn from the frames' own detail back over them.

    Args:
        frames: ``(batch, height, width, channels)`` float tensor on its own scale.
        radius: Gaussian radius in pixels the detail is separated at. The glow is blurred at
            twice this. 0 leaves the frames as they are.
        intensity: Strength of the glow, 0.0 to 1.0, truncated onto 0 to 255.
        scale: What the frames are divided by to bring them inside 0 to 1, and multiplied
            by again at the end. 1.0 for frames already inside.

    Returns:
        A float32 tensor the shape of ``frames`` on its device. A fourth channel is carried
        through untouched.
    """
    frames = _folded(frames, scale)
    colour = _codes(frames[..., :3])
    if radius > 0.0:
        glow = (colour - _gaussian_blurred(colour, float(radius))).clamp_(min=0)
        glow = _gaussian_blurred(glow, float(radius) * 2.0)
    else:
        glow = torch.zeros_like(colour)
    level = int(255 * intensity)
    glow = (glow * 2).clamp_(max=255) * level // 255
    colour = 255 - (255 - colour) * (255 - glow) // 255
    result = _from_codes(colour)
    if frames.shape[-1] > 3:
        result = torch.cat((result, frames[..., 3:]), dim=-1)
    return _unfolded(result, scale)
