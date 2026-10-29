"""Vivid-light sharpening, the high-pass frequency-separation retouch.

:func:`vivid_sharpen` sharpens one picture, :func:`sharpen_batch` a whole batch with each
stage of the stack exposed. Both run in torch on ComfyUI's compute device, a run of frames
at a time.
"""

from __future__ import annotations

__all__ = ["sharpen_batch", "vivid_sharpen"]

import torch
import torch.nn.functional as functional
from PIL import Image

from .accelerate import run_frames
from .blend_modes import blend, ceiling_of

#: Float32 copies of a frame the sharpening stack holds at its peak.
WORKING_COPIES = 16
from .filters import _blurred, _picture, _planes


def _kernel(radius: float, mode: str, span: int) -> torch.Tensor:
    """Build one axis of a separable blur kernel.

    Args:
        radius: Blur radius in pixels, which is the Gaussian's standard deviation.
        mode: ``"box"`` for a flat kernel, anything else for a Gaussian.
        span: Samples the kernel reaches either side of its centre.

    Returns:
        A ``(2 * span + 1,)`` tensor summing to 1.
    """
    size = span * 2 + 1
    if mode == "box":
        curve = torch.ones(size, dtype=torch.float32)
    else:
        offsets = torch.arange(size, dtype=torch.float32) - span
        curve = torch.exp(-offsets ** 2 / (2 * radius ** 2))
    return curve / curve.sum()


def _blur(x: torch.Tensor, radius: float, mode: str) -> torch.Tensor:
    """Blur a ``(batch, channels, height, width)`` tensor, reflecting at the edges."""
    channels = x.shape[1]
    # A reflecting pad reaches one short of the side it mirrors, so a frame narrower than the
    # radius takes the widest kernel that fits it.
    span = min(max(1, int(radius)), min(x.shape[-2], x.shape[-1]) - 1)
    if span < 1:
        return x
    curve = _kernel(radius, mode, span).to(x.device, x.dtype)
    rows = curve.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    columns = curve.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    across = functional.conv2d(
        functional.pad(x, (span, span, 0, 0), mode="reflect"), rows, groups=channels
    )
    return functional.conv2d(
        functional.pad(across, (0, 0, span, span), mode="reflect"), columns, groups=channels
    )


def _high_pass(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Invert a picture and blur it by a gaussian radius.

    Args:
        plane: ``(height, width, channels)`` or ``(batch, height, width, channels)``
            floats on a 0 to 1 scale.
        radius: Gaussian blur radius in pixels.

    Returns:
        The blurred inverse, the shape that went in.
    """
    inverted = 1.0 - plane
    if inverted.dim() == 3:
        return _blurred(inverted, radius)
    height, width, channels = inverted.shape[-3:]
    # The batch rides on the channel axis, which each blur pass treats independently.
    folded = inverted.reshape(-1, height, width, channels).permute(1, 2, 0, 3)
    blurred = _blurred(folded.reshape(height, width, -1), radius)
    spread = blurred.reshape(height, width, -1, channels).permute(2, 0, 1, 3)
    return spread.reshape(inverted.shape)


def _adjust(x: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    """Trim a layer's contrast around mid grey, then its brightness."""
    return ((x - 0.5) * contrast + 0.5) * brightness


def _weighted(base: torch.Tensor, layer: torch.Tensor, weight: float) -> torch.Tensor:
    """Mix a layer over its base at a weight, answering the layer alone at 1.0."""
    return layer if weight == 1.0 else base * (1.0 - weight) + layer * weight


def _stacked(
    plane: torch.Tensor,
    high_pass: torch.Tensor,
    vivid_opacity: float,
    overlay_opacity: float,
    strength: float,
) -> torch.Tensor:
    """Blend a high-pass layer back over a picture in vivid light, then in overlay.

    Args:
        plane: ``(..., 3)`` colours on a 0 to 1 scale.
        high_pass: The high-pass layer, shaped as ``plane``.
        vivid_opacity: Weight of the vivid-light pass.
        overlay_opacity: Weight of the overlay pass.
        strength: Weight the finished stack is mixed back at.

    Returns:
        The sharpened plane, the shape that went in.
    """
    result = _weighted(plane, blend(plane, high_pass, "vivid-light"), vivid_opacity)
    result = _weighted(plane, blend(plane, result, "overlay"), overlay_opacity)
    return _weighted(plane, result, strength)


def vivid_sharpen(image, radius: float = 5, strength: float = 1.0):
    """Sharpen a picture through an inverted, blurred high-pass layer.

    Args:
        image: An 8-bit PIL image, or a float tensor shaped ``(height, width, channels)``
            or ``(batch, height, width, channels)``. Light above 1.0 is carried through
            the stack and restored on the way out, and a fourth channel is carried through
            untouched.
        radius: Gaussian blur radius in pixels used to build the high-pass layer. Larger
            radii accent broader structure; a radius near 1 accents fine texture.
        strength: How far the result is mixed back over the original, in ``[0, 1]``.

    Returns:
        The sharpened picture, in the form that arrived, the same size and held inside the
        range that arrived.
    """
    picture = isinstance(image, Image.Image)
    plane = _planes(image) if picture else image
    colour, carried = plane[..., :3], plane[..., 3:]
    ceiling = ceiling_of(colour)

    def work(bands: torch.Tensor) -> torch.Tensor:
        bands = bands / ceiling
        stacked = _stacked(bands, _high_pass(bands, radius), 1.0, 1.0, strength)
        return stacked.clamp(0.0, 1.0) * ceiling

    result = run_frames(colour, work, factor=WORKING_COPIES, node="Vivid Sharpen")
    if carried.shape[-1]:
        result = torch.cat((result, carried), dim=-1)
    return _picture(result, image.mode) if picture else result


def sharpen_batch(
    images: torch.Tensor,
    radius_highpass: float,
    radius_blur: float,
    blur_mode: str,
    hp_brightness: float,
    hp_contrast: float,
    vivid_opacity: float,
    overlay_opacity: float,
    strength: float,
) -> torch.Tensor:
    """Sharpen a whole batch, with each stage of the stack exposed.

    Args:
        images: ``(batch, height, width, channels)`` tensor. Light above 1.0 is carried
            through the stack and restored on the way out.
        radius_highpass: Blur radius for the inverted high-pass layer, in pixels.
        radius_blur: Blur radius applied to that layer a second time, in pixels.
        blur_mode: ``"box"`` or ``"gaussian"``, used for both blurs.
        hp_brightness: Brightness multiplier on the high-pass layer.
        hp_contrast: Contrast factor on the high-pass layer, around mid grey.
        vivid_opacity: Weight of the vivid-light blend.
        overlay_opacity: Weight of the overlay blend.
        strength: How far the finished result is mixed over the original. Above 1.0 pushes
            past it, which exaggerates the accent.

    Returns:
        A tensor of the same shape, held inside the range that arrived.
    """
    ceiling = ceiling_of(images)

    def work(batch: torch.Tensor) -> torch.Tensor:
        plane = batch / ceiling
        high_pass = _blur((1.0 - plane).permute(0, 3, 1, 2), radius_highpass, blur_mode)
        high_pass = _adjust(high_pass, hp_brightness, hp_contrast).clamp(0.0, 1.0)
        high_pass = _blur(high_pass, radius_blur, blur_mode).permute(0, 2, 3, 1)
        result = _stacked(plane, high_pass, vivid_opacity, overlay_opacity, strength)
        return result.clamp(0.0, 1.0) * ceiling

    return run_frames(images, work, factor=WORKING_COPIES, node="Vivid Sharpen (V2)")
