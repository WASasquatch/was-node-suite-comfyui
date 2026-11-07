"""The size BluePixel edits a picture at, and the resampling that brings it there."""

from __future__ import annotations

import torch

from .model import COMPRESSION

#: The pixel count BluePixel edits at when released, 4096 x 4096.
TARGET_PIXELS = 4096 * 4096


def edit_size(width: int, height: int, target_pixels: int = TARGET_PIXELS) -> tuple[int, int]:
    """The size BluePixel's own pipeline picks: near ``target_pixels``, multiples of 64, aspect kept best.

    Args:
        width: The picture's width.
        height: The picture's height.
        target_pixels: The pixel count to scale to, or 0 to keep the picture's own.

    Returns:
        ``(width, height)``, each a multiple of 64 and at least 64.
    """
    w, h = int(width), int(height)
    if target_pixels > 0:
        scale = (target_pixels / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
    ratio = width / height
    step = COMPRESSION
    w_lo, w_hi = max(step, (w - step) // step * step), (w + step) // step * step
    h_lo, h_hi = max(step, (h - step) // step * step), (h + step) // step * step
    best, best_err = (w, h), 1e9
    for cw in range(w_lo, w_hi + 1, step):
        for ch in range(h_lo, h_hi + 1, step):
            err = abs(cw / ch - ratio)
            if err < best_err:
                best_err, best = err, (cw, ch)
    return best


def resize(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Pictures ``(B, H, W, C)`` in ``[0, 1]`` brought to ``width`` x ``height`` RGB by 8-bit Lanczos."""
    import numpy as np
    from PIL import Image

    out = []
    for image in images:
        array = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        picture = Image.fromarray(array, "RGB")
        if picture.size != (width, height):
            picture = picture.resize((width, height), Image.LANCZOS)
        out.append(torch.from_numpy(np.asarray(picture).astype(np.float32) / 255.0))
    return torch.stack(out)
