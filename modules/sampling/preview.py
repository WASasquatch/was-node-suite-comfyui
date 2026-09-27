"""The per-step sampling callback, with a cached previewer and a preview decoded at its shown size.

Mirrors ComfyUI's ``latent_preview.prepare_callback``: progress every step, a preview picture on
every ``every``-th step and the last.
"""

from __future__ import annotations

import math

import comfy.utils
import latent_preview
import torch
from comfy.cli_args import args

#: Pixels per latent cell assumed when sizing a latent down to the preview.
LATENT_SCALE = 8

#: The previewer last built, as ``(key, previewer)``. One at a time, so a model switch frees it.
_cached: list = []


def previewer_for(model):
    """The previewer ComfyUI would build for a model, built once and kept.

    Args:
        model: A ``ModelPatcher``.

    Returns:
        A ``latent_preview.LatentPreviewer``, or None when previews are off.
    """
    latent_format = model.model.latent_format
    key = (
        str(args.preview_method),
        getattr(latent_format, "taesd_decoder_name", None),
        type(latent_format).__name__,
        str(model.load_device),
    )
    if _cached and _cached[0][0] == key:
        return _cached[0][1]
    previewer = latent_preview.get_previewer(model.load_device, latent_format)
    _cached[:] = [(key, previewer)]
    return previewer


def shrunk(x0: torch.Tensor, edge: int) -> torch.Tensor:
    """The first latent of a batch, scaled down so its decode is no larger than the preview.

    Args:
        x0: ``(batch, channels, height, width)`` or ``(batch, channels, frames, height, width)``.
        edge: Longest side of the shown preview, in pixels.

    Returns:
        A batch of one, spatially no larger than ``edge`` pixels once decoded.
    """
    first = x0[:1]
    height, width = int(first.shape[-2]), int(first.shape[-1])
    limit = max(1, math.ceil(edge / LATENT_SCALE))
    if max(height, width) <= limit:
        return first
    scale = limit / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    if first.ndim == 5:
        frames = first.movedim(2, 1).reshape(-1, first.shape[1], height, width)
        frames = torch.nn.functional.interpolate(frames, size=size, mode="area")
        return frames.reshape(1, first.shape[2], first.shape[1], *size).movedim(1, 2)
    return torch.nn.functional.interpolate(first, size=size, mode="area")


def prepare_callback(model, steps: int, x0_output_dict=None, every: int = 1, edge: int | None = None):
    """A sampler callback reporting progress and a preview.

    Args:
        model: A ``ModelPatcher``.
        steps: Steps the progress bar counts to.
        x0_output_dict: A dictionary that receives the latest ``x0``, as ComfyUI's does.
        every: Decode a preview on every this many steps, and always on the last.
        edge: Longest side of the preview in pixels, ComfyUI's preview size when None.

    Returns:
        ``callback(step, x0, x, total_steps)``.
    """
    previewer = previewer_for(model)
    pbar = comfy.utils.ProgressBar(steps)
    every = max(1, int(every))
    edge = int(edge or latent_preview.MAX_PREVIEW_RESOLUTION)
    decodes = isinstance(previewer, latent_preview.TAESDPreviewerImpl)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        picture = None
        if previewer is not None and (step % every == 0 or step + 1 >= total_steps):
            if x0.is_nested:
                x0 = x0.tensors[0]
            latent = shrunk(x0, edge) if decodes else x0
            picture = previewer.decode_latent_to_preview_image("JPEG", latent)
        pbar.update_absolute(step + 1, total_steps, picture)

    return callback
