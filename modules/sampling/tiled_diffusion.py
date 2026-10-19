"""A model predicting tile by tile inside any sampler, the tiles blended under weights summing to one.

Tiling bounds are sigmas on the model's own curve; a flow shift covers only the steps before them.
"""

from __future__ import annotations

import math

import torch

from .. import log
from . import schedules, tiles

__all__ = ["ALIGN", "SETTINGS", "WRAPPER", "patch", "settle"]

logger = log.get_logger("sampling.tiled_diffusion")

#: Key the wrappers are registered under.
WRAPPER = "was_tiled_diffusion"

#: ``model_options`` key holding the patch's ``{"start": ..., "end": ...}``.
SETTINGS = "was_tiled_diffusion"

#: Tile starts and sizes fall on multiples of this many latent pixels.
ALIGN = 8

#: Image pixels per latent pixel where the model names none.
DEFAULT_SCALE = 8

#: Relative difference within which a step's sigma counts as on a noise bound.
SIGMA_TOLERANCE = 1e-3


def patch(model, tile: tuple, overlap: int, start: float, end: float):
    """A clone of a model that predicts over overlapping tiles while the noise is in range.

    Args:
        model: A model patcher.
        tile: ``(height, width)`` of a tile in image pixels.
        overlap: Least image pixels neighbouring tiles share.
        start: Share of the model's own schedule, 0 to 1, where tiling begins.
        end: Share of the model's own schedule, 0 to 1, where tiling ends.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.WrappersMP
    latent_format = model.get_model_object("latent_format")
    scale = float(getattr(latent_format, "spacial_downscale_ratio", 0) or DEFAULT_SCALE)
    throughout = start <= 0.0 and end >= 1.0
    high, low = _bounds(schedules.native_sampling(model), start, end)
    plans = {}

    def planned(height: int, width: int):
        key = (height, width)
        if key not in plans:
            grid = tiles.plan((height, width), scale, tile, overlap, ALIGN)
            plans[key] = (grid, tiles.tile_weights(grid))
        return plans[key]

    def prepare(executor, patcher, noise_shape, conds, *args, **kwargs):
        shape = list(noise_shape)
        if throughout and len(shape) >= 4:
            tile_h, tile_w = planned(int(shape[-2]), int(shape[-1]))[0].tile
            shape[-2], shape[-1] = min(shape[-2], tile_h), min(shape[-1], tile_w)
        return executor(patcher, shape, conds, *args, **kwargs)

    def apply(executor, x, t, c_concat=None, c_crossattn=None, control=None,
              transformer_options={}, **kwargs):
        whole = (x, t, c_concat, c_crossattn, control, transformer_options)
        if x.ndim < 4 or "latent_shapes" in kwargs or not (
            throughout or _in_range(t, transformer_options, high, low)
        ):
            return executor(*whole, **kwargs)
        size = (int(x.shape[-2]), int(x.shape[-1]))
        grid, weights = planned(*size)
        if grid.count == 1 or not _croppable(control):
            return executor(*whole, **kwargs)
        return _tiled(executor, grid, weights, size, whole, kwargs)

    patched = model.clone()
    for kind in (wrappers.PREPARE_SAMPLING, wrappers.APPLY_MODEL):
        patched.remove_wrappers_with_key(kind, WRAPPER)
    patched.add_wrapper_with_key(wrappers.PREPARE_SAMPLING, WRAPPER, prepare)
    patched.add_wrapper_with_key(wrappers.APPLY_MODEL, WRAPPER, apply)
    patched.model_options[SETTINGS] = {"start": float(start), "end": float(end)}
    settle(patched, "Tiled Diffusion Model Patch")
    return patched


def settle(model, node: str) -> None:
    """Join a tiled model's flow shift onto its own curve where the tiles take over.

    Args:
        model: A model patcher, patched in place when it carries :data:`SETTINGS`.
        node: The node's display name, for the log line.
    """
    settings = model.model_options.get(SETTINGS)
    if not settings:
        return
    start = float(settings.get("start", 0.0))
    own = schedules.native_sampling(model)
    current = model.get_model_object("model_sampling")
    joined = schedules.handoff(current, own, start)
    if joined is None:
        return
    model.add_object_patch("model_sampling", joined)
    shift = schedules.shift_factor(current)
    native = schedules.shift_factor(own)
    if start >= 1.0:
        return
    if start <= 0.0:
        logger.warning(
            "%s: the tiles run from the first step at the model's own shift %.2f, so the "
            "shift of %.2f set on the model is not used. Raise start_percent on Tiled "
            "Diffusion Model Patch to give it whole-frame steps.", node, native, shift,
        )
        return
    logger.info(
        "%s: steps above sigma %.3f follow shift %.2f; the tiles below follow the model's "
        "own %.2f.", node, float(own.percent_to_sigma(start)), shift, native,
    )


def _bounds(own, start: float, end: float) -> tuple[float, float]:
    """The sigmas a tiled step starts at or below, and above, read off the model's own curve.

    Args:
        own: The model's own ``model_sampling`` object.
        start: Share of the schedule where tiling begins.
        end: Share of the schedule where tiling ends.

    Returns:
        ``(high, low)``: infinite at a share of 0 and 1 respectively.
    """
    high = math.inf if start <= 0.0 else float(own.percent_to_sigma(start))
    low = -math.inf if end >= 1.0 else float(own.percent_to_sigma(end))
    return high, low


def _tiled(executor, grid, weights: list, size: tuple, whole: tuple, kwargs: dict) -> torch.Tensor:
    """One model call per tile, the predictions blended into one the size of the latent.

    Args:
        executor: The next step of the model call.
        grid: The tiling.
        weights: :func:`tiles.tile_weights` of the grid.
        size: ``(height, width)`` of the latent.
        whole: The call's ``(x, t, c_concat, c_crossattn, control, transformer_options)``.
        kwargs: The call's further conditioning.

    Returns:
        The blended prediction, shaped like ``x``.
    """
    import comfy.model_management

    x, t, c_concat, c_crossattn, control, options = whole
    blended = total = None
    result = x.dtype
    for box, weight in zip(grid.boxes(), weights):
        comfy.model_management.throw_exception_if_processing_interrupted()
        h0, h1, w0, w1 = box
        part = executor(
            x[..., h0:h1, w0:w1], t, _cropped(c_concat, box, size), c_crossattn,
            _cropped_control(control, box, size), options,
            **{key: _cropped(value, box, size) for key, value in kwargs.items()},
        )
        if blended is None:
            blended = torch.zeros((*part.shape[:-2], *size), device=part.device, dtype=torch.float32)
            total = torch.zeros(size, device=part.device, dtype=torch.float32)
            result = part.dtype
        weight = weight.to(part.device)
        blended[..., h0:h1, w0:w1] += part.float() * weight
        total[h0:h1, w0:w1] += weight
    return (blended / total).to(result)


def _in_range(t: torch.Tensor, options: dict, high: float, low: float) -> bool:
    """Whether the step being sampled starts at or below ``high`` and above ``low``.

    Args:
        t: The call's sigmas.
        options: The call's transformer options, holding the sampler's schedule.
        high: Sigma at or below which a step is tiled.
        low: Sigma at or below which a step is no longer tiled.

    Returns:
        True to tile this call.
    """
    sigma = float(t.max())
    schedule = (options or {}).get("sample_sigmas")
    if torch.is_tensor(schedule) and schedule.numel() > 1:
        sigmas = schedule.flatten().float().tolist()
        steps = len(sigmas) - 1
        # A step runs from its sigma down to the next; a sampler's sub-steps fall inside it.
        index = next((i for i in range(steps) if sigmas[i + 1] < sigma * (1.0 - 1e-6)), steps - 1)
        sigma = sigmas[index]
    return low * (1.0 + SIGMA_TOLERANCE) < sigma <= high * (1.0 + SIGMA_TOLERANCE)


def _cropped(value, box: tuple, size: tuple):
    """The part of a picture-sized tensor under a tile; any other value as it is."""
    if torch.is_tensor(value) and value.ndim >= 4 and tuple(value.shape[-2:]) == size:
        h0, h1, w0, w1 = box
        return value[..., h0:h1, w0:w1]
    return value


def _croppable(control) -> bool:
    """Whether every tensor of a control's output is picture-shaped."""
    if control is None:
        return True
    if not isinstance(control, dict):
        return False
    for values in control.values():
        for value in values or ():
            if value is not None and (not torch.is_tensor(value) or value.ndim != 4):
                return False
    return True


def _cropped_control(control, box: tuple, size: tuple):
    """A control output with each feature map cut to the tile in proportion to its size."""
    if control is None:
        return None
    return {
        key: [None if value is None else tiles.crop_plane(value, box, size) for value in values]
        if isinstance(values, list) else values
        for key, values in control.items()
    }
