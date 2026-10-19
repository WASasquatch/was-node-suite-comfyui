"""A model whose RoPE is scaled call by call, so each side stays within a native length.

A side's scale is the native side over the call's side in pixels, at most 1, multiplied into any
``rope_options`` already set.
"""

from __future__ import annotations

import inspect

from .. import log

__all__ = ["DEFAULT_SIDE", "READERS", "WRAPPER", "patch", "scales"]

logger = log.get_logger("sampling.rope_extension")

#: Key the wrapper is registered under.
WRAPPER = "was_rope_extension"

#: Native side in pixels the node starts on: the long side of MOSS-Image's widest 2 MP size, 21:9.
DEFAULT_SIDE = 2160

#: Transformers that read RoPE scales from ``rope_options``, as ``(module, class)``, with what
#: a scale below 1 changes in them.
READERS = {
    ("comfy.ldm.moss_image.model", "MOSSImage"): "frequencies",
    ("comfy.ldm.flux.model", "Flux"): "positions",
}

#: Image pixels per latent pixel where the model names none.
_DEFAULT_RATIO = 8


def _reader(transformer) -> str | None:
    """What a transformer's RoPE scale acts on, or None when it reads no scale."""
    for kind in type(transformer).__mro__:
        found = READERS.get((kind.__module__, kind.__name__))
        if found:
            return found
    return None


def scales(height: int, width: int, native: int) -> tuple[float, float]:
    """RoPE scales for a picture of one size.

    Args:
        height: Picture height in pixels.
        width: Picture width in pixels.
        native: Longest side in pixels sampled unscaled.

    Returns:
        ``(scale_y, scale_x)``, each at most 1.
    """
    return min(1.0, native / max(1, height)), min(1.0, native / max(1, width))


def patch(model, native: int):
    """A clone of a model whose RoPE is scaled to ``native`` on every call longer than it.

    Args:
        model: A model patcher whose transformer is one of :data:`READERS`.
        native: Longest side in pixels sampled unscaled.

    Returns:
        ``(model, kind)``: the patched clone and what the scale acts on, from :data:`READERS`.

    Raises:
        ValueError: The transformer reads no RoPE scale.
    """
    import comfy.patcher_extension

    transformer = model.get_model_object("diffusion_model")
    kind = _reader(transformer)
    if kind is None:
        raise ValueError(
            f"RoPE Extension Patch scales the RoPE of MOSS-Image and FLUX 1 and 2 models, and this "
            f"model's transformer is {type(transformer).__name__}, which reads no RoPE scale. Leave "
            f"the patch out for it."
        )
    parameters = list(inspect.signature(type(transformer)._forward).parameters)[1:]
    index = parameters.index("transformer_options") if "transformer_options" in parameters else None
    latent_format = model.get_model_object("latent_format")
    ratio = int(getattr(latent_format, "spacial_downscale_ratio", 0) or _DEFAULT_RATIO)
    logged = set()

    def wrapper(executor, *args, **kwargs):
        x = args[0]
        height, width = int(x.shape[-2]) * ratio, int(x.shape[-1]) * ratio
        scale_y, scale_x = scales(height, width, native)
        if scale_y >= 1.0 and scale_x >= 1.0:
            return executor(*args, **kwargs)
        positional = "transformer_options" not in kwargs and index is not None and len(args) > index
        options = args[index] if positional else kwargs.get("transformer_options", {})
        rope = dict((options or {}).get("rope_options") or {})
        rope["scale_y"] = float(rope.get("scale_y", 1.0)) * scale_y
        rope["scale_x"] = float(rope.get("scale_x", 1.0)) * scale_x
        options = {**(options or {}), "rope_options": rope}
        if positional:
            args = (*args[:index], options, *args[index + 1:])
        else:
            kwargs = {**kwargs, "transformer_options": options}
        if (height, width) not in logged:
            logged.add((height, width))
            logger.info("RoPE Extension Patch: a %d x %d call samples with RoPE scale %.3f across "
                        "and %.3f down (%s)", width, height, scale_x, scale_y, kind)
        return executor(*args, **kwargs)

    wrappers = comfy.patcher_extension.WrappersMP
    patched = model.clone()
    patched.remove_wrappers_with_key(wrappers.DIFFUSION_MODEL, WRAPPER)
    patched.add_wrapper_with_key(wrappers.DIFFUSION_MODEL, WRAPPER, wrapper)
    return patched, kind
