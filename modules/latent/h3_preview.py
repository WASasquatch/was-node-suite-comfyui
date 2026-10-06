"""An H3 segment's prediction drawn through a preview VAE at each sampling step."""

from __future__ import annotations

import time

from .. import log
from ..interface import segment_preview

__all__ = ["DECODER_NAME", "MIN_INTERVAL", "VIDEO_CHANNELS", "decoder", "step_of", "watched"]

logger = log.get_logger("latent.h3_preview")

#: Seconds between two previews of one segment; the last step is always drawn.
MIN_INTERVAL = 0.75

#: Channels of the video half of an H3 latent.
VIDEO_CHANNELS = 24

#: What the preview decoder's file name starts with in ``vae_approx``, as core's previewer reads it.
DECODER_NAME = "taeh3"

_decoders: dict = {}


def decoder(name: str = DECODER_NAME):
    """The preview decoder in ComfyUI's ``vae_approx`` folder, loaded once and kept.

    Args:
        name: What the file's name starts with.

    Returns:
        The loaded VAE, or ``None`` where no such file is installed.
    """
    import os

    import folder_paths

    found = next((entry for entry in folder_paths.get_filename_list("vae_approx")
                  if entry.startswith(name)), None)
    path = folder_paths.get_full_path("vae_approx", found) if found else None
    if not path:
        return None
    key = (path, os.path.getmtime(path))
    held = _decoders.get(key)
    if held is None:
        import comfy.sd
        import comfy.utils

        held = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path, safe_load=True))
        if hasattr(held.first_stage_model, "show_progress_bar"):
            held.first_stage_model.show_progress_bar = False
        _decoders.clear()
        _decoders[key] = held
    return held


def step_of(sigma, sigmas) -> tuple[int, int, bool]:
    """Which step of a schedule a model call is on.

    Args:
        sigma: The noise level the model is called at.
        sigmas: The schedule, ending on its last level, or None.

    Returns:
        ``(step, steps, last)``, the step counted from 1. ``(0, 0, False)`` without a schedule.
    """
    if sigmas is None or len(sigmas) < 2:
        return 0, 0, False
    level = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
    levels = [float(value) for value in sigmas]
    steps = len(levels) - 1
    nearest = min(range(steps), key=lambda index: abs(levels[index] - level))
    return nearest + 1, steps, nearest == steps - 1


def watched(model, vae, owner, segment: int, head: int):
    """A copy of a model that publishes what each step predicts the segment to be.

    Args:
        model: The segment's model.
        vae: A video VAE that decodes H3 latents, as the tiny preview one.
        owner: The id of the node the segment belongs to.
        segment: Segment number, from 0.
        head: Frames at the start of the window that come from the segment before.

    Returns:
        The patched copy, or ``model`` itself where it takes no sampler hook.
    """
    if vae is None or owner is None or not hasattr(model, "set_model_sampler_post_cfg_function"):
        return model
    patched = model.clone()
    state = {"last": 0.0, "failed": False}

    def preview(args):
        denoised = args["denoised"]
        if state["failed"]:
            return denoised
        options = args.get("model_options") or {}
        sigmas = (options.get("transformer_options") or {}).get("sample_sigmas")
        step, steps, last = step_of(args["sigma"], sigmas)
        if not last and time.monotonic() - state["last"] < MIN_INTERVAL:
            return denoised
        try:
            video = denoised.tensors[0] if getattr(denoised, "is_nested", False) else denoised
            # Video and audio sample packed as one flat tensor; the model holds their shapes.
            shapes = getattr(args.get("model"), "latent_shapes", None)
            if video.ndim == 3 and shapes and len(shapes) > 1:
                import comfy.utils

                video = comfy.utils.unpack_latents(video, shapes)[0]
            frames = vae.decode(video[:1, :VIDEO_CHANNELS].detach())
            if frames.ndim == 5:
                frames = frames[0]
            segment_preview.publish(owner, segment, frames, step, steps, head, last)
        except Exception as error:
            state["failed"] = True
            logger.warning("segment %d could not be previewed (%s: %s); sampling goes on without "
                           "previews", int(segment) + 1, type(error).__name__, error)
        state["last"] = time.monotonic()
        return denoised

    patched.set_model_sampler_post_cfg_function(preview)
    return patched
