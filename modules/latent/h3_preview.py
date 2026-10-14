"""An H3 segment's prediction drawn through a preview VAE at each sampling step."""

from __future__ import annotations

import time

from .. import log
from ..interface import segment_preview

__all__ = ["DECODER_NAME", "FAILED", "INVALID", "MIN_INTERVAL", "MISSING", "SOURCE",
           "VIDEO_CHANNELS", "Unavailable", "decoder", "previewed", "step_of", "unwatched",
           "watched"]

logger = log.get_logger("latent.h3_preview")

#: Seconds between two previews of one segment; the last step is always drawn.
MIN_INTERVAL = 0.75

#: Channels of the video half of an H3 latent.
VIDEO_CHANNELS = 24

#: What the preview decoder's file name starts with in ``vae_approx``, as core's previewer reads it.
DECODER_NAME = "taeh3"

#: Where the preview decoder is downloaded from.
SOURCE = "safetensors/taeh3.safetensors from https://github.com/madebyollin/taehv"

#: What the Prompt Timeline shows on a segment that cannot be previewed.
MISSING = "No MiniMax H3 Preview VAE"
INVALID = "Invalid MiniMax H3 Preview VAE"
FAILED = "MiniMax H3 Preview Failed"

_decoders: dict = {}
_reported: set = set()


class Unavailable(Exception):
    """No preview decoder can be used.

    Attributes:
        title: What the Prompt Timeline shows, as :data:`MISSING` or :data:`INVALID`.
        detail: The cause and the fix.
    """

    def __init__(self, title: str, detail: str):
        super().__init__(f"{title}. {detail}")
        self.title = title
        self.detail = detail


def decoder(name: str = DECODER_NAME):
    """The preview decoder in ComfyUI's ``vae_approx`` folders, loaded once and kept.

    Args:
        name: What the file's name starts with, in any case and in any subfolder.

    Returns:
        The loaded VAE.

    Raises:
        Unavailable: No such file is installed, or ComfyUI cannot load the one that is.
    """
    import os

    import folder_paths

    entries = [entry.replace("\\", "/") for entry in folder_paths.get_filename_list("vae_approx")]
    matches = [entry for entry in entries
               if entry.rsplit("/", 1)[-1].lower().startswith(name.lower())]
    # A file at the top of the folder is taken before one in a subfolder.
    found = min(matches, key=lambda entry: ("/" in entry, entry), default=None)
    path = folder_paths.get_full_path("vae_approx", found) if found else None
    if not path:
        folders = tuple(folder_paths.get_folder_paths("vae_approx"))
        problem = Unavailable(MISSING, (
            f"No file named {name}* is in the vae_approx folders ComfyUI reads "
            f"({', '.join(folders) or 'none configured'}). Put {SOURCE} in one of them; it is "
            "found on the next run without a restart."))
        if folders not in _reported:
            _reported.add(folders)
            logger.warning("live_preview is on but the Prompt Timeline draws no segments: %s",
                           problem)
        raise problem
    key = (path, os.path.getmtime(path))
    if key not in _decoders:
        import comfy.sd
        import comfy.utils

        held = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path, safe_load=True))
        if getattr(held, "first_stage_model", None) is None:
            held = Unavailable(INVALID, (
                f"{path} is not a decoder ComfyUI can load, likely the older per-frame taeh3. "
                f"Replace it with {SOURCE}; it is read on the next run without a restart."))
            logger.warning("live_preview is on but the Prompt Timeline draws no segments: %s",
                           held)
        elif hasattr(held.first_stage_model, "show_progress_bar"):
            held.first_stage_model.show_progress_bar = False
        _decoders.clear()
        _decoders[key] = held
    held = _decoders[key]
    if isinstance(held, Unavailable):
        raise Unavailable(held.title, held.detail)
    return held


def previewed(model, owner, segment: int, head: int):
    """A copy of a model that previews a segment, or a problem held in the segment's place.

    Args:
        model: The segment's model.
        owner: The id of the node the segment belongs to.
        segment: Segment number, from 0.
        head: Frames at the start of the window that come from the segment before.

    Returns:
        The patched copy from :func:`watched`, or ``model`` where no preview decoder can be
        used, the problem then published for the Prompt Timeline in place of a sheet.
    """
    try:
        vae = decoder()
    except Unavailable as problem:
        segment_preview.report(owner, segment, problem.title, problem.detail)
        return model
    return watched(model, vae, owner, segment, head)


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
            reason = " ".join(str(error).split())[:300]
            segment_preview.report(owner, segment, FAILED, (
                f"{type(error).__name__}: {reason}. Sampling went on without previews."))
        state["last"] = time.monotonic()
        return denoised

    preview.was_segment_preview = True
    patched.set_model_sampler_post_cfg_function(preview)
    return patched


def unwatched(model):
    """A model with the segment previews :func:`watched` added taken off again.

    Args:
        model: A model patcher, watched or not.

    Returns:
        A copy without the preview hook, or ``model`` itself where it carries none.
    """
    hooks = (getattr(model, "model_options", None) or {}).get("sampler_post_cfg_function") or []
    if not any(getattr(hook, "was_segment_preview", False) for hook in hooks):
        return model
    plain = model.clone()
    plain.model_options["sampler_post_cfg_function"] = [
        hook for hook in hooks if not getattr(hook, "was_segment_preview", False)]
    return plain
