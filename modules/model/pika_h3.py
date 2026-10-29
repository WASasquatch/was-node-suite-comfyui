"""Pika H3 Flex on MiniMax H3: the released knots and shifts, canvas sizes and the fp32 AdaLN wrapper.

Sigmas are video sigmas on MiniMax H3's flow schedule; core derives each audio sigma with the
audio shift.
"""

from __future__ import annotations

import re

import torch

__all__ = [
    "ASPECTS", "AUDIO_SHIFT", "CANVAS", "MAX_SECONDS", "MIN_SECONDS", "RELEASE_KNOTS", "VIDEO_SHIFT",
    "apply", "audio_sigma", "frames_for_seconds", "is_minimax_h3", "knots_text", "parse_knots",
    "sigmas_from_knots",
]

#: The released schedule's knots on the 0 to 1000 grid, ending in 0.
RELEASE_KNOTS = (1000, 750, 500, 250, 100, 40, 0)

#: The release's video and audio shifts.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

#: Frame rate, and the ``17k + 5`` clip grid every length lands on.
FPS = 24
CLIP_FRAMES = 17
CLIP_LEAD = 5

#: The shortest and longest clips the release covers, in seconds.
MIN_SECONDS = 5
MAX_SECONDS = 15

#: Width to height, widest first.
ASPECTS = ("21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21")

#: ``(width, height)`` per resolution and aspect.
CANVAS = {
    "480p": dict(zip(ASPECTS, (
        (960, 416), (832, 480), (704, 480), (640, 480), (608, 480), (480, 480),
        (480, 608), (480, 640), (480, 704), (480, 832), (416, 960),
    ))),
    "768p": dict(zip(ASPECTS, (
        (1536, 672), (1344, 768), (1152, 768), (1024, 768), (960, 768), (768, 768),
        (768, 960), (768, 1024), (768, 1152), (768, 1344), (672, 1536),
    ))),
}

#: Key of the AdaLN wrapper on a patched model.
WRAPPER_KEY = "was_pika_h3_flex_adaln"

_SAMPLING_CLASS = None


def knots_text(knots=RELEASE_KNOTS) -> str:
    """Knots as the widget writes them, as ``1000, 750, 500, 250, 100, 40, 0``."""
    return ", ".join(f"{float(k):g}" for k in knots)


def shift_sigma(base: torch.Tensor, shift: float) -> torch.Tensor:
    """The flow shift ``s*b / (1 + (s-1)*b)`` of grid values in ``[0, 1]``."""
    return shift * base / (1.0 + (shift - 1.0) * base)


def audio_sigma(video: float, video_shift: float = VIDEO_SHIFT, audio_shift: float = AUDIO_SHIFT) -> float:
    """The audio sigma core samples beside one video sigma.

    Args:
        video: A video sigma in ``[0, 1]``.
        video_shift: The video shift it was made with.
        audio_shift: The audio shift.

    Returns:
        The audio sigma at the same point of the base grid.
    """
    video = float(video)
    base = video / (video_shift - (video_shift - 1.0) * video) if video < 1.0 else 1.0
    return audio_shift * base / (1.0 + (audio_shift - 1.0) * base)


def parse_knots(text: str) -> list[float]:
    """Knots read from text, as ``1000, 750, 500, 250, 100, 40, 0``.

    Args:
        text: Numbers separated by commas or spaces.

    Returns:
        The knots, each in ``(0, 1000]`` and the last one 0.

    Raises:
        ValueError: The text holds something else.
    """
    parts = [part for part in re.split(r"[,\s]+", str(text or "").strip()) if part]
    try:
        knots = [float(part) for part in parts]
    except ValueError:
        raise ValueError(
            f"knots are numbers separated by commas, as {knots_text()}, and the text was "
            f"{str(text)[:120]!r}") from None
    if len(knots) < 2 or knots[-1] != 0.0:
        raise ValueError(
            f"knots need at least one value before a final 0, as {knots_text()} or 100, 40, 0, "
            f"and the text was {str(text)[:120]!r}")
    if any(not 0.0 < knot <= 1000.0 for knot in knots[:-1]):
        raise ValueError(
            f"every knot before the final 0 lies above 0 and at most 1000, and the text was "
            f"{str(text)[:120]!r}")
    return knots


def sigmas_from_knots(knots, video_shift: float = VIDEO_SHIFT) -> torch.Tensor:
    """Video sigmas for knots on the 0 to 1000 grid.

    Args:
        knots: From :func:`parse_knots`.
        video_shift: The video shift.

    Returns:
        A float32 tensor ending in 0.
    """
    base = torch.tensor([float(k) for k in knots], dtype=torch.float64) / 1000.0
    return shift_sigma(base, video_shift).to(torch.float32)


def frames_for_seconds(seconds: int) -> int:
    """The shortest ``17k + 5`` frame count covering a length at 24 fps.

    Args:
        seconds: The clip's length.

    Returns:
        The frame count, as 124 for 5 seconds.
    """
    frames = int(seconds) * FPS
    return frames + (CLIP_LEAD - frames) % CLIP_FRAMES


def is_minimax_h3(model) -> bool:
    """Whether a ``ModelPatcher`` holds a MiniMax H3 transformer."""
    import comfy.model_base

    return isinstance(getattr(model, "model", None), comfy.model_base.MiniMaxH3)


def _sampling_class():
    """The packed audio and video flow sampling the release runs at, built once from core's classes."""
    global _SAMPLING_CLASS
    if _SAMPLING_CLASS is None:
        import comfy.model_sampling

        class PikaFlexSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            """MiniMax H3 audio and video flow sampling at fixed video and audio shifts."""

        _SAMPLING_CLASS = PikaFlexSampling
    return _SAMPLING_CLASS


def _with_release_shifts(model):
    """A clone of a model sampling at the release's video and audio shifts."""
    patched = model.clone()
    original = patched.get_model_object("model_sampling")
    sampling = _sampling_class()(patched.model.model_config)
    sampling.set_parameters(shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT)
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    patched.add_object_patch("model_sampling", sampling)
    options = dict(patched.model_options.get("transformer_options", {}))
    options["minimax_h3_sigma_shift_video"] = VIDEO_SHIFT
    options["minimax_h3_sigma_shift_audio"] = AUDIO_SHIFT
    patched.model_options["transformer_options"] = options
    return patched


def _fp32_projection(module, held: dict):
    """A forward for one AdaLN projection reading the fp32 time embedding of the current call."""

    def forward(t_emb):
        emb = held.get("emb")
        if emb is None or emb.shape != t_emb.shape:
            emb = t_emb
        x = module.linear(torch.nn.functional.silu(emb.float()).to(t_emb.dtype))
        x = x.view(x.shape[0] * module.modalities, module.expand * module.hidden)
        return x.chunk(module.expand, dim=-1)

    return forward


def fp32_adaln(executor, *args, **kwargs):
    """Diffusion model wrapper: one forward with each AdaLN SiLU taken on the fp32 time embedding."""
    model = executor.class_obj
    embedder = getattr(model, "time_embedder", None)
    if embedder is None or getattr(model, "use_adaln_curves", False):
        return executor(*args, **kwargs)
    projections = [block.adaln_proj for block in model.blocks] + [model.final_layer.adaln_proj]
    projections = [p for p in projections if getattr(p, "apply_silu", False)]
    held = {}
    embed = type(embedder).forward

    def keep(t):
        held["emb"] = embed(embedder, t)
        return held["emb"]

    embedder.forward = keep
    for projection in projections:
        projection.forward = _fp32_projection(projection, held)
    try:
        return executor(*args, **kwargs)
    finally:
        held.clear()
        embedder.__dict__.pop("forward", None)
        for projection in projections:
            projection.__dict__.pop("forward", None)


def apply(model, lora: dict, strength: float, mode: str, adaln: bool):
    """A MiniMax H3 model carrying a Pika H3 Flex adapter at the release's shifts.

    Args:
        model: A ``ModelPatcher`` of MiniMax H3, fl2va or ref2va.
        lora: The adapter's state dict, or None for none.
        strength: The adapter's scale.
        mode: ``bypass`` to compute it beside each layer, ``merge`` to add it to the weights.
        adaln: Whether each AdaLN SiLU runs on the fp32 time embedding.

    Returns:
        ``(model, curves)``: the patched clone, and whether the model keeps its AdaLN as a table.
    """
    import comfy.patcher_extension
    import comfy.sd

    patched = model
    if lora is not None and strength != 0:
        load = comfy.sd.load_bypass_lora_for_models if mode == "bypass" else comfy.sd.load_lora_for_models
        patched, _ = load(model, None, lora, strength, 0.0)
    patched = _with_release_shifts(patched)
    diffusion = patched.get_model_object("diffusion_model")
    curves = bool(getattr(diffusion, "use_adaln_curves", False))
    if adaln:
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, fp32_adaln)
    return patched, curves
