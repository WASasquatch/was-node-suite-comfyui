"""Per-window structural control for MiniMax H3 from one control video spanning the finished clip.

Control, mask and source frames run at 24 fps. A control latent stacks 24 control channels, one
visibility channel and 24 masked source channels.
"""

from __future__ import annotations

import contextlib
import math

from . import h3_extend

__all__ = [
    "CONTROL_CHANNELS",
    "CONTROL_WRAPPER",
    "INPAINT_CHANNELS",
    "checked",
    "control_latent",
    "controlled",
    "describe",
    "fit",
    "placement",
    "taken",
]

#: Channels of an encoded control video, one H3 video latent.
CONTROL_CHANNELS = 24

#: Channels a control latent carries with inpainting: control, visibility and masked source.
INPAINT_CHANNELS = 49

#: Key the control's model wrapper is registered under.
CONTROL_WRAPPER = "was_h3_control"


def placement(window: dict) -> dict:
    """Where a window's frames sit on the control video.

    Args:
        window: The H3 joint latent a pass samples, placed by H3 Extend Window or not.

    Returns:
        ``{"first", "frames", "head", "start", "placed"}``: the control frame at the window's
        frame 0, the window's frame count, the frames ahead of its first kept frame, that kept
        frame on the finished clip, and whether the window carried a place.

    Raises:
        ValueError: The latent is not a MiniMax H3 joint latent.
    """
    video, _ = h3_extend.split(window)
    frames = h3_extend.frames_for(int(video.shape[2]))
    place = window.get(h3_extend.WINDOW_KEY) if isinstance(window, dict) else None
    if not isinstance(place, dict):
        return {"first": 0, "frames": frames, "head": 0, "start": 0, "placed": False}
    start, head = int(place.get("start", 0)), int(place.get("head", 0))
    return {"first": max(0, start - head), "frames": frames, "head": head, "start": start,
            "placed": True}


def taken(video, first: int, count: int):
    """A stretch of frames, holding the first or last frame past either end.

    Args:
        video: A ``[N, ...]`` batch of frames or mask planes.
        first: Index of the first frame wanted, which may lie outside the batch.
        count: Frames wanted.

    Returns:
        A ``[count, ...]`` copy.

    Raises:
        ValueError: The batch holds no frames.
    """
    import torch

    total = int(video.shape[0])
    if total < 1:
        raise ValueError(
            "a video or mask wired into H3 Control holds no frames. Wire one that holds frames"
        )
    index = torch.arange(int(first), int(first) + int(count)).clamp(0, total - 1)
    return video[index.to(video.device)]


def fit(frames, width: int, height: int):
    """Frames cropped about their centre to a canvas's shape and scaled onto it.

    Args:
        frames: A ``[T, H, W, C]`` tensor; one channel is read as grey.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        A ``[T, height, width, 3]`` float32 tensor.
    """
    import comfy.utils

    colour = frames[..., :3].float()
    if colour.shape[-1] == 1:
        colour = colour.expand(*colour.shape[:-1], 3)
    scaled = comfy.utils.common_upscale(colour.movedim(-1, 1), int(width), int(height),
                                        "bilinear", "center")
    return scaled.movedim(1, -1)


def _encoded(vae, pixels, target: tuple, name: str):
    """One stretch of frames encoded by the H3 video VAE, checked against the window.

    Args:
        vae: The H3 video VAE.
        pixels: A ``[T, H, W, 3]`` tensor at the window's canvas.
        target: The window's video latent shape with a batch of 1.
        name: The input the frames came from, for the error.

    Returns:
        A float32 latent shaped ``target``.

    Raises:
        ValueError: The VAE answered another shape.
    """
    latent = vae.encode(pixels).float()
    if tuple(latent.shape) != tuple(target):
        raise ValueError(
            f"the vae encoded {name} to {tuple(latent.shape)} and the window is "
            f"{tuple(target)}. Wire the MiniMax H3 video VAE into H3 Control's vae"
        )
    return latent


def control_latent(vae, shape, first: int, control_video=None, mask=None, source_video=None,
                   post_norm: bool = False):
    """The control latent for one window, from the stretch of each video that lands on it.

    Args:
        vae: The H3 video VAE.
        shape: The window's video latent shape, ``[B, 24, T, h, w]``.
        first: The video frame at the window's frame 0.
        control_video: ``[N, H, W, C]`` control frames, or None.
        mask: ``[N, H, W]`` mask, 1 where the window regenerates, or None.
        source_video: ``[N, H, W, C]`` frames the mask keeps, read with ``mask``.
        post_norm: True where masked pixels take the VAE's zero point rather than black.

    Returns:
        ``(latent, regenerated)``: a ``[1, C, T, h, w]`` float32 latent with 24 channels, or 49
        with a mask, and the share of the window regenerated, None without a mask.

    Raises:
        ValueError: Neither control_video nor mask is given, mask is given without
            source_video, or the VAE answers another shape.
    """
    import torch
    import torch.nn.functional as functional

    if control_video is None and mask is None:
        raise ValueError(
            "H3 Control has nothing to steer with. Wire the control frames into control_video, "
            "or a mask and source_video to inpaint"
        )
    if mask is not None and source_video is None:
        raise ValueError(
            "H3 Control has a mask and no source_video. Inpainting keeps source_video outside "
            "the mask and regenerates inside it: wire the video the mask was drawn on into "
            "source_video, or disconnect mask"
        )
    tokens, high, wide = int(shape[2]), int(shape[3]), int(shape[4])
    frames = h3_extend.frames_for(tokens)
    scale = int(vae.spacial_compression_encode())
    width, height = wide * scale, high * scale
    target = (1, CONTROL_CHANNELS, tokens, high, wide)

    hint = None
    if control_video is not None:
        hint = _encoded(vae, fit(taken(control_video, first, frames), width, height), target,
                        "control_video")
    if mask is None:
        return hint, None

    import comfy.utils
    from comfy.ldm.minimax.vae import IMAGENET_MEAN

    planes = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    holes = (taken(planes, first, frames) > 0.5).float().unsqueeze(1)
    holes = comfy.utils.common_upscale(holes, width, height, "bilinear", "center")
    kept = 1.0 - (holes > 0.5).float()
    del holes
    source = fit(taken(source_video, first, frames), width, height).to(kept.device)
    visible = kept.movedim(1, -1)
    masked = source * visible
    del source
    if post_norm:
        # Masked pixels sit at the colour the VAE normalises to zero.
        grey = torch.tensor(IMAGENET_MEAN, dtype=masked.dtype, device=masked.device)
        masked += (1.0 - visible) * grey
    masked_latent = _encoded(vae, masked, target, "source_video")
    del masked
    visibility = functional.interpolate(kept[:, 0][None, None], size=(tokens, high, wide),
                                        mode="trilinear", align_corners=False)
    regenerated = float(1.0 - kept.mean())
    if hint is None:
        hint = torch.zeros(target, dtype=torch.float32)
    latent = torch.cat([hint, visibility.to(hint), masked_latent.to(hint)], dim=1)
    return latent, regenerated


def _adaln_width(block) -> int | None:
    """Width of the time embedding a DiT block reads, or None where it cannot be found."""
    linear = getattr(getattr(block, "adaln_proj", None), "linear", None)
    width = getattr(linear, "in_features", None)
    return int(width) if width else None


def checked(model, patch):
    """The control network inside a model patch, checked against the H3 model it steers.

    Args:
        model: A MiniMax H3 model patcher.
        patch: A ``MODEL_PATCH`` from Load Model Patch.

    Returns:
        The control network.

    Raises:
        ValueError: The patch is not an H3 Fun ControlNet, the model is not MiniMax H3, or the
            two were built for checkpoints with different time embeddings.
    """
    network = getattr(patch, "model", None)
    if not all(hasattr(network, name) for name in
               ("init_stream", "step", "injection_layers", "control_in_dim", "control_blocks")):
        raise ValueError(
            f"H3 Control needs the MiniMax H3 Fun ControlNet-Union patch, and the model_patch "
            f"wired in holds a {type(network).__name__}. Set Load Model Patch to a "
            f"minimax_h3_fun_controlnet_union file in models/model_patches"
        )
    base = model.get_model_object("diffusion_model")
    blocks = getattr(base, "blocks", None)
    if "minimax" not in type(base).__module__ or blocks is None:
        raise ValueError(
            f"H3 Control steers a MiniMax H3 model, and the model wired in is a "
            f"{type(base).__name__}. Wire the model output of H3 Extend Window"
        )
    layers = tuple(int(layer) for layer in network.injection_layers)
    if not layers or layers[0] != 0 or layers[-1] >= len(blocks):
        raise ValueError(
            f"this control patch injects at blocks {list(layers)} and the model has "
            f"{len(blocks)}. Load the control patch made for this MiniMax H3 checkpoint"
        )
    wanted, given = _adaln_width(network.control_blocks[0]), _adaln_width(blocks[0])
    if wanted and given and wanted != given:
        raise ValueError(
            f"this control patch reads a time embedding {wanted} wide and the model's blocks "
            f"read one {given} wide, so the two were made for different MiniMax H3 checkpoints. "
            f"Load the control patch made for this checkpoint, or the checkpoint it was made for"
        )
    return network


def _outside_graph():
    """A context whose allocations stay out of ComfyUI's per-block allocation record."""
    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return contextlib.nullcontext()
    pause = getattr(prefetch, "pause_malloc_graph", None)
    return pause() if pause is not None else contextlib.nullcontext()


def _sigma_of(timestep, transformer_options) -> float:
    """The noise level a model call runs at."""
    sigmas = (transformer_options or {}).get("sigmas")
    if sigmas is not None:
        return float(sigmas.flatten()[0]) if hasattr(sigmas, "flatten") else float(sigmas[0])
    return float(timestep.flatten()[0]) / 1000.0


class _Injection:
    """One window's control latent and the running control stream it feeds."""

    def __init__(self, patch, latent, strength: float, high: float, low: float):
        self.patch = patch
        self.network = patch.model
        self.layers = tuple(int(layer) for layer in self.network.injection_layers)
        self.latent = latent
        self.strength = float(strength)
        self.high, self.low = float(high), float(low)
        self.active = False
        self.stash = None
        self.stream = None
        self.resident = None

    def forward(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        """The H3 model call, with control on where the call's sigma is inside the range.

        Args:
            executor: The next wrapper, or the model's own forward.
            x: The ``[video, audio]`` latents the model is called on.
            timestep: The call's timestep.
            context: The text conditioning.
            transformer_options: The call's transformer options.
            **kwargs: Passed on unchanged.

        Returns:
            The model's output.

        Raises:
            ValueError: The model is called on a latent of another shape than the window.
        """
        video = x[0] if isinstance(x, (list, tuple)) else x
        if tuple(video.shape[2:]) != tuple(self.latent.shape[2:]):
            raise ValueError(
                f"H3 Control encoded its control for a window of {self.latent.shape[2]} tokens "
                f"on a {self.latent.shape[4]}x{self.latent.shape[3]} latent grid and the sampler "
                f"called the model on {video.shape[2]} tokens on {video.shape[4]}x"
                f"{video.shape[3]}. Wire the window the sampler samples into H3 Control's "
                f"window, and sample it untiled"
            )
        sigma = _sigma_of(timestep, transformer_options)
        self.active = self.low <= sigma <= self.high
        self.stash = self.stream = None
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            self.active = False
            self.stash = self.stream = None

    def inject(self, order: int, args: dict, out: dict) -> dict:
        """Run one control block and add its output to the base block's.

        Args:
            order: The control block's place among the injection layers, from 0.
            args: The base block's arguments.
            out: The base block's output, added to in place.

        Returns:
            ``out``.
        """
        if order == 0:
            device = out["img"].device
            if self.resident is None or self.resident.device != device:
                self.resident = self.latent.to(device)
            self.stream = self.network.init_stream(self.stash, self.resident, args["layout"],
                                                   args["t_emb"])
            self.stash = None
        if self.stream is None:
            return out
        self.stream, skip = self.network.step(
            order, self.stream, args["t_emb"], args["mod_segments"], args["rope_freqs"],
            transformer_options=args["transformer_options"],
        )
        skip[args["layout"].audio_pos.to(skip.device)] = 0
        out["img"].add_(skip, alpha=self.strength)
        return out

    def release(self):
        """Drop the stream and the copy of the latent on the compute device."""
        self.active = False
        self.stash = self.stream = self.resident = None


class _Block:
    """A block replacement running one H3 block, then the control block paired with it."""

    def __init__(self, injection: _Injection, order: int, previous=None):
        self.injection = injection
        self.order = int(order)
        self.previous = previous

    def __call__(self, args, extra_args):
        injection = self.injection
        if injection.active and self.order == 0:
            with _outside_graph():
                injection.stash = args["img"].clone()
        if self.previous is None:
            out = extra_args["original_block"](args)
        else:
            out = self.previous(args, extra_args)
        if not injection.active:
            return out
        with _outside_graph():
            return injection.inject(self.order, args, out)

    def to(self, device_or_dtype):
        """Pass a device or dtype move on to the replacement this one wraps."""
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def models(self) -> list:
        """The control patch, and whatever the wrapped replacement loads."""
        models = [self.injection.patch]
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models

    def cleanup(self):
        """Release the control state after sampling."""
        self.injection.release()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()


def controlled(model, patch, latent, strength: float, start_percent: float = 0.0,
               end_percent: float = 1.0):
    """A clone of an H3 model whose blocks add a control network's output.

    Args:
        model: A MiniMax H3 model patcher.
        patch: The H3 Fun ControlNet ``MODEL_PATCH``.
        latent: The window's control latent from :func:`control_latent`.
        strength: Multiplier on every control block's output.
        start_percent: Point in the noise schedule control starts, 0.0 for the first step.
        end_percent: Point in the noise schedule control stops, 1.0 for the last step.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension

    sampling = model.get_model_object("model_sampling")
    high = math.inf if start_percent <= 0.0 else float(sampling.percent_to_sigma(start_percent))
    low = -math.inf if end_percent >= 1.0 else float(sampling.percent_to_sigma(end_percent))
    patched = model.clone()
    injection = _Injection(patch, latent, strength, high, low)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 CONTROL_WRAPPER, injection.forward)
    for order, layer in enumerate(injection.layers):
        options = patched.model_options.get("transformer_options", {})
        previous = options.get("patches_replace", {}).get("dit", {}).get(("double_block", layer))
        patched.set_model_patch_replace(_Block(injection, order, previous), "dit",
                                        "double_block", layer)
    return patched


def describe(place: dict, available: int, read: str, strength: float, start_percent: float,
             end_percent: float, regenerated) -> str:
    """One line naming the frames a window reads and how control is applied.

    Args:
        place: The window's :func:`placement`.
        available: Frames the read video holds.
        read: What the frames are read from, as ``control video``.
        strength: The control strength.
        start_percent: Point in the schedule control starts.
        end_percent: Point in the schedule control stops.
        regenerated: Share of the window regenerated, or None without a mask.

    Returns:
        The report line.
    """
    first, frames = int(place["first"]), int(place["frames"])
    last = first + frames - 1
    line = (f"frames {first} to {last} of the {read} (head {int(place['head'])}) "
            f"at {float(strength):.2f}")
    if start_percent > 0.0 or end_percent < 1.0:
        line += f" from {start_percent:.0%} to {end_percent:.0%} of the schedule"
    if last >= available:
        held = last - max(first, available) + 1
        line += f", the last {held} holding its frame {available - 1}"
    if regenerated is None:
        line += "; inpaint off"
    else:
        line += f"; inpaint on, {regenerated:.0%} regenerated"
    if not place.get("placed"):
        line += "; the window carries no place on the clip, so control reads from frame 0"
    return line
