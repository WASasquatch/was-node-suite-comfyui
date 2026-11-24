"""Structure video conditioning for MiniMax H3 with the SolarRenderer LoRAs: canvas, frames, latents, prompt.

Pictures are ``[T, H, W, C]`` floats in ``[0, 1]``. Latents are core's MiniMax H3 layout, video
``[1, 24, T, H/16, W/16]``.
"""

from __future__ import annotations

import math

from .. import log

__all__ = [
    "AS_TRAINED", "CUSTOM", "DESCRIPTION_WORDS", "ENCODINGS", "FPS", "KINDS", "LANDSCAPE",
    "LEAST_FRAMES", "MEAN", "MOST_FRAMES", "PORTRAIT", "STILL_FRAMES", "TRAINED_FRAMES", "WRITTEN", "build", "canvas", "fit", "frames_for", "negative",
    "on_grid", "prompt", "words_in",
]

logger = log.get_logger("latent.h3_structure")

#: Frames a second the structure video and the render run at.
FPS = 24

#: Frames the LoRAs were trained on, 6.6 seconds.
TRAINED_FRAMES = 158

#: The two canvases the LoRAs were trained on, as ``(width, height)``.
LANDSCAPE = (1344, 768)
PORTRAIT = (768, 1344)

#: Frames a first frame is held for when it is encoded and shown to the text encoder.
STILL_FRAMES = 13

#: Frames between two pictures the text encoder is shown, 2 a second.
SHOWN_EVERY = FPS // 2

#: Seed of the posterior draw the training latents were encoded with.
ENCODE_SEED = 42

#: Fewest frames a structure video may have, one ``17k + 5`` clip past the lead.
LEAST_FRAMES = 22

#: Most frames one render takes, 15.08 seconds.
MOST_FRAMES = 362

#: The menu label of each trained structure type, and its key in the trained sentences.
KINDS = {
    "grey box": "scene_graybox",
    "semantic map": "semantic",
    "white box with character": "scene_whitebox",
    "masks and skeletons": "mask_skeleton",
    "body mesh": "smpl_mesh",
    "near depth": "near_depth",
}

#: The structure type described in the node's own words.
CUSTOM = "described below"

#: The structure type whose description is the whole prompt, sent as typed.
WRITTEN = "prompt as written"

#: How the structure and the first frame are encoded, the first being the default.
AS_TRAINED = "as trained"
MEAN = "posterior mean"
ENCODINGS = (AS_TRAINED, MEAN)

#: Words the trained prompts spend on the detailed description.
DESCRIPTION_WORDS = (350, 500)


def canvas(width: int, height: int) -> tuple[int, int]:
    """The trained canvas a picture of one shape renders on.

    Args:
        width: The picture's width in pixels.
        height: Its height.

    Returns:
        :data:`LANDSCAPE` for a picture at least as wide as it is tall, else :data:`PORTRAIT`.
    """
    return LANDSCAPE if width >= height else PORTRAIT


def on_grid(frames, fps: float):
    """Frames put on 24 fps by dropping or holding whole frames.

    Args:
        frames: ``[T, H, W, C]`` at ``fps``.
        fps: Their rate.

    Returns:
        The frames at :data:`FPS`, the same tensor when ``fps`` is already that.
    """
    import torch

    if abs(float(fps) - FPS) < 1e-6:
        return frames
    scale = FPS / float(fps)
    count = int(frames.shape[0])
    slots = [math.floor(i * scale + 0.5) for i in range(count)] + [math.floor(count * scale + 0.5)]
    repeats = torch.tensor([slots[i + 1] - slots[i] for i in range(count)], dtype=torch.long)
    return torch.repeat_interleave(frames, repeats.to(frames.device), dim=0)


def fit(frames, width: int, height: int):
    """Frames centre-cropped to a canvas's shape and LANCZOS-resized onto it, on 8-bit samples.

    Args:
        frames: ``[T, H, W, C]`` in ``[0, 1]``; channels past the third are dropped.
        width: Canvas width.
        height: Canvas height.

    Returns:
        ``[T, height, width, 3]`` in ``[0, 1]``, holding multiples of ``1/255``.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from PIL import Image

    pixels = frames[..., :3]
    rows, cols = int(pixels.shape[1]), int(pixels.shape[2])
    wanted = width / height
    if abs(cols / rows - wanted) >= 1e-3:
        if cols / rows > wanted:
            keep = int(round(rows * wanted))
            start = (cols - keep) // 2
            pixels = pixels[:, :, start:start + keep]
        else:
            keep = int(round(cols / wanted))
            start = (rows - keep) // 2
            pixels = pixels[:, start:start + keep]
    codes = pixels.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    if codes.shape[1:3] != (height, width):

        def one(frame):
            return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))

        with ThreadPoolExecutor(8) as pool:
            codes = np.stack(list(pool.map(one, codes)))
    return torch.from_numpy(np.ascontiguousarray(codes)).float().div(255.0)


def frames_for(length: int, available: int) -> int:
    """Frames one render covers: the asked length cut to the structure video and snapped down onto H3's grid.

    Args:
        length: Frames asked for.
        available: Frames the structure video holds at 24 fps.

    Returns:
        A ``17k + 5`` count.

    Raises:
        ValueError: Fewer than :data:`LEAST_FRAMES` frames are left.
    """
    frames = min(int(length), int(available), MOST_FRAMES)
    if frames < LEAST_FRAMES:
        raise ValueError(
            f"H3 Structure Video needs at least {LEAST_FRAMES} frames of structure video at 24 fps, "
            f"and has {frames}. Feed a longer clip, or set fps to the rate the frames were made at."
        )
    return frames - (frames - 5) % 17


def words_in(text: str) -> int:
    """Words in a text, split on whitespace."""
    return len(str(text or "").split())


def prompt(kind: str, summary: str, description: str, soundscape: str, looks_like: str,
           not_shown: str, first_frame: bool) -> str:
    """The six-part prompt the LoRAs were trained on, from its free parts.

    Args:
        kind: A key of :data:`KINDS`, :data:`CUSTOM` or :data:`WRITTEN`.
        summary: One paragraph: who does what, how the camera moves.
        description: The look, then ``[Shot 1]`` and what is seen in order; the whole prompt under
            :data:`WRITTEN`.
        soundscape: The ambience, or ``N/A``.
        looks_like: What a custom structure video looks like.
        not_shown: What of its look must not appear.
        first_frame: Whether a first frame is given.

    Returns:
        The prompt.

    Raises:
        ValueError: The kind is unknown, a part the kind needs is empty.
    """
    if kind == WRITTEN:
        if not str(description or "").strip():
            raise ValueError("H3 Structure Video's structure is 'prompt as written', and description is empty. "
                             "Paste the whole six-part prompt into description.")
        return str(description).strip()
    from ..vendor.solarrenderer import conditions

    if kind == CUSTOM:
        if not str(looks_like or "").strip() or not str(not_shown or "").strip():
            raise ValueError("H3 Structure Video's structure is 'described below', which needs looks_like, "
                             "as 'a black pencil line sketch of the scene on white paper', and not_shown, as "
                             "'the pencil lines and white paper'.")
        condition = conditions.custom_condition(str(looks_like).strip(), str(not_shown).strip())
    elif kind in KINDS:
        condition = KINDS[kind]
    else:
        raise ValueError(f"H3 Structure Video has no structure type {kind!r}. Choose one of: "
                         f"{', '.join([*KINDS, CUSTOM, WRITTEN])}.")
    if not str(summary or "").strip() or not str(description or "").strip():
        raise ValueError("H3 Structure Video needs a summary, one paragraph of who does what and how the "
                         "camera moves, and a description of the look and the shot.")
    free = {"summary": summary, "detailed_description": description, "overall_soundscape": soundscape}
    return conditions.assemble(free, "full" if first_frame else "structure", condition)


def _shown(frames) -> dict:
    """A video reference as MiniMax H3's text encoder reads it: two pictures a second, timestamped."""
    index = list(range(0, int(frames.shape[0]), SHOWN_EVERY))
    return {"type": "video", "data": frames[index], "timestamps": [i / 2.0 for i in range(len(index))]}


def _latents(vae, frames, sampled: bool):
    """Frames as normalized MiniMax H3 latents, the posterior drawn at the training seed or its mean.

    Args:
        vae: A ComfyUI ``VAE``.
        frames: ``[T, H, W, 3]`` in ``[0, 1]``.
        sampled: True to draw from the posterior at :data:`ENCODE_SEED` and round to fp16.

    Returns:
        ``(latents, drawn)``: ``[1, 24, F, h, w]``, and whether the posterior was drawn from.
    """
    import comfy.model_management
    import torch

    model = vae.first_stage_model
    if not sampled or not hasattr(model, "encode_temporal") or not hasattr(model, "latents_mean"):
        return vae.encode(frames), False
    pixels = frames.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
    memory = vae.memory_used_encode(pixels.shape, vae.vae_dtype)
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory,
                                           force_full_load=vae.disable_offload)
    with torch.no_grad():
        moments = model.encode_temporal(vae.process_input(pixels).to(vae.vae_dtype), vae.device).float()
        mean, logvar = moments.chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(ENCODE_SEED)).to(mean.device)
        drawn = torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)
        drawn = drawn.to(torch.float16).float()
        shift = model.latents_mean.view(1, -1, 1, 1, 1).to(drawn)
        scale = model.latents_std.view(1, -1, 1, 1, 1).to(drawn)
        latents = (drawn - shift) / scale
    return latents.to(vae.output_device), True


def negative(positive) -> list:
    """The positive conditioning with its prompt zeroed, its structure and first frame kept.

    Args:
        positive: Conditioning from :func:`build`.

    Returns:
        A conditioning list for a sampler's negative input.
    """
    import torch

    out = []
    for tensor, extras in positive:
        extras = dict(extras)
        if extras.get("pooled_output") is not None:
            extras["pooled_output"] = torch.zeros_like(extras["pooled_output"])
        out.append([torch.zeros_like(tensor), extras])
    return out


def build(clip, vae, structure, text: str, width: int, height: int, length: int, fps: float,
          first_frame=None, structure_vae=None, encoding: str = AS_TRAINED) -> tuple:
    """Conditioning and an empty latent rendering a structure video, as SolarRenderer encodes a clip.

    Args:
        clip: MiniMax H3's text encoder.
        vae: MiniMax H3's video VAE.
        structure: ``[T, H, W, C]`` structure video at ``fps``.
        text: The six-part prompt.
        width: Canvas width, 0 to take the trained canvas of the structure video's shape.
        height: Canvas height, 0 likewise.
        length: Frames to render at 24 fps.
        fps: The structure video's rate.
        first_frame: ``[B, H, W, C]``, its first picture taken, or None.
        structure_vae: A VAE that encodes the structure video in place of ``vae``, or None.
        encoding: One of :data:`ENCODINGS`.

    Returns:
        ``(positive, latent, facts)``: the conditioning, the empty joint latent, and what was used.

    Raises:
        ValueError: The structure video is too short, a side is not a multiple of 32, or the
            encoding is unknown.
    """
    import comfy.model_management

    from . import h3_conditioning

    if encoding not in ENCODINGS:
        raise ValueError(f"H3 Structure Video has no encoding {encoding!r}. Choose one of: {', '.join(ENCODINGS)}.")
    if not width or not height:
        width, height = canvas(int(structure.shape[2]), int(structure.shape[1]))
    video = on_grid(structure, fps)
    frames = frames_for(length, int(video.shape[0]))
    video = fit(video[:frames], int(width), int(height))
    sampled = encoding == AS_TRAINED
    items = [_shown(video)]
    still = None
    if first_frame is not None:
        still = fit(first_frame[:1], int(width), int(height))
        items.append(_shown(still.repeat(STILL_FRAMES, 1, 1, 1)))
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(text, minimax_ref_items=items))

    structure_latent, drawn = _latents(structure_vae or vae, video, sampled)
    keyframes = [{"resolved_frame_index": 0, "latent": structure_latent}]
    if still is not None:
        # A drawn still takes the held clip's own noise; its mean is the one frame's encode.
        held, _ = _latents(vae, still.repeat(STILL_FRAMES, 1, 1, 1) if sampled else still, sampled)
        keyframes.append({"resolved_frame_index": 0, "latent": held[:, :, :1]})
    positive = [[c[0], {**c[1], "minimax_keyframes": keyframes}] for c in positive]
    latent, frames = h3_conditioning.empty_latent(int(width), int(height), frames)
    comfy.model_management.soft_empty_cache()
    facts = {
        "canvas": f"{int(width)}x{int(height)}",
        "frames": frames,
        "seconds": round(frames / FPS, 2),
        "latents": int(structure_latent.shape[2]),
        "first frame": "given" if still is not None else "none",
        "encoding": "posterior drawn at seed 42, fp16" if drawn else "posterior mean",
    }
    return positive, latent, facts
