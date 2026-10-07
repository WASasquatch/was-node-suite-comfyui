"""Kandinsky 6 prompts, joint latents and image-to-video references.

Pixel sizes are multiples of 16 and clip lengths ``4n + 1`` frames; audio runs at 44.1 kHz, one
latent frame per 1024 samples.
"""

from __future__ import annotations

import math

import torch

from .model import (
    AUDIO_CHANNELS,
    AUDIO_FRAME_SAMPLES,
    AUDIO_SAMPLE_RATE,
    REFERENCE_KEY,
    SPATIAL_FACTOR,
    TEMPORAL_FACTOR,
    VIDEO_CHANNELS,
)

#: The system prompt Kandinsky 6 was trained under, Qwen chat format.
TEMPLATE = (
    "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.\n"
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\n"
    "Describe the location of the video, main characters or objects and their action.\n"
    "Describe the dynamism of the video and presented actions.\n"
    "Name the visual style of the video: whether it is a professional footage, user generated content, "
    "some kind of animation, video game or scren content.\n"
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.\n"
    "Pay attention to the order of key actions shown in the scene.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>"
)

#: Qwen tokens kept per prompt: the 129 the template opens with and 1024 of caption.
MAX_PROMPT_TOKENS = 1153


def prompt_text(video_prompt: str, audio_prompt: str) -> str:
    """One Kandinsky 6 caption: the picture, then the sound between ``<AUDCAP>`` tags."""
    video_prompt = video_prompt.strip()
    audio_prompt = audio_prompt.strip()
    if not audio_prompt:
        return video_prompt
    return f"{video_prompt} <AUDCAP>{audio_prompt}<ENDAUDCAP>"


def encode(clip, video_prompt: str, audio_prompt: str):
    """Encode a Kandinsky 6 prompt with a Kandinsky 5 type text encoder.

    Args:
        clip: A ``CLIP`` holding Qwen 2.5 VL 7B and CLIP-L, as Load CLIP builds them for
            ``kandinsky5``.
        video_prompt: What is seen.
        audio_prompt: What is heard, or ``""``.

    Returns:
        ComfyUI conditioning carrying the prompt and the CLIP-L pooled output.

    Raises:
        ValueError: ``clip`` is missing or not that text encoder pair.
    """
    if clip is None:
        raise ValueError("Kandinsky 6 Text Encode needs a CLIP. Connect Load CLIP or DualCLIPLoader.")
    tokens = clip.tokenize(prompt_text(video_prompt, audio_prompt), llama_template=TEMPLATE)
    if "qwen25_7b" not in tokens or "l" not in tokens:
        raise ValueError(
            "Kandinsky 6 reads its prompt with Qwen 2.5 VL 7B and CLIP-L. Load qwen_2.5_vl_7b and "
            "clip_l in DualCLIPLoader with type 'kandinsky5'."
        )
    tokens["qwen25_7b"] = [row[:MAX_PROMPT_TOKENS] for row in tokens["qwen25_7b"]]
    return clip.encode_from_tokens_scheduled(tokens)


def latent_size(width: int, height: int, length: int) -> tuple[int, int, int]:
    """Latent frames, height and width for a clip, each pixel side floored to a multiple of 16."""
    frames = (max(int(length), 1) - 1) // TEMPORAL_FACTOR + 1
    return frames, (int(height) // 16) * 2, (int(width) // 16) * 2


def audio_frames(video_frames: int, fps: float) -> int:
    """Audio latent frames for the duration of ``video_frames`` pixel frames at ``fps``."""
    return int(math.ceil(video_frames / float(fps) * AUDIO_SAMPLE_RATE / AUDIO_FRAME_SAMPLES))


def empty_latent(width: int, height: int, length: int, fps: float, batch_size: int) -> dict:
    """A zeroed joint latent, video and audio of one duration.

    Raises:
        ValueError: The size is under 16 pixels on a side, or ``fps`` is not positive.
    """
    import comfy.model_management
    import comfy.nested_tensor

    frames, latent_height, latent_width = latent_size(width, height, length)
    if latent_height < 2 or latent_width < 2:
        raise ValueError(f"Kandinsky 6 needs at least 16x16 pixels, not {width}x{height}.")
    if fps <= 0:
        raise ValueError(f"fps must be above 0, not {fps}.")
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(batch_size, VIDEO_CHANNELS, frames, latent_height, latent_width, device=device)
    sound = audio_frames((frames - 1) * TEMPORAL_FACTOR + 1, fps)
    audio = torch.zeros(batch_size, AUDIO_CHANNELS, sound, device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def with_reference(positive, negative, vae, image: torch.Tensor, width: int, height: int):
    """Attach a start image, encoded at the clip's size, to both conditionings.

    Args:
        positive: Conditioning for the prompt.
        negative: Conditioning for the negative prompt.
        vae: The HunyuanVideo VAE.
        image: ``IMAGE`` batch; its first picture is used.
        width: Clip width in pixels.
        height: Clip height in pixels.

    Returns:
        ``(positive, negative)`` with the reference latent set on both.

    Raises:
        ValueError: ``vae`` is missing.
    """
    import comfy.utils
    import node_helpers

    if vae is None or not hasattr(vae, "encode"):
        raise ValueError(
            "Kandinsky 6 Image To Video encodes the start image with the HunyuanVideo VAE. Connect "
            "Load VAE with hunyuan_video_vae_bf16 to its vae input."
        )
    _, latent_height, latent_width = latent_size(width, height, 1)
    picture = comfy.utils.common_upscale(
        image[:1, :, :, :3].movedim(-1, 1),
        latent_width * SPATIAL_FACTOR,
        latent_height * SPATIAL_FACTOR,
        "bilinear",
        "center",
    ).movedim(1, -1)
    reference = vae.encode(picture)
    if reference.ndim == 4:
        reference = reference.unsqueeze(2)
    values = {REFERENCE_KEY: reference[:, :, :1]}
    return (
        node_helpers.conditioning_set_values(positive, values),
        node_helpers.conditioning_set_values(negative, values),
    )
