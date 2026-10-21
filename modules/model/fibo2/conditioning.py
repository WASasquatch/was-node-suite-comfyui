"""Fibo 2.0 prompts encoded on ComfyUI's Qwen3-VL 4B, and the images an edit reads.

Conditioning is ``(1, L, 6 * 7680)``, the :data:`dit.BUNDLES` of Qwen3-VL layers at unit RMS.
Images to edit ride on both conditionings as ``reference_latents``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from PIL import Image

from . import prompts, text_images
from .dit import BUNDLES, TEXT_LAYERS, TEXT_WIDTH

#: Where the fallback fonts for caption text live.
FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "fibo2")

#: The most tokens a prompt keeps, images included.
MAX_TOKENS = 4096

#: Qwen3-VL's image grid: sides are multiples of this, and the pixel count stays within these.
VISION_FACTOR = 32
VISION_MIN_PIXELS = 3136
VISION_MAX_PIXELS = 1605632

#: The token every image is spliced in at.
IMAGE_PAD = 151655

#: Channels of a FLUX.2 latent.
LATENT_CHANNELS = 128


def _check_clip(clip):
    """The tokenizer class to drive, once the CLIP is known to be Qwen3-VL 4B.

    Raises:
        ValueError: ``clip`` is missing or holds another text encoder.
    """
    from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer

    tokenizer = getattr(clip, "tokenizer", None)
    if not isinstance(tokenizer, Qwen3VLTokenizer) or getattr(tokenizer, "clip_name", "") != "qwen3vl_4b":
        raise ValueError(
            "Fibo 2.0 reads its prompt with Qwen3-VL 4B. Load qwen3vl_4b_bf16.safetensors in Load CLIP "
            "with type fibo2 where ComfyUI offers it, otherwise krea2."
        )
    return Qwen3VLTokenizer


def to_pil(frame: torch.Tensor) -> Image.Image:
    """One ``(H, W, C)`` frame of an IMAGE as an 8-bit RGB picture."""
    pixels = (frame[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(pixels, "RGB")


def to_image(picture: Image.Image) -> torch.Tensor:
    """An 8-bit picture as a ``(1, H, W, 3)`` IMAGE."""
    return torch.from_numpy(np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def vision_size(height: int, width: int) -> tuple[int, int]:
    """The ``(height, width)`` Qwen3-VL reads a picture at."""
    rows = round(height / VISION_FACTOR) * VISION_FACTOR
    columns = round(width / VISION_FACTOR) * VISION_FACTOR
    if rows * columns > VISION_MAX_PIXELS:
        beta = math.sqrt(height * width / VISION_MAX_PIXELS)
        rows = max(VISION_FACTOR, math.floor(height / beta / VISION_FACTOR) * VISION_FACTOR)
        columns = max(VISION_FACTOR, math.floor(width / beta / VISION_FACTOR) * VISION_FACTOR)
    elif rows * columns < VISION_MIN_PIXELS:
        beta = math.sqrt(VISION_MIN_PIXELS / (height * width))
        rows = math.ceil(height * beta / VISION_FACTOR) * VISION_FACTOR
        columns = math.ceil(width * beta / VISION_FACTOR) * VISION_FACTOR
    return rows, columns


def vision_image(picture: Image.Image) -> torch.Tensor:
    """A picture at the size Qwen3-VL reads it, resampled bicubic with antialiasing on 8-bit values."""
    pixels = torch.from_numpy(np.array(picture.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)
    size = vision_size(pixels.shape[2], pixels.shape[3])
    if tuple(pixels.shape[2:]) != size:
        pixels = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", align_corners=False, antialias=True)
    return (pixels.float() / 255.0).permute(0, 2, 3, 1)


def encode(clip, text: str, images=()) -> torch.Tensor:
    """One prompt's hidden states as Fibo 2.0 reads them.

    Args:
        clip: Qwen3-VL 4B from Load CLIP.
        text: The full chat text, a vision marker where each image stands.
        images: ``(1, H, W, 3)`` pictures, in marker order.

    Returns:
        ``(1, L, 15 * 2560)`` float32 on ComfyUI's intermediate device.
    """
    import comfy.model_management

    tokenizer_class = _check_clip(clip)
    tokens = tokenizer_class.tokenize_with_weights(clip.tokenizer, text, images=list(images), skip_template=True)
    clip.load_model(tokens)
    outer = clip.cond_stage_model
    inner = getattr(outer, outer.clip)
    device = clip.patcher.load_device
    rows = [[pair[0] for pair in row] for row in tokens[outer.clip_name]]
    with torch.no_grad(), comfy.model_management.cuda_device_context(device):
        embeds, _, count, info = inner.process_tokens(rows, device)
        _, taps = inner.transformer(None, None, embeds=embeds, num_tokens=count, intermediate_output=list(TEXT_LAYERS),
                                    final_layer_norm_intermediate=False, dtype=torch.float32, embeds_info=info)[:2]
        if taps.shape[-1] != TEXT_WIDTH:
            raise ValueError(f"Fibo 2.0 reads Qwen3-VL 4B, whose width is {TEXT_WIDTH}, not {taps.shape[-1]}.")
        taps = taps[:, :, :MAX_TOKENS].float()
        taps = taps * torch.rsqrt(taps.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        out = torch.cat([taps[:, TEXT_LAYERS.index(layer)] for bundle in BUNDLES for layer in bundle], dim=-1)
    return out.to(comfy.model_management.intermediate_device())


def edit_images(pictures, width: int | None, height: int | None, mask: Image.Image | None):
    """The images to edit: crops for the VAE and copies for Qwen3-VL.

    Args:
        pictures: 1 to 5 RGB pictures.
        width: Crop width for a single picture, ``None`` for its trained size.
        height: Crop height for a single picture, ``None`` for its trained size.
        mask: White where the single picture is repainted, or ``None``.

    Returns:
        ``(crops, thumbnails)``.
    """
    if mask is not None:
        grey = Image.new("RGB", pictures[0].size, (128, 128, 128))
        binary = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        pictures = [Image.composite(grey, pictures[0].convert("RGB"), binary)]
    if len(pictures) == 1:
        if width is None and height is None:
            width, height = prompts.trained_size(pictures[0].width / pictures[0].height)
        sizes = [(width or 1024, height or 1024)]
    else:
        sizes = [prompts.trained_size(picture.width / picture.height) for picture in pictures]
    crops = []
    for picture, (crop_width, crop_height) in zip(pictures, sizes):
        picture = picture.convert("RGB")
        scale = max(crop_height / picture.height, crop_width / picture.width)
        cover = (math.ceil(picture.width * scale), math.ceil(picture.height * scale))
        picture = picture.resize(cover, resample=Image.LANCZOS)
        left, top = (cover[0] - crop_width) // 2, (cover[1] - crop_height) // 2
        crops.append(picture.crop((left, top, left + crop_width, top + crop_height)))
    thumbnails = []
    for crop in crops:
        thumbnail = crop.copy()
        thumbnail.thumbnail((prompts.VISION_SIDE, prompts.VISION_SIDE))
        thumbnails.append(thumbnail)
    return crops, thumbnails


def _entry(cond: torch.Tensor, **values) -> list:
    return [cond, {"pooled_output": None, **values}]


def conditioning(clip, prompt: str, negative_prompt: str = "", images=(), mask=None, vae=None, width: int = 0,
                 height: int = 0, batch_size: int = 1, draw_text: bool = True):
    """Positive and negative conditioning and an empty latent for a Fibo 2.0 generation or edit.

    Args:
        clip: Qwen3-VL 4B from Load CLIP.
        prompt: A JSON caption or plain text; for an edit, the instruction or a caption with one.
        negative_prompt: What guidance runs against, empty for Bria's own.
        images: IMAGE tensors to edit, every frame an image, at most 5 in all.
        mask: A MASK, white where the single image to edit is repainted.
        vae: The FLUX.2 VAE, needed to edit.
        width: Image width, 0 for 1024 or the first image's trained size.
        height: Image height, 0 for 1024 or the first image's trained size.
        batch_size: Latents in the batch.
        draw_text: Whether Qwen3-VL also sees pictures of the caption's text.

    Returns:
        ``(positive, negative, latent)``.

    Raises:
        ValueError: More than 5 images, a mask with more than one image, or an edit without the
            FLUX.2 VAE.
    """
    import comfy.model_management

    frames = [frame for image in images if image is not None for frame in image]
    if len(frames) > prompts.MOST_IMAGES:
        raise ValueError(f"Fibo 2.0 edits up to {prompts.MOST_IMAGES} images at once, and was given {len(frames)}.")
    if mask is not None and len(frames) != 1:
        raise ValueError("A mask repaints one image. Connect exactly one image with it.")
    pictures = [to_pil(frame) for frame in frames]

    caption, _ = prompts.split(prompt)
    drawn = []
    if draw_text:
        fallbacks = text_images.fallback_fonts(FONTS)
        drawn = [vision_image(text_images.draw(text, fallbacks)) for text in prompts.texts(caption)]

    if pictures:
        if vae is None or getattr(vae, "latent_channels", LATENT_CHANNELS) != LATENT_CHANNELS:
            raise ValueError("An edit encodes its images with the FLUX.2 VAE. Load flux2-vae.safetensors in Load VAE "
                             "and connect it to vae.")
        layer = None
        if mask is not None:
            values = (mask[0].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            layer = Image.fromarray(values, "L")
            if layer.size != pictures[0].size:
                layer = layer.resize(pictures[0].size, resample=Image.NEAREST)
        if width or height:
            crop_size = (int(width) or None, int(height) or None)
        else:
            crop_size = (None, None)
        crops, thumbnails = edit_images(pictures, crop_size[0], crop_size[1], layer)
        if len(crops) == 1:
            width, height = crops[0].size
        elif not (width or height):
            width, height = prompts.trained_size(pictures[0].width / pictures[0].height)
        text = prompts.CHAT.format(prompts.edit(prompt, len(pictures), bool(drawn)))
        positive_cond = encode(clip, text, [vision_image(thumbnail) for thumbnail in thumbnails] + drawn)
        negative_cond = encode(clip, prompts.CHAT.format(negative_prompt.strip() or prompts.NULL_CAPTION))
        references = [vae.encode(to_image(crop)) for crop in crops]
        positive = [_entry(positive_cond, reference_latents=references)]
        negative = [_entry(negative_cond, reference_latents=references)]
    else:
        text, early = prompts.generation(prompt, bool(drawn))
        if early is None:
            positive = [_entry(encode(clip, prompts.CHAT.format(text)))]
        else:
            switch = float(np.float32(prompts.TEXT_IMAGE_SIGMA))
            above = float(np.nextafter(np.float32(prompts.TEXT_IMAGE_SIGMA), np.float32(1.0)))
            positive = [
                _entry(encode(clip, prompts.CHAT.format(early)), timestep_end=above),
                _entry(encode(clip, prompts.CHAT.format(text), drawn), timestep_start=switch),
            ]
        negative = [_entry(encode(clip, prompts.CHAT.format(negative_prompt.strip() or prompts.GENERATION_NEGATIVE)))]

    width = max(16, (int(width) or 1024) // 16 * 16)
    height = max(16, (int(height) or 1024) // 16 * 16)
    latent = torch.zeros([batch_size, LATENT_CHANNELS, height // 16, width // 16],
                         device=comfy.model_management.intermediate_device())
    return positive, negative, {"samples": latent}
