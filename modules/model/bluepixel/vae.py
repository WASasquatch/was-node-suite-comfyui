"""DC-AE f64c128, the autoencoder BluePixel samples on, as a ComfyUI VAE.

Latents are ``(B, 128, H / 64, W / 64)``, unscaled. Pictures go in and out as ComfyUI
images in ``[0, 1]``.
"""

from __future__ import annotations

import functools

import torch

from .model import CHANNELS, COMPRESSION

#: The folder the autoencoder file is listed from.
FOLDER = "vae"

#: The released configuration, ``mit-han-lab/dc-ae-f64c128-mix-1.0``.
CONFIG = {
    "in_channels": 3,
    "latent_channels": CHANNELS,
    "attention_head_dim": 32,
    "encoder_block_types": ["ResBlock"] * 3 + ["EfficientViTBlock"] * 4,
    "decoder_block_types": ["ResBlock"] * 3 + ["EfficientViTBlock"] * 4,
    "encoder_block_out_channels": (128, 256, 512, 512, 1024, 1024, 2048),
    "decoder_block_out_channels": (128, 256, 512, 512, 1024, 1024, 2048),
    "encoder_layers_per_block": (0, 4, 8, 2, 2, 2, 2),
    "decoder_layers_per_block": (0, 5, 10, 2, 2, 2, 2),
    "encoder_qkv_multiscales": ((),) * 7,
    "decoder_qkv_multiscales": ((),) * 7,
    "upsample_block_type": "pixel_shuffle",
    "downsample_block_type": "pixel_unshuffle",
    "decoder_norm_types": ["batch_norm"] * 3 + ["rms_norm"] * 4,
    "decoder_act_fns": ["relu"] * 3 + ["silu"] * 4,
    "scaling_factor": 1.0,
}

#: The smallest decode tile, in latent units, and the overlap blended between tiles.
TILE = 8
OVERLAP = 1

#: Bytes of working memory per picture pixel, per byte of the working dtype.
ENCODE_BYTES_PER_PIXEL = 340
DECODE_BYTES_PER_PIXEL = 375

#: Working memory on top of the per-pixel figure, per byte of the working dtype.
BASE_BYTES = 64 * 1024 * 1024


def offered() -> tuple[list[str], str | None]:
    """The files in ComfyUI's vae folders, and the one most likely to be DC-AE f64c128."""
    import folder_paths

    files = list(folder_paths.get_filename_list(FOLDER))
    flat = {name: name.lower().replace("-", "_") for name in files}
    likely = [name for name, low in flat.items() if "f64c128" in low]
    likely = likely or [name for name, low in flat.items() if "dc_ae" in low or "dcae" in low or "bluepixel" in low]
    return files, (likely[0] if likely else None)


def is_dcae(state_dict) -> bool:
    """Whether a state dict holds DC-AE f64c128's encoder and decoder."""
    conv_in = state_dict.get("decoder.conv_in.weight")
    return (
        conv_in is not None
        and tuple(conv_in.shape[:2]) == (2048, CHANNELS)
        and "encoder.conv_in.conv.weight" in state_dict
        and "decoder.conv_out.conv.weight" in state_dict
    )


@functools.cache
def _autoencoder_class():
    from comfy.ldm.ace.vae.autoencoder_dc import AutoencoderDC

    class DCAutoencoder(AutoencoderDC):
        """DC-AE f64c128, decoding whole or in tiles blended across their overlap."""

        comfy_decode_estimate_is_reliable = True

        def encode(self, x):
            """Pixels ``(B, 3, H, W)`` in ``[-1, 1]`` to latents, other models moved out first when short of room."""
            _make_room(self, x.shape[0] * x.shape[-2] * x.shape[-1], ENCODE_BYTES_PER_PIXEL, x)
            return super().encode(x)

        def decode(self, z):
            """Latents ``(B, 128, h, w)`` to pixels in ``[-1, 1]``, other models moved out first when short of room."""
            pixels = z.shape[0] * z.shape[-2] * z.shape[-1] * COMPRESSION * COMPRESSION
            _make_room(self, pixels, DECODE_BYTES_PER_PIXEL, z)
            return super().decode(z)

        def decode_tiled(self, z, tile_x=TILE, tile_y=TILE, overlap=OVERLAP, **_):
            """Decode ``z`` tile by tile, each tile's overlap blended linearly into the one before.

            Args:
                z: Latents ``(B, 128, h, w)``.
                tile_x: Tile width in latent units, at least :data:`TILE`.
                tile_y: Tile height in latent units, at least :data:`TILE`.
                overlap: Latent units each tile shares with the one before it.

            Returns:
                Pixels ``(B, 3, 64 h, 64 w)`` in ``[-1, 1]``.
            """
            tile_h, tile_w = max(int(tile_y), TILE), max(int(tile_x), TILE)
            overlap = max(0, min(int(overlap), min(tile_h, tile_w) - 1))
            stride_h, stride_w = tile_h - overlap, tile_w - overlap
            blend = overlap * COMPRESSION
            height, width = z.shape[-2:]
            rows = []
            for top in range(0, height, stride_h):
                rows.append([
                    self.decoder(z[:, :, top:top + tile_h, left:left + tile_w])
                    for left in range(0, width, stride_w)
                ])
            out_rows = []
            for i, row in enumerate(rows):
                kept = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = _blend(rows[i - 1][j], tile, blend, axis=2)
                    if j > 0:
                        tile = _blend(row[j - 1], tile, blend, axis=3)
                    kept.append(tile[:, :, : stride_h * COMPRESSION, : stride_w * COMPRESSION])
                out_rows.append(torch.cat(kept, dim=3))
            return torch.cat(out_rows, dim=2)

    return DCAutoencoder


def _make_room(model, pixels: int, per_pixel: int, sample: torch.Tensor) -> None:
    """Move every other model's weights out of the GPU when ``pixels`` of work exceed its physically free memory."""
    device = sample.device
    patcher = getattr(model, "current_patcher", None)
    if device.type != "cuda" or patcher is None:
        return
    import comfy.model_management as mm

    from .. import release_others

    need = (int(pixels) * per_pixel + BASE_BYTES) * sample.element_size() + int(patcher.model_size())
    try:
        used = torch.cuda.device_memory_used(device)
    except Exception:
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
    spare = torch.cuda.get_device_properties(device).total_memory - used - mm.EXTRA_RESERVED_VRAM
    if need > spare:
        release_others([patcher], device, unpin=True)


def _blend(before: torch.Tensor, after: torch.Tensor, extent: int, axis: int) -> torch.Tensor:
    """``after``, changed in place, its first ``extent`` pixels along ``axis`` faded in from the end of ``before``."""
    extent = min(before.shape[axis], after.shape[axis], extent)
    if extent <= 0:
        return after
    for index in range(extent):
        weight = index / extent
        tail = before.narrow(axis, before.shape[axis] - extent + index, 1)
        head = after.narrow(axis, index, 1)
        head.copy_(tail * (1 - weight) + head * weight)
    return after


def build(state_dict, device=None, dtype=None, disable_dynamic: bool = False):
    """A ComfyUI VAE around DC-AE f64c128.

    Args:
        state_dict: The autoencoder's tensors, as ``mit-han-lab/dc-ae-f64c128-mix-1.0`` releases them.
        device: Where it runs, ComfyUI's VAE device when ``None``.
        dtype: Its working dtype, ComfyUI's choice for the device when ``None``.
        disable_dynamic: Build a patcher that loads every weight at once.

    Returns:
        A ``comfy.sd.VAE``.

    Raises:
        ValueError: The tensors are not DC-AE f64c128's.
    """
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.sd

    if not is_dcae(state_dict):
        raise ValueError(
            "This file is not the DC-AE f64c128 autoencoder. Pick vae/diffusion_pytorch_model."
            "safetensors from openvivo/BluePixel-Turbo or mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers."
        )

    vae = comfy.sd.VAE(no_init=True)
    vae.first_stage_model = _autoencoder_class()(**CONFIG).eval()
    vae.downscale_ratio = COMPRESSION
    vae.upscale_ratio = COMPRESSION
    vae.latent_channels = CHANNELS
    vae.latent_dim = 2
    vae.output_channels = 3
    vae.pad_channel_value = None
    vae.process_input = lambda image: image * 2.0 - 1.0
    vae.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
    vae.working_dtypes = [torch.bfloat16, torch.float32]
    vae.memory_used_encode = lambda shape, dtype: _memory(shape, dtype, ENCODE_BYTES_PER_PIXEL, 1)
    vae.memory_used_decode = lambda shape, dtype: _memory(shape, dtype, DECODE_BYTES_PER_PIXEL, COMPRESSION)
    vae.disable_offload = False
    vae.not_video = True
    vae.size = None
    vae.downscale_index_formula = None
    vae.upscale_index_formula = None
    vae.extra_1d_channel = None
    vae.crop_input = True
    vae.handles_tiling = True
    vae.format_encoded = None
    vae.audio_sample_rate = 44100

    vae.device = device if device is not None else mm.vae_device()
    vae.vae_dtype = dtype if dtype is not None else mm.vae_dtype(vae.device, vae.working_dtypes)
    vae.output_device = mm.intermediate_device()
    vae.first_stage_model.to(vae.vae_dtype)
    mm.archive_model_dtypes(vae.first_stage_model)
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    vae.patcher = patcher_class(vae.first_stage_model, load_device=vae.device, offload_device=mm.vae_offload_device())
    vae.first_stage_model.current_patcher = vae.patcher

    missing, unexpected = vae.first_stage_model.load_state_dict(state_dict, strict=False, assign=vae.patcher.is_dynamic())
    if not vae.patcher.is_dynamic():
        vae.first_stage_model.to(vae.vae_dtype)
    if missing or unexpected:
        first = (missing or unexpected)[0]
        raise ValueError(
            f"The DC-AE file does not match f64c128: {len(missing)} tensors missing and "
            f"{len(unexpected)} unexpected, starting with {first}. Download "
            "vae/diffusion_pytorch_model.safetensors from openvivo/BluePixel-Turbo again."
        )
    vae.model_size()
    return vae


def _memory(shape, dtype, per_pixel: int, scale: int) -> int:
    import comfy.model_management as mm

    pixels = int(shape[0]) * int(shape[-2]) * int(shape[-1]) * scale * scale
    return (pixels * per_pixel + BASE_BYTES) * mm.dtype_size(dtype)


def load(path: str, disable_dynamic: bool = False):
    """Load DC-AE f64c128 from a file as a ComfyUI VAE.

    Raises:
        ValueError: The file is not DC-AE f64c128.
    """
    import comfy.utils

    vae = build(comfy.utils.load_torch_file(path, safe_load=True), disable_dynamic=disable_dynamic)
    vae.patcher.cached_patcher_init = (load_patcher, (path,))
    return vae


def load_patcher(path: str, disable_dynamic: bool = False):
    """The patcher of DC-AE f64c128 freshly loaded from ``path``."""
    return load(path, disable_dynamic=disable_dynamic).patcher
