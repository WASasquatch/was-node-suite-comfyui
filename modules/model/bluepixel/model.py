"""The BluePixel transformer: its key layout, its latent format and its loader.

Weights load in Black Forest Labs' FLUX.2 layout, Diffusers files converted to it. Latents are
DC-AE f64c128's: 128 channels at a 64th of the width and height.
"""

from __future__ import annotations

import torch

#: Model latents are the autoencoder's latents times this.
SCALING_FACTOR = 0.4538

#: Pixels per latent unit along each side.
COMPRESSION = 64

#: Latent channels.
CHANNELS = 128

#: Weight storage choices, as ComfyUI's own diffusion model loader offers them.
WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")

#: The folder the transformer files are listed from.
FOLDER = "diffusion_models"

_TOP = {
    "x_embedder": "img_in",
    "context_embedder": "txt_in",
    "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
    "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
    "time_guidance_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
    "time_guidance_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
    "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
    "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
    "single_stream_modulation.linear": "single_stream_modulation.lin",
    "proj_out": "final_layer.linear",
}

_DOUBLE = {
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}

_DOUBLE_NORMS = {
    "attn.norm_q": "img_attn.norm.query_norm",
    "attn.norm_k": "img_attn.norm.key_norm",
    "attn.norm_added_q": "txt_attn.norm.query_norm",
    "attn.norm_added_k": "txt_attn.norm.key_norm",
}

_DOUBLE_FUSED = {
    "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}

_SINGLE = {"attn.to_qkv_mlp_proj": "linear1", "attn.to_out": "linear2"}

_SINGLE_NORMS = {"attn.norm_q": "norm.query_norm", "attn.norm_k": "norm.key_norm"}


def offered() -> tuple[list[str], str | None]:
    """The files in ComfyUI's diffusion_models folders, and the one most likely to be BluePixel."""
    import folder_paths

    files = list(folder_paths.get_filename_list(FOLDER))
    likely = [name for name in files if "bluepixel" in name.lower().replace("-", "").replace("_", "")]
    preferred = [name for name in likely if "int8" in name.lower()] or likely
    return files, (preferred[0] if preferred else None)


def layout(state_dict) -> str | None:
    """``"native"``, ``"diffusers"``, or ``None`` for tensors that are neither FLUX.2 layout."""
    if "double_stream_modulation_img.lin.weight" in state_dict and "img_in.weight" in state_dict:
        return "native"
    if "double_stream_modulation_img.linear.weight" in state_dict and "x_embedder.weight" in state_dict:
        return "diffusers"
    return None


def to_native(state_dict: dict) -> dict:
    """A Diffusers FLUX.2 transformer under the native names; any other layout comes back as it is.

    Raises:
        ValueError: A Diffusers tensor has no native name.
    """
    if layout(state_dict) != "diffusers":
        return state_dict
    source = dict(state_dict)
    out = {}

    def move(name: str, target: str) -> None:
        for suffix in ("weight", "bias"):
            if f"{name}.{suffix}" in source:
                out[f"{target}.{suffix}"] = source.pop(f"{name}.{suffix}")

    def fuse(names, target: str) -> None:
        for suffix in ("weight", "bias"):
            keys = [f"{name}.{suffix}" for name in names]
            if all(key in source for key in keys):
                out[f"{target}.{suffix}"] = torch.cat([source.pop(key) for key in keys])

    for name, target in _TOP.items():
        move(name, target)
    # The final modulation is stored scale first in Diffusers and shift first natively.
    for suffix in ("weight", "bias"):
        key = f"norm_out.linear.{suffix}"
        if key in source:
            scale, shift = source.pop(key).chunk(2, dim=0)
            out[f"final_layer.adaLN_modulation.1.{suffix}"] = torch.cat([shift, scale])

    index = 0
    while f"transformer_blocks.{index}.attn.to_q.weight" in source:
        src, dst = f"transformer_blocks.{index}", f"double_blocks.{index}"
        for name, target in _DOUBLE.items():
            move(f"{src}.{name}", f"{dst}.{target}")
        for name, target in _DOUBLE_NORMS.items():
            out[f"{dst}.{target}.scale"] = source.pop(f"{src}.{name}.weight")
        for target, names in _DOUBLE_FUSED.items():
            fuse([f"{src}.{name}" for name in names], f"{dst}.{target}")
        index += 1

    index = 0
    while f"single_transformer_blocks.{index}.attn.to_out.weight" in source:
        src, dst = f"single_transformer_blocks.{index}", f"single_blocks.{index}"
        for name, target in _SINGLE.items():
            move(f"{src}.{name}", f"{dst}.{target}")
        for name, target in _SINGLE_NORMS.items():
            out[f"{dst}.{target}.scale"] = source.pop(f"{src}.{name}.weight")
        index += 1

    if source:
        raise ValueError(
            f"{len(source)} tensors of this Diffusers FLUX.2 transformer have no place in the "
            f"native layout, starting with {sorted(source)[0]}. The file is not a FLUX.2 Klein "
            "transformer."
        )
    return out


def _latent_format_class():
    import comfy.latent_formats

    class BluePixelLatent(comfy.latent_formats.LatentFormat):
        """DC-AE f64c128 latents as the BluePixel transformer samples them."""

        latent_channels = CHANNELS
        spacial_downscale_ratio = COMPRESSION
        scale_factor = SCALING_FACTOR

        def __init__(self):
            from .preview import RGB_BIAS, RGB_FACTORS

            self.latent_rgb_factors = [list(row) for row in RGB_FACTORS]
            self.latent_rgb_factors_bias = list(RGB_BIAS)

    return BluePixelLatent


def latent_format():
    """A fresh DC-AE f64c128 latent format for one loaded model."""
    return _latent_format_class()()


def _options(weight_dtype: str) -> dict:
    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"No weight storage {weight_dtype!r}. Choose one of: {', '.join(WEIGHT_DTYPES)}.")
    options = {}
    if weight_dtype in ("fp8_e4m3fn", "fp8_e4m3fn_fast"):
        options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    if weight_dtype == "fp8_e4m3fn_fast":
        options["fp8_optimizations"] = True
    return options


def load(path: str, weight_dtype: str = "default", disable_dynamic: bool = False):
    """Load a BluePixel transformer as a ComfyUI MODEL sampling DC-AE f64c128 latents.

    Args:
        path: The file, native or Diffusers layout, plain or quantized.
        weight_dtype: One of :data:`WEIGHT_DTYPES`.
        disable_dynamic: Build a patcher that loads every weight at once.

    Returns:
        The model's ``ModelPatcher``.

    Raises:
        ValueError: The file is not a FLUX.2 transformer taking 128-channel latents, or
            ``weight_dtype`` names nothing offered.
    """
    import os

    import comfy.sd
    import comfy.utils

    options = _options(weight_dtype)
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    if layout(state_dict) == "diffusers":
        state_dict = to_native(state_dict)
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options=options, metadata=metadata, disable_dynamic=disable_dynamic,
    )
    name = os.path.basename(path)
    if patcher is None:
        raise ValueError(
            f"{name} is not a diffusion model ComfyUI can read. Pick a BluePixel transformer, "
            "such as bluepixel_turbo_bf16.safetensors."
        )
    config = patcher.model.model_config.unet_config
    if config.get("image_model") != "flux2" or config.get("in_channels") != CHANNELS:
        raise ValueError(
            f"{name} is a {config.get('image_model', 'different')} model, not a FLUX.2 transformer "
            f"taking {CHANNELS}-channel latents. Pick a BluePixel transformer, such as "
            "bluepixel_turbo_bf16.safetensors, or load this file with Load Diffusion Model."
        )
    adopt(patcher)
    patcher.cached_patcher_init = (load, (path, weight_dtype))
    return patcher


def adopt(patcher) -> None:
    """Give a freshly loaded FLUX.2 model the DC-AE f64c128 latent format."""
    fmt = latent_format()
    patcher.model.latent_format = fmt
    patcher.model.model_config.latent_format = fmt


def parameters(patcher) -> int:
    """How many weights the loaded transformer holds, counting a quantized tensor once per value."""
    total = 0
    for value in patcher.model.diffusion_model.state_dict().values():
        shape = getattr(value, "shape", None)
        if shape is not None:
            count = 1
            for size in shape:
                count *= int(size)
            total += count
    return total


#: Quantized format tags under the names the pack's quantizer offers them.
FORMAT_NAMES = {"w6a8_int8": "w6a8", "asym_w4a8_int8": "w4a8", "convrot_w4a4": "w4a4"}


def weight_formats(path: str) -> dict:
    """How many weight matrices a safetensors file stores in each format: a quantized format, or a dtype.

    Returns:
        Format name -> count, empty for a file that is not safetensors.
    """
    import json

    from safetensors import safe_open

    if not path.lower().endswith(".safetensors"):
        return {}
    counts = {}
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
        quantized = {key[: -len(".comfy_quant")] for key in keys if key.endswith(".comfy_quant")}
        for layer in quantized:
            conf = json.loads(bytes(handle.get_tensor(f"{layer}.comfy_quant").tolist()).decode("utf-8"))
            tag = str(conf.get("format", "quantized"))
            name = "int8 convrot" if tag == "int8_tensorwise" and conf.get("convrot") else FORMAT_NAMES.get(tag, tag)
            counts[name] = counts.get(name, 0) + 1
        for key in keys:
            if not key.endswith(".weight") or key[: -len(".weight")] in quantized:
                continue
            tensor = handle.get_slice(key)
            if len(tensor.get_shape()) == 2:
                name = str(tensor.get_dtype()).lower()
                counts[name] = counts.get(name, 0) + 1
    return counts
