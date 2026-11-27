"""Kandinsky 6 video super-resolution: the transformer as a MODEL, the K-VAE as a VAE, the latent upscalers.

Latents are ``(B, 64, 1 + (T - 1) / 4, H / 16, W / 16)``, sampled times :data:`SCALING_FACTOR`.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import torch

#: The ``image_model`` value of the transformer's unet config.
IMAGE_MODEL = "was_kandinsky6_sr"

CHANNELS = 64
SPATIAL_FACTOR = 16
TEMPORAL_FACTOR = 4
SCALING_FACTOR = 0.910344004631042
#: Sampling shift of the VSR 2.0 checkpoints.
SHIFT = 5.0
#: Pixel tile sizes, ``(height, width)``, the transformer was trained on.
TILE_SIZES = ((512, 512), (512, 768), (768, 512))
#: Last PiFlow step's share of an ordinary step.
FINAL_STEP_SCALE = 0.5

FOLDER = "diffusion_models"
VAE_FOLDER = "vae"
UPSCALER_FOLDER = "latent_upscale_models"

#: Weight storage choices, as ComfyUI's own diffusion model loader offers them.
WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")

_REQUIRED = (
    "pooled_bias",
    "visual_embeddings.in_layer.weight",
    "time_embeddings.timestep_embedder.linear_2.weight",
    "visual_transformer_blocks.0.self_attention.to_query.weight",
    "visual_transformer_blocks.0.self_attention.query_norm.weight",
    "visual_transformer_blocks.0.visual_modulation.out_layer.weight",
    "out_layer.out_layer.weight",
)
_PREFIXES = ("model.diffusion_model.", "diffusion_model.")

#: Encoder and decoder widths of the released K-VAE.
VAE_CONFIG = {
    "encoder_block_out_channels": (16, 128, 256, 512, 1024),
    "decoder_block_out_channels": (16, 256, 512, 1024, 2048),
}


def _stripped(state_dict: dict) -> dict:
    for prefix in _PREFIXES:
        if any(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    return state_dict


def detect(state_dict) -> dict | None:
    """The unet config of a Kandinsky 6 super-resolution transformer, or ``None`` for other weights."""
    if any(name not in state_dict for name in _REQUIRED):
        return None

    def shape(name):
        return tuple(state_dict[name].shape)

    model_dim, visual_in = shape("visual_embeddings.in_layer.weight")
    in_visual_dim = (visual_in - 1) // 2
    out_visual_dim = shape("out_layer.out_layer.weight")[0]
    head_dim = shape("visual_transformer_blocks.0.self_attention.query_norm.weight")[0]
    blocks = 0
    while f"visual_transformer_blocks.{blocks}.self_attention.to_query.weight" in state_dict:
        blocks += 1
    if in_visual_dim != CHANNELS or out_visual_dim % in_visual_dim:
        raise ValueError(
            f"This file has Kandinsky 6 upscaler layer names but a {in_visual_dim}-channel input and a "
            f"{out_visual_dim}-channel output, which no release has. Load the transformer of a "
            "kandinskylab/Kandinsky-6.0-VSR repository."
        )
    return {
        "image_model": IMAGE_MODEL,
        "in_visual_dim": in_visual_dim,
        "out_visual_dim": out_visual_dim,
        "time_dim": shape("time_embeddings.timestep_embedder.linear_2.weight")[0],
        "patch_size": (1, 1, 1),
        "model_dim": model_dim,
        "ff_dim": shape("visual_transformer_blocks.0.feed_forward.net.0.proj.weight")[0],
        "num_visual_blocks": blocks,
        "axes_dims": (head_dim // 4, head_dim * 3 // 8, head_dim * 3 // 8),
        "scale_factor": (1.0, 2.0, 2.0),
        "nabla_threshold": 0.8,
        "nabla_window": (11, 7, 7),
        "n_grid": out_visual_dim // in_visual_dim,
    }


def piflow_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """The distilled checkpoints' schedule: even raw steps, the last at :data:`FINAL_STEP_SCALE`, shifted."""
    segment = 1.0 / (steps - (1.0 - FINAL_STEP_SCALE))
    raw = torch.tensor([max(1.0 - index * segment, 1e-6) for index in range(steps)], dtype=torch.float32)
    return torch.cat([shift * raw / (1.0 + (shift - 1.0) * raw), torch.zeros(1)])


def flow_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """Evenly spaced flow-matching sigmas, shifted, ending at 0."""
    raw = torch.linspace(1.0, 0.0, steps + 1)
    return shift * raw / (1.0 + (shift - 1.0) * raw)


@functools.cache
def _classes():
    import comfy.latent_formats
    import comfy.model_base
    import comfy.patcher_extension
    import comfy.supported_models_base

    from ...vendor.kandinsky6 import piflow, sr_dit
    from .model import SHIFT_OPTION
    from .transformer import PIFLOW_EPS, PIFLOW_SUBSTEPS, next_sigma, unwarp

    class KVAELatent(comfy.latent_formats.LatentFormat):
        """K-VAE latents as the Kandinsky 6 upscaler samples them."""

        latent_channels = CHANNELS
        latent_dimensions = 3
        scale_factor = SCALING_FACTOR
        spacial_downscale_ratio = SPATIAL_FACTOR
        temporal_downscale_ratio = TEMPORAL_FACTOR

    class UpscaleTransformer(sr_dit.SRTransformer3D):
        """The super-resolution transformer reading ComfyUI's channels-first latent."""

        def __init__(self, n_grid: int = 1, **kwargs):
            super().__init__(**kwargs)
            self.n_grid = int(n_grid)

        def forward(self, x, timestep, context=None, y=None, control=None, transformer_options={}, **kwargs):
            return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self._forward,
                self,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
                ),
            ).execute(x, timestep, transformer_options)

        def _forward(self, x, timestep, transformer_options):
            if x.shape[-2] % 8 or x.shape[-1] % 8:
                raise ValueError(
                    "The Kandinsky 6 upscaler refines tiles in multiples of 128 pixels; this latent is "
                    f"{x.shape[-1] * SPATIAL_FACTOR}x{x.shape[-2] * SPATIAL_FACTOR}."
                )
            anchor = torch.zeros_like(x)
            mask = torch.zeros_like(x[:, :1])
            out = super().forward(torch.cat([x, anchor, mask], dim=1).movedim(1, -1), timestep).movedim(-1, 1)
            if self.n_grid == 1:
                return out
            grid = out.float().unflatten(1, (self.n_grid, -1))
            sigma = timestep.float().to(x.device) / 1000.0
            shift = float(transformer_options.get(SHIFT_OPTION, SHIFT))
            target = next_sigma(transformer_options.get("sample_sigmas", None), float(sigma[0]))
            if target is None:
                return grid[:, -1]
            raw_src = unwarp(sigma, shift)
            raw_dst = unwarp(torch.full_like(sigma, target), shift).clamp(min=PIFLOW_EPS)
            state = x.float()
            policy = piflow.DXPolicy(grid, state, sigma, raw_src - raw_dst, shift=shift, eps=PIFLOW_EPS)
            end, _, _ = piflow.policy_rollout_fm(state, sigma, raw_src, raw_dst, PIFLOW_SUBSTEPS, policy)
            return (end - state) / (target - sigma).reshape(-1, *([1] * (state.ndim - 1)))

    class Kandinsky6Upscaler(comfy.model_base.BaseModel):
        """ComfyUI model wrapping the Kandinsky 6 super-resolution transformer."""

        def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
            super().__init__(model_config, model_type, device=device, unet_model=UpscaleTransformer)
            self.n_grid = int(model_config.unet_config.get("n_grid", 1))

        def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            transformer_options = dict(transformer_options)
            transformer_options[SHIFT_OPTION] = float(getattr(self.model_sampling, "shift", SHIFT))
            return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    class Kandinsky6UpscalerConfig(comfy.supported_models_base.BASE):
        """Model config for the unet config :func:`detect` writes."""

        unet_config = {"image_model": IMAGE_MODEL}
        unet_extra_config = {}
        sampling_settings = {"shift": SHIFT}
        latent_format = KVAELatent
        memory_usage_factor = 0.6
        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        def get_model(self, state_dict, prefix="", device=None):
            return Kandinsky6Upscaler(self, device=device)

        @staticmethod
        def was_rebuild(state_dict):
            """Build the model again from rewritten tensors, such as Quantize Model's."""
            return from_state_dict(state_dict)

    return KVAELatent, Kandinsky6UpscalerConfig


def _options(weight_dtype: str) -> dict:
    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"No weight storage {weight_dtype!r}. Choose one of: {', '.join(WEIGHT_DTYPES)}.")
    options = {}
    if weight_dtype in ("fp8_e4m3fn", "fp8_e4m3fn_fast"):
        options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    return options


def load_model(path: str, weight_dtype: str = "default", disable_dynamic: bool = False):
    """Load a Kandinsky 6 super-resolution transformer, plain or quantized, as a ComfyUI MODEL.

    Returns:
        The model's ``ModelPatcher``.

    Raises:
        ValueError: The file is not a Kandinsky 6 super-resolution transformer, or ``weight_dtype``
            names nothing offered.
    """
    import comfy.utils

    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    patcher = from_state_dict(state_dict, metadata, weight_dtype, disable_dynamic, os.path.basename(path))
    patcher.cached_patcher_init = (load_model, (path, weight_dtype))
    return patcher


def from_state_dict(state_dict, metadata=None, weight_dtype: str = "default", disable_dynamic: bool = False,
                    name: str = "These weights"):
    """Build a Kandinsky 6 super-resolution MODEL from its tensors, plain or quantized.

    Returns:
        The model's ``ModelPatcher``.

    Raises:
        ValueError: The tensors are not a Kandinsky 6 super-resolution transformer, or
            ``weight_dtype`` names nothing offered.
    """
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.storage
    import comfy.utils

    options = _options(weight_dtype)
    state_dict = _stripped(state_dict)
    state_dict, metadata = comfy.utils.convert_old_quants(state_dict, "", metadata=metadata)
    unet_config = detect(state_dict)
    if unet_config is None:
        raise ValueError(
            f"{name} is not a Kandinsky 6 upscaler transformer. Pick the transformer of a "
            "kandinskylab/Kandinsky-6.0-VSR repository, such as kandinsky6_vsr2_int8_convrot.safetensors."
        )
    _, config_class = _classes()
    model_config = config_class(unet_config)
    quant = comfy.utils.detect_layer_quantization(state_dict, "")
    if quant:
        model_config.quant_config = quant

    parameters = comfy.utils.calculate_parameters(state_dict)
    stored = None if model_config.quant_config is not None else comfy.utils.weight_dtype(state_dict)
    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    supported = list(model_config.supported_inference_dtypes)
    unet_dtype = options.get("dtype") or mm.unet_dtype(model_params=parameters, supported_dtypes=supported, weight_dtype=stored)
    cast_from = None if model_config.quant_config is not None else unet_dtype
    manual_cast = mm.unet_manual_cast(cast_from, load_device, supported)
    model_config.set_inference_dtype(unet_dtype, manual_cast, device=load_device)
    if weight_dtype == "fp8_e4m3fn_fast":
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(state_dict, "")
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device,
                            fast_disk=comfy.storage.state_dict_fast_disk(state_dict))
    if not mm.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(state_dict, "", assign=patcher.is_dynamic())
    return patcher


def is_upscaler_model(patcher) -> bool:
    """Whether a MODEL is the Kandinsky 6 super-resolution transformer."""
    config = getattr(getattr(getattr(patcher, "model", None), "model_config", None), "unet_config", {}) or {}
    return config.get("image_model") == IMAGE_MODEL


def is_kvae(state_dict) -> bool:
    """Whether the tensors are the Kandinsky 6 K-VAE."""
    return ("decoder.norm_out.conv_y.weight" in state_dict and "encoder.conv_out.conv.weight" in state_dict
            and tuple(state_dict["encoder.conv_out.conv.weight"].shape[:1]) == (2 * CHANNELS,))


def _vae_memory(shape, dtype, per_pixel: int, scale: int) -> int:
    import comfy.model_management as mm

    return int(shape[-2]) * int(shape[-1]) * scale * scale * per_pixel * mm.dtype_size(dtype)


def build_vae(state_dict, device=None, dtype=None, disable_dynamic: bool = False):
    """A ComfyUI VAE around the K-VAE.

    Raises:
        ValueError: The tensors are not the K-VAE's.
    """
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.sd

    from ...vendor.kandinsky6.kvae import KVAE, decode_segment

    if not is_kvae(state_dict):
        raise ValueError(
            "This file is not the Kandinsky 6 K-VAE. Pick vae/diffusion_pytorch_model.safetensors from a "
            "kandinskylab/Kandinsky-6.0-VSR repository, or kandinsky6_vsr_kvae_bf16.safetensors."
        )
    vae = comfy.sd.VAE(no_init=True)
    vae.first_stage_model = KVAE(**VAE_CONFIG).eval()
    vae.upscale_ratio = (lambda a: max(0, a * TEMPORAL_FACTOR - 3), SPATIAL_FACTOR, SPATIAL_FACTOR)
    vae.upscale_index_formula = (TEMPORAL_FACTOR, SPATIAL_FACTOR, SPATIAL_FACTOR)
    vae.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / TEMPORAL_FACTOR)), SPATIAL_FACTOR, SPATIAL_FACTOR)
    vae.downscale_index_formula = (TEMPORAL_FACTOR, SPATIAL_FACTOR, SPATIAL_FACTOR)
    vae.latent_channels = CHANNELS
    vae.latent_dim = 3
    vae.output_channels = 3
    vae.pad_channel_value = None
    vae.process_input = lambda image: image * (255.0 / 128.0) - 1.0
    vae.process_output = lambda image: image.add_(1.0).mul_(128.0 / 255.0).clamp_(0.0, 1.0)
    vae.working_dtypes = [torch.bfloat16, torch.float32]
    vae.memory_used_encode = lambda shape, dtype: _vae_memory(shape, dtype, 2600, 1)
    vae.memory_used_decode = lambda shape, dtype: _vae_memory(
        shape, dtype, 6 * 256 * (decode_segment(shape[-2], shape[-1]) + 1), SPATIAL_FACTOR)
    vae.disable_offload = False
    vae.not_video = False
    vae.size = None
    vae.extra_1d_channel = None
    vae.crop_input = True
    vae.handles_tiling = False
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
            f"The K-VAE file holds {len(missing)} tensors too few and {len(unexpected)} too many, starting "
            f"with {first}. Download vae/diffusion_pytorch_model.safetensors from a "
            "kandinskylab/Kandinsky-6.0-VSR repository again."
        )
    vae.model_size()
    return vae


def load_vae(path: str, disable_dynamic: bool = False):
    """Load the K-VAE from ``path`` as a ComfyUI VAE."""
    import comfy.utils

    vae = build_vae(comfy.utils.load_torch_file(path, safe_load=True), disable_dynamic=disable_dynamic)
    vae.patcher.cached_patcher_init = (load_vae_patcher, (path,))
    return vae


def load_vae_patcher(path: str, disable_dynamic: bool = False):
    """The patcher of the K-VAE freshly loaded from ``path``."""
    return load_vae(path, disable_dynamic=disable_dynamic).patcher


@dataclass
class LatentUpscaler:
    """A loaded latent upscaler.

    Attributes:
        patcher: Its ``ModelPatcher``.
        scale: How many times it enlarges a latent's width and height.
        name: The file it came from.
    """

    patcher: object
    scale: int
    name: str

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Enlarge scaled K-VAE latents ``(B, 64, T, H, W)`` on the upscaler's device."""
        import comfy.model_management as mm

        mm.load_models_gpu([self.patcher])
        module = self.patcher.model
        dtype = next(module.parameters()).dtype
        with torch.inference_mode():
            out = module(latents.to(self.patcher.load_device, dtype))
        return out.float()


def upscaler_scales(state_dict) -> dict[int, str]:
    """The scales a latent upscaler file holds, with the key prefix of each."""
    found = {}
    for prefix in ("", "_models.0.", "_models.1."):
        if f"{prefix}x2_branch.adapter.0.conv1.weight" in state_dict:
            found[2] = prefix
        elif f"{prefix}input_proj.0.weight" in state_dict and f"{prefix}pre_blocks.0.conv1.weight" in state_dict:
            found[4] = prefix
    return found


def load_upscaler(path: str, scale: int) -> LatentUpscaler:
    """Load the ``scale`` latent upscaler from a file holding it alone or in the released bank.

    Raises:
        ValueError: The file holds no Kandinsky 6 latent upscaler for ``scale``.
    """
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.utils

    from ...vendor.kandinsky6.latent_upscaler import LatentUpscaler as Network

    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    found = upscaler_scales(state_dict)
    name = os.path.basename(path)
    if scale not in found:
        held = ", ".join(f"x{s}" for s in sorted(found)) or "no Kandinsky 6 latent upscaler"
        raise ValueError(
            f"{name} holds {held}, and this upscale needs the x{scale} latent upscaler. Pick "
            f"kandinsky6_vsr_latent_upscaler_x{scale}.safetensors from models/latent_upscale_models."
        )
    prefix = found[scale]
    network = Network(scale).eval()
    wanted = network.state_dict().keys()
    weights = {key: state_dict[prefix + key] for key in wanted if prefix + key in state_dict}
    missing = [key for key in wanted if key not in weights]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} tensors of the x{scale} upscaler, starting with {missing[0]}.")
    device = mm.get_torch_device()
    dtype = torch.bfloat16 if mm.should_use_bf16(device) else torch.float32
    network.load_state_dict(weights, strict=True, assign=True)
    network.to(dtype)
    patcher = comfy.model_patcher.ModelPatcher(network, load_device=device, offload_device=mm.unet_offload_device())
    return LatentUpscaler(patcher, scale, name)
