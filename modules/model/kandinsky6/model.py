"""Kandinsky 6 checkpoint detection, model class and registration.

The joint latent holds video ``(B, 16, T, H / 8, W / 8)`` and audio ``(B, 40, L)``, an audio
frame per 1024 samples at 44.1 kHz.
"""

from __future__ import annotations

import functools

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.nested_tensor
import comfy.supported_models_base
import comfy.utils

#: The ``image_model`` value detection writes into the unet config.
IMAGE_MODEL = "was_kandinsky6"

VIDEO_CHANNELS = 16
AUDIO_CHANNELS = 40
SPATIAL_FACTOR = 8
TEMPORAL_FACTOR = 4
PATCH_SIZE = (1, 2, 2)
AUDIO_SAMPLE_RATE = 44100
AUDIO_FRAME_SAMPLES = 1024
FRAME_RATE = 24.0
SHIFT = 5.0

#: Audio latent scale of the distilled and pretrained checkpoints.
AUDIO_SCALE_DISTILLED = 0.417
#: Audio latent scale of the released Pro and Lite checkpoints.
AUDIO_SCALE = 0.5302

#: Conditioning key holding the reference frame latent for image-to-video.
REFERENCE_KEY = "kandinsky6_reference"
#: transformer_options key carrying the sampling shift to the PiFlow policy.
SHIFT_OPTION = "kandinsky6_shift"

#: Qwen2.5-VL-7B's final RMSNorm weight, bundled in ``modules/data/models``.
TEXT_NORM_FILE = "qwen25_vl_7b_final_norm.safetensors"
TEXT_NORM_EPS = 1e-6

_REQUIRED_KEYS = (
    "visual_embeddings.in_layer.weight",
    "audio_embeddings.in_layer.weight",
    "out_layer.out_layer.weight",
    "audio_out_layer.out_layer.weight",
    "video_time_embeddings.timestep_embedder.linear_2.weight",
    "audio_time_embeddings.timestep_embedder.linear_2.weight",
    "video_text_embeddings.in_layer.weight",
    "video_pooled_text_embeddings.in_layer.weight",
    "visual_transformer_blocks.0.va_modulation.out_layer.weight",
    "visual_transformer_blocks.0.video_dec_block.self_attention.query_norm.weight",
    "visual_transformer_blocks.0.audio_dec_block.self_attention.query_norm.weight",
    "visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight",
    "visual_transformer_blocks.0.audio_dec_block.feed_forward.net.0.proj.weight",
)

_DETECTOR_MARK = "_was_kandinsky6_detector"


def _axes(head_dim: int) -> tuple[int, int, int]:
    return head_dim // 4, head_dim * 3 // 8, head_dim * 3 // 8


@functools.cache
def _text_norm_weight() -> torch.Tensor:
    from ...data import paths
    from safetensors.torch import load_file

    return load_file(str(paths.data_directory() / "models" / TEXT_NORM_FILE))["weight"].float()


def text_norm(hidden: torch.Tensor) -> torch.Tensor:
    """Qwen2.5-VL-7B's final RMSNorm over the last hidden layer, the text features Kandinsky 6 reads.

    Args:
        hidden: ``(B, L, 3584)`` hidden states before the final norm, as ComfyUI's
            ``kandinsky5`` text encoder answers them.

    Returns:
        The normalised states, float32.
    """
    weight = _text_norm_weight()
    if hidden.shape[-1] != weight.shape[0]:
        return hidden
    x = hidden.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + TEXT_NORM_EPS) * weight.to(x.device)


def detect(state_dict, key_prefix: str = "") -> dict | None:
    """Read a Kandinsky 6 transformer's configuration off its weights.

    Args:
        state_dict: The checkpoint's tensors, Diffusers key names.
        key_prefix: Prefix the transformer's keys carry, ``""`` for a bare transformer.

    Returns:
        The unet config, or ``None`` when the weights are not a Kandinsky 6 transformer.

    Raises:
        ValueError: The weights carry Kandinsky 6's keys in shapes no release has.
    """
    if any(key_prefix + name not in state_dict for name in _REQUIRED_KEYS):
        return None

    def shape(name):
        return tuple(state_dict[key_prefix + name].shape)

    def count(pattern):
        index = 0
        while key_prefix + pattern.format(index) in state_dict:
            index += 1
        return index

    patch_volume = PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]
    model_dim, visual_in = shape("visual_embeddings.in_layer.weight")
    model_dim_a, in_audio_dim = shape("audio_embeddings.in_layer.weight")
    in_visual_dim = (visual_in // patch_volume - 1) // 2
    out_visual_dim = shape("out_layer.out_layer.weight")[0] // patch_volume
    out_audio_dim = shape("audio_out_layer.out_layer.weight")[0]
    n_grid = out_visual_dim // max(in_visual_dim, 1)
    va_modulation = shape("visual_transformer_blocks.0.va_modulation.out_layer.weight")[0]
    head_dim = shape("visual_transformer_blocks.0.video_dec_block.self_attention.query_norm.weight")[0]
    head_dim_a = shape("visual_transformer_blocks.0.audio_dec_block.self_attention.query_norm.weight")[0]

    if (
        visual_in != (2 * in_visual_dim + 1) * patch_volume
        or in_visual_dim != VIDEO_CHANNELS
        or in_audio_dim != AUDIO_CHANNELS
        or out_visual_dim != in_visual_dim * n_grid
        or out_audio_dim != in_audio_dim * n_grid
        or va_modulation not in (2 * model_dim + model_dim_a, 3 * model_dim)
    ):
        raise ValueError(
            "This file has Kandinsky 6's layer names but not the shapes of any released Kandinsky 6 "
            f"transformer (model width {model_dim}, video input {visual_in}, video output "
            f"{out_visual_dim}, audio output {out_audio_dim}). Load a transformer from one of the "
            "kandinskylab/Kandinsky-6.0 repositories."
        )

    type_key = key_prefix + "visual_token_type_embeddings.weight"
    return {
        "image_model": IMAGE_MODEL,
        "in_visual_dim": in_visual_dim,
        "out_visual_dim": out_visual_dim,
        "in_text_dim": shape("video_text_embeddings.in_layer.weight")[1],
        "in_text_dim2": shape("video_pooled_text_embeddings.in_layer.weight")[1],
        "time_dim": shape("video_time_embeddings.timestep_embedder.linear_2.weight")[0],
        "patch_size": PATCH_SIZE,
        "model_dim": model_dim,
        "ff_dim": shape("visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj.weight")[0],
        "num_text_blocks": count("video_text_transformer_blocks.{}.attn.to_query.weight"),
        "num_visual_blocks": count("visual_transformer_blocks.{}.va_modulation.out_layer.weight"),
        "axes_dims": _axes(head_dim),
        "visual_cond": True,
        "in_audio_dim": in_audio_dim,
        "out_audio_dim": out_audio_dim,
        "model_dim_a": model_dim_a,
        "time_dim_a": shape("audio_time_embeddings.timestep_embedder.linear_2.weight")[0],
        "ff_dim_a": shape("visual_transformer_blocks.0.audio_dec_block.feed_forward.net.0.proj.weight")[0],
        "axes_dims_a": _axes(head_dim_a),
        "audio_freqs_scaling": 0.144,
        "scale_factor": (1.0, 2.0, 2.0),
        "ca_rope": True,
        "cross_gates": va_modulation == 2 * model_dim + model_dim_a,
        "fix_modulation": True,
        "visual_token_type_num_embeddings": state_dict[type_key].shape[0] if type_key in state_dict else 0,
        "n_grid": n_grid,
    }


class Kandinsky6(comfy.model_base.BaseModel):
    """ComfyUI model wrapping the Kandinsky 6 transformer and its joint video and audio latent."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        from .transformer import Kandinsky6Transformer

        super().__init__(model_config, model_type, device=device, unet_model=Kandinsky6Transformer)
        self.n_grid = int(model_config.unet_config.get("n_grid", 1))
        self.audio_scale = AUDIO_SCALE_DISTILLED if self.n_grid > 1 else AUDIO_SCALE

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        if self.n_grid > 1:
            transformer_options = dict(transformer_options)
            transformer_options[SHIFT_OPTION] = float(getattr(self.model_sampling, "shift", SHIFT))
        return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def encode_adm(self, **kwargs):
        return kwargs.get("pooled_output", None)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(text_norm(cross_attn))
        latent_shapes = kwargs.get("latent_shapes", None)
        if latent_shapes is not None:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)
        reference = kwargs.get(REFERENCE_KEY, None)
        if reference is not None:
            out[REFERENCE_KEY] = comfy.conds.CONDRegular(self.latent_format.process_in(reference))
        return out

    def memory_required(self, input_shape, cond_shapes={}):
        input_shape = list(input_shape)
        if len(input_shape) == 3 and input_shape[1] == 1:
            input_shape[2] = max(1, input_shape[2] // VIDEO_CHANNELS)
        return super().memory_required(input_shape, cond_shapes=cond_shapes)

    def _map_streams(self, latent, video_fn, audio_fn):
        if getattr(latent, "is_nested", False):
            streams = list(latent.unbind())
            streams[0] = video_fn(streams[0])
            if len(streams) > 1:
                streams[1] = audio_fn(streams[1])
            return comfy.nested_tensor.NestedTensor(streams)
        shapes = self.latent_shapes
        if shapes is not None and len(shapes) > 1:
            streams = comfy.utils.unpack_latents(latent, shapes)
            streams = [video_fn(streams[0]), audio_fn(streams[1]), *streams[2:]]
            return comfy.utils.pack_latents(streams)[0]
        return video_fn(latent)

    def process_latent_in(self, latent):
        return self._map_streams(latent, self.latent_format.process_in, lambda audio: audio * self.audio_scale)

    def process_latent_out(self, latent):
        return self._map_streams(latent, self.latent_format.process_out, lambda audio: audio / self.audio_scale)


class Kandinsky6Config(comfy.supported_models_base.BASE):
    """Supported-model entry matching the unet config :func:`detect` writes."""

    unet_config = {"image_model": IMAGE_MODEL}
    unet_extra_config = {}
    sampling_settings = {"shift": SHIFT}
    latent_format = comfy.latent_formats.HunyuanVideo
    memory_usage_factor = 1.25
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        self.memory_usage_factor = 1.25 * unet_config.get("model_dim", 4096) / 4096

    def get_model(self, state_dict, prefix="", device=None):
        return Kandinsky6(self, device=device)

    def clip_target(self, state_dict={}):
        return None


def register() -> None:
    """Teach ComfyUI's diffusion model loaders to recognise and build Kandinsky 6."""
    import comfy.model_detection
    import comfy.supported_models

    models = comfy.supported_models.models
    for index, entry in enumerate(models):
        if getattr(entry, "unet_config", {}).get("image_model") == IMAGE_MODEL:
            models[index] = Kandinsky6Config
            break
    else:
        models.append(Kandinsky6Config)

    current = comfy.model_detection.detect_unet_config
    if getattr(current, _DETECTOR_MARK, False):
        return

    @functools.wraps(current)
    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        config = detect(state_dict, key_prefix)
        if config is not None:
            return config
        return current(state_dict, key_prefix, *args, **kwargs)

    setattr(detect_unet_config, _DETECTOR_MARK, True)
    comfy.model_detection.detect_unet_config = detect_unet_config
