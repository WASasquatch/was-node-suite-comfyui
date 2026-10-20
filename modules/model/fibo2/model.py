"""Fibo 2.0 checkpoint detection, key layout, model class and registration.

The native layout is Comfy-Org's: fused ``qkv``, ``to_kv`` and ``w13``, the feed-forward width
padded to a multiple of 256. Diffusers and unfused files are converted to it on loading.
"""

from __future__ import annotations

import functools

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base

from .dit import BUNDLE_WIDTH

#: The ``image_model`` detection writes, and the one ComfyUI's own Fibo 2.0 support writes.
IMAGE_MODEL = "was_fibo2"
CORE_IMAGE_MODEL = "fibo2"

#: The flow shift a one megapixel image samples at.
SHIFT = 1.15

#: Conditioning keys the text-only work travels under once it has run.
GIST_KEY = "fibo2_gist"
TEXT_KV_KEY = "fibo2_text_kv"

#: The multiple the feed-forward width is padded to.
FFN_MULTIPLE = 256

_SHARED = ("perceiver.latent_queries", "final_layer.linear.weight", "x_embedder.weight")
_DETECTOR_MARK = "_was_fibo2_detector"
_LORA_MARK = "_was_fibo2_lora_keys"


def _layout(state_dict, prefix: str = "") -> str | None:
    """``"native"``, ``"unfused"``, ``"diffusers"``, or ``None`` for weights that are not Fibo 2.0's."""
    if any(prefix + key not in state_dict for key in _SHARED):
        return None
    if prefix + "layers.0.attention.qkv.weight" in state_dict:
        return "native" if prefix + "layers.0.feed_forward.w13.weight" in state_dict else "unfused"
    if prefix + "layers.0.attention.to_q.weight" in state_dict:
        return "diffusers"
    return None


def _padded(hidden: int) -> int:
    return -(-hidden // FFN_MULTIPLE) * FFN_MULTIPLE


def raw_ffn_dim(dim: int) -> int:
    """The feed-forward width a Fibo 2.0 transformer of width ``dim`` was trained with."""
    return (dim * 8 // 3) // 64 * 64


def detect(state_dict, key_prefix: str = "") -> dict | None:
    """Read a Fibo 2.0 transformer's configuration off its weights, in any layout.

    Args:
        state_dict: The checkpoint's tensors.
        key_prefix: Prefix the transformer's keys carry.

    Returns:
        The unet config, or ``None`` when the weights are not a Fibo 2.0 transformer.
    """
    layout = _layout(state_dict, key_prefix)
    if layout is None:
        return None

    def count(pattern):
        index = 0
        while key_prefix + pattern.format(index) in state_dict:
            index += 1
        return index

    max_gist, dim = state_dict[key_prefix + "perceiver.latent_queries"].shape
    norm = "norm_q" if layout == "diffusers" else "q_norm"
    head_dim = state_dict[f"{key_prefix}layers.0.attention.{norm}.weight"].shape[0]
    if layout == "native":
        ffn_dim = state_dict[key_prefix + "layers.0.feed_forward.w13.weight"].shape[0] // 2
    else:
        ffn_dim = _padded(state_dict[key_prefix + "layers.0.feed_forward.w1.weight"].shape[0])
    n_layers = count("layers.{}.attention_norm1.weight")
    return {
        "image_model": IMAGE_MODEL,
        "in_channels": int(state_dict[key_prefix + "final_layer.linear.weight"].shape[0]),
        "dim": int(dim),
        "n_heads": int(dim // head_dim),
        "n_layers": n_layers,
        "n_refiner_layers": count("noise_refiner.{}.attention_norm1.weight"),
        "injection_layers": [i for i in range(n_layers) if f"{key_prefix}layers.{i}.cross_attn_gate" in state_dict],
        "cap_feat_dim": BUNDLE_WIDTH,
        "perceiver_layers": count("perceiver.layers.{}.cross_attn_norm.weight"),
        "max_gist_tokens": int(max_gist),
        "ffn_dim": int(ffn_dim),
    }


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat([tensor, tensor.new_zeros(rows - tensor.shape[0], *tensor.shape[1:])]) if tensor.shape[0] < rows else tensor


def _ffn_prefixes(state_dict) -> list[str]:
    return [key[: -len(".w1.weight")] for key in state_dict if key.endswith((".feed_forward.w1.weight", ".ffn.w1.weight"))]


def to_native(state_dict: dict) -> dict:
    """A Diffusers or unfused Fibo 2.0 transformer in the native layout; native weights come back as they are."""
    layout = _layout(state_dict)
    if layout not in ("diffusers", "unfused"):
        return state_dict
    out = dict(state_dict)
    if layout == "diffusers":
        for key in [key for key in state_dict if key.endswith(".to_q.weight")]:
            prefix = key[: -len(".to_q.weight")]
            q, k, v = (out.pop(f"{prefix}.to_{name}.weight") for name in "qkv")
            output = out.pop(f"{prefix}.to_out.0.weight")
            if prefix.endswith(".attention"):
                out[f"{prefix}.qkv.weight"] = torch.cat([q, k, v])
                out[f"{prefix}.out.weight"] = output
                out[f"{prefix}.q_norm.weight"] = out.pop(f"{prefix}.norm_q.weight")
                out[f"{prefix}.k_norm.weight"] = out.pop(f"{prefix}.norm_k.weight")
            else:
                out[f"{prefix}.to_q.weight"] = q
                out[f"{prefix}.to_kv.weight"] = torch.cat([k, v])
                out[f"{prefix}.to_out.weight"] = output
    for prefix in _ffn_prefixes(state_dict):
        w1, w2, w3 = (out.pop(f"{prefix}.{name}.weight") for name in ("w1", "w2", "w3"))
        hidden = _padded(w1.shape[0])
        out[f"{prefix}.w13.weight"] = torch.cat([_pad_rows(w1, hidden), _pad_rows(w3, hidden)])
        out[f"{prefix}.w2.weight"] = _pad_rows(w2.t(), hidden).t().contiguous()
    return out


def diffusers_lora_keys(unet_config: dict, output_prefix: str = "diffusion_model.") -> dict:
    """Diffusers layer names mapped to the native weights, as slices where layers are fused or padded.

    Args:
        unet_config: The model's unet config, this pack's or ComfyUI's own.
        output_prefix: Prefix the model's weights carry.

    Returns:
        Layer name -> weight key, or ``(weight key, (dim, offset, size))`` for part of one.
    """
    dim = unet_config["dim"]
    ffn_dim = unet_config["ffn_dim"]
    fused = unet_config.get("fused_ffn", True)
    raw = raw_ffn_dim(dim)
    blocks = [f"layers.{i}" for i in range(unet_config["n_layers"])]
    blocks += [f"noise_refiner.{i}" for i in range(unet_config["n_refiner_layers"])]
    perceiver = [f"perceiver.layers.{i}" for i in range(unet_config["perceiver_layers"])]
    texts = [f"layers.{i}.cross_attention" for i in unet_config["injection_layers"]]
    texts += [f"{layer}.{kind}" for layer in perceiver for kind in ("cross_attention", "self_attention")]
    ffns = [f"{block}.feed_forward" for block in blocks] + [f"{layer}.ffn" for layer in perceiver]

    out = {}
    for block in blocks:
        fused_qkv = f"{output_prefix}{block}.attention.qkv.weight"
        for index, name in enumerate("qkv"):
            out[f"{block}.attention.to_{name}"] = (fused_qkv, (0, index * dim, dim))
        out[f"{block}.attention.to_out.0"] = f"{output_prefix}{block}.attention.out.weight"
    for prefix in texts:
        out[f"{prefix}.to_q"] = f"{output_prefix}{prefix}.to_q.weight"
        out[f"{prefix}.to_k"] = (f"{output_prefix}{prefix}.to_kv.weight", (0, 0, dim))
        out[f"{prefix}.to_v"] = (f"{output_prefix}{prefix}.to_kv.weight", (0, dim, dim))
        out[f"{prefix}.to_out.0"] = f"{output_prefix}{prefix}.to_out.weight"
    if fused:
        for prefix in ffns:
            out[f"{prefix}.w1"] = (f"{output_prefix}{prefix}.w13.weight", (0, 0, raw))
            out[f"{prefix}.w3"] = (f"{output_prefix}{prefix}.w13.weight", (0, ffn_dim, raw))
            out[f"{prefix}.w2"] = (f"{output_prefix}{prefix}.w2.weight", (1, 0, raw))
    return out


class Fibo2Model(comfy.model_base.BaseModel):
    """ComfyUI model wrapping the Fibo 2.0 transformer on the FLUX.2 latent."""

    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLUX, device=None):
        from .dit import Fibo2

        super().__init__(model_config, model_type, device=device, unet_model=Fibo2)
        self.memory_usage_factor_conds = ("ref_latents",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        references = kwargs.get("reference_latents", None)
        if references is not None:
            out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(latent) for latent in references])
        text = kwargs.get("cross_attn", None)
        if text is None:
            return out
        # Cond and uncond of different lengths run apart: the gist count follows the text's length.
        out["c_crossattn"] = comfy.conds.CONDRegular(text)
        if torch.is_inference_mode_enabled() and kwargs.get("hooks") is None:
            device, dtype = kwargs["device"], self.get_dtype_inference()
            gist, text_kv = self.diffusion_model.preprocess_conds(text.to(device=device, dtype=dtype))
            del out["c_crossattn"]
            out[GIST_KEY] = comfy.conds.CONDRegular(gist)
            out[TEXT_KV_KEY] = comfy.conds.CONDRegular(text_kv)
        return out

    def extra_conds_shapes(self, **kwargs):
        out = {}
        references = kwargs.get("reference_latents", None)
        if references is not None:
            out["ref_latents"] = [1, 16, sum(int(latent.shape[-1] * latent.shape[-2]) for latent in references)]
        return out


class Fibo2Config(comfy.supported_models_base.BASE):
    """Supported-model entry matching the unet config :func:`detect` writes."""

    unet_config = {"image_model": IMAGE_MODEL}
    unet_extra_config = {}
    sampling_settings = {"shift": SHIFT}
    latent_format = comfy.latent_formats.Flux2
    memory_usage_factor = 17.0
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def get_model(self, state_dict, prefix="", device=None):
        return Fibo2Model(self, device=device)

    def process_unet_state_dict(self, state_dict):
        return to_native(state_dict)

    def clip_target(self, state_dict={}):
        return None


def core_supports() -> bool:
    """Whether the running ComfyUI loads Fibo 2.0 itself."""
    import comfy.supported_models

    return any(getattr(entry, "unet_config", {}).get("image_model") == CORE_IMAGE_MODEL
               for entry in comfy.supported_models.models)


def register() -> None:
    """Teach ComfyUI's loaders Fibo 2.0 where it lacks it, and its LoRA loader Diffusers layer names."""
    import comfy.lora
    import comfy.model_detection
    import comfy.supported_models

    keys = comfy.lora.model_lora_keys_unet
    if not getattr(keys, _LORA_MARK, False):
        @functools.wraps(keys)
        def model_lora_keys_unet(model, key_map={}):
            key_map = keys(model, key_map)
            config = getattr(getattr(model, "model_config", None), "unet_config", {}) or {}
            if config.get("image_model") in (IMAGE_MODEL, CORE_IMAGE_MODEL):
                for name, target in diffusers_lora_keys(config).items():
                    for alias in (f"transformer.{name}", f"diffusion_model.{name}", name,
                                  f"lycoris_{name.replace('.', '_')}"):
                        key_map[alias] = target
                for key in model.state_dict():
                    if key.startswith("diffusion_model.") and key.endswith(".weight"):
                        key_map.setdefault(f"transformer.{key[len('diffusion_model.'):-len('.weight')]}", key)
            return key_map

        setattr(model_lora_keys_unet, _LORA_MARK, True)
        comfy.lora.model_lora_keys_unet = model_lora_keys_unet

    if core_supports():
        return

    models = comfy.supported_models.models
    for index, entry in enumerate(models):
        if getattr(entry, "unet_config", {}).get("image_model") == IMAGE_MODEL:
            models[index] = Fibo2Config
            break
    else:
        models.append(Fibo2Config)

    current = comfy.model_detection.detect_unet_config
    if not getattr(current, _DETECTOR_MARK, False):
        @functools.wraps(current)
        def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
            config = detect(state_dict, key_prefix)
            if config is not None:
                return config
            return current(state_dict, key_prefix, *args, **kwargs)

        setattr(detect_unet_config, _DETECTOR_MARK, True)
        comfy.model_detection.detect_unet_config = detect_unet_config
