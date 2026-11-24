"""SolarRenderer's two LoRAs on MiniMax H3, and the soundtrack held at encoded silence while it samples.

LoRA files carry diffusers PEFT names; :func:`patches` maps them onto core's fused projections.
"""

from __future__ import annotations

import re

from .. import log

__all__ = ["KEY", "SAMPLES_PER_LATENT", "apply", "grid", "hold_silence", "patches", "read_lora"]

logger = log.get_logger("model.h3_structure")

#: The key the pack's wrappers are filed under on a patched model.
KEY = "was_h3_structure"

#: Audio samples one H3 audio latent frame covers, at the audio VAE's 32 kHz.
SAMPLES_PER_LATENT = 800

#: Flow shifts MiniMax H3 samples video and audio at.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

_BLOCK = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


def read_lora(path: str) -> tuple[dict, float | None]:
    """A LoRA file's tensors and the alpha its header names.

    Args:
        path: Absolute path of a ``.safetensors`` LoRA.

    Returns:
        ``(state dict, alpha)``, the alpha None when the header names none.
    """
    import comfy.utils
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        meta = handle.metadata() or {}
    try:
        alpha = float(meta["alpha"]) if "alpha" in meta else None
    except ValueError:
        alpha = None
    return comfy.utils.load_torch_file(path, safe_load=True), alpha


def patches(model, lora: dict, alpha: float | None, name: str) -> dict:
    """A SolarRenderer LoRA as core's patches for one MiniMax H3 model.

    Args:
        model: The ``ModelPatcher``.
        lora: The LoRA's tensors, in diffusers PEFT names or core's own.
        alpha: Its alpha, None for a scale of 1.
        name: Its file name, for messages.

    Returns:
        ``{weight key: patch}``, as ``comfy.lora.load_lora`` returns them.

    Raises:
        ValueError: A module has no counterpart in MiniMax H3.
    """
    import comfy.lora
    import torch

    if not any(key.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.")) for key in lora):
        return comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model.model, {}), log_missing=False)
    diffusion = model.get_model_object("diffusion_model")
    inner = int(diffusion.blocks[0].attn.out_proj.weight.shape[1])
    ffn = int(diffusion.blocks[0].mlp.fc2.weight.shape[1])
    entries, key_map = {}, {}

    def add(label: str, down, up, target):
        entries[f"{label}.lora_A.weight"] = down
        entries[f"{label}.lora_B.weight"] = up
        if alpha is not None:
            entries[f"{label}.alpha"] = torch.tensor(float(alpha))
        key_map[label] = target

    for key in lora:
        if not key.endswith(".lora_A.weight"):
            continue
        base = key[: -len(".lora_A.weight")]
        match = _BLOCK.match(base)
        if match is None:
            raise ValueError(f"{name} adapts {base}, which is not a MiniMax H3 block or text refiner module. "
                             f"Choose a SolarRenderer LoRA.")
        prefix, index, module = match.groups()
        root = ("diffusion_model.blocks." if prefix == "transformer_blocks" else
                "diffusion_model.token_refiner.blocks.") + index
        down, up = lora[key], lora[base + ".lora_B.weight"]
        if module in ("attn.to_q", "attn.to_k", "attn.to_v"):
            third = "qkv".index(module[-1])
            add(base, down, up, (f"{root}.attn.qkv_proj.weight", (0, third * inner, inner)))
        elif module == "ff.net.0.proj":
            # Diffusers holds the feed-forward input as [value; gate], core as [gate; value].
            half = up.shape[0] // 2
            add(f"{base}.value", down, up[:half], (f"{root}.mlp.fc1.weight", (0, ffn, ffn)))
            add(f"{base}.gate", down, up[half:], (f"{root}.mlp.fc1.weight", (0, 0, ffn)))
        elif module == "attn.to_out.0":
            add(base, down, up, f"{root}.attn.out_proj.weight")
        elif module == "ff.net.2":
            add(base, down, up, f"{root}.mlp.fc2.weight")
        else:
            raise ValueError(f"{name} adapts {module}, which MiniMax H3 has no counterpart for. Choose a "
                             f"SolarRenderer LoRA.")
    return comfy.lora.load_lora(entries, key_map, log_missing=False)


def apply(model, loras: list[tuple[str, dict, float]]):
    """A clone of a model carrying LoRA patches.

    Args:
        model: The ``ModelPatcher``.
        loras: ``(name, patches, strength)`` per LoRA, in the order they are added.

    Returns:
        ``(clone, weights)``: the patched clone, and how many weights each LoRA reached.

    Raises:
        ValueError: A LoRA's patches find no weight in the model.
    """
    clone = model.clone()
    reached = []
    for name, found, strength in loras:
        added = clone.add_patches(found, float(strength))
        missed = set(found) - set(added)
        if missed:
            raise ValueError(f"{name}: {len(missed)} of its {len(found)} weights are not in this model, as "
                             f"{sorted(missed)[0]}. Load MiniMax H3 Ref2VA, as "
                             f"minimax_h3_ref2va_pruned_int8_convrot.")
        reached.append(len(found))
    return clone, reached


def grid(steps: int, shift: float = VIDEO_SHIFT) -> list[float]:
    """The schedule the four-step LoRA was trained on, uniform in time under ``shift``, ending at 0."""
    return [shift * (k / steps) / (1.0 + (shift - 1.0) * (k / steps)) for k in range(steps, -1, -1)]


def _off_grid(schedule) -> bool:
    """Whether a sampler schedule leaves the shifted uniform grid of its own length."""
    values = [float(v) for v in schedule]
    if len(values) < 2:
        return False
    return any(abs(a - b) > 1e-3 for a, b in zip(values, grid(len(values) - 1)))


class _Silence:
    """The soundtrack H3 is held at: the audio VAE's encoded silence, noised to each step's audio time."""

    def __init__(self, audio_vae):
        self.audio_vae = audio_vae
        self.clean = {}
        self.noise = {}
        self.current = None
        self.warned = set()

    def prepare(self, length: int, seed: int) -> None:
        """Encode silence ``length`` audio latents long and draw its noise from ``seed``, once each."""
        import torch

        if length not in self.clean:
            with torch.no_grad():
                waveform = torch.zeros(1, length * SAMPLES_PER_LATENT, 2)
                encoded = self.audio_vae.encode(waveform).float().cpu()
            if encoded.shape[-1] != length:
                raise ValueError(
                    f"The audio VAE encoded {encoded.shape[-1]} latent frames of silence for a {length}-frame "
                    f"soundtrack. Wire Load VAE set to minimax_h3_audio_vae_fp32 into audio_vae.")
            self.clean[length] = encoded.to(torch.bfloat16).float()
        key = (length, int(seed or 0))
        if key not in self.noise:
            generator = torch.Generator("cpu").manual_seed(key[1])
            self.noise[key] = torch.randn(self.clean[length].shape, generator=generator, dtype=torch.float32)
        self.current = (self.clean[length], self.noise[key])

    def warn(self, kind: str, message: str, *values) -> None:
        """Log one warning of a kind once per run."""
        if kind not in self.warned:
            self.warned.add(kind)
            logger.warning(message, *values)


def hold_silence(model, audio_vae):
    """A clone of a MiniMax H3 model whose soundtrack is held at encoded silence on every evaluation.

    Args:
        model: The ``ModelPatcher``.
        audio_vae: MiniMax H3's audio VAE.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension
    from comfy.ldm.minimax.model import time_shift_sigma

    silence = _Silence(audio_vae)
    patched = model.clone()

    def outer(executor, noise, latent_image, sampler, sigmas, *args, **kwargs):
        shapes = kwargs.get("latent_shapes")
        seed = args[3] if len(args) > 3 else kwargs.get("seed", 0)
        silence.warned.clear()
        silence.current = None
        if shapes is not None and len(shapes) > 1:
            silence.prepare(int(shapes[1][-1]), seed)
        if _off_grid(sigmas):
            silence.warn("grid", "H3 Structure Renderer: the sampler's sigmas %s are off the grid the four-step "
                                 "LoRA was trained on, %s. Use KSampler with scheduler simple.",
                         [round(float(s), 4) for s in sigmas], [round(s, 4) for s in grid(len(sigmas) - 1)])
        name = getattr(getattr(sampler, "sampler_function", None), "__name__", "")
        if name and name != "sample_lcm":
            silence.warn("sampler", "H3 Structure Renderer: the sampler is %s; the four-step LoRA was trained "
                                    "on lcm, which takes the clean picture at each step and noises it afresh.",
                         name.removeprefix("sample_"))
        return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)

    def hold(executor, x, timestep, context, transformer_options={}, **kwargs):
        if silence.current is None:
            silence.warn("unprepared", "H3 Structure Renderer: the latent carries no sound stream, so the "
                                       "soundtrack is not held at silence.")
            return executor(x, timestep, context, transformer_options, **kwargs)
        video, audio = x[0], x[1]
        clean, noise = silence.current
        if audio.shape[-1] != clean.shape[-1]:
            silence.warn("length", "H3 Structure Renderer: the model was handed %d audio frames of a %d-frame "
                                   "soundtrack, so this call keeps its own sound.", audio.shape[-1], clean.shape[-1])
            return executor(x, timestep, context, transformer_options, **kwargs)
        owner = getattr(executor, "class_obj", None)
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video",
                                                getattr(owner, "sigma_shift_video", VIDEO_SHIFT)))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio",
                                                getattr(owner, "sigma_shift_audio", AUDIO_SHIFT)))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = float(time_shift_sigma(sigma_v, shift_v, shift_a))
        clean = clean.to(audio.device)
        held = ((1.0 - sigma_a) * clean + sigma_a * noise.to(audio.device)).to(audio.dtype).expand_as(audio)
        return executor([video, held], timestep, context, transformer_options, **kwargs)

    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, outer)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, hold)
    return patched
