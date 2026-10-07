"""The Kandinsky 6 audio autoencoder behind ComfyUI's audio VAE calls.

Latents are ``(B, 40, L)``; waveforms go out as ``(B, N, 1)`` at 44.1 kHz, ``N = 1024 * L``.
"""

from __future__ import annotations

import torch

from ... import log

logger = log.get_logger("model.kandinsky6.audio_vae")

SAMPLE_RATE = 44100
FRAME_SAMPLES = 1024
LATENT_CHANNELS = 40


def offered() -> tuple[list[str], str | None]:
    """The files in ComfyUI's vae folders, and the one most likely to be the Kandinsky 6 audio VAE."""
    import folder_paths

    files = list(folder_paths.get_filename_list("vae"))
    likely = [name for name in files if "kandinsky" in name.lower() and "audio" in name.lower()]
    likely = likely or [name for name in files if "audio_vae" in name.lower().replace("-", "_")]
    return files, (likely[0] if likely else None)


def load(name: str) -> "Kandinsky6AudioVAE":
    """Build the audio VAE from a file in ComfyUI's vae folders.

    Raises:
        FileNotFoundError: No vae folder holds ``name``.
        ValueError: The file is not the Kandinsky 6 audio VAE.
    """
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("vae", name)
    return Kandinsky6AudioVAE(comfy.utils.load_torch_file(path, safe_load=True))


def is_audio_vae(state_dict) -> bool:
    """Whether a state dict holds the Kandinsky 6 audio autoencoder and its vocoder."""
    return (
        "vae.decoder.conv_in.weight" in state_dict
        and "vocoder.conv_pre.weight" in state_dict
        and tuple(state_dict["vae.decoder.conv_in.weight"].shape[:2]) == (2048, LATENT_CHANNELS)
    )


class Kandinsky6AudioVAE:
    """Decodes and encodes Kandinsky 6 audio latents, placed on the GPU by ComfyUI's model manager."""

    def __init__(self, state_dict):
        """Build the autoencoder from the released audio VAE file.

        Args:
            state_dict: Tensors of the ``audio_vae`` Diffusers file, ``vae.``, ``vocoder.`` and
                ``mel_converter.`` keys.

        Raises:
            ValueError: The tensors are not the Kandinsky 6 audio autoencoder.
        """
        import comfy.model_management as mm
        import comfy.model_patcher

        from ...vendor.kandinsky6.audio import AudioAutoencoder

        if not is_audio_vae(state_dict):
            raise ValueError(
                "This file is not the Kandinsky 6 audio VAE. Pick the audio_vae file from a "
                "kandinskylab/Kandinsky-6.0 repository, the one holding vae., vocoder. and "
                "mel_converter. weights."
            )
        model = AudioAutoencoder()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise ValueError(
                f"The Kandinsky 6 audio VAE file is missing {len(missing)} tensors, starting with "
                f"{missing[0]}. Download the audio_vae file again."
            )
        if unexpected:
            logger.debug("ignored %d tensors the audio VAE does not use", len(unexpected))
        model.eval().requires_grad_(False)

        self.first_stage_model = model
        self.device = mm.vae_device()
        self.offload_device = mm.vae_offload_device()
        patcher_class = getattr(comfy.model_patcher, "CoreModelPatcher", comfy.model_patcher.ModelPatcher)
        self.patcher = patcher_class(model, load_device=self.device, offload_device=self.offload_device)
        self.audio_sample_rate = SAMPLE_RATE
        self.audio_sample_rate_output = SAMPLE_RATE
        self.latent_channels = LATENT_CHANNELS
        self.latent_dim = 1
        self.output_channels = 1
        self.upscale_ratio = FRAME_SAMPLES
        self.downscale_ratio = FRAME_SAMPLES

    def throw_exception_if_invalid(self):
        return None

    def get_sd(self):
        return self.first_stage_model.state_dict()

    def _load(self, samples: int):
        import comfy.model_management as mm

        mm.load_models_gpu([self.patcher], memory_required=512 * 1024 * 1024 + samples * 4096,
                           force_full_load=True)
        return self.patcher.load_device

    def decode(self, samples: torch.Tensor) -> torch.Tensor:
        """Latents ``(B, 40, L)`` to a waveform ``(B, 1024 * L, 1)`` in ``[-1, 1]``.

        Raises:
            ValueError: The latent is not Kandinsky 6 audio.
        """
        import comfy.model_management as mm

        if getattr(samples, "is_nested", False):
            samples = samples.unbind()[-1]
        if samples.ndim != 3 or samples.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                f"The Kandinsky 6 audio VAE decodes latents shaped (batch, {LATENT_CHANNELS}, frames), "
                f"not {tuple(samples.shape)}. Connect the latent a Kandinsky 6 sampler produced."
            )
        device = self._load(samples.shape[0] * samples.shape[-1] * FRAME_SAMPLES)
        with torch.inference_mode():
            waveform = self.first_stage_model.decode(samples.to(device=device, dtype=torch.float32))
        return waveform.movedim(1, -1).to(device=mm.intermediate_device(), dtype=torch.float32)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """A waveform ``(B, N, C)`` at 44.1 kHz to latents ``(B, 40, ceil(N / 1024))``, channels mixed to mono."""
        import comfy.model_management as mm

        mono = waveform.float().mean(dim=-1)
        pad = -mono.shape[-1] % FRAME_SAMPLES
        if pad:
            mono = torch.nn.functional.pad(mono, (0, pad))
        device = self._load(mono.numel())
        with torch.inference_mode():
            latent = self.first_stage_model.encode(mono.to(device))
        return latent.to(device=mm.intermediate_device(), dtype=torch.float32)
