"""The Kandinsky 6 transformer on ComfyUI's joint latent.

A distilled checkpoint answers each step with the velocity that carries the latent to the next
sigma along its PiFlow policy.
"""

from __future__ import annotations

import torch

import comfy.patcher_extension
import comfy.utils

from ...vendor.kandinsky6 import dit, piflow
from .model import REFERENCE_KEY, SHIFT, SHIFT_OPTION

PIFLOW_EPS = 1e-6
PIFLOW_SUBSTEPS = 128


def next_sigma(sample_sigmas, sigma: float) -> float | None:
    """The sigma after ``sigma`` in the running schedule, or ``None`` when it is not on it."""
    if sample_sigmas is None or sample_sigmas.numel() < 2:
        return None
    sigmas = sample_sigmas.flatten().float()
    distance = (sigmas[:-1] - sigma).abs()
    index = int(torch.argmin(distance))
    if float(distance[index]) > 1e-4 * max(1.0, sigma):
        return None
    following = float(sigmas[index + 1])
    return following if following < sigma else None


def unwarp(sigma, shift: float):
    """Raw flow time for a shifted sigma."""
    return sigma / (shift + (1.0 - shift) * sigma)


class Kandinsky6Transformer(dit.DiffusionTransformer3D):
    """Kandinsky 6 DiT reading ComfyUI's video and audio latents channels-first."""

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
        ).execute(x, timestep, context, y, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, y, transformer_options, **kwargs):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "Kandinsky 6 samples video and sound together and needs a joint latent from "
                "Empty Kandinsky 6 Latent or Kandinsky 6 Image To Video."
            )
        video, audio = x
        if y is None:
            raise ValueError("Kandinsky 6 needs its prompt encoded by Kandinsky 6 Text Encode.")
        batch, _, frames, height, width = video.shape
        if height % 2 or width % 2:
            raise ValueError(
                f"Kandinsky 6 needs width and height in multiples of 16 pixels; this latent is "
                f"{width * 8}x{height * 8}."
            )

        reference = kwargs.get(REFERENCE_KEY, None)
        if reference is not None:
            reference = comfy.utils.repeat_to_batch_size(reference[:, :, :1], batch).to(video)
            if reference.shape[-2:] != video.shape[-2:]:
                raise ValueError(
                    f"The Kandinsky 6 start image encodes to {reference.shape[-1] * 8}x"
                    f"{reference.shape[-2] * 8} but the latent is {width * 8}x{height * 8}. Give "
                    "Kandinsky 6 Image To Video the same width and height as the clip."
                )
            clip = torch.cat([video, reference], dim=2)
        else:
            clip = video
        mask = torch.zeros_like(clip[:, :1])
        type_ids = None
        if reference is not None:
            mask[:, :, -1] = 1
            if self.visual_token_type_num_embeddings > 1:
                type_ids = torch.zeros(clip.shape[2], dtype=torch.long, device=video.device)
                type_ids[-1] = 1
        model_input = torch.cat([clip, torch.zeros_like(clip), mask], dim=1).movedim(1, -1)

        out_video, out_audio = super().forward(
            model_input,
            audio.movedim(1, -1),
            context,
            y,
            timestep,
            visual_token_type_ids=type_ids,
            reference_tail=reference is not None,
            transformer_options=transformer_options,
        )
        out_video = out_video.movedim(-1, 1)[:, :, :frames]
        out_audio = out_audio.movedim(-1, 1)
        if self.n_grid > 1:
            out_video, out_audio = self._piflow(video, audio, out_video, out_audio, timestep, transformer_options)
        return [out_video, out_audio]

    def _piflow(self, video, audio, out_video, out_audio, timestep, transformer_options):
        """Velocities that carry each stream to the next sigma along the predicted policy."""
        grid_video = out_video.float().unflatten(1, (self.n_grid, -1))
        grid_audio = out_audio.float().unflatten(1, (self.n_grid, -1))
        sigma = timestep.float().to(video.device) / 1000.0
        shift = float(transformer_options.get(SHIFT_OPTION, SHIFT))
        target = next_sigma(transformer_options.get("sample_sigmas", None), float(sigma[0]))
        if target is None:
            return grid_video[:, -1], grid_audio[:, -1]

        raw_src = unwarp(sigma, shift)
        raw_dst = unwarp(torch.full_like(sigma, target), shift).clamp(min=PIFLOW_EPS)
        step = target - sigma
        velocities = []
        for state, grid in ((video, grid_video), (audio, grid_audio)):
            state = state.float()
            policy = piflow.DXPolicy(grid, state, sigma, raw_src - raw_dst, shift=shift, eps=PIFLOW_EPS)
            end, _, _ = piflow.policy_rollout_fm(state, sigma, raw_src, raw_dst, PIFLOW_SUBSTEPS, policy)
            velocities.append((end - state) / step.reshape(-1, *([1] * (state.ndim - 1))))
        return velocities
