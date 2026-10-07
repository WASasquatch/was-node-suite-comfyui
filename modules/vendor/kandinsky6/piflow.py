"""Pure-torch math used by the base-model π-Flow sampler."""

from __future__ import annotations

import torch


class DXPolicy:
    """Network-free DX policy over one flow-matching segment."""

    def __init__(  # noqa: PLR0913
        self,
        denoising_output: torch.Tensor,
        x_t_src: torch.Tensor,
        sigma_t_src: torch.Tensor,
        segment_size: float | torch.Tensor = 1.0,
        shift: float = 1.0,
        mode: str = "grid",
        eps: float = 1e-4,
    ) -> None:
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.shift = shift
        self.eps = eps
        if mode not in ("grid", "polynomial"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.raw_t_src = self._unwarp_t(self.sigma_t_src)
        segment = segment_size
        if isinstance(segment, torch.Tensor) and segment.dim() < self.raw_t_src.dim():
            segment = segment.reshape(*segment.size(), *((self.raw_t_src.dim() - segment.dim()) * [1]))
        self.raw_t_dst = (self.raw_t_src - segment).clamp(min=0)
        self.segment_size = (self.raw_t_src - self.raw_t_dst).clamp(min=eps)
        self.denoising_output_x_0 = self._u_to_x_0(denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _interpolate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        if n < 2:  # noqa: PLR2004
            return x.squeeze(1)
        t = t.clamp(min=0, max=1) * (n - 1)
        t0 = t.floor().to(torch.long).clamp(min=0, max=n - 2)
        t1 = t0 + 1
        indices = torch.stack([t0, t1], dim=1)
        values = torch.gather(x, dim=1, index=indices.expand(-1, -1, *x.shape[2:]))
        return (t1 - t) * values[:, 0] + (t - t0) * values[:, 1]

    def _unwarp_t(self, sigma_t: torch.Tensor) -> torch.Tensor:
        return sigma_t / (self.shift + (1 - self.shift) * sigma_t)

    @staticmethod
    def _u_to_x_0(
        denoising_output: torch.Tensor,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        return x_t.unsqueeze(1) - sigma_t.unsqueeze(1) * denoising_output

    def pi(self, x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        sigma_t = sigma_t.reshape(*sigma_t.size(), *((self.ndim - sigma_t.dim()) * [1]))
        raw_t = self._unwarp_t(sigma_t)
        if self.mode == "grid":
            x_0 = self._interpolate(
                self.denoising_output_x_0,
                (raw_t - self.raw_t_dst) / self.segment_size,
            )
        else:
            p_order = self.denoising_output_x_0.size(1)
            diff_t = self.raw_t_src - raw_t
            basis = torch.stack([diff_t**i for i in range(p_order)], dim=1)
            x_0 = torch.sum(basis * self.denoising_output_x_0, dim=1)
        return (x_t - x_0) / sigma_t.clamp(min=self.eps)

    def copy(self) -> DXPolicy:
        new_policy = DXPolicy.__new__(DXPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.shift = self.shift
        new_policy.eps = self.eps
        new_policy.mode = self.mode
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.raw_t_src = self.raw_t_src
        new_policy.raw_t_dst = self.raw_t_dst
        new_policy.segment_size = self.segment_size
        new_policy.denoising_output_x_0 = self.denoising_output_x_0
        return new_policy

    def detach_(self) -> DXPolicy:
        self.denoising_output_x_0 = self.denoising_output_x_0.detach()
        return self

    def detach(self) -> DXPolicy:
        return self.copy().detach_()


class MultimodalDXPolicy:
    """Independent video and audio DX policies for the fused DiT."""

    def __init__(  # noqa: PLR0913
        self,
        denoising_output_v: torch.Tensor,
        x_t_src_v: torch.Tensor,
        denoising_output_a: torch.Tensor,
        x_t_src_a: torch.Tensor,
        sigma_t_v_src: torch.Tensor,
        segment_v_size: torch.Tensor,
        sigma_t_a_exp: torch.Tensor,
        segment_a_size: torch.Tensor,
        shift: float = 1.0,
        mode: str = "grid",
        eps: float = 1e-4,
    ) -> None:
        self.policy_V = DXPolicy(
            denoising_output_v,
            x_t_src_v,
            sigma_t_v_src,
            segment_v_size,
            shift,
            mode,
            eps,
        )
        self.policy_A = DXPolicy(
            denoising_output_a,
            x_t_src_a,
            sigma_t_a_exp,
            segment_a_size,
            shift,
            mode,
            eps,
        )
        self.shift = shift
        self.eps = eps

    def detach_(self) -> MultimodalDXPolicy:
        self.policy_V.detach_()
        self.policy_A.detach_()
        return self

    def detach(self) -> MultimodalDXPolicy:
        new_policy = MultimodalDXPolicy.__new__(MultimodalDXPolicy)
        new_policy.policy_V = self.policy_V.detach()
        new_policy.policy_A = self.policy_A.detach()
        new_policy.shift = self.shift
        new_policy.eps = self.eps
        return new_policy


def shift_timesteps(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Map raw flow-matching time to the shifted DiT time."""
    return shift * t / (1 + (shift - 1) * t)


def policy_rollout_fm(  # noqa: PLR0913
    x_t_start: torch.Tensor,
    sigma_t_start: torch.Tensor,
    raw_t_start: torch.Tensor,
    raw_t_end: torch.Tensor,
    total_substeps: int,
    policy: DXPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate ``policy.pi`` from ``raw_t_start`` to ``raw_t_end``."""
    num_batches = x_t_start.size(0)
    ndim = x_t_start.dim()
    shape = (num_batches, *((ndim - 1) * [1]))
    raw_t_start = raw_t_start.reshape(shape)
    raw_t_end = raw_t_end.reshape(shape)
    sigma_t = sigma_t_start.reshape(shape)

    delta_raw_t = raw_t_start - raw_t_end
    num_substeps = (delta_raw_t * total_substeps).round().to(torch.long).clamp(min=1)
    substep_size = delta_raw_t / num_substeps
    max_num_substeps = num_substeps.max()

    raw_t = raw_t_start
    x_t = x_t_start
    for substep_id in range(max_num_substeps.item()):
        velocity = policy.pi(x_t, sigma_t)
        raw_t_minus = (raw_t - substep_size).clamp(min=0)
        sigma_t_minus = shift_timesteps(raw_t_minus, policy.shift)
        x_t_minus = x_t + velocity * (sigma_t_minus - sigma_t)

        active_mask = num_substeps > substep_id
        x_t = torch.where(active_mask, x_t_minus, x_t)
        sigma_t = torch.where(active_mask, sigma_t_minus, sigma_t)
        raw_t = torch.where(active_mask, raw_t_minus, raw_t)

    return x_t, sigma_t, sigma_t.flatten() * 1_000
