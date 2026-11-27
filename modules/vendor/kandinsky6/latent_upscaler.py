"""Latent upscalers of Kandinsky 6 video super-resolution, on ComfyUI's layer operations.

Latents are K-VAE latents ``(B, 64, T, H, W)`` scaled by the K-VAE scaling factor; the output is
``(B, 64, T, H * scale, W * scale)``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

import comfy.model_management
import comfy.ops

ops = comfy.ops.disable_weight_init


class Conv3d(ops.Conv3d):
    """``Conv3d`` replicating the edge frame along time and zero-padding height and width."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=(0, kernel_size // 2, kernel_size // 2))
        self.temporal_pad = kernel_size // 2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = functional.pad(hidden_states, (0, 0, 0, 0, self.temporal_pad, self.temporal_pad), mode="replicate")
        return super().forward(hidden_states)


class RMSNorm(nn.Module):
    """Channel-first RMS normalisation with a learnable gain, in float32."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.empty(num_channels, 1, 1, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = functional.normalize(hidden_states.float(), dim=1).to(hidden_states.dtype)
        gamma = comfy.model_management.cast_to(self.gamma, dtype=normalized.dtype, device=normalized.device)
        return normalized * self.scale * gamma


class ModulatedNorm(nn.Module):
    """RMS norm followed by a FiLM modulation from the input latent ``zq``."""

    def __init__(self, num_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(num_channels)
        self.conv_y = ops.Conv3d(zq_channels, num_channels, kernel_size=1)
        self.conv_b = ops.Conv3d(zq_channels, num_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if zq.shape[2:] != hidden_states.shape[2:]:
            zq = functional.interpolate(zq, size=hidden_states.shape[2:], mode="nearest")
        return self.norm(hidden_states) * self.conv_y(zq) + self.conv_b(zq)


class ResidualBlock(nn.Module):
    """``norm, SiLU, conv3, norm, SiLU, conv3`` with a 1x1 shortcut when the width changes."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm1 = ModulatedNorm(in_channels, zq_channels)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = ModulatedNorm(out_channels, zq_channels)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3)
        self.shortcut = nn.Identity() if in_channels == out_channels else ops.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(functional.silu(self.norm1(hidden_states, zq)))
        residual = self.conv2(functional.silu(self.norm2(residual, zq)))
        return self.shortcut(hidden_states) + residual


class Upsample(nn.Module):
    """Spatial 2x: nearest upsample plus a per-frame convolutional residual, mixed pointwise."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_conv = ops.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.linear = ops.Conv3d(channels, channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = functional.interpolate(hidden_states, scale_factor=(1, 2, 2), mode="nearest")
        return self.linear(hidden_states + self.spatial_conv(hidden_states))


class OutputHead(nn.Module):
    """``modulated norm, SiLU, conv3`` back to the latent channels."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm = ModulatedNorm(in_channels, zq_channels)
        self.activation = nn.SiLU()
        self.conv = Conv3d(in_channels, out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        return self.conv(self.activation(self.norm(hidden_states, zq)))


def _blocks(widths, zq_channels):
    return nn.ModuleList([ResidualBlock(widths[i], widths[i + 1], zq_channels) for i in range(len(widths) - 1)])


class X2Branch(nn.Module):
    """The x2 upscaler: adapter blocks and a finisher on the input grid, then one 2x stage."""

    def __init__(self, in_channels, stage_channels, num_adapter_blocks, num_mid_blocks, num_post_blocks) -> None:
        super().__init__()
        width_1, width_2, width_3 = stage_channels
        self.adapter = _blocks([width_1] * (num_adapter_blocks + 1), in_channels)
        self.finisher = nn.Module()
        self.finisher.spatial_conv = ops.Conv3d(width_1, width_1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.finisher.linear = ops.Conv3d(width_1, width_1, kernel_size=1)
        self.mid_blocks = _blocks([width_1] + [width_2] * num_mid_blocks, in_channels)
        self.upsample = Upsample(width_2)
        self.blocks = _blocks([width_2] + [width_3] * num_post_blocks, in_channels)
        self.output_proj = OutputHead(width_3, in_channels, in_channels)

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        for block in self.adapter:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.finisher.linear(hidden_states + self.finisher.spatial_conv(hidden_states))
        for block in self.mid_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, zq)
        return self.output_proj(hidden_states, zq)


class LatentUpscaler(nn.Module):
    """One latent upscaler: the x4 two-stage cascade, or the x2 entry and its own branch."""

    def __init__(
        self,
        scale: int,
        in_channels: int = 64,
        stage_channels: tuple = (2048, 1024, 512),
        num_pre_blocks: int = 5,
        num_mid_blocks: int = 3,
        num_post_blocks: int = 3,
        num_x2_adapter_blocks: int = 2,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        self.scale = scale
        width_1, width_2, width_3 = stage_channels

        if scale == 2:
            self.mid_input_proj = nn.Sequential(Conv3d(in_channels, width_1, kernel_size=3))
            self.x2_branch = X2Branch(in_channels, stage_channels, num_x2_adapter_blocks, num_mid_blocks, num_post_blocks)
            return

        self.input_proj = nn.Sequential(Conv3d(in_channels, width_1, kernel_size=3))
        self.pre_blocks = _blocks([width_1] * (num_pre_blocks + 1), in_channels)
        self.upsample_1 = Upsample(width_1)
        self.mid_blocks = _blocks([width_1] + [width_2] * num_mid_blocks, in_channels)
        self.upsample_2 = Upsample(width_2)
        self.post_blocks = _blocks([width_2] + [width_3] * num_post_blocks, in_channels)
        self.output_proj = OutputHead(width_3, in_channels, in_channels)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        zq = latents
        if self.scale == 2:
            return self.x2_branch(self.mid_input_proj(latents), zq)

        hidden_states = self.input_proj(latents)
        for block in self.pre_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample_1(hidden_states)
        for block in self.mid_blocks:
            hidden_states = block(hidden_states, zq)
        hidden_states = self.upsample_2(hidden_states)
        for block in self.post_blocks:
            hidden_states = block(hidden_states, zq)
        return self.output_proj(hidden_states, zq)
