"""The MiniMax H3 learned latent upscaler: a 3D convolutional resizer for 24-channel video latents.

Inputs are H3 video latents normalised by :data:`LATENTS_MEAN` and :data:`LATENTS_STD`, shaped
``(batch, 24, time, height, width)``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["LATENTS_MEAN", "LATENTS_STD", "LatentResizer3D", "ResBlockEmb3D", "TemporalConv"]

#: Per-channel mean of the H3 video latents the network was trained on.
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)

#: Per-channel standard deviation of the same latents.
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def normalization(channels: int) -> nn.GroupNorm:
    """Group normalisation over 32 groups."""
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    """A 3D residual block whose normalised activations are scaled and shifted by an embedding."""

    def __init__(self, channels: int, emb_channels: int, out_channels: int | None = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Identity(),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(batch, channels, time, height, width)``.
            emb: ``(batch, emb_channels)``.

        Returns:
            ``(batch, out_channels, time, height, width)``.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out.reshape(*emb_out.shape, 1, 1, 1)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    """A residual depthwise convolution along time, followed by a pointwise one."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``(batch, channels, time, height, width)``."""
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """Encode a latent, resize its features trilinearly, and decode them at the new size.

    Args:
        in_channels: Latent channels.
        in_blocks: Residual blocks before the resize.
        out_blocks: Residual blocks after it.
        channels: Feature width.
        temporal_every: A temporal convolution follows every this many residual blocks, ``0``
            for none.
        temporal_kernel: Frames the temporal convolution spans.
    """

    def __init__(self, in_channels: int = 24, in_blocks: int = 12, out_blocks: int = 12,
                 channels: int = 512, temporal_every: int = 2, temporal_kernel: int = 5):
        super().__init__()
        embed_dim = 64
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for index in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and index % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for index in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and index % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Pass ``x`` through ``blocks``, the residual ones taking ``emb``."""
        for block in blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        return x

    def forward(self, x: torch.Tensor, scale: float, size: tuple) -> torch.Tensor:
        """Resize a normalised latent.

        Args:
            x: ``(batch, in_channels, time, height, width)``.
            scale: Size multiplier the embedding is conditioned on.
            size: ``(time, height, width)`` to resize to.

        Returns:
            ``(batch, in_channels, *size)``.
        """
        emb = self.embed(torch.tensor([[float(scale) - 1.0]], dtype=x.dtype, device=x.device))
        x = self._run(self.in_blocks, self.conv_in(x), emb)
        x = F.interpolate(x, size=tuple(size), mode="trilinear", align_corners=False)
        x = self._run(self.out_blocks, x, emb)
        return self.conv_out(F.silu(self.norm_out(x)))
