"""Causal 3D K-VAE of Kandinsky 6 video super-resolution, on ComfyUI's layer operations.

Videos are ``(B, C, T, H, W)`` in ``[-1, 1]``; latents are ``(B, 64, 1 + (T - 1) / 4, H / 16, W / 16)``,
unscaled.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

import comfy.model_management
import comfy.ops

ops = comfy.ops.disable_weight_init

SEGMENT_FRAMES = 16
CONV_CHUNK_ELEMENTS = 2 * 10**9
#: Elements one widest full-resolution decoder activation may hold before segments shorten.
DECODE_SEGMENT_ELEMENTS = 6 * 10**8
#: Channels of the widest decoder activation at full resolution.
DECODE_WIDEST = 256
#: Channels squared in float32 at a time by the RMS norm.
NORM_CHUNK = 32
#: Elements the widest encoder activation, counted per full-resolution pixel, may hold before segments shorten.
ENCODE_SEGMENT_ELEMENTS = 3 * 10**8
#: Channels per full-resolution pixel of the widest encoder activation.
ENCODE_WIDEST = 32


def encode_segment(height: int, width: int) -> int:
    """Pixel frames encoded per segment for frames of this size: 4 to 16, a multiple of 4."""
    frames = ENCODE_SEGMENT_ELEMENTS // max(1, int(height) * int(width) * ENCODE_WIDEST)
    return int(max(4, min(SEGMENT_FRAMES, frames // 4 * 4)))


def decode_segment(latent_height: int, latent_width: int) -> int:
    """Pixel frames decoded per segment for a latent of this size: 4 to 16, a multiple of 4."""
    pixels = int(latent_height) * int(latent_width) * 256
    frames = DECODE_SEGMENT_ELEMENTS // max(1, pixels * DECODE_WIDEST)
    return int(max(4, min(SEGMENT_FRAMES, frames // 4 * 4)))


def _gamma(gamma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return comfy.model_management.cast_to(gamma, dtype=like.dtype, device=like.device)


class SafeConv3d(ops.Conv3d):
    """``Conv3d`` that convolves very large inputs in temporal chunks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= CONV_CHUNK_ELEMENTS:
            return super().forward(x)

        kernel_size = self.kernel_size[0]
        chunks = torch.chunk(x, math.ceil(x.numel() / CONV_CHUNK_ELEMENTS), dim=2)
        if kernel_size > 1 and any(chunk.size(2) < kernel_size for chunk in chunks):
            if chunks[0].numel() * (kernel_size / chunks[0].size(2)) >= CONV_CHUNK_ELEMENTS:
                raise ValueError("frames are too big for Conv3d")
            stride = self.stride[0]
            windows = range(0, x.size(2) - kernel_size + 1, stride)
            return torch.cat([super(SafeConv3d, self).forward(x[:, :, i : i + kernel_size]) for i in windows], dim=2)

        outputs = []
        for i, chunk in enumerate(chunks):
            if i == 0 or kernel_size == 1:
                carried = chunk
            else:
                carried = torch.cat([carried[:, :, -kernel_size + 1 :], chunk], dim=2)
            outputs.append(super().forward(carried))
        return torch.cat(outputs, dim=2)


class CausalConv3d(nn.Module):
    """Causal 3D convolution carrying its temporal padding across segments in ``cache``."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1)) -> None:
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size,) * 3
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.time_stride = stride[0]
        self.conv = SafeConv3d(in_channels, out_channels, kernel_size, stride=stride)

    @staticmethod
    def make_cache() -> dict:
        return {"padding": None}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        batch_size, _, num_frames, height, width = hidden_states.shape
        hidden_states = F.pad(hidden_states, (self.width_pad, self.width_pad, self.height_pad, self.height_pad))

        if cache["padding"] is None:
            first_frame = hidden_states[:, :, :1]
            padding = first_frame.expand(-1, -1, self.time_pad, -1, -1)
        else:
            padding = cache["padding"]

        stride = self.time_stride
        output_frames = (num_frames + 1) // 2 if stride == 2 else num_frames
        output = torch.empty(
            (batch_size, self.conv.out_channels, output_frames, height, width),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        offset_out = math.ceil(padding.size(2) / stride)
        offset_in = offset_out * stride - padding.size(2)
        if offset_out > 0:
            padded = torch.cat([padding, hidden_states[:, :, : offset_in + self.time_kernel_size - stride]], dim=2)
            output[:, :, :offset_out] = self.conv(padded)
        if offset_out < output_frames:
            output[:, :, offset_out:] = self.conv(hidden_states[:, :, offset_in:])

        pad_offset = offset_in + stride * math.trunc((num_frames - offset_in - self.time_kernel_size) / stride) + stride
        if pad_offset < 0:
            cache["padding"] = torch.cat([padding[:, :, pad_offset:], hidden_states], dim=2)
        else:
            cache["padding"] = hidden_states[:, :, pad_offset:].clone()
        return output


class RMSNorm(nn.Module):
    """RMS normalisation over the channel axis of ``(B, C, T, H, W)``, in float32."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.empty(num_channels, 1, 1, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        squares = None
        for part in torch.split(hidden_states, NORM_CHUNK, dim=1):
            summed = part.float().square().sum(1, keepdim=True)
            squares = summed if squares is None else squares.add_(summed)
        gain = (self.scale / squares.sqrt_().clamp_min_(1e-12)).to(hidden_states.dtype)
        return (hidden_states * gain).mul_(_gamma(self.gamma, hidden_states))


class SpatialNorm3D(nn.Module):
    """RMS norm modulated by convolutions of the nearest-upsampled latent ``zq``."""

    def __init__(self, num_channels: int, zq_channels: int) -> None:
        super().__init__()
        self.norm_layer = RMSNorm(num_channels)
        self.conv_y = SafeConv3d(zq_channels, num_channels, kernel_size=1)
        self.conv_b = SafeConv3d(zq_channels, num_channels, kernel_size=1)

    @staticmethod
    def make_cache() -> dict:
        return {"is_first_segment": True}

    def forward(self, hidden_states: torch.Tensor, zq: torch.Tensor, cache: dict) -> torch.Tensor:
        if cache["is_first_segment"]:
            zq_first = F.interpolate(zq[:, :, :1], size=hidden_states[:, :, :1].shape[-3:], mode="nearest")
            if zq.size(2) > 1:
                zq_rest = torch.cat(
                    [
                        F.interpolate(split, size=hidden_states[:, :, 1:].shape[-3:], mode="nearest")
                        for split in torch.split(zq[:, :, 1:], 32, dim=1)
                    ],
                    dim=1,
                )
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            zq = torch.cat(
                [F.interpolate(split, size=hidden_states.shape[-3:], mode="nearest") for split in torch.split(zq, 32, dim=1)],
                dim=1,
            )
        output = self.norm_layer(hidden_states).mul_(self.conv_y(zq)).add_(self.conv_b(zq))
        cache["is_first_segment"] = False
        return output


class ResnetBlock3D(nn.Module):
    """Causal residual block; decoder blocks modulate their norms with ``zq``."""

    def __init__(self, in_channels: int, out_channels: int, zq_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if zq_channels is None:
            self.norm1 = RMSNorm(in_channels)
            self.norm2 = RMSNorm(out_channels)
        else:
            self.norm1 = SpatialNorm3D(in_channels, zq_channels)
            self.norm2 = SpatialNorm3D(out_channels, zq_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.nin_shortcut = SafeConv3d(in_channels, out_channels, kernel_size=1)

    def make_cache(self) -> dict:
        cache = {"conv1": self.conv1.make_cache(), "conv2": self.conv2.make_cache()}
        if isinstance(self.norm1, SpatialNorm3D):
            cache["norm1"] = self.norm1.make_cache()
            cache["norm2"] = self.norm2.make_cache()
        return cache

    def forward(self, hidden_states: torch.Tensor, cache: dict, zq: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        if zq is None:
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states = self.norm1(hidden_states, zq, cache["norm1"])
        hidden_states = self.conv1(F.silu(hidden_states), cache["conv1"])
        if zq is None:
            hidden_states = self.norm2(hidden_states)
        else:
            hidden_states = self.norm2(hidden_states, zq, cache["norm2"])
        hidden_states = self.conv2(F.silu(hidden_states), cache["conv2"])
        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class Downsample(nn.Module):
    """Spatial 2x downsample with an optional causal temporal 2x."""

    def __init__(self, in_channels: int, compress_time: bool) -> None:
        super().__init__()
        out_channels = 2 * in_channels
        self.compress_time = compress_time
        self.unshuffle = nn.PixelUnshuffle(2)
        self.spatial_conv = SafeConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        if compress_time:
            self.temporal_conv = nn.ModuleList(
                [
                    CausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(1, 1, 1)),
                    CausalConv3d(out_channels, out_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                ]
            )
        self.linear = SafeConv3d(out_channels, out_channels, kernel_size=1)

    def make_cache(self) -> dict:
        if not self.compress_time:
            return {}
        return {"temporal_conv": [conv.make_cache() for conv in self.temporal_conv]}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = hidden_states.shape
        frames = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        frames = self.unshuffle(frames)
        frames = frames.view(batch_size * num_frames, 2 * channels, 2, height // 2, width // 2).mean(dim=2)
        frames = frames.view(batch_size, num_frames, 2 * channels, height // 2, width // 2).permute(0, 2, 1, 3, 4)
        hidden_states = self.spatial_conv(hidden_states) + frames

        if not self.compress_time:
            return self.linear(hidden_states)

        batch_size, channels, num_frames, height, width = hidden_states.shape
        sequence = hidden_states.permute(0, 3, 4, 1, 2).reshape(-1, channels, num_frames)
        if cache["temporal_conv"][0]["padding"] is None:
            first, rest = sequence[..., :1], sequence[..., 1:]
            pooled = torch.cat([first, F.avg_pool1d(rest, kernel_size=2, stride=2)], dim=-1) if rest.size(-1) else first
        else:
            pooled = F.avg_pool1d(sequence, kernel_size=2, stride=2)
        pooled = pooled.reshape(batch_size, height, width, channels, -1).permute(0, 3, 4, 1, 2)

        conv_out = self.temporal_conv[0](hidden_states, cache["temporal_conv"][0])
        conv_out = self.temporal_conv[1](conv_out, cache["temporal_conv"][1])
        return self.linear(conv_out + pooled)


class Upsample(nn.Module):
    """Spatial 2x nearest upsample with a convolutional residual, after an optional causal temporal 2x."""

    def __init__(self, channels: int, compress_time: bool) -> None:
        super().__init__()
        self.compress_time = compress_time
        self.spatial_conv = SafeConv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        if compress_time:
            self.temporal_conv = CausalConv3d(channels, channels, kernel_size=(3, 1, 1))
        self.linear = SafeConv3d(channels, channels, kernel_size=1)

    def make_cache(self) -> dict:
        return {"temporal_conv": self.temporal_conv.make_cache()} if self.compress_time else {}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        if self.compress_time:
            repeated = hidden_states.repeat_interleave(2, dim=2)
            if cache["temporal_conv"]["padding"] is None:
                repeated = repeated[:, :, 1:]
            conv_out = self.temporal_conv(repeated, cache["temporal_conv"])
            hidden_states = conv_out + repeated

        hidden_states = F.interpolate(hidden_states, scale_factor=(1, 2, 2), mode="nearest")
        hidden_states = hidden_states + self.spatial_conv(hidden_states)
        return self.linear(hidden_states)


class Encoder3D(nn.Module):
    """Causal encoder: ``conv_in``, downsampling resnet levels, a bottleneck, ``norm_out`` and ``conv_out``."""

    def __init__(self, in_channels, latent_channels, block_out_channels, layers_per_block,
                 temporal_compression_ratio, temporal_compression_start_level) -> None:
        super().__init__()
        num_levels = len(block_out_channels)
        end_level = int(math.log2(temporal_compression_ratio)) + temporal_compression_start_level

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.down = nn.ModuleList()
        for level in range(num_levels):
            block_in = block_out_channels[0] if level == 0 else 2 * block_out_channels[level - 1]
            block_out = block_out_channels[level]
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(ResnetBlock3D(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = blocks
            if level != num_levels - 1:
                compress_time = temporal_compression_start_level <= level < end_level
                down.downsample = Downsample(block_in, compress_time=compress_time)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(block_in, block_in)
        self.mid.block_2 = ResnetBlock3D(block_in, block_in)
        self.norm_out = RMSNorm(block_in)
        self.conv_out = CausalConv3d(block_in, 2 * latent_channels, kernel_size=3)

    def make_cache(self) -> dict:
        return {
            "conv_in": self.conv_in.make_cache(),
            "down": [
                {
                    "block": [block.make_cache() for block in down.block],
                    "downsample": down.downsample.make_cache() if hasattr(down, "downsample") else {},
                }
                for down in self.down
            ],
            "mid_1": self.mid.block_1.make_cache(),
            "mid_2": self.mid.block_2.make_cache(),
            "conv_out": self.conv_out.make_cache(),
        }

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states, cache["conv_in"])
        for down, level_cache in zip(self.down, cache["down"]):
            for i, block in enumerate(down.block):
                hidden_states = block(hidden_states, level_cache["block"][i])
            if hasattr(down, "downsample"):
                hidden_states = down.downsample(hidden_states, level_cache["downsample"])

        hidden_states = self.mid.block_1(hidden_states, cache["mid_1"])
        hidden_states = self.mid.block_2(hidden_states, cache["mid_2"])

        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states, cache["conv_out"])


class Decoder3D(nn.Module):
    """Causal decoder: ``conv_in``, a ``zq``-conditioned bottleneck, upsampling resnet levels and the output."""

    def __init__(self, out_channels, latent_channels, block_out_channels, layers_per_block,
                 temporal_compression_ratio, temporal_compression_start_level) -> None:
        super().__init__()
        num_levels = len(block_out_channels)
        end_level = int(math.log2(temporal_compression_ratio)) + temporal_compression_start_level

        block_in = block_out_channels[-1]
        self.conv_in = CausalConv3d(latent_channels, block_in, kernel_size=3)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(block_in, block_in, zq_channels=latent_channels)
        self.mid.block_2 = ResnetBlock3D(block_in, block_in, zq_channels=latent_channels)

        self.up = nn.ModuleList()
        for level in reversed(range(num_levels)):
            block_out = block_out_channels[level]
            blocks = nn.ModuleList()
            for _ in range(layers_per_block + 1):
                blocks.append(ResnetBlock3D(block_in, block_out, zq_channels=latent_channels))
                block_in = block_out
            up = nn.Module()
            up.block = blocks
            if level != 0:
                compress_time = num_levels - temporal_compression_start_level > level >= num_levels - end_level
                up.upsample = Upsample(block_in, compress_time=compress_time)
            self.up.insert(0, up)

        self.norm_out = SpatialNorm3D(block_in, latent_channels)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

    def make_cache(self) -> dict:
        return {
            "conv_in": self.conv_in.make_cache(),
            "mid_1": self.mid.block_1.make_cache(),
            "mid_2": self.mid.block_2.make_cache(),
            "up": [
                {
                    "block": [block.make_cache() for block in up.block],
                    "upsample": up.upsample.make_cache() if hasattr(up, "upsample") else {},
                }
                for up in self.up
            ],
            "norm_out": self.norm_out.make_cache(),
            "conv_out": self.conv_out.make_cache(),
        }

    def forward(self, latents: torch.Tensor, cache: dict) -> torch.Tensor:
        hidden_states = self.conv_in(latents, cache["conv_in"])
        hidden_states = self.mid.block_1(hidden_states, cache["mid_1"], zq=latents)
        hidden_states = self.mid.block_2(hidden_states, cache["mid_2"], zq=latents)

        for level in reversed(range(len(self.up))):
            up, level_cache = self.up[level], cache["up"][level]
            for i, block in enumerate(up.block):
                hidden_states = block(hidden_states, level_cache["block"][i], zq=latents)
            if hasattr(up, "upsample"):
                hidden_states = up.upsample(hidden_states, level_cache["upsample"])

        hidden_states = self.norm_out(hidden_states, latents, cache["norm_out"])
        hidden_states = F.silu(hidden_states)
        return self.conv_out(hidden_states, cache["conv_out"])


class KVAE(nn.Module):
    """The K-VAE: ``encode`` answers the posterior mean, ``decode`` the video, both in 16-frame segments."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 64,
        encoder_block_out_channels: tuple = (16, 128, 256, 512, 1024),
        decoder_block_out_channels: tuple = (16, 256, 512, 1024, 2048),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        temporal_compression_start_level: int = 1,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression_ratio = temporal_compression_ratio
        self.encoder = Encoder3D(in_channels, latent_channels, encoder_block_out_channels, layers_per_block,
                                 temporal_compression_ratio, temporal_compression_start_level)
        self.decoder = Decoder3D(out_channels, latent_channels, decoder_block_out_channels, layers_per_block,
                                 temporal_compression_ratio, temporal_compression_start_level)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        cache = self.encoder.make_cache()
        segment = encode_segment(x.size(-2), x.size(-1))
        lengths = [min(segment + 1, x.size(2))]
        remaining = x.size(2) - lengths[0]
        while remaining > 0:
            lengths.append(min(segment, remaining))
            remaining -= segment
        moments = torch.cat([self.encoder(segment, cache) for segment in torch.split(x, lengths, dim=2)], dim=2)
        return moments[:, : self.latent_channels]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        cache = self.decoder.make_cache()
        latent_segment = decode_segment(z.shape[-2], z.shape[-1]) // self.temporal_compression_ratio
        frames = z.size(2)
        if frames == 1:
            lengths = [1]
        else:
            lengths = [latent_segment] * ((frames - 1) // latent_segment)
            if (frames - 1) % latent_segment:
                lengths.append((frames - 1) % latent_segment)
            lengths[0] += 1
        return torch.cat([self.decoder(segment, cache) for segment in torch.split(z, lengths, dim=2)], dim=2)
