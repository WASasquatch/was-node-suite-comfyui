"""The AnySplat network: patch tokens from a ViT-L/14, alternating frame and global attention, and heads
for cameras, depth and per-pixel Gaussians.

Parameter names match the published checkpoint once its ``encoder.`` prefix is removed.
"""

from __future__ import annotations

__all__ = ["GAUSSIAN_CHANNELS", "PATCH", "SH_BANDS", "TAPS", "AnySplatNetwork"]

import math

import torch
import torch.nn.functional as F
from torch import nn

#: Token width of the transformer, and twice that once frame and global tokens are joined.
WIDTH = 1024

#: Attention heads in every transformer.
HEADS = 16

#: Blocks in the patch encoder and in each attention stream.
DEPTH = 24

#: Register tokens beside each frame's camera token.
REGISTERS = 4

#: Pixels per patch side.
PATCH = 14

#: Blocks whose joined frame and global tokens the heads read.
TAPS = (4, 11, 17, 23)

#: Spherical harmonic coefficients per colour channel, degree 4.
SH_BANDS = 25

#: Per-pixel Gaussian channels: density, three scales, a quaternion and the harmonics.
GAUSSIAN_CHANNELS = 1 + 3 + 4 + 3 * SH_BANDS

#: Positional frequency base of the rotary embedding and of the head grids.
ROPE_BASE = 100.0

#: Image statistics the patch encoder was trained on.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Frames a dense head decodes at once.
HEAD_FRAMES = 8


class LayerScale(nn.Module):
    """A learned per-channel gain."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    """Two linear layers with a GELU between."""

    def __init__(self, dim: int, hidden: int, out: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out or dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_tables(positions: torch.Tensor, head_dim: int, dtype) -> tuple:
    """Cosine and sine tables for 2D rotary embedding.

    Args:
        positions: ``(N, 2)`` integer row and column of each token.
        head_dim: Width of one attention head; rows rotate its first half, columns its second.
        dtype: Dtype of the tables.

    Returns:
        ``(cos, sin)``, each ``(N, head_dim)``.
    """
    half = head_dim // 2
    rates = 1.0 / (ROPE_BASE ** (torch.arange(0, half, 2, device=positions.device).float() / half))
    parts_cos, parts_sin = [], []
    for axis in (0, 1):
        angles = positions[:, axis].float()[:, None] * rates[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        parts_cos.append(angles.cos())
        parts_sin.append(angles.sin())
    return torch.cat(parts_cos, -1).to(dtype), torch.cat(parts_sin, -1).to(dtype)


def _apply_rope(x, tables):
    """Rotate ``(B, heads, N, head_dim)`` queries or keys by per-token tables ``(N, head_dim)``."""
    cos, sin = tables
    half = x.shape[-1] // 2
    rows, cols = x[..., :half], x[..., half:]
    rows = rows * cos[:, :half] + _rotate_half(rows) * sin[:, :half]
    cols = cols * cos[:, half:] + _rotate_half(cols) * sin[:, half:]
    return torch.cat((rows, cols), dim=-1)


class Attention(nn.Module):
    """Multi-head self attention, optionally with normalised queries and keys."""

    def __init__(self, dim: int, heads: int, qk_norm: bool):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        head_dim = dim // heads
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x, rope=None):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = _apply_rope(q, rope), _apply_rope(k, rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    """Pre-norm transformer block with layer scale."""

    def __init__(self, dim: int, heads: int, eps: float, qk_norm: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, qk_norm)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, dim * 4)
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchProjection(nn.Module):
    """A strided convolution cutting a picture into patch tokens."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, WIDTH, PATCH, PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchEncoder(nn.Module):
    """ViT-L/14 with four registers, answering normalised patch tokens."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchProjection()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 37 * 37, WIDTH))
        self.register_tokens = nn.Parameter(torch.zeros(1, REGISTERS, WIDTH))
        self.mask_token = nn.Parameter(torch.zeros(1, WIDTH))
        self.blocks = nn.ModuleList(Block(WIDTH, HEADS, 1e-6) for _ in range(DEPTH))
        self.norm = nn.LayerNorm(WIDTH, eps=1e-6)

    def _positions(self, rows: int, cols: int, dtype):
        grid = self.pos_embed[:, 1:].float()
        side = int(math.sqrt(grid.shape[1]))
        if rows == side and cols == side:
            return self.pos_embed.to(dtype)
        grid = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(rows, cols), mode="bicubic", antialias=True)
        grid = grid.permute(0, 2, 3, 1).reshape(1, rows * cols, -1)
        return torch.cat((self.pos_embed[:, :1].float(), grid), dim=1).to(dtype)

    def forward(self, x):
        rows, cols = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1).to(tokens.dtype), tokens), dim=1)
        tokens = tokens + self._positions(rows, cols, tokens.dtype)
        tokens = torch.cat((tokens[:, :1], self.register_tokens.expand(tokens.shape[0], -1, -1).to(tokens.dtype),
                            tokens[:, 1:]), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, 1 + REGISTERS:]


class Aggregator(nn.Module):
    """Patch tokens plus camera and register tokens, refined by alternating frame and global attention."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEncoder()
        self.frame_blocks = nn.ModuleList(Block(WIDTH, HEADS, 1e-5, qk_norm=True) for _ in range(DEPTH))
        self.global_blocks = nn.ModuleList(Block(WIDTH, HEADS, 1e-5, qk_norm=True) for _ in range(DEPTH))
        self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, WIDTH))
        self.register_token = nn.Parameter(torch.zeros(1, 2, REGISTERS, WIDTH))

    @staticmethod
    def _special(token, frames: int):
        """The first frame's copy of a special token, then the shared copy for every later frame."""
        return torch.cat((token[:, :1], token[:, 1:].expand(1, frames - 1, *token.shape[2:])), dim=1)[0]

    def forward(self, images):
        """Joined frame and global tokens at :data:`TAPS`, each ``(frames, tokens, 2 * WIDTH)``.

        Args:
            images: ``(frames, 3, H, W)`` in ``[0, 1]``, sides multiples of :data:`PATCH`.
        """
        frames, _, height, width = images.shape
        mean = torch.tensor(MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        patches = self.patch_embed((images - mean) / std)
        dtype = patches.dtype
        tokens = torch.cat((self._special(self.camera_token, frames).to(dtype),
                            self._special(self.register_token, frames).to(dtype), patches), dim=1)
        rows, cols = height // PATCH, width // PATCH
        grid = torch.cartesian_prod(torch.arange(rows, device=images.device), torch.arange(cols, device=images.device)) + 1
        positions = torch.cat((torch.zeros(1 + REGISTERS, 2, dtype=grid.dtype, device=grid.device), grid))
        head_dim = WIDTH // HEADS
        per_frame = rope_tables(positions, head_dim, dtype)
        across = tuple(table.repeat(frames, 1) for table in per_frame)
        count = tokens.shape[1]
        taps = []
        for index in range(DEPTH):
            tokens = self.frame_blocks[index](tokens, per_frame)
            seen_frame = tokens
            tokens = self.global_blocks[index](tokens.reshape(1, frames * count, WIDTH), across).reshape(frames, count, WIDTH)
            if index in TAPS:
                taps.append(torch.cat((seen_frame, tokens), dim=-1))
        return taps


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class CameraHead(nn.Module):
    """Iteratively refines each frame's pose encoding: translation, xyzw quaternion, two fields of view."""

    def __init__(self, dim: int = 2 * WIDTH):
        super().__init__()
        self.trunk = nn.Sequential(*(Block(dim, HEADS, 1e-5) for _ in range(4)))
        self.token_norm = nn.LayerNorm(dim)
        self.trunk_norm = nn.LayerNorm(dim)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, 9))
        self.embed_pose = nn.Linear(9, dim)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.adaln_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(dim, dim // 2, 9)

    def forward(self, tokens, iterations: int = 4):
        """``(frames, 9)`` pose encodings from the last tap's camera tokens ``(frames, tokens, dim)``."""
        pose_tokens = self.token_norm(tokens[None, :, 0])
        encoding = None
        for _ in range(iterations):
            source = self.empty_pose_tokens.expand(1, pose_tokens.shape[1], -1) if encoding is None else encoding
            shift, scale, gate = self.poseLN_modulation(self.embed_pose(source)).chunk(3, dim=-1)
            modulated = gate * _modulate(self.adaln_norm(pose_tokens), shift, scale) + pose_tokens
            delta = self.pose_branch(self.trunk_norm(self.trunk(modulated)))
            encoding = delta if encoding is None else encoding + delta
        return torch.cat((encoding[..., :7], F.relu(encoding[..., 7:])), dim=-1)[0]


class ResidualUnit(nn.Module):
    """Two 3x3 convolutions over a rectified input, added back to that rectified input."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x):
        x = F.relu(x)
        return self.conv2(F.relu(self.conv1(x))) + x


class Fusion(nn.Module):
    """Merges a coarser map with a skip connection and upsamples the result."""

    def __init__(self, features: int, residual: bool = True):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1)
        if residual:
            self.resConfUnit1 = ResidualUnit(features)
        self.resConfUnit2 = ResidualUnit(features)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        if size is None:
            size = (x.shape[-2] * 2, x.shape[-1] * 2)
        return self.out_conv(F.interpolate(x, size=size, mode="bilinear", align_corners=True))


def _uv_embedding(rows: int, cols: int, channels: int, aspect: float, device, dtype):
    """A sinusoidal embedding ``(channels, rows, cols)`` of a centred grid scaled to unit diagonal."""
    diagonal = math.sqrt(aspect * aspect + 1.0)
    span_x, span_y = aspect / diagonal, 1.0 / diagonal
    xs = torch.linspace(-span_x * (cols - 1) / cols, span_x * (cols - 1) / cols, cols, device=device)
    ys = torch.linspace(-span_y * (rows - 1) / rows, span_y * (rows - 1) / rows, rows, device=device)
    u, v = torch.meshgrid(xs, ys, indexing="xy")
    quarter = channels // 4
    omega = 1.0 / ROPE_BASE ** (torch.arange(quarter, dtype=torch.float64, device=device) / quarter)
    parts = []
    for axis in (u, v):
        angles = axis.reshape(-1).double()[:, None] * omega[None, :]
        parts.append(torch.cat((angles.sin(), angles.cos()), dim=1))
    return torch.cat(parts, dim=1).float().reshape(rows, cols, channels).permute(2, 0, 1).to(dtype)


class DenseHead(nn.Module):
    """Decodes tapped tokens into a full-resolution map, optionally mixing in the picture itself."""

    def __init__(self, out_channels: int, last: int, picture: bool = False, dim: int = 2 * WIDTH):
        super().__init__()
        widths = (256, 512, 1024, 1024)
        self.norm = nn.LayerNorm(dim)
        self.projects = nn.ModuleList(nn.Conv2d(dim, width, 1) for width in widths)
        self.resize_layers = nn.ModuleList((
            nn.ConvTranspose2d(widths[0], widths[0], 4, 4),
            nn.ConvTranspose2d(widths[1], widths[1], 2, 2),
            nn.Identity(),
            nn.Conv2d(widths[3], widths[3], 3, 2, 1),
        ))
        self.scratch = nn.Module()
        for index, width in enumerate(widths, start=1):
            setattr(self.scratch, f"layer{index}_rn", nn.Conv2d(width, 256, 3, padding=1, bias=False))
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", Fusion(256, residual=index != 4))
        self.scratch.output_conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(128, last, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(last, out_channels, 1))
        self.picture = picture
        if picture:
            self.input_merger = nn.Sequential(nn.Conv2d(3, 128, 7, 1, 3), nn.ReLU())

    def _embed(self, x, aspect: float):
        return x + 0.1 * _uv_embedding(x.shape[-2], x.shape[-1], x.shape[1], aspect, x.device, x.dtype)[None]

    def forward(self, taps, images):
        """``(frames, out_channels, H, W)`` from taps ``(frames, tokens, dim)`` and pictures ``(frames, 3, H, W)``."""
        frames, _, height, width = images.shape
        rows, cols = height // PATCH, width // PATCH
        aspect = width / height
        maps = []
        for index, tap in enumerate(taps):
            x = self.norm(tap[:, 1 + REGISTERS:])
            x = x.permute(0, 2, 1).reshape(frames, x.shape[-1], rows, cols)
            x = self._embed(self.projects[index](x), aspect)
            maps.append(self.resize_layers[index](x))
        s = self.scratch
        first, second, third, fourth = (getattr(s, f"layer{i}_rn")(m) for i, m in enumerate(maps, start=1))
        out = s.refinenet4(fourth, size=third.shape[2:])
        out = s.refinenet3(out, third, size=second.shape[2:])
        out = s.refinenet2(out, second, size=first.shape[2:])
        out = s.output_conv1(s.refinenet1(out, first))
        out = F.interpolate(out, size=(rows * PATCH, cols * PATCH), mode="bilinear", align_corners=True)
        if self.picture:
            out = out + self.input_merger(images)
        return s.output_conv2(self._embed(out, aspect))


class AnySplatNetwork(nn.Module):
    """The four parts AnySplat runs: aggregator, camera head, depth head and Gaussian head."""

    def __init__(self):
        super().__init__()
        self.aggregator = Aggregator()
        self.camera_head = CameraHead()
        self.depth_head = DenseHead(2, 32)
        self.gaussian_param_head = DenseHead(GAUSSIAN_CHANNELS + 1, 128, picture=True)

    def forward(self, images, autocast: bool = True):
        """Pose encodings, depth with confidence, and raw per-pixel Gaussians.

        Args:
            images: ``(frames, 3, H, W)`` in ``[0, 1]``, sides multiples of :data:`PATCH`.
            autocast: Run the aggregator in bfloat16 on CUDA.

        Returns:
            ``(poses (F, 9), depth (F, H, W), confidence (F, H, W), gaussians (F, C, H, W))``,
            ``C`` = :data:`GAUSSIAN_CHANNELS` + 1, the last channel a fusion confidence.
        """
        device_type = images.device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast and device_type == "cuda"):
            taps = self.aggregator(images)
        taps = [tap.float() for tap in taps]
        images = images.float()
        poses = self.camera_head(taps[-1])
        depths, confidences, raw = [], [], []
        for start in range(0, images.shape[0], HEAD_FRAMES):
            part = slice(start, start + HEAD_FRAMES)
            chunk = [tap[part] for tap in taps]
            depth = self.depth_head(chunk, images[part])
            depths.append(torch.exp(depth[:, 0]))
            confidences.append(1.0 + torch.exp(depth[:, 1]))
            raw.append(self.gaussian_param_head(chunk, images[part]))
        return poses, torch.cat(depths), torch.cat(confidences), torch.cat(raw)
