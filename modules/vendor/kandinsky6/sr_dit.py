"""Kandinsky 6 video super-resolution transformer and its NABLA sparse attention, on ComfyUI's layer operations.

Tiles are channels-last ``(B, T, H, W, C)`` with ``C = 2 * 64 + 1``: the noisy latent, an anchor latent
and an anchor mask. Tokens run in 8x8 spatial block order; attention blocks hold 64 tokens.
"""

from __future__ import annotations

import logging
import math
import threading

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import comfy.model_management

from .dit import (
    FeedForward,
    Modulation,
    OutLayer,
    TimeEmbeddings,
    VisualEmbeddings,
    _ops,
    apply_gate_sum,
    apply_rotary,
    apply_scale_shift_norm,
    rope_3d,
)

FRACTAL_BLOCK_SIZE = 8
BLOCK_TOKENS = FRACTAL_BLOCK_SIZE**2
#: Query blocks per masked attention call when flex attention is unavailable.
FALLBACK_QUERY_BLOCKS = 4

_LOCK = threading.Lock()
_FLEX = {"kernel": None, "failed": False}


def local_patch(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, g1, H // g2, g2, W // g3, g3, *x.shape[dim + 3 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 2, d + 4, d + 1, d + 3, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def local_merge(x: Tensor, shape: tuple, group_size: tuple, dim: int = 0) -> Tensor:
    T, H, W = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], T // g1, H // g2, W // g3, g1, g2, g3, *x.shape[dim + 2 :])
    d = len(x.shape[:dim])
    x = x.permute(*range(d), d, d + 3, d + 1, d + 4, d + 2, d + 5, *range(d + 6, len(x.shape)))
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


def sliding_tile_mask(num_frames: int, height: int, width: int, window: tuple, device) -> Tensor:
    """Static sliding-tile prior over the ``(t, h, w)`` grid of 8x8 token blocks, as a boolean matrix."""
    window_t, window_h, window_w = window
    positions = torch.arange(max(num_frames, height, width), device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    near_t = (distance[:num_frames, :num_frames] <= window_t // 2).flatten()
    near_h = (distance[:height, :height] <= window_h // 2).flatten()
    near_w = (distance[:width, :width] <= window_w // 2).flatten()
    near_hw = (near_h[:, None] & near_w[None, :]).reshape(height, height, width, width).transpose(1, 2).flatten()
    near = (near_t[:, None] & near_hw[None, :]).reshape(num_frames, num_frames, height * width, height * width)
    return near.transpose(1, 2).reshape(num_frames * height * width, num_frames * height * width)


def nabla_blocks(q: Tensor, k: Tensor, sta: Tensor, threshold: float) -> Tensor:
    """Boolean ``(B, heads, blocks, blocks)`` NABLA pattern from ``(B, heads, S, D)`` queries and keys."""
    B, h, S, D = q.shape
    s1 = S // BLOCK_TOKENS
    qa = q.reshape(B, h, s1, BLOCK_TOKENS, D).float().mean(-2)
    ka = k.reshape(B, h, s1, BLOCK_TOKENS, D).float().mean(-2).transpose(-2, -1)
    attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)
    vals, inds = attn_map.sort(-1)
    mask = (vals.cumsum_(-1) >= 1 - threshold).int().gather(-1, inds.argsort(-1))
    return torch.logical_or(mask, sta)


def _flex_kernel():
    with _LOCK:
        if _FLEX["kernel"] is None and not _FLEX["failed"]:
            try:
                from torch.nn.attention.flex_attention import flex_attention

                _FLEX["kernel"] = torch.compile(flex_attention, dynamic=False)
            except Exception as error:
                _FLEX["failed"] = True
                logging.warning("Kandinsky 6 upscaling runs its sparse attention without flex attention (%s).", error)
        return _FLEX["kernel"]


def _flex_attention(q: Tensor, k: Tensor, v: Tensor, blocks: Tensor) -> Tensor:
    from torch.nn.attention.flex_attention import BlockMask

    kv_nb = blocks.sum(-1).to(torch.int32)
    kv_inds = blocks.int().argsort(dim=-1, descending=True).to(torch.int32)
    mask = BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=BLOCK_TOKENS, mask_mod=None)
    return _flex_kernel()(q, k, v, block_mask=mask, kernel_options={"BLOCK_M": 64, "BLOCK_N": 64})


def _masked_attention(q: Tensor, k: Tensor, v: Tensor, blocks: Tensor) -> Tensor:
    out = torch.empty_like(q)
    step = FALLBACK_QUERY_BLOCKS
    for start in range(0, blocks.shape[2], step):
        rows = blocks[:, :, start : start + step]
        mask = rows.repeat_interleave(BLOCK_TOKENS, dim=2).repeat_interleave(BLOCK_TOKENS, dim=3)
        span = slice(start * BLOCK_TOKENS, (start + rows.shape[2]) * BLOCK_TOKENS)
        out[:, :, span] = F.scaled_dot_product_attention(q[:, :, span], k, v, attn_mask=mask)
    return out


def nabla_attention(q: Tensor, k: Tensor, v: Tensor, sta: Tensor, threshold: float) -> Tensor:
    """NABLA sparse self-attention over ``(B, S, heads, D)`` tensors, answered as ``(B, S, heads * D)``."""
    q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    blocks = nabla_blocks(q, k, sta, threshold)
    out = None
    if q.is_cuda and _flex_kernel() is not None:
        try:
            out = _flex_attention(q, k, v, blocks)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            with _LOCK:
                _FLEX["kernel"], _FLEX["failed"] = None, True
            logging.warning("Kandinsky 6 upscaling runs its sparse attention without flex attention (%s).", error)
    if out is None:
        out = _masked_attention(q, k, v, blocks)
    return out.transpose(1, 2).flatten(2)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, head_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.num_heads = num_channels // head_dim
        self.to_query = ops.Linear(num_channels, num_channels, device=device, dtype=dtype)
        self.to_key = ops.Linear(num_channels, num_channels, device=device, dtype=dtype)
        self.to_value = ops.Linear(num_channels, num_channels, device=device, dtype=dtype)
        self.query_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.key_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.out_layer = ops.Linear(num_channels, num_channels, device=device, dtype=dtype)

    def forward(self, x: Tensor, rope: Tensor, sta: Tensor, threshold: float) -> Tensor:
        q = self.to_query(x).unflatten(-1, (self.num_heads, -1))
        k = self.to_key(x).unflatten(-1, (self.num_heads, -1))
        v = self.to_value(x).unflatten(-1, (self.num_heads, -1))
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        q = apply_rotary(q, rope)
        k = apply_rotary(k, rope)
        return self.out_layer(nabla_attention(q, k, v, sta, threshold))


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.visual_modulation = Modulation(time_dim, model_dim, 6, operation_settings)
        self.self_attention_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.self_attention = SelfAttention(model_dim, head_dim, operation_settings)
        self.feed_forward_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings)

    def forward(self, x: Tensor, temb: Tensor, rope: Tensor, sta: Tensor, threshold: float) -> Tensor:
        attention_params, feed_forward_params = torch.chunk(self.visual_modulation(temb), 2, dim=-1)
        shift, scale, gate = torch.chunk(attention_params, 3, dim=-1)
        attended = self.self_attention(apply_scale_shift_norm(self.self_attention_norm, x, scale, shift), rope, sta, threshold)
        x = apply_gate_sum(x, attended, gate)
        shift, scale, gate = torch.chunk(feed_forward_params, 3, dim=-1)
        return apply_gate_sum(x, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)), gate)


class SRTransformer3D(nn.Module):
    """Text-free transformer refining K-VAE latent tiles; the output holds ``out_visual_dim`` channels."""

    def __init__(
        self,
        in_visual_dim: int = 64,
        out_visual_dim: int = 640,
        time_dim: int = 512,
        patch_size: tuple = (1, 1, 1),
        model_dim: int = 1792,
        ff_dim: int = 7168,
        num_visual_blocks: int = 32,
        axes_dims: tuple = (16, 24, 24),
        scale_factor: tuple = (1.0, 2.0, 2.0),
        nabla_threshold: float = 0.8,
        nabla_window: tuple = (11, 7, 7),
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        op = {"operations": operations, "device": device, "dtype": dtype}
        head_dim = sum(axes_dims)
        self.dtype = dtype
        self.axes_dims = tuple(axes_dims)
        self.scale_factor = tuple(scale_factor)
        self.nabla_threshold = float(nabla_threshold)
        self.nabla_window = tuple(nabla_window)
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim, operation_settings=op)
        self.pooled_bias = nn.Parameter(torch.empty(time_dim, device=device, dtype=torch.float32))
        self.visual_embeddings = VisualEmbeddings(2 * in_visual_dim + 1, model_dim, tuple(patch_size), op)
        self.visual_transformer_blocks = nn.ModuleList(
            [TransformerBlock(model_dim, time_dim, ff_dim, head_dim, op) for _ in range(num_visual_blocks)]
        )
        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, tuple(patch_size), op)

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        device = x.device
        bias = comfy.model_management.cast_to(self.pooled_bias, dtype=torch.float32, device=device)
        temb = self.time_embeddings(timestep, torch.float32) + bias

        visual = self.visual_embeddings(x)
        batch = visual.shape[0]
        shape = tuple(int(size) for size in visual.shape[1:4])
        num_frames, height, width = shape
        if height % FRACTAL_BLOCK_SIZE or width % FRACTAL_BLOCK_SIZE:
            raise ValueError(f"NABLA attention needs the token grid {(height, width)} to be divisible by {FRACTAL_BLOCK_SIZE}")
        positions = tuple(torch.arange(size, device=device) for size in shape)
        rope = rope_3d(self.axes_dims, positions, self.scale_factor)
        group = (1, FRACTAL_BLOCK_SIZE, FRACTAL_BLOCK_SIZE)
        visual = local_patch(visual, shape, group, dim=1).flatten(1, 2)
        rope = local_patch(rope, shape, group, dim=0).flatten(0, 1)
        sta = sliding_tile_mask(num_frames, height // FRACTAL_BLOCK_SIZE, width // FRACTAL_BLOCK_SIZE, self.nabla_window, device)

        for block in self.visual_transformer_blocks:
            visual = block(visual, temb, rope, sta, self.nabla_threshold)

        visual = local_merge(visual.reshape(batch, -1, BLOCK_TOKENS, visual.shape[-1]), shape, group, dim=1)
        return self.out_layer(visual, temb)
