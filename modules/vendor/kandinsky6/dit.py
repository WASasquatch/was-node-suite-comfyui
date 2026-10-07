"""Kandinsky 6 joint video and audio diffusion transformer, on ComfyUI's layer operations.

Video tensors are channels-last ``(B, T, H, W, C)``; audio and text are ``(B, L, C)``. Rotary
tables are ``(L, 1, D/2, 2, 2)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import comfy.ops
from comfy.ldm.modules import attention as comfy_attention


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim
    )


def apply_scale_shift_norm(norm, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).to(dtype=x.dtype)


def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(dtype=x.dtype)


def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    out = rope[..., 0] * x_[..., 0]
    out.addcmul_(rope[..., 1], x_[..., 1])
    return out.reshape(*x.shape).to(dtype=x.dtype)


def _rope_from_args(args: Tensor) -> Tensor:
    rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
    return rope.view(*rope.shape[:-1], 2, 2).unsqueeze(-4)


def rope_1d(dim: int, positions: Tensor, freqs_scaling: float = 1.0) -> Tensor:
    """Rotary table ``(L, 1, dim/2, 2, 2)`` for 1-D positions."""
    freq = get_freqs(dim // 2).to(positions.device) * freqs_scaling
    return _rope_from_args(torch.outer(positions.float(), freq))


def rope_3d(axes_dims, positions, scale_factor) -> Tensor:
    """Rotary table ``(T, H, W, 1, sum(axes)/2, 2, 2)`` for a frame grid."""
    T, H, W = (p.shape[0] for p in positions)
    args = [
        torch.outer(pos.float(), get_freqs(d // 2).to(pos.device)) / s
        for pos, d, s in zip(positions, axes_dims, scale_factor)
    ]
    args = torch.cat(
        [
            args[0].view(T, 1, 1, -1).expand(T, H, W, -1),
            args[1].view(1, H, 1, -1).expand(T, H, W, -1),
            args[2].view(1, 1, W, -1).expand(T, H, W, -1),
        ],
        dim=-1,
    )
    return _rope_from_args(args)


def linear_fp32(layer, x: Tensor) -> Tensor:
    x = x.float()
    context = getattr(comfy.ops, "CastBiasWeightContext", None)
    if context is not None:
        with context(layer, x, dtype=torch.float32, bias_dtype=torch.float32, offloadable=True) as (weight, bias):
            return F.linear(x, weight, bias)
    weight, bias = comfy.ops.cast_bias_weight(layer, x, dtype=torch.float32, bias_dtype=torch.float32)
    return F.linear(x, weight, bias)


def attention(q: Tensor, k: Tensor, v: Tensor, heads: int, transformer_options=None) -> Tensor:
    return comfy_attention.optimized_attention(
        q.flatten(-2), k.flatten(-2), v.flatten(-2), heads,
        transformer_options=transformer_options or {},
    )


def _ops(operation_settings):
    return (
        operation_settings.get("operations"),
        operation_settings.get("device"),
        operation_settings.get("dtype"),
    )


# ---------------------------------------------------------------------------
# Small embedding / projection modules
# ---------------------------------------------------------------------------

class _TimestepEmbedder(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.linear_1 = ops.Linear(model_dim, time_dim, device=device, dtype=dtype)
        self.act = nn.SiLU()
        self.linear_2 = ops.Linear(time_dim, time_dim, device=device, dtype=dtype)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0, operation_settings=None):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.timestep_embedder = _TimestepEmbedder(model_dim, time_dim, operation_settings)

    def forward(self, time: Tensor, dtype: torch.dtype) -> Tensor:
        embedder = self.timestep_embedder
        freqs = get_freqs(self.model_dim // 2, self.max_period).to(time.device)
        args = torch.outer(time.float(), freqs)
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = linear_fp32(embedder.linear_1, embed)
        out = linear_fp32(embedder.linear_2, embedder.act(h))
        return out.to(dtype=dtype)


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim: int, model_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.in_layer = ops.Linear(text_dim, model_dim, device=device, dtype=dtype)
        self.norm = ops.LayerNorm(model_dim, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.in_layer(x))


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.patch_size = patch_size
        self.in_layer = ops.Linear(math.prod(patch_size) * visual_dim, model_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, T, H, W, C = x.shape
        pT, pH, pW = self.patch_size
        x = (
            x.view(B, T // pT, pT, H // pH, pH, W // pW, pW, C)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.act = nn.SiLU()
        self.out_layer = ops.Linear(time_dim, num_params * model_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return linear_fp32(self.out_layer, self.act(x.float())).to(dtype=x.dtype)


class _GELUProjection(nn.Module):
    def __init__(self, dim: int, ff_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.proj = ops.Linear(dim, ff_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.net = nn.ModuleList([
            _GELUProjection(dim, ff_dim, operation_settings),
            nn.Dropout(0.0),
            ops.Linear(ff_dim, dim, bias=False, device=device, dtype=dtype),
        ])

    def forward(self, x: Tensor) -> Tensor:
        for module in self.net:
            x = module(x)
        return x


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class MultiheadCrossAttention(nn.Module):
    def __init__(self, q_dim: int, head_dim: int, kv_dim: int | None = None, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        kv_dim = kv_dim or q_dim
        self.num_heads = q_dim // head_dim
        self.to_query = ops.Linear(q_dim, q_dim, device=device, dtype=dtype)
        self.to_key = ops.Linear(kv_dim, q_dim, device=device, dtype=dtype)
        self.to_value = ops.Linear(kv_dim, q_dim, device=device, dtype=dtype)
        self.query_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.key_norm = ops.RMSNorm(head_dim, device=device, dtype=dtype)
        self.out_layer = ops.Linear(q_dim, q_dim, device=device, dtype=dtype)

    def forward(self, x: Tensor, cond: Tensor, rope_q: Tensor | None = None, rope_kv: Tensor | None = None,
                transformer_options=None) -> Tensor:
        q = self.to_query(x).unflatten(-1, (self.num_heads, -1))
        k = self.to_key(cond).unflatten(-1, (self.num_heads, -1))
        v = self.to_value(cond).unflatten(-1, (self.num_heads, -1))
        q = self.query_norm(q)
        k = self.key_norm(k)
        if rope_q is not None:
            q = apply_rotary(q, rope_q)
        if rope_kv is not None:
            k = apply_rotary(k, rope_kv)
        return self.out_layer(attention(q, k, v, self.num_heads, transformer_options))


class MultiheadSelfAttentionEnc(MultiheadCrossAttention):
    def __init__(self, dim: int, head_dim: int, operation_settings=None):
        super().__init__(dim, head_dim, operation_settings=operation_settings)

    def forward(self, x: Tensor, rope: Tensor, transformer_options=None) -> Tensor:
        return super().forward(x, x, rope, rope, transformer_options)


class MultiheadSelfAttentionDec(MultiheadSelfAttentionEnc):
    pass


# ---------------------------------------------------------------------------
# Output layers
# ---------------------------------------------------------------------------

class OutLayer(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2, operation_settings)
        self.norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.out_layer = ops.Linear(model_dim, math.prod(patch_size) * visual_dim, device=device, dtype=dtype)

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, visual_embed, scale, shift).type_as(visual_embed)
        x = self.out_layer(x)
        batch, T, H, W, _ = x.shape
        pT, pH, pW = self.patch_size
        return (
            x.view(batch, T, H, W, -1, pT, pH, pW)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )


class OutLayerAudio(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, audio_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.modulation = Modulation(time_dim, model_dim, 2, operation_settings)
        self.norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.out_layer = ops.Linear(model_dim, audio_dim, device=device, dtype=dtype)

    def forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, audio_embed, scale, shift).type_as(audio_embed)
        x = self.norm(x)
        return self.out_layer(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.text_modulation = Modulation(time_dim, model_dim, 6, operation_settings)
        self.attn_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttentionEnc(model_dim, head_dim, operation_settings)
        self.feed_forward_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, transformer_options=None) -> Tensor:
        sa_p, ff_p = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        out = self.attn(apply_scale_shift_norm(self.attn_norm, x, scale, shift), rope, transformer_options)
        x = apply_gate_sum(x, out, gate)
        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        out = self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift))
        return apply_gate_sum(x, out, gate)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, operation_settings=None):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.visual_modulation = Modulation(time_dim, model_dim, 9, operation_settings)
        self.self_attention_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, operation_settings)
        self.cross_attention_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, operation_settings=operation_settings)
        self.feed_forward_norm = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings)


class FusedTransformerDecoderBlock(nn.Module):
    """Video and audio blocks joined by cross-modal attention."""

    def __init__(
        self,
        model_dim: int, time_dim: int, ff_dim: int, head_dim: int,
        model_dim_a: int, time_dim_a: int, ff_dim_a: int, head_dim_a: int,
        ca_rope: bool = False, cross_gates: bool = False, fix_modulation: bool = False,
        operation_settings=None,
    ):
        super().__init__()
        ops, device, dtype = _ops(operation_settings)
        self.video_dec_block = TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, operation_settings)
        self.audio_dec_block = TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a, operation_settings)

        self.va_cross_attention = MultiheadCrossAttention(model_dim, head_dim, model_dim_a, operation_settings)
        self.av_cross_attention = MultiheadCrossAttention(model_dim_a, head_dim_a, model_dim, operation_settings)

        self.va_modulation = Modulation(
            time_dim, model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3, operation_settings,
        )
        self.av_modulation = Modulation(
            time_dim_a, model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3, operation_settings,
        )
        self.va_normalization = ops.LayerNorm(model_dim, elementwise_affine=False, device=device, dtype=dtype)
        self.av_normalization = ops.LayerNorm(model_dim_a, elementwise_affine=False, device=device, dtype=dtype)

        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(self, vis, aud, text_v, text_a, time_embed, vis_rope, aud_rope, transformer_options=None):
        t_v, t_a = time_embed
        video_block, audio_block = self.video_dec_block, self.audio_dec_block

        # ---- video backbone ----
        sa_p, ca_p, ff_p = torch.chunk(video_block.visual_modulation(t_v), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        out = video_block.self_attention(
            apply_scale_shift_norm(video_block.self_attention_norm, vis, scale, shift), vis_rope, transformer_options
        )
        vis = apply_gate_sum(vis, out, gate).type_as(vis)
        shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
        vis_pre_ca = apply_scale_shift_norm(video_block.cross_attention_norm, vis, scale, shift).type_as(vis)
        vis_out_t = video_block.cross_attention(vis_pre_ca, text_v, transformer_options=transformer_options)

        # ---- audio backbone ----
        sa_p, ca_p, ff_p_a = torch.chunk(audio_block.visual_modulation(t_a), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        out = audio_block.self_attention(
            apply_scale_shift_norm(audio_block.self_attention_norm, aud, scale, shift), aud_rope, transformer_options
        )
        aud = apply_gate_sum(aud, out, gate).type_as(aud)
        shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
        aud_pre_ca = apply_scale_shift_norm(audio_block.cross_attention_norm, aud, scale, shift).type_as(aud)
        aud_out_t = audio_block.cross_attention(aud_pre_ca, text_a, transformer_options=transformer_options)
        aud = apply_gate_sum(aud, aud_out_t, gate_a).type_as(aud)

        # ---- cross-modal attention ----
        t_va_mod = t_a if not self.fix_modulation else t_v
        t_av_mod = t_v if not self.fix_modulation else t_a
        va_params = self.va_modulation(t_va_mod)
        av_params = self.av_modulation(t_av_mod)
        if self.cross_gates:
            va_shift, va_scale, va_gate = torch.split(va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1)
            av_shift, av_scale, av_gate = torch.split(av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1)
        else:
            va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
            av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

        vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)
        vis_for_va = apply_scale_shift_norm(self.va_normalization, vis, va_scale, va_shift).type_as(vis)
        aud_for_av = apply_scale_shift_norm(self.av_normalization, aud, av_scale, av_shift).type_as(aud)

        rq_v = vis_rope if self.ca_rope else None
        rk_a = aud_rope if self.ca_rope else None
        vis_from_aud = self.va_cross_attention(vis_for_va, aud_pre_ca, rope_q=rq_v, rope_kv=rk_a,
                                               transformer_options=transformer_options)
        aud_from_vis = self.av_cross_attention(aud_for_av, vis_pre_ca, rope_q=rk_a, rope_kv=rq_v,
                                               transformer_options=transformer_options)

        va_g = va_gate if not self.cross_gates else av_gate
        av_g = av_gate if not self.cross_gates else va_gate
        vis = apply_gate_sum(vis, vis_from_aud, va_g).type_as(vis)
        aud = apply_gate_sum(aud, aud_from_vis, av_g).type_as(aud)

        # ---- FFN ----
        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        out = video_block.feed_forward(apply_scale_shift_norm(video_block.feed_forward_norm, vis, scale, shift))
        vis = apply_gate_sum(vis, out, gate).type_as(vis)

        shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
        out = audio_block.feed_forward(apply_scale_shift_norm(audio_block.feed_forward_norm, aud, scale, shift))
        aud = apply_gate_sum(aud, out, gate).type_as(aud)
        return vis, aud


# ---------------------------------------------------------------------------
# Joint video and audio transformer
# ---------------------------------------------------------------------------

class DiffusionTransformer3D(nn.Module):
    """Kandinsky 6 DiT predicting video and audio velocities together."""

    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 1024,
        patch_size: tuple = (1, 2, 2),
        model_dim: int = 4096,
        ff_dim: int = 16384,
        num_text_blocks: int = 4,
        num_visual_blocks: int = 60,
        axes_dims: tuple = (32, 48, 48),
        visual_cond: bool = True,
        in_audio_dim: int = 40,
        out_audio_dim: int | None = None,
        model_dim_a: int | None = None,
        time_dim_a: int | None = None,
        ff_dim_a: int | None = None,
        axes_dims_a: tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        scale_factor: tuple = (1.0, 2.0, 2.0),
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        visual_token_type_num_embeddings: int = 0,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        operations = operations or comfy.ops.disable_weight_init
        op = {"operations": operations, "device": device, "dtype": dtype}
        self.dtype = dtype
        self.patch_size = tuple(patch_size)
        self.visual_cond = visual_cond
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        self.axes_dims = tuple(axes_dims)
        self.scale_factor = tuple(float(s) for s in scale_factor)
        self.audio_freqs_scaling = float(audio_freqs_scaling)
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)

        head_dim = sum(axes_dims)
        model_dim_a = model_dim_a or model_dim
        time_dim_a = time_dim_a or time_dim
        ff_dim_a = ff_dim_a or ff_dim
        axes_dims_a = axes_dims_a or axes_dims
        head_dim_a = sum(axes_dims_a)
        self.head_dim = head_dim
        self.head_dim_a = head_dim_a

        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings = VisualEmbeddings(vis_in_dim, model_dim, self.patch_size, op)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = operations.Embedding(
                self.visual_token_type_num_embeddings, model_dim, device=device, dtype=dtype
            )
        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, self.patch_size, op)

        self.audio_embeddings = TextEmbeddings(in_audio_dim, model_dim_a, op)
        self.audio_out_layer = OutLayerAudio(model_dim_a, time_dim_a, out_audio_dim or in_audio_dim, op)

        for prefix, md, td, fd, hd in [
            ("video", model_dim, time_dim, ff_dim, head_dim),
            ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
        ]:
            setattr(self, f"{prefix}_time_embeddings", TimeEmbeddings(md, td, operation_settings=op))
            setattr(self, f"{prefix}_text_embeddings", TextEmbeddings(in_text_dim, md, op))
            setattr(self, f"{prefix}_pooled_text_embeddings", TextEmbeddings(in_text_dim2, td, op))
            setattr(self, f"{prefix}_text_transformer_blocks", nn.ModuleList([
                TransformerEncoderBlock(md, td, fd, hd, op) for _ in range(num_text_blocks)
            ]))

        self.visual_transformer_blocks = nn.ModuleList([
            FusedTransformerDecoderBlock(
                model_dim, time_dim, ff_dim, head_dim,
                model_dim_a, time_dim_a, ff_dim_a, head_dim_a,
                ca_rope=ca_rope, cross_gates=cross_gates, fix_modulation=fix_modulation,
                operation_settings=op,
            )
            for _ in range(num_visual_blocks)
        ])

    def _encode_text(self, prefix: str, text_embed: Tensor, pooled: Tensor, time: Tensor, dtype, transformer_options):
        te = getattr(self, f"{prefix}_text_embeddings")(text_embed)
        pe = getattr(self, f"{prefix}_pooled_text_embeddings")(pooled)
        tm = getattr(self, f"{prefix}_time_embeddings")(time, dtype) + pe
        head_dim = self.head_dim if prefix == "video" else self.head_dim_a
        rope = rope_1d(head_dim, torch.arange(te.shape[1], device=te.device))
        for block in getattr(self, f"{prefix}_text_transformer_blocks"):
            te = block(te, tm, rope, transformer_options)
        return te, tm

    def visual_rope(self, frames: int, height: int, width: int, device, reference_tail: bool = False) -> Tensor:
        """Rotary table for a patch grid, with an optional reference frame reusing position 0."""
        positions = [torch.arange(n, device=device) for n in (frames, height, width)]
        rope = rope_3d(self.axes_dims, positions, self.scale_factor)
        if reference_tail:
            rope = torch.cat([rope, rope[:1]], dim=0)
        return rope.flatten(0, 2)

    def run_blocks(self, vis, aud, video_te, audio_te, video_tm, audio_tm, vis_rope, aud_rope, transformer_options):
        for block in self.visual_transformer_blocks:
            vis, aud = block(vis, aud, video_te, audio_te, (video_tm, audio_tm), vis_rope, aud_rope, transformer_options)
        return vis, aud

    def forward(
        self,
        x_video: Tensor,
        x_audio: Tensor,
        text_embed: Tensor,
        pooled_text_embed: Tensor,
        time: Tensor,
        visual_token_type_ids: Tensor | None = None,
        reference_tail: bool = False,
        transformer_options=None,
    ) -> tuple[Tensor, Tensor]:
        """Video ``(B, T, H, W, 2C+1)`` and audio ``(B, L, C)`` to their velocities."""
        transformer_options = {} if transformer_options is None else transformer_options
        dtype = x_video.dtype
        video_te, video_tm = self._encode_text("video", text_embed, pooled_text_embed, time, dtype, transformer_options)
        audio_te, audio_tm = self._encode_text("audio", text_embed, pooled_text_embed, time, dtype, transformer_options)

        vis_embed = self.visual_embeddings(x_video)
        if hasattr(self, "visual_token_type_embeddings") and visual_token_type_ids is not None:
            type_embed = self.visual_token_type_embeddings(visual_token_type_ids.to(device=vis_embed.device))
            vis_embed = vis_embed + type_embed[None, :, None, None, :].to(dtype=vis_embed.dtype)
        vis_shape = vis_embed.shape[1:-1]
        frames = vis_shape[0] - (1 if reference_tail else 0)
        vis_rope = self.visual_rope(frames, vis_shape[1], vis_shape[2], vis_embed.device, reference_tail)
        vis_embed = vis_embed.flatten(1, 3)

        aud_embed = self.audio_embeddings(x_audio)
        aud_rope = rope_1d(
            self.head_dim_a, torch.arange(aud_embed.shape[1], device=aud_embed.device), self.audio_freqs_scaling
        )

        vis_embed, aud_embed = self.run_blocks(
            vis_embed, aud_embed, video_te, audio_te, video_tm, audio_tm, vis_rope, aud_rope, transformer_options
        )

        vis_embed = vis_embed.reshape(vis_embed.shape[0], *vis_shape, vis_embed.shape[-1])
        return self.out_layer(vis_embed, video_tm), self.audio_out_layer(aud_embed, audio_tm)
