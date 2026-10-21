"""Fibo 2.0's diffusion transformer on ComfyUI's Z-Image layers, in Comfy-Org's file layout.

Image tokens are the FLUX.2 latent, one per position. The text is ``(B, L, 6 * 7680)``: the
Perceiver's bundle, then one per injection block.
"""

from __future__ import annotations

import math

import torch
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.lumina.model import FinalLayer, JointAttention, apply_gate, clamp_fp16, modulate
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder

#: The Qwen3-VL layers behind each text bundle: the Perceiver's, then one per injection block.
BUNDLES = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

#: Every Qwen3-VL layer a bundle reads, in order.
TEXT_LAYERS = tuple(sorted({layer for bundle in BUNDLES for layer in bundle}))

#: Qwen3-VL's hidden size, and one bundle's width.
TEXT_WIDTH = 2560
BUNDLE_WIDTH = 3 * TEXT_WIDTH

#: Gist tokens: the fewest, the step they rise by, and the text lengths the rise spans.
MIN_GIST = 256
GIST_STEP = 64
SHORT_TEXT, LONG_TEXT = 256, 2048

#: Width of the timestep embedding every modulation reads, and the timestep's scale.
ADALN_WIDTH = 256
T_SCALE = 1000.0


def gist_count(text_length: int, largest: int) -> int:
    """Gist tokens for a text of ``text_length`` tokens."""
    ratio = min(1.0, max(0.0, (text_length - SHORT_TEXT) / (LONG_TEXT - SHORT_TEXT)))
    return max(MIN_GIST, min(largest, math.ceil((MIN_GIST + ratio * (largest - MIN_GIST)) / GIST_STEP) * GIST_STEP))


def residual(x, y, gate=None):
    """``x + gate * y``, in place outside training."""
    if comfy.model_management.in_training:
        return x + y if gate is None else torch.addcmul(x, y, gate)
    return x.add_(y) if gate is None else x.addcmul_(y, gate)


class TextAttention(nn.Module):
    """Attention without position, to the text or among gist tokens: ``to_q``, ``to_kv``, ``to_out``."""

    def __init__(self, dim: int, heads: int, kv_dim: int, norm_eps: float, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.heads = heads
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_kv = operations.Linear(kv_dim, 2 * dim, bias=False, dtype=dtype, device=device)
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(dim // heads, eps=norm_eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim // heads, eps=norm_eps, dtype=dtype, device=device)

    def keys_values(self, context: torch.Tensor) -> torch.Tensor:
        """Normalised keys beside values for ``context``, ``(B, L, 2, heads, head_dim)``."""
        key, value = self.to_kv(context).unflatten(-1, (2, self.heads, -1)).unbind(2)
        return torch.stack([self.norm_k(key), value], dim=2)

    def forward(self, x, context=None, kv=None, transformer_options={}):
        query = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        key, value = (kv if kv is not None else self.keys_values(context)).to(query.dtype).unbind(2)
        query, key, value = (AttentionTensorContainer(t.movedim(1, 2)) for t in (query, key, value))
        return self.to_out(optimized_attention(query, key, value, self.heads, skip_reshape=True,
                                               transformer_options=transformer_options,
                                               preferred_attention=self.comfy_attention))


class FeedForward(nn.Module):
    """SwiGLU with the gate and up projections fused as ``w13``, the hidden width padded."""

    def __init__(self, dim: int, hidden: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.w13 = operations.Linear(dim, 2 * hidden, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.w2, self.w13(x), "swiglu")


class Block(nn.Module):
    """Z-Image's block; an injection block also reads its own text bundle between attention and feed-forward."""

    def __init__(self, dim, heads, ffn_dim, norm_eps, text_dim=None, operation_settings={}):
        super().__init__()
        ops, dtype, device = (operation_settings[key] for key in ("operations", "dtype", "device"))
        self.attention = JointAttention(dim, heads, None, True, operation_settings=operation_settings)
        self.attention.q_norm.eps = self.attention.k_norm.eps = norm_eps
        self.feed_forward = FeedForward(dim, ffn_dim, dtype=dtype, device=device, operations=ops)
        self.attention_norm1 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm1 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.attention_norm2 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm2 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(ops.Linear(ADALN_WIDTH, 4 * dim, bias=True, dtype=dtype, device=device))
        self.cross_attention = None
        if text_dim is not None:
            self.cross_attn_norm1 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attention = TextAttention(dim, heads, text_dim, norm_eps, dtype=dtype, device=device, operations=ops)
            self.cross_attn_norm2 = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attn_gate = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

    def forward(self, x, freqs_cis, adaln, clean=None, text_kv=None, transformer_options={}):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln).chunk(4, dim=1)
        y = self.attention(modulate(self.attention_norm1(x), scale_msa, timestep_zero_index=clean), None, freqs_cis,
                           transformer_options=transformer_options)
        x = residual(x, apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(clamp_fp16(y)),
                                   timestep_zero_index=clean))
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), kv=text_kv, transformer_options=transformer_options)
            x = residual(x, self.cross_attn_norm2(clamp_fp16(y)), comfy.ops.cast_to_input(self.cross_attn_gate, y).tanh())
        y = self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp, timestep_zero_index=clean))
        return residual(x, apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(clamp_fp16(y)),
                                      timestep_zero_index=clean))


class PerceiverLayer(nn.Module):
    """Gist tokens read the text, then each other, then a feed-forward."""

    def __init__(self, dim, heads, ffn_dim, text_dim, norm_eps, operation_settings={}):
        super().__init__()
        ops, dtype, device = (operation_settings[key] for key in ("operations", "dtype", "device"))
        self.cross_attn_norm = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.cross_attention = TextAttention(dim, heads, text_dim, norm_eps, dtype=dtype, device=device, operations=ops)
        self.self_attn_norm = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.self_attention = TextAttention(dim, heads, dim, norm_eps, dtype=dtype, device=device, operations=ops)
        self.ffn_norm = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn = FeedForward(dim, ffn_dim, dtype=dtype, device=device, operations=ops)

    def forward(self, x, text, transformer_options={}):
        x = x + self.cross_attention(self.cross_attn_norm(x), text, transformer_options=transformer_options)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y, transformer_options=transformer_options)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    """Condenses the first text bundle into gist tokens."""

    def __init__(self, dim, heads, ffn_dim, text_dim, queries, layers, norm_eps, operation_settings={}):
        super().__init__()
        ops, dtype, device = (operation_settings[key] for key in ("operations", "dtype", "device"))
        self.latent_queries = nn.Parameter(torch.empty(queries, dim, dtype=dtype, device=device))
        self.layers = nn.ModuleList([PerceiverLayer(dim, heads, ffn_dim, text_dim, norm_eps, operation_settings)
                                     for _ in range(layers)])
        self.norm_out = ops.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, text, count, transformer_options={}):
        x = comfy.ops.cast_to_input(self.latent_queries[:count], text).unsqueeze(0).expand(text.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, text, transformer_options=transformer_options)
        return self.norm_out(x)


class Fibo2(nn.Module):
    """Fibo 2.0's single-stream DiT: gist tokens, the noisy image and any images to edit in one sequence."""

    #: The ComfyUI module whose weight layout and quantization rules this model shares.
    comfy_family = "comfy.ldm.fibo2.model"

    def __init__(self, in_channels=128, dim=3584, n_heads=28, n_layers=44, n_refiner_layers=2,
                 injection_layers=(6, 14, 22, 30, 38), cap_feat_dim=BUNDLE_WIDTH, perceiver_layers=2,
                 max_gist_tokens=512, ffn_dim=9728, norm_eps=1e-5, image_model=None, dtype=None, device=None,
                 operations=None, **kwargs):
        super().__init__()
        settings = {"operations": operations, "device": device, "dtype": dtype}
        self.dtype = dtype
        self.in_channels = in_channels
        self.max_gist_tokens = max_gist_tokens
        self.injection_layers = tuple(injection_layers)
        self.x_embedder = operations.Linear(in_channels, dim, bias=True, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(1024, output_size=ADALN_WIDTH, **settings)
        self.noise_refiner = nn.ModuleList([Block(dim, n_heads, ffn_dim, norm_eps, operation_settings=settings)
                                            for _ in range(n_refiner_layers)])
        self.perceiver = Perceiver(dim, n_heads, ffn_dim, cap_feat_dim, max_gist_tokens, perceiver_layers, norm_eps,
                                   operation_settings=settings)
        self.layers = nn.ModuleList([
            Block(dim, n_heads, ffn_dim, norm_eps, cap_feat_dim if index in self.injection_layers else None,
                  operation_settings=settings)
            for index in range(n_layers)
        ])
        self.final_layer = FinalLayer(dim, 1, in_channels, z_image_modulation=True, operation_settings=settings)
        head = dim // n_heads
        self.rope_embedder = EmbedND(dim=head, theta=256.0, axes_dim=[head // 4, head * 3 // 8, head * 3 // 8])

    def preprocess_conds(self, context: torch.Tensor, transformer_options={}):
        """What depends on the text alone: the gist tokens, and each injection block's keys and values.

        Args:
            context: The text, ``(B, L, 6 * 7680)``.
            transformer_options: ComfyUI's options for this call.

        Returns:
            ``(gist, text_kv)``: ``(B, G, dim)`` and ``(B, blocks, L, 2, heads, head_dim)``.
        """
        bundles = context.chunk(1 + len(self.injection_layers), dim=-1)
        gist = self.perceiver(bundles[0], gist_count(context.shape[1], self.max_gist_tokens), transformer_options)
        text_kv = torch.stack([self.layers[layer].cross_attention.keys_values(bundle)
                               for layer, bundle in zip(self.injection_layers, bundles[1:])], dim=1)
        return gist, text_kv

    def positions(self, gist: int, shapes, device) -> torch.Tensor:
        """RoPE ids: gist token ``i`` at ``(1 + i, 0, 0)``, each image on its own plane past the gist."""
        plane = self.max_gist_tokens + 1
        zeros = torch.zeros(gist, dtype=torch.long, device=device)
        ids = [torch.stack([torch.arange(1, gist + 1, device=device), zeros, zeros], dim=-1)]
        for index, (height, width) in enumerate(shapes):
            rows = torch.arange(height, device=device).repeat_interleave(width)
            columns = torch.arange(width, device=device).repeat(height)
            ids.append(torch.stack([torch.full_like(rows, plane + index), rows, columns], dim=-1))
        return torch.cat(ids, dim=0).float().unsqueeze(0)

    def forward(self, x, timestep, context=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                                     transformer_options),
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context=None, transformer_options={}, ref_latents=None, **kwargs):
        batch, channels, height, width = x.shape
        dtype, device = x.dtype, x.device
        if channels != self.in_channels:
            raise ValueError(
                f"Fibo 2.0 samples a FLUX.2 latent of {self.in_channels} channels and was given one of {channels}. "
                "Start from the latent Text Encode Fibo 2 hands on, or encode with the FLUX.2 VAE."
            )
        if context is None:
            raise ValueError("Fibo 2.0 needs a prompt from Text Encode Fibo 2.")
        transformer_options = transformer_options.copy()
        context = comfy.utils.repeat_to_batch_size(context, batch).to(dtype)
        gist, text_kv = self.preprocess_conds(context, transformer_options)

        # The noisy image reads the first half of the modulation batch, every clean token the second.
        adaln = self.t_embedder(torch.cat((timestep * T_SCALE, torch.zeros_like(timestep))), dtype)
        noisy = adaln[:batch]

        tokens = [x.flatten(2).transpose(1, 2)]
        shapes = [(height, width)]
        for latent in ref_latents or []:
            tokens.append(comfy.utils.repeat_to_batch_size(latent, batch).to(x).flatten(2).transpose(1, 2))
            shapes.append(tuple(latent.shape[-2:]))
        image = height * width
        img = self.x_embedder(torch.cat(tokens, dim=1))
        del tokens
        span = img.shape[1]
        count = gist.shape[1]
        freqs = self.rope_embedder(self.positions(count, shapes, device)).movedim(1, 2)

        references = [(image, span)] if span > image else None
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.noise_refiner), device, transformer_options)
        for block in self.noise_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, device, block, dtype)
            img = block(img, freqs[:, count:], adaln if references else noisy, references,
                        transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, device, None)

        x = torch.cat((gist, img), dim=1)
        del img
        clean = [(0, count)] + ([(count + image, count + span)] if references else [])
        replace = transformer_options.get("patches_replace", {}).get("dit", {})
        injected = {layer: index for index, layer in enumerate(self.injection_layers)}
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for index, block in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = index
            own_kv = text_kv[:, injected[index]] if index in injected else None
            if ("single_block", index) in replace:
                def wrapped(args, block=block, own_kv=own_kv):
                    return {"img": block(args["img"], args["pe"], args["vec"], clean, own_kv, args["transformer_options"])}

                x = replace[("single_block", index)](
                    {"img": x, "vec": adaln, "pe": freqs, "transformer_options": transformer_options},
                    {"original_block": wrapped},
                )["img"]
            else:
                x = block(x, freqs, adaln, clean, own_kv, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.final_layer(x[:, count:count + image], noisy)
        return out.transpose(1, 2).reshape(batch, channels, height, width)
