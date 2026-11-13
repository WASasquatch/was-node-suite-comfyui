"""EmbeddingGemma 2's text and vision towers and its picture preprocessing, for inference.

Module and parameter names follow the checkpoint's ``model.safetensors``. Text is embedded from
token ids, a picture from 16 pixel patches; both come out as one 768 number vector per input.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EmbeddingGemma2", "picture_patches", "read_config"]

#: Token ids the picture sequence is built from.
BOS = 2
EOS = 1
PAD = 0


def read_config(folder: str) -> dict:
    """The checkpoint folder's ``config.json``."""
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _gelu(hidden: torch.Tensor) -> torch.Tensor:
    return F.gelu(hidden, approximate="tanh")


def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    half = hidden.shape[-1] // 2
    return torch.cat((-hidden[..., half:], hidden[..., :half]), dim=-1)


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Unscaled attention, key and value heads repeated to the query heads.

    Args:
        query: ``(batch, heads, length, head_dim)``.
        key: ``(batch, kv_heads, length, head_dim)``.
        value: ``(batch, kv_heads, length, head_dim)``.
        mask: Boolean ``(batch, 1, length, length)``, True where a query may read a key.
    """
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=1.0)


class RMSNorm(nn.Module):
    """Root mean square normalisation computed in float32, with or without a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = hidden.float()
        normed = normed * torch.pow(normed.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.type_as(hidden)


class _Linear(nn.Module):
    """A bias-free linear layer held under ``.linear``, as the vision tower's weights name it."""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class _MLP(nn.Module):
    """Gated feed-forward block with the tanh GELU."""

    def __init__(self, hidden: int, inner: int, wrapped: bool = False):
        super().__init__()
        make = _Linear if wrapped else (lambda inputs, outputs: nn.Linear(inputs, outputs, bias=False))
        self.gate_proj = make(hidden, inner)
        self.up_proj = make(hidden, inner)
        self.down_proj = make(inner, hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_gelu(self.gate_proj(hidden)) * self.up_proj(hidden))


# ---------------------------------------------------------------------- text tower


class TextAttention(nn.Module):
    """Bidirectional attention with normalised queries, keys and values and rotary positions."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.v_norm = RMSNorm(head_dim, eps, with_scale=False)

    def forward(self, hidden, cos, sin, mask):
        batch, length, _ = hidden.shape
        shape = (batch, length, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden).view(shape).transpose(1, 2))
        key = self.k_norm(self.k_proj(hidden).view(shape).transpose(1, 2))
        value = self.v_norm(self.v_proj(hidden).view(shape).transpose(1, 2))
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        out = _attend(query, key, value, mask)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class PLEBlock(nn.Module):
    """Mixes one layer's slice of the per-layer embeddings into the residual stream."""

    def __init__(self, hidden: int, per_layer: int, eps: float):
        super().__init__()
        self.per_layer_input_gate = nn.Linear(hidden, per_layer, bias=False)
        self.per_layer_projection = nn.Linear(per_layer, hidden, bias=False)
        self.post_per_layer_input_norm = RMSNorm(hidden, eps)

    def forward(self, hidden, per_layer_input):
        mixed = _gelu(self.per_layer_input_gate(hidden)) * per_layer_input
        return hidden + self.post_per_layer_input_norm(self.per_layer_projection(mixed))


class TextLayer(nn.Module):
    """Attention, feed-forward and per-layer embedding blocks, each wrapped in its norms."""

    def __init__(self, config: dict, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.self_attn = TextAttention(hidden, heads, kv_heads, head_dim, eps)
        self.mlp = _MLP(hidden, config["intermediate_size"])
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)
        self.ple_block = PLEBlock(hidden, config["hidden_size_per_layer_input"], eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden, per_layer_input, cos, sin, mask):
        hidden = hidden + self.post_attention_layernorm(self.self_attn(self.input_layernorm(hidden), cos, sin, mask))
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        hidden = self.ple_block(hidden, per_layer_input)
        return hidden * self.layer_scalar


class PLE(nn.Module):
    """The per-layer embeddings, projected from the input embeddings alone."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.width = config["hidden_size_per_layer_input"]
        self.scale = hidden ** -0.5
        self.per_layer_model_projection = nn.Linear(hidden, self.layers * self.width, bias=False)
        self.per_layer_projection_norm = RMSNorm(self.width, config["rms_norm_eps"])

    def forward(self, embeds):
        projected = self.per_layer_model_projection(embeds) * self.scale
        return self.per_layer_projection_norm(projected.reshape(*embeds.shape[:-1], self.layers, self.width))


class TextModel(nn.Module):
    """The encoder every input runs through, ending in a projection to the embedding width."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        self.hidden = hidden
        self.layer_types = list(config["layer_types"])
        self.sliding_window = int(config["sliding_window"])
        overrides = {int(index): value for index, value in (config.get("per_layer_config") or {}).items()}
        self.head_dims = {}
        layers = []
        for index, kind in enumerate(self.layer_types):
            override = overrides.get(index, {})
            head_dim = int(override.get("head_dim", config["head_dim"]))
            kv_heads = int(override.get("num_key_value_heads", config["num_key_value_heads"]))
            self.head_dims[kind] = head_dim
            layers.append(TextLayer(config, config["num_attention_heads"], kv_heads, head_dim))
        self.embed_tokens = nn.Embedding(config["vocab_size"], hidden, config.get("pad_token_id", PAD))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden, config["rms_norm_eps"])
        self.ple = PLE(config)
        self.embedding_projection = nn.Linear(hidden, config["embedding_dim"], bias=False)
        self.rope_theta = {kind: float(value["rope_theta"]) for kind, value in config["rope_parameters"].items()}

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings scaled by the square root of the width, in the weights' precision."""
        scale = torch.tensor(self.hidden ** 0.5, dtype=torch.float32).to(self.embed_tokens.weight.dtype)
        return self.embed_tokens(ids) * scale

    def _rotary(self, kind: str, length: int, device, dtype):
        dim = self.head_dims[kind]
        inverse = 1.0 / (self.rope_theta[kind] ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None]
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def forward(self, embeds: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Token vectors at the embedding width.

        Args:
            embeds: ``(batch, length, hidden)`` input embeddings.
            valid: Boolean ``(batch, length)``, False on padding.
        """
        batch, length, _ = embeds.shape
        device = embeds.device
        per_layer = self.ple(embeds)
        keys = valid[:, None, None, :].expand(batch, 1, length, length)
        offsets = torch.arange(length, device=device)
        near = (offsets[:, None] - offsets[None, :]).abs() <= self.sliding_window
        masks = {"full_attention": keys, "sliding_attention": keys & near[None, None]}
        rotary = {kind: self._rotary(kind, length, device, embeds.dtype) for kind in set(self.layer_types)}
        hidden = embeds
        for index, layer in enumerate(self.layers):
            kind = self.layer_types[index]
            cos, sin = rotary[kind]
            hidden = layer(hidden, per_layer[:, :, index, :], cos, sin, masks[kind])
        return self.embedding_projection(self.norm(hidden))


# ---------------------------------------------------------------------- vision tower


class VisionAttention(nn.Module):
    """Bidirectional attention over patches with two-axis rotary positions."""

    def __init__(self, config: dict):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        kv_heads = config["num_key_value_heads"]
        self.q_proj = _Linear(hidden, self.heads * self.head_dim)
        self.k_proj = _Linear(hidden, kv_heads * self.head_dim)
        self.v_proj = _Linear(hidden, kv_heads * self.head_dim)
        self.o_proj = _Linear(self.heads * self.head_dim, hidden)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)
        self.v_norm = RMSNorm(self.head_dim, eps, with_scale=False)

    @staticmethod
    def _rope(hidden, cos, sin):
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        width = hidden.shape[-1] // 2
        parts = []
        for start in (0, width):
            piece = hidden[..., start:start + width]
            parts.append(piece * cos[..., start:start + width] + _rotate_half(piece) * sin[..., start:start + width])
        return torch.cat(parts, dim=-1)

    def forward(self, hidden, cos, sin, mask):
        batch, length, _ = hidden.shape
        shape = (batch, length, -1, self.head_dim)
        query = self._rope(self.q_norm(self.q_proj(hidden).view(shape)), cos, sin).transpose(1, 2)
        key = self._rope(self.k_norm(self.k_proj(hidden).view(shape)), cos, sin).transpose(1, 2)
        value = self.v_norm(self.v_proj(hidden).view(shape)).transpose(1, 2)
        out = _attend(query, key, value, mask)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class VisionLayer(nn.Module):
    """Attention and feed-forward blocks, each wrapped in its norms."""

    def __init__(self, config: dict):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.self_attn = VisionAttention(config)
        self.mlp = _MLP(hidden, config["intermediate_size"], wrapped=True)
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)

    def forward(self, hidden, cos, sin, mask):
        hidden = hidden + self.post_attention_layernorm(self.self_attn(self.input_layernorm(hidden), cos, sin, mask))
        return hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))


class PatchEmbedder(nn.Module):
    """Pixel patches projected to the vision width, plus learned x and y position tables."""

    def __init__(self, config: dict):
        super().__init__()
        patch = config["patch_size"]
        self.input_proj = nn.Linear(3 * patch * patch, config["hidden_size"], bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config["position_embedding_size"], config["hidden_size"])
        )

    def forward(self, pixels, positions, padding):
        hidden = self.input_proj((2 * (pixels - 0.5)).to(self.input_proj.weight.dtype))
        clamped = positions.clamp(min=0)
        placed = (
            F.embedding(clamped[..., 0], self.position_embedding_table[0])
            + F.embedding(clamped[..., 1], self.position_embedding_table[1])
        )
        return hidden + torch.where(padding.unsqueeze(-1), 0.0, placed)


class VisionEncoder(nn.Module):
    """The vision layers."""

    def __init__(self, config: dict):
        super().__init__()
        self.layers = nn.ModuleList(VisionLayer(config) for _ in range(config["num_hidden_layers"]))


class VisionModel(nn.Module):
    """Pixel patches to soft tokens: encoded, pooled in 3 by 3 patch cells and scaled."""

    def __init__(self, config: dict):
        super().__init__()
        self.hidden = config["hidden_size"]
        self.head_dim = config["head_dim"]
        self.kernel = config["pooling_kernel_size"]
        self.theta = float(config["rope_parameters"]["rope_theta"])
        self.patch_embedder = PatchEmbedder(config)
        self.encoder = VisionEncoder(config)

    def _rotary(self, positions, dtype):
        spatial = self.head_dim // 2
        inverse = 1.0 / (self.theta ** (torch.arange(0, spatial, 2, dtype=torch.float32, device=positions.device) / spatial))
        angles = positions[..., None].float() * inverse
        cos, sin = angles.cos(), angles.sin()
        cos = torch.cat([cos[:, :, 0], cos[:, :, 0], cos[:, :, 1], cos[:, :, 1]], dim=-1)
        sin = torch.cat([sin[:, :, 0], sin[:, :, 0], sin[:, :, 1], sin[:, :, 1]], dim=-1)
        return cos.to(dtype), sin.to(dtype)

    def _pool(self, hidden, positions, length):
        cells = self.kernel * self.kernel
        clamped = positions.clamp(min=0)
        widest = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        cell = torch.div(clamped, self.kernel, rounding_mode="floor")
        index = cell[..., 0] + (widest // self.kernel) * cell[..., 1]
        weights = F.one_hot(index.long(), length).float() / cells
        pooled = weights.transpose(1, 2) @ hidden.float()
        return pooled, torch.logical_not((weights == 0).all(dim=1))

    def forward(self, pixels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Every picture's soft tokens, joined along the first axis in picture order.

        Args:
            pixels: ``(batch, patches, 3 * 16 * 16)`` in ``[0, 1]``.
            positions: ``(batch, patches, 2)`` patch ``(x, y)``, ``-1`` on padding.
        """
        padding = (positions == -1).all(dim=-1)
        hidden = self.patch_embedder(pixels, positions, padding)
        batch, length, _ = hidden.shape
        mask = (~padding)[:, None, None, :].expand(batch, 1, length, length)
        cos, sin = self._rotary(positions, hidden.dtype)
        for layer in self.encoder.layers:
            hidden = layer(hidden, cos, sin, mask)
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        tokens = length // (self.kernel * self.kernel)
        pooled, kept = self._pool(hidden, positions, tokens)
        pooled = pooled * (self.hidden ** 0.5)
        return pooled[kept]


class Embedder(nn.Module):
    """Soft tokens normalised without a scale and projected to the text width."""

    def __init__(self, inputs: int, outputs: int, eps: float):
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(inputs, eps, with_scale=False)
        self.embedding_projection = nn.Linear(inputs, outputs, bias=False)

    def forward(self, tokens):
        return self.embedding_projection(self.embedding_pre_projection_norm(tokens))


class EmbeddingGemma2(nn.Module):
    """Text and pictures embedded into one space of ``embedding_dim`` numbers."""

    def __init__(self, config: dict):
        super().__init__()
        text, vision = config["text_config"], config["vision_config"]
        self.config = config
        self.language_model = TextModel(text)
        self.vision_tower = VisionModel(vision)
        self.embed_vision = Embedder(vision["hidden_size"], text["hidden_size"], vision["rms_norm_eps"])
        self.image_token = int(config["image_token_id"])
        self.boi_token = int(config["boi_token_id"])
        self.eoi_token = int(config["eoi_token_id"])

    @classmethod
    def load(cls, folder: str, dtype: torch.dtype = torch.float32) -> "EmbeddingGemma2":
        """The model with its weights read from ``model.safetensors``, the audio tower left out.

        Raises:
            ValueError: A weight the model needs is missing from the file.
        """
        from safetensors import safe_open

        model = cls(read_config(folder))
        wanted = model.state_dict()
        state = {}
        with safe_open(os.path.join(folder, "model.safetensors"), "pt") as handle:
            for name in handle.keys():
                if name in wanted:
                    state[name] = handle.get_tensor(name)
        missing = [name for name in wanted if name not in state]
        if missing:
            raise ValueError(f"EmbeddingGemma 2 weights are missing {len(missing)} tensors, such as {missing[0]}")
        model.load_state_dict(state)
        return model.to(dtype).eval()

    def _pooled(self, embeds: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = self.language_model(embeds, valid).float()
        weights = valid.unsqueeze(-1).float()
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)

    @torch.inference_mode()
    def embed_ids(self, ids: list[list[int]], device) -> torch.Tensor:
        """Unit vectors for token id sequences, BOS and EOS already placed."""
        length = max(len(row) for row in ids)
        batch = torch.full((len(ids), length), PAD, dtype=torch.long, device=device)
        valid = torch.zeros((len(ids), length), dtype=torch.bool, device=device)
        for row, sequence in enumerate(ids):
            batch[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            valid[row, :len(sequence)] = True
        return self._pooled(self.language_model.embed(batch), valid)

    @torch.inference_mode()
    def embed_pictures(self, pixels: torch.Tensor, positions: torch.Tensor, device) -> torch.Tensor:
        """Unit vectors for pictures from :func:`picture_patches`, stacked along the first axis."""
        dtype = self.language_model.embed_tokens.weight.dtype
        pixels = pixels.to(device=device, dtype=dtype)
        positions = positions.to(device)
        cells = self.vision_tower.kernel ** 2
        counts = ((positions != -1).all(dim=-1).sum(dim=-1) // cells).tolist()
        soft = self.embed_vision(self.vision_tower(pixels, positions).to(dtype))
        ids = [[BOS, self.boi_token] + [self.image_token] * count + [self.eoi_token, EOS] for count in counts]
        length = max(len(row) for row in ids)
        batch = torch.full((len(ids), length), PAD, dtype=torch.long, device=device)
        valid = torch.zeros((len(ids), length), dtype=torch.bool, device=device)
        for row, sequence in enumerate(ids):
            batch[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            valid[row, :len(sequence)] = True
        slots = batch == self.image_token
        embeds = self.language_model.embed(torch.where(slots, PAD, batch))
        embeds = embeds.masked_scatter(slots.unsqueeze(-1).expand_as(embeds), soft.to(embeds.dtype))
        return self._pooled(embeds, valid)


def _picture_size(height: int, width: int, patch: int, max_patches: int, kernel: int) -> tuple[int, int]:
    """The largest size keeping the aspect ratio within the patch budget, a multiple of ``kernel * patch``."""
    target = max_patches * patch * patch
    factor = math.sqrt(target / (height * width))
    side = kernel * patch
    rows = int(math.floor(factor * height / side)) * side
    columns = int(math.floor(factor * width / side)) * side
    longest = (max_patches // (kernel * kernel)) * side
    if rows == 0 and columns == 0:
        raise ValueError("the picture is too small to embed")
    if rows == 0:
        rows, columns = side, min(int(math.floor(width / height)) * side, longest)
    elif columns == 0:
        columns, rows = side, min(int(math.floor(height / width)) * side, longest)
    return rows, columns


def picture_patches(picture, patch: int = 16, soft_tokens: int = 280, kernel: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """A picture resized within the patch budget and cut into patches, padded to the budget.

    Args:
        picture: An RGB PIL picture.
        patch: Patch side in pixels.
        soft_tokens: Most soft tokens a picture becomes.
        kernel: Pooling cell side in patches.

    Returns:
        ``(pixels, positions)``: float32 ``(max_patches, 3 * patch * patch)`` in ``[0, 1]`` and
        long ``(max_patches, 2)`` patch ``(x, y)``, ``-1`` on padding.
    """
    from PIL import Image

    max_patches = soft_tokens * kernel * kernel
    picture = picture.convert("RGB")
    rows, columns = _picture_size(picture.height, picture.width, patch, max_patches, kernel)
    if (rows, columns) != (picture.height, picture.width):
        picture = picture.resize((columns, rows), Image.BICUBIC)
    image = torch.from_numpy(np.asarray(picture, dtype=np.uint8).copy()).permute(2, 0, 1)
    values = image.float() * (1.0 / 255.0)
    across, down = columns // patch, rows // patch
    pixels = values.reshape(3, down, patch, across, patch).permute(1, 3, 2, 4, 0).reshape(down * across, -1)
    grid_x, grid_y = torch.meshgrid(torch.arange(across), torch.arange(down), indexing="xy")
    positions = torch.stack((grid_x, grid_y), dim=-1).reshape(down * across, 2)
    pad = max_patches - pixels.shape[0]
    pixels = F.pad(pixels, (0, 0, 0, pad), value=0.0)
    positions = F.pad(positions, (0, 0, 0, pad), value=-1)
    return pixels, positions.long()
