"""Sapiens2 body-part segmentation: a vision transformer and the head that upsamples it.

The network takes ``(batch, 3, height, width)`` normalised RGB, sides a multiple of
:data:`PATCH`, and answers ``(batch, 29, height, width)`` logits named by :data:`CLASSES`.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional

from . import (
    ModelUnavailable,
    checkpoint_shapes,
    managed_module,
    model_directories,
    model_file_path,
    model_files,
    network_enabled,
    published_checkpoint,
    unresolved_links,
)

__all__ = [
    "CLASSES",
    "DEFAULT",
    "FEATURE",
    "FOLDER",
    "MEAN",
    "PATCH",
    "PUBLISHED",
    "READ_SIZE",
    "STD",
    "Network",
    "fits",
    "load",
    "locate",
    "offered",
]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for checkpoints.
FOLDER = "sapiens"

#: Published file -> repository holding it, smallest first.
PUBLISHED = {
    "sapiens2_0.4b_seg.safetensors": "facebook/sapiens2-seg-0.4b",
    "sapiens2_0.8b_seg.safetensors": "facebook/sapiens2-seg-0.8b",
    "sapiens2_1b_seg.safetensors": "facebook/sapiens2-seg-1b",
    "sapiens2_5b_seg.safetensors": "facebook/sapiens2-seg-5b",
}

#: The file a new node selects.
DEFAULT = "sapiens2_1b_seg.safetensors"

#: The 29 classes the head answers, in channel order.
CLASSES = (
    "Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand",
    "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm",
    "Left_Upper_Leg", "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg",
    "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth",
    "Tongue",
)

#: Height and width the network was trained at, and its patch side.
READ_SIZE = (1024, 768)
PATCH = 16

#: Normalisation the network was trained with, on a 0 to 1 scale.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Head channel widths after each upsampling step, and after the two pointwise steps.
UPSAMPLE = (512, 256, 128, 64)
POINTWISE = (64, 64)

#: Epsilon of every RMS norm.
NORM_EPSILON = 1e-6

#: Tensor whose shape marks a body-part checkpoint.
SIGNATURE = ("decode_head.conv_seg.weight", [len(CLASSES), POINTWISE[-1], 1, 1])


class _RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learned scale."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last axis in float32 and scale it."""
        held = x.float()
        held = held * torch.rsqrt(held.pow(2).mean(-1, keepdim=True) + NORM_EPSILON)
        return self.weight * held.to(x.dtype)


class _Scale(nn.Module):
    """A learned per-channel multiplier."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply the last axis."""
        return x * self.weight


class _Attention(nn.Module):
    """Self-attention with normalised queries and keys, rotary positions and grouped values."""

    def __init__(self, width: int, heads: int, groups: int):
        super().__init__()
        self.heads = heads
        self.groups = groups
        self.head_width = width // heads
        self.wq = nn.Linear(width, width)
        self.wk = nn.Linear(width, groups * self.head_width)
        self.wv = nn.Linear(width, groups * self.head_width)
        self.proj = nn.Linear(width, width)
        self.q_norm = _RMSNorm(self.head_width)
        self.k_norm = _RMSNorm(self.head_width)
        self.gamma = _Scale(width)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Attend over every token, rotating only the patch tokens."""
        batch, tokens, _ = x.shape
        q = self.wq(x).view(batch, tokens, self.heads, self.head_width).transpose(1, 2)
        k = self.wk(x).view(batch, tokens, self.groups, self.head_width).transpose(1, 2)
        v = self.wv(x).view(batch, tokens, self.groups, self.head_width).transpose(1, 2)
        q = _rotate(self.q_norm(q), cos, sin)
        k = _rotate(self.k_norm(k), cos, sin)
        if self.groups != self.heads:
            repeat = self.heads // self.groups
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, tokens, -1)
        return self.gamma(self.proj(out))


class _FeedForward(nn.Module):
    """A gated SiLU feed-forward whose gate and value share one projection."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(width, 2 * hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate the value with the SiLU of the gate."""
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(functional.silu(gate) * value)


class _Block(nn.Module):
    """One transformer layer: attention, then the feed-forward, each around a residual."""

    def __init__(self, width: int, heads: int, groups: int, hidden: int):
        super().__init__()
        self.ln1 = _RMSNorm(width)
        self.attn = _Attention(width, heads, groups)
        self.ln2 = _RMSNorm(width)
        self.ffn = _FeedForward(width, hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Run the layer."""
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ffn(self.ln2(x))


class _Patches(nn.Module):
    """The patch projection."""

    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Conv2d(3, width, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(b, 3, h, w)`` into ``(b, h / 16 * w / 16, width)`` tokens."""
        return self.projection(x).flatten(2).transpose(1, 2)


class _Rope(nn.Module):
    """The periods of the two-axis rotary position code."""

    def __init__(self, head_width: int):
        super().__init__()
        self.register_buffer("periods", torch.ones(head_width // 4))

    def forward(self, rows: int, columns: int, device, dtype):
        """Cosine and sine for every patch, ``(rows * columns, head_width)`` each."""
        ys = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows
        xs = (torch.arange(columns, device=device, dtype=torch.float32) + 0.5) / columns
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).flatten(0, 1)
        grid = 2.0 * grid - 1.0
        periods = self.periods.to(device=device, dtype=torch.float32)
        angles = (2.0 * math.pi * grid[:, :, None] / periods[None, None, :]).flatten(1, 2)
        angles = angles.tile(2)
        return angles.cos().to(dtype), angles.sin().to(dtype)


class _Backbone(nn.Module):
    """Patch tokens, a class token and register tokens through the transformer layers."""

    def __init__(self, width: int, heads: int, groups: list[int], hidden: int, registers: int):
        super().__init__()
        self.patch_embed = _Patches(width)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.storage_tokens = nn.Parameter(torch.zeros(1, registers, width))
        self.rope_embed = _Rope(width // heads)
        self.blocks = nn.ModuleList(_Block(width, heads, group, hidden) for group in groups)
        self.ln1 = _RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the patch features as ``(b, width, h / 16, w / 16)``."""
        batch, _, height, width = x.shape
        rows, columns = height // PATCH, width // PATCH
        patches = self.patch_embed(x)
        prefix = torch.cat(
            [self.cls_token.expand(batch, -1, -1), self.storage_tokens.expand(batch, -1, -1)], 1
        )
        tokens = torch.cat([prefix.to(patches.dtype), patches], 1)
        cos, sin = self.rope_embed(rows, columns, x.device, tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, cos, sin)
        tokens = self.ln1(tokens)[:, prefix.shape[1]:]
        return tokens.transpose(1, 2).reshape(batch, -1, rows, columns)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotary code to the trailing patch tokens of ``(b, heads, tokens, d)``."""
    count = cos.shape[0]
    head, patches = x[:, :, :-count], x[:, :, -count:]
    half = patches.shape[-1] // 2
    turned = torch.cat([-patches[..., half:], patches[..., :half]], dim=-1)
    return torch.cat([head, patches * cos + turned * sin], dim=2)


def _upsampling(width: int) -> nn.Sequential:
    """Four doubling transposed convolutions, each normalised and gated by SiLU."""
    layers = []
    for channels in UPSAMPLE:
        layers += [
            nn.ConvTranspose2d(width, channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.SiLU(),
        ]
        width = channels
    return nn.Sequential(*layers)


def _pointwise() -> nn.Sequential:
    """Two pointwise convolutions, each normalised and gated by SiLU."""
    layers = []
    width = UPSAMPLE[-1]
    for channels in POINTWISE:
        layers += [nn.Conv2d(width, channels, 1), nn.InstanceNorm2d(channels), nn.SiLU()]
        width = channels
    return nn.Sequential(*layers)


class _Head(nn.Module):
    """The decoder taking patch features back to full size, one logit per class."""

    def __init__(self, width: int):
        super().__init__()
        self.deconv_layers = _upsampling(width)
        self.conv_layers = _pointwise()
        self.conv_seg = nn.Conv2d(POINTWISE[-1], len(CLASSES), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer ``(b, 29, 16 * rows, 16 * columns)`` logits."""
        return self.conv_seg(self.conv_layers(self.deconv_layers(x)))


class Network(nn.Module):
    """A Sapiens2 body-part segmentation network.

    Attributes:
        backbone: The vision transformer.
        decode_head: The upsampling decoder.
    """

    def __init__(self, width: int, heads: int, groups: list[int], hidden: int, registers: int):
        super().__init__()
        self.backbone = _Backbone(width, heads, groups, hidden, registers)
        self.decode_head = _Head(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the class logits for normalised ``(b, 3, h, w)`` RGB."""
        return self.decode_head(self.backbone(x))


def _shape_of(shapes: dict) -> dict:
    """The constructor arguments a checkpoint's tensor shapes describe."""
    width = shapes["backbone.patch_embed.projection.weight"][0]
    head_width = shapes["backbone.blocks.0.attn.q_norm.weight"][0]
    depth = sum(1 for key in shapes if key.endswith(".attn.wq.weight"))
    return {
        "width": width,
        "heads": width // head_width,
        "groups": [
            shapes[f"backbone.blocks.{index}.attn.wk.weight"][0] // head_width
            for index in range(depth)
        ],
        "hidden": shapes["backbone.blocks.0.ffn.w3.weight"][1],
        "registers": shapes["backbone.storage_tokens"][1],
    }


def fits(path: str | os.PathLike) -> bool:
    """Whether a safetensors file holds a body-part network this module builds.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        True when its header carries the body-part head.
    """
    key, shape = SIGNATURE
    return checkpoint_shapes(path).get(key) == shape


def offered() -> list[str]:
    """What the model menu lists: usable files on disk, then published files a run can fetch.

    Returns:
        File names in the ``sapiens`` folder, published files first and smallest first.
    """
    found = []
    for name in model_files(FOLDER, suffixes=(".safetensors",)):
        parts = name.replace("\\", "/").split("/")
        if any(part.startswith(("models--", ".")) for part in parts[:-1]):
            continue
        path = model_file_path(FOLDER, name)
        if path is not None and fits(path):
            found.append(name.replace(os.sep, "/"))
    order = {name: index for index, name in enumerate(PUBLISHED)}
    found.sort(key=lambda name: (order.get(os.path.basename(name), len(order)), name.lower()))
    if network_enabled():
        present = {os.path.basename(name) for name in found}
        found += [name for name in PUBLISHED if name not in present]
    return found


def locate(name: str = DEFAULT) -> str:
    """The path of one checkpoint, fetching a published file when that is allowed.

    Args:
        name: A file in the ``sapiens`` model folder, relative to it.

    Returns:
        An absolute path.

    Raises:
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    found = model_file_path(FOLDER, name)
    if found is not None:
        return str(found)
    base = os.path.basename(name)
    if base in PUBLISHED:
        return published_checkpoint(
            FOLDER, PUBLISHED[base], base, feature=FEATURE, what="The body-part network"
        )
    roots = model_directories(FOLDER)
    searched = "\n".join(f"    {root}" for root in roots) or "    nowhere, ComfyUI is not running"
    raise ModelUnavailable(
        "\n".join([f"The body-part network needs {name}, which is not in:", searched,
                   *unresolved_links(roots, name)])
    )


def load(name: str = DEFAULT, device: str | None = None):
    """Build the network and read one checkpoint into it.

    Args:
        name: A file in the ``sapiens`` model folder, relative to it.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``model`` is the network in eval mode, in
        bfloat16 on a GPU that computes in it and float32 elsewhere, built once per file.

    Raises:
        ValueError: The file does not hold a body-part network.
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    path = locate(name)
    return managed_module(
        ("sapiens2", path), lambda: _build(path, device), device=device, name="Sapiens2 Body Parts"
    )


def _build(path: str, device) -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    from safetensors.torch import load_file

    from . import compute_device

    if not fits(path):
        raise ValueError(
            f"{path} is not a Sapiens2 body-part checkpoint: its head does not answer "
            f"{len(CLASSES)} classes. Pick a sapiens2_*_seg.safetensors file."
        )
    network = Network(**_shape_of(checkpoint_shapes(path)))
    network.load_state_dict(load_file(path), strict=True)
    target = compute_device(device)
    if target.type == "cuda" and torch.cuda.is_bf16_supported():
        network.to(torch.bfloat16)
    return network.eval()
