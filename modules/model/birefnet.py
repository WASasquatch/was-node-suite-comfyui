"""Foreground segmentation, on the BiRefNet network.

:func:`load` answers a network taking ``(batch, 3, height, width)`` RGB on a 0 to 1 scale,
each side a multiple of :data:`MULTIPLE`. :data:`MODELS` names each published file.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import deform_conv2d

from . import (
    ModelUnavailable,
    checkpoint_shapes,
    managed_module,
    model_file_path,
    model_directories,
    published_checkpoint,
    unresolved_links,
)
from .swin import (
    DEPTHS,
    EMBED_DIM_LARGE,
    HEADS_LARGE,
    PATCH_SIZE,
    WINDOW_SIZE,
    SwinTransformer,
)

__all__ = [
    "DEFAULT_SIDE",
    "FEATURE",
    "FILENAME",
    "FOLDER",
    "fits",
    "load",
    "locate",
    "MODELS",
    "MULTIPLE",
    "Network",
    "REPO_ID",
    "RESOLUTIONS",
    "SIGNATURE",
    "side",
    "SUBFOLDER",
]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "birefnet"

#: Repository publishing the weights, the directory inside it, and the default file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "birefnet"
FILENAME = "General.safetensors"

#: Each published file, by the name the model menu gave it before it listed files.
MODELS = {
    "BiRefNet General": FILENAME,
    "BiRefNet General HR": "General-HR.safetensors",
    "BiRefNet General Dynamic": "General-dynamic.safetensors",
    "BiRefNet General 512": "General-reso_512.safetensors",
    "BiRefNet Portrait": "Portrait.safetensors",
    "BiRefNet Matting HR": "Matting-HR.safetensors",
    "BiRefNet Fine Detail": "DIS.safetensors",
    "BiRefNet Fine Detail Extended": "DIS-TR_TEs.safetensors",
    "BiRefNet Camouflage": "COD.safetensors",
    "BiRefNet Salient Object": "HRSOD.safetensors",
}

#: Channels the four Swin large stages answer, coarsest first.
BACKBONE_CHANNELS = (1536, 768, 384, 192)

#: Channels one decoder scale reads: a stage's own map beside the same stage read at half
#: the frame.
LATERAL_CHANNELS = tuple(channels * 2 for channels in BACKBONE_CHANNELS)

#: Channels the three finer maps carry into the coarsest scale, finest first.
CONTEXT_CHANNELS = tuple(reversed(LATERAL_CHANNELS[1:]))

#: Width every decoder block and every input block works at inside.
INTER_CHANNELS = 64

#: Width each branch of the deformable ASPP answers.
ASPP_CHANNELS = 256

#: Kernel side of the three deformable ASPP branches.
ASPP_KERNELS = (1, 3, 7)

#: Width the gradient branch of each scale works at.
GRADIENT_CHANNELS = 16

#: Tiles along each side the frame is cut into for the five decoder reads, coarsest first.
SPLIT_GRIDS = (32, 16, 8, 4, 1)

#: Mean and standard deviation the frame is standardised by, red first.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Side a published file is read at, by filename. Any other file is read at
#: :data:`DEFAULT_SIDE`.
RESOLUTIONS = {
    "General-HR.safetensors": 2048,
    "General-reso_512.safetensors": 512,
    "Matting-HR.safetensors": 2048,
}

#: Side a file absent from :data:`RESOLUTIONS` is read at.
DEFAULT_SIDE = 1024

#: A tensor every checkpoint this network loads carries, and its shape.
SIGNATURE = ("bb.patch_embed.proj.weight", [192, 3, 4, 4])

#: Multiple both sides of a frame must be.
MULTIPLE = PATCH_SIZE * 2 ** (len(DEPTHS) - 1)


def _resize(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Resample a frame onto another frame's rows and columns.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        like: Tensor whose last two axes give the size to answer at.

    Returns:
        A tensor of ``x``'s channels at ``like``'s rows and columns.
    """
    return functional.interpolate(x, size=like.shape[2:], mode="bilinear", align_corners=True)


def _patches(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Cut a frame into a square grid of tiles, stacked along the channel axis.

    Args:
        x: ``(batch, channels, height, width)`` tensor, both sides a multiple of ``grid``.
        grid: Tiles along each side.

    Returns:
        A ``(batch, channels * grid * grid, height // grid, width // grid)`` tensor.
    """
    batch, channels, height, width = x.shape
    rows, columns = height // grid, width // grid
    tiles = x.view(batch, channels, grid, rows, grid, columns)
    tiles = tiles.permute(0, 1, 2, 4, 3, 5)
    return tiles.reshape(batch, channels * grid * grid, rows, columns)


def _gradient(in_channels: int) -> nn.Sequential:
    """A 3x3 down to :data:`GRADIENT_CHANNELS`, a batch norm and a ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, GRADIENT_CHANNELS, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(GRADIENT_CHANNELS),
        nn.ReLU(inplace=True),
    )


def _readout() -> nn.Sequential:
    """A 1x1 from :data:`GRADIENT_CHANNELS` down to one channel."""
    return nn.Sequential(nn.Conv2d(GRADIENT_CHANNELS, 1, kernel_size=1, stride=1, padding=0))


class DeformableConv2d(nn.Module):
    """A convolution sampling at learnt offsets, every tap scaled by a learnt modulator."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        taps = kernel_size * kernel_size
        self.padding = padding
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * taps, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.modulator_conv = nn.Conv2d(
            in_channels, taps, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve at the offsets and the modulators read off the frame.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            A ``(batch, out_channels, height, width)`` tensor.
        """
        return deform_conv2d(
            input=x,
            offset=self.offset_conv(x),
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            stride=1,
            padding=self.padding,
            mask=2.0 * torch.sigmoid(self.modulator_conv(x)),
        )


class ASPPBranch(nn.Module):
    """One deformable branch of the ASPP: the convolution, a batch norm and a ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.atrous_conv = DeformableConv2d(
            in_channels, out_channels, kernel_size, kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the frame at this branch's kernel side.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            A ``(batch, out_channels, height, width)`` tensor.
        """
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    """Three deformable branches beside a 1x1 and the pooled frame, joined and projected."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.aspp1 = ASPPBranch(in_channels, ASPP_CHANNELS, 1)
        self.aspp_deforms = nn.ModuleList(
            ASPPBranch(in_channels, ASPP_CHANNELS, kernel) for kernel in ASPP_KERNELS
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, ASPP_CHANNELS, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(ASPP_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            ASPP_CHANNELS * (2 + len(ASPP_KERNELS)), in_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the frame at every branch and join what they answer.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            A ``(batch, in_channels, height, width)`` tensor.
        """
        first = self.aspp1(x)
        branches = [branch(x) for branch in self.aspp_deforms]
        pooled = _resize(self.global_avg_pool(x), first)
        joined = self.conv1(torch.cat([first, *branches, pooled], dim=1))
        return self.dropout(self.relu(self.bn1(joined)))


class DecoderBlock(nn.Module):
    """A 3x3 down to :data:`INTER_CHANNELS`, the deformable ASPP, and a 3x3 back out."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, INTER_CHANNELS, kernel_size=3, stride=1, padding=1
        )
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(INTER_CHANNELS)
        self.conv_out = nn.Conv2d(
            INTER_CHANNELS, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn_in = nn.BatchNorm2d(INTER_CHANNELS)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Narrow the frame, attend over it and widen it again.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            A ``(batch, out_channels, height, width)`` tensor.
        """
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        return self.bn_out(self.conv_out(self.dec_att(x)))


class LateralBlock(nn.Module):
    """A 1x1 projection of one backbone scale, keeping its channel count."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a skip onto the scale it is added to.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        return self.conv(x)


class InputBlock(nn.Module):
    """Two 3x3 convolutions carrying the frame's own tiles into one decoder scale."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, INTER_CHANNELS, kernel_size=3, stride=1, padding=1
        )
        self.conv_out = nn.Conv2d(
            INTER_CHANNELS, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read a grid of tiles as one stack of channels.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor of stacked tiles.

        Returns:
            A ``(batch, out_channels, height, width)`` tensor.
        """
        return self.conv_out(self.conv1(x))


class Decoder(nn.Module):
    """Four scales climbed back to full size, each read beside the frame's own tiles."""

    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.ipt_blk5 = InputBlock(3 * SPLIT_GRIDS[0] ** 2, channels[0] // 8)
        self.ipt_blk4 = InputBlock(3 * SPLIT_GRIDS[1] ** 2, channels[0] // 8)
        self.ipt_blk3 = InputBlock(3 * SPLIT_GRIDS[2] ** 2, channels[1] // 8)
        self.ipt_blk2 = InputBlock(3 * SPLIT_GRIDS[3] ** 2, channels[2] // 8)
        self.ipt_blk1 = InputBlock(3 * SPLIT_GRIDS[4] ** 2, channels[3] // 8)
        self.decoder_block4 = DecoderBlock(channels[0] + channels[0] // 8, channels[1])
        self.decoder_block3 = DecoderBlock(channels[1] + channels[0] // 8, channels[2])
        self.decoder_block2 = DecoderBlock(channels[2] + channels[1] // 8, channels[3])
        self.decoder_block1 = DecoderBlock(channels[3] + channels[2] // 8, channels[3] // 2)
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(
                channels[3] // 2 + channels[3] // 8, 1, kernel_size=1, stride=1, padding=0
            )
        )
        self.lateral_block4 = LateralBlock(channels[1])
        self.lateral_block3 = LateralBlock(channels[2])
        self.lateral_block2 = LateralBlock(channels[3])
        self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, kernel_size=1, stride=1, padding=0)
        self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, kernel_size=1, stride=1, padding=0)
        self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, kernel_size=1, stride=1, padding=0)
        self.gdt_convs_4 = _gradient(channels[1])
        self.gdt_convs_3 = _gradient(channels[2])
        self.gdt_convs_2 = _gradient(channels[3])
        self.gdt_convs_pred_4 = _readout()
        self.gdt_convs_pred_3 = _readout()
        self.gdt_convs_pred_2 = _readout()
        self.gdt_convs_attn_4 = _readout()
        self.gdt_convs_attn_3 = _readout()
        self.gdt_convs_attn_2 = _readout()

    def forward(
        self, frame: torch.Tensor, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Climb from the coarsest scale to full size, gated by the gradient of each scale.

        Args:
            frame: ``(batch, 3, height, width)`` standardised tensor.
            features: One map per backbone scale, finest first, the coarsest squeezed.

        Returns:
            A one element list holding the ``(batch, 1, height, width)`` logits.
        """
        x1, x2, x3, x4 = features
        tiles = _resize(_patches(frame, SPLIT_GRIDS[0]), x4)
        p4 = self.decoder_block4(torch.cat([x4, self.ipt_blk5(tiles)], dim=1))
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()

        p3 = _resize(p4, x3) + self.lateral_block4(x3)
        tiles = _resize(_patches(frame, SPLIT_GRIDS[1]), x3)
        p3 = self.decoder_block3(torch.cat([p3, self.ipt_blk4(tiles)], dim=1))
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()

        p2 = _resize(p3, x2) + self.lateral_block3(x2)
        tiles = _resize(_patches(frame, SPLIT_GRIDS[2]), x2)
        p2 = self.decoder_block2(torch.cat([p2, self.ipt_blk3(tiles)], dim=1))
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()

        p1 = _resize(p2, x1) + self.lateral_block2(x1)
        tiles = _resize(_patches(frame, SPLIT_GRIDS[3]), x1)
        p1 = self.decoder_block1(torch.cat([p1, self.ipt_blk2(tiles)], dim=1))
        p1 = _resize(p1, frame)
        tiles = _patches(frame, SPLIT_GRIDS[4])
        return [self.conv_out1(torch.cat([p1, self.ipt_blk1(tiles)], dim=1))]


class Network(nn.Module):
    """BiRefNet: a Swin large backbone read at two scales, climbed back to full size."""

    def __init__(self):
        super().__init__()
        self.bb = SwinTransformer(
            embed_dim=EMBED_DIM_LARGE,
            depths=DEPTHS,
            num_heads=HEADS_LARGE,
            window_size=WINDOW_SIZE,
        )
        self.squeeze_module = nn.Sequential(
            DecoderBlock(LATERAL_CHANNELS[0] + sum(CONTEXT_CHANNELS), LATERAL_CHANNELS[0])
        )
        self.decoder = Decoder(LATERAL_CHANNELS)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Read a frame and answer how much of every pixel belongs to the foreground.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with both
                sides a multiple of :data:`MULTIPLE`.

        Returns:
            The scaled predictions, each ``(batch, 1, height, width)`` on a 0 to 1 scale in
            the dtype of ``x``. A caller reads the last one, which is at the frame's size.
        """
        source = x.dtype
        dtype = self.decoder.conv_out1[0].weight.dtype
        mean = torch.tensor(MEAN, device=x.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=x.device, dtype=dtype).view(1, 3, 1, 1)
        frame = (x.to(dtype) - mean) / std
        features = self._encode(frame)
        features[3] = self.squeeze_module(features[3])
        return [out.sigmoid().to(source) for out in self.decoder(frame, features)]

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Read the frame at its own size and at half of it, joined scale by scale.

        Args:
            x: ``(batch, 3, height, width)`` standardised tensor.

        Returns:
            One map per backbone scale, finest first, the coarsest carrying the three finer
            maps resampled onto it.
        """
        height, width = x.shape[2], x.shape[3]
        half = functional.interpolate(
            x, size=(height // 2, width // 2), mode="bilinear", align_corners=True
        )
        maps = [
            torch.cat([whole, _resize(part, whole)], dim=1)
            for whole, part in zip(self.bb(x)[1:], self.bb(half)[1:])
        ]
        context = [_resize(one, maps[3]) for one in maps[:3]]
        maps[3] = torch.cat([*context, maps[3]], dim=1)
        return maps


def side(name: str) -> int:
    """The square side a checkpoint is read at.

    Args:
        name: A checkpoint name, with or without a subdirectory.

    Returns:
        Pixels per side.
    """
    return RESOLUTIONS.get(os.path.basename(name), DEFAULT_SIDE)


def fits(path: str | os.PathLike) -> bool:
    """Whether a safetensors file holds weights this network loads.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        True when its header carries :data:`SIGNATURE`.
    """
    key, shape = SIGNATURE
    return checkpoint_shapes(path).get(key) == shape


def locate(name: str = FILENAME) -> str:
    """The path of one checkpoint, fetching a published file when that is allowed.

    Args:
        name: A file in the ``birefnet`` model folder, relative to it, or a key of
            :data:`MODELS`.

    Returns:
        An absolute path.

    Raises:
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    name = MODELS.get(name, name)
    found = model_file_path(FOLDER, name)
    if found is not None:
        return str(found)
    if os.path.basename(name) in MODELS.values():
        return published_checkpoint(
            FOLDER, REPO_ID, os.path.basename(name), subfolder=SUBFOLDER, feature=FEATURE,
            what="The segmentation network",
        )
    roots = model_directories(FOLDER)
    searched = "\n".join(f"    {root}" for root in roots) or "    nowhere, ComfyUI is not running"
    raise ModelUnavailable(
        "\n".join([f"The segmentation network needs {name}, which is not in:", searched,
                   *unresolved_links(roots, name)])
    )


def load(name: str = FILENAME, device: str | None = None):
    """Build the network and read one checkpoint into it.

    Args:
        name: A file in the ``birefnet`` model folder, relative to it, or a key of
            :data:`MODELS`.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``model`` is the network in eval mode, at
        the dtype the file holds, built once per file and kept for the process.

    Raises:
        ValueError: The file does not hold BiRefNet weights of the size built here.
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    path = locate(name)
    return managed_module(("birefnet", path), lambda: _build(path), device=device)


def _build(path: str) -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    from safetensors.torch import load_file

    if not fits(path):
        key, shape = SIGNATURE
        raise ValueError(
            f"{path} is not a BiRefNet checkpoint this node can build: it has no {key} "
            f"shaped {shape}. BiRefNet Lite files and other architectures are not supported; "
            "pick a full size BiRefNet file."
        )
    weights = load_file(path)
    network = Network()
    network.to(next(one.dtype for one in weights.values() if one.is_floating_point()))
    network.load_state_dict(weights, strict=True)
    return network.eval()
