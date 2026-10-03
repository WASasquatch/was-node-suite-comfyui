"""Image denoising, on the NAFNet restoration network.

:func:`load` answers the network with the published weights already in it, taking and
answering ``(batch, 3, height, width)`` on a 0 to 1 scale.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from . import managed_module, published_checkpoint

__all__ = ["CHECKPOINTS", "FEATURE", "FILENAME", "FOLDER", "load", "Network", "REPO_ID", "SUBFOLDER"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoints.
FOLDER = "denoise"

#: Repository publishing the weights, the directory inside it, and the default file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "denoise"
FILENAME = "NAFNet-SIDD-width64.pth"

#: Widget option -> the file inside the repository and the width it was trained at.
CHECKPOINTS = {
    "NAFNet SIDD width64": (FILENAME, 64),
    "NAFNet SIDD width32": ("NAFNet-SIDD-width32.pth", 32),
}

#: Blocks each of the four encoder stages stacks, ``(2, 2, 4, 8)``, sixteen in all.
ENCODER_BLOCKS = (2, 2, 4, 8)

#: Blocks the bottleneck stacks between the encoder and the decoder, twelve.
MIDDLE_BLOCKS = 12

#: Blocks each of the four decoder stages stacks, ``(2, 2, 2, 2)``, eight in all.
DECODER_BLOCKS = (2, 2, 2, 2)

#: Channel multiple of the depthwise branch, and of the feed-forward branch.
DW_EXPAND = 2
FFN_EXPAND = 2

#: Epsilon added to the variance of the channel norm.
NORM_EPS = 1e-6


class LayerNorm2d(nn.Module):
    """Layer normalisation across the channels of one pixel, with a scale and a shift."""

    def __init__(self, channels: int, eps: float = NORM_EPS):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each pixel over its channels, then scale and shift it.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        channels = x.shape[1]
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalised = (x - mean) / (variance + self.eps).sqrt()
        weight = self.weight.view(1, channels, 1, 1)
        bias = self.bias.view(1, channels, 1, 1)
        return weight * normalised + bias


class SimpleGate(nn.Module):
    """Halves the channels by splitting them in two and multiplying the halves together."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate the first half of the channels by the second.

        Args:
            x: ``(batch, channels, height, width)`` tensor with an even channel count.

        Returns:
            A ``(batch, channels // 2, height, width)`` tensor.
        """
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """A gated depthwise branch with channel attention, then a gated feed-forward branch."""

    def __init__(self, channels: int):
        super().__init__()
        dw_channel = channels * DW_EXPAND
        self.conv1 = nn.Conv2d(
            channels, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            dw_channel // 2, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_EXPAND * channels
        self.conv4 = nn.Conv2d(
            channels, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            ffn_channel // 2, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Run both branches, each added back to what it was handed.

        Args:
            inp: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        x = self.norm1(inp)
        x = self.conv2(self.conv1(x))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class Network(nn.Module):
    """The restoration U-net, answering a frame the size of the one handed to it."""

    def __init__(self, width: int = 64, img_channel: int = 3):
        super().__init__()
        self.intro = nn.Conv2d(
            img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            width, img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in ENCODER_BLOCKS:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(MIDDLE_BLOCKS)]
        )
        for num in DECODER_BLOCKS:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the bottom and the right up to a multiple of :attr:`padder_size`.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor whose two frame sides are both multiples of the padder size.
        """
        _, _, height, width = x.size()
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return functional.pad(x, (0, pad_width, 0, pad_height))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Restore a frame, adding the network's own output back to it.

        Args:
            inp: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` tensor on the same scale and of the same size.
        """
        _, _, height, width = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)
        x = self.ending(x) + inp
        return x[:, :, :height, :width]


def load(name: str = "NAFNet SIDD width64", device: str | None = None) -> Network:
    """Build the network and read one of the published checkpoints into it.

    Args:
        name: One of the keys of :data:`CHECKPOINTS`.
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once per checkpoint and kept for the process.

    Raises:
        ValueError: ``name`` is not a key of :data:`CHECKPOINTS`.
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    if name not in CHECKPOINTS:
        raise ValueError(
            f"NAFNet model must be one of {', '.join(CHECKPOINTS)}, not {name!r}"
        )
    filename, width = CHECKPOINTS[name]
    return managed_module(
        ("nafnet", REPO_ID, filename), lambda: _build(filename, width), name="NAFNet"
    )


def _build(filename: str, width: int) -> Network:
    """Read one checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER,
        REPO_ID,
        filename,
        subfolder=SUBFOLDER,
        feature=FEATURE,
        what="The denoising network",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # The published file wraps the weights under a params key.
    state = checkpoint.get("params", checkpoint)
    network = Network(width=width)
    network.load_state_dict(state, strict=True)
    return network.float().eval()
