"""Low light enhancement and deblurring, on the DarkIR restoration network.

:func:`load` answers the network with its published weights, taking and answering
``(batch, 3, height, width)`` on a 0 to 1 scale, each side a multiple of
:data:`MULTIPLE`.
"""

from __future__ import annotations

import torch
from torch import nn

from . import managed_module, published_checkpoint

__all__ = [
    "FEATURE",
    "FILENAME",
    "FOLDER",
    "MULTIPLE",
    "Network",
    "REPO_ID",
    "SUBFOLDER",
    "load",
]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "low_light"

#: Repository publishing the weights, the directory inside it, and the file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "low_light"
FILENAME = "darkir-m.pt"

#: Channels at the finest scale, doubling at every scale below it.
WIDTH = 32

#: Blocks each encoder stage stacks, finest first, and each decoder stage, coarsest first.
ENCODER_BLOCKS = (1, 2, 3)
DECODER_BLOCKS = (3, 1, 1)

#: Blocks each of the two bottleneck stacks holds, the encoder side first.
MIDDLE_ENCODER_BLOCKS = 2
MIDDLE_DECODER_BLOCKS = 2

#: Dilation each decoder branch runs its depthwise convolution at.
DILATIONS = (1, 4, 9)

#: Channel multiple of the depthwise branch, and of the feed-forward branch.
DW_EXPAND = 2
FFN_EXPAND = 2

#: Channel multiple between the frequency branch's two 1x1 convolutions.
FREQ_EXPAND = 2

#: Slope the frequency branch's activation gives a negative value.
NEGATIVE_SLOPE = 0.1

#: Epsilon added to the variance of the channel norm.
NORM_EPS = 1e-6

#: Side multiple: ``downs`` halves each side three times, 32 to 64 to 128 to 256.
MULTIPLE = 2 ** len(ENCODER_BLOCKS)


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


class Frequency(nn.Module):
    """Two 1x1 convolutions over the magnitudes of a spectrum, with the phases kept."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * FREQ_EXPAND
        self.process1 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1, 0),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True),
            nn.Conv2d(hidden, channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale the magnitude of every frequency and transform back to pixels.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm="backward")
        magnitude = self.process1(torch.abs(spectrum))
        phase = torch.angle(spectrum)
        real = magnitude * torch.cos(phase)
        imaginary = magnitude * torch.sin(phase)
        rescaled = torch.complex(real, imaginary)
        return torch.fft.irfft2(rescaled, s=(height, width), norm="backward")


class Branch(nn.Module):
    """One depthwise 3x3 convolution, at the dilation the branch was built for."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=dilation, stride=1,
                groups=channels, bias=True, dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve each channel with its own dilated kernel.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        return self.branch(x)


class EBlock(nn.Module):
    """A gated depthwise branch with channel attention, then a frequency branch."""

    def __init__(self, channels: int):
        super().__init__()
        dw_channel = channels * DW_EXPAND
        self.extra_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, stride=1, groups=channels,
            bias=True, dilation=1,
        )
        self.conv1 = nn.Conv2d(
            channels, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.branches = nn.ModuleList([Branch(dw_channel, 1)])
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1,
                groups=1, bias=True, dilation=1,
            ),
        )
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(
            dw_channel // 2, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.freq = Frequency(channels)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Gate the depthwise path by channel attention, then modulate by the frequency path.

        Args:
            inp: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        x = self.conv1(self.extra_conv(self.norm1(inp)))
        z = self.sg(sum(branch(x) for branch in self.branches))
        x = self.conv3(self.sca(z) * z)
        y = inp + self.beta * x
        x = y * self.freq(self.norm2(y))
        return y + x * self.gamma


class DBlock(nn.Module):
    """Gated dilated branches with channel attention, then a gated feed-forward branch."""

    def __init__(self, channels: int, dilations: tuple[int, ...]):
        super().__init__()
        dw_channel = channels * DW_EXPAND
        ffn_channel = channels * FFN_EXPAND
        self.conv1 = nn.Conv2d(
            channels, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True,
            dilation=1,
        )
        self.extra_conv = nn.Conv2d(
            dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=channels,
            bias=True, dilation=1,
        )
        self.branches = nn.ModuleList(Branch(dw_channel, dilation) for dilation in dilations)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1,
                groups=1, bias=True, dilation=1,
            ),
        )
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(
            dw_channel // 2, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True,
            dilation=1,
        )
        self.conv4 = nn.Conv2d(
            channels, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            ffn_channel // 2, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Gate the dilated paths by channel attention, then run the feed-forward path.

        Args:
            inp: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        x = self.extra_conv(self.conv1(self.norm1(inp)))
        z = self.sg(sum(branch(x) for branch in self.branches))
        x = self.conv3(self.sca(z) * z)
        y = inp + self.beta * x
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class Stack(nn.Module):
    """The blocks of one stage, held under ``modules_list`` and run in order."""

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.modules_list = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block in turn over one tensor.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        for block in self.modules_list:
            x = block(x)
        return x


class Network(nn.Module):
    """The DarkIR U-net, answering the restored frame rather than the ``side_out`` map."""

    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(
            3, WIDTH, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            WIDTH, 3, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = WIDTH
        for count in ENCODER_BLOCKS:
            self.encoders.append(Stack(*(EBlock(chan) for _ in range(count))))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks_enc = Stack(*(EBlock(chan) for _ in range(MIDDLE_ENCODER_BLOCKS)))
        self.middle_blks_dec = Stack(
            *(DBlock(chan, DILATIONS) for _ in range(MIDDLE_DECODER_BLOCKS))
        )
        for count in DECODER_BLOCKS:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(Stack(*(DBlock(chan, DILATIONS) for _ in range(count))))
        self.side_out = nn.Conv2d(WIDTH * MULTIPLE, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Carry a frame down three stages, through both bottleneck stacks and back up.

        Args:
            inp: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with
                both sides a multiple of :data:`MULTIPLE`.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.
        """
        x = self.intro(inp)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        light = self.middle_blks_enc(x)
        x = self.middle_blks_dec(light) + light
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        return self.ending(x) + inp


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, answering one restored frame per frame it is given, built
        once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("darkir", REPO_ID, FILENAME), _build, name="DarkIR")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, subfolder=SUBFOLDER, feature=FEATURE,
        what="The low light network",
    )
    weights = torch.load(path, map_location="cpu", weights_only=True)
    network = Network()
    network.load_state_dict(weights.get("params", weights), strict=True)
    return network.float().eval()
