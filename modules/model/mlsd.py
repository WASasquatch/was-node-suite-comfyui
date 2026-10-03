"""Line segment detection, on the ControlNet M-LSD network.

:func:`load` answers the network with the published weights already in it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import managed_module, published_checkpoint

__all__ = ["FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "load"]

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "mlsd"

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: Repository publishing the weights, and the file inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "mlsd_large_512_fp32.pth"

#: Square side the frame is resized to before the network sees it.
INPUT_SIZE = 512

#: Planes the input carries: three colour planes and a fourth constant plane.
INPUT_CHANNELS = 4

#: Channels the opening convolution answers with.
STEM_CHANNELS = 32

#: Backbone stages: expansion, output channels, repeats, stride of the first repeat.
STAGES = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1))

#: Backbone positions the decoder taps, finest first.
TAPS = (1, 3, 6, 10, 13)

#: Channels each decoder rung works at, and the width of a merged pair.
DECODER_CHANNELS = 64
MERGED_CHANNELS = 128

#: Dilation of the first convolution in the head.
DILATION = 5

#: Channels the head emits, and how many leading ones the detector drops.
HEAD_CHANNELS = 16
HEAD_OFFSET = 7


class ConvBlock(nn.Sequential):
    """A convolution with batch norm and a ReLU6, padded the way TFLite pads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = 0 if stride == 2 else (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve, normalise and clamp.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor, halved in both axes at stride two.
        """
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        return super().forward(x)


class InvertedResidual(nn.Module):
    """A MobileNetV2 bottleneck: expand, convolve per channel, project back down."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int):
        super().__init__()
        hidden = int(round(in_channels * expansion))
        self.residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBlock(in_channels, hidden, kernel_size=1))
        layers.extend([
            ConvBlock(hidden, hidden, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bottleneck, adding the input back where the shapes allow.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor, halved in both axes at stride two.
        """
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)


class Backbone(nn.Module):
    """A truncated MobileNetV2 answering the five feature maps the decoder merges."""

    def __init__(self):
        super().__init__()
        features: list[nn.Module] = [ConvBlock(INPUT_CHANNELS, STEM_CHANNELS, stride=2)]
        channels = STEM_CHANNELS
        for expansion, outputs, repeats, stride in STAGES:
            for repeat in range(repeats):
                first = repeat == 0
                features.append(
                    InvertedResidual(channels, outputs, stride if first else 1, expansion)
                )
                channels = outputs
        self.features = nn.Sequential(*features)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the stack and keep the tapped maps.

        Args:
            x: ``(batch, 4, height, width)`` tensor.

        Returns:
            Five feature maps at 16, 24, 32, 64 and 96 channels, finest first.
        """
        taps = []
        for index, feature in enumerate(self.features):
            x = feature(x)
            if index in TAPS:
                taps.append(x)
        return taps


class LateralBlock(nn.Module):
    """Projects a fine and a coarse map to a common width and concatenates them."""

    def __init__(
        self,
        fine_channels: int,
        coarse_channels: int,
        fine_out: int,
        coarse_out: int,
        upscale: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(coarse_channels, coarse_out, kernel_size=1),
            nn.BatchNorm2d(coarse_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fine_channels, fine_out, kernel_size=1),
            nn.BatchNorm2d(fine_out),
            nn.ReLU(inplace=True),
        )
        self.upscale = upscale

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """Merge one backbone tap with the rung above it.

        Args:
            fine: ``(batch, fine_channels, height, width)`` tensor from the backbone.
            coarse: ``(batch, coarse_channels, height, width)`` tensor from the rung above,
                half the frame where the block upscales.

        Returns:
            ``(batch, fine_out + coarse_out, height, width)`` tensor.
        """
        coarse = self.conv1(coarse)
        fine = self.conv2(fine)
        if self.upscale:
            coarse = F.interpolate(
                coarse, scale_factor=2.0, mode="bilinear", align_corners=True
            )
        return torch.cat((fine, coarse), dim=1)


class RefineBlock(nn.Module):
    """A residual 3x3 convolution followed by a 3x3 that narrows the map."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a merged map.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor.
        """
        return self.conv2(self.conv1(x) + x)


class HeadBlock(nn.Module):
    """A dilated 3x3, a plain 3x3 and a 1x1 down to the output channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=DILATION, dilation=DILATION
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Turn the finest decoder map into the raw centre and displacement channels.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor.
        """
        return self.conv3(self.conv2(self.conv1(x)))


class Network(nn.Module):
    """The M-LSD large network: a MobileNetV2 trunk under a four rung merging decoder."""

    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        widths = [stage[1] for stage in STAGES]
        self.block15 = LateralBlock(
            widths[3], widths[4], DECODER_CHANNELS, DECODER_CHANNELS, upscale=False
        )
        self.block16 = RefineBlock(MERGED_CHANNELS, DECODER_CHANNELS)
        self.block17 = LateralBlock(
            widths[2], DECODER_CHANNELS, DECODER_CHANNELS, DECODER_CHANNELS
        )
        self.block18 = RefineBlock(MERGED_CHANNELS, DECODER_CHANNELS)
        self.block19 = LateralBlock(
            widths[1], DECODER_CHANNELS, DECODER_CHANNELS, DECODER_CHANNELS
        )
        self.block20 = RefineBlock(MERGED_CHANNELS, DECODER_CHANNELS)
        self.block21 = LateralBlock(
            widths[0], DECODER_CHANNELS, DECODER_CHANNELS, DECODER_CHANNELS
        )
        self.block22 = RefineBlock(MERGED_CHANNELS, DECODER_CHANNELS)
        self.block23 = HeadBlock(DECODER_CHANNELS, HEAD_CHANNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score line segment centres and their end point displacements.

        Args:
            x: ``(batch, 4, 512, 512)`` tensor, three colour planes red first on a -1 to 1
                scale and a fourth constant plane.

        Returns:
            ``(batch, 9, 256, 256)`` tensor. Channel 0 is the centre score before the
            sigmoid, channels 1 to 4 are the start and end displacements in x and y.
        """
        c1, c2, c3, c4, c5 = self.backbone(x)
        x = self.block16(self.block15(c4, c5))
        x = self.block18(self.block17(c3, x))
        x = self.block20(self.block19(c2, x))
        x = self.block22(self.block21(c1, x))
        return self.block23(x)[:, HEAD_OFFSET:, :, :]


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("mlsd", REPO_ID, FILENAME), _build, name="M-LSD")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, feature=FEATURE, what="The line segment network"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    network = Network()
    network.load_state_dict(state)
    return network.float().eval()
