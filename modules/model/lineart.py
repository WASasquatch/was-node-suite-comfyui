"""Lineart extraction, on the ControlNet lineart generator.

:func:`load` answers the network with the published weights already in it.
"""

from __future__ import annotations


import torch
from torch import nn

from . import managed_module, published_checkpoint

__all__ = ["COARSE_FILENAME", "FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "lineart"

#: Repository publishing the weights, and the two files inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "sk_model.pth"
COARSE_FILENAME = "sk_model2.pth"

#: Channel count the opening convolution lifts a frame to.
BASE_CHANNELS = 64

#: Halvings on the way down, matched by as many doublings on the way back up.
SCALES = 2

#: Residual blocks stacked at the smallest scale.
RESIDUAL_BLOCKS = 3

#: Reflection padding either side of the 7x7 convolutions that open and close the network.
WIDE_PAD = 3


class ResidualBlock(nn.Module):
    """Two reflection-padded 3x3 convolutions, instance normalised, added back to the input."""

    def __init__(self, features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the block's own output back to what came in.

        Args:
            x: ``(batch, features, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        return x + self.conv_block(x)


class Network(nn.Module):
    """The residual generator, answering one stroke map per frame."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        residual_blocks: int = RESIDUAL_BLOCKS,
    ):
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(WIDE_PAD),
            nn.Conv2d(input_channels, BASE_CHANNELS, 7),
            nn.InstanceNorm2d(BASE_CHANNELS),
            nn.ReLU(inplace=True),
        )

        channels = BASE_CHANNELS
        down: list[nn.Module] = []
        for _ in range(SCALES):
            down += [
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
        self.model1 = nn.Sequential(*down)

        self.model2 = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(residual_blocks)]
        )

        up: list[nn.Module] = []
        for _ in range(SCALES):
            up += [
                nn.ConvTranspose2d(
                    channels, channels // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2
        self.model3 = nn.Sequential(*up)

        self.model4 = nn.Sequential(
            nn.ReflectionPad2d(WIDE_PAD),
            nn.Conv2d(BASE_CHANNELS, output_channels, 7),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trace strokes across a frame.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with both
                sides a multiple of four.

        Returns:
            A ``(batch, 1, height, width)`` map on a 0 to 1 scale, bright where a stroke is.
        """
        h = self.model0(x)
        h = self.model1(h)
        h = self.model2(h)
        h = self.model3(h)
        return self.model4(h)


def load(coarse: bool = False, device: str | None = None) -> Network:
    """Build the network and read one of the two published checkpoints into it.

    Args:
        coarse: True for the heavier strokes of ``sk_model2.pth``, False for the finer
            ones of ``sk_model.pth``.
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once per checkpoint and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    filename = COARSE_FILENAME if coarse else FILENAME
    return managed_module(("lineart", REPO_ID, filename), lambda: _build(filename), name="Lineart")


def _build(filename: str) -> Network:
    """Read one checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, filename, feature=FEATURE, what="The lineart network"
    )
    network = Network()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.float().eval()
