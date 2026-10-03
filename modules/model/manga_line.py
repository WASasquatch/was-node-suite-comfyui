"""Manga line extraction, on the MangaLineExtraction residual skip network.

:func:`load` answers the network with the published weights already in it.
"""

from __future__ import annotations


import torch
from torch import nn

from . import managed_module, published_checkpoint

__all__ = ["FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "manga_line"

#: Repository publishing the weights, and the file inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "erika.pth"

#: Channels each contracting stage works at, and how many blocks it stacks. Every stage but the
#: first halves the frame in its last block.
STAGES = ((1, 24, 2), (24, 48, 3), (48, 96, 5), (96, 192, 7), (192, 384, 12))

#: Channels each expanding stage works at, and how many blocks it stacks. Each doubles the frame
#: in its first block, and its output is added to the contracting stage of the same width.
UPSAMPLING_STAGES = ((384, 192, 7), (192, 96, 5), (96, 48, 3), (48, 24, 2))

#: Channels and depth of the stage between the last skip merge and the one-channel projection.
HEAD = (24, 16, 2)

#: Slope of the negative half of every activation.
NEGATIVE_SLOPE = 0.2

#: Epsilon of every batch normalisation.
EPSILON = 1e-3

SIZE_MULTIPLE = 16


class BnReluConv(nn.Module):
    """Batch normalisation, a leaky activation and a convolution, in that order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 1,
        upsample: bool = False,
    ):
        super().__init__()
        layers = [
            nn.BatchNorm2d(in_channels, eps=EPSILON),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel, kernel),
                stride=stride,
                padding=(kernel // 2, kernel // 2),
                padding_mode="zeros",
            ),
        ]
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve the normalised, activated input.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor, the frame halved by a stride of two
            and doubled by an upsampling layer.
        """
        return self.model(x)


class Shortcut(nn.Module):
    """The path around a block, projecting its input to the block's shape where they differ."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        upsample: bool = False,
    ):
        super().__init__()
        self.model = None
        if in_channels != out_channels or stride != 1:
            layers = [nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride)]
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add a block's input to its output.

        Args:
            x: The block's input.
            y: The block's output.

        Returns:
            Their sum, ``x`` projected by a 1x1 convolution first where the two shapes differ.
        """
        if self.model is None:
            return x + y
        return self.model(x) + y


class BasicBlock(nn.Module):
    """Two convolutions with a shortcut around them."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        upsample: bool = False,
    ):
        super().__init__()
        self.conv1 = BnReluConv(in_channels, out_channels, stride=stride, upsample=upsample)
        self.residual = BnReluConv(out_channels, out_channels)
        self.shortcut = Shortcut(in_channels, out_channels, stride=stride, upsample=upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both convolutions and merge the shortcut into the result.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor, at the block's own scale.
        """
        return self.shortcut(x, self.residual(self.conv1(x)))


class Stage(nn.Module):
    """A run of blocks at one width, one of which changes the scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        layers = [
            BasicBlock(
                in_channels if index == 0 else out_channels,
                out_channels,
                stride=2 if downsample and index == blocks - 1 else 1,
                upsample=upsample and index == 0,
            )
            for index in range(blocks)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stage's blocks in order.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            ``(batch, out_channels, height, width)`` tensor, halved by a contracting stage and
            doubled by an expanding one.
        """
        return self.model(x)


class Network(nn.Module):
    """Five contracting stages, four expanding ones merged with their skips, and a head."""

    def __init__(self):
        super().__init__()
        for index, (inputs, outputs, blocks) in enumerate(STAGES):
            setattr(self, f"block{index}", Stage(inputs, outputs, blocks, downsample=index > 0))
        for index, (inputs, outputs, blocks) in enumerate(UPSAMPLING_STAGES, start=1):
            setattr(self, f"block{index + 4}", Stage(inputs, outputs, blocks, upsample=True))
            setattr(self, f"res{index}", Shortcut(outputs, outputs))
        self.block9 = Stage(*HEAD)
        self.conv15 = BnReluConv(HEAD[1], 1, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trace the drawn lines.

        Args:
            x: ``(batch, 1, height, width)`` grey tensor on a 0 to 255 scale, both sides a
                multiple of :data:`SIZE_MULTIPLE`.

        Returns:
            A ``(batch, 1, height, width)`` map on a 0 to 255 scale, dark lines on a light
            ground, for the caller to clamp.
        """
        skips = []
        h = x
        for index in range(len(STAGES)):
            h = getattr(self, f"block{index}")(h)
            skips.append(h)
        for index in range(1, len(UPSAMPLING_STAGES) + 1):
            h = getattr(self, f"block{index + 4}")(h)
            h = getattr(self, f"res{index}")(skips[len(STAGES) - 1 - index], h)
        return self.conv15(self.block9(h))


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("manga_line", REPO_ID, FILENAME), _build, name="Manga Line")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, feature=FEATURE, what="The manga line network"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    network = Network()
    network.load_state_dict(state)
    return network.float().eval()
