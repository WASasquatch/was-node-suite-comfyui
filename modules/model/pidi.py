"""Pixel difference edge detection, on the ControlNet PiDiNet network.

:func:`load` answers the network with the published weights already in it.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from . import managed_module, published_checkpoint

__all__ = ["FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "pidi"

#: Repository publishing the weights, and the file inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "table5_pidinet.pth"

#: Channels the first stage works at.
CHANNELS = 60

#: Channels each stage is narrowed to before it becomes an edge map.
DILATION_CHANNELS = 24

#: Channels each of the four stages answers.
FUSE_CHANNELS = (CHANNELS, CHANNELS * 2, CHANNELS * 4, CHANNELS * 4)

#: Blocks each stage stacks.
STAGE_BLOCKS = (3, 4, 4, 4)

#: Pixel difference operator of the sixteen convolutions, the ``carv4`` configuration.
STAGES = ("cd", "ad", "rd", "cv") * 4

#: Clockwise neighbour of every cell of a flattened 3x3 kernel.
ROTATION = [3, 0, 1, 6, 4, 2, 7, 8, 5]

#: Cells of a flattened 5x5 kernel the ring of a 3x3 kernel is written to.
OUTER_RING = [0, 2, 4, 10, 14, 20, 22, 24]

#: Cells of a flattened 5x5 kernel the same ring is subtracted from.
INNER_RING = [6, 7, 8, 11, 13, 16, 17, 18]


def _vanilla(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Convolve a 3x3 kernel over the pixels themselves.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        weight: ``(outputs, channels // groups, 3, 3)`` kernel.
        groups: Channel groups the convolution is split into.

    Returns:
        ``(batch, outputs, height, width)`` tensor.
    """
    return functional.conv2d(x, weight, None, 1, 1, 1, groups)


def _central(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Convolve a 3x3 kernel over each pixel less the centre of its window.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        weight: ``(outputs, channels // groups, 3, 3)`` kernel.
        groups: Channel groups the convolution is split into.

    Returns:
        ``(batch, outputs, height, width)`` tensor.
    """
    centre = weight.sum(dim=[2, 3], keepdim=True)
    y = functional.conv2d(x, weight, None, 1, 1, 1, groups)
    return y - functional.conv2d(x, centre, None, 1, 0, 1, groups)


def _angular(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Convolve a 3x3 kernel over each pixel less its clockwise neighbour.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        weight: ``(outputs, channels // groups, 3, 3)`` kernel.
        groups: Channel groups the convolution is split into.

    Returns:
        ``(batch, outputs, height, width)`` tensor.
    """
    shape = weight.shape
    flat = weight.view(shape[0], shape[1], -1)
    turned = (flat - flat[:, :, ROTATION]).view(shape)
    return functional.conv2d(x, turned, None, 1, 1, 1, groups)


def _radial(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Convolve a 5x5 kernel over each pixel of the outer ring less the inner ring.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        weight: ``(outputs, channels // groups, 3, 3)`` kernel.
        groups: Channel groups the convolution is split into.

    Returns:
        ``(batch, outputs, height, width)`` tensor.
    """
    shape = weight.shape
    flat = weight.view(shape[0], shape[1], -1)
    spread = torch.zeros(
        shape[0], shape[1], 5 * 5, dtype=weight.dtype, device=weight.device
    )
    spread[:, :, OUTER_RING] = flat[:, :, 1:]
    spread[:, :, INNER_RING] = -flat[:, :, 1:]
    spread = spread.view(shape[0], shape[1], 5, 5)
    return functional.conv2d(x, spread, None, 1, 2, 1, groups)


#: The operator each entry of ``STAGES`` names.
OPERATORS = {"cv": _vanilla, "cd": _central, "ad": _angular, "rd": _radial}


class PixelDifference(nn.Module):
    """A 3x3 convolution reading differences between pixels rather than pixels."""

    def __init__(
        self, operator: str, in_channels: int, out_channels: int, groups: int = 1
    ):
        super().__init__()
        self.operator = operator
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, 3, 3)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolution named by the operator.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, outputs, height, width)`` tensor.
        """
        return OPERATORS[self.operator](x, self.weight, self.groups)


class Block(nn.Module):
    """One residual block: a depthwise pixel difference and a 1x1 projection."""

    def __init__(
        self, operator: str, in_channels: int, out_channels: int, stride: int = 1
    ):
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.conv1 = PixelDifference(
            operator, in_channels, in_channels, groups=in_channels
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block, halving the frame first where the stride asks.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, outputs, height, width)`` tensor, halved when the stride is two.
        """
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(functional.relu(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class Dilation(nn.Module):
    """Four dilated 3x3 convolutions over a narrowed stage, summed into one map."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        for index, spacing in enumerate((5, 7, 9, 11), start=1):
            setattr(
                self,
                f"conv2_{index}",
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=spacing,
                    padding=spacing,
                    bias=False,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Narrow the stage and read it at four spacings.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, outputs, height, width)`` tensor.
        """
        x = self.conv1(functional.relu(x))
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class Attention(nn.Module):
    """Compact spatial attention: a one-channel gate multiplied back into the stage."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weight the stage by its own gate.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, channels, height, width)`` tensor.
        """
        gate = self.conv2(self.conv1(functional.relu(x)))
        return x * torch.sigmoid(gate)


class MapReduce(nn.Module):
    """A 1x1 projection of a stage down to a single edge channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the stage.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            ``(batch, 1, height, width)`` tensor.
        """
        return self.conv(x)


class Network(nn.Module):
    """The four-stage edge network, answering one edge map per stage and one fused map."""

    def __init__(self):
        super().__init__()
        self.init_block = PixelDifference(STAGES[0], 3, CHANNELS)
        index = 1
        inputs = CHANNELS
        for stage, outputs in enumerate(FUSE_CHANNELS, start=1):
            for position in range(1, STAGE_BLOCKS[stage - 1] + 1):
                stride = 2 if stage > 1 and position == 1 else 1
                setattr(
                    self,
                    f"block{stage}_{position}",
                    Block(STAGES[index], inputs, outputs, stride),
                )
                inputs = outputs
                index += 1
        self.conv_reduces = nn.ModuleList(
            MapReduce(DILATION_CHANNELS) for _ in FUSE_CHANNELS
        )
        self.attentions = nn.ModuleList(
            Attention(DILATION_CHANNELS) for _ in FUSE_CHANNELS
        )
        self.dilations = nn.ModuleList(
            Dilation(channels, DILATION_CHANNELS) for channels in FUSE_CHANNELS
        )
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Trace edges at four scales and fuse them.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, blue first.

        Returns:
            Five ``(batch, 1, height, width)`` maps on a 0 to 1 scale, one per stage and
            the fused map last.
        """
        height, width = x.shape[2:]
        h = self.init_block(x)
        stages = []
        for stage, count in enumerate(STAGE_BLOCKS, start=1):
            for position in range(1, count + 1):
                h = getattr(self, f"block{stage}_{position}")(h)
            stages.append(h)
        maps = []
        for index, stage in enumerate(stages):
            reduced = self.conv_reduces[index](
                self.attentions[index](self.dilations[index](stage))
            )
            maps.append(
                functional.interpolate(
                    reduced, (height, width), mode="bilinear", align_corners=False
                )
            )
        maps.append(self.classifier(torch.cat(maps, dim=1)))
        return [torch.sigmoid(edges) for edges in maps]


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("pidi", REPO_ID, FILENAME), _build, name="PiDiNet")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, feature=FEATURE, what="The pixel difference network"
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # The published file wraps the weights and carries a DataParallel prefix on every key.
    state = checkpoint.get("state_dict", checkpoint)
    network = Network()
    network.load_state_dict({
        key.removeprefix("module."): value for key, value in state.items()
    })
    return network.float().eval()
