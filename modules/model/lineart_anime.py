"""Anime line extraction, on the ControlNet lineart anime network.

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
FOLDER = "lineart_anime"

#: Repository publishing the weights, and the file inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "netG.pth"

#: Channels the network reads, and the channels it answers with.
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

#: Filters the outermost level convolves the frame down to.
BASE_FILTERS = 64

#: Outer and inner channel counts of every level below the outermost, top down. The last
#: pair is the bottleneck.
STAGES = (
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 512),
    (512, 512),
    (512, 512),
    (512, 512),
)

#: Levels the frame is halved through, counting the outermost.
DEPTH = len(STAGES) + 1

#: Multiple both frame dimensions have to be, one halving per level.
FRAME_MULTIPLE = 2**DEPTH

#: Side of every convolution kernel, its stride, and its padding.
KERNEL = 4
STRIDE = 2
PADDING = 1

#: Slope the rectifier keeps below zero on the way down.
LEAK = 0.2

#: Prefix ``torch.nn.DataParallel`` left on every key of the published checkpoint.
CHECKPOINT_PREFIX = "module."


def _norm(channels: int) -> nn.Module:
    """Instance normalisation with no learned scale and no running statistics.

    Args:
        channels: Channels the normalisation runs over.

    Returns:
        A parameter-free ``nn.InstanceNorm2d``.
    """
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)


class SkipBlock(nn.Module):
    """One level of the U: a halving convolution, the level below it, a doubling one."""

    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        input_channels: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_channels is None:
            input_channels = outer_channels
        down = nn.Conv2d(input_channels, inner_channels, KERNEL, STRIDE, PADDING)
        if outermost:
            layers = [
                down,
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels * 2, outer_channels, KERNEL, STRIDE, PADDING
                ),
                nn.Tanh(),
            ]
        elif innermost:
            layers = [
                nn.LeakyReLU(LEAK, True),
                down,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels, outer_channels, KERNEL, STRIDE, PADDING
                ),
                _norm(outer_channels),
            ]
        else:
            layers = [
                nn.LeakyReLU(LEAK, True),
                down,
                _norm(inner_channels),
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels * 2, outer_channels, KERNEL, STRIDE, PADDING
                ),
                _norm(outer_channels),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the level, carrying its input forward beside its output.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            The level's own output on the outermost level, otherwise that output with the
            input concatenated in front of it along the channel axis.
        """
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class Network(nn.Module):
    """The U-net generator that reads a colour frame and answers with a line drawing."""

    def __init__(self):
        super().__init__()
        outer, inner = STAGES[-1]
        block = SkipBlock(outer, inner, innermost=True)
        for outer, inner in reversed(STAGES[:-1]):
            block = SkipBlock(outer, inner, submodule=block)
        self.model = SkipBlock(
            OUTPUT_CHANNELS,
            BASE_FILTERS,
            input_channels=INPUT_CHANNELS,
            submodule=block,
            outermost=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw the lines of a frame.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a -1 to 1 scale, red first. Both
                dimensions have to be multiples of ``FRAME_MULTIPLE``.

        Returns:
            A ``(batch, 1, height, width)`` tensor on a -1 to 1 scale, lines towards -1
            against a background towards 1.
        """
        return self.model(x)


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("lineart_anime", REPO_ID, FILENAME), _build, name="Lineart Anime")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, feature=FEATURE, what="The anime lineart network"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    network = Network()
    network.load_state_dict(state)
    return network.float().eval()
