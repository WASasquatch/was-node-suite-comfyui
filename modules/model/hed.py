"""Holistically-nested edge detection, on the ControlNet HED network.

:func:`load` answers a five-stage network taking ``(batch, 3, height, width)`` on a 0 to 255
scale and answering one edge map per stage.
"""

from __future__ import annotations

import torch
from torch import nn

from . import managed_module, published_checkpoint

__all__ = ["FILENAME", "Network", "REPO_ID", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "hed"

#: Repository publishing the weights, and the file inside it.
REPO_ID = "lllyasviel/Annotators"
FILENAME = "ControlNetHED.pth"

#: Channels each stage works at, and how many convolutions it stacks.
STAGES = ((3, 64, 2), (64, 128, 2), (128, 256, 3), (256, 512, 3), (512, 512, 3))


class DoubleConvBlock(nn.Module):
    """One stage: a stack of 3x3 convolutions and a 1x1 projection down to one channel."""

    def __init__(self, input_channel: int, output_channel: int, layer_number: int):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(input_channel, output_channel, (3, 3), (1, 1), 1))
        for _ in range(1, layer_number):
            self.convs.append(nn.Conv2d(output_channel, output_channel, (3, 3), (1, 1), 1))
        self.projection = nn.Conv2d(output_channel, 1, (1, 1), (1, 1), 0)

    def forward(self, x: torch.Tensor, down_sampling: bool = False):
        """Run the stage, halving the frame first where asked.

        Args:
            x: ``(batch, channels, height, width)`` tensor.
            down_sampling: True to max-pool by two before the convolutions.

        Returns:
            ``(features, projection)``, the stage's own output and its one-channel edge map.
        """
        h = torch.nn.functional.max_pool2d(x, (2, 2), (2, 2)) if down_sampling else x
        for conv in self.convs:
            h = torch.nn.functional.relu(conv(h))
        return h, self.projection(h)


class Network(nn.Module):
    """The five-stage edge network, answering one edge map per stage."""

    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        for index, (inputs, outputs, layers) in enumerate(STAGES, start=1):
            setattr(self, f"block{index}", DoubleConvBlock(inputs, outputs, layers))

    def forward(self, x: torch.Tensor):
        """Trace edges at five scales.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 255 scale, red first.

        Returns:
            Five ``(batch, 1, height, width)`` maps, each half the size of the one before.
        """
        h = x - self.norm
        projections = []
        for index in range(1, len(STAGES) + 1):
            h, projection = getattr(self, f"block{index}")(h, down_sampling=index > 1)
            projections.append(projection)
        return projections


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The network is small enough to rest where it is built, and the
            caller moves it.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("hed", REPO_ID, FILENAME), _build, name="HED")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, feature=FEATURE, what="The edge network"
    )
    network = Network()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.float().eval()


