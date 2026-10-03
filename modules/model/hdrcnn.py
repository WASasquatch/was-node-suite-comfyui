"""HDR reconstruction from a single exposure, on the HDRCNN autoencoder.

:func:`load` answers a network taking ``(batch, 3, height, width)`` RGB on a 0 to 1 scale
and answering linear light of the same size, unbounded above.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from . import Backend, managed_module, published_checkpoint

__all__ = ["FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "SUBFOLDER", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "hdr"

#: Repository publishing the weights, the directory inside it, and the file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "hdr"
FILENAME = "hdrcnn.safetensors"

#: Multiple both frame sides are padded up to before the stages run.
SIZE_MULTIPLE = 32

#: Channels of each encoder stage, its input then one entry per 3x3 convolution.
ENCODER = (
    (3, 64, 64),
    (64, 128, 128),
    (128, 256, 256, 256),
    (256, 512, 512, 512),
    (512, 512, 512, 512),
)

#: Input and output channels of each decoder stage, deepest first.
DECODER = ((512, 512), (512, 512), (512, 256), (256, 128), (128, 64))

#: Channel mean the encoder subtracts, on a 0 to 255 scale, blue then green then red.
VGG_MEAN = (103.939, 116.779, 123.68)

#: Offset added inside the logarithm the decoder works through.
LOG_OFFSET = 1.0 / 255.0

#: Distance below a channel maximum of 1 over which the prediction takes over.
HIGHLIGHT_THRESHOLD = 0.05


def log_domain(x: torch.Tensor) -> torch.Tensor:
    """Carry an activation on a 0 to 255 scale into the decoder's log domain.

    Args:
        x: Tensor of any shape, holding non-negative values.

    Returns:
        A tensor of the same shape.
    """
    return torch.log((x / 255.0) ** 2 + LOG_OFFSET)


class Upsample(nn.Module):
    """One decoder stage: a doubling transposed convolution fused with an encoder skip."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.fuse = nn.Conv2d(2 * out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Double both sides of the frame and mix the skip into the result.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.
            skip: ``(batch, out_channels, 2 * height, 2 * width)`` tensor in the log domain.

        Returns:
            A ``(batch, out_channels, 2 * height, 2 * width)`` tensor.
        """
        x = functional.relu(self.norm(self.deconv(x)))
        return self.fuse(torch.cat((x, skip), dim=1))


class Network(nn.Module):
    """A VGG16 encoder and a log domain decoder, blended into the input by highlight."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(stage[index], stage[index + 1], 3, padding=1)
                for index in range(len(stage) - 1)
            )
            for stage in ENCODER
        )
        self.latent = nn.Conv2d(512, 512, 3, padding=1)
        self.latent_norm = nn.BatchNorm2d(512)
        self.decoder = nn.ModuleList(
            Upsample(in_channels, out_channels) for in_channels, out_channels in DECODER
        )
        self.output = nn.Conv2d(64, 3, 1)
        self.output_norm = nn.BatchNorm2d(3)
        self.blend = nn.Conv2d(6, 3, 1)
        self.register_buffer(
            "channel_mean", torch.tensor(VGG_MEAN).reshape(1, 3, 1, 1), persistent=False
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run the autoencoder over a frame whose sides are multiples of 32.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` tensor in the log domain.
        """
        scaled = x * 255.0
        # Channels are reversed to blue, green, red and each mean is removed.
        h = scaled.flip(1) - self.channel_mean
        skips = []
        for stage in self.encoder:
            for conv in stage:
                h = functional.relu(conv(h))
            skips.append(log_domain(h))
            h = functional.max_pool2d(h, 2, ceil_mode=True)
        h = functional.relu(self.latent_norm(self.latent(h)))
        for stage, skip in zip(self.decoder, reversed(skips)):
            h = stage(h, skip)
        h = functional.relu(self.output_norm(self.output(h)))
        return self.blend(torch.cat((h, log_domain(scaled)), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the light the exposure clipped.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` linear light tensor, unbounded above.
        """
        height, width = x.shape[-2:]
        padding = (0, -width % SIZE_MULTIPLE, 0, -height % SIZE_MULTIPLE)
        padded = functional.pad(x, padding, mode="replicate")
        prediction = self.predict(padded)[..., :height, :width]
        # The weight runs from 0 to 1 over the last 0.05 of the brightest channel.
        weight = x.amax(dim=1, keepdim=True) - 1.0 + HIGHLIGHT_THRESHOLD
        weight = (weight / HIGHLIGHT_THRESHOLD).clamp(0.0, 1.0)
        linear = x ** 2
        highlight = torch.exp(prediction) - LOG_OFFSET
        blended = (1.0 - weight) * linear + weight * highlight
        # Clipping discards light above the cut, never below it, so the recorded level is
        # the floor.
        lifted = torch.maximum(blended, linear)
        return torch.where(weight > 0.0, lifted, linear)


def load(device: str | None = None) -> Backend:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The backend holding the network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("hdrcnn", REPO_ID, FILENAME), _build, name="HDRCNN")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    from safetensors.torch import load_file

    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, subfolder=SUBFOLDER, feature=FEATURE,
        what="The HDR reconstruction network",
    )
    network = Network()
    network.load_state_dict(load_file(path), strict=True)
    return network.float().eval()
