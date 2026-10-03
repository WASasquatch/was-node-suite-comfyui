"""TEED soft edge, on the tiny and efficient edge detector network.

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
FOLDER = "teed"

#: Repository publishing the weights, the directory inside it, and the file.
REPO_ID = "bdsqlsz/qinglong_controlnet-lllite"
SUBFOLDER = "Annotators"
FILENAME = "7_model.pth"

#: The AnyLine variant of the same network, published in another repository.
MISTO_REPO_ID = "TheMistoAI/MistoLine"
MISTO_SUBFOLDER = "Anyline"
MISTO_FILENAME = "MTEED.pth"

#: Channels each side branch reads, and how many doublings it takes back to full size.
STAGES = ((16, 1), (32, 1), (48, 2))

#: Transposed-convolution padding, indexed by the number of doublings a branch makes.
DECONV_PADDING = (0, 0, 1, 3, 7)

#: Channels a doubling carries, until the last one narrows to a single map.
UPSAMPLE_CHANNELS = 16

#: Channels the dense block works at.
DENSE_CHANNELS = 48

#: Multiple both frame sides must be for the three branches to come back the same size.
SIZE_MULTIPLE = 8


def smish(x: torch.Tensor) -> torch.Tensor:
    """Apply the smish activation, ``x * tanh(log(1 + sigmoid(x)))``.

    Args:
        x: Tensor of any shape.

    Returns:
        A tensor of the same shape.
    """
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class Smish(nn.Module):
    """Smish as a layer, for the sequences that hold their activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply smish.

        Args:
            x: Tensor of any shape.

        Returns:
            A tensor of the same shape.
        """
        return smish(x)


class DoubleConvBlock(nn.Module):
    """Two 3x3 convolutions with smish between them, and after the second where asked."""

    def __init__(
        self,
        in_features: int,
        mid_features: int,
        out_features: int | None = None,
        stride: int = 1,
        use_act: bool = True,
    ):
        super().__init__()
        self.use_act = use_act
        out_features = mid_features if out_features is None else out_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both convolutions.

        Args:
            x: ``(batch, in_features, height, width)`` tensor.

        Returns:
            A ``(batch, out_features, height, width)`` tensor, halved on each side when the
            block was built with a stride of two.
        """
        h = self.conv2(smish(self.conv1(x)))
        return smish(h) if self.use_act else h


class SingleConvBlock(nn.Module):
    """A 1x1 convolution projecting channels, striding where asked."""

    def __init__(self, in_features: int, out_features: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the channels.

        Args:
            x: ``(batch, in_features, height, width)`` tensor.

        Returns:
            A ``(batch, out_features, height, width)`` tensor, subsampled by the stride.
        """
        return self.conv(x)


class DenseLayer(nn.Module):
    """A 3x3 convolution pair averaging its result with the skip it is handed."""

    def __init__(self, input_features: int, out_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True
        )
        self.smish1 = Smish()
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve the first tensor and average it with the second.

        Args:
            x: ``(features, skip)``, both ``(batch, channels, height, width)`` tensors.

        Returns:
            ``(averaged, skip)``, the skip passed on untouched for the next layer.
        """
        features, skip = x
        convolved = self.conv2(self.smish1(self.conv1(smish(features))))
        return 0.5 * (convolved + skip), skip


class DenseBlock(nn.Module):
    """A run of dense layers, each one averaging against the same skip."""

    def __init__(self, num_layers: int, input_features: int, out_features: int):
        super().__init__()
        for index in range(1, num_layers + 1):
            self.add_module(f"denselayer{index}", DenseLayer(input_features, out_features))
            input_features = out_features

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run every layer in turn.

        Args:
            x: ``(features, skip)``, both ``(batch, channels, height, width)`` tensors.

        Returns:
            ``(features, skip)`` from the last layer.
        """
        for layer in self.children():
            x = layer(x)
        return x


class UpConvBlock(nn.Module):
    """Projects a branch down to one channel and doubles it back to the frame size."""

    def __init__(self, in_features: int, up_scale: int):
        super().__init__()
        self.features = nn.Sequential(*_doublings(in_features, up_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and upsample.

        Args:
            x: ``(batch, in_features, height, width)`` tensor.

        Returns:
            A ``(batch, 1, height * 2 ** up_scale, width * 2 ** up_scale)`` tensor.
        """
        return self.features(x)


def _doublings(in_features: int, up_scale: int) -> list[nn.Module]:
    """The projection, activation and transposed convolution of each doubling."""
    kernel_size = 2 ** up_scale
    padding = DECONV_PADDING[up_scale]
    layers: list[nn.Module] = []
    for step in range(up_scale):
        out_features = 1 if step == up_scale - 1 else UPSAMPLE_CHANNELS
        layers.append(nn.Conv2d(in_features, out_features, 1))
        layers.append(Smish())
        layers.append(
            nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=padding
            )
        )
        in_features = out_features
    return layers


class DoubleFusion(nn.Module):
    """Weights the stacked branch maps by two depthwise convolutions and sums them to one."""

    def __init__(self, in_ch: int):
        super().__init__()
        # Named for the checkpoint rather than for the style of this pack.
        self.DWconv1 = nn.Conv2d(
            in_ch, in_ch * 8, kernel_size=3, stride=1, padding=1, groups=in_ch
        )
        self.DWconv2 = nn.Conv2d(
            in_ch * 8, in_ch * 8, kernel_size=3, stride=1, padding=1, groups=in_ch * 8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse the stacked maps.

        Args:
            x: ``(batch, in_ch, height, width)`` tensor of stacked branch maps.

        Returns:
            A ``(batch, 1, height, width)`` tensor.
        """
        attention = self.DWconv1(smish(x))
        weighted = self.DWconv2(smish(attention))
        return smish((weighted + attention).sum(1).unsqueeze(1))


class Network(nn.Module):
    """The tiny edge detector, answering one map per branch and their fusion."""

    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = DenseBlock(1, 32, DENSE_CHANNELS)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, DENSE_CHANNELS, 1)
        for index, (channels, up_scale) in enumerate(STAGES, start=1):
            setattr(self, f"up_block_{index}", UpConvBlock(channels, up_scale))
        self.block_cat = DoubleFusion(len(STAGES))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Trace edges on three branches and fuse them.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 255 scale, red first, with both
                sides a multiple of :data:`SIZE_MULTIPLE`.

        Returns:
            Four ``(batch, 1, height, width)`` maps, one per branch and the fusion of the
            three, as logits.
        """
        block_1 = self.block_1(x)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + self.side_1(block_1)
        block_3, _ = self.dblock_3((block_2_add, self.pre_dense_3(block_2_down)))
        results = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
        ]
        results.append(self.block_cat(torch.cat(results, dim=1)))
        return results


def fuse(maps: list[torch.Tensor], steps: int = 2) -> torch.Tensor:
    """Average the branch maps into one edge image.

    Args:
        maps: The four ``(batch, 1, height, width)`` logit maps the network answers.
        steps: Levels to quantise the result to, or 0 to leave it as it is.

    Returns:
        A ``(batch, 1, height, width)`` tensor on a 0 to 1 scale.
    """
    edge = torch.sigmoid(torch.stack(maps, dim=0).mean(dim=0).double()).float()
    if steps:
        edge = (edge * float(steps + 1)).to(torch.int32).float() / float(steps)
    return edge.clamp(0.0, 1.0)


def load_misto(device: str | None = None) -> Network:
    """Build the network and read the AnyLine weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(
        ("teed_misto", MISTO_REPO_ID, MISTO_FILENAME), _build_misto, name="AnyLine"
    )


def _build_misto() -> Network:
    """Read the AnyLine checkpoint into a freshly built network."""
    path = published_checkpoint(
        FOLDER, MISTO_REPO_ID, MISTO_FILENAME, subfolder=MISTO_SUBFOLDER,
        feature=FEATURE, what="The AnyLine network",
    )
    network = Network()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.float().eval()


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("teed", REPO_ID, FILENAME), _build, name="TEED")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, FILENAME, subfolder=SUBFOLDER, feature=FEATURE, what="The TEED network"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    network = Network()
    network.load_state_dict(state)
    return network.float().eval()
