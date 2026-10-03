"""Background removal, on the BEN2 matting network.

:func:`load` answers the network with its published weights, taking ``(batch, 3, height,
width)`` on a 0 to 1 scale, each side a multiple of :data:`MULTIPLE`, and answering one
alpha channel.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional

from . import (
    ModelUnavailable,
    checkpoint_shapes,
    managed_module,
    model_directories,
    model_file_path,
    published_checkpoint,
    unresolved_links,
)
from .swin import DEPTHS, EMBED_DIM_BASE, HEADS_BASE, WINDOW_SIZE, SwinTransformer

__all__ = [
    "FEATURE",
    "FILENAME",
    "FOLDER",
    "fits",
    "load",
    "locate",
    "MULTIPLE",
    "Network",
    "REPO_ID",
    "SIGNATURE",
    "SMALLEST_SIDE",
    "SUBFOLDER",
    "TRAINED_SIDE",
]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "ben2"

#: Repository publishing the weights, the directory inside it, and the file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "ben2"
FILENAME = "ben2-base.safetensors"

#: A tensor every checkpoint this network loads carries, and its shape.
SIGNATURE = ("backbone.patch_embed.proj.weight", [128, 3, 4, 4])

#: Channels every decoder stage and every head works at.
EMBED_DIM = EMBED_DIM_BASE

#: Channels the four backbone stages answer, finest first.
STAGE_CHANNELS = tuple(EMBED_DIM_BASE * 2**index for index in range(len(DEPTHS)))

#: Attention heads the two cross attention blocks split their channels across.
HEADS = 1

#: Ratios the joined quarters are pooled to inside the localisation block.
LOCALISATION_POOLS = (1, 4, 8)

#: Ratios the whole view is pooled to inside each refinement block.
REFINEMENT_POOLS = (2, 4, 8)

#: Channels the instance mask head widens to between its three convolutions.
HEAD_WIDTH = 384

#: Multiplier from a block's channels to the width of its feed-forward layer.
FFN_RATIO = 2

#: Wavelength range of the sine position embedding, and the epsilon normalising it.
POSITION_TEMPERATURE = 10000
POSITION_EPS = 1e-6

#: Per-channel mean and standard deviation the frame is standardised by.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Multiple both sides of a frame must be.
MULTIPLE = 128

#: Shortest side the pooled fields survive at. Below this the coarsest of them rounds to
#: nothing and the block attends over fewer scales than it holds weights for.
SMALLEST_SIDE = 384

#: Side the published weights were trained at.
TRAINED_SIDE = 1024


def _cbr(in_dim: int, out_dim: int) -> nn.Sequential:
    """A 3x3 convolution, an instance norm and a GELU.

    Args:
        in_dim: Channels read.
        out_dim: Channels answered.

    Returns:
        The three layers in sequence.
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.GELU(),
    )


def _patches(x: torch.Tensor) -> torch.Tensor:
    """Cut each frame into its four quarters, stacked along the batch axis.

    Args:
        x: ``(batch, channels, height, width)`` tensor, both sides even.

    Returns:
        A ``(batch * 4, channels, height // 2, width // 2)`` tensor, top left, top right,
        bottom left, bottom right.
    """
    batch, channels, height, width = x.shape
    quads = x.reshape(batch, channels, 2, height // 2, 2, width // 2)
    quads = quads.permute(2, 4, 0, 1, 3, 5)
    return quads.reshape(batch * 4, channels, height // 2, width // 2)


def _unpatch(x: torch.Tensor) -> torch.Tensor:
    """Lay four quarters back out as one frame.

    Args:
        x: ``(batch * 4, channels, height, width)`` tensor.

    Returns:
        A ``(batch, channels, height * 2, width * 2)`` tensor.
    """
    total, channels, height, width = x.shape
    frame = x.reshape(2, 2, total // 4, channels, height, width)
    frame = frame.permute(2, 3, 0, 4, 1, 5)
    return frame.reshape(total // 4, channels, height * 2, width * 2)


def _resize_as(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Resample a frame onto another frame's grid.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        other: Tensor whose last two dimensions give the target grid.

    Returns:
        A ``(batch, channels, other height, other width)`` tensor.
    """
    return functional.interpolate(x, size=other.shape[-2:], mode="bilinear")


def _double(x: torch.Tensor) -> torch.Tensor:
    """Repeat every pixel over a 2x2 square.

    Args:
        x: ``(batch, channels, height, width)`` tensor.

    Returns:
        A ``(batch, channels, height * 2, width * 2)`` tensor.
    """
    return functional.interpolate(x, scale_factor=2, mode="nearest")


def _interleave(angles: torch.Tensor) -> torch.Tensor:
    """Take the sine of every even channel and the cosine of every odd one.

    Args:
        angles: Tensor whose last axis holds an even count of angles.

    Returns:
        A tensor of the same shape.
    """
    paired = torch.stack((angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1)
    return paired.flatten(-2)


def _position_embedding(
    height: int, width: int, features: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Answer the sine position embedding of a grid, laid out as a token sequence.

    Args:
        height: Rows of the grid.
        width: Columns of the grid.
        features: Channels the row takes, matched by the column.
        device: Where to build the embedding.
        dtype: Type to answer in.

    Returns:
        A ``(height * width, 1, features * 2)`` tensor, broadcasting across the batch.
    """
    scale = 2 * math.pi
    index = torch.arange(features, dtype=torch.float32, device=device)
    dim_t = POSITION_TEMPERATURE ** (
        2 * torch.div(index, 2, rounding_mode="floor") / features
    )
    rows = torch.arange(1, height + 1, dtype=torch.float32, device=device) - 0.5
    columns = torch.arange(1, width + 1, dtype=torch.float32, device=device) - 0.5
    rows = (rows / (height + POSITION_EPS) * scale).view(height, 1, 1) / dim_t
    columns = (columns / (width + POSITION_EPS) * scale).view(1, width, 1) / dim_t
    down = _interleave(rows).expand(height, width, features)
    across = _interleave(columns).expand(height, width, features)
    embedded = torch.cat([down, across], dim=-1)
    return embedded.reshape(height * width, 1, features * 2).to(dtype)


class LocalisationBlock(nn.Module):
    """The whole view and its four quarters rewriting each other, at the coarsest scale."""

    def __init__(self, dim: int, num_heads: int, pools: tuple[int, ...]):
        super().__init__()
        self.pools = pools
        self.features = dim // 2
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads) for _ in range(5)
        )
        self.linear1 = nn.Linear(dim, dim * FFN_RATIO)
        self.linear2 = nn.Linear(dim * FFN_RATIO, dim)
        self.linear3 = nn.Linear(dim, dim * FFN_RATIO)
        self.linear4 = nn.Linear(dim * FFN_RATIO, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, loc: torch.Tensor, glb: torch.Tensor) -> torch.Tensor:
        """Attend the view over the joined quarters, then each quarter over the view.

        Args:
            loc: ``(batch * 4, dim, height, width)`` tensor, one row per quarter.
            glb: ``(batch, dim, height, width)`` tensor holding the whole view.

        Returns:
            A ``(batch * 5, dim, height, width)`` tensor, the four quarters then the view.
        """
        _, channels, height, width = loc.shape
        joined = _unpatch(loc)
        pools = []
        positions = []
        for ratio in self.pools:
            target = (round(height / ratio), round(width / ratio))
            pooled = functional.adaptive_avg_pool2d(joined, target)
            pools.append(pooled.flatten(2).permute(2, 0, 1))
            positions.append(
                _position_embedding(*target, self.features, loc.device, loc.dtype)
            )
        keys = torch.cat(pools, 0)
        key_pos = torch.cat(positions, 0)
        query_pos = _position_embedding(
            glb.shape[2], glb.shape[3], self.features, loc.device, loc.dtype
        )

        view = glb.flatten(2).permute(2, 0, 1)
        view = view + self.attention[0](view + query_pos, keys + key_pos, keys)[0]
        view = self.norm1(view)
        view = view + self.linear2(functional.gelu(self.linear1(view)))
        view = self.norm2(view)

        quarters = loc.flatten(2).permute(2, 0, 1)
        split = view.reshape(2, height // 2, 2, width // 2, -1, channels)
        split = split.permute(1, 3, 0, 2, 4, 5).reshape(height * width // 4, -1, channels)
        answered = [
            self.attention[index + 1](query, region, region)[0]
            for index, (query, region) in enumerate(
                zip(quarters.chunk(4, dim=1), split.chunk(4, dim=1))
            )
        ]
        quarters = quarters + torch.cat(answered, 1)
        quarters = self.norm1(quarters)
        quarters = quarters + self.linear4(functional.gelu(self.linear3(quarters)))
        quarters = self.norm2(quarters)
        merged = torch.cat((quarters, view), 1)
        return merged.permute(1, 2, 0).reshape(-1, channels, height, width)


class RefinementBlock(nn.Module):
    """Each quarter attending over the pooled view, which is then refreshed from them."""

    def __init__(self, dim: int, num_heads: int, pools: tuple[int, ...]):
        super().__init__()
        self.pools = pools
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads) for _ in range(4)
        )
        self.linear3 = nn.Linear(dim, dim * FFN_RATIO)
        self.linear4 = nn.Linear(dim * FFN_RATIO, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.sal_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate each quarter by the view's salience, attend it over the view, refresh both.

        Args:
            x: ``(batch * 5, dim, height, width)`` tensor, the four quarters then the view.

        Returns:
            A ``(batch * 5, dim, height, width)`` tensor in the same order.
        """
        _, channels, height, width = x.shape
        loc, glb = x.split([4, 1], dim=0)
        patched = _patches(glb)
        salience = torch.sigmoid(self.sal_conv(glb))
        salience = functional.interpolate(
            salience, size=(height * 2, width * 2), mode="nearest"
        )
        loc = loc * _patches(salience)
        pools = [
            functional.adaptive_avg_pool2d(
                patched, (round(height / ratio), round(width / ratio))
            ).flatten(2)
            for ratio in self.pools
        ]
        keys = torch.cat(pools, 2).permute(0, 2, 1).unsqueeze(2)
        queries = loc.flatten(2).permute(0, 2, 1).unsqueeze(2)
        answered = [
            self.attention[index](query, key, key)[0]
            for index, (query, key) in enumerate(zip(queries.unbind(0), keys.unbind(0)))
        ]
        tokens = loc.reshape(4, channels, -1).permute(2, 0, 1) + torch.cat(answered, 1)
        tokens = self.norm1(tokens)
        tokens = tokens + self.linear4(functional.gelu(self.linear3(tokens)))
        tokens = self.norm2(tokens)
        refined = tokens.permute(1, 2, 0).reshape(4, channels, height, width)
        spread = functional.interpolate(
            _unpatch(refined), size=glb.shape[-2:], mode="nearest"
        )
        return torch.cat((refined, glb + spread), 0)


class Network(nn.Module):
    """The BEN2 body, answering one alpha mask per frame it is given."""

    def __init__(self):
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim=EMBED_DIM_BASE,
            depths=DEPTHS,
            num_heads=HEADS_BASE,
            window_size=WINDOW_SIZE,
        )
        for index, width in enumerate((STAGE_CHANNELS[0],) + STAGE_CHANNELS, start=1):
            setattr(self, f"output{index}", _cbr(width, EMBED_DIM))
            setattr(
                self,
                f"sideout{index}",
                nn.Sequential(nn.Conv2d(EMBED_DIM, 1, kernel_size=3, padding=1)),
            )
        self.multifieldcrossatt = LocalisationBlock(EMBED_DIM, HEADS, LOCALISATION_POOLS)
        for index in range(1, len(DEPTHS) + 1):
            setattr(self, f"conv{index}", _cbr(EMBED_DIM, EMBED_DIM))
            setattr(
                self,
                f"dec_blk{index}",
                RefinementBlock(EMBED_DIM, HEADS, REFINEMENT_POOLS),
            )
        self.insmask_head = nn.Sequential(
            nn.Conv2d(EMBED_DIM, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.InstanceNorm2d(HEAD_WIDTH),
            nn.GELU(),
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.InstanceNorm2d(HEAD_WIDTH),
            nn.GELU(),
            nn.Conv2d(HEAD_WIDTH, EMBED_DIM, kernel_size=3, padding=1),
        )
        self.shallow = nn.Sequential(nn.Conv2d(3, EMBED_DIM, kernel_size=3, padding=1))
        self.upsample1 = _cbr(EMBED_DIM, EMBED_DIM)
        self.upsample2 = _cbr(EMBED_DIM, EMBED_DIM)
        self.output = nn.Sequential(nn.Conv2d(EMBED_DIM, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read every frame as four quarters beside a halved whole view, and cut it out.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with both
                sides a multiple of :data:`MULTIPLE`.

        Returns:
            A ``(batch, 1, height, width)`` alpha mask on a 0 to 1 scale.
        """
        mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        frames = (x - mean) / std
        halved = functional.interpolate(frames, scale_factor=0.5, mode="bilinear")
        views = torch.cat(
            [
                torch.cat(
                    (_patches(frames[index : index + 1]), halved[index : index + 1]),
                    dim=0,
                )
                for index in range(frames.shape[0])
            ],
            dim=0,
        )
        features = self.backbone(views)
        masks = []
        for index in range(frames.shape[0]):
            span = slice(index * 5, index * 5 + 5)
            e5 = self.output5(features[4][span])
            e4 = self.output4(features[3][span])
            e3 = self.output3(features[2][span])
            e2 = self.output2(features[1][span])
            e1 = self.output1(features[0][span])
            quarters, view = e5.split([4, 1], dim=0)
            e5 = self.multifieldcrossatt(quarters, view)
            e4 = self.conv4(self.dec_blk4(e4 + _resize_as(e5, e4)))
            e3 = self.conv3(self.dec_blk3(e3 + _resize_as(e4, e3)))
            e2 = self.conv2(self.dec_blk2(e2 + _resize_as(e3, e2)))
            e1 = self.conv1(self.dec_blk1(e1 + _resize_as(e2, e1)))
            quarters, view = e1.split([4, 1], dim=0)
            joined = _unpatch(quarters)
            joined = joined + _resize_as(view, joined)
            detail = self.shallow(frames[index : index + 1])
            fine = self.insmask_head(joined)
            fine = fine + _resize_as(detail, fine)
            fine = self.upsample1(_double(fine))
            fine = _double(fine + _resize_as(detail, fine))
            masks.append(self.output(self.upsample2(fine)).sigmoid())
        return torch.cat(masks, dim=0)


def fits(path: str | os.PathLike) -> bool:
    """Whether a safetensors file holds weights this network loads.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        True when its header carries :data:`SIGNATURE`.
    """
    key, shape = SIGNATURE
    return checkpoint_shapes(path).get(key) == shape


def locate(name: str = FILENAME) -> str:
    """The path of one checkpoint, fetching the published file when that is allowed.

    Args:
        name: A file in the ``ben2`` model folder, relative to it.

    Returns:
        An absolute path.

    Raises:
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    found = model_file_path(FOLDER, name)
    if found is not None:
        return str(found)
    if os.path.basename(name) == FILENAME:
        return published_checkpoint(
            FOLDER, REPO_ID, FILENAME, subfolder=SUBFOLDER, feature=FEATURE,
            what="The background removal network",
        )
    roots = model_directories(FOLDER)
    searched = "\n".join(f"    {root}" for root in roots) or "    nowhere, ComfyUI is not running"
    raise ModelUnavailable(
        "\n".join([f"The background removal network needs {name}, which is not in:",
                   searched, *unresolved_links(roots, name)])
    )


def load(name: str = FILENAME, device: str | None = None):
    """Build the network and read one checkpoint into it.

    Args:
        name: A file in the ``ben2`` model folder, relative to it.
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        A :class:`~modules.model.Backend` whose ``model`` is the network in eval mode,
        answering one alpha mask per frame it is given, built once per file and kept for
        the process.

    Raises:
        ValueError: The file does not hold BEN2 weights.
        ModelUnavailable: The file is not on disk, and is not published or
            ``features.network`` is off.
    """
    path = locate(name)
    return managed_module(("ben2", path), lambda: _build(path), name="BEN2")


def _build(path: str) -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    from safetensors.torch import load_file

    if not fits(path):
        key, shape = SIGNATURE
        raise ValueError(
            f"{path} is not a BEN2 checkpoint this node can build: it has no {key} "
            f"shaped {shape}. Pick the BEN2 base file."
        )
    network = Network()
    network.load_state_dict(load_file(path), strict=True)
    return network.float().eval()
