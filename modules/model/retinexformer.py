"""Low light enhancement, on the Retinexformer network.

:func:`load` answers a network taking ``(batch, 3, height, width)`` RGB on a 0 to 1 scale
and answering the same shape. :data:`MODELS` names the scenario each file was trained for.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from . import managed_module, published_checkpoint

__all__ = [
    "FEATURE",
    "FILENAME",
    "FOLDER",
    "MODELS",
    "Network",
    "REPO_ID",
    "SUBFOLDER",
    "load",
]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "low_light"

#: Repository publishing the weights, the directory inside it, and the default file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "low_light"
FILENAME = "retinexformer-ntire.pth"

#: The file holding each scenario's weights, by the widget option naming it.
MODELS = {
    "Retinexformer NTIRE": FILENAME,
    "Retinexformer LOL v1": "retinexformer-lol-v1.pth",
    "Retinexformer LOL v2 Real": "retinexformer-lol-v2-real.pth",
    "Retinexformer LOL v2 Synthetic": "retinexformer-lol-v2-synthetic.pth",
    "Retinexformer FiveK": "retinexformer-fivek.pth",
    "Retinexformer Extreme Dark": "retinexformer-sid.pth",
    "Retinexformer Dark Motion": "retinexformer-smid.pth",
    "Retinexformer Indoor Night": "retinexformer-sdsd-indoor.pth",
    "Retinexformer Outdoor Night": "retinexformer-sdsd-outdoor.pth",
}

#: A single stage, 40 features at the finest scale, and two halvings below it.
STAGE = 1
N_FEAT = 40
LEVEL = 2

#: Attention blocks stacked at the two encoder scales and at the bottleneck.
BLOCKS = (1, 2, 2)

#: Channels the estimator reads: the three colour channels and their mean.
PRIOR_CHANNELS = 4

#: Multiplier from a block's channels to the width of its feed-forward layer.
EXPANSION = 4


class Estimator(nn.Module):
    """Illumination estimator: a 1x1 lift, a grouped 5x5 and a 1x1 back to three channels."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.conv1 = nn.Conv2d(PRIOR_CHANNELS, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=PRIOR_CHANNELS
        )
        self.conv2 = nn.Conv2d(n_feat, 3, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a frame beside its mean channel and answer the illumination it carries.

        Args:
            img: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            ``(feature, map)``: a ``(batch, n_feat, height, width)`` feature the attention
            is guided by, and a ``(batch, 3, height, width)`` per-channel illumination.
        """
        prior = img.mean(dim=1).unsqueeze(1)
        lifted = self.conv1(torch.cat([img, prior], dim=1))
        feature = self.depth_conv(lifted)
        return feature, self.conv2(feature)


class Attention(nn.Module):
    """Illumination-guided attention across channels, split over heads of ``dim_head``."""

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x: torch.Tensor, illu: torch.Tensor) -> torch.Tensor:
        """Attend channel against channel, with the values scaled by the illumination.

        Args:
            x: ``(batch, height, width, dim)`` tensor.
            illu: ``(batch, height, width, dim)`` illumination feature.

        Returns:
            A ``(batch, height, width, dim)`` tensor.
        """
        batch, height, width, channels = x.shape
        flat = x.reshape(batch, height * width, channels)
        q_inp, k_inp, v_inp = self.to_q(flat), self.to_k(flat), self.to_v(flat)
        q, k, v, guide = [
            self._heads(t) for t in (q_inp, k_inp, v_inp, illu.flatten(1, 2))
        ]
        v = (v * guide).transpose(-2, -1)
        q = functional.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = functional.normalize(k.transpose(-2, -1), dim=-1, p=2)
        scores = (k @ q.transpose(-2, -1)) * self.rescale
        attended = scores.softmax(dim=-1) @ v
        merged = attended.permute(0, 3, 1, 2).reshape(
            batch, height * width, self.num_heads * self.dim_head
        )
        projected = self.proj(merged).view(batch, height, width, channels)
        position = v_inp.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return projected + self.pos_emb(position).permute(0, 2, 3, 1)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        """Split ``(batch, pixels, dim)`` into ``(batch, heads, pixels, dim_head)``."""
        batch, pixels, _ = t.shape
        return t.reshape(batch, pixels, self.num_heads, self.dim_head).permute(0, 2, 1, 3)


class FeedForward(nn.Module):
    """Two 1x1 projections around a depthwise 3x3, widened by :data:`EXPANSION` between."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = dim * EXPANSION
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix each pixel's channels and then its neighbours.

        Args:
            x: ``(batch, height, width, dim)`` tensor.

        Returns:
            A ``(batch, height, width, dim)`` tensor.
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class PreNorm(nn.Module):
    """A layer norm over the channel axis, run before the layer it wraps."""

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise every pixel across its channels and run the wrapped layer.

        Args:
            x: ``(batch, height, width, dim)`` tensor.

        Returns:
            A ``(batch, height, width, dim)`` tensor.
        """
        return self.fn(self.norm(x))


class GuidedBlock(nn.Module):
    """A stack of attention and feed-forward pairs, each one added back to its input."""

    def __init__(self, dim: int, dim_head: int, heads: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleList([Attention(dim, dim_head, heads), PreNorm(dim, FeedForward(dim))])
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, illu: torch.Tensor) -> torch.Tensor:
        """Run every pair over the frame, guided by the illumination feature.

        Args:
            x: ``(batch, dim, height, width)`` tensor.
            illu: ``(batch, dim, height, width)`` illumination feature.

        Returns:
            A ``(batch, dim, height, width)`` tensor.
        """
        x = x.permute(0, 2, 3, 1)
        guide = illu.permute(0, 2, 3, 1)
        for attention, feed_forward in self.blocks:
            x = attention(x, guide) + x
            x = feed_forward(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    """A two-scale U-net of guided attention blocks, adding its output to its input."""

    def __init__(self, dim: int, level: int, num_blocks: tuple[int, ...]):
        super().__init__()
        self.level = level
        self.embedding = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList()
        width = dim
        for index in range(level):
            self.encoder_layers.append(
                nn.ModuleList([
                    GuidedBlock(width, dim, width // dim, num_blocks[index]),
                    nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
                    nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
                ])
            )
            width *= 2
        self.bottleneck = GuidedBlock(width, dim, width // dim, num_blocks[-1])
        self.decoder_layers = nn.ModuleList()
        for index in range(level):
            narrow = width // 2
            self.decoder_layers.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(width, narrow, kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(width, narrow, 1, 1, bias=False),
                    GuidedBlock(narrow, dim, narrow // dim, num_blocks[level - 1 - index]),
                ])
            )
            width = narrow
        self.mapping = nn.Conv2d(dim, 3, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, illu: torch.Tensor) -> torch.Tensor:
        """Carry a frame and its illumination down two scales and back up.

        Args:
            x: ``(batch, 3, height, width)`` tensor, both sides a multiple of four.
            illu: ``(batch, dim, height, width)`` illumination feature.

        Returns:
            A ``(batch, 3, height, width)`` tensor.
        """
        fea = self.embedding(x)
        skips: list[torch.Tensor] = []
        guides: list[torch.Tensor] = []
        for blocks, down_fea, down_illu in self.encoder_layers:
            fea = blocks(fea, illu)
            guides.append(illu)
            skips.append(fea)
            fea = down_fea(fea)
            illu = down_illu(illu)
        fea = self.bottleneck(fea, illu)
        for index, (up, fuse, blocks) in enumerate(self.decoder_layers):
            fea = up(fea)
            fea = fuse(torch.cat([fea, skips[self.level - 1 - index]], dim=1))
            fea = blocks(fea, guides[self.level - 1 - index])
        return self.mapping(fea) + x


class Stage(nn.Module):
    """One stage: the illumination estimator and the U-net it lights a frame for."""

    def __init__(self, n_feat: int, level: int, num_blocks: tuple[int, ...]):
        super().__init__()
        self.estimator = Estimator(n_feat)
        self.denoiser = Denoiser(n_feat, level, num_blocks)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Light a frame by the illumination estimated from it, then denoise it.

        Args:
            img: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.
        """
        feature, illumination = self.estimator(img)
        return self.denoiser(img * illumination + img, feature)


class Network(nn.Module):
    """The Retinexformer body, answering one enhanced RGB frame per frame it is given."""

    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList(Stage(N_FEAT, LEVEL, BLOCKS) for _ in range(STAGE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Brighten a frame, one stage at a time.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with both
                sides a multiple of four.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.
        """
        for stage in self.body:
            x = stage(x)
        return x


def load(name: str = "Retinexformer NTIRE", device: str | None = None) -> Network:
    """Build the network and read one scenario's published weights into it.

    Args:
        name: A key of :data:`MODELS`, naming the file to read.
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once per scenario and kept for the process.

    Raises:
        ValueError: ``name`` is not a key of :data:`MODELS`.
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    if name not in MODELS:
        raise ValueError(
            f"Retinexformer model must be one of {', '.join(MODELS)}, not {name!r}"
        )
    filename = MODELS[name]
    return managed_module(
        ("retinexformer", REPO_ID, filename), lambda: _build(filename), name="Retinexformer"
    )


def _build(filename: str) -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER, REPO_ID, filename, subfolder=SUBFOLDER, feature=FEATURE,
        what="The low light network",
    )
    weights = torch.load(path, map_location="cpu", weights_only=True)
    network = Network()
    network.load_state_dict(weights.get("params", weights), strict=True)
    return network.float().eval()
