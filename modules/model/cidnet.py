"""Low light enhancement, on the HVI-CIDNet network.

:func:`load` answers a network taking ``(batch, 3, height, width)`` RGB on a 0 to 1 scale
and answering the same shape. :data:`CHECKPOINTS` names the scenario each file was trained
for.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from . import managed_module, published_checkpoint

__all__ = [
    "CHECKPOINTS",
    "FEATURE",
    "FILENAME",
    "FOLDER",
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
FILENAME = "hvi-cidnet-generalization.safetensors"

#: The file holding each scenario's weights, by the name :func:`load` takes.
CHECKPOINTS = {
    "generalization": FILENAME,
    "fivek": "hvi-cidnet-fivek.safetensors",
    "sice": "hvi-cidnet-sice.safetensors",
    "sony-total-dark": "hvi-cidnet-sony-total-dark.safetensors",
}

#: Channels the four scales work at.
CHANNELS = (36, 36, 72, 144)

#: Attention heads the same four scales split their channels across.
HEADS = (1, 2, 4, 8)

#: Scale each of the six attention stages works at, indexing :data:`CHANNELS`.
STAGES = (1, 2, 3, 3, 2, 1)

#: Multiplier from a block's channels to the width of its gated layer.
EXPANSION = 2.66

#: Constant added to every denominator and root of the colour transform.
EPS = 1e-8

#: Which of ``(value, q, p, t)`` the red, green and blue channels take in each hue sextant.
SEXTANTS = ((0, 3, 2), (1, 0, 2), (2, 0, 3), (2, 1, 0), (3, 2, 0), (0, 2, 1))


class LayerNorm(nn.Module):
    """Channels-first layer norm, standardising each pixel across its channels."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise every pixel and apply the scale and the shift.

        Args:
            x: ``(batch, channels, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalised = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalised + self.bias[:, None, None]


class NormDownsample(nn.Module):
    """A 3x3 convolution, a bilinear halving and a PReLU."""

    def __init__(self, in_ch: int, out_ch: int, scale: float = 0.5):
        super().__init__()
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the frame.

        Args:
            x: ``(batch, in_ch, height, width)`` tensor.

        Returns:
            A ``(batch, out_ch, height // 2, width // 2)`` tensor.
        """
        return self.prelu(self.down(x))


class NormUpsample(nn.Module):
    """A 3x3 convolution and a bilinear doubling, joined to a skip and projected back."""

    def __init__(self, in_ch: int, out_ch: int, scale: float = 2):
        super().__init__()
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True),
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Double the frame and merge the skip into it.

        Args:
            x: ``(batch, in_ch, height, width)`` tensor.
            y: ``(batch, out_ch, height * 2, width * 2)`` skip from the encoder.

        Returns:
            A ``(batch, out_ch, height * 2, width * 2)`` tensor.
        """
        doubled = self.up_scale(x)
        return self.prelu(self.up(torch.cat([doubled, y], dim=1)))


class CrossAttention(nn.Module):
    """Multi-head attention taking its queries from one branch and its keys from the other."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False
        )
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=False
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Attend across channels, one head at a time.

        Args:
            x: ``(batch, dim, height, width)`` tensor the queries come from.
            y: ``(batch, dim, height, width)`` tensor the keys and values come from.

        Returns:
            A ``(batch, dim, height, width)`` tensor.
        """
        batch, channels, height, width = x.shape
        queries = self.q_dwconv(self.q(x))
        keys, values = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
        shape = (batch, self.num_heads, channels // self.num_heads, height * width)
        queries = functional.normalize(queries.reshape(shape), dim=-1)
        keys = functional.normalize(keys.reshape(shape), dim=-1)
        values = values.reshape(shape)
        scores = (queries @ keys.transpose(-2, -1)) * self.temperature
        attended = functional.softmax(scores, dim=-1) @ values
        return self.project_out(attended.reshape(batch, channels, height, width))


class Enhancement(nn.Module):
    """A gated layer: two depthwise branches, each with a tanh residual, multiplied together."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = int(dim * EXPANSION)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.dwconv1 = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False
        )
        self.dwconv2 = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Widen the channels, gate the two halves against each other and narrow them back.

        Args:
            x: ``(batch, dim, height, width)`` tensor.

        Returns:
            A ``(batch, dim, height, width)`` tensor.
        """
        first, second = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        first = torch.tanh(self.dwconv1(first)) + first
        second = torch.tanh(self.dwconv2(second)) + second
        return self.project_out(first * second)


class ChromaAttention(nn.Module):
    """One stage of the hue and saturation branch: cross attention, then the gated layer."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.gdfn = Enhancement(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CrossAttention(dim, num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Read the intensity branch and rewrite the chroma branch from it.

        Args:
            x: ``(batch, dim, height, width)`` chroma tensor.
            y: ``(batch, dim, height, width)`` intensity tensor.

        Returns:
            A ``(batch, dim, height, width)`` tensor.
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        return self.gdfn(self.norm(x))


class IntensityAttention(nn.Module):
    """One stage of the intensity branch: cross attention and the gated layer, both residual."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = Enhancement(dim)
        self.ffn = CrossAttention(dim, num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Read the chroma branch and rewrite the intensity branch from it.

        Args:
            x: ``(batch, dim, height, width)`` intensity tensor.
            y: ``(batch, dim, height, width)`` chroma tensor.

        Returns:
            A ``(batch, dim, height, width)`` tensor.
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        return x + self.gdfn(self.norm(x))


class ColourTransform(nn.Module):
    """The HVI colour space, holding the learned density its two directions share."""

    def __init__(self):
        super().__init__()
        self.density_k = nn.Parameter(torch.full([1], 0.2))

    def to_hvi(self, img: torch.Tensor) -> torch.Tensor:
        """Convert a frame from RGB to HVI.

        Args:
            img: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` tensor holding the horizontal, the vertical and
            the intensity channel.
        """
        value = img.max(1)[0]
        floor = img.min(1)[0]
        span = value - floor + EPS
        hue = torch.zeros_like(value)
        blue_max = img[:, 2] == value
        green_max = img[:, 1] == value
        red_max = img[:, 0] == value
        hue[blue_max] = 4.0 + ((img[:, 0] - img[:, 1]) / span)[blue_max]
        hue[green_max] = 2.0 + ((img[:, 2] - img[:, 0]) / span)[green_max]
        hue[red_max] = (0.0 + ((img[:, 1] - img[:, 2]) / span)[red_max]) % 6
        hue[floor == value] = 0.0
        hue = (hue / 6.0).unsqueeze(1)
        saturation = (value - floor) / (value + EPS)
        saturation[value == 0] = 0
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        density = ((value * 0.5 * math.pi).sin() + EPS).pow(self.density_k)
        horizontal = density * saturation * (2.0 * math.pi * hue).cos()
        vertical = density * saturation * (2.0 * math.pi * hue).sin()
        return torch.cat([horizontal, vertical, value], dim=1)

    def to_rgb(self, img: torch.Tensor) -> torch.Tensor:
        """Convert a frame from HVI back to RGB.

        Args:
            img: ``(batch, 3, height, width)`` tensor holding the horizontal, the vertical
                and the intensity channel.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.
        """
        horizontal = torch.clamp(img[:, 0, :, :], -1, 1)
        vertical = torch.clamp(img[:, 1, :, :], -1, 1)
        value = torch.clamp(img[:, 2, :, :], 0, 1)
        density = ((value * 0.5 * math.pi).sin() + EPS).pow(self.density_k)
        horizontal = torch.clamp(horizontal / (density + EPS), -1, 1)
        vertical = torch.clamp(vertical / (density + EPS), -1, 1)
        hue = (torch.atan2(vertical + EPS, horizontal + EPS) / (2 * math.pi)) % 1
        saturation = torch.sqrt(horizontal**2 + vertical**2 + EPS)
        saturation = torch.clamp(saturation, 0, 1)
        value = torch.clamp(value, 0, 1)
        sextant = torch.floor(hue * 6.0)
        offset = hue * 6.0 - sextant
        choices = (
            value,
            value * (1.0 - offset * saturation),
            value * (1.0 - saturation),
            value * (1.0 - (1.0 - offset) * saturation),
        )
        channels = [torch.zeros_like(hue) for _ in range(3)]
        for index, picks in enumerate(SEXTANTS):
            mask = sextant == index
            for channel, pick in enumerate(picks):
                channels[channel][mask] = choices[pick][mask]
        return torch.stack(channels, dim=1)


class Network(nn.Module):
    """The two-branch HVI network, answering one enhanced RGB frame per frame it is given."""

    def __init__(self):
        super().__init__()
        first = CHANNELS[0]
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, first, 3, stride=1, padding=0, bias=False),
        )
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(first, 2, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, first, 3, stride=1, padding=0, bias=False),
        )
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(first, 1, 3, stride=1, padding=0, bias=False),
        )
        for index in range(1, len(CHANNELS)):
            narrow, wide = CHANNELS[index - 1], CHANNELS[index]
            setattr(self, f"HVE_block{index}", NormDownsample(narrow, wide))
            setattr(self, f"IE_block{index}", NormDownsample(narrow, wide))
            setattr(self, f"HVD_block{index}", NormUpsample(wide, narrow))
            setattr(self, f"ID_block{index}", NormUpsample(wide, narrow))
        for index, scale in enumerate(STAGES, start=1):
            dim, heads = CHANNELS[scale], HEADS[scale]
            setattr(self, f"HV_LCA{index}", ChromaAttention(dim, heads))
            setattr(self, f"I_LCA{index}", IntensityAttention(dim, heads))
        self.trans = ColourTransform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Brighten a frame, working in HVI and answering in RGB.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first, with both
                sides a multiple of eight.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.
        """
        hvi = self.trans.to_hvi(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(x.dtype)

        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        return self.trans.to_rgb(torch.cat([hv_0, i_dec0], dim=1) + hvi)


def load(scenario: str = "generalization", device: str | None = None) -> Network:
    """Build the network and read one scenario's published weights into it.

    Args:
        scenario: A key of :data:`CHECKPOINTS`, naming the file to read.
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once per scenario and kept for the process.

    Raises:
        KeyError: ``scenario`` names none of :data:`CHECKPOINTS`.
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    filename = CHECKPOINTS[scenario]
    return managed_module(
        ("cidnet", REPO_ID, filename), lambda: _build(filename), name="HVI-CIDNet"
    )


def _build(filename: str) -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    from safetensors.torch import load_file

    path = published_checkpoint(
        FOLDER, REPO_ID, filename, subfolder=SUBFOLDER, feature=FEATURE,
        what="The low light network",
    )
    network = Network()
    network.load_state_dict(load_file(path), strict=True)
    return network.float().eval()
