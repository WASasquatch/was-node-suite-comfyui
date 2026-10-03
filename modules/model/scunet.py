"""Blind image denoising, on the SCUNet swin-conv UNet.

:func:`load` answers a network taking ``(batch, 3, height, width)`` on a 0 to 1 scale and
answering a frame of the same size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from . import managed_module, published_checkpoint

__all__ = ["FEATURE", "FILENAME", "FOLDER", "Network", "REPO_ID", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "denoise"

#: Repository publishing the weights, the directory inside it, and the file.
REPO_ID = "WAS/was-node-suite-weights"
SUBFOLDER = "denoise"
FILENAME = "scunet_color_real_psnr.pth"

#: Channels the outermost stage works at.
DIM = 64

#: Channels one attention head reads.
HEAD_DIM = 32

#: Side of the square window attention is computed inside.
WINDOW_SIZE = 8

#: Blocks each of the seven stages stacks, outermost down and back up.
CONFIG = (4, 4, 4, 4, 4, 4, 4)

#: Multiple both frame sides are padded up to before the stages run.
SIZE_MULTIPLE = 64


class WindowAttention(nn.Module):
    """Self-attention inside a square window, shifted by half a window where asked."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.shifted = shifted
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1)
        )
        self.linear = nn.Linear(input_dim, output_dim)
        rows, columns = torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        )
        cells = torch.stack((rows, columns), dim=-1).reshape(-1, 2)
        # Row and column offset of every ordered pair of cells, biased to index from zero.
        self.register_buffer(
            "relation",
            cells[:, None, :] - cells[None, :, :] + window_size - 1,
            persistent=False,
        )

    def relative_embedding(self) -> torch.Tensor:
        """Read the learned bias of every ordered pair of cells in a window.

        Returns:
            A ``(heads, window_size ** 2, window_size ** 2)`` tensor.
        """
        return self.relative_position_params[
            :, self.relation[:, :, 0], self.relation[:, :, 1]
        ]

    def generate_mask(
        self, windows_h: int, windows_w: int, window: int, shift: int
    ) -> torch.Tensor:
        """Mark the pairs of cells a shifted window brought together from opposite edges.

        Args:
            windows_h: Windows the frame divides into down its height.
            windows_w: Windows the frame divides into across its width.
            window: Side of one square window.
            shift: Cells the frame was rolled by.

        Returns:
            A ``(1, 1, windows, window ** 2, window ** 2)`` boolean tensor, True on each
            pair to be hidden.
        """
        mask = torch.zeros(
            windows_h,
            windows_w,
            window,
            window,
            window,
            window,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        edge = window - shift
        mask[-1, :, :edge, :, edge:, :] = True
        mask[-1, :, edge:, :, :edge, :] = True
        mask[:, -1, :, :edge, :, edge:] = True
        mask[:, -1, :, edge:, :, :edge] = True
        cells = window * window
        return mask.reshape(1, 1, windows_h * windows_w, cells, cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend within each window.

        Args:
            x: ``(batch, height, width, input_dim)`` tensor, both sides a multiple of the
                window size.

        Returns:
            A ``(batch, height, width, output_dim)`` tensor.
        """
        window = self.window_size
        shift = window // 2
        if self.shifted:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        batch, height, width, _ = x.shape
        windows_h, windows_w = height // window, width // window
        windows = windows_h * windows_w
        cells = window * window
        x = x.reshape(batch, windows_h, window, windows_w, window, self.input_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, windows, cells, self.input_dim)
        qkv = self.embedding_layer(x)
        qkv = qkv.reshape(batch, windows, cells, 3 * self.n_heads, self.head_dim)
        query, key, value = qkv.permute(3, 0, 1, 2, 4).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", query, key) * self.scale
        sim = sim + self.relative_embedding()[:, None, None, :, :]
        if self.shifted:
            mask = self.generate_mask(windows_h, windows_w, window, shift)
            sim = sim.masked_fill(mask, float("-inf"))
        probs = functional.softmax(sim, dim=-1)
        out = torch.einsum("hbwij,hbwjc->hbwic", probs, value)
        out = out.permute(1, 2, 3, 0, 4).reshape(batch, windows, cells, self.input_dim)
        out = self.linear(out)
        out = out.reshape(batch, windows_h, windows_w, window, window, self.output_dim)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, self.output_dim)
        if self.shifted:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))
        return out


class SwinBlock(nn.Module):
    """One transformer block: residual window attention, then a residual two-layer MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WindowAttention(input_dim, input_dim, head_dim, window_size, shifted)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention and the MLP.

        Args:
            x: ``(batch, height, width, input_dim)`` tensor.

        Returns:
            A ``(batch, height, width, output_dim)`` tensor.
        """
        x = x + self.msa(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ConvTransBlock(nn.Module):
    """One block splitting its channels between a residual conv branch and a swin branch."""

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
    ):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.trans_block = SwinBlock(trans_dim, trans_dim, head_dim, window_size, shifted)
        self.conv1_1 = nn.Conv2d(
            conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True
        )
        self.conv1_2 = nn.Conv2d(
            conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches over their own share of the channels and add the result back.

        Args:
            x: ``(batch, conv_dim + trans_dim, height, width)`` tensor.

        Returns:
            A tensor of the same shape.
        """
        conv_x, trans_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1
        )
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


def _blocks(channels: int, count: int) -> list[nn.Module]:
    """One stage's blocks.

    Args:
        channels: Channels each branch of one block works at.
        count: Blocks the stage stacks.

    Returns:
        The stage's blocks, in order.
    """
    return [
        ConvTransBlock(channels, channels, HEAD_DIM, WINDOW_SIZE, bool(index % 2))
        for index in range(count)
    ]


class Network(nn.Module):
    """The seven-stage swin-conv UNet, answering a frame the size it was handed."""

    def __init__(self, in_nc: int = 3):
        super().__init__()
        self.config = CONFIG
        self.dim = DIM
        self.head_dim = HEAD_DIM
        self.window_size = WINDOW_SIZE
        dim = DIM
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *_blocks(dim // 2, CONFIG[0]),
            nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False),
        )
        self.m_down2 = nn.Sequential(
            *_blocks(dim, CONFIG[1]),
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False),
        )
        self.m_down3 = nn.Sequential(
            *_blocks(2 * dim, CONFIG[2]),
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False),
        )
        self.m_body = nn.Sequential(*_blocks(4 * dim, CONFIG[3]))
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *_blocks(2 * dim, CONFIG[4]),
        )
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *_blocks(dim, CONFIG[5]),
        )
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
            *_blocks(dim // 2, CONFIG[6]),
        )
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a frame.

        Args:
            x: ``(batch, 3, height, width)`` tensor on a 0 to 1 scale, red first.

        Returns:
            A ``(batch, 3, height, width)`` tensor on a 0 to 1 scale.
        """
        height, width = x.shape[-2:]
        padding = (0, -width % SIZE_MULTIPLE, 0, -height % SIZE_MULTIPLE)
        x = functional.pad(x, padding, mode="replicate")
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        h = self.m_body(x4)
        h = self.m_up3(h + x4)
        h = self.m_up2(h + x3)
        h = self.m_up1(h + x2)
        h = self.m_tail(h + x1)
        return h[..., :height, :width]


def load(device: str | None = None) -> Network:
    """Build the network and read the published weights into it.

    Args:
        device: Ignored. The caller moves the network to where it runs.

    Returns:
        The network in eval mode, built once and kept for the process.

    Raises:
        ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
    """
    return managed_module(("scunet", REPO_ID, FILENAME), _build, name="SCUNet")


def _build() -> Network:
    """Read the checkpoint and load it into a freshly built network."""
    path = published_checkpoint(
        FOLDER,
        REPO_ID,
        FILENAME,
        subfolder=SUBFOLDER,
        feature=FEATURE,
        what="The denoise network",
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    network = Network()
    network.load_state_dict(state, strict=True)
    return network.float().eval()
