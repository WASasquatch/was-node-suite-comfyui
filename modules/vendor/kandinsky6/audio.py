"""MMAudio 44.1 kHz mel autoencoder and BigVGAN v2 vocoder, as the Kandinsky 6 audio VAE.

Waveforms are ``(B, N)`` at 44.1 kHz, mel spectrograms ``(B, 128, F)``, latents ``(B, 40, F / 2)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Magnitude-preserving layers (MMAudio, EDM2)
# ---------------------------------------------------------------------------


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_silu(x):
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


class MPConv1D(nn.Module):
    """Convolution whose stored weight already carries its weight normalisation."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))

    def forward(self, x, gain=1):
        w = self.weight * gain
        return F.conv1d(x, w, padding=(w.shape[-1] // 2,))


class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_dim, out_dim=None, conv_shortcut=False, kernel_size=3, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim
        self.use_conv_shortcut = conv_shortcut
        self.use_norm = use_norm
        self.conv1 = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
        self.conv2 = MPConv1D(out_dim, out_dim, kernel_size=kernel_size)
        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                self.conv_shortcut = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
            else:
                self.nin_shortcut = MPConv1D(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.use_norm:
            x = normalize(x, dim=1)
        h = mp_silu(x)
        h = self.conv1(h)
        h = mp_silu(h)
        h = self.conv2(h)
        if self.in_dim != self.out_dim:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return mp_sum(x, h, t=0.3)


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.qkv = MPConv1D(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = MPConv1D(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        y = self.qkv(x)
        y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[-1])
        q, k, v = normalize(y, dim=2).unbind(3)
        q, k, v = (t.permute(0, 1, 3, 2) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(h.shape[0], -1, h.shape[2])
        h = self.proj_out(h)
        return mp_sum(x, h, t=0.3)


class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = MPConv1D(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv1 = MPConv1D(in_channels, in_channels, kernel_size=1)
            self.conv2 = MPConv1D(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv1(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        if self.with_conv:
            x = self.conv2(x)
        return x


class Encoder1D(nn.Module):
    def __init__(self, *, dim, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_layers=(), down_layers=(),
                 resamp_with_conv=True, in_dim, embed_dim, double_z=True, kernel_size=3, clip_act=256.0):
        super().__init__()
        self.dim = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = MPConv1D(in_dim, self.dim, kernel_size=kernel_size)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = dim * in_ch_mult[i_level]
            block_out = dim * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, kernel_size=kernel_size, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, kernel_size=kernel_size, use_norm=True)
        self.conv_out = MPConv1D(block_in, 2 * embed_dim if double_z else embed_dim, kernel_size=kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)
        h = mp_silu(h)
        return self.conv_out(h, gain=(self.learnable_gain + 1))


class Decoder1D(nn.Module):
    def __init__(self, *, dim, out_dim, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_layers=(), down_layers=(),
                 kernel_size=3, resamp_with_conv=True, in_dim, embed_dim, clip_act=256.0):
        super().__init__()
        self.ch = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = [i + 1 for i in down_layers]

        block_in = dim * ch_mult[self.num_layers - 1]
        self.conv_in = MPConv1D(embed_dim, block_in, kernel_size=kernel_size)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = dim * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)

        self.conv_out = MPConv1D(block_in, out_dim, kernel_size=kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)
        h = mp_silu(h)
        return self.conv_out(h, gain=(self.learnable_gain + 1))


class VAE(nn.Module):
    """The 44.1 kHz mel autoencoder: 128 mel bands to 40 latent channels at half the frame rate."""

    def __init__(self, *, data_dim=128, embed_dim=40, hidden_dim=512):
        super().__init__()
        self.register_buffer("data_mean", torch.zeros(1, data_dim, 1))
        self.register_buffer("data_std", torch.ones(1, data_dim, 1))
        self.encoder = Encoder1D(dim=hidden_dim, ch_mult=(1, 2, 4), num_res_blocks=2, attn_layers=[3],
                                 down_layers=[0], in_dim=data_dim, embed_dim=embed_dim)
        self.decoder = Decoder1D(dim=hidden_dim, ch_mult=(1, 2, 4), num_res_blocks=2, attn_layers=[3],
                                 down_layers=[0], in_dim=data_dim, out_dim=data_dim, embed_dim=embed_dim)
        self.embed_dim = embed_dim

    def encode(self, x):
        """Mean of the latent posterior for a mel spectrogram."""
        moments = self.encoder((x - self.data_mean) / self.data_std)
        return torch.chunk(moments, 2, dim=1)[0]

    def decode(self, z):
        return self.decoder(z) * self.data_std + self.data_mean


class MelConverter(nn.Module):
    """Log mel spectrogram at 44.1 kHz: 2048-point FFT, hop 512, 128 bands."""

    def __init__(self, *, n_fft=2048, num_mels=128, hop_size=512, win_size=2048):
        super().__init__()
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("mel_basis", torch.zeros(num_mels, 1 + n_fft // 2))
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def forward(self, waveform):
        waveform = waveform.clamp(min=-1.0, max=1.0)
        pad = int((self.n_fft - self.hop_size) / 2)
        waveform = F.pad(waveform.unsqueeze(1), [pad, pad], mode="reflect").squeeze(1)
        spec = torch.stft(waveform, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.hann_window, center=False, pad_mode="reflect", normalized=False,
                          onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9).float()
        spec = torch.matmul(self.mel_basis, spec)
        return torch.log(torch.clamp(spec, min=1e-5))


# ---------------------------------------------------------------------------
# BigVGAN v2 (NVIDIA)
# ---------------------------------------------------------------------------


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, padding=True, padding_mode="replicate", kernel_size=12):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        )

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        return self.lowpass(x)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        init = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        self.alpha = nn.Parameter(init * alpha)
        self.beta = nn.Parameter(init * alpha)
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)


class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), snake_logscale=True):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
            for _ in range(len(dilation))
        ])
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([
            Activation1d(activation=SnakeBeta(channels, alpha_logscale=snake_logscale))
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    """BigVGAN v2 generator, 44.1 kHz, 128 mel bands, 512x upsampling."""

    def __init__(
        self,
        num_mels=128,
        upsample_rates=(8, 4, 2, 2, 2, 2),
        upsample_kernel_sizes=(16, 8, 4, 4, 4, 4),
        upsample_initial_channel=1536,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        snake_logscale=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                   k, u, padding=(k - u) // 2)
            ]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d, snake_logscale=snake_logscale))

        self.activation_post = Activation1d(activation=SnakeBeta(ch, alpha_logscale=snake_logscale))
        self.use_bias_at_final = use_bias_at_final
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=use_bias_at_final)
        self.use_tanh_at_final = use_tanh_at_final

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            return torch.tanh(x)
        return torch.clamp(x, min=-1.0, max=1.0)


# ---------------------------------------------------------------------------
# The combined audio VAE
# ---------------------------------------------------------------------------


class AudioAutoencoder(nn.Module):
    """Mel converter, mel autoencoder and vocoder under the key prefixes the released file uses."""

    sample_rate = 44100
    downsample_factor = 1024
    latent_channels = 40

    def __init__(self):
        super().__init__()
        self.mel_converter = MelConverter()
        self.vae = VAE()
        self.vocoder = BigVGAN()

    def decode(self, z):
        """Latents ``(B, 40, L)`` to a waveform ``(B, 1, 1024 * L)`` in ``[-1, 1]``."""
        return self.vocoder(self.vae.decode(z))

    def encode(self, waveform):
        """A mono waveform ``(B, N)`` to latents ``(B, 40, N / 1024)``."""
        return self.vae.encode(self.mel_converter(waveform))
