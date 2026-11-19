"""Frames lined up on a neighbour along measured flow, and their noise level.

Frames are picture codes. Flows are ``(1, 2, height, width)`` in pixels, x first. Noise levels
are on the 0 to 1 scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import motion as motion_field
from . import optical_flow

__all__ = [
    "BINS",
    "FLOW_SIDE",
    "Flows",
    "align",
    "from_opponent",
    "noise_map",
    "noise_table",
    "opponent",
    "spatial_noise",
]

#: Long side, in pixels, flow is measured at by default.
FLOW_SIDE = motion_field.MOTION_SIDE

#: Gaussian sigma, in pixels of the measured size, applied to frames before flow is measured.
PRESMOOTH = 1.0

#: Frame pairs the texture flow measures together.
PAIR_BATCH = 8

#: Share of pixels whose two flows agree below which a neighbouring pair is a cut.
CUT_SHARE = motion_field.CUT_SHARE

#: Coefficient of the cubic convolution kernel ``grid_sample`` interpolates with.
CUBIC = -0.75

#: Rows of the orthonormal opponent transform: brightness, red against blue, green against both.
OPPONENT = (
    (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    (1.0 / math.sqrt(2.0), 0.0, -1.0 / math.sqrt(2.0)),
    (1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0)),
)

#: Brightness bands a noise table holds a level for.
BINS = 8

#: Residual samples a noise table is measured from at most.
NOISE_SAMPLES = 1 << 20

#: Samples a band needs before its own level is measured.
BAND_SAMPLES = 2000

#: Smallest noise level a table holds.
NOISE_FLOOR = 0.2 / 255.0

#: Gaussian sigma, in pixels, of the brightness a noise table is read at.
LEVEL_SMOOTH = 2.0

#: Share of a frame's least textured pixels the spatial noise estimate reads.
FLAT_SHARE = 0.5

#: Median absolute deviation of a normal distribution, as a share of its deviation.
MAD_SCALE = 1.4826


def _memory(device):
    """``(free, total)`` bytes on ``device`` as ComfyUI counts them, or ``None`` on the CPU."""
    from ..model import flow_networks

    return flow_networks._memory(torch.device(device))


class Flows:
    """Flow between any two frames of one clip, measured when first asked for.

    Attributes:
        count: Frames in the clip.
        frame_size: ``(height, width)`` of the frames.
        size: ``(h, w)`` the flows are measured and held at.
        device: Where flows are answered.
        network: The :class:`~modules.model.flow_networks.FlowNetwork`, or ``None`` for the
            texture flow.
    """

    def __init__(self, frames, side: int, device, network=None):
        """Prepare to measure flow between frames of ``frames``.

        Args:
            frames: ``(frames, height, width, channels)`` picture codes, float or uint8.
            side: Long side flow is measured at; 0 measures at the frames' own size.
            device: Where the texture flow runs and flows are answered.
            network: A flow network to measure with, or ``None`` for the texture flow.
        """
        count, height, width = (int(v) for v in frames.shape[:3])
        self.frames = frames
        self.count = count
        self.frame_size = (height, width)
        self.size = motion_field.working_size(height, width, int(side))
        self.device = torch.device(device)
        self.network = network
        self._pictures: dict = {}
        self._encoded: dict = {}
        self._flows: dict = {}
        self._agreement: dict = {}
        if network is not None:
            self._prepare_network()

    @property
    def engine(self) -> str:
        """What measures the flow."""
        return motion_field.CLASSICAL if self.network is None else self.network.describe()

    def _prepare_network(self) -> None:
        """Size the network's working memory and make its weights resident."""
        from ..model import flow_networks
        from ..vendor.sea_raft.corr import LOCAL_CHUNK_BYTES

        height, width = self.size
        scale = max(1.0, flow_networks.SHORTEST_SIDE / max(1, min(height, width)))
        self._net_size = (round(height * scale), round(width * scale))
        net_height, net_width = self._net_size
        net = self.network.backend.model
        frame_bytes = net_height * net_width * flow_networks.FRAME_BYTES.get(
            type(net).__name__, max(flow_networks.FRAME_BYTES.values())
        )
        pair_bytes = net_height * net_width * flow_networks.PAIR_BYTES
        before = _memory(self.network.backend.load_device)
        local = 0
        while before is not None and local < flow_networks.LEVELS and (
            pair_bytes + flow_networks._volume(net_height, net_width, local)
            + (2 * LOCAL_CHUNK_BYTES if local else 0) > flow_networks.SPARE * before[1]
        ):
            local += 1
        self._local = local
        self._frame_bytes = frame_bytes
        self._per_pair = pair_bytes + flow_networks._volume(net_height, net_width, local)
        self._overhead = 2 * LOCAL_CHUNK_BYTES if local else 0
        wanted = max(
            flow_networks.MOST_FRAMES * frame_bytes,
            flow_networks.WORKING_PAIRS * self._per_pair + self._overhead,
        )
        if before is not None:
            wanted = min(wanted, flow_networks.SPARE * before[1])
        self._net_device = self.network.backend.load(memory_required=int(wanted))

    def _picture(self, index: int):
        """Frame ``index`` as the flow measurement reads it: resized, smoothed, on its device."""
        found = self._pictures.get(index)
        if found is not None:
            return found
        height, width = self.size
        frame = optical_flow.unit(self.frames[index:index + 1, ..., :3], self.device)
        frame = frame.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        if frame.shape[1] < 3:
            frame = frame[:, :1].expand(-1, 3, -1, -1)
        frame = optical_flow.gaussian(optical_flow.resize(frame, height, width), PRESMOOTH)
        if self.network is None:
            weights = torch.tensor((0.299, 0.587, 0.114), device=frame.device).view(1, 3, 1, 1)
            found = (frame * weights).sum(1, keepdim=True) * 255.0
        else:
            net_height, net_width = self._net_size
            if (net_height, net_width) != (height, width):
                frame = F.interpolate(frame, size=self._net_size, mode="bilinear", align_corners=False)
            found = (frame * 255.0).to(self._net_device)
        self._pictures[index] = found
        return found

    def measure(self, pairs) -> None:
        """Measure every ordered pair not measured yet.

        Args:
            pairs: ``(a, b)`` frame numbers; the flow measured maps frame ``a`` onto ``b``.
        """
        wanted = [pair for pair in dict.fromkeys(pairs) if pair not in self._flows and pair[0] != pair[1]]
        if not wanted:
            return
        if self.network is None:
            self._measure_texture(wanted)
        else:
            self._measure_network(wanted)

    def _measure_texture(self, wanted) -> None:
        """Measure pairs with the texture flow."""
        for start in range(0, len(wanted), PAIR_BATCH):
            chunk = wanted[start:start + PAIR_BATCH]
            first = torch.cat([self._picture(a) for a, _ in chunk])
            second = torch.cat([self._picture(b) for _, b in chunk])
            found = optical_flow.estimate(first, second)
            for slot, pair in enumerate(chunk):
                self._flows[pair] = found[slot:slot + 1]

    def _measure_network(self, wanted) -> None:
        """Measure pairs with the flow network."""
        from ..model import flow_networks

        net = self.network.backend.model
        indices = sorted({index for pair in wanted for index in pair} - set(self._encoded))
        height, width = self.size
        with torch.inference_mode():
            budget = _memory(self._net_device)
            at_once = flow_networks.MOST_FRAMES
            if budget is not None:
                at_once = max(1, min(at_once, int(budget[0] * flow_networks.SPARE) // self._frame_bytes))
            for start in range(0, len(indices), at_once):
                chunk = indices[start:start + at_once]
                encoded = net.encode(torch.cat([self._picture(index) for index in chunk]))
                for slot, index in enumerate(chunk):
                    self._encoded[index] = flow_networks.select(encoded, [slot])
                del encoded
            budget = _memory(self._net_device)
            at_once = flow_networks.MOST_PAIRS
            if budget is not None:
                spare = int(budget[0] * flow_networks.SPARE) - self._overhead
                at_once = max(1, min(at_once, spare // self._per_pair))
            for start in range(0, len(wanted), at_once):
                chunk = wanted[start:start + at_once]
                first = _joined([self._encoded[a] for a, _ in chunk])
                second = _joined([self._encoded[b] for _, b in chunk])
                found = net.estimate(first, second, self.network.iterations, local=self._local)
                found = optical_flow.resize_flow(found.float(), height, width).to(self.device)
                for slot, pair in enumerate(chunk):
                    self._flows[pair] = found[slot:slot + 1].clone()

    def flow(self, a: int, b: int):
        """The measured flow from frame ``a`` onto frame ``b``, ``(1, 2, h, w)`` on :attr:`device`."""
        self.measure([(a, b)])
        return self._flows[(a, b)]

    def agreement(self, a: int, b: int) -> float:
        """Share of frame ``a``'s pixels whose flows to and from frame ``b`` agree."""
        key = (min(a, b), max(a, b))
        found = self._agreement.get(key)
        if found is None:
            self.measure([key, key[::-1]])
            found = float(optical_flow.consistent(self._flows[key], self._flows[key[::-1]]).float().mean())
            self._agreement[key] = found
        return found

    def reachable(self, index: int, radius: int) -> list[int]:
        """Offsets of the frames within ``radius`` of ``index`` that no cut separates from it.

        Args:
            index: The frame.
            radius: Frames looked at on each side.

        Returns:
            Offsets, nearest first on each side, negative for earlier frames.
        """
        found = []
        for direction in (-1, 1):
            for step in range(1, int(radius) + 1):
                other = index + direction * step
                if other < 0 or other >= self.count:
                    break
                if self.agreement(other - direction, other) < CUT_SHARE:
                    break
                found.append(direction * step)
        return sorted(found, key=lambda offset: (abs(offset), offset))

    def forget(self, before: int) -> None:
        """Release everything held for frames numbered below ``before`` alone.

        Args:
            before: The first frame still wanted.
        """
        for index in [index for index in self._pictures if index < before]:
            del self._pictures[index]
        for index in [index for index in self._encoded if index < before]:
            del self._encoded[index]
        for pair in [pair for pair in self._flows if max(pair) < before]:
            del self._flows[pair]


def _joined(encodings: list) -> dict:
    """Single-frame encodings stacked into one batch."""
    first = encodings[0]
    return {
        key: torch.cat([one[key] for one in encodings]) if hasattr(value, "shape") else value
        for key, value in first.items()
    }


def _spread(fraction):
    """Share of a sample's noise variance left after cubic interpolation at ``fraction`` of a pixel."""
    a = CUBIC
    far = fraction + 1.0
    near = 1.0 - fraction
    beyond = 2.0 - fraction
    weights = (
        ((a * far - 5.0 * a) * far + 8.0 * a) * far - 4.0 * a,
        ((a + 2.0) * fraction - (a + 3.0)) * fraction * fraction + 1.0,
        ((a + 2.0) * near - (a + 3.0)) * near * near + 1.0,
        ((a * beyond - 5.0 * a) * beyond + 8.0 * a) * beyond - 4.0 * a,
    )
    return sum(weight * weight for weight in weights)


def align(picture, flow, back, frame_size):
    """A neighbour pulled onto the frame along ``flow``, with where the pull can be trusted.

    Args:
        picture: The neighbour, ``(1, channels, height, width)`` on the device.
        flow: Flow from the frame onto the neighbour, at the measured size.
        back: Flow from the neighbour onto the frame, at the measured size.
        frame_size: ``(height, width)`` of the frames.

    Returns:
        ``(aligned, seen, spread)``: the neighbour resampled onto the frame with a cubic filter;
        a ``(1, 1, height, width)`` weight, 1 where both flows agree and the pull stays inside
        the frame; and a ``(1, 1, height, width)`` share of the neighbour's noise variance the
        resampling keeps.
    """
    height, width = frame_size
    agree = optical_flow.consistent(flow, back).float()
    agree = -F.max_pool2d(-agree, 3, stride=1, padding=1)
    full = optical_flow.resize_flow(flow, height, width)
    xs = torch.arange(width, device=full.device, dtype=full.dtype).view(1, 1, width)
    ys = torch.arange(height, device=full.device, dtype=full.dtype).view(1, height, 1)
    px = xs + full[:, 0]
    py = ys + full[:, 1]
    grid = torch.stack(
        [px * (2.0 / max(width - 1, 1)) - 1.0, py * (2.0 / max(height - 1, 1)) - 1.0], -1
    )
    aligned = F.grid_sample(picture, grid, mode="bicubic", padding_mode="border", align_corners=True)
    inside = ((px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)).unsqueeze(1)
    seen = F.interpolate(agree, size=(height, width), mode="bilinear", align_corners=False)
    seen = seen.clamp(0.0, 1.0) * inside
    spread = (_spread(px - px.floor()) * _spread(py - py.floor())).unsqueeze(1)
    return aligned, seen, spread


def opponent(planes):
    """RGB planes in the orthonormal opponent space of :data:`OPPONENT`, ``(n, 3, h, w)``."""
    matrix = torch.tensor(OPPONENT, device=planes.device, dtype=planes.dtype)
    return torch.einsum("oc,nchw->nohw", matrix, planes)


def from_opponent(planes):
    """Opponent planes back to RGB, ``(n, 3, h, w)``."""
    matrix = torch.tensor(OPPONENT, device=planes.device, dtype=planes.dtype)
    return torch.einsum("oc,nohw->nchw", matrix, planes)


def _brightness(centre):
    """Smoothed brightness of opponent planes, ``(height, width)`` in 0 to 1."""
    plane = optical_flow.gaussian(centre[:, :1] / math.sqrt(3.0), LEVEL_SMOOTH)
    return plane[0, 0].clamp(0.0, 1.0)


def _filled(table):
    """A ``(3, BINS)`` table with each unmeasured band given its nearest measured band's level."""
    measured = [band for band in range(BINS) if not bool(torch.isnan(table[0, band]))]
    if not measured:
        return None
    for band in range(BINS):
        if band not in measured:
            nearest = min(measured, key=lambda other: (abs(other - band), other))
            table[:, band] = table[:, nearest]
    return table.clamp(min=NOISE_FLOOR)


def noise_table(centre, residuals):
    """Noise level of each opponent channel in each brightness band, read off aligned residuals.

    Args:
        centre: The frame's opponent planes, ``(1, 3, height, width)``.
        residuals: ``(difference, seen, spread)`` triples: an aligned neighbour's opponent
            planes minus ``centre``, and the weights :func:`align` answered for it.

    Returns:
        A ``(3, BINS)`` tensor, or ``None`` where too few pixels lined up to measure.
    """
    bands = (_brightness(centre) * BINS).long().clamp(max=BINS - 1)
    values, labels = [], []
    for difference, seen, spread in residuals:
        keep = seen[0, 0] > 0.99
        values.append((difference[0] / torch.sqrt(1.0 + spread[0]))[:, keep])
        labels.append(bands[keep])
    if not values:
        return None
    values = torch.cat(values, 1)
    labels = torch.cat(labels)
    if int(labels.numel()) > NOISE_SAMPLES:
        generator = torch.Generator(device="cpu").manual_seed(0)
        pick = torch.randperm(int(labels.numel()), generator=generator)[:NOISE_SAMPLES].to(labels.device)
        values, labels = values[:, pick], labels[pick]
    table = torch.full((3, BINS), float("nan"), device=centre.device)
    for band in range(BINS):
        chosen = values[:, labels == band]
        if int(chosen.shape[1]) < BAND_SAMPLES:
            continue
        centred = chosen - chosen.median(dim=1, keepdim=True).values
        table[:, band] = MAD_SCALE * centred.abs().median(dim=1).values
    return _filled(table)


def spatial_noise(centre):
    """Noise level of each opponent channel read from one frame alone, the same in every band.

    Args:
        centre: The frame's opponent planes, ``(1, 3, height, width)``.

    Returns:
        A ``(3, BINS)`` tensor.
    """
    kernel = torch.tensor(
        ((1.0, -2.0, 1.0), (-2.0, 4.0, -2.0), (1.0, -2.0, 1.0)), device=centre.device
    ).view(1, 1, 3, 3).expand(3, 1, 3, 3)
    response = F.conv2d(centre, kernel, groups=3).abs()[0]
    gx, gy = optical_flow._gradients(centre[:, :1])
    texture = (gx * gx + gy * gy)[0, 0, 1:-1, 1:-1]
    flat = texture <= torch.quantile(texture.flatten()[:: max(1, texture.numel() // 1_000_000)], FLAT_SHARE)
    levels = math.sqrt(math.pi / 2.0) / 6.0 * response[:, flat].mean(1)
    return levels.view(3, 1).expand(3, BINS).clone().clamp(min=NOISE_FLOOR)


def noise_map(table, centre):
    """Noise level of each opponent channel at every pixel, read from a table at its brightness.

    Args:
        table: A ``(3, BINS)`` tensor from :func:`noise_table` or :func:`spatial_noise`.
        centre: The frame's opponent planes, ``(1, 3, height, width)``.

    Returns:
        ``(1, 3, height, width)``.
    """
    position = (_brightness(centre) * BINS - 0.5).clamp(0.0, BINS - 1.0)
    low = position.floor().long()
    high = (low + 1).clamp(max=BINS - 1)
    share = position - low
    table = table.to(centre.device)
    return (table[:, low] * (1.0 - share) + table[:, high] * share).unsqueeze(0)
