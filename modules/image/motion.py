"""A clip's measured motion: flow between every pair of neighbouring frames, at a reduced size.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. Flows are ``(1, 2, h, w)`` in
pixels of the measured size, x first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import optical_flow

__all__ = [
    "CUT_SHARE",
    "HOLD_CHANGE",
    "MOTION_SIDE",
    "Motion",
    "measure",
    "visualise",
    "working_size",
]

#: Long side, in pixels, motion is measured at by default.
MOTION_SIDE = 768

#: Frame pairs measured together.
PAIR_BATCH = 8

#: Pixels of frame pairs a flow network measures together.
NETWORK_PIXELS = 32 * 768 * 432

#: What :attr:`Motion.engine` holds for motion measured without a network.
CLASSICAL = "texture flow"

#: Share of agreeing pixels below which a frame pair is treated as a cut.
CUT_SHARE = 0.4

#: Mean luminance change, in levels of 255, below which a frame repeats the one before it.
HOLD_CHANGE = 0.5

#: Distances, in pixels of the measured motion, a pixel looks for a motion that matches better.
SNAP_REACH = (4, 8, 16, 24)

#: Half the window a motion's match against the neighbouring frame is summed over, in pixels of
#: the measured motion.
SNAP_RADIUS = 3

#: Spread of motion, in pixels per frame of the measured motion, within :data:`SNAP_REACH` of a
#: pixel that marks it as near a motion edge.
SNAP_EDGE = 3.0

#: Share of its own match cost a neighbour's motion has to beat to be taken.
SNAP_BIAS = 0.85

#: How much worse, as a share plus a floor in levels of 255, one side's match may be than the
#: other's and still count as seen from that side.
SIDE_MATCH = (1.5, 2.0)

#: Path length, in pixels per frame, drawn at half brightness by :func:`visualise`.
VISUAL_REACH = 8.0


def working_size(height: int, width: int, side: int) -> tuple[int, int]:
    """The size motion is measured at: long side ``side``, never above the frame's own, even sides.

    Args:
        height: Frame height.
        width: Frame width.
        side: Long side wanted; 0 or anything above the frame's own keeps the frame's size.

    Returns:
        ``(height, width)``.
    """
    longest = max(height, width)
    if side <= 0 or side >= longest:
        return height, width
    scale = side / longest
    return max(16, 2 * int(round(height * scale / 2))), max(16, 2 * int(round(width * scale / 2)))


def _shifted(x, dx: int, dy: int):
    """``x`` read ``(dx, dy)`` pixels away, edges held."""
    height, width = x.shape[-2:]
    pad = max(abs(dx), abs(dy))
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return padded[..., pad + dy:pad + dy + height, pad + dx:pad + dx + width]


def _match_cost(here, flow, target):
    """Windowed mean difference between a frame and its neighbour pulled back along ``flow``.

    Returns:
        ``(1, 1, h, w)`` in levels of 255, averaged over a ``2 * SNAP_RADIUS + 1`` window.
    """
    mismatch = (optical_flow.warp(target, flow) - here).abs()
    window = 2 * SNAP_RADIUS + 1
    return F.avg_pool2d(F.pad(mismatch, (SNAP_RADIUS,) * 4, mode="replicate"), window, stride=1)


def _snap(here, flow, target):
    """Give pixels near a motion edge the motion, their own or a neighbour's, that best matches.

    Args:
        here: Luminance of the frame, ``(1, 1, h, w)``.
        flow: Flow from the frame onto ``target``, ``(1, 2, h, w)``.
        target: Luminance of the frame the flow maps onto.

    Returns:
        ``(flow, cost)``: the flow with its edges moved onto the edges the frames agree on, and
        its match cost from :func:`_match_cost`.
    """
    reach = max(SNAP_REACH)
    spread = (
        F.max_pool2d(flow, 2 * reach + 1, stride=1, padding=reach)
        + F.max_pool2d(-flow, 2 * reach + 1, stride=1, padding=reach)
    ).amax(1, keepdim=True)
    edge = spread > SNAP_EDGE
    own_cost = _match_cost(here, flow, target)
    if not bool(edge.any()):
        return flow, own_cost
    best = flow
    best_cost = own_cost * SNAP_BIAS
    for distance in SNAP_REACH:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            candidate = _shifted(flow, dx * distance, dy * distance)
            candidate_cost = _match_cost(here, candidate, target)
            take = (candidate_cost < best_cost) & edge
            best = torch.where(take, candidate, best)
            best_cost = torch.where(take, candidate_cost, best_cost)
    return best, torch.minimum(best_cost, own_cost)


@dataclass
class Motion:
    """Forward and backward flow for every neighbouring pair of one clip.

    Attributes:
        frame_size: ``(height, width)`` of the frames measured.
        size: ``(h, w)`` the flows are held at.
        forward: ``(frames - 1, 2, h, w)`` float16, frame ``i`` onto ``i + 1``.
        backward: ``(frames - 1, 2, h, w)`` float16, frame ``i + 1`` onto ``i``.
        luma: ``(frames, 1, h, w)`` uint8 luminance at the measured size.
        agreement: ``(frames - 1,)`` share of pixels whose forward and backward flow agree.
        mismatch: ``(frames - 1,)`` mean luminance difference, in levels of 255, left after
            following the forward flow.
        change: ``(frames - 1,)`` mean luminance difference, in levels of 255, between the two
            frames as they stand.
        engine: What measured the flow.
        snap: Whether flows have their edges snapped by :func:`_snap` when used; set for the
            texture flow only.
    """

    frame_size: tuple[int, int]
    size: tuple[int, int]
    forward: torch.Tensor
    backward: torch.Tensor
    luma: torch.Tensor
    agreement: torch.Tensor
    mismatch: torch.Tensor
    change: torch.Tensor
    engine: str = CLASSICAL
    snap: bool = True

    @property
    def count(self) -> int:
        """How many frames were measured."""
        return int(self.luma.shape[0])

    def describe(self) -> str:
        """One line naming the clip this motion was measured from."""
        height, width = self.frame_size
        h, w = self.size
        return f"{self.count} frame(s) of {width}x{height}, measured at {w}x{h} with {self.engine}"

    def check(self, count: int, height: int, width: int, name: str) -> None:
        """Raise unless this motion was measured from a clip of this length and size.

        Args:
            count: Frames in the clip it is about to be used with.
            height: Their height.
            width: Their width.
            name: The node using it, for the message.

        Raises:
            ValueError: The clip and the motion do not match.
        """
        if self.count != int(count) or tuple(self.frame_size) != (int(height), int(width)):
            raise ValueError(
                f"{name} was given motion measured from {self.describe()}, and a clip of "
                f"{int(count)} frame(s) of {int(width)}x{int(height)}. Measure the motion with "
                f"Video Motion from this same clip, or leave the motion input empty."
            )

    def cuts(self, share: float = CUT_SHARE) -> list[bool]:
        """Whether each neighbouring pair is a cut.

        Args:
            share: Agreement below which a pair counts as a cut.

        Returns:
            One flag per pair, ``frames - 1`` in all.
        """
        return [float(value) < float(share) for value in self.agreement.tolist()]

    def holds(self, change: float = HOLD_CHANGE) -> list[bool]:
        """Whether each frame after the first repeats the one before it.

        Args:
            change: Mean luminance change, in levels of 255, below which a frame repeats.

        Returns:
            One flag per pair, ``frames - 1`` in all.
        """
        return [float(value) < float(change) for value in self.change.tolist()]

    def _pair(self, index: int, device):
        """Flows of pair ``index`` as float32 on ``device``, ``(ahead, behind)``."""
        ahead = self.forward[index:index + 1].to(device=device, dtype=torch.float32)
        behind = self.backward[index:index + 1].to(device=device, dtype=torch.float32)
        return ahead, behind

    def _plane(self, index: int, device):
        """Luminance of frame ``index`` as float32 in ``[0, 255]`` on ``device``."""
        return self.luma[index:index + 1].to(device=device, dtype=torch.float32)

    def _edges(self, here, flow, target):
        """``(flow, cost)`` for a flow from ``here`` onto ``target``, snapped where :attr:`snap` is set."""
        if self.snap:
            return _snap(here, flow, target)
        return flow, _match_cost(here, flow, target)

    def toward(self, index: int, step: int, device):
        """Flow from frame ``index`` onto its neighbour ``index + step``, edges snapped per :attr:`snap`.

        Args:
            index: The frame.
            step: ``1`` for the next frame, ``-1`` for the previous one.
            device: Where the answer is placed.

        Returns:
            ``(flow, seen)``: the flow at the measured size, and a ``(1, 1, h, w)`` float that
            is 1 where the pixel is seen in the neighbour. ``None`` where there is no neighbour
            or the pair is a cut.
        """
        other = index + step
        if other < 0 or other >= self.count:
            return None
        pair = min(index, other)
        if float(self.agreement[pair]) < CUT_SHARE:
            return None
        ahead, behind = self._pair(pair, device)
        flow, back = (ahead, behind) if step > 0 else (behind, ahead)
        agree = optical_flow.consistent(flow, back)
        if self.snap:
            flow, _ = _snap(self._plane(index, device), flow, self._plane(other, device))
        return flow, agree.float()

    def paths(self, index: int, device):
        """Path terms and the hidden band for one frame, at the measured size.

        Args:
            index: The frame.
            device: Where the answer is placed.

        Returns:
            ``(a, c, hidden)``: a pixel sits at ``p + a t + c t^2`` for ``t`` in ``[-1, 1]``
            across one frame interval either side, and ``hidden`` is a ``(1, 1, h, w)`` float
            that is 1 where the pixel is not seen in a neighbour.
        """
        height, width = self.size
        ahead = self.forward[index:index + 1] if index < self.count - 1 else None
        behind = self.backward[index - 1:index] if index > 0 else None
        if ahead is not None and float(self.agreement[index]) < CUT_SHARE:
            ahead = None
        if behind is not None and float(self.agreement[index - 1]) < CUT_SHARE:
            behind = None
        if ahead is None and behind is None:
            zero = torch.zeros(1, 2, height, width, device=device)
            return zero, zero, torch.zeros(1, 1, height, width, device=device)
        here = self._plane(index, device)
        if ahead is not None:
            ahead_flow, ahead_back = self._pair(index, device)
            ahead_ok = optical_flow.consistent(ahead_flow, ahead_back)
            ahead_flow, ahead_cost = self._edges(here, ahead_flow, self._plane(index + 1, device))
        if behind is not None:
            behind_back, behind_flow = self._pair(index - 1, device)
            behind_ok = optical_flow.consistent(behind_flow, behind_back)
            behind_flow, behind_cost = self._edges(here, behind_flow, self._plane(index - 1, device))
        if behind is None:
            return ahead_flow, torch.zeros_like(ahead_flow), (~ahead_ok).float()
        if ahead is None:
            return -behind_flow, torch.zeros_like(behind_flow), (~behind_ok).float()
        share, floor = SIDE_MATCH
        ahead_ok = ahead_ok & (ahead_cost <= share * behind_cost + floor)
        behind_ok = behind_ok & (behind_cost <= share * ahead_cost + floor)
        both = ahead_ok & behind_ok
        central = 0.5 * (ahead_flow - behind_flow)
        a = torch.where(
            both, central,
            torch.where(ahead_ok, ahead_flow, torch.where(behind_ok, -behind_flow, central)),
        )
        c = torch.where(both, 0.5 * (ahead_flow + behind_flow), torch.zeros_like(ahead_flow))
        return a, c, (~both).float()


def measure(frames, side: int = MOTION_SIDE, device=None, progress=None, network=None) -> Motion:
    """Measure the flow between every pair of neighbouring frames.

    Args:
        frames: ``(frames, height, width, channels)`` in ``[0, 1]``.
        side: Long side motion is measured at; 0 measures at the frame's own size.
        device: Where the work runs. Defaults to the frames' own device.
        progress: Optional callable taking a step count, called once per pair measured.
        network: A :class:`~modules.model.flow_networks.FlowNetwork` to measure the flow with,
            or ``None`` for the texture flow.

    Returns:
        The :class:`Motion`, held on the CPU.
    """
    count, height, width = (int(v) for v in frames.shape[:3])
    device = frames.device if device is None else torch.device(device)
    size = working_size(height, width, int(side))
    h, w = size
    pairs = max(count - 1, 0)
    forward = torch.zeros(pairs, 2, h, w, dtype=torch.float16)
    backward = torch.zeros_like(forward)
    luma = torch.zeros(count, 1, h, w, dtype=torch.uint8)
    agreement = torch.ones(pairs)
    mismatch = torch.zeros(pairs)
    change = torch.zeros(pairs)
    if count == 1:
        plane = optical_flow.resize(optical_flow.luminance(frames[:1].to(device)), h, w)
        luma[0] = plane.round().clamp(0, 255).to(torch.uint8).cpu()[0]
    batch = PAIR_BATCH if network is None else max(1, NETWORK_PIXELS // (h * w))
    for start in range(0, pairs, batch):
        stop = min(start + batch, pairs)
        chunk = frames[start:stop + 1].to(device)
        planes = optical_flow.resize(optical_flow.luminance(chunk), h, w)
        first, second = planes[:-1], planes[1:]
        if network is None:
            del chunk
            ahead = optical_flow.estimate(first, second)
            behind = optical_flow.estimate(second, first)
        else:
            from ..model import flow_networks

            colour = chunk[..., :3].permute(0, 3, 1, 2).float().clamp(0.0, 1.0)
            del chunk
            colour = optical_flow.resize(colour, h, w) * 255.0
            ahead, behind = flow_networks.flows(network, colour)
            ahead, behind = ahead.to(device), behind.to(device)
            del colour
        agreement[start:stop] = optical_flow.consistent(ahead, behind).float().mean((1, 2, 3)).cpu()
        mismatch[start:stop] = (first - optical_flow.warp(second, ahead)).abs().mean((1, 2, 3)).cpu()
        change[start:stop] = (first - second).abs().mean((1, 2, 3)).cpu()
        forward[start:stop] = ahead.to(torch.float16).cpu()
        backward[start:stop] = behind.to(torch.float16).cpu()
        luma[start:stop + 1] = planes.round().clamp(0, 255).to(torch.uint8).cpu()
        if progress is not None:
            progress(stop - start)
    return Motion(
        frame_size=(height, width),
        size=size,
        forward=forward,
        backward=backward,
        luma=luma,
        agreement=agreement,
        mismatch=mismatch,
        change=change,
        engine=CLASSICAL if network is None else network.describe(),
        snap=network is None,
    )


def visualise(a, c=None):
    """A picture of one frame's motion: hue for direction, brightness for length.

    Args:
        a: Motion, ``(1, 2, height, width)`` in pixels.
        c: Optional curved term, shaped like ``a``, added to the length.

    Returns:
        ``(height, width, 3)`` in ``[0, 1]``.
    """
    reach = a.norm(dim=1)[0]
    if c is not None:
        reach = reach + c.norm(dim=1)[0]
    hue = (torch.atan2(a[0, 1], a[0, 0]) / (2.0 * math.pi)) % 1.0
    value = reach / (reach + VISUAL_REACH)
    sector = hue * 6.0
    channels = []
    for shift in (5.0, 3.0, 1.0):
        k = (shift + sector) % 6.0
        channels.append(value * (1.0 - torch.clamp(torch.minimum(k, 4.0 - k), 0.0, 1.0)))
    return torch.stack(channels, -1)
