"""A fixed-size window that follows a masked subject through a clip, and the way back.

Frames are ``(frames, height, width, channels)`` in 0 to 1. A window is its centre and
height in frame pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "Track",
    "crop_frames",
    "crop_one",
    "detail",
    "native",
    "register",
    "scaled",
    "shifted",
    "paste_frames",
    "plan",
    "soft_alpha",
]

#: Smallest share of a frame a mask must cover to place the window on it.
MIN_COVER = 0.00002

#: Smallest window height, in frame pixels.
MIN_SIDE = 8.0

#: Largest window height, in frame sides; past the frame the edge pixels repeat.
MAX_FRAMES = 3.0

#: Side of the plane a mask is measured on.
MEASURE_SIDE = 512

#: Shrink factor past which a picture is first reduced by area before it is resampled.
REDUCE_FROM = 1.5

#: Blur used to register two crops, as a divisor of the crop's short side.
ALIGN_BLUR = 96.0

#: Least phase correlation peak that counts as a match.
ALIGN_PEAK = 0.02


@dataclass
class Track:
    """Where the window sits on every frame of a clip.

    Attributes:
        frame: ``(height, width)`` of the clip.
        size: ``(height, width)`` of the crop.
        centres: ``(x, y)`` window centre per frame, in frame pixels.
        sides: Window height per frame, in frame pixels.
        found: Whether the mask held the subject on each frame.
        reference: Index of the frame where the subject is largest.
    """

    frame: tuple
    size: tuple
    centres: list
    sides: list
    found: list
    reference: int = 0

    @property
    def count(self) -> int:
        """Frames the track covers."""
        return len(self.sides)

    def scale(self, index: int) -> float:
        """Frame pixels per crop pixel on one frame."""
        return float(self.sides[index]) / float(self.size[0])

    def box(self, index: int) -> tuple:
        """``(left, top, right, bottom)`` of the window on one frame, in frame pixels."""
        x, y = self.centres[index]
        height = float(self.sides[index])
        width = height * self.size[1] / self.size[0]
        return x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0


def _interpolated(values: list, known: list) -> list:
    """Values at unknown positions drawn linearly between known ones, ends held."""
    marks = [index for index, ok in enumerate(known) if ok]
    out = list(values)
    for index in range(len(values)):
        if known[index]:
            continue
        before = [mark for mark in marks if mark < index]
        after = [mark for mark in marks if mark > index]
        if before and after:
            a, b = before[-1], after[0]
            share = (index - a) / (b - a)
            out[index] = values[a] * (1.0 - share) + values[b] * share
        else:
            out[index] = values[before[-1] if before else after[0]]
    return out


def plan(masks: torch.Tensor, count: int, frame: tuple, size: tuple, margin: float, smoothing: float) -> Track:
    """Place the window on every frame from the subject's mask.

    Args:
        masks: ``(frames, height, width)`` subject mask, or one frame for all.
        count: Frames in the clip.
        frame: ``(height, width)`` of the clip.
        size: ``(height, width)`` of the crop.
        margin: Window side against the side of a square holding the subject's area.
        smoothing: Gaussian sigma, in frames, of the window's path and size.

    Returns:
        The :class:`Track`.

    Raises:
        ValueError: No frame's mask holds the subject.
    """
    from ..image import camera_motion

    height, width = frame
    aspect = size[1] / size[0]
    factor = min(1.0, MEASURE_SIDE / max(height, width))
    small = (max(8, round(height * factor)), max(8, round(width * factor)))
    ys = (torch.arange(small[0], dtype=torch.float32).view(-1, 1) + 0.5) * (height / small[0])
    xs = (torch.arange(small[1], dtype=torch.float32).view(1, -1) + 0.5) * (width / small[1])
    cell = (height / small[0]) * (width / small[1])
    centres_x, centres_y, sides, found = [], [], [], []
    for index in range(count):
        plane = masks[min(index, int(masks.shape[0]) - 1)].float().cpu()
        if tuple(plane.shape[-2:]) != (height, width):
            plane = F.interpolate(plane.view(1, 1, *plane.shape[-2:]), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        plane = F.interpolate(plane.view(1, 1, height, width), size=small, mode="area")[0, 0].clamp(0.0, 1.0)
        total = float(plane.sum())
        if total < MIN_COVER * small[0] * small[1]:
            centres_x.append(0.0)
            centres_y.append(0.0)
            sides.append(0.0)
            found.append(False)
            continue
        centres_x.append(float((plane * xs).sum()) / total)
        centres_y.append(float((plane * ys).sum()) / total)
        sides.append(math.log(max(MIN_SIDE, margin * math.sqrt(total * cell / aspect))))
        found.append(True)
    if not any(found):
        raise ValueError("no frame's mask holds the subject")
    centres_x = camera_motion.smooth(_interpolated(centres_x, found), smoothing)
    centres_y = camera_motion.smooth(_interpolated(centres_y, found), smoothing)
    logs = camera_motion.smooth(_interpolated(sides, found), smoothing)
    tallest = MAX_FRAMES * max(float(height), float(width) / aspect)
    heights = [min(tallest, max(MIN_SIDE, math.exp(value))) for value in logs]
    measured = [math.exp(value) if ok else 0.0 for value, ok in zip(sides, found)]
    reference = max(range(count), key=lambda index: measured[index])
    return Track(
        frame=(int(height), int(width)),
        size=(int(size[0]), int(size[1])),
        centres=list(zip(centres_x, centres_y)),
        sides=heights,
        found=found,
        reference=reference,
    )


def _reduced(picture: torch.Tensor, factor: float) -> torch.Tensor:
    """A ``(1, channels, height, width)`` picture reduced by area when ``factor`` is large."""
    if factor < REDUCE_FROM:
        return picture
    step = int(factor)
    height, width = int(picture.shape[-2]), int(picture.shape[-1])
    return F.interpolate(picture, size=(max(1, height // step), max(1, width // step)), mode="area")


def _grid(xs: torch.Tensor, ys: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """A ``grid_sample`` grid from pixel-centre coordinates in a picture of the given size."""
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([2.0 * xx / width - 1.0, 2.0 * yy / height - 1.0], -1).unsqueeze(0)


def crop_one(picture: torch.Tensor, track: Track, index: int, device) -> tuple:
    """Cut the window of one frame out of a picture, resampled to the crop size.

    Args:
        picture: ``(height, width, channels)``, the frame or a plane laid over it.
        track: The window's track.
        index: The frame the window is taken from.
        device: Where to work.

    Returns:
        ``(crop, inside)``: ``(crop height, crop width, channels)`` and
        ``(crop height, crop width)``, 1 where the window lies inside the frame.
    """
    crop_h, crop_w = track.size
    height, width = track.frame
    planes = picture.to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if tuple(planes.shape[-2:]) != (height, width):
        planes = F.interpolate(planes, size=(height, width), mode="bilinear", align_corners=False)
    scale = track.scale(index)
    x, y = track.centres[index]
    xs = x + (torch.arange(crop_w, device=device, dtype=torch.float32) + 0.5 - crop_w / 2.0) * scale
    ys = y + (torch.arange(crop_h, device=device, dtype=torch.float32) + 0.5 - crop_h / 2.0) * scale
    mode = "bicubic" if scale < REDUCE_FROM else "bilinear"
    cut = F.grid_sample(_reduced(planes, scale), _grid(xs, ys, width, height), mode=mode,
                        padding_mode="border", align_corners=False)
    inside = ((xs >= 0) & (xs <= width)).view(1, -1) & ((ys >= 0) & (ys <= height)).view(-1, 1)
    return cut[0].permute(1, 2, 0).clamp(0.0, 1.0), inside.float()


def crop_frames(frames: torch.Tensor, track: Track, indices, device) -> tuple:
    """Cut the window out of some frames of a clip.

    Args:
        frames: ``(frames, height, width, channels)`` of the whole clip.
        track: The window's track.
        indices: Frame indices to cut.
        device: Where to work.

    Returns:
        ``(crops, inside)``: ``(n, crop height, crop width, channels)`` and
        ``(n, crop height, crop width)``, 1 where the window lies inside the frame.
    """
    cuts = [crop_one(frames[index], track, index, device) for index in indices]
    return torch.stack([cut for cut, _ in cuts]), torch.stack([inside for _, inside in cuts])


def detail(crop: torch.Tensor, mask: torch.Tensor) -> float:
    """Fine detail of the subject in one crop: Laplacian energy of its luma inside the mask.

    Args:
        crop: ``(height, width, channels)`` in 0 to 1.
        mask: ``(height, width)`` subject mask in crop space.

    Returns:
        Mean squared Laplacian over the subject, 0 where the mask is empty.
    """
    weight = mask.to(crop.device, torch.float32).clamp(0.0, 1.0)
    covered = float(weight.sum())
    if covered < 16.0:
        return 0.0
    luma = (crop[..., :3].float() * torch.tensor([0.2126, 0.7152, 0.0722], device=crop.device)).sum(-1)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=crop.device).view(1, 1, 3, 3)
    edges = F.conv2d(luma.view(1, 1, *luma.shape), kernel, padding=1)[0, 0]
    inner = -F.max_pool2d(-weight.view(1, 1, *weight.shape), 5, stride=1, padding=2)[0, 0]
    return float((edges.pow(2) * inner).sum()) / max(1.0, float(inner.sum()))


def scaled(track: Track, frame: tuple) -> Track:
    """The same track on a clip enlarged or reduced uniformly to ``frame``.

    Args:
        track: The window's track.
        frame: ``(height, width)`` of the other clip.

    Returns:
        A :class:`Track` on ``frame``, its crop size unchanged.
    """
    ratio = frame[1] / track.frame[1]
    return Track(
        frame=(int(frame[0]), int(frame[1])), size=track.size,
        centres=[(x * ratio, y * ratio) for x, y in track.centres],
        sides=[side * ratio for side in track.sides], found=list(track.found), reference=track.reference,
    )


def register(moving: torch.Tensor, fixed: torch.Tensor, weight: torch.Tensor, limit: float) -> tuple:
    """The shift that lays ``moving`` over ``fixed``, by phase correlation of their blurred luma.

    Args:
        moving: ``(height, width, channels)`` picture to shift.
        fixed: ``(height, width, channels)`` picture it should land on.
        weight: ``(height, width)`` where to compare, 0 to 1.
        limit: Largest shift allowed on either axis, in pixels.

    Returns:
        ``(dy, dx)`` in pixels, subpixel; ``(0, 0)`` when no clear peak lies within ``limit``.
    """
    from ..image import develop

    height, width = int(moving.shape[0]), int(moving.shape[1])
    coefficients = torch.tensor([0.2126, 0.7152, 0.0722], device=moving.device)
    sigma = max(1.0, min(height, width) / ALIGN_BLUR)
    planes = []
    for picture in (fixed, moving):
        luma = (picture[..., :3].float() * coefficients).sum(-1)
        planes.append(develop.soft_blur(luma.view(1, 1, height, width), sigma)[0, 0])
    window = torch.hann_window(height, device=moving.device).view(-1, 1) * torch.hann_window(width, device=moving.device).view(1, -1)
    window = window * (0.25 + 0.75 * weight.to(moving.device, torch.float32))
    spectra = [torch.fft.rfft2((plane - (plane * window).sum() / window.sum()) * window) for plane in planes]
    cross = spectra[0] * spectra[1].conj()
    surface = torch.fft.irfft2(cross / (cross.abs() + 1e-9), s=(height, width))
    surface = torch.fft.fftshift(surface)
    centre_y, centre_x = height // 2, width // 2
    reach = int(min(limit, centre_y - 1, centre_x - 1))
    local = surface[centre_y - reach:centre_y + reach + 1, centre_x - reach:centre_x + reach + 1]
    peak = int(torch.argmax(local))
    row, col = divmod(peak, int(local.shape[1]))
    if float(local[row, col]) < ALIGN_PEAK:
        return 0.0, 0.0

    def offset(values, at):
        if at <= 0 or at >= len(values) - 1:
            return 0.0
        left, middle, right = (float(values[at + step]) for step in (-1, 0, 1))
        bend = left - 2.0 * middle + right
        return 0.0 if abs(bend) < 1e-9 else 0.5 * (left - right) / bend

    dy = row - reach + offset(local[:, col], row)
    dx = col - reach + offset(local[row, :], col)
    return float(dy), float(dx)


def shifted(picture: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """A ``(height, width, channels)`` picture moved by ``(dy, dx)`` pixels, edges repeated."""
    height, width = int(picture.shape[0]), int(picture.shape[1])
    xs = torch.arange(width, device=picture.device, dtype=torch.float32) + 0.5 - dx
    ys = torch.arange(height, device=picture.device, dtype=torch.float32) + 0.5 - dy
    planes = picture.float().permute(2, 0, 1).unsqueeze(0)
    moved = F.grid_sample(planes, _grid(xs, ys, width, height), mode="bicubic", padding_mode="border", align_corners=False)
    return moved[0].permute(1, 2, 0).clamp(0.0, 1.0)


def native(track: Track, index: int, limit: int) -> Track:
    """A one-frame track whose crop is the window at the frame's own pixel size, up to ``limit``.

    Args:
        track: The window's track.
        index: The frame.
        limit: Longest side of the crop.

    Returns:
        A :class:`Track` holding only ``index``, its crop at least the track's size.
    """
    crop_h, crop_w = track.size
    grow = max(1.0, min(track.scale(index), limit / max(crop_h, crop_w)))
    size = (max(crop_h, int(round(crop_h * grow / 8)) * 8), max(crop_w, int(round(crop_w * grow / 8)) * 8))
    return Track(
        frame=track.frame, size=size, centres=[track.centres[index]], sides=[track.sides[index]],
        found=[track.found[index]], reference=0,
    )


def soft_alpha(masks, count: int, size: tuple, grow: float, feather: float, device) -> torch.Tensor:
    """How much of each crop is laid back, in crop pixels: grown, feathered, faded at the edge.

    Args:
        masks: ``(frames, h, w)`` subject mask in crop space, one for all, or ``None`` for an
            ellipse filling the crop.
        count: Frames.
        size: ``(height, width)`` of the crop.
        grow: Pixels the mask is grown by before feathering; negative shrinks it.
        feather: Gaussian sigma of the edge, in crop pixels.
        device: Where to work.

    Returns:
        ``(frames, height, width)`` in 0 to 1.
    """
    from ..image import develop

    crop_h, crop_w = size
    if masks is None:
        ys = (torch.arange(crop_h, device=device, dtype=torch.float32) + 0.5) / crop_h * 2.0 - 1.0
        xs = (torch.arange(crop_w, device=device, dtype=torch.float32) + 0.5) / crop_w * 2.0 - 1.0
        plane = ((ys.view(-1, 1) ** 2 + xs.view(1, -1) ** 2) <= 0.8 ** 2).float()
        planes = plane.expand(count, crop_h, crop_w).clone()
    else:
        planes = masks.to(device=device, dtype=torch.float32)
        if tuple(planes.shape[-2:]) != (crop_h, crop_w):
            planes = F.interpolate(planes.unsqueeze(1), size=(crop_h, crop_w), mode="bilinear", align_corners=False)[:, 0]
        if int(planes.shape[0]) < count:
            planes = torch.cat([planes, planes[-1:].expand(count - int(planes.shape[0]), crop_h, crop_w)])
        planes = planes[:count]
    reach = int(round(abs(grow)))
    if reach:
        pool = 2 * reach + 1
        if grow > 0:
            planes = F.max_pool2d(planes.unsqueeze(1), pool, stride=1, padding=reach)[:, 0]
        else:
            planes = -F.max_pool2d(-planes.unsqueeze(1), pool, stride=1, padding=reach)[:, 0]
    if feather > 0:
        planes = develop.soft_blur(planes.unsqueeze(1), float(feather))[:, 0]
    edge = max(1.0, 2.0 * float(feather))
    ys = torch.arange(crop_h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(crop_w, device=device, dtype=torch.float32) + 0.5
    fade_y = torch.minimum(ys, crop_h - ys).clamp(max=edge) / edge
    fade_x = torch.minimum(xs, crop_w - xs).clamp(max=edge) / edge
    return (planes.clamp(0.0, 1.0) * fade_y.view(-1, 1) * fade_x.view(1, -1)).clamp(0.0, 1.0)


def paste_frames(frames: torch.Tensor, crops: torch.Tensor, alpha: torch.Tensor, track: Track, indices, device) -> None:
    """Lay crops back onto their frames in place, through their alpha.

    Args:
        frames: ``(frames, height, width, channels)``, written in place.
        crops: ``(n, crop height, crop width, channels)`` for ``indices``.
        alpha: ``(n, crop height, crop width)`` in 0 to 1.
        track: The window's track.
        indices: Frame indices the crops belong to.
        device: Where to work.
    """
    crop_h, crop_w = track.size
    height, width = track.frame
    channels = int(frames.shape[-1])
    for slot, index in enumerate(indices):
        weight = alpha[slot]
        if float(weight.max()) <= 0.0:
            continue
        left, top, right, bottom = track.box(index)
        x0, y0 = max(0, int(math.floor(left)) - 1), max(0, int(math.floor(top)) - 1)
        x1, y1 = min(width, int(math.ceil(right)) + 1), min(height, int(math.ceil(bottom)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        scale = track.scale(index)
        x, y = track.centres[index]
        xs = (torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5 - x) / scale + crop_w / 2.0
        ys = (torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5 - y) / scale + crop_h / 2.0
        grid = _grid(xs, ys, crop_w, crop_h)
        picture = crops[slot].to(device=device, dtype=torch.float32)[..., :channels].permute(2, 0, 1).unsqueeze(0)
        cover = weight.to(device=device, dtype=torch.float32).view(1, 1, crop_h, crop_w)
        shrink = 1.0 / scale
        laid = F.grid_sample(_reduced(picture, shrink), grid, mode="bicubic" if shrink < REDUCE_FROM else "bilinear",
                             padding_mode="border", align_corners=False)[0].permute(1, 2, 0).clamp(0.0, 1.0)
        amount = F.grid_sample(_reduced(cover, shrink), grid, mode="bilinear", padding_mode="zeros",
                               align_corners=False)[0, 0].clamp(0.0, 1.0).unsqueeze(-1)
        region = frames[index, y0:y1, x0:x1]
        mixed = region.to(device=device, dtype=torch.float32) * (1.0 - amount) + laid * amount
        region.copy_(mixed.to(region.device, region.dtype))
