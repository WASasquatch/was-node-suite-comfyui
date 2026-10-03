"""Cropping a clip to another aspect along a smoothed path that follows its subject.

Crop windows are ``(left, top)`` in pixels of the source frame, all one size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["ASPECTS", "FOLLOW", "crop_size", "cropped", "follow", "marked"]

#: Aspects offered, as ``label: width / height``.
ASPECTS = {
    "9:16": 9 / 16,
    "4:5": 4 / 5,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}

#: Brightness kept outside the window on the marked preview.
OUTSIDE_SHADE = 0.3

#: Width of the window's outline on the marked preview, in pixels.
OUTLINE = 2

#: Share of the frame a mask has to cover before its centre is followed.
MIN_COVER = 0.0005

#: What of a mask the window follows: its largest region, held from frame to frame, or the
#: centre of all of it.
FOLLOW = ("largest", "everything")

#: Long side, in pixels, a mask's regions are told apart at.
REGION_SIDE = 192

#: Share of the largest region's size a region already followed must keep to go on being followed.
HOLD_SHARE = 0.5


def _regions(on):
    """Connected regions of a boolean plane, each pixel labelled by its region's id, 0 outside."""
    height, width = on.shape
    labels = torch.arange(1, height * width + 1, dtype=torch.float32).view(height, width) * on
    for _ in range(height + width):
        grown = F.max_pool2d(labels[None, None], 3, stride=1, padding=1)[0, 0] * on
        if torch.equal(grown, labels):
            break
        labels = grown
    return labels


def _subject(plane, held):
    """The region of a coarse mask the window follows, and its weight per pixel.

    Args:
        plane: ``(h, w)`` mask in ``[0, 1]``.
        held: ``(h, w)`` bool of the region followed on the frame before, or ``None``.

    Returns:
        ``(weight, region)``: the mask kept to the chosen region, and that region as bool.
    """
    labels = _regions(plane > 0.5)
    ids, areas = torch.unique(labels[labels > 0], return_counts=True)
    if ids.numel() == 0:
        return plane, None
    chosen = ids[int(torch.argmax(areas))]
    if held is not None:
        overlap = torch.stack([(held & (labels == region)).sum() for region in ids])
        best = int(torch.argmax(overlap))
        if int(overlap[best]) > 0 and int(areas[best]) >= HOLD_SHARE * int(areas.max()):
            chosen = ids[best]
    region = labels == chosen
    return plane * region, region


def crop_size(height: int, width: int, aspect: float, zoom: float = 1.0) -> tuple[int, int]:
    """The largest even-sided window of ``aspect`` that fits, enlarged by ``zoom``.

    Args:
        height: Frame height.
        width: Frame width.
        aspect: Width over height.
        zoom: How much tighter than the largest fit, at least 1.

    Returns:
        ``(crop height, crop width)``.
    """
    if width / height > aspect:
        crop_h = height / max(zoom, 1.0)
        crop_w = crop_h * aspect
    else:
        crop_w = width / max(zoom, 1.0)
        crop_h = crop_w / aspect
    even = lambda value, limit: max(2, min(limit - limit % 2, 2 * int(value / 2)))
    return even(crop_h, height), even(crop_w, width)


def follow(mask, count: int, frame, crop, cuts, smoothing_frames: float, what: str = "largest"):
    """Where the window sits on every frame: on the subject, smoothed within each scene.

    Args:
        mask: ``(frames, height, width)`` subject mask, one frame for all, or ``None`` to centre.
        count: Frames in the clip.
        frame: ``(height, width)`` of the clip.
        crop: ``(height, width)`` of the window.
        cuts: One flag per neighbouring pair, True where the clip cuts.
        smoothing_frames: Gaussian sigma, in frames, of the path.
        what: One of :data:`FOLLOW`.

    Returns:
        ``(left, top)`` per frame, floats, inside the frame.
    """
    from ..image import camera_motion

    height, width = frame
    crop_h, crop_w = crop
    scale = REGION_SIDE / max(height, width)
    small = (max(8, int(round(height * scale))), max(8, int(round(width * scale))))
    ys = (torch.arange(small[0], dtype=torch.float32).view(-1, 1) + 0.5) * (height / small[0])
    xs = (torch.arange(small[1], dtype=torch.float32).view(1, -1) + 0.5) * (width / small[1])
    bounds = [0] + [index + 1 for index, cut in enumerate(cuts) if cut] + [count]
    starts = set(bounds[:-1])
    centres = []
    previous = (width / 2.0, height / 2.0)
    held = None
    for index in range(count):
        if index in starts:
            held = None
        centre = previous
        if mask is not None:
            plane = mask[min(index, mask.shape[0] - 1)].float().cpu()
            plane = F.interpolate(plane.view(1, 1, *plane.shape[-2:]), size=small, mode="area")[0, 0]
            if float(plane.sum()) > MIN_COVER * small[0] * small[1]:
                weight = plane
                if what == "largest":
                    weight, region = _subject(plane, held)
                    if region is not None:
                        held = F.max_pool2d(region.float()[None, None], 5, stride=1, padding=2)[0, 0] > 0
                total = float(weight.sum())
                if total > 0:
                    centre = (float((weight * xs).sum()) / total, float((weight * ys).sum()) / total)
        centres.append(centre)
        previous = centre
    path = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        xs = camera_motion.smooth([c[0] for c in centres[start:stop]], smoothing_frames)
        ys = camera_motion.smooth([c[1] for c in centres[start:stop]], smoothing_frames)
        for x, y in zip(xs, ys):
            left = min(max(x - crop_w / 2.0, 0.0), float(width - crop_w))
            top = min(max(y - crop_h / 2.0, 0.0), float(height - crop_h))
            path.append((left, top))
    return path


def cropped(frame, left: float, top: float, crop):
    """One window cut from a frame, whole pixels where it sits on them.

    Args:
        frame: ``(1, channels, height, width)``.
        left: Window's left edge in pixels.
        top: Window's top edge in pixels.
        crop: ``(height, width)``.

    Returns:
        ``(1, channels, crop height, crop width)``.
    """
    crop_h, crop_w = crop
    if abs(left - round(left)) < 1e-3 and abs(top - round(top)) < 1e-3:
        x, y = int(round(left)), int(round(top))
        return frame[..., y:y + crop_h, x:x + crop_w]
    _, _, height, width = frame.shape
    xs = (torch.arange(crop_w, device=frame.device, dtype=torch.float32) + left) * (2.0 / max(width - 1, 1)) - 1.0
    ys = (torch.arange(crop_h, device=frame.device, dtype=torch.float32) + top) * (2.0 / max(height - 1, 1)) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], -1).unsqueeze(0)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


def marked(frame, left: float, top: float, crop):
    """A frame with everything outside the window dimmed and the window outlined.

    Args:
        frame: ``(1, channels, height, width)``.
        left: Window's left edge in pixels.
        top: Window's top edge in pixels.
        crop: ``(height, width)``.

    Returns:
        A frame shaped like ``frame``.
    """
    crop_h, crop_w = crop
    _, _, height, width = frame.shape
    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = min(width, x0 + crop_w), min(height, y0 + crop_h)
    out = frame * OUTSIDE_SHADE
    out[..., y0:y1, x0:x1] = frame[..., y0:y1, x0:x1]
    out[..., y0:y0 + OUTLINE, x0:x1] = 1.0
    out[..., max(y1 - OUTLINE, 0):y1, x0:x1] = 1.0
    out[..., y0:y1, x0:x0 + OUTLINE] = 1.0
    out[..., y0:y1, max(x1 - OUTLINE, 0):x1] = 1.0
    return out
