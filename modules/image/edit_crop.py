"""Crop the regions an edit changes, sized for an edit model, and paste each edit back through them.

Windows are ``(left, top, right, bottom)`` in source pixels, right and bottom exclusive. Pictures
are ``(height, width, channels)`` floats in ``[0, 1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .. import log
from ..mask.regions import connected

__all__ = [
    "GROUPINGS",
    "EditCrop",
    "Region",
    "crop_picture",
    "find_regions",
    "outline",
    "paste",
    "plan",
]

logger = log.get_logger("image.edit_crop")

#: Matte level a sample must pass to belong to a region.
REGION_LEVEL = 0.5

#: Longest side in pixels a mask is labelled at when its regions are found.
LABEL_SIDE = 1024

#: Least share of a mask's area a region keeps; smaller specks are passed over.
LEAST_REGION = 0.0005

#: How regions become crops: `auto` joins regions whose boxes overlap, and others while one crop
#: still holds them at full resolution; `one crop` joins them all; `each region` keeps them apart.
GROUPINGS = ("auto", "one crop", "each region")

#: Least share of a crop outside its regions that drift is measured on.
LEAST_REFERENCE = 0.02

#: Most a crop is enlarged; a smaller window is widened with the picture around it first.
MOST_ENLARGEMENT = 4.0

#: Side of the grid drift is measured on, in pixels of the crop.
DRIFT_STEP = 8

#: Least share of an edit a pixel takes; below one 8-bit step it takes none and stays the original.
ALPHA_FLOOR = 1.0 / 255.0

#: Share of its box's longer side a region pasted whole is grown by.
WHOLE_MARGIN = 0.15

#: Least share of an edited area lying over a region for the area to be what replaced it.
REPLACEMENT_SHARE = 0.2


@dataclass
class Region:
    """One area an edit may change.

    Attributes:
        box: Its bounding box in source pixels.
        plane: The mask plane it was found on.
        support: Where it lies, as a boolean plane at :data:`LABEL_SIDE` or less.
        matte: Its soft matte over its crop's window, filled in by :func:`plan`.
        whole: Its edit is pasted through its box, grown by :data:`WHOLE_MARGIN`, rather than its shape.
    """

    box: tuple[int, int, int, int]
    plane: int
    support: torch.Tensor | None = None
    matte: torch.Tensor | None = None
    whole: bool = False


@dataclass
class EditCrop:
    """One crop handed to an edit model, and what puts its edit back.

    Attributes:
        source: ``(width, height)`` of the picture cropped.
        window: The crop's window in source pixels.
        size: ``(width, height)`` the crop is handed over at.
        regions: The regions inside it.
        index: Its place among the crops of one picture, from 0.
        count: How many crops the picture gave.
    """

    source: tuple[int, int]
    window: tuple[int, int, int, int]
    size: tuple[int, int]
    regions: list[Region] = field(default_factory=list)
    index: int = 0
    count: int = 1

    @property
    def scale(self) -> float:
        """Crop pixels per source pixel."""
        return self.size[0] / max(1, self.window[2] - self.window[0])


def _union(first, second):
    """The box holding two boxes."""
    return (min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3]))


def _overlap(first, second) -> bool:
    """Whether two boxes share any area."""
    return first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3]


def find_regions(masks: torch.Tensor | None, boxes: list | None, width: int, height: int) -> list[Region]:
    """The separate areas of a stack of masks, or the boxes where no mask marks anything.

    Args:
        masks: ``(planes, height, width)`` mattes in ``[0, 1]``, or ``None``.
        boxes: Bounding boxes as ``{"x", "y", "width", "height"}`` in source pixels, or ``None``.
        width: The picture's width.
        height: The picture's height.

    Returns:
        The regions, plane by plane, each plane's in raster order.
    """
    found: list[Region] = []
    if masks is not None:
        for plane_index in range(masks.shape[0]):
            plane = masks[plane_index].float()
            if float(plane.max()) < REGION_LEVEL:
                continue
            scale = min(1.0, LABEL_SIDE / max(height, width))
            small = plane
            if scale < 1.0:
                size = (max(1, round(height * scale)), max(1, round(width * scale)))
                small = F.interpolate(plane[None, None], size=size, mode="area")[0, 0]
            marked = small > REGION_LEVEL
            labels = connected(marked, diagonal=True)
            names, areas = torch.unique(labels[labels >= 0], return_counts=True)
            least = max(1, int(LEAST_REGION * marked.numel()))
            for name, area in zip(names.tolist(), areas.tolist()):
                if area < least:
                    continue
                support = labels == name
                rows = torch.nonzero(support.any(dim=1)).flatten()
                cols = torch.nonzero(support.any(dim=0)).flatten()
                sy, sx = small.shape
                left = max(0, math.floor(int(cols[0]) * width / sx) - 1)
                top = max(0, math.floor(int(rows[0]) * height / sy) - 1)
                right = min(width, math.ceil((int(cols[-1]) + 1) * width / sx) + 1)
                bottom = min(height, math.ceil((int(rows[-1]) + 1) * height / sy) + 1)
                found.append(Region((left, top, right, bottom), plane_index, support))
    if not found:
        for box in boxes or ():
            left = max(0, int(round(float(box.get("x", 0)))))
            top = max(0, int(round(float(box.get("y", 0)))))
            right = min(width, int(round(float(box.get("x", 0)) + float(box.get("width", 0)))))
            bottom = min(height, int(round(float(box.get("y", 0)) + float(box.get("height", 0)))))
            if right > left and bottom > top:
                found.append(Region((left, top, right, bottom), -1))
    return found


def _padded(box, context: float, width: int, height: int):
    """A box grown by ``context`` times its longer side, held inside the picture."""
    pad = int(round(context * max(box[2] - box[0], box[3] - box[1])))
    return (max(0, box[0] - pad), max(0, box[1] - pad), min(width, box[2] + pad), min(height, box[3] + pad))


def _fits(window, most: int) -> bool:
    """Whether a window is handed over at full resolution: its longest side within ``most``."""
    return max(window[2] - window[0], window[3] - window[1]) <= most


def _groups(regions: list[Region], context: float, grouping: str, width: int, height: int, most: int):
    """The regions joined into crops, each with its window before sizing."""
    groups = [([region], _padded(region.box, context, width, height)) for region in regions]
    if grouping == "one crop" and groups:
        members = [region for group, _ in groups for region in group]
        window = groups[0][1]
        for _, other in groups[1:]:
            window = _union(window, other)
        return [(members, window)]
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                window = _union(groups[a][1], groups[b][1])
                forced = grouping == "auto" and any(
                    _overlap(first.box, second.box) for first in groups[a][0] for second in groups[b][0]
                )
                if grouping != "auto" and not forced:
                    continue
                if not forced and not _fits(window, most):
                    continue
                growth = (window[2] - window[0]) * (window[3] - window[1])
                if best is None or (forced, -growth) > (best[0], -best[1]):
                    best = (forced, growth, a, b, window)
        if best is None:
            break
        _, _, a, b, window = best
        groups[a] = (groups[a][0] + groups[b][0], window)
        del groups[b]
    return groups


def _sized(window, width: int, height: int, least: int, most: int, multiple: int):
    """A window and the size it is handed over at: its own where it fits, else scaled into range."""
    left, top, right, bottom = window
    span_x, span_y = right - left, bottom - top
    floor = min(width, height, math.ceil(least / MOST_ENLARGEMENT))
    if min(span_x, span_y) < floor:
        grown_x, grown_y = max(span_x, floor), max(span_y, floor)
        left = min(max(0, left - (grown_x - span_x) // 2), width - grown_x)
        top = min(max(0, top - (grown_y - span_y) // 2), height - grown_y)
        right, bottom = left + grown_x, top + grown_y
        window = (left, top, right, bottom)
        span_x, span_y = grown_x, grown_y
    short, long = min(span_x, span_y), max(span_x, span_y)
    scale = least / max(1, short) if short < least else 1.0
    scale = min(scale, most / max(1, long), MOST_ENLARGEMENT)
    if scale == 1.0:
        # Grown with real pixels to the next multiple, where the picture has them.
        new_x = min(width, math.ceil(span_x / multiple) * multiple)
        new_y = min(height, math.ceil(span_y / multiple) * multiple)
        fits = max(new_x, new_y) <= most and new_x % multiple == 0 and new_y % multiple == 0
        if fits and new_x >= span_x and new_y >= span_y:
            left = min(max(0, left - (new_x - span_x) // 2), width - new_x)
            top = min(max(0, top - (new_y - span_y) // 2), height - new_y)
            return (left, top, left + new_x, top + new_y), (new_x, new_y)
    size = tuple(max(multiple, int(side * scale / multiple) * multiple) for side in (span_x, span_y))
    return window, size


def _support_matte(region: Region, plane: torch.Tensor, window) -> torch.Tensor:
    """A region's soft matte over a window: its plane where its own support reaches."""
    left, top, right, bottom = window
    part = plane[top:bottom, left:right].float()
    if region.support is None:
        matte = torch.zeros_like(part)
        box = region.box
        matte[max(0, box[1] - top):max(0, box[3] - top), max(0, box[0] - left):max(0, box[2] - left)] = 1.0
        return matte
    height, width = plane.shape
    pad = max(2, math.ceil(max(height, width) / max(region.support.shape)) + 2)
    sy, sx = region.support.shape
    # The support is scaled to the window alone rather than the whole picture.
    y0, y1 = math.floor(top * sy / height), math.ceil(bottom * sy / height)
    x0, x1 = math.floor(left * sx / width), math.ceil(right * sx / width)
    piece = region.support[y0:y1, x0:x1].float()
    full = (round((y1 - y0) * height / sy), round((x1 - x0) * width / sx))
    reach = F.interpolate(piece[None, None], size=full, mode="nearest")[0, 0]
    oy, ox = max(0, top - round(y0 * height / sy)), max(0, left - round(x0 * width / sx))
    reach = _grown(reach, pad)[oy:oy + (bottom - top), ox:ox + (right - left)]
    if reach.shape != part.shape:
        reach = F.pad(reach, (0, part.shape[1] - reach.shape[1], 0, part.shape[0] - reach.shape[0]))
    return (part * reach).clamp(0, 1)


def plan(
    masks: torch.Tensor | None, boxes: list | None, width: int, height: int, context: float,
    grouping: str, least: int, most: int, multiple: int,
) -> list[EditCrop]:
    """The crops one picture's regions are edited in.

    Args:
        masks: ``(planes, height, width)`` mattes, or ``None``.
        boxes: Bounding boxes used where no mask marks anything, or ``None``.
        width: The picture's width.
        height: The picture's height.
        context: Room around the regions, as a share of their longer side.
        grouping: One of :data:`GROUPINGS`.
        least: Shortest side a small crop is enlarged toward.
        most: Longest side a crop is handed over at.
        multiple: Step both sides of a crop are held to.

    Returns:
        One :class:`EditCrop` per crop, empty where nothing is marked.
    """
    regions = find_regions(masks, boxes, width, height)
    crops = []
    groups = _groups(regions, context, grouping, width, height, most)
    for index, (members, window) in enumerate(groups):
        window, size = _sized(window, width, height, least, most, multiple)
        for region in members:
            plane = masks[region.plane] if masks is not None and region.plane >= 0 else torch.zeros((height, width))
            region.matte = _support_matte(region, plane, window)
            region.support = None
        crops.append(EditCrop((width, height), window, size, members, index, len(groups)))
    return crops


def _resized(picture: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """A ``(height, width, channels)`` picture at ``(width, height)``; unchanged at its own size."""
    if (picture.shape[1], picture.shape[0]) == tuple(size):
        return picture
    shrinking = size[0] < picture.shape[1] or size[1] < picture.shape[0]
    moved = picture.permute(2, 0, 1)[None].float()
    moved = F.interpolate(moved, size=(size[1], size[0]), mode="bicubic", align_corners=False, antialias=shrinking)
    return moved[0].permute(1, 2, 0).clamp(0, 1)


def crop_picture(picture: torch.Tensor, crop: EditCrop) -> torch.Tensor:
    """The window of a picture at the size the crop is handed over at."""
    left, top, right, bottom = crop.window
    return _resized(picture[top:bottom, left:right], crop.size)


def _blur(plane: torch.Tensor, sigma: float) -> torch.Tensor:
    """A ``(height, width)`` plane blurred by a Gaussian of ``sigma`` pixels."""
    if sigma <= 0:
        return plane
    radius = max(1, int(math.ceil(sigma * 3)))
    steps = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(steps ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    moved = plane[None, None].float()
    moved = F.conv2d(F.pad(moved, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    moved = F.conv2d(F.pad(moved, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return moved[0, 0]


def _grown(plane: torch.Tensor, pixels: int) -> torch.Tensor:
    """A plane dilated by ``pixels`` over a square, one axis at a time."""
    if pixels <= 0:
        return plane
    moved = plane[None, None].float()
    moved = F.max_pool2d(moved, (1, 2 * pixels + 1), stride=1, padding=(0, pixels))
    moved = F.max_pool2d(moved, (2 * pixels + 1, 1), stride=1, padding=(pixels, 0))
    return moved[0, 0]


def _replacements(later: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The separate areas of an edited crop's matte, labelled on a copy at most :data:`LABEL_SIDE` long."""
    height, width = later.shape
    scale = min(1.0, LABEL_SIDE / max(height, width))
    small = later
    if scale < 1.0:
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        small = F.interpolate(later[None, None], size=size, mode="area")[0, 0]
    return connected(small > REGION_LEVEL, diagonal=True), scale


def _cut(labels: torch.Tensor, inner: tuple[bool, bool, bool, bool]) -> set:
    """The edited areas that reach a window edge lying inside the picture, as label names."""
    edges = []
    if inner[0]:
        edges.append(labels[:, 0])
    if inner[1]:
        edges.append(labels[0, :])
    if inner[2]:
        edges.append(labels[:, -1])
    if inner[3]:
        edges.append(labels[-1, :])
    if not edges:
        return set()
    joined = torch.cat(edges)
    return set(torch.unique(joined[joined >= 0]).tolist())


def _edge_fade(span, inner: tuple[bool, bool, bool, bool], pixels: int) -> torch.Tensor:
    """A ``(height, width)`` ramp from 0 at each window edge inside the picture to 1 ``pixels`` in."""
    width, height = span
    across = torch.arange(width, dtype=torch.float32)
    down = torch.arange(height, dtype=torch.float32)
    ramp_x = torch.ones(width)
    ramp_y = torch.ones(height)
    reach = max(1, pixels)
    if inner[0]:
        ramp_x = torch.minimum(ramp_x, (across / reach).clamp(0, 1))
    if inner[2]:
        ramp_x = torch.minimum(ramp_x, ((width - 1 - across) / reach).clamp(0, 1))
    if inner[1]:
        ramp_y = torch.minimum(ramp_y, (down / reach).clamp(0, 1))
    if inner[3]:
        ramp_y = torch.minimum(ramp_y, ((height - 1 - down) / reach).clamp(0, 1))
    return ramp_y[:, None] * ramp_x[None, :]


def _reach(labels: torch.Tensor, scale: float, span, cut: set, own: torch.Tensor) -> tuple[list, list]:
    """The edited areas that replaced a region: those lying over it by :data:`REPLACEMENT_SHARE`.

    Args:
        labels: The edited areas, labelled at ``scale``.
        scale: Labelled pixels per window pixel.
        span: ``(width, height)`` of the window.
        cut: Areas reaching a window edge inside the picture, never kept.
        own: Where the region lies, at the labels' size.

    Returns:
        ``(names, boxes)``: the kept areas' labels, and their boxes in window pixels.
    """
    touched = labels[own & (labels >= 0)]
    names, boxes = [], []
    for name in torch.unique(touched).tolist():
        if name in cut:
            continue
        area = labels == name
        if float((area & own).sum()) < REPLACEMENT_SHARE * float(area.sum()):
            continue
        rows = torch.nonzero(area.any(dim=1)).flatten()
        cols = torch.nonzero(area.any(dim=0)).flatten()
        names.append(name)
        boxes.append((
            max(0, math.floor(int(cols[0]) / scale) - 1), max(0, math.floor(int(rows[0]) / scale) - 1),
            min(span[0], math.ceil((int(cols[-1]) + 1) / scale) + 1), min(span[1], math.ceil((int(rows[-1]) + 1) / scale) + 1),
        ))
    return names, boxes


def _drift(original: torch.Tensor, edited: torch.Tensor, reference: torch.Tensor) -> torch.Tensor | None:
    """The smooth difference between an edit and its source, measured where nothing was to change."""
    if float(reference.mean()) < LEAST_REFERENCE:
        return None
    height, width = reference.shape
    step = max(1, min(height, width) // (DRIFT_STEP * 8))
    small = (max(1, height // step), max(1, width // step))
    weight = F.interpolate(reference[None, None], size=small, mode="area")[0, 0]
    difference = (edited - original).permute(2, 0, 1)[None]
    difference = F.interpolate(difference * reference[None, None], size=small, mode="area")[0]
    sigma = max(2.0, min(small) / 8)
    held = _blur(weight, sigma)
    overall = (difference.sum(dim=(1, 2)) / weight.sum().clamp(min=1e-6))
    field = torch.stack([_blur(channel, sigma) for channel in difference])
    field = torch.where(held[None] > 1e-3, field / held[None].clamp(min=1e-3), overall[:, None, None])
    field = F.interpolate(field[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return field.permute(1, 2, 0)


def paste(
    picture: torch.Tensor, edited: torch.Tensor, crop: EditCrop, after: torch.Tensor | None,
    grow: int, feather: int, correct_drift: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put one crop's edit back into a picture through its regions alone.

    Args:
        picture: The ``(height, width, channels)`` picture the crop was taken from.
        edited: The edited crop, any size.
        crop: What :func:`plan` recorded for it.
        after: A matte of the edited crop, any size; an area of it lying over a region is pasted
            whole, and an area beside a region or reaching a window edge inside the picture is left out.
        grow: Pixels each region is grown by.
        feather: Pixels each region's edge is softened over.
        correct_drift: Remove the edit's colour drift, measured outside the regions.

    Returns:
        ``(picture, alpha)``: the result, and how much of the edit each pixel took.
    """
    left, top, right, bottom = crop.window
    span = (right - left, bottom - top)
    channels = picture.shape[-1]
    edit = _resized(edited[..., :channels] if edited.shape[-1] >= channels else F.pad(edited, (0, channels - edited.shape[-1]), value=1.0), span)
    before = picture[top:bottom, left:right].float()
    later = None
    if after is not None:
        later = F.interpolate(after.float()[None, None], size=(span[1], span[0]), mode="bilinear", align_corners=False)[0, 0]
    alpha = torch.zeros((span[1], span[0]))
    reach = grow + feather
    width, height = picture.shape[1], picture.shape[0]
    inner = (left > 0, top > 0, right < width, bottom < height)
    labels, label_scale = _replacements(later) if later is not None else (None, 1.0)
    cut = _cut(labels, inner) if labels is not None else set()
    for region in crop.regions:
        own = (region.box[0] - left, region.box[1] - top, region.box[2] - left, region.box[3] - top)
        if region.whole:
            margin = round(WHOLE_MARGIN * max(own[2] - own[0], own[3] - own[1]))
            own = (own[0] - margin, own[1] - margin, own[2] + margin, own[3] + margin)
        if region.whole:
            matte = torch.zeros_like(alpha)
            matte[max(0, own[1]):max(0, own[3]), max(0, own[0]):max(0, own[2])] = 1.0
        else:
            matte = region.matte.float() if region.matte is not None else torch.zeros_like(alpha)
            if region.matte is None:
                matte[max(0, own[1]):max(0, own[3]), max(0, own[0]):max(0, own[2])] = 1.0
        rects = [own]
        replaced = None
        if labels is not None:
            # What replaced a region, found again on the edit, may paste its whole extent.
            lies = F.interpolate(matte[None, None], size=labels.shape, mode="area")[0, 0] > 0.25
            names, boxes = _reach(labels, label_scale, span, cut, _grown(lies.float(), 2) > 0)
            rects += boxes
            if names:
                kept = torch.isin(labels, torch.tensor(names, dtype=labels.dtype)).float()
                replaced = later * _grown(F.interpolate(kept[None, None], size=(span[1], span[0]), mode="nearest")[0, 0], 2)
        allowed = torch.zeros_like(alpha)
        for x0, y0, x1, y1 in rects:
            allowed[max(0, y0 - reach):max(0, y1 + reach), max(0, x0 - reach):max(0, x1 + reach)] = 1.0
        allowed = _blur(allowed, feather / 2.0).clamp(0, 1)
        if replaced is not None:
            matte = torch.maximum(matte, replaced * allowed)
        matte = _blur(_grown(matte, grow), feather / 2.0) * allowed
        alpha = torch.maximum(alpha, matte.clamp(0, 1))
    alpha = alpha * _edge_fade(span, inner, max(2, feather))
    alpha = torch.where(alpha >= ALPHA_FLOOR, alpha, torch.zeros_like(alpha))
    if correct_drift:
        touched = _grown((alpha > 0.002).float(), reach + 4)
        field = _drift(before[..., :3], edit[..., :3], 1.0 - touched)
        if field is not None:
            edit = edit.clone()
            edit[..., :3] = (edit[..., :3] - field).clamp(0, 1)
    blended = before * (1.0 - alpha[..., None]) + edit * alpha[..., None]
    result = picture.clone()
    result[top:bottom, left:right] = torch.where(alpha[..., None] > 0, blended, before).to(picture.dtype)
    full = torch.zeros(picture.shape[:2])
    full[top:bottom, left:right] = alpha
    return result, full


def outline(picture: torch.Tensor, crops: list[EditCrop]) -> torch.Tensor:
    """A picture with each crop's window and each region's box drawn over it."""
    shown = picture[..., :3].clone().float()
    height, width = shown.shape[:2]
    line = max(2, round(max(height, width) / 400))
    colours = (torch.tensor([0.29, 0.62, 1.0]), torch.tensor([1.0, 0.75, 0.2]))
    for crop in crops:
        for box, colour in [(crop.window, colours[0])] + [(region.box, colours[1]) for region in crop.regions]:
            x0, y0, x1, y1 = box
            x1, y1 = min(width, x1), min(height, y1)
            shown[y0:min(y1, y0 + line), x0:x1] = colour
            shown[max(y0, y1 - line):y1, x0:x1] = colour
            shown[y0:y1, x0:min(x1, x0 + line)] = colour
            shown[y0:y1, max(x0, x1 - line):x1] = colour
    return shown
