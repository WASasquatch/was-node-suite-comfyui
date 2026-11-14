"""Region operations on mask images.

Every function takes and returns a PIL image, apart from :func:`dominant_plane`, which
takes and answers picture codes. Return mode is ``L`` for some and ``RGB`` for others, and
inversion varies by function.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageFilter, ImageOps

from .. import log

__all__ = [
    "arbitrary_region",
    "ceiling_region",
    "combine_masks",
    "crop_dominant_region",
    "crop_minority_region",
    "crop_region",
    "dominant_plane",
    "dominant_region",
    "fill_region",
    "floor_region",
    "gaussian_region",
    "minority_region",
    "morph_region",
    "smooth_region",
    "threshold_region",
]

logger = log.get_logger("mask.regions")

#: Level a sample must exceed for the two cropping operations to count it as set.
CROP_LEVEL = 128

#: How far a Gaussian kernel reaches, as a multiple of its standard deviation.
KERNEL_REACH = 4.0

#: Accumulators a pairwise sum interleaves, and the longest run it adds without halving.
SUM_LANES = 8
SUM_BLOCK = 128

_DEVICE: list = []


def crop_dominant_region(image: Image.Image, padding: int = 0) -> Image.Image:
    """Crop the largest connected white region and centre it on a square canvas.

    Args:
        image: Mask image, converted to ``L`` internally.
        padding: Pixels added to every side of the square canvas.

    Returns:
        An inverted ``L`` image, square, with a side of
        ``max(crop width, crop height) + 2 * padding``.

    Raises:
        ValueError: The thresholded image has no white region, so there is nothing to
            crop to.
    """
    return _crop_extreme_region(image, padding, "Mask Crop Dominant Region", largest=True)


def crop_minority_region(image: Image.Image, padding: int = 0) -> Image.Image:
    """Crop the smallest connected white region and centre it on a square canvas.

    Args:
        image: Mask image, converted to ``L`` internally.
        padding: Pixels added to every side of the square canvas.

    Returns:
        An inverted ``L`` image, square, with a side of
        ``max(crop width, crop height) + 2 * padding``.

    Raises:
        ValueError: The thresholded image has no white region, so there is nothing to
            crop to.
    """
    return _crop_extreme_region(image, padding, "Mask Crop Minority Region", largest=False)


def crop_region(
    mask: Image.Image, region_type: str, padding: int = 0
) -> tuple[Image.Image, tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """Crop a mask to a square window centred on its bounding box.

    Args:
        mask: Mask image. The chosen region's bounding box drives the crop.
        region_type: ``"dominant"`` measures the largest connected region, ``"minority"``
            the smallest. Anything else measures everything the mask marks.
        padding: Pixels added to each side of the bounding box before it is squared.

    Returns:
        ``(cropped_mask, crop_data)``, where ``crop_data`` is
        ``(cropped size, (x1, y1, x2, y2))`` in source coordinates, the form
        ``Mask Paste Region`` reads back. A mask with no bounding box returns unchanged
        alongside ``(mask.size, (0, 0, 0, 0))``.
    """
    measured = mask
    if region_type == "dominant":
        measured = dominant_region(mask)
    elif region_type == "minority":
        measured = minority_region(mask)

    bbox = measured.getbbox() or mask.getbbox()
    if bbox is None:
        return mask, (mask.size, (0, 0, 0, 0))

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    side_length = max(bbox_width, bbox_height) + 2 * padding

    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2

    crop_x = center_x - side_length // 2
    crop_y = center_y - side_length // 2

    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    crop_x2 = min(crop_x + side_length, mask.width)
    crop_y2 = min(crop_y + side_length, mask.height)

    cropped_mask = mask.crop((crop_x, crop_y, crop_x2, crop_y2))
    crop_data = (cropped_mask.size, (crop_x, crop_y, crop_x2, crop_y2))

    return cropped_mask, crop_data


def dominant_region(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Keep only the largest connected region of the inverted mask.

    Args:
        image: Mask image, converted to ``L`` internally.
        threshold: Grey level above which an inverted pixel counts as part of a region.

    Returns:
        An ``RGB`` image, white over the winning region and black everywhere else. When
        labelling finds nothing, the background is returned as the region.
    """
    plane = dominant_plane(_plane(image), threshold)
    return _picture(plane.unsqueeze(-1).expand(-1, -1, 3), "RGB")


def dominant_plane(plane: torch.Tensor, threshold: int = 128) -> torch.Tensor:
    """Keep only the largest connected region of an inverted grey plane.

    Args:
        plane: ``(height, width)`` int64 picture codes.
        threshold: Grey level above which an inverted sample counts as part of a region.

    Returns:
        Int64 picture codes of the same shape, 255 over the winning region and 0 elsewhere.
        When labelling finds nothing, the background is returned as the region.
    """
    return _extreme_plane(plane, threshold, invert=True, largest=True)


def minority_region(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Keep only the smallest connected region of the mask.

    Unlike :func:`dominant_region`, the mask is labelled as it arrives, without an
    inversion first.

    Args:
        image: Mask image, converted to ``L`` internally.
        threshold: Grey level above which a pixel counts as part of a region.

    Returns:
        An ``RGB`` image, white over the smallest region and black everywhere else. When
        labelling finds nothing, the background is returned as the region.
    """
    plane = _extreme_plane(_plane(image), threshold, invert=False, largest=False)
    return _picture(plane.unsqueeze(-1).expand(-1, -1, 3), "RGB")


def arbitrary_region(image: Image.Image, size: int, threshold: int = 128) -> Image.Image:
    """Keep the smallest connected region that is still at least ``size`` big.

    Args:
        image: Mask image, converted to ``L`` internally.
        size: Minimum region area, in ten-thousandths of the image area.
        threshold: Grey level above which a pixel counts as part of a region.

    Returns:
        An ``L`` image of the winning region, white over black. When no region reaches
        the scaled size, the greyscale source image is returned instead of a mask.
    """
    plane = _plane(image)
    marked = plane > threshold
    roots = _label(marked, diagonal=True)
    found, areas = _areas(roots)

    wanted = size * plane.shape[0] * plane.shape[1] / 10000
    reached = areas.to(torch.float64) >= wanted
    if bool(reached.any()):
        ranked = torch.where(reached, areas, int(areas.max()) + 1)
        winner = found[_first(ranked, ranked.min())]
        return _picture(torch.where(roots == winner, 255, 0))

    return _picture(plane)


def smooth_region(image: Image.Image, tolerance: float) -> Image.Image:
    """Blur a mask and re-threshold it, rounding off its edges.

    Args:
        image: Mask image, converted to ``L`` internally.
        tolerance: Gaussian sigma. Larger values round off more.

    Returns:
        An inverted ``RGB`` image, hard black and white with no intermediate levels.
    """
    blurred = _gaussian(_plane(image), tolerance)
    level = float(blurred.max()) / 2 if blurred.numel() else 0.0
    marked = torch.where(blurred.to(torch.float64) >= level, 255, 0)
    return _picture((255 - marked).unsqueeze(-1).expand(-1, -1, 3), "RGB")


def morph_region(
    image: Image.Image, iterations: int = 1, grow: bool = True, blur: float = 0.0
) -> Image.Image:
    """Grow or shrink the white area of a mask by binary morphology.

    Any non-zero pixel counts as set. A shrink holds the frame edge.

    Args:
        image: Mask image, converted to ``L`` internally.
        iterations: Passes over the four-neighbour cross. Zero and below run until the
            result stops changing.
        grow: Dilate the set area when true, erode it when false.
        blur: Gaussian radius in pixels, applied once the shape has settled. 0 leaves the
            hard binary edge.

    Returns:
        An inverted ``RGB`` image of the reshaped mask.
    """
    shaped = _morphed(_plane(image) > 0, iterations, grow)
    codes = torch.where(shaped, 255, 0)
    # gaussian_region inverts what it is handed and does not invert it back.
    if blur > 0:
        return gaussian_region(_picture(codes), blur)
    return _picture((255 - codes).unsqueeze(-1).expand(-1, -1, 3), "RGB")


def fill_region(image: Image.Image) -> Image.Image:
    """Fill enclosed holes in a mask.

    A hole touching the image border is open, not enclosed, and is left alone.

    Args:
        image: Mask image, converted to ``L`` internally.

    Returns:
        An inverted ``RGB`` image of the filled mask.
    """
    marked = _plane(image) > 0
    codes = torch.where(_filled(marked), 255, 0)
    return _picture((255 - codes).unsqueeze(-1).expand(-1, -1, 3), "RGB")


def combine_masks(*masks: Image.Image) -> Image.Image:
    """Intersect masks by taking the darkest pixel of each set.

    Args:
        *masks: Two or more mask images of identical dimensions, converted to ``L``
            internally.

    Returns:
        An ``L`` image, the pixel-wise minimum of the inputs over a white canvas.

    Raises:
        ValueError: No mask was given, or the masks differ in size.
    """
    if len(masks) < 1:
        raise ValueError("\033[34mWAS NS\033[0m Error: At least one mask must be provided.")
    dimensions = masks[0].size
    for mask in masks:
        if mask.size != dimensions:
            raise ValueError("\033[34mWAS NS\033[0m Error: All masks must have the same dimensions.")

    return _picture(torch.stack([_plane(mask) for mask in masks]).amin(0))


def threshold_region(
    image: Image.Image, black_threshold: int = 0, white_threshold: int = 255
) -> Image.Image:
    """Clip a mask's extremes to black and white.

    Args:
        image: Mask image, converted to ``L`` internally.
        black_threshold: Levels below this become 0.
        white_threshold: Levels above this become 255.

    Returns:
        An inverted ``L`` image.
    """
    plane = _plane(image)
    plane = torch.where(plane < black_threshold, 0, plane)
    plane = torch.where(plane > white_threshold, 255, plane)
    return _picture(255 - plane)


def floor_region(image: Image.Image) -> Image.Image:
    """Binarize a mask at its own darkest non-black level.

    Args:
        image: Mask image, converted to ``L`` internally.

    Returns:
        An inverted ``L`` image.
    """
    plane = _plane(image)
    lit = plane > 0
    if bool(lit.any()):
        plane = torch.where(plane > plane[lit].min(), 255, 0)
    return _picture(255 - plane)


def ceiling_region(image: Image.Image, offset: int = 30) -> Image.Image:
    """Keep only the brightest band of a mask.

    Args:
        image: Mask image, converted to ``L`` internally.
        offset: Width of the surviving band below white, clamped to 0-255.

    Returns:
        An inverted ``L`` image.
    """
    if offset < 0:
        offset = 0
    elif offset > 255:
        offset = 255
    plane = _plane(image)
    plane = torch.where(plane < 255 - offset, 0, plane)
    plane = torch.where(plane >= 250, 255, plane)
    return _picture(255 - plane)


def gaussian_region(image: Image.Image, radius: float = 5.0) -> Image.Image:
    """Feather a mask with a Gaussian blur.

    The radius is truncated to an integer, so 5.9 blurs as 5.

    Args:
        image: Mask image, converted to ``L`` internally.
        radius: Blur radius in pixels.

    Returns:
        An ``RGB`` image of the blurred mask. The input is inverted before the blur and
        the result is not inverted back.
    """
    image = ImageOps.invert(image.convert("L"))
    image = image.filter(ImageFilter.GaussianBlur(radius=int(radius)))
    return image.convert("RGB")


def _crop_extreme_region(
    image: Image.Image, padding: int, node_name: str, *, largest: bool
) -> Image.Image:
    """Crop one connected white region of a mask, chosen by area, onto a square canvas.

    Args:
        image: Mask image, converted to ``L`` internally.
        padding: Pixels added to every side of the square canvas.
        node_name: Name of the caller, used in the error raised for a mask with no white
            region.
        largest: Take the region with the greatest area. False takes the smallest.

    Returns:
        An inverted ``L`` image, square, with a side of
        ``max(crop width, crop height) + 2 * padding``.

    Raises:
        ValueError: The thresholded image has no white region, so there is nothing to
            crop to.
    """
    plane = _plane(image)
    roots = _label(plane > CROP_LEVEL)
    found, areas = _areas(roots)
    if areas.numel() < 1:
        raise ValueError(
            f"This mask has nothing above half brightness for {node_name} to crop to. "
            "Paint or threshold a region into it first."
        )
    winner = found[_first(areas, areas.max() if largest else areas.min())]
    region = roots == winner
    rows = torch.nonzero(region.any(1)).flatten()
    columns = torch.nonzero(region.any(0)).flatten()
    cropped = plane[int(rows[0]):int(rows[-1]) + 1, int(columns[0]):int(columns[-1]) + 1]

    height, width = cropped.shape
    side = max(height, width) + 2 * padding
    canvas = cropped.new_zeros((side, side))
    left = (side - width) // 2
    top = (side - height) // 2
    # Pasting a mask over itself squares every sample and divides the square by 255.
    carried = cropped * cropped + 128
    canvas[top:top + height, left:left + width] = ((carried >> 8) + carried) >> 8
    return _picture(255 - canvas)


def _extreme_plane(
    plane: torch.Tensor, threshold: int, *, invert: bool, largest: bool
) -> torch.Tensor:
    """Keep one connected region of a thresholded plane, chosen by area.

    Args:
        plane: ``(height, width)`` int64 picture codes.
        threshold: Grey level above which a sample counts as part of a region.
        invert: Invert the plane before it is thresholded and labelled.
        largest: Take the region with the greatest area. False takes the smallest.

    Returns:
        Int64 picture codes of the same shape, 255 over the winning region and 0
        elsewhere. When labelling finds nothing, the background is returned as the region.
    """
    marked = (255 - plane if invert else plane) > threshold
    roots = _label(marked)
    found, areas = _areas(roots)
    if areas.numel() < 1:
        return torch.where(marked, 0, 255)
    winner = found[_first(areas, areas.max() if largest else areas.min())]
    return torch.where(roots == winner, 255, 0)


def connected(marked: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
    """Label the connected regions of a boolean plane, as :func:`_label` does.

    Args:
        marked: ``(height, width)`` boolean plane, true over the regions.
        diagonal: Join samples that touch only at a corner.

    Returns:
        An int64 plane holding each set sample's region label, and -1 for every clear sample.
    """
    return _label(marked, diagonal)


def _label(marked: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
    """Label the connected regions of a boolean plane.

    Args:
        marked: ``(height, width)`` boolean plane, true over the regions.
        diagonal: Join samples that touch only at a corner. False joins the four samples
            sharing an edge.

    Returns:
        An int64 plane carrying, for every set sample, the flat index of the first sample
        of its region in raster order, and -1 for every clear sample. Labelling a region
        by its first sample orders the regions the way a raster scan meets them.
    """
    height, width = marked.shape
    device = marked.device
    index = torch.arange(height * width, device=device, dtype=torch.int64).view(height, width)
    unreached = torch.full_like(index, torch.iinfo(torch.int64).max)
    parent = index.flatten().clone()

    while True:
        seen = torch.where(marked, parent.view(height, width), unreached)
        best = seen.clone()
        best[:-1] = torch.minimum(best[:-1], seen[1:])
        best[1:] = torch.minimum(best[1:], seen[:-1])
        best[:, :-1] = torch.minimum(best[:, :-1], seen[:, 1:])
        best[:, 1:] = torch.minimum(best[:, 1:], seen[:, :-1])
        if diagonal:
            best[:-1, :-1] = torch.minimum(best[:-1, :-1], seen[1:, 1:])
            best[1:, 1:] = torch.minimum(best[1:, 1:], seen[:-1, :-1])
            best[:-1, 1:] = torch.minimum(best[:-1, 1:], seen[1:, :-1])
            best[1:, :-1] = torch.minimum(best[1:, :-1], seen[:-1, 1:])
        best = torch.where(marked, best, index).flatten()

        moved = parent.clone()
        moved.scatter_reduce_(0, parent, best, reduce="amin", include_self=True)
        moved = torch.minimum(moved, best)
        while True:
            jumped = moved[moved]
            if bool(torch.equal(jumped, moved)):
                break
            moved = jumped
        if bool(torch.equal(moved, parent)):
            break
        parent = moved

    return torch.where(marked, parent.view(height, width), torch.full_like(index, -1))


def _areas(roots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """How many samples each labelled region holds.

    Args:
        roots: The plane :func:`_label` answered.

    Returns:
        ``(labels, areas)``, the labels ascending, so their order is the order a raster
        scan meets the regions. Both are empty where nothing was labelled.
    """
    flat = roots.flatten()
    kept = flat[flat >= 0]
    if kept.numel() < 1:
        empty = roots.new_empty(0)
        return empty, empty
    return torch.unique(kept, sorted=True, return_counts=True)


def _first(values: torch.Tensor, wanted) -> int:
    """The position of the first sample equal to a value.

    Args:
        values: One-dimensional tensor.
        wanted: The value to look for, which must be present.

    Returns:
        The lowest index holding it.
    """
    return int(torch.nonzero(values == wanted)[0])


def _morphed(marked: torch.Tensor, iterations: int, grow: bool) -> torch.Tensor:
    """Dilate or erode a boolean plane over the four-neighbour cross.

    Args:
        marked: ``(height, width)`` boolean plane.
        iterations: Passes to run. Zero and below run until the plane stops changing.
        grow: Dilate when true, erode when false. A dilation reads the samples outside
            the frame as clear and an erosion reads them as set.

    Returns:
        A boolean plane of the same shape.
    """
    passes = iterations if iterations > 0 else -1
    outside = not grow
    while passes != 0:
        above = torch.full_like(marked, outside)
        above[:-1] = marked[1:]
        below = torch.full_like(marked, outside)
        below[1:] = marked[:-1]
        left = torch.full_like(marked, outside)
        left[:, :-1] = marked[:, 1:]
        right = torch.full_like(marked, outside)
        right[:, 1:] = marked[:, :-1]
        if grow:
            shifted = marked | above | below | left | right
        else:
            shifted = marked & above & below & left & right
        if passes < 0 and bool(torch.equal(shifted, marked)):
            return marked
        marked = shifted
        passes -= 1
    return marked


def _filled(marked: torch.Tensor) -> torch.Tensor:
    """Set every clear sample a border-connected run cannot reach.

    Args:
        marked: ``(height, width)`` boolean plane.

    Returns:
        A boolean plane of the same shape, set over the regions and over their holes. A
        plane with no area is answered unchanged.
    """
    if not marked.numel():
        return marked
    roots = _label(~marked)
    frame = torch.cat([roots[0], roots[-1], roots[:, 0], roots[:, -1]])
    open_air = torch.unique(frame[frame >= 0])
    if open_air.numel() < 1:
        return torch.ones_like(marked)
    return ~torch.isin(roots, open_air)


def _gaussian(plane: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur picture codes with a truncated Gaussian kernel, one axis after the other.

    Args:
        plane: ``(height, width)`` int64 picture codes.
        sigma: Standard deviation in pixels. Zero and below answer the plane unchanged.

    Returns:
        Int64 picture codes of the same shape. Samples off an edge mirror the samples
        inside it, and each axis is truncated back to picture codes before the next runs.
        A plane with no area is answered unchanged.
    """
    if sigma <= 0 or not plane.numel():
        return plane
    reach = int(KERNEL_REACH * sigma + 0.5)
    steps = torch.arange(-reach, reach + 1, dtype=torch.float64)
    weights = torch.exp(steps * steps * (-0.5 / (sigma * sigma)))
    weights = (weights / _total(weights.tolist())).to(plane.device)

    for axis in (0, 1):
        length = plane.shape[axis]
        mirrored = torch.arange(-reach, length + reach, device=plane.device) % (2 * length)
        mirrored = torch.where(mirrored >= length, 2 * length - 1 - mirrored, mirrored)
        padded = torch.index_select(plane, axis, mirrored).to(torch.float64)
        total = padded.narrow(axis, reach, length) * weights[reach]
        # The kernel is symmetric, so each pair of samples is added before it is weighted,
        # working inwards from the pair furthest out.
        for step in range(reach, 0, -1):
            pair = padded.narrow(axis, reach + step, length) + padded.narrow(axis, reach - step, length)
            total = total + pair * weights[reach + step]
        plane = total.to(torch.int64).clamp(0, 255)
    return plane


def _total(values: list[float], start: int = 0, count: int | None = None) -> float:
    """Add up a run of doubles, pairwise.

    Args:
        values: The doubles.
        start: First position of the run.
        count: Length of the run. ``None`` runs to the end of ``values``.

    Returns:
        Their sum. Under :data:`SUM_LANES` values are added in order, up to
        :data:`SUM_BLOCK` are added in that many interleaved accumulators, and a longer run
        is halved first, the split rounded down to a multiple of :data:`SUM_LANES`.
    """
    if count is None:
        count = len(values) - start
    if count < SUM_LANES:
        total = 0.0
        for step in range(count):
            total += values[start + step]
        return total
    if count <= SUM_BLOCK:
        lanes = [values[start + lane] for lane in range(SUM_LANES)]
        step = SUM_LANES
        while step <= count - SUM_LANES:
            for lane in range(SUM_LANES):
                lanes[lane] += values[start + step + lane]
            step += SUM_LANES
        total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + \
                ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        while step < count:
            total += values[start + step]
            step += 1
        return total
    half = count // 2
    half -= half % SUM_LANES
    return _total(values, start, half) + _total(values, start + half, count - half)


def _plane(image: Image.Image) -> torch.Tensor:
    """One greyscale picture-code plane from a PIL image.

    Args:
        image: Any PIL image, converted to ``L`` first.

    Returns:
        A ``(height, width)`` int64 tensor on the device :func:`_device` answered. An image
        with no area answers an empty tensor carrying its shape.
    """
    grey = image.convert("L")
    data = bytearray(grey.tobytes())
    samples = (torch.frombuffer(data, dtype=torch.uint8) if data
               else torch.empty(0, dtype=torch.uint8))
    return samples.view(grey.height, grey.width).to(_device(), torch.int64)


def _picture(plane: torch.Tensor, mode: str = "L") -> Image.Image:
    """A PIL image from picture codes.

    Args:
        plane: ``(height, width)`` int64 picture codes for mode ``L``, or
            ``(height, width, 3)`` for mode ``RGB``.
        mode: PIL mode the image is built in.

    Returns:
        A PIL image in ``mode``.
    """
    samples = plane.to(torch.uint8).contiguous().cpu()
    return Image.frombytes(mode, (plane.shape[1], plane.shape[0]), samples.numpy().tobytes())


def _device():
    """The device the region maths runs on.

    Returns:
        A ``torch.device``: ComfyUI's compute device where it carries the whole-number
        operations a region pass needs, and the CPU where it does not.
    """
    if _DEVICE:
        return _DEVICE[0]
    chosen = torch.device("cpu")
    try:
        from ..model import compute_device

        candidate = compute_device()
        probe = torch.zeros(2, dtype=torch.int64, device=candidate)
        probe.scatter_reduce_(0, probe, probe, reduce="amin", include_self=True)
        chosen = candidate
    except Exception as error:
        logger.debug("region maths runs on the CPU: %s", error)
    _DEVICE.append(chosen)
    return chosen
