"""Screen-space occlusion behind the occlusion nodes.

:func:`horizon_occlusion` traces rays over a height field in float32, and
:func:`blurred_field` and :func:`quantised` finish a float result. The neighbourhood kernel
the direct pass measures with is float64. Picture codes are int64, 0 to 255.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from .. import log

__all__ = [
    "PRECISIONS",
    "blurred_codes",
    "blurred_field",
    "calculate_direct_occlusion_factor",
    "darkened_codes",
    "grey_codes",
    "horizon_occlusion",
    "quantised",
    "screened_codes",
    "smoothed_codes",
    "stretched_codes",
]

logger = log.get_logger("image.occlusion")

#: Bytes one neighbourhood pass holds at a time. The row count each pass covers follows
#: from it, the image width, the channel count and the neighbourhood width.
PASS_BYTES = 48 << 20

#: Weights of the 5x5 smoothing kernel, row by row, and the number they are divided by.
SMOOTH_WEIGHTS = (
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 5.0, 5.0, 5.0, 1.0,
    1.0, 5.0, 44.0, 5.0, 1.0,
    1.0, 5.0, 5.0, 5.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
)
SMOOTH_SCALE = 100.0

#: Passes the Gaussian blur approximates a kernel with, and the weights of the red, green
#: and blue channels in a grey sample, over 65536.
BLUR_PASSES = 3
GREY_WEIGHTS = (19595, 38470, 7471)

#: Widget option -> steps one unit of an output is divided into. 0 keeps every value.
PRECISIONS = {"8 bit": 255, "16 bit": 65535, "32 bit float": 0}


def quantised(picture: torch.Tensor, levels: int) -> torch.Tensor:
    """Snap a picture to evenly spaced levels.

    Args:
        picture: Float tensor on a 0 to 1 scale, unbounded above.
        levels: Steps one unit is divided into, from :data:`PRECISIONS`. 0 or less
            answers the picture untouched.

    Returns:
        A float tensor of the same shape. Nothing is clamped, so light above 1.0 keeps
        its magnitude and lands on the same ladder of steps.
    """
    if levels <= 0:
        return picture
    return torch.round(picture * levels) / levels


def blurred_field(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Blur a float plane with a run of box filters standing in for a Gaussian kernel.

    Args:
        plane: ``(height, width)`` float tensor.
        radius: Standard deviation of the kernel the run approximates, in pixels.

    Returns:
        A float32 tensor of the same shape. A radius of 0 answers the plane unchanged,
        and the cost does not grow with the radius.
    """
    if radius <= 0:
        return plane
    box = _box_width(radius)
    if box <= 0:
        return plane
    field = plane.to(torch.float32)
    # The running sums below are differences of large partial sums, so the plane is
    # centred first and the level put back at the end.
    level = field.mean()
    field = field - level
    for _ in range(BLUR_PASSES):
        field = _box_pass_field(field, box)
    field = field.T.contiguous()
    for _ in range(BLUR_PASSES):
        field = _box_pass_field(field, box)
    return field.T.contiguous() + level


def _box_width(radius: float) -> float:
    """The box half-width whose repeated passes approximate a Gaussian kernel.

    Args:
        radius: Standard deviation of the kernel the run approximates, in pixels.

    Returns:
        The half-width, which is 0 or less where the radius is too small to blur.
    """
    variance = radius * radius / BLUR_PASSES
    length = math.sqrt(12.0 * variance + 1.0)
    whole = float(int((length - 1.0) / 2.0))
    part = (2 * whole + 1) * (whole * (whole + 1) - 3 * variance)
    part /= 6 * (variance - (whole + 1) * (whole + 1))
    return whole + part


def _box_pass_field(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """One horizontal box-filter pass over a float plane.

    Args:
        plane: ``(rows, columns)`` float32 tensor.
        radius: Box half-width in pixels. The fractional part weights the two samples just
            outside the whole box.

    Returns:
        A float32 tensor of the same shape. Samples off the ends of a row read as the end
        sample.
    """
    whole = int(radius)
    weight = 1.0 / (radius * 2.0 + 1.0)
    edge = (1.0 - (whole * 2 + 1) * weight) / 2.0
    rows, width = plane.shape
    reach = torch.arange(-whole - 1, width + whole + 1, device=plane.device)
    padded = plane[:, reach.clamp(0, width - 1)]
    running = torch.cat(
        [plane.new_zeros((rows, 1)), torch.cumsum(padded, dim=1)], dim=1
    )
    far = 2 * whole + 2
    inner = running[:, far:far + width] - running[:, 1:1 + width]
    return inner * weight + (padded[:, 0:width] + padded[:, far:far + width]) * edge


def horizon_occlusion(height_field: torch.Tensor, radius: float, rays: int, steps: int,
                      relief: float, bias: float = 0.0) -> torch.Tensor:
    """Occlude a height field by the horizon each pixel sees around the full circle.

    Args:
        height_field: ``(height, width)`` float tensor scaled to ``[0, 1]``, bright high.
        radius: How far across the surface each ray travels, in pixels.
        rays: Directions traced, spaced evenly around the circle.
        steps: Samples taken along each ray.
        relief: Pixels the full 0 to 1 height range stands above the base plane.
        bias: Slope, as a rise over a run, taken off every horizon before it counts.

    Returns:
        A ``(height, width)`` float32 tensor. 0 is a pixel the open sky reaches from every
        direction and 1 is one walled in on all of them. A ray leaving the frame reads the
        edge height from there on.
    """
    field = height_field.to(torch.float32) * relief
    rows, columns = field.shape
    plane = field.unsqueeze(0).unsqueeze(0)
    grid = _sample_grid(rows, columns, field.device)
    shifted = torch.empty_like(grid)
    offset = torch.zeros(2, dtype=torch.float32, device=field.device)
    total = torch.zeros_like(field)
    highest = torch.empty_like(field)

    for ray in range(rays):
        angle = 2.0 * math.pi * ray / rays
        across, down = math.cos(angle), math.sin(angle)
        highest.zero_()
        for step in range(1, steps + 1):
            reach = radius * step / steps
            offset[0] = 2.0 * across * reach / columns
            offset[1] = 2.0 * down * reach / rows
            torch.add(grid, offset, out=shifted)
            sampled = functional.grid_sample(plane, shifted, mode="bilinear",
                                             padding_mode="border", align_corners=False)
            # The tangent of the elevation angle to this sample, kept if it is the steepest.
            torch.maximum(highest, (sampled[0, 0] - field) / reach, out=highest)
        if bias > 0.0:
            highest.sub_(bias).clamp_min_(0.0)
        # sin of the horizon angle is the share of that direction's sky the ridge covers.
        total += highest * torch.rsqrt(highest * highest + 1.0)

    return total / rays


def _sample_grid(rows: int, columns: int, device) -> torch.Tensor:
    """The identity grid a shifted ``grid_sample`` reads a plane through.

    Args:
        rows: Row count.
        columns: Column count.
        device: Device the grid is built on.

    Returns:
        A ``(1, rows, columns, 2)`` float32 tensor holding each pixel's own position on the
        ``align_corners=False`` scale, x before y.
    """
    across = (torch.arange(columns, dtype=torch.float32, device=device) * 2.0 + 1.0) / columns - 1.0
    down = (torch.arange(rows, dtype=torch.float32, device=device) * 2.0 + 1.0) / rows - 1.0
    grid = torch.empty((1, rows, columns, 2), dtype=torch.float32, device=device)
    grid[0, :, :, 0] = across.unsqueeze(0)
    grid[0, :, :, 1] = down.unsqueeze(1)
    return grid


def calculate_direct_occlusion_factor(rgb_normalized, depth_normalized, height: int,
                                      width: int, radius: float):
    """Estimate direct occlusion as brightness: bright where occluded, dark where open.

    Args:
        rgb_normalized: ``(height, width, 3)`` array or tensor scaled to ``[0, 1]``.
        depth_normalized: ``(height, width, planes)`` array or tensor scaled to ``[0, 1]``.
            Only plane 0 is read.
        height: Row count.
        width: Column count.
        radius: Neighbourhood half-width in pixels.

    Returns:
        A ``(height, width)`` uint8 array for an array argument and a uint8 tensor for a
        tensor argument, contrast-stretched to a darkest pixel of 0 and a brightest of 255.
        A pixel whose combined depth and colour difference exceeds 1.0 is more occluded
        than the scale holds and reaches the stretch fully white; summing three colour
        channels reaches that on ordinary input. When every pixel carries the same
        occlusion value there is no range to stretch and the array comes back black.
    """
    factor = _factors(rgb_normalized, depth_normalized[:, :, 0], height, width, radius)
    codes = _codes(factor * 255)

    lowest = int(codes.min())
    span = int(codes.max()) - lowest
    # A uniform field has no range to stretch, and every value in it is already its own
    # minimum, so dividing by 1 leaves the black the numerator holds. The divisor is a
    # float64 tensor, which divides rather than multiplying by a reciprocal.
    divisor = torch.tensor(float(span or 1), dtype=torch.float64, device=codes.device)
    stretched = _codes((codes - lowest) / divisor * 255)
    return _answer(stretched, rgb_normalized)


def grey_codes(rgb: torch.Tensor) -> torch.Tensor:
    """Flatten colour picture codes to one grey plane.

    Args:
        rgb: ``(..., 3)`` int64 picture codes.

    Returns:
        Int64 picture codes without the channel axis.
    """
    red, green, blue = GREY_WEIGHTS
    return (rgb[..., 0] * red + rgb[..., 1] * green + rgb[..., 2] * blue + 32768) >> 16


def blurred_codes(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Blur picture codes with a run of box filters standing in for a Gaussian kernel.

    Args:
        plane: ``(height, width)`` int64 picture codes.
        radius: Standard deviation of the kernel the run approximates, in pixels.

    Returns:
        Int64 picture codes of the same shape. A radius of 0 answers the plane unchanged.
    """
    if radius <= 0:
        return plane
    box = _box_width(radius)
    if box <= 0:
        return plane
    for _ in range(BLUR_PASSES):
        plane = _box_pass(plane, box)
    plane = plane.T.contiguous()
    for _ in range(BLUR_PASSES):
        plane = _box_pass(plane, box)
    return plane.T.contiguous()


def smoothed_codes(plane: torch.Tensor) -> torch.Tensor:
    """Soften picture codes with the 5x5 smoothing kernel.

    Args:
        plane: ``(height, width)`` int64 picture codes.

    Returns:
        Int64 picture codes of the same shape. The two-pixel frame the kernel cannot reach
        into carries the samples it arrived with, and a plane under five pixels on either
        side comes back unchanged.
    """
    height, width = plane.shape
    if height < 5 or width < 5:
        return plane
    values = plane.to(torch.float32)
    scale = torch.tensor(SMOOTH_SCALE, dtype=torch.float32, device=plane.device)
    weights = [torch.tensor(weight, dtype=torch.float32, device=plane.device) / scale
               for weight in SMOOTH_WEIGHTS]
    # The half added here is what turns the truncation below into a rounding.
    total = torch.full((height - 4, width - 4), 0.5, dtype=torch.float32, device=plane.device)
    for band, row in enumerate((4, 3, 2, 1, 0)):
        line = values[row:height - 4 + row, 0:width - 4] * weights[band * 5]
        for column in range(1, 5):
            line = line + values[row:height - 4 + row, column:width - 4 + column] * weights[band * 5 + column]
        total = total + line
    filtered = plane.clone()
    filtered[2:height - 2, 2:width - 2] = total.clamp(0.0, 255.0).to(torch.int64)
    return filtered


def darkened_codes(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fade colour picture codes towards black wherever a mask is bright.

    Args:
        rgb: ``(height, width, channels)`` int64 picture codes.
        mask: ``(height, width)`` int64 picture codes. 255 blacks a pixel out and 0 leaves
            it alone.

    Returns:
        Int64 picture codes shaped like ``rgb``.
    """
    return _over_255(rgb * (255 - mask.unsqueeze(-1)))


def screened_codes(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Combine two sets of picture codes so neither can darken the other.

    Args:
        first: Int64 picture codes.
        second: Int64 picture codes of a broadcastable shape.

    Returns:
        Int64 picture codes, never darker than either argument.
    """
    return 255 - ((255 - first) * (255 - second)) // 255


def stretched_codes(rgb: torch.Tensor, cutoff: tuple[float, float]) -> torch.Tensor:
    """Pull each channel's darkest and brightest surviving level out to black and white.

    Args:
        rgb: ``(height, width, channels)`` int64 picture codes. Each channel is measured
            and stretched on its own.
        cutoff: Percentages of the samples discarded from the dark end and the bright end
            before the surviving range is measured.

    Returns:
        Int64 picture codes shaped like ``rgb``. A channel whose surviving range is a
        single level comes back untouched.
    """
    levels = torch.arange(256, device=rgb.device)
    stretched = torch.empty_like(rgb)
    for channel in range(rgb.shape[-1]):
        plane = rgb[..., channel]
        histogram = torch.bincount(plane.flatten(), minlength=256)
        total = int(histogram.sum())
        rising = torch.cumsum(histogram, 0)
        falling = torch.flip(torch.cumsum(torch.flip(histogram, (0,)), 0), (0,))
        alive = histogram > 0
        low = alive & (rising > int(total * cutoff[0] // 100))
        high = alive & (falling > int(total * cutoff[1] // 100))
        darkest = int(levels[low][0]) if bool(low.any()) else 255
        brightest = int(levels[high][-1]) if bool(high.any()) else 0
        if brightest <= darkest:
            stretched[..., channel] = plane
            continue
        scale = 255.0 / (brightest - darkest)
        offset = -darkest * scale
        table = (levels.to(torch.float64) * scale + offset).to(torch.int64).clamp(0, 255)
        stretched[..., channel] = table[plane]
    return stretched


def _factors(rgb_normalized, depth_normalized, height: int, width: int,
             radius: float) -> torch.Tensor:
    """The occlusion factor of every pixel.

    Args:
        rgb_normalized: ``(height, width, channels)`` array or tensor scaled to ``[0, 1]``.
        depth_normalized: ``(height, width)`` or ``(height, width, planes)`` array or
            tensor scaled to ``[0, 1]``. The depth term is a mean over the planes as well
            as over the neighbourhood.
        height: Row count.
        width: Column count.
        radius: Neighbourhood half-width in pixels.

    Returns:
        A ``(height, width)`` float64 tensor. 0 is a pixel with no occluder near it. Depth
        is measured from each neighbour towards the centre pixel.
    """
    device = _device()
    rgb = torch.as_tensor(rgb_normalized, dtype=torch.float64, device=device)
    depth = torch.as_tensor(depth_normalized, dtype=torch.float64, device=device)
    if depth.dim() == 2:
        depth = depth.unsqueeze(2)

    span = _offsets(radius)
    reach = len(span)
    before, after = -span.start, span.stop - 1
    rgb_window = _windows(rgb, before, after, reach)
    depth_window = _windows(depth, before, after, reach)
    inside = _inside(width, span, device)

    depth_total = torch.zeros((height, width), dtype=torch.float64, device=device)
    colour_total = torch.zeros((height, width), dtype=torch.float64, device=device)
    widest = max(rgb.shape[2], depth.shape[2])
    step = max(1, PASS_BYTES // max(1, width * widest * reach * 8))

    for down in span:
        top = max(0, -down)
        bottom = min(height, height - down)
        for start in range(top, bottom, step):
            stop = min(start + step, bottom)
            centre = depth[start:stop].unsqueeze(3)
            neighbour = depth_window[start + down:stop + down]
            gap = neighbour - centre
            depth_total[start:stop] += gap.clamp_min_(0).mul_(inside).sum((2, 3))
            centre = rgb[start:stop].unsqueeze(3)
            neighbour = rgb_window[start + down:stop + down]
            colour_total[start:stop] += (centre - neighbour).abs_().mul_(inside).sum((2, 3))

    rows = _counts(height, span, device)
    columns = _counts(width, span, device)
    count = rows.unsqueeze(1) * columns.unsqueeze(0)
    return depth_total / (count * depth.shape[2]) + colour_total / count


def _windows(planes: torch.Tensor, before: int, after: int, reach: int) -> torch.Tensor:
    """Every horizontal neighbourhood of a plane stack, as one strided view.

    Args:
        planes: ``(height, width, channels)`` float64 tensor.
        before: Columns the neighbourhood reaches to the left.
        after: Columns it reaches to the right.
        reach: Columns it spans, ``before + after + 1``.

    Returns:
        A ``(height, width, channels, reach)`` view whose last axis runs from the leftmost
        neighbour to the rightmost. Samples off the edge read as 0 and are dropped by the
        mask :func:`_inside` builds.
    """
    height, width, channels = planes.shape
    padded = planes.new_zeros((height, width + before + after, channels))
    padded[:, before:before + width] = planes
    return padded.unfold(1, reach, 1)


def _inside(width: int, span: range, device) -> torch.Tensor:
    """Which horizontal neighbours of each column fall inside the image.

    Args:
        width: Column count.
        span: Offsets the neighbourhood covers, from :func:`_offsets`.
        device: Device the mask is built on.

    Returns:
        A ``(1, width, 1, len(span))`` float64 tensor of 1 and 0, shaped to multiply a
        window stack.
    """
    column = torch.arange(width, device=device).unsqueeze(1)
    shift = torch.arange(span.start, span.stop, device=device).unsqueeze(0)
    reached = column + shift
    mask = (reached >= 0) & (reached < width)
    return mask.to(torch.float64).unsqueeze(0).unsqueeze(2)


def _offsets(radius: float) -> range:
    """The offsets from a centre pixel that one axis of its neighbourhood covers.

    Args:
        radius: Neighbourhood half-width in pixels.

    Returns:
        ``range(-ceil(radius), floor(radius) + 1)``: symmetric about the centre for a whole
        radius, and one pixel further towards the low side for a fractional one. A radius of
        0 covers the centre pixel alone.
    """
    return range(-math.ceil(radius), math.floor(radius) + 1)


def _counts(length: int, offsets: range, device) -> torch.Tensor:
    """How many neighbours each position along one axis has inside the image.

    Args:
        length: Extent of the axis in pixels.
        offsets: Offsets the neighbourhood covers, from :func:`_offsets`.
        device: Device the vector is built on.

    Returns:
        A float64 tensor of ``length`` counts, each ``offsets`` clipped to the axis. The
        product of the two axes' counts is the neighbourhood size a mean divides by.
    """
    index = torch.arange(length, device=device)
    first = (index + offsets.start).clamp_min(0)
    last = (index + offsets.stop - 1).clamp_max(length - 1)
    return (last - first + 1).to(torch.float64)


def _codes(values: torch.Tensor) -> torch.Tensor:
    """Truncate occlusion values towards zero into picture codes.

    Args:
        values: Float tensor of occlusion values.

    Returns:
        An int64 tensor of the same shape. A value outside 0-255 saturates at the end it
        passed, so an occlusion stronger than the scale holds reads as the strongest the
        scale has rather than as a value picture codes cannot store.
    """
    return values.to(torch.int64).clamp(0, 255)


def _answer(codes: torch.Tensor, given):
    """Picture codes on the side of the torch boundary the caller works on.

    Args:
        codes: Int64 picture codes.
        given: The argument the caller passed, which decides the answer's kind.

    Returns:
        A uint8 tensor where ``given`` is a tensor, and a uint8 numpy array otherwise.
    """
    narrow = codes.to(torch.uint8)
    return narrow if isinstance(given, torch.Tensor) else narrow.cpu().numpy()


def _box_pass(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """One horizontal box-filter pass over picture codes.

    Args:
        plane: ``(rows, columns)`` int64 picture codes.
        radius: Box half-width in pixels. The fractional part weights the two samples just
            outside the whole box.

    Returns:
        Int64 picture codes of the same shape. Samples off the ends of a row read as the
        end sample.
    """
    whole = int(radius)
    weight = int((1 << 24) / (radius * 2 + 1))
    edge = ((1 << 24) - (whole * 2 + 1) * weight) // 2
    rows, width = plane.shape
    reach = torch.arange(-whole - 1, width + whole + 1, device=plane.device)
    padded = plane[:, reach.clamp(0, width - 1)]
    running = torch.cat(
        [plane.new_zeros((rows, 1)), torch.cumsum(padded, dim=1)], dim=1
    )
    far = 2 * whole + 2
    inner = running[:, far:far + width] - running[:, 1:1 + width]
    bulk = inner * weight + (padded[:, 0:width] + padded[:, far:far + width]) * edge
    return (bulk + (1 << 23)) >> 24


def _over_255(product: torch.Tensor) -> torch.Tensor:
    """Divide a product of two picture codes by 255, rounding to nearest.

    Args:
        product: Int64 tensor of products in 0 to 65025.

    Returns:
        An int64 tensor of the same shape.
    """
    carried = product + 128
    return ((carried >> 8) + carried) >> 8


def _device():
    """The device the occlusion kernels run on.

    Returns:
        A ``torch.device``: ComfyUI's compute device where it carries float64, and the CPU
        where it does not. MPS is one device without it.
    """
    from ..model import compute_device

    device = compute_device()
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
    except (RuntimeError, TypeError) as error:
        logger.debug("%s has no float64, so this computes on the CPU: %s", device, error)
        return torch.device("cpu")
    return device
