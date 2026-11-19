"""Lens blur from a depth map: circles of confusion rendered as depth layers in linear light.

Frames are ``(height, width, 3)`` floats, depth maps and mattes ``(height, width)`` on 0 to 1
with 1 nearest. Diameters are in pixels.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch.nn import functional

from . import raw

__all__ = [
    "APERTURES",
    "FOCUS_SOURCES",
    "NEAR_ENDS",
    "Settings",
    "Split",
    "aperture_kernel",
    "canvas_margin",
    "depth_plane",
    "depth_planes",
    "diameter_grid",
    "fill",
    "focus_depth_at",
    "matte_planes",
    "render",
    "split_edges",
    "split_matte",
    "split_radius",
    "steadied",
    "whole",
]

#: Aperture menu entry -> blade count, 0 for a round opening.
APERTURES = {
    "circle": 0,
    "5 blades": 5,
    "6 blades": 6,
    "7 blades": 7,
    "8 blades": 8,
    "9 blades": 9,
}

#: Which end of a depth map is nearest the camera, in menu order.
NEAR_ENDS = ("bright", "dark")

#: Where the plane of focus comes from, in menu order.
FOCUS_SOURCES = ("picked point", "focus depth")

#: Diameters under this many pixels are drawn sharp.
SHARP_DIAMETER = 1.0

#: Layer diameters step by half a pixel up to :data:`LINEAR_TOP`, then by :data:`RATIO`.
LINEAR_STEP = 0.5
LINEAR_TOP = 6.0
RATIO = 1.07

#: A layer is blurred at a reduced size once its kernel stays this wide there.
REDUCED_DIAMETER = 24.0

#: Largest factor a layer is reduced by.
MAX_REDUCTION = 32

#: The padded canvas is a multiple of this many pixels on each side.
CANVAS_MULTIPLE = 64

#: Highlight lift: picture code where it starts, and stops gained by a clipped pixel at full bloom.
BLOOM_KNEE = 0.92
BLOOM_STOPS = 4.0

#: Power the highlight lift rises by between the knee and white.
BLOOM_POWER = 3.0

#: How far a light stands above the mean of its surroundings, in picture codes, for the full lift.
BLOOM_STANDOUT = 0.5

#: Diameter over which a pixel's highlight lift ramps in from nothing.
BLOOM_RAMP = 3.0

#: Share of the short side the depth under the focus point is read over.
FOCUS_WINDOW = 0.01

#: Edge splitting: blur spread in pixels where a window starts and finishes counting as an edge.
SPLIT_LOW = 2.0
SPLIT_HIGH = 4.0

#: Edge splitting: squared colour distance two sides need before a pixel leans to one of them.
SPLIT_CONTRAST = 0.0075

#: Edge splitting: how hard a pixel leans to the side its colour matches.
SPLIT_GAIN = 3.0

#: Edge splitting reaches full strength this many times the share of the step a pixel sits from
#: either side.
SPLIT_ENDS = 50.0

#: Share of the step from either side inside which a pixel counts as on that side.
SPLIT_FLOOR = 0.01

#: Window radius of edge splitting as a share of the short side, and its bounds in pixels.
SPLIT_SHARE = 1.0 / 128.0
SPLIT_RADIUS = (2, 16)

#: A matte's sides trust their own depth from this many splitting radii inside the edge.
CORE_RADII = 3

#: Half the window the depth map's median is taken over, in pixels.
MEDIAN_RADIUS = 2

#: Largest change the median makes to a depth; a pixel it would move further keeps its own.
MEDIAN_LIMIT = 0.08

#: Values one band of the depth median gathers at once.
MEDIAN_BAND = 1 << 24

#: Matte levels from which a pixel counts as certainly subject, or certainly not.
SUBJECT_LEVEL = 0.98
BACKGROUND_LEVEL = 0.02

#: Below this a weight is read as nothing.
EPSILON = 1e-6

#: Pixels a layer is carried behind its neighbours past half the widest nearer blur.
EXTEND_MARGIN = 2.0

#: Steps a reach is measured over at the size it is worked out at.
REACH_STEPS = 8

#: Pixels a layer's working region reaches past what its blur and its carry need.
REGION_MARGIN = 8

#: Share of the window round a pixel a layer fills to be carried behind its neighbours in full.
EXTEND_SHARE = 0.02

#: Share of a pixel a layer holds from which that pixel carries on behind in its own colour.
OWN_SHARE = 0.05

#: Mean spread between a depth map's channels past which it is a colour picture.
COLOUR_SPREAD = 0.03

#: Share by which a map's shape may differ from the frames' and still be resized onto them.
SHAPE_TOLERANCE = 0.02

#: How far outside 0 to 1 a depth map may reach before it is stretched over the batch.
RANGE_TOLERANCE = 1e-3

#: Least spread a depth map holds, on 0 to 1, below which it is flat.
FLAT = 1.0 / 255.0


class Settings(NamedTuple):
    """What one frame is rendered with.

    Attributes:
        focus: Depth held sharp, 0 farthest to 1 nearest.
        largest: Blur diameter in pixels a full depth range from the focus reaches.
        focus_range: Depth either side of the focus, summed, that stays sharp.
        blades: Aperture blade count, 0 for a circle.
        rotation: Aperture rotation in degrees.
        bloom: Highlight bloom, 0 to 1.
    """

    focus: float
    largest: float
    focus_range: float
    blades: int
    rotation: float
    bloom: float


def diameter_grid(largest: float) -> list[float]:
    """The blur diameters layers are drawn at, smallest first.

    Args:
        largest: The widest diameter any pixel reaches, in pixels.

    Returns:
        Ascending diameters from :data:`SHARP_DIAMETER` to at least ``largest``.
    """
    grid = []
    value = SHARP_DIAMETER
    while value < LINEAR_TOP and value <= largest + LINEAR_STEP:
        grid.append(value)
        value += LINEAR_STEP
    if value >= LINEAR_TOP:
        value = LINEAR_TOP
        while True:
            grid.append(value)
            if value >= largest:
                break
            value *= RATIO
    return grid or [SHARP_DIAMETER]


def aperture_kernel(diameter: float, blades: int = 0, rotation: float = 0.0, device=None) -> torch.Tensor:
    """The picture a point of light becomes, normalised to sum to 1.

    Args:
        diameter: Width of the opening in pixels; a polygon's corners sit on this circle.
        blades: Blade count, 0 for a circle.
        rotation: Turn of the blades in degrees, 0 putting one flat edge at the bottom.
        device: Device the kernel is built on.

    Returns:
        A square ``(size, size)`` float32 tensor with an odd side.
    """
    radius = max(float(diameter), SHARP_DIAMETER) / 2.0
    half = int(math.ceil(radius + 1.0))
    steps = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    rows, columns = torch.meshgrid(steps, steps, indexing="ij")
    if blades < 3:
        distance = torch.hypot(rows, columns) - radius
    else:
        apothem = radius * math.cos(math.pi / blades)
        turn = math.radians(float(rotation))
        distance = torch.full_like(rows, -float("inf"))
        for blade in range(blades):
            angle = turn + math.pi / 2.0 + 2.0 * math.pi * blade / blades
            reach = columns * math.cos(angle) + rows * math.sin(angle) - apothem
            distance = torch.maximum(distance, reach)
    cover = (0.5 - distance).clamp(0.0, 1.0)
    return cover / cover.sum()


def depth_plane(depth: torch.Tensor, near_end: str) -> torch.Tensor:
    """A depth map read with 1 nearest.

    Args:
        depth: ``(height, width)`` on 0 to 1.
        near_end: A value of :data:`NEAR_ENDS`.

    Returns:
        The map, turned over where dark is near.
    """
    return 1.0 - depth if near_end == NEAR_ENDS[1] else depth


def focus_depth_at(depth: torch.Tensor, x: float, y: float, matte=None) -> float:
    """The depth under a point, as the median of a small window round it.

    Args:
        depth: ``(height, width)`` with 1 nearest.
        x: Share of the width, 0 left to 1 right.
        y: Share of the height, 0 top to 1 bottom.
        matte: Optional subject matte; a point on the subject reads only the subject's pixels.

    Returns:
        The median depth, 0 to 1.
    """
    height, width = int(depth.shape[0]), int(depth.shape[1])
    column = min(width - 1, max(0, int(float(x) * width)))
    row = min(height - 1, max(0, int(float(y) * height)))
    reach = max(2, int(round(FOCUS_WINDOW * min(height, width))))
    rows = slice(max(0, row - reach), row + reach + 1)
    columns = slice(max(0, column - reach), column + reach + 1)
    window = depth[rows, columns].reshape(-1)
    if matte is not None:
        side = matte[rows, columns].reshape(-1) >= 0.5
        if bool(matte[row, column] < 0.5):
            side = ~side
        if bool(side.any()):
            window = window[side]
    return float(window.median())


def depth_planes(depth, count: int, height: int, width: int, node: str) -> torch.Tensor:
    """A depth input as one grey plane per frame, at the frames' size, on 0 to 1.

    Args:
        depth: An ``IMAGE`` ``(maps, height, width, channels)`` or a ``MASK`` of two or three axes.
        count: Frames the planes are for.
        height: Frame height.
        width: Frame width.
        node: Display name of the calling node.

    Returns:
        ``(count, height, width)`` float32 on the CPU, stretched to 0 to 1 over the batch where
        the map runs past that range.

    Raises:
        ValueError: The map is in colour, of another shape, of another length or flat.
    """
    if depth is None or not hasattr(depth, "ndim") or depth.numel() == 0:
        raise ValueError(f"{node} needs a depth map on `depth`. Wire Power Preprocessor's depth_map into it.")
    held = depth.detach().to(device="cpu", dtype=torch.float32)
    # A mask batch may arrive as (maps, 1, height, width).
    if held.ndim == 4 and int(held.shape[1]) == 1 and int(held.shape[-1]) > 4:
        held = held[:, 0]
    if held.ndim == 4:
        channels = int(held.shape[-1])
        if channels >= 3:
            colour = held[..., :3]
            spread = float((colour.amax(-1) - colour.amin(-1)).mean())
            if spread > COLOUR_SPREAD:
                raise ValueError(
                    f"{node} was handed a depth map whose three channels differ by {spread:.3f} on "
                    f"average, so it is a colour picture rather than a grey depth map. Wire the grey "
                    f"depth map from Power Preprocessor (depth_map) or Depth Anything, not the picture "
                    f"itself or a coloured visualisation of depth."
                )
            held = colour[..., 0] if spread == 0.0 else colour.mean(-1)
        else:
            held = held[..., 0]
    elif held.ndim == 2:
        held = held[None]
    elif held.ndim != 3:
        raise ValueError(f"{node} cannot read a depth map shaped {tuple(held.shape)}; wire an IMAGE or a MASK.")
    maps = int(held.shape[0])
    if maps not in (1, count):
        raise ValueError(
            f"{node} was handed {maps} depth maps for {count} pictures. Wire one depth map for every "
            f"picture, or a single map to use for all of them."
        )
    tall, wide = int(held.shape[1]), int(held.shape[2])
    if abs(tall * width - wide * height) > SHAPE_TOLERANCE * height * wide:
        raise ValueError(
            f"{node} was handed a {wide}x{tall} depth map for {width}x{height} pictures, which is a "
            f"different shape. Make the depth map from these pictures, or crop both alike."
        )
    low, high = float(held.amin()), float(held.amax())
    if low < -RANGE_TOLERANCE or high > 1.0 + RANGE_TOLERANCE:
        held = (held - low) / max(high - low, EPSILON)
    if (tall, wide) != (height, width):
        held = functional.interpolate(held[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    held = held.clamp(0.0, 1.0)
    for index in range(maps):
        plane = held[index]
        if float(plane.amax() - plane.amin()) < FLAT:
            raise ValueError(
                f"{node} was handed a flat depth map (frame {index}): every value is within "
                f"{FLAT:.4f} of {float(plane.mean()):.3f}, so nothing is nearer or farther than anything "
                f"else. Wire a real depth map from Power Preprocessor (depth_map) or Depth Anything."
            )
    if maps == 1 and count > 1:
        held = held.expand(count, height, width)
    return held


def matte_planes(matte, count: int, height: int, width: int, node: str):
    """A subject matte as one plane per frame, at the frames' size.

    Args:
        matte: A ``MASK`` of two or three axes, or None.
        count: Frames the planes are for.
        height: Frame height.
        width: Frame width.
        node: Display name of the calling node.

    Returns:
        ``(count, height, width)`` float32 on the CPU on 0 to 1, or None.

    Raises:
        ValueError: The matte is of another shape or another length.
    """
    if matte is None:
        return None
    held = matte.detach().to(device="cpu", dtype=torch.float32)
    if held.ndim == 2:
        held = held[None]
    if held.ndim == 4:
        held = held[..., 0] if int(held.shape[-1]) == 1 else held[:, 0]
    maps = int(held.shape[0])
    if maps not in (1, count):
        raise ValueError(
            f"{node} was handed {maps} subject mattes for {count} pictures. Wire one matte for every "
            f"picture, or a single matte to use for all of them."
        )
    tall, wide = int(held.shape[1]), int(held.shape[2])
    if abs(tall * width - wide * height) > SHAPE_TOLERANCE * height * wide:
        raise ValueError(
            f"{node} was handed a {wide}x{tall} subject_mask for {width}x{height} pictures, which is a "
            f"different shape. Make the matte from these pictures."
        )
    if (tall, wide) != (height, width):
        held = functional.interpolate(held[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    held = held.clamp(0.0, 1.0)
    if maps == 1 and count > 1:
        held = held.expand(count, height, width)
    return held


def _box(values: torch.Tensor, radius: int) -> torch.Tensor:
    """The mean over each ``2 * radius + 1`` window, edges counting only what exists."""
    size = 2 * radius + 1
    across = functional.avg_pool2d(values, (1, size), 1, (0, radius), count_include_pad=False)
    return functional.avg_pool2d(across, (size, 1), 1, (radius, 0), count_include_pad=False)


def _window_max(values: torch.Tensor, radius: int) -> torch.Tensor:
    """The largest value in each ``2 * radius + 1`` window."""
    size = 2 * radius + 1
    across = functional.max_pool2d(values, (1, size), 1, (0, radius))
    return functional.max_pool2d(across, (size, 1), 1, (radius, 0))


def _smoothstep(low: float, high: float, values: torch.Tensor) -> torch.Tensor:
    """0 below ``low``, 1 above ``high``, a smooth ramp between."""
    t = ((values - low) / max(high - low, EPSILON)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def split_radius(height: int, width: int) -> int:
    """The window edges are split over and matte cores are measured with.

    Args:
        height: Frame height.
        width: Frame width.

    Returns:
        A radius in pixels.
    """
    low, high = SPLIT_RADIUS
    return int(min(high, max(low, round(min(height, width) * SPLIT_SHARE))))


def steadied(depth: torch.Tensor, radius: int = MEDIAN_RADIUS) -> torch.Tensor:
    """A depth map with pixel-scale ripples taken out by a median, edges kept where they are.

    Args:
        depth: ``(height, width)``.
        radius: Half the median window.

    Returns:
        ``(height, width)``, each pixel the median of its window where that moves it by under
        :data:`MEDIAN_LIMIT`, and unchanged elsewhere.
    """
    size = 2 * radius + 1
    height, width = int(depth.shape[0]), int(depth.shape[1])
    padded = functional.pad(depth[None, None], (radius, radius, radius, radius), mode="replicate")[0, 0]
    rows = max(1, MEDIAN_BAND // max(1, width * size * size))
    answer = torch.empty_like(depth)
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        band = padded[top: bottom + 2 * radius]
        windows = band.unfold(0, size, 1).unfold(1, size, 1)
        answer[top:bottom] = windows.reshape(bottom - top, width, size * size).median(dim=-1).values
    return torch.where((answer - depth).abs() < MEDIAN_LIMIT, answer, depth)


class Split(NamedTuple):
    """A frame shared between a near side and a far side, pixel by pixel.

    Attributes:
        share: ``(height, width)`` of each pixel on the near side, 0 to 1.
        near_colour: ``(height, width, 3)`` light-linear, premultiplied by ``share``.
        far_colour: ``(height, width, 3)`` light-linear, premultiplied by ``1 - share``.
        near_depth: ``(height, width)`` depth of the near side.
        far_depth: ``(height, width)`` depth of the far side.
    """

    share: torch.Tensor
    near_colour: torch.Tensor
    far_colour: torch.Tensor
    near_depth: torch.Tensor
    far_depth: torch.Tensor


def whole(light: torch.Tensor, depth: torch.Tensor) -> Split:
    """A frame with every pixel wholly on one side, at its own depth.

    Args:
        light: ``(height, width, 3)`` light-linear.
        depth: ``(height, width)``.

    Returns:
        A :class:`Split`.
    """
    return Split(torch.ones_like(depth), light, torch.zeros_like(light), depth, depth)


def split_edges(depth: torch.Tensor, codes: torch.Tensor, light: torch.Tensor, spread: float, radius: int) -> Split:
    """Share the pixels of a soft depth edge between its two sides by their colour.

    Args:
        depth: ``(height, width)`` with 1 nearest.
        codes: ``(height, width, 3)`` picture codes.
        light: The same frame light-linear.
        spread: Blur diameter in pixels one full unit of depth makes.
        radius: Half the window an edge is looked for in.

    Returns:
        A :class:`Split`, every pixel wholly near at its own depth wherever the blur across a
        window varies by under :data:`SPLIT_LOW` pixels or the pixel already sits at one side.
    """
    planes = depth[None, None]
    high = _window_max(planes, radius)
    low = -_window_max(-planes, radius)
    zone = _smoothstep(SPLIT_LOW, SPLIT_HIGH, (high - low) * spread)
    if float(zone.amax()) <= 0.0:
        return whole(light, depth)
    # Each pixel is sided against the midpoint its neighbourhood shares, not its own.
    upper = (planes > _box((high + low) * 0.5, radius)).to(planes.dtype)
    lower = 1.0 - upper
    colour = codes.permute(2, 0, 1)[None].clamp(0.0, 1.0)
    upper_share = _box(upper, radius)
    lower_share = _box(lower, radius)
    upper_colour = _box(colour * upper, radius) / upper_share.clamp_min(EPSILON)
    lower_colour = _box(colour * lower, radius) / lower_share.clamp_min(EPSILON)
    upper_depth = _box(planes * upper, radius) / upper_share.clamp_min(EPSILON)
    lower_depth = _box(planes * lower, radius) / lower_share.clamp_min(EPSILON)
    # Where the colour falls on the line from the far side's colour to the near side's.
    axis = upper_colour - lower_colour
    contrast = axis.square().sum(1, keepdim=True)
    reach = ((colour - lower_colour) * axis).sum(1, keepdim=True)
    alpha = (0.5 + SPLIT_GAIN * (reach - 0.5 * contrast) / (contrast + SPLIT_CONTRAST)).clamp(0.0, 1.0)
    alpha = torch.where(upper_share < EPSILON, torch.zeros_like(alpha), alpha)
    alpha = torch.where(lower_share < EPSILON, torch.ones_like(alpha), alpha)
    # A pixel already at one side's depth is left where it is.
    place = ((planes - lower_depth) / (upper_depth - lower_depth).clamp_min(EPSILON)).clamp(0.0, 1.0)
    between = ((torch.minimum(place, 1.0 - place) - SPLIT_FLOOR) * SPLIT_ENDS).clamp(0.0, 1.0)
    strength = (zone * between)[0, 0]
    share = torch.lerp(torch.ones_like(depth), alpha[0, 0], strength)
    front = raw.linearise(upper_colour[0].permute(1, 2, 0)[None])[0]
    back = raw.linearise(lower_colour[0].permute(1, 2, 0)[None])[0]
    held = share[..., None]
    # What the two side colours leave unexplained goes to both sides by their shares.
    rest = light - (held * front + (1.0 - held) * back)
    return Split(
        share,
        (held * (front + rest)).clamp(min=0.0),
        ((1.0 - held) * (back + rest)).clamp(min=0.0),
        torch.lerp(depth, high[0, 0], strength),
        torch.lerp(depth, low[0, 0], strength),
    )


def fill(colour: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Extend premultiplied colour into the places its weight leaves empty.

    Args:
        colour: ``(channels, height, width)`` premultiplied by ``weight``.
        weight: ``(1, height, width)`` on 0 to 1.

    Returns:
        ``(channels, height, width)`` at full weight everywhere any weight exists.
    """
    levels = []
    held, cover = colour[None], weight[None].clamp(0.0, 1.0)
    while max(int(held.shape[-2]), int(held.shape[-1])) > 1:
        levels.append((held, cover))
        held = functional.avg_pool2d(held, 2, ceil_mode=True)
        cover = functional.avg_pool2d(cover, 2, ceil_mode=True)
    estimate = held / cover.clamp_min(EPSILON)
    estimate = torch.where(cover > EPSILON, estimate, torch.zeros_like(estimate))
    for held, cover in reversed(levels):
        grown = functional.interpolate(estimate, size=held.shape[-2:], mode="bilinear", align_corners=False)
        estimate = held + (1.0 - cover) * grown
    return estimate[0]


def _reduction(diameter: float) -> int:
    """The factor a layer of this diameter is blurred at a reduced size by."""
    factor = 1
    while factor * 2 <= MAX_REDUCTION and diameter / (factor * 2) >= REDUCED_DIAMETER:
        factor *= 2
    return factor


def canvas_margin(diameter: float) -> int:
    """Pixels a frame is grown by on each side to blur at a diameter with no wrap.

    Args:
        diameter: The widest layer diameter, in pixels.

    Returns:
        The margin in pixels.
    """
    return int(math.ceil(diameter / 2.0)) + 2 * _reduction(diameter) + 2


def _convolve(planes: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Circular convolution of every plane with one centred kernel.

    Args:
        planes: ``(channels, height, width)``.
        kernel: ``(size, size)`` with an odd side no longer than either side of ``planes``.

    Returns:
        ``(channels, height, width)``.
    """
    height, width = int(planes.shape[-2]), int(planes.shape[-1])
    size = int(kernel.shape[-1])
    centre = size // 2
    placed = planes.new_zeros(height, width)
    placed[:size, :size] = kernel
    placed = torch.roll(placed, shifts=(-centre, -centre), dims=(0, 1))
    spectrum = torch.fft.rfft2(planes) * torch.fft.rfft2(placed)
    return torch.fft.irfft2(spectrum, s=(height, width))


def _blur_layer(planes: torch.Tensor, diameter: float, blades: int, rotation: float) -> torch.Tensor:
    """One layer's colour and coverage spread through the aperture.

    Args:
        planes: ``(4, height, width)``, premultiplied colour then coverage, both sides
            multiples of :data:`MAX_REDUCTION`.
        diameter: Blur diameter in pixels.
        blades: Blade count, 0 for a circle.
        rotation: Blade turn in degrees.

    Returns:
        ``(4, height, width)``, colour held at or above 0 and coverage to 0 to 1.
    """
    factor = _reduction(diameter)
    working = planes
    if factor > 1:
        working = functional.avg_pool2d(planes[None], factor)[0]
    kernel = aperture_kernel(diameter / factor, blades, rotation, device=planes.device)
    spread = _convolve(working, kernel)
    if factor > 1:
        spread = functional.interpolate(spread[None], scale_factor=factor, mode="bilinear", align_corners=False)[0]
    spread[:3].clamp_(min=0.0)
    spread[3:].clamp_(0.0, 1.0)
    return spread


def _region(coverage: torch.Tensor, grow: float):
    """The part of the canvas a layer reaches, aligned to :data:`MAX_REDUCTION`.

    Args:
        coverage: ``(height, width)`` of the layer.
        grow: Pixels to reach past the layer's own on every side.

    Returns:
        ``(rows, columns)`` as slices, or None for a layer with no coverage.
    """
    rows = torch.nonzero((coverage > EPSILON).any(dim=1)).flatten()
    if rows.numel() == 0:
        return None
    columns = torch.nonzero((coverage > EPSILON).any(dim=0)).flatten()
    reach = int(math.ceil(grow)) + REGION_MARGIN

    def span(found, length):
        low = max(0, int(found[0]) - reach) // MAX_REDUCTION * MAX_REDUCTION
        high = -(-min(length, int(found[-1]) + 1 + reach) // MAX_REDUCTION) * MAX_REDUCTION
        return slice(low, min(length, high))

    return span(rows, int(coverage.shape[0])), span(columns, int(coverage.shape[1]))


def _blurred_region(planes: torch.Tensor, diameter: float, blades: int, rotation: float) -> torch.Tensor:
    """One region of a layer blurred as if the canvas round it were empty.

    Args:
        planes: ``(4, height, width)``, both sides multiples of :data:`MAX_REDUCTION`.
        diameter: Blur diameter in pixels.
        blades: Blade count, 0 for a circle.
        rotation: Blade turn in degrees.

    Returns:
        ``(4, height, width)``, as :func:`_blur_layer` answers.
    """
    pad = -(-(canvas_margin(diameter)) // MAX_REDUCTION) * MAX_REDUCTION
    grown = functional.pad(planes, (pad, pad, pad, pad))
    return _blur_layer(grown, diameter, blades, rotation)[:, pad:-pad, pad:-pad]


def _signed_layers(depth: torch.Tensor, settings: Settings, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each pixel's blur diameter and its place between layers, negative behind the focus.

    Args:
        depth: ``(height, width)`` with 1 nearest.
        settings: The frame's settings.
        grid: Layer diameters from :func:`diameter_grid`.

    Returns:
        ``(diameters, places)``: float diameters in pixels, and float layer places where 0 is
        sharp and ``n`` is drawn at ``grid[n - 1]``; a fraction is shared between two layers.
    """
    offset = depth - settings.focus
    beyond = (offset.abs() - settings.focus_range * 0.5).clamp(min=0.0)
    diameters = (beyond * settings.largest).contiguous()
    steps = torch.cat([grid.new_tensor([0.0]), grid])
    below = (torch.bucketize(diameters, steps, right=True) - 1).clamp(0, int(steps.numel()) - 2)
    low, high = steps[below], steps[below + 1]
    places = below.to(diameters.dtype) + ((diameters - low) / (high - low)).clamp(0.0, 1.0)
    return diameters, torch.where(offset < 0, -places, places)


def _layer_share(places: torch.Tensor, layer: int) -> torch.Tensor:
    """How much of each pixel a layer holds, from the places :func:`_signed_layers` answers.

    Args:
        places: Signed float layer places.
        layer: A signed layer number.

    Returns:
        A tensor of the shape of ``places``, 1 on the layer falling to 0 one layer away.
    """
    return (1.0 - (places - float(layer)).abs()).clamp(min=0.0)


def _highlights(codes: torch.Tensor, radius: int) -> torch.Tensor:
    """How much each pixel reads as a light: near white and standing above its surroundings.

    Args:
        codes: ``(height, width, 3)`` picture codes.
        radius: Half the window the surroundings are averaged over, in pixels.

    Returns:
        ``(height, width)`` on 0 to 1.
    """
    peak = codes.amax(dim=-1)
    rise = ((peak - BLOOM_KNEE) / (1.0 - BLOOM_KNEE)).clamp(0.0, 1.0) ** BLOOM_POWER
    if float(rise.amax()) <= 0.0:
        return rise
    level = codes.clamp(0.0, 1.0).mean(dim=-1)[None, None]
    size = 2 * radius + 1
    around = functional.avg_pool2d(level, (1, size), 1, (0, radius), count_include_pad=False)
    around = functional.avg_pool2d(around, (size, 1), 1, (radius, 0), count_include_pad=False)[0, 0]
    standout = ((peak.clamp(max=1.0) - around) / BLOOM_STANDOUT).clamp(0.0, 1.0)
    return rise * standout


def _bloom(colour: torch.Tensor, lights: torch.Tensor, diameters: torch.Tensor, bloom: float) -> torch.Tensor:
    """Lift light sources by how far they will spread.

    Args:
        colour: ``(height, width, 3)`` light-linear.
        lights: ``(height, width)`` from :func:`_highlights`.
        diameters: ``(height, width)`` blur diameters in pixels.
        bloom: 0 to 1.

    Returns:
        ``colour`` scaled, unchanged wherever the diameter is under :data:`SHARP_DIAMETER`.
    """
    if bloom <= 0.0:
        return colour
    ramp = ((diameters - SHARP_DIAMETER) / BLOOM_RAMP).clamp(0.0, 1.0)
    gain = 1.0 + (2.0 ** (BLOOM_STOPS * bloom) - 1.0) * lights * ramp
    return colour * gain[..., None]


def split_matte(codes, light, depth, matte, radius) -> tuple[Split, torch.Tensor]:
    """Share a frame between a subject and its background along a matte.

    Args:
        codes: ``(height, width, 3)`` picture codes.
        light: ``(height, width, 3)`` light-linear.
        depth: ``(height, width)`` with 1 nearest.
        matte: ``(height, width)`` on 0 to 1, white on the subject.
        radius: How far inside each side its own depth is trusted from the edge.

    Returns:
        ``(split, edge)``: a :class:`Split` with the subject near, and a boolean
        ``(height, width)`` map of the pixels within ``radius`` of the matte's edge.
    """
    from . import matting

    subject = (matte >= SUBJECT_LEVEL).to(depth.dtype)[None, None]
    background = (matte <= BACKGROUND_LEVEL).to(depth.dtype)[None, None]
    subject_core = -_window_max(-subject, radius)
    background_core = -_window_max(-background, radius)
    planes = depth[None]
    subject_depth = fill(planes * subject_core[0], subject_core[0])[0]
    background_depth = fill(planes * background_core[0], background_core[0])[0]
    if float(subject_core.amax()) <= 0.0:
        subject_depth = depth
    if float(background_core.amax()) <= 0.0:
        background_depth = depth
    subject_depth = torch.where(subject_core[0, 0] > 0.5, depth, subject_depth)
    background_depth = torch.where(background_core[0, 0] > 0.5, depth, background_depth)
    band = (matte > BACKGROUND_LEVEL) & (matte < SUBJECT_LEVEL)
    front = light
    if bool(band.any()):
        unmixed = raw.linearise(matting.foreground(codes.clamp(0.0, 1.0), matte)[None])[0]
        front = torch.where(band[..., None], unmixed, light)
    subject_colour = front * matte[..., None]
    edge = (subject_core[0, 0] < 0.5) & (background_core[0, 0] < 0.5)
    split = Split(matte, subject_colour, (light - subject_colour).clamp(min=0.0), subject_depth, background_depth)
    return split, edge


def _local_mean(planes: torch.Tensor, radius: float) -> torch.Tensor:
    """The mean over a square window round every pixel, worked out at a reduced size.

    Args:
        planes: ``(channels, height, width)``, both sides multiples of :data:`MAX_REDUCTION`.
        radius: Half the window, in pixels.

    Returns:
        ``(channels, height, width)``.
    """
    factor = 1
    while factor * 2 <= MAX_REDUCTION and radius / (factor * 2) >= REACH_STEPS:
        factor *= 2
    held = planes[None]
    if factor > 1:
        held = functional.avg_pool2d(held, factor)
    steps = max(1, int(math.ceil(radius / factor)))
    held = functional.avg_pool2d(held, (1, 2 * steps + 1), 1, (0, steps), count_include_pad=False)
    held = functional.avg_pool2d(held, (2 * steps + 1, 1), 1, (steps, 0), count_include_pad=False)
    if factor > 1:
        held = functional.interpolate(held, scale_factor=factor, mode="bilinear", align_corners=False)
    return held[0]


def _extended(planes: torch.Tensor, nearer: torch.Tensor, radius: float) -> torch.Tensor:
    """A layer carried on behind the layers in front of it, from its own nearby colours.

    Args:
        planes: ``(4, height, width)``, premultiplied colour then coverage.
        nearer: ``(1, height, width)``, the coverage of every nearer layer.
        radius: How far from its own pixels the layer is carried, in pixels.

    Returns:
        ``(4, height, width)`` with coverage added under the nearer layers, full where the
        layer fills :data:`EXTEND_SHARE` of the window round a pixel and less where it is a sliver.
    """
    member = planes[3:]
    if float(member.amax()) <= EPSILON:
        return planes
    local = _local_mean(planes, radius)
    strength = (local[3:] / EXTEND_SHARE).clamp(0.0, 1.0)
    added = torch.minimum(nearer * strength, 1.0 - member).clamp(min=0.0)
    if float(added.amax()) <= EPSILON:
        return planes
    colour = local[:3] / local[3:].clamp_min(EPSILON)
    # A pixel the layer already holds part of carries on in its own colour.
    own = planes[:3] / member.clamp_min(EPSILON)
    colour = torch.where(member > OWN_SHARE, own, colour)
    return torch.cat([planes[:3] + added * colour, member + added])


def _padded(planes: torch.Tensor, margin: int) -> torch.Tensor:
    """Planes mirrored out by a margin, then zero-padded to a multiple of :data:`CANVAS_MULTIPLE`."""
    height, width = int(planes.shape[-2]), int(planes.shape[-1])
    mode = "reflect" if margin < min(height, width) else "replicate"
    grown = functional.pad(planes[None], (margin, margin, margin, margin), mode=mode)[0]
    tall = -(-int(grown.shape[-2]) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    wide = -(-int(grown.shape[-1]) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    return functional.pad(grown, (0, wide - int(grown.shape[-1]), 0, tall - int(grown.shape[-2])))


def render(codes, depth, settings: Settings, matte=None, interrupted=None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Defocus one frame through its depth map.

    Args:
        codes: ``(height, width, 3)`` picture codes, above 1 where the frame carries more light.
        depth: ``(height, width)`` with 1 nearest, on the frame's device.
        settings: What the frame is rendered with.
        matte: Optional ``(height, width)`` subject matte on 0 to 1.
        interrupted: Optional callable raising when the run is cancelled.

    Returns:
        ``(codes, focus, facts)``: the defocused frame as picture codes, a ``(height, width)``
        map 1 where sharp falling to 0 at ``settings.largest``, and a dict with ``layers`` and
        ``widest`` (the widest diameter drawn).
    """
    height, width = int(codes.shape[0]), int(codes.shape[1])
    spread = max(float(settings.largest), 0.0)
    if spread < SHARP_DIAMETER:
        return codes.clone(), torch.ones_like(depth), {"layers": 1, "widest": 0.0}
    light = raw.linearise(codes)
    radius = split_radius(height, width)
    if spread > SPLIT_LOW:
        depth = steadied(depth)
        split = split_edges(depth, codes, light, spread, radius)
    else:
        split = whole(light, depth)
    if matte is not None:
        by_matte, edge = split_matte(codes, light, depth, matte, CORE_RADII * radius)
        split = Split(*(
            torch.where(edge[..., None] if ours.ndim == 3 else edge, theirs, ours)
            for ours, theirs in zip(split, by_matte)
        ))
    parts = (
        (split.near_colour, split.share, split.near_depth),
        (split.far_colour, 1.0 - split.share, split.far_depth),
    )

    grid_values = diameter_grid(spread)
    grid = torch.tensor(grid_values, device=codes.device, dtype=torch.float32)
    lights = _highlights(codes, radius) if settings.bloom > 0.0 else None
    stacked = []
    spans = []
    used = set()
    for colour, coverage, part_depth in parts:
        diameters, places = _signed_layers(part_depth, settings, grid)
        spans.append(diameters)
        held = places[coverage > EPSILON]
        if held.numel() == 0:
            continue
        used.update(torch.unique(torch.cat([
            torch.sign(held) * held.abs().floor(), torch.sign(held) * held.abs().ceil(),
        ])).to(torch.int64).tolist())
        stacked.append(torch.cat([
            _bloom(colour, lights, diameters, settings.bloom).permute(2, 0, 1),
            coverage[None],
            places[None],
        ]))
    visible = torch.where(split.share >= 0.5, spans[0], spans[1])

    order = sorted(used)
    sizes = {layer: grid_values[abs(layer) - 1] if layer else 0.0 for layer in order}
    widest = max(sizes.values(), default=0.0)
    margin = canvas_margin(widest)
    canvases = [_padded(planes, margin) for planes in stacked]
    # The widest blur among the layers nearer than each one.
    nearer_widest = {}
    running = 0.0
    for layer in reversed(order):
        nearer_widest[layer] = running
        running = max(running, sizes[layer])

    tall, wide = int(canvases[0].shape[-2]), int(canvases[0].shape[-1])
    composite = canvases[0].new_zeros(3, tall, wide)
    cover = canvases[0].new_zeros(1, tall, wide)
    remaining = sum(canvas[3:4] for canvas in canvases).clamp(0.0, 1.0)
    for layer in order:
        if interrupted is not None:
            interrupted()
        planes = None
        for canvas in canvases:
            part = canvas[:4] * _layer_share(canvas[4], layer)
            planes = part if planes is None else planes + part
        remaining = (remaining - planes[3:4]).clamp(0.0, 1.0)
        reach = max(sizes[layer], nearer_widest[layer]) / 2.0 + EXTEND_MARGIN
        box = _region(planes[3], 2.0 * reach + sizes[layer] / 2.0)
        if box is None:
            continue
        rows, columns = box
        held = planes[:, rows, columns]
        nearer = remaining[:, rows, columns]
        if bool((nearer > EPSILON).any()):
            held = _extended(held, nearer, reach)
        if sizes[layer] >= SHARP_DIAMETER:
            held = _blurred_region(held, sizes[layer], settings.blades, settings.rotation)
        colour, coverage = held[:3], held[3:]
        composite[:, rows, columns] = colour + (1.0 - coverage) * composite[:, rows, columns]
        cover[:, rows, columns] = coverage + (1.0 - coverage) * cover[:, rows, columns]
        del planes, held, colour, coverage

    filled = fill(composite, cover)
    result = filled[:, margin: margin + height, margin: margin + width].permute(1, 2, 0)
    focus = (1.0 - visible / spread).clamp(0.0, 1.0)
    return raw.encode(result), focus, {"layers": len(used), "widest": widest}
