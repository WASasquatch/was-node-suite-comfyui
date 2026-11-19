"""Portrait retouching inside a skin mask: tone evened, spots healed, texture kept.

Pictures are ``(batch, height, width, channels)`` picture codes, worked in Oklab of linear
light. Sizes are shares of a skin region's size, the square root of its area.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import dynamic, raw
from .accelerate import working_device
from .develop import from_oklab, matte_for, smoothstep, soft_blur, to_oklab

__all__ = [
    "BLEMISHES",
    "Region",
    "find_spots",
    "guided",
    "heal_spots",
    "regions",
    "retouch",
    "retouch_frame",
    "smooth_tone",
]

#: Spot healing a menu offers.
BLEMISHES = ("off", "red spots", "all spots")

#: Radius splitting fine texture from tone, and the radius tone is evened over, as shares of a
#: region's size.
TEXTURE_RADIUS = 0.004
TONE_RADIUS = 0.05

#: Lightness variation each split still treats as flat, in Oklab L.
TEXTURE_FLAT = 0.03
TONE_FLAT = 0.1

#: Colour variation the tone base treats as flat, and the texture split treats as flat, in
#: Oklab a and b.
COLOUR_FLAT = 0.12
COLOUR_EDGE = 0.03

#: Blotch depth half evened at full strength, in Oklab L and in Oklab a and b. Shading deeper
#: than this is mostly kept.
BLOTCH_LIGHTNESS = 0.05
BLOTCH_COLOUR = 0.035

#: Share of the evening lightness receives; colour receives all of it.
LIGHTNESS_SHARE = 0.8

#: Smallest region size retouched, in pixels.
MIN_REGION = 32

#: Most regions retouched in one frame, largest first.
MOST_REGIONS = 32

#: Longest side a mask is labelled at.
LABEL_SIDE = 1024

#: Region size spots are searched at, in pixels.
SEARCH_SIZE = 800

#: Smallest spot diameter, as a share of the region size and in search pixels.
MIN_SPOT = 0.006
MIN_SPOT_PIXELS = 3.0

#: Disc radius a centre-surround response peaks at, in units of its inner sigma, and the share
#: of a disc's contrast it reads there.
DISC_RADIUS = 2.22
DISC_RESPONSE = 0.675

#: Outer sigma of the centre-surround pair, in units of the inner one.
SURROUND = 3.0

#: Weight of redness (Oklab a) against darkness (Oklab L) in a spot's contrast.
RED_WEIGHT = 1.5

#: Faintest spot contrast healed, in Oklab L, and how far above the region's own texture at that
#: scale a spot must stand, in robust deviations.
MIN_CONTRAST = 0.025
ABOVE_TEXTURE = 1.5

#: Redness a red spot carries at least, in Oklab a, and its least share of the contrast.
MIN_REDNESS = 0.008
RED_SHARE = 0.35

#: Darkest spot without redness that is healed, in Oklab L.
MAX_DARKNESS = 0.16

#: Darkest pixel a spot's core may hold below its surround, in Oklab L; the dimmest surround
#: searched, in Oklab L and as a share of the region's median lightness.
MAX_CORE_DARKNESS = 0.2
MIN_LIGHTNESS = 0.22
SHADOW_SHARE = 0.6

#: Least ratio of a spot's gentler curvature to its steeper one.
MIN_ROUNDNESS = 0.3

#: Widest surround whose shape is read, in units of the spot's own sigma, and the most line-like
#: curvature it may hold against the spot's own gentler curvature.
CONTEXT_SCALE = 4.0
MAX_LINE = 0.5

#: Most gradient a point on a crease's ridge holds, against the crease's curvature.
ON_RIDGE = 0.35

#: Most of a spot's contrast still read at twice and at four times its scale.
MID_PERSIST = 0.85
WIDE_PERSIST = 0.45

#: Most fine detail a spot's surround may hold, against the region's plainer skin, the 30th
#: percentile of its detail.
MAX_BUSY = 3.2

#: Least spot contrast against the deviation of its own surround, and the least share of its
#: contrast it keeps against the surround's median.
STAND_OUT = 2.0
ROBUST_SHARE = 0.6

#: Skin a spot and its surround must sit in, as a share of the mask.
MIN_SKIN = 0.95

#: Least distance from a spot to the edge of the mask, as a share of the region size.
EDGE_MARGIN = 0.02

#: Most spots healed in one region, strongest first.
MOST_SPOTS = 400

#: Heal radius against the measured spot, the feather against the heal radius, and the angles and
#: distances a source patch is sought at.
HEAL_GROWTH = 1.15
FEATHER = 0.5
SOURCE_ANGLES = 16
SOURCE_STEPS = (2.0, 2.5, 3.0)

#: Weights of a texture mismatch and of a mean tone difference against a tone shape mismatch
#: when choosing a source.
TEXTURE_MATCH = 2.0
OFFSET_MATCH = 1.0

#: Fine texture a source may hold, as shares of the texture around the spot.
TEXTURE_RANGE = (0.67, 1.5)

#: Most tone shape mismatch around a source, against the tone spread around the spot.
STRUCTURE_FIT = 2.5

#: Most contrast a source patch, and a healed spot, may keep against the spot's own, and how
#: many of the best sources are tried.
SOURCE_SPOT = 0.3
HEALED_SPOT = 0.25
SOURCE_TRIES = 4

#: Most tone spread a source patch may hold against the spread around the spot.
LUMPS = 1.3

#: Skin a source patch must sit in, as a share of the mask.
SOURCE_SKIN = 0.98


class Region:
    """One connected skin region of a frame.

    Attributes:
        box: ``(top, bottom, left, right)`` in frame pixels, bottom and right exclusive.
        scale: Square root of the region's area, in pixels.
        weight: ``(height, width)`` mask over the box, zero away from this region.
    """

    def __init__(self, box: tuple[int, int, int, int], scale: float, weight: torch.Tensor):
        self.box = box
        self.scale = scale
        self.weight = weight


def _box(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Box mean of ``(channels, height, width)`` with zeros past the edge."""
    size = 2 * radius + 1
    y = F.avg_pool2d(x[None], (1, size), stride=1, padding=(0, radius), count_include_pad=True)
    y = F.avg_pool2d(y, (size, 1), stride=1, padding=(radius, 0), count_include_pad=True)
    return y[0]


def _weighted(x: torch.Tensor, weight: torch.Tensor, radius: int, total: torch.Tensor) -> torch.Tensor:
    """Box mean of ``x`` weighted by ``weight``, given the box mean of the weight."""
    return _box(x * weight, radius) / total


def guided(source: torch.Tensor, guide: torch.Tensor, weight: torch.Tensor, radius: int, flats) -> torch.Tensor:
    """Smooth ``source`` where an Oklab ``guide`` is flat, counting only weighted pixels. Unclamped.

    Args:
        source: ``(channels, height, width)``.
        guide: ``(3, height, width)`` Oklab whose edges are followed.
        weight: ``(1, height, width)`` in 0 to 1.
        radius: Window half size in pixels.
        flats: Variation of L, a and b still treated as flat, in Oklab units.

    Returns:
        A tensor shaped like ``source``.
    """
    height, width = int(source.shape[-2]), int(source.shape[-1])
    channels = int(source.shape[0])
    scale = torch.as_tensor(flats, dtype=guide.dtype, device=guide.device).reshape(3, 1, 1)
    lead = guide / scale
    factor = max(1, radius // 4)
    if factor > 1 and min(height, width) // factor < 4:
        factor = 1
    if factor > 1:
        size = (max(1, height // factor), max(1, width // factor))
        w = F.interpolate(weight[None], size=size, mode="area")[0]
        held = w.clamp(min=1e-6)
        g = F.interpolate((lead * weight)[None], size=size, mode="area")[0] / held
        p = F.interpolate((source * weight)[None], size=size, mode="area")[0] / held
        small = max(1, round(radius / factor))
    else:
        w, g, p, small = weight, lead, source, max(1, radius)
    total = _box(w, small).clamp(min=1e-6)
    rows, cols = int(g.shape[1]), int(g.shape[2])
    mean_g = _weighted(g, w, small, total)
    mean_p = _weighted(p, w, small, total)
    # Covariance of each guide channel with each source channel, and of the guide with itself.
    cross = _weighted((g[:, None] * p[None]).reshape(3 * channels, rows, cols), w, small, total)
    cross = cross.reshape(3, channels, rows, cols) - mean_g[:, None] * mean_p[None]
    own = _weighted((g[:, None] * g[None]).reshape(9, rows, cols), w, small, total)
    own = own.reshape(3, 3, rows, cols) - mean_g[:, None] * mean_g[None]
    own = own + torch.eye(3, dtype=g.dtype, device=g.device)[:, :, None, None]
    slope = torch.linalg.solve(own.permute(2, 3, 0, 1), cross.permute(2, 3, 0, 1))
    slope = slope.permute(2, 3, 0, 1)
    offset = mean_p - (slope * mean_g[:, None]).sum(0)
    slope = _weighted(slope.reshape(3 * channels, rows, cols), w, small, total)
    offset = _weighted(offset, w, small, total)
    if factor > 1:
        slope = F.interpolate(slope[None], size=(height, width), mode="bilinear", align_corners=False)[0]
        offset = F.interpolate(offset[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return (slope.reshape(3, channels, height, width) * lead[:, None]).sum(0) + offset


def _masked_blur(x: torch.Tensor, weight: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of ``(channels, height, width)`` counting only weighted pixels."""
    if sigma < 0.5:
        return x
    num = soft_blur((x * weight)[None], sigma)[0]
    den = soft_blur(weight[None], sigma)[0]
    return num / den.clamp(min=1e-6)


def smooth_tone(lab: torch.Tensor, weight: torch.Tensor, scale: float, even: float, texture: float) -> torch.Tensor:
    """Even blotchy tone and scale fine texture inside one region.

    Args:
        lab: ``(3, height, width)`` Oklab.
        weight: ``(1, height, width)`` skin weight.
        scale: Region size in pixels.
        even: 0 to 1, how much of the blotch band is removed.
        texture: 0 to 1, how much fine texture is kept.

    Returns:
        A tensor shaped like ``lab``.
    """
    if even <= 0.0 and texture >= 1.0:
        return lab
    fine = max(1, round(TEXTURE_RADIUS * scale))
    low = guided(lab, lab, weight, fine, (TEXTURE_FLAT, COLOUR_EDGE, COLOUR_EDGE))
    out = lab
    if texture < 1.0:
        out = out + (texture - 1.0) * (lab - low)
    if even > 0.0:
        wide = max(fine + 1, round(TONE_RADIUS * scale))
        base = guided(low, low, weight, wide, (TONE_FLAT, COLOUR_FLAT, COLOUR_FLAT))
        band = low - base
        limits = torch.tensor([BLOTCH_LIGHTNESS, BLOTCH_COLOUR, BLOTCH_COLOUR], dtype=lab.dtype, device=lab.device)[:, None, None]
        reach = 1.0 / (1.0 + (_masked_blur(band.abs(), weight, fine) / limits).pow(2))
        shares = torch.tensor([LIGHTNESS_SHARE, 1.0, 1.0], dtype=lab.dtype, device=lab.device)[:, None, None]
        change = _masked_blur(-band * reach, weight, 2.0 * fine)
        out = out + even * shares * change
    return out


def _eroded(weight: torch.Tensor, radius: int) -> torch.Tensor:
    """Least weight within a square of half size ``radius``, outside counting as none."""
    if radius < 1:
        return weight
    size = 2 * radius + 1
    gap = F.pad((1.0 - weight)[None], (radius, radius, radius, radius), value=1.0)
    gap = F.max_pool2d(gap, (1, size), stride=1)
    gap = F.max_pool2d(gap, (size, 1), stride=1)
    return 1.0 - gap[0]


def _sums(table: torch.Tensor, y: torch.Tensor, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Sum of a plane over squares, read from its padded summed-area table."""
    height, width = int(table.shape[0]) - 1, int(table.shape[1]) - 1
    top = (y - radius).clamp(0, height)
    bottom = (y + radius + 1).clamp(0, height)
    left = (x - radius).clamp(0, width)
    right = (x + radius + 1).clamp(0, width)
    return table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]


def _table(plane: torch.Tensor) -> torch.Tensor:
    """Summed-area table of a ``(height, width)`` plane, padded with a leading zero row and column."""
    table = F.pad(plane.double().cumsum(0).cumsum(1), (1, 0, 1, 0))
    return table


def _curvatures(centre: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, sigma: float):
    """Both principal curvatures of a blurred signal at given points, scaled by ``sigma`` squared.

    Args:
        centre: ``(height, width)`` signal blurred at ``sigma``.
        rows: Point rows.
        cols: Point columns.
        sigma: Blur the curvature is read at.

    Returns:
        ``(upper, lower, slope)``: the larger and the smaller eigenvalue of the Hessian at each
        point, and the gradient's length scaled by ``sigma``. A round peak has both eigenvalues
        negative and similar; a line has one near 0.
    """
    height, width = int(centre.shape[0]), int(centre.shape[1])
    step = max(1, round(sigma))
    up = (rows - step).clamp(0, height - 1)
    down = (rows + step).clamp(0, height - 1)
    left = (cols - step).clamp(0, width - 1)
    right = (cols + step).clamp(0, width - 1)
    middle = centre[rows, cols]
    hxx = centre[rows, right] + centre[rows, left] - 2.0 * middle
    hyy = centre[down, cols] + centre[up, cols] - 2.0 * middle
    hxy = (centre[down, right] + centre[up, left] - centre[down, left] - centre[up, right]) / 4.0
    trace = hxx + hyy
    root = torch.sqrt(((hxx - hyy) / 2.0) ** 2 + hxy * hxy)
    scale = (sigma / step) ** 2
    slope = torch.sqrt((centre[rows, right] - centre[rows, left]) ** 2 + (centre[down, cols] - centre[up, cols]) ** 2)
    return (trace / 2.0 + root) * scale, (trace / 2.0 - root) * scale, slope * sigma / (2.0 * step)


def _ring_samples(plane: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, radius: float) -> torch.Tensor:
    """Samples of a plane on three circles around each point, 1.8 to 3 radii out.

    Args:
        plane: ``(height, width)``.
        rows: Point rows.
        cols: Point columns.
        radius: Radius the circles are scaled by, in pixels.

    Returns:
        ``(points, 48)``.
    """
    height, width = int(plane.shape[0]), int(plane.shape[1])
    angles = torch.linspace(0.0, 2.0 * math.pi, 17, dtype=plane.dtype, device=plane.device)[:-1]
    reach = torch.tensor([1.8, 2.4, 3.0], dtype=plane.dtype, device=plane.device) * radius
    ys = rows.to(plane.dtype)[:, None, None] + reach[None, :, None] * torch.sin(angles)[None, None, :]
    xs = cols.to(plane.dtype)[:, None, None] + reach[None, :, None] * torch.cos(angles)[None, None, :]
    grid = torch.stack([(xs + 0.5) / width * 2.0 - 1.0, (ys + 0.5) / height * 2.0 - 1.0], dim=-1)
    samples = F.grid_sample(plane[None, None], grid.reshape(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
    return samples.reshape(int(rows.shape[0]), -1)


def find_spots(lab: torch.Tensor, weight: torch.Tensor, largest: float, mode: str):
    """Small round spots darker or redder than the skin around them.

    Args:
        lab: ``(3, height, width)`` Oklab at the search size.
        weight: ``(1, height, width)`` skin weight.
        largest: Largest spot diameter in pixels.
        mode: One of :data:`BLEMISHES` other than ``off``.

    Returns:
        ``(spots, kept)``: ``(spots, 3)`` holding row, column and radius in pixels, strongest
        first, and a ``(height, width)`` map of marks that stand out but are not healed, such as
        moles and feature corners.
    """
    height, width = int(lab.shape[-2]), int(lab.shape[-1])
    empty = torch.zeros((0, 3), dtype=lab.dtype, device=lab.device)
    marks = torch.zeros((height, width), dtype=torch.bool, device=lab.device)
    inside = weight[0] > 0.5
    area = int(inside.sum())
    if area < 64:
        return empty, marks
    smallest = max(MIN_SPOT_PIXELS, MIN_SPOT * math.sqrt(area))
    low_sigma = smallest / (2.0 * DISC_RADIUS)
    high_sigma = largest / (2.0 * DISC_RADIUS)
    if high_sigma <= low_sigma or min(height, width) < 8:
        return empty, marks
    steps = max(1, math.ceil(2.0 * math.log2(high_sigma / low_sigma)))
    ratio = (high_sigma / low_sigma) ** (1.0 / steps)
    # Levels one step either side of the searched range, plus wider levels for the surround.
    context = max(2, round(math.log(CONTEXT_SCALE) / math.log(ratio)))
    sigmas = [low_sigma * ratio ** (i - 1) for i in range(steps + 2 + context)]
    signal = torch.cat([lab[0:1], lab[1:2]], dim=0)
    spotty = RED_WEIGHT * lab[1] - lab[0]
    counted = inside.to(lab.dtype)
    skin_table = _table(counted)
    fine = _masked_blur(lab[0:1], weight, 0.7)
    spotted = _masked_blur(spotty[None], weight, 0.7)[0]
    detail = (lab[0] - _masked_blur(lab[0:1], weight, 1.0)[0]).abs() * counted
    typical = float(detail[inside].quantile(0.3)) + 1e-6
    shade = SHADOW_SHARE * float(lab[0][inside].median())
    detail_table = _table(detail)
    responses, reds, darks, centres, surrounds, lights = [], [], [], [], [], []
    for sigma in sigmas:
        inner = _masked_blur(signal, weight, sigma)
        outer = _masked_blur(signal, weight, SURROUND * sigma)
        dark = outer[0] - inner[0]
        red = inner[1] - outer[1]
        responses.append(dark + RED_WEIGHT * red)
        reds.append(red)
        darks.append(dark)
        centres.append(RED_WEIGHT * inner[1] - inner[0])
        surrounds.append(RED_WEIGHT * outer[1] - outer[0])
        lights.append(outer[0])
    stack = torch.stack(responses)

    peaks = F.max_pool3d(stack[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    found = []
    for level in range(1, steps + 2):
        sigma = sigmas[level]
        response = stack[level]
        values = response[inside]
        if values.numel() < 16:
            continue
        spread = 1.4826 * (values - values.median()).abs().median()
        floor = max(MIN_CONTRAST * DISC_RESPONSE, float(ABOVE_TEXTURE * spread))
        radius = DISC_RADIUS * sigma
        skin = _eroded(weight, max(1, math.ceil(2.5 * radius), math.ceil(EDGE_MARGIN * math.sqrt(area))))[0]
        hit = (response >= peaks[level]) & (response >= floor) & (skin >= MIN_SKIN)
        rows, cols = torch.nonzero(hit, as_tuple=True)
        if rows.numel() == 0:
            continue
        strength = response[rows, cols]
        red = reds[level][rows, cols] / DISC_RESPONSE
        dark = darks[level][rows, cols] / DISC_RESPONSE
        reddish = red >= MIN_REDNESS
        tests = {}
        if mode == "red spots":
            tests["kind"] = reddish & (RED_WEIGHT * reds[level][rows, cols] >= RED_SHARE * strength)
        else:
            tests["kind"] = reddish | (dark <= MAX_DARKNESS)

        # Surround lit, and the core no more than MAX_CORE_DARKNESS below it.
        surround_light = lights[level][rows, cols]
        tests["lit"] = surround_light >= max(MIN_LIGHTNESS, shade)
        reach = max(1, math.ceil(radius))
        darkest = -F.max_pool2d(-fine[None], 2 * reach + 1, stride=1, padding=reach)[0, 0]
        tests["core"] = surround_light - darkest[rows, cols] <= MAX_CORE_DARKNESS

        # Round at its own scale, with no crease ridge through it at the wider levels.
        upper, lower, _slope = _curvatures(centres[level], rows, cols, sigma)
        tests["round"] = (lower < 0) & ((-upper).clamp(min=0.0) >= MIN_ROUNDNESS * (-lower))
        crease = torch.zeros_like(tests["round"])
        for wider in (level + context // 2, level + context):
            wide_upper, wide_lower, slope = _curvatures(centres[wider], rows, cols, sigmas[wider])
            ridge = (-wide_lower) - wide_upper.abs()
            on_ridge = (wide_lower < 0) & (slope <= ON_RIDGE * (-wide_lower))
            crease = crease | (on_ridge & (ridge > MAX_LINE * (-upper).clamp(min=0.0)))
        tests["line"] = ~crease

        # Response at the two wider levels, against the spot's own.
        tests["alone"] = (stack[level + context // 2][rows, cols] <= MID_PERSIST * strength) & (
            stack[level + context][rows, cols] <= WIDE_PERSIST * strength
        )

        near = max(1, round(1.6 * radius))
        far = max(near + 1, round(3.2 * radius))
        near_rows, far_rows = torch.full_like(rows, near), torch.full_like(rows, far)
        ring_skin = (_sums(skin_table, rows, cols, far_rows) - _sums(skin_table, rows, cols, near_rows)).clamp(min=1.0)

        # Fine detail in the surround, against the region's plainer skin.
        busy = _sums(detail_table, rows, cols, far_rows) - _sums(detail_table, rows, cols, near_rows)
        tests["busy"] = (busy / ring_skin).to(response.dtype) <= MAX_BUSY * typical

        # Contrast against the deviation of the surround.
        varied = (spotty - surrounds[level]) * counted
        first = _table(varied)
        second = _table(varied * varied)
        mean = (_sums(first, rows, cols, far_rows) - _sums(first, rows, cols, near_rows)) / ring_skin
        square = (_sums(second, rows, cols, far_rows) - _sums(second, rows, cols, near_rows)) / ring_skin
        deviation = (square - mean * mean).clamp(min=0.0).sqrt().to(response.dtype)
        tests["clear"] = strength / DISC_RESPONSE >= STAND_OUT * deviation

        # Contrast against the median of the surround, sampled on three circles.
        ring = _ring_samples(spotted, rows, cols, radius)
        middle = ring.median(-1).values
        scatter = 1.4826 * (ring - middle[:, None]).abs().median(-1).values
        plain = centres[level][rows, cols] - middle
        tests["plain"] = (plain >= ROBUST_SHARE * strength / DISC_RESPONSE) & (plain >= STAND_OUT * scatter)

        keep = torch.stack(list(tests.values())).all(0)
        refused = tests["clear"] & ~keep
        if bool(refused.any()):
            plane = torch.zeros((height, width), dtype=lab.dtype, device=lab.device)
            plane[rows[refused], cols[refused]] = 1.0
            grow = max(1, math.ceil(1.5 * radius))
            plane = F.max_pool2d(plane[None, None], 2 * grow + 1, stride=1, padding=grow)[0, 0]
            marks |= plane > 0
        rows, cols = rows[keep], cols[keep]
        strength = strength[keep] / (deviation[keep] + MIN_CONTRAST)
        found.append(torch.stack([rows.to(lab.dtype), cols.to(lab.dtype), torch.full_like(strength, radius), strength], dim=1))
    if not found:
        return empty, marks
    spots = torch.cat(found)
    spots = spots[spots[:, 3].argsort(descending=True)][: 4 * MOST_SPOTS]

    kept: list[int] = []
    centres_kept = spots[:, :2].cpu().tolist()
    radii = spots[:, 2].cpu().tolist()
    for index in range(int(spots.shape[0])):
        y, x = centres_kept[index]
        r = radii[index]
        if all((y - centres_kept[k][0]) ** 2 + (x - centres_kept[k][1]) ** 2 > (0.8 * (r + radii[k])) ** 2 for k in kept):
            kept.append(index)
            if len(kept) >= MOST_SPOTS:
                break
    spots = spots[kept]
    for y, x, r in spots[:, :3].cpu().tolist():
        span = math.ceil(2.0 * r)
        top, left = max(0, round(y) - span), max(0, round(x) - span)
        marks[top:round(y) + span + 1, left:round(x) + span + 1] = False
    return torch.stack([spots[:, 0], spots[:, 1], _extent(lab, weight, spots)], dim=1), marks


def _extent(lab: torch.Tensor, weight: torch.Tensor, spots: torch.Tensor) -> torch.Tensor:
    """Radius at which each spot fades into the skin around it, in pixels."""
    if spots.shape[0] == 0:
        return spots[:, 2]
    signal = _masked_blur(RED_WEIGHT * lab[1:2] - lab[0:1], weight, 0.7)
    height, width = int(lab.shape[-2]), int(lab.shape[-1])
    radius = spots[:, 2]
    fractions = torch.linspace(0.0, 4.0, 25, dtype=lab.dtype, device=lab.device)
    angles = torch.linspace(0.0, 2.0 * math.pi, 17, dtype=lab.dtype, device=lab.device)[:-1]
    reach = radius[:, None, None] * fractions[None, :, None]
    ys = spots[:, 0, None, None] + reach * torch.sin(angles)[None, None, :]
    xs = spots[:, 1, None, None] + reach * torch.cos(angles)[None, None, :]
    grid = torch.stack([(xs + 0.5) / width * 2.0 - 1.0, (ys + 0.5) / height * 2.0 - 1.0], dim=-1)
    samples = F.grid_sample(signal[None], grid[None].reshape(1, -1, len(angles), 2), mode="bilinear", padding_mode="border", align_corners=False)
    profile = samples.reshape(int(spots.shape[0]), len(fractions), len(angles)).mean(-1)
    background = profile[:, fractions >= 3.0].mean(-1, keepdim=True)
    peak = profile[:, :3].amax(-1, keepdim=True)
    level = background + 0.15 * (peak - background)
    below = profile <= level
    below[:, 0] = False
    first = torch.where(below.any(-1), below.float().argmax(-1), torch.full_like(below[:, 0], len(fractions) - 1, dtype=torch.long))
    measured = fractions[first] * radius
    return measured.clamp(min=0.7 * radius, max=1.8 * radius)


def heal_spots(lab: torch.Tensor, weight: torch.Tensor, spots: torch.Tensor, largest: float, kept=None):
    """Replace each spot with nearby skin, matched in tone to its own surround.

    Args:
        lab: ``(3, height, width)`` Oklab at full size.
        weight: ``(1, height, width)`` skin weight.
        spots: ``(spots, 3)`` row, column and measured radius in pixels.
        largest: Largest spot diameter in pixels.
        kept: ``(height, width)`` map of marks never copied from, or None.

    Returns:
        ``(lab, healed, count)``: the healed Oklab, ``(height, width)`` heal weights, and how many
        spots were healed.
    """
    height, width = int(lab.shape[-2]), int(lab.shape[-1])
    if spots.shape[0] == 0:
        return lab, torch.zeros((height, width), dtype=lab.dtype, device=lab.device), 0
    radii = (HEAL_GROWTH * spots[:, 2] + 1.0).clamp(max=0.8 * largest + 1.0)
    feathers = (FEATHER * radii).clamp(min=1.5)
    reach = radii + feathers
    pad = int(math.ceil(float(reach.max()) * (1.0 + max(SOURCE_STEPS) + 1.0))) + 2

    work = F.pad(lab[None], (pad, pad, pad, pad), mode="replicate")[0]
    skin = F.pad(weight, (pad, pad, pad, pad), value=0.0)[0]
    marks = torch.zeros_like(skin)
    if kept is None:
        taken = torch.zeros_like(skin, dtype=torch.bool)
    else:
        taken = F.pad(kept.to(torch.bool), (pad, pad, pad, pad), value=False)
    rows = (spots[:, 0].round().long() + pad).tolist()
    cols = (spots[:, 1].round().long() + pad).tolist()
    for row, col, outer in zip(rows, cols, reach.tolist()):
        span = int(math.ceil(outer))
        taken[row - span:row + span + 1, col - span:col + span + 1] |= _disc(span, outer, lab.device)

    # Tone without fine texture, and the energy of that texture.
    texture_sigma = max(1.0, largest / 16.0)
    smooth = _masked_blur(work, skin[None], texture_sigma)
    energy = (work[0] - smooth[0]) ** 2

    count = 0
    angles = [2.0 * math.pi * k / SOURCE_ANGLES for k in range(SOURCE_ANGLES)]
    for row, col, inner, feather in zip(rows, cols, radii.tolist(), feathers.tolist()):
        outer = inner + feather
        half = int(math.ceil(outer + inner)) + 1
        size = 2 * half + 1
        axis = torch.arange(size, device=lab.device, dtype=lab.dtype) - half
        rho = torch.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
        hole = rho <= outer
        core = rho <= inner

        here = (slice(row - half, row + half + 1), slice(col - half, col + half + 1))
        dest = work[:, here[0], here[1]]
        dest_skin = skin[here]
        dest_taken = taken[here]
        known = ~hole & (dest_skin > 0.5) & ~dest_taken
        if int(known.sum()) < 8:
            continue
        dest_smooth = smooth[:, here[0], here[1]]
        dest_texture = float(energy[here][known].mean().sqrt())
        dest_lumps = _lumps(dest_smooth, known)
        spot = _contrast(dest_smooth, core, known)
        candidates = []
        for step in SOURCE_STEPS:
            distance = step * outer + 1.0
            for angle in angles:
                y = row + int(round(distance * math.sin(angle)))
                x = col + int(round(distance * math.cos(angle)))
                there = (slice(y - half, y + half + 1), slice(x - half, x + half + 1))
                source_skin = skin[there]
                if float(source_skin[hole].min()) < SOURCE_SKIN or bool(taken[there][hole].any()):
                    continue
                usable = known & (source_skin > 0.5)
                if int(usable.sum()) < 8:
                    continue
                source_smooth = smooth[:, there[0], there[1]]
                if _contrast(source_smooth, core, usable) > max(SOURCE_SPOT * spot, 0.5 * dest_lumps) + MIN_CONTRAST / 8.0:
                    continue
                # Tone spread inside the source, against the spread around the spot.
                source_lumps = _lumps(source_smooth, hole)
                if source_lumps > LUMPS * dest_lumps + MIN_CONTRAST / 8.0:
                    continue
                source_texture = float(energy[there][hole].mean().sqrt())
                if not TEXTURE_RANGE[0] * dest_texture <= source_texture <= TEXTURE_RANGE[1] * dest_texture + 1e-4:
                    continue
                difference = (dest_smooth - source_smooth)[:, usable]
                shift = difference.mean(-1, keepdim=True)
                structure = float((difference - shift).pow(2).mean(-1).sum())
                if structure > STRUCTURE_FIT * (dest_lumps + MIN_CONTRAST / 8.0) ** 2:
                    continue
                cost = (
                    structure
                    + TEXTURE_MATCH * (source_texture - dest_texture) ** 2
                    + OFFSET_MATCH * float(shift.pow(2).sum())
                    + (source_lumps - dest_lumps) ** 2
                )
                candidates.append((cost * (1.0 + 0.05 * (step - 1.0)), there))
        candidates.sort(key=lambda item: item[0])
        blend = (1.0 - smoothstep(inner, outer, rho))[None]
        weights = known.to(lab.dtype)[None]
        everywhere = torch.ones_like(weights)
        reach_blur = max(0.7 * inner, 1.0)
        # Tone the surround implies across the spot.
        implied = _masked_blur(dest_smooth, weights, reach_blur)
        allowed = HEALED_SPOT * spot + MIN_CONTRAST / 8.0
        for _cost, there in candidates[:SOURCE_TRIES]:
            source = work[:, there[0], there[1]]
            healed = source + _masked_blur(dest - source, weights, reach_blur)
            # A source whose healed tone strays from the implied tone is refused.
            if _strays(_masked_blur(healed, everywhere, max(texture_sigma, 0.5 * inner)), implied, hole) > allowed:
                continue
            dest.add_(blend * (healed - dest))
            mark = marks[here]
            torch.maximum(mark, blend[0], out=mark)
            count += 1
            break
    inside = (slice(pad, pad + height), slice(pad, pad + width))
    return work[:, inside[0], inside[1]], marks[inside], count


def _contrast(patch: torch.Tensor, inside: torch.Tensor, around: torch.Tensor) -> float:
    """How far the mean colour of one part of an Oklab patch sits from another's.

    Args:
        patch: ``(3, height, width)`` Oklab.
        inside: Boolean ``(height, width)`` selecting the part measured.
        around: Boolean ``(height, width)`` selecting what it is measured against.

    Returns:
        The distance in Oklab, with redness weighted by :data:`RED_WEIGHT`.
    """
    difference = patch[:, inside].mean(-1) - patch[:, around].mean(-1)
    scale = torch.tensor([1.0, RED_WEIGHT, 1.0], dtype=patch.dtype, device=patch.device)
    return float((difference * scale).norm())


def _strays(patch: torch.Tensor, expected: torch.Tensor, region: torch.Tensor) -> float:
    """Root mean square distance between two Oklab patches over part of them.

    Args:
        patch: ``(3, height, width)`` Oklab.
        expected: ``(3, height, width)`` Oklab it is measured against.
        region: Boolean ``(height, width)`` selecting the part measured.

    Returns:
        The distance, redness weighted by :data:`RED_WEIGHT`.
    """
    scale = torch.tensor([1.0, RED_WEIGHT, 1.0], dtype=patch.dtype, device=patch.device)[:, None]
    gap = (patch[:, region] - expected[:, region]) * scale
    return float(gap.pow(2).sum(0).mean().sqrt())


def _lumps(patch: torch.Tensor, region: torch.Tensor) -> float:
    """Spread of an Oklab patch's tone over part of it.

    Args:
        patch: ``(3, height, width)`` Oklab.
        region: Boolean ``(height, width)`` selecting the part measured.

    Returns:
        The root of the summed channel variances, redness weighted by :data:`RED_WEIGHT`.
    """
    values = patch[:, region]
    scale = torch.tensor([1.0, RED_WEIGHT, 1.0], dtype=patch.dtype, device=patch.device)[:, None]
    return float(((values - values.mean(-1, keepdim=True)) * scale).pow(2).mean(-1).sum().sqrt())


def _disc(span: int, radius: float, device) -> torch.Tensor:
    """A ``(2 * span + 1)`` square holding a filled disc."""
    axis = torch.arange(-span, span + 1, device=device, dtype=torch.float32)
    return (axis[:, None] ** 2 + axis[None, :] ** 2) <= radius * radius


def regions(weight: torch.Tensor) -> list[Region]:
    """The connected skin regions of one frame's mask, largest first.

    Args:
        weight: ``(height, width)`` mask in 0 to 1 on the CPU.

    Returns:
        Up to :data:`MOST_REGIONS` regions of at least :data:`MIN_REGION` pixels a side.
    """
    from scipy import ndimage

    height, width = int(weight.shape[0]), int(weight.shape[1])
    factor = max(1, math.ceil(max(height, width) / LABEL_SIDE))
    small = F.interpolate(weight[None, None].float(), size=(max(1, height // factor), max(1, width // factor)), mode="area")[0, 0]
    labels, count = ndimage.label((small > 0.5).numpy())
    if count == 0:
        return []
    sizes = ndimage.sum_labels(labels > 0, labels, index=range(1, count + 1))
    boxes = ndimage.find_objects(labels)
    order = sorted(range(count), key=lambda i: -sizes[i])
    sy, sx = height / small.shape[0], width / small.shape[1]
    found = []
    for index in order[:MOST_REGIONS]:
        scale = math.sqrt(float(sizes[index]) * sy * sx)
        if scale < MIN_REGION:
            continue
        rows, cols = boxes[index]
        top = max(0, int(rows.start * sy) - 4 * factor)
        bottom = min(height, int(math.ceil(rows.stop * sy)) + 4 * factor)
        left = max(0, int(cols.start * sx) - 4 * factor)
        right = min(width, int(math.ceil(cols.stop * sx)) + 4 * factor)
        own = torch.from_numpy(ndimage.binary_dilation(labels == index + 1, iterations=3)).float()
        near = F.interpolate(own[None, None], size=(height, width), mode="nearest")[0, 0][top:bottom, left:right]
        found.append(Region((top, bottom, left, right), scale, weight[top:bottom, left:right].float() * near))
    return found


def retouch_frame(frame: torch.Tensor, weight: torch.Tensor, even: float, texture: float, mode: str, size: float,
                  ceiling: bool = True):
    """Retouch the skin of one frame.

    Args:
        frame: ``(height, width, channels)`` picture codes, on the device to work on.
        weight: ``(height, width)`` skin mask on the same device.
        even: 0 to 1, how much blotchy tone is evened.
        texture: 0 to 1, how much fine texture is kept.
        mode: One of :data:`BLEMISHES`.
        size: Largest spot diameter, as a share of a region's size.
        ceiling: Whether changed pixels are held to 0 to 1.

    Returns:
        ``(frame, healed, count)``: the retouched frame, the ``(height, width)`` heal weights, and
        how many spots were healed.
    """
    height, width = int(frame.shape[0]), int(frame.shape[1])
    healed = torch.zeros((height, width), dtype=torch.float32, device=frame.device)
    out = frame.clone()
    total = 0
    for region in regions(weight.float().cpu()):
        top, bottom, left, right = region.box
        skin = region.weight.to(frame.device)[None]
        codes = frame[top:bottom, left:right, :3].float()
        lab = to_oklab(raw.linearise(codes)).permute(2, 0, 1)
        result = lab
        largest = size * region.scale
        if mode != "off":
            factor = min(1.0, SEARCH_SIZE / region.scale)
            if factor < 1.0:
                small = (max(8, round(lab.shape[1] * factor)), max(8, round(lab.shape[2] * factor)))
                search = F.interpolate(lab[None], size=small, mode="area")[0]
                search_skin = F.interpolate(skin[None], size=small, mode="area")[0]
                fy, fx = small[0] / lab.shape[1], small[1] / lab.shape[2]
            else:
                search, search_skin, fy, fx = lab, skin, 1.0, 1.0
            spots, kept = find_spots(search, search_skin, largest * min(fy, fx), mode)
            if spots.shape[0]:
                spots = torch.stack([(spots[:, 0] + 0.5) / fy - 0.5, (spots[:, 1] + 0.5) / fx - 0.5, spots[:, 2] / min(fy, fx)], dim=1)
            if kept.shape != lab.shape[1:]:
                kept = F.interpolate(kept[None, None].float(), size=tuple(lab.shape[1:]), mode="nearest")[0, 0] > 0
            result, heal_map, count = heal_spots(result, skin, spots, largest, kept)
            healed[top:bottom, left:right] = torch.maximum(healed[top:bottom, left:right], heal_map * (skin[0] > 0))
            total += count
        result = smooth_tone(result, skin, region.scale, even, texture)
        if result is lab:
            continue
        before = raw.encode(from_oklab(lab.permute(1, 2, 0)).clamp(min=0.0))
        after = raw.encode(from_oklab(result.permute(1, 2, 0)).clamp(min=0.0))
        change = (after - before) * skin[0][..., None]
        patch = out[top:bottom, left:right, :3]
        changed = patch + change.to(patch.dtype)
        if ceiling:
            changed = changed.clamp(0.0, 1.0)
        patch.copy_(torch.where(skin[0][..., None] > 0, changed, patch))
    return out, healed, total


def retouch(images: torch.Tensor, mask: torch.Tensor, even: float, texture: float, mode: str, size: float, advance=None):
    """Retouch the skin of every frame of a batch.

    Args:
        images: ``(batch, height, width, channels)`` picture codes on the CPU.
        mask: ``MASK`` of the skin, one per frame or one for all, any size.
        even: 0 to 1, how much blotchy tone is evened.
        texture: 0 to 1, how much fine texture is kept.
        mode: One of :data:`BLEMISHES`.
        size: Largest spot diameter, as a share of a region's size.
        advance: Called with 1 after each frame, or None.

    Returns:
        ``(images, healed, count)``: the retouched batch, ``(batch, height, width)`` heal weights,
        and how many spots were healed in all.
    """
    from . import scratch

    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    healed = torch.zeros((batch, height, width), dtype=torch.float32)
    if even <= 0.0 and texture >= 1.0 and mode == "off":
        return images, healed, 0
    weights = matte_for(images, mask)
    device = working_device()
    out = scratch.allocate(tuple(images.shape), images.dtype, node="Image Portrait Retouch", advice="Retouch fewer frames at a time.")
    held = not dynamic.carries(images)
    total = 0
    for index in range(batch):
        frame = images[index]
        weight = weights[index]
        if float(weight.max()) <= 0.0:
            out[index].copy_(frame)
        else:
            try:
                result, heal_map, count = retouch_frame(frame.to(device), weight.to(device), even, texture, mode, size, held)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if device.type == "cpu":
                    raise
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                result, heal_map, count = retouch_frame(frame, weight, even, texture, mode, size, held)
            out[index].copy_(result.to(out.device))
            healed[index] = heal_map.cpu()
            total += count
        scratch.trim(out[index])
        if advance is not None:
            advance(1)
    return out, healed, total
