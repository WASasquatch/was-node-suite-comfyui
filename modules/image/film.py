"""Film looks: a stock's tone, colour and grain, and the optics, light and wear of film.

Pictures are ``(batch, height, width, channels)`` float picture codes, a fourth channel carried
through. Sizes are fractions of a frame's long edge.
"""

from __future__ import annotations

import math

import torch

from . import dynamic, raw
from .accelerate import run_frames
from .develop import (
    contrast_curve,
    fit_lab,
    frame_by_frame,
    grade_lab,
    luma,
    mix_light,
    relight,
    saturate,
    smoothstep,
    soft_blur,
    to_oklab,
    white_balance,
)
from .film_stocks import BANDS_INDEX, NEUTRAL, Stock
from .grain import film_grain

__all__ = [
    "DIFFUSION_STYLES",
    "LEAK_POSITIONS",
    "LEAK_STYLES",
    "LENS_FILTERS",
    "PAPERS",
    "SPECK_TONES",
    "TONERS",
    "apply_stock",
    "darkroom_print",
    "diffuse",
    "film_damage",
    "grain_for",
    "halation",
    "light_leak",
    "parse_hex",
]

#: Long edge, in pixels, that a stock's grain size is written for.
GRAIN_EDGE = 2048

#: Grain strength and clump size at each film speed, at :data:`GRAIN_EDGE`.
GRAIN_BY_SPEED = (
    (25, 0.010, 1.00),
    (50, 0.012, 1.05),
    (100, 0.016, 1.15),
    (200, 0.021, 1.30),
    (400, 0.028, 1.50),
    (800, 0.036, 1.75),
    (1600, 0.046, 2.05),
    (3200, 0.058, 2.40),
)

#: Linear light a stock's shoulder starts bending at.
SHOULDER_KNEE = 0.75

#: Picture code a stock's halation starts at.
STOCK_GLOW_START = 0.85

#: Dust specks and hairs on one frame at full strength, whatever its resolution.
DUST_PER_FRAME = 400
HAIRS_PER_FRAME = 12

#: How far a mark at full strength moves a picture toward white or black.
MARK_OPACITY = 0.75

#: Diffusion filters, in the order the menu lists them.
DIFFUSION_STYLES = ("black mist", "white mist", "glimmer", "soft focus", "orton")

#: Light leak shapes and where they enter the frame.
LEAK_STYLES = ("edge glow", "corner flare", "end of roll", "streak", "prism")
LEAK_POSITIONS = (
    "random", "top", "bottom", "left", "right",
    "top left", "top right", "bottom left", "bottom right",
)

#: How specks, hairs and scratches print: light as on a print from a dusty negative, dark as
#: on a dusty slide, or some of each.
SPECK_TONES = ("light", "dark", "mixed")

#: Coloured lens filters for black and white, as the share of red, green and blue each passes.
LENS_FILTERS = {
    "none": (0.2126, 0.7152, 0.0722),
    "yellow": (0.30, 0.66, 0.04),
    "orange": (0.45, 0.52, 0.03),
    "red": (0.70, 0.29, 0.01),
    "green": (0.15, 0.80, 0.05),
    "blue": (0.10, 0.30, 0.60),
}

#: Toners, as the picture codes of a toned black, middle grey and white.
TONERS = {
    "none": None,
    "sepia": ((0.06, 0.045, 0.035), (0.58, 0.46, 0.34), (0.99, 0.95, 0.86)),
    "selenium": ((0.07, 0.045, 0.06), (0.50, 0.47, 0.49), (0.97, 0.97, 0.97)),
    "split": None,
    "gold": ((0.03, 0.04, 0.07), (0.45, 0.48, 0.56), (0.95, 0.97, 1.00)),
    "platinum": ((0.12, 0.10, 0.08), (0.56, 0.50, 0.43), (0.96, 0.93, 0.87)),
    "cyanotype": ((0.02, 0.12, 0.28), (0.18, 0.42, 0.66), (0.92, 0.95, 0.97)),
    "copper": ((0.12, 0.04, 0.02), (0.66, 0.38, 0.27), (0.99, 0.90, 0.84)),
    "blue": ((0.02, 0.04, 0.12), (0.35, 0.45, 0.62), (0.95, 0.97, 1.00)),
}

#: Paper bases, as the colour of the paper's white.
PAPERS = {
    "neutral white": (0.985, 0.985, 0.985),
    "warm fibre": (0.985, 0.955, 0.90),
    "cream": (0.98, 0.94, 0.84),
    "cool white": (0.95, 0.97, 1.00),
}


def parse_hex(text: str) -> tuple[float, float, float]:
    """Read a ``#rrggbb`` or ``#rgb`` colour.

    Args:
        text: The colour as typed.

    Returns:
        Three picture codes in 0 to 1.

    Raises:
        ValueError: The text is not a hexadecimal colour.
    """
    digits = str(text or "").strip().lstrip("#")
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        raise ValueError(f"'{text}' is not a hex colour such as #ff4b1f")
    return tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _planes(x: torch.Tensor) -> torch.Tensor:
    """``(batch, height, width, channels)`` to ``(batch, channels, height, width)``."""
    return x.permute(0, 3, 1, 2)


def _pixels(x: torch.Tensor) -> torch.Tensor:
    """``(batch, channels, height, width)`` to ``(batch, height, width, channels)``."""
    return x.permute(0, 2, 3, 1)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of a ``(batch, height, width, channels)`` tensor."""
    return _pixels(soft_blur(_planes(x), sigma))


def _long_edge(x: torch.Tensor) -> int:
    """The longer side of a ``(batch, height, width, channels)`` tensor."""
    return max(int(x.shape[1]), int(x.shape[2]))


def _rejoin(codes: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    """Put a run's untouched fourth channel back beside its new colour."""
    codes = codes.to(chunk.dtype)
    return torch.cat([codes, chunk[..., 3:]], dim=-1) if chunk.shape[-1] > 3 else codes


def grain_for(stock: Stock, long_edge: int) -> tuple[float, float]:
    """The grain strength and clump size a stock shows on a frame of a given size.

    Args:
        stock: The stock.
        long_edge: The frame's longer side in pixels.

    Returns:
        ``(strength, size)`` for :func:`modules.image.grain.film_grain`; strength 0 for a stock
        with no grain.
    """
    if stock.iso <= 0 or stock.grain <= 0.0:
        return 0.0, 0.0
    speed = math.log2(max(stock.iso, 1))
    points = [(math.log2(iso), strength, size) for iso, strength, size in GRAIN_BY_SPEED]
    if speed <= points[0][0]:
        strength, size = points[0][1], points[0][2]
    elif speed >= points[-1][0]:
        strength, size = points[-1][1], points[-1][2]
    else:
        for (x0, s0, z0), (x1, s1, z1) in zip(points, points[1:]):
            if x0 <= speed <= x1:
                t = (speed - x0) / (x1 - x0)
                strength, size = s0 + (s1 - s0) * t, z0 + (z1 - z0) * t
                break
    return strength * stock.grain, max(0.55, size * long_edge / GRAIN_EDGE)


def _curve_table(points, size: int = 1024, device=None) -> torch.Tensor:
    """A monotone cubic through five levels set at 0, 0.25, 0.5, 0.75 and 1.

    Args:
        points: Five output levels.
        size: Entries in the table.
        device: Where the table is built.

    Returns:
        A ``(size,)`` float32 table.
    """
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [float(y) for y in points]
    deltas = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(4)]
    tangents = [deltas[0]] + [(deltas[i - 1] + deltas[i]) / 2.0 for i in range(1, 4)] + [deltas[3]]
    for i, delta in enumerate(deltas):
        if delta == 0.0:
            tangents[i] = tangents[i + 1] = 0.0
            continue
        alpha, beta = tangents[i] / delta, tangents[i + 1] / delta
        if alpha * alpha + beta * beta > 9.0:
            scale = 3.0 / math.sqrt(alpha * alpha + beta * beta)
            tangents[i], tangents[i + 1] = scale * alpha * delta, scale * beta * delta
    grid = torch.linspace(0.0, 1.0, size, dtype=torch.float64)
    slot = (grid * 4.0).floor().clamp(max=3).long()
    x0 = slot.double() * 0.25
    t = (grid - x0) / 0.25
    y0 = torch.tensor(ys, dtype=torch.float64)[slot]
    y1 = torch.tensor(ys, dtype=torch.float64)[slot + 1]
    m0 = torch.tensor(tangents, dtype=torch.float64)[slot] * 0.25
    m1 = torch.tensor(tangents, dtype=torch.float64)[slot + 1] * 0.25
    t2, t3 = t * t, t * t * t
    table = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1
    return table.to(torch.float32).to(device)


def _through(v: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Read codes through a curve table, extending its last slope past 1."""
    size = table.numel()
    held = v.clamp(0.0, 1.0)
    scaled = held * (size - 1)
    low = scaled.floor().long().clamp(0, size - 2)
    mapped = torch.lerp(table[low], table[low + 1], scaled - low.to(scaled.dtype))
    slope = float(table[-1] - table[-2]) * (size - 1)
    return mapped + (v - held).clamp(min=0.0) * slope


def _shoulder(light: torch.Tensor, latitude: float) -> torch.Tensor:
    """Fold light above :data:`SHOULDER_KNEE` under white, holding ``latitude`` stops of it."""
    if latitude <= 0.0:
        return light
    level = luma(light)
    span = 1.0 - SHOULDER_KNEE
    reach = (2.0 ** latitude - SHOULDER_KNEE) / span
    over = ((level - SHOULDER_KNEE) / span).clamp(min=0.0)
    folded = over * (1.0 + over / (reach * reach)) / (1.0 + over)
    target = SHOULDER_KNEE + span * folded
    return torch.where(level > SHOULDER_KNEE, light * (target / level.clamp(min=1e-6)), light)


def _stock_light(light: torch.Tensor, stock: Stock, ceiling: bool) -> torch.Tensor:
    """Develop linear light through a stock's balance, curve and colour."""
    if stock.exposure:
        light = light * 2.0 ** stock.exposure
    if stock.temperature or stock.tint:
        light = white_balance(light, stock.temperature, stock.tint)
    if stock.matrix:
        light = (light @ torch.tensor(stock.matrix, dtype=light.dtype, device=light.device).T)
        light = light.clamp(min=0.0)
    light = _shoulder(light, stock.latitude)
    if stock.mono:
        weights = torch.tensor(stock.mono, dtype=light.dtype, device=light.device)
        light = (light @ (weights / weights.sum())).clamp(min=0.0).unsqueeze(-1).expand(
            *light.shape[:-1], 3
        ).contiguous()
    table = _curve_table(stock.curve, device=light.device)
    level = luma(light)
    floor = float(raw.linearise(table[:1].clamp(min=0.0)))
    toned = relight(light, level, _through(raw.encode(level), table), floor)
    if stock.per_channel > 0.0 and not stock.mono:
        channel = raw.linearise(_through(raw.encode(light), table).clamp(min=0.0))
        toned = torch.lerp(toned, channel, stock.per_channel)
    light = toned
    if stock.mono:
        return light
    if stock.saturation != 1.0:
        light = saturate(light, stock.saturation, ceiling)
    if stock.bands:
        hues, saturations, luminances = [0.0] * 8, [0.0] * 8, [0.0] * 8
        for band, hue, saturation, luminance in stock.bands:
            index = BANDS_INDEX[band]
            hues[index], saturations[index], luminances[index] = hue / 100, saturation / 100, luminance / 100
        light = mix_light(light, hues, saturations, luminances, ceiling)
    wheels = {
        tone: (float(hue), float(saturation) / 100.0, 0.0)
        for tone, (hue, saturation) in (
            ("shadows", stock.shadows), ("midtones", stock.midtones), ("highlights", stock.highlights)
        )
        if saturation
    }
    if wheels:
        light = fit_lab(grade_lab(to_oklab(light), wheels, 0.5, 0.0), ceiling)
    return light


def _halate(light: torch.Tensor, amount: float, radius: float, threshold: float, colour) -> torch.Tensor:
    """Spread a tinted glow out from the highlights of linear light.

    Args:
        light: ``(batch, height, width, 3)`` linear light.
        amount: How bright the glow is. 1.0 is strong.
        radius: How far it reaches, as a fraction of the long edge.
        threshold: Picture code the highlights start at.
        colour: The glow's colour as three picture codes.

    Returns:
        Linear light with the glow added.
    """
    if amount <= 0.0:
        return light
    level = luma(light)
    start = float(raw.linearise(torch.tensor(float(threshold))))
    energy = ((level - start) / max(1.0 - start, 0.05)).clamp(min=0.0)
    sigma = max(0.6, float(radius) * _long_edge(light))
    halo = 0.6 * _blur(energy, sigma) + 0.4 * _blur(energy, 3.0 * sigma)
    # The glow shows round a highlight rather than on it.
    halo = halo * (1.0 - smoothstep(0.85 * float(threshold), float(threshold), raw.encode(level)))
    tint = raw.linearise(torch.tensor(colour, dtype=light.dtype, device=light.device))
    tint = tint / tint.max().clamp(min=1e-6)
    return light + float(amount) * halo * tint


def apply_stock(images: torch.Tensor, stock: Stock, strength: float = 1.0, exposure: float = 0.0,
                grain: float = 1.0, halation_amount: float = 1.0, seed: int = 0) -> torch.Tensor:
    """Develop a batch as a film stock: its exposure, curve, colour, halation and grain.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        stock: The stock.
        strength: 0 = the picture as exposed, 1 = the stock's look, up to 2 to exaggerate it.
        exposure: Stops over (positive) or under (negative) the stock's own speed.
        grain: 0 = none, 1 = the stock's own grain, 2 = twice it.
        halation_amount: 0 = none, 1 = the stock's own halation.
        seed: Which grain pattern.

    Returns:
        The developed batch, or ``images`` itself when nothing applies.
    """
    strength = max(0.0, float(strength))
    exposure = float(exposure)
    glow = stock.halation * max(0.0, float(halation_amount))
    styled = stock is not NEUTRAL and strength > 0.0
    if images.numel() == 0 or not (styled or exposure or glow):
        out = images
    else:
        ceiling = not dynamic.carries(images)
        glow_colour = (1.0, 1.0, 1.0) if stock.mono else (1.0, 0.29, 0.12)

        def work(chunk: torch.Tensor) -> torch.Tensor:
            light = raw.linearise(chunk[..., :3].float())
            if exposure:
                light = light * 2.0 ** exposure
            base = raw.encode(light)
            if styled:
                looked = raw.encode(_stock_light(light, stock, ceiling))
                codes = base + (looked - base) * strength
            else:
                codes = base
            if glow:
                codes = raw.encode(_halate(raw.linearise(codes), glow, 0.006, STOCK_GLOW_START, glow_colour))
            return _rejoin(codes, chunk)

        out = run_frames(images, frame_by_frame(work), factor=20.0, node="Image Film Stock")
        out = dynamic.hold(out, images)
    strength_grain, size = grain_for(stock, _long_edge(images) if images.ndim == 4 else 0)
    if grain > 0.0 and strength_grain > 0.0 and images.numel():
        out = film_grain(out, strength_grain * float(grain), size, stock.grain_colour, int(seed))
    return out


def halation(images: torch.Tensor, amount: float = 0.5, radius: float = 1.0,
             threshold: float = 0.75, colour=(1.0, 0.29, 0.12)) -> torch.Tensor:
    """Add the red glow film forms around bright light.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        amount: 0 = none, 1 = strong, 2 = heavy.
        radius: How far it spreads, in hundredths of the long edge.
        threshold: Picture code the glow starts at.
        colour: The glow's colour as three picture codes.

    Returns:
        The batch with halation, or ``images`` itself at amount 0.
    """
    if amount <= 0.0 or images.numel() == 0:
        return images

    def work(chunk: torch.Tensor) -> torch.Tensor:
        light = raw.linearise(chunk[..., :3].float())
        return _rejoin(raw.encode(_halate(light, amount, radius / 100.0, threshold, colour)), chunk)

    return dynamic.hold(run_frames(images, frame_by_frame(work), factor=12.0, node="Image Halation"), images)


def _corner_mask(height: int, width: int, device, dtype) -> torch.Tensor:
    """0 at the centre rising to 1 at the corners, shaped to the frame."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, width)
    radius = torch.sqrt(xs * xs + ys * ys) / math.sqrt(2.0)
    return smoothstep(0.25, 1.0, radius).pow(1.5).view(1, height, width, 1)


def diffuse(images: torch.Tensor, style: str = "black mist", strength: float = 0.25,
            size: float = 1.0, corners: float = 0.0) -> torch.Tensor:
    """Soften a batch the way a diffusion filter or a soft lens does.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        style: One of :data:`DIFFUSION_STYLES`.
        strength: 0 = none, 0.125, 0.25, 0.5 and 1 match the filter grades sold.
        size: How far the glow spreads. 1 = a hundredth of the long edge.
        corners: 0 = sharp to the edges, 1 = the corners blurred as by a soft lens.

    Returns:
        The softened batch, or ``images`` itself when both strengths are 0.
    """
    if style not in DIFFUSION_STYLES:
        raise ValueError(f"unknown diffusion style '{style}'; pick one of {', '.join(DIFFUSION_STYLES)}")
    strength, corners = max(0.0, float(strength)), max(0.0, float(corners))
    if images.numel() == 0 or (strength == 0.0 and corners == 0.0):
        return images

    def work(chunk: torch.Tensor) -> torch.Tensor:
        light = raw.linearise(chunk[..., :3].float())
        sigma = max(0.6, 0.01 * float(size) * _long_edge(chunk))
        s = strength
        if s > 0.0:
            if style == "black mist":
                source = light * smoothstep(0.3, 0.95, raw.encode(luma(light))).pow(1.5)
                halo = 0.55 * _blur(source, sigma) + 0.45 * _blur(source, 4.0 * sigma)
                light = light - 0.35 * s * source + 0.9 * s * halo
            elif style == "white mist":
                halo = 0.5 * _blur(light, sigma) + 0.5 * _blur(light, 5.0 * sigma)
                light = light * (1.0 - 0.45 * s) + 0.45 * s * halo
            elif style == "glimmer":
                source = light * smoothstep(0.55, 1.0, raw.encode(luma(light))).pow(2.0)
                light = light + 0.8 * s * (0.7 * _blur(source, 0.35 * sigma) + 0.3 * _blur(source, 1.5 * sigma))
            elif style == "soft focus":
                codes = raw.encode(light)
                soft = raw.encode(_blur(light, 0.5 * sigma))
                codes = torch.lerp(codes, torch.maximum(codes, soft), min(s, 1.0))
                codes = torch.lerp(codes, soft, 0.25 * min(s, 1.0))
                light = raw.linearise(codes)
            else:
                codes = raw.encode(light).clamp(min=0.0)
                bright = 1.0 - (1.0 - codes.clamp(max=1.0)) ** 2
                dreamy = bright * _blur(bright, 0.8 * sigma)
                light = raw.linearise(torch.lerp(codes, dreamy, min(s, 1.0)))
        if corners > 0.0:
            mask = _corner_mask(int(chunk.shape[1]), int(chunk.shape[2]), light.device, light.dtype)
            light = torch.lerp(light, _blur(light, 1.5 * sigma), (corners * mask).clamp(max=1.0))
        return _rejoin(raw.encode(light.clamp(min=0.0)), chunk)

    return dynamic.hold(run_frames(images, frame_by_frame(work), factor=16.0, node="Image Diffusion Filter"), images)


def _smooth_noise(generator: torch.Generator, height: int, width: int, cells: int, device) -> torch.Tensor:
    """Smooth value noise in 0 to 1, ``cells`` lattice cells along the long edge."""
    long_edge = max(height, width)
    rows = max(2, round(cells * height / long_edge) + 1)
    columns = max(2, round(cells * width / long_edge) + 1)
    lattice = torch.rand((1, 1, rows, columns), generator=generator).to(device)
    field = torch.nn.functional.interpolate(lattice, size=(height, width), mode="bicubic", align_corners=True)
    return field.clamp(0.0, 1.0)[0, 0]


def _leak_field(generator, height, width, style, position, size, device, dtype):
    """A light leak's intensity and, for a prism, its hue, over one frame."""
    if position == "random":
        position = LEAK_POSITIONS[1 + int(torch.randint(0, len(LEAK_POSITIONS) - 1, (1,), generator=generator))]
    short = float(min(height, width))
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5).view(height, 1) / short
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5).view(1, width) / short
    far_x, far_y = width / short, height / short
    edges = {
        "top": ys.expand(height, width),
        "bottom": (far_y - ys).expand(height, width),
        "left": xs.expand(height, width),
        "right": (far_x - xs).expand(height, width),
    }
    corners = {
        "top left": (0.0, 0.0), "top right": (far_x, 0.0),
        "bottom left": (0.0, far_y), "bottom right": (far_x, far_y),
    }
    if position in edges:
        distance = edges[position]
        along = xs.expand(height, width) / far_x if position in ("top", "bottom") else ys.expand(height, width) / far_y
    else:
        cx, cy = corners[position]
        distance = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        along = torch.atan2(ys - cy, xs - cx) / math.pi
    coarse = _smooth_noise(generator, height, width, 4, device).to(dtype)
    fine = _smooth_noise(generator, height, width, 11, device).to(dtype)
    reach = 0.45 * max(float(size), 0.05)
    hue = None
    if style == "edge glow":
        field = torch.exp(-(distance / (reach * (0.6 + 0.8 * coarse))) ** 2) * (0.6 + 0.8 * fine)
    elif style == "corner flare":
        field = torch.exp(-((distance / (reach * (0.7 + 0.6 * coarse))).clamp(min=0.0) ** 1.6))
    elif style == "end of roll":
        edge = distance - 0.5 * reach * (coarse - 0.5)
        field = 1.0 - smoothstep(0.0, 1.2 * reach, edge)
        field = field * (0.75 + 0.5 * fine)
    elif style == "streak":
        centre = float(torch.rand((1,), generator=generator)) * 0.7 + 0.15
        width_share = 0.04 + 0.05 * float(torch.rand((1,), generator=generator))
        band = torch.exp(-(((along - centre) / width_share) ** 2))
        second = 0.5 * torch.exp(-(((along - centre - 2.5 * width_share) / (0.5 * width_share)) ** 2))
        field = (band + second) * torch.exp(-distance / (2.0 * reach)) * (0.7 + 0.6 * fine)
    else:
        ring = torch.exp(-(((distance - 0.8 * reach) / (0.28 * reach)) ** 2))
        field = ring * (0.6 + 0.8 * coarse)
        hue = ((distance - 0.5 * reach) / (0.6 * reach)).clamp(0.0, 1.0)
    return field.clamp(min=0.0), hue


def _rainbow(hue: torch.Tensor) -> torch.Tensor:
    """Picture codes of a spectrum running red to violet over 0 to 1."""
    turn = hue.unsqueeze(-1) * 0.8
    offsets = torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0], dtype=hue.dtype, device=hue.device)
    return (0.5 + 0.5 * torch.cos(2.0 * math.pi * (turn + offsets))).clamp(0.0, 1.0)


def light_leak(images: torch.Tensor, style: str = "edge glow", position: str = "random",
               colour=(1.0, 0.42, 0.17), amount: float = 0.6, size: float = 0.5,
               seed: int = 0, each_frame: bool = False) -> torch.Tensor:
    """Burn light leaking into the camera into a batch.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        style: One of :data:`LEAK_STYLES`.
        position: One of :data:`LEAK_POSITIONS`.
        colour: The leak's colour as three picture codes; a prism draws a spectrum instead.
        amount: 0 = none, 1 = bright, 2 = burned out.
        size: How far into the frame it reaches, as a share of the short side.
        seed: Which leak.
        each_frame: Whether every frame draws its own leak rather than sharing the first.

    Returns:
        The batch with the leak, or ``images`` itself at amount 0.
    """
    if style not in LEAK_STYLES:
        raise ValueError(f"unknown light leak style '{style}'; pick one of {', '.join(LEAK_STYLES)}")
    if position not in LEAK_POSITIONS:
        raise ValueError(f"unknown light leak position '{position}'; pick one of {', '.join(LEAK_POSITIONS)}")
    if amount <= 0.0 or images.numel() == 0:
        return images
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    tone = torch.tensor(colour, dtype=torch.float32)
    hot = torch.tensor((1.0, 0.93, 0.78), dtype=torch.float32)
    layers = []
    for index in range(batch if each_frame else 1):
        generator = torch.Generator(device="cpu").manual_seed((int(seed) + index) % (1 << 63))
        field, hue = _leak_field(generator, height, width, style, position, size, "cpu", torch.float32)
        f = field.unsqueeze(-1)
        tint = _rainbow(hue) if hue is not None else tone
        layers.append((tint * f + hot * f.pow(3) * 0.7).clamp(0.0, 1.0))
    out = images.clone()
    for index in range(batch):
        leak = layers[index if each_frame else 0].to(images.device)
        codes = images[index, :, :, :3].float()
        screened = 1.0 - (1.0 - codes.clamp(max=1.0)) * (1.0 - (float(amount) * leak).clamp(0.0, 1.0))
        out[index, :, :, :3] = (screened + (codes - codes.clamp(max=1.0))).to(out.dtype)
    return dynamic.hold(out, images)


def _splat(canvas: torch.Tensor, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> None:
    """Add weights at fractional positions into a ``(height, width)`` canvas, bilinearly."""
    height, width = canvas.shape
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    flat = canvas.view(-1)
    for dx, dy, share in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        cx, cy = (x0 + dx).long(), (y0 + dy).long()
        inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        flat.index_put_((cy[inside] * width + cx[inside],), (weight * share)[inside], accumulate=True)


def _peaked(canvas: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur splatted points and scale them so a single point peaks near 1."""
    if sigma < 0.3:
        return canvas.clamp(0.0, 1.0)
    blurred = soft_blur(canvas.view(1, 1, *canvas.shape), sigma)[0, 0]
    return (blurred * 2.0 * math.pi * sigma * sigma).clamp(0.0, 1.0)


def _shaped(layer: torch.Tensor, sigma: float, low: float = 0.25, high: float = 0.7) -> torch.Tensor:
    """Splatted points blurred and given a firm edge."""
    return smoothstep(low, high, _peaked(layer, sigma))


def _damage_masks(generator, height, width, dust, scratches, hairs, scratch_seed):
    """Masks of specks, hairs and scratches for one frame, each in 0 to 1."""
    scale = max(0.5, max(height, width) / GRAIN_EDGE)
    marks = torch.zeros((height, width))
    if dust > 0.0:
        for share, sigma, cluster in ((0.78, 0.55, 1), (0.18, 1.0, 3), (0.04, 1.8, 5)):
            count = int(round(dust * DUST_PER_FRAME * share))
            if count < 1:
                continue
            centre_x = torch.rand(count, generator=generator) * width
            centre_y = torch.rand(count, generator=generator) * height
            strength = 0.35 + 0.55 * torch.rand(count, generator=generator)
            spread = sigma * scale * 1.2
            layer = torch.zeros((height, width))
            for _ in range(cluster):
                offset_x = torch.randn(count, generator=generator) * spread if cluster > 1 else 0.0
                offset_y = torch.randn(count, generator=generator) * spread if cluster > 1 else 0.0
                _splat(layer, centre_x + offset_x, centre_y + offset_y, strength)
            marks = torch.maximum(marks, _shaped(layer, sigma * scale))
    if hairs > 0.0:
        count = max(1, int(round(hairs * HAIRS_PER_FRAME)))
        layer = torch.zeros((height, width))
        for _ in range(count):
            length = (40.0 + 180.0 * float(torch.rand((1,), generator=generator))) * scale
            steps = max(4, int(length / 0.5))
            turn = float(torch.rand((1,), generator=generator)) * 2.0 * math.pi
            angles = turn + torch.cumsum(torch.randn(steps, generator=generator) * 0.05, dim=0)
            start_x = float(torch.rand((1,), generator=generator)) * width
            start_y = float(torch.rand((1,), generator=generator)) * height
            fade = 0.4 + 0.6 * smoothstep(0.0, 0.5, _smooth_noise(generator, 1, steps, 6, "cpu")[0])
            _splat(layer, start_x + torch.cumsum(0.5 * torch.cos(angles), 0),
                   start_y + torch.cumsum(0.5 * torch.sin(angles), 0), 0.45 * fade)
        marks = torch.maximum(marks, _shaped(layer, 0.45 * scale, 0.15, 0.6) * 0.7)
    if scratches > 0.0:
        lines = torch.Generator(device="cpu").manual_seed(scratch_seed)
        count = max(1, int(round(scratches * 5.0)))
        layer = torch.zeros((height, width))
        rows = torch.arange(height, dtype=torch.float32)
        for line in range(count):
            x0 = (0.05 + 0.9 * float(torch.rand((1,), generator=lines))) * width
            wobble = 3.0 * scale * float(torch.rand((1,), generator=lines))
            cycles = 0.5 + 2.0 * float(torch.rand((1,), generator=lines))
            phase = 2.0 * math.pi * float(torch.rand((1,), generator=lines))
            drift = (float(torch.rand((1,), generator=lines)) - 0.5) * 0.02 * width
            jitter = float(torch.randn((1,), generator=generator)) * scale
            xs = x0 + jitter + wobble * torch.sin(2.0 * math.pi * cycles * rows / height + phase) + drift * rows / height
            breaks = _smooth_noise(generator, height, 8, 12, "cpu")[:, 0]
            strength = smoothstep(0.35, 0.55, breaks) * (0.3 + 0.5 * float(torch.rand((1,), generator=lines)))
            _splat(layer, xs, rows, strength * 2.0)
        marks = torch.maximum(marks, _peaked(layer, 0.5 * scale) * scratches)
    return marks


def film_damage(images: torch.Tensor, dust: float = 0.3, scratches: float = 0.0, hairs: float = 0.2,
                mottle: float = 0.0, tone: str = "light", seed: int = 0) -> torch.Tensor:
    """Lay dust, hairs, scratches and uneven density over a batch, as on old film.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        dust: 0 = none, 1 = heavy specking.
        scratches: 0 = none, 1 = five running scratches.
        hairs: 0 = none, 1 = about ten fibres per megapixel.
        mottle: 0 = none, 1 = strongly uneven density.
        tone: One of :data:`SPECK_TONES`.
        seed: Which marks. Every frame draws its own specks; scratches run through them all.

    Returns:
        The marked batch, or ``images`` itself when every amount is 0.
    """
    if tone not in SPECK_TONES:
        raise ValueError(f"unknown tone '{tone}'; pick one of {', '.join(SPECK_TONES)}")
    if images.numel() == 0 or not any(v > 0.0 for v in (dust, scratches, hairs, mottle)):
        return images
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    out = images.clone()
    for index in range(batch):
        generator = torch.Generator(device="cpu").manual_seed((int(seed) * 7919 + index) % (1 << 63))
        marks = _damage_masks(generator, height, width, dust, scratches, hairs, int(seed) % (1 << 63))
        codes = images[index, :, :, :3].float()
        if mottle > 0.0:
            field = sum(
                weight * (_smooth_noise(generator, height, width, cells, "cpu") - 0.5)
                for cells, weight in ((6, 0.5), (14, 0.3), (32, 0.2))
            )
            light = raw.linearise(codes) * (1.0 + 0.36 * float(mottle) * field.to(codes.device)).unsqueeze(-1)
            codes = raw.encode(light.clamp(min=0.0))
        marks = marks.to(codes.device).unsqueeze(-1)
        if tone == "mixed":
            split = (_smooth_noise(generator, height, width, 40, "cpu") > 0.5).to(codes.device).unsqueeze(-1)
            light_marks, dark_marks = marks * split, marks * (~split)
        elif tone == "light":
            light_marks, dark_marks = marks, torch.zeros_like(marks)
        else:
            light_marks, dark_marks = torch.zeros_like(marks), marks
        codes = codes + (1.0 - codes.clamp(max=1.0)) * light_marks * MARK_OPACITY
        codes = codes * (1.0 - MARK_OPACITY * dark_marks)
        out[index, :, :, :3] = codes.to(out.dtype)
    return dynamic.hold(out, images)


def _toned(grey: torch.Tensor, stops) -> torch.Tensor:
    """Picture codes of a toner's colour at each grey level, from its three stops."""
    dark, middle, light = (torch.tensor(s, dtype=grey.dtype, device=grey.device) for s in stops)
    low = (grey / 0.5).clamp(0.0, 1.0)
    high = ((grey - 0.5) / 0.5).clamp(0.0, 1.0)
    return torch.where(grey < 0.5, torch.lerp(dark, middle, low), torch.lerp(middle, light, high))


def darkroom_print(images: torch.Tensor, lens_filter: str = "none", paper_grade: float = 2.0,
                   exposure: float = 0.0, toner: str = "none", toner_amount: float = 0.6,
                   sabattier: float = 0.0, paper: str = "neutral white") -> torch.Tensor:
    """Print a batch in black and white as a darkroom would, then tone it.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        lens_filter: A key of :data:`LENS_FILTERS`, the coloured filter the scene was shot through.
        paper_grade: 0 = very soft, 2 = normal, 5 = very hard.
        exposure: Stops of print exposure. Positive prints darker, as more light does.
        toner: A key of :data:`TONERS`.
        toner_amount: 0 = untoned, 1 = fully toned.
        sabattier: 0 = none, 1 = the brightest tones turned back toward grey.
        paper: A key of :data:`PAPERS`.

    Returns:
        The printed batch, always three colour channels plus any fourth carried through.
    """
    for value, table, name in ((lens_filter, LENS_FILTERS, "lens filter"), (toner, TONERS, "toner"),
                               (paper, PAPERS, "paper")):
        if value not in table:
            raise ValueError(f"unknown {name} '{value}'; pick one of {', '.join(table)}")
    if images.numel() == 0:
        return images

    def work(chunk: torch.Tensor) -> torch.Tensor:
        light = raw.linearise(chunk[..., :3].float())
        weights = torch.tensor(LENS_FILTERS[lens_filter], dtype=light.dtype, device=light.device)
        level = (light @ (weights / weights.sum())).clamp(min=0.0).unsqueeze(-1)
        grey = raw.encode(level * 2.0 ** (-float(exposure)))
        grey = contrast_curve(grey.clamp(0.0, 1.0), max(-1.0, min(1.0, (float(paper_grade) - 2.0) / 3.0)))
        if sabattier > 0.0:
            ramp = torch.linspace(0.0, 1.0, 256, dtype=grey.dtype, device=grey.device)
            reversal = float(sabattier) * 1.3
            peak = float((ramp - reversal * (ramp - 0.42).clamp(min=0.0) ** 1.15).max())
            # The brightest reversed tone is scaled back up to paper white.
            grey = (grey - reversal * (grey - 0.42).clamp(min=0.0) ** 1.15) / max(peak, 1e-3)
        grey = (0.012 + 0.988 * grey).clamp(0.0, 1.0)
        codes = grey.expand(*grey.shape[:-1], 3)
        if toner != "none" and toner_amount > 0.0:
            if toner == "split":
                warm = _toned(grey, TONERS["sepia"])
                cold = _toned(grey, TONERS["selenium"])
                toned = torch.lerp(cold, warm, smoothstep(0.25, 0.65, grey))
                reach = torch.ones_like(grey)
            else:
                toned = _toned(grey, TONERS[toner])
                if toner == "sepia":
                    reach = smoothstep(0.1, 0.6, grey)
                elif toner == "selenium":
                    reach = 1.0 - smoothstep(0.3, 0.8, grey)
                elif toner == "gold":
                    reach = smoothstep(0.3, 0.9, grey)
                else:
                    reach = torch.ones_like(grey)
            codes = torch.lerp(codes, toned, (float(toner_amount) * reach).clamp(0.0, 1.0))
        base = torch.tensor(PAPERS[paper], dtype=codes.dtype, device=codes.device)
        return _rejoin(codes * base, chunk)

    return run_frames(images, frame_by_frame(work), factor=10.0, node="Image Darkroom Print").clamp(0.0, 1.0)
