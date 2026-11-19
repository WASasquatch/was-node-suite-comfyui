"""Raw-editor development: tone, presence, a colour mixer and colour grading.

Pictures are ``(batch, height, width, channels)`` float picture codes, a fourth channel carried
through. Sliders run -100 to 100, 0 changing nothing.
"""

from __future__ import annotations

import colorsys
import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from . import dynamic, raw
from .accelerate import run_frames
from .convolve import gaussian_blur
from .guided import filter_with_guide

__all__ = [
    "BANDS",
    "BASIC",
    "GRADE_TONES",
    "band_weights",
    "contrast_curve",
    "develop",
    "fit_gamut",
    "fit_lab",
    "frame_by_frame",
    "from_oklab",
    "grade",
    "grade_lab",
    "hue_direction",
    "luma",
    "masked",
    "matte_for",
    "mix",
    "mix_light",
    "natural_finish",
    "relight",
    "saturate",
    "smoothstep",
    "soft_blur",
    "spotlight",
    "to_oklab",
    "white_balance",
]

#: The eight colour ranges a mixer works on, in hue order.
BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")

#: Picture codes of each band's centre colour.
_BAND_CODES = (
    (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 1.0),
    (1.0, 0.0, 1.0),
)

#: The basic and presence sliders, in the order they are applied.
BASIC = (
    "temperature",
    "tint",
    "exposure",
    "dehaze",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "contrast",
    "clarity",
    "texture",
    "vibrance",
    "saturation",
)

#: The tonal ranges colour grading tints, then the one that tints everything.
GRADE_TONES = ("shadows", "midtones", "highlights", "global")

#: Rec. 709 weights for the brightness of linear light.
LUMA = (0.2126, 0.7152, 0.0722)

#: Long edge, in pixels, that the pixel radii here are written for.
REFERENCE_EDGE = 2048

#: Picture code of 18% grey, the pivot contrast turns about.
MID_GREY = 0.46

#: Furthest a mixer hue slider turns a band, in degrees.
MIX_HUE_REACH = 30.0

#: Stops a mixer luminance slider moves a fully coloured band at 100.
MIX_LUMINANCE_STOPS = 0.8

#: Oklab chroma a tonal grading wheel adds at saturation 100, and the global wheel.
GRADE_REACH = 0.07
GRADE_GLOBAL_REACH = 0.045

#: Oklab lightness a grading luminance slider moves at 100.
GRADE_LIGHTNESS = 0.12

#: Halvings of chroma tried when bringing a colour back inside the gamut.
GAMUT_STEPS = 12

_TO_LMS = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_LMS_TO_LAB = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_LAB_TO_LMS = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_LMS_TO_RGB = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


def smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Hermite step from 0 at ``edge0`` to 1 at ``edge1``.

    Args:
        edge0: Where the step starts.
        edge1: Where it reaches 1. May sit below ``edge0`` for a falling step.
        x: Values to step.

    Returns:
        A tensor shaped like ``x`` in 0 to 1.
    """
    span = edge1 - edge0
    t = ((x - edge0) / (span if abs(span) > 1e-9 else 1e-9)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _matrix(x: torch.Tensor, rows) -> torch.Tensor:
    """Multiply a ``(..., 3)`` tensor by a 3 by 3 matrix given as rows."""
    return x @ torch.tensor(rows, dtype=x.dtype, device=x.device).T


def luma(light: torch.Tensor) -> torch.Tensor:
    """Rec. 709 brightness of linear light, keeping a trailing axis of one.

    Args:
        light: ``(..., 3)`` linear light.

    Returns:
        A ``(..., 1)`` tensor.
    """
    return _matrix(light, (LUMA,))


def to_oklab(light: torch.Tensor) -> torch.Tensor:
    """Oklab lightness and opponent axes of linear sRGB light.

    Args:
        light: ``(..., 3)`` linear light.

    Returns:
        ``(..., 3)`` holding L, a and b.
    """
    lms = _matrix(light, _TO_LMS)
    return _matrix(lms.sign() * lms.abs().pow(1.0 / 3.0), _LMS_TO_LAB)


def from_oklab(lab: torch.Tensor) -> torch.Tensor:
    """Linear sRGB light from Oklab.

    Args:
        lab: ``(..., 3)`` holding L, a and b.

    Returns:
        ``(..., 3)`` linear light, possibly outside the gamut.
    """
    lms = _matrix(lab, _LAB_TO_LMS)
    return _matrix(lms * lms * lms, _LMS_TO_RGB)


def _oklab_of_code(code) -> tuple[float, float, float]:
    """Oklab of one picture-code colour, in plain floats."""
    light = []
    for value in code:
        value = float(value)
        light.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    lms = [sum(row[i] * light[i] for i in range(3)) for row in _TO_LMS]
    lms = [math.copysign(abs(v) ** (1.0 / 3.0), v) for v in lms]
    return tuple(sum(row[i] * lms[i] for i in range(3)) for row in _LMS_TO_LAB)


@lru_cache(maxsize=1)
def band_hues() -> tuple[float, ...]:
    """Oklab hue angle, in radians, of each band's centre colour.

    Returns:
        Eight angles in :data:`BANDS` order.
    """
    return tuple(math.atan2(lab[2], lab[1]) for lab in map(_oklab_of_code, _BAND_CODES))


@lru_cache(maxsize=720)
def hue_direction(degrees: float) -> tuple[float, float]:
    """The Oklab a and b direction of a colour-wheel hue.

    Args:
        degrees: Hue as a colour picker names it. 0 = red, 120 = green, 240 = blue.

    Returns:
        A unit ``(a, b)`` pair.
    """
    red, green, blue = colorsys.hsv_to_rgb((float(degrees) % 360.0) / 360.0, 1.0, 1.0)
    _, a, b = _oklab_of_code((red, green, blue))
    norm = math.hypot(a, b) or 1.0
    return a / norm, b / norm


def band_weights(hue: torch.Tensor) -> torch.Tensor:
    """How far each hue belongs to each of the eight bands.

    Args:
        hue: Oklab hue angles in radians, any shape.

    Returns:
        ``hue.shape + (8,)``, each row summing to 1 and easing between neighbouring bands.
    """
    centres = torch.tensor(band_hues(), dtype=hue.dtype, device=hue.device)
    turn = 2.0 * math.pi
    right = torch.remainder(torch.roll(centres, -1) - centres, turn)
    left = torch.remainder(centres - torch.roll(centres, 1), turn)
    offset = torch.remainder(hue.unsqueeze(-1) - centres + math.pi, turn) - math.pi
    reach = torch.where(offset >= 0, offset / right, -offset / left).clamp(max=1.0)
    return torch.cos(0.5 * math.pi * reach) ** 2


def fit_gamut(light: torch.Tensor, grey: torch.Tensor, ceiling: bool) -> torch.Tensor:
    """Bring colours back inside the gamut by draining them toward a grey of their own level.

    Args:
        light: ``(..., 3)`` linear light.
        grey: ``(..., 1)`` linear grey level each colour is drained toward.
        ceiling: Whether 1.0 is a limit as well as 0.0.

    Returns:
        ``light`` with every out-of-range colour moved toward ``grey`` until it fits.
    """
    grey = grey.clamp(min=0.0)
    keep = torch.ones_like(grey)
    low = light.amin(dim=-1, keepdim=True)
    under = low < 0.0
    keep = torch.where(under, torch.minimum(keep, grey / (grey - low).clamp(min=1e-9)), keep)
    if ceiling:
        high = light.amax(dim=-1, keepdim=True)
        over = high > 1.0
        room = (1.0 - grey).clamp(min=0.0)
        keep = torch.where(over, torch.minimum(keep, room / (high - grey).clamp(min=1e-9)), keep)
    return grey + (light - grey) * keep.clamp(0.0, 1.0)


def fit_lab(lab: torch.Tensor, ceiling: bool) -> torch.Tensor:
    """Linear light from Oklab, out-of-gamut colours losing chroma at their own hue and lightness.

    Args:
        lab: ``(..., 3)`` Oklab.
        ceiling: Whether 1.0 is a limit as well as 0.0.

    Returns:
        ``(..., 3)`` linear light, never negative.
    """
    light = from_oklab(lab)
    bad = light.amin(dim=-1) < -1e-5
    if ceiling:
        bad = bad | (light.amax(dim=-1) > 1.0 + 1e-5)
    if not bool(bad.any()):
        return light.clamp(min=0.0)
    picked = lab[bad]
    low = torch.zeros(picked.shape[0], dtype=lab.dtype, device=lab.device)
    high = torch.ones_like(low)
    for _ in range(GAMUT_STEPS):
        middle = 0.5 * (low + high)
        trial = from_oklab(torch.cat([picked[:, :1], picked[:, 1:] * middle[:, None]], dim=-1))
        fits = trial.amin(dim=-1) >= 0.0
        if ceiling:
            fits = fits & (trial.amax(dim=-1) <= 1.0)
        low = torch.where(fits, middle, low)
        high = torch.where(fits, high, middle)
    light = light.clone()
    light[bad] = from_oklab(torch.cat([picked[:, :1], picked[:, 1:] * low[:, None]], dim=-1))
    return light.clamp(min=0.0)


def soft_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """A gaussian blur that stays cheap at a wide radius by working on a reduced copy.

    Args:
        x: ``(batch, channels, height, width)``.
        sigma: Standard deviation in pixels of ``x``.

    Returns:
        A tensor shaped like ``x``.
    """
    if sigma < 0.5:
        return x
    height, width = int(x.shape[-2]), int(x.shape[-1])
    factor = max(1, int(sigma // 3.0))
    if factor > 1 and min(height, width) // factor >= 4:
        small = F.interpolate(x, size=(height // factor, width // factor), mode="area")
        small = gaussian_blur(small, sigma=sigma / factor)
        return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    return gaussian_blur(x, sigma=sigma)


def frame_by_frame(work):
    """Wrap a per-frame operation so a run of frames is worked one frame at a time.

    Args:
        work: A callable taking ``(frames, height, width, channels)`` and answering one row per frame.

    Returns:
        A callable of the same shape whose answer for each frame matches a run of that frame alone.
    """

    def run(chunk: torch.Tensor) -> torch.Tensor:
        if int(chunk.shape[0]) <= 1:
            return work(chunk)
        return torch.cat([work(chunk[index:index + 1]) for index in range(int(chunk.shape[0]))], dim=0)

    return run


def _planes(x: torch.Tensor) -> torch.Tensor:
    """``(batch, height, width, channels)`` to ``(batch, channels, height, width)``."""
    return x.permute(0, 3, 1, 2)


def _pixels(x: torch.Tensor) -> torch.Tensor:
    """``(batch, channels, height, width)`` to ``(batch, height, width, channels)``."""
    return x.permute(0, 2, 3, 1)


def white_balance(light: torch.Tensor, temperature: float, tint: float) -> torch.Tensor:
    """Warm or cool, and move toward magenta or green, holding brightness.

    Args:
        light: ``(..., 3)`` linear light.
        temperature: -100 = blue, 100 = amber.
        tint: -100 = green, 100 = magenta.

    Returns:
        The balanced light.
    """
    warm, pink = float(temperature) / 100.0, float(tint) / 100.0
    gains = (
        2.0 ** (0.42 * warm + 0.12 * pink),
        2.0 ** (-0.30 * pink),
        2.0 ** (-0.55 * warm + 0.12 * pink),
    )
    norm = sum(weight * gain for weight, gain in zip(LUMA, gains))
    scale = torch.tensor([gain / norm for gain in gains], dtype=light.dtype, device=light.device)
    return light * scale


def contrast_curve(v: torch.Tensor, amount: float, pivot: float = MID_GREY) -> torch.Tensor:
    """An S-curve about a pivot that holds black and white where they are.

    Args:
        v: Picture codes. Anything above 1 is carried past the curve unchanged.
        amount: -1 = flat, 0 = unchanged, 1 = twice as steep at the pivot.
        pivot: The code the curve turns about.

    Returns:
        A tensor shaped like ``v``.
    """
    if amount == 0.0:
        return v
    x = v.clamp(0.0, 1.0)
    beyond = v - x
    if amount > 0.0:
        power = 1.0 + float(amount)
        low = pivot * (x / pivot).clamp(min=0.0) ** power
        high = 1.0 - (1.0 - pivot) * ((1.0 - x) / (1.0 - pivot)).clamp(min=0.0) ** power
        shaped = torch.where(x < pivot, low, high)
    else:
        shaped = x + float(amount) * 1.6 * (x - pivot) * x * (1.0 - x)
    return shaped + beyond


def relight(light: torch.Tensor, level: torch.Tensor, codes: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Scale colours so their brightness becomes a new picture-code level.

    Args:
        light: ``(..., 3)`` linear light.
        level: ``(..., 1)`` the brightness ``light`` has now.
        codes: ``(..., 1)`` the brightness wanted, as picture codes.
        floor: Linear level a curve lifts black to. It is added as grey, fading out by four
            times its own level, so dark colours keep their chroma.

    Returns:
        ``(..., 3)`` linear light at the wanted brightness. A colour too dark to scale that far
        is scaled 64 times and the rest of the brightness is added as grey.
    """
    target = raw.linearise(codes.clamp(min=0.0))
    held = level.clamp(min=0.0)
    fog = 0.0
    if floor > 0.0:
        fog = floor * (1.0 - held / (4.0 * floor)).clamp(min=0.0) ** 2
    lift = (target - fog).clamp(min=0.0)
    ratio = (lift / held.clamp(min=1e-12)).clamp(max=64.0)
    return light * ratio + (lift - held * ratio).clamp(min=0.0) + fog


def _long_edge(x: torch.Tensor) -> int:
    """The longer side of a ``(batch, height, width, channels)`` tensor."""
    return max(int(x.shape[1]), int(x.shape[2]))


def _dehaze(light: torch.Tensor, amount: float) -> torch.Tensor:
    """Remove haze with a dark-channel estimate, or lay a veil of it over the frame.

    Args:
        light: ``(batch, height, width, 3)`` linear light.
        amount: -1 = add a heavy veil, 1 = remove most of the haze.

    Returns:
        Linear light, never negative.
    """
    batch, height, width, _ = light.shape
    planes = _planes(light)
    shrink = min(1.0, 384.0 / max(height, width))
    small = planes
    if shrink < 1.0:
        small = F.interpolate(
            planes, size=(max(4, round(height * shrink)), max(4, round(width * shrink))), mode="area"
        )
    dark = -F.max_pool2d(-small.amin(dim=1, keepdim=True), 9, 1, 4)
    flat = dark.flatten(1)
    count = max(1, int(0.005 * flat.shape[1]))
    brightest = flat.topk(count, dim=1).indices
    colours = small.flatten(2)
    air = torch.gather(colours, 2, brightest.unsqueeze(1).expand(-1, 3, -1)).mean(dim=2)
    air = air.clamp(min=0.05)
    if amount < 0.0:
        veil = -amount * 0.5
        return light * (1.0 - veil) + air.view(batch, 1, 1, 3) * veil
    normalised = small / air.view(batch, 3, 1, 1)
    carried = 1.0 - 0.95 * (-F.max_pool2d(-normalised.amin(dim=1, keepdim=True), 9, 1, 4))
    carried = F.interpolate(carried, size=(height, width), mode="bilinear", align_corners=False)
    guide = raw.encode(luma(light)).clamp(0.0, 1.0)
    carried = filter_with_guide(
        _pixels(carried), guide, max(2, round(0.02 * max(height, width))), 1e-3
    )
    carried = 1.0 - amount * (1.0 - carried.clamp(0.0, 1.0))
    air = air.view(batch, 1, 1, 3)
    return ((light - air) / carried.clamp(min=0.15) + air).clamp(min=0.0)


def _tone(v: torch.Tensor, values: dict, long_edge: int) -> torch.Tensor:
    """Run the tone sliders over a brightness plane of picture codes.

    Args:
        v: ``(batch, height, width, 1)`` codes.
        values: Slider values, -1 to 1, keyed as :data:`BASIC` names them.
        long_edge: The frame's longer side, which local radii scale with.

    Returns:
        The toned plane.
    """
    highlights, shadows = values["highlights"], values["shadows"]
    if highlights or shadows:
        radius = max(4, round(0.025 * long_edge))
        base = filter_with_guide(v, v, radius, 0.015)
        detail = v - base
        if highlights:
            weight = smoothstep(0.35, 1.0, base)
            if highlights < 0.0:
                base = base * (1.0 + 0.32 * highlights * weight)
            else:
                base = base + highlights * 0.35 * weight * (1.05 - base.clamp(max=1.0))
        if shadows:
            held = base.clamp(min=0.0)
            weight = 1.0 - smoothstep(0.0, 0.65, held)
            if shadows > 0.0:
                base = base + shadows * 0.32 * weight * held / (held + 0.06)
            else:
                base = base + shadows * 0.22 * weight * held
        v = base + detail
    if values["whites"]:
        v = v + values["whites"] * 0.18 * smoothstep(0.45, 1.0, v)
    if values["blacks"]:
        v = v + values["blacks"] * 0.10 * (1.0 - smoothstep(0.0, 0.5, v))
    v = contrast_curve(v, values["contrast"])
    if values["clarity"]:
        radius = max(3, round(0.012 * long_edge))
        held = v.clamp(0.0, 1.0)
        middle = (4.0 * held * (1.0 - held)).clamp(0.0, 1.0).sqrt()
        v = v + values["clarity"] * 0.9 * (v - filter_with_guide(held, held, radius, 0.01)) * middle
    if values["texture"]:
        sigma = max(0.8, 2.0 * long_edge / REFERENCE_EDGE)
        smooth = _pixels(gaussian_blur(_planes(v), sigma=sigma))
        v = v + values["texture"] * (v - smooth)
    return v


def _develop_frames(chunk: torch.Tensor, values: dict, ceiling: bool) -> torch.Tensor:
    """Develop one run of frames on whatever device it arrived on."""
    rgb = chunk[..., :3].float()
    light = raw.linearise(rgb)
    if values["temperature"] or values["tint"]:
        light = white_balance(light, values["temperature"] * 100.0, values["tint"] * 100.0)
    if values["exposure"]:
        light = light * 2.0 ** values["exposure"]
    if values["dehaze"]:
        light = _dehaze(light, values["dehaze"])
    tonal = ("highlights", "shadows", "whites", "blacks", "contrast", "clarity", "texture")
    if any(values[name] for name in tonal):
        level = luma(light)
        floor = float(raw.linearise(torch.tensor(0.10 * values["blacks"]))) if values["blacks"] > 0.0 else 0.0
        light = relight(light, level, _tone(raw.encode(level), values, _long_edge(chunk)), floor)
    if values["vibrance"] or values["saturation"]:
        factor = torch.full_like(light[..., :1], 1.0 + values["saturation"])
        vibrance = values["vibrance"]
        if vibrance:
            codes = raw.encode(light)
            room = 1.0 - ((codes.amax(-1, keepdim=True) - codes.amin(-1, keepdim=True)) / 0.6).clamp(0.0, 1.0)
            if vibrance > 0.0:
                lab = to_oklab(light)
                weights = band_weights(torch.atan2(lab[..., 2], lab[..., 1]))
                skin = 0.6 * weights[..., 1:2] + 0.2 * weights[..., 0:1]
                factor = factor * (1.0 + vibrance * room * (1.0 - skin))
            else:
                factor = factor * (1.0 + vibrance * (0.5 + 0.5 * room))
        light = saturate(light, factor.clamp(min=0.0), ceiling)
    elif ceiling:
        light = fit_gamut(light, luma(light), True)
    codes = raw.encode(light).to(chunk.dtype)
    return torch.cat([codes, chunk[..., 3:]], dim=-1) if chunk.shape[-1] > 3 else codes


def develop(images: torch.Tensor, **sliders: float) -> torch.Tensor:
    """Develop a batch with the basic and presence sliders of a raw editor.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        **sliders: Any of :data:`BASIC`. ``exposure`` is in stops, -5 to 5; the rest run
            -100 to 100.

    Returns:
        The developed batch, or ``images`` itself when every slider is 0.
    """
    values = {name: float(sliders.get(name, 0.0) or 0.0) for name in BASIC}
    if not any(values.values()) or images.numel() == 0:
        return images
    for name in BASIC:
        if name != "exposure":
            values[name] /= 100.0
    ceiling = not dynamic.carries(images)
    out = run_frames(
        images, frame_by_frame(lambda chunk: _develop_frames(chunk, values, ceiling)), factor=24.0,
        node="Image Develop", advice="Develop fewer frames at a time.",
    )
    return dynamic.hold(out, images)


def matte_for(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """A mask brought to a batch's size and length.

    Args:
        images: ``(batch, height, width, channels)``.
        mask: ``(height, width)`` or ``(frames, height, width)``, any size. Frame ``i`` of the batch
            takes mask ``i``, or the last mask once they run out.

    Returns:
        ``(batch, height, width)`` floats in 0 to 1 on the CPU.
    """
    held = mask.float().cpu()
    if held.ndim == 2:
        held = held[None]
    if held.ndim == 4:
        held = held[..., 0] if held.shape[-1] == 1 else held[:, 0]
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    if tuple(held.shape[1:]) != (height, width):
        held = F.interpolate(held[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    order = torch.clamp(torch.arange(batch), max=int(held.shape[0]) - 1)
    return held[order].clamp(0.0, 1.0)


def masked(original: torch.Tensor, result: torch.Tensor, mask) -> torch.Tensor:
    """Keep an adjustment only where a mask is white.

    Args:
        original: The batch before the adjustment.
        result: The batch after it.
        mask: A ``MASK``, or None to keep the whole adjustment.

    Returns:
        ``result`` where the mask is 1, ``original`` where it is 0, mixed between.
    """
    if mask is None or result is original:
        return result
    weight = matte_for(original, mask).to(result.device, result.dtype)[..., None]
    out = result.clone()
    out[..., :3] = original[..., :3].to(result.dtype) + (result[..., :3] - original[..., :3].to(result.dtype)) * weight
    return out


def _light_field(matte: torch.Tensor, spill: float, softness: float) -> torch.Tensor:
    """Where a subject's light falls: full on the subject, fading over ``spill`` around it.

    Args:
        matte: ``(batch, height, width)`` subject matte.
        spill: How far the light reaches past the subject, in hundredths of the short side.
        softness: How soft the subject's own edge is, in hundredths of the short side.

    Returns:
        ``(batch, height, width)`` in 0 to 1.
    """
    short = min(int(matte.shape[1]), int(matte.shape[2]))
    planes = matte[:, None]
    edge = float(softness) / 100.0 * short
    inner = soft_blur(planes, edge) if edge >= 0.5 else planes
    if spill <= 0.0:
        return inner[:, 0].clamp(0.0, 1.0)
    halo = (2.0 * soft_blur(planes, max(0.5, float(spill) / 100.0 * short * 0.5))).clamp(0.0, 1.0)
    return torch.maximum(inner, halo)[:, 0].clamp(0.0, 1.0)


def spotlight(images: torch.Tensor, mask: torch.Tensor, subject: dict, surround: dict,
              spill: float = 0.0, softness: float = 0.3):
    """Light a subject and hold back what surrounds it.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        mask: The subject's ``MASK``, one per frame or one for all.
        subject: :data:`BASIC` sliders for the subject, as :func:`develop` takes them.
        surround: :data:`BASIC` sliders for everything else.
        spill: How far the subject's light reaches past it, in hundredths of the short side.
        softness: How soft the subject's own edge is, in hundredths of the short side.

    Returns:
        ``(images, light)``: the lit batch, and where the subject's light fell as a ``MASK``.
    """
    matte = matte_for(images, mask)
    if images.numel() == 0:
        return images, matte

    def scaled(sliders):
        values = {name: float(sliders.get(name, 0.0) or 0.0) for name in BASIC}
        for name in BASIC:
            if name != "exposure":
                values[name] /= 100.0
        return values

    lit, held = scaled(subject), scaled(surround)
    if not any(lit.values()) and not any(held.values()):
        return images, _light_field(matte, spill, softness)
    ceiling = not dynamic.carries(images)
    channels = int(images.shape[-1])

    def work(chunk: torch.Tensor) -> torch.Tensor:
        frame = chunk[..., :channels]
        light = _light_field(chunk[..., channels], spill, softness)[..., None]
        bright = raw.linearise(_develop_frames(frame, lit, ceiling)[..., :3].float()) if any(lit.values()) else raw.linearise(frame[..., :3].float())
        dim = raw.linearise(_develop_frames(frame, held, ceiling)[..., :3].float()) if any(held.values()) else raw.linearise(frame[..., :3].float())
        codes = raw.encode(dim + (bright - dim) * light).to(chunk.dtype)
        parts = [codes, frame[..., 3:]] if channels > 3 else [codes]
        return torch.cat(parts + [light.to(chunk.dtype)], dim=-1)

    joined = torch.cat([images, matte.to(images.device, images.dtype)[..., None]], dim=-1)
    out = run_frames(joined, frame_by_frame(work), factor=30.0, node="Image Spotlight",
                     advice="Light fewer frames at a time.")
    result = out[..., :channels]
    if ceiling:
        result = result.clamp(0.0, 1.0)
    return result, out[..., channels].clamp(0.0, 1.0)


def saturate(light: torch.Tensor, factor, ceiling: bool) -> torch.Tensor:
    """Move colours away from or toward grey, measured in picture codes.

    Args:
        light: ``(..., 3)`` linear light.
        factor: 0 = grey, 1 = unchanged, 2 = twice the colour. A number or a ``(..., 1)`` tensor.
        ceiling: Whether 1.0 is a limit as well as 0.0.

    Returns:
        ``(..., 3)`` linear light.
    """
    codes = raw.encode(light)
    grey = _matrix(codes, (LUMA,))
    return raw.linearise(fit_gamut(grey + (codes - grey) * factor, grey, ceiling))


def mix_light(light: torch.Tensor, hues, saturations, luminances, ceiling: bool) -> torch.Tensor:
    """Turn, saturate and brighten each of the eight colour bands of linear light.

    Args:
        light: ``(..., 3)`` linear light.
        hues: Eight values, -1 to 1, each turning its band up to :data:`MIX_HUE_REACH` degrees.
        saturations: Eight values, -1 = grey to 1 = twice the colour.
        luminances: Eight values, -1 = darker to 1 = brighter.
        ceiling: Whether 1.0 is a limit as well as 0.0.

    Returns:
        ``(..., 3)`` linear light.
    """
    lab = to_oklab(light)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = torch.hypot(a, b)
    hue = torch.atan2(b, a)
    weights = band_weights(hue)

    def spread(values) -> torch.Tensor:
        return weights @ torch.tensor([float(v) for v in values], dtype=lab.dtype, device=lab.device)

    if any(hues) or any(luminances):
        if any(hues):
            hue = hue + spread(hues) * math.radians(MIX_HUE_REACH) * smoothstep(0.0, 0.05, chroma)
        if any(luminances):
            stops = spread(luminances) * MIX_LUMINANCE_STOPS * smoothstep(0.0, 0.1, chroma)
            lightness = lightness * 2.0 ** stops
        lab = torch.stack([lightness, chroma * torch.cos(hue), chroma * torch.sin(hue)], dim=-1)
        light = fit_lab(lab, ceiling)
    if any(saturations):
        light = saturate(light, (1.0 + spread(saturations)).clamp(min=0.0).unsqueeze(-1), ceiling)
    return light


def _mono_lab(lab: torch.Tensor, luminances) -> torch.Tensor:
    """Grey Oklab, each band rendered brighter or darker by its slider.

    Args:
        lab: ``(..., 3)`` Oklab.
        luminances: Eight values, -1 = the band goes dark, 1 = it goes light.

    Returns:
        ``(..., 3)`` Oklab with no chroma.
    """
    lightness = lab[..., 0]
    if any(luminances):
        chroma = lab[..., 1:].norm(dim=-1)
        weights = band_weights(torch.atan2(lab[..., 2], lab[..., 1]))
        shift = weights @ torch.tensor(
            [float(v) for v in luminances], dtype=lab.dtype, device=lab.device
        )
        lightness = lightness * 2.0 ** (0.9 * shift * smoothstep(0.0, 0.12, chroma))
    return torch.stack([lightness, torch.zeros_like(lightness), torch.zeros_like(lightness)], -1)


def mix(images: torch.Tensor, hues, saturations, luminances, mono: bool = False) -> torch.Tensor:
    """Adjust the eight colour bands of a batch, or render it in black and white through them.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        hues: Eight values, -100 to 100, in :data:`BANDS` order.
        saturations: Eight values, -100 to 100.
        luminances: Eight values, -100 to 100. In black and white these set each band's grey.
        mono: Whether to render black and white.

    Returns:
        The mixed batch, or ``images`` itself in colour with every slider at 0.
    """
    hues = [float(v) / 100.0 for v in hues]
    saturations = [float(v) / 100.0 for v in saturations]
    luminances = [float(v) / 100.0 for v in luminances]
    if images.numel() == 0 or (not mono and not any(hues + saturations + luminances)):
        return images
    ceiling = not dynamic.carries(images)

    def work(chunk: torch.Tensor) -> torch.Tensor:
        light = raw.linearise(chunk[..., :3].float())
        if mono:
            light = fit_lab(_mono_lab(to_oklab(light), luminances), ceiling)
        else:
            light = mix_light(light, hues, saturations, luminances, ceiling)
        codes = raw.encode(light).to(chunk.dtype)
        return torch.cat([codes, chunk[..., 3:]], dim=-1) if chunk.shape[-1] > 3 else codes

    out = run_frames(images, frame_by_frame(work), factor=16.0, node="Image Color Mixer")
    return dynamic.hold(out, images)


def grade_lab(lab: torch.Tensor, wheels: dict, blending: float = 0.5, balance: float = 0.0):
    """Tint shadows, midtones, highlights and the whole picture, as colour grading wheels do.

    Args:
        lab: ``(..., 3)`` Oklab.
        wheels: :data:`GRADE_TONES` name to ``(hue degrees, saturation 0 to 1, luminance -1 to 1)``.
        blending: 0 = ranges meet sharply, 1 = they overlap widely.
        balance: -1 = more of the picture counts as shadow, 1 = more counts as highlight.

    Returns:
        ``(..., 3)`` Oklab.
    """
    lightness = lab[..., 0]
    pivot = 0.5 - 0.25 * float(balance)
    width = 0.10 + 0.35 * float(blending)
    shadows = 1.0 - smoothstep(pivot - 0.25 - width, pivot - 0.25 + width, lightness)
    highlights = smoothstep(pivot + 0.25 - width, pivot + 0.25 + width, lightness)
    masks = {
        "shadows": shadows,
        "midtones": (1.0 - shadows - highlights).clamp(min=0.0),
        "highlights": highlights,
        "global": torch.ones_like(lightness),
    }
    taper = smoothstep(0.0, 0.08, lightness) * (1.0 - 0.6 * smoothstep(0.92, 1.0, lightness))
    shift_a = torch.zeros_like(lightness)
    shift_b = torch.zeros_like(lightness)
    shift_l = torch.zeros_like(lightness)
    for tone, mask in masks.items():
        hue, saturation, luminance = wheels.get(tone, (0.0, 0.0, 0.0))
        reach = GRADE_GLOBAL_REACH if tone == "global" else GRADE_REACH
        if saturation:
            across, up = hue_direction(float(hue))
            shift_a = shift_a + mask * float(saturation) * reach * across
            shift_b = shift_b + mask * float(saturation) * reach * up
        if luminance:
            shift_l = shift_l + mask * float(luminance) * GRADE_LIGHTNESS
    return torch.stack(
        [lightness + shift_l, lab[..., 1] + shift_a * taper, lab[..., 2] + shift_b * taper], dim=-1
    )


def grade(images: torch.Tensor, wheels: dict, blending: float = 50.0, balance: float = 0.0):
    """Colour grade a batch with shadow, midtone, highlight and global wheels.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        wheels: :data:`GRADE_TONES` name to ``(hue 0 to 360, saturation 0 to 100,
            luminance -100 to 100)``.
        blending: 0 to 100, how widely the three ranges overlap.
        balance: -100 to 100, which way the split between shadows and highlights leans.

    Returns:
        The graded batch, or ``images`` itself when no wheel carries saturation or luminance.
    """
    scaled = {
        tone: (float(h), float(s) / 100.0, float(l) / 100.0) for tone, (h, s, l) in wheels.items()
    }
    if images.numel() == 0 or not any(s or l for _, s, l in scaled.values()):
        return images
    ceiling = not dynamic.carries(images)

    def work(chunk: torch.Tensor) -> torch.Tensor:
        lab = grade_lab(
            to_oklab(raw.linearise(chunk[..., :3].float())), scaled,
            float(blending) / 100.0, float(balance) / 100.0,
        )
        light = fit_lab(lab, ceiling)
        codes = raw.encode(light).to(chunk.dtype)
        return torch.cat([codes, chunk[..., 3:]], dim=-1) if chunk.shape[-1] > 3 else codes

    out = run_frames(images, frame_by_frame(work), factor=16.0, node="Image Color Grading")
    return dynamic.hold(out, images)


def natural_finish(images: torch.Tensor, amount: float = 0.5, detail: float = 0.3) -> torch.Tensor:
    """Pull a generated render's overcooked contrast, colour and crunch toward a photograph.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        amount: 0 to 1, how far tone and colour are calmed.
        detail: 0 to 1, how far over-sharpened texture and local contrast are softened.

    Returns:
        The finished batch, or ``images`` itself when both are 0.
    """
    amount = min(max(float(amount), 0.0), 1.0)
    detail = min(max(float(detail), 0.0), 1.0)
    if images.numel() == 0 or (amount == 0.0 and detail == 0.0):
        return images
    ceiling = not dynamic.carries(images)

    def work(chunk: torch.Tensor) -> torch.Tensor:
        light = raw.linearise(chunk[..., :3].float())
        level = luma(light)
        v = raw.encode(level)
        if amount:
            v = (
                v
                + (0.45 - v) * 0.12 * amount * smoothstep(0.02, 0.95, v)
                + 0.015 * amount * torch.exp(-((v / 0.1) ** 2))
                + 0.025 * amount * torch.exp(-(((v - 0.45) / 0.25) ** 2))
                - 0.045 * amount * smoothstep(0.75, 1.0, v) ** 2
            )
        if detail:
            long_edge = _long_edge(chunk)
            held = v.clamp(0.0, 1.0)
            v = v - detail * 0.45 * (v - filter_with_guide(held, held, max(3, round(0.012 * long_edge)), 0.01))
            sigma = max(0.8, 2.0 * long_edge / REFERENCE_EDGE)
            v = v - detail * 0.35 * (v - _pixels(gaussian_blur(_planes(v), sigma=sigma)))
        light = relight(light, level, v)
        if amount:
            lab = to_oklab(light)
            chroma = lab[..., 1:].norm(dim=-1, keepdim=True)
            weights = band_weights(torch.atan2(lab[..., 2], lab[..., 1]))
            warm = weights[..., 0:1] + weights[..., 1:2]
            factor = (1.0 - 0.08 * amount) / (1.0 + 2.5 * amount * (chroma - 0.07).clamp(min=0.0))
            factor = factor * (1.0 - 0.22 * amount * warm * smoothstep(0.03, 0.14, chroma))
            lab = torch.cat([lab[..., :1], lab[..., 1:] * factor], dim=-1)
            light = fit_lab(lab, ceiling)
        codes = raw.encode(light).to(chunk.dtype)
        return torch.cat([codes, chunk[..., 3:]], dim=-1) if chunk.shape[-1] > 3 else codes

    out = run_frames(images, frame_by_frame(work), factor=20.0, node="Image Natural Finish")
    return dynamic.hold(out, images)
