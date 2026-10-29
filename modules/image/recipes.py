"""The photo looks: a colour grade, and over most of them a halation.

:data:`RECIPES` holds the grades, :data:`HALATION` the glow laid over them, and
:func:`apply` runs one look over a batch of ``(..., 3)`` planes in 0 to 1.
"""

from __future__ import annotations

__all__ = ["HALATION", "RECIPES", "apply", "known"]

from collections.abc import Callable

import torch

from .accelerate import run_frames
from .blend_modes import ceiling_of

#: Float32 copies of a frame a look holds at its peak.
WORKING_COPIES = 16
from .fills import (
    blend_colour,
    blend_fill,
    blend_hue,
    blend_image,
    blend_opacity,
    composite_mask,
    fill,
    linear_gradient,
    linear_mask,
    quarter_turn,
    radial_gradient,
    radial_mask,
    rgb8,
)
from .looks import (
    bloom,
    brightness,
    contrast,
    fade,
    greyscale,
    hue_rotate,
    saturation,
    sepia,
    split_tone,
    temperature,
    tint_shadows,
)


def _1977(plane: torch.Tensor) -> torch.Tensor:
    """A magenta screen wash at 30 per cent, firmed up, brightened and saturated."""
    graded = blend_fill(plane, "screen", rgb8(243, 106, 188), 0.3)
    return saturation(brightness(contrast(graded, 1.1), 1.1), 1.3)


def _aden(plane: torch.Tensor) -> torch.Tensor:
    """A dark red darken fill fading off across the frame, hue turned back, flat and bright."""
    tinted = blend_fill(plane, "darken", rgb8(66, 10, 14))
    fading = quarter_turn(linear_mask(plane, start=0.8, horizontal=False))
    graded = composite_mask(tinted, plane, fading)
    graded = saturation(contrast(hue_rotate(graded, -20.0), 0.9), 0.85)
    return brightness(graded, 1.2)


def _bleach_bypass(plane: torch.Tensor) -> torch.Tensor:
    """Colour drained to a silver, the highlights blown and the contrast driven hard."""
    blown = blend_image(plane, greyscale(plane), "screen")
    graded = blend_opacity(plane, blown, 0.3)
    graded = saturation(graded, 0.48)
    graded = contrast(graded, 1.46, 0.40)
    return split_tone(graded, (0.64, 0.50, 0.38), (0.40, 0.50, 0.62), 2.1)


def _brannan(plane: torch.Tensor) -> torch.Tensor:
    """A purple lighten fill, half sepia, on hard contrast."""
    graded = blend_fill(plane, "lighten", rgb8(161, 44, 199), 0.31)
    return contrast(sepia(graded, 0.5), 1.4)


def _brooklyn(plane: torch.Tensor) -> torch.Tensor:
    """A green overlay in the middle over a grey violet one outside, flat and bright."""
    middle = blend_fill(plane, "overlay", rgb8(168, 223, 193), 0.4)
    outside = blend_fill(plane, "overlay", rgb8(196, 183, 200))
    graded = composite_mask(middle, outside, radial_mask(plane, length=0.7))
    return brightness(contrast(graded, 0.9), 1.1)


def _clarendon(plane: torch.Tensor) -> torch.Tensor:
    """A blue overlay at 20 per cent, firmed up and strongly saturated."""
    graded = blend_fill(plane, "overlay", rgb8(127, 187, 227), 0.2)
    return saturation(contrast(graded, 1.2), 1.35)


def _clean_punch(plane: torch.Tensor) -> torch.Tensor:
    """Blacks pulled deep and cool under a lifted white, vivid and firm."""
    graded = contrast(plane, 1.30, 0.24)
    graded = saturation(graded, 1.36)
    graded = split_tone(graded, (0.44, 0.46, 0.82), (0.48, 0.46, 0.72))
    return brightness(graded, 1.06)


def _cross_process(plane: torch.Tensor) -> torch.Tensor:
    """Green driven into the shadows and yellow into the highlights, hard and vivid."""
    graded = split_tone(plane, (0.50, 0.86, 0.50), (0.68, 0.84, 0.40), 1.6)
    graded = saturation(contrast(graded, 1.20, 0.44), 1.25)
    return blend_fill(graded, "soft-light", rgb8(150, 210, 80), 0.10)


def _earlybird(plane: torch.Tensor) -> torch.Tensor:
    """A sand to near black radial gradient overlaid, flattened and lightly sepia."""
    ramp = radial_gradient(
        plane,
        [rgb8(208, 186, 142), rgb8(54, 3, 9), rgb8(29, 2, 16)],
        [0.2, 0.85, 1.0],
    )
    graded = blend_image(plane, ramp, "overlay")
    return sepia(contrast(graded, 0.9), 0.2)


def _faded_film(plane: torch.Tensor) -> torch.Tensor:
    """Blacks lifted to a cream grey, flat and drained, with the yellows left in."""
    graded = contrast(saturation(plane, 0.85), 0.85, 0.45)
    graded = fade(graded, 0.075)
    graded = split_tone(graded, (0.81, 0.56, 0.02), (0.78, 0.56, 0.08))
    return brightness(graded, 1.02)


def _film_noir(plane: torch.Tensor) -> torch.Tensor:
    """Black and white on a hard curve, cold in the shadows and dark at the edges."""
    graded = contrast(greyscale(plane), 1.4, 0.18)
    darkened = blend_fill(graded, "multiply", rgb8(46, 50, 58))
    graded = composite_mask(graded, darkened, radial_mask(plane, length=0.35, scale=1.05))
    return tint_shadows(graded, (0.40, 0.56, 0.60), 0.8)


def _gingham(plane: torch.Tensor) -> torch.Tensor:
    """A lavender soft light, lifted, with every hue turned back a little."""
    graded = blend_fill(plane, "soft-light", rgb8(230, 230, 250))
    return hue_rotate(brightness(graded, 1.05), -10.0)


def _golden_hour(plane: torch.Tensor) -> torch.Tensor:
    """A low sun across the frame, gold in the highlights and amber in the blacks."""
    sun = linear_gradient(plane, rgb8(255, 196, 120), rgb8(112, 116, 140))
    graded = blend_opacity(plane, blend_image(plane, sun, "soft-light"), 0.5)
    graded = split_tone(graded, (0.86, 0.52, 0.24), (0.92, 0.54, 0.18))
    return saturation(contrast(graded, 1.06), 1.12)


def _hudson(plane: torch.Tensor) -> torch.Tensor:
    """A blue radial gradient multiplied in at half strength, bright and flat."""
    ramp = radial_gradient(plane, [rgb8(166, 177, 255), rgb8(52, 33, 52)], [0.5, 1.0])
    darkened = blend_image(plane, ramp, "multiply")
    graded = blend_opacity(plane, darkened, 0.5)
    return saturation(contrast(brightness(graded, 1.2), 0.9), 1.1)


def _inkwell(plane: torch.Tensor) -> torch.Tensor:
    """Sepia, firmed up and brightened, then drained to black and white."""
    graded = brightness(contrast(sepia(plane, 0.3), 1.1), 1.1)
    return greyscale(graded)


def _kelvin(plane: torch.Tensor) -> torch.Tensor:
    """A dark plum colour dodge under an amber overlay."""
    graded = blend_fill(plane, "color-dodge", rgb8(56, 44, 52))
    return blend_fill(graded, "overlay", rgb8(183, 125, 33))


def _lark(plane: torch.Tensor) -> torch.Tensor:
    """A navy colour dodge, capped just below white and flattened."""
    graded = blend_fill(plane, "color-dodge", rgb8(34, 37, 63))
    graded = blend_fill(graded, "darken", rgb8(242, 242, 242), 0.8)
    return contrast(graded, 0.9)


def _lofi(plane: torch.Tensor) -> torch.Tensor:
    """The middle left alone and the edges multiplied dark, hard and vivid."""
    darkened = blend_fill(plane, "multiply", rgb8(34, 34, 34))
    graded = composite_mask(plane, darkened, radial_mask(plane, length=0.7, scale=1.5))
    return contrast(saturation(graded, 1.1), 1.5)


def _maven(plane: torch.Tensor) -> torch.Tensor:
    """A green hue wash at 20 per cent, lightly sepia, dark, flat and vivid."""
    graded = blend_hue(plane, rgb8(3, 230, 26), 0.2)
    graded = contrast(brightness(sepia(graded, 0.25), 0.95), 0.95)
    return saturation(graded, 1.5)


def _mayfair(plane: torch.Tensor) -> torch.Tensor:
    """A warm off-centre glow ringed by a dark overlay, mixed back at 40 per cent."""
    white = blend_fill(plane, "overlay", rgb8(255, 255, 255), 0.8)
    pink = blend_fill(plane, "overlay", rgb8(255, 200, 200), 0.6)
    dark = blend_fill(plane, "overlay", rgb8(17, 17, 17))
    lit = composite_mask(
        white, pink, radial_mask(plane, scale=0.3, centre_x=0.4, centre_y=0.4)
    )
    lit = composite_mask(
        lit, dark, radial_mask(plane, length=0.3, scale=0.6, centre_x=0.4, centre_y=0.4)
    )
    graded = blend_opacity(plane, lit, 0.4)
    return saturation(contrast(graded, 1.1), 1.1)


def _moody_blue(plane: torch.Tensor) -> torch.Tensor:
    """A cold blue cast with the blacks deepened and the colour drawn down."""
    graded = temperature(plane, -1.2)
    graded = tint_shadows(graded, (0.32, 0.46, 0.90), 0.6)
    return saturation(contrast(graded, 1.12, 0.42), 0.85)


def _moon(plane: torch.Tensor) -> torch.Tensor:
    """A grey soft light with the blacks lifted, drained to black and white."""
    graded = blend_fill(plane, "soft-light", rgb8(160, 160, 160))
    graded = blend_fill(graded, "lighten", rgb8(56, 56, 56))
    return brightness(contrast(greyscale(graded), 1.1), 1.1)


def _nashville(plane: torch.Tensor) -> torch.Tensor:
    """A peach darken over a navy lighten, lightly sepia, firm, bright and vivid."""
    graded = blend_fill(plane, "darken", rgb8(247, 176, 153), 0.56)
    graded = blend_fill(graded, "lighten", rgb8(0, 70, 150), 0.4)
    graded = brightness(contrast(sepia(graded, 0.2), 1.2), 1.05)
    return saturation(graded, 1.2)


def _neon_night(plane: torch.Tensor) -> torch.Tensor:
    """Magenta driven through the frame and strongest in the lights, vivid and hard."""
    graded = split_tone(plane, (0.25, 0.42, 0.98), (0.90, 0.40, 0.85))
    graded = blend_fill(graded, "screen", rgb8(46, 0, 20), 0.5)
    return saturation(contrast(graded, 1.15, 0.40), 1.3)


def _perpetua(plane: torch.Tensor) -> torch.Tensor:
    """A blue to yellow vertical gradient soft lit in at half strength."""
    ramp = linear_gradient(plane, rgb8(0, 91, 154), rgb8(230, 193, 61), horizontal=False)
    lit = blend_image(plane, ramp, "soft-light")
    return blend_opacity(plane, lit, 0.5)


def _reyes(plane: torch.Tensor) -> torch.Tensor:
    """A cream soft light at half strength, lightly sepia, pale and washed out."""
    lit = blend_fill(plane, "soft-light", rgb8(239, 205, 173))
    graded = blend_opacity(plane, lit, 0.5)
    graded = contrast(brightness(sepia(graded, 0.22), 1.1), 0.85)
    return saturation(graded, 0.75)


def _rise(plane: torch.Tensor) -> torch.Tensor:
    """A warm centre inside a dark surround, glowing in the middle, sepia and soft."""
    middle = blend_fill(plane, "multiply", rgb8(236, 205, 169), 0.15)
    outside = blend_fill(plane, "multiply", rgb8(50, 30, 7), 0.4)
    shaded = composite_mask(middle, outside, radial_mask(plane, length=0.55))
    glow = blend_fill(shaded, "overlay", rgb8(232, 197, 152), 0.8)
    lit = composite_mask(glow, shaded, radial_mask(plane, scale=0.9))
    graded = blend_opacity(shaded, lit, 0.6)
    graded = contrast(sepia(brightness(graded, 1.05), 0.2), 0.9)
    return saturation(graded, 0.9)


def _slumber(plane: torch.Tensor) -> torch.Tensor:
    """A brown lighten under an olive soft light, drained of colour and lifted."""
    graded = blend_fill(plane, "lighten", rgb8(69, 41, 12), 0.4)
    graded = blend_fill(graded, "soft-light", rgb8(125, 105, 24), 0.5)
    return brightness(saturation(graded, 0.66), 1.05)


def _soft_portrait(plane: torch.Tensor) -> torch.Tensor:
    """A pink cast, flat and pale, shaded away from the middle."""
    graded = split_tone(plane, (0.74, 0.50, 0.68), (0.92, 0.52, 0.58), 1.6)
    graded = contrast(graded, 0.88, 0.52)
    shaded = blend_fill(graded, "multiply", rgb8(168, 148, 154))
    graded = composite_mask(graded, shaded, radial_mask(plane, length=0.25, scale=1.0))
    return saturation(fade(graded, 0.05), 0.86)


def _stinson(plane: torch.Tensor) -> torch.Tensor:
    """A salmon soft light at 20 per cent, very flat, pale and bright."""
    graded = blend_fill(plane, "soft-light", rgb8(240, 149, 128), 0.2)
    return brightness(saturation(contrast(graded, 0.75), 0.85), 1.15)


def _teal_and_orange(plane: torch.Tensor) -> torch.Tensor:
    """Teal driven into the shadows and orange into the lights, firm and vivid."""
    graded = contrast(plane, 1.18, 0.42)
    graded = split_tone(graded, (0.16, 0.58, 0.76), (0.96, 0.48, 0.08), 2.4)
    return saturation(graded, 1.10)


def _toaster(plane: torch.Tensor) -> torch.Tensor:
    """A brown to purple radial gradient screened in, hard and dark."""
    ramp = radial_gradient(plane, [rgb8(128, 78, 15), rgb8(59, 0, 59)])
    graded = blend_image(plane, ramp, "screen")
    return brightness(contrast(graded, 1.5), 0.9)


def _valencia(plane: torch.Tensor) -> torch.Tensor:
    """A plum exclusion at half strength, a touch firmer, brighter and sepia."""
    lifted = blend_fill(plane, "exclusion", rgb8(58, 3, 57))
    graded = blend_opacity(plane, lifted, 0.5)
    return sepia(brightness(contrast(graded, 1.08), 1.08), 0.08)


def _walden(plane: torch.Tensor) -> torch.Tensor:
    """A blue screen wash at 30 per cent, bright, sepia toned and strongly saturated."""
    washed = blend_fill(plane, "screen", rgb8(0, 68, 204))
    graded = blend_opacity(plane, washed, 0.3)
    graded = sepia(hue_rotate(brightness(graded, 1.1), -10.0), 0.3)
    return saturation(graded, 1.6)


def _willow(plane: torch.Tensor) -> torch.Tensor:
    """A pink to black radial gradient overlaid, recoloured warm grey and half drained."""
    ramp = radial_gradient(plane, [rgb8(212, 169, 175), rgb8(0, 0, 0)], [0.55, 1.5])
    graded = blend_image(plane, ramp, "overlay")
    graded = blend_colour(graded, rgb8(216, 205, 203))
    return brightness(contrast(greyscale(graded, 0.5), 0.95), 0.9)


def _xpro2(plane: torch.Tensor) -> torch.Tensor:
    """A paper white centre and a blue surround burned in, then sepia toned."""
    paper = fill(plane, rgb8(230, 231, 224))
    surround = blend_opacity(plane, fill(plane, rgb8(43, 42, 161)), 0.6)
    mask = radial_mask(plane, length=0.4, scale=1.1)
    layer = composite_mask(paper, surround, mask)
    burned = blend_image(plane, layer, "color-burn")
    softened = blend_opacity(plane, burned, 0.6)
    return sepia(composite_mask(burned, softened, mask), 0.3)


#: Every look, under the name the style menu carries, in the order the menu lists them.
RECIPES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "1977": _1977,
    "aden": _aden,
    "bleach bypass": _bleach_bypass,
    "brannan": _brannan,
    "brooklyn": _brooklyn,
    "clarendon": _clarendon,
    "clean punch": _clean_punch,
    "cross process": _cross_process,
    "earlybird": _earlybird,
    "faded film": _faded_film,
    "film noir": _film_noir,
    "gingham": _gingham,
    "golden hour": _golden_hour,
    "hudson": _hudson,
    "inkwell": _inkwell,
    "kelvin": _kelvin,
    "lark": _lark,
    "lofi": _lofi,
    "maven": _maven,
    "mayfair": _mayfair,
    "moody blue": _moody_blue,
    "moon": _moon,
    "nashville": _nashville,
    "neon night": _neon_night,
    "perpetua": _perpetua,
    "reyes": _reyes,
    "rise": _rise,
    "slumber": _slumber,
    "soft portrait": _soft_portrait,
    "stinson": _stinson,
    "teal and orange": _teal_and_orange,
    "toaster": _toaster,
    "valencia": _valencia,
    "walden": _walden,
    "willow": _willow,
    "xpro2": _xpro2,
}


#: The colour a golden hour halation is tinted, on a 0 to 1 scale.
GOLD = (1.00, 0.78, 0.46)

#: The deeper amber the hottest looks bleed.
EMBER = (1.00, 0.62, 0.28)

#: A warm white, for a glow that carries no colour of its own.
PEARL = (1.00, 0.95, 0.88)

#: The cool daylight the cold looks bleed.
FROST = (0.80, 0.90, 1.00)

#: The magenta a lit sign bleeds.
NEON = (1.00, 0.52, 0.92)

#: The pink a diffused portrait light bleeds.
ROSE = (1.00, 0.84, 0.88)

#: The halation over each grade: where the highlights start as a fraction of the range, how
#: far the glow reaches as a fraction of the shorter side, how strongly it returns, and the
#: colour it is tinted. A look absent from this carries no glow.
HALATION: dict[str, tuple[float, float, float, tuple[float, float, float] | None]] = {
    "1977": (0.66, 0.050, 0.55, GOLD),
    "aden": (0.74, 0.040, 0.28, PEARL),
    "bleach bypass": (0.80, 0.030, 0.22, PEARL),
    "brannan": (0.74, 0.035, 0.28, GOLD),
    "brooklyn": (0.76, 0.035, 0.22, PEARL),
    "clarendon": (0.80, 0.030, 0.20, FROST),
    "clean punch": (0.82, 0.028, 0.20, PEARL),
    "cross process": (0.70, 0.045, 0.40, GOLD),
    "earlybird": (0.62, 0.060, 0.70, GOLD),
    "faded film": (0.62, 0.060, 0.40, PEARL),
    "film noir": (0.60, 0.060, 0.55, None),
    "gingham": (0.74, 0.040, 0.26, FROST),
    "golden hour": (0.58, 0.070, 0.65, EMBER),
    "hudson": (0.62, 0.050, 0.42, FROST),
    "inkwell": (0.72, 0.040, 0.35, None),
    "kelvin": (0.66, 0.050, 0.55, EMBER),
    "lark": (0.72, 0.040, 0.30, FROST),
    "lofi": (0.76, 0.030, 0.25, PEARL),
    "maven": (0.72, 0.045, 0.28, GOLD),
    "mayfair": (0.68, 0.050, 0.45, GOLD),
    "moody blue": (0.66, 0.055, 0.45, FROST),
    "moon": (0.72, 0.040, 0.35, None),
    "nashville": (0.68, 0.045, 0.50, GOLD),
    "neon night": (0.56, 0.075, 0.75, NEON),
    "perpetua": (0.74, 0.040, 0.24, PEARL),
    "reyes": (0.70, 0.050, 0.35, GOLD),
    "rise": (0.62, 0.065, 0.70, GOLD),
    "slumber": (0.70, 0.050, 0.35, GOLD),
    "soft portrait": (0.42, 0.090, 0.62, ROSE),
    "stinson": (0.72, 0.045, 0.30, GOLD),
    "teal and orange": (0.72, 0.045, 0.35, GOLD),
    "toaster": (0.64, 0.055, 0.60, EMBER),
    "valencia": (0.70, 0.045, 0.45, GOLD),
    "walden": (0.70, 0.045, 0.40, PEARL),
    "willow": (0.62, 0.055, 0.50, None),
    "xpro2": (0.76, 0.035, 0.26, PEARL),
}


def _glow(plane: torch.Tensor, style: str) -> torch.Tensor:
    """Bleed a look's highlights over the grade it just made.

    Args:
        plane: ``(..., 3)`` colours a grade answered.
        style: The look's name.

    Returns:
        The plane with its highlights spread, or unchanged where the look has no glow.
    """
    settings = HALATION.get(style)
    return plane if settings is None else bloom(plane, *settings)


def known(style: str) -> bool:
    """Whether a name has a recipe behind it.

    Args:
        style: The name to test.

    Returns:
        Whether :data:`RECIPES` holds it.
    """
    return style in RECIPES


def apply(
    images: torch.Tensor, style: str, strength: float = 1.0, prefer_gpu: bool = True
) -> torch.Tensor:
    """Run one look over a batch.

    Args:
        images: ``(batch, height, width, 3)`` colours.
        style: A name :data:`RECIPES` holds.
        strength: How far towards the look the result sits. 1.0 is the whole look, 0.5
            half of it, 0.0 the images unchanged.
        prefer_gpu: Whether to run on ComfyUI's compute device. False keeps it on the CPU.

    Returns:
        A tensor of the shape and dtype it was given, on the device it arrived on.

    Raises:
        KeyError: The style has no recipe.
    """
    recipe = RECIPES[style]
    if strength <= 0.0:
        return images
    # A look is written against 0 to 1, so a batch carrying light above white is scaled into
    # that range and scaled back afterwards.
    top = ceiling_of(images)

    def graded(batch: torch.Tensor) -> torch.Tensor:
        source = batch.float() / top
        plane = _glow(recipe(source), style)
        if strength < 1.0:
            plane = source + (plane - source) * strength
        return (plane * top).clamp(0.0, top).to(batch.dtype)

    return run_frames(images, graded, prefer_gpu, factor=WORKING_COPIES, node="Image Style Filter")
