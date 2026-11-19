"""Transfer functions and gamuts of camera, display and ACES colour spaces.

Curves act on float64 ``(..., 3)`` tensors, unclamped. Linear light is relative: 1.0 is
reference white, 0.18 mid grey.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = [
    "ADAPTATIONS",
    "DEFAULT_WHITE_NITS",
    "GAMUTS",
    "LEVELS",
    "TRANSFERS",
    "Gamut",
    "Transfer",
    "convert",
    "gamut_matrix",
    "pq_nits",
]

#: CIE 1931 chromaticity of the D65 white, to the four places every standard prints.
D65 = (0.3127, 0.3290)

#: The ACES white point, SMPTE ST 2065-1.
ACES_WHITE = (0.32168, 0.33767)

#: The Blackmagic Wide Gamut Gen 5 white point, as Blackmagic Design prints it.
BLACKMAGIC_WHITE = (0.3127170, 0.3290312)


@dataclass(frozen=True)
class Gamut:
    """Primaries and white point of one RGB colour space.

    Attributes:
        primaries: ``((xr, yr), (xg, yg), (xb, yb))``, CIE 1931 chromaticities.
        white: ``(x, y)`` of the white point.
        to_xyz: The maker's published RGB to XYZ matrix, or ``None`` to derive it from the
            primaries.
        from_xyz: The maker's published XYZ to RGB matrix, or ``None`` to invert ``to_xyz``.
    """

    primaries: tuple
    white: tuple
    to_xyz: tuple | None = None
    from_xyz: tuple | None = None


#: Widget option -> gamut. Apple Log is recorded in Rec.2020.
GAMUTS = {
    "Rec.709 / sRGB": Gamut(((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), D65),
    "Display P3": Gamut(((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), D65),
    "Rec.2020": Gamut(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), D65),
    "ACES2065-1 (AP0)": Gamut(
        ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770)),
        ACES_WHITE,
        to_xyz=(
            (0.9525523959, 0.0, 0.0000936786),
            (0.3439664498, 0.7281660966, -0.0721325464),
            (0.0, 0.0, 1.0088251844),
        ),
        from_xyz=(
            (1.0498110175, 0.0, -0.0000974845),
            (-0.4959030231, 1.3733130458, 0.0982400361),
            (0.0, 0.0, 0.9912520182),
        ),
    ),
    "ACEScg (AP1)": Gamut(((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), ACES_WHITE),
    "Sony S-Gamut3": Gamut(
        ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)),
        D65,
        to_xyz=(
            (0.7064827132, 0.1288010498, 0.1151721641),
            (0.2709796708, 0.7866064112, -0.0575860820),
            (-0.0096778454, 0.0046000375, 1.0941355587),
        ),
        from_xyz=(
            (1.5073998991, -0.2458221374, -0.1716116808),
            (-0.5181517271, 1.3553912409, 0.1258786682),
            (0.0155116982, -0.0078727714, 0.9119163656),
        ),
    ),
    "Sony S-Gamut3.Cine": Gamut(
        ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)),
        D65,
        to_xyz=(
            (0.5990839208, 0.2489255161, 0.1024464902),
            (0.2150758201, 0.8850685017, -0.1001443219),
            (-0.0320658495, -0.0276583907, 1.1487819910),
        ),
        from_xyz=(
            (1.8467789693, -0.5259861230, -0.2105452114),
            (-0.4441532629, 1.2594429028, 0.1493999729),
            (0.0408554212, 0.0156408893, 0.8682072487),
        ),
    ),
    "ARRI Wide Gamut 3": Gamut(
        ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)),
        D65,
        to_xyz=(
            (0.638008, 0.214704, 0.097744),
            (0.291954, 0.823841, -0.115795),
            (0.002798, -0.067034, 1.153294),
        ),
        from_xyz=(
            (1.789066, -0.482534, -0.200076),
            (-0.639849, 1.396400, 0.194432),
            (-0.041532, 0.082335, 0.878868),
        ),
    ),
    "ARRI Wide Gamut 4": Gamut(((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)), D65),
    "Panasonic V-Gamut": Gamut(
        ((0.730, 0.280), (0.165, 0.840), (0.100, -0.030)),
        D65,
        to_xyz=(
            (0.679644, 0.152211, 0.118600),
            (0.260686, 0.774894, -0.035580),
            (-0.009310, -0.004612, 1.102980),
        ),
        from_xyz=(
            (1.589012, -0.313204, -0.180965),
            (-0.534053, 1.396011, 0.102458),
            (0.011179, 0.003194, 0.905535),
        ),
    ),
    "Fujifilm F-Gamut": Gamut(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), D65),
    "Canon Cinema Gamut": Gamut(((0.740, 0.270), (0.170, 1.140), (0.080, -0.100)), D65),
    "RED Wide Gamut RGB": Gamut(
        ((0.780308, 0.304253), (0.121595, 1.493994), (0.095612, -0.084589)),
        D65,
        to_xyz=(
            (0.735275, 0.068609, 0.146571),
            (0.286694, 0.842979, -0.129673),
            (-0.079681, -0.347343, 1.516082),
        ),
    ),
    "Blackmagic Wide Gamut Gen 5": Gamut(
        ((0.7177215, 0.3171181), (0.2280410, 0.8615690), (0.1005841, -0.0820452)),
        BLACKMAGIC_WHITE,
    ),
    "DaVinci Wide Gamut": Gamut(((0.8000, 0.3130), (0.1682, 0.9877), (0.0790, -0.1155)), D65),
}

#: Widget option -> cone response matrix of a von Kries adaptation, or ``None`` for none.
ADAPTATIONS = {
    "Bradford": ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296)),
    "CAT02": ((0.7328, 0.4296, -0.1624), (-0.7036, 1.6975, 0.0061), (0.0030, 0.0136, 0.9834)),
    "CAT16": (
        (0.401288, 0.650173, -0.051461),
        (-0.250268, 1.204414, 0.045854),
        (-0.002079, 0.048952, 0.953127),
    ),
    "Von Kries": ((0.40024, 0.70760, -0.08081), (-0.22630, 1.16532, 0.04570), (0.0, 0.0, 0.91822)),
    "XYZ Scaling": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "None": None,
}

#: Widget option for how image codes sit against a curve's own values.
LEVELS = ("full", "legal")

#: 10-bit code values of video black and white, and the code count.
LEGAL_BLACK = 64.0
LEGAL_SPAN = 876.0
CODE_MAX = 1023.0

#: Display luminance, in cd/m2, that linear 1.0 stands for on the PQ and HLG side.
DEFAULT_WHITE_NITS = 203.0


# ---------------------------------------------------------------------------------------
# Curves. Each takes and answers float64; ``white`` is reference white in cd/m2.


def _mirror(curve: Callable, values: torch.Tensor) -> torch.Tensor:
    """Apply ``curve``, answering ``-curve(-v)`` for a negative ``v``."""
    magnitude = curve(values.abs())
    return torch.where(values < 0, -magnitude, magnitude)


def _power(exponent: float) -> Callable:
    """A pure power law, mirrored below zero."""
    return lambda values, white=None: _mirror(lambda v: v**exponent, values)


def _linear(values: torch.Tensor, white=None) -> torch.Tensor:
    """The identity."""
    return values


# IEC 61966-2-1 sRGB.
SRGB_CUT_LINEAR = 0.0031308
SRGB_CUT_CODE = 0.04045


def _srgb_encode(light, white=None):
    """sRGB inverse EOTF; the straight segment continues below zero."""
    curved = 1.055 * light.clamp(min=0.0) ** (1.0 / 2.4) - 0.055
    return torch.where(light <= SRGB_CUT_LINEAR, light * 12.92, curved)


def _srgb_decode(codes, white=None):
    """sRGB EOTF; the straight segment continues below zero."""
    curved = ((codes.clamp(min=0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(codes <= SRGB_CUT_CODE, codes / 12.92, curved)


# ITU-R BT.709 camera OETF.
BT709_CUT = 0.018
BT709_CUT_CODE = 1.099 * BT709_CUT**0.45 - 0.099


def _bt709_encode(light, white=None):
    """BT.709 OETF; the straight segment continues below zero."""
    curved = 1.099 * light.clamp(min=0.0) ** 0.45 - 0.099
    return torch.where(light < BT709_CUT, light * 4.5, curved)


def _bt709_decode(codes, white=None):
    """Inverse of the BT.709 OETF."""
    curved = ((codes.clamp(min=0.0) + 0.099) / 1.099) ** (1.0 / 0.45)
    return torch.where(codes < BT709_CUT_CODE, codes / 4.5, curved)


# SMPTE ST 2084 perceptual quantiser.
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0
PQ_PEAK = 10000.0


def _pq_signal(luminance):
    """PQ inverse EOTF of luminance normalised to 10000 cd/m2."""
    power = luminance**PQ_M1
    return ((PQ_C1 + PQ_C2 * power) / (1.0 + PQ_C3 * power)) ** PQ_M2


def _pq_luminance(codes):
    """PQ EOTF as luminance normalised to 10000 cd/m2, infinite past the curve's pole."""
    power = codes ** (1.0 / PQ_M2)
    denominator = PQ_C2 - PQ_C3 * power
    ratio = (power - PQ_C1).clamp(min=0.0) / denominator.clamp(min=1e-300)
    return torch.where(denominator > 0, ratio ** (1.0 / PQ_M1), torch.full_like(codes, math.inf))


def _pq_encode(light, white):
    """Relative light to PQ codes, linear 1.0 at ``white`` cd/m2, mirrored below zero."""
    return _mirror(_pq_signal, light * (white / PQ_PEAK))


def _pq_decode(codes, white):
    """PQ codes to relative light, ``white`` cd/m2 at 1.0, mirrored below zero."""
    return _mirror(_pq_luminance, codes) * (PQ_PEAK / white)


def pq_nits(codes: torch.Tensor) -> torch.Tensor:
    """Display luminance of PQ codes.

    Args:
        codes: PQ signal values, any shape.

    Returns:
        cd/m2, float64, mirrored below zero.
    """
    return _mirror(_pq_luminance, codes.to(torch.float64)) * PQ_PEAK


# ITU-R BT.2100 hybrid log-gamma, decoded through its EOTF on the reference display.
HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4.0 * HLG_A)
HLG_PEAK = 1000.0
HLG_GAMMA = 1.2 + 0.42 * math.log10(HLG_PEAK / 1000.0)
HLG_WEIGHTS = (0.2627, 0.6780, 0.0593)


def _hlg_oetf(scene):
    """BT.2100 HLG OETF of scene light in 0 to 1, mirrored below zero."""

    def curve(e):
        root = (3.0 * e).sqrt()
        logged = HLG_A * torch.log((12.0 * e - HLG_B).clamp(min=1e-300)) + HLG_C
        return torch.where(e <= 1.0 / 12.0, root, logged)

    return _mirror(curve, scene)


def _hlg_oetf_inverse(codes):
    """Inverse of the BT.2100 HLG OETF, mirrored below zero."""

    def curve(v):
        squared = v * v / 3.0
        exponential = (torch.exp((v - HLG_C) / HLG_A) + HLG_B) / 12.0
        return torch.where(v <= 0.5, squared, exponential)

    return _mirror(curve, codes)


def _luminance(rgb):
    """BT.2100 luminance of the last axis, keeping it as a size-one axis."""
    weights = torch.tensor(HLG_WEIGHTS, dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _hlg_decode(codes, white):
    """HLG codes to relative display light through the BT.2100 EOTF at 1000 cd/m2."""
    scene = _hlg_oetf_inverse(codes)
    display = HLG_PEAK * scene * _luminance(scene).abs() ** (HLG_GAMMA - 1.0)
    return display / white


def _hlg_encode(light, white):
    """Relative display light to HLG codes through the inverse BT.2100 EOTF at 1000 cd/m2."""
    display = light * white
    luminance = _luminance(display)
    gain = (luminance / HLG_PEAK).abs() ** ((1.0 - HLG_GAMMA) / HLG_GAMMA)
    scene = torch.where(luminance == 0, torch.zeros_like(display), gain * display / HLG_PEAK)
    return _hlg_oetf(scene)


# Academy S-2016-001 ACEScct.
ACESCCT_X_BRK = 0.0078125
ACESCCT_Y_BRK = 0.155251141552511
ACESCCT_A = 10.5402377416545
ACESCCT_B = 0.0729055341958355


def _acescct_encode(light, white=None):
    """ACEScct from linear AP1."""
    logged = (torch.log2(light.clamp(min=1e-300)) + 9.72) / 17.52
    return torch.where(light <= ACESCCT_X_BRK, ACESCCT_A * light + ACESCCT_B, logged)


def _acescct_decode(codes, white=None):
    """Linear AP1 from ACEScct."""
    return torch.where(
        codes > ACESCCT_Y_BRK,
        torch.exp2(codes * 17.52 - 9.72),
        (codes - ACESCCT_B) / ACESCCT_A,
    )


# Sony S-Log3, technical summary for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3.
SLOG3_CUT = 0.01125
SLOG3_TOE = 171.2102946929


def _slog3_encode(light, white=None):
    """S-Log3 code values over 1023, from reflectance."""
    logged = (420.0 + torch.log10((light + 0.01).clamp(min=1e-300) / 0.19) * 261.5) / CODE_MAX
    straight = (light * (SLOG3_TOE - 95.0) / SLOG3_CUT + 95.0) / CODE_MAX
    return torch.where(light >= SLOG3_CUT, logged, straight)


def _slog3_decode(codes, white=None):
    """Reflectance from S-Log3 code values over 1023."""
    logged = 10.0 ** ((codes * CODE_MAX - 420.0) / 261.5) * 0.19 - 0.01
    straight = (codes * CODE_MAX - 95.0) * SLOG3_CUT / (SLOG3_TOE - 95.0)
    return torch.where(codes >= SLOG3_TOE / CODE_MAX, logged, straight)


# ARRI LogC3, SUP 3.x, exposure index 800, scene linear.
LOGC3 = {"cut": 0.010591, "a": 5.555556, "b": 0.052272, "c": 0.247190, "d": 0.385537,
         "e": 5.367655, "f": 0.092809}


def _logc3_encode(light, white=None):
    """ARRI LogC3 at EI 800 from scene linear."""
    k = LOGC3
    logged = k["c"] * torch.log10((k["a"] * light + k["b"]).clamp(min=1e-300)) + k["d"]
    return torch.where(light > k["cut"], logged, k["e"] * light + k["f"])


def _logc3_decode(codes, white=None):
    """Scene linear from ARRI LogC3 at EI 800."""
    k = LOGC3
    logged = (10.0 ** ((codes - k["d"]) / k["c"]) - k["b"]) / k["a"]
    return torch.where(codes > k["e"] * k["cut"] + k["f"], logged, (codes - k["f"]) / k["e"])


# ARRI LogC4 specification.
LOGC4_A = (2.0**18 - 16.0) / 117.45
LOGC4_B = (1023.0 - 95.0) / 1023.0
LOGC4_C = 95.0 / 1023.0
LOGC4_S = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * LOGC4_C / LOGC4_B)) / (LOGC4_A * LOGC4_B)
LOGC4_T = (2.0 ** (14.0 * (-LOGC4_C / LOGC4_B) + 6.0) - 64.0) / LOGC4_A


def _logc4_encode(light, white=None):
    """ARRI LogC4 from scene linear."""
    inner = (LOGC4_A * light + 64.0).clamp(min=1e-300)
    logged = (torch.log2(inner) - 6.0) / 14.0 * LOGC4_B + LOGC4_C
    return torch.where(light >= LOGC4_T, logged, (light - LOGC4_T) / LOGC4_S)


def _logc4_decode(codes, white=None):
    """Scene linear from ARRI LogC4."""
    logged = (torch.exp2(14.0 * ((codes - LOGC4_C) / LOGC4_B) + 6.0) - 64.0) / LOGC4_A
    return torch.where(codes >= 0, logged, codes * LOGC4_S + LOGC4_T)


# Panasonic V-Log/V-Gamut reference manual.
VLOG_CUT1 = 0.01
VLOG_CUT2 = 0.181
VLOG_B = 0.00873
VLOG_C = 0.241514
VLOG_D = 0.598206


def _vlog_encode(light, white=None):
    """V-Log code values over 1023, from reflectance."""
    logged = VLOG_C * torch.log10((light + VLOG_B).clamp(min=1e-300)) + VLOG_D
    return torch.where(light < VLOG_CUT1, 5.6 * light + 0.125, logged)


def _vlog_decode(codes, white=None):
    """Reflectance from V-Log code values over 1023."""
    logged = 10.0 ** ((codes - VLOG_D) / VLOG_C) - VLOG_B
    return torch.where(codes < VLOG_CUT2, (codes - 0.125) / 5.6, logged)


# Fujifilm F-Log2 data sheet.
FLOG2 = {"cut1": 0.000889, "cut2": 0.100686685370811, "a": 5.555556, "b": 0.064829,
         "c": 0.245281, "d": 0.384316, "e": 8.799461, "f": 0.092864}


def _flog2_encode(light, white=None):
    """F-Log2 code values over 1023, from reflectance."""
    k = FLOG2
    logged = k["c"] * torch.log10((k["a"] * light + k["b"]).clamp(min=1e-300)) + k["d"]
    return torch.where(light < k["cut1"], k["e"] * light + k["f"], logged)


def _flog2_decode(codes, white=None):
    """Reflectance from F-Log2 code values over 1023."""
    k = FLOG2
    logged = 10.0 ** ((codes - k["d"]) / k["c"]) / k["a"] - k["b"] / k["a"]
    return torch.where(codes < k["cut2"], (codes - k["f"]) / k["e"], logged)


# Apple Log profile white paper.
APPLE_R0 = -0.05641088
APPLE_RT = 0.01
APPLE_SIGMA = 47.28711236
APPLE_BETA = 0.00964052
APPLE_GAMMA = 0.08550479
APPLE_DELTA = 0.69336945
APPLE_PT = APPLE_SIGMA * (APPLE_RT - APPLE_R0) ** 2


def _apple_encode(light, white=None):
    """Apple Log from scene linear; light below R0 encodes as 0, as the profile defines."""
    logged = APPLE_GAMMA * torch.log2((light + APPLE_BETA).clamp(min=1e-300)) + APPLE_DELTA
    toe = APPLE_SIGMA * (light - APPLE_R0) ** 2
    answer = torch.where(light >= APPLE_RT, logged, toe)
    return torch.where(light < APPLE_R0, torch.zeros_like(light), answer)


def _apple_decode(codes, white=None):
    """Scene linear from Apple Log; codes below 0 decode to R0, as the profile defines."""
    logged = torch.exp2((codes - APPLE_DELTA) / APPLE_GAMMA) - APPLE_BETA
    toe = (codes.clamp(min=0.0) / APPLE_SIGMA).sqrt() + APPLE_R0
    answer = torch.where(codes >= APPLE_PT, logged, toe)
    return torch.where(codes < 0, torch.full_like(codes, APPLE_R0), answer)


# Canon Log 3, version 1.2 constants, which are code values over 1023.
CLOG3_LOW = 0.097465473
CLOG3_HIGH = 0.15277891
CLOG3_SLOPE = 1.9754798
CLOG3_OFFSET = 0.12512219
CLOG3_LIGHT_LOW = (CLOG3_LOW - CLOG3_OFFSET) / CLOG3_SLOPE
CLOG3_LIGHT_HIGH = (CLOG3_HIGH - CLOG3_OFFSET) / CLOG3_SLOPE
CANON_REFLECTANCE = 0.9


def _clog3_encode(light, white=None):
    """Canon Log 3 code values over 1023, from reflectance."""
    x = light / CANON_REFLECTANCE
    below = -0.36726845 * torch.log10((-x * 14.98325 + 1.0).clamp(min=1e-300)) + 0.12783901
    above = 0.36726845 * torch.log10((x * 14.98325 + 1.0).clamp(min=1e-300)) + 0.12240537
    middle = CLOG3_SLOPE * x + CLOG3_OFFSET
    return torch.where(x < CLOG3_LIGHT_LOW, below, torch.where(x <= CLOG3_LIGHT_HIGH, middle, above))


def _clog3_decode(codes, white=None):
    """Reflectance from Canon Log 3 code values over 1023."""
    below = -(10.0 ** ((0.12783901 - codes) / 0.36726845) - 1.0) / 14.98325
    above = (10.0 ** ((codes - 0.12240537) / 0.36726845) - 1.0) / 14.98325
    middle = (codes - CLOG3_OFFSET) / CLOG3_SLOPE
    x = torch.where(codes < CLOG3_LOW, below, torch.where(codes <= CLOG3_HIGH, middle, above))
    return x * CANON_REFLECTANCE


# RED Log3G10, IPP2 version 3.
LOG3G10_A = 0.224282
LOG3G10_B = 155.975327
LOG3G10_C = 0.01
LOG3G10_G = 15.1927


def _log3g10_encode(light, white=None):
    """RED Log3G10 from scene linear."""
    x = light + LOG3G10_C
    logged = torch.sign(x) * LOG3G10_A * torch.log10(x.abs() * LOG3G10_B + 1.0)
    return torch.where(x < 0.0, x * LOG3G10_G, logged)


def _log3g10_decode(codes, white=None):
    """Scene linear from RED Log3G10."""
    logged = torch.sign(codes) * (10.0 ** (codes.abs() / LOG3G10_A) - 1.0) / LOG3G10_B - LOG3G10_C
    return torch.where(codes < 0.0, codes / LOG3G10_G - LOG3G10_C, logged)


# Blackmagic Generation 5 Color Science technical reference.
BMD_A = 0.08692876065491224
BMD_B = 0.005494072432257808
BMD_C = 0.5300133392291939
BMD_D = 8.283605932402494
BMD_E = 0.09246575342465753
BMD_LIN_CUT = 0.005
BMD_LOG_CUT = BMD_D * BMD_LIN_CUT + BMD_E


def _bmd_encode(light, white=None):
    """Blackmagic Film Gen 5 from scene linear."""
    logged = BMD_A * torch.log((light + BMD_B).clamp(min=1e-300)) + BMD_C
    return torch.where(light < BMD_LIN_CUT, BMD_D * light + BMD_E, logged)


def _bmd_decode(codes, white=None):
    """Scene linear from Blackmagic Film Gen 5."""
    logged = torch.exp((codes - BMD_C) / BMD_A) - BMD_B
    return torch.where(codes < BMD_LOG_CUT, (codes - BMD_E) / BMD_D, logged)


# DaVinci Resolve Wide Gamut Intermediate.
DI_A = 0.0075
DI_B = 7.0
DI_C = 0.07329248
DI_M = 10.44426855
DI_LIN_CUT = 0.00262409
DI_LOG_CUT = 0.02740668


def _di_encode(light, white=None):
    """DaVinci Intermediate from scene linear."""
    logged = DI_C * (torch.log2((light + DI_A).clamp(min=1e-300)) + DI_B)
    return torch.where(light <= DI_LIN_CUT, light * DI_M, logged)


def _di_decode(codes, white=None):
    """Scene linear from DaVinci Intermediate."""
    return torch.where(codes <= DI_LOG_CUT, codes / DI_M, torch.exp2(codes / DI_C - DI_B) - DI_A)


@dataclass(frozen=True)
class Transfer:
    """One transfer function in both directions.

    Attributes:
        encode: ``(light, white_nits) -> codes``.
        decode: ``(codes, white_nits) -> light``.
    """

    encode: Callable
    decode: Callable


#: Widget option -> transfer function.
TRANSFERS = {
    "Linear": Transfer(_linear, _linear),
    "sRGB": Transfer(_srgb_encode, _srgb_decode),
    "Gamma 2.2": Transfer(_power(1.0 / 2.2), _power(2.2)),
    "Gamma 2.4 (BT.1886)": Transfer(_power(1.0 / 2.4), _power(2.4)),
    "Gamma 2.6": Transfer(_power(1.0 / 2.6), _power(2.6)),
    "Rec.709 camera (BT.709 OETF)": Transfer(_bt709_encode, _bt709_decode),
    "Rec.2100 PQ (ST 2084)": Transfer(_pq_encode, _pq_decode),
    "Rec.2100 HLG": Transfer(_hlg_encode, _hlg_decode),
    "ACEScct": Transfer(_acescct_encode, _acescct_decode),
    "Sony S-Log3": Transfer(_slog3_encode, _slog3_decode),
    "ARRI LogC3 (EI 800)": Transfer(_logc3_encode, _logc3_decode),
    "ARRI LogC4": Transfer(_logc4_encode, _logc4_decode),
    "Panasonic V-Log": Transfer(_vlog_encode, _vlog_decode),
    "Fujifilm F-Log2": Transfer(_flog2_encode, _flog2_decode),
    "Apple Log": Transfer(_apple_encode, _apple_decode),
    "Canon Log 3": Transfer(_clog3_encode, _clog3_decode),
    "RED Log3G10": Transfer(_log3g10_encode, _log3g10_decode),
    "Blackmagic Film Gen 5": Transfer(_bmd_encode, _bmd_decode),
    "DaVinci Intermediate": Transfer(_di_encode, _di_decode),
}


# ---------------------------------------------------------------------------------------
# Gamut matrices.


def _xyz(xy) -> torch.Tensor:
    """XYZ of a chromaticity at Y = 1."""
    x, y = (float(value) for value in xy)
    return torch.tensor([x / y, 1.0, (1.0 - x - y) / y], dtype=torch.float64)


def _to_xyz(gamut: Gamut) -> torch.Tensor:
    """RGB to XYZ: the published matrix, or SMPTE RP 177's derivation from the primaries."""
    if gamut.to_xyz is not None:
        return torch.tensor(gamut.to_xyz, dtype=torch.float64)
    primaries = torch.stack([_xyz(xy) for xy in gamut.primaries], dim=1)
    scale = torch.linalg.solve(primaries, _xyz(gamut.white))
    return primaries * scale


def _from_xyz(gamut: Gamut) -> torch.Tensor:
    """XYZ to RGB: the published matrix, or the inverse of :func:`_to_xyz`."""
    if gamut.from_xyz is not None:
        return torch.tensor(gamut.from_xyz, dtype=torch.float64)
    return torch.linalg.inv(_to_xyz(gamut))


def _adaptation(source_white, target_white, method: str) -> torch.Tensor:
    """Von Kries adaptation in the cone space ``method`` names, identity for ``None``."""
    cones = ADAPTATIONS[method]
    if cones is None:
        return torch.eye(3, dtype=torch.float64)
    response = torch.tensor(cones, dtype=torch.float64)
    gains = (response @ _xyz(target_white)) / (response @ _xyz(source_white))
    return torch.linalg.inv(response) @ torch.diag(gains) @ response


@functools.lru_cache(maxsize=256)
def gamut_matrix(source: str, target: str, adaptation: str = "Bradford") -> tuple:
    """The matrix taking linear RGB in one gamut to linear RGB in another.

    Args:
        source: A key of :data:`GAMUTS`.
        target: A key of :data:`GAMUTS`.
        adaptation: A key of :data:`ADAPTATIONS`, applied between the two white points.

    Returns:
        Three rows of three float64 values.

    Raises:
        KeyError: A name is not in its table.
    """
    from_gamut, to_gamut = GAMUTS[source], GAMUTS[target]
    adapt = _adaptation(from_gamut.white, to_gamut.white, adaptation)
    matrix = _from_xyz(to_gamut) @ adapt @ _to_xyz(from_gamut)
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


# ---------------------------------------------------------------------------------------
# Conversion.


def _to_curve(codes: torch.Tensor, levels: str) -> torch.Tensor:
    """Image codes to the curve's own values: ``legal`` reads them as 10-bit video levels."""
    if levels == "legal":
        return (LEGAL_BLACK + LEGAL_SPAN * codes) / CODE_MAX
    return codes


def _from_curve(values: torch.Tensor, levels: str) -> torch.Tensor:
    """The curve's own values to image codes: inverse of :func:`_to_curve`."""
    if levels == "legal":
        return (values * CODE_MAX - LEGAL_BLACK) / LEGAL_SPAN
    return values


def convert(
    images: torch.Tensor,
    source_gamut: str,
    source_transfer: str,
    target_gamut: str,
    target_transfer: str,
    adaptation: str = "Bradford",
    source_levels: str = "full",
    target_levels: str = "full",
    white_nits: float = DEFAULT_WHITE_NITS,
    device=None,
) -> torch.Tensor:
    """Convert a batch from one colour space to another, frame by frame in float64.

    Args:
        images: ``(batch, height, width, channels)``, three or more channels; any past the
            third pass through untouched.
        source_gamut: A key of :data:`GAMUTS`.
        source_transfer: A key of :data:`TRANSFERS`.
        target_gamut: A key of :data:`GAMUTS`.
        target_transfer: A key of :data:`TRANSFERS`.
        adaptation: A key of :data:`ADAPTATIONS`.
        source_levels: A member of :data:`LEVELS` for the input codes.
        target_levels: A member of :data:`LEVELS` for the output codes.
        white_nits: cd/m2 that linear 1.0 stands for on a PQ or HLG side.
        device: Where the arithmetic runs, the images' own device when ``None``.

    Returns:
        A tensor of the input's shape, dtype and device.

    Raises:
        KeyError: A name is not in its table.
        ValueError: The images have fewer than three channels.
    """
    if images.shape[-1] < 3:
        raise ValueError(
            f"Image Color Space needs RGB images, and these have {images.shape[-1]} channel(s). "
            f"Convert greyscale to RGB first, for example with Images to RGB."
        )
    if source_levels not in LEVELS or target_levels not in LEVELS:
        raise KeyError(f"levels must be one of {', '.join(LEVELS)}")
    decode = TRANSFERS[source_transfer].decode
    encode = TRANSFERS[target_transfer].encode
    same_gamut = source_gamut == target_gamut
    matrix = None if same_gamut else torch.tensor(
        gamut_matrix(source_gamut, target_gamut, adaptation), dtype=torch.float64
    )
    if same_gamut and source_transfer == target_transfer and source_levels == target_levels:
        return images

    device = images.device if device is None else device
    matrix = None if matrix is None else matrix.to(device).T
    white = float(white_nits)
    result = images.clone()
    for index, frame in enumerate(images):
        codes = frame[..., :3].to(device=device, dtype=torch.float64)
        light = decode(_to_curve(codes, source_levels), white)
        if matrix is not None:
            light = light @ matrix
        answer = _from_curve(encode(light, white), target_levels)
        result[index, ..., :3] = answer.to(device=images.device, dtype=images.dtype)
    return result
