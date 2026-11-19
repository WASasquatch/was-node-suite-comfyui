"""The ACES 2.0 Output Transform, scene-linear light to display code values, in torch.

A port of the Academy's aces-core CTL (Apache-2.0, ``LICENSE.aces``) following OpenColorIO's
implementation (BSD-3-Clause, ``LICENSE.opencolorio``). Tables are float32, built once per preset.
"""

from __future__ import annotations

__all__ = ["PRESETS", "SOURCES", "WHITES", "Preset", "render"]

import functools
import math
from dataclasses import dataclass

import numpy as np
import torch

from .accelerate import run_frames

#: Chromaticities as ``(red, green, blue, white)`` xy pairs.
AP0 = ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.077), (0.32168, 0.33767))
AP1 = ((0.713, 0.293), (0.165, 0.83), (0.128, 0.044), (0.32168, 0.33767))
REC709 = ((0.64, 0.33), (0.3, 0.6), (0.15, 0.06), (0.3127, 0.329))
P3_D65 = ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06), (0.3127, 0.329))
REC2020 = ((0.708, 0.292), (0.17, 0.797), (0.131, 0.046), (0.3127, 0.329))
XYZ_E = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0))
CAM16 = ((0.8336, 0.1735), (2.3854, -1.4659), (0.087, -0.125), (0.333, 0.333))

#: The ACES white, which the ``D60 sim`` renders hold.
D60 = (0.32168, 0.33767)

#: Bradford cone response, XYZ to LMS by rows.
BRADFORD = ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296))

#: Scene-linear spaces the transform reads, in menu order.
SOURCES = {
    "Linear Rec.709 (sRGB)": REC709,
    "ACEScg": AP1,
    "ACES2065-1": AP0,
    "Linear P3-D65": P3_D65,
    "Linear Rec.2020": REC2020,
}

#: Creative whites, in menu order.
WHITES = ("D65", "D60 sim")


@dataclass(frozen=True)
class Preset:
    """One display the transform renders for.

    Attributes:
        peak: Peak luminance the tone scale targets, in nits.
        limit: Primaries the colours are held inside.
        encoding: Primaries of the display signal, ``XYZ_E`` for DCDM.
        eotf: ``bt1886``, ``srgb``, ``gamma22``, ``gamma26``, ``pq`` or ``hlg``.
        scale: Linear scale applied before the encoding.
        d60_scale_white: Whether the ``D60 sim`` render scales white down to fit.
    """

    peak: float
    limit: tuple
    encoding: tuple
    eotf: str
    scale: float = 1.0
    d60_scale_white: bool = True


#: The ACES 2.0 reference outputs, in menu order.
PRESETS = {
    "SDR 100 nit sRGB": Preset(100.0, REC709, REC709, "srgb"),
    "SDR 100 nit sRGB gamma 2.2": Preset(100.0, REC709, REC709, "gamma22"),
    "SDR 100 nit Rec.709 BT.1886": Preset(100.0, REC709, REC709, "bt1886"),
    "SDR 100 nit Display P3": Preset(100.0, P3_D65, P3_D65, "srgb"),
    "SDR 100 nit Display P3 gamma 2.2": Preset(100.0, P3_D65, P3_D65, "gamma22"),
    "SDR 100 nit Display P3, Rec.709 limited": Preset(100.0, REC709, P3_D65, "srgb"),
    "SDR 100 nit P3-D65 PQ, Rec.709 limited": Preset(100.0, REC709, P3_D65, "pq"),
    "SDR 100 nit Rec.2100 PQ, Rec.709 limited": Preset(100.0, REC709, REC2020, "pq"),
    "SDR 48 nit cinema P3-D65 gamma 2.6": Preset(100.0, P3_D65, P3_D65, "gamma26"),
    "SDR 48 nit cinema P3-D65 gamma 2.6, Rec.709 limited": Preset(100.0, REC709, P3_D65, "gamma26"),
    "SDR 48 nit cinema DCDM XYZ, P3 limited": Preset(
        100.0, P3_D65, XYZ_E, "gamma26", 48.0 / 52.37, False
    ),
    "HDR 108 nit cinema P3-D65 PQ": Preset(225.0, P3_D65, P3_D65, "pq", 0.48),
    "HDR 300 nit cinema DCDM XYZ PQ, P3 limited": Preset(625.0, P3_D65, XYZ_E, "pq", 0.48),
    "HDR 500 nit P3-D65 PQ": Preset(500.0, P3_D65, P3_D65, "pq"),
    "HDR 1000 nit P3-D65 PQ": Preset(1000.0, P3_D65, P3_D65, "pq"),
    "HDR 2000 nit P3-D65 PQ": Preset(2000.0, P3_D65, P3_D65, "pq"),
    "HDR 4000 nit P3-D65 PQ": Preset(4000.0, P3_D65, P3_D65, "pq"),
    "HDR 500 nit Rec.2100 PQ, P3 limited": Preset(500.0, P3_D65, REC2020, "pq"),
    "HDR 1000 nit Rec.2100 PQ, P3 limited": Preset(1000.0, P3_D65, REC2020, "pq"),
    "HDR 2000 nit Rec.2100 PQ, P3 limited": Preset(2000.0, P3_D65, REC2020, "pq"),
    "HDR 4000 nit Rec.2100 PQ, P3 limited": Preset(4000.0, P3_D65, REC2020, "pq"),
    "HDR 1000 nit Rec.2100 HLG, P3 limited": Preset(1000.0, P3_D65, REC2020, "hlg"),
    "HDR 500 nit Rec.2100 PQ": Preset(500.0, REC2020, REC2020, "pq"),
    "HDR 1000 nit Rec.2100 PQ": Preset(1000.0, REC2020, REC2020, "pq"),
    "HDR 2000 nit Rec.2100 PQ": Preset(2000.0, REC2020, REC2020, "pq"),
    "HDR 4000 nit Rec.2100 PQ": Preset(4000.0, REC2020, REC2020, "pq"),
    "HDR 1000 nit Display P3 extended": Preset(1000.0, P3_D65, P3_D65, "srgb"),
}

_F = np.float32


def _f(value) -> float:
    """``value`` rounded to float32, as a Python float."""
    return float(_F(value))


PI = _f(3.14159265358979)
EXP_042 = _f(0.42)
INV_042 = float(_F(1.0) / _F(0.42))
CAM_NL_OFFSET = float(_F(0.2713) * _F(100.0))
SURROUND = (_f(0.9), _f(0.59), _f(0.9))
MODEL_GAMMA = float(_F(SURROUND[1]) * (_F(1.48) + np.sqrt(_F(20.0) / _F(100.0))))
MODEL_GAMMA_INV = float(_F(1.0) / _F(MODEL_GAMMA))

SMOOTH_CUSPS = _f(0.12)
SMOOTH_M = _f(0.27)
CUSP_MID_BLEND = _f(1.3)
FOCUS_GAIN_BLEND = _f(0.3)
COMPRESSION_THRESHOLD = _f(0.75)
SIXTH = float(_F(1.0) / _F(6.0))

#: Chroma normalisation weights for cos h, cos 2h, cos 3h, then sin h, sin 2h, sin 3h, 1.
NORM_WEIGHTS = tuple(_f(w) for w in (11.34072, 16.46899, 7.88380, 14.66441, -6.37224, 9.19364, 77.12896))

TABLE = 360
TABLE_TOTAL = TABLE + 3
CORNERS = 6

#: SMPTE ST 2084 constants.
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0

#: Pixels evaluated per pass.
PIXELS_PER_PASS = 1 << 21


class _NumpyOps:
    """Elementwise maths on float32 numpy arrays."""

    abs = staticmethod(np.abs)
    sqrt = staticmethod(np.sqrt)
    copysign = staticmethod(np.copysign)
    atan2 = staticmethod(np.arctan2)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    log10 = staticmethod(np.log10)
    where = staticmethod(np.where)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)


class _TorchOps:
    """Elementwise maths on float32 tensors."""

    abs = staticmethod(torch.abs)
    sqrt = staticmethod(torch.sqrt)
    copysign = staticmethod(torch.copysign)
    atan2 = staticmethod(torch.atan2)
    cos = staticmethod(torch.cos)
    sin = staticmethod(torch.sin)
    log10 = staticmethod(torch.log10)
    where = staticmethod(torch.where)

    @staticmethod
    def minimum(x, y):
        """Elementwise minimum against a tensor or a number."""
        return torch.minimum(x, y) if isinstance(y, torch.Tensor) else torch.clamp(x, max=y)

    @staticmethod
    def maximum(x, y):
        """Elementwise maximum against a tensor or a number."""
        return torch.maximum(x, y) if isinstance(y, torch.Tensor) else torch.clamp(x, min=y)


NP = _NumpyOps
TO = _TorchOps


# Matrices, built in double from chromaticities.

def _rgb_to_xyz(prims) -> np.ndarray:
    """RGB to XYZ for a set of primaries, white at Y = 1, in float64."""
    (rx, ry), (gx, gy), (bx, by), (wx, wy) = prims
    m = np.array(
        [[rx, gx, bx], [ry, gy, by], [1.0 - rx - ry, 1.0 - gx - gy, 1.0 - bx - by]], np.float64
    )
    white = np.array([wx / wy, 1.0, (1.0 - wx - wy) / wy], np.float64)
    gains = np.linalg.inv(m) @ white
    return m * gains[None, :]


def _conversion(src, dst, adapt: bool = False) -> np.ndarray:
    """RGB in ``src`` primaries to RGB in ``dst``, in float64, Bradford adapted when asked."""
    src_xyz = _rgb_to_xyz(src)
    dst_rgb = np.linalg.inv(_rgb_to_xyz(dst))
    if not adapt or tuple(src[3]) == tuple(dst[3]):
        return dst_rgb @ src_xyz
    cone = np.array(BRADFORD, np.float64)
    src_white = cone @ (src_xyz @ np.ones(3))
    dst_white = cone @ (_rgb_to_xyz(dst) @ np.ones(3))
    adaptation = np.linalg.inv(cone) @ np.diag(dst_white / src_white) @ cone
    return dst_rgb @ adaptation @ src_xyz


def _single(m) -> tuple:
    """A 3x3 matrix as nested tuples of float32 values."""
    m = np.asarray(m, np.float64).astype(np.float32)
    return tuple(tuple(float(v) for v in row) for row in m)


def _mul33(a, b) -> tuple:
    """Float32 product of two 3x3 matrices, summed left to right."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return tuple(
        tuple(float(a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]) for j in range(3))
        for i in range(3)
    )


def _mul3(m, v) -> tuple:
    """Float32 product of a 3x3 matrix and a vector, summed left to right."""
    m = np.asarray(m, np.float32)
    v = [_F(x) for x in v]
    return tuple(float(v[0] * m[i, 0] + v[1] * m[i, 1] + v[2] * m[i, 2]) for i in range(3))


def _invert(m) -> tuple:
    """Inverse of a float32 matrix, taken in float64 and rounded to float32."""
    return _single(np.linalg.inv(np.asarray(m, np.float64)))


def _apply(m, x, y, z):
    """``m`` times ``(x, y, z)``, each row summed left to right."""
    return (
        x * m[0][0] + y * m[0][1] + z * m[0][2],
        x * m[1][0] + y * m[1][1] + z * m[1][2],
        x * m[2][0] + y * m[2][1] + z * m[2][2],
    )


# The colour appearance model.

@dataclass(frozen=True)
class _JMh:
    """Conversion constants between RGB in one set of primaries and JMh."""

    to_cam: tuple
    from_cam: tuple
    to_aab: tuple
    from_aab: tuple
    f_l_n: float
    cz: float
    inv_cz: float
    a_w_j: float
    inv_a_w_j: float


def _compress_scalar(v):
    """Cone response compression of a non-negative float32 value."""
    f = _F(v) ** _F(EXP_042)
    return f / (_F(CAM_NL_OFFSET) + f)


def _jmh_params(prims) -> _JMh:
    """JMh conversion constants for a set of primaries."""
    base = (
        (_F(2.0), _F(1.0), _F(1.0) / _F(20.0)),
        (_F(1.0), _F(-12.0) / _F(11.0), _F(1.0) / _F(11.0)),
        (_F(1.0) / _F(9.0), _F(1.0) / _F(9.0), _F(-2.0) / _F(9.0)),
    )
    to_e = _conversion(CAM16, XYZ_E)
    m16 = _single(np.linalg.inv(to_e))
    rgb_to_xyz = _single(_conversion(prims, XYZ_E))
    xyz_w = _mul3(rgb_to_xyz, (100.0, 100.0, 100.0))
    y_w = _F(xyz_w[1])
    rgb_w = _mul3(m16, xyz_w)

    k = _F(1.0) / (_F(5.0) * _F(100.0) + _F(1.0))
    k4 = k * k * k * k
    f_l = (_F(0.2) * k4 * (_F(5.0) * _F(100.0))
           + _F(0.1) * ((_F(1.0) - k4) ** _F(2.0)) * ((_F(5.0) * _F(100.0)) ** (_F(1.0) / _F(3.0))))
    f_l_n = f_l / _F(100.0)
    cz = _F(MODEL_GAMMA)

    d_rgb = tuple(f_l_n * y_w / _F(c) for c in rgb_w)
    rgb_aw = tuple(_compress_scalar(d * _F(c)) for d, c in zip(d_rgb, rgb_w))

    cone_to_aab = _mul33(np.diag([400.0, 400.0, 400.0]), base)
    c = np.asarray(cone_to_aab, np.float32)
    a_w = c[0, 0] * rgb_aw[0] + c[0, 1] * rgb_aw[1] + c[0, 2] * rgb_aw[2]
    a_w_j = _compress_scalar(f_l)

    rgb_to_cam = _mul33(_mul33(_single(np.linalg.inv(to_e)), rgb_to_xyz), np.diag([100.0, 100.0, 100.0]))
    rgb_to_cam = _mul33(np.diag([float(d) for d in d_rgb]), rgb_to_cam)
    to_aab = (
        tuple(float(c[0, j] / a_w) for j in range(3)),
        tuple(float(c[1, j] * _F(43.0) * _F(SURROUND[2])) for j in range(3)),
        tuple(float(c[2, j] * _F(43.0) * _F(SURROUND[2])) for j in range(3)),
    )
    return _JMh(
        to_cam=rgb_to_cam,
        from_cam=_invert(rgb_to_cam),
        to_aab=to_aab,
        from_aab=_invert(to_aab),
        f_l_n=float(f_l_n),
        cz=float(cz),
        inv_cz=float(_F(1.0) / cz),
        a_w_j=float(a_w_j),
        inv_a_w_j=float(_F(1.0) / a_w_j),
    )


def _compress_fwd(v, xp):
    """Post-adaptation cone response compression, mirrored about zero."""
    f = xp.abs(v) ** EXP_042
    return xp.copysign(f / (CAM_NL_OFFSET + f), v)


def _compress_inv(v, xp):
    """Inverse of :func:`_compress_fwd`."""
    ra = xp.minimum(xp.abs(v), 0.99)
    return xp.copysign(((CAM_NL_OFFSET * ra) / (1.0 - ra)) ** INV_042, v)


def _rgb_to_aab(r, g, b, p: _JMh, xp):
    """RGB to achromatic response and opponent dimensions."""
    r, g, b = _apply(p.to_cam, r, g, b)
    return _apply(p.to_aab, _compress_fwd(r, xp), _compress_fwd(g, xp), _compress_fwd(b, xp))


def _aab_to_jmh(A, a, b, p: _JMh, xp):
    """Achromatic and opponent values to J, M and hue in degrees; zero where A is not positive."""
    positive = A > 0.0
    j = 100.0 * xp.abs(A) ** p.cz
    m = xp.sqrt(a * a + b * b)
    h = (180.0 * xp.atan2(b, a)) / PI
    h = xp.where(h < 0.0, h + 360.0, h)
    return xp.where(positive, j, 0.0), xp.where(positive, m, 0.0), xp.where(positive, h, 0.0)


def _rgb_to_jmh(r, g, b, p: _JMh, xp):
    """RGB to J, M and hue."""
    return _aab_to_jmh(*_rgb_to_aab(r, g, b, p, xp), p, xp)


def _jmh_to_rgb_trig(j, m, cos_h, sin_h, p: _JMh, xp):
    """J and M with the hue's cosine and sine back to RGB."""
    A = (j * 0.01) ** p.inv_cz
    r, g, b = _apply(p.from_aab, A, m * cos_h, m * sin_h)
    return _apply(p.from_cam, _compress_inv(r, xp), _compress_inv(g, xp), _compress_inv(b, xp))


def _jmh_to_rgb(j, m, h, p: _JMh, xp):
    """J, M and hue in degrees back to RGB."""
    radians = (PI * h) / 180.0
    return _jmh_to_rgb_trig(j, m, xp.cos(radians), xp.sin(radians), p, xp)


def _y_to_j(y, p: _JMh, xp):
    """Luminance to J, mirrored about zero."""
    ra = _compress_fwd(xp.abs(y) * p.f_l_n, xp)
    return xp.copysign(100.0 * (ra * p.inv_a_w_j) ** p.cz, y)


# The tone scale and chroma compression.

@dataclass(frozen=True)
class _Tone:
    """Tone scale and chroma compression constants for one peak luminance."""

    peak: float
    g: float
    t_1: float
    c_t: float
    s_2: float
    m_2: float
    forward_limit: float
    log_peak: float
    sat: float
    sat_thr: float
    compr: float
    chroma_scale: float


def _tone_params(peak: float) -> _Tone:
    """Tone scale constants for a peak luminance in nits."""
    n = _F(peak)
    n_r = _F(100.0)
    g = _F(1.15)
    c = _F(0.18)
    c_d = _F(10.013)
    w_g = _F(0.14)
    t_1 = _F(0.04)
    r_hit_min = _F(128.0)
    r_hit_max = _F(896.0)
    r_hit = r_hit_min + (r_hit_max - r_hit_min) * (np.log(n / n_r) / np.log(_F(10000.0) / _F(100.0)))
    m_0 = n / n_r
    m_1 = _F(0.5) * (m_0 + np.sqrt(m_0 * (m_0 + _F(4.0) * t_1)))
    u = ((r_hit / m_1) / ((r_hit / m_1) + _F(1.0))) ** g
    m = m_1 / u
    w_i = np.log(n / _F(100.0)) / np.log(_F(2.0))
    c_t = c_d / n_r * (_F(1.0) + w_i * w_g)
    g_ip = _F(0.5) * (c_t + np.sqrt(c_t * (c_t + _F(4.0) * t_1)))
    g_ipp2 = -(m_1 * ((g_ip / m) ** (_F(1.0) / g))) / (((g_ip / m) ** (_F(1.0) / g)) - _F(1.0))
    w_2 = c / g_ipp2
    s_2 = w_2 * m_1 * _F(100.0)
    u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)) ** g
    m_2 = m_1 / u_2
    log_peak = np.log10(n / n_r)
    compr = _F(2.4) + (_F(2.4) * _F(3.3)) * log_peak
    sat = max(_F(0.2), _F(1.3) - (_F(1.3) * _F(0.69)) * log_peak)
    chroma_scale = (_F(0.03379) * n) ** _F(0.30596) - _F(0.45135)
    return _Tone(
        peak=float(n), g=float(g), t_1=float(t_1), c_t=float(c_t), s_2=float(s_2),
        m_2=float(m_2), forward_limit=float(_F(8.0) * r_hit), log_peak=float(log_peak),
        sat=float(sat), sat_thr=float(_F(0.5) / n), compr=float(compr),
        chroma_scale=float(chroma_scale),
    )


def _tonescale_a_to_j(A, p: _JMh, t: _Tone, xp):
    """J after the tone scale, from the achromatic response; zero where A is not positive."""
    y = _compress_inv(p.a_w_j * A, xp) / p.f_l_n
    f = t.m_2 * (y / (y + t.s_2)) ** t.g
    y_ts = xp.maximum(f * f / (f + t.t_1), 0.0) * 100.0
    j = 100.0 * (_compress_fwd(y_ts * p.f_l_n, xp) * p.inv_a_w_j) ** p.cz
    return xp.where(A > 0.0, j, 0.0)


def _chroma_norm(cos_h, sin_h, scale: float):
    """Hue-dependent chroma normalisation."""
    cos_2 = 2.0 * cos_h * cos_h - 1.0
    sin_2 = 2.0 * cos_h * sin_h
    cos_3 = 4.0 * cos_h * cos_h * cos_h - 3.0 * cos_h
    sin_3 = 3.0 * sin_h - 4.0 * sin_h * sin_h * sin_h
    w = NORM_WEIGHTS
    m = (w[0] * cos_h + w[1] * cos_2 + w[2] * cos_3 + w[3] * sin_h + w[4] * sin_2
         + w[5] * sin_3 + w[6])
    return m * scale


def _toe(x, limit, k1_in, k2_in, xp):
    """The chroma compression toe, identity above ``limit``."""
    k2 = xp.maximum(k2_in, 0.001)
    k1 = xp.sqrt(k1_in * k1_in + k2 * k2)
    k3 = (limit + k1) / (limit + k2)
    minus_b = k3 * x - k1
    minus_ac = k2 * k3 * x
    toed = 0.5 * (minus_b + xp.sqrt(minus_b * minus_b + 4.0 * minus_ac))
    return xp.where(x > limit, x, toed)


def _chroma_compress(j, m, j_ts, m_norm, reach_m, limit_j_max: float, t: _Tone, xp):
    """Colourfulness after the tone scale's chroma compression."""
    nj = j_ts / limit_j_max
    snj = xp.maximum(1.0 - nj, 0.0)
    limit = nj ** MODEL_GAMMA_INV * reach_m / m_norm
    mc = m * (j_ts / j) ** MODEL_GAMMA_INV
    mc = mc / m_norm
    mc = limit - _toe(limit - mc, limit - 0.001, snj * t.sat, xp.sqrt(nj * nj + t.sat_thr), xp)
    mc = _toe(mc, limit, nj * t.compr, snj, xp)
    return xp.where(m != 0.0, mc * m_norm, m)


# Gamut compression.

def _focus_gain(j, threshold, limit_j_max: float, gain: float, xp):
    """Slope gain of the compression vector, raised above the analytical threshold."""
    adjust = xp.log10((limit_j_max - threshold) / xp.maximum(limit_j_max - j, 0.0001))
    adjust = adjust * adjust + 1.0
    return xp.where(j > threshold, gain * adjust, gain)


def _solve_j_intersect(j, m, focus_j, max_j: float, slope_gain, xp):
    """J where the compression vector through ``(j, m)`` meets the J axis."""
    m_scaled = m / slope_gain
    a = m_scaled / focus_j
    b_low = 1.0 - m_scaled
    c_low = -j
    low = -2.0 * c_low / (b_low + xp.sqrt(b_low * b_low - 4.0 * a * c_low))
    b_high = -(1.0 + m_scaled + max_j * a)
    c_high = max_j * m_scaled + j
    high = -2.0 * c_high / (b_high - xp.sqrt(b_high * b_high - 4.0 * a * c_high))
    return xp.where(j < focus_j, low, high)


def _vector_slope(intersect_j, focus_j, limit_j_max: float, slope_gain, xp):
    """Slope of the compression vector leaving the J axis at ``intersect_j``."""
    scaler = xp.where(intersect_j < focus_j, intersect_j, limit_j_max - intersect_j)
    return scaler * (intersect_j - focus_j) / (focus_j * slope_gain)


def _boundary_m(axis_j, slope, inv_gamma, j_max, m_max, j_reference):
    """Approximate M where the vector meets a boundary ``J = j_max (M / m_max) ^ (1 / inv_gamma)``."""
    shifted = j_reference * (axis_j / j_reference) ** inv_gamma
    return shifted * m_max / (j_max - slope * m_max)


def _gamut_boundary(cusp_j, cusp_m, j_max: float, gamma_top_inv, gamma_bottom_inv: float,
                    j_source, slope, j_cusp, xp):
    """M where the compression vector leaves the limiting gamut, smoothed at the cusp."""
    lower = _boundary_m(j_source, slope, gamma_bottom_inv, cusp_j, cusp_m, j_cusp)
    upper = _boundary_m(j_max - j_source, -slope, gamma_top_inv, j_max - cusp_j, cusp_m, j_max - j_cusp)
    s = SMOOTH_CUSPS * cusp_m
    h = xp.maximum(s - xp.abs(lower - upper), 0.0) / s
    return xp.minimum(lower, upper) - h * h * h * s * SIXTH


def _remap_m(m, gamut_m, reach_m, xp):
    """Colourfulness pulled from the reach boundary inside the gamut boundary."""
    proportion = xp.maximum(gamut_m / reach_m, COMPRESSION_THRESHOLD)
    threshold = proportion * gamut_m
    offset = m - threshold
    gamut_offset = gamut_m - threshold
    reach_offset = reach_m - threshold
    scale = reach_offset / ((reach_offset / gamut_offset) - 1.0)
    nd = offset / scale
    remapped = threshold + scale * nd / (1.0 + nd)
    return xp.where((m <= threshold) | (proportion >= 1.0), m, remapped)


def _lerp(a, b, t):
    """Linear interpolation, ``(b - a) * t + a``."""
    return (b - a) * t + a


@dataclass(frozen=True)
class _Gamut:
    """Gamut compression constants and tables for one preset."""

    limit_j_max: float
    mid_j: float
    focus_dist: float
    focus_gain: float
    lower_gamma_inv: float
    reach: np.ndarray
    hues: np.ndarray
    cusps: np.ndarray
    search: tuple


def _focus_j(cusp_j, mid_j: float, limit_j_max: float, xp):
    """J the compression vectors converge on."""
    return _lerp(cusp_j, mid_j, xp.minimum(CUSP_MID_BLEND - (cusp_j / limit_j_max), 1.0))


def _compress_gamut(j, m, cusp_j, cusp_m, gamma_top_inv, reach_m, g: _Gamut, xp):
    """J and M inside the limiting gamut."""
    focus_j = _focus_j(cusp_j, g.mid_j, g.limit_j_max, xp)
    threshold = _lerp(cusp_j, g.limit_j_max, FOCUS_GAIN_BLEND)
    gain = _focus_gain(j, threshold, g.limit_j_max, g.focus_gain, xp)
    j_source = _solve_j_intersect(j, m, focus_j, g.limit_j_max, gain, xp)
    slope = _vector_slope(j_source, focus_j, g.limit_j_max, gain, xp)
    j_cusp = _solve_j_intersect(cusp_j, cusp_m, focus_j, g.limit_j_max, gain, xp)
    gamut_m = _gamut_boundary(cusp_j, cusp_m, g.limit_j_max, gamma_top_inv, g.lower_gamma_inv,
                              j_source, slope, j_cusp, xp)
    reach_boundary = _boundary_m(j_source, slope, MODEL_GAMMA_INV, g.limit_j_max, reach_m, g.limit_j_max)
    remapped = _remap_m(m, gamut_m, reach_boundary, xp)
    inside = gamut_m > 0.0
    return (xp.where(inside, j_source + remapped * slope, j),
            xp.where(inside, remapped, 0.0))


# Table construction, in float32 on the CPU.

def _corner(index: int) -> tuple:
    """Unit cube corner ``index`` in the order red, yellow, green, cyan, blue, magenta."""
    return (float(((index + 1) % CORNERS) < 3), float(((index + 5) % CORNERS) < 3),
            float(((index + 3) % CORNERS) < 3))


def _arr(value, size: int = 1) -> np.ndarray:
    """A float32 array of ``size`` copies of ``value``."""
    return np.full(size, value, np.float32)


def _lowest_first(jmh: list) -> int:
    """Index of the first corner holding the lowest hue."""
    index = 0
    for i in range(1, CORNERS):
        if jmh[i][2] < jmh[index][2]:
            index = i
    return index


def _rotate(items: list, start: int) -> list:
    """``items`` rotated to begin at ``start``, with a wrapped copy at each end."""
    turned = [items[(i + start) % CORNERS] for i in range(CORNERS)]
    return [turned[-1]] + turned + [turned[0]]


def _wrap_hues(jmh: list) -> list:
    """Corner JMh with the first hue moved down a turn and the last up a turn."""
    jmh = list(jmh)
    jmh[0] = (jmh[0][0], jmh[0][1], float(_F(jmh[0][2]) - _F(360.0)))
    jmh[-1] = (jmh[-1][0], jmh[-1][1], float(_F(jmh[-1][2]) + _F(360.0)))
    return jmh


def _jmh_of(rgb, p: _JMh) -> tuple:
    """JMh of one RGB triple, in float32."""
    j, m, h = _rgb_to_jmh(_arr(rgb[0]), _arr(rgb[1]), _arr(rgb[2]), p, NP)
    return float(j[0]), float(m[0]), float(h[0])


def _limit_corners(p: _JMh, peak: float):
    """RGB and JMh of the limiting cube's six corners at peak, cycled by hue."""
    scale = _F(peak) / _F(100.0)
    rgb = [tuple(float(scale * _F(c)) for c in _corner(i)) for i in range(CORNERS)]
    jmh = [_jmh_of(c, p) for c in rgb]
    start = _lowest_first(jmh)
    return _rotate(rgb, start), _wrap_hues(_rotate(jmh, start))


def _reach_corners(p: _JMh, limit_j: float, maximum: float) -> list:
    """JMh of the reach cube's corners where they reach ``limit_j``, cycled by hue."""
    limit_a = (_F(limit_j) * _F(0.01)) ** _F(p.inv_cz)
    vectors = np.array([_corner(i) for i in range(CORNERS)], np.float32)
    lower = _arr(0.0, CORNERS)
    upper = _arr(maximum, CORNERS)
    done = np.zeros(CORNERS, bool)
    tolerance = _F(1e-3)
    while True:
        active = ~done & ((upper - lower) > tolerance)
        if not active.any():
            break
        test = (lower + upper) / _F(2.0)
        a = _rgb_to_aab(test * vectors[:, 0], test * vectors[:, 1], test * vectors[:, 2], p, NP)[0]
        lower = np.where(active & (a < limit_a), test, lower)
        upper = np.where(active & ~(a < limit_a), test, upper)
        done |= active & (a == limit_a)
    jmh = [_jmh_of(tuple(float(upper[i] * vectors[i, c]) for c in range(3)), p) for i in range(CORNERS)]
    return _wrap_hues(_rotate(jmh, _lowest_first(jmh)))


def _sorted_hues(reach: list, display: list) -> list:
    """The two corner hue lists merged, a hue both share listed once."""
    hues = []
    r = d = 1
    while r < CORNERS + 1 or d < CORNERS + 1:
        rh, dh = reach[r][2], display[d][2]
        if rh == dh:
            hues.append(rh)
            r += 1
            d += 1
        elif rh < dh:
            hues.append(rh)
            r += 1
        else:
            hues.append(dh)
            d += 1
    return hues


def _hue_table(hues: list) -> np.ndarray:
    """Hue samples near one degree apart that land exactly on every corner hue."""
    table = np.zeros(TABLE_TOTAL, np.float32)
    mask = 0xFFFFFFFF
    counts = [0] * (2 * CORNERS + 2)
    last = mask
    least = 0 if hues[0] == 0.0 else 1
    for index, hue in enumerate(hues):
        nominal = min(max(int(math.floor(float(_F(hue) * _F(1.0)) + 0.5)), least), TABLE - 1)
        if last == nominal:
            if index > 1 and counts[index - 2] != ((counts[index - 1] - 1) & mask):
                counts[index - 1] = (counts[index - 1] - 1) & mask
            else:
                nominal = nominal + 1
        counts[index] = min(nominal, TABLE - 1)
        last = least = nominal

    def fill(samples: int, lower: float, upper: float, base: int) -> None:
        delta = (_F(upper) - _F(lower)) / _F(samples) if samples else _F(0.0)
        for i in range(samples):
            table[base + i] = _F(lower) + _F(i) * delta

    total = 0
    fill(counts[0], 0.0, hues[0], total + 1)
    total += counts[0]
    for index in range(1, len(hues)):
        samples = (counts[index] - counts[index - 1]) & mask
        fill(samples, hues[index - 1], hues[index], total + 1)
        total += samples
    fill(TABLE - total, hues[-1], 360.0, total + 1)
    table[0] = table[TABLE] - _F(360.0)
    table[TABLE + 1] = table[1] + _F(360.0)
    table[TABLE + 2] = table[2] + _F(360.0)
    return table


def _display_cusp(hue: float, rgb: list, jmh: list, p: _JMh, previous: list) -> tuple:
    """J and M of the limiting gamut's cusp at ``hue``, searched along the cube's edges."""
    upper_corner = 1
    for i in range(1, CORNERS + 2):
        if jmh[i][2] > hue:
            upper_corner = i
            break
    lower_corner = upper_corner - 1
    if jmh[lower_corner][2] == hue:
        return jmh[lower_corner][0], jmh[lower_corner][1]
    low_rgb = [_F(v) for v in rgb[lower_corner]]
    high_rgb = [_F(v) for v in rgb[upper_corner]]
    low_hue = _F(jmh[lower_corner][2])
    high_hue = _F(jmh[upper_corner][2])
    target = _F(hue)
    lower_t = _F(previous[1]) if float(upper_corner) == previous[0] else _F(0.0)
    upper_t = _F(1.0)
    tolerance = _F(1e-7)
    two = _F(2.0)

    def sample(t):
        point = [_arr((high_rgb[c] - low_rgb[c]) * t + low_rgb[c]) for c in range(3)]
        return _rgb_to_jmh(point[0], point[1], point[2], p, NP)

    while (upper_t - lower_t) > tolerance:
        t = (lower_t + upper_t) / two
        h = _F(sample(t)[2][0])
        if h < low_hue:
            upper_t = t
        elif h >= high_hue:
            lower_t = t
        elif h > target:
            upper_t = t
        else:
            lower_t = t
    t = (lower_t + upper_t) / two
    j, m, _ = sample(t)
    previous[0] = float(upper_corner)
    previous[1] = float(t)
    return float(j[0]), float(m[0])


def _reach_table(p: _JMh, limit_j_max: float) -> np.ndarray:
    """Largest M inside the reach gamut at ``limit_j_max``, per whole degree of hue."""
    hues = np.arange(TABLE, dtype=np.float32)
    j = _arr(limit_j_max, TABLE)
    step = _F(50.0)
    low = _arr(0.0, TABLE)
    high = low + step
    outside = np.zeros(TABLE, bool)

    def beyond(m):
        r, g, b = _jmh_to_rgb(j, m, hues, p, NP)
        return (r < 0.0) | (g < 0.0) | (b < 0.0)

    while True:
        active = ~outside & (high < _F(1300.0))
        if not active.any():
            break
        found = beyond(high)
        outside = np.where(active, found, outside)
        grow = active & ~found
        low = np.where(grow, high, low)
        high = np.where(grow, high + step, high)
    while True:
        active = (high - low).astype(np.float64) > 1e-2
        if not active.any():
            break
        sample = (high + low) / _F(2.0)
        found = beyond(sample)
        high = np.where(active & found, sample, high)
        low = np.where(active & ~found, sample, low)
    table = np.zeros(TABLE_TOTAL, np.float32)
    table[1:TABLE + 1] = high
    table[0] = table[TABLE]
    table[TABLE + 1] = table[1]
    table[TABLE + 2] = table[2]
    return table


def _upper_gammas(hues: np.ndarray, cusps: np.ndarray, peak: float, g: _Gamut, p: _JMh) -> np.ndarray:
    """Inverse gamma of the gamut's upper hull per hue, the smallest that clears the hull."""
    hue = hues[1:TABLE + 1]
    cusp_j = cusps[1:TABLE + 1, 0]
    cusp_m = cusps[1:TABLE + 1, 1]
    threshold = _lerp(cusp_j, _F(g.limit_j_max), _F(FOCUS_GAIN_BLEND))
    focus_j = _focus_j(cusp_j, g.mid_j, g.limit_j_max, NP)
    tests = []
    for position in (0.01, 0.1, 0.5, 0.8, 0.99):
        test_j = _lerp(cusp_j, _F(g.limit_j_max), _F(position))
        gain = _focus_gain(test_j, threshold, g.limit_j_max, g.focus_gain, NP)
        j_source = _solve_j_intersect(test_j, cusp_m, focus_j, g.limit_j_max, gain, NP)
        slope = _vector_slope(j_source, focus_j, g.limit_j_max, gain, NP)
        j_cusp = _solve_j_intersect(cusp_j, cusp_m, focus_j, g.limit_j_max, gain, NP)
        tests.append((j_source, slope, j_cusp))
    ceiling = _F(peak) / _F(100.0)

    def clears(gamma):
        top_inv = _F(1.0) / gamma
        result = np.ones(TABLE, bool)
        for j_source, slope, j_cusp in tests:
            m = _gamut_boundary(cusp_j, cusp_m, g.limit_j_max, top_inv, g.lower_gamma_inv,
                                j_source, slope, j_cusp, NP)
            j = j_source + slope * m
            r, gg, b = _jmh_to_rgb(j, m, hue, p, NP)
            result &= (r > ceiling) | (gg > ceiling) | (b > ceiling)
        return result

    step = _F(0.4)
    low = _arr(0.0, TABLE)
    high = low + step
    found = np.zeros(TABLE, bool)
    while True:
        active = ~found & (high < _F(5.0))
        if not active.any():
            break
        fit = clears(high)
        found = np.where(active, fit, found)
        grow = active & ~fit
        low = np.where(grow, high, low)
        high = np.where(grow, high + step, high)
    while True:
        active = (high - low) > _F(1e-5)
        if not active.any():
            break
        test = (high + low) / _F(2.0)
        fit = clears(test)
        high = np.where(active & fit, test, high)
        low = np.where(active & ~fit, test, low)
    return _F(1.0) / high


@dataclass(frozen=True)
class _Setup:
    """Everything one preset renders with.

    Attributes:
        tone: Tone scale and chroma compression constants.
        gamut: Gamut compression constants and tables.
        input: JMh constants for ACES2065-1.
        limit: JMh constants for the limiting primaries.
        to_ap0: AP1 to AP0, float32.
        display: Limiting RGB to display RGB with white and linear scale, float32, or ``None``.
        eotf: The display encoding.
    """

    tone: _Tone
    gamut: _Gamut
    input: _JMh
    limit: _JMh
    to_ap0: tuple
    display: tuple | None
    eotf: str


@functools.lru_cache(maxsize=None)
def _core(peak: float, limit_prims: tuple) -> tuple:
    """Tone, gamut and JMh constants for a peak and limiting primaries."""
    with np.errstate(all="ignore"):
        tone = _tone_params(peak)
        source = _jmh_params(AP0)
        reach_params = _jmh_params(AP1)
        limit = _jmh_params(limit_prims)
        limit_j_max = float(_y_to_j(_arr(peak), source, NP)[0])
        mid_j = float(_y_to_j(_arr(float(_F(tone.c_t) * _F(100.0))), source, NP)[0])
        focus_dist = _F(1.35) + _F(1.35) * _F(1.75) * _F(tone.log_peak)
        reach = _reach_table(reach_params, limit_j_max)

        reach_jmh = _reach_corners(reach_params, limit_j_max, tone.forward_limit)
        rgb_corners, limit_jmh = _limit_corners(limit, peak)
        hues = _hue_table(_sorted_hues(reach_jmh, limit_jmh))
        cusps = np.zeros((TABLE_TOTAL, 3), np.float32)
        previous = [0.0, 0.0]
        grow = _F(1.0) + _F(SMOOTH_M) * _F(SMOOTH_CUSPS)
        for i in range(1, TABLE + 1):
            j, m = _display_cusp(float(hues[i]), rgb_corners, limit_jmh, limit, previous)
            cusps[i] = (j, _F(m) * grow, hues[i])
        cusps[0] = (cusps[TABLE, 0], cusps[TABLE, 1], hues[0])
        cusps[TABLE + 1] = (cusps[1, 0], cusps[1, 1], hues[TABLE + 1])
        cusps[TABLE + 2] = (cusps[2, 0], cusps[2, 1], hues[TABLE + 2])

        low, high = 0, 1
        for i in range(1, TABLE + 1):
            delta = i - (1 + int(cusps[i, 2]))
            low = min(low, delta)
            high = max(high, delta + 1)

        gamut = _Gamut(
            limit_j_max=limit_j_max,
            mid_j=mid_j,
            focus_dist=float(focus_dist),
            focus_gain=float(_F(limit_j_max) * focus_dist),
            lower_gamma_inv=float(_F(1.0) / (_F(1.14) + _F(0.07) * _F(tone.log_peak))),
            reach=reach,
            hues=hues,
            cusps=cusps,
            search=(low, high),
        )
        gammas = _upper_gammas(hues, cusps, peak, gamut, limit)
        cusps[1:TABLE + 1, 2] = gammas
        cusps[0, 2] = cusps[TABLE, 2]
        cusps[TABLE + 1, 2] = cusps[1, 2]
        cusps[TABLE + 2, 2] = cusps[2, 2]
    return tone, gamut, source, limit


def _limit_primaries(preset: Preset, white: str) -> tuple:
    """The preset's limiting primaries, with the D60 white when simulated."""
    return preset.limit if white != WHITES[1] else (*preset.limit[:3], D60)


@functools.lru_cache(maxsize=None)
def _setup(preset_name: str, white: str) -> _Setup:
    """Render constants and tables for a preset and creative white."""
    preset = PRESETS[preset_name]
    prims = _limit_primaries(preset, white)
    single = tuple(tuple(_f(v) for v in pair) for pair in prims)
    tone, gamut, source, limit = _core(_f(preset.peak), single)

    display = _conversion(prims, preset.encoding)
    scale = float(preset.scale)
    if white == WHITES[1] and preset.d60_scale_white:
        scale = scale / float(np.max(display @ np.ones(3)))
    if tuple(prims) == tuple(preset.encoding) and scale == 1.0:
        display = None
    else:
        display = _single(display * scale)
    return _Setup(
        tone=tone, gamut=gamut, input=source, limit=limit,
        to_ap0=_single(_conversion(AP1, AP0)), display=display, eotf=preset.eotf,
    )


@functools.lru_cache(maxsize=None)
def _source_matrix(source: str) -> tuple:
    """Source RGB to AP1, float32."""
    return _single(_conversion(SOURCES[source], AP1, adapt=True))


_DEVICE_TABLES: dict = {}


def _tables(setup: _Setup, device: torch.device) -> tuple:
    """Reach, hue and cusp tables as tensors on ``device``."""
    key = (id(setup), str(device))
    tables = _DEVICE_TABLES.get(key)
    if tables is None:
        g = setup.gamut
        tables = (
            torch.from_numpy(g.reach.copy()).to(device),
            torch.from_numpy(g.hues.copy()).to(device),
            torch.from_numpy(np.ascontiguousarray(g.cusps.T)).to(device),
        )
        _DEVICE_TABLES[key] = tables
    return tables


# Per-pixel rendering in torch.

def _reach_at(h, reach_table):
    """Reach gamut M at ``limit_j_max`` for hues in degrees."""
    base = h.to(torch.int64)
    return _lerp(reach_table[base + 1], reach_table[base + 2], h - base.to(h.dtype))


def _cusp_at(h, hue_table, cusp_table, search: tuple):
    """Cusp J, cusp M and upper hull inverse gamma for hues in degrees."""
    probe = h.to(torch.int64) + 1
    lower = torch.clamp(probe + search[0], min=0)
    upper = torch.clamp(probe + search[1], max=TABLE + 1)
    for _ in range(search[1] - search[0] + 3):
        active = lower + 1 < upper
        above = h > hue_table[probe]
        lower = torch.where(active & above, probe, lower)
        upper = torch.where(active & ~above, probe, upper)
        probe = torch.where(active, (lower + upper) // 2, probe)
    upper = torch.clamp(upper, min=1)
    lower = upper - 1
    hue_lo = hue_table[lower]
    w = (h - hue_lo) / (hue_table[upper] - hue_lo)
    return tuple(_lerp(cusp_table[c][lower], cusp_table[c][upper], w) for c in range(3))


def _gamut_stage(j, m, h, reach_m, gam: _Gamut, hue_table, cusp_table):
    """Tone-mapped J and M brought inside the limiting gamut."""
    cusp_j, cusp_m, gamma_top = _cusp_at(h, hue_table, cusp_table, gam.search)
    j_g, m_g = _compress_gamut(j, m, cusp_j, cusp_m, gamma_top, reach_m, gam, TO)
    compress = (j > 0.0) & (m > 0.0) & (j <= gam.limit_j_max)
    return (torch.where(j > 0.0, torch.where(compress, j_g, j), 0.0),
            torch.where(compress, m_g, 0.0))


def _display_light(r, g, b, setup: _Setup, tables: tuple):
    """ACES2065-1 to linear light in the limiting primaries, 1.0 = 100 nits."""
    reach_table, hue_table, cusp_table = tables
    tone = setup.tone
    A, a, bb = _rgb_to_aab(r, g, b, setup.input, TO)
    j, m, h = _aab_to_jmh(A, a, bb, setup.input, TO)
    reach_m = _reach_at(h, reach_table)
    radians = (PI * h) / 180.0
    cos_h = torch.cos(radians)
    sin_h = torch.sin(radians)
    m_norm = _chroma_norm(cos_h, sin_h, tone.chroma_scale)

    j_ts = _tonescale_a_to_j(A, setup.input, tone, TO)
    m_cp = _chroma_compress(j, m, j_ts, m_norm, reach_m, setup.gamut.limit_j_max, tone, TO)
    j_out, m_out = _gamut_stage(j_ts, m_cp, h, reach_m, setup.gamut, hue_table, cusp_table)

    r, g, b = _jmh_to_rgb_trig(j_out, m_out, cos_h, sin_h, setup.limit, TO)
    ceiling = tone.peak / 100.0
    return (torch.clamp(r, 0.0, ceiling), torch.clamp(g, 0.0, ceiling), torch.clamp(b, 0.0, ceiling))


def _encode(rgb: torch.Tensor, setup: _Setup) -> torch.Tensor:
    """Display light, 1.0 = 100 nits, to display code values."""
    x = rgb
    if setup.display is not None:
        x = torch.stack(_apply(setup.display, *x.unbind(-1)), dim=-1)
    x = torch.clamp(x, min=0.0)
    eotf = setup.eotf
    if eotf == "bt1886":
        return x ** (1.0 / 2.4)
    if eotf == "gamma22":
        return x ** (1.0 / 2.2)
    if eotf == "gamma26":
        return x ** (1.0 / 2.6)
    if eotf == "srgb":
        gamma, offset = 2.4, 0.055
        knee = (offset * gamma / ((gamma - 1.0) * (1.0 + offset))) ** gamma
        slope = ((gamma - 1.0) / offset) ** (gamma - 1.0) * ((1.0 + offset) / gamma) ** gamma
        return torch.where(x >= knee, (1.0 + offset) * x ** (1.0 / gamma) - offset, x * slope)
    if eotf == "pq":
        return _pq(x * 100.0)
    nits = x * 100.0
    y = 0.2627 * nits[..., 0:1] + 0.6780 * nits[..., 1:2] + 0.0593 * nits[..., 2:3]
    gain = torch.where(y == 0.0, 0.0, (y / 1000.0) ** ((1.0 - 1.2) / 1.2))
    scene = gain * (nits / 1000.0)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return torch.where(scene <= 1.0 / 12.0, torch.sqrt(3.0 * scene),
                       a * torch.log(torch.clamp(12.0 * scene - b, min=1e-30)) + c)


def _pq(nits: torch.Tensor) -> torch.Tensor:
    """SMPTE ST 2084 code values for absolute luminance in nits."""
    lm = (nits / 10000.0) ** PQ_M1
    # (c1 + c2 Lm) / (1 + c3 Lm) - 1, with c1 - 1 = -(c2 - c3) = -0.1640625 exactly.
    below_one = (PQ_C2 - PQ_C3) * (lm - 1.0) / (1.0 + PQ_C3 * lm)
    return torch.exp(PQ_M2 * torch.log1p(below_one))


def _render_pixels(pixels: torch.Tensor, matrix: tuple, setup: _Setup) -> torch.Tensor:
    """``(n, 3)`` scene-linear source RGB to ``(n, 3)`` display code values."""
    tables = _tables(setup, pixels.device)
    out = torch.empty(pixels.shape, dtype=torch.float32, device=pixels.device)
    limit = setup.tone.forward_limit
    for start in range(0, int(pixels.shape[0]), PIXELS_PER_PASS):
        chunk = pixels[start:start + PIXELS_PER_PASS].to(torch.float32)
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=1e30, neginf=-1e30)
        r, g, b = chunk.unbind(-1)
        r, g, b = (torch.clamp(c, 0.0, limit) for c in _apply(matrix, r, g, b))
        r, g, b = _apply(setup.to_ap0, r, g, b)
        light = torch.stack(_display_light(r, g, b, setup, tables), dim=-1)
        out[start:start + PIXELS_PER_PASS] = _encode(light, setup).to(torch.float32)
    return out


def render(images: torch.Tensor, source: str, preset: str, white: str = WHITES[0]) -> torch.Tensor:
    """Render scene-linear frames to display code values through the ACES 2.0 Output Transform.

    Args:
        images: ``(batch, height, width, channels)`` scene-linear light. A fourth channel is
            carried through; one channel is read as grey.
        source: A key of :data:`SOURCES`.
        preset: A key of :data:`PRESETS`.
        white: A member of :data:`WHITES`.

    Returns:
        Float32 display code values, the same shape as ``images`` with at least three channels.

    Raises:
        ValueError: An unknown source, preset or white, or two channels.
    """
    if source not in SOURCES:
        raise ValueError(f"Image ACES Output source must be one of {', '.join(SOURCES)}, not {source!r}.")
    if preset not in PRESETS:
        raise ValueError(
            f"Image ACES Output preset {preset!r} is not one of the {len(PRESETS)} ACES 2.0 outputs "
            "in the menu. Pick one there."
        )
    if white not in WHITES:
        raise ValueError(f"Image ACES Output white must be one of {', '.join(WHITES)}, not {white!r}.")
    if images.shape[-1] == 2:
        raise ValueError("Image ACES Output needs RGB or RGBA images, or one grey channel, not two channels.")
    setup = _setup(preset, white)
    matrix = _source_matrix(source)
    if images.shape[-1] == 1:
        images = images.expand(*images.shape[:-1], 3)

    def work(chunk: torch.Tensor) -> torch.Tensor:
        flat = chunk[..., :3].reshape(-1, 3)
        codes = _render_pixels(flat, matrix, setup).reshape(*chunk.shape[:-1], 3)
        if chunk.shape[-1] > 3:
            codes = torch.cat((codes, chunk[..., 3:].to(codes.dtype)), dim=-1)
        return codes

    if images.numel() == 0:
        return images.to(torch.float32)
    return run_frames(images, work, factor=6.0, node="Image ACES Output")
