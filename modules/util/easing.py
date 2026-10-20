"""Easing curves mapping ``[0, 1]`` onto ``[0, 1]``, with ``f(0) == 0`` and ``f(1) == 1``.

The ``back`` and ``elastic`` families overshoot in between. Sines, cosines and powers are
computed from addition, multiplication and division alone.
"""

from __future__ import annotations

import math

__all__ = ["EASINGS", "EASING_NAMES", "ease", "ease_series"]

#: Overshoot constant for the ``back`` family, from the original Penner equations.
_BACK = 1.70158

#: The same constant scaled for the symmetric ``ease_in_out_back``.
_BACK_INOUT = _BACK * 1.525

#: Period of the ``elastic`` oscillation, as a fraction of the curve.
_ELASTIC_PERIOD = 0.3

#: Angular frequency the elastic curves oscillate at, derived from :data:`_ELASTIC_PERIOD`.
_ELASTIC_OMEGA = (2 * math.pi) / _ELASTIC_PERIOD

#: Angular frequency of ``ease_in_out_elastic``.
_ELASTIC_INOUT_OMEGA = (2 * math.pi) / 4.5

#: Two over pi, and pi over two in two parts: the first holds 33 bits, so a multiple of it
#: by a quadrant count below 2 to the 20 is exact.
_TWO_OVER_PI = 6.36619772367581382433e-01
_PIO2_HIGH = 1.57079632673412561417e00
_PIO2_LOW = 6.07710050650619224932e-11

#: The natural logarithm of 2.
_LN2 = 0.6931471805599453

#: Taylor coefficients of sine past ``x``, cosine past 1 and the exponential past ``1 + y``,
#: each the reciprocal of a factorial below 2 to the 53.
_SIN = (
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5040.0,
    1.0 / 362880.0,
    -1.0 / 39916800.0,
    1.0 / 6227020800.0,
    -1.0 / 1307674368000.0,
    1.0 / 355687428096000.0,
)
_COS = (
    -1.0 / 2.0,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40320.0,
    -1.0 / 3628800.0,
    1.0 / 479001600.0,
    -1.0 / 87178291200.0,
    1.0 / 20922789888000.0,
    -1.0 / 6402373705728000.0,
)
_EXP = (
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
    1.0 / 87178291200.0,
    1.0 / 1307674368000.0,
    1.0 / 20922789888000.0,
    1.0 / 355687428096000.0,
)


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    """A polynomial in ``x`` with the given coefficients, constant first, by Horner's rule."""
    total = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        total = coefficient + x * total
    return total


def _quadrant(x: float) -> tuple[int, float]:
    """``x`` as a quarter turn count, from 0 to 3, and a remainder within pi over four."""
    turns = math.floor(x * _TWO_OVER_PI + 0.5)
    remainder = (x - turns * _PIO2_HIGH) - turns * _PIO2_LOW
    return int(turns) % 4, remainder


def _sin_near(r: float) -> float:
    """Sine of a value within pi over four."""
    r2 = r * r
    return r + r * r2 * _horner(_SIN, r2)


def _cos_near(r: float) -> float:
    """Cosine of a value within pi over four."""
    r2 = r * r
    return 1.0 + r2 * _horner(_COS, r2)


def _sin(x: float) -> float:
    """Sine of ``x``."""
    quadrant, r = _quadrant(x)
    if quadrant == 0:
        return _sin_near(r)
    if quadrant == 1:
        return _cos_near(r)
    if quadrant == 2:
        return -_sin_near(r)
    return -_cos_near(r)


def _cos(x: float) -> float:
    """Cosine of ``x``."""
    quadrant, r = _quadrant(x)
    if quadrant == 0:
        return _cos_near(r)
    if quadrant == 1:
        return -_sin_near(r)
    if quadrant == 2:
        return -_cos_near(r)
    return _sin_near(r)


def _exp2(x: float) -> float:
    """Two to the power ``x``."""
    whole = math.floor(x)
    y = (x - whole) * _LN2
    scale = 1.0
    step = 2.0 if whole > 0 else 0.5
    for _ in range(abs(int(whole))):
        scale *= step
    return (1.0 + y + y * y * _horner(_EXP, y)) * scale


def _linear(t: float) -> float:
    return t


def _in_sine(t: float) -> float:
    return 1 - _cos((t * math.pi) / 2)


def _out_sine(t: float) -> float:
    return _sin((t * math.pi) / 2)


def _in_out_sine(t: float) -> float:
    return -(_cos(math.pi * t) - 1) / 2


def _in_quad(t: float) -> float:
    return t * t


def _out_quad(t: float) -> float:
    u = 1 - t
    return 1 - u * u


def _in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    u = -2 * t + 2
    return 1 - (u * u) / 2


def _in_cubic(t: float) -> float:
    return t * t * t


def _out_cubic(t: float) -> float:
    u = 1 - t
    return 1 - u * u * u


def _in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    u = -2 * t + 2
    return 1 - (u * u * u) / 2


def _in_quart(t: float) -> float:
    return t * t * t * t


def _out_quart(t: float) -> float:
    u = 1 - t
    return 1 - u * u * u * u


def _in_out_quart(t: float) -> float:
    if t < 0.5:
        return 8 * t * t * t * t
    u = -2 * t + 2
    return 1 - (u * u * u * u) / 2


def _in_quint(t: float) -> float:
    return t * t * t * t * t


def _out_quint(t: float) -> float:
    u = 1 - t
    return 1 - u * u * u * u * u


def _in_out_quint(t: float) -> float:
    if t < 0.5:
        return 16 * t * t * t * t * t
    u = -2 * t + 2
    return 1 - (u * u * u * u * u) / 2


def _in_expo(t: float) -> float:
    return 0.0 if t == 0 else _exp2(10 * t - 10)


def _out_expo(t: float) -> float:
    return 1.0 if t == 1 else 1 - _exp2(-10 * t)


def _in_out_expo(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return _exp2(20 * t - 10) / 2 if t < 0.5 else (2 - _exp2(-20 * t + 10)) / 2


def _in_circ(t: float) -> float:
    return 1 - math.sqrt(max(0.0, 1 - t * t))


def _out_circ(t: float) -> float:
    u = t - 1
    return math.sqrt(max(0.0, 1 - u * u))


def _in_out_circ(t: float) -> float:
    if t < 0.5:
        u = 2 * t
        return (1 - math.sqrt(max(0.0, 1 - u * u))) / 2
    u = -2 * t + 2
    return (math.sqrt(max(0.0, 1 - u * u)) + 1) / 2


def _in_back(t: float) -> float:
    return (_BACK + 1) * t * t * t - _BACK * t * t


def _out_back(t: float) -> float:
    u = t - 1
    return 1 + (_BACK + 1) * u * u * u + _BACK * u * u


def _in_out_back(t: float) -> float:
    if t < 0.5:
        u = 2 * t
        return (u * u * ((_BACK_INOUT + 1) * 2 * t - _BACK_INOUT)) / 2
    u = 2 * t - 2
    return (u * u * ((_BACK_INOUT + 1) * (t * 2 - 2) + _BACK_INOUT) + 2) / 2


def _in_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return -_exp2(10 * t - 10) * _sin((t * 10 - 10.75) * _ELASTIC_OMEGA)


def _out_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    return _exp2(-10 * t) * _sin((t * 10 - 0.75) * _ELASTIC_OMEGA) + 1


def _in_out_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return float(t)
    if t < 0.5:
        return -(_exp2(20 * t - 10) * _sin((20 * t - 11.125) * _ELASTIC_INOUT_OMEGA)) / 2
    return (_exp2(-20 * t + 10) * _sin((20 * t - 11.125) * _ELASTIC_INOUT_OMEGA)) / 2 + 1


def _out_bounce(t: float) -> float:
    """The bounce curve, and the definition the other two bounce curves are built from.

    Args:
        t: Progress in ``[0, 1]``.

    Returns:
        The eased value, in ``[0, 1]``.
    """
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def _in_bounce(t: float) -> float:
    return 1 - _out_bounce(1 - t)


def _in_out_bounce(t: float) -> float:
    if t < 0.5:
        return (1 - _out_bounce(1 - 2 * t)) / 2
    return (1 + _out_bounce(2 * t - 1)) / 2


#: Curve name -> the function computing it. Ordered as it is presented to the user: the
#: identity first, then the four polynomial families from gentlest to steepest, then the
#: three that do something other than accelerate.
EASINGS = {
    "linear": _linear,
    "ease_in_sine": _in_sine,
    "ease_out_sine": _out_sine,
    "ease_in_out_sine": _in_out_sine,
    "ease_in_quad": _in_quad,
    "ease_out_quad": _out_quad,
    "ease_in_out_quad": _in_out_quad,
    "ease_in_cubic": _in_cubic,
    "ease_out_cubic": _out_cubic,
    "ease_in_out_cubic": _in_out_cubic,
    "ease_in_quart": _in_quart,
    "ease_out_quart": _out_quart,
    "ease_in_out_quart": _in_out_quart,
    "ease_in_quint": _in_quint,
    "ease_out_quint": _out_quint,
    "ease_in_out_quint": _in_out_quint,
    "ease_in_expo": _in_expo,
    "ease_out_expo": _out_expo,
    "ease_in_out_expo": _in_out_expo,
    "ease_in_circ": _in_circ,
    "ease_out_circ": _out_circ,
    "ease_in_out_circ": _in_out_circ,
    "ease_in_back": _in_back,
    "ease_out_back": _out_back,
    "ease_in_out_back": _in_out_back,
    "ease_in_elastic": _in_elastic,
    "ease_out_elastic": _out_elastic,
    "ease_in_out_elastic": _in_out_elastic,
    "ease_in_bounce": _in_bounce,
    "ease_out_bounce": _out_bounce,
    "ease_in_out_bounce": _in_out_bounce,
}

#: The curve names in presentation order, for a combo's ``options``.
EASING_NAMES = tuple(EASINGS)


def ease(name: str, t: float) -> float:
    """Apply one curve to a progress value.

    Args:
        name: A key of :data:`EASINGS`.
        t: Progress. Values outside ``[0, 1]`` are clamped before the curve is applied.

    Returns:
        The eased value. Within ``[0, 1]`` for every curve except the ``back`` and
        ``elastic`` families, which overshoot both ends by design.

    Raises:
        KeyError: ``name`` is not a known curve.
    """
    curve = EASINGS[name]
    return float(curve(min(1.0, max(0.0, float(t)))))


def ease_series(name: str, count: int, endpoint: bool = True) -> list[float]:
    """The curve sampled at evenly spaced points.

    Args:
        name: A key of :data:`EASINGS`.
        count: How many samples to take. A ``count`` of 1 yields ``[0.0]``, since one
            sample cannot describe a span.
        endpoint: Whether the last sample sits at ``t = 1``. ``False`` stops one step
            short, which is what a seamless loop needs: the final frame of the loop is
            the first frame of the next pass, so emitting both repeats it.

    Returns:
        ``count`` eased values.

    Raises:
        KeyError: ``name`` is not a known curve.
        ValueError: ``count`` is less than 1.
    """
    if count < 1:
        raise ValueError(f"count must be 1 or more, not {count}")
    if count == 1:
        return [0.0]
    divisor = (count - 1) if endpoint else count
    return [ease(name, step / divisor) for step in range(count)]
