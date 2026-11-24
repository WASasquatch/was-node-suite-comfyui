/**
 * Easing curves mapping 0 to 1 onto 0 to 1, worked out again in the browser.
 *
 * One function per curve in `modules/util/easing.py`, with the same operations in the same
 * order, sines and powers included. No imports and no DOM.
 */

// Overshoot constant for the `back` family.
const BACK = 1.70158;

// The same constant scaled for `ease_in_out_back`.
const BACK_INOUT = BACK * 1.525;

// Period of the `elastic` oscillation, as a fraction of the curve.
const ELASTIC_PERIOD = 0.3;

// Angular frequency the elastic curves oscillate at.
const ELASTIC_OMEGA = (2 * Math.PI) / ELASTIC_PERIOD;

// Angular frequency of `ease_in_out_elastic`.
const ELASTIC_INOUT_OMEGA = (2 * Math.PI) / 4.5;

// Two over pi, and pi over two in two parts, the first holding 33 bits.
const TWO_OVER_PI = 6.36619772367581382433e-1;
const PIO2_HIGH = 1.57079632673412561417e0;
const PIO2_LOW = 6.07710050650619224932e-11;

// The natural logarithm of 2.
const LN2 = 0.6931471805599453;

// Taylor coefficients of sine past `x`, cosine past 1 and the exponential past `1 + y`.
const SIN = [
  -1.0 / 6.0,
  1.0 / 120.0,
  -1.0 / 5040.0,
  1.0 / 362880.0,
  -1.0 / 39916800.0,
  1.0 / 6227020800.0,
  -1.0 / 1307674368000.0,
  1.0 / 355687428096000.0,
];
const COS = [
  -1.0 / 2.0,
  1.0 / 24.0,
  -1.0 / 720.0,
  1.0 / 40320.0,
  -1.0 / 3628800.0,
  1.0 / 479001600.0,
  -1.0 / 87178291200.0,
  1.0 / 20922789888000.0,
  -1.0 / 6402373705728000.0,
];
const EXP = [
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
];

/**
 * A polynomial by Horner's rule.
 *
 * @param {number[]} coefficients - Constant term first.
 * @param {number} x - Where it is read.
 * @returns {number} Its value.
 */
function horner(coefficients, x) {
  let total = coefficients[coefficients.length - 1];
  for (let index = coefficients.length - 2; index >= 0; index -= 1) {
    total = coefficients[index] + x * total;
  }
  return total;
}

/**
 * A value as a quarter turn count and a remainder within pi over four.
 *
 * @param {number} x - The angle.
 * @returns {[number, number]} The count from 0 to 3, and the remainder.
 */
function quadrant(x) {
  const turns = Math.floor(x * TWO_OVER_PI + 0.5);
  const remainder = (x - turns * PIO2_HIGH) - turns * PIO2_LOW;
  return [((turns % 4) + 4) % 4, remainder];
}

/**
 * Sine of a value within pi over four.
 *
 * @param {number} r - The value.
 * @returns {number} Its sine.
 */
function sinNear(r) {
  const r2 = r * r;
  return r + r * r2 * horner(SIN, r2);
}

/**
 * Cosine of a value within pi over four.
 *
 * @param {number} r - The value.
 * @returns {number} Its cosine.
 */
function cosNear(r) {
  const r2 = r * r;
  return 1.0 + r2 * horner(COS, r2);
}

/**
 * Sine.
 *
 * @param {number} x - The angle.
 * @returns {number} Its sine.
 */
function sin(x) {
  const [turn, r] = quadrant(x);
  if (turn === 0) return sinNear(r);
  if (turn === 1) return cosNear(r);
  if (turn === 2) return -sinNear(r);
  return -cosNear(r);
}

/**
 * Cosine.
 *
 * @param {number} x - The angle.
 * @returns {number} Its cosine.
 */
function cos(x) {
  const [turn, r] = quadrant(x);
  if (turn === 0) return cosNear(r);
  if (turn === 1) return -sinNear(r);
  if (turn === 2) return -cosNear(r);
  return sinNear(r);
}

/**
 * Two to a power.
 *
 * @param {number} x - The power.
 * @returns {number} Two to the power `x`.
 */
function exp2(x) {
  const whole = Math.floor(x);
  const y = (x - whole) * LN2;
  let scale = 1.0;
  const step = whole > 0 ? 2.0 : 0.5;
  for (let count = Math.abs(whole); count > 0; count -= 1) scale *= step;
  return (1.0 + y + y * y * horner(EXP, y)) * scale;
}

/**
 * The bounce curve the other two bounce curves are built from.
 *
 * @param {number} t - Progress in 0 to 1.
 * @returns {number} The eased value.
 */
function outBounce(t) {
  const n1 = 7.5625;
  const d1 = 2.75;
  if (t < 1 / d1) return n1 * t * t;
  if (t < 2 / d1) {
    t -= 1.5 / d1;
    return n1 * t * t + 0.75;
  }
  if (t < 2.5 / d1) {
    t -= 2.25 / d1;
    return n1 * t * t + 0.9375;
  }
  t -= 2.625 / d1;
  return n1 * t * t + 0.984375;
}

/** Curve name to the function computing it, in the order a menu offers them. */
export const EASINGS = Object.freeze({
  linear: (t) => t,
  ease_in_sine: (t) => 1 - cos((t * Math.PI) / 2),
  ease_out_sine: (t) => sin((t * Math.PI) / 2),
  ease_in_out_sine: (t) => -(cos(Math.PI * t) - 1) / 2,
  ease_in_quad: (t) => t * t,
  ease_out_quad: (t) => {
    const u = 1 - t;
    return 1 - u * u;
  },
  ease_in_out_quad: (t) => {
    if (t < 0.5) return 2 * t * t;
    const u = -2 * t + 2;
    return 1 - (u * u) / 2;
  },
  ease_in_cubic: (t) => t * t * t,
  ease_out_cubic: (t) => {
    const u = 1 - t;
    return 1 - u * u * u;
  },
  ease_in_out_cubic: (t) => {
    if (t < 0.5) return 4 * t * t * t;
    const u = -2 * t + 2;
    return 1 - (u * u * u) / 2;
  },
  ease_in_quart: (t) => t * t * t * t,
  ease_out_quart: (t) => {
    const u = 1 - t;
    return 1 - u * u * u * u;
  },
  ease_in_out_quart: (t) => {
    if (t < 0.5) return 8 * t * t * t * t;
    const u = -2 * t + 2;
    return 1 - (u * u * u * u) / 2;
  },
  ease_in_quint: (t) => t * t * t * t * t,
  ease_out_quint: (t) => {
    const u = 1 - t;
    return 1 - u * u * u * u * u;
  },
  ease_in_out_quint: (t) => {
    if (t < 0.5) return 16 * t * t * t * t * t;
    const u = -2 * t + 2;
    return 1 - (u * u * u * u * u) / 2;
  },
  ease_in_expo: (t) => (t === 0 ? 0 : exp2(10 * t - 10)),
  ease_out_expo: (t) => (t === 1 ? 1 : 1 - exp2(-10 * t)),
  ease_in_out_expo: (t) => {
    if (t === 0 || t === 1) return t;
    return t < 0.5 ? exp2(20 * t - 10) / 2 : (2 - exp2(-20 * t + 10)) / 2;
  },
  ease_in_circ: (t) => 1 - Math.sqrt(Math.max(0, 1 - t * t)),
  ease_out_circ: (t) => {
    const u = t - 1;
    return Math.sqrt(Math.max(0, 1 - u * u));
  },
  ease_in_out_circ: (t) => {
    if (t < 0.5) {
      const u = 2 * t;
      return (1 - Math.sqrt(Math.max(0, 1 - u * u))) / 2;
    }
    const u = -2 * t + 2;
    return (Math.sqrt(Math.max(0, 1 - u * u)) + 1) / 2;
  },
  ease_in_back: (t) => (BACK + 1) * t * t * t - BACK * t * t,
  ease_out_back: (t) => {
    const u = t - 1;
    return 1 + (BACK + 1) * u * u * u + BACK * u * u;
  },
  ease_in_out_back: (t) => {
    if (t < 0.5) {
      const u = 2 * t;
      return (u * u * ((BACK_INOUT + 1) * 2 * t - BACK_INOUT)) / 2;
    }
    const u = 2 * t - 2;
    return (u * u * ((BACK_INOUT + 1) * (t * 2 - 2) + BACK_INOUT) + 2) / 2;
  },
  ease_in_elastic: (t) => {
    if (t === 0 || t === 1) return t;
    return -exp2(10 * t - 10) * sin((t * 10 - 10.75) * ELASTIC_OMEGA);
  },
  ease_out_elastic: (t) => {
    if (t === 0 || t === 1) return t;
    return exp2(-10 * t) * sin((t * 10 - 0.75) * ELASTIC_OMEGA) + 1;
  },
  ease_in_out_elastic: (t) => {
    if (t === 0 || t === 1) return t;
    if (t < 0.5) return -(exp2(20 * t - 10) * sin((20 * t - 11.125) * ELASTIC_INOUT_OMEGA)) / 2;
    return (exp2(-20 * t + 10) * sin((20 * t - 11.125) * ELASTIC_INOUT_OMEGA)) / 2 + 1;
  },
  ease_in_bounce: (t) => 1 - outBounce(1 - t),
  ease_out_bounce: outBounce,
  ease_in_out_bounce: (t) => (t < 0.5 ? (1 - outBounce(1 - 2 * t)) / 2 : (1 + outBounce(2 * t - 1)) / 2),
});

/** The curve names in menu order. */
export const EASING_NAMES = Object.freeze(Object.keys(EASINGS));

/**
 * Apply one curve to a progress value.
 *
 * @param {string} name - A key of `EASINGS`.
 * @param {number} t - Progress, clamped to 0 to 1 before the curve is applied.
 * @returns {number} The eased value.
 * @throws {Error} When the curve is not one of `EASINGS`.
 */
export function ease(name, t) {
  const curve = EASINGS[name];
  if (!curve) throw new Error(`unknown curve ${name}`);
  return curve(Math.min(1, Math.max(0, Number(t))));
}
