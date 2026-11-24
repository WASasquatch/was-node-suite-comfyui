/**
 * The arithmetic rules Python follows and JavaScript does not.
 *
 * `roundHalfEven` for Python's `round()`, `roundDecimals` for its `round(x, n)`, `truncate`
 * for its `int()`, and `floorMod` for its `%`. Every function takes and answers a plain number.
 */

// The most decimal places `toFixed` writes, which holds the exact value of any number at or
// above 2 to the -48.
const EXACT_PLACES = 100;

// The size past which `toFixed` writes an exponent rather than digits.
const FIXED_LIMIT = 1e21;

/**
 * Round to decimal places as Python's `round(value, places)` does, ties to the even digit.
 *
 * 0.125 to two places answers 0.12 where `toFixed` answers 0.13.
 *
 * @param {number} value - Value to round.
 * @param {number} places - Decimal places, 0 or more.
 * @returns {number} The number nearest the rounded decimal, its sign kept.
 */
export function roundDecimals(value, places) {
  if (!Number.isFinite(value)) return value;
  const digits = Math.min(EXACT_PLACES, Math.max(0, Math.trunc(places)));
  const magnitude = Math.abs(value);
  if (magnitude >= FIXED_LIMIT) return value;
  let text = magnitude.toFixed(digits);
  const exact = magnitude.toFixed(EXACT_PLACES);
  const point = exact.indexOf(".");
  // `toFixed` takes a tie away from zero; Python keeps the neighbour ending in an even digit.
  if (/^50*$/.test(exact.slice(point + 1 + digits))) {
    const kept = exact.slice(0, point + 1 + digits).replace(/\.$/, "");
    if (Number(kept[kept.length - 1]) % 2 === 0) text = kept;
  }
  const answer = Number(text);
  return value < 0 || Object.is(value, -0) ? -answer : answer;
}

/**
 * Round half to even, the rule Python's `round()` follows.
 *
 * 2.5 answers 2 and 3.5 answers 4, where `Math.round` answers 3 and 4.
 *
 * @param {number} value - Value to round.
 * @returns {number} The value rounded to a whole number.
 */
export function roundHalfEven(value) {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Drop the fraction toward zero, the rule Python's `int()` follows.
 *
 * It differs from a floor below zero: `int(-0.5)` is 0 and `Math.floor(-0.5)` is -1.
 *
 * @param {number} value - Value to truncate.
 * @returns {number} The whole part, with the sign kept.
 */
export function truncate(value) {
  return Math.trunc(value);
}

/**
 * The remainder of a division, with the sign of the divisor, as Python's and numpy's `%` give it.
 *
 * @param {number} value - Value to reduce.
 * @param {number} modulus - Positive modulus.
 * @returns {number} The remainder, 0 up to the modulus.
 */
export function floorMod(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}
