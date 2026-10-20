/**
 * Number sequences through keyframes, worked out again in the browser.
 *
 * One function per python one in `modules/number/sequence.py`, named after it, plus the
 * reading of a values box and the rounding Number Sequence emits. No DOM.
 */

import { EASING_NAMES, ease } from "./easing.js";
import { roundDecimals } from "./python_arithmetic.js";

/** The curve that keeps each value until the next one is reached. */
export const HOLD = "hold";

/** Curve names in menu order. */
export const CURVE_NAMES = Object.freeze([...EASING_NAMES, HOLD]);

/** What a curve shapes. */
export const CURVE_TARGETS = Object.freeze(["transitions", "values"]);

/** How one cycle follows the one before it. */
export const LOOPS = Object.freeze(["restart", "ping_pong", "continue"]);

/** Most entries one sequence holds, and most keyframes it passes through. */
export const MAX_COUNT = 10000;

/** Most cycles one sequence holds. */
export const MAX_REPEAT = 1000;

// One entry of a values box, as Python's `float()` reads one: digits with optional single
// underscores between them, a point, an exponent.
const DIGITS = String.raw`\d(?:_?\d)*`;
const NUMBER_TEXT = new RegExp(
  String.raw`^[+-]?(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:[eE][+-]?${DIGITS})?$`,
);

// What separates entries in a values box.
const ENTRY = /[^\s,]+/g;

/**
 * One curve applied to a progress value.
 *
 * @param {string} curve - A name from `CURVE_NAMES`.
 * @param {number} t - Progress, clamped to 0 to 1.
 * @returns {number} The shaped progress; `hold` answers 0 until `t` reaches 1.
 */
export function shaped(curve, t) {
  if (curve === HOLD) return t >= 1 ? 1 : 0;
  return ease(curve, t);
}

/**
 * The value one cycle holds at a position.
 *
 * @param {number[]} keys - The keyframes, at least one.
 * @param {number} position - From 0 at the first keyframe to 1 at the last.
 * @param {string} curve - A name from `CURVE_NAMES`.
 * @param {string} target - `transitions` or `values`.
 * @param {number} low - The smallest keyframe.
 * @param {number} high - The largest keyframe.
 * @returns {number} The value at that position.
 */
export function cycleValue(keys, position, curve, target, low, high) {
  const last = keys.length - 1;
  if (last === 0) return keys[0];
  const scaled = position * last;
  const index = Math.min(Math.floor(scaled), last - 1);
  const fraction = scaled - index;
  const start = keys[index];
  const span = keys[index + 1] - start;
  if (target === "transitions") return start + span * shaped(curve, fraction);
  const value = start + span * fraction;
  if (high === low) return value;
  return low + (high - low) * shaped(curve, (value - low) / (high - low));
}

/**
 * How many entries a sequence holds.
 *
 * @param {number} keyframes - Keyframes in one cycle.
 * @param {number} count - Entries asked for, or 0 for one per keyframe in every cycle.
 * @param {number} repeat - Cycles.
 * @param {string} loop - A name from `LOOPS`.
 * @returns {number} The number of entries.
 */
export function lengthOf(keyframes, count, repeat, loop) {
  if (count > 0) return count;
  if (loop === "restart") return keyframes * repeat;
  return (keyframes - 1) * repeat + 1;
}

/**
 * Where each entry sits: its cycle and its position within that cycle.
 *
 * @param {number} length - Entries, at least 1.
 * @param {number} repeat - Cycles, at least 1.
 * @param {string} loop - A name from `LOOPS`.
 * @returns {Array<[number, number]>} One `[cycle, position]` per entry.
 */
export function positions(length, repeat, loop) {
  const placed = [];
  if (loop === "restart") {
    const base = Math.floor(length / repeat);
    const extra = length % repeat;
    for (let cycle = 0; cycle < repeat; cycle += 1) {
      const size = base + (cycle < extra ? 1 : 0);
      for (let step = 0; step < size; step += 1) {
        placed.push([cycle, size > 1 ? step / (size - 1) : 0]);
      }
    }
    return placed;
  }
  if (length === 1) return [[0, 0]];
  for (let step = 0; step < length; step += 1) {
    const travelled = (step * repeat) / (length - 1);
    const cycle = Math.min(Math.floor(travelled), repeat - 1);
    placed.push([cycle, travelled - cycle]);
  }
  return placed;
}

/**
 * The sequence, unrounded.
 *
 * @param {number[]} keys - The keyframes, at least one.
 * @param {number} count - Entries, or 0 for one per keyframe in every cycle.
 * @param {string} curve - A name from `CURVE_NAMES`.
 * @param {string} target - A name from `CURVE_TARGETS`.
 * @param {number} repeat - Cycles.
 * @param {string} loop - A name from `LOOPS`.
 * @param {boolean} reverse - Whether the list runs back to front.
 * @returns {number[]} The values in order.
 */
export function build(keys, count, curve, target, repeat, loop, reverse) {
  const length = lengthOf(keys.length, count, repeat, loop);
  const low = Math.min(...keys);
  const high = Math.max(...keys);
  const rise = cycleValue(keys, 1, curve, target, low, high)
    - cycleValue(keys, 0, curve, target, low, high);
  const values = [];
  for (let [cycle, position] of positions(length, repeat, loop)) {
    if (loop === "ping_pong" && cycle % 2 === 1) position = 1 - position;
    let value = cycleValue(keys, position, curve, target, low, high);
    if (loop === "continue") value += cycle * rise;
    values.push(value);
  }
  if (reverse) values.reverse();
  return values;
}

/**
 * The sequence as Number Sequence emits it, each value rounded.
 *
 * @param {number[]} keys - The keyframes.
 * @param {object} settings - `count`, `curve`, `target`, `repeat`, `loop`, `reverse` and
 *   `decimals`, as the node's widgets hold them.
 * @returns {number[]} The values.
 */
export function buildRounded(keys, settings) {
  const built = build(
    keys,
    settings.count,
    settings.curve,
    settings.target,
    settings.repeat,
    settings.loop,
    settings.reverse,
  );
  return built.map((value) => roundDecimals(value, settings.decimals));
}

/**
 * Where a keyframe of the first cycle sits in the whole list.
 *
 * @param {number} keyframe - The keyframe, from 0.
 * @param {number} keyframes - Keyframes in a cycle.
 * @param {number} length - Entries in the list.
 * @param {number} repeat - Cycles.
 * @param {string} loop - A name from `LOOPS`.
 * @param {boolean} reverse - Whether the list runs back to front.
 * @returns {number} A fractional entry index, from 0, for the keyframe's place in the first
 *   cycle.
 */
export function keyframeIndex(keyframe, keyframes, length, repeat, loop, reverse) {
  const position = keyframes > 1 ? keyframe / (keyframes - 1) : 0;
  let index;
  if (loop === "restart") {
    const size = Math.floor(length / repeat) + (length % repeat > 0 ? 1 : 0);
    index = position * Math.max(0, size - 1);
  } else {
    index = length > 1 ? (position * (length - 1)) / repeat : 0;
  }
  return reverse ? length - 1 - index : index;
}

/**
 * The entries of a values box and where each sits in the text.
 *
 * @param {string} text - What the box holds.
 * @returns {Array<{text: string, start: number, end: number, value: number|null}>} One per
 *   entry, `value` null for one Python's `float()` would refuse or that is not finite.
 */
export function readEntries(text) {
  const entries = [];
  for (const match of String(text ?? "").matchAll(ENTRY)) {
    const raw = match[0];
    let value = null;
    if (NUMBER_TEXT.test(raw)) {
      const number = Number(raw.replace(/_/g, ""));
      if (Number.isFinite(number)) value = number;
    }
    entries.push({ text: raw, start: match.index, end: match.index + raw.length, value });
  }
  return entries;
}
