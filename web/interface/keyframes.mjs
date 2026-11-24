/**
 * Keyframes for an amount along a clip, worked out again in the browser.
 *
 * One function per python one in `modules/media/keyframes.py`, named after it, with the same
 * operations in the same order. No DOM.
 */

import { CURVE_NAMES, shaped } from "./number_sequence.mjs";

/** The easing of an entry that names none. */
export const DEFAULT_EASING = "linear";

/** Most keyframes one text holds. */
export const MAX_KEYFRAMES = 1000;

/** Furthest a keyframe may sit from frame 0, either way. */
export const MAX_FRAME = 10000000;

// What separates two entries.
const ENTRY = /[;\n\r]/;

// A whole frame number, signed, in ASCII digits.
const FRAME = /^[+-]?[0-9]+$/;

// A decimal amount in ASCII digits.
const AMOUNT = /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/;

/**
 * One entry read, as `_entry` reads it.
 *
 * @param {string} text - The entry, already trimmed.
 * @returns {{frame: number, amount: number, easing: string}|null} The keyframe, or null where it
 *   does not read.
 */
export function readEntry(text) {
  const parts = String(text).split(":").map((part) => part.trim());
  if (parts.length !== 2 && parts.length !== 3) return null;
  if (!FRAME.test(parts[0]) || !AMOUNT.test(parts[1])) return null;
  const frame = Number(parts[0]) + 0;
  const amount = Number(parts[1]) + 0;
  const easing = parts.length === 3 ? parts[2].toLowerCase() : DEFAULT_EASING;
  if (Math.abs(frame) > MAX_FRAME || !Number.isFinite(amount) || !(amount >= 0 && amount <= 1)) {
    return null;
  }
  if (!CURVE_NAMES.includes(easing)) return null;
  return { frame, amount, easing };
}

/**
 * Every entry of a keyframes text.
 *
 * @param {string} text - Entries separated by `;` or a new line; blank entries and entries
 *   starting with `#` are skipped.
 * @returns {{keys: object[], refused: string[]}} The readable keyframes in written order, and
 *   the text of every entry that does not read.
 */
export function parse(text) {
  const keys = [];
  const refused = [];
  for (const raw of String(text ?? "").split(ENTRY)) {
    const entry = raw.trim();
    if (!entry || entry.startsWith("#")) continue;
    const key = readEntry(entry);
    if (key) keys.push(key);
    else refused.push(entry);
  }
  return { keys, refused };
}

/**
 * Keyframes at the frames they land on in a clip, in frame order.
 *
 * @param {object[]} keys - Keyframes in written order.
 * @param {number} count - Frames in the clip.
 * @returns {Array<[number, number, string]>} `[frame, amount, easing]` sorted by frame; a
 *   negative frame lands at `count` plus it, and the later of two on one frame is kept.
 */
export function placed(keys, count) {
  const held = new Map();
  for (const key of keys) {
    const frame = key.frame < 0 ? key.frame + count : key.frame;
    held.set(frame, [frame, key.amount, key.easing]);
  }
  return [...held.keys()].sort((a, b) => a - b).map((frame) => held.get(frame));
}

/**
 * The amount at one frame.
 *
 * @param {Array<[number, number, string]>} points - What `placed` returned, at least one.
 * @param {number} frame - The frame, whole or between two.
 * @returns {number} The amount, 0 to 1.
 */
export function amountAt(points, frame) {
  if (frame <= points[0][0]) return points[0][1];
  if (frame >= points[points.length - 1][0]) return points[points.length - 1][1];
  let low = 0;
  let high = points.length - 1;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (points[middle][0] <= frame) low = middle;
    else high = middle;
  }
  const [start, startAmount, easing] = points[low];
  if (frame === start) return startAmount;
  const [end, endAmount] = points[high];
  if (startAmount === endAmount) return startAmount;
  const progress = (frame - start) / (end - start);
  const value = startAmount + (endAmount - startAmount) * shaped(easing, progress);
  return Math.min(1, Math.max(0, value));
}

/**
 * The amount at every frame of a clip.
 *
 * @param {object[]} keys - Keyframes in written order, at least one.
 * @param {number} count - Frames in the clip.
 * @returns {number[]} `count` amounts.
 */
export function amounts(keys, count) {
  const points = placed(keys, count);
  const out = [];
  for (let frame = 0; frame < count; frame += 1) out.push(amountAt(points, frame));
  return out;
}
