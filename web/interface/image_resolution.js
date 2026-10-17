/**
 * The width and height an aspect ratio and one measurement come to, worked out again in the
 * browser: one function per python one in `modules/image/resolution.py`.
 */

import { roundHalfEven } from "./python_arithmetic.js";

/** Largest side either edge may reach. */
export const MAX_EDGE = 16384;

/**
 * The same shape the other way round.
 *
 * @param {string} ratio - A ratio written `width:height`.
 * @returns {string} `height:width`, and `ratio` itself where it holds no `:`.
 */
export function mirrored(ratio) {
  const text = String(ratio);
  const at = text.indexOf(":");
  return at === -1 ? text : `${text.slice(at + 1)}:${text.slice(0, at)}`;
}

/**
 * Read one number the way python's `float` reads a menu entry.
 *
 * @param {string} text - The number as written.
 * @returns {number} The value, or NaN where python would refuse it.
 */
function pythonFloat(text) {
  return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(text) ? Number(text) : Number.NaN;
}

/**
 * Read an aspect ratio written as two numbers.
 *
 * @param {string} text - `16:9`, `1.85:1`, `16/9` or a bare number meaning width over one.
 * @returns {number[]} `[width, height]` as positive numbers, `[1, 1]` for anything unreadable.
 */
export function parseRatio(text) {
  const cleaned = String(text ?? "").trim().replaceAll("x", ":").replaceAll("/", ":").replaceAll(",", ".");
  if (!cleaned) return [1, 1];
  const parts = cleaned.split(":").map((part) => part.trim()).filter(Boolean);
  const wide = pythonFloat(parts[0] ?? "");
  const high = parts.length === 1 ? 1 : pythonFloat(parts[1]);
  if (!(wide > 0 && high > 0) || !Number.isFinite(wide) || !Number.isFinite(high)) return [1, 1];
  return [wide, high];
}

/**
 * The nearest whole step of `multiple`, never below one step and never past the cap.
 *
 * @param {number} value - A side in pixels.
 * @param {number} multiple - The step.
 * @returns {number} The snapped side.
 */
function snap(value, multiple) {
  const step = Math.max(1, Math.trunc(Number(multiple) || 0));
  const snapped = roundHalfEven(Number(value) / step) * step;
  return Math.max(step, Math.min(snapped, Math.floor(MAX_EDGE / step) * step));
}

/**
 * The width and height a ratio and one measurement come to.
 *
 * @param {string} ratio - A menu entry, width first.
 * @param {string} driver - `long edge`, `short edge`, `width`, `height` or `megapixels`.
 * @param {number} size - The measurement in pixels, read by every driver but `megapixels`.
 * @param {number} megapixels - The measurement in millions of pixels.
 * @param {number} multiple - Step both sides land on.
 * @returns {number[]} `[width, height]`, both a whole number of steps.
 */
export function resolve(ratio, driver = "long edge", size = 1024, megapixels = 1.0, multiple = 64) {
  const [wide, high] = parseRatio(ratio);
  const shape = wide / high;
  let width;
  let height;
  if (driver === "megapixels") {
    const area = Math.max(Number(megapixels) || 0, 0) * 1_000_000;
    width = area > 0 ? Math.sqrt(area * shape) : 0;
    height = shape > 0 ? width / shape : 0;
  } else {
    const edge = Math.max(Number(size) || 0, 0);
    if (driver === "width" || (driver === "long edge" && shape >= 1) || (driver === "short edge" && shape < 1)) {
      width = edge;
      height = edge / shape;
    } else {
      height = edge;
      width = edge * shape;
    }
  }
  return [snap(width, multiple), snap(height, multiple)];
}
