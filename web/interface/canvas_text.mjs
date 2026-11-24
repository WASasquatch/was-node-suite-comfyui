/**
 * Text drawn on a canvas that has less room than the text wants.
 *
 * `elideText` measures against the context's current font and answers what to draw.
 */

// What a line trimmed to its room ends with, and the fewest characters of it kept by default.
const ELLIPSIS = "...";
const MIN_ELIDED = 12;

/**
 * Trim a line to the room it has rather than squeezing it into it.
 *
 * @param {CanvasRenderingContext2D} ctx - Context the line is measured against, carrying the
 *   font it will be drawn in.
 * @param {string} text - The line.
 * @param {number} room - How much room it has, in element pixels.
 * @param {number} [least] - Characters that must survive for a stub to be worth drawing.
 * @returns {string} The line, as much of it as fits ending in an ellipsis, or nothing at all
 *   where fewer than `least` characters of it would survive.
 */
export function elideText(ctx, text, room, least = MIN_ELIDED) {
  if (!text || !(room > 0)) return "";
  if (ctx.measureText(text).width <= room) return text;

  const ellipsis = ctx.measureText(ELLIPSIS).width;
  if (ellipsis > room) return "";

  let kept = 0;
  let rest = text.length;
  while (kept < rest) {
    const middle = Math.ceil((kept + rest) / 2);
    if (ctx.measureText(text.slice(0, middle)).width + ellipsis <= room) kept = middle;
    else rest = middle - 1;
  }
  if (kept < Math.min(least, text.length)) return "";
  return text.slice(0, kept).trimEnd() + ELLIPSIS;
}
