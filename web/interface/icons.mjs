/**
 * Small glyphs an interface draws in place of a sentence, and the hover text behind them.
 *
 * `drawIcon` paints one glyph into a canvas. `hoverTitles` puts the matching text in an
 * element's `title`, from regions supplied fresh on every repaint.
 */

import { elementPoint } from "./pointer.mjs";
import { surfaceRatio } from "./resolution.mjs";
import { readTheme } from "./theme.mjs";

/** Glyph names `drawIcon` knows. */
export const ICON = {
  EXACT: "exact",
  APPROXIMATE: "approximate",
  WARNING: "warning",
};

/** The hover text each glyph carries, so every interface words it alike. */
export const ICON_TITLES = {
  [ICON.EXACT]: "Preview matches the render",
  [ICON.APPROXIMATE]: "Approximate preview",
  [ICON.WARNING]: "Preview",
};

/**
 * The pack's words for a condition, with the measurement one interface has for it.
 *
 * @param {string} name - A value of `ICON`.
 * @param {string} [detail] - What this interface measured, in its own words.
 * @returns {string} The hover text.
 */
export function iconTitle(name, detail = "") {
  const lead = ICON_TITLES[name] ?? "";
  const rest = typeof detail === "string" ? detail.trim() : "";
  if (!lead) return rest;
  return rest ? `${lead}: ${rest}` : lead;
}

/** Side of a glyph in element pixels, which is the footer's line height. */
export const ICON_SIZE = 11;

/**
 * Draw one glyph.
 *
 * @param {CanvasRenderingContext2D} ctx - Target context.
 * @param {string} name - A value of `ICON`.
 * @param {number} x - Left edge, in element pixels.
 * @param {number} y - Top edge, in element pixels.
 * @param {number} size - Side length. `ICON_SIZE` unless the caller has a reason.
 * @param {string} colour - Stroke and fill colour.
 * @returns {{x: number, y: number, width: number, height: number}} The area it covers, ready
 *   to hand to `hoverTitles`.
 */
export function drawIcon(ctx, name, x, y, size = ICON_SIZE, colour = "#888") {
  // Every glyph below is drawn from paths rather than from a font, so they are the same on
  // every machine and need no asset.
  const box = { x, y, width: size, height: size };
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = Math.max(1, size / 11);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  if (name === ICON.EXACT) {
    // Two bars, the arithmetic sign for equal to.
    const bar = (offset) => {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.12, y + size * offset);
      ctx.lineTo(x + size * 0.88, y + size * offset);
      ctx.stroke();
    };
    bar(0.38);
    bar(0.68);
  } else if (name === ICON.APPROXIMATE) {
    // Two tildes, the arithmetic sign for approximately.
    const wave = (offset) => {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.12, y + size * offset);
      ctx.bezierCurveTo(
        x + size * 0.32, y + size * (offset - 0.16),
        x + size * 0.56, y + size * (offset + 0.16),
        x + size * 0.88, y + size * offset,
      );
      ctx.stroke();
    };
    wave(0.38);
    wave(0.68);
  } else if (name === ICON.WARNING) {
    // A triangle with a bar, which reads as a caution mark at this size.
    ctx.beginPath();
    ctx.moveTo(x + size / 2, y + size * 0.1);
    ctx.lineTo(x + size * 0.94, y + size * 0.86);
    ctx.lineTo(x + size * 0.06, y + size * 0.86);
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + size / 2, y + size * 0.38);
    ctx.lineTo(x + size / 2, y + size * 0.64);
    ctx.stroke();
  }

  ctx.restore();
  return box;
}

/**
 * Draw the glyph a panel claims a fidelity with, into a canvas of its own.
 *
 * @param {HTMLCanvasElement} canvas - A canvas laid out at `ICON_SIZE` on both sides.
 * @param {{icon: string, detail: string}} claim - The glyph and what it says on hover.
 * @returns {void}
 */
export function drawFidelityGlyph(canvas, claim) {
  if (!canvas) return;
  const ratio = surfaceRatio(canvas);
  const side = Math.max(1, Math.round(ICON_SIZE * ratio));
  if (canvas.width !== side || canvas.height !== side) {
    canvas.width = side;
    canvas.height = side;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, side, side);
  ctx.scale(ratio, ratio);
  drawIcon(ctx, claim?.icon, 0, 0, ICON_SIZE, readTheme().fgMuted);
  canvas.title = iconTitle(claim?.icon, claim?.detail);
}

/**
 * Put the text of whichever glyph is under the pointer into an element's ``title``.
 *
 * @param {HTMLElement} element - The element the glyphs are drawn on.
 * @returns {{set: (regions: Array<object>) => void, dispose: () => void}} ``set`` replaces the
 *   regions, which a repaint does every frame; ``dispose`` removes the listeners and clears
 *   the title.
 */
export function hoverTitles(element) {
  let regions = [];
  let current = "";
  // The last pointer position is kept, so a repaint that changes what a region says under a
  // stationary pointer updates the tooltip in place instead of clearing it until the pointer is
  // moved again.
  let point = null;

  const titleAt = (at) => {
    if (!at) return "";
    for (const region of regions) {
      if (
        at.x >= region.x && at.x <= region.x + region.width &&
        at.y >= region.y && at.y <= region.y + region.height
      ) {
        return region.title ?? "";
      }
    }
    return "";
  };

  // The one title a focused element can be named by, where there is exactly one. That gives an
  // interface reached from the keyboard an accessible name rather than an empty one. Where
  // several regions carry different text there is nothing for focus to choose between, so none
  // of it is named.
  const soleTitle = () => {
    const titled = regions.filter((region) => region.title);
    return titled.length === 1 ? titled[0].title : "";
  };

  const show = (next) => {
    if (next === current) return;
    current = next;
    // The browser draws the tooltip from this, so it is styled like every other tooltip and
    // nothing goes on the canvas that would have to be cleared. Assigned rather than removed,
    // which keeps the element from inheriting the tooltip of whatever is behind it.
    element.title = next;
  };

  const settle = () => {
    const under = titleAt(point);
    show(under || (document.activeElement === element ? soleTitle() : ""));
  };

  const onMove = (event) => {
    point = elementPoint(element, event);
    settle();
  };
  const onLeave = () => {
    point = null;
    settle();
  };
  const onFocus = () => settle();
  // Focus is gone, so the sole title goes with it, but a pointer still on a glyph keeps its own.
  const onBlur = () => show(titleAt(point));

  element.addEventListener("pointermove", onMove);
  element.addEventListener("pointerleave", onLeave);
  element.addEventListener("focus", onFocus);
  element.addEventListener("blur", onBlur);

  return {
    set(next) {
      // Replaced rather than merged, and called from every repaint, which drops the regions
      // a resize left behind.
      regions = Array.isArray(next) ? next : [];
      settle();
    },
    dispose() {
      element.removeEventListener("pointermove", onMove);
      element.removeEventListener("pointerleave", onLeave);
      element.removeEventListener("focus", onFocus);
      element.removeEventListener("blur", onBlur);
      element.title = "";
      regions = [];
      point = null;
      current = "";
    },
  };
}
