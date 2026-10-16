/**
 * The shared surface a filter preview draws on, and the test card it draws over.
 *
 * `createFilterSurface` builds one element that shows a picture twice, before and after, and
 * hands the filter an `ImageData` to read and one to write.
 */

import { app } from "../../../scripts/app.js";
import { imageBackdrop, normaliseFrame } from "./backdrop.js";
import { elideText } from "./canvas_text.js";
import { ICON, ICON_SIZE, ICON_TITLES, drawIcon, hoverTitles, iconTitle } from "./icons.js";
import { createMetricsSection } from "./metrics_panel.js";
import { captureWheel, elementPoint } from "./pointer.js";
import { LABELS, PREVIEW_STATE } from "./preview.js";
import { floorMod } from "./python_arithmetic.js";
import { contentRatio, surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme } from "./theme.js";
import { appendInterfaceWidget } from "./widget.js";

//: The two greys a transparent area is drawn against, and the side of one square.
const CHECKER_LIGHT = "#999999";
const CHECKER_DARK = "#666666";
const CHECKER_SIDE = 8;

/**
 * Fill a rectangle with the checkerboard a transparent area is read against.
 *
 * @param {CanvasRenderingContext2D} ctx - The context to fill.
 * @param {number} x - Left edge, in the context's own units.
 * @param {number} y - Top edge.
 * @param {number} w - Width.
 * @param {number} h - Height.
 * @returns {void}
 */
function paintChecker(ctx, x, y, w, h) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.fillStyle = CHECKER_LIGHT;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = CHECKER_DARK;
  const side = CHECKER_SIDE;
  for (let row = 0; row * side < h; row += 1) {
    for (let col = row % 2; col * side < w; col += 2) {
      ctx.fillRect(x + col * side, y + row * side, side, side);
    }
  }
  ctx.restore();
}

const LOG_NAME = "WASNodeSuite.FilterSurface";

/**
 * Which backdrop the picture came from.
 *
 * A pixel-scale setting is literal on `CARD` and reduced on `IMAGE`.
 */
export const BACKDROP_KIND = {
  IMAGE: "image",
  CARD: "card",
};

// The card's width in card pixels. 288 is 16 columns of clipped black, one column for each of
// the 256 grey levels, and 16 columns of clipped white, and it divides exactly by 6, 8, 12, 16,
// 24, 36 and 48, so every swatch, block, cell and chequer boundary in the band table below
// lands on a whole pixel. The height is the sum of that table.
const CARD_WIDTH = 288;

// Columns held at pure black and pure white at the two ends of the grey ramp.
const RAMP_CLIP = 16;

// Eight skin tones, lightest to deepest, and the order they are laid down in. The order
// alternates between the two ends of the range, leaving any crop two swatches wide carrying
// both a light tone and a deep one.
const SKIN_TONES = [
  [255, 224, 196],
  [241, 194, 165],
  [224, 172, 138],
  [198, 134, 105],
  [172, 112, 82],
  [141, 85, 60],
  [107, 63, 43],
  [72, 42, 28],
];
const SKIN_ORDER = [0, 7, 1, 6, 2, 5, 3, 4];

// Red, green, blue, cyan, magenta and yellow, as the channels each one lights.
const HUES = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 0],
];

// Stroke widths of the four concentric square cells, in card pixels.
const EDGE_STROKES = [1, 2, 4, 8];

// Pitch of the three chequers, and of the value noise lattice, in card pixels.
const CHEQUER_PITCHES = [1, 2, 4];
const NOISE_PITCH = 8;

// Seed of the value noise lattice. Fixed, so two nodes show the same card and a twin written
// in another language can reproduce it.
const NOISE_SEED = 0x5741534e;

// Height of the appended widget in node units, and the margin a DOM widget element is inset by
// on every side, which makes the element itself shorter than the widget by twice it.
const DEFAULT_HEIGHT = 240;
const UI_MARGIN = 10;

const DEFAULT_WIDGET_NAME = "was_filter_ui";
const DEFAULT_WIDGET_TYPE = "was_filter_surface";

// Layout bands, measured in element pixels.
const PAD = 4;
const HEADER_HEIGHT = 12;
const FOOTER_HEIGHT = 13;
const FOOTER_LINES = 1;
const PANEL_GAP = 6;
const MIN_PANEL = 24;

const BODY_FONT = "10px sans-serif";
const TAG_FONT = "9px sans-serif";
const LABEL_FONT = "11px sans-serif";

// The largest margin the buffers are grown by, in preview pixels on each side. Three times the
// radius is what a canvas blur needs, so this covers every radius a preview can afford to draw.
const MAX_MARGIN = 192;

// How long to wait before asking for a picture again while the answer is not the picture. A
// node queued before the socket opened publishes on its next run, so an answer of `waiting` is
// never the last word.
const RETRY_INTERVAL = 3000;

// The gap kept between the two halves of a footer line.
const FOOTER_GAP = 8;

// The gap kept between a glyph and whatever follows it on the same line.
const GLYPH_GAP = 4;

/**
 * How far the factor has to stand off one before the picture counts as reduced or as magnified.
 */
export const REDUCED_AT = 1.02;
export const MAGNIFIED_AT = 0.98;

// Pixels the filter may walk on one repaint while nobody is holding the surface at full detail.
const FILTER_PIXEL_BUDGET = 320000;

// Room the measurement band adds below the two panels, in node units.
const METRICS_HEIGHT = 104;

// The longest edge the band measures at, and how long a slider rests before it measures.
const METRICS_EDGE = 512;
const METRICS_DELAY = 150;

// The footer button that lifts the picture to the image's own detail while it is held, and the
// room it takes. Its words change while it is down.
const HOLD_LABEL = "hold for full detail";
const HOLD_HELD = "full detail";
const HOLD_HEIGHT = 13;
const HOLD_PADDING = 5;
const HOLD_GAP = 6;

// What the two panels are called when the adopter names neither.
const DEFAULT_LABELS = { before: "before", after: "after" };

// What the glyph says when the adopter states no fidelity. A preview that approximates and does
// not say so is the one failure this surface cannot detect on the adopter's behalf, so the
// absence itself is reported, in the warning glyph rather than in silence.
const NO_FIDELITY = "fidelity not stated";

// What a panel says when the buffer behind it could not be made, which is the one way a panel is
// left with nothing in it: a buffer or a canvas context the browser would not give.
const NO_BUFFER = "the picture could not be built";

/**
 * Clamp a number into a range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} low - Lower bound.
 * @param {number} high - Upper bound.
 * @returns {number} The value, held inside the bounds.
 */
function clamp(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

/**
 * Read a number, answering a fallback for anything that is not one.
 *
 * @param {*} value - Value to read.
 * @param {number} fallback - What to answer when the value is not a finite number.
 * @returns {number} The number, or the fallback.
 */
function toNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/**
 * Hold a value to a whole channel.
 *
 * @param {number} value - Channel value.
 * @returns {number} A whole number, 0 to 255.
 */
function level(value) {
  return clamp(roundHalfUp(value), 0, 255);
}

/**
 * Round half up.
 *
 * The one rounding rule the card is drawn with.
 *
 * @param {number} value - Value to round.
 * @returns {number} The value rounded to a whole number.
 */
function roundHalfUp(value) {
  return Math.floor(value + 0.5);
}

/**
 * One lattice value of the noise field.
 *
 * Integer arithmetic throughout.
 *
 * @param {number} ix - Lattice column.
 * @param {number} iy - Lattice row.
 * @returns {number} A value from 0 to 1.
 */
function lattice(ix, iy) {
  let hash = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + NOISE_SEED) | 0;
  hash = Math.imul(hash ^ (hash >>> 13), 1274126177);
  return ((hash ^ (hash >>> 16)) >>> 0) / 4294967295;
}

/**
 * Value noise at one point of the card.
 *
 * @param {number} x - Position across the card.
 * @param {number} y - Position down the card.
 * @returns {number} A value from 0 to 1, smooth across each lattice cell.
 */
function noiseAt(x, y) {
  const gx = Math.floor(x / NOISE_PITCH);
  const gy = Math.floor(y / NOISE_PITCH);
  const fx = (x - gx * NOISE_PITCH) / NOISE_PITCH;
  const fy = (y - gy * NOISE_PITCH) / NOISE_PITCH;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);

  const topLeft = lattice(gx, gy);
  const topRight = lattice(gx + 1, gy);
  const bottomLeft = lattice(gx, gy + 1);
  const bottomRight = lattice(gx + 1, gy + 1);
  const top = topLeft + (topRight - topLeft) * sx;
  const bottom = bottomLeft + (bottomRight - bottomLeft) * sx;
  return top + (bottom - top) * sy;
}

/**
 * The nine bands of the card, in the order they are laid down.
 *
 * Every painter answers one opaque colour.
 */
const BANDS = [
  {
    name: "ramp",
    height: 20,
    purpose:
      "Two rows carrying all 256 grey levels, forward on top and reversed underneath, with a" +
      " run of pure black and a run of pure white at each end. Any tonal curve reads here:" +
      " brightness, contrast, a nova LUT, a gradient map lookup, high pass clipping. The two" +
      " rows are complementary, so a crop of any width carries both ends of the scale, and a" +
      " clipped run that grows is a filter losing tones it cannot get back.",
    paint(x, y, height) {
      const forward = y < height / 2;
      const value = forward ? x - RAMP_CLIP : CARD_WIDTH - 1 - RAMP_CLIP - x;
      const grey = level(value);
      return [grey, grey, grey];
    },
  },
  {
    name: "skin",
    height: 24,
    purpose:
      "Eight skin tones from fair to deep, each 36 wide, each shaded by five levels from top" +
      " to bottom so a saturation or contrast change reads as a shift rather than a step." +
      " Skin is the subject a wrong grade shows on first, and the style, dragan, saturation" +
      " and median filters are all judged on faces, so a setting chosen here is a setting" +
      " that transfers to a photograph.",
    paint(x, y, height) {
      const tone = SKIN_TONES[SKIN_ORDER[Math.floor(x / (CARD_WIDTH / SKIN_ORDER.length))]];
      const shade = 5 - roundHalfUp((10 * y) / Math.max(1, height - 1));
      return [level(tone[0] + shade), level(tone[1] + shade), level(tone[2] + shade)];
    },
  },
  {
    name: "primaries",
    height: 20,
    purpose:
      "Red, green, blue, cyan, magenta and yellow, each at full value and immediately after it" +
      " at half value, in twelve blocks of 24. Saturation, hue rotation, a gradient map's" +
      " endpoints and a chromatic fringe all read off fully saturated colour and nowhere else." +
      " The half value blocks catch a filter that clips only at the top, and pairing each one" +
      " with its own hue puts both values inside any crop two blocks wide.",
    paint(x) {
      const block = Math.floor(x / (CARD_WIDTH / (HUES.length * 2)));
      const hue = HUES[Math.floor(block / 2)];
      const value = block % 2 === 1 ? 128 : 255;
      return [hue[0] * value, hue[1] * value, hue[2] * value];
    },
  },
  {
    name: "edges",
    height: 32,
    purpose:
      "Black on white concentric squares with strokes of 1, 2, 4 and 8 pixels, a 45 degree" +
      " wedge for the diagonal case, and a quadrant giving one vertical and one horizontal" +
      " hard edge. Sharpening, edge detection, a canny threshold, a high pass radius and a" +
      " chromatic offset are all judged against a known stroke width, and four of them side by" +
      " side let a radius be read straight off the picture.",
    paint(x, y, height) {
      const cells = EDGE_STROKES.length * height;
      if (x < cells) {
        const stroke = EDGE_STROKES[Math.floor(x / height)];
        const lx = x % height;
        const distance = Math.min(lx, y, height - 1 - lx, height - 1 - y);
        const grey = Math.floor(distance / stroke) % 2 === 0 ? 0 : 255;
        return [grey, grey, grey];
      }
      const rest = (CARD_WIDTH - cells) / 2;
      if (x < cells + rest) {
        const grey = x - cells - y >= 24 ? 0 : 255;
        return [grey, grey, grey];
      }
      const lx = x - cells - rest;
      const grey = (lx >= rest / 2) !== (y >= height / 2) ? 0 : 255;
      return [grey, grey, grey];
    },
  },
  {
    name: "texture",
    height: 24,
    purpose:
      "Chequers at a 1, 2 and 4 pixel pitch, then a value noise field on an 8 pixel lattice," +
      " all of it in the mid tones where nothing clips. A blur radius, a median diameter, a" +
      " bloom radius and a grain supersample factor are all frequency selective, and a" +
      " frequency selective filter cannot be judged at all without a known frequency to point" +
      " it at.",
    paint(x, y) {
      const strip = CARD_WIDTH / 6;
      if (x < strip * CHEQUER_PITCHES.length) {
        const pitch = CHEQUER_PITCHES[Math.floor(x / strip)];
        const grey = (Math.floor(x / pitch) + Math.floor(y / pitch)) % 2 === 0 ? 64 : 192;
        return [grey, grey, grey];
      }
      const grey = 96 + roundHalfUp(64 * noiseAt(x, y));
      return [grey, grey, grey];
    },
  },
  {
    name: "gradient",
    height: 20,
    purpose:
      "A shallow diagonal gradient from level 104 to 144, one level every eight pixels or so." +
      " Banding is invisible on a steep ramp and obvious on a slow one, so this is the band" +
      " that shows a contrast or a curve quantising the mid tones into steps, and the only" +
      " place a preview can show it before the render does.",
    paint(x, y, height) {
      const span = CARD_WIDTH - 1 + 2 * (height - 1);
      const grey = 104 + roundHalfUp((40 * (x + 2 * y)) / span);
      return [grey, grey, grey];
    },
  },
  {
    name: "highlight",
    height: 24,
    purpose:
      "A bright field carrying a fine chequer of eight levels, with three discs whose cores" +
      " clip at 255 and whose falloff runs out over 20 pixels. Bloom has nothing to bloom on a" +
      " picture with no highlight, and a filter that crushes highlight detail shows it here as" +
      " the chequer disappearing outside the cores as well as inside them.",
    paint(x, y, height) {
      const detail = (Math.floor(x / 3) + Math.floor(y / 3)) % 2 === 0 ? -8 : 8;
      const centreY = height / 2;
      let bump = 0;
      for (let disc = 0; disc < 3; disc++) {
        const centreX = 40 + disc * 104;
        const dx = x - centreX;
        const dy = y - centreY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 20) bump = Math.max(bump, roundHalfUp(96 * (1 - distance / 20)));
      }
      const grey = level(176 + detail + bump);
      return [grey, grey, grey];
    },
  },
  {
    name: "shadow",
    height: 20,
    purpose:
      "A field between levels 4 and 22 carrying a two pixel chequer, so the detail is there" +
      " but only just. Contrast, dragan, shadows and highlights and a high pass all crush the" +
      " bottom of the scale before they touch anything else, and a preview with no readable" +
      " shadow cannot show the one thing those settings are most often got wrong on.",
    paint(x, y, height) {
      const span = CARD_WIDTH - 1 + height - 1;
      const base = 6 + roundHalfUp((10 * (x + y)) / span);
      const detail = (Math.floor(x / 2) + Math.floor(y / 2)) % 2 === 0 ? -2 : 6;
      const grey = level(base + detail);
      return [grey, grey, grey];
    },
  },
  {
    name: "flat",
    height: 8,
    purpose:
      "An unbroken field of 128. What a filter does to nothing at all: grain, monitor effects" +
      " and a high pass over a neutral area produce their whole signature here and are" +
      " invisible everywhere else.",
    paint() {
      return [128, 128, 128];
    },
  },
];

// The card's height in card pixels, which is the band table's own. Nine bands at the shortest
// height each stays readable at comes to 192, which is short enough that the whole card is
// visible at 1:1 inside a node of the default width, and it repeats cleanly where there is more
// room.
const CARD_HEIGHT = BANDS.reduce((total, band) => total + band.height, 0);

/**
 * The card's size and the band table, for an interface that names a band or a twin that checks
 * one.
 */
export const TEST_CARD = Object.freeze({
  width: CARD_WIDTH,
  height: CARD_HEIGHT,
  regions: Object.freeze(
    BANDS.map((band, index) => {
      const y = BANDS.slice(0, index).reduce((total, earlier) => total + earlier.height, 0);
      return Object.freeze({
        name: band.name,
        y,
        height: band.height,
        purpose: band.purpose,
      });
    }),
  ),
});

let cardPixels = null;

/**
 * Build the card once, as one opaque RGBA buffer.
 *
 * @returns {Uint8ClampedArray} The card's pixels, `CARD_WIDTH` by `CARD_HEIGHT`.
 */
function buildCard() {
  if (cardPixels) return cardPixels;

  const pixels = new Uint8ClampedArray(CARD_WIDTH * CARD_HEIGHT * 4);
  let top = 0;
  for (const band of BANDS) {
    for (let y = 0; y < band.height; y++) {
      let offset = (top + y) * CARD_WIDTH * 4;
      for (let x = 0; x < CARD_WIDTH; x++) {
        const colour = band.paint(x, y, band.height);
        pixels[offset] = colour[0];
        pixels[offset + 1] = colour[1];
        pixels[offset + 2] = colour[2];
        pixels[offset + 3] = 255;
        offset += 4;
      }
    }
    top += band.height;
  }

  cardPixels = pixels;
  return pixels;
}

/**
 * Make an `ImageData` without needing a context to hand.
 *
 * @param {number} width - Width in pixels.
 * @param {number} height - Height in pixels.
 * @returns {ImageData|null} The buffer, or null where neither route is available.
 */
function makeImageData(width, height) {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  try {
    if (typeof ImageData === "function") return new ImageData(w, h);
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    return scratch.getContext("2d")?.createImageData(w, h) ?? null;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to make a buffer of ${w} by ${h}:`, error);
    return null;
  }
}

/**
 * Draw the test card at 1:1, tiled to fill the area asked for.
 *
 * The card is never scaled.
 *
 * @param {number} width - Width to fill, in card pixels.
 * @param {number} height - Height to fill, in card pixels.
 * @param {{originX?: number, originY?: number}} [origin] - Which card pixel the top left of the
 *   area is, so a caller wanting a margin around the visible picture asks for a negative one.
 * @returns {ImageData|null} The pixels, or null where a buffer could not be made.
 */
export function createTestCard(width, height, origin = {}) {
  const image = makeImageData(width, height);
  if (!image) return null;

  const card = buildCard();
  const originX = Math.trunc(toNumber(origin.originX, 0));
  const originY = Math.trunc(toNumber(origin.originY, 0));
  const stride = CARD_WIDTH * 4;

  for (let y = 0; y < image.height; y++) {
    // The origin is negative for a caller asking for a margin, and a coordinate to the left of
    // or above the card indexes past the start of the tile unless the wrap takes the sign of
    // the divisor.
    const row = floorMod(originY + y, CARD_HEIGHT) * stride;
    let written = 0;
    while (written < image.width) {
      const column = floorMod(originX + written, CARD_WIDTH);
      const run = Math.min(CARD_WIDTH - column, image.width - written);
      image.data.set(
        card.subarray(row + column * 4, row + (column + run) * 4),
        (y * image.width + written) * 4,
      );
      written += run;
    }
  }
  return image;
}

/**
 * The backdrop that draws the test card.
 *
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
export function testCardBackdrop() {
  const answer = {
    state: PREVIEW_STATE.READY,
    label: "",
    image: null,
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
    scale: 1,
    kind: BACKDROP_KIND.CARD,
  };
  return {
    async load() {
      return answer;
    },
  };
}

/**
 * Work out where the header, the two panels and the footer sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the header, both panels and the footer.
 */
function computeLayout(width, height) {
  const footerY = Math.max(0, height - PAD - FOOTER_HEIGHT * FOOTER_LINES);
  const areaX0 = PAD;
  const areaX1 = Math.max(areaX0 + 1, width - PAD);
  const areaY0 = PAD + HEADER_HEIGHT;
  const areaY1 = Math.max(areaY0 + MIN_PANEL, footerY - 2);
  const panelWidth = Math.max(1, (areaX1 - areaX0 - PANEL_GAP) / 2);
  const panelHeight = areaY1 - areaY0;

  return {
    width,
    height,
    areaX0,
    areaX1,
    areaY0,
    areaY1,
    areaWidth: areaX1 - areaX0,
    headerY: PAD,
    footerY,
    before: { x: areaX0, y: areaY0, w: panelWidth, h: panelHeight },
    after: { x: areaX0 + panelWidth + PANEL_GAP, y: areaY0, w: panelWidth, h: panelHeight },
  };
}

/**
 * Fit the picture into one panel.
 *
 * The card fills the panel. A published picture keeps its aspect and is centred.
 *
 * @param {{x: number, y: number, w: number, h: number}} panel - Panel in element pixels.
 * @param {object} frame - Frame from `normaliseFrame`, carrying its `kind`.
 * @returns {{x: number, y: number, w: number, h: number}|null} Where the picture is drawn, or
 *   null while there is nothing to draw it at.
 */
function fitView(panel, frame) {
  if (!(panel.w > 0) || !(panel.h > 0)) return null;
  if (frame.kind === BACKDROP_KIND.CARD) return { ...panel };
  if (!(frame.width > 0) || !(frame.height > 0) || !frame.image) return null;

  const fit = Math.min(panel.w / frame.width, panel.h / frame.height);
  const w = Math.max(1, frame.width * fit);
  const h = Math.max(1, frame.height * fit);
  return { x: panel.x + (panel.w - w) / 2, y: panel.y + (panel.h - h) / 2, w, h };
}

/**
 * Convert a rectangle in element pixels into whole pixels at one ratio.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - Rectangle in element pixels.
 * @param {number} ratio - Pixels per element pixel.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle at that ratio.
 */
function deviceRect(rect, ratio) {
  // Each edge is rounded on its own, so a rectangle never gains or loses a pixel to its
  // neighbour and the two panels stay the same size as each other.
  const x = Math.round(rect.x * ratio);
  const y = Math.round(rect.y * ratio);
  return {
    x,
    y,
    w: Math.max(0, Math.round((rect.x + rect.w) * ratio) - x),
    h: Math.max(0, Math.round((rect.y + rect.h) * ratio) - y),
  };
}

/**
 * Write a reduction factor the way the pack states it.
 *
 * @param {number} value - Source pixels per preview pixel.
 * @returns {string} The factor with at most one decimal.
 */
export function formatFactor(value) {
  return String(Math.round(value * 10) / 10);
}

/**
 * Repeat the edge pixels of a picture outwards into the margin around it.
 *
 * @param {ImageData} image - Buffer holding the picture inset by the margin.
 * @param {number} margin - Margin in pixels on every side.
 * @returns {void}
 */
function clampEdges(image, margin) {
  if (margin <= 0) return;
  const { width, height, data } = image;
  const innerWidth = width - margin * 2;
  const innerHeight = height - margin * 2;
  if (innerWidth <= 0 || innerHeight <= 0) return;

  for (let y = margin; y < margin + innerHeight; y++) {
    const row = y * width * 4;
    const left = row + margin * 4;
    const right = row + (margin + innerWidth - 1) * 4;
    for (let x = 0; x < margin; x++) {
      data.copyWithin(row + x * 4, left, left + 4);
      data.copyWithin(row + (margin + innerWidth + x) * 4, right, right + 4);
    }
  }

  const topRow = margin * width * 4;
  const bottomRow = (margin + innerHeight - 1) * width * 4;
  for (let y = 0; y < margin; y++) {
    data.copyWithin(y * width * 4, topRow, topRow + width * 4);
    data.copyWithin((margin + innerHeight + y) * width * 4, bottomRow, bottomRow + width * 4);
  }
}

/**
 * Build the surface for one node.
 *
 * @param {object} options - What is drawn, and what draws it.
 * @param {object} [options.node] - The node the surface is drawn on. Used to ask for the
 *   picture that node published, and to repaint the graph.
 * @param {{load: () => Promise<object>}} [options.backdrop] - What the filter is applied to.
 *   `imageBackdrop` from `backdrop.js` by default when a node is given, and the test card
 *   otherwise. Whatever it answers, the card stands in whenever no picture arrives, so the
 *   surface always has something to show and always says which it is showing.
 * @param {(source: ImageData, target: ImageData, info: object) => void} [options.filter] - The
 *   filter, called once per repaint. `source` is the picture with the margin around it,
 *   `target` starts as a copy of it and is what gets drawn, and `info` carries `kind`, `scale`,
 *   `margin`, `width` and `height`. A filter reads the node's widgets itself and writes
 *   nothing. Left out, the two panels show the same picture.
 * @param {((info: object) => {icon: string, detail: string, note?: string})} [options.fidelity] -
 *   What this preview does not reproduce, in the adopter's own words, with the bound it was
 *   measured at. `icon` is the value of `ICON` for the class the preview is in, `detail` is the
 *   sentence the glyph carries on hover, and `note` is the one part worth room on screen: a state
 *   that changes what somebody should do now, such as a setting the run reads off a link. Asked
 *   on every repaint, so a claim that changes with the settings changes with them.
 * @param {number|((info: object) => number)} [options.margin] - How many pixels beyond the
 *   visible picture the filter needs, on every side. Three times a blur radius, in preview
 *   pixels, for anything that blurs.
 * @param {{before?: string, after?: string}} [options.labels] - What to call the two panels.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {boolean} [options.metrics] - Draw a band under the panels measuring the picture
 *   against the filtered one: PSNR, SSIM, RMSE, MAE, colour shift and both histograms. Off by
 *   default.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   refresh: () => void, dispose: () => void}} The element to hand to `addDOMWidget`, the
 *   height it was built for, a coalesced repaint, a fresh ask for the backdrop, and teardown.
 */
export function createFilterSurface(options = {}) {
  const measured = options.metrics === true;
  const settings = {
    node: options.node ?? null,
    backdrop: options.backdrop ?? (options.node ? imageBackdrop(options.node) : testCardBackdrop()),
    filter: typeof options.filter === "function" ? options.filter : null,
    fidelity: typeof options.fidelity === "function" ? options.fidelity : null,
    margin: options.margin ?? 0,
    labels: { ...DEFAULT_LABELS, ...(options.labels ?? {}) },
    height: Math.max(UI_MARGIN * 2 + MIN_PANEL, toNumber(options.height, DEFAULT_HEIGHT))
      + (measured ? METRICS_HEIGHT : 0),
  };

  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${settings.height - UI_MARGIN * 2}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
    ...(measured ? ["display:flex", "flex-direction:column"] : []),
  ].join(";");

  // The panels are drawn on a stage at the top of the element, the measurement band under it.
  const stage = measured ? document.createElement("div") : root;
  if (measured) {
    stage.style.cssText = "position:relative;flex:1 1 auto;min-height:0;overflow:hidden";
    root.appendChild(stage);
  }

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  stage.appendChild(canvas);

  const metrics = measured ? createMetricsSection({ workingEdge: METRICS_EDGE }) : null;
  if (metrics) {
    // Sized to its figures, which wrap to a second row on a narrow node.
    metrics.element.style.flex = "0 0 auto";
    metrics.element.style.padding = `${PAD}px ${PAD}px 0`;
    root.appendChild(metrics.element);
  }

  // The footer's glyphs state their sentence through the element's own title. The regions are
  // handed over again on every repaint, since a glyph moves whenever the node is resized.
  const titles = hoverTitles(root);

  const state = {
    frame: cardFrame(normaliseFrame({ state: PREVIEW_STATE.LOADING })),
    loading: false,
    reloadWanted: false,
    retryTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    source: null,
    sourceKey: "",
    target: null,
    scratch: null,
    token: 0,
    note: "",
    holding: false,
    holdBox: null,
    disposed: false,
    measure: null,
    measureTimer: 0,
    measureCanvases: null,
  };

  // The footer's two hover regions, rebuilt on every paint.
  let glyphTitle = null;
  let holdTitle = null;

  /**
   * Fall back to the card whenever the answer is not the picture.
   *
   * @param {object} frame - Frame from `normaliseFrame`.
   * @returns {object} The frame, with its `kind` and, for the card, the card's own size.
   */
  function cardFrame(frame) {
    if (frame.image && frame.width > 0 && frame.height > 0) {
      return { ...frame, kind: BACKDROP_KIND.IMAGE };
    }
    return {
      ...frame,
      kind: BACKDROP_KIND.CARD,
      image: null,
      width: CARD_WIDTH,
      height: CARD_HEIGHT,
      scale: 1,
    };
  }

  /**
   * Whether two answers stand for the same picture in the same state.
   *
   * @param {object} a - One frame.
   * @param {object} b - The other.
   * @returns {boolean} True while nothing the surface draws differs between them.
   */
  function sameFrame(a, b) {
    return (
      a.state === b.state
      && a.kind === b.kind
      && a.image === b.image
      && a.width === b.width
      && a.height === b.height
      && a.scale === b.scale
      && a.label === b.label
    );
  }

  /**
   * Ask the backdrop for the picture, and again later while the answer is not one.
   *
   * @returns {void}
   */
  function loadFrame() {
    if (state.disposed) return;
    if (state.loading) {
      state.reloadWanted = true;
      return;
    }
    state.loading = true;
    let changed = false;
    Promise.resolve()
      .then(() => settings.backdrop?.load?.())
      .then((answer) => {
        if (state.disposed) return;
        const frame = cardFrame(normaliseFrame(answer));
        // A node that has not run answers the same waiting frame every three seconds for as long
        // as it is left alone, and taking that answer as new drops the buffers, samples the
        // backdrop again and runs the whole filter again, which on a wide node is a visible hitch
        // every three seconds with nothing on screen changing.
        changed = !sameFrame(state.frame, frame);
        if (!changed) return;
        state.frame = frame;
        // The buffers hold pixels sampled from the picture that has just been replaced.
        state.token += 1;
        state.source = null;
        state.sourceKey = "";
      })
      .catch((error) => {
        if (state.disposed) return;
        console.error(`[${LOG_NAME}] Failed to read the backdrop:`, error);
        const frame = cardFrame(normaliseFrame({ state: PREVIEW_STATE.FAILED }));
        changed = !sameFrame(state.frame, frame);
        state.frame = frame;
      })
      .finally(() => {
        state.loading = false;
        if (state.disposed) return;
        if (state.reloadWanted) {
          state.reloadWanted = false;
          loadFrame();
          return;
        }
        scheduleRetry();
        if (changed) schedulePaint();
      });
  }

  /**
   * Ask again for a backdrop that is not the picture yet.
   *
   * @returns {void}
   */
  function scheduleRetry() {
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.retryTimer = 0;
    if (state.disposed || state.frame.state === PREVIEW_STATE.READY) return;
    state.retryTimer = setTimeout(() => {
      state.retryTimer = 0;
      // A hidden tab publishes nothing new to ask about, so the wait starts again instead.
      if (document.hidden) scheduleRetry();
      else loadFrame();
    }, RETRY_INTERVAL);
  }

  /**
   * Ask for the backdrop again now.
   *
   * @returns {void}
   */
  function refresh() {
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.retryTimer = 0;
    loadFrame();
  }

  /**
   * What the filter is told about the picture it has been handed.
   *
   * @param {object} view - The picture buffer, from `deviceRect` at the picture's own ratio.
   * @param {number} margin - Margin the buffers carry on every side.
   * @returns {{kind: string, scale: number, width: number, height: number, margin: number,
   *   source: {width: number, height: number}}} What the picture is and what it stands for.
   */
  function describe(view, margin) {
    const frame = state.frame;
    const scale = frame.kind === BACKDROP_KIND.CARD ? 1 : frame.width / Math.max(1, view.w);
    return {
      kind: frame.kind,
      scale,
      width: view.w,
      height: view.h,
      margin,
      source: { width: frame.width, height: frame.height },
    };
  }

  /**
   * How far beyond the visible picture the filter asked for.
   *
   * @param {object} info - Description from `describe`, built with no margin.
   * @returns {number} The margin in buffer pixels, held to what a buffer can carry.
   */
  function readMargin(info) {
    try {
      const value = typeof settings.margin === "function" ? settings.margin(info) : settings.margin;
      return clamp(Math.ceil(toNumber(value, 0)), 0, MAX_MARGIN);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the margin:`, error);
      return 0;
    }
  }

  /**
   * The resolution the picture is worked out at, in buffer pixels per layout pixel.
   *
   * @param {object} view - Where the picture goes in the panel, in layout units.
   * @returns {number} A ratio of at least what the screen's density asks for.
   */
  function pictureRatio(view) {
    const frame = state.frame;
    const across = view.w > 0 && frame.width > 0 ? frame.width / view.w : 1;
    // The filter walks every pixel of this buffer on every repaint, and a slider dragged over
    // a picture nobody can wait for is worse than one drawn from less detail, so the ask is
    // held to a budget worked back from the panel the buffer is measured against.
    // Held, the budget is stood down and the picture is worked out at the image's own detail.
    if (state.holding) return Math.max(contentRatio(root), across);
    const affordable = Math.sqrt(FILTER_PIXEL_BUDGET / Math.max(1, view.w * view.h));
    return Math.max(contentRatio(root), Math.min(across, affordable));
  }

  /**
   * Build the picture the filter reads, with the margin around it.
   *
   * @param {object} view - The picture inside the buffer, in buffer pixels.
   * @param {number} margin - Margin on every side, in buffer pixels.
   * @returns {ImageData|null} The pixels, or null when there are none to sample.
   */
  function ensureSource(view, margin) {
    const frame = state.frame;
    // Everything the buffer's contents depend on is in the key and nothing else is.
    const key = `${frame.kind}|${state.token}|${view.w}x${view.h}|${margin}`;
    if (state.source && state.sourceKey === key) return state.source;

    const width = view.w + margin * 2;
    const height = view.h + margin * 2;
    if (!(view.w > 0) || !(view.h > 0)) return null;

    let image = null;
    if (frame.kind === BACKDROP_KIND.CARD) {
      image = createTestCard(width, height, { originX: -margin, originY: -margin });
    } else {
      image = sampleImage(frame.image, width, height, view, margin);
    }
    if (!image) return null;

    state.source = image;
    state.sourceKey = key;
    return image;
  }

  /**
   * Sample the published picture into a buffer, reduced to the panel and clamped at its edges.
   *
   * @param {HTMLImageElement} image - The picture the node published.
   * @param {number} width - Buffer width in buffer pixels.
   * @param {number} height - Buffer height in buffer pixels.
   * @param {object} view - Where the picture goes inside the buffer, in buffer pixels.
   * @param {number} margin - Margin on every side.
   * @returns {ImageData|null} The pixels, or null when the picture could not be read.
   */
  function sampleImage(image, width, height, view, margin) {
    try {
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      const ctx = scratch.getContext("2d", { willReadFrequently: true });
      if (!ctx) return null;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      // The picture is the published thumbnail, capped at 512 on its longest edge, so a buffer
      // larger than that magnifies it. Only a wide node and a dense screen reach that now, the
      // zoom having no say in this size, and it is a ceiling on the backdrop rather than on the
      // surface: the panels, the labels and the glyph go on sharpening where the picture cannot.
      ctx.drawImage(image, margin, margin, view.w, view.h);
      const pixels = ctx.getImageData(0, 0, width, height);
      clampEdges(pixels, margin);
      return pixels;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the picture:`, error);
      return null;
    }
  }

  /**
   * Run the filter over a copy of the picture.
   *
   * @param {ImageData} source - The picture with its margin.
   * @param {object} info - Description from `describe`.
   * @returns {ImageData} What to draw in the second panel.
   */
  function runFilter(source, info) {
    if (!settings.filter) return source;

    // The buffer is kept between repaints.
    let target = state.target;
    if (!target || target.width !== source.width || target.height !== source.height) {
      target = makeImageData(source.width, source.height);
      state.target = target;
    }
    if (!target) return source;
    target.data.set(source.data);

    try {
      settings.filter(source, target, info);
      return target;
    } catch (error) {
      console.error(`[${LOG_NAME}] The filter failed:`, error);
      state.note = "the filter failed, see the console";
      return source;
    }
  }

  /**
   * The canvas a picture drawn larger than it was worked out goes through.
   *
   * @param {number} width - Width in buffer pixels.
   * @param {number} height - Height in buffer pixels.
   * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D}|null} The canvas, or
   *   null where a context could not be had.
   */
  function ensureScratch(width, height) {
    try {
      if (!state.scratch) {
        const element = document.createElement("canvas");
        const context = element.getContext("2d");
        if (!context) return null;
        state.scratch = { canvas: element, ctx: context };
      }
      if (state.scratch.canvas.width !== width) state.scratch.canvas.width = width;
      if (state.scratch.canvas.height !== height) state.scratch.canvas.height = height;
      return state.scratch;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to make the scratch canvas:`, error);
      return null;
    }
  }

  /**
   * Draw one panel's picture into the room the zoom gives it.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into, transformed by `ratio`.
   * @param {ImageData} image - Buffer holding the picture inset by the margin.
   * @param {number} margin - Margin the buffer carries on every side.
   * @param {object} picture - The picture inside that buffer, in buffer pixels.
   * @param {object} device - Where it goes on the surface, in device pixels.
   * @param {number} ratio - Device pixels per element pixel.
   * @returns {void}
   */
  function drawPicture(ctx, image, margin, picture, device, ratio) {
    try {
      if (device.w === picture.w && device.h === picture.h) {
        ctx.putImageData(
          image,
          device.x - margin,
          device.y - margin,
          margin,
          margin,
          device.w,
          device.h,
        );
        return;
      }

      const holder = ensureScratch(image.width, image.height);
      if (!holder) return;
      holder.ctx.putImageData(image, 0, 0);

      // Magnified, an unsmoothed pixel becomes a block of its own colour and the picture reads
      // coarse, which is the truth: it is the arithmetic, at the size the arithmetic was done at.
      // Smoothing would put tones between the filter's own on screen, and this preview exists to
      // be judged on levels and edge widths, so it would show banding the filter does not cause
      // and edges softer than the ones it leaves.
      //
      // Reduced it is smoothed, and the reverse looks more honest than it is. Measured on the
      // card's own alternating 64 and 192 band taken to half size: smoothed it reads 128 over the
      // whole band, the mean of what is there, while unsmoothed it reads a uniform 64 or a uniform
      // 192 depending only on which column the sampling lands on, so a one pixel shift flips the
      // band from dark to light. Averaging is also what `sampleImage` already does to reach this
      // size. Keeping the real levels is worth nothing when which of them survives is arbitrary.
      const reducing = device.w < picture.w || device.h < picture.h;
      ctx.save();
      // Only where the picture lands, so a letterboxed panel keeps its own flat background
      // and a cut-out still reads against something.
      paintChecker(
        ctx, device.x / ratio, device.y / ratio, device.w / ratio, device.h / ratio,
      );
      ctx.imageSmoothingEnabled = reducing;
      ctx.imageSmoothingQuality = "high";
      // `drawImage` is under the transform and `device` is not, so the destination is stated in
      // element pixels that land back on those whole device pixels. The source rectangle is the
      // picture alone, leaving the margin where it is: it is there for the filter to read past
      // the edge, and it was never part of what is shown.
      ctx.drawImage(
        holder.canvas,
        margin,
        margin,
        picture.w,
        picture.h,
        device.x / ratio,
        device.y / ratio,
        device.w / ratio,
        device.h / ratio,
      );
      ctx.restore();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to draw the picture:`, error);
    }
  }

  /**
   * The adopter's fidelity, or the warning that says there is not one.
   *
   * @param {object|null} info - Description from `describe`, where there is a picture.
   * @returns {{icon: string, detail: string, note: string, stated: boolean}} Which glyph to draw,
   *   the sentence it carries on hover, the part that belongs on screen, and whether the adopter
   *   stated any of it.
   */
  function fidelityClaim(info) {
    let claim = null;
    try {
      claim = settings.fidelity ? settings.fidelity(info) : null;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the fidelity:`, error);
    }

    const detail = typeof claim?.detail === "string" ? claim.detail.trim() : "";
    if (!detail) return { icon: ICON.WARNING, detail: NO_FIDELITY, note: "", stated: false };
    return {
      icon: ICON_TITLES[claim.icon] ? claim.icon : ICON.APPROXIMATE,
      detail,
      note: typeof claim.note === "string" ? claim.note.trim() : "",
      stated: true,
    };
  }

  /**
   * Draw the glyph for the fidelity into the footer line, and hand its hover text over.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} claim - Claim from `fidelityClaim`.
   * @param {boolean} drew - Whether the panels were given a picture this pass.
   * @param {number} x - Left edge of the line, in element pixels.
   * @param {number} y - Middle of the line, in element pixels.
   * @returns {number} How much of the line the glyph took, gap included.
   */
  // Named apart from `drawFidelityGlyph` in `interface/icons.js`, which is the same glyph
  // owning a canvas of its own. This one is an item in a line and answers the room it took.
  function drawInlineFidelityGlyph(ctx, theme, claim, drew, x, y) {
    if (!drew) {
      // A claim about how truly a preview stands for the render is worth nothing where there is
      // no preview to stand for it.
      glyphTitle = null;
      return 0;
    }
    const colour = claim.stated && claim.icon !== ICON.WARNING ? theme.fgMuted : theme.warning;
    const box = drawIcon(ctx, claim.icon, x, y - ICON_SIZE / 2, ICON_SIZE, colour);
    glyphTitle = { ...box, title: iconTitle(claim.icon, claim.detail) };
    return ICON_SIZE + GLYPH_GAP;
  }

  /**
   * What the picture is, in the words the footer states it in.
   *
   * @param {object|null} info - Description from `describe`.
   * @returns {string} The name and size of the backdrop.
   */
  function sourceSentence(info) {
    const frame = state.frame;
    if (frame.kind !== BACKDROP_KIND.CARD) {
      return `source ${Math.round(frame.width)}x${Math.round(frame.height)}`;
    }

    const card = `test card ${CARD_WIDTH}x${CARD_HEIGHT}`;
    if (!info) return card;
    const width = Math.round(info.width);
    const height = Math.round(info.height);
    const repeats = width > CARD_WIDTH || height > CARD_HEIGHT;
    const cut = width < CARD_WIDTH || height < CARD_HEIGHT;
    if (repeats && cut) return `${card}, tiled and cropped`;
    if (repeats) return `${card}, tiled`;
    if (cut) return `${card}, cropped`;
    return card;
  }

  /**
   * Draw the button that lifts the picture to the image's own detail while it is held.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {boolean} drew - Whether a picture was drawn this pass.
   * @param {number} middle - Middle of the footer line, in layout pixels.
   * @returns {number} The room it took, gap included, so the note is not drawn under it.
   */
  function drawHoldButton(ctx, theme, drew, middle) {
    const layout = state.layout;
    // Nothing to hold while there is no picture, and a button that does nothing is worse than
    // no button, so the hit box goes too and the hold cannot be started.
    if (!drew || state.frame.kind === BACKDROP_KIND.CARD) {
      state.holdBox = null;
      return 0;
    }

    const label = state.holding ? HOLD_HELD : HOLD_LABEL;
    const width = Math.ceil(ctx.measureText(label).width) + HOLD_PADDING * 2;
    const box = {
      x: layout.areaX0 + layout.areaWidth - width,
      y: middle - HOLD_HEIGHT / 2,
      width,
      height: HOLD_HEIGHT,
    };
    state.holdBox = box;

    ctx.save();
    ctx.fillStyle = state.holding ? theme.accent : theme.inputBg;
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeStyle = theme.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.width - 1, box.height - 1);
    ctx.fillStyle = state.holding ? theme.selectionText : theme.fgMuted;
    ctx.textAlign = "center";
    ctx.fillText(label, box.x + box.width / 2, middle);
    ctx.restore();

    holdTitle = {
      ...box,
      title: "Hold to work the picture out at the size the node received, which is the detail "
        + "the render acts on. Released, it goes back to a smaller copy so a slider still "
        + "answers while it is dragged.",
    };
    return width + HOLD_GAP;
  }

  /**
   * Start or stop holding, and rebuild the picture at the other resolution.
   *
   * @param {boolean} holding - Whether the button is down.
   * @returns {void}
   */
  function setHolding(holding) {
    if (state.holding === holding) return;
    state.holding = holding;
    // The buffer is keyed on its own size, so a different resolution is a different buffer.
    state.source = null;
    state.sourceKey = "";
    schedulePaint();
  }

  /**
   * Draw the footer line.
   *
   * The glyph, then the state or the size of the picture on the left, and the note on the right.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object|null} info - Description from `describe`.
   * @param {boolean} drew - Whether the panels were given a picture this pass.
   * @returns {void}
   */
  function drawFooter(ctx, theme, info, drew) {
    const layout = state.layout;
    const middle = layout.footerY + FOOTER_HEIGHT / 2;
    const claim = fidelityClaim(info);
    const note = state.note || claim.note || "";
    // The state's words go here rather than across the panels, since the card is what is drawn
    // where the picture would be. A node that has not run has no words at all, and the card named
    // on the left is the whole of what there is to say about that.
    const standIn = drew ? LABELS[state.frame.state] || "" : "";
    const failed = state.frame.state === PREVIEW_STATE.FAILED;
    const warns = failed || Boolean(note);

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    glyphTitle = null;
    holdTitle = null;
    const glyphWidth = drawInlineFidelityGlyph(ctx, theme, claim, drew, layout.areaX0, middle);
    const holdWidth = drawHoldButton(ctx, theme, drew, middle);
    titles.set([glyphTitle, holdTitle].filter(Boolean));

    // Two things can want words here at once, so they take a half each rather than one hiding
    // the other. The right holds the note, which is the filter's own failure or a setting read
    // off a link, and it is given the room it needs first.
    const shownNote = elideText(ctx, note, layout.areaWidth - glyphWidth - holdWidth);
    let noteWidth = 0;
    if (shownNote) {
      noteWidth = ctx.measureText(shownNote).width + FOOTER_GAP;
      ctx.textAlign = "right";
      ctx.fillStyle = warns ? theme.warning : theme.fgMuted;
      ctx.fillText(shownNote, layout.areaX1, middle);
    }

    // The left holds the state the card is standing in for while it is standing in for one, and
    // the size of the picture once it is the picture, since that is there to be read rather than
    // acted on.
    const lead = standIn || (info ? sourceSentence(info) : "");
    const shownLead = elideText(ctx, lead, layout.areaWidth - glyphWidth - noteWidth);
    if (shownLead) {
      ctx.fillStyle = standIn && failed ? theme.warning : theme.fgMuted;
      ctx.textAlign = "left";
      ctx.fillText(shownLead, layout.areaX0 + glyphWidth, middle);
    }
  }

  /**
   * Draw the words for a panel that was given nothing to show.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} panel - Panel in element pixels.
   * @returns {void}
   */
  function drawEmpty(ctx, theme, panel) {
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.warning;
    const shown = elideText(ctx, NO_BUFFER, panel.w - FOOTER_GAP);
    if (shown) ctx.fillText(shown, panel.x + panel.w / 2, panel.y + panel.h / 2);
  }

  /**
   * Copy the picture inside a buffer, without its margin, onto a canvas the band can read.
   *
   * @param {number} index - Which of the two kept canvases, 0 or 1.
   * @param {ImageData} image - Buffer holding the picture inset by the margin.
   * @param {number} margin - Margin the buffer carries on every side.
   * @param {{w: number, h: number}} picture - The picture inside the buffer.
   * @returns {HTMLCanvasElement|null} The canvas, or null where a context could not be had.
   */
  function measureCanvas(index, image, margin, picture) {
    if (!state.measureCanvases) {
      state.measureCanvases = [document.createElement("canvas"), document.createElement("canvas")];
    }
    const held = state.measureCanvases[index];
    if (held.width !== picture.w) held.width = picture.w;
    if (held.height !== picture.h) held.height = picture.h;
    const ctx = held.getContext("2d");
    if (!ctx) return null;
    ctx.putImageData(image, -margin, -margin, margin, margin, picture.w, picture.h);
    return held;
  }

  /**
   * Measure the last painted pair once the settings have rested.
   *
   * @returns {void}
   */
  function scheduleMeasure() {
    if (!metrics) return;
    if (state.measureTimer) clearTimeout(state.measureTimer);
    state.measureTimer = setTimeout(() => {
      state.measureTimer = 0;
      if (state.disposed) return;
      try {
        const pair = state.measure;
        if (!pair) {
          metrics.clear();
          return;
        }
        const before = measureCanvas(0, pair.source, pair.margin, pair.picture);
        const after = measureCanvas(1, pair.after, pair.margin, pair.picture);
        if (!before || !after) {
          metrics.clear();
          return;
        }
        metrics.update(before, after);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to measure the filter:`, error);
      }
    }, METRICS_DELAY);
  }

  /**
   * Draw the whole surface.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (!width || !height) return;

    // The graph's zoom is in here as well as the screen's density, so a magnified node is drawn
    // at the resolution it is shown at. Everything below `setTransform` stays in layout units.
    const ratio = surfaceRatio(root);
    const deviceWidth = Math.max(1, Math.round(width * ratio));
    const deviceHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== deviceWidth) canvas.width = deviceWidth;
    if (canvas.height !== deviceHeight) canvas.height = deviceHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    state.layout = computeLayout(width, height);
    const layout = state.layout;
    const theme = readTheme();
    state.note = "";

    // Both panels are laid down whole. A picture keeping the aspect of the image the node
    // received does not fill its panel, and what it leaves over is the surface's own background
    // rather than the node body showing through a hole.
    ctx.fillStyle = theme.inputBg;
    for (const panel of [layout.before, layout.after]) {
      ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
    }

    const view = fitView(layout.before, state.frame);
    // Two resolutions. The picture is worked out at the detail the node's own image holds and
    // lands on the surface scaled, so a setting that acts on fine detail, sharpness above all,
    // acts on the detail the render will act on rather than on a panel sized copy where there
    // is none left to find. Neither the layout rectangle nor the image's size moves when the
    // graph is zoomed, so the same widgets still give the same pixels however far in somebody
    // is looking: a buffer that grew with the viewport would move a fixed kernel's reach into
    // the subject, change how much work a cost bound affords and change the factor a fidelity
    // is claimed against. Everything else is drawn at the zoom and sharpens with it: the
    // panels, the labels, the borders, the glyph and the footer.
    const picture = view ? deviceRect(view, pictureRatio(view)) : null;
    const device = view ? deviceRect(view, ratio) : null;
    let info = null;
    let drew = false;
    state.measure = null;

    if (picture && device && picture.w > 0 && picture.h > 0 && device.w > 0 && device.h > 0) {
      const margin = readMargin(describe(picture, 0));
      const source = ensureSource(picture, margin);
      info = describe(picture, margin);
      if (source) {
        const after = runFilter(source, info);
        const afterDevice = {
          ...device,
          x: Math.round((view.x + layout.after.x - layout.before.x) * ratio),
        };
        drawPicture(ctx, source, margin, picture, device, ratio);
        drawPicture(ctx, after, margin, picture, afterDevice, ratio);
        drew = true;
        // Only the node's own picture is measured, never the test card.
        if (state.frame.kind === BACKDROP_KIND.IMAGE) {
          state.measure = { source, after, margin, picture: { w: picture.w, h: picture.h } };
        }
      } else {
        drawEmpty(ctx, theme, layout.before);
        drawEmpty(ctx, theme, layout.after);
      }
    } else {
      drawEmpty(ctx, theme, layout.before);
      drawEmpty(ctx, theme, layout.after);
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    for (const panel of [layout.before, layout.after]) {
      ctx.strokeRect(
        Math.round(panel.x) + 0.5,
        Math.round(panel.y) + 0.5,
        Math.max(1, Math.round(panel.w) - 1),
        Math.max(1, Math.round(panel.h) - 1),
      );
    }

    ctx.font = TAG_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.fgMuted;
    const tagY = layout.headerY + HEADER_HEIGHT / 2;
    ctx.fillText(settings.labels.before, layout.before.x, tagY, Math.max(1, layout.before.w));
    ctx.fillText(settings.labels.after, layout.after.x, tagY, Math.max(1, layout.after.w));

    drawFooter(ctx, theme, info, drew);
    scheduleMeasure();
  }

  /**
   * Repaint on the next frame, coalescing repeated requests into one.
   *
   * @returns {void}
   */
  function schedulePaint() {
    if (state.disposed || state.paintHandle) return;
    state.paintHandle = requestAnimationFrame(() => {
      state.paintHandle = 0;
      try {
        paint();
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw the filter surface:`, error);
      }
    });
  }

  /**
   * Wrap an event handler so a failure is logged rather than thrown at the browser.
   *
   * @param {(event: Event) => void} handler - Handler to wrap.
   * @returns {(event: Event) => void} The wrapped handler.
   */
  function guard(handler) {
    return (event) => {
      try {
        handler(event);
      } catch (error) {
        console.error(`[${LOG_NAME}] Filter surface input failed:`, error);
      }
    };
  }

  root.addEventListener(
    "contextmenu",
    guard((event) => {
      // The graph canvas suppresses its own context menu on its own element, and this is a
      // separate element, so the browser menu would otherwise open over the node.
      event.preventDefault();
      event.stopPropagation();
    }),
  );

  // The surface scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);

  root.addEventListener(
    "pointerdown",
    guard((event) => {
      // Middle button panning belongs to the canvas underneath.
      if (event.button === 1) {
        app.canvas?.processMouseDown?.(event);
        return;
      }
      if (event.button !== 0 || !state.holdBox) return;
      const at = elementPoint(root, event);
      const box = state.holdBox;
      if (
        at.x < box.x || at.x > box.x + box.width
        || at.y < box.y || at.y > box.y + box.height
      ) return;
      // The press is the surface's own, so the canvas must not also read it as a click on the
      // node, which would select and start dragging the node under the button.
      event.preventDefault();
      event.stopPropagation();
      setHolding(true);
    }),
  );

  root.addEventListener(
    "pointermove",
    guard((event) => {
      if (event.buttons & 4) app.canvas?.processMouseMove?.(event);
    }),
  );

  root.addEventListener(
    "pointerup",
    guard((event) => {
      if (event.button === 1) app.canvas?.processMouseUp?.(event);
      else setHolding(false);
    }),
  );

  // Every other way the button can be lost while it is down. A hold left on costs a slow
  // repaint on every later change, so it is released by the pointer leaving, by the pointer
  // being taken away, by the window going to the background, and by the tab being hidden.
  for (const name of ["pointerleave", "pointercancel"]) {
    root.addEventListener(name, guard(() => setHolding(false)));
  }
  const releaseHold = () => {
    if (!state.disposed) setHolding(false);
  };
  window.addEventListener("blur", releaseHold);
  document.addEventListener("visibilitychange", releaseHold);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => {
      // A different width is a different picture, so the buffers go with it.
      state.source = null;
      state.sourceKey = "";
      schedulePaint();
    });
    observer.observe(root);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here. The buffers are not dropped alongside it the way a
  // resize drops them: a zoom changes how much room the picture is drawn across and nothing
  // about the picture, so the repaint draws the one already worked out.
  let unwatchRatio = watchSurfaceRatio(root, schedulePaint);

  // The picture is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the timers, observers and buffers the surface holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    if (state.retryTimer) clearTimeout(state.retryTimer);
    if (state.measureTimer) clearTimeout(state.measureTimer);
    state.paintHandle = 0;
    state.retryTimer = 0;
    state.measureTimer = 0;
    state.source = null;
    state.sourceKey = "";
    state.target = null;
    state.scratch = null;
    state.measure = null;
    state.measureCanvases = null;
    metrics?.dispose();
    titles.dispose();
    // Taken off the page rather than off the root, so a disposed surface leaves neither behind.
    window.removeEventListener("blur", releaseHold);
    document.removeEventListener("visibilitychange", releaseHold);
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
  }

  loadFrame();
  schedulePaint();

  return {
    element: root,
    height: settings.height,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    refresh,
    dispose,
  };
}

/**
 * Append a surface to a node as a DOM widget that carries no data.
 *
 * @param {object} node - The node being created.
 * @param {{element: HTMLElement, height: number}} surface - Surface from `createFilterSurface`.
 * @param {{name?: string, type?: string}} [names] - What to call the widget and its type. The
 *   type collides with no key of the frontend's widget registry, which is what a type string
 *   has to avoid, so several nodes may share it.
 * @returns {object} The widget that was added.
 */
export function appendFilterWidget(node, surface, names = {}) {
  return appendInterfaceWidget(node, surface, {
    name: names.name ?? DEFAULT_WIDGET_NAME,
    type: names.type ?? DEFAULT_WIDGET_TYPE,
  });
}
