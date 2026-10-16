/**
 * The two panels Image Style Filter draws, one per value of `contact_sheet`.
 *
 * The sheet holds all 37 styles over the image the node received, in tile pixels and drawn in
 * device pixels. Beside it, a before and after band.
 */

// This file sits at the top of `web/`, so ComfyUI's own modules are reached with
// `../../scripts/`, one level shallower than the shared components under `web/interface/`.
import { app } from "../../scripts/app.js";
import { imageBackdrop, normaliseFrame } from "./interface/backdrop.js";
import { createBeforeAfterPanel } from "./interface/before_after.js";
import { elideText } from "./interface/canvas_text.js";
import {
  BACKDROP_KIND,
  TEST_CARD,
  appendFilterWidget,
  createTestCard,
} from "./interface/filter_surface.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { LABELS, PREVIEW_STATE, fetchInputPreview } from "./interface/preview.js";
import { contentRatio, surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeDone } from "./interface/run_events.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.StyleFilterUI";
const NODE_NAME = "Image Style Filter";
const SETTING_ID = "WAS.StyleFilter.ShowInterface";

const STYLE_WIDGET = "style";
const SHEET_WIDGET = "contact_sheet";

const UI_WIDGET_NAME = "was_style_ui";
const UI_WIDGET_TYPE = "was_style_sheet";

// The second panel, drawn in place of the sheet while `contact_sheet` is off.
const BAND_WIDGET_NAME = "was_style_band_ui";
const BAND_WIDGET_TYPE = "was_style_band";

// The slot the node files the picture it was handed and the picture it answered with under,
// which is the name of its picture input.
const PAIR_SLOT = "image";

// Height of the appended widget in node units, and the margin a DOM widget element is inset by
// on every side, which makes the element itself shorter than the widget by twice it.
const UI_HEIGHT = 320;
const UI_MARGIN = 10;

// Layout bands, measured in element pixels.
const PAD = 4;
const HEADER_HEIGHT = 12;
const FOOTER_HEIGHT = 13;
const FOOTER_LINES = 1;
const CAPTION_HEIGHT = 9;
const CELL_GAP = 3;

// The gap kept between the two halves of a footer line, and between a glyph and the words after
// it.
const FOOTER_GAP = 8;
const GLYPH_GAP = 4;

// The smallest tile worth drawing. Below this a swatch carries no shape and no grade is
// readable off it, so the sheet says it has no room rather than drawing 38 specks.
const MIN_TILE = 14;

// The longest edge, in tile pixels, that a cell's picture is reduced to. Past this the tiles
// are sampled at this size and the canvas draws them up to the cell, which the glyph says.
//
// The size this bounds is worked out without the graph's zoom in it, so which side of the bound
// a node sits on is a property of the node and not of how closely somebody is looking at it.
const MAX_TILE = 128;

// The shapes the grid can hold a picture to. A panorama or a column would otherwise give every
// tile one row of pixels.
const MIN_ASPECT = 0.6;
const MAX_ASPECT = 2.2;

const BODY_FONT = "10px sans-serif";
const CAPTION_FONT = "8px sans-serif";
const LABEL_FONT = "11px sans-serif";

const MESSAGE_TIMEOUT = 4000;

// How long to wait before asking for a picture again while the answer is not the picture. A
// node queued before the socket opened publishes on its next run, so an answer of `waiting` is
// never the last word.
const RETRY_INTERVAL = 3000;

const ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

// The cell that shows the picture with nothing done to it. It is not a style, so it is drawn
// first, captioned as itself and never written to the widget.
const SOURCE_CELL = "source";

// Proper nouns: `brannan`, `perpetua` and `xpro2` say nothing about what they do to a picture.
/** The style menu, in alphabetical order. */
const STYLES = [
  "1977",
  "aden",
  "bleach bypass",
  "brannan",
  "brooklyn",
  "clarendon",
  "clean punch",
  "cross process",
  "earlybird",
  "faded film",
  "fairy tale",
  "film noir",
  "gingham",
  "golden hour",
  "hudson",
  "inkwell",
  "kelvin",
  "lark",
  "lofi",
  "maven",
  "mayfair",
  "moody blue",
  "moon",
  "nashville",
  "neon night",
  "perpetua",
  "reyes",
  "rise",
  "slumber",
  "soft portrait",
  "stinson",
  "teal and orange",
  "toaster",
  "valencia",
  "walden",
  "willow",
  "xpro2",
];

// This one is not a colour grade at all. It draws 10000 unseeded random dots over a bloom, so
// the node itself gives a different result every run and no preview can be true.
/** The style drawn as random glitter rather than as a colour grade. */
const SPARKLE_STYLE = "fairy tale";

/** How truly a tile stands for what the node will render. */
const FIDELITY = {
  // A chain of per pixel operations with no term that depends on the image's size, so the tile
  // is the node's own arithmetic on the colours under it.
  FLAT: "flat",
  // A gradient mask built from the image's size, which is a function of the shape alone, so the
  // tile is exact for a picture of the tile's shape.
  SHAPED: "shaped",
  // Drawn from an unseeded generator inside the node, so no preview of it can be true.
  RANDOM: "random",
};

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

/** Which styles bleed their highlights into the pixels around them. */
const GLOW_STYLES = new Set([
  "1977",
  "aden",
  "bleach bypass",
  "brannan",
  "brooklyn",
  "clarendon",
  "clean punch",
  "cross process",
  "earlybird",
  "faded film",
  "film noir",
  "gingham",
  "golden hour",
  "hudson",
  "inkwell",
  "kelvin",
  "lark",
  "lofi",
  "maven",
  "mayfair",
  "moody blue",
  "moon",
  "nashville",
  "neon night",
  "perpetua",
  "reyes",
  "rise",
  "slumber",
  "soft portrait",
  "stinson",
  "teal and orange",
  "toaster",
  "valencia",
  "walden",
  "willow",
  "xpro2",
]);

/**
 * Whether one style bleeds its highlights.
 *
 * @param {string} style - A style name.
 * @returns {boolean} True while the style carries a halation.
 */
function hasGlow(style) {
  return GLOW_STYLES.has(style);
}

/** Which styles carry a mask built from the image's own size. */
const SHAPED_STYLES = new Set([
  "aden",
  "brooklyn",
  "earlybird",
  "film noir",
  "golden hour",
  "hudson",
  "lofi",
  "mayfair",
  "perpetua",
  "rise",
  "soft portrait",
  "toaster",
  "willow",
  "xpro2",
]);

/**
 * How truly one style's tile stands for what the node will render.
 *
 * @param {string} style - A style name.
 * @returns {string} One of `FIDELITY`.
 */
function fidelityOf(style) {
  if (style === SPARKLE_STYLE) return FIDELITY.RANDOM;
  return SHAPED_STYLES.has(style) || hasGlow(style) ? FIDELITY.SHAPED : FIDELITY.FLAT;
}

/**
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * Test whether one of a node's inputs is filled by a link.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  // `style` is a combo the schema also offers as a socket, and the prompt the frontend builds
  // writes the link into the input in place of the widget's value, so a linked socket is what
  // the run reads and the widget beside it is read by nothing. A sheet that rings a tile and
  // calls it exact would then be naming a style the run never sees.
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  for (const input of inputs) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

/**
 * Work out where the header, the grid and the footer sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the header, the grid area and the footer.
 */
function computeLayout(width, height) {
  const footerY = Math.max(0, height - PAD - FOOTER_HEIGHT * FOOTER_LINES);
  const areaX0 = PAD;
  const areaX1 = Math.max(areaX0 + 1, width - PAD);
  const areaY0 = PAD + HEADER_HEIGHT;
  const areaY1 = Math.max(areaY0, footerY - 2);

  return {
    width,
    height,
    areaX0,
    areaX1,
    areaY0,
    areaY1,
    areaWidth: areaX1 - areaX0,
    areaHeight: areaY1 - areaY0,
    headerY: PAD,
    footerY,
  };
}

/**
 * Choose the grid that draws the largest tile.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} count - How many cells there are.
 * @param {number} aspect - The shape a tile is cut to.
 * @returns {object|null} The grid, or null when no column count leaves a readable tile.
 */
function computeGrid(layout, count, aspect) {
  let best = null;
  for (let columns = 1; columns <= count; columns++) {
    const rows = Math.ceil(count / columns);
    const cellWidth = (layout.areaWidth - CELL_GAP * (columns - 1)) / columns;
    const cellHeight = (layout.areaHeight - CELL_GAP * (rows - 1)) / rows;
    const tileHeight = cellHeight - CAPTION_HEIGHT;
    if (cellWidth < MIN_TILE || tileHeight < MIN_TILE) continue;

    const width = Math.min(cellWidth, tileHeight * aspect);
    const height = width / aspect;
    if (width < MIN_TILE || height < MIN_TILE) continue;

    const area = width * height;
    // A wider grid wins ties, since more of the set is visible in one row of the eye.
    if (!best || area > best.area + 0.5) {
      best = { columns, rows, cellWidth, cellHeight, tileWidth: width, tileHeight: height, area };
    }
  }
  return best;
}

/**
 * Where a cell's tile lands on the surface, in whole device pixels.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - The tile in element pixels.
 * @param {number} ratio - Device pixels per element pixel.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle at that ratio.
 */
function deviceRect(rect, ratio) {
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
 * Build the surface for one node.
 *
 * @param {object} options - What is drawn, and what it writes.
 * @param {object} options.node - The node the sheet is drawn on.
 * @param {{load: () => Promise<object>}} options.backdrop - What the styles are applied to.
 * @param {() => string} options.read - Answers the style the node's widget holds.
 * @param {(style: string) => void} options.write - Writes one style into that widget.
 * @param {() => boolean} [options.linked] - Answers whether the run reads the style off a link
 *   rather than off the widget, in which case the sheet says so and stops ringing a tile in the
 *   accent colour, since a chosen tile the run ignores is the same untruth drawn twice.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {() => boolean} [options.active] - Answers whether the sheet is the panel the node is
 *   drawing. While it answers false the sheet asks for no picture and arms no retry, and
 *   `refresh` is the call that starts it again.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   refresh: () => void, dispose: () => void}} The element to hand to `addDOMWidget`, the
 *   height it was built for, a coalesced repaint, a fresh ask for the backdrop, and teardown.
 */
function createStyleSheet(options) {
  const settings = {
    node: options.node ?? null,
    backdrop: options.backdrop,
    read: options.read,
    write: options.write,
    linked: typeof options.linked === "function" ? options.linked : null,
    active: typeof options.active === "function" ? options.active : null,
    height: Math.max(UI_MARGIN * 2 + MIN_TILE, toNumber(options.height, UI_HEIGHT)),
  };

  const cells = [SOURCE_CELL, ...STYLES];

  const root = document.createElement("div");
  root.tabIndex = 0;
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${settings.height - UI_MARGIN * 2}px`,
    "overflow:hidden",
    "outline:none",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The footer's glyph states its sentence through the element's own title. The regions are
  // handed over again on every repaint, since a glyph moves whenever the node is resized.
  const titles = hoverTitles(root);

  let tileScratch = null;

  const state = {
    frame: cardFrame(normaliseFrame({ state: PREVIEW_STATE.LOADING })),
    loading: false,
    reloadWanted: false,
    retryTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    grid: null,
    tiles: null,
    tilesKey: "",
    // One drawable per style, as the node rendered them. The sheet draws what the node makes
    // rather than a second opinion of it.
    styled: null,
    styledToken: -1,
    styledLoading: false,
    tileSize: null,
    failed: new Set(),
    token: 0,
    hover: -1,
    pending: "",
    pressed: -1,
    message: "",
    messageTimer: 0,
    disposed: false,
  };

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
      width: TEST_CARD.width,
      height: TEST_CARD.height,
      scale: 1,
    };
  }

  /**
   * Show a short note in the footer.
   *
   * @param {string} text - Note to show.
   * @returns {void}
   */
  function setMessage(text) {
    state.message = text;
    if (state.messageTimer) clearTimeout(state.messageTimer);
    state.messageTimer = setTimeout(() => {
      state.messageTimer = 0;
      state.message = "";
      schedulePaint();
    }, MESSAGE_TIMEOUT);
    schedulePaint();
  }

  /**
   * Whether two answers stand for the same picture in the same state.
   *
   * @param {object} a - One frame.
   * @param {object} b - The other.
   * @returns {boolean} True while nothing the sheet draws differs between them.
   */
  function sameFrame(a, b) {
    return (
      a.state === b.state
      && a.kind === b.kind
      && a.image === b.image
      && a.width === b.width
      && a.height === b.height
      && a.label === b.label
    );
  }

  /**
   * Ask the node for the styled picture behind every cell.
   *
   * @returns {void}
   */
  function loadStyled() {
    // Frame 0 is the source and one frame per style follows it, in menu order.
    if (state.disposed || state.styledLoading) return;
    const token = state.token;
    if (state.styledToken === token) return;
    state.styledLoading = true;

    Promise.all(
      STYLES.map((_, index) =>
        fetchInputPreview(settings.node, "", index + 1)
          .then((answer) => normaliseFrame(answer)?.image ?? null)
          .catch(() => null)
      )
    )
      .then((images) => {
        if (state.disposed || state.token !== token) return;
        state.styled = images;
        state.styledToken = token;
        state.tiles = null;
        state.tilesKey = "";
        schedulePaint();
      })
      .finally(() => {
        state.styledLoading = false;
      });
  }

  /**
   * Whether the sheet is the panel the node is drawing.
   *
   * @returns {boolean} True while no caller said otherwise, or the caller's answer.
   */
  function drawn() {
    if (!settings.active) return true;
    try {
      return settings.active() !== false;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read which panel the node draws:`, error);
      return true;
    }
  }

  /**
   * Ask the backdrop for the picture, and again later while the answer is not one.
   *
   * @returns {void}
   */
  function loadFrame() {
    if (state.disposed || !drawn()) return;
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
        changed = !sameFrame(state.frame, frame);
        if (!changed) return;
        state.frame = frame;
        // The tiles hold pixels sampled from the picture that has just been replaced.
        state.token += 1;
        state.tiles = null;
        state.tilesKey = "";
        state.styled = null;
        state.styledToken = -1;
      })
      .catch((error) => {
        if (state.disposed) return;
        console.error(`[${EXT_NAME}] Failed to read the backdrop:`, error);
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
        // The node publishes its own styled tiles beside the picture, so they are asked for
        // once the picture is the answer.
        if (state.frame.state === PREVIEW_STATE.READY) loadStyled();
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
    if (state.disposed || !drawn() || state.frame.state === PREVIEW_STATE.READY) return;
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
   * The style the sheet is showing as chosen, which is the one an unfinished keyboard gesture
   * holds while there is one.
   *
   * @returns {string} A style name, empty when the widget holds nothing readable.
   */
  function currentStyle() {
    if (state.pending) return state.pending;
    try {
      const value = settings.read?.();
      return typeof value === "string" ? value : "";
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the style:`, error);
      return "";
    }
  }

  /**
   * Whether the run reads the style off a link rather than off the widget the sheet writes.
   *
   * @returns {boolean} True while the socket is connected.
   */
  function styleLinked() {
    try {
      return settings.linked?.() === true;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read whether the style is linked:`, error);
      return false;
    }
  }

  /**
   * The shape a tile is cut to, which is the picture's own within what the grid can hold.
   *
   * @returns {{aspect: number, held: boolean}} The shape, and whether the picture's own was
   *   too far outside the range to use.
   */
  function tileAspect() {
    const frame = state.frame;
    const wanted = frame.width > 0 && frame.height > 0 ? frame.width / frame.height : 1.5;
    const held = clamp(wanted, MIN_ASPECT, MAX_ASPECT);
    return { aspect: held, held: Math.abs(held - wanted) > 0.001 };
  }

  /**
   * Draw the backdrop into a buffer of the tile's size.
   *
   * @param {number} width - Tile width in tile pixels.
   * @param {number} height - Tile height in tile pixels.
   * @returns {object|null} The buffer, or null when the picture could not be read.
   */
  function sampleBackdrop(width, height) {
    try {
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      const ctx = scratch.getContext("2d", { willReadFrequently: true });
      if (!ctx) return null;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";

      // The buffer is a reduction of the picture, so its pixels are averages of the picture's.
      // It is the source cell's tile, and the tile a style with no published picture falls back
      // to under the hatch.
      //
      // The card is drawn whole and reduced, rather than cropped at one to one. Nothing
      // in this node is measured in pixels: every style is a colour transform, so a tile
      // carrying all nine bands of the card says more than a tile carrying the left edge of two
      // of them.
      if (state.frame.kind === BACKDROP_KIND.CARD) {
        const card = createTestCard(TEST_CARD.width, TEST_CARD.height);
        if (!card) return null;
        const source = document.createElement("canvas");
        source.width = TEST_CARD.width;
        source.height = TEST_CARD.height;
        const sourceCtx = source.getContext("2d");
        if (!sourceCtx) return null;
        sourceCtx.putImageData(card, 0, 0);
        ctx.drawImage(source, 0, 0, width, height);
      } else {
        ctx.drawImage(state.frame.image, 0, 0, width, height);
      }
      const pixels = ctx.getImageData(0, 0, width, height);
      // The node converts what it receives to RGB before any style touches it, and a canvas
      // hands back whatever alpha the picture carried, so the tile is made opaque here rather
      // than drawn through.
      for (let i = 3; i < pixels.data.length; i += 4) pixels.data[i] = 255;
      return pixels;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the picture:`, error);
      return null;
    }
  }

  /**
   * Reduce one styled picture to a tile.
   *
   * @param {CanvasImageSource} drawn - The picture the node made of this style.
   * @param {number} width - Tile width in tile pixels.
   * @param {number} height - Tile height in tile pixels.
   * @returns {object|null} The buffer, or null where it could not be read.
   */
  function sampleStyled(drawn, width, height) {
    try {
      const scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      const ctx = scratch.getContext("2d", { willReadFrequently: true });
      if (!ctx) return null;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(drawn, 0, 0, width, height);
      const pixels = ctx.getImageData(0, 0, width, height);
      for (let i = 3; i < pixels.data.length; i += 4) pixels.data[i] = 255;
      return pixels;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read a styled tile:`, error);
      return null;
    }
  }

  /**
   * The size the tiles are computed at, which is the size they are drawn at until that costs
   * too much.
   *
   * @param {number} width - Tile width in tile pixels.
   * @param {number} height - Tile height in tile pixels.
   * @returns {{width: number, height: number, capped: boolean}} The size to compute at.
   */
  function tileBudget(width, height) {
    // Every cell is its own reduction of a picture and there are 38 of them, so the work grows
    // with the tile's area. Past this bound the tiles are sampled at the bound and the canvas
    // draws them larger, which is the reduction the footer already declares.
    const longest = Math.max(width, height);
    if (longest <= MAX_TILE) return { width, height, capped: false };
    const factor = MAX_TILE / longest;
    return {
      width: Math.max(1, Math.round(width * factor)),
      height: Math.max(1, Math.round(height * factor)),
      capped: true,
    };
  }

  /**
   * Build every tile, held between repaints.
   *
   * @param {number} width - Tile width in tile pixels.
   * @param {number} height - Tile height in tile pixels.
   * @returns {object[]|null} One entry per cell, or null when there is nothing to sample.
   */
  function ensureTiles(width, height) {
    const key = `${state.frame.kind}|${state.token}|${state.styledToken}|${width}x${height}`;
    if (state.tiles && state.tilesKey === key) return state.tiles;
    if (!(width > 0) || !(height > 0)) return null;

    const source = sampleBackdrop(width, height);
    if (!source) return null;

    const failed = new Set();
    const tiles = [];
    for (const cell of cells) {
      if (cell === SOURCE_CELL || cell === SPARKLE_STYLE) {
        tiles.push({ name: cell, image: source, previewed: cell === SOURCE_CELL });
        continue;
      }
      const drawn = state.styled?.[STYLES.indexOf(cell)] ?? null;
      let image = source;
      let previewed = true;
      if (drawn) {
        const sampled = sampleStyled(drawn, width, height);
        if (sampled) {
          image = sampled;
        } else {
          previewed = false;
          failed.add(cell);
        }
      } else {
        // The picture with nothing done to it is what is left, and drawn plainly it reads as a
        // style that happens to change nothing. It carries the hatch instead, and the footer
        // says so rather than calling it exact.
        previewed = false;
        failed.add(cell);
      }
      tiles.push({ name: cell, image, previewed });
    }

    state.failed = failed;
    state.tiles = tiles;
    state.tilesKey = key;
    return tiles;
  }

  /**
   * Where one cell sits, in element pixels.
   *
   * @param {number} index - Which cell.
   * @returns {{x: number, y: number, w: number, h: number, tile: object}|null} The cell and the
   *   tile inside it, or null while there is no grid.
   */
  function cellRect(index) {
    const grid = state.grid;
    if (!grid) return null;
    const column = index % grid.columns;
    const row = Math.floor(index / grid.columns);
    if (row >= grid.rows) return null;

    const x = state.layout.areaX0 + column * (grid.cellWidth + CELL_GAP);
    const y = state.layout.areaY0 + row * (grid.cellHeight + CELL_GAP);
    return {
      x,
      y,
      w: grid.cellWidth,
      h: grid.cellHeight,
      tile: {
        x: x + (grid.cellWidth - grid.tileWidth) / 2,
        y,
        w: grid.tileWidth,
        h: grid.tileHeight,
      },
    };
  }

  /**
   * Find the cell under a point.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {number} The cell's index, or -1 when the point is on none.
   */
  function hitTest(point) {
    const grid = state.grid;
    if (!grid) return -1;
    for (let index = 0; index < cells.length; index++) {
      const rect = cellRect(index);
      if (!rect) continue;
      if (
        point.x >= rect.x &&
        point.x <= rect.x + rect.w &&
        point.y >= rect.y &&
        point.y <= rect.y + rect.h
      ) {
        return index;
      }
    }
    return -1;
  }

  /**
   * Read the pointer position in element pixels.
   *
   * @param {PointerEvent|MouseEvent} event - Event to read.
   * @returns {{x: number, y: number}} Position inside the element.
   */
  function localPoint(event) {
    return elementPoint(root, event);
  }

  /**
   * Write one style into the node's widget, with its own undo entry.
   *
   * @param {string} style - The style to write.
   * @returns {void}
   */
  function commit(style) {
    // Every path out of here has just dropped the held choice, and the ring and the header are
    // drawn from it, so every path repaints. A click on the tile the widget already holds
    // writes nothing and still has to put the sheet back on the widget's own style.
    const pending = state.pending;
    state.pending = "";
    if (state.disposed) return;
    if (!style) {
      if (pending) schedulePaint();
      return;
    }
    if (style === SOURCE_CELL) {
      setMessage("the picture as it arrived");
      return;
    }
    if (style === currentStyle()) {
      if (pending) schedulePaint();
      return;
    }

    const canvas = app.canvas;
    const transactional =
      typeof canvas?.emitBeforeChange === "function" &&
      typeof canvas?.emitAfterChange === "function";

    if (transactional) canvas.emitBeforeChange();
    try {
      settings.write?.(style);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to write the style:`, error);
    } finally {
      if (transactional) canvas.emitAfterChange();
    }
    settings.node?.setDirtyCanvas?.(true, true);
    schedulePaint();
  }

  /**
   * Move the chosen style by one cell of the grid, without writing it.
   *
   * @param {number} dx - Cells across.
   * @param {number} dy - Cells down.
   * @returns {void}
   */
  function nudge(dx, dy) {
    const grid = state.grid;
    if (!grid) {
      setMessage("no room to draw the sheet");
      return;
    }
    const style = currentStyle();
    const index = Math.max(0, cells.indexOf(style));
    const next = clamp(index + dx + dy * grid.columns, 1, cells.length - 1);
    if (cells[next] === state.pending) return;
    state.pending = cells[next];
    schedulePaint();
  }

  /**
   * The canvas a tile held to the cost bound is drawn up to the cell through.
   *
   * @param {number} width - Width in tile pixels.
   * @param {number} height - Height in tile pixels.
   * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D}|null} The canvas, or
   *   null where a context could not be had.
   */
  function ensureScratch(width, height) {
    try {
      if (!tileScratch) {
        const element = document.createElement("canvas");
        const context = element.getContext("2d");
        if (!context) return null;
        tileScratch = { canvas: element, ctx: context };
      }
      if (tileScratch.canvas.width !== width) tileScratch.canvas.width = width;
      if (tileScratch.canvas.height !== height) tileScratch.canvas.height = height;
      return tileScratch;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to make the scratch canvas:`, error);
      return null;
    }
  }

  /**
   * Draw one tile's picture.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} tile - Entry from `ensureTiles`.
   * @param {{x: number, y: number, w: number, h: number}} device - Where the tile goes, in
   *   device pixels.
   * @param {number} ratio - Device pixels per element pixel.
   * @param {boolean} written - Whether the tile can be written through rather than drawn.
   * @returns {void}
   */
  function drawTile(ctx, tile, device, ratio, written) {
    try {
      // A tile the surface is drawing at the tile's own resolution is written through as pixels,
      // so nothing resamples it a second time.
      if (written) {
        ctx.putImageData(tile.image, device.x, device.y);
        return;
      }
      // Anywhere else the canvas draws it across the cell, which is every zoom past one as well
      // as the tile held to the cost bound. Smoothing is left on for that, which is the
      // context's own setting: a tile is a colour grade and the eye reads it as one, so a tone
      // between two of the tile's own is a truer answer for the pixel it lands on than the
      // nearer of the two repeated across a block.
      const holder = ensureScratch(tile.image.width, tile.image.height);
      if (!holder) return;
      holder.ctx.putImageData(tile.image, 0, 0);
      // `drawImage` is under the transform and `device` is not, so the destination is stated in
      // the element pixels that land back on those whole device pixels.
      ctx.drawImage(
        holder.canvas,
        device.x / ratio,
        device.y / ratio,
        device.w / ratio,
        device.h / ratio,
      );
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to draw a tile:`, error);
    }
  }

  /**
   * Lay a hatch over a tile that stands for nothing the node will draw twice.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} rect - The tile in element pixels.
   * @returns {void}
   */
  function drawHatch(ctx, theme, rect) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = theme.bg;
    ctx.lineWidth = 2;
    for (let offset = -rect.h; offset < rect.w; offset += 6) {
      ctx.beginPath();
      ctx.moveTo(rect.x + offset, rect.y + rect.h);
      ctx.lineTo(rect.x + offset + rect.h, rect.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * Draw the words for a sheet with no tiles in it, where the grid would go.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {string} text - What to say.
   * @param {boolean} [warns] - Whether what it says is a fault rather than a state.
   * @returns {void}
   */
  function drawNotice(ctx, theme, text, warns = false) {
    const layout = state.layout;
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = warns ? theme.warning : theme.fgMuted;
    ctx.fillText(
      text,
      layout.areaX0 + layout.areaWidth / 2,
      layout.areaY0 + layout.areaHeight / 2,
      Math.max(1, layout.areaWidth - 8),
    );
  }

  /**
   * How truly the tile on screen stands for the render, ignoring what the run reads.
   *
   * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
   */
  function tileClaim() {
    // The claim follows the chosen style: a grade carries a glow drawn for the picture's size,
    // and several a vignette drawn for its shape as well, while one is drawn at random inside
    // the node and cannot be previewed at all.
    const style = currentStyle();
    const shape = tileAspect();
    // A tile held to the cost bound is drawn larger than it was worked out, which the glyph
    // carries as well, since the size it was computed at is the whole of what makes that tile
    // softer than the render.
    const capped = state.tileSize?.capped
      ? `, worked out at ${state.tileSize.width}x${state.tileSize.height} and drawn larger`
      : "";

    if (style === SPARKLE_STYLE) {
      return {
        icon: ICON.WARNING,
        detail: "fairy tale is random glitter, so no tile can show it",
      };
    }
    if (!STYLES.includes(style)) {
      return { icon: ICON.WARNING, detail: "no tile for this style" };
    }
    if (state.failed.has(style)) {
      return {
        icon: ICON.WARNING,
        detail: `${style} could not be drawn, so its tile is the picture`,
      };
    }
    if (fidelityOf(style) === FIDELITY.SHAPED) {
      const traits = [];
      if (SHAPED_STYLES.has(style)) traits.push("fades from the middle");
      if (hasGlow(style)) traits.push("bleeds its highlights");
      const what = traits.join(" and ");
      if (shape.held) {
        return {
          icon: ICON.WARNING,
          detail: `${style} ${what}, drawn at the tile's shape${capped}`,
        };
      }
      // The vignette is a function of the picture's shape, so a tile drawn over the test card is
      // exact for the card and for nothing the node will be given. It is close rather than exact
      // until there is a picture under it.
      const card = state.frame.kind === BACKDROP_KIND.CARD;
      const own = card ? "the card's" : "the picture's";
      // A glow reaches across pixels rather than standing on one, so a tile of it is drawn at
      // the tile's own size and is close to the render rather than equal to it.
      if (hasGlow(style)) {
        return {
          icon: ICON.APPROXIMATE,
          detail: `${style} ${what}, drawn at the tile's size${capped}`,
        };
      }
      return {
        icon: capped || card ? ICON.APPROXIMATE : ICON.EXACT,
        detail: `${style} ${what}, exact at ${own} shape${capped}`,
      };
    }
    return {
      icon: capped ? ICON.APPROXIMATE : ICON.EXACT,
      detail: `${style} is the node's own bytes${capped}`,
    };
  }

  /**
   * The claim for the glyph, with what the run reads leading it where that differs.
   *
   * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
   */
  function fidelityClaim() {
    const claim = tileClaim();
    // That the widget is not the value the run reads is drawn beside the style's own name as
    // well, since it is the one of these somebody acts on.
    if (!styleLinked()) return claim;
    return {
      icon: ICON.WARNING,
      detail: "the run reads the linked value, so no tile stands for what it will draw."
        + ` ${claim.detail}`,
    };
  }

  /**
   * Draw the footer line.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {boolean} drew - Whether a tile was drawn. Where none was, there is nothing for a
   *   fidelity glyph to make a claim about and none is drawn.
   * @returns {void}
   */
  function drawFooter(ctx, theme, drew) {
    const layout = state.layout;
    const middle = layout.footerY + FOOTER_HEIGHT / 2;
    const note = state.message || LABELS[state.frame.state] || "";
    const warns = state.frame.state === PREVIEW_STATE.FAILED;

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    let glyphWidth = 0;
    if (drew) {
      const claim = fidelityClaim();
      const box = drawIcon(
        ctx,
        claim.icon,
        layout.areaX0,
        middle - ICON_SIZE / 2,
        ICON_SIZE,
        claim.icon === ICON.WARNING ? theme.warning : theme.fgMuted,
      );
      titles.set([{ ...box, title: iconTitle(claim.icon, claim.detail) }]);
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    } else {
      titles.set([]);
    }

    // The note is given the room it needs first. A `maxWidth` handed to `fillText` condenses the
    // glyphs sideways rather than shortening the line, so a state somebody has to act on would be
    // drawn as a smear on a narrow node while the size of the picture kept its room.
    const shownNote = elideText(ctx, note, layout.areaWidth - glyphWidth);
    let noteWidth = 0;
    if (shownNote) {
      noteWidth = ctx.measureText(shownNote).width + FOOTER_GAP;
      ctx.textAlign = "right";
      ctx.fillStyle = warns ? theme.warning : theme.fgMuted;
      ctx.fillText(shownNote, layout.areaX1, middle);
    }

    const frame = state.frame;
    const source = elideText(
      ctx,
      frame.kind === BACKDROP_KIND.CARD
        ? `test card ${TEST_CARD.width}x${TEST_CARD.height}`
        : `source ${Math.round(frame.width)}x${Math.round(frame.height)}`,
      layout.areaWidth - glyphWidth - noteWidth,
    );
    if (source) {
      ctx.fillStyle = theme.fgMuted;
      ctx.textAlign = "left";
      ctx.fillText(source, layout.areaX0 + glyphWidth, middle);
    }
  }

  /**
   * Draw the whole sheet.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
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
    const shape = tileAspect();
    state.grid = computeGrid(layout, cells.length, shape.aspect);

    const chosen = currentStyle();
    const hovered = state.hover >= 0 ? cells[state.hover] : "";
    const linked = styleLinked();

    ctx.font = BODY_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const headerMiddle = layout.headerY + HEADER_HEIGHT / 2;
    ctx.fillStyle = theme.fg;
    ctx.fillText(hovered || chosen || "no style", layout.areaX0, headerMiddle, layout.areaWidth);
    ctx.textAlign = "right";
    ctx.fillStyle = linked ? theme.warning : theme.fgMuted;
    // Only the two readings that change: the widget the run does not read, and the one cell that
    // is not a style at all. That the name on the left is the chosen style is what the accent
    // ring around its tile already says, so it is not written out beside it. A tile under the
    // pointer that is neither of those is named and nothing more: a tile that lights up under
    // the pointer is a tile that can be clicked.
    const headerNote = linked
      ? "the widget, which the run does not read"
      : hovered === SOURCE_CELL
        ? "the picture as it arrived"
        : "";
    if (headerNote) {
      ctx.fillText(headerNote, layout.areaX1, headerMiddle, Math.max(1, layout.areaWidth / 2));
    }

    if (!state.grid) {
      // Nothing is drawn, so the footer has no tile size to name.
      state.tileSize = null;
      drawNotice(ctx, theme, "make the node taller to see the styles");
      drawFooter(ctx, theme, false);
      return;
    }

    // Two resolutions, and the whole of what keeps a swatch standing for the render. The tiles
    // are worked out at one the graph's zoom cannot reach, so the same picture gives the same 36
    // grades however closely somebody is looking: every tile pixel averages the same part of the
    // picture, the cost bound affords the same tiles, and the size the footer names for a capped
    // tile is a fact about the node rather than about the viewport. They land on the surface at
    // the other, which is where the zoom belongs.
    const content = contentRatio(root);
    const tileWidth = Math.max(1, Math.round(state.grid.tileWidth * content));
    const tileHeight = Math.max(1, Math.round(state.grid.tileHeight * content));
    const budget = tileBudget(tileWidth, tileHeight);
    state.tileSize = budget;
    const tiles = ensureTiles(budget.width, budget.height);

    if (!tiles) {
      // The card stands in for every state that has no picture of its own, so a sheet with no
      // tiles is a buffer or a context the browser would not give rather than a picture that has
      // not arrived, and running the node would not change it.
      state.tileSize = null;
      drawNotice(ctx, theme, "the tiles could not be built", true);
      drawFooter(ctx, theme, false);
      return;
    }

    // A tile is already the size the surface wants only where the surface draws at the tile's
    // own resolution. There every cell is written through byte for byte, and anywhere else the
    // canvas draws each of them across its cell instead.
    const written = !budget.capped && ratio === content;

    for (let index = 0; index < cells.length; index++) {
      const rect = cellRect(index);
      if (!rect) continue;
      const tile = tiles[index];
      const name = cells[index];
      const isChosen = name === chosen;
      const isHovered = index === state.hover;

      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(rect.tile.x, rect.tile.y, rect.tile.w, rect.tile.h);
      drawTile(ctx, tile, deviceRect(rect.tile, ratio), ratio, written);
      // The one tile that stands for nothing the node will draw twice is hatched over, so it
      // does not read as a style that happens to change nothing.
      if (!tile.previewed) drawHatch(ctx, theme, rect.tile);

      ctx.lineWidth = isChosen ? 2 : 1;
      // A ring in the accent colour says the run will use this tile. While the socket is
      // connected it will not, so the chosen tile is marked in the muted colour instead and the
      // sheet stops making a claim the prompt contradicts.
      const chosenColour = linked ? theme.fgMuted : theme.accent;
      ctx.strokeStyle = isChosen ? chosenColour : isHovered ? theme.fg : theme.border;
      ctx.strokeRect(
        Math.round(rect.tile.x) + 0.5,
        Math.round(rect.tile.y) + 0.5,
        Math.max(1, Math.round(rect.tile.w) - 1),
        Math.max(1, Math.round(rect.tile.h) - 1),
      );

      ctx.font = CAPTION_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = isChosen
        ? chosenColour
        : name === SPARKLE_STYLE
          ? theme.warning
          : isHovered
            ? theme.fg
            : theme.fgMuted;
      ctx.fillText(
        name,
        rect.x + rect.w / 2,
        rect.y + rect.h - CAPTION_HEIGHT / 2,
        Math.max(1, rect.w),
      );
    }

    drawFooter(ctx, theme, true);

    if (document.activeElement === root) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = theme.accent;
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    }
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
        console.error(`[${EXT_NAME}] Failed to draw the style sheet:`, error);
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
        console.error(`[${EXT_NAME}] Style sheet input failed:`, error);
      }
    };
  }

  root.addEventListener(
    "pointerdown",
    guard((event) => {
      // Middle button panning belongs to the canvas underneath.
      if (event.button === 1) {
        app.canvas?.processMouseDown?.(event);
        return;
      }
      if (event.button !== 0) return;
      state.pressed = hitTest(localPoint(event));
      root.setPointerCapture?.(event.pointerId);
      schedulePaint();
    }),
  );

  root.addEventListener(
    "pointermove",
    guard((event) => {
      if (event.buttons & 4) {
        app.canvas?.processMouseMove?.(event);
        return;
      }
      // A press whose button was released elsewhere delivers no pointerup here.
      if (state.pressed >= 0 && !(event.buttons & 1)) state.pressed = -1;
      const index = hitTest(localPoint(event));
      if (index === state.hover) return;
      state.hover = index;
      schedulePaint();
    }),
  );

  root.addEventListener(
    "pointerup",
    guard((event) => {
      if (event.button === 1) {
        app.canvas?.processMouseUp?.(event);
        return;
      }
      const pressed = state.pressed;
      state.pressed = -1;
      if (root.hasPointerCapture?.(event.pointerId)) root.releasePointerCapture?.(event.pointerId);
      if (pressed < 0) return;
      // The gesture is one press and one release on the same cell, so a press that slid off
      // writes nothing.
      const index = hitTest(localPoint(event));
      if (index !== pressed) {
        schedulePaint();
        return;
      }
      commit(cells[index]);
    }),
  );

  root.addEventListener(
    "pointercancel",
    guard(() => {
      state.pressed = -1;
      schedulePaint();
    }),
  );

  root.addEventListener(
    "lostpointercapture",
    guard(() => {
      state.pressed = -1;
      schedulePaint();
    }),
  );

  root.addEventListener(
    "pointerleave",
    guard(() => {
      if (state.hover === -1) return;
      state.hover = -1;
      schedulePaint();
    }),
  );

  root.addEventListener(
    "contextmenu",
    guard((event) => {
      // The graph canvas suppresses its own context menu on its own element, and this is a
      // separate element, so the browser menu would otherwise open over the node.
      event.preventDefault();
      event.stopPropagation();
    }),
  );

  // The sheet scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);

  root.addEventListener(
    "keydown",
    guard((event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      let handled = true;
      switch (event.key) {
        case "ArrowLeft":
          nudge(-1, 0);
          break;
        case "ArrowRight":
          nudge(1, 0);
          break;
        case "ArrowUp":
          nudge(0, -1);
          break;
        case "ArrowDown":
          nudge(0, 1);
          break;
        case "Enter":
        case " ":
          commit(state.pending || currentStyle());
          break;
        case "Escape":
          state.pending = "";
          schedulePaint();
          break;
        case "Delete":
        case "Backspace":
          // Consumed whether or not it has anything to do. Left unhandled these reach ComfyUI's
          // own binding, which deletes the node the sheet is drawn on.
          setMessage("nothing to delete here");
          break;
        default:
          handled = false;
      }
      if (handled) {
        event.preventDefault();
        event.stopPropagation();
      }
    }),
  );

  root.addEventListener(
    "keyup",
    guard((event) => {
      if (!ARROW_KEYS.has(event.key)) return;
      // One write per gesture: the arrows move a held choice and the release commits it.
      if (state.pending) commit(state.pending);
    }),
  );

  root.addEventListener("focus", guard(schedulePaint));
  root.addEventListener(
    "blur",
    guard(() => {
      state.pending = "";
      state.pressed = -1;
      schedulePaint();
    }),
  );

  let observer = null;
  if (typeof ResizeObserver === "function") {
    // The tiles are keyed by the size they were built at, so a resize that leaves that size
    // alone keeps them. Dropping them here instead would resample all 38 cells on every frame
    // of a drag, including the frames where the rounded tile does not move at all.
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here. The two answer different events: the observer answers a
  // node that was resized or collapsed, this answers the same box drawn at another size.
  let unwatchRatio = watchSurfaceRatio(root, schedulePaint);

  // The panel is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the timers, observers and buffers the sheet holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    if (state.retryTimer) clearTimeout(state.retryTimer);
    if (state.messageTimer) clearTimeout(state.messageTimer);
    state.paintHandle = 0;
    state.retryTimer = 0;
    state.messageTimer = 0;
    state.tiles = null;
    state.tilesKey = "";
    tileScratch = null;
    titles.dispose();
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
 * Ask for the picture again as soon as the node finishes, or when a run ends without it.
 *
 * @param {object} node - The node the picture is published for.
 * @param {{refresh: () => void}} sheet - Sheet from `createStyleSheet`.
 * @returns {() => void} Unhooks the listener.
 */
function watchRuns(node, sheet) {
  return onNodeDone(node, () => {
    try {
      sheet.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to ask for the image again:`, error);
    }
  });
}

/**
 * Read which of the node's two panels it is set to draw.
 *
 * @param {object} node - The node the panels are on.
 * @returns {boolean} True for the contact sheet, false for the before and after. True as well
 *   for a node carrying no `contact_sheet` widget.
 */
function sheetWanted(node) {
  const widget = findWidget(node, SHEET_WIDGET);
  if (!widget) return true;
  return String(widget.value ?? "true") === "true";
}

/**
 * Read whether the sheet is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * Append the sheet to a node and wire it to the widget it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachStyleSheet(node) {
  if (!findWidget(node, STYLE_WIDGET)) return;

  const sheet = createStyleSheet({
    node,
    backdrop: imageBackdrop(node),
    read: () => findWidget(node, STYLE_WIDGET)?.value ?? "",
    write: (style) => {
      const widget = findWidget(node, STYLE_WIDGET);
      // Through `value` and nothing else. The setter behind it writes the store the frontend
      // reads and calls the widget's own callback, so a second call here would fire the
      // callback twice for one choice.
      if (widget && widget.value !== style) widget.value = style;
    },
    linked: () => inputLinked(node, STYLE_WIDGET),
    active: () => sheetWanted(node),
    height: UI_HEIGHT,
  });

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendFilterWidget` is for.
  const sheetWidget = appendFilterWidget(node, sheet, {
    name: UI_WIDGET_NAME,
    type: UI_WIDGET_TYPE,
  });

  // The band the pack's other filters carry, opened at the sheet's own height.
  const band = createBeforeAfterPanel(node, {
    slot: PAIR_SLOT,
    height: UI_HEIGHT,
    logName: EXT_NAME,
  });
  const bandWidget = appendInterfaceWidget(node, band, {
    name: BAND_WIDGET_NAME,
    type: BAND_WIDGET_TYPE,
  });

  /**
   * Draw the panel the `contact_sheet` widget asks for and fold the other away.
   *
   * @returns {void}
   */
  function showMode() {
    const sheeting = sheetWanted(node);
    setWidgetHidden(sheetWidget, !sheeting);
    setWidgetHidden(bandWidget, sheeting);
    // Grown to the shown panel's own floor, and never shrunk.
    const computed = node.computeSize?.();
    if (computed) {
      node.setSize([
        Math.max(node.size[0], computed[0]),
        Math.max(node.size[1], computed[1]),
      ]);
    }
    if (sheeting) {
      sheet.refresh();
      sheet.schedulePaint();
    } else {
      band.refresh();
    }
    node.graph?.setDirtyCanvas(true, true);
  }

  chainWidgetCallback(node, STYLE_WIDGET, () => sheet.schedulePaint(), EXT_NAME);
  chainWidgetCallback(node, SHEET_WIDGET, () => showMode(), EXT_NAME);

  const stopWatchingRuns = watchRuns(node, sheet);

  // Linking the style input leaves its widget read by nothing, and attaching a link changes no
  // widget value, so the callback above never hears about it.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      sheet.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      showMode();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    // One release each, so a throw in the first still leaves the second one released.
    for (const release of [stopWatchingRuns, () => sheet.dispose(), () => band.dispose()]) {
      try {
        release();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to release a style panel:`, error);
      }
    }
    return result;
  };

  showMode();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Style Filter", "Panel"],
      name: "Show the panel on Image Style Filter",
      tooltip:
        "Draw a panel on Image Style Filter. The node's contact_sheet widget chooses which " +
        "one: on, all 37 styles over the image, and a click on a tile picks that style; off, " +
        "the image beside the graded result with the difference between them. The style " +
        "widget itself is always available. This applies to nodes added after the setting " +
        "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second sheet.
    if (proto.__was_style_sheet_wrapped) return;
    proto.__was_style_sheet_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachStyleSheet(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the style sheet:`, error);
      }
      return result;
    };
  },
});
