/**
 * Transfer curve for the Image Shadows and Highlights node.
 *
 * Drawn on a 256 by 256 lattice of input level against output level, from four of the node's
 * own widgets. The level is the unit throughout.
 */

import { app } from "../../scripts/app.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { roundHalfEven } from "./interface/python_arithmetic.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ShadowsAndHighlightsUI";
const NODE_NAME = "Image Shadows and Highlights";
const SETTING_ID = "WAS.ShadowsAndHighlights.ShowInterface";

const SHADOW_THRESHOLD = "shadow_threshold";
const SHADOW_FACTOR = "shadow_factor";
const HIGHLIGHT_THRESHOLD = "highlight_threshold";
const HIGHLIGHT_FACTOR = "highlight_factor";
const SHADOW_SMOOTHING = "shadow_smoothing";
const HIGHLIGHT_SMOOTHING = "highlight_smoothing";
const SIMPLIFY_ISOLATION = "simplify_isolation";

// The widgets the curve is computed from, and the three it is blind to but reports on.
const CURVE_WIDGETS = [SHADOW_THRESHOLD, SHADOW_FACTOR, HIGHLIGHT_THRESHOLD, HIGHLIGHT_FACTOR];
const BLUR_WIDGETS = [SHADOW_SMOOTHING, HIGHLIGHT_SMOOTHING, SIMPLIFY_ISOLATION];

const UI_WIDGET_NAME = "was_shadows_highlights_ui";
const UI_WIDGET_TYPE = "was_shadows_highlights_curve";

// The schema's own defaults, read only when a widget cannot be. The two smoothing radii are
// here as well so a widget that cannot be read is treated as the blur it ships with rather than
// as no blur at all, which would understate what the curve leaves out.
const DEFAULTS = {
  [SHADOW_THRESHOLD]: 75,
  [SHADOW_FACTOR]: 1.5,
  [HIGHLIGHT_THRESHOLD]: 175,
  [HIGHLIGHT_FACTOR]: 0.5,
  [SHADOW_SMOOTHING]: 0.25,
  [HIGHLIGHT_SMOOTHING]: 0.25,
  [SIMPLIFY_ISOLATION]: 0,
};

// An 8 bit image holds these levels and no others, so the lattice is every question the node
// can be asked and every answer it can give rather than a sample of either.
const LEVELS = 256;
const MAX_LEVEL = 255;

const THRESHOLD_MIN = 0;
const THRESHOLD_MAX = 255;

// A threshold is written as a whole level. The node tests an integer level against it, so
// every cut the node can make is reachable from a whole number and a fraction moves the cut
// only when it crosses one.
const THRESHOLD_STEP = 1;
const THRESHOLD_COARSE_STEP = 10;

const FACTOR_FLOOR = 0;
const FACTOR_MAX = 12;
const FACTOR_STEP = 0.1;
const FACTOR_COARSE_STEP = 0.5;

// A factor drag pivots on the level under the pointer, so the curve follows the pointer. The
// pivot is held at this level or above: below it the whole factor range spans a handful of
// output levels and the curve could not track the pointer at all.
const PIVOT_MIN_LEVEL = 8;

const SHADOW = "shadow";
const KEEP = "keep";
const HIGHLIGHT = "highlight";

const BAND_LABELS = { [SHADOW]: "shadow", [KEEP]: "kept", [HIGHLIGHT]: "highlight" };
const REGION_FACTOR = { [SHADOW]: SHADOW_FACTOR, [HIGHLIGHT]: HIGHLIGHT_FACTOR };
const REGION_THRESHOLD = { [SHADOW]: SHADOW_THRESHOLD, [HIGHLIGHT]: HIGHLIGHT_THRESHOLD };

const ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

// Height of the appended widget in node units. A DOM widget element is inset by the widget's
// margin on every side, so the element itself is shorter by twice that margin. The footer's one
// line is paid for out of the widget rather than out of the plot, and the room the footer once
// took for a second line stays in the widget, where the plot has it.
const UI_HEIGHT = 181;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout bands, measured in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const GUTTER_WIDTH = 18;
const TICK_HEIGHT = 10;
const FOOTER_HEIGHT = 13;

// The footer's one line. It opens with the glyph for what the curve leaves out, which is a
// standing fact about every curve this file can draw, then carries the readout, and on the right
// whatever the gesture, the links or the thresholds have to say.
const FOOTER_LINES = 1;
const MIN_PLOT_HEIGHT = 24;

// The gap kept between the glyph and the readout beside it.
const GLYPH_GAP = 4;

const HANDLE_HALF_WIDTH = 4;
const HANDLE_HEIGHT = 5;
const HIT_RADIUS = 6;
const SAMPLE_THICKNESS = 1.6;
const FLOOR_THICKNESS = 3;
const LABEL_PADDING = 4;

const BODY_FONT = "10px sans-serif";
const AXIS_FONT = "9px sans-serif";

const BAND_ALPHA = 0.07;
const BAND_SELECTED_ALPHA = 0.16;
const GRID_ALPHA = 0.4;
const FLOOR_ALPHA = 0.3;

const MESSAGE_TIMEOUT = 4000;

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
 * Test whether one of a node's inputs is linked.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  for (const input of inputs) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

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
 * Hold a lookup table entry to the range PIL stores it in.
 *
 * @param {number} value - Rounded table entry.
 * @returns {number} The entry, held to 0 through 255.
 */
function clipToByte(value) {
  return value <= 0 ? 0 : value > MAX_LEVEL ? MAX_LEVEL : value;
}

/**
 * Snap a value to a multiple of a step.
 *
 * This is pointer and key arithmetic rather than the node's, so it rounds half up.
 *
 * @param {number} value - Value to snap.
 * @param {number} step - Step to snap to.
 * @returns {number} The snapped value.
 */
function snap(value, step) {
  if (!(step > 0)) return value;
  // Two decimals, so a step of 0.1 stores 1.5 rather than 1.5000000000000002 and a repeated
  // gesture can be compared against what is already in the widget. Adding zero turns the
  // negative zero a drag just under the axis produces back into a plain one.
  return Number((Math.round(value / step) * step).toFixed(2)) + 0;
}

/**
 * Format a widget value for the footer.
 *
 * @param {number} value - Value to write.
 * @returns {string} The value with at most two decimals.
 */
function formatNumber(value) {
  if (!Number.isFinite(value)) return "?";
  return String(Math.round(value * 100) / 100);
}

/**
 * The region one input level falls in.
 *
 * @param {object} values - The four values the curve is drawn from.
 * @param {number} level - Input level.
 * @returns {string} `SHADOW`, `KEEP` or `HIGHLIGHT`.
 */
function regionAt(values, level) {
  if (level > values[HIGHLIGHT_THRESHOLD]) return HIGHLIGHT;
  if (level < values[SHADOW_THRESHOLD]) return SHADOW;
  return KEEP;
}

/**
 * Build the transfer curve for one set of widget values.
 *
 * @param {object} values - The four values the curve is drawn from.
 * @returns {{outputs: number[], regions: string[], overlap: number}} The output level for each
 *   of the 256 inputs, the region each input falls in, and how many levels both masks claim.
 */
function buildCurve(values) {
  // A neutral pixel converts to its own value in mode `L`, so each of the node's two masks is
  // that value tested against a threshold and one level is all the state a cell needs. The
  // colour blend that finishes the node puts the source's hue and saturation over the adjusted
  // luminosity and reads both luminosities through the same integer conversion, so for a grey
  // it hands back the level the lookup table already produced.
  const outputs = new Array(LEVELS);
  const regions = new Array(LEVELS);
  let overlap = 0;

  for (let level = 0; level < LEVELS; level++) {
    if (level < values[SHADOW_THRESHOLD] && level > values[HIGHLIGHT_THRESHOLD]) overlap++;

    const region = regionAt(values, level);
    regions[level] = region;
    // Each multiply runs through a 256 entry lookup table. `Image.point` rounds every entry
    // before storing it, so the node's two multiplies land half to even: a factor of 0.5 sends
    // 253 to 126 rather than to the 127 half up would give.
    if (region === HIGHLIGHT) {
      outputs[level] = clipToByte(roundHalfEven(level * values[HIGHLIGHT_FACTOR]));
    } else if (region === SHADOW) {
      outputs[level] = clipToByte(roundHalfEven(level * values[SHADOW_FACTOR]));
    } else {
      outputs[level] = level;
    }
  }

  return { outputs, regions, overlap };
}

/**
 * Collect the runs of levels that share a region.
 *
 * @param {string[]} regions - Region of each level.
 * @returns {Array<{region: string, from: number, to: number}>} Runs in level order, `to`
 *   being the first level past the run.
 */
function regionRuns(regions) {
  const runs = [];
  let start = 0;
  for (let level = 1; level <= LEVELS; level++) {
    if (level === LEVELS || regions[level] !== regions[start]) {
      runs.push({ region: regions[start], from: start, to: level });
      start = level;
    }
  }
  return runs;
}

/**
 * The lattice edge the shadow cut is drawn on.
 *
 * @param {object} values - The four values the curve is drawn from.
 * @returns {number} An edge index, 0 through 256.
 */
function shadowEdge(values) {
  return clamp(Math.ceil(values[SHADOW_THRESHOLD]), 0, LEVELS);
}

/**
 * The lattice edge the highlight cut is drawn on.
 *
 * @param {object} values - The four values the curve is drawn from.
 * @returns {number} An edge index, 0 through 256.
 */
function highlightEdge(values) {
  return clamp(Math.floor(values[HIGHLIGHT_THRESHOLD]) + 1, 0, LEVELS);
}

/**
 * The lattice edge one region's cut is drawn on.
 *
 * @param {object} values - The four values the curve is drawn from.
 * @param {string} region - `SHADOW` or `HIGHLIGHT`.
 * @returns {number} An edge index, 0 through 256.
 */
function edgeFor(values, region) {
  return region === SHADOW ? shadowEdge(values) : highlightEdge(values);
}

/**
 * Read whether the curve is drawn at all.
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
 * Work out where the plot, the tick row and the footer sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry, including the size of one lattice cell.
 */
function computeLayout(width, height) {
  const plotX0 = PAD_X + GUTTER_WIDTH;
  const plotX1 = Math.max(plotX0 + 1, width - PAD_X);
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT * FOOTER_LINES);
  const tickY = Math.max(0, footerY - TICK_HEIGHT);
  const plotY0 = PAD_Y;
  const plotY1 = Math.max(plotY0 + MIN_PLOT_HEIGHT, tickY - 2);
  const plotWidth = plotX1 - plotX0;
  const plotHeight = plotY1 - plotY0;

  return {
    width,
    height,
    plotX0,
    plotX1,
    plotY0,
    plotY1,
    plotWidth,
    plotHeight,
    cellWidth: plotWidth / LEVELS,
    cellHeight: plotHeight / LEVELS,
    tickY,
    footerY,
  };
}

/**
 * Build the transfer curve interface for one node.
 *
 * @param {object} node - The node the curve decorates.
 * @returns {{element: HTMLElement, schedulePaint: () => void,
 *   handleWidgetChanged: (name: string) => void, dispose: () => void}} The element to hand to
 *   `addDOMWidget`, a coalesced repaint, the repaint to run when a widget changed, and
 *   teardown.
 */
function createCurveEditor(node) {
  const root = document.createElement("div");
  root.tabIndex = 0;
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${ELEMENT_MIN_HEIGHT}px`,
    "overflow:hidden",
    "outline:none",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The footer's glyph states what the curve leaves out through the element's own title. The
  // region is handed over again on every repaint, since the glyph moves whenever the node is
  // resized.
  const titles = hoverTitles(root);

  const state = {
    selected: null,
    hoverRegion: null,
    hoverLevel: null,
    drag: null,
    pending: null,
    lastWritten: {},
    message: "",
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Read one widget as a number.
   *
   * @param {string} name - Widget name.
   * @returns {number} The widget's value, or the schema default when it cannot be read.
   */
  function readValue(name) {
    const value = Number(findWidget(node, name)?.value);
    return Number.isFinite(value) ? value : (DEFAULTS[name] ?? 0);
  }

  /**
   * Read the four values the curve is drawn from, with any unfinished gesture applied.
   *
   * @returns {object} The values keyed by widget name.
   */
  function readValues() {
    const values = {};
    for (const name of CURVE_WIDGETS) values[name] = readValue(name);
    if (state.pending) values[state.pending.name] = state.pending.value;
    return values;
  }

  /**
   * Read a widget's own bounds, falling back to the schema's.
   *
   * @param {string} name - Widget name.
   * @param {number} low - Fallback minimum.
   * @param {number} high - Fallback maximum.
   * @returns {{min: number, max: number}} The range a written value is held to.
   */
  function readBounds(name, low, high) {
    const options = findWidget(node, name)?.options ?? {};
    const min = Number.isFinite(options.min) ? options.min : low;
    const max = Number.isFinite(options.max) ? options.max : high;
    return { min, max };
  }

  /**
   * Test whether any of the three blur inputs is doing something.
   *
   * @returns {boolean} True while a blur radius is not zero or comes from a link.
   */
  function blurActive() {
    // All three blur in image space and none of them can be drawn on a curve, so the footer says
    // so whenever one of them is set, which at the shipped defaults is always. A linked radius
    // counts as set: the widget beside it is not the number the run will use, and claiming no
    // blur on the strength of it would understate what the curve leaves out.
    for (const name of BLUR_WIDGETS) {
      if (inputLinked(node, name) || readValue(name) !== 0) return true;
    }
    return false;
  }

  /**
   * Name the curve inputs a link fills in.
   *
   * @returns {string[]} The linked names, in the order the curve reads them.
   */
  function linkedCurves() {
    // A linked input is read by nothing on the run, so the value under the pointer is not the
    // value the image is made with: the footer names it and the gestures that would write it
    // are refused.
    return CURVE_WIDGETS.filter((name) => inputLinked(node, name));
  }

  /**
   * Write one widget, once.
   *
   * @param {string} name - Widget name.
   * @param {number} value - Value to store.
   * @returns {void}
   */
  function writeValue(name, value) {
    if (state.disposed) return;
    const widget = findWidget(node, name);
    if (!widget) return;
    // A widget whose input is linked is never written. The gestures refuse it before they reach
    // here, and this catches the one that cannot: a gesture held on the keyboard while the link
    // is attached, since attaching one changes no widget value and drops nothing.
    if (inputLinked(node, name)) return;
    // The value is compared first, so a repaint driven by a widget's own callback can never
    // write anything back.
    if (!Number.isFinite(value) || value === widget.value) return;

    // The write is bracketed by the canvas change events the graph's change tracker listens for,
    // which is what gives the edit its own undo entry. The tracker's own snapshot triggers are a
    // document `mouseup` and the release of a bare modifier key, so a commit made with the
    // keyboard reaches none of them and would otherwise be folded into whatever the previous
    // snapshot held.
    const canvas = app.canvas;
    const transactional =
      typeof canvas?.emitBeforeChange === "function" &&
      typeof canvas?.emitAfterChange === "function";

    state.lastWritten[name] = value;
    if (transactional) canvas.emitBeforeChange();
    try {
      widget.value = value;
    } finally {
      if (transactional) canvas.emitAfterChange();
    }
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Write the value an unfinished gesture holds.
   *
   * @returns {void}
   */
  function commitPending() {
    const pending = state.pending;
    state.pending = null;
    if (!pending) return;
    writeValue(pending.name, pending.value);
    schedulePaint();
  }

  /**
   * Hold a value for a gesture in progress, committing one already held for another widget.
   *
   * @param {string} name - Widget name.
   * @param {number} value - Value the gesture has reached.
   * @returns {void}
   */
  function holdPending(name, value) {
    if (state.pending && state.pending.name !== name) commitPending();
    if (state.pending?.value === value) return;
    state.pending = { name, value };
    schedulePaint();
  }

  /**
   * Repaint after a widget changed, dropping a gesture the change invalidated.
   *
   * @param {string} name - Widget name.
   * @returns {void}
   */
  function handleWidgetChanged(name) {
    const current = findWidget(node, name)?.value;
    if (state.lastWritten[name] !== current) {
      delete state.lastWritten[name];
      if (state.pending?.name === name) state.pending = null;
    }
    schedulePaint();
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
   * Horizontal position of a lattice edge.
   *
   * @param {number} edge - Edge index, 0 through 256.
   * @returns {number} Position in element pixels.
   */
  function xFromEdge(edge) {
    const layout = state.layout;
    return layout.plotX0 + clamp(edge, 0, LEVELS) * layout.cellWidth;
  }

  /**
   * Horizontal centre of one input level's cell.
   *
   * @param {number} level - Input level.
   * @returns {number} Position in element pixels.
   */
  function xFromLevel(level) {
    const layout = state.layout;
    return layout.plotX0 + (level + 0.5) * layout.cellWidth;
  }

  /**
   * Vertical centre of one output level's cell.
   *
   * @param {number} output - Output level.
   * @returns {number} Position in element pixels.
   */
  function yFromOutput(output) {
    const layout = state.layout;
    return layout.plotY1 - (output + 0.5) * layout.cellHeight;
  }

  /**
   * The input level a horizontal position sits on.
   *
   * @param {number} x - Position in element pixels.
   * @returns {number} An input level, 0 through 255.
   */
  function levelFromX(x) {
    const layout = state.layout;
    const level = Math.floor((x - layout.plotX0) / layout.cellWidth);
    return clamp(level, 0, MAX_LEVEL);
  }

  /**
   * The output level a vertical position stands for, unclamped so a drag past the plot keeps
   * pulling the factor towards its bound.
   *
   * @param {number} y - Position in element pixels.
   * @returns {number} An output level.
   */
  function outputFromY(y) {
    const layout = state.layout;
    return (layout.plotY1 - y) / layout.cellHeight - 0.5;
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
   * Test whether a point is inside the plot.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {boolean} True when the point is over the lattice.
   */
  function insidePlot(point) {
    const layout = state.layout;
    return (
      point.x >= layout.plotX0 &&
      point.x <= layout.plotX1 &&
      point.y >= layout.plotY0 &&
      point.y <= layout.plotY1
    );
  }

  /**
   * How far to one side of a threshold line the pointer still lands on the line.
   *
   * @param {object} values - Values from `readValues`.
   * @param {string} region - `SHADOW` or `HIGHLIGHT`.
   * @param {boolean} leftSide - Whether the pointer is to the left of the line.
   * @returns {number} The reach in element pixels.
   */
  function thresholdReach(values, region, leftSide) {
    const shadow = shadowEdge(values);
    const highlight = highlightEdge(values);
    const levels =
      region === SHADOW
        ? leftSide
          ? shadow
          : highlight - shadow
        : leftSide
          ? highlight - shadow
          : LEVELS - highlight;
    // A lattice cell is well under a pixel wide, so a band of a few levels is narrower than the
    // hit radius and would lie entirely inside its own line's hit zone, leaving its factor
    // unreachable with the pointer. The reach is held to half the band on that side instead,
    // which always leaves the outer half of the band to press on.
    return clamp((Math.max(0, levels) * state.layout.cellWidth) / 2, 1, HIT_RADIUS);
  }

  /**
   * Find the threshold line under a point.
   *
   * The nearer line wins.
   *
   * @param {object} values - Values from `readValues`.
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {string|null} The region whose line is under the point, or null.
   */
  function hitThreshold(values, point) {
    const layout = state.layout;
    if (point.y < layout.plotY0 - 2 || point.y > layout.plotY1 + 2) return null;
    // The handle drawn at the top of a line answers at the full radius whatever the bands either
    // side of it are doing, so a line between two narrow bands is always somewhere to grab.
    const onHandle = point.y <= layout.plotY0 + HANDLE_HEIGHT;

    // The selected line is tested last and wins a tie, so a line dragged on top of the other
    // can still be picked up.
    const order = state.selected === SHADOW ? [HIGHLIGHT, SHADOW] : [SHADOW, HIGHLIGHT];
    let best = null;
    let bestDistance = HIT_RADIUS;
    for (const region of order) {
      const x = xFromEdge(edgeFor(values, region));
      const distance = Math.abs(x - point.x);
      const reach = onHandle ? HIT_RADIUS : thresholdReach(values, region, point.x < x);
      if (distance <= reach && distance <= bestDistance) {
        best = region;
        bestDistance = distance;
      }
    }
    return best;
  }

  /**
   * The threshold a horizontal position stands for.
   *
   * @param {number} x - Position in element pixels.
   * @param {string} region - Region being dragged.
   * @param {boolean} coarse - Snap to a coarser step.
   * @returns {number} A threshold, inside the widget's bounds.
   */
  function thresholdFromX(x, region, coarse) {
    const layout = state.layout;
    const edge = clamp(Math.round((x - layout.plotX0) / layout.cellWidth), 0, LEVELS);
    // The shadow cut is drawn on the edge before the first level it leaves alone, the
    // highlight cut on the edge before the first level it multiplies.
    const raw = region === SHADOW ? edge : edge - 1;
    const bounds = readBounds(REGION_THRESHOLD[region], THRESHOLD_MIN, THRESHOLD_MAX);
    const step = coarse ? THRESHOLD_COARSE_STEP : THRESHOLD_STEP;
    return clamp(snap(raw, step), bounds.min, bounds.max);
  }

  /**
   * The factor a vertical position stands for.
   *
   * @param {number} y - Position in element pixels.
   * @param {string} region - Region being dragged.
   * @param {number} pivot - Input level the curve is pinned to.
   * @param {boolean} coarse - Snap to a coarser step.
   * @returns {number} A factor, inside the widget's bounds.
   */
  function factorFromY(y, region, pivot, coarse) {
    const bounds = readBounds(REGION_FACTOR[region], FACTOR_FLOOR, FACTOR_MAX);
    const step = coarse ? FACTOR_COARSE_STEP : FACTOR_STEP;
    const low = Math.max(bounds.min, FACTOR_FLOOR);
    return clamp(snap(outputFromY(y) / pivot, step), low, bounds.max);
  }

  /**
   * Draw the three bands, and the floor of any band that renders black.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} values - Values from `readValues`.
   * @param {Array<object>} runs - Runs from `regionRuns`.
   * @returns {void}
   */
  function drawBands(ctx, theme, values, runs) {
    const layout = state.layout;
    ctx.font = AXIS_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    for (const run of runs) {
      const x0 = xFromEdge(run.from);
      const width = xFromEdge(run.to) - x0;
      if (width <= 0) continue;

      const factorName = REGION_FACTOR[run.region];
      const dead = Boolean(factorName) && values[factorName] <= 0;

      if (run.region !== KEEP) {
        ctx.globalAlpha = state.selected === run.region ? BAND_SELECTED_ALPHA : BAND_ALPHA;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(x0, layout.plotY0, width, layout.plotHeight);
        ctx.globalAlpha = 1;
      }

      if (dead) {
        // Every level in the band is on the bottom row of the lattice. The floor is drawn as a
        // floor so the band reads as a dead one rather than as a curve that happens to be flat.
        ctx.globalAlpha = FLOOR_ALPHA;
        ctx.fillStyle = theme.error;
        ctx.fillRect(x0, layout.plotY1 - FLOOR_THICKNESS, width, FLOOR_THICKNESS);
        ctx.globalAlpha = 1;
      }

      const label = dead ? "black" : BAND_LABELS[run.region];
      if (label && ctx.measureText(label).width + LABEL_PADDING * 2 <= width) {
        ctx.fillStyle = dead ? theme.error : theme.fgMuted;
        ctx.fillText(label, x0 + width / 2, layout.plotY0 + 2);
      }
    }
  }

  /**
   * Draw the quarter grid and the line an untouched image would follow.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawGrid(ctx, theme) {
    const layout = state.layout;

    ctx.globalAlpha = GRID_ALPHA;
    ctx.strokeStyle = theme.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const level of [64, 128, 192]) {
      const x = Math.round(xFromEdge(level)) + 0.5;
      ctx.moveTo(x, layout.plotY0);
      ctx.lineTo(x, layout.plotY1);
      const y = Math.round(yFromOutput(level)) + 0.5;
      ctx.moveTo(layout.plotX0, y);
      ctx.lineTo(layout.plotX1, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = theme.fgMuted;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(xFromLevel(0), yFromOutput(0));
    ctx.lineTo(xFromLevel(MAX_LEVEL), yFromOutput(MAX_LEVEL));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw one cell per input level.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {number[]} outputs - Output level of each input.
   * @returns {void}
   */
  function drawCurve(ctx, theme, outputs) {
    const layout = state.layout;
    const width = Math.max(layout.cellWidth, 1);
    const height = Math.max(layout.cellHeight, SAMPLE_THICKNESS);

    // Nothing here is fitted. At the shipped defaults the cell for input 175 sits at output 175
    // and the cell for 176 sits at 88, and the 87 level gap between them is the gap the two
    // lookup tables have. The steepest settings read the same way, as widely spaced cells: a
    // factor of 12 skips 11 output levels out of every 12.
    ctx.beginPath();
    for (let level = 0; level < LEVELS; level++) {
      ctx.rect(
        xFromLevel(level) - width / 2,
        yFromOutput(outputs[level]) - height / 2,
        width,
        height,
      );
    }
    ctx.fillStyle = theme.fg;
    ctx.fill();
  }

  /**
   * Draw the two threshold lines and their handles.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} values - Values from `readValues`.
   * @returns {void}
   */
  function drawThresholds(ctx, theme, values) {
    const layout = state.layout;

    for (const region of [SHADOW, HIGHLIGHT]) {
      const x = Math.round(xFromEdge(edgeFor(values, region))) + 0.5;
      const selected = state.selected === region;
      const hovered = state.hoverRegion === region;
      const colour = selected ? theme.accent : hovered ? theme.fg : theme.fgMuted;

      ctx.strokeStyle = colour;
      ctx.lineWidth = selected ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x, layout.plotY0);
      ctx.lineTo(x, layout.plotY1);
      ctx.stroke();

      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.moveTo(x - HANDLE_HALF_WIDTH, layout.plotY0);
      ctx.lineTo(x + HANDLE_HALF_WIDTH, layout.plotY0);
      ctx.lineTo(x, layout.plotY0 + HANDLE_HEIGHT);
      ctx.closePath();
      ctx.fill();
    }
    ctx.lineWidth = 1;
  }

  /**
   * Draw the two axes.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawAxes(ctx, theme) {
    const layout = state.layout;

    ctx.font = AXIS_FONT;
    ctx.fillStyle = theme.fgMuted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const level of [MAX_LEVEL, 128, 0]) {
      // The top and bottom rows of the lattice are half a cell from the edge of the plot, so
      // their labels are held inside it rather than drawn half outside.
      const y = clamp(yFromOutput(level), layout.plotY0 + 4, layout.plotY1 - 4);
      ctx.fillText(String(level), layout.plotX0 - 2, y);
    }

    const middle = layout.tickY + TICK_HEIGHT / 2;
    ctx.textAlign = "left";
    ctx.fillText("0", layout.plotX0, middle);
    ctx.textAlign = "center";
    ctx.fillText("128", xFromEdge(128), middle);
    ctx.textAlign = "right";
    ctx.fillText("255", layout.plotX1, middle);
  }

  /**
   * The left half of the footer: what the gesture, the pointer or the selection is on.
   *
   * @param {object} values - Values from `readValues`.
   * @param {object} curve - Curve from `buildCurve`.
   * @returns {string} Text to draw.
   */
  function footerReadout(values, curve) {
    const drag = state.drag;
    if (drag) {
      if (drag.kind === "threshold") {
        const value = formatNumber(values[REGION_THRESHOLD[drag.region]]);
        return drag.region === SHADOW ? `shadow < ${value}` : `highlight > ${value}`;
      }
      const factor = values[REGION_FACTOR[drag.region]];
      return `${drag.region} x ${formatNumber(factor)}${factor <= 0 ? "   black" : ""}`;
    }

    if (state.hoverLevel !== null) {
      return `in ${state.hoverLevel}   out ${curve.outputs[state.hoverLevel]}`;
    }

    if (state.selected) {
      const threshold = formatNumber(values[REGION_THRESHOLD[state.selected]]);
      const factor = formatNumber(values[REGION_FACTOR[state.selected]]);
      const sign = state.selected === SHADOW ? "<" : ">";
      return `${state.selected} ${sign} ${threshold}   x ${factor}`;
    }

    const shadow = `<${formatNumber(values[SHADOW_THRESHOLD])}`;
    const highlight = `>${formatNumber(values[HIGHLIGHT_THRESHOLD])}`;
    const shadowFactor = `x${formatNumber(values[SHADOW_FACTOR])}`;
    const highlightFactor = `x${formatNumber(values[HIGHLIGHT_FACTOR])}`;
    return `${shadow} ${shadowFactor}   ${highlight} ${highlightFactor}`;
  }

  /**
   * What the curve leaves out, for the glyph and the hover text behind it.
   *
   * Both omissions are named.
   *
   * @returns {string} The sentence the glyph carries.
   */
  function fidelityText() {
    // Every channel is multiplied and held to 0 through 255 on its own before any luminosity is
    // read, so a saturated pixel is off this curve however neutral its brightness. The three
    // blur radii work in image space, so a pixel beside a brighter or darker one lands between
    // two treatments and every step drawn here is softened.
    return blurActive()
      ? "the curve is one neutral grey level at a time, with no clipped colour and no blur, and"
        + " the three radii soften every step drawn here"
      : "the curve is one neutral grey level at a time, so a channel that clips is off it however"
        + " neutral the pixel's brightness";
  }

  /**
   * The note the footer line carries on the right.
   *
   * @param {object} curve - Curve from `buildCurve`.
   * @returns {string} The note, empty when there is nothing to report.
   */
  function footerNote(curve) {
    if (state.message) return state.message;

    const linked = linkedCurves();
    if (linked.length === 1) return `${linked[0]} is linked`;
    if (linked.length > 1) return `${linked.length} inputs are linked`;

    // Both masks claim these levels and the highlight copy is pasted second, so it is the one
    // that survives. Nothing else on the node says which.
    if (curve.overlap > 0) return "crossed, highlight wins";
    return "";
  }

  /**
   * Draw the footer line.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} values - Values from `readValues`.
   * @param {object} curve - Curve from `buildCurve`.
   * @returns {void}
   */
  function drawFooter(ctx, theme, values, curve) {
    const layout = state.layout;
    const middle = layout.footerY + FOOTER_HEIGHT / 2;
    const note = footerNote(curve);

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    const box = drawIcon(
      ctx,
      ICON.APPROXIMATE,
      layout.plotX0,
      middle - ICON_SIZE / 2,
      ICON_SIZE,
      theme.fgMuted,
    );
    titles.set([{ ...box, title: iconTitle(ICON.APPROXIMATE, fidelityText()) }]);
    const glyphWidth = ICON_SIZE + GLYPH_GAP;

    let noteWidth = 0;
    if (note) {
      noteWidth = ctx.measureText(note).width;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.warning;
      ctx.fillText(note, layout.plotX1, middle);
    }

    const available = layout.plotWidth - glyphWidth - noteWidth - 8;
    if (available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = state.drag || state.selected ? theme.fg : theme.fgMuted;
      ctx.fillText(footerReadout(values, curve), layout.plotX0 + glyphWidth, middle, available);
    }
  }

  /**
   * Draw the whole interface.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return;

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
    if (layout.plotWidth <= 0 || layout.plotHeight <= 0) {
      // Nothing was drawn over the cleared canvas, so no glyph is under the pointer either.
      titles.set([]);
      return;
    }

    const theme = readTheme();
    const values = readValues();
    const curve = buildCurve(values);

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.plotX0, layout.plotY0, layout.plotWidth, layout.plotHeight);

    drawBands(ctx, theme, values, regionRuns(curve.regions));
    drawGrid(ctx, theme);
    drawCurve(ctx, theme, curve.outputs);
    drawThresholds(ctx, theme, values);

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      layout.plotX0 + 0.5,
      layout.plotY0 + 0.5,
      Math.max(1, layout.plotWidth - 1),
      Math.max(1, layout.plotHeight - 1),
    );

    drawAxes(ctx, theme);
    drawFooter(ctx, theme, values, curve);

    if (document.activeElement === root) {
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
        console.error(`[${EXT_NAME}] Failed to draw the transfer curve:`, error);
      }
    });
  }

  /**
   * Begin a drag on one control.
   *
   * @param {PointerEvent} event - The press that starts the drag.
   * @param {object} drag - What is being dragged: a `kind`, its `region` and, for a factor,
   *   the `pivot` level the curve is pinned to.
   * @returns {void}
   */
  function startDrag(event, drag) {
    state.selected = drag.region;
    state.drag = { pointerId: event.pointerId, ...drag };
    state.hoverRegion = null;
    state.hoverLevel = null;
    root.setPointerCapture?.(event.pointerId);
    schedulePaint();
  }

  /**
   * End a drag, releasing the pointer capture it holds.
   *
   * @param {boolean} commit - Write the value the drag reached. False discards it and leaves
   *   the widget as it was.
   * @returns {void}
   */
  function endDrag(commit) {
    const drag = state.drag;
    if (!drag) return;
    state.drag = null;
    root.style.cursor = "default";
    if (root.hasPointerCapture?.(drag.pointerId)) root.releasePointerCapture?.(drag.pointerId);
    if (commit) {
      commitPending();
      return;
    }
    state.pending = null;
    schedulePaint();
  }

  /**
   * Refuse an edit to a widget a link fills in, and say so.
   *
   * @param {string} name - Widget the gesture would write.
   * @param {string} region - Region the gesture was aimed at.
   * @returns {boolean} True when the gesture was refused.
   */
  function refuseLinked(name, region) {
    if (!inputLinked(node, name)) return false;
    state.selected = region;
    setMessage(`${name} is linked`);
    return true;
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
        console.error(`[${EXT_NAME}] Transfer curve input failed:`, error);
      }
    };
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;

    root.focus?.({ preventScroll: true });
    if (!state.drag) commitPending();

    // The pointer default action is left alone throughout. Cancelling it would suppress the
    // mouse events that follow, which carry the graph snapshot that gives the gesture its
    // undo entry.
    const point = localPoint(event);
    const values = readValues();

    const line = hitThreshold(values, point);
    if (line) {
      if (refuseLinked(REGION_THRESHOLD[line], line)) return;
      startDrag(event, { kind: "threshold", region: line });
      return;
    }

    if (!insidePlot(point)) return;

    const level = levelFromX(point.x);
    const region = regionAt(values, level);
    if (region === KEEP) {
      state.selected = null;
      schedulePaint();
      return;
    }

    if (refuseLinked(REGION_FACTOR[region], region)) return;
    startDrag(event, { kind: "factor", region, pivot: Math.max(level, PIVOT_MIN_LEVEL) });
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    const point = localPoint(event);
    const drag = state.drag;

    if (drag) {
      // A button released over another window, or a capture the browser took away, ends the
      // gesture without a pointerup. Without this the value would keep following an unpressed
      // pointer and commit a setting nobody chose.
      if (!(event.buttons & 1)) {
        endDrag(false);
        return;
      }
      if (drag.kind === "threshold") {
        root.style.cursor = "ew-resize";
        holdPending(
          REGION_THRESHOLD[drag.region],
          thresholdFromX(point.x, drag.region, event.shiftKey),
        );
        return;
      }
      root.style.cursor = "ns-resize";
      holdPending(
        REGION_FACTOR[drag.region],
        factorFromY(point.y, drag.region, drag.pivot, event.shiftKey),
      );
      return;
    }

    const values = readValues();
    const line = hitThreshold(values, point);
    const level = insidePlot(point) ? levelFromX(point.x) : null;
    const region = level === null ? null : regionAt(values, level);

    root.style.cursor = line ? "ew-resize" : region && region !== KEEP ? "ns-resize" : "default";

    if (line !== state.hoverRegion || level !== state.hoverLevel) {
      state.hoverRegion = line;
      state.hoverLevel = level;
      schedulePaint();
    }
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    endDrag(true);
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  };

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;

    let handled = true;

    switch (event.key) {
      case "ArrowLeft":
      case "ArrowRight": {
        if (!state.selected) {
          state.selected = SHADOW;
          schedulePaint();
          break;
        }
        const name = REGION_THRESHOLD[state.selected];
        if (refuseLinked(name, state.selected)) break;
        const bounds = readBounds(name, THRESHOLD_MIN, THRESHOLD_MAX);
        const step =
          (event.shiftKey ? THRESHOLD_COARSE_STEP : THRESHOLD_STEP) *
          (event.key === "ArrowLeft" ? -1 : 1);
        holdPending(
          name,
          clamp(snap(readValues()[name] + step, THRESHOLD_STEP), bounds.min, bounds.max),
        );
        break;
      }
      case "ArrowUp":
      case "ArrowDown": {
        if (!state.selected) {
          state.selected = SHADOW;
          schedulePaint();
          break;
        }
        const name = REGION_FACTOR[state.selected];
        if (refuseLinked(name, state.selected)) break;
        const bounds = readBounds(name, FACTOR_FLOOR, FACTOR_MAX);
        const step =
          (event.shiftKey ? FACTOR_COARSE_STEP : FACTOR_STEP) *
          (event.key === "ArrowDown" ? -1 : 1);
        const current = readValues()[name];
        // The floor is where a drag stops, not where the value is held. A key press starts
        // from whatever the widget holds, so a hand typed negative factor steps inside the
        // widget's own range rather than being lifted to the floor by the first press in
        // either direction.
        const low = current < FACTOR_FLOOR ? bounds.min : Math.max(bounds.min, FACTOR_FLOOR);
        holdPending(name, clamp(snap(current + step, FACTOR_STEP), low, bounds.max));
        break;
      }
      case "Home":
      case "End": {
        // Tab is left to the browser, so the two bands are reached by the two keys that
        // already mean the ends of a range.
        state.selected = event.key === "Home" ? SHADOW : HIGHLIGHT;
        schedulePaint();
        break;
      }
      case "Delete":
      case "Backspace": {
        // Consumed whether or not it has anything to do. Left unhandled these reach ComfyUI's
        // own binding, which deletes the node the curve is drawn on.
        setMessage("nothing to delete here");
        break;
      }
      case "Escape": {
        // An unfinished key gesture is dropped rather than written, which leaves the widget
        // holding what it held before the first key press.
        if (state.drag) endDrag(false);
        else if (state.pending) state.pending = null;
        else state.selected = null;
        schedulePaint();
        break;
      }
      default:
        handled = false;
    }

    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const onKeyUp = (event) => {
    if (!ARROW_KEYS.has(event.key)) return;
    if (state.drag) return;
    commitPending();
  };

  const onBlur = () => {
    // Focus can only leave mid-drag when the gesture has been interrupted, by another window
    // taking the pointer for example, so the drag is discarded rather than kept.
    if (state.drag) endDrag(false);
    else commitPending();
    state.hoverRegion = null;
    state.hoverLevel = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(() => endDrag(false)));
  root.addEventListener("lostpointercapture", guard(() => endDrag(false)));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hoverRegion === null && state.hoverLevel === null) return;
    state.hoverRegion = null;
    state.hoverLevel = null;
    schedulePaint();
  }));
  root.addEventListener("contextmenu", guard(onContextMenu));
  // The curve scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);
  root.addEventListener("keydown", guard(onKeyDown));
  root.addEventListener("keyup", guard(onKeyUp));
  root.addEventListener("focus", guard(schedulePaint));
  root.addEventListener("blur", guard(onBlur));

  let observer = null;
  if (typeof ResizeObserver === "function") {
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
   * Release the timers, observers, listeners and hover text the interface holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    if (state.messageTimer) clearTimeout(state.messageTimer);
    state.paintHandle = 0;
    state.messageTimer = 0;
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
    titles.dispose();
  }

  return {
    element: root,
    height: UI_HEIGHT,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    handleWidgetChanged,
    dispose,
  };
}

/**
 * Chain a repaint onto a widget's callback.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {() => void} onChange - Called after the original callback.
 * @returns {void}
 */
function chainWidgetCallback(node, name, onChange) {
  const widget = findWidget(node, name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a widget change:`, error);
    }
    return result;
  };
}

/**
 * Append the curve to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachCurveEditor(node) {
  for (const name of CURVE_WIDGETS) {
    if (!findWidget(node, name)) return;
  }

  const editor = createCurveEditor(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for. The widget holds no value, so it adds no entry to the saved
  // workflow and no key to the API prompt.
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  for (const name of [...CURVE_WIDGETS, ...BLUR_WIDGETS]) {
    chainWidgetCallback(node, name, () => editor.handleWidgetChanged(name));
  }

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link
  // changes no widget value, so the callbacks above never hear about it.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      editor.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      editor.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered
  // and its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      editor.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the transfer curve:`, error);
    }
    return result;
  };

  editor.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Shadows and Highlights", "Transfer curve"],
      name: "Show the transfer curve",
      tooltip:
        "Draw the transfer curve under the widgets of Image Shadows and Highlights. The " +
        "widgets themselves are always available. This applies to nodes added after the " +
        "setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second curve.
    if (proto.__was_shadows_highlights_wrapped) return;
    proto.__was_shadows_highlights_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachCurveEditor(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the transfer curve:`, error);
      }
      return result;
    };
  },
});
