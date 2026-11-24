/**
 * The tone curve editor drawn on a node.
 *
 * `mountCurveEditor` attaches a grid of control points and writes them as
 * `rgb:0,0;255,255|r:...` on the 0 to 255 scale `modules/image/curves.py` reads. On
 * `channels: false` it draws one curve, written as `0,0;255,255`.
 */

import { app } from "../../../scripts/app.js";
import { ICON } from "./icons.mjs";
import { histogram, readPixels } from "./image_metrics.mjs";
import { captureWheel, elementPoint } from "./pointer.mjs";
import { PREVIEW_STATE, fetchInputPreview } from "./preview.mjs";
import { withGraphChange } from "./region.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.mjs";
import { onRunEnded } from "./run_events.mjs";
import { onThemeChange, readTheme } from "./theme.mjs";
import { appendInterfaceWidget, boundTextBoxes } from "./widget.mjs";

const LOG_NAME = "WASNodeSuite.CurveEditor";

// The widget the editor is added as: one name and one type for the interface, whatever node
// carries it.
const UI_WIDGET_NAME = "was_curves_ui";
const UI_WIDGET_TYPE = "was_curves_editor";

// The channel keys, in the order the node applies them. `modules/image/curves.py` holds the
// same four and the same text format.
const CHANNELS = ["rgb", "r", "g", "b"];
const CHANNEL_LABEL = { rgb: "RGB", r: "R", g: "G", b: "B" };

// The key the one curve is held under with `channels: false`, which nothing writes to text.
const SINGLE = "curve";

// What each channel's curve is drawn in, so the plot says which one is in front without
// reading the button row.
const CHANNEL_STROKE = {
  rgb: "#d8d8d8",
  r: "#ff6b6b",
  g: "#6bd66b",
  b: "#6b9dff",
};

// The scale a control point is stored on, matching the node's.
const MAX_LEVEL = 255;
const MIN_POINTS = 2;
const MAX_POINTS = 16;

// Height of the appended widget in node units, and the room the channel row takes out of it.
const UI_HEIGHT = 260;
const MAX_UI_HEIGHT = 460;
const MIN_UI_WIDTH = 220;
const TAB_HEIGHT = 22;
const FOOTER_HEIGHT = 16;
const PAD = 10;

// How near the pointer has to come to a point, in plot pixels, to take hold of it.
const GRAB_RADIUS = 9;
const POINT_RADIUS = 4;

// Grid lines per axis, counting the border, so quarters.
const GRID_STEPS = 4;

// Bins in the histogram drawn behind the grid. 128 is finer than the plot is wide at the
// smallest the node draws, so the shape is the picture's rather than the binning's.
const HISTOGRAM_BINS = 128;

// The size the input picture is read back at to count its levels. A histogram is a shape
// rather than a measurement, and a thumbnail settles into the same shape as the full frame
// for a fraction of the decode.
const SAMPLE_EDGE = 256;

// How much of the plot's height the tallest bin fills.
const HISTOGRAM_FILL = 0.62;

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
 * Find one of a node's widgets by name.
 *
 * @param {object} node - The node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * The straight line, on the stored scale.
 *
 * @returns {Array<Array<number>>} Two points, black to white.
 */
function straightLine() {
  return [[0, 0], [MAX_LEVEL, MAX_LEVEL]];
}

/**
 * Sort points by input, drop duplicate inputs and hold them on scale.
 *
 * @param {Array<Array<number>>} points - Control points, in any order.
 * @returns {Array<Array<number>>} At least two points, at most `MAX_POINTS`.
 */
function cleanPoints(points) {
  const seen = new Map();
  for (const pair of Array.isArray(points) ? points : []) {
    const x = Math.round(Number(pair?.[0]));
    const y = Math.round(Number(pair?.[1]));
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    seen.set(clamp(x, 0, MAX_LEVEL), clamp(y, 0, MAX_LEVEL));
  }
  if (seen.size < MIN_POINTS) return straightLine();
  return [...seen.entries()].sort((a, b) => a[0] - b[0]).slice(0, MAX_POINTS);
}

/**
 * Whether one channel's points are still the straight line.
 *
 * @param {Array<Array<number>>} points - Control points.
 * @returns {boolean} True when the channel is untouched.
 */
function isLine(points) {
  const cleaned = cleanPoints(points);
  return (
    cleaned.length === MIN_POINTS &&
    cleaned[0][0] === 0 && cleaned[0][1] === 0 &&
    cleaned[1][0] === MAX_LEVEL && cleaned[1][1] === MAX_LEVEL
  );
}

/**
 * Read `0,0;128,200;255,255` into control points.
 *
 * @param {string} body - Points separated by `;`, each `input,output`.
 * @returns {Array<Array<number>>} At least two points.
 */
function parsePoints(body) {
  return cleanPoints(
    String(body ?? "")
      .split(";")
      .map((item) => item.split(",").map((part) => Number(part.trim()))),
  );
}

/**
 * Which channel a block of curve text names.
 *
 * @param {string} block - One `name:points` block.
 * @returns {string|null} A key of `CHANNELS`, or null when the block names none.
 */
function blockChannel(block) {
  const at = String(block ?? "").indexOf(":");
  if (at < 0) return null;
  const name = block.slice(0, at).trim().toLowerCase();
  return CHANNELS.includes(name) ? name : null;
}

/**
 * Read a curve widget's text into control points.
 *
 * @param {string} text - `rgb:0,0;255,255|r:...`, or `0,0;255,255` on one curve.
 * @param {boolean} [channels] - False to answer the one curve under `SINGLE`.
 * @returns {object} Every key the mode holds, each with at least two points.
 */
function parseCurves(text, channels = true) {
  const raw = String(text ?? "");
  if (!channels) {
    const blocks = raw.split("|");
    if (!blocks.some((block) => blockChannel(block))) {
      return { [SINGLE]: raw.trim() ? parsePoints(raw) : straightLine() };
    }
    const named = blocks.find((block) => blockChannel(block) === CHANNELS[0]);
    return { [SINGLE]: named ? parsePoints(named.slice(named.indexOf(":") + 1)) : straightLine() };
  }
  const curves = {};
  for (const name of CHANNELS) curves[name] = straightLine();
  for (const block of raw.split("|")) {
    const name = blockChannel(block);
    if (!name) continue;
    curves[name] = parsePoints(block.slice(block.indexOf(":") + 1));
  }
  return curves;
}

/**
 * Write control points as the text a curve widget stores.
 *
 * @param {object} curves - Control points per channel.
 * @param {boolean} [channels] - False to write the one curve under `SINGLE`.
 * @returns {string} Every channel present, so the text round-trips.
 */
function serialiseCurves(curves, channels = true) {
  const body = (name) => cleanPoints(curves?.[name]).map(([x, y]) => `${x},${y}`).join(";");
  if (!channels) return body(SINGLE);
  return CHANNELS.map((name) => `${name}:${body(name)}`).join("|");
}

/**
 * Whether every curve is still the straight line.
 *
 * @param {object} curves - Control points per channel.
 * @param {boolean} [channels] - False to look at the one curve under `SINGLE`.
 * @returns {boolean} True when the node would pass its input through untouched.
 */
function isIdentity(curves, channels = true) {
  return (channels ? CHANNELS : [SINGLE]).every((name) => isLine(curves?.[name]));
}

/**
 * Fritsch-Carlson tangents for a monotone cubic through the points.
 *
 * @param {Array<number>} xs - Input levels, strictly increasing.
 * @param {Array<number>} ys - Output levels.
 * @returns {Array<number>} One tangent per point, limited against overshoot.
 */
function slopes(xs, ys) {
  const deltas = [];
  for (let i = 0; i < xs.length - 1; i += 1) deltas.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));

  const tangents = new Array(xs.length);
  tangents[0] = deltas[0];
  tangents[xs.length - 1] = deltas[deltas.length - 1];
  for (let i = 1; i < xs.length - 1; i += 1) tangents[i] = (deltas[i - 1] + deltas[i]) / 2;

  for (let i = 0; i < deltas.length; i += 1) {
    if (deltas[i] === 0) {
      tangents[i] = 0;
      tangents[i + 1] = 0;
      continue;
    }
    const alpha = tangents[i] / deltas[i];
    const beta = tangents[i + 1] / deltas[i];
    const size = alpha * alpha + beta * beta;
    if (size > 9) {
      const scale = 3 / Math.sqrt(size);
      tangents[i] = scale * alpha * deltas[i];
      tangents[i + 1] = scale * beta * deltas[i];
    }
  }
  return tangents;
}

/**
 * Evaluate one channel's curve across the whole input range.
 *
 * @param {Array<Array<number>>} points - Control points.
 * @param {number} samples - How many outputs to answer.
 * @returns {Array<number>} Outputs on a 0-255 scale, one per evenly spaced input.
 */
function curveSamples(points, samples) {
  const cleaned = cleanPoints(points);
  const xs = cleaned.map(([x]) => x);
  const ys = cleaned.map(([, y]) => y);
  const out = new Array(samples);

  if (cleaned.length === MIN_POINTS) {
    for (let i = 0; i < samples; i += 1) {
      const x = (i / (samples - 1)) * MAX_LEVEL;
      const span = xs[1] - xs[0] || 1;
      const step = clamp((x - xs[0]) / span, 0, 1);
      out[i] = ys[0] + (ys[1] - ys[0]) * step;
    }
    return out;
  }

  const tangents = slopes(xs, ys);
  for (let i = 0; i < samples; i += 1) {
    const x = (i / (samples - 1)) * MAX_LEVEL;
    if (x <= xs[0]) {
      out[i] = ys[0];
      continue;
    }
    if (x >= xs[xs.length - 1]) {
      out[i] = ys[ys.length - 1];
      continue;
    }
    let slot = 0;
    while (slot < xs.length - 2 && x >= xs[slot + 1]) slot += 1;
    const span = xs[slot + 1] - xs[slot];
    const step = (x - xs[slot]) / span;
    const step2 = step * step;
    const step3 = step2 * step;
    out[i] =
      (2 * step3 - 3 * step2 + 1) * ys[slot] +
      (step3 - 2 * step2 + step) * span * tangents[slot] +
      (-2 * step3 + 3 * step2) * ys[slot + 1] +
      (step3 - step2) * span * tangents[slot + 1];
  }
  return out;
}

/**
 * Where the plot, the channel row and the footer sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @param {boolean} [tabs] - False to give the channel row's band to the plot.
 * @returns {object} Pixel geometry of the three bands.
 */
function computeLayout(width, height, tabs = true) {
  const tabY = PAD;
  const plotY = tabs ? tabY + TAB_HEIGHT + 6 : tabY;
  const footerY = Math.max(plotY, height - PAD - FOOTER_HEIGHT);
  const plotSize = Math.max(0, footerY - plotY - 4);
  const plotX = PAD;
  const plotWidth = Math.max(0, width - PAD * 2);
  return { width, height, tabY, plotX, plotY, plotWidth, plotSize, footerY };
}

/**
 * Build the curve editor for one node.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {object} [options] - What the editor reads, writes and draws.
 * @param {string} [options.pointsWidget] - The widget holding the curve text, `curve_points`
 *   where nothing else is named.
 * @param {boolean} [options.channels] - False for one curve with no button row, written as
 *   `0,0;255,255`. Left out, there are four and they are written as `rgb:0,0;255,255|r:...`.
 * @param {boolean} [options.histogram] - Draw the levels of the picture the node was handed
 *   behind the grid. Off for a node that takes no picture.
 * @param {string} [options.logName] - What a console error from the editor is prefixed with.
 * @param {number} [options.height] - Height in node units the panel opens at.
 * @param {number} [options.maxHeight] - Height it grows to as the node is dragged taller.
 * @param {number} [options.minWidth] - The narrowest the panel is drawn in.
 * @returns {object} A panel for `appendInterfaceWidget`, with the hooks the node chains onto.
 */
export function createCurveEditor(node, options = {}) {
  const pointsWidget = String(options.pointsWidget || "curve_points");
  const hasChannels = options.channels !== false;
  const wantsHistogram = options.histogram === true;
  const logName = options.logName || LOG_NAME;
  const height = Number(options.height) > 0 ? Number(options.height) : UI_HEIGHT;
  const maxHeight = Number(options.maxHeight) > 0 ? Number(options.maxHeight) : MAX_UI_HEIGHT;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : MIN_UI_WIDTH;

  const element = document.createElement("div");
  element.style.cssText = "width:100%;height:100%;position:relative;overflow:hidden;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;height:100%;display:block;touch-action:none;";
  element.appendChild(canvas);

  const state = {
    channel: hasChannels ? CHANNELS[0] : SINGLE,
    curves: parseCurves(findWidget(node, pointsWidget)?.value ?? "", hasChannels),
    dragging: -1,
    pointer: -1,
    hover: -1,
    disposed: false,
    painting: false,
    writing: false,
    histogram: null,
  };

  let releaseRatio = null;
  let releaseRun = null;

  /**
   * The channel the plot is drawing.
   *
   * @returns {string} A key of `CHANNELS`, or `SINGLE` where there is one curve.
   */
  function currentChannel() {
    return hasChannels ? state.channel : SINGLE;
  }

  /**
   * Bring a channel to the front.
   *
   * @param {string} name - A key of `CHANNELS`.
   * @returns {void}
   */
  function setChannel(name) {
    if (!CHANNELS.includes(name)) return;
    state.channel = name;
  }

  /**
   * Take the points back off the widget, dropping whatever gesture was in hand.
   *
   * @returns {void}
   */
  function syncFromWidget() {
    state.curves = parseCurves(findWidget(node, pointsWidget)?.value ?? "", hasChannels);
    state.dragging = -1;
    state.hover = -1;
  }

  /**
   * Put the current points back on the widget, as one undo step.
   *
   * @returns {void}
   */
  function commitPoints() {
    const widget = findWidget(node, pointsWidget);
    if (!widget) return;
    const next = isIdentity(state.curves, hasChannels)
      ? ""
      : serialiseCurves(state.curves, hasChannels);
    if (widget.value === next) return;
    // The widget's callback is chained to `handlePointsChanged`, which resyncs from the
    // widget. Held off while the editor is the one writing.
    state.writing = true;
    try {
      withGraphChange(() => {
        widget.value = next;
        widget.callback?.(next);
      });
    } finally {
      state.writing = false;
    }
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Read the levels of the picture this node was last handed.
   *
   * @returns {Promise<void>} Resolves once the plot has the shape to draw, or has given up.
   */
  async function loadHistogram() {
    if (!wantsHistogram) return;
    try {
      const answer = await fetchInputPreview(node);
      if (state.disposed) return;
      if (answer?.state !== PREVIEW_STATE.READY || !answer.image) {
        state.histogram = null;
        schedulePaint();
        return;
      }
      const source = answer.sourceWidth || answer.width || SAMPLE_EDGE;
      const tall = answer.sourceHeight || answer.height || SAMPLE_EDGE;
      const scale = Math.min(1, SAMPLE_EDGE / Math.max(source, tall));
      const rgba = readPixels(answer.image, source * scale, tall * scale);
      state.histogram = rgba ? histogram(rgba, HISTOGRAM_BINS) : null;
    } catch (error) {
      console.error(`[${logName}] Failed to read the input levels:`, error);
      state.histogram = null;
    }
    schedulePaint();
  }

  /**
   * Repaint on the next frame, once however many changes arrive this one.
   *
   * @returns {void}
   */
  function schedulePaint() {
    if (state.painting || state.disposed) return;
    state.painting = true;
    requestAnimationFrame(() => {
      state.painting = false;
      if (!state.disposed) paint();
    });
  }

  /**
   * Convert a stored point to plot pixels.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {number} x - Input level, 0-255.
   * @param {number} y - Output level, 0-255.
   * @returns {{x: number, y: number}} Position on the canvas.
   */
  function toPixels(layout, x, y) {
    return {
      x: layout.plotX + (x / MAX_LEVEL) * layout.plotWidth,
      y: layout.plotY + layout.plotSize - (y / MAX_LEVEL) * layout.plotSize,
    };
  }

  /**
   * Convert plot pixels back to a stored point.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {number} px - Position across the canvas.
   * @param {number} py - Position down the canvas.
   * @returns {Array<number>} Input and output level, both 0-255 and whole.
   */
  function toLevels(layout, px, py) {
    const x = layout.plotWidth ? ((px - layout.plotX) / layout.plotWidth) * MAX_LEVEL : 0;
    const y = layout.plotSize
      ? ((layout.plotY + layout.plotSize - py) / layout.plotSize) * MAX_LEVEL
      : 0;
    return [clamp(Math.round(x), 0, MAX_LEVEL), clamp(Math.round(y), 0, MAX_LEVEL)];
  }

  /**
   * Draw the whole editor.
   *
   * @returns {void}
   */
  function paint() {
    const ratio = surfaceRatio(element);
    const width = Math.max(1, Math.round(element.clientWidth));
    const down = Math.max(1, Math.round(element.clientHeight));
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(down * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(down * ratio);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, down);

    const theme = readTheme();
    const layout = computeLayout(width, down, hasChannels);
    if (hasChannels) drawChannelRow(ctx, layout, theme);
    drawPlot(ctx, layout, theme);
    drawFooter(ctx, layout, theme);
  }

  /**
   * Draw the four channel buttons.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {void}
   */
  function drawChannelRow(ctx, layout, theme) {
    const front = currentChannel();
    const slot = layout.plotWidth / CHANNELS.length;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    CHANNELS.forEach((name, index) => {
      const x = layout.plotX + slot * index;
      const active = name === front;
      const bent = !isLine(state.curves[name]);
      ctx.fillStyle = active ? theme.accent ?? "#4a9eff" : theme.inputBg ?? "#222222";
      ctx.fillRect(x + 1, layout.tabY, slot - 2, TAB_HEIGHT);
      ctx.fillStyle = active ? theme.selectionText ?? "#ffffff" : theme.fg ?? "#cccccc";
      ctx.fillText(
        bent && !active ? `${CHANNEL_LABEL[name]}${ICON?.dot ?? "*"}` : CHANNEL_LABEL[name],
        x + slot / 2,
        layout.tabY + TAB_HEIGHT / 2,
      );
    });
  }

  /**
   * Draw the grid, the diagonal, the curve and its points.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {void}
   */
  function drawPlot(ctx, layout, theme) {
    if (!(layout.plotWidth > 0) || !(layout.plotSize > 0)) return;
    const front = currentChannel();

    ctx.fillStyle = theme.inputBg ?? "#1a1a1a";
    ctx.fillRect(layout.plotX, layout.plotY, layout.plotWidth, layout.plotSize);

    ctx.strokeStyle = theme.border ?? "#333333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let step = 0; step <= GRID_STEPS; step += 1) {
      const x = layout.plotX + (layout.plotWidth * step) / GRID_STEPS;
      const y = layout.plotY + (layout.plotSize * step) / GRID_STEPS;
      ctx.moveTo(Math.round(x) + 0.5, layout.plotY);
      ctx.lineTo(Math.round(x) + 0.5, layout.plotY + layout.plotSize);
      ctx.moveTo(layout.plotX, Math.round(y) + 0.5);
      ctx.lineTo(layout.plotX + layout.plotWidth, Math.round(y) + 0.5);
    }
    ctx.stroke();

    drawHistogram(ctx, layout, front);

    // The untouched response, so a bend is read against where it started.
    ctx.strokeStyle = theme.border ?? "#333333";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    const start = toPixels(layout, 0, 0);
    const end = toPixels(layout, MAX_LEVEL, MAX_LEVEL);
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.setLineDash([]);

    const points = cleanPoints(state.curves[front]);
    const samples = Math.max(2, Math.round(layout.plotWidth));
    const values = curveSamples(points, samples);
    ctx.strokeStyle = CHANNEL_STROKE[front] ?? theme.accent ?? "#4a9eff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < samples; i += 1) {
      const at = toPixels(layout, (i / (samples - 1)) * MAX_LEVEL, clamp(values[i], 0, MAX_LEVEL));
      if (i === 0) ctx.moveTo(at.x, at.y);
      else ctx.lineTo(at.x, at.y);
    }
    ctx.stroke();

    points.forEach(([x, y], index) => {
      const at = toPixels(layout, x, y);
      const held = index === state.dragging || index === state.hover;
      ctx.beginPath();
      ctx.arc(at.x, at.y, held ? POINT_RADIUS + 1.5 : POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = held ? theme.accent ?? "#4a9eff" : theme.bg ?? "#1a1a1a";
      ctx.fill();
      ctx.strokeStyle = CHANNEL_STROKE[front] ?? theme.accent ?? "#4a9eff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  /**
   * Draw the levels of the input picture behind the grid.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {string} front - The channel the plot is drawing.
   * @returns {void}
   */
  function drawHistogram(ctx, layout, front) {
    const counts = state.histogram;
    if (!counts?.peak) return;

    // On a colour channel only that channel is behind the curve being bent; on the composite
    // all three are drawn over each other, which is where a cast shows as a split shoulder.
    const channels = front === "rgb" || !hasChannels
      ? [["r", "#ff5f5f"], ["g", "#5fd65f"], ["b", "#5f9dff"]]
      : [[front, CHANNEL_STROKE[front]]];

    ctx.save();
    ctx.beginPath();
    ctx.rect(layout.plotX, layout.plotY, layout.plotWidth, layout.plotSize);
    ctx.clip();
    ctx.globalCompositeOperation = "lighter";
    for (const [name, colour] of channels) {
      const bins = counts[name];
      if (!bins) continue;
      ctx.fillStyle = colour;
      ctx.globalAlpha = channels.length > 1 ? 0.22 : 0.3;
      ctx.beginPath();
      ctx.moveTo(layout.plotX, layout.plotY + layout.plotSize);
      for (let i = 0; i < bins.length; i += 1) {
        const x = layout.plotX + (i / (bins.length - 1)) * layout.plotWidth;
        const up = (bins[i] / counts.peak) * layout.plotSize * HISTOGRAM_FILL;
        ctx.lineTo(x, layout.plotY + layout.plotSize - up);
      }
      ctx.lineTo(layout.plotX + layout.plotWidth, layout.plotY + layout.plotSize);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  /**
   * Draw the line under the plot.
   *
   * @param {CanvasRenderingContext2D} ctx - Where to draw.
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {void}
   */
  function drawFooter(ctx, layout, theme) {
    const points = cleanPoints(state.curves[currentChannel()]);
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.descripText ?? theme.fg ?? "#999999";
    const held = state.dragging >= 0 ? state.dragging : state.hover;
    const text = held >= 0 && points[held]
      ? `in ${points[held][0]}  out ${points[held][1]}`
      : `${points.length} points`;
    ctx.fillText(text, layout.plotX, layout.footerY + FOOTER_HEIGHT / 2);
  }

  /**
   * The point nearest a position, when one is near enough to grab.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {{x: number, y: number}} at - Pointer position in element pixels.
   * @returns {number} Index into the channel's points, or -1.
   */
  function pointAt(layout, at) {
    const points = cleanPoints(state.curves[currentChannel()]);
    let best = -1;
    let bestDistance = GRAB_RADIUS;
    points.forEach(([x, y], index) => {
      const pixel = toPixels(layout, x, y);
      const distance = Math.hypot(pixel.x - at.x, pixel.y - at.y);
      if (distance <= bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  /**
   * Whether a position is inside the plot.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {{x: number, y: number}} at - Pointer position in element pixels.
   * @returns {boolean} True when the plot would take the gesture.
   */
  function insidePlot(layout, at) {
    return (
      at.x >= layout.plotX && at.x <= layout.plotX + layout.plotWidth &&
      at.y >= layout.plotY && at.y <= layout.plotY + layout.plotSize
    );
  }

  /**
   * Which channel button a position is over.
   *
   * @param {object} layout - Geometry from `computeLayout`.
   * @param {{x: number, y: number}} at - Pointer position in element pixels.
   * @returns {string|null} A channel key, or null.
   */
  function channelAt(layout, at) {
    if (!hasChannels) return null;
    if (at.y < layout.tabY || at.y > layout.tabY + TAB_HEIGHT) return null;
    const slot = layout.plotWidth / CHANNELS.length;
    const index = Math.floor((at.x - layout.plotX) / slot);
    return CHANNELS[index] ?? null;
  }

  /**
   * Give back a pointer capture the editor holds.
   *
   * @param {number} pointerId - The pointer to release.
   * @returns {void}
   */
  function releasePointer(pointerId) {
    if (!(pointerId >= 0)) return;
    if (state.pointer === pointerId) state.pointer = -1;
    if (element.hasPointerCapture?.(pointerId)) element.releasePointerCapture?.(pointerId);
  }

  /**
   * End a drag, writing where it reached or taking it back.
   *
   * @param {boolean} commit - Write the points the drag reached. False reads them back off
   *   the widget, which is what an interrupted gesture leaves.
   * @returns {void}
   */
  function endDrag(commit) {
    const held = state.dragging >= 0;
    const pointerId = state.pointer;
    // Cleared before the capture is given back, since letting go of one raises
    // `lostpointercapture`, which ends the gesture again.
    state.dragging = -1;
    releasePointer(pointerId);
    if (!held) return;
    if (commit) commitPoints();
    else syncFromWidget();
    schedulePaint();
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    const layout = computeLayout(element.clientWidth, element.clientHeight, hasChannels);
    const at = elementPoint(element, event);

    const channel = channelAt(layout, at);
    if (channel) {
      endDrag(true);
      setChannel(channel);
      state.hover = -1;
      schedulePaint();
      return;
    }
    if (!insidePlot(layout, at)) return;

    const front = currentChannel();
    const points = cleanPoints(state.curves[front]);
    const index = pointAt(layout, at);

    if (event.button === 2) {
      // The two ends anchor the curve: removing one would leave the range undefined past it.
      if (index > 0 && index < points.length - 1) {
        points.splice(index, 1);
        state.curves[front] = points;
        // The held index counts into the list this drops a point from.
        state.dragging = -1;
        state.hover = -1;
        commitPoints();
        schedulePaint();
      }
      event.preventDefault();
      return;
    }
    if (event.button !== 0) return;

    if (index >= 0) {
      state.dragging = index;
    } else if (points.length < MAX_POINTS) {
      const added = toLevels(layout, at.x, at.y);
      points.push(added);
      const sorted = cleanPoints(points);
      state.curves[front] = sorted;
      state.dragging = sorted.findIndex(([x]) => x === added[0]);
    }
    state.pointer = event.pointerId;
    element.setPointerCapture?.(event.pointerId);
    event.stopPropagation();
    event.preventDefault();
    schedulePaint();
  };

  const onPointerMove = (event) => {
    const layout = computeLayout(element.clientWidth, element.clientHeight, hasChannels);
    const at = elementPoint(element, event);

    if (state.dragging < 0) {
      const hover = insidePlot(layout, at) ? pointAt(layout, at) : -1;
      if (hover !== state.hover) {
        state.hover = hover;
        schedulePaint();
      }
      return;
    }

    // A button released over another window, or a capture the browser took away, ends the
    // gesture with no pointerup.
    if (!(event.buttons & 1)) {
      endDrag(false);
      return;
    }

    const front = currentChannel();
    const points = cleanPoints(state.curves[front]);
    const moved = toLevels(layout, at.x, at.y);
    const last = points.length - 1;
    // The ends keep their input level, so the curve always spans the whole range and the
    // point cannot be dragged past its neighbours into a fold the spline cannot describe.
    if (state.dragging === 0) moved[0] = 0;
    else if (state.dragging === last) moved[0] = MAX_LEVEL;
    else {
      moved[0] = clamp(moved[0], points[state.dragging - 1][0] + 1, points[state.dragging + 1][0] - 1);
    }
    points[state.dragging] = moved;
    state.curves[front] = points;
    // Written to the widget on the release, so a drag is one undo step rather than one per
    // frame it moved through.
    schedulePaint();
    event.preventDefault();
  };

  const onPointerUp = (event) => {
    if (state.dragging < 0) {
      releasePointer(event.pointerId);
      return;
    }
    endDrag(true);
  };

  const onPointerCancel = () => {
    endDrag(false);
  };

  const onPointerLeave = () => {
    if (state.hover === -1) return;
    state.hover = -1;
    schedulePaint();
  };

  const onDoubleClick = (event) => {
    const layout = computeLayout(element.clientWidth, element.clientHeight, hasChannels);
    const at = elementPoint(element, event);
    if (!insidePlot(layout, at)) return;
    state.curves[currentChannel()] = straightLine();
    state.dragging = -1;
    state.hover = -1;
    commitPoints();
    schedulePaint();
    event.preventDefault();
  };

  const onContextMenu = (event) => {
    const layout = computeLayout(element.clientWidth, element.clientHeight, hasChannels);
    const at = elementPoint(element, event);
    if (insidePlot(layout, at)) event.preventDefault();
  };

  element.addEventListener("pointerdown", onPointerDown);
  element.addEventListener("pointermove", onPointerMove);
  element.addEventListener("pointerup", onPointerUp);
  element.addEventListener("pointercancel", onPointerCancel);
  element.addEventListener("lostpointercapture", onPointerCancel);
  element.addEventListener("pointerleave", onPointerLeave);
  element.addEventListener("dblclick", onDoubleClick);
  element.addEventListener("contextmenu", onContextMenu);
  // The editor scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(element);

  try {
    releaseRatio = watchSurfaceRatio(element, schedulePaint);
  } catch (error) {
    console.error(`[${logName}] Failed to watch the drawing resolution:`, error);
  }
  // The curve is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let releaseTheme = onThemeChange(schedulePaint);
  if (wantsHistogram) {
    try {
      releaseRun = onRunEnded(() => { loadHistogram(); });
    } catch (error) {
      console.error(`[${logName}] Failed to watch for the end of a run:`, error);
    }
  }

  return {
    element,
    height,
    maxHeight,
    minWidth,
    schedulePaint,
    loadHistogram,
    handlePointsChanged() {
      if (state.writing) return;
      syncFromWidget();
      schedulePaint();
    },
    dispose() {
      state.disposed = true;
      element.removeEventListener("pointerdown", onPointerDown);
      element.removeEventListener("pointermove", onPointerMove);
      element.removeEventListener("pointerup", onPointerUp);
      element.removeEventListener("pointercancel", onPointerCancel);
      element.removeEventListener("lostpointercapture", onPointerCancel);
      element.removeEventListener("pointerleave", onPointerLeave);
      element.removeEventListener("dblclick", onDoubleClick);
      element.removeEventListener("contextmenu", onContextMenu);
      releaseWheel();
      try {
        releaseRatio?.();
        releaseRun?.();
        releaseTheme?.();
      } catch (error) {
        console.error(`[${logName}] Failed to release the editor's watches:`, error);
      }
      releaseRatio = null;
      releaseRun = null;
    },
  };
}

/**
 * Chain a repaint onto a widget's own callback.
 *
 * @param {object} node - The node the widget belongs to.
 * @param {string} name - Widget name.
 * @param {Function} onChange - What to run after the widget's own callback.
 * @param {string} logName - What a console error is prefixed with.
 * @returns {void}
 */
function chainWidgetCallback(node, name, onChange, logName) {
  const widget = findWidget(node, name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${logName}] Failed to repaint after a widget change:`, error);
    }
    return result;
  };
}

/**
 * Put a curve editor on a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @param {object} [options] - Everything `createCurveEditor` takes.
 * @returns {object|null} The editor, or null when the node holds no curve widget to write.
 */
export function mountCurveEditor(node, options = {}) {
  const pointsWidget = String(options.pointsWidget || "curve_points");
  const logName = options.logName || LOG_NAME;
  if (!findWidget(node, pointsWidget)) return null;

  const editor = createCurveEditor(node, options);
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
  // A multiline box left uncapped takes the room a node is dragged taller for, and the plot
  // beside it never grows.
  boundTextBoxes(node);
  chainWidgetCallback(node, pointsWidget, editor.handlePointsChanged, logName);

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      editor.handlePointsChanged();
    } catch (error) {
      console.error(`[${logName}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      editor.dispose();
    } catch (error) {
      console.error(`[${logName}] Failed to release the curve editor:`, error);
    }
    return result;
  };

  editor.schedulePaint();
  // A node added to a graph that has already run has levels waiting for it.
  editor.loadHistogram();
  return editor;
}
