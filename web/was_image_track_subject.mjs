/**
 * The frames Image Track Subject read, with the matte it tracked drawn over each, a scrubber
 * through them, the text candidates found on the key frame and the points placed on it.
 */

import { app } from "../../scripts/app.js";
import { findWidget } from "./interface/audio_view.mjs";
import { CARD_LOOK, CHIP_TONES, cardChip, cardRule } from "./interface/card.mjs";
import { ICON, drawFidelityGlyph, ICON_SIZE } from "./interface/icons.mjs";
import { drawStandIn, loadPlaceholder, standInDetail } from "./interface/placeholder.mjs";
import { captureWheel, elementPoint } from "./interface/pointer.mjs";
import { LABELS, PREVIEW_STATE, fetchInputPreview, fetchOutputPreview } from "./interface/preview.mjs";
import { withGraphChange } from "./interface/region.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onNodeFinished } from "./interface/run_events.mjs";
import { fetchRunResult, whenReadable } from "./interface/run_result.mjs";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.mjs";
import { SANS, el } from "./interface/window_kit.mjs";

const EXT_NAME = "WASNodeSuite.ImageTrackSubjectUI";
const NODE_ID = "WASImageTrackSubject";

const UI_WIDGET_NAME = "was_track_subject_ui";
const UI_WIDGET_TYPE = "was_track_subject";

// The slots the node files its frames and mattes under.
const FRAMES_SLOT = "images";
const MATTE_SLOT = "mask";

// The widgets the panel reads and writes.
const KEY_FRAME = "key_frame";
const PICK = "pick";
const POINTS = "points";
const REPAINT_ON = ["subject", KEY_FRAME, PICK, POINTS, "track", "grow", "feather"];

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 420;
const PANEL_WIDTH = 360;

// Height of the scrubber strip, in CSS pixels.
const STRIP_HEIGHT = 40;

// Narrowest a thumbnail is drawn, and the fewest that must fit before the slim track takes over.
const LEAST_CELL = 36;
const LEAST_CELLS = 4;

// Height of the coverage band under the thumbnails and the gap above it.
const BAND_HEIGHT = 3;
const BAND_GAP = 2;

// Height of the slim track, and the least room between two of its ticks.
const TRACK_HEIGHT = 6;
const LEAST_TICK = 6;

// Side of the reduced copy a matte's coverage is measured on.
const COVERAGE_SIDE = 32;

// Room left round the picture inside its well, in CSS pixels.
const MARGIN = 6;

// Size of a point marker and how near a click has to land to remove one, in CSS pixels.
const MARKER = 5;
const HIT = 9;

// How far a press may move and still count as a click, in CSS pixels.
const CLICK_SLOP = 4;

// Opacity of the matte's tint, and of its outline.
const TINT_ALPHA = 0.45;
const EDGE_ALPHA = 0.95;

// Requests for frames and mattes in flight at once.
const CONCURRENCY = 3;

// The most candidates and points read out of a report or a widget.
const MAX_CANDIDATES = 32;
const MAX_POINTS = 64;

// Decimals a written point keeps.
const POINT_DECIMALS = 4;

// The two ways a click on the picture is read.
const MODE = { PICK: "pick", POINTS: "points" };

// The sources a report may name, and nothing else is drawn.
const SOURCES = new Set(["points", "text", "mask"]);

/**
 * A number held inside a range.
 *
 * @param {number} value - The number.
 * @param {number} low - The least it may be.
 * @param {number} high - The most it may be.
 * @returns {number} The number, held to the range.
 */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * A JSON value from report or widget text, or null where it is not JSON.
 *
 * @param {*} text - The text.
 * @returns {*} The parsed value, or null.
 */
function parsed(text) {
  if (typeof text !== "string" || !text.trim()) return null;
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

/**
 * The candidates a report names, each with a box in shares of the key frame.
 *
 * @param {*} text - The `candidates` fact.
 * @returns {Array<{index: number, box: number[], score: number}>} The readable ones, at most
 *   `MAX_CANDIDATES`.
 */
function readCandidates(text) {
  const list = parsed(text);
  if (!Array.isArray(list)) return [];
  const kept = [];
  for (const entry of list.slice(0, MAX_CANDIDATES)) {
    const box = Array.isArray(entry?.box) ? entry.box.map(Number) : [];
    const index = Number(entry?.index);
    if (box.length !== 4 || !box.every(Number.isFinite) || !Number.isInteger(index)) continue;
    const [x0, y0, x1, y1] = box.map((value) => clamp(value, 0, 1));
    if (x1 <= x0 || y1 <= y0) continue;
    kept.push({ index, box: [x0, y0, x1, y1], score: Number(entry?.score) || 0 });
  }
  return kept;
}

/**
 * The batch frame each stored frame stands for.
 *
 * @param {*} text - The `shown` fact.
 * @param {number} held - How many frames the store holds.
 * @returns {number[]} One batch index per stored frame; the stored order itself where the
 *   report names none or names something unreadable.
 */
function readShown(text, held) {
  const list = parsed(text);
  const identity = Array.from({ length: held }, (_, index) => index);
  if (!Array.isArray(list) || list.length < held) return identity;
  const indices = list.slice(0, held).map(Number);
  return indices.every((value) => Number.isInteger(value) && value >= 0) ? indices : identity;
}

/**
 * The points a widget holds.
 *
 * @param {*} value - The `points` widget's value.
 * @returns {{points: Array<{x: number, y: number, keep: boolean}>, readable: boolean}} The points,
 *   and whether the text could be read whole, which is what allows the panel to rewrite it.
 */
function readPoints(value) {
  const text = String(value ?? "").trim();
  if (!text) return { points: [], readable: true };
  const list = parsed(text);
  if (!Array.isArray(list) || list.length > MAX_POINTS) return { points: [], readable: false };
  const points = [];
  for (const entry of list) {
    const x = Number(entry?.x);
    const y = Number(entry?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) {
      return { points: [], readable: false };
    }
    points.push({ x, y, keep: entry?.keep !== false });
  }
  return { points, readable: true };
}

/**
 * Points as the text the `points` widget holds.
 *
 * @param {Array<{x: number, y: number, keep: boolean}>} points - The points.
 * @returns {string} JSON, or an empty string for no points.
 */
function writePoints(points) {
  if (!points.length) return "";
  const round = (value) => Number(value.toFixed(POINT_DECIMALS));
  return JSON.stringify(points.map((point) => ({ x: round(point.x), y: round(point.y), keep: point.keep })));
}

/**
 * The red, green and blue a CSS colour resolves to.
 *
 * @param {string} colour - Any colour a canvas accepts.
 * @returns {number[]} Three channels, 0 to 255.
 */
function channels(colour) {
  const probe = document.createElement("canvas");
  probe.width = 1;
  probe.height = 1;
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [255, 160, 60];
  ctx.fillStyle = colour;
  ctx.fillRect(0, 0, 1, 1);
  const data = ctx.getImageData(0, 0, 1, 1).data;
  return [data[0], data[1], data[2]];
}

/**
 * A matte drawn as a tint inside it and a line round its edge, at the matte's own size.
 *
 * @param {HTMLImageElement} matte - The published matte, grey where the subject is.
 * @param {number[]} tint - Channels of the colour it is drawn in.
 * @returns {{canvas: HTMLCanvasElement, covered: number}|null} The overlay and the share of the
 *   frame it covers, or null when it cannot be read.
 */
function buildOverlay(matte, tint) {
  const width = matte.naturalWidth || matte.width;
  const height = matte.naturalHeight || matte.height;
  if (!(width > 0) || !(height > 0)) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(matte, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height);
  const data = pixels.data;
  const inside = new Uint8Array(width * height);
  let count = 0;
  for (let index = 0; index < width * height; index += 1) {
    const level = data[index * 4];
    inside[index] = level >= 128 ? 1 : 0;
    count += inside[index];
    data[index * 4] = tint[0];
    data[index * 4 + 1] = tint[1];
    data[index * 4 + 2] = tint[2];
    data[index * 4 + 3] = Math.round(level * TINT_ALPHA);
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!inside[index]) continue;
      const edge = x === 0 || y === 0 || x === width - 1 || y === height - 1
        || !inside[index - 1] || !inside[index + 1] || !inside[index - width] || !inside[index + width];
      if (edge) data[index * 4 + 3] = Math.round(255 * EDGE_ALPHA);
    }
  }
  ctx.putImageData(pixels, 0, 0);
  return { canvas, covered: count / (width * height) };
}

/**
 * A small rounded button in the card's colours.
 *
 * @param {string} label - Its words.
 * @param {string} hint - Hover text.
 * @param {() => void} onPress - Called on a press.
 * @returns {{element: HTMLButtonElement, setOn: (on: boolean) => void}} The button, and a way to
 *   show it switched on.
 */
function pillButton(label, hint, onPress) {
  const press = document.createElement("button");
  press.type = "button";
  press.textContent = label;
  press.title = hint;
  let on = false;
  const style = (hovered) => {
    const tone = on ? CHIP_TONES.accent : null;
    press.style.cssText = `flex:none;padding:2px 10px;border-radius:999px;cursor:pointer;white-space:nowrap;`
      + `font:600 11px/18px ${SANS};border:1px solid ${hovered || on ? CARD_LOOK.rule : CARD_LOOK.edge};`
      + `background:${tone ? tone.background : CARD_LOOK.inset};`
      + `color:${tone ? tone.color : hovered ? CARD_LOOK.title : CARD_LOOK.muted}`;
  };
  style(false);
  press.onmouseenter = () => style(true);
  press.onmouseleave = () => style(false);
  press.addEventListener("pointerdown", (event) => event.stopPropagation());
  press.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      onPress();
    } catch (error) {
      console.error(`[${EXT_NAME}] ${label} failed:`, error);
    }
  });
  return {
    element: press,
    setOn(next) {
      on = !!next;
      style(false);
    },
  };
}

/**
 * Build the panel for one node.
 *
 * @param {object} node - The Image Track Subject node.
 * @returns {object} The panel `appendInterfaceWidget` hosts.
 */
function createPanel(node) {
  let disposed = false;
  let serial = 0;
  let frameHandle = 0;
  let mode = MODE.POINTS;
  let current = 0;
  let pictureState = PREVIEW_STATE.WAITING;
  let promptId = "";
  let held = 0;
  let total = 0;
  let shown = [];
  let report = null;
  let press = null;
  let scrubbing = false;
  let tint = null;

  const frames = new Map();
  const mattes = new Map();
  const overlays = new Map();
  const coverage = new Map();
  const queue = [];
  let inflight = 0;

  const root = el("div", "position:relative;box-sizing:border-box;width:100%;height:100%;display:flex;"
    + "flex-direction:column;gap:6px;padding:7px 8px 8px;min-width:0;min-height:0;outline:none;"
    + `font:11px/1.35 ${SANS};user-select:none`);
  root.tabIndex = 0;

  const header = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0");
  const title = el("span", `flex:none;font:650 12px/1.4 ${SANS};color:${CARD_LOOK.title};white-space:nowrap`,
    "Track Subject");
  const chips = el("div", "flex:1 1 auto;display:flex;flex-wrap:wrap;gap:4px;min-width:0");
  const controls = el("div", "flex:0 1 auto;display:flex;flex-wrap:wrap;gap:4px;align-items:center;min-width:0");
  const segment = el("div", `flex:none;display:flex;gap:2px;padding:2px;border-radius:999px;`
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well}`);
  const pickButton = pillButton("Pick", "Click a numbered find", () => setMode(MODE.PICK));
  const pointsButton = pillButton("Points", "Shift-click excludes", () => setMode(MODE.POINTS));
  segment.append(pickButton.element, pointsButton.element);
  const clearButton = pillButton("Clear points", "Remove every point", () => {
    write({ [POINTS]: "" });
  });
  const keyButton = pillButton("Key frame here", "Pick on this frame", () => {
    const batch = batchIndex(current);
    if (batch !== null) write({ [KEY_FRAME]: batch, [POINTS]: "" });
  });
  controls.append(segment, clearButton.element, keyButton.element);
  header.append(title, chips, controls);

  const stage = el("div", "position:relative;flex:1 1 auto;min-height:0;min-width:0;border-radius:6px;"
    + `overflow:hidden;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:crosshair`);
  const canvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  stage.append(canvas);

  const stripRow = el("div", `flex:0 0 ${STRIP_HEIGHT}px;display:flex;align-items:center;gap:4px;min-width:0`);
  const strip = el("div", "position:relative;flex:1 1 auto;align-self:stretch;min-width:0;border-radius:6px;"
    + `overflow:hidden;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:ew-resize`);
  const stripCanvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  strip.append(stripCanvas);
  const backButton = pillButton("‹", "Previous frame", () => showFrame(current - 1));
  const forwardButton = pillButton("›", "Next frame", () => showFrame(current + 1));
  stripRow.append(backButton.element, strip, forwardButton.element);

  const footer = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:6px;min-width:0");
  const glyph = el("canvas", `flex:none;width:${ICON_SIZE}px;height:${ICON_SIZE}px`);
  const status = el("span", `flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;`
    + `white-space:nowrap;color:${CARD_LOOK.muted}`);
  footer.append(glyph, status);

  root.append(header, cardRule(), stage, stripRow, footer);

  /**
   * The batch frame a stored frame stands for.
   *
   * @param {number} index - A stored frame.
   * @returns {number|null} Its batch index, or null where nothing is stored there.
   */
  function batchIndex(index) {
    return index >= 0 && index < shown.length ? shown[index] : null;
  }

  /**
   * A number a widget holds.
   *
   * @param {string} name - The widget's name.
   * @returns {number} The value, or NaN.
   */
  function widgetNumber(name) {
    return Number(findWidget(node, name)?.value);
  }

  /**
   * A fact the last run reported.
   *
   * @param {string} name - The fact's name.
   * @returns {string} Its text, or an empty string.
   */
  function fact(name) {
    return report?.facts.get(name) ?? "";
  }

  /**
   * A count the last run reported.
   *
   * @param {string} name - The count's name.
   * @returns {number} Its value, or NaN.
   */
  function count(name) {
    return Number(report?.counts.get(name));
  }

  /**
   * The key frame the last run picked its subject on.
   *
   * @returns {number|null} A batch index, or null before a run.
   */
  function runKeyFrame() {
    const value = Number(fact(KEY_FRAME) || count(KEY_FRAME));
    return Number.isInteger(value) ? value : null;
  }

  /**
   * Write widgets as one undoable edit.
   *
   * @param {Object<string, *>} values - Widget name to value.
   * @returns {boolean} True when any widget took a new value.
   */
  function write(values) {
    const writes = [];
    for (const [name, raw] of Object.entries(values)) {
      const widget = findWidget(node, name);
      if (!widget) continue;
      let value = raw;
      if (typeof raw === "number") {
        const low = Number(widget.options?.min);
        const high = Number(widget.options?.max);
        if (Number.isFinite(low)) value = Math.max(low, value);
        if (Number.isFinite(high)) value = Math.min(high, value);
      }
      if (widget.value !== value) writes.push([widget, value]);
    }
    if (!writes.length) return false;
    withGraphChange(() => {
      for (const [widget, value] of writes) {
        widget.value = value;
        widget.callback?.(value);
      }
    });
    node.setDirtyCanvas?.(true, true);
    redraw();
    return true;
  }

  /**
   * Choose how a click on the picture is read.
   *
   * @param {string} next - A value of `MODE`.
   * @returns {void}
   */
  function setMode(next) {
    mode = next;
    pickButton.setOn(mode === MODE.PICK);
    pointsButton.setOn(mode === MODE.POINTS);
    stage.style.cursor = mode === MODE.PICK ? "pointer" : "crosshair";
    redraw();
  }

  /**
   * Repaint once, on the next frame.
   *
   * @returns {void}
   */
  function redraw() {
    if (disposed || frameHandle) return;
    frameHandle = requestAnimationFrame(() => {
      frameHandle = 0;
      try {
        paint();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the panel:`, error);
      }
    });
  }

  /**
   * Size a canvas to its element at the graph's zoom.
   *
   * @param {HTMLCanvasElement} target - The canvas.
   * @returns {{ctx: CanvasRenderingContext2D, width: number, height: number}|null} Its context,
   *   scaled to layout pixels, and its size in them.
   */
  function sized(target) {
    const width = target.clientWidth;
    const height = target.clientHeight;
    if (!(width > 0) || !(height > 0)) return null;
    const ratio = surfaceRatio(target);
    const backingWidth = Math.max(1, Math.round(width * ratio));
    const backingHeight = Math.max(1, Math.round(height * ratio));
    if (target.width !== backingWidth) target.width = backingWidth;
    if (target.height !== backingHeight) target.height = backingHeight;
    const ctx = target.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  /**
   * Where the shown frame sits inside the well.
   *
   * @returns {{x: number, y: number, width: number, height: number}} The rectangle, in layout
   *   pixels, fitted to the frame's own shape.
   */
  function pictureRect() {
    const image = frames.get(current)?.image;
    const across = image?.naturalWidth || 16;
    const down = image?.naturalHeight || 9;
    const width = Math.max(1, canvas.clientWidth - MARGIN * 2);
    const height = Math.max(1, canvas.clientHeight - MARGIN * 2);
    const fit = Math.min(width / across, height / down);
    const drawWidth = across * fit;
    const drawHeight = down * fit;
    return {
      x: MARGIN + (width - drawWidth) / 2,
      y: MARGIN + (height - drawHeight) / 2,
      width: drawWidth,
      height: drawHeight,
    };
  }

  /**
   * The overlay for one stored frame's matte, built once per palette.
   *
   * @param {number} index - A stored frame.
   * @returns {{canvas: HTMLCanvasElement, covered: number}|null} The overlay, or null.
   */
  function overlayOf(index) {
    if (overlays.has(index)) return overlays.get(index);
    const matte = mattes.get(index)?.image;
    if (!matte) return null;
    tint ??= channels(readTheme().warning);
    const overlay = buildOverlay(matte, tint);
    overlays.set(index, overlay);
    return overlay;
  }

  /**
   * Draw the picture well.
   *
   * @param {object} theme - The palette.
   * @returns {{icon: string, detail: string}} What the well claims about its fidelity.
   */
  function paintStage(theme) {
    const surface = sized(canvas);
    if (!surface) return { icon: ICON.WARNING, detail: standInDetail(pictureState) };
    const { ctx } = surface;
    const rect = pictureRect();
    const image = frames.get(current)?.image;
    if (!image) {
      const state = frames.get(current)?.state ?? pictureState;
      drawStandIn(ctx, { x: rect.x, y: rect.y, w: rect.width, h: rect.height });
      const words = state === PREVIEW_STATE.WAITING ? "Run once to load frames" : LABELS[state] ?? "";
      if (words) {
        ctx.fillStyle = theme.fgMuted;
        ctx.font = `600 12px ${SANS}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(words, rect.x + rect.width / 2, rect.y + rect.height - 6);
      }
      return { icon: ICON.WARNING, detail: standInDetail(state) };
    }
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, rect.x, rect.y, rect.width, rect.height);
    const overlay = overlayOf(current);
    if (overlay) ctx.drawImage(overlay.canvas, rect.x, rect.y, rect.width, rect.height);

    const batch = batchIndex(current);
    const runKey = runKeyFrame();
    if (batch !== null && batch === runKey) paintCandidates(ctx, rect, theme);
    if (batch !== null && batch === widgetNumber(KEY_FRAME)) paintPoints(ctx, rect, theme);
    return { icon: ICON.EXACT, detail: overlay ? "tracked matte" : "input frame" };
  }

  /**
   * Draw the numbered text candidates found on the run's key frame.
   *
   * @param {CanvasRenderingContext2D} ctx - The well's context.
   * @param {{x: number, y: number, width: number, height: number}} rect - Where the frame is.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintCandidates(ctx, rect, theme) {
    const candidates = readCandidates(fact("candidates"));
    if (!candidates.length) return;
    const picked = Number(fact("picked"));
    const usingPoints = readPoints(findWidget(node, POINTS)?.value).points.length > 0;
    ctx.save();
    ctx.font = `700 10px ${SANS}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    for (const candidate of candidates) {
      const [x0, y0, x1, y1] = candidate.box;
      const chosen = !usingPoints && candidate.index === picked;
      const left = rect.x + x0 * rect.width;
      const top = rect.y + y0 * rect.height;
      ctx.lineWidth = chosen ? 2 : 1;
      ctx.setLineDash(chosen ? [] : [4, 3]);
      ctx.strokeStyle = chosen ? theme.accent : theme.fgMuted;
      ctx.strokeRect(left, top, (x1 - x0) * rect.width, (y1 - y0) * rect.height);
      ctx.setLineDash([]);
      const label = String(candidate.index);
      const side = Math.max(14, ctx.measureText(label).width + 8);
      ctx.fillStyle = chosen ? theme.accent : theme.bgDark;
      ctx.fillRect(left, top, side, 14);
      ctx.fillStyle = chosen ? theme.bgDark : theme.fg;
      ctx.fillText(label, left + side / 2, top + 7.5);
    }
    ctx.restore();
  }

  /**
   * Draw the points the widget holds, a cross for one kept and an x for one excluded.
   *
   * @param {CanvasRenderingContext2D} ctx - The well's context.
   * @param {{x: number, y: number, width: number, height: number}} rect - Where the frame is.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintPoints(ctx, rect, theme) {
    const { points } = readPoints(findWidget(node, POINTS)?.value);
    ctx.save();
    ctx.lineCap = "round";
    for (const point of points) {
      const x = rect.x + point.x * rect.width;
      const y = rect.y + point.y * rect.height;
      const arms = point.keep
        ? [[x - MARKER, y, x + MARKER, y], [x, y - MARKER, x, y + MARKER]]
        : [[x - MARKER, y - MARKER, x + MARKER, y + MARKER], [x - MARKER, y + MARKER, x + MARKER, y - MARKER]];
      for (const [width, colour] of [[4, theme.bgDark], [2, point.keep ? theme.success : theme.error]]) {
        ctx.lineWidth = width;
        ctx.strokeStyle = colour;
        for (const [ax, ay, bx, by] of arms) {
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  /**
   * The share of one stored frame's matte that holds the subject, measured once.
   *
   * @param {number} index - A stored frame.
   * @returns {number|null} 0 to 1, or null while its matte has not arrived.
   */
  function coverageOf(index) {
    if (coverage.has(index)) return coverage.get(index);
    const matte = mattes.get(index)?.image;
    if (!matte) return null;
    const probe = document.createElement("canvas");
    probe.width = COVERAGE_SIDE;
    probe.height = COVERAGE_SIDE;
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(matte, 0, 0, COVERAGE_SIDE, COVERAGE_SIDE);
    const data = ctx.getImageData(0, 0, COVERAGE_SIDE, COVERAGE_SIDE).data;
    let set = 0;
    for (let at = 0; at < data.length; at += 4) set += data[at] >= 128 ? 1 : 0;
    const share = set / (COVERAGE_SIDE * COVERAGE_SIDE);
    coverage.set(index, share);
    return share;
  }

  /**
   * The width one thumbnail takes at a given height, at the shown frames' own shape.
   *
   * @param {number} height - The thumbnail's height, in layout pixels.
   * @returns {number} Its width, never under `LEAST_CELL`.
   */
  function cellWidth(height) {
    const image = frames.get(current)?.image ?? frames.get(0)?.image;
    const aspect = image ? (image.naturalWidth || 16) / (image.naturalHeight || 9) : 16 / 9;
    return Math.max(LEAST_CELL, height * aspect);
  }

  /**
   * Whether the timeline is drawn as a slim track rather than as thumbnails.
   *
   * @returns {boolean} True when fewer than `LEAST_CELLS` thumbnails fit the row.
   */
  function compactTimeline() {
    const width = stripRow.clientWidth;
    if (!(width > 0)) return false;
    return Math.floor(width / cellWidth(STRIP_HEIGHT - BAND_HEIGHT - BAND_GAP)) < LEAST_CELLS;
  }

  /**
   * Draw each stored frame's share of the track, tinted where its matte holds the subject.
   *
   * @param {CanvasRenderingContext2D} ctx - The strip's context.
   * @param {number} x - Left edge, in layout pixels.
   * @param {number} y - Top edge.
   * @param {number} width - Width of the band.
   * @param {number} height - Height of the band.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintCoverage(ctx, x, y, width, height, theme) {
    if (!held) return;
    const slice = width / held;
    ctx.save();
    ctx.fillStyle = theme.warning;
    for (let index = 0; index < held; index += 1) {
      const share = coverageOf(index);
      if (!(share > 0)) continue;
      ctx.fillRect(x + index * slice, y, Math.max(1, Math.ceil(slice)), height);
    }
    ctx.restore();
  }

  /**
   * Draw the key frame marker and the playhead over a span of the strip.
   *
   * @param {CanvasRenderingContext2D} ctx - The strip's context.
   * @param {number} x - Left edge of the span, in layout pixels.
   * @param {number} width - Width of the span.
   * @param {number} top - Where the marker hangs from.
   * @param {number} bottom - Where the playhead ends.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintMarkers(ctx, x, width, top, bottom, theme) {
    const slice = width / Math.max(1, held);
    const key = nearestStored(widgetNumber(KEY_FRAME));
    if (key >= 0) {
      const centre = x + (key + 0.5) * slice;
      ctx.fillStyle = theme.accent;
      ctx.beginPath();
      ctx.moveTo(centre - 5, top);
      ctx.lineTo(centre + 5, top);
      ctx.lineTo(centre, top + 7);
      ctx.closePath();
      ctx.fill();
    }
    const head = x + (current + 0.5) * slice;
    ctx.fillStyle = theme.fg;
    ctx.fillRect(Math.round(head) - 1, top, 2, bottom - top);
  }

  /**
   * Draw the timeline as thumbnails at their own shape, as many as fit, over a coverage band.
   *
   * @param {CanvasRenderingContext2D} ctx - The strip's context.
   * @param {number} width - The strip's width, in layout pixels.
   * @param {number} height - The strip's height.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintThumbnails(ctx, width, height, theme) {
    const thumbHeight = height - BAND_HEIGHT - BAND_GAP;
    const cell = cellWidth(thumbHeight);
    const slots = clamp(Math.floor(width / cell), 1, held);
    const slot = width / slots;
    const key = nearestStored(widgetNumber(KEY_FRAME));
    for (let place = 0; place < slots; place += 1) {
      const first = Math.floor((place * held) / slots);
      const last = Math.max(first, Math.floor(((place + 1) * held) / slots) - 1);
      const index = key >= first && key <= last ? key : Math.floor((first + last) / 2);
      const left = place * slot + (slot - cell) / 2;
      const image = frames.get(index)?.image;
      if (image) {
        const across = image.naturalWidth;
        const down = image.naturalHeight;
        const shownAcross = Math.min(across, down * (cell / thumbHeight));
        const shownDown = Math.min(down, across * (thumbHeight / cell));
        ctx.drawImage(image, (across - shownAcross) / 2, (down - shownDown) / 2, shownAcross, shownDown,
          left, 0, cell, thumbHeight);
      } else {
        ctx.fillStyle = theme.bgDark;
        ctx.fillRect(left, 0, cell, thumbHeight);
      }
      if (current >= first && current <= last) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = theme.fg;
        ctx.strokeRect(left + 1, 1, cell - 2, thumbHeight - 2);
      }
    }
    paintCoverage(ctx, 0, height - BAND_HEIGHT, width, BAND_HEIGHT, theme);
    paintMarkers(ctx, 0, width, 0, height, theme);
  }

  /**
   * Draw the timeline as a slim track: coverage, ticks, key frame, playhead and the frame label.
   *
   * @param {CanvasRenderingContext2D} ctx - The strip's context.
   * @param {number} width - The strip's width, in layout pixels.
   * @param {number} height - The strip's height.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintTrack(ctx, width, height, theme) {
    const batch = batchIndex(current);
    const frameCount = Number.isFinite(count("frames")) ? count("frames") : total;
    const label = batch === null ? "" : `f ${batch} / ${frameCount}`;
    ctx.fillStyle = theme.fgMuted;
    ctx.font = `600 10px ${SANS}`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(label, 2, 1);
    const trackTop = height - TRACK_HEIGHT - 4;
    ctx.fillStyle = theme.bgDark;
    ctx.fillRect(0, trackTop, width, TRACK_HEIGHT);
    paintCoverage(ctx, 0, trackTop, width, TRACK_HEIGHT, theme);
    const slice = width / Math.max(1, held);
    const every = Math.max(1, Math.ceil(LEAST_TICK / slice));
    ctx.fillStyle = theme.fgMuted;
    for (let index = 0; index < held; index += every) {
      ctx.fillRect(Math.round((index + 0.5) * slice), trackTop - 3, 1, 3);
    }
    paintMarkers(ctx, 0, width, trackTop - 9, height - 2, theme);
  }

  /**
   * Draw the timeline in whichever layout its width allows.
   *
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintStrip(theme) {
    const compact = compactTimeline();
    const buttons = compact ? "" : "none";
    if (backButton.element.style.display !== buttons) backButton.element.style.display = buttons;
    if (forwardButton.element.style.display !== buttons) forwardButton.element.style.display = buttons;
    const surface = sized(stripCanvas);
    if (!surface || !held) return;
    const { ctx, width, height } = surface;
    if (compact) paintTrack(ctx, width, height, theme);
    else paintThumbnails(ctx, width, height, theme);
  }

  /**
   * The stored frame nearest a batch frame.
   *
   * @param {number} batch - A batch index.
   * @returns {number} The stored frame whose batch index is closest, or -1 with none stored.
   */
  function nearestStored(batch) {
    if (!Number.isFinite(batch) || !shown.length) return -1;
    let best = -1;
    let distance = Infinity;
    shown.forEach((value, index) => {
      const gap = Math.abs(value - batch);
      if (gap < distance) {
        distance = gap;
        best = index;
      }
    });
    return best;
  }

  /**
   * Fill the header's chips from the report and the shown frame.
   *
   * @returns {void}
   */
  function paintChips() {
    chips.replaceChildren();
    const frameCount = Number.isFinite(count("frames")) ? count("frames") : total;
    const batch = batchIndex(current);
    if (batch !== null) chips.append(cardChip(`frame ${batch} / ${frameCount}`, "accent"));
    const tracked = count("tracked");
    if (Number.isFinite(tracked) && frameCount > 0) {
      chips.append(cardChip(`tracked ${tracked} of ${frameCount}`, tracked > 0 ? "success" : "warning"));
    }
    const source = fact("source");
    if (SOURCES.has(source)) chips.append(cardChip(source));
    if (!readPoints(findWidget(node, POINTS)?.value).readable) {
      chips.append(cardChip("points unreadable", "warning"));
    }
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    if (disposed) return;
    const theme = readTheme();
    const claim = paintStage(theme);
    paintStrip(theme);
    paintChips();
    drawFidelityGlyph(glyph, claim);
    const key = widgetNumber(KEY_FRAME);
    const parts = [];
    if (Number.isFinite(key)) parts.push(`key frame ${key}`);
    if (held && total > held) parts.push(`${held} of ${total} shown`);
    status.textContent = parts.join("  ·  ");
    keyButton.element.disabled = batchIndex(current) === null;
  }

  /**
   * Ask for a stored frame and its matte, ahead of anything already waiting.
   *
   * @param {number} index - A stored frame.
   * @returns {void}
   */
  function wantNow(index) {
    queue.unshift({ kind: MATTE_SLOT, index }, { kind: FRAMES_SLOT, index });
    pump();
  }

  /**
   * Start waiting requests, up to `CONCURRENCY` at once.
   *
   * @returns {void}
   */
  function pump() {
    while (!disposed && inflight < CONCURRENCY && queue.length) {
      const job = queue.shift();
      const store = job.kind === FRAMES_SLOT ? frames : mattes;
      if (store.has(job.index) || job.index < 0 || job.index >= held) continue;
      store.set(job.index, { state: PREVIEW_STATE.LOADING, image: null });
      inflight += 1;
      const ticket = serial;
      const ask = job.kind === FRAMES_SLOT
        ? fetchInputPreview(node, FRAMES_SLOT, job.index, promptId)
        : fetchOutputPreview(node, MATTE_SLOT, job.index, promptId);
      ask.then((answer) => {
        if (ticket !== serial || disposed) return;
        const ready = answer?.state === PREVIEW_STATE.READY;
        store.set(job.index, { state: answer?.state ?? PREVIEW_STATE.FAILED, image: ready ? answer.image : null });
        if (job.kind === MATTE_SLOT) {
          overlays.delete(job.index);
          coverage.delete(job.index);
        }
        redraw();
      }).catch((error) => {
        console.error(`[${EXT_NAME}] Failed to read a frame:`, error);
        if (ticket === serial) store.set(job.index, { state: PREVIEW_STATE.FAILED, image: null });
      }).finally(() => {
        if (ticket === serial) inflight -= 1;
        pump();
      });
    }
  }

  /**
   * Read the last run's report and its first frame, then queue the rest.
   *
   * @returns {Promise<void>} Settles once the first answers are in.
   */
  async function load() {
    const ticket = ++serial;
    inflight = 0;
    queue.length = 0;
    pictureState = PREVIEW_STATE.LOADING;
    redraw();
    try {
      const [run, first] = await Promise.all([
        fetchRunResult(node),
        fetchInputPreview(node, FRAMES_SLOT, 0),
      ]);
      if (ticket !== serial || disposed) return;
      frames.clear();
      mattes.clear();
      overlays.clear();
      coverage.clear();
      report = run?.state === PREVIEW_STATE.READY && run.result ? {
        facts: new Map((run.result.facts ?? []).map((row) => [row.name, row.value])),
        counts: new Map((run.result.counts ?? []).map((row) => [row.name, row.value])),
      } : null;
      pictureState = first?.state ?? PREVIEW_STATE.FAILED;
      if (first?.state !== PREVIEW_STATE.READY) {
        held = 0;
        total = 0;
        shown = [];
        redraw();
        return;
      }
      promptId = first.promptId ?? "";
      held = Math.max(1, Number(first.frameCount) || 1);
      total = Math.max(held, Number(first.frameTotal) || held);
      shown = readShown(fact("shown"), held);
      frames.set(0, { state: PREVIEW_STATE.READY, image: first.image });
      const key = nearestStored(runKeyFrame() ?? NaN);
      current = key >= 0 ? key : clamp(current, 0, held - 1);
      wantNow(current);
      for (let index = 0; index < held; index += 1) queue.push({ kind: FRAMES_SLOT, index });
      for (let index = 0; index < held; index += 1) queue.push({ kind: MATTE_SLOT, index });
      pump();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the last run:`, error);
      pictureState = PREVIEW_STATE.FAILED;
    }
    redraw();
  }

  /**
   * Show another stored frame.
   *
   * @param {number} index - The stored frame wanted.
   * @returns {boolean} True when the shown frame moved.
   */
  function showFrame(index) {
    if (!held) return false;
    const next = clamp(Math.round(index), 0, held - 1);
    if (next === current) return false;
    current = next;
    wantNow(current);
    redraw();
    return true;
  }

  /**
   * Where a pointer is on the shown frame.
   *
   * @param {PointerEvent} event - The event.
   * @returns {{u: number, v: number, x: number, y: number}|null} Shares of the frame and the
   *   point in layout pixels, or null outside the frame.
   */
  function framePoint(event) {
    const rect = pictureRect();
    const point = elementPoint(canvas, event);
    if (!point) return null;
    const u = (point.x - rect.x) / rect.width;
    const v = (point.y - rect.y) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { u, v, x: point.x, y: point.y };
  }

  /**
   * Act on a click on the picture.
   *
   * @param {{u: number, v: number, x: number, y: number}} point - Where it landed.
   * @param {boolean} exclude - Whether it places an exclude point.
   * @returns {void}
   */
  function clickAt(point, exclude) {
    const batch = batchIndex(current);
    if (batch === null || !frames.get(current)?.image) return;
    if (mode === MODE.PICK) {
      if (batch !== runKeyFrame()) return;
      const hits = readCandidates(fact("candidates")).filter(({ box }) => (
        point.u >= box[0] && point.u <= box[2] && point.v >= box[1] && point.v <= box[3]
      ));
      if (!hits.length) return;
      const smallest = hits.reduce((best, one) => (
        (one.box[2] - one.box[0]) * (one.box[3] - one.box[1]) < (best.box[2] - best.box[0]) * (best.box[3] - best.box[1])
          ? one : best
      ));
      write({ [PICK]: smallest.index, [POINTS]: "", [KEY_FRAME]: batch });
      return;
    }
    const { points, readable } = readPoints(findWidget(node, POINTS)?.value);
    if (!readable) return;
    const onKey = batch === widgetNumber(KEY_FRAME);
    if (onKey) {
      const rect = pictureRect();
      const near = points.findIndex((one) => Math.hypot(
        rect.x + one.x * rect.width - point.x, rect.y + one.y * rect.height - point.y,
      ) <= HIT);
      if (near >= 0) {
        write({ [POINTS]: writePoints(points.filter((_, index) => index !== near)) });
        return;
      }
      if (points.length >= MAX_POINTS) return;
      write({ [POINTS]: writePoints([...points, { x: point.u, y: point.v, keep: !exclude }]) });
      return;
    }
    write({ [KEY_FRAME]: batch, [POINTS]: writePoints([{ x: point.u, y: point.v, keep: !exclude }]) });
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    event.stopPropagation();
    root.focus({ preventScroll: true });
    press = { x: event.clientX, y: event.clientY, exclude: event.shiftKey || event.button === 2 };
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
  });
  const endPress = (event, cancelled) => {
    const pressed = press;
    press = null;
    try {
      canvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
    }
    if (!pressed || cancelled) return;
    if (Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) > CLICK_SLOP) return;
    const point = framePoint(event);
    if (point) clickAt(point, pressed.exclude);
  };
  canvas.addEventListener("pointerup", (event) => endPress(event, false));
  canvas.addEventListener("pointercancel", (event) => endPress(event, true));
  canvas.addEventListener("lostpointercapture", () => { press = null; });
  canvas.addEventListener("pointermove", (event) => {
    if (press && event.buttons === 0) press = null;
  });

  /**
   * The stored frame under a pointer on the scrubber.
   *
   * @param {PointerEvent} event - The event.
   * @returns {number} The stored frame.
   */
  function stripIndex(event) {
    const point = elementPoint(stripCanvas, event);
    const width = stripCanvas.clientWidth || 1;
    return Math.floor(clamp(point.x / width, 0, 0.9999) * Math.max(1, held));
  }

  stripCanvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    root.focus({ preventScroll: true });
    scrubbing = true;
    try {
      stripCanvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
    showFrame(stripIndex(event));
  });
  stripCanvas.addEventListener("pointermove", (event) => {
    if (!scrubbing) return;
    if (event.buttons === 0) {
      scrubbing = false;
      return;
    }
    showFrame(stripIndex(event));
  });
  const endScrub = (event) => {
    scrubbing = false;
    try {
      stripCanvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
    }
  };
  stripCanvas.addEventListener("pointerup", endScrub);
  stripCanvas.addEventListener("pointercancel", endScrub);
  stripCanvas.addEventListener("lostpointercapture", () => { scrubbing = false; });

  root.addEventListener("contextmenu", (event) => event.preventDefault());
  root.addEventListener("keydown", (event) => {
    const steps = { ArrowLeft: -1, ArrowRight: 1 };
    if (event.key in steps) {
      event.preventDefault();
      event.stopPropagation();
      showFrame(current + steps[event.key]);
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      event.stopPropagation();
      showFrame(event.key === "Home" ? 0 : held - 1);
    }
  });

  const releaseWheel = captureWheel(root, (event) => {
    if (!stripRow.contains(event.target) || !event.deltaY) return false;
    return showFrame(current + (event.deltaY > 0 ? 1 : -1));
  });

  const resizeWatch = new ResizeObserver(() => redraw());
  resizeWatch.observe(root);
  const stopStageRatio = watchSurfaceRatio(canvas, () => redraw());
  const stopStripRatio = watchSurfaceRatio(stripCanvas, () => redraw());
  const stopTheme = onThemeChange(() => {
    tint = null;
    overlays.clear();
    redraw();
  });
  const stopFinished = onNodeFinished(node, () => load());
  const stopReadable = whenReadable(node, () => load());
  for (const name of REPAINT_ON) chainWidgetCallback(node, name, () => redraw(), EXT_NAME);
  loadPlaceholder().then(() => redraw());
  setMode(MODE.POINTS);

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: () => load(),
    dispose() {
      disposed = true;
      serial += 1;
      queue.length = 0;
      if (frameHandle) cancelAnimationFrame(frameHandle);
      frameHandle = 0;
      resizeWatch.disconnect();
      stopStageRatio?.();
      stopStripRatio?.();
      stopTheme?.();
      stopFinished?.();
      stopReadable?.();
      releaseWheel?.();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_track_subject_wrapped) return;
    proto.__was_track_subject_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
