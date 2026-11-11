/**
 * Gradient editor for the nodes that read `position:r,g,b` stops.
 *
 * Reads the node's `gradient_stops` widget, previews the ramp and writes edits back as
 * `position:r,g,b` lines. Lines the parser cannot read pass through byte for byte.
 */

import { app } from "../../scripts/app.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { openColourPicker } from "./interface/colour_picker.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { roundHalfEven } from "./interface/python_arithmetic.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme, themeVar, themeVarName } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.GradientEditorUI";
const NODE_NAMES = [
  "Image Generate Gradient",
  "WASImageGradientMapNative",
];
const SETTING_ID = "WAS.Gradient.ShowInterface";

const STOPS_WIDGET = "gradient_stops";
const DIRECTION_WIDGET = "direction";
const TOLERANCE_WIDGET = "tolerance";

// A picture on this input is the ramp a run reads, and the stops are left alone.
const PICTURE_INPUT = "gradient_image";

// The node ids that blur their answer after drawing the ramp, which is what the footer glyph
// claims. A gradient map indexes the table directly and blurs nothing.
const BLURRING_NODES = new Set(["Image Generate Gradient"]);

const UI_WIDGET_NAME = "was_gradient_ui";
const UI_WIDGET_TYPE = "was_gradient_editor";

// Height of the appended widget in node units. A DOM widget element is inset by the
// widget's margin on every side, so the element itself is shorter by twice that margin.
const UI_HEIGHT = 104;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout bands, measured in element pixels from the top.
const PAD_X = 4;
const PAD_Y = 4;
const CAP_WIDTH = 10;
const BAND_HEIGHT = 20;
const FOOTER_HEIGHT = 13;
const MIN_PREVIEW_HEIGHT = 10;

// The gap kept between the footer's glyph and the words after it.
const GLYPH_GAP = 4;

const MARKER_HALF_WIDTH = 5;
const MARKER_TIP_HEIGHT = 5;
const HIT_RADIUS = 6;

const BODY_FONT = "10px sans-serif";
const CAP_FONT = "9px sans-serif";
const MENU_FONT = "11px sans-serif";

const MESSAGE_TIMEOUT = 4000;

const CAPTIONS = {
  horizontal: { caption: "left to right", start: "L", end: "R" },
  vertical: { caption: "top to bottom", start: "T", end: "B" },
};

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
 * Read an integer the way Python's `int()` reads one.
 *
 * @param {string} text - Text to read.
 * @returns {number|null} The integer, or null when the text is not one.
 */
function toInt(text) {
  const trimmed = String(text).trim();
  // ASCII digits only. `int()` reads a few numbers this rejects, such as one written with
  // underscore separators or in another numeral system, so a line holding one of those is
  // counted as unreadable here while the node still renders it. The line is left exactly as
  // written, so the rendered image is unaffected.
  return /^[+-]?\d+$/.test(trimmed) ? parseInt(trimmed, 10) : null;
}

/**
 * Read `position:r,g,b` lines into stops, keeping the original lines.
 *
 * @param {string} raw - Full text of the `gradient_stops` widget.
 * @returns {{lines: string[], stops: Array<{line: number, position: number, rgb: number[]}>,
 *   unreadable: number[]}} The raw lines, the stops read from them, and the indices of the
 *   lines that could not be read.
 */
function parseStops(raw) {
  // Python reads the text with `str.splitlines`, which breaks on more separators than the
  // line feed. A line ended by one of the others arrives here joined to the line after it and
  // is counted as one unreadable line, while the node still renders the stops in both. The
  // text is left exactly as written, so the rendered image is unaffected.
  const lines = String(raw ?? "").split("\n");
  const stops = [];
  const unreadable = [];

  for (let index = 0; index < lines.length; index++) {
    // Spaces are removed before anything is read, as the Python parser removes them, so
    // `50 : 255, 0, 0` is the stop `50:255,0,0`. A carriage return goes with them, since
    // `str.splitlines` breaks on `\r\n` and never leaves one at the end of a line.
    const line = lines[index].replace(/\r/g, "").replace(/ /g, "");
    if (line.trim() === "") continue;

    // The position is the text before the first colon and the channels the comma separated
    // text after it. A line with no colon, with fewer than three channels, or whose numbers
    // are not integers is skipped, exactly as the Python parser skips it.
    const parts = line.split(":");
    if (parts.length < 2) {
      unreadable.push(index);
      continue;
    }

    const channels = parts[1].split(",");
    if (channels.length < 3) {
      unreadable.push(index);
      continue;
    }

    const position = toInt(parts[0]);
    const rgb = [toInt(channels[0]), toInt(channels[1]), toInt(channels[2])];
    if (position === null || rgb.some((channel) => channel === null)) {
      unreadable.push(index);
      continue;
    }

    // The line number indexes the raw text and travels with the stop, so an edit rewrites
    // the one line that stop owns and touches no other.
    stops.push({ line: index, position, rgb });
  }

  return { lines, stops, unreadable };
}

/**
 * Write a stop as the one line that represents it.
 *
 * @param {{position: number, rgb: number[]}} stop - Stop to format.
 * @returns {string} The stop as `position:r,g,b`.
 */
function formatStop(stop) {
  return `${stop.position}:${stop.rgb[0]},${stop.rgb[1]},${stop.rgb[2]}`;
}

/**
 * Build the 256 entry spectrum the node interpolates through.
 *
 * @param {Array<{position: number, rgb: number[]}>} stops - Stops in line order.
 * @returns {number[][]|null} 256 RGB triples, or null when there is no stop to draw.
 */
function buildSpectrum(stops) {
  if (!stops.length) return null;

  const colors = new Map();
  for (const stop of stops) colors.set(stop.position, stop.rgb);

  colors.set(0, colors.get(Math.min(...colors.keys())));
  colors.set(255, colors.get(Math.max(...colors.keys())));

  const positions = [...colors.keys()].sort((a, b) => a - b);
  const spectrum = new Array(256);
  let cursor = 0;

  for (let index = 0; index < 256; index++) {
    while (cursor + 1 < positions.length && positions[cursor + 1] <= index) cursor++;
    const startPos = positions[cursor];
    const endPos = startPos === index ? index : positions[cursor + 1] ?? startPos;
    const start = colors.get(startPos);
    const end = colors.get(endPos);
    const factor = startPos === endPos ? 0 : (index - startPos) / (endPos - startPos);

    spectrum[index] = [
      roundHalfEven(start[0] + (end[0] - start[0]) * factor),
      roundHalfEven(start[1] + (end[1] - start[1]) * factor),
      roundHalfEven(start[2] + (end[2] - start[2]) * factor),
    ];
  }

  return spectrum;
}

/**
 * Round a channel to a multiple of the tolerance, as the node does.
 *
 * @param {number} channel - Channel value.
 * @param {number} tolerance - Quantisation step. 0 leaves the channel alone.
 * @returns {number} The quantised channel.
 */
function quantise(channel, tolerance) {
  if (!(tolerance > 0)) return channel;
  return roundHalfEven(channel / tolerance) * tolerance;
}

/**
 * Format an RGB triple as a canvas fill style.
 *
 * @param {number[]} rgb - Three channels.
 * @returns {string} An `rgb()` colour, with each channel held to 0 to 255.
 */
function cssColour(rgb) {
  const r = clamp(Math.round(rgb[0]), 0, 255);
  const g = clamp(Math.round(rgb[1]), 0, 255);
  const b = clamp(Math.round(rgb[2]), 0, 255);
  return `rgb(${r},${g},${b})`;
}

/**
 * Read whether the editor is drawn at all.
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
 * Work out where each band of the editor sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the preview strip, the stop band and the footer.
 */
function computeLayout(width, height) {
  const trackX0 = PAD_X + CAP_WIDTH;
  const trackX1 = Math.max(trackX0 + 1, width - PAD_X - CAP_WIDTH);
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT);
  const bandY = Math.max(0, footerY - BAND_HEIGHT);
  const previewY = PAD_Y;
  const previewHeight = Math.max(MIN_PREVIEW_HEIGHT, bandY - previewY - 2);

  return {
    width,
    height,
    trackX0,
    trackX1,
    trackWidth: trackX1 - trackX0,
    previewY,
    previewHeight,
    bandY,
    bandHeight: BAND_HEIGHT,
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * Build the gradient editor for one node.
 *
 * @param {object} node - The node the editor decorates.
 * @returns {{element: HTMLElement, schedulePaint: () => void,
 *   handleStopsChanged: () => void, dispose: () => void}} The element to hand to
 *   `addDOMWidget`, a coalesced repaint, the repaint to run when the stops text changed,
 *   and teardown.
 */
function createGradientEditor(node) {
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

  // The footer's glyph states what the strip is worth through the element's own title. The region
  // is handed over again on every repaint, since the glyph moves whenever the node is resized.
  const titles = hoverTitles(root);

  const state = {
    selectedLine: null,
    pending: null,
    drag: null,
    hoverLine: null,
    lastWritten: null,
    message: "",
    messageTimer: 0,
    footerLink: null,
    menu: null,
    menuDismiss: null,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Read the widget text, the stops in it, and the position an unfinished gesture moves a
   * stop to.
   *
   * @returns {object} Parsed lines and stops, with any pending move applied.
   */
  function readModel() {
    const model = parseStops(findWidget(node, STOPS_WIDGET)?.value);
    if (state.pending) {
      const stop = model.stops.find((candidate) => candidate.line === state.pending.line);
      if (stop) stop.position = state.pending.position;
      else state.pending = null;
    }
    return model;
  }

  /**
   * Whether a picture is connected, which a run reads in place of the stops.
   *
   * @returns {boolean} True while `gradient_image` carries a link.
   */
  function pictureConnected() {
    const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
    return inputs.some((input) => input?.name === PICTURE_INPUT && input.link != null);
  }

  /**
   * Read the `tolerance` widget.
   *
   * @returns {number} The quantisation step, 0 when it cannot be read.
   */
  function readTolerance() {
    const value = Number(findWidget(node, TOLERANCE_WIDGET)?.value);
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
  }

  /**
   * Read the `direction` widget as the caption and end caps that describe it.
   *
   * @returns {{caption: string, start: string, end: string}} Wording for the footer and
   *   the two ends of the track.
   */
  function readDirection() {
    const value = findWidget(node, DIRECTION_WIDGET)?.value;
    return CAPTIONS[value] ?? CAPTIONS.horizontal;
  }

  /**
   * Find the selected stop in a parsed model.
   *
   * @param {object} model - Model from `readModel`.
   * @returns {object|null} The selected stop, or null when nothing is selected.
   */
  function selectedStop(model) {
    if (state.selectedLine === null) return null;
    return model.stops.find((stop) => stop.line === state.selectedLine) ?? null;
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
   * Write the widget once, leaving every line the editor does not own untouched.
   *
   * @param {string[]} lines - Full set of lines to store.
   * @returns {void}
   */
  function writeLines(lines) {
    if (state.disposed) return;
    const widget = findWidget(node, STOPS_WIDGET);
    if (!widget) return;
    const next = lines.join("\n");
    // The value is compared first, so a repaint driven by the widget's own callback can never
    // write anything back.
    if (next === widget.value) return;

    // The write is bracketed by the canvas change events the graph's change tracker listens
    // for, which is what gives the edit its own undo entry. The tracker's own snapshot
    // triggers are a document `mouseup` and the release of a bare modifier key, so a commit
    // from the keyboard, from the stop menu or from the colour picker reaches none of them
    // and would otherwise be folded into whatever the previous snapshot held.
    const canvas = app.canvas;
    const transactional =
      typeof canvas?.emitBeforeChange === "function" &&
      typeof canvas?.emitAfterChange === "function";

    state.lastWritten = next;
    if (transactional) canvas.emitBeforeChange();
    try {
      widget.value = next;
    } finally {
      if (transactional) canvas.emitAfterChange();
    }
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Repaint after the stops text changed, dropping a selection the change invalidated.
   *
   * @returns {void}
   */
  function handleStopsChanged() {
    const current = findWidget(node, STOPS_WIDGET)?.value;
    if (current !== state.lastWritten) {
      state.lastWritten = typeof current === "string" ? current : null;
      state.selectedLine = null;
      state.hoverLine = null;
      state.pending = null;
    }
    schedulePaint();
  }

  /**
   * Convert a horizontal element position into a stop position.
   *
   * @param {number} x - Position in element pixels.
   * @param {boolean} snap - Snap to multiples of five.
   * @returns {number} A stop position, 0 to 100.
   */
  function positionFromX(x, snap) {
    const layout = state.layout;
    const span = layout.trackWidth;
    const ratio = span > 0 ? (x - layout.trackX0) / span : 0;
    const raw = ratio * 100;
    const stepped = snap ? Math.round(raw / 5) * 5 : Math.round(raw);
    return clamp(stepped, 0, 100);
  }

  /**
   * Convert a stop position into a horizontal element position.
   *
   * @param {number} position - Stop position.
   * @returns {number} Position in element pixels.
   */
  function xFromPosition(position) {
    const layout = state.layout;
    return layout.trackX0 + (clamp(position, 0, 100) / 100) * layout.trackWidth;
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
   * Convert an element position back to a position on screen.
   *
   * @param {number} x - Horizontal position in element pixels.
   * @param {number} y - Vertical position in element pixels.
   * @returns {{clientX: number, clientY: number}} Position on screen.
   */
  function screenPoint(x, y) {
    const rect = root.getBoundingClientRect();
    const scaleX = root.clientWidth ? rect.width / root.clientWidth : 1;
    const scaleY = root.clientHeight ? rect.height / root.clientHeight : 1;
    return { clientX: rect.left + x * scaleX, clientY: rect.top + y * scaleY };
  }

  /**
   * Test whether the `gradient_stops` textarea is in the page and can take focus.
   *
   * @returns {boolean} True when the textarea can take focus.
   */
  function stopsTextFocusable() {
    const element = findWidget(node, STOPS_WIDGET)?.element;
    // The classic canvas mounts the element the widget was built with. Nodes 2.0 draws the
    // widget from its own component and leaves that element detached, where focusing it does
    // nothing, so the unreadable line count is a plain status line there rather than a
    // control that advertises an action it cannot perform.
    return Boolean(element?.isConnected && typeof element.focus === "function");
  }

  /**
   * Test whether a point is over the unreadable line count in the footer.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {boolean} True when the point is over the count.
   */
  function hitFooterLink(point) {
    const link = state.footerLink;
    if (!link) return false;
    return point.x >= link.x0 && point.x <= link.x1 && point.y >= link.y0 && point.y <= link.y1;
  }

  /**
   * Find the stop marker under a point.
   *
   * @param {object} model - Model from `readModel`.
   * @param {number} x - Horizontal position in element pixels.
   * @param {number} y - Vertical position in element pixels.
   * @returns {object|null} The stop under the point, or null.
   */
  function hitTestStop(model, x, y) {
    const layout = state.layout;
    if (y < layout.bandY - 2 || y > layout.bandY + layout.bandHeight) return null;

    let best = null;
    let bestDistance = HIT_RADIUS;
    for (const stop of model.stops) {
      const distance = Math.abs(xFromPosition(stop.position) - x);
      if (distance <= bestDistance) {
        best = stop;
        bestDistance = distance;
      }
    }
    return best;
  }

  /**
   * Draw the ramp the node will render, one column per device pixel.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {number} ratio - Backing store pixels per layout pixel. The ramp is written with
   *   `putImageData`, which ignores the context transform, so it applies the ratio itself.
   * @param {number[][]} spectrum - Spectrum from `buildSpectrum`.
   * @param {number} tolerance - Quantisation step.
   * @returns {void}
   */
  function drawRamp(ctx, ratio, spectrum, tolerance) {
    const layout = state.layout;
    const width = Math.max(1, Math.round(layout.trackWidth * ratio));
    const height = Math.max(1, Math.round(layout.previewHeight * ratio));
    const image = ctx.createImageData(width, height);
    const row = new Uint8ClampedArray(width * 4);

    for (let column = 0; column < width; column++) {
      // The node maps a pixel onto 0 to 100, so a ramp of any width holds 101 colours.
      const position = width > 1 ? Math.floor((column * 100) / (width - 1)) : 0;
      const colour = spectrum[position];
      const offset = column * 4;
      row[offset] = quantise(colour[0], tolerance);
      row[offset + 1] = quantise(colour[1], tolerance);
      row[offset + 2] = quantise(colour[2], tolerance);
      row[offset + 3] = 255;
    }

    for (let line = 0; line < height; line++) image.data.set(row, line * width * 4);
    ctx.putImageData(image, Math.round(layout.trackX0 * ratio), Math.round(layout.previewY * ratio));
  }

  /**
   * Draw one stop marker.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} stop - Stop to draw.
   * @param {number} tolerance - Quantisation step, applied to the swatch so the handle
   *   carries the colour the strip above it and the rendered image both hold.
   * @param {boolean} selected - Whether the stop is selected.
   * @param {boolean} hovered - Whether the pointer is over the stop.
   * @returns {void}
   */
  function drawMarker(ctx, theme, stop, tolerance, selected, hovered) {
    const layout = state.layout;
    const x = xFromPosition(stop.position);
    const top = layout.bandY + 1;
    const shoulder = top + MARKER_TIP_HEIGHT;
    const bottom = layout.bandY + layout.bandHeight - 3;
    const inRange = stop.position >= 0 && stop.position <= 100;

    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x + MARKER_HALF_WIDTH, shoulder);
    ctx.lineTo(x + MARKER_HALF_WIDTH, bottom);
    ctx.lineTo(x - MARKER_HALF_WIDTH, bottom);
    ctx.lineTo(x - MARKER_HALF_WIDTH, shoulder);
    ctx.closePath();

    ctx.fillStyle = cssColour([
      quantise(stop.rgb[0], tolerance),
      quantise(stop.rgb[1], tolerance),
      quantise(stop.rgb[2], tolerance),
    ]);
    ctx.fill();
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeStyle = selected
      ? theme.accent
      : !inRange
        ? theme.warning
        : hovered
          ? theme.fg
          : theme.border;
    ctx.stroke();
  }

  /**
   * Draw the footer line.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readModel`.
   * @param {number} tolerance - Quantisation step.
   * @param {boolean} drew - Whether a strip was drawn. Where none was, there is nothing for the
   *   banding claim to be about and no glyph is drawn.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model, tolerance, drew) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    const stop = selectedStop(model);
    const unreadable = model.unreadable.length;

    let rightText = "";
    let rightWarns = false;
    let rightLinks = false;

    if (state.message) {
      rightText = state.message;
      rightWarns = true;
    } else if (pictureConnected()) {
      rightText = "gradient_image is the ramp";
      rightWarns = true;
    } else if (unreadable > 0) {
      rightText = `${unreadable} ${unreadable === 1 ? "line" : "lines"} unreadable`;
      rightWarns = true;
      rightLinks = stopsTextFocusable();
    }

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    // The node blurs the ramp on every run, at a radius that grows with the output size past 512
    // pixels, so the strip is always sharper than the image it stands for. That holds whatever the
    // stops are, so it is the glyph rather than a line that would take the readout's room.
    let glyphWidth = 0;
    if (!drew || !BLURRING_NODES.has(node?.type)) {
      titles.set([]);
    } else {
      const banding = tolerance > 0
        ? `the strip bands at a tolerance of ${tolerance} and the output is blurred after it, at a`
          + " radius that grows with the output size past 512 pixels"
        : "the output is blurred after the ramp, at a radius that grows with the output size past"
          + " 512 pixels, so the strip is the sharper of the two";
      const box = drawIcon(
        ctx,
        ICON.APPROXIMATE,
        layout.trackX0,
        middle - ICON_SIZE / 2,
        ICON_SIZE,
        theme.fgMuted,
      );
      titles.set([{ ...box, title: iconTitle(ICON.APPROXIMATE, banding) }]);
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    }

    let rightWidth = 0;
    state.footerLink = null;
    if (rightText) {
      rightWidth = ctx.measureText(rightText).width;
      ctx.textAlign = "right";
      ctx.fillStyle = rightWarns ? theme.warning : theme.fgMuted;
      ctx.fillText(rightText, layout.trackX1, middle);
      if (rightLinks) {
        state.footerLink = {
          x0: layout.trackX1 - rightWidth,
          x1: layout.trackX1,
          y0: layout.footerY,
          y1: layout.footerY + layout.footerHeight,
        };
      }
    }

    const leftText = stop
      ? `pos ${stop.position}   ${stop.rgb[0]},${stop.rgb[1]},${stop.rgb[2]}`
      : readDirection().caption;
    const available = layout.trackWidth - glyphWidth - rightWidth - 8;
    if (available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = stop ? theme.fg : theme.fgMuted;
      ctx.fillText(leftText, layout.trackX0 + glyphWidth, middle, available);
    }
  }

  /**
   * Draw the whole editor.
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
    const theme = readTheme();
    const model = readModel();
    const tolerance = readTolerance();
    const spectrum = buildSpectrum(model.stops);

    if (spectrum) {
      drawRamp(ctx, ratio, spectrum, tolerance);
    } else {
      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(layout.trackX0, layout.previewY, layout.trackWidth, layout.previewHeight);
      ctx.font = BODY_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = theme.warning;
      ctx.fillText(
        "no readable stop",
        layout.trackX0 + layout.trackWidth / 2,
        layout.previewY + layout.previewHeight / 2,
        layout.trackWidth - 4,
      );
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      layout.trackX0 + 0.5,
      layout.previewY + 0.5,
      Math.max(1, layout.trackWidth - 1),
      Math.max(1, layout.previewHeight - 1),
    );

    ctx.beginPath();
    ctx.moveTo(layout.trackX0, layout.bandY + 0.5);
    ctx.lineTo(layout.trackX1, layout.bandY + 0.5);
    ctx.strokeStyle = theme.border;
    ctx.stroke();

    const direction = readDirection();
    const capMiddle = layout.bandY + layout.bandHeight / 2;
    ctx.font = CAP_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(direction.start, layout.trackX0 - CAP_WIDTH / 2 - 1, capMiddle);
    ctx.fillText(direction.end, layout.trackX1 + CAP_WIDTH / 2 + 1, capMiddle);

    for (const stop of model.stops) {
      if (stop.line === state.selectedLine) continue;
      drawMarker(ctx, theme, stop, tolerance, false, stop.line === state.hoverLine);
    }
    const selected = selectedStop(model);
    if (selected) {
      drawMarker(ctx, theme, selected, tolerance, true, selected.line === state.hoverLine);
    }

    drawFooter(ctx, theme, model, tolerance, Boolean(spectrum));

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
        console.error(`[${EXT_NAME}] Failed to draw the gradient editor:`, error);
      }
    });
  }

  /**
   * Close the stop menu if it is open.
   *
   * @returns {void}
   */
  function closeMenu() {
    if (state.menuDismiss) {
      document.removeEventListener("pointerdown", state.menuDismiss, true);
      state.menuDismiss = null;
    }
    if (state.menu) {
      state.menu.remove();
      state.menu = null;
    }
  }

  /**
   * Open the two item menu for a stop.
   *
   * @param {object} stop - Stop the menu acts on.
   * @param {number} x - Horizontal position in element pixels.
   * @param {number} y - Vertical position in element pixels.
   * @param {number} clientX - Horizontal position on screen, for the colour picker.
   * @param {number} clientY - Vertical position on screen, for the colour picker.
   * @returns {void}
   */
  function openMenu(stop, x, y, clientX, clientY) {
    closeMenu();

    const menu = document.createElement("div");
    const menuWidth = 104;
    const left = clamp(x, 0, Math.max(0, root.clientWidth - menuWidth));
    const top = clamp(y, 0, Math.max(0, root.clientHeight - 44));
    menu.style.cssText = [
      "position:absolute",
      `left:${Math.round(left)}px`,
      `top:${Math.round(top)}px`,
      `width:${menuWidth}px`,
      `background:${themeVar("panelBg")}`,
      `border:1px solid ${themeVar("border")}`,
      "border-radius:3px",
      "padding:2px 0",
      "z-index:10",
      `font:${MENU_FONT}`,
      `color:${themeVar("fg")}`,
    ].join(";");

    const addItem = (label, action) => {
      const item = document.createElement("div");
      item.textContent = label;
      item.style.cssText = "padding:2px 8px;cursor:pointer;white-space:nowrap";
      item.onmouseenter = () => {
        item.style.background = `var(${themeVarName("accentBg")}, ${themeVar("border")})`;
      };
      item.onmouseleave = () => {
        item.style.background = "transparent";
      };
      item.addEventListener("pointerdown", (event) => {
        event.stopPropagation();
        event.preventDefault();
        closeMenu();
        try {
          action();
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to run a menu action:`, error);
        }
      });
      menu.appendChild(item);
    };

    addItem("Edit colour", () => pickStopColour(stop, clientX, clientY));
    addItem("Remove stop", () => removeStop(stop));

    root.appendChild(menu);
    state.menu = menu;
    state.menuDismiss = (event) => {
      if (!menu.contains(event.target)) closeMenu();
    };
    document.addEventListener("pointerdown", state.menuDismiss, true);
  }

  /**
   * Open the colour picker for a stop and write the colour it returns.
   *
   * @param {object} stop - Stop to recolour.
   * @param {number} clientX - Horizontal position on screen.
   * @param {number} clientY - Vertical position on screen.
   * @returns {void}
   */
  function pickStopColour(stop, clientX, clientY) {
    openColourPicker(clientX, clientY, stop.rgb, (rgb) => {
      const model = readModel();
      const target = model.stops.find((candidate) => candidate.line === stop.line);
      if (!target) return;
      const lines = model.lines.slice();
      lines[target.line] = formatStop({ position: target.position, rgb });
      state.selectedLine = target.line;
      writeLines(lines);
      schedulePaint();
    });
  }

  /**
   * Remove a stop, unless it is the only one that can be read.
   *
   * @param {object} stop - Stop to remove.
   * @returns {void}
   */
  function removeStop(stop) {
    const model = readModel();
    const target = model.stops.find((candidate) => candidate.line === stop.line);
    if (!target) return;
    if (model.stops.length <= 1) {
      setMessage("at least one stop is required");
      return;
    }
    const lines = model.lines.slice();
    lines.splice(target.line, 1);
    state.selectedLine = null;
    state.hoverLine = null;
    state.pending = null;
    writeLines(lines);
    schedulePaint();
  }

  /**
   * Add a stop at a position, coloured with the ramp already there.
   *
   * @param {number} position - Stop position, 0 to 100.
   * @returns {void}
   */
  function addStop(position) {
    const model = readModel();
    // A stop already at that position is selected rather than added, since a second line at
    // the same position would replace it.
    const existing = model.stops.find((stop) => stop.position === position);
    if (existing) {
      state.selectedLine = existing.line;
      schedulePaint();
      return;
    }

    const spectrum = buildSpectrum(model.stops);
    const rgb = spectrum ? spectrum[clamp(position, 0, 255)] : [255, 255, 255];
    const lines = model.lines.slice();
    // The new line is appended, so no existing line moves and no existing line is consumed,
    // including a blank or half typed last line the editor does not own.
    lines.push(formatStop({ position, rgb }));
    state.selectedLine = lines.length - 1;

    writeLines(lines);
    schedulePaint();
  }

  /**
   * Move the selected stop to the position an unfinished gesture holds, and write it.
   *
   * @returns {void}
   */
  function commitPending() {
    const pending = state.pending;
    state.pending = null;
    if (!pending) return;

    const model = parseStops(findWidget(node, STOPS_WIDGET)?.value);
    const stop = model.stops.find((candidate) => candidate.line === pending.line);
    if (!stop || stop.position === pending.position) {
      schedulePaint();
      return;
    }

    const taken = model.stops.some(
      (candidate) =>
        candidate.line !== pending.line && candidate.position === pending.position,
    );
    if (taken) {
      setMessage(`position ${pending.position} is taken`);
      return;
    }

    const lines = model.lines.slice();
    lines[stop.line] = formatStop({ position: pending.position, rgb: stop.rgb });
    writeLines(lines);
    schedulePaint();
  }

  /**
   * End a drag, releasing the pointer capture it holds.
   *
   * @param {boolean} commit - Write the position the drag reached. False discards it and
   *   leaves the stop where it started.
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
        console.error(`[${EXT_NAME}] Gradient editor input failed:`, error);
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

    closeMenu();
    root.focus?.({ preventScroll: true });

    const point = localPoint(event);
    const layout = state.layout;
    const model = readModel();

    // The count of unreadable lines leads to the text they are in, and is only offered as
    // a target while that text is reachable. The focus is taken on the next frame, after
    // the click has finished moving focus around by itself.
    if (hitFooterLink(point)) {
      const stopsWidget = findWidget(node, STOPS_WIDGET);
      requestAnimationFrame(() => stopsWidget?.element?.focus?.());
      return;
    }

    // The pointer default action is left alone throughout. Cancelling it would suppress
    // the mouse events that follow, which carry both the double click and the graph
    // snapshot that gives the gesture its undo entry.
    const stop = hitTestStop(model, point.x, point.y);
    if (stop) {
      state.selectedLine = stop.line;
      state.drag = { pointerId: event.pointerId, line: stop.line };
      state.pending = { line: stop.line, position: stop.position };
      root.setPointerCapture?.(event.pointerId);
      schedulePaint();
      return;
    }

    if (point.y >= layout.bandY && point.y <= layout.bandY + layout.bandHeight) {
      state.selectedLine = null;
      schedulePaint();
    }
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    const point = localPoint(event);

    if (state.drag) {
      // A button released over another window, or a capture the browser took away, ends
      // the gesture without a pointerup. Without this the stop would keep following an
      // unpressed pointer and commit a position nobody chose.
      if (!(event.buttons & 1)) {
        endDrag(false);
        return;
      }
      root.style.cursor = "grabbing";
      const position = positionFromX(point.x, event.shiftKey);
      if (!state.pending || state.pending.position !== position) {
        state.pending = { line: state.drag.line, position };
        schedulePaint();
      }
      return;
    }

    const model = readModel();
    const stop = hitTestStop(model, point.x, point.y);
    const hoverLine = stop ? stop.line : null;
    root.style.cursor = stop ? "grab" : hitFooterLink(point) ? "pointer" : "default";
    if (hoverLine !== state.hoverLine) {
      state.hoverLine = hoverLine;
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

  const onPointerCancel = () => endDrag(false);

  const onDoubleClick = (event) => {
    const point = localPoint(event);
    const layout = state.layout;
    const model = readModel();

    const stop = hitTestStop(model, point.x, point.y);
    if (stop) {
      state.selectedLine = stop.line;
      pickStopColour(stop, event.clientX, event.clientY);
      event.preventDefault();
      return;
    }

    const inTrack = point.x >= layout.trackX0 - HIT_RADIUS && point.x <= layout.trackX1 + HIT_RADIUS;
    const inBands = point.y >= layout.previewY && point.y <= layout.bandY + layout.bandHeight;
    if (inTrack && inBands) {
      addStop(positionFromX(point.x, event.shiftKey));
      event.preventDefault();
    }
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();

    const point = localPoint(event);
    const stop = hitTestStop(readModel(), point.x, point.y);
    if (!stop) return;
    state.selectedLine = stop.line;
    openMenu(stop, point.x, point.y, event.clientX, event.clientY);
    schedulePaint();
  };

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;

    const model = readModel();
    const stop = selectedStop(model);
    let handled = true;

    switch (event.key) {
      case "ArrowLeft":
      case "ArrowRight": {
        if (!stop) {
          handled = false;
          break;
        }
        const step = (event.shiftKey ? 10 : 1) * (event.key === "ArrowLeft" ? -1 : 1);
        state.pending = { line: stop.line, position: clamp(stop.position + step, 0, 100) };
        schedulePaint();
        break;
      }
      case "ArrowUp":
      case "ArrowDown": {
        const ordered = model.stops.slice().sort((a, b) => a.position - b.position);
        if (!ordered.length) {
          handled = false;
          break;
        }
        const current = stop ? ordered.findIndex((candidate) => candidate.line === stop.line) : -1;
        const next =
          current < 0
            ? event.key === "ArrowUp"
              ? ordered.length - 1
              : 0
            : clamp(current + (event.key === "ArrowUp" ? -1 : 1), 0, ordered.length - 1);
        state.selectedLine = ordered[next].line;
        schedulePaint();
        break;
      }
      case "Enter":
      case " ": {
        if (!stop) {
          handled = false;
          break;
        }
        const point = screenPoint(xFromPosition(stop.position), state.layout.bandY);
        pickStopColour(stop, point.clientX, point.clientY);
        break;
      }
      case "Delete":
      case "Backspace": {
        // Consumed whether or not a stop is selected. Left unhandled these reach ComfyUI's
        // own binding, which deletes the node the editor is drawn on.
        if (!stop) {
          setMessage("select a stop first");
          break;
        }
        removeStop(stop);
        break;
      }
      case "Escape": {
        if (state.drag) endDrag(false);
        else if (state.menu) closeMenu();
        else state.selectedLine = null;
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
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (state.drag) return;
    commitPending();
  };

  const onBlur = () => {
    closeMenu();
    // Focus can only leave mid-drag when the gesture has been interrupted, by another
    // window taking the pointer for example, so the drag is discarded rather than kept.
    if (state.drag) endDrag(false);
    else commitPending();
    state.hoverLine = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(onPointerCancel));
  root.addEventListener("lostpointercapture", guard(() => endDrag(false)));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hoverLine === null) return;
    state.hoverLine = null;
    schedulePaint();
  }));
  root.addEventListener("dblclick", guard(onDoubleClick));
  root.addEventListener("contextmenu", guard(onContextMenu));
  // The editor scrolls nothing of its own, so it takes every wheel gesture over it and the
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

  // The strip is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the timers, observers, listeners and hover text the editor holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    closeMenu();
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
    handleStopsChanged,
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
 * Append the editor to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachGradientEditor(node) {
  if (!findWidget(node, STOPS_WIDGET)) return;

  const editor = createGradientEditor(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Every multiline box on the node bounded the same way, so the panel above takes
  // the room past their ceiling instead of losing all of it to them.
  boundTextBoxes(node);

  chainWidgetCallback(node, STOPS_WIDGET, editor.handleStopsChanged);
  for (const name of [DIRECTION_WIDGET, TOLERANCE_WIDGET]) {
    chainWidgetCallback(node, name, editor.schedulePaint);
  }

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

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      editor.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a link changed:`, error);
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
      console.error(`[${EXT_NAME}] Failed to release the gradient editor:`, error);
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
      category: ["WAS Node Suite", "Gradients", "Gradient editor"],
      name: "Show the gradient editor",
      tooltip:
        "Draw the gradient editor under the gradient_stops widget of Image Generate " +
        "Gradient and Image Gradient Map. " +
        "The widget itself is always available. This applies to nodes added after the " +
        "setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would
    // otherwise wrap the prototype a second time and append a second editor.
    if (proto.__was_gradient_wrapped) return;
    proto.__was_gradient_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachGradientEditor(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the gradient editor:`, error);
      }
      return result;
    };
  },
});
