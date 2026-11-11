/**
 * Palette strip for the Image Palette Map node.
 *
 * Draws one cell per colour the node's `palette` widget keeps and rewrites the one comma or
 * newline separated field an edit owns. Colours are written back as `#rrggbb`.
 */

import { app } from "../../scripts/app.js";
import {
  STATUS,
  drawCell,
  drawSwatch,
  formatColour,
  outlineCell,
  parseColor,
  pickColour,
  residualNote,
  tallyColours,
} from "./interface/colour_cell.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.PaletteMapUI";
const NODE_NAME = "WASImagePaletteMap";
const SETTING_ID = "WAS.PaletteMap.ShowInterface";

const PALETTE_WIDGET = "palette";
const MODE_WIDGET = "mode";
const REVERSE_WIDGET = "reverse";
const SMOOTH_WIDGET = "smooth";
const BLEND_WIDGET = "blend";

// The socket that replaces the text box. Connected, the box is not read at all.
const PALETTES_INPUT = "color_palettes";

// The widgets that change what the footer and its glyph say about the palette rather than what
// the strip draws.
const CAPTION_WIDGETS = [MODE_WIDGET, REVERSE_WIDGET, SMOOTH_WIDGET, BLEND_WIDGET];

// The one mode that reads the palette as an ordered gradient. `apply_palette` treats every
// other value as a nearest match, so anything unrecognised is captioned as one.
const RAMP_MODE = "Luminance Ramp";

const UI_WIDGET_NAME = "was_palette_map_ui";
const UI_WIDGET_TYPE = "was_palette_strip";

// Height of the appended widget in node units. A DOM widget element is inset by the widget's
// margin on every side, so the element itself is shorter by twice that margin.
const UI_HEIGHT = 76;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout bands, measured in element pixels from the top.
const PAD_X = 4;
const PAD_Y = 4;
const RAIL_HEIGHT = 9;
const FOOTER_HEIGHT = 13;
const MIN_STRIP_HEIGHT = 8;

// A gap tick is drawn only while the cells either side of it are wide enough for the tick to
// stand between them rather than over them. The caret for the gap under the pointer is drawn
// at every width.
const MIN_TICK_SPAN = 4;
const CARET_INSET = 3;

// The gap kept between the footer's glyph and the words after it.
const GLYPH_GAP = 4;

const BODY_FONT = "10px sans-serif";
const MARK_FONT = "9px sans-serif";
const RAIL_FONT = "9px sans-serif";

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
 * Test whether two colours are the same three channels.
 *
 * @param {number[]|null} left - First colour.
 * @param {number[]|null} right - Second colour.
 * @returns {boolean} True when both hold the same red, green and blue.
 */
function sameRgb(left, right) {
  if (!left || !right) return false;
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

/**
 * Read the whitespace a field opens with.
 *
 * @param {string} text - Field text.
 * @returns {string} The leading whitespace, never including a line feed, which ends a field.
 */
function leadingPad(text) {
  return /^\s*/.exec(String(text ?? ""))[0];
}

/**
 * Read the whitespace a field closes with.
 *
 * @param {string} text - Field text.
 * @returns {string} The trailing whitespace.
 */
function trailingPad(text) {
  return /\s*$/.exec(String(text ?? ""))[0];
}

/**
 * Choose the separator an insert uses where the field it attaches to has none.
 *
 * @param {string} text - Full text of the widget.
 * @returns {string} A line feed for a box holding lines, a comma for a single line list, and a
 *   line feed for a box holding one field and no separator at all.
 */
function dominantSeparator(text) {
  if (text.includes("\n")) return "\n";
  return text.includes(",") ? "," : "\n";
}

/**
 * Find the commas of one line that separate one field from the next.
 *
 * @param {string} line - One line of the widget text, with no line feed in it.
 * @returns {number[]} Offsets into the line, in order.
 */
function separatingCommas(line) {
  const outside = [];
  const every = [];
  let depth = 0;

  for (let index = 0; index < line.length; index++) {
    const character = line[index];
    if (character === "(") {
      depth += 1;
    } else if (character === ")") {
      depth = Math.max(0, depth - 1);
    } else if (character === ",") {
      every.push(index);
      if (depth === 0) outside.push(index);
    }
  }

  return depth > 0 ? every : outside;
}

/**
 * Split the widget text into the fields the node reads.
 *
 * @param {string} raw - Full text of the `palette` widget.
 * @returns {Array<{index: number, start: number, end: number, text: string,
 *   separator: string}>} Every field in order, with the character range it occupies and the
 *   separator that ended it. The last field's separator is empty, and text holding nothing
 *   still holds one field.
 */
function splitFields(raw) {
  const text = String(raw ?? "");
  const fields = [];
  let lineStart = 0;

  for (;;) {
    // The node reaches its lines through `str.splitlines`, which breaks on several separators a
    // line feed is only one of, so a field holding a bare carriage return, a form feed or a line
    // separator is two entries to the node and one unreadable field here. Splitting on the same
    // outside commas also inherits the PIL divergence `interface/colour_cell.js` documents.
    const feed = text.indexOf("\n", lineStart);
    const lineEnd = feed === -1 ? text.length : feed;
    let start = lineStart;

    for (const offset of separatingCommas(text.slice(lineStart, lineEnd))) {
      const end = lineStart + offset;
      fields.push({
        index: fields.length,
        start,
        end,
        text: text.slice(start, end),
        separator: ",",
      });
      start = end + 1;
    }

    fields.push({
      index: fields.length,
      start,
      end: lineEnd,
      text: text.slice(start, lineEnd),
      separator: feed === -1 ? "" : "\n",
    });

    if (feed === -1) return fields;
    lineStart = feed + 1;
  }
}

/**
 * Read the palette the node will build out of the widget text.
 *
 * @param {string} raw - Full text of the `palette` widget.
 * @returns {{text: string, fields: Array<object>, cells: Array<{field: object, parsed: object,
 *   rgb: number[]|null, repeats: Array<object>}>, dropped: Array<{field: object,
 *   repeat: boolean}>, tally: object, repeats: number}} The raw text, its fields, one cell per
 *   colour the node keeps with the repeated fields that resolve to it, the fields the node
 *   drops and whether each was dropped for repeating a colour, the tally of every field that
 *   held something, and how many of the drops were repeats.
 */
function readPalette(raw) {
  const text = String(raw ?? "");
  const fields = splitFields(text);
  const cells = [];
  const dropped = [];
  const results = [];
  const seen = new Map();
  let repeats = 0;

  for (const field of fields) {
    const parsed = parseColor(field.text);
    // `parse_palette` passes over an empty field without comment, so it is not counted here
    // either and the footer never reports one.
    if (parsed.status === STATUS.EMPTY) continue;
    results.push(parsed);

    if (parsed.status === STATUS.COLOUR) {
      // The node resolves in mode `RGB`, which drops the alpha before the repeat test, so a
      // field carrying one repeats the same field written opaque, and the cell is drawn opaque,
      // as the node paints it.
      const rgb = [parsed.rgba[0], parsed.rgba[1], parsed.rgba[2]];
      const key = rgb.join(",");
      const owner = seen.get(key);
      if (owner) {
        // `parse_palette` keeps the first field of each colour and drops every later one,
        // wherever they sit, so `#f80` written under `#ff8800` is a repeat rather than a second
        // cell. The cell keeps those later fields as well as the one the node reads, so an
        // edit to the colour reaches all of
        // them: rewriting only the first would hand the old colour to the repeat behind it, and
        // removing only the first would leave the colour exactly where it was.
        owner.repeats.push(field);
        dropped.push({ field, repeat: true });
        repeats += 1;
        continue;
      }
      const cell = { field, parsed, rgb, repeats: [] };
      seen.set(key, cell);
      cells.push(cell);
      continue;
    }

    // The node resolves a function form and this file does not, so a declined field is drawn
    // as a cell without a colour rather than dropped. A colour with no colour cannot be
    // compared, so a declined field is never counted as a repeat of anything. That has one
    // visible consequence: `rgb(255, 136, 0)` and `#ff8800` in the same box are one colour to
    // the node, which keeps the first of them, and two cells here, so the count of cells is one
    // ahead of the palette in that case. A marker never claims a colour, so nothing on screen is
    // wrong about what the node will paint.
    if (parsed.status === STATUS.DECLINED) cells.push({ field, parsed, rgb: null, repeats: [] });
    else dropped.push({ field, repeat: false });
  }

  return { text, fields, cells, dropped, tally: tallyColours(results), repeats };
}

/**
 * Rewrite one field, keeping the whitespace around it.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {object} field - Field to rewrite.
 * @param {string} value - Text to put in its place.
 * @returns {string} The full widget text, with every other byte untouched.
 */
function replaceField(model, field, value) {
  const open = field.start + leadingPad(field.text).length;
  const close = Math.max(open, field.end - trailingPad(field.text).length);
  return model.text.slice(0, open) + value + model.text.slice(close);
}

/**
 * Rewrite several fields with the same text.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {Array<object>} fields - Fields to rewrite.
 * @param {string} value - Text to put in each of their places.
 * @returns {string} The full widget text, with every other byte untouched.
 */
function replaceFields(model, fields, value) {
  let text = model.text;
  for (const field of fields.slice().sort((left, right) => right.index - left.index)) {
    text = replaceField({ text, fields: model.fields }, field, value);
  }
  return text;
}

/**
 * Put a colour in front of a field.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {object} field - Field the colour goes in front of.
 * @param {string} value - Colour to insert.
 * @returns {{text: string, fieldIndex: number}} The full widget text and the index the new
 *   field holds in it.
 */
function insertBefore(model, field, value) {
  const previous = model.fields[field.index - 1];
  const separator = previous?.separator || field.separator || dominantSeparator(model.text);
  const pad = leadingPad(field.text);
  const at = field.start;
  return {
    text: model.text.slice(0, at) + pad + value + separator + model.text.slice(at),
    fieldIndex: field.index,
  };
}

/**
 * Put a colour after a field.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {object} field - Field the colour goes after.
 * @param {string} value - Colour to insert.
 * @returns {{text: string, fieldIndex: number}} The full widget text and the index the new
 *   field holds in it.
 */
function insertAfter(model, field, value) {
  if (field.text.trim() === "") {
    const at = field.start + leadingPad(field.text).length;
    return {
      text: model.text.slice(0, at) + value + model.text.slice(at),
      fieldIndex: field.index,
    };
  }

  const separator = field.separator || dominantSeparator(model.text);
  const pad = leadingPad(field.text);
  const at = field.end;
  return {
    text: model.text.slice(0, at) + separator + pad + value + model.text.slice(at),
    fieldIndex: field.index + 1,
  };
}

/**
 * Remove a field and the one separator that held it in place.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {object} field - Field to remove.
 * @returns {string} The full widget text without it.
 */
function removeField(model, field) {
  const previous = model.fields[field.index - 1];
  if (previous) return model.text.slice(0, previous.end) + model.text.slice(field.end);
  // The first field has nothing in front of it, so it gives up the separator behind it instead.
  return model.text.slice(0, field.start) + model.text.slice(field.end + field.separator.length);
}

/**
 * Remove several fields.
 *
 * @param {object} model - Model from `readPalette`.
 * @param {Array<object>} fields - Fields to remove.
 * @returns {string} The full widget text without them.
 */
function removeFields(model, fields) {
  let text = model.text;
  for (const field of fields.slice().sort((left, right) => right.index - left.index)) {
    text = removeField({ text, fields: model.fields }, field);
  }
  return text;
}

/**
 * Word what the node drops, for the footer.
 *
 * @param {object} model - Model from `readPalette`.
 * @returns {string} The residual line, empty when the node drops nothing.
 */
function residualText(model) {
  const counted = residualNote(model.tally, "field");
  const repeated =
    model.repeats > 0
      ? `${model.repeats} ${model.repeats === 1 ? "repeat" : "repeats"} dropped`
      : "";
  return [counted, repeated].filter(Boolean).join(", ");
}

/**
 * Choose the colour a gap's picker opens on.
 *
 * @param {Array<{rgb: number[]|null}>} cells - Cells from `readPalette`.
 * @param {number} gap - Gap index, 0 in front of the first cell and `cells.length` after the
 *   last.
 * @returns {number[]} Three channels: the midpoint of the cells either side, one neighbour's
 *   colour at an end, and white for an empty palette.
 */
function gapSeed(cells, gap) {
  const before = cells[gap - 1]?.rgb ?? null;
  const after = cells[gap]?.rgb ?? null;
  if (before && after) {
    return [0, 1, 2].map((channel) => Math.round((before[channel] + after[channel]) / 2));
  }
  return before ?? after ?? [255, 255, 255];
}

/**
 * Read whether the strip is drawn at all.
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
 * Work out where each band of the strip sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the cell strip, the insert rail and the footer.
 */
function computeLayout(width, height) {
  const stripX0 = PAD_X;
  const stripX1 = Math.max(stripX0 + 1, width - PAD_X);
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT);
  const railY = Math.max(0, footerY - RAIL_HEIGHT);
  const stripY = PAD_Y;

  return {
    width,
    height,
    stripX0,
    stripX1,
    stripWidth: stripX1 - stripX0,
    stripY,
    stripHeight: Math.max(MIN_STRIP_HEIGHT, railY - stripY - 2),
    railY,
    railHeight: RAIL_HEIGHT,
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * Work out where one cell sits in the strip.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} count - How many cells share the strip.
 * @param {number} index - Which cell.
 * @returns {{x: number, y: number, width: number, height: number}} The cell in element pixels.
 */
function cellRect(layout, count, index) {
  const span = layout.stripWidth / Math.max(1, count);
  const x = layout.stripX0 + index * span;
  return {
    x,
    y: layout.stripY,
    width: layout.stripX0 + (index + 1) * span - x,
    height: layout.stripHeight,
  };
}

/**
 * Work out where one gap sits along the rail.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} count - How many cells share the strip.
 * @param {number} gap - Gap index, 0 in front of the first cell and `count` after the last.
 * @returns {number} Position in element pixels. An empty palette has one gap, drawn in the
 *   middle of the strip.
 */
function gapX(layout, count, gap) {
  if (count <= 0) return layout.stripX0 + layout.stripWidth / 2;
  return layout.stripX0 + gap * (layout.stripWidth / count);
}

/**
 * Build the palette strip for one node.
 *
 * @param {object} node - The node the strip decorates.
 * @returns {{element: HTMLElement, schedulePaint: () => void, handlePaletteChanged: () => void,
 *   dispose: () => void}} The element to hand to `addDOMWidget`, a coalesced repaint, the
 *   repaint to run when the palette text changed, and teardown.
 */
function createPaletteStrip(node) {
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
    selectedField: null,
    hoverField: null,
    hoverGap: null,
    press: null,
    lastWritten: null,
    message: "",
    messageTimer: 0,
    footerLink: null,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Read the `palette` widget.
   *
   * @returns {string} The text it holds, empty when it cannot be read.
   */
  function paletteText() {
    const value = findWidget(node, PALETTE_WIDGET)?.value;
    return typeof value === "string" ? value : "";
  }

  /**
   * Read the palette the node will build.
   *
   * @returns {object} Model from `readPalette`.
   */
  function readModel() {
    return readPalette(paletteText());
  }

  /**
   * Read whether the palette comes from the socket rather than from the text box.
   *
   * @returns {boolean} True while `color_palettes` is linked, which is when the box is read by
   *   nothing.
   */
  function socketDrives() {
    return inputLinked(node, PALETTES_INPUT);
  }

  /**
   * Read whether the palette is being read as an ordered ramp.
   *
   * @returns {boolean} True in `Luminance Ramp`. Every other value is a nearest match, which
   *   is how `apply_palette` treats one it does not recognise.
   */
  function rampMode() {
    return findWidget(node, MODE_WIDGET)?.value === RAMP_MODE;
  }

  /**
   * Read the `reverse` widget.
   *
   * @returns {boolean} True while the palette is read in the opposite direction.
   */
  function reversed() {
    return findWidget(node, REVERSE_WIDGET)?.value === true;
  }

  /**
   * Read the `smooth` widget.
   *
   * @returns {boolean} True while `Luminance Ramp` interpolates between neighbouring colours,
   *   which is the schema's default and leaves most output pixels on no cell of the strip.
   */
  function smoothRamp() {
    return findWidget(node, SMOOTH_WIDGET)?.value !== false;
  }

  /**
   * Read the `blend` widget.
   *
   * @returns {number} How much of the repainted image replaces the original, 0 to 1, taken as
   *   1 when it cannot be read.
   */
  function blendAmount() {
    const value = Number(findWidget(node, BLEND_WIDGET)?.value);
    return Number.isFinite(value) ? clamp(value, 0, 1) : 1;
  }

  /**
   * Find the selected cell in a model.
   *
   * @param {object} model - Model from `readPalette`.
   * @returns {object|null} The selected cell, or null when nothing is selected.
   */
  function selectedCell(model) {
    if (state.selectedField === null) return null;
    return model.cells.find((cell) => cell.field.index === state.selectedField) ?? null;
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
   * Write the widget once, leaving every byte the strip does not own untouched.
   *
   * @param {string} text - Full text to store.
   * @returns {void}
   */
  function writeText(text) {
    if (state.disposed) return;
    const widget = findWidget(node, PALETTE_WIDGET);
    if (!widget) return;
    if (text === widget.value) return;

    // Bracketing the write in the canvas change events the graph's change tracker listens for is
    // what gives the edit its own undo entry. Every edit here is committed from the native colour
    // picker or from a right click, so none of them reaches the tracker's own snapshot triggers
    // on their own.
    const canvas = app.canvas;
    const transactional =
      typeof canvas?.emitBeforeChange === "function" &&
      typeof canvas?.emitAfterChange === "function";

    state.lastWritten = text;
    if (transactional) canvas.emitBeforeChange();
    try {
      widget.value = text;
    } finally {
      if (transactional) canvas.emitAfterChange();
    }
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Repaint after the palette text changed, dropping a selection the change invalidated.
   *
   * @returns {void}
   */
  function handlePaletteChanged() {
    const current = findWidget(node, PALETTE_WIDGET)?.value;
    if (current !== state.lastWritten) {
      state.lastWritten = typeof current === "string" ? current : null;
      state.selectedField = null;
      state.hoverField = null;
    }
    schedulePaint();
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
   * Find the cell under a point.
   *
   * @param {object} model - Model from `readPalette`.
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {object|null} The cell under the point, or null.
   */
  function hitCell(model, point) {
    const layout = state.layout;
    const count = model.cells.length;
    if (!count) return null;
    if (point.y < layout.stripY || point.y > layout.stripY + layout.stripHeight) return null;
    if (point.x < layout.stripX0 || point.x > layout.stripX1) return null;
    const span = layout.stripWidth / count;
    const index = clamp(Math.floor((point.x - layout.stripX0) / span), 0, count - 1);
    return model.cells[index];
  }

  /**
   * Find the gap under a point.
   *
   * @param {object} model - Model from `readPalette`.
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {number|null} The gap index, or null when the point is not on the rail.
   */
  function hitGap(model, point) {
    const layout = state.layout;
    if (point.y < layout.railY || point.y > layout.railY + layout.railHeight) return null;
    if (point.x < layout.stripX0 - CARET_INSET || point.x > layout.stripX1 + CARET_INSET) {
      return null;
    }
    const count = model.cells.length;
    if (!count) return 0;
    const span = layout.stripWidth / count;
    return clamp(Math.round((point.x - layout.stripX0) / span), 0, count);
  }

  /**
   * Test whether the `palette` textarea is in the page and can take focus.
   *
   * @returns {boolean} True when the textarea can take focus.
   */
  function paletteTextFocusable() {
    // The classic canvas mounts the element the widget was built with. Nodes 2.0 draws the widget
    // from its own component and leaves that element detached, where focusing it does nothing, so
    // the count of fields the node drops is a plain status line there rather than a control that
    // advertises an action it cannot perform.
    const element = findWidget(node, PALETTE_WIDGET)?.element;
    return Boolean(element?.isConnected && typeof element.focus === "function");
  }

  /**
   * Test whether a point is over the count of dropped fields in the footer.
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
   * Recolour one cell with the colour the picker returns.
   *
   * @param {object} cell - Cell to recolour.
   * @param {number} clientX - Horizontal position on screen.
   * @param {number} clientY - Vertical position on screen.
   * @returns {void}
   */
  function pickCellColour(cell, clientX, clientY) {
    pickColour(clientX, clientY, cell.rgb ?? [255, 255, 255], (rgb) => {
      const model = readModel();
      const target = model.cells.find(
        (candidate) => candidate.field.index === cell.field.index,
      );
      if (!target) return;

      const hex = formatColour([rgb[0], rgb[1], rgb[2], 255]);
      // A pick that lands on the colour the cell already holds is not written, since rewriting
      // `red` as `#ff0000` moves no pixel and would still take an undo entry.
      if (sameRgb(target.rgb, rgb)) {
        state.selectedField = target.field.index;
        schedulePaint();
        return;
      }
      // A colour another cell already holds is refused. The node keeps the first field of
      // each colour: the recoloured field would either replace that one or stop reaching the
      // palette itself, and the strip would lose a cell with nothing on screen to say why.
      const taken = model.cells.some(
        (candidate) => candidate.field.index !== target.field.index && sameRgb(candidate.rgb, rgb),
      );
      if (taken) {
        setMessage(`${hex} is already in the palette`);
        return;
      }

      // Rewriting the first field alone would hand the colour being replaced to the repeat
      // standing behind it, so a palette of one ink would come back as two.
      const fields = [target.field, ...target.repeats];
      state.selectedField = target.field.index;
      writeText(replaceFields(model, fields, hex));
      if (target.repeats.length > 0) {
        const count = target.repeats.length;
        setMessage(`${count} ${count === 1 ? "repeat" : "repeats"} rewritten as well`);
      }
      schedulePaint();
    });
  }

  /**
   * Insert a colour into a gap, taking it from the picker.
   *
   * @param {number} gap - Gap index, 0 in front of the first cell and one past the last cell
   *   for the end of the palette.
   * @param {number} clientX - Horizontal position on screen.
   * @param {number} clientY - Vertical position on screen.
   * @returns {void}
   */
  function pickGapColour(gap, clientX, clientY) {
    const seed = gapSeed(readModel().cells, gap);
    pickColour(clientX, clientY, seed, (rgb) => {
      const model = readModel();
      const hex = formatColour([rgb[0], rgb[1], rgb[2], 255]);
      if (model.cells.some((candidate) => sameRgb(candidate.rgb, rgb))) {
        setMessage(`${hex} is already in the palette`);
        return;
      }

      // A gap past the last cell, and every gap in a palette whose text has since been edited
      // down, puts the colour at the end of the text.
      const target = model.cells[gap];
      const edit = target
        ? insertBefore(model, target.field, hex)
        : insertAfter(model, model.fields[model.fields.length - 1], hex);

      state.selectedField = edit.fieldIndex;
      writeText(edit.text);
      schedulePaint();
    });
  }

  /**
   * Remove a cell, unless it is the only colour the node has.
   *
   * @param {object} cell - Cell to remove.
   * @returns {void}
   */
  function removeCell(cell) {
    const model = readModel();
    const target = model.cells.find((candidate) => candidate.field.index === cell.field.index);
    if (!target) return;
    if (model.cells.length <= 1) {
      // `parse_palette` raises on a palette holding no colour, so taking the last one away
      // would stop the run rather than render anything.
      setMessage("at least one colour is required");
      return;
    }

    const label = target.rgb ? formatColour([...target.rgb, 255]) : target.parsed.text;
    const count = target.repeats.length;
    state.selectedField = null;
    state.hoverField = null;
    // Taking the first field alone would leave the colour exactly where it was, with the repeat
    // behind it becoming the field the node reads, so the strip would be unchanged and the
    // removal would have removed nothing.
    writeText(removeFields(model, [target.field, ...target.repeats]));
    setMessage(
      count > 0
        ? `removed ${label} and ${count} ${count === 1 ? "repeat" : "repeats"}`
        : `removed ${label}`,
    );
    schedulePaint();
  }

  /**
   * What the current mode ignores, for the footer to draw.
   *
   * @param {object} model - Model from `readPalette`.
   * @returns {string} The note, empty where nothing somebody set is being ignored.
   */
  function orderNote(model) {
    // What each mode does with the order is the mode's own definition, the mode is a widget the
    // reader is looking at, and no gesture on this strip reorders anything, so the rest of this
    // goes on the glyph's hover text instead. Nothing here reorders the text, for the same reason.
    if (rampMode()) return "";
    // `apply_palette` reads the order in every other mode only to settle an exact tie between two
    // equally distant colours, so `reverse` does not flip the strip: with `reverse` on, and with
    // each of the three dither settings, the same image comes back.
    return reversed() ? "reverse changes no pixel in this mode" : "";
  }

  /**
   * How truly the strip stands for the output, for the glyph and the hover behind it.
   *
   * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
   */
  function fidelityClaim() {
    const order = reversed() ? "light to dark" : "dark to light";
    const blend = blendAmount();
    // A `blend` below 1 mixes the repainted image back over the original, which leaves no output
    // pixel on any cell of the strip, so it downgrades an exact claim to an approximate one in
    // every mode. Its number stays on screen beside the count, since it is a value rather than an
    // explanation.
    const mixed = blend < 1
      ? `, and the result is then mixed back over the original at blend ${blend.toFixed(2)},`
        + " so no output pixel is exactly one of these"
      : "";
    const whole = mixed ? ICON.APPROXIMATE : ICON.EXACT;
    if (rampMode()) {
      // The ramp itself is not drawn. `smooth` decides which of two different pictures the same
      // strip stands for, and it is the schema's default: on, the ramp interpolates in Oklab,
      // which this file does not compute, and most output pixels are on no cell here; off, every
      // pixel snaps to a cell and the strip is the whole of the output. Either way the palette
      // goes through `sorted_by_lightness` first, so the typed order reaches no pixel and
      // shuffling the five default colours renders byte for byte the same image.
      if (smoothRamp()) {
        return {
          icon: ICON.APPROXIMATE,
          detail: `the ramp sorts these colours by lightness, ${order}, and blends between them`
            + ` in Oklab, so most output pixels are on no cell here${mixed}`,
        };
      }
      return {
        icon: whole,
        detail: `the ramp sorts these colours by lightness, ${order}, and every output pixel`
          + ` snaps to one of them${mixed}`,
      };
    }
    return {
      icon: whole,
      detail: "every output pixel takes the nearest of these colours, so the order they are"
        + ` written in reaches no pixel${mixed}`,
    };
  }

  /**
   * Draw the cells, or a line saying there are none.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {number} ratio - Device pixels per layout pixel the canvas is scaled by. The swatches
   *   write pixels straight into the backing store, so they need the same number `setTransform`
   *   was given.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readPalette`.
   * @returns {void}
   */
  function drawStrip(ctx, ratio, theme, model) {
    const layout = state.layout;
    const count = model.cells.length;

    if (!count) {
      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(layout.stripX0, layout.stripY, layout.stripWidth, layout.stripHeight);
      ctx.font = BODY_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = theme.warning;
      // `parse_palette` raises on a palette holding no colour, so where every field is empty
      // this is not an empty preview, it is the node stopping. Where a field held something
      // this split cannot read, the node's own split can still find a colour in it, so the
      // strip says what it sees rather than what the run will do. The footer keeps its room
      // for the cause either way.
      const readsNothing = model.tally.invalid === 0 && model.tally.declined === 0;
      ctx.fillText(
        readsNothing ? "no colour here, the node stops" : "nothing here reads as a colour",
        layout.stripX0 + layout.stripWidth / 2,
        layout.stripY + layout.stripHeight / 2,
        layout.stripWidth - 4,
      );
      return;
    }

    const options = {
      markColour: theme.warning,
      markFont: MARK_FONT,
    };

    for (let index = 0; index < count; index++) {
      const cell = model.cells[index];
      const rect = cellRect(layout, count, index);
      if (cell.rgb) drawSwatch(ctx, ratio, rect, cell.rgb, options);
      else drawCell(ctx, ratio, rect, cell.parsed, options);
    }

    for (let index = 0; index < count; index++) {
      const cell = model.cells[index];
      const selected = cell.field.index === state.selectedField;
      const hovered = cell.field.index === state.hoverField;
      if (!selected && !hovered) continue;
      outlineCell(ctx, cellRect(layout, count, index), selected ? theme.accent : theme.fg);
    }
  }

  /**
   * Draw the rail an insert is taken from.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readPalette`.
   * @returns {void}
   */
  function drawRail(ctx, theme, model) {
    const layout = state.layout;
    const count = model.cells.length;
    const span = count > 0 ? layout.stripWidth / count : layout.stripWidth;
    const top = layout.railY + 2;
    const bottom = layout.railY + layout.railHeight - 2;

    if (span >= MIN_TICK_SPAN) {
      ctx.beginPath();
      for (let gap = 0; gap <= count; gap++) {
        const x = Math.round(gapX(layout, count, gap)) + 0.5;
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
      }
      ctx.lineWidth = 1;
      ctx.strokeStyle = theme.border;
      ctx.stroke();
    }

    if (state.hoverGap === null) return;
    ctx.font = RAIL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.accent;
    ctx.fillText(
      "+",
      clamp(
        gapX(layout, count, state.hoverGap),
        layout.stripX0 + CARET_INSET,
        layout.stripX1 - CARET_INSET,
      ),
      layout.railY + layout.railHeight / 2,
    );
  }

  /**
   * Draw the footer line.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readPalette`.
   * @param {boolean} fromSocket - Whether the palette comes from the socket.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model, fromSocket) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    const count = model.cells.length;

    let rightText = "";
    let rightWarns = false;
    let rightLinks = false;
    const residual = fromSocket ? "" : residualText(model);

    // What the node drops outranks a widget the mode ignores. A palette that reads as nothing says
    // so inside the strip instead of here, so the two never compete for this line.
    if (state.message) {
      rightText = state.message;
      rightWarns = true;
    } else if (fromSocket) {
      rightText = "the palette box is not read";
    } else if (residual) {
      rightText = residual;
      rightWarns = true;
      rightLinks = paletteTextFocusable();
    } else if (count) {
      rightText = orderNote(model);
      rightWarns = Boolean(rightText);
    }

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    // The socket's palette is not on the node at all, so there is nothing here to claim anything
    // about and no glyph is drawn.
    let glyphWidth = 0;
    if (fromSocket || !count) {
      titles.set([]);
    } else {
      const claim = fidelityClaim();
      const box = drawIcon(
        ctx,
        claim.icon,
        layout.stripX0,
        middle - ICON_SIZE / 2,
        ICON_SIZE,
        theme.fgMuted,
      );
      titles.set([{ ...box, title: iconTitle(claim.icon, claim.detail) }]);
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    }

    let rightWidth = 0;
    state.footerLink = null;
    if (rightText) {
      rightWidth = ctx.measureText(rightText).width;
      ctx.textAlign = "right";
      ctx.fillStyle = rightWarns ? theme.warning : theme.fgMuted;
      ctx.fillText(rightText, layout.stripX1, middle);
      if (rightLinks) {
        state.footerLink = {
          x0: layout.stripX1 - rightWidth,
          x1: layout.stripX1,
          y0: layout.footerY,
          y1: layout.footerY + layout.footerHeight,
        };
      }
    }

    const leftText = leftFooterText(model, fromSocket);
    const available = layout.stripWidth - glyphWidth - rightWidth - 8;
    if (leftText && available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(leftText, layout.stripX0 + glyphWidth, middle, available);
    }
  }

  /**
   * Word the left of the footer.
   *
   * @param {object} model - Model from `readPalette`.
   * @param {boolean} fromSocket - Whether the palette comes from the socket.
   * @returns {string} The wording, empty where there is nothing to say.
   */
  function leftFooterText(model, fromSocket) {
    if (fromSocket) return "";

    // A `blend` below 1 is carried alongside whatever else this line says. It is the one
    // note that says no pixel of the output is the colour of any cell, and it applies for as long
    // as the widget holds that value. It cannot wait for the line to be free: an edit committed
    // here selects the cell it changed, so the readout would take this line over for good and the
    // note would never be seen again.
    const blend = blendAmount();
    const caveat = blend < 1 ? `blend ${blend.toFixed(2)}` : "";
    const withCaveat = (text) => (caveat ? `${text}   ${caveat}` : text);

    const cell =
      model.cells.find((candidate) => candidate.field.index === state.hoverField) ??
      selectedCell(model);
    if (cell) {
      const hex = cell.rgb ? formatColour([...cell.rgb, 255]) : "";
      const text = cell.parsed.text;
      return withCaveat(hex && hex !== text.toLowerCase() ? `${text}   ${hex}` : text);
    }

    if (state.hoverGap !== null) return withCaveat("insert a colour here");

    const count = model.cells.length;
    // A palette nothing reads from stops the node, so there is no blend to qualify.
    if (!count) return "";
    return withCaveat(`${count} ${count === 1 ? "colour" : "colours"}`);
  }

  /**
   * Draw the whole strip.
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
    const fromSocket = socketDrives();
    const model = readModel();

    if (fromSocket) {
      ctx.fillStyle = theme.inputBg;
      ctx.fillRect(layout.stripX0, layout.stripY, layout.stripWidth, layout.stripHeight);
      ctx.font = BODY_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(
        "color_palettes is connected",
        layout.stripX0 + layout.stripWidth / 2,
        layout.stripY + layout.stripHeight / 2,
        layout.stripWidth - 4,
      );
    } else {
      drawStrip(ctx, ratio, theme, model);
      drawRail(ctx, theme, model);
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      layout.stripX0 + 0.5,
      layout.stripY + 0.5,
      Math.max(1, layout.stripWidth - 1),
      Math.max(1, layout.stripHeight - 1),
    );

    drawFooter(ctx, theme, model, fromSocket);

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
        console.error(`[${EXT_NAME}] Failed to draw the palette strip:`, error);
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
        console.error(`[${EXT_NAME}] Palette strip input failed:`, error);
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
    state.press = null;

    const point = localPoint(event);

    // The count of dropped fields leads to the text they are in, and is only offered as a
    // target while that text is reachable. The focus is taken on the next frame, after the
    // click has finished moving focus around by itself.
    if (hitFooterLink(point)) {
      const widget = findWidget(node, PALETTE_WIDGET);
      requestAnimationFrame(() => widget?.element?.focus?.());
      return;
    }

    if (socketDrives()) return;

    // The pointer default action is left alone throughout. Cancelling it would suppress the
    // mouse events that follow, which carry the graph snapshot that gives a gesture made here
    // its own undo entry.
    const model = readModel();
    const cell = hitCell(model, point);
    if (cell) {
      state.selectedField = cell.field.index;
      state.press = { pointerId: event.pointerId, field: cell.field.index, gap: null };
      schedulePaint();
      return;
    }

    const gap = hitGap(model, point);
    if (gap !== null) {
      state.press = { pointerId: event.pointerId, field: null, gap };
      schedulePaint();
      return;
    }

    state.selectedField = null;
    schedulePaint();
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    // A button released off the element, over the graph canvas for example, delivers no
    // pointerup here and no cancel, since the strip captures nothing. Without this the press
    // would stay armed and the next release over a cell, from any gesture at all, would open
    // the picker for it.
    if (state.press && !(event.buttons & 1)) {
      state.press = null;
      return;
    }

    const point = localPoint(event);
    if (socketDrives()) {
      root.style.cursor = "default";
      return;
    }

    const model = readModel();
    const cell = hitCell(model, point);
    const gap = cell ? null : hitGap(model, point);
    const hoverField = cell ? cell.field.index : null;

    root.style.cursor = cell
      ? "pointer"
      : gap !== null
        ? "copy"
        : hitFooterLink(point)
          ? "pointer"
          : "default";

    if (hoverField !== state.hoverField || gap !== state.hoverGap) {
      state.hoverField = hoverField;
      state.hoverGap = gap;
      schedulePaint();
    }
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    if (event.button !== 0) return;

    const press = state.press;
    state.press = null;
    if (!press || press.pointerId !== event.pointerId || socketDrives()) return;

    // The text can have been edited by hand between the two halves of the gesture, so what is
    // under the pointer now is what the picker is opened for, and only when it is still what
    // the gesture started on.
    const point = localPoint(event);
    const model = readModel();

    if (press.field !== null) {
      const cell = hitCell(model, point);
      if (cell && cell.field.index === press.field) {
        pickCellColour(cell, event.clientX, event.clientY);
      }
      return;
    }

    const gap = hitGap(model, point);
    if (gap !== null && gap === press.gap) pickGapColour(gap, event.clientX, event.clientY);
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
    if (socketDrives()) return;

    const model = readModel();
    const cell = hitCell(model, localPoint(event));
    if (!cell) return;
    state.selectedField = cell.field.index;
    removeCell(cell);
  };

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;

    // A linked socket leaves no cell to select, so the keys that move a selection are left to
    // the canvas rather than consumed for nothing. The two that act on a cell still answer for
    // themselves: Delete reaches the binding that removes the node.
    const model = socketDrives() ? readPalette("") : readModel();
    const cells = model.cells;
    const cell = selectedCell(model);
    const current = cell ? cells.indexOf(cell) : -1;
    let handled = true;

    switch (event.key) {
      case "ArrowLeft":
      case "ArrowRight": {
        if (!cells.length) {
          handled = false;
          break;
        }
        const step = event.key === "ArrowLeft" ? -1 : 1;
        const next =
          current < 0
            ? step < 0
              ? cells.length - 1
              : 0
            : clamp(current + step, 0, cells.length - 1);
        state.selectedField = cells[next].field.index;
        schedulePaint();
        break;
      }
      case "Home":
      case "End": {
        if (!cells.length) {
          handled = false;
          break;
        }
        state.selectedField = cells[event.key === "Home" ? 0 : cells.length - 1].field.index;
        schedulePaint();
        break;
      }
      case "Enter":
      case " ": {
        if (socketDrives() || !cell) {
          setMessage(socketDrives() ? "the palette box is not read" : "select a colour first");
          break;
        }
        const rect = cellRect(state.layout, cells.length, current);
        const point = screenPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
        pickCellColour(cell, point.clientX, point.clientY);
        break;
      }
      case "Delete":
      case "Backspace": {
        // Consumed whether or not a cell is selected. Left unhandled these reach ComfyUI's own
        // binding, which deletes the node the strip is drawn on.
        if (socketDrives() || !cell) {
          setMessage(socketDrives() ? "the palette box is not read" : "select a colour first");
          break;
        }
        removeCell(cell);
        break;
      }
      case "Escape": {
        state.press = null;
        state.selectedField = null;
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

  const onBlur = () => {
    state.press = null;
    state.hoverField = null;
    state.hoverGap = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(() => {
    state.press = null;
  }));
  root.addEventListener("lostpointercapture", guard(() => {
    state.press = null;
  }));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hoverField === null && state.hoverGap === null) return;
    state.hoverField = null;
    state.hoverGap = null;
    schedulePaint();
  }));
  root.addEventListener("contextmenu", guard(onContextMenu));
  // The strip scrolls nothing of its own, so it takes every wheel gesture over it and the
  // graph zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);
  root.addEventListener("keydown", guard(onKeyDown));
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
   * Release the timers, observers, listeners and hover text the strip holds.
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
    handlePaletteChanged,
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
 * Append the strip to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPaletteStrip(node) {
  if (!findWidget(node, PALETTE_WIDGET)) return;

  const strip = createPaletteStrip(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, strip, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Every multiline box on the node bounded the same way, so the panel above takes
  // the room past their ceiling instead of losing all of it to them.
  boundTextBoxes(node);

  chainWidgetCallback(node, PALETTE_WIDGET, strip.handlePaletteChanged);
  for (const name of CAPTION_WIDGETS) {
    chainWidgetCallback(node, name, strip.schedulePaint);
  }

  // Linking `color_palettes` leaves the text box read by nothing, which the strip says instead
  // of previewing a palette the node will not use.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      strip.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      strip.schedulePaint();
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
      strip.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the palette strip:`, error);
    }
    return result;
  };

  strip.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Palette Map", "Palette strip"],
      name: "Show the palette strip",
      tooltip:
        "Draw the palette strip under the palette widget of Image Palette Map. The widget " +
        "itself is always available. This applies to nodes added after the setting changes, " +
        "so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second strip.
    if (proto.__was_palette_map_wrapped) return;
    proto.__was_palette_map_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachPaletteStrip(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the palette strip:`, error);
      }
      return result;
    };
  },
});
