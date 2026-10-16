/**
 * Interface for the Image Draw Text node: the text over the picture, with its colours editable.
 *
 * The layout is in pixels of the image the node draws on; alpha is a byte and the rows are in
 * element pixels.
 */

import { app } from "../../scripts/app.js";
import { imageBackdrop, normaliseFrame } from "./interface/backdrop.js";
import {
  STATUS,
  drawCell,
  formatColour,
  outlineCell,
  parseColor,
  pickColour,
  residualNote,
  tallyColours,
} from "./interface/colour_cell.js";
import { faceMetrics, fontFamily, loadFont } from "./interface/fonts.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { drawStandIn, loadPlaceholder, standInDetail } from "./interface/placeholder.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { PREVIEW_STATE } from "./interface/preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeDone } from "./interface/run_events.js";
import { ALIGN, layoutText, strokeLineWidth } from "./interface/text_layout.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.DrawTextUI";
const NODE_ID = "WASImageDrawText";
const SETTING_ID = "WAS.DrawText.ShowInterface";

const TEXT_WIDGET = "text";
const FONT_SIZE_WIDGET = "font_size";
const FONT_PATH_WIDGET = "font_path";
const TEXT_COLOUR_WIDGET = "text_color";
const POSITION_WIDGET = "position";
const ALIGN_WIDGET = "align";
const OFFSET_X_WIDGET = "offset_x";
const OFFSET_Y_WIDGET = "offset_y";
const MARGIN_WIDGET = "margin";
const LINE_SPACING_WIDGET = "line_spacing";
const WRAP_WIDTH_WIDGET = "wrap_width";
const STROKE_WIDTH_WIDGET = "stroke_width";
const STROKE_COLOUR_WIDGET = "stroke_color";
const PANEL_COLOUR_WIDGET = "background_color";
const PANEL_PADDING_WIDGET = "background_padding";
const OPACITY_WIDGET = "opacity";
const FONT_WIDGET = "font";

const UI_WIDGET_NAME = "was_draw_text_ui";
const UI_WIDGET_TYPE = "was_draw_text_preview";

// Every input the preview reads. All of them are schema inputs a link can fill, and a link is what
// the run reads instead of the widget, so the panel says so rather than drawing a value nothing
// reads.
const READ_WIDGETS = [
  TEXT_WIDGET,
  FONT_SIZE_WIDGET,
  FONT_PATH_WIDGET,
  TEXT_COLOUR_WIDGET,
  POSITION_WIDGET,
  ALIGN_WIDGET,
  OFFSET_X_WIDGET,
  OFFSET_Y_WIDGET,
  MARGIN_WIDGET,
  LINE_SPACING_WIDGET,
  WRAP_WIDTH_WIDGET,
  STROKE_WIDTH_WIDGET,
  STROKE_COLOUR_WIDGET,
  PANEL_COLOUR_WIDGET,
  PANEL_PADDING_WIDGET,
  OPACITY_WIDGET,
  FONT_WIDGET,
];

// What `parse_color` returns for a value it cannot read, at each of the three call sites. The
// node passes no default for the glyphs or the outline, so both fall back to `draw.FALLBACK`,
// and passes `draw.TRANSPARENT` for the panel, so an unreadable panel colour draws nothing.
const FALLBACK = [255, 255, 255, 255];
const TRANSPARENT = [0, 0, 0, 0];

// The three colours, in the order the node's widgets carry them. `slot` names the layer the
// colour belongs to, and `fallback` is what the node draws with when the value is unreadable.
const ROWS = [
  { slot: "fill", label: "text", widget: TEXT_COLOUR_WIDGET, fallback: FALLBACK },
  { slot: "stroke", label: "stroke", widget: STROKE_COLOUR_WIDGET, fallback: FALLBACK },
  { slot: "panel", label: "panel", widget: PANEL_COLOUR_WIDGET, fallback: TRANSPARENT },
];

const COLOUR_WIDGETS = ROWS.map((row) => row.widget);

// `modules.image.draw.DEFAULT_FONT`, which is what the node draws with when the optional `font`
// input is absent from the prompt.
const DEFAULT_FONT = "DejaVu Sans";

// What the preview draws in while the node's own face is not available. A generic family rather
// than a named one, so it is whatever this machine calls its sans face.
const SUBSTITUTE_FAMILY = "sans-serif";

// The size the layout is worked out against until the node has run and published the image it was
// given. Nothing here is written in frame units, so an assumed frame costs a stored value nothing,
// and the footer names it as assumed for as long as it is one.
const ASSUMED_FRAME = 512;

// The built-in tokens `TextTokens` resolves before the node draws.
const BUILT_IN_TOKENS = /\[(?:time(?:\([^)\]]*\))?|hostname|user|cuda_device|cuda_name)\]/g;

// Any other name in brackets. `Text Add Tokens` stores a token under the name typed into it and
// `parseTokens` replaces that name wherever it appears, so a bracketed name is a custom token
// while one of that name is defined and is drawn as written while none is. Which of the two it is
// cannot be read from here: the table lives in the settings database and outlives the prompt that
// wrote it. The brackets are the convention rather than the rule, so a token stored as `project`
// is plain text to this pattern, which the hover says wherever one of these is found.
const BRACKETED_NAMES = /\[[^[\]]*\]/g;

// Height of the appended widget in node units. A DOM widget element is inset by the widget's
// margin on every side, so the element itself is shorter by twice that margin.
const UI_HEIGHT = 300;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout, measured in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const GAP = 5;
const ROW_HEIGHT = 15;
const LABEL_WIDTH = 40;
const CELL_WIDTH = 40;
const CELL_INSET = 2;
const CLEAR_SIZE = 11;
const PREVIEW_GAP = 5;
const MIN_PREVIEW = 40;

// The footer's two lines. The first carries the block the widgets produce and whatever the last
// gesture, the links or an empty text have to say. The second carries the face being drawn in and
// the size the layout is measured against, which are standing facts and never give up their room.
const FOOTER_HEIGHT = 13;
const FOOTER_LINES = 2;

// The gap kept between the glyph and the words after it.
const GLYPH_GAP = 4;

const BODY_FONT = "10px sans-serif";
const SMALL_FONT = "9px sans-serif";
const LABEL_FONT = "11px sans-serif";

const SELECTED_ALPHA = 0.14;

// Alpha bytes covered by one element pixel of drag, and the step Shift snaps to.
const ALPHA_PER_PIXEL = 2;
const ALPHA_COARSE_STEP = 16;

// How far a press has to travel before it is a drag rather than a click.
const DRAG_THRESHOLD = 3;

// A click opens the picker, so a double click would open it twice and the second would take
// over the handler of the one already on screen.
const PICKER_COOLDOWN = 350;

const MESSAGE_TIMEOUT = 4000;

const CLEAR_GLYPH = "x";

// How long to wait before asking for the picture again while the answer is not the picture, and
// the ceiling that wait doubles up to. A node placed and never run answers nothing for the life of
// the page, and there is no point asking it at the rate a node about to run is asked at.
const RETRY_INTERVAL = 3000;
const RETRY_MAX_INTERVAL = 30000;
const RETRY_BACKOFF = 2;

// What the panel says where the text could not be drawn at all, which is a canvas or a context the
// browser would not give rather than anything about the node.
const NO_LAYER = "the text could not be drawn";

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
 * Whether an input is filled in by a link.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {string} name - Input name, which is the widget's name as well.
 * @returns {boolean} True while a link is connected to that input.
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
 * Round a value to a multiple of a step.
 *
 * @param {number} value - Value to snap.
 * @param {number} step - Step to snap to.
 * @returns {number} The nearest multiple of the step.
 */
function snap(value, step) {
  return step > 0 ? Math.round(value / step) * step : value;
}

/**
 * Write an RGBA quadruple as the colour a canvas takes.
 *
 * @param {number[]} rgba - Four channels, 0 to 255.
 * @returns {string} A CSS colour, with the alpha as a fraction.
 */
function cssColour(rgba) {
  return `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3] / 255})`;
}

/**
 * The RGBA the node will draw one row with.
 *
 * @param {object} parsed - Result from `parseColor`.
 * @param {number[]} fallback - What the node's own call to `parse_color` falls back to.
 * @returns {number[]|null} Four channels, or null when the colour is not known here.
 */
function effectiveColour(parsed, fallback) {
  if (parsed?.status === STATUS.COLOUR) return parsed.rgba;
  // An empty value is fully transparent at all three rows, which is no panel, an outline that
  // cuts through whatever is under it, and glyphs that leave only their own hole.
  if (parsed?.status === STATUS.EMPTY) return TRANSPARENT;
  // An unreadable value resolves to whatever `parse_color` was given as a default at that call
  // site, so the fallback travels with the row rather than being decided here.
  if (parsed?.status === STATUS.INVALID) return fallback;
  // A spelling the cell declines has no answer here: the node resolves it and the interface did
  // not read it, so nothing that stands for the render may claim a colour for it.
  return null;
}

/**
 * Read whether the interface is drawn at all.
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
 * Work out where each part of the interface sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the picture, the rows and the footer.
 */
function computeLayout(width, height) {
  const contentX0 = PAD_X;
  const contentX1 = Math.max(contentX0 + 1, width - PAD_X);
  const contentWidth = contentX1 - contentX0;
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT * FOOTER_LINES);
  const rowsHeight = ROW_HEIGHT * ROWS.length;
  const rowsY = Math.max(PAD_Y, footerY - rowsHeight - PREVIEW_GAP);
  const cellWidth = Math.max(
    8,
    Math.min(CELL_WIDTH, contentWidth - LABEL_WIDTH - GAP * 2 - CLEAR_SIZE),
  );

  return {
    width,
    height,
    contentX0,
    contentX1,
    contentWidth,
    rowsY,
    rowsHeight,
    cellWidth,
    footerY,
    preview: {
      x: contentX0,
      y: PAD_Y,
      w: contentWidth,
      h: Math.max(MIN_PREVIEW, rowsY - PREVIEW_GAP - PAD_Y),
    },
  };
}

/**
 * The strip one row occupies.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} index - Row index.
 * @returns {{x: number, y: number, width: number, height: number}} The row in element pixels.
 */
function rowRect(layout, index) {
  return {
    x: layout.contentX0,
    y: layout.rowsY + index * ROW_HEIGHT,
    width: layout.contentWidth,
    height: ROW_HEIGHT,
  };
}

/**
 * The swatch inside one row.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} index - Row index.
 * @returns {{x: number, y: number, width: number, height: number}} The cell in element pixels.
 */
function cellRect(layout, index) {
  return {
    x: layout.contentX0 + LABEL_WIDTH + GAP,
    y: layout.rowsY + index * ROW_HEIGHT + CELL_INSET,
    width: layout.cellWidth,
    height: ROW_HEIGHT - CELL_INSET * 2,
  };
}

/**
 * The control at the end of one row that empties it.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {number} index - Row index.
 * @returns {{x: number, y: number, width: number, height: number}} The control in element
 *   pixels.
 */
function clearRect(layout, index) {
  return {
    x: layout.contentX1 - CLEAR_SIZE,
    y: layout.rowsY + index * ROW_HEIGHT + Math.round((ROW_HEIGHT - CLEAR_SIZE) / 2),
    width: CLEAR_SIZE,
    height: CLEAR_SIZE,
  };
}

/**
 * Where one row writes its value out, and how much room it has.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @returns {{x: number, width: number}} Left edge and available width in element pixels.
 */
function valueSlot(layout) {
  const x = layout.contentX0 + LABEL_WIDTH + GAP + layout.cellWidth + GAP;
  return { x, width: Math.max(0, layout.contentX1 - CLEAR_SIZE - GAP - x) };
}

/**
 * Test whether a point is inside a rectangle.
 *
 * @param {{x: number, y: number}} point - Position in element pixels.
 * @param {{x: number, y: number, width: number, height: number}} rect - Rectangle to test.
 * @returns {boolean} True when the point is inside.
 */
function inside(point, rect) {
  return (
    point.x >= rect.x &&
    point.x <= rect.x + rect.width &&
    point.y >= rect.y &&
    point.y <= rect.y + rect.height
  );
}

/**
 * Fit the image into the panel, keeping the image's own aspect.
 *
 * @param {{x: number, y: number, w: number, h: number}} area - The panel in element pixels.
 * @param {object} frame - Frame from `normaliseFrame`.
 * @returns {{x: number, y: number, w: number, h: number}|null} Where the picture goes, or null
 *   while there is no size to draw it at.
 */
function fitFrame(area, frame) {
  if (!(area.w > 0) || !(area.h > 0)) return null;
  if (!(frame?.width > 0) || !(frame?.height > 0)) return null;
  const fit = Math.min(area.w / frame.width, area.h / frame.height);
  const w = Math.max(1, frame.width * fit);
  const h = Math.max(1, frame.height * fit);
  return { x: area.x + (area.w - w) / 2, y: area.y + (area.h - h) / 2, w, h };
}

/**
 * Convert a rectangle in element pixels into whole device pixels.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - Rectangle in element pixels.
 * @param {number} ratio - Device pixels per element pixel.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle in device pixels.
 */
function deviceRect(rect, ratio) {
  // Each edge is rounded on its own, so the picture never gains or loses a pixel to its neighbour.
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
 * The names in brackets a string carries, split by what the run puts in their place.
 *
 * @param {string} text - The text widget's value.
 * @returns {{builtIn: string[], custom: string[]}} The built-in tokens, and the bracketed names
 *   that are not built-in, each once, in the order they first appear.
 */
function tokensIn(text) {
  const value = String(text ?? "");
  const builtIn = [];
  for (const match of value.matchAll(BUILT_IN_TOKENS)) {
    if (!builtIn.includes(match[0])) builtIn.push(match[0]);
  }
  const custom = [];
  for (const match of value.matchAll(BRACKETED_NAMES)) {
    if (builtIn.includes(match[0]) || custom.includes(match[0])) continue;
    custom.push(match[0]);
  }
  return { builtIn, custom };
}

/**
 * Word a list of linked inputs the way the pack words it.
 *
 * @param {string[]} linked - Input names a link fills in.
 * @returns {string} The note, empty when nothing is linked.
 */
function linkedNote(linked) {
  if (linked.length === 1) return `${linked[0]} is linked`;
  if (linked.length > 1) return `${linked.length} inputs are linked`;
  return "";
}

/**
 * Build the interface for one node.
 *
 * @param {object} node - The node the interface decorates.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   handleWidgetChanged: (name: string) => void, refresh: () => void, dispose: () => void}} The
 *   element to hand to `appendInterfaceWidget`, the height it was built for, a coalesced repaint,
 *   the repaint to run when a widget changed, a fresh ask for the picture, and teardown.
 */
function createTextInterface(node) {
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

  // The footer's glyph states what the preview is worth, and the footer band states what its
  // numbers are measured in, both through the element's own title. The regions are handed over
  // again on every repaint, since they move whenever the node is resized.
  const titles = hoverTitles(root);

  const backdrop = imageBackdrop(node, { width: ASSUMED_FRAME, height: ASSUMED_FRAME });

  const state = {
    frame: normaliseFrame({
      state: PREVIEW_STATE.LOADING,
      width: ASSUMED_FRAME,
      height: ASSUMED_FRAME,
    }),
    // The size of the image the node was given, once it has published one. Held past a later
    // answer that carries none, since the store a published picture sits in is bounded and evicts
    // the least recently used. A thumbnail ageing out of a busy graph is not a size change.
    known: null,
    loading: false,
    reloadWanted: false,
    retryTimer: 0,
    retryWait: RETRY_INTERVAL,
    fontAsked: "",
    fontMissing: "",
    selected: null,
    hover: null,
    hoverClear: null,
    press: null,
    pending: null,
    lastWritten: {},
    pickedAt: 0,
    message: "",
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    layer: null,
    disposed: false,
  };

  /**
   * Read one widget as a number.
   *
   * @param {string} name - Widget name.
   * @param {number} fallback - Used when the widget is missing or holds no number.
   * @returns {number} The widget's value.
   */
  function readNumber(name, fallback) {
    const value = Number(findWidget(node, name)?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  /**
   * Read one widget as a string.
   *
   * @param {string} name - Widget name.
   * @param {string} fallback - Used when the widget is missing or holds nothing.
   * @returns {string} The widget's value.
   */
  function readString(name, fallback = "") {
    const value = findWidget(node, name)?.value;
    return typeof value === "string" && value !== "" ? value : fallback;
  }

  /**
   * Read everything the preview draws from, in the node's own units.
   *
   * @returns {object} The widget values as the node reads them, the three resolved colours, the
   *   rows the swatches are drawn from, and which inputs a link fills in.
   */
  function readModel() {
    const rows = readRows();
    const layer = {};
    for (const row of rows) layer[row.spec.slot] = row.live ? row.effective : TRANSPARENT;

    const strokeWidth = readStrokeWidth();
    const padding = Math.max(0, Math.trunc(readNumber(PANEL_PADDING_WIDGET, 8)));
    const text = String(findWidget(node, TEXT_WIDGET)?.value ?? "");
    const fontPath = readString(FONT_PATH_WIDGET).trim();

    return {
      text,
      tokens: tokensIn(text),
      // Every fallback here is the default `execute` carries, since a widget the node does not
      // have is a key absent from the prompt, which is the case that reaches the Python default.
      // `draw.load_font` holds the size at one or more, as the schema's own minimum does.
      fontSize: Math.max(1, Math.trunc(readNumber(FONT_SIZE_WIDGET, 32))),
      fontName: readString(FONT_WIDGET, DEFAULT_FONT),
      fontPath,
      position: readString(POSITION_WIDGET, "bottom center"),
      align: readString(ALIGN_WIDGET, ALIGN.CENTER),
      offset: [
        Math.trunc(readNumber(OFFSET_X_WIDGET, 0)),
        Math.trunc(readNumber(OFFSET_Y_WIDGET, 0)),
      ],
      margin: Math.trunc(readNumber(MARGIN_WIDGET, 16)),
      // `draw.text_block` holds the spacing at 0.1 or more.
      lineSpacing: Math.max(0.1, readNumber(LINE_SPACING_WIDGET, 1)),
      wrapWidth: Math.trunc(readNumber(WRAP_WIDTH_WIDGET, 0)),
      strokeWidth,
      backgroundPadding: padding,
      // `draw.composite` holds the opacity at zero or more, and applies it only below 1.0.
      opacity: Math.max(0, readNumber(OPACITY_WIDGET, 1)),
      rows,
      layer,
      linked: READ_WIDGETS.filter((name) => inputLinked(node, name)),
    };
  }

  /**
   * Read `stroke_width`, held to whole pixels at or above zero as the node reads it.
   *
   * @returns {number} The outline width in pixels.
   */
  function readStrokeWidth() {
    return Math.max(0, Math.trunc(readNumber(STROKE_WIDTH_WIDGET, 0)));
  }

  /**
   * Read the three colours, with any unfinished alpha gesture applied.
   *
   * @returns {object[]} One entry per row: its specification, the parse the cell reads, the
   *   RGBA the node will draw with, the text the row shows, whether the node reads it, and
   *   whether it is dimmed.
   */
  function readRows() {
    const strokeWidth = readStrokeWidth();

    // A row shows its colour as the widget holds it, alpha and all. `opacity` is left off it,
    // since `opacity` applies to the finished layer rather than to one colour in it, and the
    // picture is where the layer fades as a whole.
    return ROWS.map((spec) => {
      const parsed = parseColor(findWidget(node, spec.widget)?.value);
      let display = parsed;

      if (state.pending?.widget === spec.widget && parsed.status === STATUS.COLOUR) {
        const rgba = [parsed.rgba[0], parsed.rgba[1], parsed.rgba[2], state.pending.alpha];
        display = { ...parsed, rgba, text: formatColour(rgba) };
      }

      const effective = effectiveColour(display, spec.fallback);

      let value = display.text;
      if (display.status === STATUS.EMPTY) value = "none";
      else if (display.status === STATUS.INVALID) {
        // The row names the colour the node will actually use, since the text it cannot read is
        // already in the widget above it.
        value = spec.fallback[3] > 0 ? formatColour(spec.fallback) : "none";
      }

      // `stroke_color` reaches a pixel only while `stroke_width` is 1 or more, since
      // `draw_text_layer` passes the outline colour to `ImageDraw` only then, so at a width of 0
      // the row is not one of the colours the render is built from. A row that is not live is
      // dimmed, left out of the picture and left out of the footer's count.
      const live = spec.widget !== STROKE_COLOUR_WIDGET || strokeWidth > 0;

      // The panel row is dimmed on a different test: it is read whenever it holds a colour, and
      // `background_padding` reaches a pixel only while a panel is actually drawn behind the text.
      let dim = !live;
      if (spec.widget === PANEL_COLOUR_WIDGET) dim = effective !== null && effective[3] === 0;

      return { spec, parsed: display, effective, value, live, dim };
    });
  }

  /**
   * What one layer replacing another underneath it does, which a canvas cannot reproduce.
   *
   * @param {object} model - Model from `readModel`.
   * @returns {string} The state to draw, empty when no mark sits over another.
   */
  function cutNote(model) {
    const layer = model.layer;
    if (!layer.fill || !layer.stroke || !layer.panel) return "";
    const outlined = model.strokeWidth > 0 && layer.stroke[3] > 0;
    const panelled = layer.panel[3] > 0;
    if (outlined && layer.fill[3] < 255) return "text cuts its stroke";
    if (panelled && model.strokeWidth > 0 && layer.stroke[3] < 255) return "stroke cuts the panel";
    if (panelled && model.strokeWidth === 0 && layer.fill[3] < 255) return "text cuts the panel";
    return "";
  }

  /**
   * Ask for the typeface the node will draw with, once per name.
   *
   * @param {object} model - Model from `readModel`.
   * @returns {void}
   */
  function ensureFont(model) {
    const wanted = model.fontName;
    if (!wanted || state.fontAsked === wanted) return;
    state.fontAsked = wanted;
    loadFont(wanted)
      .then((family) => {
        // A name the pack does not serve resolves to null, which is a state to draw rather than a
        // fault, and it is what separates a typeface still crossing from one that is not coming.
        if (!family) state.fontMissing = wanted;
        if (!state.disposed) schedulePaint();
      })
      .catch((error) => {
        console.error(`[${EXT_NAME}] Failed to load the font ${wanted}:`, error);
      });
  }

  /**
   * The face the preview draws in, and how much of it is the node's own.
   *
   * @param {CanvasRenderingContext2D} ctx - Context already carrying the font at `family`.
   * @param {object} model - Model from `readModel`.
   * @param {string|null} family - The family the node's own face was loaded under, or null.
   * @returns {{ascent: number, descent: number, family: string, own: boolean, exact: boolean}|null}
   *   The metrics to lay out with, the family drawn in, whether that family is the node's own face,
   *   and whether the metrics are FreeType's own. Null when neither source has metrics.
   */
  function readFace(ctx, model, family) {
    const own = Boolean(family);
    const drawn = family ?? SUBSTITUTE_FAMILY;
    const metrics = faceMetrics(model.fontName, model.fontSize);
    if (metrics) return { ...metrics, family: drawn, own, exact: true };

    // No tables to read, so the browser's own idea of the face has to do. It is a different
    // quantity from the one `text_block` reads, and is reached from here alone.
    const box = ctx.measureText("Hg");
    const ascent = Number(box.fontBoundingBoxAscent);
    const descent = Number(box.fontBoundingBoxDescent);
    if (!Number.isFinite(ascent) || !(ascent > 0)) return null;
    return {
      ascent: Math.ceil(ascent),
      descent: Math.ceil(Number.isFinite(descent) ? descent : 0),
      family: drawn,
      own,
      exact: false,
    };
  }

  /**
   * Ask the backdrop for the picture, and again later while the answer is not the picture.
   *
   * @returns {void}
   */
  function loadFrame() {
    if (state.disposed) return;
    // An ask made while one is already in flight is recorded rather than dropped, since the answer
    // on its way was asked for before whatever prompted the second one.
    if (state.loading) {
      state.reloadWanted = true;
      return;
    }
    state.loading = true;
    state.reloadWanted = false;
    Promise.resolve()
      .then(() => backdrop.load())
      .then((answer) => {
        if (state.disposed) return;
        state.frame = keepSize(normaliseFrame(answer));
      })
      .catch((error) => {
        if (state.disposed) return;
        console.error(`[${EXT_NAME}] Failed to read the picture:`, error);
        state.frame = keepSize(normaliseFrame({
          state: PREVIEW_STATE.FAILED,
          width: ASSUMED_FRAME,
          height: ASSUMED_FRAME,
        }));
      })
      .finally(() => {
        state.loading = false;
        if (state.disposed) return;
        schedulePaint();
        if (state.reloadWanted) loadFrame();
        else scheduleRetry();
      });
  }

  /**
   * Record the size a published picture states, and keep it once one has.
   *
   * @param {object} frame - The answer, from `normaliseFrame`.
   * @returns {object} The answer, carrying the last published size where it states none of its own.
   */
  function keepSize(frame) {
    if (frame.state === PREVIEW_STATE.READY && frame.width > 0 && frame.height > 0) {
      state.known = { width: frame.width, height: frame.height };
      return frame;
    }
    // The state is the answer's own, so the stand-in comes back over a size the layout goes on
    // being worked out in.
    return state.known ? { ...frame, ...state.known } : frame;
  }

  /**
   * Ask again for a picture that has not arrived.
   *
   * @returns {void}
   */
  function scheduleRetry() {
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.retryTimer = 0;
    if (state.disposed) return;
    if (state.frame.state === PREVIEW_STATE.READY) {
      state.retryWait = RETRY_INTERVAL;
      return;
    }
    const wait = state.retryWait;
    state.retryWait = Math.min(RETRY_MAX_INTERVAL, wait * RETRY_BACKOFF);
    state.retryTimer = setTimeout(() => {
      state.retryTimer = 0;
      // A hidden tab publishes nothing new to ask about, so the wait starts again instead.
      if (document.hidden) scheduleRetry();
      else loadFrame();
    }, wait);
  }

  /**
   * Ask for the picture again now.
   *
   * @returns {void}
   */
  function refresh() {
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.retryTimer = 0;
    state.retryWait = RETRY_INTERVAL;
    loadFrame();
  }

  /**
   * Whether the stand-in picture is what goes behind the text.
   *
   * @returns {boolean} True where the backdrop has no picture and a picture is what is missing,
   *   rather than a fault to report.
   */
  function standingIn() {
    const frame = state.frame;
    if (frame.image) return false;
    // A failure keeps the words that say so. It is the one state of the five that is a fault, and
    // a picture over it would say that nothing is wrong.
    return frame.state !== PREVIEW_STATE.FAILED;
  }

  /**
   * Whether the size the layout is worked out against is the image the node received.
   *
   * @returns {boolean} True while the frame is the assumed one rather than a published picture's.
   */
  function frameAssumed() {
    return state.known === null;
  }

  /**
   * Show a short note in the footer.
   *
   * @param {string} text - Note to show.
   * @returns {void}
   */
  function setMessage(text) {
    if (state.disposed) return;
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
   * Write one colour widget, once.
   *
   * @param {string} name - Widget name.
   * @param {string} text - Text to store.
   * @returns {void}
   */
  function writeColour(name, text) {
    if (state.disposed) return;
    const widget = findWidget(node, name);
    if (!widget) return;
    // The value is compared first, so a repaint driven by the widget's own callback can never
    // write anything back.
    if (typeof text !== "string" || text === widget.value) return;

    const graph = app.canvas;
    const transactional =
      typeof graph?.emitBeforeChange === "function" &&
      typeof graph?.emitAfterChange === "function";

    // The write is bracketed by the canvas change events the graph's change tracker listens for,
    // which is what gives the edit its own undo entry. The tracker's own snapshot triggers are a
    // document `mouseup` and the release of a bare modifier key, so a commit from the keyboard or
    // from the colour picker reaches none of them and would otherwise be folded into whatever the
    // previous snapshot held.
    state.lastWritten[name] = text;
    if (transactional) graph.emitBeforeChange();
    try {
      widget.value = text;
    } finally {
      if (transactional) graph.emitAfterChange();
    }
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Write the alpha byte an unfinished gesture holds.
   *
   * @returns {void}
   */
  function commitPending() {
    const pending = state.pending;
    state.pending = null;
    if (!pending) return;

    // The widget is read again here, so a value that changed under the gesture keeps what it
    // changed to rather than being overwritten with an alpha meant for the colour it held before.
    const parsed = parseColor(findWidget(node, pending.widget)?.value);
    // A gesture that ends on the alpha it started from writes nothing. An alpha edit has to be
    // rewritten as hex, since the alpha byte has nowhere else to live, and a gesture that moved no
    // alpha would otherwise spend that rewrite on nothing: a name or a short hex would be replaced
    // by a longer spelling of the same colour, an undo entry would be taken for a gesture that
    // changed no pixel, and the prompt string would change enough to make the node render an
    // identical image again.
    if (parsed.status === STATUS.COLOUR && parsed.rgba[3] !== pending.alpha) {
      writeColour(
        pending.widget,
        formatColour([parsed.rgba[0], parsed.rgba[1], parsed.rgba[2], pending.alpha]),
      );
    }
    schedulePaint();
  }

  /**
   * Hold an alpha byte for a gesture in progress, committing one already held for another row.
   *
   * @param {string} name - Widget name.
   * @param {number} alpha - Alpha the gesture has reached.
   * @returns {void}
   */
  function holdPending(name, alpha) {
    if (state.pending && state.pending.widget !== name) commitPending();
    if (state.pending?.alpha === alpha) return;
    state.pending = { widget: name, alpha };
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
      if (state.pending?.widget === name) state.pending = null;
    }
    schedulePaint();
  }

  /**
   * The index of the selected row.
   *
   * @returns {number} The row index, or -1 when nothing is selected.
   */
  function selectedIndex() {
    return ROWS.findIndex((spec) => spec.widget === state.selected);
  }

  /**
   * The row a point is over.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {number} The row index, or -1 when the point is not over a row.
   */
  function rowIndexAt(point) {
    const layout = state.layout;
    if (point.x < layout.contentX0 || point.x > layout.contentX1) return -1;
    if (point.y < layout.rowsY || point.y >= layout.rowsY + layout.rowsHeight) return -1;
    return Math.floor((point.y - layout.rowsY) / ROW_HEIGHT);
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
   * Empty one row, which is how a widget says draw nothing here.
   *
   * @param {number} index - Row index.
   * @returns {void}
   */
  function clearRow(index) {
    const spec = ROWS[index];
    if (!spec) return;
    if (state.pending?.widget === spec.widget) state.pending = null;
    writeColour(spec.widget, "");
    schedulePaint();
  }

  /**
   * Open the native picker for one row and write the colour it returns.
   *
   * @param {number} index - Row index.
   * @param {number} clientX - Horizontal position on screen.
   * @param {number} clientY - Vertical position on screen.
   * @returns {void}
   */
  function openPicker(index, clientX, clientY) {
    const rows = readRows();
    const row = rows[index];
    if (!row) return;

    // A declined spelling is refused and left exactly as written: the node resolves it and the
    // interface does not, so the picker would open on a colour the row does not hold and the write
    // would replace a value that renders perfectly well.
    if (row.parsed.status === STATUS.DECLINED) {
      setMessage(row.parsed.note);
      return;
    }

    const now = Date.now();
    if (now - state.pickedAt < PICKER_COOLDOWN) return;
    state.pickedAt = now;

    // The picker opens on the colour the node draws for that row now, which for an unreadable
    // value is the fallback it falls back to and for an empty one is black, since a native picker
    // carries no alpha and has no way to open on nothing.
    pickColour(clientX, clientY, row.effective ?? row.spec.fallback, (rgb) => {
      // The alpha the row already held is kept, read again here rather than captured when the
      // picker opened, and a row with no colour of its own becomes opaque.
      const current = parseColor(findWidget(node, row.spec.widget)?.value);
      const alpha = current.status === STATUS.COLOUR ? current.rgba[3] : 255;
      state.selected = row.spec.widget;
      writeColour(row.spec.widget, formatColour([rgb[0], rgb[1], rgb[2], alpha]));
      schedulePaint();
    });
  }

  /**
   * The canvas the text layer is built on, at the size asked for.
   *
   * @param {number} width - Width in device pixels.
   * @param {number} height - Height in device pixels.
   * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D}|null} The canvas and its
   *   context, or null where the browser would not give one.
   */
  function ensureLayer(width, height) {
    try {
      if (!state.layer) {
        const element = document.createElement("canvas");
        const context = element.getContext("2d");
        if (!context) return null;
        state.layer = { canvas: element, ctx: context };
      }
      if (state.layer.canvas.width !== width) state.layer.canvas.width = width;
      if (state.layer.canvas.height !== height) state.layer.canvas.height = height;
      return state.layer;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to make the text layer:`, error);
      return null;
    }
  }

  /**
   * Draw the text layer and lay it over the picture, the way the node composites it.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into, transformed by `ratio`.
   * @param {number} ratio - Device pixels per element pixel.
   * @param {object} model - Model from `readModel`.
   * @param {{x: number, y: number, w: number, h: number}} view - Where the picture is drawn.
   * @returns {object|null} What was laid out, from `layoutText`, with the face it was measured in.
   *   Null where the layer could not be built.
   */
  function drawLayer(ctx, ratio, model, view) {
    const frame = state.frame;
    const device = deviceRect(view, ratio);
    if (!(device.w > 0) || !(device.h > 0)) return null;

    const holder = ensureLayer(device.w, device.h);
    if (!holder) return null;

    const lctx = holder.ctx;
    // Image pixels to layer pixels. The layer is the whole picture, so it clips the text exactly
    // as the node's own layer clips it: a block anchored past the edge is cut off, not moved.
    const scale = device.w / frame.width;
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.clearRect(0, 0, device.w, device.h);
    lctx.setTransform(scale, 0, 0, scale, 0, 0);

    // Set before anything is measured, since every width follows from it, and before the face is
    // read, since the metrics the browser answers with are the metrics of whatever is set.
    const family = fontFamily(model.fontName);
    lctx.font = `${model.fontSize}px ${family ?? SUBSTITUTE_FAMILY}`;
    lctx.textAlign = "left";
    lctx.textBaseline = "alphabetic";

    const face = readFace(lctx, model, family);
    if (!face) return null;

    const laid = layoutText(lctx, {
      text: model.text,
      face,
      canvas: [frame.width, frame.height],
      position: model.position,
      align: model.align,
      offset: model.offset,
      margin: model.margin,
      lineSpacing: model.lineSpacing,
      wrapWidth: model.wrapWidth,
      strokeWidth: model.strokeWidth,
      backgroundPadding: model.backgroundPadding,
    });

    if (model.layer.panel && model.layer.panel[3] > 0) {
      lctx.fillStyle = cssColour(model.layer.panel);
      lctx.fillRect(laid.panel.x, laid.panel.y, laid.panel.w, laid.panel.h);
    }

    // The node draws one line's outline and then that line's glyphs, in line order, so a line
    // whose spacing brings it over the one above covers it in the same order here.
    const outlined = model.strokeWidth > 0 && model.layer.stroke && model.layer.stroke[3] > 0;
    lctx.lineJoin = "round";
    lctx.lineCap = "round";
    lctx.lineWidth = strokeLineWidth(model.strokeWidth);
    for (const row of laid.rows) {
      if (outlined) {
        lctx.strokeStyle = cssColour(model.layer.stroke);
        lctx.strokeText(row.text, row.x, row.baseline);
      }
      if (model.layer.fill && model.layer.fill[3] > 0) {
        lctx.fillStyle = cssColour(model.layer.fill);
        lctx.fillText(row.text, row.x, row.baseline);
      }
    }

    ctx.save();
    // `opacity` fades the finished layer rather than each mark in it, so the layer is built
    // whole and laid over the picture in one go.
    ctx.globalAlpha = clamp(model.opacity, 0, 1);
    try {
      // The destination is the device rectangle back in layout units, so the layer's pixels land
      // one for one on the device pixels they were drawn at and nothing resamples them.
      ctx.drawImage(
        holder.canvas,
        device.x / ratio,
        device.y / ratio,
        device.w / ratio,
        device.h / ratio,
      );
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to lay the text over the picture:`, error);
    }
    ctx.restore();

    return { ...laid, face };
  }

  /**
   * Draw the picture and the text over it.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {number} ratio - Device pixels per element pixel.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readModel`.
   * @returns {{laid: object|null, failed: boolean}} What was laid out, and whether the layer could
   *   not be built.
   */
  function drawPicture(ctx, ratio, theme, model) {
    const area = state.layout.preview;
    const frame = state.frame;
    const view = fitFrame(area, frame);
    const rect = view ?? area;

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(area.x, area.y, area.w, area.h);

    let stood = false;
    if (view && frame.image) {
      try {
        ctx.drawImage(frame.image, rect.x, rect.y, rect.w, rect.h);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the picture:`, error);
      }
    } else if (standingIn()) {
      // Fitted into the frame rather than sizing it. The frame is what the layout is measured in,
      // and a stand-in that carried its own size into it would put the text somewhere else.
      stood = drawStandIn(ctx, rect);
    }

    // A state's words are what is left where there is no picture standing in its place, so a
    // stand-in that has not arrived or could not be decoded leaves the words that were there.
    const label = frame.image || stood ? "" : frame.label;
    if (label) {
      ctx.font = LABEL_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = frame.state === PREVIEW_STATE.FAILED ? theme.warning : theme.fgMuted;
      ctx.fillText(label, rect.x + rect.w / 2, rect.y + rect.h / 2, Math.max(1, rect.w - 8));
    }

    let laid = null;
    let failed = false;
    if (view && model.text !== "") {
      laid = drawLayer(ctx, ratio, model, view);
      failed = laid === null;
    }

    outlineCell(ctx, { x: area.x, y: area.y, width: area.w, height: area.h }, theme.border);
    if (failed) {
      ctx.font = LABEL_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = theme.warning;
      ctx.fillText(NO_LAYER, area.x + area.w / 2, area.y + area.h / 2, Math.max(1, area.w - 8));
    }
    return { laid, failed };
  }

  /**
   * Draw the three colour rows.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {number} ratio - Device pixels the canvas holds per layout pixel.
   * @param {object} theme - Theme tokens.
   * @param {object[]} rows - Rows from `readRows`.
   * @returns {void}
   */
  function drawRows(ctx, ratio, theme, rows) {
    const layout = state.layout;
    const slot = valueSlot(layout);

    for (let index = 0; index < rows.length; index++) {
      const row = rows[index];
      const rect = rowRect(layout, index);
      const cell = cellRect(layout, index);
      const clear = clearRect(layout, index);
      const selected = state.selected === row.spec.widget;
      const hovered = state.hover === row.spec.widget;
      const middle = rect.y + rect.height / 2;

      if (selected) {
        ctx.globalAlpha = SELECTED_ALPHA;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        ctx.globalAlpha = 1;
      }

      ctx.font = SMALL_FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = row.dim ? theme.fgDisabled : theme.fgMuted;
      ctx.fillText(row.spec.label, rect.x, middle, LABEL_WIDTH - 2);

      // The swatch is written as pixels, so it replaces whatever is under it and the selected
      // row's wash stops at its edge.
      drawCell(ctx, ratio, cell, row.parsed, { markColour: theme.warning });
      outlineCell(
        ctx,
        cell,
        selected
          ? theme.accent
          : hovered
            ? theme.fg
            : row.dim
              ? theme.fgDisabled
              : theme.border,
      );

      if (slot.width > 8) {
        ctx.font = BODY_FONT;
        ctx.textAlign = "left";
        ctx.fillStyle =
          row.parsed.status === STATUS.INVALID
            ? theme.warning
            : row.parsed.status === STATUS.EMPTY
              ? theme.fgDisabled
              : row.dim
                ? theme.fgDisabled
                : row.parsed.status === STATUS.DECLINED
                  ? theme.fgMuted
                  : theme.fg;
        ctx.fillText(row.value, slot.x, middle, slot.width);
      }

      const clearHovered = state.hoverClear === index;
      ctx.font = SMALL_FONT;
      ctx.textAlign = "center";
      ctx.fillStyle =
        row.parsed.status === STATUS.EMPTY
          ? theme.fgDisabled
          : clearHovered
            ? theme.fg
            : theme.fgMuted;
      ctx.fillText(CLEAR_GLYPH, clear.x + clear.width / 2, clear.y + clear.height / 2);
      if (clearHovered) outlineCell(ctx, clear, theme.border);
    }
  }

  /**
   * What the preview is worth, and the measurement behind it.
   *
   * @param {object} model - Model from `readModel`.
   * @param {object|null} laid - What was laid out, from `drawLayer`.
   * @returns {{icon: string, detail: string}} The glyph and the sentence it carries on hover.
   */
  function fidelityClaim(model, laid) {
    const reasons = [];

    // A linked input is the one of these somebody can act on, so it leads the hover as well as
    // taking the words on screen, and the measurement follows it rather than being replaced by it.
    if (model.linked.length) {
      reasons.push(
        "an input is filled in by a link, so the run reads the link and not the widget this drew",
      );
    }

    if (model.tokens.builtIn.length) {
      reasons.push(
        `the text carries ${model.tokens.builtIn.join(" and ")}, which the run resolves before`
        + " drawing",
      );
    }
    if (model.tokens.custom.length) {
      reasons.push(
        `the text carries ${model.tokens.custom.join(" and ")}, which the run replaces where a`
        + " custom token of that name is defined and draws as written where none is, and this"
        + " cannot read the tokens Text Add Tokens stored, so the block width, the wrapping and"
        + " the placement here are the written form's; a token is matched under the name it was"
        + " stored under, so one stored without brackets is not looked for here",
      );
    }
    if (model.fontPath) {
      reasons.push(
        "font_path names a file this cannot ask for by name, since only the picked fonts are"
        + " served: where that file opens the run draws in a face nothing here was measured in,"
        + " and where it does not the run falls back to the picked font",
      );
    }
    // Read whatever font_path holds. The face this drew is the picked one either way, so a picked
    // face that is missing is a second thing the widths on screen do not stand for, and the one
    // the footer's second line marks.
    if (laid && !laid.face.own) {
      reasons.push(
        state.fontMissing === model.fontName
          ? `${model.fontName} could not be loaded, so this draws a substitute face and the`
            + " widths and the wrapping on screen are that face's"
          : `${model.fontName} has not arrived yet, so this draws a substitute face until it does`,
      );
    } else if (laid && !laid.face.exact) {
      reasons.push("the face reports no metrics, so the line height here is the browser's");
    }
    if (standingIn()) reasons.push(standInDetail(state.frame.state));
    if (frameAssumed()) {
      reasons.push(
        `the layout is against an assumed ${ASSUMED_FRAME} by ${ASSUMED_FRAME} until the node`
        + " runs, so every position, margin and wrap width is measured in the wrong image",
      );
    }

    // What is left once none of the above applies. The layout is the node's arithmetic and the
    // line height is FreeType's own, so what remains is how a line is measured and how the same
    // outline is filled in.
    const measured = [];
    // Measured against Pillow over 676 lines a face at each of six sizes from 12 to 96 px, and the
    // wrap rate over every one of those lines taken as a wrap width.
    const gap =
      "a line is measured through the canvas, which sums unhinted fractional advances and its own"
      + " kerning where Pillow rounds each advance to a whole pixel and reads the older kern"
      + " table: a mean of 1.6 to 8.4 px a line on DejaVu Sans and Liberation Sans, and up to 15"
      + " px on the monospaced and the version 2 faces";
    if (model.wrapWidth > 0) {
      measured.push(`${gap}, which moved a break in 11.6 per cent of 9360 wrappings`);
    } else {
      measured.push(
        `${gap}; wrap_width is 0, so no break can move, and what is left of it shifts a centred`
        + " block by half the gap and widens the panel by all of it",
      );
    }
    if (model.strokeWidth > 0) {
      measured.push(
        "the outline is drawn at twice stroke_width and centred on the glyph, which puts"
        + " stroke_width of it outside as Pillow's stroker does, measured at exactly that on"
        + " every side at 1 to 16 px",
      );
    }
    const cut = cutNote(model);
    if (cut) {
      measured.push(
        "a mark drawn over another replaces it in the render, alpha included, and composites over"
        + " it here: a #FFFFFF80 fill over an opaque outline is (255,255,255,128) where the glyph"
        + " covers, so the render shows the picture through the letters and this does not",
      );
    }
    if (model.opacity < 1) {
      measured.push("opacity truncates the layer's alpha byte in the node and not here, 1 of 255");
    }
    measured.push(
      "the same outline is hinted and antialiased differently: over one line's ink, a mean"
      + " coverage difference of 8.5 to 60 of 255, with the ink box itself inside 3 px of the"
      + " render's in width and 1 px in where it starts",
    );

    return {
      icon: reasons.length ? ICON.WARNING : ICON.APPROXIMATE,
      detail: [...reasons, ...measured].join(". "),
    };
  }

  /**
   * The note the first footer line carries on the right.
   *
   * @param {object} model - Model from `readModel`.
   * @param {object|null} laid - What was laid out, from `drawLayer`.
   * @returns {{text: string, warns: boolean}} The note and whether it is a warning.
   */
  function footerNote(model, laid) {
    if (state.message) return { text: state.message, warns: true };

    const linked = linkedNote(model.linked);
    if (linked) return { text: linked, warns: true };

    const residual = residualNote(
      tallyColours(model.rows.filter((row) => row.live).map((row) => row.parsed)),
      "colour",
    );
    if (residual) return { text: residual, warns: true };
    if (model.text === "") return { text: "no text, nothing is drawn", warns: false };
    if (laid && past(model, laid)) return { text: "text past the edge", warns: true };
    return { text: cutNote(model), warns: false };
  }

  /**
   * Whether anything drawn reaches outside the image.
   *
   * @param {object} model - Model from `readModel`.
   * @param {object} laid - What was laid out, from `drawLayer`.
   * @returns {boolean} True when the block or its panel is not wholly inside the frame.
   */
  function past(model, laid) {
    const frame = state.frame;
    const box = model.layer.panel?.[3] > 0
      ? laid.panel
      : { x: laid.x, y: laid.y, w: laid.block.width, h: laid.block.height };
    return box.x < 0 || box.y < 0 || box.x + box.w > frame.width || box.y + box.h > frame.height;
  }

  /**
   * Draw the two footer lines.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `readModel`.
   * @param {object} drawn - What `drawPicture` answered.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model, drawn) {
    const layout = state.layout;
    const first = layout.footerY + FOOTER_HEIGHT / 2;
    const second = layout.footerY + FOOTER_HEIGHT + FOOTER_HEIGHT / 2;
    const laid = drawn.laid;
    const claim = fidelityClaim(model, laid);
    const note = footerNote(model, laid);
    const regions = [];

    ctx.font = SMALL_FONT;
    ctx.textBaseline = "middle";

    // A claim about how truly a preview stands for the render is worth nothing on a pass that drew
    // no preview, which is a layer the browser refused rather than anything about the node.
    let glyphWidth = 0;
    if (!drawn.failed) {
      const colour = claim.icon === ICON.WARNING ? theme.warning : theme.fgMuted;
      const top = first - ICON_SIZE / 2;
      const box = drawIcon(ctx, claim.icon, layout.contentX0, top, ICON_SIZE, colour);
      regions.push({ ...box, title: iconTitle(claim.icon, claim.detail) });
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    }

    let noteWidth = 0;
    if (note.text) {
      noteWidth = ctx.measureText(note.text).width + GAP;
      ctx.textAlign = "right";
      ctx.fillStyle = note.warns ? theme.warning : theme.fgMuted;
      ctx.fillText(note.text, layout.contentX1, first);
    }

    // The block the widgets produce, which is the value somebody is working with, and never the
    // note's room: a line of numbers nobody can read is a line nobody was given.
    const readout = laid
      ? `${laid.lines.length} ${laid.lines.length === 1 ? "line" : "lines"},`
        + ` ${laid.block.width}x${laid.block.height} at ${laid.x},${laid.y}`
      : "";
    const room = layout.contentWidth - glyphWidth - noteWidth;
    if (readout && room > 24) {
      ctx.textAlign = "left";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(readout, layout.contentX0 + glyphWidth, first, room);
    }

    // The second line carries two standing facts: the face the text is drawn in, and the size the
    // layout is measured against. Neither is given up for a note.
    const frameText = frameAssumed()
      ? `assumed ${Math.round(state.frame.width)}x${Math.round(state.frame.height)}`
      : `${Math.round(state.frame.width)}x${Math.round(state.frame.height)}`;
    const frameWidth = ctx.measureText(frameText).width;
    ctx.textAlign = "right";
    ctx.fillStyle = frameAssumed() ? theme.warning : theme.fgMuted;
    ctx.fillText(frameText, layout.contentX1, second);

    // The picked face is not always the one on screen: it is asked for once per name and a face
    // still crossing, or one this browser will not read, is drawn in a substitute. The name here
    // is the one that was picked either way, so the substitution is marked rather than left to the
    // colour, and the glyph's hover says which of the two it is.
    const substituted = Boolean(laid) && !laid.face.own;
    const suffix = substituted ? " (substituted)" : "";
    const faceText = model.fontPath
      ? `font_path, drawn in ${model.fontName}${suffix}`
      : `${model.fontName} ${model.fontSize}${suffix}`;
    const faceRoom = layout.contentWidth - frameWidth - GAP;
    if (faceRoom > 24) {
      ctx.textAlign = "left";
      ctx.fillStyle = substituted ? theme.warning : theme.fgMuted;
      ctx.fillText(faceText, layout.contentX0, second, faceRoom);
    }

    // The footer band itself carries what its numbers are measured in, since a unit does not change
    // as somebody works and there is nothing on the line to point a glyph at.
    regions.push({
      x: layout.contentX0,
      y: layout.footerY,
      width: layout.contentWidth,
      height: FOOTER_HEIGHT * FOOTER_LINES,
      title:
        "Sizes and positions are pixels of the image the node draws on, worked out at that size"
        + " and drawn reduced to fit the panel, so the block, the wrap points and the placement"
        + " are the render's own rather than this panel's. A size shown as assumed is the one the"
        + " layout is measured against until the node runs and publishes the image it was given.",
    });
    titles.set(regions);
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
    const theme = readTheme();
    const model = readModel();
    ensureFont(model);

    const drawn = drawPicture(ctx, ratio, theme, model);
    drawRows(ctx, ratio, theme, model.rows);
    drawFooter(ctx, theme, model, drawn);

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
        console.error(`[${EXT_NAME}] Failed to draw the interface:`, error);
      }
    });
  }

  /**
   * The alpha byte a drag has reached.
   *
   * @param {object} press - The press being tracked.
   * @param {number} x - Pointer position in element pixels.
   * @param {boolean} coarse - Snap to a coarser step.
   * @returns {number} An alpha byte, 0 to 255.
   */
  function alphaFromDrag(press, x, coarse) {
    const moved = (x - press.startX) * ALPHA_PER_PIXEL;
    const raw = press.startAlpha + moved;
    return clamp(Math.round(coarse ? snap(raw, ALPHA_COARSE_STEP) : raw), 0, 255);
  }

  /**
   * End the press being tracked, releasing the pointer capture it holds.
   *
   * @param {boolean} commit - Write the alpha the drag reached. False discards it and leaves the
   *   widget as it was.
   * @returns {void}
   */
  function endPress(commit) {
    const press = state.press;
    if (!press) return;
    state.press = null;
    root.style.cursor = "default";
    if (root.hasPointerCapture?.(press.pointerId)) root.releasePointerCapture?.(press.pointerId);
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
        console.error(`[${EXT_NAME}] Interface input failed:`, error);
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
    if (!state.press) commitPending();

    // The pointer default action is left alone throughout. Cancelling it would suppress the
    // mouse events that follow, which carry the graph snapshot that gives the gesture its undo
    // entry.
    const point = localPoint(event);
    const index = rowIndexAt(point);
    if (index < 0) {
      state.selected = null;
      schedulePaint();
      return;
    }

    state.selected = ROWS[index].widget;

    if (inside(point, clearRect(state.layout, index))) {
      state.press = { pointerId: event.pointerId, index, kind: "clear" };
      root.setPointerCapture?.(event.pointerId);
      schedulePaint();
      return;
    }

    const row = readRows()[index];
    state.press = {
      pointerId: event.pointerId,
      index,
      kind: "row",
      startX: point.x,
      startAlpha: row.parsed.status === STATUS.COLOUR ? row.parsed.rgba[3] : null,
      dragging: false,
    };
    root.setPointerCapture?.(event.pointerId);
    schedulePaint();
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    const point = localPoint(event);
    const press = state.press;

    if (press) {
      // A button released over another window, or a capture the browser took away, ends the
      // gesture without a pointerup. Without this the alpha would keep following an unpressed
      // pointer and commit a value nobody chose.
      if (!(event.buttons & 1)) {
        endPress(false);
        return;
      }
      if (press.kind !== "row") return;
      if (!press.dragging && Math.abs(point.x - press.startX) < DRAG_THRESHOLD) return;

      if (!press.dragging) {
        if (press.startAlpha === null) {
          // Nothing was read, so there is no alpha byte to move and no hex to pack it into.
          // The press is spent rather than left to open the picker on release.
          press.kind = "refused";
          const row = readRows()[press.index];
          setMessage(
            row.parsed.status === STATUS.DECLINED ? row.parsed.note : "alpha needs a colour",
          );
          return;
        }
        press.dragging = true;
        root.style.cursor = "ew-resize";
      }

      holdPending(ROWS[press.index].widget, alphaFromDrag(press, point.x, event.shiftKey));
      return;
    }

    const index = rowIndexAt(point);
    const overClear = index >= 0 && inside(point, clearRect(state.layout, index));
    const hover = index >= 0 ? ROWS[index].widget : null;
    const hoverClear = overClear ? index : null;
    root.style.cursor = index >= 0 ? "pointer" : "default";

    if (hover !== state.hover || hoverClear !== state.hoverClear) {
      state.hover = hover;
      state.hoverClear = hoverClear;
      schedulePaint();
    }
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }

    const press = state.press;
    if (!press) return;

    const point = localPoint(event);
    const dragging = Boolean(press.dragging);
    const kind = press.kind;
    const index = press.index;

    endPress(dragging);
    if (dragging) return;

    if (kind === "clear") {
      if (inside(point, clearRect(state.layout, index))) clearRow(index);
      return;
    }
    if (kind === "row") openPicker(index, event.clientX, event.clientY);
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  };

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;

    const index = selectedIndex();
    let handled = true;

    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown": {
        const step = event.key === "ArrowUp" ? -1 : 1;
        const next =
          index < 0
            ? step < 0
              ? ROWS.length - 1
              : 0
            : clamp(index + step, 0, ROWS.length - 1);
        commitPending();
        state.selected = ROWS[next].widget;
        schedulePaint();
        break;
      }
      case "ArrowLeft":
      case "ArrowRight": {
        if (index < 0) {
          setMessage("select a colour first");
          break;
        }
        const row = readRows()[index];
        if (row.parsed.status !== STATUS.COLOUR) {
          setMessage(
            row.parsed.status === STATUS.DECLINED ? row.parsed.note : "alpha needs a colour",
          );
          break;
        }
        const step =
          (event.shiftKey ? ALPHA_COARSE_STEP : 1) * (event.key === "ArrowLeft" ? -1 : 1);
        holdPending(row.spec.widget, clamp(row.parsed.rgba[3] + step, 0, 255));
        break;
      }
      case "Enter":
      case " ": {
        if (index < 0) {
          setMessage("select a colour first");
          break;
        }
        const cell = cellRect(state.layout, index);
        const point = screenPoint(cell.x, cell.y + cell.height);
        openPicker(index, point.clientX, point.clientY);
        break;
      }
      case "Delete":
      case "Backspace": {
        // Consumed whether or not there is a colour to empty. Left unhandled these reach
        // ComfyUI's own binding, which deletes the node the interface is drawn on.
        if (index < 0) {
          setMessage("select a colour first");
          break;
        }
        clearRow(index);
        break;
      }
      case "Escape": {
        // An unfinished key gesture is dropped rather than written, which leaves the widget
        // holding what it held before the first key press.
        if (state.press) endPress(false);
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
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (state.press) return;
    commitPending();
  };

  const onBlur = () => {
    // Focus can only leave mid-press when the gesture has been interrupted, by another window
    // taking the pointer for example, so the press is discarded rather than kept.
    if (state.press) endPress(false);
    else commitPending();
    state.hover = null;
    state.hoverClear = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(() => endPress(false)));
  root.addEventListener("lostpointercapture", guard(() => endPress(false)));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hover === null && state.hoverClear === null) return;
    state.hover = null;
    state.hoverClear = null;
    schedulePaint();
  }));
  root.addEventListener("contextmenu", guard(onContextMenu));
  // The interface scrolls nothing of its own, so it takes every wheel gesture over it and the
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
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.paintHandle = 0;
    state.messageTimer = 0;
    state.retryTimer = 0;
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
    // The layer is the megabytes here, and whoever still holds the surface after a teardown holds
    // them with it, so it is let go rather than left to the element's own lifetime.
    state.layer = null;
    titles.dispose();
  }

  // The stand-in is one picture for the page, asked for by the first interface that needs it. The
  // ask is made here rather than at import, since every module under `web/` is imported on every
  // page whether a node wants it or not.
  loadPlaceholder().then(() => {
    if (!state.disposed) schedulePaint();
  });
  loadFrame();

  return {
    element: root,
    height: UI_HEIGHT,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    handleWidgetChanged,
    refresh,
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
 * Ask for the picture again as soon as the node finishes, or when a run ends without it.
 *
 * @param {object} node - The node the picture is published for.
 * @param {{refresh: () => void}} surface - Interface from `createTextInterface`.
 * @returns {() => void} Unhooks the listener.
 */
function watchRuns(node, surface) {
  return onNodeDone(node, () => {
    try {
      surface.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to ask for the image again:`, error);
    }
  });
}

/**
 * Append the interface to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachInterface(node) {
  for (const name of [TEXT_WIDGET, ...COLOUR_WIDGETS]) {
    if (!findWidget(node, name)) return;
  }

  const surface = createTextInterface(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, surface, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Every multiline box on the node bounded the same way, so the panel above takes
  // the room past their ceiling instead of losing all of it to them.
  boundTextBoxes(node);

  for (const name of READ_WIDGETS) {
    chainWidgetCallback(node, name, () => surface.handleWidgetChanged(name));
  }

  const unwatchRuns = watchRuns(node, surface);

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link changes
  // no widget value, so the callbacks above never hear about it. Without this the glyph, its
  // hover and the footer would go on standing for the widgets until something else asked for a
  // repaint.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      surface.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a connection change:`, error);
    }
    return result;
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      surface.schedulePaint();
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
    try {
      unwatchRuns();
      surface.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the interface:`, error);
    }
    return result;
  };

  surface.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Draw Text", "Text preview"],
      name: "Show the text preview",
      tooltip:
        "Draw the text over the image under the widgets of Image Draw Text, in the typeface the "
        + "node will use, with rows for its three colours. Click a row for the colour picker, "
        + "drag a row sideways for its transparency, and use the x at the end of a row to empty "
        + "it. The widgets themselves are always available. This applies to nodes added after "
        + "the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second interface.
    if (proto.__was_draw_text_wrapped) return;
    proto.__was_draw_text_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachInterface(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the interface:`, error);
      }
      return result;
    };
  },
});
