/**
 * The Bounding Box Editor's interface: boxes on a frame of the run's shape, layers, an inspector
 * and colour coding, written into the layout widget.
 *
 * The frame comes from the image input, or from width and height through wired nodes.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  KINDS,
  MAX_BOXES,
  MAX_PALETTE,
  PALETTE,
  RATIOS,
  RESIZE_MODES,
  inkOn,
  mapping,
  nextColour,
  normaliseColour,
  parseLayout,
  placed,
  ratioValue,
  unplaced,
  writeLayout,
} from "./interface/box_layout.js";
import { drawLabel } from "./interface/canvas_label.js";
import { pickColour } from "./interface/colour_cell.js";
import { createFloatingWindow } from "./interface/floating_window.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.js";
import { captureWheel, elementPoint, wheelPixels } from "./interface/pointer.js";
import { openPopupMenu } from "./interface/popup_menu.js";
import { PREVIEW_STATE, executionId, fetchInputPreview, nodeLocator, ownOutput } from "./interface/preview.js";
import { withGraphChange } from "./interface/region.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onRunEnded } from "./interface/run_events.js";
import { joinTicking } from "./interface/switch_board.js";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.js";
import { inputValue, isWired, sourceOf } from "./interface/upstream.js";
import { appendInterfaceWidget, boundTextBoxes, graphLoading, keepOnNode } from "./interface/widget.js";
import { LOOK, SANS, SCROLLBARS, SELECTED_FILL, button, createSplitter, el } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.BoundingBoxEditorUI";
const NODE_ID = "WASBoundingBoxEditor";
const UI_NAME = "was_box_editor_ui";
const UI_TYPE = "was_box_editor";

// The key the node files its run report under, and the node properties the view is kept in.
const REPORT_KEY = "was_box_layout";
const COLOUR_BY_KEY = "was_box_colour_by";
const SIZES_KEY = "was_box_panes";
const SIZES_WINDOW_KEY = "was_box_panes_window";

// Height of every control on the panel, and the looks shared by them, in CSS pixels.
const CONTROL = 26;
const CONTROL_BOX = `box-sizing:border-box;height:${CONTROL}px;border-radius:6px;font:12px ${SANS};outline:none;`
  + `border:1px solid ${LOOK.border};background:${LOOK.surface};color:${LOOK.text}`;
const SECTION_TITLE = `font:600 11px ${SANS};letter-spacing:0.06em;text-transform:uppercase;color:${LOOK.muted}`;
const FIELD_LABEL = `color:${LOOK.muted};white-space:nowrap`;

// Smallest the side panel, the panel under the canvas and the layer list are dragged to.
const SIDE_MIN = 190;
const UNDER_MIN = 110;
const LAYERS_MIN = 52;

// Where the expanded window keeps its place on the page.
const WINDOW_KEY = "was.boundingBoxEditor.window";

// Height the panel opens at on the node, and the narrowest it reads at.
const PANEL_HEIGHT = 480;
const PANEL_MIN_WIDTH = 300;

// Panel width from which the layer list stands beside the canvas.
const SIDE_BY_SIDE = 560;

// Pixels kept clear around the frame inside the canvas.
const FRAME_PAD = 10;

// Half the side of a drawn handle, and how near the pointer comes to take a handle or an edge.
const HANDLE = 3.5;
const GRAB = 7;

// Screen pixels within which a moving edge is drawn to another box or to a frame line.
const SNAP_PX = 6;

// Smallest drag that draws a box, in screen pixels; anything less is a click.
const MIN_DRAG_PX = 4;

// Screen pixels the pointer travels before a press becomes a drag.
const DRAG_START_PX = 3;

// Idle milliseconds after a keystroke in a caption before it is written.
const TYPING_IDLE_MS = 700;

const GRID_CHOICES = [[1, "Off"], [8, "8 px"], [16, "16 px"], [32, "32 px"], [64, "64 px"]];
const COLOUR_MODES = [["box", "Box"], ["kind", "Kind"], ["depth", "Depth"]];
const KIND_LABELS = { object: "Object", character: "Character", text: "Text" };
const KIND_SHORT = { object: "Obj", character: "Char", text: "Text" };
const KIND_COLOURS = { object: "#56B4E9", character: "#E69F00", text: "#CC79A7" };
const DEPTH_STOPS = ["#5E81AC", "#A3BE8C", "#EBCB8B"];
const CURSORS = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
};

// Glyph outlines in a 24 unit box.
const SVG_NS = "http://www.w3.org/2000/svg";
const EYE = ["M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z", "M12 9a3 3 0 1 0 0 6 3 3 0 1 0 0-6z"];
const EYE_OFF = ["M3 3l18 18", "M10.6 5.1A10 10 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.2",
  "M6.6 6.6A17 17 0 0 0 2 12s3.6 7 10 7a10 10 0 0 0 5.4-1.6"];
const LOCK = ["M6 11h12v10H6z", "M8 11V8a4 4 0 0 1 8 0v3"];
const UNLOCK = ["M6 11h12v10H6z", "M8 11V8a4 4 0 0 1 7.6-1.8"];
const EXPAND = ["M15 3h6v6", "M9 21H3v-6", "M21 3l-7 7", "M3 21l7-7"];
const POINTER = ["M5 3l14 7.5-6.2 1.6L10 19z", "M12.8 12.1l5 5"];
const BOX_PLUS = ["M3 7h12v12H3z", "M19 3v6", "M16 6h6"];
const PICTURE = ["M3 5h18v14H3z", "M3 16l5-5 4 4 3-3 6 6"];
const SLIDERS = ["M4 7h9", "M17 7h3", "M15 5v4", "M4 17h3", "M11 17h9", "M9 15v4"];

/**
 * A node's widget by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|null} The widget.
 */
function widgetOf(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

/**
 * Write one widget as a single undo step.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @param {*} value - The new value.
 * @returns {boolean} True when the widget took a new value.
 */
function writeWidget(node, name, value) {
  const widget = widgetOf(node, name);
  if (!widget || widget.value === value) return false;
  withGraphChange(() => {
    widget.value = value;
    widget.callback?.(value);
  });
  node.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * A colour's channels.
 *
 * @param {string} colour - `#rrggbb`.
 * @returns {number[]} `[r, g, b]`.
 */
function channels(colour) {
  const hex = normaliseColour(colour) ?? "#888888";
  return [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16));
}

/**
 * A colour with an alpha, for a canvas.
 *
 * @param {string} colour - `#rrggbb`.
 * @param {number} alpha - 0 to 1.
 * @returns {string} `rgba(...)`.
 */
function withAlpha(colour, alpha) {
  const [r, g, b] = channels(colour);
  return `rgba(${r},${g},${b},${alpha})`;
}

/**
 * A colour along the back to front ramp.
 *
 * @param {number} at - 0 at the back to 1 at the front.
 * @returns {string} `#rrggbb`.
 */
function depthColour(at) {
  const span = Math.max(0, Math.min(1, at)) * (DEPTH_STOPS.length - 1);
  const index = Math.min(DEPTH_STOPS.length - 2, Math.floor(span));
  const share = span - index;
  const a = channels(DEPTH_STOPS[index]);
  const b = channels(DEPTH_STOPS[index + 1]);
  const mixed = a.map((value, channel) => Math.round(value + (b[channel] - value) * share));
  return `#${mixed.map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * A line glyph as an SVG element.
 *
 * @param {string[]} paths - Path data in a 24 unit box.
 * @param {number} [size] - Side in CSS pixels.
 * @returns {SVGElement} The glyph.
 */
function glyph(paths, size = 13) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  for (const data of paths) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", data);
    svg.appendChild(path);
  }
  return svg;
}

/**
 * Snap a value to a grid.
 *
 * @param {number} value - The value.
 * @param {number} step - The grid; 1 or less leaves the value alone.
 * @returns {number} The snapped value.
 */
function snapTo(value, step) {
  return step > 1 ? Math.round(value / step) * step : value;
}

/**
 * The nearest of some targets to any of some points, within a tolerance.
 *
 * @param {number[]} points - Positions on the moving box.
 * @param {number[]} targets - Positions it may land on.
 * @param {number} tolerance - Largest distance taken.
 * @returns {{delta: number, target: number}|null} How far to move, and the line landed on.
 */
function nearest(points, targets, tolerance) {
  let best = null;
  for (const point of points) {
    for (const target of targets) {
      const delta = target - point;
      if (Math.abs(delta) <= tolerance && (!best || Math.abs(delta) < Math.abs(best.delta))) {
        best = { delta, target };
      }
    }
  }
  return best;
}

/**
 * A ratio written for a reader, as `16:9` when it is one of the listed ratios.
 *
 * @param {number} width - Width.
 * @param {number} height - Height.
 * @returns {string} The ratio.
 */
function ratioText(width, height) {
  if (!(width > 0 && height > 0)) return "";
  const value = width / height;
  for (const [entry] of RATIOS) {
    const listed = ratioValue(entry, [1, 1]);
    if (entry !== "frame" && listed && Math.abs(listed - value) / listed < 0.005) return entry;
  }
  return value >= 1 ? `${value.toFixed(2)}:1` : `1:${(1 / value).toFixed(2)}`;
}

/**
 * A select at the panel's control height.
 *
 * @param {Array<[*, string]>} entries - `[value, label]` pairs.
 * @param {*} value - The chosen value.
 * @param {string} hint - What the hover says.
 * @returns {HTMLSelectElement} The select.
 */
function controlSelect(entries, value, hint) {
  const element = el("select", `${CONTROL_BOX};padding:0 6px;cursor:pointer;min-width:0;max-width:100%`);
  for (const [option, label] of entries) {
    const entry = document.createElement("option");
    entry.value = String(option);
    entry.textContent = label;
    element.appendChild(entry);
  }
  element.value = String(value);
  element.title = hint;
  return element;
}

/**
 * A text or number field at the panel's control height.
 *
 * @param {string} type - `text` or `number`.
 * @param {string} hint - What the hover says.
 * @returns {HTMLInputElement} The field.
 */
function controlInput(type, hint) {
  const input = el("input", `${CONTROL_BOX};padding:0 7px;width:100%;min-width:0;font-variant-numeric:tabular-nums`);
  input.type = type;
  if (type === "number") input.step = "1";
  input.title = hint;
  input.addEventListener("focus", () => { input.style.borderColor = LOOK.accent; });
  input.addEventListener("blur", () => { input.style.borderColor = LOOK.border; });
  return input;
}

/**
 * A button at the panel's control height, holding glyphs and words.
 *
 * @param {Array<string|SVGElement>} parts - What it shows, in order.
 * @param {string} hint - What the hover says.
 * @param {(event: MouseEvent) => void} onPress - What it does.
 * @returns {HTMLButtonElement} The button.
 */
function controlButton(parts, hint, onPress) {
  const element = button("", onPress, { hint, kind: "tool", logName: EXT_NAME });
  element.style.cssText += `;height:${CONTROL}px;padding:0 ${parts.length === 1 && typeof parts[0] !== "string" ? 6 : 9}px;`
    + `border-radius:6px;font:12px ${SANS};gap:6px`;
  for (const part of parts) element.appendChild(typeof part === "string" ? document.createTextNode(part) : part);
  return element;
}

/**
 * One side of the tool switch.
 *
 * @param {SVGElement} icon - Its glyph.
 * @param {string} hint - What the hover says.
 * @param {() => void} onPress - What it does.
 * @returns {HTMLButtonElement} The button.
 */
function segmentButton(icon, hint, onPress) {
  const element = el("button", "display:inline-flex;align-items:center;justify-content:center;width:30px;height:100%;"
    + `padding:0;border:0;cursor:pointer;background:transparent;color:${LOOK.muted}`);
  element.type = "button";
  element.title = hint;
  element.appendChild(icon);
  element.addEventListener("click", (event) => {
    event.preventDefault();
    try {
      onPress();
    } catch (error) {
      console.error(`[${EXT_NAME}] ${hint} failed:`, error);
    }
  });
  return element;
}

/**
 * A control with a short label before it.
 *
 * @param {string} label - The label.
 * @param {HTMLElement} control - The control.
 * @returns {HTMLElement} The pair.
 */
function labelled(label, control) {
  const box = el("label", "display:flex;align-items:center;gap:6px;min-width:0");
  box.appendChild(el("span", `color:${LOOK.muted};flex:0 0 auto`, label));
  box.appendChild(control);
  return box;
}

/**
 * The pane sizes kept on a node, for the node's panel or its window.
 *
 * @param {object} node - The node.
 * @param {boolean} inWindow - True for the window's panes.
 * @returns {{side: number, under: number, layers: number}} Pixels: the side panel's width, its
 *   height when it sits under the canvas, and the layer list's height.
 */
function readSizes(node, inWindow) {
  const kept = node.properties?.[inWindow ? SIZES_WINDOW_KEY : SIZES_KEY];
  const number = (value, fallback) => (Number.isFinite(Number(value)) && Number(value) > 0 ? Number(value) : fallback);
  return {
    side: number(kept?.side, inWindow ? 320 : 250),
    under: number(kept?.under, 170),
    layers: number(kept?.layers, inWindow ? 240 : 120),
  };
}

/**
 * Keep the pane sizes on the node.
 *
 * @param {object} node - The node.
 * @param {boolean} inWindow - True for the window's panes.
 * @param {object} sizes - What `readSizes` answered, after a drag.
 * @returns {void}
 */
function keepSizes(node, inWindow, sizes) {
  keepOnNode(node, inWindow ? SIZES_WINDOW_KEY : SIZES_KEY, {
    side: Math.round(sizes.side), under: Math.round(sizes.under), layers: Math.round(sizes.layers),
  });
}

/**
 * Show a value in a select, adding it as an entry when the menu does not list it.
 *
 * @param {HTMLSelectElement} element - The select.
 * @param {string} value - The value.
 * @param {string} [label] - What a missing entry is called.
 * @returns {void}
 */
function showValue(element, value, label) {
  const text = String(value ?? "");
  if (![...element.options].some((option) => option.value === text)) {
    const entry = document.createElement("option");
    entry.value = text;
    entry.textContent = label || text;
    element.appendChild(entry);
  }
  element.value = text;
}

/**
 * A colour swatch button.
 *
 * @param {string} hint - What the hover says.
 * @returns {HTMLButtonElement} The swatch.
 */
function swatchButton(hint) {
  const element = el("button", `width:22px;height:22px;flex:0 0 auto;padding:0;border-radius:5px;cursor:pointer;`
    + `border:1px solid ${LOOK.border}`);
  element.type = "button";
  element.title = hint;
  return element;
}

/**
 * Build the editor for one node, at either size.
 *
 * @param {object} node - The Bounding Box Editor node.
 * @param {{inWindow?: boolean, onExpand?: () => void}} [options] - Whether it fills a window, and
 *   what the expand button opens.
 * @returns {object} The element, its height, its minimum width, `refresh` and `dispose`.
 */
function createBoxEditor(node, options = {}) {
  const inWindow = options.inWindow === true;
  let theme = readTheme();
  let tool = "select";
  let selected = -1;
  let hover = -1;
  let gesture = null;
  let current = null;
  let seenText = null;
  let lastWritten = null;
  let report = null;
  let backdrop = { state: PREVIEW_STATE.WAITING, image: null };
  let drawRatio = "";
  let frameRequest = 0;
  let typingTimer = 0;
  let lastKey = "";
  let view = null;
  let disposed = false;

  // ---- elements --------------------------------------------------------------------------------

  const root = el("div", "position:relative;width:100%;height:100%;display:flex;flex-direction:column;gap:8px;"
    + "padding:6px;box-sizing:border-box;border-radius:6px;overflow:hidden;outline:none;"
    + `background:${themeVar("panelBg")};color:${themeVar("fg")};font:12px ${SANS}`);
  const toolbar = el("div", "display:flex;flex-wrap:wrap;align-items:center;gap:8px;flex:0 0 auto");
  const body = el("div", "flex:1 1 auto;min-height:0;display:flex;flex-direction:column");
  const stage = el("div", "position:relative;flex:1 1 auto;min-width:0;min-height:120px;border-radius:6px;"
    + `overflow:hidden;background:${themeVar("bgDark")};border:1px solid ${LOOK.border}`);
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block;outline:none;"
    + "touch-action:none");
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "Bounding boxes");
  stage.appendChild(canvas);
  const side = el("div", "display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden");
  const layersHead = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;padding:0 2px 5px");
  const layersTitle = el("span", SECTION_TITLE, "Layers");
  const layersCount = el("span", `margin-left:auto;color:${LOOK.muted};font-variant-numeric:tabular-nums`);
  layersHead.append(layersTitle, layersCount);
  const layers = el("div", "position:relative;display:flex;flex-direction:column;gap:2px;overflow:auto;"
    + `flex:0 0 auto;min-height:44px;${SCROLLBARS}`);
  const dropLine = el("div", `position:absolute;left:0;right:0;height:2px;background:${LOOK.accent};display:none;`
    + "pointer-events:none");
  const inspector = el("div", `display:flex;flex-direction:column;gap:6px;flex:1 1 auto;min-height:0;overflow:auto;${SCROLLBARS}`);
  const footer = el("div", `display:flex;align-items:center;gap:8px;flex:0 0 auto;min-height:24px;color:${LOOK.muted}`);
  const glyphCanvas = el("canvas", `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`);
  const frameText = el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
    + "font-variant-numeric:tabular-nums");
  const notice = el("span", "display:flex;align-items:center;gap:6px;flex:0 1 auto;min-width:0;white-space:nowrap");
  footer.append(glyphCanvas, frameText, notice);

  // The two dividers: canvas against the side panel, and the layer list against the inspector.
  const sizes = readSizes(node, inWindow);
  const sideSplit = createSplitter(() => (body.style.flexDirection === "row" ? "x" : "y"), (delta) => {
    if (body.style.flexDirection === "row") {
      sizes.side = Math.max(SIDE_MIN, Math.min(root.clientWidth * 0.7, sizes.side - delta));
    } else {
      sizes.under = Math.max(UNDER_MIN, Math.min(root.clientHeight * 0.6, sizes.under - delta));
    }
    layoutFor();
  }, () => keepSizes(node, inWindow, sizes));
  const listSplit = createSplitter(() => "y", (delta) => {
    sizes.layers = Math.max(LAYERS_MIN, Math.min(side.clientHeight - 60, sizes.layers + delta));
    layoutFor();
  }, () => keepSizes(node, inWindow, sizes));
  side.append(layersHead, layers, listSplit.element, inspector);
  body.append(stage, sideSplit.element, side);
  root.append(toolbar, body, footer);

  // ---- toolbar ---------------------------------------------------------------------------------

  const selectTool = segmentButton(glyph(POINTER, 15), "Select, move and resize (V)", () => setTool("select"));
  const drawTool = segmentButton(glyph(BOX_PLUS, 15), "Draw a box, even over others (B)", () => setTool("box"));
  const tools = el("span", `display:inline-flex;height:${CONTROL}px;border:1px solid ${LOOK.border};border-radius:6px;`
    + `overflow:hidden;background:${LOOK.surface};flex:0 0 auto`);
  tools.append(selectTool, el("span", `width:1px;background:${LOOK.border}`), drawTool);
  const ratioPick = controlSelect(RATIOS, "", "Ratio new boxes are drawn at. Shift keeps a ratio too");
  const backgroundButton = controlButton([glyph(PICTURE, 14), "Background"], "Add a box filling the frame, at the back",
    () => addBackground());
  const viewButton = controlButton([glyph(SLIDERS, 14), "View"], "Grid and colours", (event) => openViewMenu(event));
  toolbar.append(tools, labelled("Draw at", ratioPick), backgroundButton, el("span", "flex:1 1 auto"), viewButton);
  if (!inWindow && typeof options.onExpand === "function") {
    toolbar.appendChild(controlButton([glyph(EXPAND, 14)], "Open in a window", options.onExpand));
  }
  ratioPick.addEventListener("change", () => { drawRatio = ratioPick.value; });

  /**
   * The menu holding the grid and the colour mode.
   *
   * @param {MouseEvent} event - The press that opened it.
   * @returns {void}
   */
  function openViewMenu(event) {
    const box = event.currentTarget?.getBoundingClientRect?.() ?? { left: event.clientX, bottom: event.clientY };
    const grid = gridValue();
    const mode = colourBy();
    openPopupMenu({
      x: box.left, y: box.bottom + 4, logName: EXT_NAME,
      items: [
        { header: "Snap to grid" },
        ...GRID_CHOICES.map(([value, label]) => ({
          label, checked: grid === value, detail: value > 1 ? "Ctrl places freely" : "",
          onSelect: () => {
            writeWidget(node, "grid", value);
            notifyEditors(node);
          },
        })),
        { separator: true },
        { header: "Colour boxes by" },
        ...COLOUR_MODES.map(([value, label]) => ({
          label, checked: mode === value,
          onSelect: () => {
            keepOnNode(node, COLOUR_BY_KEY, value);
            notifyEditors(node);
          },
        })),
      ],
    });
  }

  // ---- inspector -------------------------------------------------------------------------------

  const emptyNote = el("div", `color:${LOOK.muted};padding:10px 2px;text-align:center`, "Select a box to edit");
  const inspectorTitle = el("span", SECTION_TITLE, "Box");
  const kindPick = controlSelect(KINDS.map((kind) => [kind, KIND_LABELS[kind]]), "object",
    "Object, character, or words the model renders");
  const caption = el("textarea", `font:12px/1.45 ${SANS};padding:5px 7px;border-radius:6px;resize:none;`
    + `width:100%;box-sizing:border-box;min-height:48px;border:1px solid ${LOOK.border};`
    + `background:${LOOK.surface};color:${LOOK.text};outline:none;${SCROLLBARS}`);
  caption.rows = 2;
  caption.placeholder = "What is in this box";
  caption.spellcheck = false;
  caption.setAttribute("data-capture-wheel", "true");
  const words = controlInput("text", "Words the box renders");
  words.placeholder = "Words the box renders";
  const xInput = controlInput("number", "Left edge, in pixels");
  const yInput = controlInput("number", "Top edge, in pixels");
  const wInput = controlInput("number", "Width, in pixels");
  const hInput = controlInput("number", "Height, in pixels");
  const boxRatio = controlSelect(RATIOS, "", "Ratio the box keeps while resized");
  const colourSwatch = swatchButton("The box's colour here and on the preview");
  const paletteRow = el("span", "display:inline-flex;align-items:center;gap:4px;flex-wrap:wrap");
  const centreButton = controlButton(["Centre"], "Centre the box in the frame", () => editGeometry((rect, frame) => ({
    ...rect, x: Math.round((frame[0] - rect.width) / 2), y: Math.round((frame[1] - rect.height) / 2),
  })));
  const fillButton = controlButton(["Fill frame"], "Make the box fill the frame", () => editGeometry((rect, frame) => ({
    x: 0, y: 0, width: frame[0], height: frame[1],
  })));
  const pair = (a, b) => {
    const row = el("div", "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;min-width:0");
    row.append(a, b);
    return row;
  };
  const grid = el("div", "display:grid;grid-template-columns:fit-content(40%) minmax(0,1fr);gap:6px 8px;"
    + "align-items:center");
  const wordsLabel = el("span", FIELD_LABEL, "Text");
  const rows = [
    [el("span", FIELD_LABEL, "Kind"), kindPick],
    [el("span", FIELD_LABEL, "Caption"), caption],
    [wordsLabel, words],
    [el("span", FIELD_LABEL, "Position"), pair(labelled("X", xInput), labelled("Y", yInput))],
    [el("span", FIELD_LABEL, "Size"), pair(labelled("W", wInput), labelled("H", hInput))],
    [el("span", FIELD_LABEL, "Ratio"), boxRatio],
    [el("span", FIELD_LABEL, "Colour"), colourSwatch],
    [el("span", FIELD_LABEL, "Palette"), paletteRow],
  ];
  for (const [label, control] of rows) grid.append(label, control);
  const actions = el("div", "display:flex;gap:6px;flex-wrap:wrap");
  actions.append(centreButton, fillButton);
  const fields = el("div", "display:flex;flex-direction:column;gap:8px");
  fields.append(inspectorTitle, grid, actions);
  inspector.append(emptyNote, fields);

  boxRatio.addEventListener("change", () => setRatio(boxRatio.value));
  kindPick.addEventListener("change", () => editSelected((box) => { box.kind = kindPick.value; }));
  const commitCaption = () => {
    clearTimeout(typingTimer);
    typingTimer = 0;
    editSelected((box) => {
      box.caption = caption.value;
      box.text = words.value;
    });
  };
  for (const field of [caption, words]) {
    field.addEventListener("input", () => {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(commitCaption, TYPING_IDLE_MS);
    });
    field.addEventListener("change", commitCaption);
    field.addEventListener("keydown", (event) => event.stopPropagation());
  }
  const geometryInputs = { x: xInput, y: yInput, width: wInput, height: hInput };
  for (const [key, input] of Object.entries(geometryInputs)) {
    input.addEventListener("change", () => {
      const value = Math.round(Number(input.value));
      if (!Number.isFinite(value)) return;
      editGeometry((rect) => ({ ...rect, [key]: value }));
    });
    input.addEventListener("keydown", (event) => event.stopPropagation());
  }
  colourSwatch.addEventListener("click", (event) => {
    const box = current?.parsed.boxes[selected];
    if (!box) return;
    pickColour(event.clientX, event.clientY, channels(box.color), (rgb) => {
      const hex = `#${rgb.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
      editSelected((target) => { target.color = hex; });
    });
  });

  // ---- reading the node ------------------------------------------------------------------------

  const gridValue = () => {
    const value = Number(widgetOf(node, "grid")?.value);
    return Number.isFinite(value) && value >= 1 ? Math.trunc(value) : 16;
  };
  const resizeMode = () => {
    const value = String(widgetOf(node, "on_resize")?.value ?? "fit");
    return RESIZE_MODES.includes(value) ? value : "fit";
  };
  const colourBy = () => {
    const value = node.properties?.[COLOUR_BY_KEY];
    return COLOUR_MODES.some(([mode]) => mode === value) ? value : "box";
  };
  const frameSide = (value) => {
    const number = Math.trunc(Number(value));
    return Number.isFinite(number) && number >= 1 ? Math.min(number, 16384) : 0;
  };

  /**
   * The frame the node will run at, and where it was read.
   *
   * @returns {{frame: number[], claim: string, from: string}} `claim` is `exact`, `last run` or `guess`.
   */
  function readFrame() {
    const fromWidgets = [frameSide(widgetOf(node, "width")?.value) || 1024,
      frameSide(widgetOf(node, "height")?.value) || 1024];
    const reported = Array.isArray(report?.frame) && report.frame.length === 2 ? report.frame.map(frameSide) : null;
    if (isWired(node, "image")) {
      const source = sourceOf(node, "image");
      const picture = source?.node?.imgs?.[0];
      if (picture?.naturalWidth > 0 && picture?.naturalHeight > 0) {
        return { frame: [picture.naturalWidth, picture.naturalHeight], claim: "exact",
          from: String(source.node.title || "the image") };
      }
      if (reported?.every(Boolean) && report?.source === "image") {
        return { frame: reported, claim: "last run", from: "the image, last run" };
      }
      return { frame: fromWidgets, claim: "guess", from: "the image, size unknown" };
    }
    const width = inputValue(node, "width");
    const height = inputValue(node, "height");
    const w = frameSide(width.value);
    const h = frameSide(height.value);
    if (width.known && height.known && w && h) {
      const named = [...new Set([width.from, height.from].filter(Boolean))].join(" and ");
      return { frame: [w, h], claim: "exact", from: named };
    }
    if (reported?.every(Boolean) && report?.source === "size") {
      return { frame: reported, claim: "last run", from: "last run" };
    }
    return { frame: [w || fromWidgets[0], h || fromWidgets[1]], claim: "guess",
      from: [width, height].find((answer) => !answer.known) ? "a wired value, read on the next run" : "" };
  }

  /**
   * Read the node again into one model.
   *
   * @returns {object} The frame, the parsed layout, the mapping onto the frame and each box as placed.
   */
  function rebuild() {
    const info = readFrame();
    const text = String(widgetOf(node, "layout")?.value ?? "");
    if (text !== seenText) {
      if (seenText !== null && text !== lastWritten) selected = -1;
      seenText = text;
    }
    const parsed = parseLayout(text, info.frame);
    const grid = gridValue();
    const map = mapping(parsed.frame, info.frame, resizeMode());
    const shown = parsed.boxes.map((box) => placed(box, map, info.frame, grid));
    if (selected >= parsed.boxes.length) selected = parsed.boxes.length - 1;
    let number = 0;
    const numbers = parsed.boxes.map((box, index) => (box.hidden || !shown[index] ? 0 : ++number));
    current = { info, text, parsed, grid, map, shown, numbers, incoming: incomingBoxes(info.frame, number) };
    return current;
  }

  /**
   * The boxes the last run read from the bboxes input, placed on the frame.
   *
   * @param {number[]} frame - The frame.
   * @param {number} after - How many layout boxes come before them.
   * @returns {object[]} Boxes with their output number.
   */
  function incomingBoxes(frame, after) {
    if (!isWired(node, "bboxes") || String(widgetOf(node, "bboxes_mode")?.value ?? "add") === "ignore") return [];
    const list = Array.isArray(report?.incoming) ? report.incoming : [];
    const from = Array.isArray(report?.frame) ? report.frame : frame;
    const sx = frame[0] / (from[0] || frame[0]);
    const sy = frame[1] / (from[1] || frame[1]);
    const replacing = String(widgetOf(node, "bboxes_mode")?.value) === "replace";
    return list.slice(0, MAX_BOXES).map((box, index) => ({
      ...box,
      x: box.x * sx,
      y: box.y * sy,
      width: box.width * sx,
      height: box.height * sy,
      number: (replacing ? 0 : after) + index + 1,
    }));
  }

  // ---- geometry --------------------------------------------------------------------------------

  /**
   * Fit the frame into the canvas.
   *
   * @param {number[]} frame - `[width, height]`.
   * @param {number} width - Canvas width in layout pixels.
   * @param {number} height - Canvas height in layout pixels.
   * @returns {object} The scale and where the frame's corner lands.
   */
  function fitView(frame, width, height) {
    const scale = Math.max(1e-6, Math.min((width - 2 * FRAME_PAD) / frame[0], (height - 2 * FRAME_PAD) / frame[1]));
    const w = frame[0] * scale;
    const h = frame[1] * scale;
    return { scale, x0: (width - w) / 2, y0: (height - h) / 2, w, h, frame };
  }

  const toScreen = (rect) => ({
    x: view.x0 + rect.x * view.scale,
    y: view.y0 + rect.y * view.scale,
    w: rect.width * view.scale,
    h: rect.height * view.scale,
  });

  const pointAt = (event) => {
    const point = elementPoint(canvas, event);
    return { px: point.x, py: point.y, x: (point.x - view.x0) / view.scale, y: (point.y - view.y0) / view.scale };
  };

  /**
   * The visible boxes under a point, front first.
   *
   * @param {object} at - A point from `pointAt`.
   * @returns {number[]} Indexes into the layout.
   */
  function boxesAt(at) {
    const hits = [];
    for (let index = current.parsed.boxes.length - 1; index >= 0; index -= 1) {
      const rect = current.shown[index];
      if (!rect || current.parsed.boxes[index].hidden) continue;
      if (at.x >= rect.x && at.x <= rect.x + rect.width && at.y >= rect.y && at.y <= rect.y + rect.height) {
        hits.push(index);
      }
    }
    return hits;
  }

  /**
   * The handle or edge of a box under a point.
   *
   * @param {object} at - A point from `pointAt`.
   * @param {object} rect - The box as placed.
   * @returns {string|null} A compass name, or null.
   */
  function handleAt(at, rect) {
    const box = toScreen(rect);
    const xs = { w: box.x, c: box.x + box.w / 2, e: box.x + box.w };
    const ys = { n: box.y, c: box.y + box.h / 2, s: box.y + box.h };
    const near = (a, b) => Math.abs(a - b) <= GRAB;
    for (const [name, x, y] of [["nw", xs.w, ys.n], ["ne", xs.e, ys.n], ["se", xs.e, ys.s], ["sw", xs.w, ys.s]]) {
      if (near(at.px, x) && near(at.py, y)) return name;
    }
    const insideX = at.px >= box.x - GRAB && at.px <= box.x + box.w + GRAB;
    const insideY = at.py >= box.y - GRAB && at.py <= box.y + box.h + GRAB;
    if (insideX && near(at.py, ys.n)) return "n";
    if (insideX && near(at.py, ys.s)) return "s";
    if (insideY && near(at.px, xs.w)) return "w";
    if (insideY && near(at.px, xs.e)) return "e";
    return null;
  }

  /**
   * Lines a moving edge may land on along one axis.
   *
   * @param {string} axis - `x` or `y`.
   * @param {number} except - The box being moved, left out.
   * @returns {number[]} Positions in frame pixels.
   */
  function targets(axis, except) {
    const side = axis === "x" ? current.info.frame[0] : current.info.frame[1];
    const lines = [0, side / 3, side / 2, (2 * side) / 3, side];
    current.shown.forEach((rect, index) => {
      if (!rect || index === except || current.parsed.boxes[index].hidden) return;
      const start = axis === "x" ? rect.x : rect.y;
      const span = axis === "x" ? rect.width : rect.height;
      lines.push(start, start + span / 2, start + span);
    });
    return lines;
  }

  /**
   * Scale a rectangle about an anchor until it fits the frame.
   *
   * @param {object} rect - `{x, y, width, height}`.
   * @param {{x: number, y: number}} anchor - The point it scales about.
   * @returns {object} The rectangle inside the frame.
   */
  function fitInside(rect, anchor) {
    const [W, H] = current.info.frame;
    let scale = 1;
    if (rect.x < 0) scale = Math.min(scale, (anchor.x - 0) / Math.max(1e-6, anchor.x - rect.x));
    if (rect.y < 0) scale = Math.min(scale, (anchor.y - 0) / Math.max(1e-6, anchor.y - rect.y));
    if (rect.x + rect.width > W) scale = Math.min(scale, (W - anchor.x) / Math.max(1e-6, rect.x + rect.width - anchor.x));
    if (rect.y + rect.height > H) scale = Math.min(scale, (H - anchor.y) / Math.max(1e-6, rect.y + rect.height - anchor.y));
    scale = Math.max(0, scale);
    return {
      x: anchor.x + (rect.x - anchor.x) * scale,
      y: anchor.y + (rect.y - anchor.y) * scale,
      width: rect.width * scale,
      height: rect.height * scale,
    };
  }

  /**
   * A rectangle from an anchor towards a pointer, held to a ratio and drawn from the centre if asked.
   *
   * @param {{x: number, y: number}} anchor - Where the drag started, or the fixed corner.
   * @param {{x: number, y: number}} pointer - Where the pointer is.
   * @param {number|null} ratio - Width over height to keep, or null.
   * @param {boolean} fromCentre - True to grow both ways from the anchor.
   * @returns {object} The rectangle in frame pixels.
   */
  function spanRect(anchor, pointer, ratio, fromCentre) {
    let dx = pointer.x - anchor.x;
    let dy = pointer.y - anchor.y;
    if (ratio) {
      if (Math.abs(dx) / ratio > Math.abs(dy)) dy = (Math.sign(dy) || 1) * (Math.abs(dx) / ratio);
      else dx = (Math.sign(dx) || 1) * Math.abs(dy) * ratio;
    }
    const rect = fromCentre
      ? { x: anchor.x - Math.abs(dx), y: anchor.y - Math.abs(dy), width: 2 * Math.abs(dx), height: 2 * Math.abs(dy) }
      : { x: Math.min(anchor.x, anchor.x + dx), y: Math.min(anchor.y, anchor.y + dy), width: Math.abs(dx), height: Math.abs(dy) };
    return ratio ? fitInside(rect, anchor) : clampRect(rect);
  }

  /**
   * Hold a rectangle inside the frame, trimming what falls outside.
   *
   * @param {object} rect - `{x, y, width, height}`.
   * @returns {object} The rectangle.
   */
  function clampRect(rect) {
    const [W, H] = current.info.frame;
    const x0 = Math.max(0, Math.min(W, rect.x));
    const y0 = Math.max(0, Math.min(H, rect.y));
    const x1 = Math.max(0, Math.min(W, rect.x + rect.width));
    const y1 = Math.max(0, Math.min(H, rect.y + rect.height));
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
  }

  /**
   * A rectangle rounded to whole pixels.
   *
   * @param {object} rect - `{x, y, width, height}`.
   * @returns {object} The rectangle, never below one pixel either way.
   */
  function whole(rect) {
    const x0 = Math.round(rect.x);
    const y0 = Math.round(rect.y);
    const x1 = Math.round(rect.x + rect.width);
    const y1 = Math.round(rect.y + rect.height);
    return { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
  }

  // ---- gestures --------------------------------------------------------------------------------

  /**
   * Work out the rectangle a gesture holds now.
   *
   * @param {PointerEvent} event - The latest pointer event.
   * @returns {void}
   */
  function updateGesture(event) {
    const at = pointAt(event);
    const free = event.ctrlKey || event.metaKey;
    const grid = current.grid;
    const tolerance = SNAP_PX / view.scale;
    const guides = [];
    const frame = current.info.frame;
    const clampPoint = (point) => ({
      x: Math.max(0, Math.min(frame[0], point.x)),
      y: Math.max(0, Math.min(frame[1], point.y)),
    });
    gesture.pointer = at;
    if (!gesture.moved && Math.hypot(at.px - gesture.pressed.px, at.py - gesture.pressed.py) < DRAG_START_PX) return;
    if (gesture.type === "draw") {
      const ratio = ratioValue(drawRatio, frame) ?? (event.shiftKey ? 1 : null);
      const pointer = clampPoint(free ? at : { x: snapTo(at.x, grid), y: snapTo(at.y, grid) });
      gesture.rect = spanRect(gesture.origin, pointer, ratio, event.altKey);
    } else if (gesture.type === "move") {
      let dx = at.x - gesture.origin.x;
      let dy = at.y - gesture.origin.y;
      if (event.shiftKey) {
        if (Math.abs(dx) > Math.abs(dy)) dy = 0;
        else dx = 0;
      }
      const from = gesture.from;
      let x = from.x + dx;
      let y = from.y + dy;
      if (!free) {
        const sx = nearest([x, x + from.width / 2, x + from.width], targets("x", gesture.index), tolerance);
        if (sx) {
          x += sx.delta;
          guides.push({ axis: "x", value: sx.target });
        } else x = snapTo(x, grid);
        const sy = nearest([y, y + from.height / 2, y + from.height], targets("y", gesture.index), tolerance);
        if (sy) {
          y += sy.delta;
          guides.push({ axis: "y", value: sy.target });
        } else y = snapTo(y, grid);
      }
      x = Math.max(0, Math.min(frame[0] - from.width, x));
      y = Math.max(0, Math.min(frame[1] - from.height, y));
      gesture.rect = { x, y, width: from.width, height: from.height };
    } else if (gesture.type === "resize") {
      gesture.rect = resized(at, event, free, tolerance, guides);
    }
    gesture.guides = guides;
    gesture.moved = true;
  }

  /**
   * The rectangle a resize gesture holds now.
   *
   * @param {object} at - The pointer, from `pointAt`.
   * @param {PointerEvent} event - The pointer event, for its modifiers.
   * @param {boolean} free - True to skip snapping.
   * @param {number} tolerance - Snapping distance in frame pixels.
   * @param {object[]} guides - Filled with the lines an edge landed on.
   * @returns {object} The rectangle in frame pixels.
   */
  function resized(at, event, free, tolerance, guides) {
    const from = gesture.from;
    const handle = gesture.handle;
    const box = current.parsed.boxes[gesture.index];
    const frame = current.info.frame;
    const snapEdge = (value, axis) => {
      if (free) return value;
      const hit = nearest([value], targets(axis, gesture.index), tolerance);
      if (hit) {
        guides.push({ axis, value: hit.target });
        return hit.target;
      }
      return snapTo(value, current.grid);
    };
    const point = {
      x: Math.max(0, Math.min(frame[0], snapEdge(at.x, "x"))),
      y: Math.max(0, Math.min(frame[1], snapEdge(at.y, "y"))),
    };
    const ratio = ratioValue(box.ratio, frame) ?? (event.shiftKey ? from.width / from.height : null);
    const centre = { x: from.x + from.width / 2, y: from.y + from.height / 2 };
    if (handle.length === 2) {
      const anchor = event.altKey ? centre : {
        x: handle.includes("w") ? from.x + from.width : from.x,
        y: handle.includes("n") ? from.y + from.height : from.y,
      };
      return spanRect(anchor, point, ratio, event.altKey);
    }
    let left = from.x;
    let top = from.y;
    let right = from.x + from.width;
    let bottom = from.y + from.height;
    if (handle === "w") left = point.x;
    if (handle === "e") right = point.x;
    if (handle === "n") top = point.y;
    if (handle === "s") bottom = point.y;
    if (event.altKey) {
      if (handle === "w") right = 2 * centre.x - left;
      if (handle === "e") left = 2 * centre.x - right;
      if (handle === "n") bottom = 2 * centre.y - top;
      if (handle === "s") top = 2 * centre.y - bottom;
    }
    let rect = {
      x: Math.min(left, right), y: Math.min(top, bottom), width: Math.abs(right - left), height: Math.abs(bottom - top),
    };
    if (ratio) {
      if (handle === "w" || handle === "e") {
        const height = rect.width / ratio;
        rect = { ...rect, y: centre.y - height / 2, height };
      } else {
        const width = rect.height * ratio;
        rect = { ...rect, x: centre.x - width / 2, width };
      }
      return fitInside(rect, { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 });
    }
    return clampRect(rect);
  }

  /**
   * End the gesture, writing what it made.
   *
   * @param {boolean} [cancelled] - True to drop it.
   * @returns {void}
   */
  function finishGesture(cancelled = false) {
    const ended = gesture;
    gesture = null;
    if (!ended) return;
    if (canvas.hasPointerCapture?.(ended.pointerId)) canvas.releasePointerCapture(ended.pointerId);
    if (cancelled || !ended.moved || !ended.rect) {
      if (ended.type === "draw" && !cancelled) selected = -1;
      paintAll();
      return;
    }
    const rect = whole(ended.rect);
    if (ended.type === "draw") {
      if (ended.rect.width * view.scale < MIN_DRAG_PX || ended.rect.height * view.scale < MIN_DRAG_PX) {
        selected = -1;
        paintAll();
        return;
      }
      addBox(rect);
      return;
    }
    const before = current.shown[ended.index];
    if (before && before.x === rect.x && before.y === rect.y && before.width === rect.width
      && before.height === rect.height) {
      paintAll();
      return;
    }
    editBox(ended.index, (box) => Object.assign(box, unplaced(rect, current.map)));
  }

  // ---- writing ---------------------------------------------------------------------------------

  /**
   * Write a set of boxes into the layout widget.
   *
   * @param {object[]} boxes - Every box, back to front.
   * @returns {void}
   */
  function writeBoxes(boxes) {
    const m = current;
    const frame = m.parsed.boxes.length ? m.parsed.frame : m.info.frame;
    const text = writeLayout(boxes.length ? frame : m.info.frame, boxes);
    lastWritten = text;
    if (!writeWidget(node, "layout", text)) lastWritten = String(widgetOf(node, "layout")?.value ?? "");
    notifyEditors(node);
  }

  /**
   * Change one box and write the layout.
   *
   * @param {number} index - The box.
   * @param {(box: object) => void} change - Mutates a copy of it.
   * @returns {void}
   */
  function editBox(index, change) {
    const m = rebuild();
    if (!m.parsed.ok || !m.parsed.boxes[index]) return;
    const boxes = m.parsed.boxes.map((box) => ({ ...box, palette: [...box.palette] }));
    change(boxes[index]);
    writeBoxes(boxes);
  }

  const editSelected = (change) => {
    if (selected >= 0) editBox(selected, change);
  };

  /**
   * Change the selected box's rectangle as placed on the frame.
   *
   * @param {(rect: object, frame: number[]) => object} change - Answers the new rectangle.
   * @returns {void}
   */
  function editGeometry(change) {
    const m = rebuild();
    const rect = m.shown[selected];
    if (selected < 0 || !rect) return;
    const next = whole(clampRect(change({ ...rect }, m.info.frame)));
    editBox(selected, (box) => Object.assign(box, unplaced(next, m.map)));
  }

  /**
   * Add a box drawn on the frame.
   *
   * @param {object} rect - Its rectangle on the frame.
   * @param {object} [extra] - Keys to set on it.
   * @param {boolean} [atBack] - True to put it behind every other box.
   * @returns {void}
   */
  function addBox(rect, extra = {}, atBack = false) {
    const m = rebuild();
    if (!m.parsed.ok) return;
    if (m.parsed.boxes.length >= MAX_BOXES) {
      paintAll();
      return;
    }
    const boxes = m.parsed.boxes.map((box) => ({ ...box, palette: [...box.palette] }));
    const box = {
      kind: "object", ...unplaced(rect, m.map), caption: "", text: "", palette: [],
      color: normaliseColour(nextColour(boxes)), ratio: drawRatio, locked: false, hidden: false, ...extra,
    };
    if (atBack) boxes.unshift(box);
    else boxes.push(box);
    selected = atBack ? 0 : boxes.length - 1;
    writeBoxes(boxes);
  }

  const addBackground = () => {
    const frame = rebuild().info.frame;
    addBox({ x: 0, y: 0, width: frame[0], height: frame[1] }, { caption: "background", locked: true }, true);
  };

  /**
   * Remove one box.
   *
   * @param {number} index - The box.
   * @returns {void}
   */
  function removeBox(index) {
    const m = rebuild();
    if (!m.parsed.ok || !m.parsed.boxes[index]) return;
    const boxes = m.parsed.boxes.filter((_, at) => at !== index);
    selected = boxes.length ? Math.min(index, boxes.length - 1) : -1;
    writeBoxes(boxes);
  }

  /**
   * Copy one box, a grid step down and to the right.
   *
   * @param {number} index - The box.
   * @returns {void}
   */
  function duplicateBox(index) {
    const m = rebuild();
    const rect = m.shown[index];
    if (!m.parsed.ok || !rect || m.parsed.boxes.length >= MAX_BOXES) return;
    const step = Math.max(8, m.grid);
    const moved = whole(clampRect({ ...rect, x: rect.x + step, y: rect.y + step }));
    const boxes = m.parsed.boxes.map((box) => ({ ...box, palette: [...box.palette] }));
    const copy = { ...boxes[index], ...unplaced(moved, m.map), color: nextColour(boxes), locked: false, hidden: false };
    boxes.splice(index + 1, 0, copy);
    selected = index + 1;
    writeBoxes(boxes);
  }

  /**
   * Move one box to another place in the stack.
   *
   * @param {number} from - Where it is, 0 at the back.
   * @param {number} to - Where it goes.
   * @returns {void}
   */
  function moveInStack(from, to) {
    const m = rebuild();
    const count = m.parsed.boxes.length;
    if (!m.parsed.ok || from < 0 || from >= count) return;
    const target = Math.max(0, Math.min(count - 1, to));
    if (target === from) return;
    const boxes = m.parsed.boxes.map((box) => ({ ...box, palette: [...box.palette] }));
    const [box] = boxes.splice(from, 1);
    boxes.splice(target, 0, box);
    selected = target;
    writeBoxes(boxes);
  }

  /**
   * Lock the selected box to a ratio and bring it to that ratio about its centre.
   *
   * @param {string} value - A value from `RATIOS`.
   * @returns {void}
   */
  function setRatio(value) {
    const m = rebuild();
    const rect = m.shown[selected];
    if (selected < 0 || !rect) return;
    const ratio = ratioValue(value, m.info.frame);
    let next = rect;
    if (ratio) {
      const centre = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
      const height = rect.width / ratio;
      next = whole(fitInside({ x: rect.x, y: centre.y - height / 2, width: rect.width, height }, centre));
    }
    editBox(selected, (box) => {
      box.ratio = value;
      Object.assign(box, unplaced(next, m.map));
    });
  }

  /**
   * Take the boxes the last run read in into the layout, and stop reading the input.
   *
   * @returns {void}
   */
  function adoptIncoming() {
    const m = rebuild();
    if (!m.parsed.ok || !m.incoming.length) return;
    const boxes = m.parsed.boxes.map((box) => ({ ...box, palette: [...box.palette] }));
    const replacing = String(widgetOf(node, "bboxes_mode")?.value) === "replace";
    const kept = replacing ? [] : boxes;
    for (const box of m.incoming) {
      if (kept.length >= MAX_BOXES) break;
      kept.push({
        kind: KINDS.includes(box.kind) ? box.kind : "object",
        ...unplaced(whole(clampRect(box)), m.map),
        caption: typeof box.caption === "string" ? box.caption : "",
        text: typeof box.text === "string" ? box.text : "",
        palette: (Array.isArray(box.palette) ? box.palette : []).map(normaliseColour).filter(Boolean).slice(0, MAX_PALETTE),
        color: normaliseColour(box.color) ?? nextColour(kept),
        ratio: "", locked: false, hidden: false,
      });
    }
    withGraphChange(() => {
      writeBoxes(kept);
      writeWidget(node, "bboxes_mode", "ignore");
    });
  }

  /**
   * Rewrite the layout at the frame it is placed on now.
   *
   * @returns {void}
   */
  function adoptFrame() {
    const m = rebuild();
    if (!m.parsed.ok) return;
    const boxes = [];
    m.parsed.boxes.forEach((box, index) => {
      const rect = m.shown[index];
      if (rect) boxes.push({ ...box, x: rect.x, y: rect.y, width: rect.width, height: rect.height });
    });
    lastWritten = writeLayout(m.info.frame, boxes);
    writeWidget(node, "layout", lastWritten);
    notifyEditors(node);
  }

  // ---- pointer ---------------------------------------------------------------------------------

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    canvas.focus();
    const m = rebuild();
    if (!view || !m.parsed.ok) return;
    const at = pointAt(event);
    const boxes = m.parsed.boxes;
    let start = null;
    if (selected >= 0 && m.shown[selected] && !boxes[selected].locked) {
      const handle = handleAt(at, m.shown[selected]);
      if (handle) start = { type: "resize", index: selected, handle };
    }
    if (!start && tool === "select") {
      const hits = boxesAt(at);
      if (hits.length) {
        let index = hits[0];
        if (event.altKey && hits.length > 1) index = hits[(hits.indexOf(selected) + 1) % hits.length];
        selected = index;
        start = { type: boxes[index].locked ? "select" : "move", index };
      }
    }
    if (!start) {
      const inside = at.x >= 0 && at.y >= 0 && at.x <= m.info.frame[0] && at.y <= m.info.frame[1];
      if (!inside || boxes.length >= MAX_BOXES) {
        selected = -1;
        paintAll();
        return;
      }
      const grid = event.ctrlKey || event.metaKey ? 1 : m.grid;
      start = { type: "draw", index: -1, origin: { x: snapTo(at.x, grid), y: snapTo(at.y, grid) } };
    }
    gesture = {
      origin: { x: at.x, y: at.y },
      ...start,
      from: start.index >= 0 ? { ...m.shown[start.index] } : null,
      rect: null,
      guides: [],
      pointer: at,
      pressed: at,
      pointerId: event.pointerId,
      moved: false,
    };
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to capture the pointer:`, error);
    }
    paintAll();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!view || !current) return;
    if (!gesture) {
      hoverAt(event);
      return;
    }
    if ((event.buttons & 1) === 0) {
      finishGesture();
      return;
    }
    updateGesture(event);
    schedule();
  });
  canvas.addEventListener("pointerup", () => finishGesture());
  canvas.addEventListener("pointercancel", () => finishGesture(true));
  canvas.addEventListener("lostpointercapture", () => finishGesture());
  canvas.addEventListener("pointerleave", () => {
    if (!gesture && hover !== -1) {
      hover = -1;
      paintLayers();
      schedule();
    }
  });
  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (selected < 0) return;
    caption.focus();
    caption.select();
  });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!view || !current?.parsed.ok) return;
    const hits = boxesAt(pointAt(event));
    if (hits.length) selected = hits[0];
    paintAll();
    openBoxMenu(event.clientX, event.clientY);
  });

  /**
   * Light the box under the pointer and set the cursor.
   *
   * @param {PointerEvent} event - The pointer event.
   * @returns {void}
   */
  function hoverAt(event) {
    const at = pointAt(event);
    let cursor = tool === "box" ? "crosshair" : "default";
    let over = -1;
    if (selected >= 0 && current.shown[selected] && !current.parsed.boxes[selected].locked) {
      const handle = handleAt(at, current.shown[selected]);
      if (handle) cursor = CURSORS[handle];
    }
    if (tool === "select") {
      over = boxesAt(at)[0] ?? -1;
      if (over >= 0 && cursor === "default") cursor = current.parsed.boxes[over].locked ? "pointer" : "move";
      if (over < 0 && cursor === "default") cursor = "crosshair";
    }
    canvas.style.cursor = cursor;
    if (over !== hover) {
      hover = over;
      const box = current.parsed.boxes[over];
      canvas.title = box ? labelOf(box) || KIND_LABELS[box.kind] : "";
      paintLayers();
      schedule();
    }
  }

  // ---- keyboard --------------------------------------------------------------------------------

  canvas.addEventListener("keydown", (event) => {
    const key = event.key;
    const command = event.ctrlKey || event.metaKey;
    let used = true;
    if (key === "Delete" || key === "Backspace") {
      if (selected >= 0) removeBox(selected);
    } else if (command && (key === "d" || key === "D")) {
      if (selected >= 0) duplicateBox(selected);
    } else if (key.startsWith("Arrow")) {
      if (selected >= 0 && !current?.parsed.boxes[selected]?.locked) {
        const step = event.shiftKey ? Math.max(1, current.grid) : 1;
        const dx = key === "ArrowLeft" ? -step : key === "ArrowRight" ? step : 0;
        const dy = key === "ArrowUp" ? -step : key === "ArrowDown" ? step : 0;
        editGeometry((rect, frame) => ({
          ...rect,
          x: Math.max(0, Math.min(frame[0] - rect.width, rect.x + dx)),
          y: Math.max(0, Math.min(frame[1] - rect.height, rect.y + dy)),
        }));
      }
    } else if (!command && (key === "v" || key === "V")) {
      setTool("select");
    } else if (!command && (key === "b" || key === "B")) {
      setTool("box");
    } else if (!command && (key === "h" || key === "H")) {
      if (selected >= 0) editSelected((box) => { box.hidden = !box.hidden; });
    } else if (!command && (key === "l" || key === "L")) {
      if (selected >= 0) editSelected((box) => { box.locked = !box.locked; });
    } else if (key === "]") {
      if (selected >= 0) moveInStack(selected, selected + 1);
    } else if (key === "[") {
      if (selected >= 0) moveInStack(selected, selected - 1);
    } else if (key === "Escape") {
      if (gesture) finishGesture(true);
      else selected = -1;
      paintAll();
    } else {
      used = false;
    }
    if (used) {
      event.preventDefault();
      event.stopPropagation();
    }
  });

  /**
   * Switch between the select and draw tools.
   *
   * @param {string} next - `select` or `box`.
   * @returns {void}
   */
  function setTool(next) {
    tool = next === "box" ? "box" : "select";
    for (const [element, on] of [[selectTool, tool === "select"], [drawTool, tool === "box"]]) {
      element.style.background = on ? SELECTED_FILL : "transparent";
      element.style.color = on ? LOOK.text : LOOK.muted;
    }
    canvas.style.cursor = tool === "box" ? "crosshair" : "default";
  }

  /**
   * The menu for the selected box.
   *
   * @param {number} x - Left edge, in client pixels.
   * @param {number} y - Top edge, in client pixels.
   * @returns {void}
   */
  function openBoxMenu(x, y) {
    const box = current.parsed.boxes[selected];
    if (!box) {
      openPopupMenu({
        x, y, logName: EXT_NAME,
        items: [{ label: "Add background", onSelect: () => addBackground() }],
      });
      return;
    }
    const count = current.parsed.boxes.length;
    openPopupMenu({
      x, y, logName: EXT_NAME,
      items: [
        { header: KIND_LABELS[box.kind] },
        ...KINDS.map((kind) => ({
          label: KIND_LABELS[kind], checked: box.kind === kind,
          onSelect: () => editSelected((target) => { target.kind = kind; }),
        })),
        { separator: true },
        { label: "Duplicate", detail: "Ctrl+D", disabled: count >= MAX_BOXES, onSelect: () => duplicateBox(selected) },
        { label: "Bring forward", detail: "]", disabled: selected >= count - 1, onSelect: () => moveInStack(selected, selected + 1) },
        { label: "Send backward", detail: "[", disabled: selected <= 0, onSelect: () => moveInStack(selected, selected - 1) },
        { label: "Bring to front", disabled: selected >= count - 1, onSelect: () => moveInStack(selected, count - 1) },
        { label: "Send to back", disabled: selected <= 0, onSelect: () => moveInStack(selected, 0) },
        { separator: true },
        { label: box.hidden ? "Show" : "Hide", detail: "H", onSelect: () => editSelected((target) => { target.hidden = !target.hidden; }) },
        { label: box.locked ? "Unlock" : "Lock", detail: "L", onSelect: () => editSelected((target) => { target.locked = !target.locked; }) },
        { label: "Fill the frame", onSelect: () => editGeometry((rect, frame) => ({ x: 0, y: 0, width: frame[0], height: frame[1] })) },
        { swatches: PALETTE, current: box.color, onPick: (colour) => editSelected((target) => { target.color = normaliseColour(colour) ?? target.color; }) },
        { separator: true },
        { label: "Delete", detail: "Del", danger: true, onSelect: () => removeBox(selected) },
      ],
    });
  }

  // ---- painting --------------------------------------------------------------------------------

  const schedule = () => {
    if (frameRequest || disposed) return;
    frameRequest = requestAnimationFrame(paint);
  };

  /**
   * The colour a box is drawn in under the colour mode.
   *
   * @param {number} index - The box's place, 0 at the back.
   * @param {object} box - The box.
   * @returns {string} `#rrggbb`.
   */
  function colourOf(index, box) {
    const mode = colourBy();
    if (mode === "kind") return KIND_COLOURS[box.kind] ?? KIND_COLOURS.object;
    if (mode === "depth") {
      const count = current?.parsed.boxes.length ?? 1;
      return depthColour(count > 1 ? index / (count - 1) : 1);
    }
    return normaliseColour(box.color) ?? PALETTE[index % PALETTE.length];
  }

  /**
   * Draw one box.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas.
   * @param {object} rect - Its rectangle on the frame.
   * @param {object} box - The box.
   * @param {string} colour - Its colour.
   * @param {object} state - `{number, selected, hovered, incoming}`.
   * @param {object[]} tags - Collects the box's tag, drawn after every box.
   * @returns {void}
   */
  function drawBox(ctx, rect, box, colour, state, tags) {
    const r = toScreen(rect);
    if (r.w < 1 || r.h < 1) return;
    const faint = box.hidden;
    ctx.save();
    ctx.fillStyle = withAlpha(colour, state.selected ? 0.26 : state.hovered ? 0.2 : faint ? 0.05 : 0.14);
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = colour;
    ctx.lineWidth = state.selected ? 2 : 1.5;
    if (faint || state.incoming) ctx.setLineDash([5, 4]);
    ctx.globalAlpha = faint ? 0.6 : 1;
    ctx.strokeRect(r.x + ctx.lineWidth / 2, r.y + ctx.lineWidth / 2, r.w - ctx.lineWidth, r.h - ctx.lineWidth);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.rect(r.x, r.y, r.w, r.h);
    ctx.clip();
    const palette = Array.isArray(box.palette) ? box.palette.slice(0, MAX_PALETTE) : [];
    if (palette.length && r.w > 8) {
      const each = r.w / palette.length;
      palette.forEach((swatch, at) => {
        ctx.fillStyle = swatch;
        ctx.fillRect(r.x + Math.round(at * each), r.y, Math.ceil(each), 4);
      });
    }
    if (box.locked && r.w > 24 && r.h > 20) drawLock(ctx, r.x + r.w - 15, r.y + r.h - 17);
    ctx.restore();
    tags.push({ r, colour, box, state });
  }

  /**
   * The words a box's tag shows.
   *
   * @param {object} box - The box.
   * @returns {string} Its caption on one line, a text box's words first.
   */
  function labelOf(box) {
    let words = typeof box.caption === "string" ? box.caption.trim() : "";
    if (box.kind === "text" && box.text) words = `"${box.text}"${words ? ` ${words}` : ""}`;
    return words.replace(/\s+/g, " ");
  }

  /**
   * Shorten text to a width, ending it with an ellipsis.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas, with its font set.
   * @param {string} text - The text.
   * @param {number} width - The width in pixels.
   * @returns {string} The text, whole or shortened.
   */
  function fitText(ctx, text, width) {
    if (ctx.measureText(text).width <= width) return text;
    let low = 0;
    let high = text.length;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (ctx.measureText(`${text.slice(0, middle)}…`).width <= width) low = middle;
      else high = middle - 1;
    }
    return low ? `${text.slice(0, low).trimEnd()}…` : "";
  }

  /**
   * Draw every box's tag, a number chip and a one line caption, each placed clear of the others.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas.
   * @param {object[]} tags - What `drawBox` collected, back to front.
   * @returns {void}
   */
  function drawTags(ctx, tags) {
    const taken = [];
    const clashes = (a) => taken.some((b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y);
    const laid = [];
    const height = 17;
    ctx.save();
    ctx.textBaseline = "middle";
    // The box in front chooses its corner first.
    for (const { r, colour, box, state } of [...tags].reverse()) {
      const number = state.incoming ? `in ${state.number}` : state.number ? String(state.number) : "off";
      ctx.font = "600 10px system-ui,sans-serif";
      const chip = Math.ceil(ctx.measureText(number).width) + 10;
      ctx.font = "11px system-ui,sans-serif";
      const room = Math.min(260, r.w - chip - 6);
      const label = labelOf(box);
      const text = label && room > 30 ? fitText(ctx, label, room - 10) : "";
      const plate = text ? Math.ceil(ctx.measureText(text).width) + 10 : 0;
      const width = chip + plate;
      const spots = [
        [r.x + 3, r.y + 3], [r.x + r.w - width - 3, r.y + 3],
        [r.x + 3, r.y + r.h - height - 3], [r.x + r.w - width - 3, r.y + r.h - height - 3],
        [r.x, r.y - height - 3],
      ];
      let spot = spots[0];
      for (const [x, y] of spots) {
        if (!clashes({ x, y, w: width, h: height })) {
          spot = [x, y];
          break;
        }
      }
      taken.push({ x: spot[0], y: spot[1], w: width, h: height });
      laid.push({ x: spot[0], y: spot[1], chip, plate, number, text, colour });
    }
    for (const tag of laid.reverse()) {
      ctx.fillStyle = tag.colour;
      ctx.fillRect(tag.x, tag.y, tag.chip, height);
      ctx.font = "600 10px system-ui,sans-serif";
      ctx.fillStyle = inkOn(tag.colour);
      ctx.fillText(tag.number, tag.x + 5, tag.y + height / 2 + 0.5);
      if (tag.text) {
        ctx.fillStyle = theme.bgDark ?? theme.bg;
        ctx.globalAlpha = 0.9;
        ctx.fillRect(tag.x + tag.chip, tag.y, tag.plate, height);
        ctx.globalAlpha = 1;
        ctx.font = "11px system-ui,sans-serif";
        ctx.fillStyle = theme.fg;
        ctx.fillText(tag.text, tag.x + tag.chip + 5, tag.y + height / 2 + 0.5);
      }
    }
    ctx.restore();
  }

  /**
   * Draw a small padlock on a plate.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas.
   * @param {number} x - Left edge.
   * @param {number} y - Top edge.
   * @returns {void}
   */
  function drawLock(ctx, x, y) {
    ctx.save();
    ctx.fillStyle = theme.bgDark ?? theme.bg;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x, y, 12, 14);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = theme.fg;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(x + 2.5, y + 6.5, 7, 5);
    ctx.beginPath();
    ctx.arc(x + 6, y + 6.5, 2.4, Math.PI, 0);
    ctx.stroke();
    ctx.restore();
  }

  /**
   * Paint the canvas.
   *
   * @returns {void}
   */
  function paint() {
    frameRequest = 0;
    if (disposed) return;
    try {
      const m = current ?? rebuild();
      const ratio = surfaceRatio(canvas);
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      if (!(width > 0 && height > 0)) return;
      const deviceW = Math.max(1, Math.round(width * ratio));
      const deviceH = Math.max(1, Math.round(height * ratio));
      if (canvas.width !== deviceW || canvas.height !== deviceH) {
        canvas.width = deviceW;
        canvas.height = deviceH;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = theme.bgDark ?? theme.bg;
      ctx.fillRect(0, 0, width, height);
      view = fitView(m.info.frame, width, height);
      const frameBox = { x: view.x0, y: view.y0, w: view.w, h: view.h };
      ctx.fillStyle = theme.bg;
      ctx.fillRect(frameBox.x, frameBox.y, frameBox.w, frameBox.h);
      const picture = backdropPicture();
      if (picture) {
        ctx.globalAlpha = 0.6;
        ctx.drawImage(picture, frameBox.x, frameBox.y, frameBox.w, frameBox.h);
        ctx.globalAlpha = 1;
      }
      drawGrid(ctx, m);
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(frameBox.x) + 0.5, Math.round(frameBox.y) + 0.5, Math.round(frameBox.w) - 1, Math.round(frameBox.h) - 1);

      const frameArea = { width, height, top: 0, theme };
      ctx.save();
      ctx.beginPath();
      ctx.rect(frameBox.x, frameBox.y, frameBox.w, frameBox.h);
      ctx.clip();
      const tags = [];
      m.parsed.boxes.forEach((box, index) => {
        let rect = m.shown[index];
        if (gesture && gesture.index === index && gesture.rect) rect = gesture.rect;
        if (!rect || (box.hidden && index !== selected)) return;
        drawBox(ctx, rect, box, colourOf(index, box), {
          number: m.numbers[index], selected: index === selected, hovered: index === hover, incoming: false,
        }, tags);
      });
      for (const box of m.incoming) {
        drawBox(ctx, box, box, normaliseColour(box.color) ?? theme.fgMuted, {
          number: box.number, selected: false, hovered: false, incoming: true,
        }, tags);
      }
      if (gesture?.type === "draw" && gesture.rect) {
        drawBox(ctx, gesture.rect, { caption: "", palette: [] }, nextColour(m.parsed.boxes), {
          number: m.numbers.filter(Boolean).length + 1, selected: true, hovered: false, incoming: false,
        }, tags);
      }
      drawTags(ctx, tags);
      for (const guide of gesture?.guides ?? []) {
        ctx.strokeStyle = theme.accent;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        if (guide.axis === "x") {
          const x = Math.round(view.x0 + guide.value * view.scale) + 0.5;
          ctx.moveTo(x, frameBox.y);
          ctx.lineTo(x, frameBox.y + frameBox.h);
        } else {
          const y = Math.round(view.y0 + guide.value * view.scale) + 0.5;
          ctx.moveTo(frameBox.x, y);
          ctx.lineTo(frameBox.x + frameBox.w, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();

      const handleRect = gesture?.rect && gesture.index === selected ? gesture.rect : m.shown[selected];
      if (selected >= 0 && handleRect && !m.parsed.boxes[selected]?.locked && gesture?.type !== "draw") {
        drawHandles(ctx, handleRect, colourOf(selected, m.parsed.boxes[selected]));
      }
      if (gesture?.rect && gesture.moved) {
        const rect = whole(gesture.rect);
        const words = `${rect.width} × ${rect.height} px · ${ratioText(rect.width, rect.height)}`;
        drawLabel(ctx, frameArea, words, (gesture.pointer?.px ?? 0) + 14, (gesture.pointer?.py ?? 0) + 16);
      }
      if (!m.parsed.ok) {
        drawLabel(ctx, frameArea, "The layout text cannot be read", width / 2 - 80, height / 2);
      } else if (!m.parsed.boxes.length && !m.incoming.length && !gesture) {
        const lead = "Drag on the frame to draw a box";
        const rest = "Shift keeps a ratio · Alt draws from the centre";
        ctx.font = "10px system-ui,sans-serif";
        drawLabel(ctx, frameArea, lead, width / 2 - (ctx.measureText(lead).width + 8) / 2, height / 2 - 9);
        drawLabel(ctx, frameArea, rest, width / 2 - (ctx.measureText(rest).width + 8) / 2, height / 2 + 9, { muted: true });
      }
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to paint the boxes:`, error);
    }
  }

  /**
   * Draw the snapping grid where it is wide enough to read.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas.
   * @param {object} m - The model.
   * @returns {void}
   */
  function drawGrid(ctx, m) {
    if (m.grid <= 1) return;
    let step = m.grid;
    while (step * view.scale < 10) step *= 2;
    const cols = Math.floor(m.info.frame[0] / step);
    const rows = Math.floor(m.info.frame[1] / step);
    if (cols * rows > 40000) return;
    ctx.save();
    ctx.fillStyle = theme.fgMuted;
    ctx.globalAlpha = 0.28;
    for (let i = 1; i <= cols; i += 1) {
      const x = view.x0 + i * step * view.scale;
      for (let j = 1; j <= rows; j += 1) {
        ctx.fillRect(x - 0.5, view.y0 + j * step * view.scale - 0.5, 1, 1);
      }
    }
    ctx.restore();
  }

  /**
   * Draw the selected box's handles.
   *
   * @param {CanvasRenderingContext2D} ctx - The canvas.
   * @param {object} rect - The box on the frame.
   * @param {string} colour - Its colour.
   * @returns {void}
   */
  function drawHandles(ctx, rect, colour) {
    const r = toScreen(rect);
    const points = [
      [r.x, r.y], [r.x + r.w / 2, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h / 2],
      [r.x + r.w, r.y + r.h], [r.x + r.w / 2, r.y + r.h], [r.x, r.y + r.h], [r.x, r.y + r.h / 2],
    ];
    ctx.save();
    for (const [x, y] of points) {
      ctx.fillStyle = theme.fg;
      ctx.fillRect(x - HANDLE, y - HANDLE, HANDLE * 2, HANDLE * 2);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - HANDLE, y - HANDLE, HANDLE * 2, HANDLE * 2);
    }
    ctx.restore();
  }

  /**
   * The picture behind the frame, when the image input has one to show.
   *
   * @returns {CanvasImageSource|null} The picture.
   */
  function backdropPicture() {
    if (!isWired(node, "image")) return null;
    const upstream = sourceOf(node, "image")?.node?.imgs?.[0];
    if (upstream?.naturalWidth > 0 && upstream.complete) return upstream;
    return backdrop.state === PREVIEW_STATE.READY ? backdrop.image : null;
  }

  // ---- side panel and footer -------------------------------------------------------------------

  /**
   * Rebuild the layer list, front at the top.
   *
   * @returns {void}
   */
  function paintLayers() {
    const m = current;
    if (!m) return;
    const boxes = m.parsed.boxes;
    const shownCount = m.numbers.filter(Boolean).length;
    layersCount.textContent = boxes.length
      ? `${shownCount} of ${boxes.length}${m.incoming.length ? ` · ${m.incoming.length} in` : ""}`
      : "";
    layers.replaceChildren(dropLine);
    if (!boxes.length) {
      layers.appendChild(el("div", `color:${LOOK.muted};padding:2px 4px`, "No boxes yet"));
      return;
    }
    for (let index = boxes.length - 1; index >= 0; index -= 1) layers.appendChild(layerRow(index, boxes[index], m));
  }

  /**
   * One row of the layer list.
   *
   * @param {number} index - The box's place, 0 at the back.
   * @param {object} box - The box.
   * @param {object} m - The model.
   * @returns {HTMLElement} The row.
   */
  function layerRow(index, box, m) {
    const on = index === selected;
    const row = el("div", "display:flex;align-items:center;gap:5px;padding:2px 4px;border-radius:4px;cursor:pointer;"
      + `user-select:none;border:1px solid ${on ? LOOK.accent : "transparent"};`
      + `background:${on ? SELECTED_FILL : index === hover ? LOOK.surface : "transparent"};`
      + `opacity:${box.hidden ? 0.55 : 1}`);
    row.dataset.index = String(index);
    const swatch = el("span", `width:9px;height:9px;flex:0 0 auto;border-radius:2px;background:${colourOf(index, box)}`);
    const number = el("span", `flex:0 0 auto;font-variant-numeric:tabular-nums;color:${LOOK.muted};min-width:14px`,
      m.numbers[index] ? String(m.numbers[index]).padStart(2, "0") : "off");
    const kind = el("span", `flex:0 0 auto;color:${LOOK.muted}`, KIND_SHORT[box.kind] ?? "Obj");
    const label = box.kind === "text" && box.text ? `"${box.text}" ${box.caption}`.trim() : box.caption;
    const name = el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
      + `color:${label ? LOOK.text : LOOK.muted}`, label || `${KIND_LABELS[box.kind] ?? "Object"} ${m.numbers[index] || ""}`.trim());
    name.title = label || "";
    const eye = rowToggle(box.hidden ? EYE_OFF : EYE, box.hidden ? "Show" : "Hide", () => editBox(index, (target) => { target.hidden = !target.hidden; }));
    const lock = rowToggle(box.locked ? LOCK : UNLOCK, box.locked ? "Unlock" : "Lock", () => editBox(index, (target) => { target.locked = !target.locked; }));
    eye.style.color = box.hidden ? LOOK.muted : LOOK.text;
    lock.style.color = box.locked ? LOOK.text : LOOK.muted;
    row.append(swatch, number, kind, name, eye, lock);
    row.addEventListener("mouseenter", () => {
      hover = index;
      schedule();
    });
    row.addEventListener("mouseleave", () => {
      if (hover === index) hover = -1;
      schedule();
    });
    row.addEventListener("dblclick", () => {
      selected = index;
      paintAll();
      caption.focus();
    });
    row.addEventListener("pointerdown", (event) => startRowDrag(event, row, index));
    return row;
  }

  /**
   * Restyle the rows for the selection, without rebuilding them.
   *
   * @returns {void}
   */
  function markRows() {
    for (const row of layers.querySelectorAll("[data-index]")) {
      const on = Number(row.dataset.index) === selected;
      row.style.borderColor = on ? LOOK.accent : "transparent";
      row.style.background = on ? SELECTED_FILL : "transparent";
    }
  }

  /**
   * A small glyph button at the end of a row.
   *
   * @param {string[]} paths - The glyph.
   * @param {string} hint - What the hover says.
   * @param {() => void} onPress - What it does.
   * @returns {HTMLButtonElement} The button.
   */
  function rowToggle(paths, hint, onPress) {
    const element = el("button", "flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;"
      + "width:18px;height:16px;padding:0;border:0;border-radius:3px;background:transparent;cursor:pointer");
    element.type = "button";
    element.title = hint;
    element.appendChild(glyph(paths, 12));
    element.addEventListener("pointerdown", (event) => event.stopPropagation());
    element.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      try {
        onPress();
      } catch (error) {
        console.error(`[${EXT_NAME}] ${hint} failed:`, error);
      }
    });
    return element;
  }

  /**
   * Select a row, and reorder the stack when the row is dragged.
   *
   * @param {PointerEvent} event - The press.
   * @param {HTMLElement} row - The row.
   * @param {number} index - Its box.
   * @returns {void}
   */
  function startRowDrag(event, row, index) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    selected = index;
    markRows();
    paintInspector();
    schedule();
    const startY = event.clientY;
    let dragging = false;
    let target = index;
    const rows = () => [...layers.querySelectorAll("[data-index]")];
    const move = (moved) => {
      if ((moved.buttons & 1) === 0) {
        end();
        return;
      }
      if (!dragging && Math.abs(moved.clientY - startY) < 4) return;
      dragging = true;
      const list = rows();
      let slot = list.length;
      for (let at = 0; at < list.length; at += 1) {
        const box = list[at].getBoundingClientRect();
        if (moved.clientY < box.top + box.height / 2) {
          slot = at;
          break;
        }
      }
      const count = current.parsed.boxes.length;
      // List slots run front to back; a drop above slot `slot` lands behind the box shown there.
      const position = list.indexOf(row);
      const landing = slot > position ? slot - 1 : slot;
      target = count - 1 - landing;
      const anchor = list[Math.min(slot, list.length - 1)];
      const top = anchor
        ? anchor.offsetTop + (slot >= list.length ? anchor.offsetHeight : 0) - 1
        : 0;
      dropLine.style.top = `${top}px`;
      dropLine.style.display = "block";
    };
    const end = () => {
      row.removeEventListener("pointermove", move);
      row.removeEventListener("pointerup", end);
      row.removeEventListener("pointercancel", end);
      row.removeEventListener("lostpointercapture", end);
      dropLine.style.display = "none";
      if (dragging && target !== index) moveInStack(index, target);
    };
    row.addEventListener("pointermove", move);
    row.addEventListener("pointerup", end);
    row.addEventListener("pointercancel", end);
    row.addEventListener("lostpointercapture", end);
    try {
      row.setPointerCapture(event.pointerId);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to capture the pointer:`, error);
    }
  }

  /**
   * Show the selected box in the inspector, leaving any field being typed in alone.
   *
   * @returns {void}
   */
  function paintInspector() {
    const m = current;
    const box = m?.parsed.boxes[selected];
    const rect = m?.shown[selected];
    emptyNote.style.display = box ? "none" : "block";
    fields.style.display = box ? "flex" : "none";
    emptyNote.textContent = m?.parsed.boxes.length ? "Select a box to edit" : "No box selected";
    showValue(ratioPick, drawRatio);
    if (!box) return;
    inspectorTitle.textContent = m.numbers[selected]
      ? `Box ${m.numbers[selected]} · ${KIND_LABELS[box.kind]}`
      : `Hidden box · ${KIND_LABELS[box.kind]}`;
    showValue(boxRatio, box.ratio, box.ratio);
    const typing = document.activeElement;
    if (typing !== kindPick) kindPick.value = box.kind;
    if (typing !== caption && !typingTimer) caption.value = box.caption;
    if (typing !== words && !typingTimer) words.value = box.text;
    const showWords = box.kind === "text";
    wordsLabel.style.display = showWords ? "" : "none";
    words.style.display = showWords ? "" : "none";
    for (const [key, input] of Object.entries(geometryInputs)) {
      if (typing !== input) input.value = rect ? String(rect[key]) : "";
      input.disabled = !rect || box.locked;
    }
    colourSwatch.style.background = normaliseColour(box.color) ?? PALETTE[0];
    paintPalette(box);
  }

  /**
   * The swatches of colours the selected box asks of the model.
   *
   * @param {object} box - The box.
   * @returns {void}
   */
  function paintPalette(box) {
    paletteRow.replaceChildren();
    box.palette.forEach((colour, at) => {
      const swatch = swatchButton("Click to change, right-click to remove");
      swatch.style.background = colour;
      swatch.addEventListener("click", (event) => {
        pickColour(event.clientX, event.clientY, channels(colour), (rgb) => {
          const hex = `#${rgb.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
          editSelected((target) => { target.palette[at] = hex; });
        });
      });
      swatch.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        editSelected((target) => { target.palette.splice(at, 1); });
      });
      paletteRow.appendChild(swatch);
    });
    if (box.palette.length < MAX_PALETTE) {
      const add = swatchButton("Add a colour the model is asked to use");
      add.textContent = "+";
      add.style.cssText += `;background:${LOOK.surface};color:${LOOK.text};font:11px/14px ${SANS}`;
      add.addEventListener("click", (event) => {
        pickColour(event.clientX, event.clientY, channels(box.color), (rgb) => {
          const hex = `#${rgb.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
          editSelected((target) => { target.palette.push(hex); });
        });
      });
      paletteRow.appendChild(add);
    }
  }

  /**
   * Write the frame readout, the fidelity glyph and whatever needs doing.
   *
   * @returns {void}
   */
  function paintFooter() {
    const m = current;
    if (!m) return;
    const { frame, claim, from } = m.info;
    frameText.textContent = `${frame[0]} × ${frame[1]}${from ? ` · ${from}` : ""}`;
    frameText.title = from ? `Frame read from ${from}` : "Frame from width and height";
    const icon = claim === "exact" ? ICON.EXACT : claim === "last run" ? ICON.APPROXIMATE : ICON.WARNING;
    const detail = claim === "exact"
      ? "boxes sit where the node places them"
      : claim === "last run"
        ? "frame size from the node's last run"
        : "frame size is a guess until the node runs";
    drawFidelityGlyph(glyphCanvas, { icon, detail });
    notice.replaceChildren();
    if (!m.parsed.ok) {
      const words = el("span", `color:${LOOK.warning};overflow:hidden;text-overflow:ellipsis`, `Unreadable: ${m.parsed.error}`);
      words.title = `${m.parsed.error}. Fix it in the layout box, or paste a valid layout.`;
      notice.appendChild(words);
      return;
    }
    if (m.incoming.length) {
      notice.append(el("span", "", `${m.incoming.length} incoming`),
        controlButton(["Adopt"], "Copy the incoming boxes into the layout and stop reading the input", adoptIncoming));
    }
    if (!m.map.same && m.parsed.boxes.length) {
      const mode = controlSelect(RESIZE_MODES.map((value) => [value, value]), resizeMode(),
        "How a layout drawn at another size lands on this frame");
      mode.style.width = "auto";
      mode.addEventListener("change", () => writeWidget(node, "on_resize", mode.value));
      const drawn = el("span", "font-variant-numeric:tabular-nums", `from ${m.parsed.frame[0]} × ${m.parsed.frame[1]}`);
      drawn.title = "Size the layout was drawn";
      notice.append(drawn, mode, controlButton(["Adopt"], "Rewrite the layout at this frame size", adoptFrame));
      return;
    }
    if (!m.incoming.length) {
      const hidden = m.parsed.boxes.filter((box) => box.hidden).length;
      const count = m.numbers.filter(Boolean).length;
      notice.appendChild(el("span", "font-variant-numeric:tabular-nums",
        `${count} box${count === 1 ? "" : "es"}${hidden ? ` · ${hidden} hidden` : ""}`));
    }
  }

  /**
   * Paint everything from the model.
   *
   * @returns {void}
   */
  function paintAll() {
    try {
      rebuild();
      paintLayers();
      paintInspector();
      paintFooter();
      schedule();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to draw the editor:`, error);
    }
  }

  // ---- run output ------------------------------------------------------------------------------

  const reportOf = (output) => {
    const list = Array.isArray(output?.[REPORT_KEY]) ? output[REPORT_KEY] : [];
    const entry = list[0];
    return entry && typeof entry === "object" && ownOutput(entry) ? entry : null;
  };

  /**
   * Read the node's last run again, and its backdrop.
   *
   * @returns {void}
   */
  function refresh() {
    try {
      const store = app?.nodeOutputs ?? {};
      const output = store[nodeLocator(node)] ?? store[executionId(node)] ?? store[String(node.id)];
      const read = reportOf(output);
      if (read) report = read;
      fetchBackdrop();
      paintAll();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the node:`, error);
    }
  }

  /**
   * Ask for the picture the image input carried on the last run.
   *
   * @returns {void}
   */
  function fetchBackdrop() {
    if (!isWired(node, "image")) {
      backdrop = { state: PREVIEW_STATE.WAITING, image: null };
      return;
    }
    fetchInputPreview(node, "image").then((answer) => {
      if (disposed) return;
      backdrop = { state: answer?.state ?? PREVIEW_STATE.FAILED, image: answer?.image ?? null };
      schedule();
    }).catch((error) => console.error(`[${EXT_NAME}] Failed to fetch the backdrop:`, error));
  }

  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      const read = reportOf(detail?.output);
      if (read) report = read;
      fetchBackdrop();
      paintAll();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the node's report:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  // ---- upkeep ----------------------------------------------------------------------------------

  /**
   * Repaint when anything the editor reads has moved: the frame, the layout, or a setting.
   *
   * @returns {void}
   */
  function tick() {
    if (gesture || !root.isConnected) return;
    const info = readFrame();
    const key = [info.frame.join("x"), info.claim, info.from, widgetOf(node, "layout")?.value ?? "", gridValue(),
      resizeMode(), String(widgetOf(node, "bboxes_mode")?.value ?? ""), isWired(node, "image"), isWired(node, "bboxes"),
      colourBy()].join("|");
    if (key === lastKey) return;
    lastKey = key;
    paintAll();
  }

  function layoutFor() {
    const wide = root.clientWidth >= SIDE_BY_SIDE;
    body.style.flexDirection = wide ? "row" : "column";
    if (wide) {
      const width = Math.round(Math.max(SIDE_MIN, Math.min(root.clientWidth * 0.7, sizes.side)));
      side.style.flex = `0 0 ${width}px`;
      side.style.width = `${width}px`;
      side.style.height = "auto";
    } else {
      const room = Math.max(UNDER_MIN, root.clientHeight * 0.6);
      const height = Math.round(Math.max(UNDER_MIN, Math.min(room, sizes.under)));
      side.style.flex = `0 0 ${height}px`;
      side.style.width = "auto";
      side.style.height = `${height}px`;
    }
    const listRoom = wide ? sizes.layers : Math.min(sizes.layers, Math.max(LAYERS_MIN, side.clientHeight * 0.5));
    layers.style.height = `${Math.round(Math.max(LAYERS_MIN, listRoom))}px`;
    sideSplit.refresh();
    listSplit.refresh();
  }
  const sizeWatch = new ResizeObserver(() => {
    layoutFor();
    schedule();
  });
  sizeWatch.observe(root);
  sizeWatch.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, () => schedule());
  const stopTheme = onThemeChange((next) => {
    theme = next;
    paintAll();
  });
  const stopRun = onRunEnded(() => refresh());
  const stopTicking = joinTicking(node, tick, EXT_NAME);
  const releaseWheel = inWindow ? null : captureWheel(root, (event) => {
    const target = event.target instanceof Node ? event.target : null;
    const scroller = target && layers.contains(target) ? layers : target && caption.contains(target) ? caption : null;
    if (!scroller || scroller.scrollHeight <= scroller.clientHeight) return false;
    const before = scroller.scrollTop;
    scroller.scrollTop += wheelPixels(event, scroller).y;
    return scroller.scrollTop !== before;
  });
  root.addEventListener("contextmenu", (event) => event.preventDefault());

  const editor = { refresh, repaint: paintAll };
  editorsOf(node).add(editor);
  setTool("select");
  layoutFor();
  // The node's widgets hold their saved values only once it is configured, so the first read waits.
  requestAnimationFrame(() => {
    if (!disposed) refresh();
  });

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    try {
      editorsOf(node).delete(editor);
      clearTimeout(typingTimer);
      if (frameRequest) cancelAnimationFrame(frameRequest);
      frameRequest = 0;
      stopTicking();
      stopRun();
      stopTheme();
      stopRatio();
      releaseWheel?.();
      sizeWatch.disconnect();
      api.removeEventListener?.("executed", onExecuted);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the editor:`, error);
    }
  };

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    refresh,
    dispose,
  };
}

// Node -> the editors open on it, so a write from one repaints the other.
const editors = new WeakMap();

/**
 * The editors open on a node.
 *
 * @param {object} node - The node.
 * @returns {Set<object>} Its editors.
 */
function editorsOf(node) {
  let set = editors.get(node);
  if (!set) {
    set = new Set();
    editors.set(node, set);
  }
  return set;
}

/**
 * Repaint every editor open on a node.
 *
 * @param {object} node - The node.
 * @returns {void}
 */
function notifyEditors(node) {
  for (const editor of editorsOf(node)) {
    try {
      editor.repaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] An editor failed to repaint:`, error);
    }
  }
}

// Node -> its open window.
const windows = new WeakMap();

/**
 * Open the editor in a window beside the graph, or bring the open one forward.
 *
 * @param {object} node - The node.
 * @returns {void}
 */
function openWindow(node) {
  const open = windows.get(node);
  if (open) {
    open.window.open();
    return;
  }
  let entry = null;
  const close = () => {
    entry?.editor.dispose();
    entry?.window.dispose();
    windows.delete(node);
  };
  const win = createFloatingWindow({
    title: "Bounding Box Editor",
    badge: `${node.title ?? "Bounding Box Editor"} #${node.id}`,
    width: 1080,
    height: 720,
    minWidth: 560,
    minHeight: 380,
    storageKey: WINDOW_KEY,
    logName: EXT_NAME,
    onClose: close,
  });
  const editor = createBoxEditor(node, { inWindow: true });
  editor.element.style.flex = "1 1 auto";
  editor.element.style.minHeight = "0";
  win.body.appendChild(editor.element);
  entry = { window: win, editor, close };
  windows.set(node, entry);
  win.open();
}

/**
 * Close a node's window, if one is open.
 *
 * @param {object} node - The node.
 * @returns {void}
 */
function closeWindow(node) {
  windows.get(node)?.close();
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_box_editor_wrapped) return;
    proto.__was_box_editor_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        const editor = createBoxEditor(node, { onExpand: () => openWindow(node) });
        appendInterfaceWidget(node, editor, { name: UI_NAME, type: UI_TYPE });
        boundTextBoxes(node);
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          if (!graphLoading()) closeWindow(node);
          editor.dispose();
          return removed;
        };
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
          const configured = originalOnConfigure?.apply(this, args);
          if (!graphLoading()) editor.refresh();
          return configured;
        };
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
          const changed = originalOnConnectionsChange?.apply(this, args);
          notifyEditors(node);
          return changed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the editor:`, error);
      }
      return result;
    };
  },
});
