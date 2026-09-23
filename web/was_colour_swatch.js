/**
 * A colour swatch and picker on every widget that names a colour.
 *
 * `COLOUR_WIDGETS` maps a node id to its colour widgets. The swatch draws on the right of
 * the value and opens the picker when clicked.
 */

import { app } from "../../scripts/app.js";
import { parseColor, STATUS } from "./interface/colour_cell.js";
import { openColourPicker } from "./interface/colour_picker.js";

const EXT_NAME = "WASNodeSuite.ColourSwatch";
const SETTING_ID = "WAS.ColourSwatch.ShowSwatch";

// Node id -> the string widgets holding a colour. `HSL to Hex` is absent: its widget takes an
// `hsl(...)` string, which a picked hex would replace with a value the node cannot read.
const COLOUR_WIDGETS = {
  "Hex to HSL": ["hex_color"],
  "Image Resize": ["pad_color"],
  WASDrawImageBounds: ["color"],
  WASImageDrawText: ["text_color", "stroke_color", "background_color"],
  WASNSImageTileExtract: ["border_color"],
  WASImageTileExtractGrid: ["border_color"],
  WASNSImageTileShuffle: ["border_color"],
  WASLoadImageSequence: ["pad_color"],
  WASLoadImagesFromZIP: ["pad_color"],
  WASLoadVideo: ["pad_color"],
  WASLoadVideoUpload: ["pad_color"],
};

// Geometry of the swatch pill, matching the one core draws on its own colour widgets.
const SWATCH_WIDTH = 40;
const SWATCH_INSET = 3;
const RIGHT_PADDING = 10;
// From the node edge to the widget edge, which is litegraph's own widget margin.
const WIDGET_MARGIN = 15;

// The checkerboard alpha is shown against, and the side of one square in graph pixels.
const CHECKER_LIGHT = "#999999";
const CHECKER_DARK = "#666666";
const CHECKER_SIZE = 4;

// Drawn in place of a colour where the value names none.
const MARK_EMPTY = "-";
const MARK_DECLINED = "()";
const MARK_INVALID = "?";
const MARK_FONT = "9px sans-serif";
const MARK_COLOUR = "#ff9800";

/**
 * Read whether the swatch is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
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
 * Work out where the swatch sits inside a widget of a given width.
 *
 * @param {number} width - Widget width in graph pixels.
 * @param {number} y - Top of the widget in node space.
 * @param {number} height - Widget height in graph pixels.
 * @returns {{x: number, y: number, width: number, height: number}} The swatch rectangle.
 */
function swatchRect(width, y, height) {
  return {
    x: width - WIDGET_MARGIN - RIGHT_PADDING - SWATCH_WIDTH,
    y: y + SWATCH_INSET,
    width: SWATCH_WIDTH,
    height: Math.max(1, height - SWATCH_INSET * 2),
  };
}

/**
 * Paint the checkerboard a partly transparent colour is shown against.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {{x: number, y: number, width: number, height: number}} rect - The swatch.
 * @returns {void}
 */
function paintChecker(ctx, rect) {
  ctx.fillStyle = CHECKER_LIGHT;
  ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  ctx.fillStyle = CHECKER_DARK;
  for (let row = 0; row * CHECKER_SIZE < rect.height; row++) {
    for (let column = 0; column * CHECKER_SIZE < rect.width; column++) {
      if ((row + column) % 2 === 0) continue;
      const x = rect.x + column * CHECKER_SIZE;
      const y = rect.y + row * CHECKER_SIZE;
      ctx.fillRect(
        x,
        y,
        Math.min(CHECKER_SIZE, rect.x + rect.width - x),
        Math.min(CHECKER_SIZE, rect.y + rect.height - y),
      );
    }
  }
}

/**
 * Paint the swatch a parse result calls for.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {{x: number, y: number, width: number, height: number}} rect - The swatch.
 * @param {object} parsed - Result from `parseColor`.
 * @returns {void}
 */
function paintSwatch(ctx, rect, parsed) {
  const radius = rect.height / 2;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(rect.x, rect.y, rect.width, rect.height, radius);
  ctx.clip();

  paintChecker(ctx, rect);
  if (parsed?.status === STATUS.COLOUR && Array.isArray(parsed.rgba)) {
    const [red, green, blue] = parsed.rgba;
    const alpha = parsed.rgba.length > 3 ? parsed.rgba[3] / 255 : 1;
    ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha})`;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
  } else {
    const mark =
      parsed?.status === STATUS.EMPTY
        ? MARK_EMPTY
        : parsed?.status === STATUS.DECLINED
          ? MARK_DECLINED
          : MARK_INVALID;
    ctx.fillStyle = MARK_COLOUR;
    ctx.font = MARK_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(mark, rect.x + rect.width / 2, rect.y + rect.height / 2);
  }

  ctx.restore();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1, radius);
  ctx.stroke();
}

/**
 * Write a picked colour back, keeping any alpha the value already carried.
 *
 * @param {number[]} rgb - The picked colour, three channels.
 * @param {object} parsed - The parse of the value before the pick.
 * @returns {string} The colour to store.
 */
function withExistingAlpha(rgb, parsed) {
  const hex = (value) => {
    const whole = Math.max(0, Math.min(255, Math.round(Number(value) || 0)));
    return whole.toString(16).padStart(2, "0");
  };
  const opaque = `#${hex(rgb[0])}${hex(rgb[1])}${hex(rgb[2])}`;
  const alpha =
    parsed?.status === STATUS.COLOUR && Array.isArray(parsed.rgba) && parsed.rgba.length > 3
      ? Math.round(parsed.rgba[3])
      : 255;
  return alpha === 255 ? opaque : `${opaque}${hex(alpha)}`;
}

/**
 * Give one widget a swatch and a picker.
 *
 * @param {object} node - The node the widget sits on.
 * @param {object} widget - The string widget holding a colour.
 * @returns {void}
 */
export function attachSwatch(node, widget) {
  if (!widget || widget.__was_colour_swatch) return;
  widget.__was_colour_swatch = true;

  const originalDraw = widget.drawWidget?.bind(widget);
  const originalClick = widget.onClick?.bind(widget);

  // Both the plain object and the concrete widget carry `drawWidget` and `onClick`.
  widget.drawWidget = function (ctx, options) {
    originalDraw?.(ctx, options);
    if (!enabled() || options?.showText === false) return;
    try {
      const rect = swatchRect(options?.width ?? node.size[0], this.y ?? 0, this.height ?? 20);
      if (rect.width <= 0 || rect.x <= WIDGET_MARGIN) return;
      paintSwatch(ctx, rect, parseColor(String(this.value ?? "")));
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to draw the swatch:`, error);
    }
  };

  widget.onClick = function (options) {
    const { e, canvas } = options ?? {};
    let onSwatch = false;
    try {
      const rect = swatchRect(node.size[0], this.y ?? 0, this.height ?? 20);
      const localX = (e?.canvasX ?? 0) - (node.pos?.[0] ?? 0);
      onSwatch = enabled() && localX >= rect.x && localX <= rect.x + rect.width;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to test the swatch:`, error);
    }

    if (!onSwatch) {
      originalClick?.(options);
      return;
    }

    const parsed = parseColor(String(this.value ?? ""));
    const opening = parsed?.status === STATUS.COLOUR ? parsed.rgba : [0, 0, 0];
    openColourPicker(e?.clientX ?? 0, e?.clientY ?? 0, opening, (rgb) => {
      const stored = withExistingAlpha(rgb, parsed);
      if (typeof this.setValue === "function") {
        this.setValue(stored, options);
      } else {
        this.value = stored;
        this.callback?.(stored);
      }
      canvas?.setDirty?.(true);
    });
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Colour", "Show the swatch"],
      name: "Show a colour swatch and picker",
      tooltip:
        "Draw the colour a widget names as a swatch on the right of its value, and open a " +
        "colour picker when the swatch is clicked. The value stays text, so a name, a " +
        "#RRGGBBAA with transparency, and an empty field all stay typeable.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const names = COLOUR_WIDGETS[nodeData?.name];
    if (!names) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap the draw and
    // the click a second time on every node of this type.
    if (proto.__was_colour_swatch_wrapped) return;
    proto.__was_colour_swatch_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        for (const name of names) {
          attachSwatch(this, (this.widgets ?? []).find((widget) => widget.name === name));
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to attach ${nodeData.name}'s swatches:`, error);
      }
      return result;
    };
  },
});

export { COLOUR_WIDGETS };
