/**
 * The drawing a mask interface holds, with the brush and the chip row.
 *
 * `createMaskPaint` answers a `layer`, a `tool` with first refusal on the pointer, and `chips`.
 * A radius is in mask pixels, and positions arrive in frame units.
 */

import { EMPTY_MASK_VALUE, decodeMask, encodeMask, maskStoreSize, readMaskHeader } from "./mask_value.mjs";
import { withGraphChange } from "./region.mjs";

const LOG_NAME = "WASNodeSuite.MaskPaint";

/**
 * What the pointer does over the frame.
 *
 * `RECT` leaves every gesture to the rectangle.
 */
export const PAINT_MODE = {
  RECT: "rect",
  PAINT: "paint",
  ERASE: "erase",
};

// The order the mode chip cycles in, for an adopter that names none.
const MODE_ORDER = [PAINT_MODE.RECT, PAINT_MODE.PAINT, PAINT_MODE.ERASE];

// What each mode does, for the mode chip's hover text. Only the offered ones are listed, so a
// node with no rectangle never reads that a chip drags one.
const MODE_MEANING = {
  [PAINT_MODE.RECT]: "rect drags the rectangle",
  [PAINT_MODE.PAINT]: "paint adds to the drawing",
  [PAINT_MODE.ERASE]: "erase takes it away",
};

// The brush, in mask pixels. The smallest is one pixel across, and the largest is what covers a
// quarter of a 512 canvas in one stamp.
const DEFAULT_RADIUS = 16;
const MIN_RADIUS = 1;
const MAX_RADIUS = 256;

// How far a drag on the size chip covers the whole range in, in element pixels. The frame's own
// width is used where there is one, so the gesture is the same length whatever the node is
// drawn at, and this is the fallback while the frame states no size.
const SIZE_DRAG_SPAN = 220;

// The chip row, in element pixels.
const CHIP_HEIGHT = 14;
const CHIP_PAD = 5;
const CHIP_GAP = 4;
const CHIP_INSET = 4;
const CHIP_FONT = "9px sans-serif";
const CHIP_ALPHA = 0.82;

// How solid the drawing is drawn over the backdrop. Light: the backdrop is already the
// finished mask, and this says which of it came from the brush rather than from the rectangle.
const LAYER_ALPHA = 0.32;

// How solid it is drawn where the combine is one whose effect the backdrop cannot show on its
// own: `subtract` takes a hole out, and `off` leaves the drawing out of the run entirely.
const LAYER_ALPHA_STATE = 0.45;

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
 * Join a drawing with a computed mask, at one pixel.
 *
 * Set arithmetic rather than addition.
 *
 * @param {number} computed - The computed mask at that pixel, 0 to 1.
 * @param {number} drawn - The drawing at that pixel, 0 to 1.
 * @param {string} mode - One of `union`, `subtract`, `intersect` or `off`. Anything else
 *   answers the computed value.
 * @returns {number} The joined value, 0 to 1.
 */
export function combine(computed, drawn, mode) {
  if (mode === "union") return Math.max(computed, drawn);
  if (mode === "subtract") return clamp(computed - drawn, 0, 1);
  if (mode === "intersect") return Math.min(computed, drawn);
  return computed;
}

// One context serves every chip row on the page, since text is measured in element pixels and
// the font is this module's own.
let measureContext = null;

/**
 * Measure a chip label.
 *
 * @param {string} text - Label to measure.
 * @returns {number} Its width in element pixels.
 */
function labelWidth(text) {
  if (!measureContext) measureContext = document.createElement("canvas").getContext("2d");
  if (!measureContext) return text.length * 5;
  measureContext.font = CHIP_FONT;
  return measureContext.measureText(text).width;
}

/**
 * Build the drawing one node holds, and the brush over it where one is offered.
 *
 * @param {object} options - What the drawing covers and where it is written.
 * @param {() => {width: number, height: number}} options.frame - The frame a position arrives
 *   in, in frame units.
 * @param {() => {width: number, height: number}} options.canvas - The mask itself, in pixels. A
 *   radius is counted in these, and the drawing is stored at this size held to the long edge
 *   cap.
 * @param {{read: () => string, write: (value: string) => void}} options.value - The accessor
 *   pair for the node's `drawn_mask` widget. `read` answers what it holds and `write` stores a
 *   value, without a bracket of its own.
 * @param {() => string} [options.combine] - What the node's `drawn_combine` widget holds, which
 *   decides the colour the drawing is drawn in and whether the run reads it at all.
 * @param {string[]} [options.modes] - The values of `PAINT_MODE` the mode chip cycles, in order,
 *   the first being the one a fresh interface starts in. An adopter with no rectangle leaves
 *   `rect` out. A list naming one mode draws no chip at all, since there is nothing to cycle,
 *   and an empty list offers no brush at all: `tool` is null, no mode, size or clear chip is
 *   drawn, and no gesture reaches the drawing. Absent, every mode is offered.
 * @param {() => Array<{key: string, label: string, title?: string, press: () => void}>}
 *   [options.actions] - Chips of the adopter's own, drawn after this module's in the order given
 *   and asked for again on every layout, so one that only means something in some states is
 *   left out of the list in the others.
 * @returns {{tool: object|null, chips: object, layer: Function, bind: (editor: object) => void,
 *   mode: () => string, radius: () => number, header: () => object|null, bytes: () => number,
 *   sample: (width: number, height: number) => Float32Array|null, version: () => number,
 *   invalidate: () => void, adopt: (canvas: HTMLCanvasElement|null) => void,
 *   dispose: () => void}} The layer and one of the two pointer surfaces to hand
 *   `createRegionEditor`, the repaint hookup, what is on screen, the drawing reduced for a
 *   picture of the finished mask, the way a drawing made elsewhere is taken on, and teardown.
 */
export function createMaskPaint(options = {}) {
  // Filtered against the known modes, so the mode the interface starts in is always one the
  // pointer knows what to do with. A list is taken at its word once it is filtered: an adopter
  // naming none of them is offering no brush, and one naming no list at all takes them all.
  const offered = Array.isArray(options.modes)
    ? [...new Set(options.modes.filter((mode) => Object.values(PAINT_MODE).includes(mode)))]
    : MODE_ORDER;

  // Whether the pointer draws at all. Without it the interface holds `rect`, the mode every
  // gesture is refused in, and the mode, size and clear chips are left undrawn.
  const painting = offered.length > 0;

  const settings = {
    frame: typeof options.frame === "function" ? options.frame : () => ({ width: 0, height: 0 }),
    canvas: typeof options.canvas === "function" ? options.canvas : () => ({ width: 0, height: 0 }),
    value: options.value ?? {},
    combine: typeof options.combine === "function" ? options.combine : () => "union",
    modes: offered,
    actions: typeof options.actions === "function" ? options.actions : () => [],
  };

  const state = {
    // `rect` stands for no brush as well as for a rectangle, since neither takes a gesture here.
    mode: settings.modes[0] ?? PAINT_MODE.RECT,
    radius: DEFAULT_RADIUS,
    // The strokes at the size they are stored at, white where drawn and transparent elsewhere.
    strokes: null,
    // A copy taken at the start of a gesture, so an interrupted one puts the pixels back.
    before: null,
    // What the reduced, tinted copy the layer draws is built in.
    tint: null,
    // The value the strokes were built from, so a repaint does not decode the same string
    // again, and the value last written, so the interface's own write is not read back as an
    // edit made somewhere else.
    loadedFrom: null,
    lastWritten: null,
    loading: false,
    stroke: null,
    sizeDrag: null,
    pointer: null,
    chips: [],
    editor: null,
    onChanged: null,
    // Bumped whenever the strokes canvas changes, so a reduced copy taken for the picture is
    // kept until the drawing under it moves.
    version: 0,
    reduced: null,
    reducedKey: "",
    disposed: false,
  };

  /**
   * Ask the editor to draw again.
   *
   * @returns {void}
   */
  function repaint() {
    state.editor?.schedulePaint?.();
  }

  /**
   * Repaint, and tell the adopter the drawing moved.
   *
   * @returns {void}
   */
  function announce() {
    repaint();
    try {
      state.onChanged?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] The adopter failed to answer a changed drawing:`, error);
    }
  }

  /**
   * The size the drawing is stored at for the mask as it stands.
   *
   * @returns {{width: number, height: number}} The store size in pixels.
   */
  function storeSize() {
    const canvas = settings.canvas();
    return maskStoreSize(canvas.width, canvas.height);
  }

  /**
   * Read the widget.
   *
   * @returns {string} What `drawn_mask` holds, empty when it cannot be read.
   */
  function readValue() {
    try {
      const value = settings.value.read?.();
      return typeof value === "string" ? value : EMPTY_MASK_VALUE;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the drawn mask:`, error);
      return EMPTY_MASK_VALUE;
    }
  }

  /**
   * The header of the drawing on screen.
   *
   * @returns {{width: number, height: number, data: string}|null} The stored size and body, or
   *   null when nothing is drawn.
   */
  function header() {
    return readMaskHeader(readValue());
  }

  /**
   * How many bytes the drawing adds to the workflow.
   *
   * @returns {number} The value's length, 0 when nothing is drawn.
   */
  function bytes() {
    const value = readValue();
    return typeof value === "string" ? value.length : 0;
  }

  /**
   * Build the strokes canvas from the widget, once per value.
   *
   * The decode is asynchronous, so the layer draws nothing until it lands and then repaints.
   *
   * @returns {void}
   */
  function ensureStrokes() {
    if (state.disposed || state.loading) return;
    const value = readValue();
    if (state.loadedFrom === value) return;

    if (!readMaskHeader(value)) {
      // Either nothing is drawn or the value is not this format, and both are drawn as nothing
      // rather than as the last drawing, which would be a picture of a value nobody holds.
      state.strokes = null;
      state.version += 1;
      state.loadedFrom = value;
      announce();
      return;
    }

    state.loading = true;
    decodeMask(value)
      .then((answer) => {
        if (state.disposed) return;
        // A stroke started while the decode was in flight owns the canvas, and the value it
        // will write already carries what was decoded, so the answer is dropped.
        if (state.stroke) return;
        state.strokes = answer?.canvas ?? null;
        state.version += 1;
        state.loadedFrom = value;
      })
      .catch((error) => {
        console.error(`[${LOG_NAME}] Failed to build the drawing:`, error);
        state.loadedFrom = value;
      })
      .finally(() => {
        state.loading = false;
        if (!state.disposed) announce();
      });
  }

  /**
   * The canvas a stroke is drawn into, made at the store size when there is none.
   *
   * @returns {HTMLCanvasElement|null} The strokes canvas, or null when one cannot be made.
   */
  function strokeTarget() {
    if (state.strokes) return state.strokes;
    const size = storeSize();
    if (!(size.width > 0) || !(size.height > 0)) return null;
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    state.strokes = canvas;
    state.version += 1;
    return canvas;
  }

  /**
   * Convert a position in frame units into the strokes canvas's own pixels.
   *
   * @param {{x: number, y: number}} point - Position in frame units.
   * @param {HTMLCanvasElement} target - The strokes canvas.
   * @returns {{x: number, y: number}|null} The position in stroke pixels, or null while the
   *   frame states no size to convert through.
   */
  function strokePoint(point, target) {
    const frame = settings.frame();
    if (!(frame.width > 0) || !(frame.height > 0)) return null;
    return {
      x: (point.x * target.width) / frame.width,
      y: (point.y * target.height) / frame.height,
    };
  }

  /**
   * The brush radius in the strokes canvas's own pixels.
   *
   * @param {HTMLCanvasElement} target - The strokes canvas.
   * @returns {number} The radius, never below half a pixel.
   */
  function strokeRadius(target) {
    const canvas = settings.canvas();
    const scale = canvas.width > 0 ? target.width / canvas.width : 1;
    return Math.max(0.5, state.radius * scale);
  }

  /**
   * Lay one segment of a stroke down.
   *
   * @param {{x: number, y: number}} from - Where the segment starts, in stroke pixels.
   * @param {{x: number, y: number}} to - Where it ends, in stroke pixels.
   * @param {boolean} erasing - Take the drawing away rather than add to it.
   * @returns {void}
   */
  function drawSegment(from, to, erasing) {
    const target = state.strokes;
    const ctx = target?.getContext("2d");
    if (!ctx) return;

    ctx.save();
    // An erase takes the alpha out rather than painting black, so the two directions of the
    // same gesture leave a canvas the encoder reads the same way.
    ctx.globalCompositeOperation = erasing ? "destination-out" : "source-over";
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = strokeRadius(target) * 2;
    if (from.x === to.x && from.y === to.y) {
      // A press that never moves is a dab, and a zero length stroke draws nothing at all.
      ctx.beginPath();
      ctx.arc(to.x, to.y, strokeRadius(target), 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
    ctx.restore();
    state.version += 1;
  }

  /**
   * Keep a copy of the drawing before a gesture changes it.
   *
   * @returns {void}
   */
  function snapshot() {
    const target = state.strokes;
    if (!target) {
      state.before = null;
      return;
    }
    if (!state.before) state.before = document.createElement("canvas");
    state.before.width = target.width;
    state.before.height = target.height;
    const ctx = state.before.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, target.width, target.height);
    ctx.drawImage(target, 0, 0);
  }

  /**
   * Put the drawing back to the copy taken before the gesture.
   *
   * @returns {void}
   */
  function restore() {
    const target = state.strokes;
    const ctx = target?.getContext("2d");
    if (!ctx || !state.before) return;
    ctx.clearRect(0, 0, target.width, target.height);
    ctx.drawImage(state.before, 0, 0);
    state.version += 1;
  }

  /**
   * Write the drawing into the widget, once, with its own undo entry.
   *
   * @param {string} value - The value to store.
   * @returns {void}
   */
  function commit(value) {
    if (state.disposed) return;
    state.lastWritten = value;
    state.loadedFrom = value;
    withGraphChange(() => {
      try {
        settings.value.write?.(value);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to store the drawing:`, error);
      }
    });
    // A single line string widget is a plain widget whose value setter runs no callback, so
    // the repaint that follows a write is asked for here rather than waited for.
    announce();
  }

  /**
   * Store whatever is on the strokes canvas.
   *
   * @returns {void}
   */
  function commitStrokes() {
    const target = state.strokes;
    if (!target) return;
    commit(encodeMask(target));
  }

  /**
   * Throw the drawing away.
   *
   * @returns {void}
   */
  function clearDrawing() {
    state.strokes = null;
    state.version += 1;
    commit(EMPTY_MASK_VALUE);
  }

  /**
   * Drop the decoded copy so the next repaint reads the widget again.
   *
   * @returns {void}
   */
  function invalidate() {
    if (readValue() === state.lastWritten) return;
    state.loadedFrom = null;
    ensureStrokes();
  }

  /**
   * The drawing's coverage, reduced to a given size.
   *
   * @param {number} width - Columns wanted.
   * @param {number} height - Rows wanted.
   * @returns {Float32Array|null} Coverage from 0 to 1, row major, or null when nothing is
   *   drawn or the reduction cannot be made.
   */
  function sample(width, height) {
    const strokes = state.strokes;
    const columns = Math.max(1, Math.round(width));
    const rows = Math.max(1, Math.round(height));
    if (!strokes) return null;

    const key = `${columns}x${rows}:${state.version}`;
    if (state.reducedKey === key && state.reduced) return state.reduced;

    const scratch = document.createElement("canvas");
    scratch.width = columns;
    scratch.height = rows;
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(strokes, 0, 0, columns, rows);

    const pixels = ctx.getImageData(0, 0, columns, rows).data;
    const coverage = new Float32Array(columns * rows);
    for (let at = 0; at < coverage.length; at++) coverage[at] = pixels[at * 4 + 3] / 255;

    state.reduced = coverage;
    state.reducedKey = key;
    return coverage;
  }

  /**
   * The chips drawn over the top left of the frame, in element pixels.
   *
   * @param {object} view - View from the region editor.
   * @returns {Array<object>} Each chip's box, its label, whether it is the active one, and what
   *   it says on hover.
   */
  function chipLayout(view) {
    const canvas = settings.canvas();
    const drawn = header();
    const chips = [];
    let x = view.x0 + CHIP_INSET;
    const y = view.y0 + CHIP_INSET;

    const add = (key, label, active, title, press) => {
      const width = labelWidth(label) + CHIP_PAD * 2;
      // A chip that would run past the frame is left out rather than drawn over the edge,
      // which is what keeps a narrow node readable instead of covered.
      if (x + width > view.x0 + view.drawWidth - CHIP_INSET) return;
      chips.push({ key, x, y, width, height: CHIP_HEIGHT, label, active, title, press });
      x += width + CHIP_GAP;
    };

    // One offered mode is the mode, so the chip would name a state that cannot be left.
    if (settings.modes.length > 1) {
      add(
        "mode",
        state.mode,
        state.mode !== PAINT_MODE.RECT,
        `What the pointer does over the picture. ${settings.modes
          .map((mode) => MODE_MEANING[mode])
          .join(", ")}. Click to change.`,
      );
    }

    if (state.mode !== PAINT_MODE.RECT) {
      // The rectangle's own softening is named only where there is a rectangle, since an
      // adopter that offers no `rect` mode carries no `blur_radius` widget either.
      const softening = settings.modes.includes(PAINT_MODE.RECT)
        ? " The brush has a hard edge; blur_radius softens the rectangle and not the drawing."
        : " The brush has a hard edge.";
      add(
        "size",
        `${Math.round(state.radius * 2)} px`,
        false,
        `How wide the brush is, in pixels of the ${canvas.width}x${canvas.height} mask. Drag `
          + `this across to change it.${softening}`,
      );
    }

    if (painting && drawn) {
      add(
        "clear",
        "clear",
        false,
        `Throw the drawing away. It was made at ${drawn.width}x${drawn.height} and the widget `
          + "holds it, so clearing it is one undo away from coming back.",
      );
    }

    // The adopter's own chips go last, so the brush's stay where they are as ones come and go.
    for (const action of readActions()) {
      add(`action:${action.key}`, action.label, false, action.title ?? "", action.press);
    }

    state.chips = chips;
    return chips;
  }

  /**
   * The chips the adopter offers as things stand.
   *
   * @returns {Array<object>} Each entry's key, label, hover text and what pressing it does.
   *   Empty for an adopter that offers none or answers something unusable.
   */
  function readActions() {
    let offeredChips = null;
    try {
      offeredChips = settings.actions();
    } catch (error) {
      console.error(`[${LOG_NAME}] The adopter failed to name its chips:`, error);
      return [];
    }
    if (!Array.isArray(offeredChips)) return [];
    return offeredChips.filter(
      (action) => action?.key && action?.label && typeof action.press === "function",
    );
  }

  /**
   * Which chip a point is on.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {object|null} The chip, or null when the point is on none.
   */
  function chipAt(point) {
    for (const chip of state.chips) {
      if (
        point.x >= chip.x && point.x <= chip.x + chip.width &&
        point.y >= chip.y && point.y <= chip.y + chip.height
      ) {
        return chip;
      }
    }
    return null;
  }

  /**
   * Whether a point is inside the picture the strokes go on.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @param {object} view - View from the region editor.
   * @returns {boolean} True while the point is over the frame.
   */
  function insideFrame(point, view) {
    return (
      point.x >= view.x0 && point.x <= view.x0 + view.drawWidth &&
      point.y >= view.y0 && point.y <= view.y0 + view.drawHeight
    );
  }

  /**
   * The colour the drawing is drawn in, which follows the combine the node states.
   *
   * @param {object} theme - Theme tokens.
   * @returns {{colour: string, alpha: number}} What to tint the drawing with and how solid to
   *   lay it over the backdrop.
   */
  function layerPaint(theme) {
    const mode = settings.combine();
    if (mode === "subtract") return { colour: theme.warning, alpha: LAYER_ALPHA_STATE };
    if (mode === "off") return { colour: theme.fgMuted, alpha: LAYER_ALPHA_STATE };
    return { colour: theme.accent, alpha: LAYER_ALPHA };
  }

  /**
   * Draw the drawing over the frame, reduced and tinted.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} view - View from the region editor.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawStrokes(ctx, view, theme) {
    const strokes = state.strokes;
    if (!strokes) return;

    const width = Math.max(1, Math.round(view.drawWidth));
    const height = Math.max(1, Math.round(view.drawHeight));
    if (!state.tint) state.tint = document.createElement("canvas");
    const tint = state.tint;
    if (tint.width !== width) tint.width = width;
    if (tint.height !== height) tint.height = height;

    const tintCtx = tint.getContext("2d");
    if (!tintCtx) return;
    tintCtx.setTransform(1, 0, 0, 1, 0, 0);
    tintCtx.globalCompositeOperation = "source-over";
    tintCtx.clearRect(0, 0, width, height);
    tintCtx.drawImage(strokes, 0, 0, width, height);

    const paint = layerPaint(theme);
    // `source-in` keeps the reduced coverage as the alpha and replaces every colour under it,
    // so a soft edge stays soft and the whole drawing takes the one colour.
    tintCtx.globalCompositeOperation = "source-in";
    tintCtx.fillStyle = paint.colour;
    tintCtx.fillRect(0, 0, width, height);

    ctx.globalAlpha = paint.alpha;
    ctx.drawImage(tint, view.x0, view.y0, view.drawWidth, view.drawHeight);
    ctx.globalAlpha = 1;
  }

  /**
   * Draw the brush's own outline where the pointer is.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} view - View from the region editor.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawCursor(ctx, view, theme) {
    if (state.mode === PAINT_MODE.RECT) return;
    const point = state.pointer;
    if (!point || !insideFrame(point, view)) return;
    if (chipAt(point)) return;

    const canvas = settings.canvas();
    const scale = canvas.width > 0 ? view.drawWidth / canvas.width : 1;
    const radius = Math.max(1, state.radius * scale);

    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    // Two rings, one dark and one light, so the outline is visible over a white mask and over
    // a black one without asking which is behind it.
    ctx.lineWidth = 3;
    ctx.strokeStyle = theme.bg;
    ctx.globalAlpha = 0.55;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = state.mode === PAINT_MODE.ERASE ? theme.warning : theme.accent;
    ctx.stroke();
  }

  /**
   * Draw the chip row.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} view - View from the region editor.
   * @param {object} theme - Theme tokens.
   * @returns {Array<object>} The hover regions the chips answer.
   */
  function drawChips(ctx, view, theme) {
    const chips = chipLayout(view);
    ctx.font = CHIP_FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";

    for (const chip of chips) {
      ctx.globalAlpha = CHIP_ALPHA;
      ctx.fillStyle = chip.active ? theme.accent : theme.panelBg;
      ctx.fillRect(chip.x, chip.y, chip.width, chip.height);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = theme.border;
      ctx.lineWidth = 1;
      ctx.strokeRect(chip.x + 0.5, chip.y + 0.5, chip.width - 1, chip.height - 1);
      ctx.fillStyle = chip.active ? theme.selectionText : theme.fg;
      ctx.fillText(chip.label, chip.x + CHIP_PAD, chip.y + chip.height / 2, chip.width - CHIP_PAD * 2);
    }

    return chips.map((chip) => ({
      x: chip.x,
      y: chip.y,
      width: chip.width,
      height: chip.height,
      title: chip.title,
    }));
  }

  /**
   * The layer the region editor draws between the backdrop and the rectangle.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} view - View from the region editor.
   * @param {object} theme - Theme tokens.
   * @returns {Array<object>} Hover regions for the chips.
   */
  function layer(ctx, view, theme) {
    ensureStrokes();
    drawStrokes(ctx, view, theme);
    const regions = drawChips(ctx, view, theme);
    drawCursor(ctx, view, theme);
    return regions;
  }

  /**
   * Act on a chip that was pressed.
   *
   * @param {object} chip - The chip from `chipAt`.
   * @param {object} context - The context the region editor hands over.
   * @returns {void}
   */
  function pressChip(chip, context) {
    if (typeof chip.press === "function") {
      try {
        chip.press();
      } catch (error) {
        console.error(`[${LOG_NAME}] The adopter failed to answer a pressed chip:`, error);
      }
      repaint();
      return;
    }
    if (chip.key === "mode") {
      const at = settings.modes.indexOf(state.mode);
      state.mode = settings.modes[(at + 1) % settings.modes.length];
      repaint();
      return;
    }
    if (chip.key === "clear") {
      clearDrawing();
      return;
    }
    if (chip.key === "size") {
      state.sizeDrag = { startX: context.point.x, startRadius: state.radius };
    }
  }

  const tool = {
    /**
     * What the pointer looks like over the frame.
     *
     * @param {object} context - The context the region editor hands over.
     * @returns {string} A CSS cursor, empty where the rectangle decides.
     */
    cursor(context) {
      if (state.sizeDrag) return "ew-resize";
      if (chipAt(context.point)) return "pointer";
      if (state.mode === PAINT_MODE.RECT) return "";
      if (!context.view || !insideFrame(context.point, context.view)) return "";
      // The outline drawn at the pointer is the cursor, so the system one is taken away rather
      // than drawn on top of it.
      return "none";
    },

    /**
     * Take a press on a chip or start a stroke.
     *
     * @param {PointerEvent} event - The press.
     * @param {object} context - The context the region editor hands over.
     * @returns {boolean} True when the brush took the press.
     */
    pointerDown(event, context) {
      if (event.button !== 0) return false;

      const chip = chipAt(context.point);
      if (chip) {
        pressChip(chip, context);
        return true;
      }

      if (state.mode === PAINT_MODE.RECT) return false;
      if (!context.view || !insideFrame(context.point, context.view)) return false;
      if (!context.frame) return false;

      const target = strokeTarget();
      if (!target) return false;
      const at = strokePoint(context.frame, target);
      if (!at) return false;

      snapshot();
      state.stroke = { last: at, erasing: state.mode === PAINT_MODE.ERASE };
      drawSegment(at, at, state.stroke.erasing);
      repaint();
      return true;
    },

    /**
     * Carry a stroke, a size drag, or the brush outline.
     *
     * @param {PointerEvent} event - The move.
     * @param {object} context - The context the region editor hands over.
     * @returns {boolean} True when the brush took the move.
     */
    pointerMove(event, context) {
      state.pointer = context.point;

      if (state.sizeDrag) {
        const span = context.view ? Math.max(1, context.view.drawWidth) : SIZE_DRAG_SPAN;
        const travelled = context.point.x - state.sizeDrag.startX;
        const step = ((MAX_RADIUS - MIN_RADIUS) * travelled) / span;
        state.radius = clamp(Math.round(state.sizeDrag.startRadius + step), MIN_RADIUS, MAX_RADIUS);
        repaint();
        return true;
      }

      if (state.stroke) {
        // A button released over another window, or a capture the browser took away, arrives
        // as a move with nothing held. The stroke is finished with what it has rather than
        // left following an unpressed pointer.
        if (!(event.buttons & 1)) {
          finishStroke(true);
          return true;
        }
        const target = state.strokes;
        const at = context.frame && target ? strokePoint(context.frame, target) : null;
        if (at) {
          drawSegment(state.stroke.last, at, state.stroke.erasing);
          state.stroke.last = at;
          repaint();
        }
        return true;
      }

      if (chipAt(context.point)) return true;
      if (state.mode === PAINT_MODE.RECT) return false;
      if (!context.view || !insideFrame(context.point, context.view)) return false;
      // The outline follows the pointer, so a repaint is asked for even though nothing was
      // drawn into the mask.
      repaint();
      return true;
    },

    /**
     * End a stroke or a size drag.
     *
     * @param {PointerEvent} event - The release.
     * @param {object} context - The context the region editor hands over.
     * @returns {boolean} True when the brush took the release.
     */
    pointerUp(event, context) {
      state.pointer = context.point;
      if (state.sizeDrag) {
        state.sizeDrag = null;
        repaint();
        return true;
      }
      if (state.stroke) {
        finishStroke(true);
        return true;
      }
      return false;
    },

    /**
     * Drop a gesture that was interrupted.
     *
     * @returns {void}
     */
    cancel() {
      state.sizeDrag = null;
      if (!state.stroke) return;
      state.stroke = null;
      restore();
      repaint();
    },

    /**
     * Stop drawing the brush outline once the pointer is somewhere else.
     *
     * @returns {void}
     */
    leave() {
      if (state.pointer === null) return;
      state.pointer = null;
      repaint();
    },
  };

  // The pointer surface for an adopter with no brush, which is the chip row and nothing else.
  // The region editor routes a press through the tool it was given, so a chip is only pressable
  // while one of these is handed to it.
  const chips = {
    /**
     * What the pointer looks like over the chip row.
     *
     * @param {object} context - The context the region editor hands over.
     * @returns {string} A CSS cursor over a chip, empty everywhere else.
     */
    cursor(context) {
      return chipAt(context.point) ? "pointer" : "";
    },

    /**
     * Take a press on a chip.
     *
     * @param {PointerEvent} event - The press.
     * @param {object} context - The context the region editor hands over.
     * @returns {boolean} True when the press was on a chip.
     */
    pointerDown(event, context) {
      if (event.button !== 0) return false;
      const chip = chipAt(context.point);
      if (!chip) return false;
      pressChip(chip, context);
      return true;
    },

    /**
     * Claim a move over a chip, so the cursor is asked for there.
     *
     * @param {PointerEvent} event - The move.
     * @param {object} context - The context the region editor hands over.
     * @returns {boolean} True while the pointer is over a chip.
     */
    pointerMove(event, context) {
      return Boolean(chipAt(context.point));
    },
  };

  /**
   * End the stroke in progress.
   *
   * @param {boolean} keep - Store what it drew. False puts the pixels back.
   * @returns {void}
   */
  function finishStroke(keep) {
    if (!state.stroke) return;
    state.stroke = null;
    if (keep) commitStrokes();
    else restore();
    repaint();
  }

  return {
    // Null rather than a tool that refuses everything, so an adopter offering no mode cannot
    // hand the region editor a brush by accident.
    tool: painting ? tool : null,
    chips,
    layer,

    /**
     * Give the drawing the editor it is drawn in, so its own writes are followed by a repaint.
     *
     * @param {object} editor - What `createRegionEditor` answered.
     * @param {() => void} [onChanged] - Called after every write, for an adopter whose backdrop
     *   is a picture of the drawing joined with whatever else the node computes.
     * @returns {void}
     */
    bind(editor, onChanged) {
      state.editor = editor;
      state.onChanged = typeof onChanged === "function" ? onChanged : null;
    },

    /**
     * What the pointer is doing over the frame.
     *
     * @returns {string} One of `PAINT_MODE`.
     */
    mode() {
      return state.mode;
    },

    /**
     * How wide the brush is.
     *
     * @returns {number} The radius in mask pixels.
     */
    radius() {
      return state.radius;
    },

    header,
    bytes,
    sample,
    invalidate,

    /**
     * Take on a drawing made elsewhere, and store it as one edit.
     *
     * @param {HTMLCanvasElement|null} canvas - Coverage in the alpha channel, at the size the
     *   drawing is to be held at. Null throws the drawing away.
     * @returns {void}
     */
    adopt(canvas) {
      if (state.disposed) return;
      if (!canvas) {
        clearDrawing();
        return;
      }
      // A gesture in flight owns the strokes canvas and would write over this on release.
      if (state.stroke) return;
      state.strokes = canvas;
      state.before = null;
      state.version += 1;
      commitStrokes();
    },

    /**
     * A number that changes whenever the drawing does.
     *
     * @returns {number} The drawing's version, for a cache keyed on it.
     */
    version() {
      return state.version;
    },

    /**
     * Release what the drawing is holding.
     *
     * @returns {void}
     */
    dispose() {
      state.disposed = true;
      state.stroke = null;
      state.sizeDrag = null;
      state.strokes = null;
      state.before = null;
      state.tint = null;
      state.reduced = null;
      state.editor = null;
      state.onChanged = null;
    },
  };
}
