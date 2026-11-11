/**
 * Region editor and brush for the two Mask Rect Area nodes.
 *
 * `Mask Rect Area` counts in percentages of a 512 by 512 mask and the advanced node in pixels
 * of its own. The brush counts in mask pixels.
 */

import { app } from "../../scripts/app.js";
import { blankBackdrop } from "./interface/backdrop.js";
import { combine, createMaskPaint } from "./interface/mask_paint.js";
import { maskStoreSize, maskValueBytes } from "./interface/mask_value.js";
import { EDGE, GESTURE, createRegionEditor } from "./interface/region.js";
import { readableBytes } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.MaskRectAreaUI";
const SETTING_ID = "WAS.MaskRectArea.ShowInterface";

const PERCENT_NODE = "Mask Rect Area";
const PIXEL_NODE = "Mask Rect Area (Advanced)";

const X_WIDGET = "x";
const Y_WIDGET = "y";
const WIDTH_WIDGET = "width";
const HEIGHT_WIDGET = "height";
const BLUR_WIDGET = "blur_radius";
const IMAGE_WIDTH_WIDGET = "image_width";
const IMAGE_HEIGHT_WIDGET = "image_height";
const DRAWN_MASK_WIDGET = "drawn_mask";
const DRAWN_COMBINE_WIDGET = "drawn_combine";

// What `drawn_combine` holds on a node whose widget cannot be read, from `DEFAULT_COMBINE` in
// `modules/mask/drawn.py`.
const DEFAULT_COMBINE = "union";

// The four the rectangle is made of. Both nodes name them alike, so one accessor pair serves
// both and only the unit they are counted in differs.
const RECT_WIDGETS = [X_WIDGET, Y_WIDGET, WIDTH_WIDGET, HEIGHT_WIDGET];

const UI_WIDGET_NAME = "was_mask_rect_area_ui";
const UI_WIDGET_TYPE = "was_mask_rect_area_region";

// Side of the canvas `Mask Rect Area` always draws on, from `RESOLUTION` in
// `nodes/mask/mask_rect_area.py`, and the span its percentages are counted over.
const RESOLUTION = 512;
const PERCENT_SPAN = 100;

// The percentages carry `step=1` and are whole percentages already, so nothing on that node is
// snapped beyond the whole number every gesture writes.
const PERCENT_STEP = 1;
const PERCENT_COARSE_STEP = 10;
const PERCENT_MIN_SIZE = 1;

// The step the advanced node's four numbers carry, which is the grid a gesture there lands on,
// and the smallest rectangle a gesture may leave, so a snapped rectangle never closes to
// nothing. `x`, `y`, `width` and `height` there carry `step=64`, inherited from the v2 node.
// Nothing on the node says so, and it is the difference between a rectangle that can be placed
// anywhere and one with nine reachable left edges on a 512 pixel canvas.
const PIXEL_GRID = 64;
const PIXEL_COARSE_STEP = PIXEL_GRID * 4;

// The widget bounds used only where a widget cannot be asked for its own, from the schemas in
// `nodes/mask/`.
const PERCENT_LIMITS = { min: 0, max: PERCENT_SPAN };
const PIXEL_LIMITS = { min: 0, max: 4096 };

// What the advanced node's frame falls back to while its size widgets cannot be read.
const DEFAULT_IMAGE_SIZE = 512;

// Longest edge of the drawn mask in picture pixels. The picture stands for the mask and is not
// the thing being measured, so it is capped rather than drawn at the mask's own size.
const PICTURE_MAX = 256;

/**
 * The kernels OpenCV answers with for the smallest widths, keyed by width.
 */
const SMALL_KERNELS = {
  // These are held to multiples of 1/256 and differ from the width's own Gaussian in the third
  // decimal, which is enough to move the drawn falloff at the smallest radii.
  3: [0.25, 0.5, 0.25],
  5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
  7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
  9: [
    0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875, 0.05078125,
    0.015625,
  ],
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
 * Read a widget's value as a number.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {number} fallback - What to answer when the widget is absent or holds no number.
 * @returns {number} The widget's value, or the fallback.
 */
function readValue(node, name, fallback) {
  const value = Number(findWidget(node, name)?.value);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Read a widget's own bounds, falling back to the schema's.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {{min: number, max: number}} fallback - The range to use when the widget states none.
 * @returns {{min: number, max: number}} The range a written value is held to.
 */
function readBounds(node, name, fallback) {
  const options = findWidget(node, name)?.options ?? {};
  return {
    min: Number.isFinite(options.min) ? options.min : fallback.min,
    max: Number.isFinite(options.max) ? options.max : fallback.max,
  };
}

/**
 * Read a widget's value as text.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {string} The widget's value, empty when it is absent or holds no string.
 */
function readText(node, name) {
  const value = findWidget(node, name)?.value;
  return typeof value === "string" ? value : "";
}

/**
 * Write a widget's value as text, once.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {string} value - Value to store.
 * @returns {void}
 */
function writeText(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget) return;
  if (widget.value === value) return;
  // A single line string widget is a plain widget whose value setter runs no callback, so the
  // repaint that follows is the caller's rather than something this write sets off.
  widget.value = value;
}

/**
 * What the node does with the painting.
 *
 * @param {object} node - Node holding the widget.
 * @returns {string} The combine, falling back to the schema default while the widget is absent.
 */
function readCombine(node) {
  const value = readText(node, DRAWN_COMBINE_WIDGET);
  return value || DEFAULT_COMBINE;
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
 * Build the kernel `cv2.GaussianBlur` uses for one radius.
 *
 * @param {number} radius - `blur_radius`, above zero.
 * @returns {Float64Array} The kernel, summing to one.
 */
function gaussianKernel(radius) {
  const size = radius * 2 + 1;
  const stored = SMALL_KERNELS[size];
  // The rule OpenCV applies to a sigma of zero, with the result normalised to sum to one below.
  const sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8;
  const scale = -0.5 / (sigma * sigma);
  const middle = (size - 1) * 0.5;

  const kernel = new Float64Array(size);
  let total = 0;
  for (let index = 0; index < size; index++) {
    const offset = index - middle;
    const weight = stored ? stored[index] : Math.exp(scale * offset * offset);
    kernel[index] = weight;
    total += weight;
  }
  for (let index = 0; index < size; index++) kernel[index] /= total;
  return kernel;
}

/**
 * Fold an index back inside a length the way OpenCV's default border does.
 *
 * @param {number} index - Sample index, inside the length or outside it.
 * @param {number} length - Number of samples along the axis.
 * @returns {number} An index inside the length.
 */
function reflect(index, length) {
  if (length === 1) return 0;
  let at = index;
  while (at < 0 || at >= length) {
    if (at < 0) at = -at;
    else at = length * 2 - at - 2;
  }
  return at;
}

/**
 * Blur one axis of the mask.
 *
 * @param {number} length - Number of samples along the axis.
 * @param {number} low - First sample the rectangle covers.
 * @param {number} high - First sample past the rectangle.
 * @param {Float64Array} kernel - Kernel from `gaussianKernel`.
 * @returns {Float64Array} The blurred box, one value per sample.
 */
function blurProfile(length, low, high, kernel) {
  const out = new Float64Array(Math.max(0, length));
  if (high <= low || length <= 0) return out;

  const size = kernel.length;
  const half = (size - 1) / 2;
  const running = new Float64Array(size + 1);
  for (let tap = 0; tap < size; tap++) running[tap + 1] = running[tap] + kernel[tap];

  for (let index = 0; index < length; index++) {
    const first = index - half;
    // Every sample the kernel covers is either one or zero, so where the kernel sits wholly
    // inside the axis the answer is a difference of two running sums of the kernel. Only the
    // ends, and an axis narrower than the kernel, need the reflected border walked tap by tap.
    if (first >= 0 && first + size <= length) {
      const lowTap = Math.max(0, low - first);
      const highTap = Math.min(size - 1, high - 1 - first);
      out[index] = highTap >= lowTap ? running[highTap + 1] - running[lowTap] : 0;
      continue;
    }
    let total = 0;
    for (let tap = 0; tap < size; tap++) {
      const at = reflect(first + tap, length);
      if (at >= low && at < high) total += kernel[tap];
    }
    out[index] = total;
  }
  return out;
}

/**
 * The largest value in a profile.
 *
 * @param {Float64Array} profile - Profile from `blurProfile`.
 * @returns {number} The largest value.
 */
function largest(profile) {
  let most = 0;
  for (let index = 0; index < profile.length; index++) {
    if (profile[index] > most) most = profile[index];
  }
  return most;
}

/**
 * The rectangle the node's widgets hold, in the unit the node counts them in.
 *
 * @param {object} node - Node holding the widgets.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle, exactly as held.
 */
function readRect(node) {
  return {
    x: readValue(node, X_WIDGET, 0),
    y: readValue(node, Y_WIDGET, 0),
    w: readValue(node, WIDTH_WIDGET, 0),
    h: readValue(node, HEIGHT_WIDGET, 0),
  };
}

/**
 * Read which edges cannot be written, and what to call them.
 *
 * @param {object} node - Node holding the widgets.
 * @param {string} gesture - What the gesture would do, one of `GESTURE`.
 * @returns {object} A map from edge to input name, holding only the edges that are locked.
 */
function readLocks(node, gesture) {
  const locks = {};
  const linked = {
    [X_WIDGET]: inputLinked(node, X_WIDGET),
    [Y_WIDGET]: inputLinked(node, Y_WIDGET),
    [WIDTH_WIDGET]: inputLinked(node, WIDTH_WIDGET),
    [HEIGHT_WIDGET]: inputLinked(node, HEIGHT_WIDGET),
  };

  // A translation carries the size through untouched and writes the origin alone, so a linked
  // `width` or `height` stops every resize and stops no move, and the one gesture that is safe
  // under that link is the one that would otherwise be refused.
  if (gesture === GESTURE.MOVE) {
    if (linked[X_WIDGET]) {
      locks[EDGE.LEFT] = X_WIDGET;
      locks[EDGE.RIGHT] = X_WIDGET;
    }
    if (linked[Y_WIDGET]) {
      locks[EDGE.TOP] = Y_WIDGET;
      locks[EDGE.BOTTOM] = Y_WIDGET;
    }
    return locks;
  }

  // Moving the left edge writes both `x` and `width`, since the right edge stays where it is;
  // moving the right edge writes `width` alone. The top and bottom edges stand in the same
  // relation to `y` and `height`.
  if (linked[X_WIDGET]) locks[EDGE.LEFT] = X_WIDGET;
  else if (linked[WIDTH_WIDGET]) locks[EDGE.LEFT] = WIDTH_WIDGET;
  if (linked[WIDTH_WIDGET]) locks[EDGE.RIGHT] = WIDTH_WIDGET;

  if (linked[Y_WIDGET]) locks[EDGE.TOP] = Y_WIDGET;
  else if (linked[HEIGHT_WIDGET]) locks[EDGE.TOP] = HEIGHT_WIDGET;
  if (linked[HEIGHT_WIDGET]) locks[EDGE.BOTTOM] = HEIGHT_WIDGET;

  return locks;
}

/**
 * Write one widget, once.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {number} value - Value to store.
 * @param {{min: number, max: number}} fallback - Bounds to use when the widget states none.
 * @returns {void}
 */
function writeValue(node, name, value, fallback) {
  const widget = findWidget(node, name);
  if (!widget) return;
  // The gestures refuse a linked input before they reach here. This catches the one that cannot:
  // a gesture held on the keyboard while the link is attached, since attaching one changes no
  // widget value.
  if (inputLinked(node, name)) return;
  if (!Number.isFinite(value)) return;

  // A value past the widget's own bounds reaches the backend as an input the node refuses to
  // run with.
  const bounds = readBounds(node, name, fallback);
  const next = Math.round(clamp(value, bounds.min, bounds.max));
  // Compared first, so a repaint driven by a widget's own callback can never write anything
  // back.
  if (next === widget.value) return;
  // The write is not bracketed here. The region editor brackets the whole of one gesture's write
  // in the canvas change events, which is what gives the gesture a single undo entry.
  widget.value = next;
}

/**
 * Work out what one axis of the rectangle is written as.
 *
 * @param {{low: boolean, high: boolean}} moved - Which of the axis's two edges moved.
 * @param {{low: number, size: number}} reached - Where the gesture left the axis.
 * @param {{low: number, size: number}} current - What the widgets hold.
 * @returns {{low: number|null, size: number|null}} What to write, with null for a widget this
 *   gesture is not about.
 */
function resolveAxis(moved, reached, current) {
  if (!moved.low && !moved.high) return { low: null, size: null };
  // Both edges moving the same distance is a move rather than a resize, so the size is carried
  // through untouched and only the origin is written. Writing both edges of a rectangle whose
  // size is not a whole number of steps would otherwise resize it for having been moved.
  if (moved.low && moved.high && reached.size === current.size) {
    return { low: Math.round(reached.low), size: null };
  }

  // A moved edge arrives already on the node's own step, since the editor is given that step and
  // lands every gesture on it. An edge the gesture did not move is carried through exactly as it
  // stood, which is what lets a number typed off the step survive a gesture on the other edge,
  // and what keeps the far edge of the rectangle where it was while the near one is dragged.
  const low = moved.low ? Math.round(reached.low) : current.low;
  const high = moved.high
    ? Math.round(reached.low + reached.size)
    : current.low + current.size;
  return { low: moved.low ? low : null, size: high - low };
}

/**
 * Write the rectangle a gesture reached.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} spec - The node's own spec.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle the gesture
 *   reached, in frame units.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @returns {void}
 */
function writeRect(node, spec, rect, moved) {
  const current = readRect(node);

  const across = resolveAxis(
    { low: Boolean(moved[EDGE.LEFT]), high: Boolean(moved[EDGE.RIGHT]) },
    { low: rect.x, size: rect.w },
    { low: current.x, size: current.w },
  );
  const down = resolveAxis(
    { low: Boolean(moved[EDGE.TOP]), high: Boolean(moved[EDGE.BOTTOM]) },
    { low: rect.y, size: rect.h },
    { low: current.y, size: current.h },
  );

  if (across.low !== null) writeValue(node, X_WIDGET, across.low, spec.limits);
  if (across.size !== null) writeValue(node, WIDTH_WIDGET, across.size, spec.limits);
  if (down.low !== null) writeValue(node, Y_WIDGET, down.low, spec.limits);
  if (down.size !== null) writeValue(node, HEIGHT_WIDGET, down.size, spec.limits);
}

/**
 * The size of the canvas the advanced node draws on, which is also its frame.
 *
 * @param {object} node - Node holding the widgets.
 * @returns {{width: number, height: number}} The mask's size in pixels.
 */
function imageSize(node) {
  return {
    width: Math.max(1, Math.trunc(readValue(node, IMAGE_WIDTH_WIDGET, DEFAULT_IMAGE_SIZE))),
    height: Math.max(1, Math.trunc(readValue(node, IMAGE_HEIGHT_WIDGET, DEFAULT_IMAGE_SIZE))),
  };
}

/**
 * Where `Mask Rect Area` cuts its rectangle, in pixels of the 512 canvas.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - The rectangle in percentages.
 * @returns {{x0: number, x1: number, y0: number, y1: number}} The pixels it fills.
 */
function percentCut(rect) {
  // The node turns a percentage into a pixel with `int(x / 100 * 512)`, which is repeated here
  // so the drawn mask cuts where the node cuts.
  const minX = rect.x / PERCENT_SPAN;
  const minY = rect.y / PERCENT_SPAN;
  const width = rect.w / PERCENT_SPAN;
  const height = rect.h / PERCENT_SPAN;
  return {
    x0: Math.trunc(minX * RESOLUTION),
    x1: Math.trunc((minX + width) * RESOLUTION),
    y0: Math.trunc(minY * RESOLUTION),
    y1: Math.trunc((minY + height) * RESOLUTION),
  };
}

/**
 * Where `Mask Rect Area (Advanced)` cuts its rectangle.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - The rectangle in pixels.
 * @returns {{x0: number, x1: number, y0: number, y1: number}} The pixels it fills.
 */
function pixelCut(rect) {
  return {
    x0: Math.trunc(rect.x),
    x1: Math.trunc(rect.x + rect.w),
    y0: Math.trunc(rect.y),
    y1: Math.trunc(rect.y + rect.h),
  };
}

/**
 * What the four numbers of `Mask Rect Area` are measured in, for the footer's hover text.
 *
 * @returns {string} The sentence.
 */
function percentUnits() {
  return `the four numbers are percentages of a ${RESOLUTION}x${RESOLUTION} mask`;
}

/**
 * What the four numbers of `Mask Rect Area (Advanced)` are measured in, for the hover text.
 *
 * @returns {string} The sentence.
 */
function pixelUnits() {
  return `the four numbers are pixels of the mask, and a gesture lands on ${PIXEL_GRID} of them`;
}

/**
 * What `Mask Rect Area (Advanced)` has to report about the mask it is drawn over.
 *
 * @param {object} node - Node holding the widgets.
 * @returns {string[]} The parts of the footer's second line.
 */
function pixelMeaning(node) {
  if (inputLinked(node, IMAGE_WIDTH_WIDGET) || inputLinked(node, IMAGE_HEIGHT_WIDGET)) {
    return ["mask size is linked"];
  }
  return [];
}

/**
 * What each node counts in, draws on and steps by.
 */
const SPECS = new Map([
  [
    PERCENT_NODE,
    {
      // The frame is 100 by 100 and one frame unit is one percent, which is what the widgets
      // hold and what the footer reads out. The mask underneath is square, so a percent across
      // and a percent down are the same number of pixels and the frame is drawn square as well.
      frame: () => ({ width: PERCENT_SPAN, height: PERCENT_SPAN }),
      canvas: () => ({ width: RESOLUTION, height: RESOLUTION }),
      limits: PERCENT_LIMITS,
      cut: percentCut,
      meaning: () => [],
      units: percentUnits,
      watch: [BLUR_WIDGET],
      grid: 0,
      step: PERCENT_STEP,
      coarseStep: PERCENT_COARSE_STEP,
      minSize: PERCENT_MIN_SIZE,
    },
  ],
  [
    PIXEL_NODE,
    {
      // The frame is `image_width` by `image_height` and one frame unit is one pixel, so the
      // mask and the frame are the same thing. Both widgets are read at every repaint, so
      // changing the mask size moves the rectangle with it rather than leaving it measured
      // against a canvas that no longer exists.
      frame: imageSize,
      canvas: imageSize,
      limits: PIXEL_LIMITS,
      cut: pixelCut,
      meaning: pixelMeaning,
      units: pixelUnits,
      watch: [BLUR_WIDGET, IMAGE_WIDTH_WIDGET, IMAGE_HEIGHT_WIDGET],
      // An edge is snapped to the nearest multiple of the grid as it is dragged rather than as
      // it is written, so the rectangle drawn and the numbers read out are the ones stored, and
      // an arrow press moves by one whole step. The mask's own edge and the smallest rectangle a
      // gesture may leave are held to after that grid, so an edge stopped by either lands where
      // it was stopped.
      grid: PIXEL_GRID,
      step: PIXEL_GRID,
      coarseStep: PIXEL_COARSE_STEP,
      minSize: PIXEL_GRID,
    },
  ],
]);

/**
 * Measure the mask the node's widgets describe, keeping the last answer.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} spec - The node's own spec.
 * @param {object} store - Where the last answer is kept.
 * @returns {{canvas: object, radius: number, peak: number, row: Float64Array|null,
 *   col: Float64Array|null}} The mask's size, the blur radius, the largest value anywhere in the
 *   mask, and the two profiles it is the product of, which are absent where there is no blur to
 *   draw.
 */
function measureMask(node, spec, store) {
  const canvas = spec.canvas(node);
  const cut = spec.cut(readRect(node));
  const bounds = {
    x0: clamp(cut.x0, 0, canvas.width),
    x1: clamp(cut.x1, 0, canvas.width),
    y0: clamp(cut.y0, 0, canvas.height),
    y1: clamp(cut.y1, 0, canvas.height),
  };
  const radius = Math.max(0, Math.trunc(readValue(node, BLUR_WIDGET, 0)));

  const key = [
    canvas.width,
    canvas.height,
    bounds.x0,
    bounds.x1,
    bounds.y0,
    bounds.y1,
    radius,
  ].join(",");
  // The answer is asked for on every repaint, for the footer, and again whenever the picture is
  // redrawn, so it is kept until one of the numbers behind it changes.
  if (store.key === key && store.answer) return store.answer;

  const empty = bounds.x1 <= bounds.x0 || bounds.y1 <= bounds.y0;
  let answer;
  if (empty) {
    answer = { canvas, radius, peak: 0, row: null, col: null };
  } else if (radius <= 0) {
    // At a `blur_radius` of zero there is no falloff to draw and the mask is exactly the
    // rectangle the editor already draws, so no profile is built and no picture is drawn.
    answer = { canvas, radius, peak: 1, row: null, col: null };
  } else {
    // `cv2.GaussianBlur` is separable and the unblurred mask is the outer product of two boxes,
    // so the whole mask is the product of two one dimensional profiles, each the node's own
    // kernel run along one axis with OpenCV's reflected border.
    const kernel = gaussianKernel(radius);
    const row = blurProfile(canvas.width, bounds.x0, bounds.x1, kernel);
    const col = blurProfile(canvas.height, bounds.y0, bounds.y1, kernel);
    // Both profiles are positive, so the largest value in their product is the product of the
    // two largest.
    answer = { canvas, radius, peak: largest(row) * largest(col), row, col };
  }

  store.key = key;
  store.answer = answer;
  return answer;
}

/**
 * The mask sample one picture pixel stands for.
 *
 * @param {number} index - Picture pixel along one axis.
 * @param {number} size - Picture size along that axis.
 * @param {number} length - Mask size along that axis.
 * @returns {number} The sample at the middle of that picture pixel.
 */
function sampleAt(index, size, length) {
  return clamp(Math.floor(((index + 0.5) * length) / size), 0, length - 1);
}

/**
 * Draw the mask the node produces, reduced to fit inside the node.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} spec - The node's own spec.
 * @param {object} store - Where the last answer and the picture's canvas are kept.
 * @param {object|null} paint - The brush, asked for the painting reduced to the picture's size.
 * @returns {{image: HTMLCanvasElement, width: number}|null} The picture and its width in pixels,
 *   or null while there is neither a falloff nor a painting to draw.
 */
function drawMask(node, spec, store, paint) {
  const mask = measureMask(node, spec, store);

  const canvas = spec.canvas(node);
  const fit = Math.min(1, PICTURE_MAX / Math.max(canvas.width, canvas.height));
  const width = Math.max(1, Math.round(canvas.width * fit));
  const height = Math.max(1, Math.round(canvas.height * fit));

  const mode = readCombine(node);
  // `off` is what the run reads, so the picture leaves the painting out of the mask as well and
  // the layer over it is what says the painting is still there.
  const drawn = mode === "off" ? null : (paint?.sample(width, height) ?? null);
  // At a `blur_radius` of zero with nothing painted there is no falloff to draw and the mask is
  // exactly the rectangle the editor already draws, so no picture is made.
  if (!drawn && (!mask.row || !mask.col)) return null;

  if (!store.picture) store.picture = document.createElement("canvas");
  const picture = store.picture;
  picture.width = width;
  picture.height = height;

  const ctx = picture.getContext("2d");
  if (!ctx) return null;
  const image = ctx.createImageData(width, height);
  const pixels = image.data;

  const across = new Float64Array(width);
  if (mask.row) {
    for (let x = 0; x < width; x++) across[x] = mask.row[sampleAt(x, width, canvas.width)];
  } else {
    // No blur, so the rectangle is the hard cut the node makes and the profile is that box.
    const cut = spec.cut(readRect(node));
    for (let x = 0; x < width; x++) {
      const sample = sampleAt(x, width, canvas.width);
      across[x] = sample >= cut.x0 && sample < cut.x1 ? 1 : 0;
    }
  }

  const cut = mask.col ? null : spec.cut(readRect(node));
  for (let y = 0; y < height; y++) {
    let down;
    if (mask.col) {
      down = mask.col[sampleAt(y, height, canvas.height)];
    } else {
      const sample = sampleAt(y, height, canvas.height);
      down = sample >= cut.y0 && sample < cut.y1 ? 1 : 0;
    }
    for (let x = 0; x < width; x++) {
      const computed = clamp(down * across[x], 0, 1);
      const at = y * width + x;
      const level = Math.round(clamp(drawn ? combine(computed, drawn[at], mode) : computed, 0, 1) * 255);
      const pixel = at * 4;
      pixels[pixel] = level;
      pixels[pixel + 1] = level;
      pixels[pixel + 2] = level;
      pixels[pixel + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  return { image: picture, width };
}

/**
 * The backdrop the rectangle is drawn over.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} spec - The node's own spec.
 * @param {object} store - Where the last answer and the picture's canvas are kept.
 * @param {object|null} paint - The brush, whose painting is joined into the picture.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function maskBackdrop(node, spec, store, paint) {
  return {
    async load() {
      // Neither node has an image input and both masks are a pure function of the widgets, so
      // the frame can always be stated and there is never an image to wait for.
      const frame = spec.frame(node);
      const answer = await blankBackdrop({ width: frame.width, height: frame.height }).load();
      // `blur_radius` runs a Gaussian over the finished mask, and a Gaussian over a rectangle
      // does not give back a rectangle. The edge becomes a ramp, the corners pull in, and as
      // soon as the blur is wide enough to reach across the rectangle the middle stops being
      // white: a 64 pixel band on the 512 canvas peaks at 0.59 with `blur_radius` at 128 and at
      // 0.32 at the widget's maximum of 255, and a 16 pixel square on a 64 pixel canvas peaks at
      // 0.06. A hard white rectangle would be the wrong picture in every one of those cases.
      const picture = drawMask(node, spec, store, paint);
      if (!picture) return answer;
      // The picture stands in for the mask at whatever size fits, so what it reports is how many
      // frame units one of its own pixels covers, which is what a gesture on it converts through.
      return {
        ...answer,
        image: picture.image,
        scale: frame.width / picture.width,
      };
    },
  };
}

/**
 * What the rectangle is doing right now, on the footer's second line.
 *
 * @param {object} node - Node holding the widgets.
 * @param {object} spec - The node's own spec.
 * @param {object} store - Where the last answer is kept.
 * @param {object|null} paint - The brush, asked what it is holding.
 * @returns {string} The line to draw.
 */
function footerMeaning(node, spec, store, paint) {
  const parts = spec.meaning(node);
  const mask = measureMask(node, spec, store);
  if (mask.radius > 0 && mask.peak > 0) parts.push(`blur peaks at ${mask.peak.toFixed(2)}`);

  const drawn = paint?.header();
  if (drawn) {
    const mode = readCombine(node);
    const size = maskStoreSize(mask.canvas.width, mask.canvas.height);
    const bytes = readableBytes(maskValueBytes(readText(node, DRAWN_MASK_WIDGET)));
    // The size the painting was made at is only worth the room when it is not the size the mask
    // is now, since that is the case where the run resizes it and a stroke has moved.
    parts.push(
      drawn.width === size.width && drawn.height === size.height
        ? `${mode} painting, ${bytes}`
        : `${mode} painting made at ${drawn.width}x${drawn.height}, resized to the mask, ${bytes}`,
    );
  }
  return parts.join(", ");
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
 * @param {object} spec - The node's own spec.
 * @returns {void}
 */
function attachRegionEditor(node, spec) {
  for (const name of RECT_WIDGETS) {
    if (!findWidget(node, name)) return;
  }

  const store = { key: "", answer: null, picture: null, refreshHandle: 0, disposed: false };

  // A frontend newer than the python beside it reaches a node with no painting widget, and the
  // rectangle is the whole of what that node can do.
  const paint = findWidget(node, DRAWN_MASK_WIDGET)
    ? createMaskPaint({
        frame: () => spec.frame(node),
        canvas: () => spec.canvas(node),
        value: {
          read: () => readText(node, DRAWN_MASK_WIDGET),
          write: (value) => writeText(node, DRAWN_MASK_WIDGET, value),
        },
        combine: () => readCombine(node),
      })
    : null;

  const editor = createRegionEditor({
    node,
    backdrop: maskBackdrop(node, spec, store, paint),
    rect: {
      read: () => readRect(node),
      write: (rect, moved) => writeRect(node, spec, rect, moved),
      locks: (gesture) => readLocks(node, gesture),
    },
    footer: () => footerMeaning(node, spec, store, paint),
    hover: spec.units,
    step: spec.step,
    coarseStep: spec.coarseStep,
    gridStep: spec.grid,
    minSize: spec.minSize,
    layers: paint ? [paint.layer] : [],
    tool: paint?.tool ?? null,
  });

  // The brush repaints through the editor, and the editor is not built until the brush exists,
  // so the two are joined here rather than at either construction. A stored stroke also changes
  // the mask the backdrop draws, which is asked for rather than pushed.
  paint?.bind(editor, () => scheduleBackdrop());

  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  /**
   * Draw the mask again, once, on the next frame.
   *
   * @returns {void}
   */
  function scheduleBackdrop() {
    if (store.disposed || store.refreshHandle) return;
    store.refreshHandle = requestAnimationFrame(() => {
      store.refreshHandle = 0;
      if (store.disposed) return;
      try {
        editor.refresh();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the mask again:`, error);
      }
    });
  }

  // The picture shows the mask the node makes as the widgets stand. A gesture in progress moves
  // the rectangle over that picture and the picture catches up when the gesture is written,
  // since a gesture writes once and writes at its end.
  for (const name of RECT_WIDGETS) {
    chainWidgetCallback(node, name, () => {
      editor.handleRectChanged();
      scheduleBackdrop();
    });
  }
  for (const name of spec.watch) {
    chainWidgetCallback(node, name, () => {
      editor.schedulePaint();
      scheduleBackdrop();
    });
  }

  // The painting is joined into the picture, so a hand edit of either widget redraws it. The
  // brush's own writes do not reach these: a single line string widget runs no callback when its
  // value is assigned, so the brush asks for its own repaint.
  if (paint) {
    for (const name of [DRAWN_MASK_WIDGET, DRAWN_COMBINE_WIDGET]) {
      chainWidgetCallback(node, name, () => {
        paint.invalidate();
        editor.schedulePaint();
        scheduleBackdrop();
      });
    }
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      // A workflow load and an undo both arrive here, and both can replace the painting with a
      // different one, so the decoded copy is dropped rather than drawn over the new value.
      paint?.invalidate();
      editor.schedulePaint();
      scheduleBackdrop();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a workflow load:`, error);
    }
    return result;
  };

  // Attaching a link changes no widget value, so nothing else here would hear about it. The
  // gestures read the links as they are pressed and are refused either way, but the footer and
  // the edges drawn for a linked input would otherwise say the old thing until something else
  // asked for a repaint.
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
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      store.disposed = true;
      if (store.refreshHandle) cancelAnimationFrame(store.refreshHandle);
      store.refreshHandle = 0;
      editor.dispose();
      paint?.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the region editor:`, error);
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
      category: ["WAS Node Suite", "Mask Rect Area", "Region editor"],
      name: "Show the region editor",
      tooltip:
        "Draw the rectangle of Mask Rect Area and Mask Rect Area (Advanced) under their " +
        "widgets, over the mask they make. Drag the rectangle or its handles to place it, and " +
        "use the arrow keys to nudge it. The chip in the top left switches to a brush, which " +
        "paints into the drawn_mask widget and joins the rectangle in whichever way " +
        "drawn_combine names. The widgets themselves are always available. This applies to " +
        "nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    const spec = SPECS.get(nodeData?.name);
    if (!spec) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise wrap
    // the prototype a second time and append a second editor.
    if (proto.__was_mask_rect_area_wrapped) return;
    proto.__was_mask_rect_area_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachRegionEditor(this, spec);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the region editor:`, error);
      }
      return result;
    };
  },
});
