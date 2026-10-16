/**
 * Region editor for the two crop by location nodes.
 *
 * Draws the rectangle `Image Crop Location` and `Image Crop Square Location` cut, over the
 * image the node received. Every number is in pixels of that image.
 */

import { app } from "../../scripts/app.js";
import { imageBackdrop } from "./interface/backdrop.js";
import { EDGE, GESTURE, createRegionEditor } from "./interface/region.js";
import { onNodeDone } from "./interface/run_events.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.CropLocationUI";
const SETTING_ID = "WAS.CropLocation.ShowInterface";

const UI_WIDGET_NAME = "was_crop_location_ui";
const UI_WIDGET_TYPE = "was_crop_location_region";

const CROP_NODE = "Image Crop Location";
const SQUARE_NODE = "Image Crop Square Location";

const TOP = "top";
const LEFT = "left";
const RIGHT = "right";
const BOTTOM = "bottom";
const X = "x";
const Y = "y";
const SIZE = "size";
const DIVISIBLE_BY = "divisible_by";

const CROP_WIDGETS = [TOP, LEFT, RIGHT, BOTTOM];
const SQUARE_WIDGETS = [X, Y, SIZE];

// Which widget of `Image Crop Location` each edge of the rectangle stands for. The two sets of
// names happen to agree, and the map is written out anyway so the editor's edges stay the
// editor's and a rename on either side is one edit rather than a coincidence to notice.
const CROP_WIDGET = {
  [EDGE.LEFT]: LEFT,
  [EDGE.TOP]: TOP,
  [EDGE.RIGHT]: RIGHT,
  [EDGE.BOTTOM]: BOTTOM,
};

const EDGES = [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT, EDGE.BOTTOM];

// The schema's own defaults, read only when a widget cannot be.
const DEFAULTS = {
  [TOP]: 0,
  [LEFT]: 0,
  [RIGHT]: 256,
  [BOTTOM]: 256,
  [X]: 0,
  [Y]: 0,
  [SIZE]: 256,
  [DIVISIBLE_BY]: 8,
};

// Height of the appended widget in node units. The picture is drawn inside it, so it is taller
// than an interface that only draws a plot.
const UI_HEIGHT = 220;

// The step to fall back on wherever the node's own `divisible_by` cannot be read, which is the
// value both schemas declare for it.
const DEFAULT_OUTPUT_STEP = 8;

// The smallest side a gesture may produce on each node, which is what each schema already
// allows: `Image Crop Location` refuses a rectangle with no area, and the square's own side
// has a minimum of five.
const CROP_MIN_SIZE = 1;
const SQUARE_MIN_SIZE = 5;

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
 * Test whether two rectangles hold the same four numbers.
 *
 * @param {object|null} left - A rectangle, or null.
 * @param {object|null} right - A rectangle, or null.
 * @returns {boolean} True when both are rectangles holding equal numbers.
 */
function sameRect(left, right) {
  if (!left || !right) return false;
  return left.x === right.x && left.y === right.y && left.w === right.w && left.h === right.h;
}

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
 * Read one widget as a number.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {number} The value the widget holds, or the schema's default for it.
 */
function widgetNumber(node, name) {
  const value = Number(findWidget(node, name)?.value);
  return Number.isFinite(value) ? value : DEFAULTS[name];
}

/**
 * Read the range one widget declares.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @returns {{min: number, max: number}} The bounds the schema declared, wide open where they
 *   cannot be read.
 */
function widgetLimits(node, name) {
  const options = findWidget(node, name)?.options ?? {};
  const low = Number(options.min);
  const high = Number(options.max);
  return {
    min: Number.isFinite(low) ? low : Number.MIN_SAFE_INTEGER,
    max: Number.isFinite(high) ? high : Number.MAX_SAFE_INTEGER,
  };
}

/**
 * Store one number in one widget.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {number} value - Value to store, rounded to a whole pixel and held to the range the
 *   schema declared.
 * @returns {void}
 */
function writeValue(node, name, value) {
  const widget = findWidget(node, name);
  if (!widget) return;
  // A widget whose input is linked is never written. The gestures refuse it before they reach
  // here, and this catches the one that cannot: attaching a link changes no widget value, so it
  // drops no gesture already in hand.
  if (inputLinked(node, name)) return;
  if (!Number.isFinite(value)) return;

  const limits = widgetLimits(node, name);
  const next = clamp(Math.round(value), limits.min, limits.max);
  // A widget already holding the value is left alone, so a gesture that ends where it started
  // marks nothing modified.
  if (next === widget.value) return;
  // The write is not bracketed here. The region editor brackets the whole of one gesture in the
  // canvas change events the graph's change tracker snapshots on, so a gesture that moves three
  // widgets is one undo entry rather than three.
  widget.value = next;
}

/**
 * The step both sides of the crop are put on, read from the node's own widget.
 *
 * @param {object} node - The node the editor is drawn on.
 * @returns {number|null} The step, or null while `divisible_by` is filled by a link, since the
 *   value the run reads is then not on the node at all.
 */
function outputStep(node) {
  if (inputLinked(node, DIVISIBLE_BY)) return null;
  const step = Math.round(widgetNumber(node, DIVISIBLE_BY));
  return Number.isFinite(step) && step >= 1 ? step : DEFAULT_OUTPUT_STEP;
}

/**
 * The side one axis of the output comes back at.
 *
 * @param {number} side - Width or height of the rectangle that was cut, in image pixels.
 * @param {number} step - The step both sides of the crop are put on.
 * @returns {number} The side of the image the node returns.
 */
function outputSide(side, step) {
  // Neither node need output the rectangle at the size it was cut. Both round each axis down to
  // a multiple of their own `divisible_by` widget and resize the crop to that rather than
  // trimming it further, floored at one whole step: at a step of 8 a crop 5 pixels wide comes
  // back 8 wide. A step of 1 rounds nothing and resizes nothing, so every side comes back at the
  // size it was cut.
  return Math.max(step, Math.floor(side / step) * step);
}

/**
 * The two sizes the node hands on, for the footer's second line.
 *
 * @param {{left: number, top: number, right: number, bottom: number}|null} box - The window the
 *   node cuts, or null while the image size is unknown.
 * @param {number|null} step - The step both sides are put on, or null while it arrives on a link.
 * @returns {string} The footer's second line.
 */
function footerMeaning(box, step) {
  // The output need not be the rectangle at the size it was cut, and `crop_data` always is,
  // which is the one thing about these nodes that neither the widgets nor the picture shows:
  // `crop_data` records the rectangle before the resize, so a paste node puts the crop back
  // where and how large it was. The clamped size is given as well, since the readout above this
  // line carries the widgets' own numbers and those can sit outside the image. Both move with
  // every gesture, so both are drawn and the rule that produces them is on hover.

  // A linked step is a value the run reads and this interface cannot, which is a state to act on
  // rather than a size to read, so it is named wherever it holds.
  if (!box) return step === null ? "divisible_by is linked" : "";

  const width = box.right - box.left;
  const height = box.bottom - box.top;
  if (!(width > 0) || !(height > 0)) return "no crop: the rectangle encloses nothing";

  if (step === null) return `crop_data ${width}x${height}, divisible_by is linked`;
  if (step === 1) return `output ${width}x${height}`;

  return (
    `output ${outputSide(width, step)}x${outputSide(height, step)}, ` +
    `crop_data ${width}x${height}`
  );
}

/**
 * What the two sizes are, and what puts them apart, for the footer's hover text.
 *
 * @param {number|null} step - The step both sides are put on, or null while it arrives on a link.
 * @returns {string} The sentence.
 */
function footerHover(step) {
  const base =
    "the numbers are pixels of the image, and both edges are held to it before anything is cut."
    + " crop_data is the window as it was cut, for Image Paste Crop by Location to put back";
  if (step === null) return `${base}. The output is resized to a multiple of divisible_by`;
  if (step === 1) return `${base}. At a divisible_by of 1 the output is that window unresized`;
  return `${base}. The output is resized to a multiple of ${step}, rounded down, never below one`
    + " whole step";
}

/**
 * The rectangle `Image Crop Location`'s widgets hold.
 *
 * @param {object} node - The node the editor is drawn on.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle in image pixels.
 */
function cropRect(node) {
  // The numbers are answered exactly as they stand, including one past the image. The node holds
  // `top`, `left`, `right` and `bottom` to the image, so an edge past it is trimmed rather than
  // padded, and the editor draws such an edge on the image's own edge in the colour that says so,
  // which is where the node cuts it. The widget keeps its number until a gesture moves it.
  const left = widgetNumber(node, LEFT);
  const top = widgetNumber(node, TOP);
  return {
    x: left,
    y: top,
    w: widgetNumber(node, RIGHT) - left,
    h: widgetNumber(node, BOTTOM) - top,
  };
}

/**
 * The window a rectangle cuts, which is its four edges held to the image.
 *
 * @param {{width: number, height: number}} frame - The image the node received.
 * @param {{x: number, y: number, w: number, h: number}|null} rect - The rectangle on screen, in
 *   image pixels.
 * @returns {{left: number, top: number, right: number, bottom: number}|null} The window, or
 *   null while the image size or the rectangle is unknown.
 */
function cutWindow(frame, rect) {
  if (!(frame.width > 0) || !(frame.height > 0) || !rect) return null;
  // Both nodes trim rather than pad at the image's own edge, so holding the four edges to the
  // image gives the window either of them cuts.
  return {
    left: Math.max(rect.x, 0),
    top: Math.max(rect.y, 0),
    right: Math.min(rect.x + rect.w, frame.width),
    bottom: Math.min(rect.y + rect.h, frame.height),
  };
}

/**
 * Write a gesture on `Image Crop Location`.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle the gesture
 *   ended on, in image pixels.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @returns {void}
 */
function writeCrop(node, rect, moved) {
  const edges = {
    [EDGE.LEFT]: rect.x,
    [EDGE.TOP]: rect.y,
    [EDGE.RIGHT]: rect.x + rect.w,
    [EDGE.BOTTOM]: rect.y + rect.h,
  };
  for (const edge of EDGES) {
    if (moved[edge]) writeValue(node, CROP_WIDGET[edge], edges[edge]);
  }
}

/**
 * Which edges of `Image Crop Location` a link has taken over.
 *
 * @param {object} node - The node the editor is drawn on.
 * @returns {object} A map from edge to input name, holding only the edges that cannot move.
 */
function cropLocks(node) {
  const locks = {};
  for (const edge of EDGES) {
    const name = CROP_WIDGET[edge];
    if (inputLinked(node, name)) locks[edge] = name;
  }
  return locks;
}

/**
 * The window `Image Crop Square Location` cuts for one centre and one side.
 *
 * This is `crop_square_location` in `nodes/image/process/image_crop_square_location.py` step
 * for step.
 *
 * @param {number} x - Horizontal centre, in image pixels.
 * @param {number} y - Vertical centre, in image pixels.
 * @param {number} size - Length of each side before any trimming, in image pixels.
 * @param {number} imageWidth - Width of the image, or `Infinity` while it is unknown, which
 *   gives the square the node would cut out of an image large enough to hold it.
 * @param {number} imageHeight - Height of the image, on the same terms.
 * @returns {{left: number, top: number, right: number, bottom: number}} The window.
 */
function squareWindowFor(x, y, size, imageWidth, imageHeight) {
  // The square is laid around the centre with a half side of `size // 2`. An odd side goes down
  // the same path as an even one, since a half side of `size // 2` doubled is one pixel short of
  // an odd `size`, so the window comes out narrow and is pushed back below.
  const half = Math.floor(size / 2);
  let left = Math.max(x - half, 0);
  let top = Math.max(y - half, 0);
  let right = Math.min(x + half, imageWidth);
  let bottom = Math.min(y + half, imageHeight);

  // A window narrower than it was asked for is pushed back the way it came rather than left
  // hanging over an edge: right first, and left only where the right has already reached the
  // image. That is what slides a square asked for at a corner into that corner rather than
  // leaving it half the size, and it trims to what fits where the image is too small to hold one.
  if (right - left < size) {
    if (right < imageWidth) right = Math.min(right + size - (right - left), imageWidth);
    else if (left > 0) left = Math.max(left - (size - (right - left)), 0);
  }
  if (bottom - top < size) {
    if (bottom < imageHeight) bottom = Math.min(bottom + size - (bottom - top), imageHeight);
    else if (top > 0) top = Math.max(top - (size - (bottom - top)), 0);
  }

  return { left, top, right, bottom };
}

/**
 * The rectangle `Image Crop Square Location` cuts.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{width: number, height: number}} frame - The image the node received.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle in image pixels.
 */
function squareRect(node, frame) {
  // The widgets hold a centre in `x` and `y` and one `size` rather than a rectangle, so the
  // rectangle is worked out the way the node works it out and the square is drawn where it
  // lands: the readout is the window that is cut and not the centre the widgets carry. While the
  // image size is unknown the square is laid out as it would fall in an image large enough to
  // hold it, which is the readout the widgets describe and the only one available before the
  // node has run.
  const box = squareWindowFor(
    widgetNumber(node, X),
    widgetNumber(node, Y),
    widgetNumber(node, SIZE),
    frame.width > 0 ? frame.width : Infinity,
    frame.height > 0 ? frame.height : Infinity,
  );
  return {
    x: box.left,
    y: box.top,
    w: box.right - box.left,
    h: box.bottom - box.top,
  };
}

/**
 * The centre and the side `Image Crop Square Location` stores for a rectangle a gesture reached.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle the gesture
 *   reached, in image pixels.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @param {{width: number, height: number}} frame - The image the node received.
 * @returns {{x: number, y: number, size: number}} The three numbers to store.
 */
function squarePlan(node, rect, moved, frame) {
  const current = squareRect(node, frame);
  const resizedAcross = rect.w !== current.w;
  const resizedDown = rect.h !== current.h;

  // The node holds one side, so a rectangle is reduced to a centre and a side before it can be
  // stored. A gesture on the square therefore writes a centre and a side rather than an edge,
  // and dragging one edge moves the opposite one with it. The side is the axis the gesture
  // resized, or the longer of the two where a corner resized both, so a corner drag gives the
  // square that covers the rectangle drawn. A gesture that moved the rectangle without resizing
  // it keeps the side it had.
  let side = widgetNumber(node, SIZE);
  if (resizedAcross && resizedDown) side = Math.max(rect.w, rect.h);
  else if (resizedAcross) side = rect.w;
  else if (resizedDown) side = rect.h;
  // All three numbers are held to the range their own widget declares before any of the others
  // is worked out from them. The side stops at 4096 while the frame a gesture is clamped to is
  // the image, which is routinely larger, and a centre placed from a side the node will never
  // see describes a square nobody dragged.
  const sides = widgetLimits(node, SIZE);
  side = clamp(Math.round(side), sides.min, sides.max);

  // Each axis is then placed on its own. An axis the gesture resized is anchored on the edge the
  // gesture did not move, so the corner nobody dragged stays where it was. An axis the gesture
  // did not resize keeps the centre it had: the gesture chose no edge on it and a square growing
  // on one axis grows on both. That is what leaves `y` alone for a gesture
  // across the image and `x` alone for one down it, which is also what the locks promise, since
  // an edge is locked by the centre of its own axis and never by the other one.
  const anchorRight = Boolean(moved[EDGE.LEFT]) && !moved[EDGE.RIGHT];
  const anchorBottom = Boolean(moved[EDGE.TOP]) && !moved[EDGE.BOTTOM];
  const left = resizedAcross
    ? anchorRight
      ? rect.x + rect.w - side
      : rect.x
    : rect.x + rect.w / 2 - side / 2;
  const top = resizedDown
    ? anchorBottom
      ? rect.y + rect.h - side
      : rect.y
    : rect.y + rect.h / 2 - side / 2;
  // The centre is worked back through the node's own half side rather than from the middle of
  // the rectangle, so the window this describes is the window the node then cuts, to the pixel,
  // for an odd side as well as an even one.
  const half = Math.floor(side / 2);
  const across = widgetLimits(node, X);
  const down = widgetLimits(node, Y);

  return {
    x: clamp(Math.round(left + half), across.min, across.max),
    y: clamp(Math.round(top + half), down.min, down.max),
    size: side,
  };
}

/**
 * The square `Image Crop Square Location` will cut for a rectangle a gesture reached.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle the gesture
 *   reached, in image pixels.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @param {{width: number, height: number}} frame - The image the node received.
 * @param {object} plan - Where the three numbers behind the answer are kept.
 * @returns {{x: number, y: number, w: number, h: number}} The window the node would cut.
 */
function squareShape(node, rect, moved, frame, plan) {
  // A gesture can reach a rectangle of any shape and this node holds a square, so the two part
  // company for the whole of a drag unless the shape the node will store is what is drawn. The
  // square that will be stored is therefore drawn while the gesture is still in hand, rather
  // than the rectangle the pointer traced, so the rubber band, the footer and the crop are one
  // shape.
  const next = squarePlan(node, rect, moved, frame);
  const box = squareWindowFor(
    next.x,
    next.y,
    next.size,
    frame.width > 0 ? frame.width : Infinity,
    frame.height > 0 ? frame.height : Infinity,
  );
  const shown = {
    x: box.left,
    y: box.top,
    w: box.right - box.left,
    h: box.bottom - box.top,
  };
  // The crop plan is kept beside the answer. The window given back is the one the node cuts,
  // slid and trimmed to the image, and the centre behind it cannot be read off a slid window.
  plan.rect = shown;
  plan.x = next.x;
  plan.y = next.y;
  plan.size = next.size;
  return shown;
}

/**
 * Write a gesture on `Image Crop Square Location`.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle the gesture
 *   ended on, in image pixels.
 * @param {object} moved - A map from edge to true, holding only the edges that moved.
 * @param {{width: number, height: number}} frame - The image the node received.
 * @param {object} plan - Where the three numbers behind the drawn window are kept.
 * @returns {void}
 */
function writeSquare(node, rect, moved, frame, plan) {
  const next = sameRect(plan.rect, rect) ? plan : squarePlan(node, rect, moved, frame);

  writeValue(node, X, next.x);
  writeValue(node, Y, next.y);
  writeValue(node, SIZE, next.size);
}

/**
 * Which edges of `Image Crop Square Location` a link has taken over.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {string} gesture - What the gesture would do, one of `GESTURE`.
 * @returns {object} A map from edge to input name, holding only the edges that cannot move.
 */
function squareLocks(node, gesture) {
  const locks = {};
  // The node's numbers are not its edges, so a link is read through what a gesture on an edge
  // would have to write. Moving either upright edge writes the horizontal centre, moving either
  // flat edge writes the vertical one, and any gesture that resizes the square writes the side.
  // A move is asked about separately: it carries the square whole and writes the two
  // centres alone, so a linked side stops every resize and stops no move.
  const sizeLinked = gesture !== GESTURE.MOVE && inputLinked(node, SIZE);
  const across = inputLinked(node, X) ? X : sizeLinked ? SIZE : "";
  const down = inputLinked(node, Y) ? Y : sizeLinked ? SIZE : "";

  if (across) {
    locks[EDGE.LEFT] = across;
    locks[EDGE.RIGHT] = across;
  }
  if (down) {
    locks[EDGE.TOP] = down;
    locks[EDGE.BOTTOM] = down;
  }
  return locks;
}

/**
 * What each node holds, reads and writes.
 */
const SPECS = {
  // The editor itself learns no widget name: a node spelling its rectangle as four edges and a
  // node spelling it as a centre and a side reach it through the same `read`, `write` and
  // `locks`.
  [CROP_NODE]: {
    widgets: CROP_WIDGETS,
    minSize: CROP_MIN_SIZE,
    read: (node) => cropRect(node),
    write: (node, rect, moved) => writeCrop(node, rect, moved),
    locks: (node) => cropLocks(node),
    // Four edges spell every rectangle a gesture can reach, so there is nothing to correct.
    shape: null,
  },
  [SQUARE_NODE]: {
    widgets: SQUARE_WIDGETS,
    minSize: SQUARE_MIN_SIZE,
    read: (node, frame) => squareRect(node, frame),
    write: (node, rect, moved, frame, plan) => writeSquare(node, rect, moved, frame, plan),
    locks: (node, gesture) => squareLocks(node, gesture),
    // `shape` is stated only by the node whose numbers cannot spell every rectangle a gesture
    // can reach, and it answers the one the node would store, so the square is drawn as a square
    // for the whole of the drag rather than at the release.
    shape: (node, rect, moved, frame, plan) => squareShape(node, rect, moved, frame, plan),
  },
};

/**
 * The published picture, with the size of the image it was reduced from kept beside it.
 *
 * @param {object} node - The node the editor is drawn on.
 * @param {{width: number, height: number}} frame - Filled in with the image's size on every
 *   answer that states one.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function trackedBackdrop(node, frame) {
  // The picture is the image the node received on its last run, published by `preview.publish`
  // from the node's own `execute` and served by `modules/interface/preview.py`.
  const source = imageBackdrop(node);
  return {
    async load() {
      // The editor converts every gesture through the image's size on its own, and both of these
      // nodes need it as well: one holds its rectangle to the image and the other slides a square
      // inside it, so neither can say what it cuts without it. It is taken from the answer the
      // editor is given rather than asked for a second time, so the two can never be reading
      // different images.
      const answer = await source.load();
      const width = Math.max(0, Number(answer?.width) || 0);
      const height = Math.max(0, Number(answer?.height) || 0);
      // An answer stating no size leaves the last one standing, as the editor's own frame does.
      // The store a picture is held in is bounded and evicts the least recently used, so a node
      // in a busy graph answers `waiting` again after it has been answering `ready`, and the
      // image it received on its last run is still the image it received.
      if (width > 0 && height > 0) {
        frame.width = width;
        frame.height = height;
      }
      return answer;
    },
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
 * @param {{refresh: () => void}} editor - Editor from `createRegionEditor`.
 * @returns {() => void} Unhooks the listener.
 */
function watchRuns(node, editor) {
  return onNodeDone(node, () => {
    try {
      editor.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to ask for the image again:`, error);
    }
  });
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
 * Append the editor to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @param {object} spec - The node's entry in `SPECS`.
 * @returns {void}
 */
function attachRegionEditor(node, spec) {
  for (const name of spec.widgets) {
    if (!findWidget(node, name)) return;
  }

  // The size of the image the node received, filled in by `trackedBackdrop`. A node that has
  // published nothing leaves it at zero, and the editor draws the stand-in picture and refuses
  // every gesture: there is no frame to measure one in, and a rectangle measured against a guess,
  // the stand-in's own size included, would write numbers that look right on screen and cut
  // somewhere else.
  const frame = { width: 0, height: 0 };

  // The three numbers behind the window a gesture is drawing, kept between the shape that drew
  // it and the write that stores it. Nothing reads it outside a gesture.
  const plan = { rect: null, x: 0, y: 0, size: 0 };

  const editor = createRegionEditor({
    node,
    backdrop: trackedBackdrop(node, frame),
    rect: {
      read: () => spec.read(node, frame),
      write: (rect, moved) => spec.write(node, rect, moved, frame, plan),
      locks: (gesture) => spec.locks(node, gesture),
    },
    // Alt on a corner holds the aspect the gesture started with, which for a square that fits
    // inside the image is the square itself, so the shape below has nothing to correct there.
    coerce: spec.shape
      ? (rect, moved) => spec.shape(node, rect, moved, frame, plan)
      : undefined,
    // The rectangle handed over is the one on screen, which is what an unfinished gesture holds
    // while there is one, so the two sizes follow the drag rather than reporting the crop the
    // widgets held before it started, and a crop being sized for a sampler is read off the line
    // that names the output rather than off the one that names the rectangle.
    footer: (rect) => footerMeaning(cutWindow(frame, rect), outputStep(node)),
    // A `divisible_by` filled by a link is not on the node at all, so the footer says that
    // rather than a size it cannot work out, and the rule the two sizes follow rides on the
    // hover text, where a rule that never changes costs the numbers no room.
    hover: () => footerHover(outputStep(node)),
    height: UI_HEIGHT,
    // Asked at each gesture rather than fixed here, so shift snaps to the step the node is set
    // to and a rectangle laid on that grid is one the resize leaves at the size it was cut. That
    // is a grid for the edge being dragged and not for the side: an axis comes back at the size
    // it was cut only when both of its edges sit on that grid, so on `Image Crop Location` the
    // opposite edge has to be there already, and on the square, whose untouched edge is
    // `x - size // 2`, it rarely is.
    coarseStep: () => outputStep(node) ?? DEFAULT_OUTPUT_STEP,
    minSize: spec.minSize,
  });

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, editor, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  for (const name of spec.widgets) {
    chainWidgetCallback(node, name, () => editor.handleRectChanged());
  }

  // The rectangle does not move with the step, but the footer's account of the output is worked
  // out from it, so a change to it repaints.
  chainWidgetCallback(node, DIVISIBLE_BY, () => editor.schedulePaint());

  const stopWatchingRuns = watchRuns(node, editor);

  // Linking one of these inputs leaves its widget read by nothing, and attaching a link changes
  // no widget value, so the callbacks above never hear about it.
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
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      stopWatchingRuns();
      editor.dispose();
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
      category: ["WAS Node Suite", "Image Crop Location", "Region editor"],
      name: "Show the region editor",
      tooltip:
        "Draw the crop rectangle over the image on Image Crop Location and Image Crop " +
        "Square Location. The widgets themselves are always available. This applies to " +
        "nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const id = nodeData?.name;
    // Asked for with `hasOwn`, so a node named after something every object inherits is not
    // handed this pack's editor.
    if (typeof id !== "string" || !Object.hasOwn(SPECS, id)) return;
    const spec = SPECS[id];

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second editor.
    if (proto.__was_crop_region_wrapped) return;
    proto.__was_crop_region_wrapped = true;

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
