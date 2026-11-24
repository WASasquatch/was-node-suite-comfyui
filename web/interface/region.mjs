/**
 * The region editor node interfaces draw a rectangle with.
 *
 * `createRegionEditor` builds one focusable element: a rectangle with handles over a backdrop,
 * editing the four numbers behind it through an accessor pair. Every number it writes is in
 * frame units.
 */

import { app } from "../../../scripts/app.js";
import { blankBackdrop, normaliseFrame } from "./backdrop.mjs";
import { ICON_SIZE, drawIcon, hoverTitles } from "./icons.mjs";
import {
  STAND_IN_ICON,
  drawStandIn,
  loadPlaceholder,
  standInTitle,
} from "./placeholder.mjs";
import { captureWheel, elementPoint } from "./pointer.mjs";
import { PREVIEW_STATE } from "./preview.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.mjs";
import { onThemeChange, readTheme } from "./theme.mjs";

const LOG_NAME = "WASNodeSuite.RegionEditor";

/**
 * The four edges of the rectangle.
 */
export const EDGE = {
  LEFT: "left",
  TOP: "top",
  RIGHT: "right",
  BOTTOM: "bottom",
};

/**
 * What a gesture does to the rectangle, which is what a lock is asked about.
 */
export const GESTURE = {
  RESIZE: "resize",
  MOVE: "move",
};

// Edge order, used wherever edges are counted or listed so a footer words them alike.
const EDGE_ORDER = [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT, EDGE.BOTTOM];

// The two edges measured across the frame. The other two are measured down it.
const ACROSS = new Set([EDGE.LEFT, EDGE.RIGHT]);

// The eight handles, as a fraction of the drawn rectangle and the edges each one moves.
// Corners carry two edges and are hit tested first, since at a small rectangle every edge
// handle overlaps one and a corner is the harder target to reach.
const HANDLES = [
  { key: "topLeft", fx: 0, fy: 0, edges: [EDGE.LEFT, EDGE.TOP], cursor: "nwse-resize" },
  { key: "topRight", fx: 1, fy: 0, edges: [EDGE.RIGHT, EDGE.TOP], cursor: "nesw-resize" },
  { key: "bottomRight", fx: 1, fy: 1, edges: [EDGE.RIGHT, EDGE.BOTTOM], cursor: "nwse-resize" },
  { key: "bottomLeft", fx: 0, fy: 1, edges: [EDGE.LEFT, EDGE.BOTTOM], cursor: "nesw-resize" },
  { key: "top", fx: 0.5, fy: 0, edges: [EDGE.TOP], cursor: "ns-resize" },
  { key: "right", fx: 1, fy: 0.5, edges: [EDGE.RIGHT], cursor: "ew-resize" },
  { key: "bottom", fx: 0.5, fy: 1, edges: [EDGE.BOTTOM], cursor: "ns-resize" },
  { key: "left", fx: 0, fy: 0.5, edges: [EDGE.LEFT], cursor: "ew-resize" },
];

// Height of the appended widget in node units, and the margin a DOM widget element is inset
// by on every side, which makes the element itself shorter than the widget by twice it.
const DEFAULT_HEIGHT = 220;
const UI_MARGIN = 10;

// Layout bands, measured in element pixels.
const PAD = 4;
const FOOTER_HEIGHT = 13;

// The footer's two lines, carrying values and states and nothing else. The first carries the
// readout and whatever the gesture, the links or a rectangle outside the frame has to say. The
// second carries what the rectangle means, beside the size of the frame those numbers are
// measured in, which is a standing fact about the node and is never given up for a note.
const FOOTER_LINES = 2;
const MIN_FRAME_HEIGHT = 24;

// The gap kept between a glyph and whatever follows it on the same line.
const GLYPH_GAP = 4;

// What the frame's own slot says while there is no frame. Every gesture is refused until then.
const NO_FRAME = "no frame yet";

const HANDLE_SIZE = 7;
const HIT_RADIUS = 8;

const BODY_FONT = "10px sans-serif";
const LABEL_FONT = "11px sans-serif";

// The area outside the rectangle is laid under a wash of the panel colour rather than of
// black, so it reads as covered in a light palette as well as in a dark one.
const SCRIM_ALPHA = 0.45;

const MESSAGE_TIMEOUT = 4000;

// How long to wait before asking for a picture again while the answer is not the picture. A
// node queued before the socket opened publishes on its next run, so an answer of `waiting` is
// never the last word.
//
// The wait doubles up to a ceiling rather than staying flat, so a node that is placed and
// never run is asked at a falling rate rather than at the rate a node about to run is asked
// at. A finished prompt calls `refresh`, which puts the wait
// back to the first one, so a node that does run is answered promptly however long the page has
// been open.
const RETRY_INTERVAL = 3000;
const RETRY_MAX_INTERVAL = 30000;
const RETRY_BACKOFF = 2;

// What one arrow press moves the rectangle by, and what a shift held with it moves it by. The
// coarse step is also what the pointer snaps to, so the two gestures agree on the grid.
const DEFAULT_STEP = 1;
const DEFAULT_COARSE_STEP = 10;

// The smallest width or height a gesture can produce. A rectangle already smaller than this is
// drawn as it stands, since the frame clamps the display and never the widget.
const DEFAULT_MIN_SIZE = 1;

const ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

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
 * Read a number, answering a fallback for anything that is not one.
 *
 * @param {*} value - Value to read.
 * @param {number} fallback - What to answer when the value is not a finite number.
 * @returns {number} The number, or the fallback.
 */
function toNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/**
 * Snap a value to a multiple of a step.
 *
 * This is pointer and key arithmetic rather than a node's, so it rounds half up.
 *
 * @param {number} value - Value to snap.
 * @param {number} step - Step to snap to. A step at or below zero leaves the value alone.
 * @returns {number} The snapped value.
 */
function snapTo(value, step) {
  if (!(step > 0)) return value;
  // The result is trimmed to two decimals so a step of 0.1 produces 1.5 rather than
  // 1.5000000000000002, and adding zero at the end turns the negative zero a drag just past an
  // edge produces back into a plain one.
  return Number((Math.round(value / step) * step).toFixed(2)) + 0;
}

/**
 * Format a number for the footer.
 *
 * @param {number} value - Value to write.
 * @returns {string} The value with at most two decimals.
 */
function formatNumber(value) {
  if (!Number.isFinite(value)) return "?";
  return String(Math.round(value * 100) / 100);
}

/**
 * Read a rectangle as its four edges.
 *
 * @param {{x: number, y: number, w: number, h: number}} rect - Rectangle in frame units.
 * @returns {{left: number, top: number, right: number, bottom: number}} The four edges.
 */
function edgesOf(rect) {
  return { left: rect.x, top: rect.y, right: rect.x + rect.w, bottom: rect.y + rect.h };
}

/**
 * Read four edges back as a rectangle.
 *
 * @param {{left: number, top: number, right: number, bottom: number}} edges - The four edges.
 * @returns {{x: number, y: number, w: number, h: number}} Rectangle in frame units.
 */
function rectOf(edges) {
  return {
    x: edges.left,
    y: edges.top,
    w: edges.right - edges.left,
    h: edges.bottom - edges.top,
  };
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
 * Work out where the frame and the footer sit inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the area the frame is fitted into and of the footer.
 */
function computeLayout(width, height) {
  const footerY = Math.max(0, height - PAD - FOOTER_HEIGHT * FOOTER_LINES);
  const areaX0 = PAD;
  const areaX1 = Math.max(areaX0 + 1, width - PAD);
  const areaY0 = PAD;
  const areaY1 = Math.max(areaY0 + MIN_FRAME_HEIGHT, footerY - 2);

  return {
    width,
    height,
    areaX0,
    areaY0,
    areaX1,
    areaY1,
    areaWidth: areaX1 - areaX0,
    areaHeight: areaY1 - areaY0,
    footerY,
  };
}

/**
 * Fit the frame into the area, keeping the frame's own aspect.
 *
 * @param {object} layout - Layout from `computeLayout`.
 * @param {object} frame - Frame from `normaliseFrame`.
 * @returns {{x0: number, y0: number, drawWidth: number, drawHeight: number, frame: object}|null}
 *   Where the frame is drawn in element pixels, or null while there is no size to draw it at.
 */
function computeView(layout, frame) {
  if (!(frame?.width > 0) || !(frame?.height > 0)) return null;
  if (!(layout.areaWidth > 0) || !(layout.areaHeight > 0)) return null;

  const fit = Math.min(layout.areaWidth / frame.width, layout.areaHeight / frame.height);
  const drawWidth = Math.max(1, frame.width * fit);
  const drawHeight = Math.max(1, frame.height * fit);

  return {
    x0: layout.areaX0 + (layout.areaWidth - drawWidth) / 2,
    y0: layout.areaY0 + (layout.areaHeight - drawHeight) / 2,
    drawWidth,
    drawHeight,
    frame,
  };
}

/**
 * Convert a horizontal element position into frame units.
 *
 * @param {object} view - View from `computeView`.
 * @param {number} x - Position in element pixels.
 * @returns {number} Position across the frame, in frame units.
 */
function unitsX(view, x) {
  return ((x - view.x0) * view.frame.width) / view.drawWidth;
}

/**
 * Convert a vertical element position into frame units.
 *
 * @param {object} view - View from `computeView`.
 * @param {number} y - Position in element pixels.
 * @returns {number} Position down the frame, in frame units.
 */
function unitsY(view, y) {
  return ((y - view.y0) * view.frame.height) / view.drawHeight;
}

/**
 * Convert a position across the frame into element pixels.
 *
 * @param {object} view - View from `computeView`.
 * @param {number} units - Position in frame units.
 * @returns {number} Position in element pixels.
 */
function pixelX(view, units) {
  return view.x0 + (units * view.drawWidth) / view.frame.width;
}

/**
 * Convert a position down the frame into element pixels.
 *
 * @param {object} view - View from `computeView`.
 * @param {number} units - Position in frame units.
 * @returns {number} Position in element pixels.
 */
function pixelY(view, units) {
  return view.y0 + (units * view.drawHeight) / view.frame.height;
}

/**
 * Run a write inside the canvas change events the graph's change tracker listens for.
 *
 * @param {() => void} write - What to run between the two events. Its exceptions are the
 *   caller's, and the bracket is closed either way.
 * @returns {void}
 */
export function withGraphChange(write) {
  // The tracker's own snapshot triggers are a document `mouseup` and the release of a bare
  // modifier key, so a commit made from the keyboard, from a menu or from a dialog reaches none
  // of them and would otherwise be folded into whatever the previous snapshot held. The two
  // events are feature detected together, since emitting only the first leaves the tracker's
  // nesting count above zero and stops it snapshotting at all.
  const canvas = app.canvas;
  const transactional =
    typeof canvas?.emitBeforeChange === "function" &&
    typeof canvas?.emitAfterChange === "function";

  if (transactional) canvas.emitBeforeChange();
  try {
    write();
  } finally {
    if (transactional) canvas.emitAfterChange();
  }
}

/**
 * Build the region editor for one node.
 *
 * @param {object} options - How the editor reads, writes and draws.
 * @param {object} [options.node] - The node the editor is drawn on. Used to repaint the graph
 *   after a write, and for nothing else.
 * @param {{load: () => Promise<object>}} options.backdrop - What is drawn behind the
 *   rectangle, from `imageBackdrop`, `blankBackdrop` or anything answering the same shape.
 * @param {{read: () => object, write: (rect: object, moved: object) => void,
 *   locks?: () => object}} options.rect - The accessor pair.
 *
 *   `read` answers the rectangle the node's widgets hold, as `{x, y, w, h}` in frame units,
 *   exactly as they are held. A number outside the frame is answered as it stands.
 *
 *   `write` is called once per gesture with the whole rectangle, including the edges the
 *   gesture left alone, and a map holding `true` for each edge that ended somewhere else. The
 *   rectangle is whole, which gives a node holding an origin and a size both edges of an axis;
 *   the map is what says which of its widgets the gesture was
 *   actually about, so a drag on one edge leaves the rest of them alone. A widget already
 *   holding the value it would be given is left unassigned, as every widget write in this pack
 *   is.
 *
 *   `locks` answers a map from edge to the name of the input that stops it moving, holding
 *   only the edges that cannot move. An edge is locked when any widget the edge would write is
 *   filled in by a link, since a link is what the run reads instead of the widget. It is asked
 *   with the gesture's own kind, one of `GESTURE`: the widgets a translation writes are
 *   not always the widgets a resize writes.
 * @param {(rect: object, moved: object) => object} [options.coerce] - The rectangle the adopter
 *   would actually write for the one a gesture reached, answered as `{x, y, w, h}` in frame
 *   units. A node holding a shape its four numbers cannot spell, a square for one, states it
 *   here: the answer is what is drawn, what the readout quotes and what `write` is given, so the
 *   rubber band, the footer and the value stored are one rectangle rather than three. Left out,
 *   a gesture is held exactly as it was made.
 * @param {string|((rect: object) => string)} [options.footer] - What the rectangle means, in the
 *   adopter's own words, drawn on the second footer line whatever else is on screen. Called with
 *   the rectangle on screen, which is the one an unfinished gesture holds while there is one, so
 *   the line describes what the numbers above it say rather than what the widgets still hold. It
 *   carries values and states: what the node will write, what it will hand on, and anything that
 *   changes what somebody should do next.
 * @param {string|((rect: object) => string)} [options.hover] - What those numbers are measured in
 *   and what the node does with them, in the adopter's own words, put on the footer's own hover
 *   text rather than drawn. A unit and a rule do not change as somebody works, so they are
 *   reachable from the numbers they are about instead of standing beside them every frame.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {number} [options.step] - What one arrow press moves the rectangle by.
 * @param {number|(() => number)} [options.coarseStep] - What an arrow press with shift moves it
 *   by, and the grid a gesture with shift snaps to. A function is asked at each gesture, which is
 *   what a node carrying its own step as a widget states.
 * @param {number} [options.gridStep] - The grid every gesture lands on, for a node whose numbers
 *   carry a step of their own. Applied while the gesture is dragged rather than as it is
 *   written, so the rectangle drawn, the readout and the smallest size all agree with the value
 *   stored. The frame's own edge and the smallest size are held to after it, so an edge stopped
 *   by either lands where it was stopped, and alt holds a corner to its aspect instead, since
 *   one corner cannot honour both. Left out, a gesture lands wherever it was made.
 * @param {number} [options.minSize] - The smallest width or height a gesture can produce.
 * @param {boolean} [options.integer] - Whether written values are whole frame units. True by
 *   default, since a node's rectangle is usually declared as integers.
 * @param {Array<(ctx: CanvasRenderingContext2D, view: object, theme: object) => Array<object>|void>}
 *   [options.layers] - What else is drawn over the frame, in order, between the backdrop and
 *   the rectangle. Each is handed the context, the view a position converts through, and the
 *   theme, and may answer hover regions in element pixels, `{x, y, width, height, title}`,
 *   which are offered to the pointer ahead of the footer's own. A layer that throws is logged
 *   and skipped, and the rest of the frame is still drawn.
 * @param {{cursor?: (context: object) => string,
 *   pointerDown?: (event: PointerEvent, context: object) => boolean,
 *   pointerMove?: (event: PointerEvent, context: object) => boolean,
 *   pointerUp?: (event: PointerEvent, context: object) => boolean,
 *   cancel?: () => void, leave?: () => void}} [options.tool] - A tool that gets first refusal on
 *   the pointer, from `mask_paint.mjs` or anything answering the same shape. Each pointer member
 *   answers true when the tool took the event, which stops the rectangle from seeing it, and the
 *   editor holds the pointer capture for a gesture the tool claimed. `cursor` answers what the
 *   pointer looks like, or an empty string to leave that to the rectangle. `cancel` ends a
 *   claimed gesture that lost its capture or its window, which arrives with no pointerup, and
 *   `leave` says the pointer is no longer over the element. The context carries `point` in
 *   element pixels, `frame`, the same position in frame units or null while there is no frame,
 *   `view` and `modifiers`.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   handleRectChanged: () => void, refresh: () => void, dispose: () => void}} The element to
 *   hand to `addDOMWidget`, the height it was built for, a coalesced repaint, the repaint to
 *   run when a widget changed, a fresh ask for the backdrop, and teardown.
 */
export function createRegionEditor(options = {}) {
  const settings = {
    node: options.node ?? null,
    backdrop: options.backdrop ?? blankBackdrop(),
    rect: options.rect ?? {},
    coerce: typeof options.coerce === "function" ? options.coerce : null,
    footer: options.footer ?? "",
    hover: options.hover ?? "",
    height: Math.max(UI_MARGIN * 2 + MIN_FRAME_HEIGHT, toNumber(options.height, DEFAULT_HEIGHT)),
    step: Math.max(0, toNumber(options.step, DEFAULT_STEP)),
    coarseStep:
      typeof options.coarseStep === "function"
        ? options.coarseStep
        : Math.max(0, toNumber(options.coarseStep, DEFAULT_COARSE_STEP)),
    gridStep: Math.max(0, toNumber(options.gridStep, 0)),
    minSize: Math.max(0, toNumber(options.minSize, DEFAULT_MIN_SIZE)),
    integer: options.integer !== false,
    layers: Array.isArray(options.layers) ? options.layers.filter((layer) => typeof layer === "function") : [],
    tool: options.tool ?? null,
  };

  const root = document.createElement("div");
  root.tabIndex = 0;
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${settings.height - UI_MARGIN * 2}px`,
    "overflow:hidden",
    "outline:none",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The footer states what its numbers are measured in through the element's own title.
  const titles = hoverTitles(root);

  /**
   * The grid a gesture with shift lands on.
   *
   * @returns {number} The step, never negative.
   */
  function coarseStep() {
    const value =
      typeof settings.coarseStep === "function" ? settings.coarseStep() : settings.coarseStep;
    return Math.max(0, toNumber(value, DEFAULT_COARSE_STEP));
  }

  const state = {
    frame: normaliseFrame({ state: PREVIEW_STATE.LOADING }),
    loading: false,
    reloadWanted: false,
    retryTimer: 0,
    retryWait: RETRY_INTERVAL,
    drag: null,
    pending: null,
    hover: null,
    lastWritten: null,
    message: "",
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    view: null,
    // The pointer a tool claimed, whose capture the editor holds until the tool releases it.
    toolPointer: null,
    // Hover regions the layers answered on the last repaint, offered to the pointer alongside
    // the footer's own.
    layerRegions: [],
    disposed: false,
  };

  /**
   * Read the rectangle the node's widgets hold.
   *
   * @returns {{x: number, y: number, w: number, h: number}|null} The rectangle in frame units,
   *   or null when the accessor answered something that is not one.
   */
  function readRect() {
    try {
      // The accessor is the only thing here that knows a widget name. A node calling its numbers
      // `top`, `left`, `right` and `bottom` and a node calling them `x`, `y`, `width` and
      // `height` adopt the same editor through it.
      const value = settings.rect.read?.();
      if (!value) return null;
      const rect = {
        x: Number(value.x),
        y: Number(value.y),
        w: Number(value.w),
        h: Number(value.h),
      };
      if (!Object.values(rect).every(Number.isFinite)) return null;
      return rect;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the rectangle:`, error);
      return null;
    }
  }

  /**
   * The rectangle on screen, which is the one an unfinished gesture holds while there is one.
   *
   * @returns {{x: number, y: number, w: number, h: number}|null} The rectangle in frame units.
   */
  function currentRect() {
    return state.pending?.rect ?? readRect();
  }

  /**
   * Read which edges cannot be written, and what to call them.
   *
   * @param {string} [gesture] - What the gesture would do, one of `GESTURE`. A resize is asked
   *   about by default, since that is what the handles, the cursor and the drawing are about.
   * @returns {object} A map from edge to input name, holding only the edges that are locked.
   */
  function readLocks(gesture = GESTURE.RESIZE) {
    const locks = {};
    try {
      // The kind of gesture goes with the question, rather than the edges alone: a link on a
      // size stops every resize and stops no move.
      const value = settings.rect.locks?.(gesture);
      if (!value) return locks;
      for (const edge of EDGE_ORDER) {
        const name = value[edge];
        if (typeof name === "string" && name) locks[edge] = name;
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read which inputs are linked:`, error);
    }
    return locks;
  }

  /**
   * Name the first locked input among a set of edges.
   *
   * @param {string[]} edges - Edges a gesture would move.
   * @param {string} [gesture] - What the gesture would do, one of `GESTURE`.
   * @returns {string} The input's name, empty when every one of those edges can be written.
   */
  function lockedName(edges, gesture) {
    const locks = readLocks(gesture);
    for (const edge of EDGE_ORDER) {
      if (edges.includes(edge) && locks[edge]) return locks[edge];
    }
    return "";
  }

  /**
   * Show a short note in the footer.
   *
   * @param {string} text - Note to show.
   * @returns {void}
   */
  function setMessage(text) {
    // The listeners outlive the teardown, since the element is only detached by whoever put it
    // on the node, so a press arriving after it would otherwise arm a timer nothing clears.
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
   * Refuse a gesture that would move an input a link fills in, and say which.
   *
   * @param {string[]} edges - Edges the gesture would move.
   * @param {string} [gesture] - What the gesture would do, one of `GESTURE`.
   * @returns {boolean} True when the gesture was refused.
   */
  function refuseLocked(edges, gesture) {
    const name = lockedName(edges, gesture);
    if (!name) return false;
    setMessage(`${name} is linked`);
    return true;
  }

  /**
   * Hold a value to a whole frame unit when the adopter asked for whole units.
   *
   * @param {number} value - Value a gesture produced.
   * @returns {number} The value, rounded when the editor writes integers.
   */
  function roundValue(value) {
    return settings.integer ? Math.round(value) : Number(value.toFixed(2)) + 0;
  }

  /**
   * Write the rectangle once, through the accessor pair, with its own undo entry.
   *
   * @param {{x: number, y: number, w: number, h: number}} rect - The whole rectangle, in frame
   *   units, including the edges the gesture left alone.
   * @param {object} moved - A map from edge to true, holding only the edges that moved.
   * @returns {void}
   */
  function writeRect(rect, moved) {
    if (state.disposed) return;

    state.lastWritten = { ...rect };
    // Bracketing the pointer path as well as the keyboard one is harmless, since the `mouseup`
    // that follows a drag finds nothing changed.
    withGraphChange(() => {
      try {
        settings.rect.write?.(rect, moved);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to write the rectangle:`, error);
      }
    });
    settings.node?.setDirtyCanvas?.(true, true);
  }

  /**
   * Write the rectangle an unfinished gesture holds.
   *
   * @returns {void}
   */
  function commitPending() {
    const pending = state.pending;
    state.pending = null;
    if (!pending) return;

    const current = readRect();
    if (current) {
      const before = edgesOf(current);
      const after = edgesOf(pending.rect);
      const moved = {};
      let any = false;
      for (const edge of EDGE_ORDER) {
        if (pending.moved[edge] && after[edge] !== before[edge]) {
          moved[edge] = true;
          any = true;
        }
      }
      if (any) writeRect(pending.rect, moved);
    }
    schedulePaint();
  }

  /**
   * Repaint after a widget changed, dropping a gesture the change invalidated.
   *
   * @returns {void}
   */
  function handleRectChanged() {
    const current = readRect();
    if (!sameRect(current, state.lastWritten)) {
      state.lastWritten = null;
      if (state.drag) endDrag(false);
      else state.pending = null;
    }
    schedulePaint();
  }

  /**
   * Ask the backdrop for the frame, and again later while the answer is not the picture.
   *
   * @returns {void}
   */
  function loadFrame() {
    if (state.disposed) return;
    // An ask made while one is already in flight is recorded rather than dropped. The answer on
    // its way was asked for before whatever prompted the second one, so a second prompt finishing
    // while the first answer is still crossing would otherwise leave the editor converting every
    // gesture through the size of an image the node no longer holds.
    if (state.loading) {
      state.reloadWanted = true;
      return;
    }
    state.loading = true;
    state.reloadWanted = false;
    Promise.resolve()
      .then(() => settings.backdrop?.load?.())
      .then((answer) => {
        if (state.disposed) return;
        state.frame = keepSize(normaliseFrame(answer));
      })
      .catch((error) => {
        if (state.disposed) return;
        console.error(`[${LOG_NAME}] Failed to read the backdrop:`, error);
        state.frame = keepSize(normaliseFrame({ state: PREVIEW_STATE.FAILED }));
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
   * The size the frame is holding, so a failure keeps whatever size was already known.
   *
   * @returns {{width: number, height: number}} The frame's size in frame units.
   */
  function frameSize() {
    return { width: state.frame.width, height: state.frame.height };
  }

  /**
   * Keep the size already known when a later answer states none.
   *
   * @param {object} frame - The answer, from `normaliseFrame`.
   * @returns {object} The answer, carrying the last known size where it states none of its own.
   */
  function keepSize(frame) {
    if (frame.width > 0 && frame.height > 0) return frame;
    if (!(state.frame.width > 0) || !(state.frame.height > 0)) return frame;
    // The store a published picture is held in is bounded and evicts the least recently used, so
    // a node whose picture has aged out of a busy graph answers `waiting` again after it has been
    // answering `ready`. The size is the image that node received on its last run either way, and
    // dropping it would take the rectangle, the handles and every gesture with it. The state is
    // the answer's own, so the stand-in comes back over a frame the editor goes on measuring in.
    return { ...frame, ...frameSize() };
  }

  /**
   * Ask again for a backdrop that is not the picture yet.
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
   * Ask for the backdrop again now.
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
   * Read the pointer position in element pixels.
   *
   * @param {PointerEvent|MouseEvent} event - Event to read.
   * @returns {{x: number, y: number}} Position inside the element.
   */
  function localPoint(event) {
    return elementPoint(root, event);
  }

  /**
   * Read the pointer position in frame units.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {{x: number, y: number}|null} Position in frame units, or null while there is no
   *   frame to convert it through.
   */
  function framePoint(point) {
    const view = state.view;
    if (!view) return null;
    return { x: unitsX(view, point.x), y: unitsY(view, point.y) };
  }

  /**
   * The rectangle as it is drawn: normalised, held inside the frame, and marked where it ran
   * past it.
   *
   * @returns {{rect: object, edges: object, box: object, pinned: object, negative: boolean}|null}
   *   The true rectangle, its true edges, the edges as drawn, which of them are pinned, and
   *   whether the rectangle has a negative width or height. Null while there is nothing to
   *   draw.
   */
  function displayModel() {
    const view = state.view;
    const rect = currentRect();
    if (!view || !rect) return null;

    const frame = view.frame;
    const edges = edgesOf(rect);
    const outside = (value, limit) => value < 0 || value > limit;

    return {
      rect,
      edges,
      box: {
        left: clamp(Math.min(edges.left, edges.right), 0, frame.width),
        right: clamp(Math.max(edges.left, edges.right), 0, frame.width),
        top: clamp(Math.min(edges.top, edges.bottom), 0, frame.height),
        bottom: clamp(Math.max(edges.top, edges.bottom), 0, frame.height),
      },
      pinned: {
        left: outside(edges.left, frame.width),
        right: outside(edges.right, frame.width),
        top: outside(edges.top, frame.height),
        bottom: outside(edges.bottom, frame.height),
      },
      negative: edges.right < edges.left || edges.bottom < edges.top,
    };
  }

  /**
   * Where the drawn rectangle sits in element pixels.
   *
   * @param {object} model - Model from `displayModel`.
   * @returns {{left: number, top: number, right: number, bottom: number}} The drawn edges.
   */
  function boxPixels(model) {
    const view = state.view;
    return {
      left: pixelX(view, model.box.left),
      right: pixelX(view, model.box.right),
      top: pixelY(view, model.box.top),
      bottom: pixelY(view, model.box.bottom),
    };
  }

  /**
   * Where one handle sits in element pixels.
   *
   * @param {object} box - Drawn edges from `boxPixels`.
   * @param {object} handle - One of `HANDLES`.
   * @returns {{x: number, y: number}} The handle's middle.
   */
  function handlePoint(box, handle) {
    return {
      x: box.left + (box.right - box.left) * handle.fx,
      y: box.top + (box.bottom - box.top) * handle.fy,
    };
  }

  /**
   * Find what is under a point.
   *
   * Corners answer first.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {{kind: string, handle: object|null, edges: string[]}|null} What the point is on,
   *   or null when it is on nothing.
   */
  function hitTest(point) {
    const model = displayModel();
    if (!model) return null;
    const box = boxPixels(model);

    for (const handle of HANDLES) {
      const middle = handlePoint(box, handle);
      if (
        Math.abs(point.x - middle.x) <= HIT_RADIUS &&
        Math.abs(point.y - middle.y) <= HIT_RADIUS
      ) {
        return { kind: "handle", handle, edges: handle.edges };
      }
    }

    if (
      point.x >= box.left &&
      point.x <= box.right &&
      point.y >= box.top &&
      point.y <= box.bottom
    ) {
      return { kind: "body", handle: null, edges: EDGE_ORDER };
    }
    return null;
  }

  /**
   * What a hit would do to the rectangle.
   *
   * @param {string} kind - What the press landed on, from `hitTest`.
   * @returns {string} One of `GESTURE`.
   */
  function gestureOf(kind) {
    return kind === "body" ? GESTURE.MOVE : GESTURE.RESIZE;
  }

  /**
   * Move the whole rectangle.
   *
   * The size is carried through untouched, so a move never rewrites a width or a height.
   *
   * @param {object} start - Edges the gesture started from.
   * @param {number} dx - How far across to move, in frame units.
   * @param {number} dy - How far down to move, in frame units.
   * @param {boolean} snap - Snap the origin to the coarse step rather than to the adopter's own
   *   grid.
   * @returns {object} The four edges after the move.
   */
  function moveEdges(start, dx, dy, snap) {
    const frame = state.frame;
    // The size the gesture started with, carried to the far edges below, so a node whose numbers
    // are an origin and a size has only its origin written for a move.
    const width = start.right - start.left;
    const height = start.bottom - start.top;
    const step = snap ? coarseStep() : settings.gridStep;

    let left = snapTo(start.left + dx, step);
    let top = snapTo(start.top + dy, step);
    left = roundValue(left);
    top = roundValue(top);
    // The origin is held so the rectangle does not leave the frame, and a rectangle already larger
    // than the frame is held so it does not stop covering it, which is the same bound read from
    // whichever side is the outer one.
    left = clamp(left, Math.min(0, frame.width - width), Math.max(0, frame.width - width));
    top = clamp(top, Math.min(0, frame.height - height), Math.max(0, frame.height - height));

    return { left, top, right: left + width, bottom: top + height };
  }

  /**
   * Hold a corner's two edges to the aspect the rectangle started the gesture with.
   *
   * @param {object} next - Edges the gesture has reached, adjusted in place.
   * @param {object} start - Edges the gesture started from.
   * @param {object} handle - The corner being dragged.
   * @returns {void}
   */
  function applyAspect(next, start, handle) {
    const frame = state.frame;
    const startWidth = start.right - start.left;
    const startHeight = start.bottom - start.top;
    if (!(startWidth > 0) || !(startHeight > 0)) return;

    const ratio = startWidth / startHeight;
    const movesLeft = handle.edges.includes(EDGE.LEFT);
    const movesTop = handle.edges.includes(EDGE.TOP);
    const anchorX = movesLeft ? start.right : start.left;
    const anchorY = movesTop ? start.bottom : start.top;

    let width = Math.abs((movesLeft ? next.left : next.right) - anchorX);
    let height = Math.abs((movesTop ? next.top : next.bottom) - anchorY);
    // The axis the pointer reached further along is the one that sets the size.
    if (width / ratio > height) height = width / ratio;
    else width = height * ratio;

    // The pair is held inside the frame by shortening both sides together, so the aspect is what
    // survives the frame rather than what the frame breaks.
    const roomX = movesLeft ? anchorX : frame.width - anchorX;
    const roomY = movesTop ? anchorY : frame.height - anchorY;
    // An anchor outside the frame leaves no room to fit inside it, so the pair is left at the
    // aspect and the frame is what holds the moved edges afterwards.
    if (roomX > 0 && roomY > 0) width = Math.min(width, roomX, roomY * ratio);
    height = width / ratio;

    if (movesLeft) next.left = anchorX - width;
    else next.right = anchorX + width;
    if (movesTop) next.top = anchorY - height;
    else next.bottom = anchorY + height;
  }

  /**
   * Move the edges one handle carries. A moved edge lands where the pointer is, held inside the
   * frame.
   *
   * @param {object} start - Edges the gesture started from.
   * @param {object} drag - The gesture in progress.
   * @param {{x: number, y: number}} point - Pointer position in frame units.
   * @param {{snap: boolean, aspect: boolean}} modifiers - Which modifiers are held.
   * @returns {object} The four edges after the resize.
   */
  function resizeEdges(start, drag, point, modifiers) {
    const frame = state.frame;
    const handle = drag.handle;
    // The edges the handle does not carry are passed through exactly as they were, which is how a
    // value outside the frame survives a gesture on another edge.
    const next = { ...start };

    for (const edge of handle.edges) {
      const across = ACROSS.has(edge);
      const limit = across ? frame.width : frame.height;
      // The grab offset keeps the edge under the point it was picked up by, so a handle
      // pressed off centre does not jump when the pointer first moves.
      const raw = (across ? point.x : point.y) + drag.grab[edge];
      const step = modifiers.snap ? coarseStep() : settings.gridStep;
      // The edge is held inside the frame, which is as far as a gesture can point.
      next[edge] = clamp(snapTo(raw, step), 0, limit);
    }

    // Only a corner preserves the aspect. On an edge handle it would move the two edges the
    // gesture never touched, which are edges the gesture was not refused for and may not even be
    // writable, so an edge handle ignores the modifier.
    if (modifiers.aspect && handle.edges.length > 1) applyAspect(next, start, handle);

    for (const edge of handle.edges) {
      const across = ACROSS.has(edge);
      const limit = across ? frame.width : frame.height;
      const opposite = across
        ? edge === EDGE.LEFT
          ? next.right
          : next.left
        : edge === EDGE.TOP
          ? next.bottom
          : next.top;
      const towards = edge === EDGE.LEFT || edge === EDGE.TOP ? -1 : 1;
      const bound = opposite + towards * settings.minSize;
      next[edge] = roundValue(next[edge]);
      // The smallest size the gesture may produce, and then the frame again, in that order:
      // a frame narrower than the smallest size keeps the frame.
      next[edge] = towards < 0 ? Math.min(next[edge], bound) : Math.max(next[edge], bound);
      next[edge] = clamp(next[edge], 0, limit);
      // An opposite edge already outside the frame leaves the frame's clamp standing past it,
      // which would turn the rectangle inside out and write a size below zero. The edge is
      // dropped from the gesture instead, so it ends where it began and nothing is written for
      // it. The rectangle is reached again by dragging the edge that is outside the frame.
      const inverted = towards < 0 ? next[edge] > opposite : next[edge] < opposite;
      if (inverted) next[edge] = start[edge];
    }

    return next;
  }

  /**
   * Put a rectangle a gesture reached into the shape the adopter will write.
   *
   * @param {{x: number, y: number, w: number, h: number}} rect - The rectangle the gesture
   *   reached, in frame units.
   * @param {object} moved - A map from edge to true, holding only the edges that moved.
   * @returns {{x: number, y: number, w: number, h: number}} The rectangle to hold, which is the
   *   one it was given when the adopter states no shape of its own.
   */
  function coerceRect(rect, moved) {
    if (!settings.coerce) return rect;
    try {
      const value = settings.coerce(rect, moved);
      const next = {
        x: Number(value?.x),
        y: Number(value?.y),
        w: Number(value?.w),
        h: Number(value?.h),
      };
      return Object.values(next).every(Number.isFinite) ? next : rect;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to shape the rectangle:`, error);
      return rect;
    }
  }

  /**
   * Hold the rectangle a gesture has reached, without writing it.
   *
   * @param {object} edges - The four edges the gesture has reached.
   * @param {string[]} moved - The edges the gesture moves.
   * @returns {void}
   */
  function holdPending(edges, moved) {
    if (state.disposed) return;
    const map = {};
    for (const edge of moved) map[edge] = true;
    const rect = coerceRect(rectOf(edges), map);
    if (state.pending && sameRect(state.pending.rect, rect)) return;
    state.pending = { rect, moved: map };
    schedulePaint();
  }

  /**
   * Begin a drag on the body or on one handle.
   *
   * Nothing is written and nothing is held yet.
   *
   * @param {PointerEvent} event - The press that starts the drag.
   * @param {object} hit - What the press landed on, from `hitTest`.
   * @param {{x: number, y: number}} point - Pointer position in frame units.
   * @returns {void}
   */
  function startDrag(event, hit, point) {
    const model = displayModel();
    const rect = model?.rect;
    if (!rect) return;
    if (refuseLocked(hit.edges, gestureOf(hit.kind))) return;

    const start = edgesOf(rect);
    const grab = {};
    if (hit.handle) {
      for (const edge of hit.handle.edges) {
        // Measured from the edge as drawn rather than as held. An edge pinned to the frame
        // was picked up where it is on screen, so the offset that keeps it under the pointer
        // is that one, and dragging it is how a number the frame cannot show comes back.
        grab[edge] = model.box[edge] - (ACROSS.has(edge) ? point.x : point.y);
      }
    }

    state.drag = {
      pointerId: event.pointerId,
      kind: hit.kind,
      handle: hit.handle,
      edges: hit.edges,
      start,
      startPoint: point,
      grab,
    };
    state.hover = hit.handle?.key ?? "body";
    root.setPointerCapture?.(event.pointerId);
    schedulePaint();
  }

  /**
   * End a drag, releasing the pointer capture it holds.
   *
   * @param {boolean} commit - Write what the drag reached. False discards it and leaves the
   *   widgets as they were.
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
   * Move the whole rectangle by one step of the keyboard.
   *
   * @param {number} dx - Steps across.
   * @param {number} dy - Steps down.
   * @param {boolean} coarse - Take the coarse step rather than the fine one.
   * @returns {void}
   */
  function nudge(dx, dy, coarse) {
    if (!state.view) {
      setMessage("no frame to edit in yet");
      return;
    }
    const rect = currentRect();
    if (!rect) return;
    // An arrow press carries the rectangle whole, so it is asked about as the translation it is
    // rather than as a gesture on all four edges.
    if (refuseLocked(EDGE_ORDER, GESTURE.MOVE)) return;

    const step = coarse ? coarseStep() : settings.step;
    holdPending(moveEdges(edgesOf(rect), dx * step, dy * step, false), EDGE_ORDER);
  }

  /**
   * Whether the stand-in picture is what goes behind the rectangle.
   *
   * @returns {boolean} True where the backdrop has no picture to draw and a picture is what is
   *   missing, rather than a fault to report or a frame the adopter draws itself.
   */
  function standingIn() {
    const frame = state.frame;
    if (frame.image) return false;
    // A failure keeps the words that say so. It is the one state of the five that is a fault, and
    // a picture over it would say that nothing is wrong.
    if (frame.state === PREVIEW_STATE.FAILED) return false;
    // A `READY` answer stating a size and carrying no picture is an adopter whose own drawing is
    // the whole of the backdrop, such as a mask the rectangle already describes, and nothing
    // stands in for a picture that was never coming.
    if (frame.state === PREVIEW_STATE.READY) return !(frame.width > 0 && frame.height > 0);
    return true;
  }

  /**
   * Where the picture goes, whether or not there is one.
   *
   * @returns {{x: number, y: number, w: number, h: number}} The frame as drawn, or the whole area
   *   while there is no frame to fit into it, in element pixels.
   */
  function backdropRect() {
    const layout = state.layout;
    const view = state.view;
    if (view) {
      return { x: view.x0, y: view.y0, w: view.drawWidth, h: view.drawHeight };
    }
    return {
      x: layout.areaX0,
      y: layout.areaY0,
      w: layout.areaWidth,
      h: layout.areaHeight,
    };
  }

  /**
   * Draw the backdrop, and the words for a state that has words of its own.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {void}
   */
  function drawBackdrop(ctx, theme) {
    const view = state.view;
    const frame = state.frame;
    const rect = backdropRect();

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

    let stood = false;
    if (view && frame.image) {
      try {
        ctx.drawImage(frame.image, rect.x, rect.y, rect.w, rect.h);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw the backdrop picture:`, error);
      }
    } else if (standingIn()) {
      // Fitted into the frame rather than sizing it. The frame is the unit every number written
      // here is measured in, so a stand-in that carried its own size into it would change what a
      // stored number means between one run and the next.
      stood = drawStandIn(ctx, rect);
    }

    // A state's words are what is left where there is no picture standing in its place, so a
    // stand-in that has not arrived or could not be decoded leaves the words that were there
    // before there was one.
    const label = frame.image || stood ? "" : frame.label;
    if (label) {
      ctx.font = LABEL_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = frame.state === PREVIEW_STATE.FAILED ? theme.warning : theme.fgMuted;
      ctx.fillText(label, rect.x + rect.w / 2, rect.y + rect.h / 2, Math.max(1, rect.w - 8));
    }
  }

  /**
   * Draw whatever the adopter puts over the frame, between the backdrop and the rectangle.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @returns {Array<object>} The hover regions the layers answered, in element pixels.
   */
  function drawLayers(ctx, theme) {
    const view = state.view;
    if (!view || settings.layers.length === 0) return [];

    const regions = [];
    for (const layer of settings.layers) {
      // Saved and restored around each one, so a layer that leaves a transform, a clip or an
      // alpha behind cannot change how the rectangle above it is drawn.
      ctx.save();
      try {
        const answer = layer(ctx, view, theme);
        if (Array.isArray(answer)) regions.push(...answer);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw a layer:`, error);
      } finally {
        ctx.restore();
      }
    }
    return regions;
  }

  /**
   * Lay a wash over everything the rectangle does not cover.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} box - Drawn edges from `boxPixels`.
   * @returns {void}
   */
  function drawScrim(ctx, theme, box) {
    const view = state.view;
    const x1 = view.x0 + view.drawWidth;
    const y1 = view.y0 + view.drawHeight;

    ctx.save();
    ctx.globalAlpha = SCRIM_ALPHA;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(view.x0, view.y0, view.drawWidth, Math.max(0, box.top - view.y0));
    ctx.fillRect(view.x0, box.bottom, view.drawWidth, Math.max(0, y1 - box.bottom));
    ctx.fillRect(
      view.x0,
      box.top,
      Math.max(0, box.left - view.x0),
      Math.max(0, box.bottom - box.top),
    );
    ctx.fillRect(
      box.right,
      box.top,
      Math.max(0, x1 - box.right),
      Math.max(0, box.bottom - box.top),
    );
    ctx.restore();
  }

  /**
   * Draw the rectangle, one edge at a time, and its eight handles.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object} model - Model from `displayModel`.
   * @returns {void}
   */
  function drawRegion(ctx, theme, model) {
    const box = boxPixels(model);
    const locks = readLocks();
    const active = Boolean(state.drag);
    const base = active ? theme.accent : theme.fg;

    drawScrim(ctx, theme, box);

    ctx.lineWidth = active ? 2 : 1;
    for (const edge of EDGE_ORDER) {
      const across = ACROSS.has(edge);
      // Each edge is drawn on its own so an edge pinned to the frame carries the colour that says
      // the widget holds a number the frame cannot show.
      ctx.strokeStyle = model.pinned[edge] ? theme.warning : locks[edge] ? theme.fgMuted : base;
      ctx.beginPath();
      if (across) {
        const x = Math.round(edge === EDGE.LEFT ? box.left : box.right) + 0.5;
        ctx.moveTo(x, box.top);
        ctx.lineTo(x, box.bottom);
      } else {
        const y = Math.round(edge === EDGE.TOP ? box.top : box.bottom) + 0.5;
        ctx.moveTo(box.left, y);
        ctx.lineTo(box.right, y);
      }
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    const half = HANDLE_SIZE / 2;
    for (const handle of HANDLES) {
      const middle = handlePoint(box, handle);
      const locked = handle.edges.some((edge) => locks[edge]);
      // The hovered handle stays the marked one through the drag, since a press records it and
      // the pointer is not asked what it is over again until the gesture ends.
      const hovered = state.hover === handle.key;
      // A handle whose edge is locked is drawn muted, so a gesture that will be refused looks
      // refused before it is made.
      ctx.fillStyle = locked ? theme.fgMuted : hovered ? theme.accent : theme.fg;
      ctx.strokeStyle = theme.bg;
      ctx.fillRect(
        Math.round(middle.x - half),
        Math.round(middle.y - half),
        HANDLE_SIZE,
        HANDLE_SIZE,
      );
      ctx.strokeRect(
        Math.round(middle.x - half) + 0.5,
        Math.round(middle.y - half) + 0.5,
        HANDLE_SIZE - 1,
        HANDLE_SIZE - 1,
      );
    }
  }

  /**
   * The left half of the top footer line: the four numbers, as they will be written.
   *
   * @param {object|null} model - Model from `displayModel`.
   * @returns {string} Text to draw.
   */
  function footerReadout(model) {
    const rect = model?.rect ?? currentRect();
    if (!rect) return typeof settings.rect.read === "function" ? "no rectangle" : "";

    const parts = [
      `${formatNumber(rect.x)}, ${formatNumber(rect.y)}`,
      `${formatNumber(rect.w)} x ${formatNumber(rect.h)}`,
    ];
    if (state.drag?.modifiers?.aspect) parts.push("aspect");
    else if (state.drag?.modifiers?.snap) parts.push("snap");
    return parts.join("   ");
  }

  /**
   * The note the top footer line carries on the right.
   *
   * @param {object|null} model - Model from `displayModel`.
   * @returns {string} The note, empty when there is nothing to report.
   */
  function footerNote(model) {
    if (state.message) return state.message;

    const locks = readLocks();
    const linked = EDGE_ORDER.filter((edge) => locks[edge]).map((edge) => locks[edge]);
    const names = [...new Set(linked)];
    if (names.length === 1) return `${names[0]} is linked`;
    if (names.length > 1) return `${names.length} inputs are linked`;

    if (model?.negative) return "negative size";
    if (model && EDGE_ORDER.some((edge) => model.pinned[edge])) return "outside the frame";
    return "";
  }

  /**
   * What the rectangle means, in the adopter's own words.
   *
   * @param {object|null} model - Model from `displayModel`.
   * @returns {string} The line to draw.
   */
  function footerMeaning(model) {
    try {
      const rect = model?.rect ?? currentRect();
      const value =
        typeof settings.footer === "function" ? settings.footer(rect) : settings.footer;
      return typeof value === "string" ? value : "";
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the footer:`, error);
      return "";
    }
  }

  /**
   * The frame, in the words that say what a number written here is measured in.
   *
   * @returns {string} The line's right hand side: the frame's size, or the words for having none.
   */
  function footerFrame() {
    const frame = state.frame;
    // The slot that names the unit says when there is not one yet, since that is what refuses
    // every gesture and it is answered by the node running rather than by anything on screen.
    if (!(frame.width > 0) || !(frame.height > 0)) return NO_FRAME;
    return `${formatNumber(frame.width)}x${formatNumber(frame.height)}`;
  }

  /**
   * What the numbers are measured in, in the adopter's own words, for the footer's hover text.
   *
   * @param {object|null} model - Model from `displayModel`.
   * @returns {string} The sentence, empty where the adopter states none.
   */
  function footerHover(model) {
    try {
      const rect = model?.rect ?? currentRect();
      const value = typeof settings.hover === "function" ? settings.hover(rect) : settings.hover;
      return typeof value === "string" ? value : "";
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the hover text:`, error);
      return "";
    }
  }

  /**
   * Draw the two footer lines.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw into.
   * @param {object} theme - Theme tokens.
   * @param {object|null} model - Model from `displayModel`.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model) {
    const layout = state.layout;
    const middle = layout.footerY + FOOTER_HEIGHT / 2;
    const note = footerNote(model);
    // A layer's own regions go in ahead of the footer's band, since they sit over the picture
    // and the band is the whole width of what is left below it.
    const regions = [...state.layerRegions];

    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    // The glyph is drawn only while the stand-in is, since what a picture stands for is a
    // condition to point at and there is nothing to point at once the node's own picture is
    // there. Its region goes in first, so a pointer on the glyph is answered by it rather than by
    // the band it sits in.
    let glyphWidth = 0;
    if (standingIn()) {
      const box = drawIcon(
        ctx,
        STAND_IN_ICON,
        layout.areaX0,
        middle - ICON_SIZE / 2,
        ICON_SIZE,
        theme.warning,
      );
      regions.push({ ...box, title: standInTitle(state.frame.state) });
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    }

    let noteWidth = 0;
    if (note) {
      noteWidth = ctx.measureText(note).width;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.warning;
      ctx.fillText(note, layout.areaX1, middle);
    }

    const available = layout.areaWidth - glyphWidth - noteWidth - 8;
    if (available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = state.drag ? theme.fg : theme.fgMuted;
      ctx.fillText(footerReadout(model), layout.areaX0 + glyphWidth, middle, available);
    }

    const second = middle + FOOTER_HEIGHT;
    const frameText = footerFrame();
    let frameWidth = 0;
    ctx.fillStyle = theme.fgMuted;
    if (frameText) {
      frameWidth = ctx.measureText(frameText).width;
      ctx.textAlign = "right";
      ctx.fillText(frameText, layout.areaX1, second);
    }

    // The footer's own band is what the hover text is attached to, since what these numbers are
    // measured in is a fact about them rather than about anything drawn over the picture.
    const hover = footerHover(model);
    if (hover) {
      regions.push({
        x: layout.areaX0,
        y: layout.footerY,
        width: layout.areaWidth,
        height: FOOTER_HEIGHT * FOOTER_LINES,
        title: hover,
      });
    }
    titles.set(regions);

    const meaning = footerMeaning(model);
    const room = layout.areaWidth - frameWidth - 8;
    if (meaning && room > 12) {
      ctx.textAlign = "left";
      ctx.fillText(meaning, layout.areaX0, second, room);
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
    state.view = computeView(state.layout, state.frame);
    const layout = state.layout;
    if (layout.areaWidth <= 0 || layout.areaHeight <= 0) {
      // Nothing was drawn over the cleared canvas, so no glyph is under the pointer either.
      state.layerRegions = [];
      titles.set([]);
      return;
    }

    const theme = readTheme();
    drawBackdrop(ctx, theme);
    state.layerRegions = drawLayers(ctx, theme);

    const model = displayModel();
    if (model) drawRegion(ctx, theme, model);

    const view = state.view;
    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    if (view) {
      ctx.strokeRect(
        Math.round(view.x0) + 0.5,
        Math.round(view.y0) + 0.5,
        Math.max(1, Math.round(view.drawWidth) - 1),
        Math.max(1, Math.round(view.drawHeight) - 1),
      );
    } else {
      ctx.strokeRect(
        layout.areaX0 + 0.5,
        layout.areaY0 + 0.5,
        Math.max(1, layout.areaWidth - 1),
        Math.max(1, layout.areaHeight - 1),
      );
    }

    drawFooter(ctx, theme, model);

    if (document.activeElement === root) {
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
        console.error(`[${LOG_NAME}] Failed to draw the region editor:`, error);
      }
    });
  }

  /**
   * Read which modifiers an event is carrying.
   *
   * @param {PointerEvent|KeyboardEvent} event - Event to read.
   * @returns {{snap: boolean, aspect: boolean}} Which modifiers are held.
   */
  function modifiersOf(event) {
    return { snap: Boolean(event.shiftKey), aspect: Boolean(event.altKey) };
  }

  /**
   * What a tool is handed with a pointer event.
   *
   * @param {PointerEvent} event - Event to read.
   * @returns {{point: object, frame: object|null, view: object|null, modifiers: object}} The
   *   position in element pixels, the same position in frame units where there is a frame, the
   *   view a position converts through, and which modifiers are held.
   */
  function toolContext(event) {
    const point = localPoint(event);
    return { point, frame: framePoint(point), view: state.view, modifiers: modifiersOf(event) };
  }

  /**
   * Offer a pointer event to the tool, if there is one.
   *
   * @param {string} name - Which member to call, one of `pointerDown`, `pointerMove`,
   *   `pointerUp`.
   * @param {PointerEvent} event - Event to offer.
   * @returns {boolean} True when the tool took the event, which stops the rectangle from
   *   seeing it. A tool that throws is logged and treated as having refused, so the rectangle
   *   still works.
   */
  function offerToTool(name, event) {
    const handler = settings.tool?.[name];
    if (typeof handler !== "function") return false;
    try {
      return handler.call(settings.tool, event, toolContext(event)) === true;
    } catch (error) {
      console.error(`[${LOG_NAME}] The tool failed on ${name}:`, error);
      return false;
    }
  }

  /**
   * End a gesture the tool claimed, releasing the pointer capture the editor holds for it.
   *
   * @param {boolean} cancelled - True when the gesture was interrupted rather than finished,
   *   which the tool is told so it can drop what it was drawing.
   * @returns {void}
   */
  function endToolGesture(cancelled) {
    const pointerId = state.toolPointer;
    if (pointerId === null) return;
    state.toolPointer = null;
    if (root.hasPointerCapture?.(pointerId)) root.releasePointerCapture?.(pointerId);
    if (!cancelled) return;
    try {
      settings.tool?.cancel?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] The tool failed to cancel:`, error);
    }
  }

  /**
   * What the tool says the pointer looks like.
   *
   * @param {PointerEvent} event - Event to read.
   * @returns {string} A CSS cursor, empty when the tool states none and the rectangle decides.
   */
  function toolCursor(event) {
    const cursor = settings.tool?.cursor;
    if (typeof cursor !== "function") return "";
    try {
      const value = cursor.call(settings.tool, toolContext(event));
      return typeof value === "string" ? value : "";
    } catch (error) {
      console.error(`[${LOG_NAME}] The tool failed to state a cursor:`, error);
      return "";
    }
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
        console.error(`[${LOG_NAME}] Region editor input failed:`, error);
      }
    };
  }

  // The gestures are the same on every adopter. Dragging the body moves the rectangle, dragging a
  // handle moves the edges that handle carries, shift snaps a gesture to the coarse step and alt
  // holds a corner to the aspect it started with.
  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;

    root.focus?.({ preventScroll: true });
    if (!state.drag) commitPending();

    // First refusal, before the frame is even checked: a tool draws its own controls over the
    // frame and has to be reachable while the frame is a stand-in.
    if (offerToTool("pointerDown", event)) {
      state.toolPointer = event.pointerId;
      root.setPointerCapture?.(event.pointerId);
      return;
    }

    // The pointer default action is left alone throughout. Cancelling it would suppress the
    // mouse events that follow, which carry the graph snapshot that gives the gesture its undo
    // entry.
    const point = localPoint(event);
    const frame = framePoint(point);
    if (!frame) {
      setMessage("no frame to edit in yet");
      return;
    }

    const hit = hitTest(point);
    if (!hit) return;
    startDrag(event, hit, frame);
  };

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }

    // Offered before the rectangle's own hover, so a tool holding a gesture keeps every move and
    // a tool that is only watching can still light up whatever is under the pointer. Not while
    // the rectangle holds a gesture: that pointer is the rectangle's until it is released, and a
    // tool claiming a move part way through would strand the drag wherever it stood.
    if (!state.drag && offerToTool("pointerMove", event)) {
      root.style.cursor = toolCursor(event) || "default";
      if (state.hover !== null) {
        state.hover = null;
        schedulePaint();
      }
      return;
    }

    const point = localPoint(event);
    const drag = state.drag;

    if (drag) {
      // A button released over another window, or a capture the browser took away, ends the
      // gesture without a pointerup. Without this the rectangle would keep following an
      // unpressed pointer and commit a position nobody chose.
      if (!(event.buttons & 1)) {
        endDrag(false);
        return;
      }
      const frame = framePoint(point);
      if (!frame) return;

      const modifiers = modifiersOf(event);
      drag.modifiers = modifiers;
      if (drag.kind === "body") {
        root.style.cursor = "move";
        const dx = frame.x - drag.startPoint.x;
        const dy = frame.y - drag.startPoint.y;
        holdPending(moveEdges(drag.start, dx, dy, modifiers.snap), drag.edges);
        return;
      }
      root.style.cursor = drag.handle.cursor;
      holdPending(resizeEdges(drag.start, drag, frame, modifiers), drag.edges);
      return;
    }

    const hit = hitTest(point);
    const hover = hit ? (hit.handle?.key ?? "body") : null;
    const locked = hit ? Boolean(lockedName(hit.edges, gestureOf(hit.kind))) : false;
    root.style.cursor = !hit
      ? "default"
      : locked
        ? "not-allowed"
        : hit.kind === "body"
          ? "move"
          : hit.handle.cursor;

    if (hover !== state.hover) {
      state.hover = hover;
      schedulePaint();
    }
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    // A gesture the tool claimed ends here whether or not the tool takes the event, so the
    // capture the editor took for it is always given back.
    if (state.toolPointer !== null && event.pointerId === state.toolPointer) {
      const taken = offerToTool("pointerUp", event);
      endToolGesture(!taken);
      return;
    }
    if (!state.drag && offerToTool("pointerUp", event)) return;
    // Only the button and the pointer that start a gesture end one. A right button pressed and
    // released during a drag, or a second finger, would otherwise write the rectangle wherever
    // it stood at that instant and leave the first button still down with nothing following it.
    if (event.button !== 0) return;
    if (state.drag && event.pointerId !== state.drag.pointerId) return;
    endDrag(true);
  };

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  };

  // On the focused element the arrow keys move the rectangle, shift takes the coarse step, and
  // escape drops whatever is unfinished.
  const onKeyDown = (event) => {
    if (event.ctrlKey || event.metaKey) return;

    let handled = true;

    switch (event.key) {
      case "ArrowLeft":
        nudge(-1, 0, event.shiftKey);
        break;
      case "ArrowRight":
        nudge(1, 0, event.shiftKey);
        break;
      case "ArrowUp":
        nudge(0, -1, event.shiftKey);
        break;
      case "ArrowDown":
        nudge(0, 1, event.shiftKey);
        break;
      case "Delete":
      case "Backspace": {
        // Consumed whether or not it has anything to do. Left unhandled these reach ComfyUI's
        // own binding, which deletes the node the editor is drawn on.
        setMessage("nothing to delete here");
        break;
      }
      case "Escape": {
        // An unfinished gesture is dropped rather than written, which leaves the widgets
        // holding what they held before it started.
        endToolGesture(true);
        if (state.drag) endDrag(false);
        else if (state.pending) state.pending = null;
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
    if (!ARROW_KEYS.has(event.key)) return;
    if (state.drag) return;
    commitPending();
  };

  const onBlur = () => {
    // Focus can only leave mid-drag when the gesture has been interrupted, by another window
    // taking the pointer for example, so the drag is discarded rather than kept.
    endToolGesture(true);
    if (state.drag) endDrag(false);
    else commitPending();
    state.hover = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener(
    "pointercancel",
    guard(() => {
      endToolGesture(true);
      endDrag(false);
    }),
  );
  root.addEventListener(
    "lostpointercapture",
    guard(() => {
      endToolGesture(true);
      endDrag(false);
    }),
  );
  root.addEventListener(
    "pointerleave",
    guard(() => {
      // The tool is told whatever the rectangle's own hover is doing, since a tool drawing at
      // the pointer has to stop drawing there once the pointer is somewhere else.
      try {
        settings.tool?.leave?.();
      } catch (error) {
        console.error(`[${LOG_NAME}] The tool failed on leave:`, error);
      }
      if (state.drag || state.hover === null) return;
      state.hover = null;
      schedulePaint();
    }),
  );
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

  // The picture is drawn into a canvas, which takes literal colours, so a palette change repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the timers, observers and listeners the editor holds.
   *
   * @returns {void}
   */
  function dispose() {
    endToolGesture(true);
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    if (state.messageTimer) clearTimeout(state.messageTimer);
    if (state.retryTimer) clearTimeout(state.retryTimer);
    state.paintHandle = 0;
    state.messageTimer = 0;
    state.retryTimer = 0;
    titles.dispose();
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
  }

  loadFrame();
  // One picture serves every editor on the page, so this is the ask on the first of them and a
  // repaint on each one after it.
  loadPlaceholder().then(() => schedulePaint());
  schedulePaint();

  return {
    element: root,
    height: settings.height,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    schedulePaint,
    handleRectChanged,
    refresh,
    dispose,
  };
}
