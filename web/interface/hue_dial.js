/**
 * The wheel a hue rotation is set on, drawn on the node.
 *
 * The outer ring holds the hues as they arrive, the inner ring where the turn sends them.
 * Dragging either one writes the turn to the node's widget.
 */

import { captureWheel, elementPoint } from "./pointer.js";
import { withGraphChange } from "./region.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme } from "./theme.js";

const LOG_NAME = "WASNodeSuite.HueDial";

// Height of the appended widget in node units, and the narrowest the wheel stays usable in.
const DEFAULT_HEIGHT = 168;
const DEFAULT_MIN_WIDTH = 200;

// Wedges each ring is drawn from. 120 puts one every three degrees, which reads as continuous.
const WEDGES = 120;

// Room left around the wheel, and the space between the two rings, in layout pixels.
const MARGIN = 10;
const GAP = 3;

// Each ring's thickness as a share of the wheel's radius.
const OUTER_SHARE = 0.22;
const INNER_SHARE = 0.16;

// The handle, in layout pixels.
const HANDLE_RADIUS = 5;
const HANDLE_LINE = 2;

// Text sizes for the turn and the angle beneath it, in layout pixels.
const VALUE_SIZE = 17;
const LABEL_SIZE = 10;

// A full turn in degrees, and the sixths of the wheel the readout names.
const DEGREES = 360;
const NAMED_TURNS = [
  [0, "no change"],
  [1 / 3, "red to green"],
  [1 / 2, "opposite colours"],
  [2 / 3, "red to blue"],
  [1, "no change"],
];

// How near a turn has to be to one of those to be named, as a fraction of the wheel.
const NAMED_SLACK = 0.004;

/**
 * One widget of a node, by name.
 *
 * @param {object} node - The node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function findWidget(node, name) {
  return (node?.widgets || []).find(widget => widget?.name === name) || null;
}

/**
 * A number held inside a range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} low - Lower bound.
 * @param {number} high - Upper bound.
 * @returns {number} The value, held inside the bounds.
 */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * What a turn does, in words, for the turns that have a name.
 *
 * @param {number} turn - Fraction of the wheel.
 * @returns {string} The name, or an empty string for a turn between two of them.
 */
function nameFor(turn) {
  const found = NAMED_TURNS.find(([at]) => Math.abs(turn - at) <= NAMED_SLACK);
  return found ? found[1] : "";
}

/**
 * Put a hue wheel on a node, wired to the widget holding the turn.
 *
 * @param {object} node - The node being created.
 * @param {object} [options] - Everything below.
 * @param {string} [options.widgetName] - Widget the wheel reads and writes. `hue_shift`.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {number} [options.minWidth] - The narrowest the wheel is drawn in, in node units.
 * @param {number} [options.step] - What a written turn is rounded to. `0.001`.
 * @returns {{element: HTMLElement, height: number, minWidth: number, refresh: () => void,
 *   dispose: () => void}} The element to hand to `appendInterfaceWidget`, the height it was
 *   built for, the narrowest it draws in, a repaint, and teardown.
 */
export function createHueDial(node, options = {}) {
  const widgetName = options.widgetName || "hue_shift";
  const height = Number(options.height) > 0 ? Number(options.height) : DEFAULT_HEIGHT;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : DEFAULT_MIN_WIDTH;
  const step = Number(options.step) > 0 ? Number(options.step) : 0.001;

  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${height - MARGIN * 2}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
    "cursor:grab",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  const state = { dragging: false, paintHandle: 0, disposed: false };

  /**
   * The turn the widget holds.
   *
   * @returns {number} A fraction of the wheel, from 0 to 1.
   */
  function readTurn() {
    const widget = findWidget(node, widgetName);
    const value = Number(widget?.value);
    return Number.isFinite(value) ? clamp(value, 0, 1) : 0;
  }

  /**
   * Write a turn onto the widget, as one undo step.
   *
   * @param {number} turn - Fraction of the wheel, clamped and rounded to `step`.
   * @returns {boolean} True where the widget moved.
   */
  function writeTurn(turn) {
    const widget = findWidget(node, widgetName);
    if (!widget) return false;
    const next = clamp(Math.round(turn / step) * step, 0, 1);
    if (Math.abs(Number(widget.value) - next) < step / 2) return false;
    withGraphChange(() => {
      widget.value = next;
      widget.callback?.(next);
    });
    node.setDirtyCanvas?.(true, true);
    return true;
  }

  /**
   * Where the wheel sits in the element, in layout pixels.
   *
   * @returns {{cx: number, cy: number, radius: number}} Centre and outer radius.
   */
  function layout() {
    const width = root.clientWidth || minWidth;
    const tall = root.clientHeight || height;
    const radius = Math.max(24, Math.min(width, tall) / 2 - MARGIN);
    return { cx: width / 2, cy: tall / 2, radius };
  }

  /**
   * One ring of hue wedges.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {object} place - Centre and radii, as `{cx, cy, outer, inner}`.
   * @param {number} offset - Turn added to each wedge's own hue.
   * @returns {void}
   */
  function ring(ctx, place, offset) {
    const sweep = (Math.PI * 2) / WEDGES;
    for (let index = 0; index < WEDGES; index += 1) {
      const turn = index / WEDGES;
      // Twelve o'clock is a hue of zero and the wheel runs clockwise, as a colour picker does.
      const from = turn * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(place.cx, place.cy, place.outer, from, from + sweep + 0.01);
      ctx.arc(place.cx, place.cy, place.inner, from + sweep + 0.01, from, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${((turn + offset) % 1) * DEGREES} 85% 52%)`;
      ctx.fill();
    }
  }

  /**
   * Draw the whole wheel.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const theme = readTheme();
    const { cx, cy, radius } = layout();
    const width = root.clientWidth || minWidth;
    const tall = root.clientHeight || height;
    const ratio = surfaceRatio(root);
    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.max(1, Math.round(tall * ratio));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, tall);

    const turn = readTurn();
    const outerThick = radius * OUTER_SHARE;
    const innerThick = radius * INNER_SHARE;
    ring(ctx, { cx, cy, outer: radius, inner: radius - outerThick }, 0);
    ring(ctx, {
      cx,
      cy,
      outer: radius - outerThick - GAP,
      inner: radius - outerThick - GAP - innerThick,
    }, turn);

    // The handle sits on the outer ring at the turn, which is where a hue of zero lands.
    const angle = turn * Math.PI * 2 - Math.PI / 2;
    const reach = radius - outerThick - GAP - innerThick;
    ctx.strokeStyle = theme.fg;
    ctx.lineWidth = HANDLE_LINE;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * reach, cy + Math.sin(angle) * reach);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + Math.cos(angle) * (radius + 1), cy + Math.sin(angle) * (radius + 1),
            HANDLE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = theme.fg;
    ctx.fill();

    ctx.fillStyle = theme.fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = `600 ${VALUE_SIZE}px sans-serif`;
    ctx.fillText(turn.toFixed(3), cx, cy + VALUE_SIZE / 3);
    ctx.fillStyle = theme.fgMuted;
    ctx.font = `${LABEL_SIZE}px sans-serif`;
    ctx.fillText(`${Math.round(turn * DEGREES)} degrees`, cx, cy + VALUE_SIZE / 3 + LABEL_SIZE + 3);
    const named = nameFor(turn);
    if (named) {
      ctx.fillText(named, cx, cy - VALUE_SIZE / 3 - LABEL_SIZE);
    }
  }

  /**
   * Repaint once, on the next frame.
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
        console.error(`[${LOG_NAME}] Failed to draw the hue wheel:`, error);
      }
    });
  }

  /**
   * The turn one pointer position stands for.
   *
   * @param {PointerEvent} event - Event to read.
   * @returns {number} A fraction of the wheel, from 0 to 1.
   */
  function turnAt(event) {
    const point = elementPoint(root, event);
    const { cx, cy } = layout();
    const angle = Math.atan2(point.x - cx, cy - point.y);
    return ((angle / (Math.PI * 2)) + 1) % 1;
  }

  /**
   * Start a drag, and set the turn the pointer went down on.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerDown(event) {
    if (event.button !== 0) return;
    state.dragging = true;
    root.style.cursor = "grabbing";
    root.setPointerCapture?.(event.pointerId);
    writeTurn(turnAt(event));
    schedulePaint();
    event.stopPropagation();
    event.preventDefault();
  }

  /**
   * Follow the pointer while the wheel is being turned.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerMove(event) {
    if (!state.dragging) return;
    writeTurn(turnAt(event));
    schedulePaint();
    event.stopPropagation();
  }

  /**
   * End the drag.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerUp(event) {
    if (!state.dragging) return;
    state.dragging = false;
    root.style.cursor = "grab";
    root.releasePointerCapture?.(event.pointerId);
    schedulePaint();
    event.stopPropagation();
  }

  /**
   * Nudge the turn by a wheel notch.
   *
   * @param {WheelEvent} event - Wheel event.
   * @returns {boolean} True where the notch moved the dial.
   */
  function onWheel(event) {
    if (!event.deltaY) return false;
    const notch = event.deltaY > 0 ? -step * 10 : step * 10;
    // At either end of the turn the dial cannot move, and the gesture is the graph's.
    if (!writeTurn(readTurn() + notch)) return false;
    schedulePaint();
    return true;
  }

  root.addEventListener("pointerdown", onPointerDown);
  root.addEventListener("pointermove", onPointerMove);
  root.addEventListener("pointerup", onPointerUp);
  root.addEventListener("pointercancel", onPointerUp);
  const releaseWheel = captureWheel(root, onWheel);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here. The two answer different events: the observer answers a
  // node that was resized or collapsed, this answers the same box drawn at another size.
  const unwatch = watchSurfaceRatio(root, schedulePaint);

  // The wheel is drawn into a canvas, which takes literal colours, so a palette change repaints.
  const unwatchTheme = onThemeChange(schedulePaint);

  // Typing in the number widget moves the wheel with it.
  const widget = findWidget(node, widgetName);
  if (widget) {
    const original = widget.callback;
    widget.callback = function (...args) {
      const result = original?.apply(this, args);
      schedulePaint();
      return result;
    };
  }

  schedulePaint();

  return {
    element: root,
    height,
    minWidth,
    refresh: schedulePaint,
    dispose() {
      state.disposed = true;
      if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
      state.paintHandle = 0;
      root.removeEventListener("pointerdown", onPointerDown);
      root.removeEventListener("pointermove", onPointerMove);
      root.removeEventListener("pointerup", onPointerUp);
      root.removeEventListener("pointercancel", onPointerUp);
      releaseWheel();
      observer?.disconnect();
      unwatch?.();
      unwatchTheme?.();
    },
  };
}
