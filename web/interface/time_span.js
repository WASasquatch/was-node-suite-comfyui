/**
 * A window of time drawn as a strip, with a handle at each end.
 *
 * The window is read and written through an accessor rather than by widget name. Times are
 * seconds and the axis begins at zero.
 */

import { captureWheel } from "./pointer.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.TimeSpan";

// Height of the strip in CSS pixels, above the scale and the readout.
const STRIP_HEIGHT = 34;

// How wide a handle is to grab, in CSS pixels either side of the edge it sits on.
const GRAB = 7;

// The largest backing store the strip is given, whatever a stray layout measures.
const MAX_EDGE = 4096;

// Ticks drawn along the axis, at most.
const MAX_TICKS = 9;

// The steps a tick spacing is chosen from, in seconds.
const TICK_STEPS = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];

// What a gesture is doing.
const GRIP = { NONE: "", START: "start", END: "end", WHOLE: "whole" };

/**
 * The tick spacing for an axis, in seconds.
 *
 * @param {number} length - Axis length in seconds.
 * @returns {number} The spacing, one of `TICK_STEPS`.
 */
function tickStep(length) {
  const wanted = Math.max(length, 1e-6) / MAX_TICKS;
  return TICK_STEPS.find((step) => step >= wanted) ?? TICK_STEPS[TICK_STEPS.length - 1];
}

/**
 * A time written for the scale under the strip.
 *
 * @param {number} seconds - The time.
 * @param {number} step - The tick spacing, which decides the decimals.
 * @returns {string} The time with its unit.
 */
function timeLabel(seconds, step) {
  const places = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return `${seconds.toFixed(places)}s`;
}

/**
 * Stop pointer and wheel events at an element.
 *
 * @param {HTMLElement} element - The element to stop them at.
 * @returns {() => void} Releases the wheel listener.
 */
function claimPointer(element) {
  // Stopped as the event bubbles out, so the strip's own handlers run first and the node's
  // widget grid, which forwards to the graph canvas, never sees it.
  for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel"]) {
    element.addEventListener(type, (event) => event.stopPropagation());
  }
  // The strip scrolls nothing of its own, so it takes every wheel gesture over it and the graph
  // zooms from the canvas around the node.
  return captureWheel(element);
}

/**
 * Build a strip that sets a window of time by dragging.
 *
 * @param {object} node - The node the strip is drawn on.
 * @param {object} options - `span.read` answers `{start, frames, fps, length}` in seconds and
 *   frames, `length` being how far the axis runs. `span.write` takes `{start, frames}` and puts
 *   them on the node's widgets. `label` is the word drawn above the strip.
 * @returns {{element: HTMLElement, refresh: Function, dispose: Function}} The strip, its
 *   repaint, and its teardown.
 */
export function createTimeSpanPanel(node, options = {}) {
  const read = typeof options.span?.read === "function" ? options.span.read : () => null;
  const write = typeof options.span?.write === "function" ? options.span.write : () => {};
  const label = String(options.label ?? "capture");

  const root = document.createElement("div");
  root.className = "was-time-span";
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "display:flex",
    "flex-direction:column",
    "gap:4px",
    "padding:6px 8px",
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    "line-height:1.4",
  ].join(";");

  const heading = document.createElement("div");
  heading.textContent = label;
  heading.style.cssText = `font-weight:600;flex:0 0 auto;color:${themeVar("fgMuted")}`;
  root.appendChild(heading);

  const frame = document.createElement("div");
  frame.style.cssText = `position:relative;height:${STRIP_HEIGHT}px;flex:0 0 auto`;
  root.appendChild(frame);

  const strip = document.createElement("canvas");
  strip.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;"
    + "border-radius:3px;touch-action:none";
  frame.appendChild(strip);

  const readout = document.createElement("div");
  readout.style.cssText = "display:flex;justify-content:space-between;gap:12px;flex:0 0 auto";
  root.appendChild(readout);

  const left = document.createElement("span");
  left.style.cssText = `color:${themeVar("fg")}`;
  const right = document.createElement("span");
  right.style.cssText = `white-space:nowrap;color:${themeVar("fgMuted")}`;
  readout.append(left, right);

  const releaseWheel = claimPointer(root);

  let disposed = false;
  let grip = GRIP.NONE;
  let hover = GRIP.NONE;
  let grabbedAt = 0;
  let grabbedStart = 0;

  /**
   * The window, with every figure sane.
   *
   * @returns {{start: number, frames: number, fps: number, length: number, end: number}|null}
   *   The window, or null where the accessor answered nothing.
   */
  const current = () => {
    let value = null;
    try {
      value = read();
    } catch (error) {
      console.error(`[${LOG_NAME}] The span accessor failed:`, error);
      return null;
    }
    if (!value) return null;
    const fps = Math.max(0.01, Number(value.fps) || 24);
    const frames = Math.max(1, Math.round(Number(value.frames) || 1));
    const start = Math.max(0, Number(value.start) || 0);
    const end = start + (frames - 1) / fps;
    const length = Math.max(Number(value.length) || 0, end, 1 / fps);
    return { start, frames, fps, length, end };
  };

  /**
   * Where a time sits across the strip.
   *
   * @param {number} seconds - The time.
   * @param {number} length - Axis length in seconds.
   * @param {number} width - Strip width in CSS pixels.
   * @returns {number} The position in CSS pixels.
   */
  const across = (seconds, length, width) => (seconds / length) * width;

  /**
   * Draw the axis, the window and its two handles.
   *
   * @returns {void}
   */
  const paint = () => {
    if (disposed) return;
    const theme = readTheme();

    const window = current();
    // Layout pixels times the ratio, which already holds the graph's zoom; both sides shrink
    // together past the largest edge so the strip keeps its shape.
    const layoutWide = Math.max(1, strip.clientWidth);
    const layoutTall = Math.max(1, strip.clientHeight);
    const wanted = surfaceRatio(strip);
    const ratio = wanted * Math.min(1, MAX_EDGE / (Math.max(layoutWide, layoutTall) * wanted));
    const wide = Math.max(1, Math.round(layoutWide * ratio));
    const tall = Math.max(1, Math.round(layoutTall * ratio));
    if (strip.width !== wide) strip.width = wide;
    if (strip.height !== tall) strip.height = tall;

    const pen = strip.getContext("2d");
    if (!pen) return;
    pen.clearRect(0, 0, wide, tall);
    pen.fillStyle = theme.bg;
    pen.fillRect(0, 0, wide, tall);

    if (!window) {
      left.textContent = "no window";
      right.textContent = "";
      return;
    }

    const width = layoutWide;
    const step = tickStep(window.length);
    pen.strokeStyle = theme.borderLight;
    pen.lineWidth = Math.max(1, Math.round(ratio));
    pen.fillStyle = theme.fgMuted;
    pen.font = `${9 * ratio}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace`;
    pen.textBaseline = "bottom";
    for (let mark = 0; mark <= window.length + 1e-9; mark += step) {
      const x = Math.round(across(mark, window.length, width) * ratio) + 0.5;
      pen.beginPath();
      pen.moveTo(x, 0);
      pen.lineTo(x, tall);
      pen.stroke();
      pen.textAlign = mark === 0 ? "left" : "center";
      pen.fillText(timeLabel(mark, step), Math.min(x + 2 * ratio, wide - 2 * ratio), tall - 2 * ratio);
    }

    // The window itself, and a rule per captured frame while they are far enough apart to read.
    const x0 = across(window.start, window.length, width) * ratio;
    const x1 = across(window.end, window.length, width) * ratio;
    pen.fillStyle = theme.accentBg;
    pen.fillRect(x0, 0, Math.max(x1 - x0, ratio), tall);

    const gap = (x1 - x0) / Math.max(1, window.frames - 1);
    if (window.frames > 1 && gap >= 3 * ratio) {
      pen.strokeStyle = theme.accent;
      pen.globalAlpha = 0.45;
      pen.lineWidth = Math.max(1, Math.round(ratio));
      for (let index = 0; index < window.frames; index++) {
        const x = Math.round(x0 + gap * index) + 0.5;
        pen.beginPath();
        pen.moveTo(x, tall * 0.25);
        pen.lineTo(x, tall * 0.75);
        pen.stroke();
      }
      pen.globalAlpha = 1;
    }

    for (const [edge, x] of [[GRIP.START, x0], [GRIP.END, x1]]) {
      pen.fillStyle = hover === edge || grip === edge ? theme.accentHover : theme.accent;
      pen.fillRect(Math.max(0, Math.min(x - 1.5 * ratio, wide - 3 * ratio)), 0, 3 * ratio, tall);
    }

    // The length a video saver makes of these frames, which is one frame's worth longer than
    // the gap between the first and the last.
    const seconds = window.frames / window.fps;
    left.textContent = `${window.start.toFixed(2)}s to ${window.end.toFixed(2)}s`;
    right.textContent = `${window.frames} frame${window.frames === 1 ? "" : "s"} `
      + `at ${window.fps.toFixed(2)} fps, ${seconds.toFixed(2)}s`;
  };

  /**
   * The time a pointer is over.
   *
   * @param {PointerEvent} event - The pointer event.
   * @param {object} window - The window `current` answered.
   * @returns {number} The time in seconds, held to the axis.
   */
  const timeAt = (event, window) => {
    const box = strip.getBoundingClientRect();
    const part = (event.clientX - box.left) / Math.max(1, box.width);
    return Math.max(0, Math.min(1, part)) * window.length;
  };

  /**
   * Which part of the window a pointer is over.
   *
   * @param {PointerEvent} event - The pointer event.
   * @param {object} window - The window `current` answered.
   * @returns {string} One of `GRIP`.
   */
  const gripAt = (event, window) => {
    const box = strip.getBoundingClientRect();
    const width = box.width || 1;
    const x = event.clientX - box.left;
    if (Math.abs(x - across(window.start, window.length, width)) <= GRAB) return GRIP.START;
    if (Math.abs(x - across(window.end, window.length, width)) <= GRAB) return GRIP.END;
    const inside = x > across(window.start, window.length, width)
      && x < across(window.end, window.length, width);
    return inside ? GRIP.WHOLE : GRIP.NONE;
  };

  /**
   * Put a window on the node and repaint.
   *
   * @param {number} start - Where the window begins, in seconds.
   * @param {number} frames - How many frames it holds.
   * @returns {void}
   */
  const commit = (start, frames) => {
    try {
      write({ start: Math.max(0, start), frames: Math.max(1, Math.round(frames)) });
    } catch (error) {
      console.error(`[${LOG_NAME}] The span accessor failed to write:`, error);
    }
    node.setDirtyCanvas?.(true, true);
    paint();
  };

  strip.addEventListener("pointerdown", (event) => {
    const window = current();
    if (!window || event.button !== 0) return;
    grip = gripAt(event, window);
    if (grip === GRIP.NONE) return;
    grabbedAt = timeAt(event, window);
    grabbedStart = window.start;
    strip.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    paint();
  });

  strip.addEventListener("pointermove", (event) => {
    const window = current();
    if (!window) return;
    if (grip === GRIP.NONE) {
      const over = gripAt(event, window);
      if (over !== hover) {
        hover = over;
        strip.style.cursor = over === GRIP.WHOLE ? "grab"
          : over === GRIP.NONE ? "default" : "ew-resize";
        paint();
      }
      return;
    }

    const at = timeAt(event, window);
    if (grip === GRIP.START) {
      const held = Math.min(at, window.end);
      commit(held, Math.round((window.end - held) * window.fps) + 1);
    } else if (grip === GRIP.END) {
      const held = Math.max(at, window.start);
      commit(window.start, Math.round((held - window.start) * window.fps) + 1);
    } else {
      commit(grabbedStart + (at - grabbedAt), window.frames);
    }
  });

  const release = (event) => {
    if (grip === GRIP.NONE) return;
    grip = GRIP.NONE;
    strip.releasePointerCapture?.(event.pointerId);
    paint();
  };
  strip.addEventListener("pointerup", release);
  strip.addEventListener("pointercancel", release);
  strip.addEventListener("pointerleave", () => {
    if (grip !== GRIP.NONE || hover === GRIP.NONE) return;
    hover = GRIP.NONE;
    paint();
  });

  const stopWatchingRatio = watchSurfaceRatio(strip, () => paint());
  // The strip is a canvas, which takes literal colours, so a palette change repaints it.
  const stopTheme = onThemeChange(() => paint());
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => paint()) : null;
  observer?.observe(frame);
  paint();

  return {
    element: root,
    refresh: paint,
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel();
      observer?.disconnect();
      stopWatchingRatio?.();
      stopTheme?.();
    },
  };
}
