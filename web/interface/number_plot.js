/**
 * A plot of lists of numbers on a node, with handles that drag up and down.
 *
 * `createNumberPlot` answers the panel and `show`, which draws series, handles and footer text.
 * A released handle is reported once, through `onCommit`.
 */

import { app } from "../../../scripts/app.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./icons.js";
import { captureWheel, elementPoint } from "./pointer.js";
import { roundDecimals } from "./python_arithmetic.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";

/** The palette token each series is drawn in, in order. */
export const SERIES_TOKENS = Object.freeze(["accent", "warning", "success", "fg"]);

// Height in node units the panel opens at, and the narrowest node it reads in.
const PANEL_HEIGHT = 170;
const PANEL_MIN_WIDTH = 240;

// Padding inside the panel and the gap between its rows, in CSS pixels.
const PADDING = 5;
const GAP = 3;

// Room around the plot for the axis labels and the handles, in CSS pixels.
const AXIS_GAP = 4;
const AXIS_BOTTOM = 18;
const PLOT_TOP = 7;
const PLOT_RIGHT = 9;
const PLOT_LEFT = 7;

// Font the axis labels are drawn in, at one CSS pixel per unit.
const LABEL_FONT = "9px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";

// Value ticks the axis aims for.
const TICKS = 4;

// Steps a dragged handle snaps to across the axis, and with Shift held.
const SNAP_STEPS = 100;
const FINE_STEPS = 1000;

// Handle radius, the distance a press reaches a handle from, and the spacing below which
// entries are drawn as a line without a dot each, in CSS pixels.
const HANDLE_RADIUS = 4.5;
const HIT_RADIUS = 9;
const DOT_SPACING = 7;

/**
 * A round step that divides a span into about `target` parts.
 *
 * @param {number} span - The span, above 0.
 * @param {number} target - Parts wanted.
 * @returns {number} 1, 2 or 5 times a power of ten.
 */
export function niceStep(span, target) {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const nice = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10;
  return nice * power;
}

/**
 * Decimal places a step needs to be written exactly.
 *
 * @param {number} step - A step from `niceStep`.
 * @returns {number} Places, 0 or more.
 */
function placesFor(step) {
  return Math.max(0, Math.min(12, -Math.floor(Math.log10(step) + 1e-9)));
}

/**
 * The value axis a set of series and handles is drawn on.
 *
 * @param {number[][]} lists - Every list of values to fit.
 * @returns {{low: number, high: number, step: number}} Bounds on round ticks.
 */
export function valueAxis(lists) {
  let low = Infinity;
  let high = -Infinity;
  for (const list of lists) {
    for (const value of list) {
      if (!Number.isFinite(value)) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  if (!Number.isFinite(low)) return { low: 0, high: 1, step: 0.25 };
  if (low === high) {
    const pad = low === 0 ? 1 : Math.abs(low) * 0.5;
    low -= pad;
    high += pad;
  }
  const step = niceStep(high - low, TICKS);
  return { low: Math.floor(low / step) * step, high: Math.ceil(high / step) * step, step };
}

/**
 * Where a model is drawn: the plot's box and the mapping between entries, values and pixels.
 *
 * @param {number} width - Canvas width, CSS pixels.
 * @param {number} height - Canvas height, CSS pixels.
 * @param {number} count - Entries along the axis.
 * @param {{low: number, high: number, step: number}} axis - The value axis.
 * @param {number} labelWidth - Room the value labels take, CSS pixels.
 * @returns {object} `left`, `right`, `top`, `bottom`, `xOf`, `yOf`, `valueAt` and `indexAt`.
 */
function layoutOf(width, height, count, axis, labelWidth) {
  const left = labelWidth + AXIS_GAP + PLOT_LEFT;
  const right = Math.max(left + 1, width - PLOT_RIGHT);
  const top = PLOT_TOP;
  const bottom = Math.max(top + 1, height - AXIS_BOTTOM);
  const span = axis.high - axis.low || 1;
  const xOf = (index) => (count > 1 ? left + (index / (count - 1)) * (right - left) : (left + right) / 2);
  const yOf = (value) => bottom - ((value - axis.low) / span) * (bottom - top);
  const valueAt = (y) => axis.low + ((bottom - y) / (bottom - top)) * span;
  const indexAt = (x) => (count > 1
    ? Math.max(0, Math.min(count - 1, Math.round(((x - left) / (right - left)) * (count - 1))))
    : 0);
  return { left, right, top, bottom, xOf, yOf, valueAt, indexAt, axis };
}

/**
 * Build a plot panel.
 *
 * @param {object} [options] - What the panel does.
 * @param {(handle: object, value: number) => void} [options.onCommit] - Called once when a handle
 *   is released at a new value. Without it no handle drags.
 * @param {string} [options.unit] - What one entry is called in the readout, `entry` by default.
 * @param {string} [options.logName] - What a failure is logged under.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   show: (model: object) => void, dispose: () => void}} The panel.
 */
export function createNumberPlot(options = {}) {
  const logName = options.logName || "WASNodeSuite.NumberPlot";
  const unit = options.unit || "entry";
  const onCommit = typeof options.onCommit === "function" ? options.onCommit : null;

  const root = document.createElement("div");
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    `gap:${GAP}px`,
    `padding:${PADDING}px`,
    "overflow:hidden",
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    "line-height:1.25",
    "border-radius:4px",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
    "user-select:none",
  ].join(";");

  const stage = document.createElement("div");
  stage.style.cssText = "flex:1 1 auto;min-height:40px;position:relative";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;"
    + "touch-action:none";
  stage.appendChild(canvas);

  const legend = document.createElement("div");
  legend.style.cssText = "flex:0 0 auto;display:flex;flex-wrap:wrap;column-gap:10px;row-gap:1px;"
    + "font-size:10px";

  const footer = document.createElement("div");
  footer.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:6px;font-size:10px";
  const summary = document.createElement("span");
  summary.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + "white-space:nowrap";
  const readout = document.createElement("span");
  readout.style.cssText = `flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;`
    + `white-space:nowrap;color:${themeVar("fgMuted")}`;
  const glyph = document.createElement("canvas");
  glyph.style.cssText = `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`;
  footer.append(summary, readout, glyph);
  root.append(stage, legend, footer);

  let model = { series: [], handles: [] };
  let layout = null;
  let hover = null;
  let drag = null;
  let paintHandle = 0;
  let disposed = false;

  /** The lists the value axis is fitted to. */
  const fitted = () => [
    ...model.series.map((series) => series.values ?? []),
    (model.handles ?? []).map((handle) => handle.value),
  ];

  /** Entries along the axis: the longest series. */
  const entryCount = () => Math.max(0, ...model.series.map((series) => series.values?.length ?? 0));

  /** Whether the handles take a drag. */
  const draggable = () => Boolean(onCommit && model.editable);

  /** The handle under a point, or null. */
  const handleAt = (point) => {
    if (!layout || !draggable()) return null;
    let best = null;
    let nearest = HIT_RADIUS;
    for (const handle of model.handles ?? []) {
      const distance = Math.hypot(layout.xOf(handle.index) - point.x, layout.yOf(handle.value) - point.y);
      if (distance <= nearest) {
        nearest = distance;
        best = handle;
      }
    }
    return best;
  };

  /** The value a handle shows: its own, or the one it is being dragged to. */
  const shownValue = (handle) => (drag && drag.handle === handle ? drag.value : handle.value);

  /** Write the readout for whatever is under the pointer. */
  const writeReadout = () => {
    if (drag) {
      readout.textContent = `${drag.handle.label ?? "value"}: ${drag.value}`;
      return;
    }
    if (hover?.handle) {
      readout.textContent = `${hover.handle.label ?? "value"}: ${hover.handle.value}`;
      return;
    }
    if (!hover || !Number.isFinite(hover.index)) {
      readout.textContent = "";
      return;
    }
    const total = entryCount();
    const parts = model.series
      .map((series) => {
        const value = series.values?.[hover.index];
        if (value === undefined) return null;
        return model.series.length > 1 ? `${series.name} ${value}` : String(value);
      })
      .filter(Boolean);
    readout.textContent = `${unit} ${hover.index + 1} of ${total}: ${parts.join(", ")}`;
  };

  /** Draw everything from the model, at the resolution the canvas is shown at. */
  const paint = () => {
    paintHandle = 0;
    if (disposed) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const ratio = surfaceRatio(canvas);
    const wide = Math.max(1, Math.round(width * ratio));
    const tall = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== wide || canvas.height !== tall) {
      canvas.width = wide;
      canvas.height = tall;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const theme = readTheme();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, wide, tall);
    ctx.scale(ratio, ratio);
    ctx.fillStyle = theme.bgDark;
    ctx.fillRect(0, 0, width, height);

    drawFidelityGlyph(glyph, model.claim ?? { icon: ICON.WARNING, detail: "fidelity not stated" });
    summary.textContent = model.summary ?? "";
    summary.style.color = model.blocking ? themeVar("warning") : themeVar("fg");
    writeReadout();

    const message = model.blocking || model.empty;
    const count = entryCount();
    if (message || count === 0) {
      layout = null;
      ctx.fillStyle = model.blocking ? theme.warning : theme.fgMuted;
      ctx.font = LABEL_FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(message || "nothing to draw", width / 2, height / 2, width - 12);
      return;
    }

    const axis = drag ? drag.axis : valueAxis(fitted());
    ctx.font = LABEL_FONT;
    const places = placesFor(axis.step);
    const ticks = [];
    for (let value = axis.low; value <= axis.high + axis.step * 1e-6; value += axis.step) {
      ticks.push(roundDecimals(value, places));
    }
    const labels = ticks.map((value) => value.toFixed(places));
    const labelWidth = Math.ceil(Math.max(...labels.map((text) => ctx.measureText(text).width)));
    layout = layoutOf(width, height, count, axis, labelWidth);

    // Value grid and labels.
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ticks.forEach((value, index) => {
      const y = Math.round(layout.yOf(value)) + 0.5;
      ctx.strokeStyle = theme.borderLight;
      ctx.globalAlpha = value === 0 ? 0.9 : 0.5;
      ctx.beginPath();
      ctx.moveTo(layout.left, y);
      ctx.lineTo(layout.right, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(labels[index], layout.left - AXIS_GAP - PLOT_LEFT, y);
    });

    // Entry numbers at both ends of the axis, below where a handle can reach.
    ctx.textBaseline = "top";
    ctx.fillStyle = theme.fgMuted;
    ctx.textAlign = "center";
    ctx.fillText("1", layout.xOf(0), layout.bottom + HANDLE_RADIUS + 3);
    if (count > 1) ctx.fillText(String(count), layout.xOf(count - 1), layout.bottom + HANDLE_RADIUS + 3);

    // The entry under the pointer.
    if (hover && Number.isFinite(hover.index) && !drag) {
      const x = Math.round(layout.xOf(hover.index)) + 0.5;
      ctx.strokeStyle = theme.fgMuted;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, layout.top);
      ctx.lineTo(x, layout.bottom);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Each series as a line, with a dot per entry where they are far enough apart.
    const spacing = count > 1 ? (layout.right - layout.left) / (count - 1) : Infinity;
    model.series.forEach((series, position) => {
      const values = series.values ?? [];
      const colour = theme[series.token ?? SERIES_TOKENS[position % SERIES_TOKENS.length]] ?? theme.accent;
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let drawing = false;
      values.forEach((value, index) => {
        if (!Number.isFinite(value)) {
          drawing = false;
          return;
        }
        const x = layout.xOf(index);
        const y = layout.yOf(value);
        if (drawing) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        drawing = true;
      });
      ctx.stroke();
      if (spacing >= DOT_SPACING || values.length === 1) {
        values.forEach((value, index) => {
          if (!Number.isFinite(value)) return;
          ctx.beginPath();
          ctx.arc(layout.xOf(index), layout.yOf(value), 1.8, 0, Math.PI * 2);
          ctx.fill();
        });
      }
      if (hover && Number.isFinite(hover.index) && Number.isFinite(values[hover.index]) && !drag) {
        ctx.beginPath();
        ctx.arc(layout.xOf(hover.index), layout.yOf(values[hover.index]), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Handles: filled where they drag, hollow where they only mark a value.
    const editable = draggable();
    for (const handle of model.handles ?? []) {
      const value = shownValue(handle);
      const x = layout.xOf(handle.index);
      const y = Math.max(layout.top, Math.min(layout.bottom, layout.yOf(value)));
      const active = (drag && drag.handle === handle) || hover?.handle === handle;
      ctx.beginPath();
      ctx.arc(x, y, active ? HANDLE_RADIUS + 1 : HANDLE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = editable ? theme.fg : theme.bgDark;
      ctx.strokeStyle = active ? theme.accent : theme.fg;
      ctx.lineWidth = active ? 2 : 1.25;
      if (editable) ctx.fill();
      ctx.stroke();
    }
  };

  /** Repaint on the next frame, once however many times it is asked. */
  const schedule = () => {
    if (disposed || paintHandle) return;
    paintHandle = requestAnimationFrame(paint);
  };

  /**
   * Show a model.
   *
   * @param {object} next - `series` as `{name, values, token}`, `handles` as `{index, value,
   *   label, key}`, `editable`, `summary`, `legend` as `{text, token}`, `claim` as
   *   `{icon, detail}`, and `blocking` or `empty` text drawn in place of the plot.
   * @returns {void}
   */
  const show = (next) => {
    if (disposed) return;
    model = { series: [], handles: [], ...(next ?? {}) };
    if (drag && !(model.handles ?? []).some((handle) => handle.key === drag.handle.key)) drag = null;
    if (drag) drag.handle = model.handles.find((handle) => handle.key === drag.handle.key);
    legend.replaceChildren();
    for (const entry of model.legend ?? []) {
      const chip = document.createElement("span");
      chip.style.cssText = "display:inline-flex;align-items:center;gap:4px;white-space:nowrap";
      const swatch = document.createElement("span");
      swatch.style.cssText = `width:8px;height:2px;background:${themeVar(entry.token ?? "accent")}`;
      const text = document.createElement("span");
      text.textContent = entry.text;
      chip.append(swatch, text);
      legend.appendChild(chip);
    }
    legend.style.display = (model.legend ?? []).length ? "flex" : "none";
    schedule();
  };

  /** End a drag, writing the value it reached when asked to. */
  const endDrag = (commit) => {
    if (!drag) return;
    const finished = drag;
    drag = null;
    canvas.style.cursor = "default";
    try {
      if (canvas.hasPointerCapture?.(finished.pointerId)) canvas.releasePointerCapture(finished.pointerId);
    } catch (error) {
      console.error(`[${logName}] Failed to release the pointer:`, error);
    }
    if (commit && finished.value !== finished.handle.value) onCommit(finished.handle, finished.value);
    schedule();
  };

  /** The value a pointer height stands for while dragging, snapped to the axis. */
  const draggedValue = (y, fine) => {
    const axis = drag.axis;
    const raw = layout.valueAt(Math.max(layout.top, Math.min(layout.bottom, y)));
    const snap = niceStep(axis.high - axis.low, fine ? FINE_STEPS : SNAP_STEPS);
    const places = placesFor(snap);
    return roundDecimals(Math.round(raw / snap) * snap, places);
  };

  /** Wrap a handler so a failure is logged rather than thrown into the graph. */
  const guard = (handler) => (event) => {
    try {
      handler(event);
    } catch (error) {
      console.error(`[${logName}] The plot failed to handle a pointer:`, error);
    }
  };

  canvas.addEventListener("pointerdown", guard((event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    const point = elementPoint(canvas, event);
    const handle = handleAt(point);
    if (!handle) return;
    drag = {
      handle,
      pointerId: event.pointerId,
      axis: layout.axis,
      value: handle.value,
    };
    canvas.setPointerCapture?.(event.pointerId);
    canvas.style.cursor = "grabbing";
    schedule();
  }));

  canvas.addEventListener("pointermove", guard((event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }
    const point = elementPoint(canvas, event);
    if (drag) {
      if (!(event.buttons & 1)) {
        endDrag(false);
        return;
      }
      const value = draggedValue(point.y, event.shiftKey);
      if (value !== drag.value) {
        drag.value = value;
        schedule();
      }
      return;
    }
    const handle = handleAt(point);
    const index = layout ? layout.indexAt(point.x) : null;
    canvas.style.cursor = handle ? "grab" : "default";
    if (hover?.handle !== handle || hover?.index !== index) {
      hover = { handle, index };
      schedule();
    }
  }));

  canvas.addEventListener("pointerup", guard((event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    endDrag(true);
  }));
  canvas.addEventListener("pointercancel", guard(() => endDrag(false)));
  canvas.addEventListener("lostpointercapture", guard(() => endDrag(false)));
  canvas.addEventListener("pointerleave", guard(() => {
    if (drag || !hover) return;
    hover = null;
    schedule();
  }));
  root.addEventListener("contextmenu", (event) => event.preventDefault());
  const releaseWheel = captureWheel(root);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedule());
    observer.observe(stage);
  }
  const unwatchRatio = watchSurfaceRatio(canvas, schedule);
  const unwatchTheme = onThemeChange(schedule);

  /**
   * Release the observers, listeners and the frame the panel holds.
   *
   * @returns {void}
   */
  const dispose = () => {
    disposed = true;
    drag = null;
    if (paintHandle) cancelAnimationFrame(paintHandle);
    paintHandle = 0;
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchTheme?.();
    releaseWheel();
  };

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    show,
    dispose,
  };
}
