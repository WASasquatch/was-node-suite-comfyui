/**
 * The layer list an arrangement is set on, drawn on the node.
 *
 * Rows run front of the stack first, each carrying the layer's thumbnail, its placement, a
 * visibility toggle, and dragging the row moves it through the stack.
 */

import { app } from "../../../scripts/app.js";
import { ICON, iconTitle } from "./icons.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { PREVIEW_STATE, fetchInputPreview } from "./preview.js";
import { statusColour } from "./report_panel.js";
import { withGraphChange } from "./region.js";
import { onNodeFinished, onRunEnded } from "./run_events.js";
import { RUN_LABELS, fetchRunResult } from "./run_result.js";
import { themeVar } from "./theme.js";
import { chainWidgetCallback } from "./widget.js";

const LOG_NAME = "WASNodeSuite.LayerArrange";

// Height in node units the panel opens at, and the narrowest it stays readable in.
const PANEL_HEIGHT = 196;
const PANEL_MIN_WIDTH = 250;

// One row, the gap under it, and the thumbnail inside it, in CSS pixels.
const ROW_HEIGHT = 42;
const ROW_GAP = 3;
const THUMB_WIDTH = 46;

// What separates the fields of a published row, and how many come before the name.
const SEPARATOR = "|";
const ROW_FIELDS = 8;

// The glyph on a drag handle, and the two a visibility toggle carries.
const HANDLE_GLYPH = "≡";
const SHOWN_GLYPH = "◉";
const HIDDEN_GLYPH = "○";

/**
 * One number as a whole number, or a fallback where it is not one.
 *
 * @param {*} value - Whatever the report or the widget carried.
 * @param {number} fallback - What to answer for anything that is not a number.
 * @returns {number} The truncated number.
 */
function whole(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : fallback;
}

/**
 * Hold a number inside a range.
 *
 * @param {number} value - The number.
 * @param {number} low - Lowest allowed.
 * @param {number} high - Highest allowed.
 * @returns {number} The number, moved into the range.
 */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * Read one published row.
 *
 * @param {string} line - The line the node published.
 * @returns {object|null} The layer's index, stacking, placement, visibility, opacity and
 *   name, or null where the line carries too few fields.
 */
function parseRow(line) {
  const parts = String(line).split(SEPARATOR);
  if (parts.length < ROW_FIELDS + 1) return null;
  const opacity = Number(parts[7]);
  return {
    index: whole(parts[0], -1),
    z: whole(parts[1]),
    x: whole(parts[2]),
    y: whole(parts[3]),
    w: Math.max(1, whole(parts[4], 1)),
    h: Math.max(1, whole(parts[5], 1)),
    visible: parts[6] === "1",
    opacity: Number.isFinite(opacity) ? clamp(opacity, 0, 1) : 1,
    name: parts.slice(ROW_FIELDS).join(SEPARATOR),
  };
}

/**
 * Every layer a report describes.
 *
 * @param {object|null} report - A `fetchRunResult` report.
 * @param {string} bodyName - The name the node published its table under.
 * @returns {object[]} One row per layer, in the order an arrangement indexes them.
 */
function readRows(report, bodyName) {
  const found = [];
  for (const part of report?.bodies ?? []) {
    if (part?.name !== bodyName) continue;
    for (const line of String(part.text ?? "").split("\n")) {
      if (!line) continue;
      const row = parseRow(line);
      if (row && row.index >= 0) found.push(row);
    }
  }
  return found.sort((a, b) => a.index - b.index);
}

/**
 * One layer as the next run will place it.
 *
 * @param {object} row - What the last run published for it.
 * @param {object} held - The whole arrangement the widget holds.
 * @returns {object} The row with the arrangement's own fields over it.
 */
function effective(row, held) {
  const entry = held?.[String(row.index)];
  const change = entry && typeof entry === "object" && !Array.isArray(entry) ? entry : {};
  const opacity = Number(change.opacity);
  return {
    ...row,
    x: whole(change.x, row.x),
    y: whole(change.y, row.y),
    w: Math.max(1, whole(change.w, row.w)),
    h: Math.max(1, whole(change.h, row.h)),
    z: whole(change.z_index, row.z),
    visible: typeof change.visible === "boolean" ? change.visible : row.visible,
    opacity: Number.isFinite(opacity) ? clamp(opacity, 0, 1) : row.opacity,
    edited: Object.keys(change).length > 0,
  };
}

/**
 * Build the panel one node arranges its stack in.
 *
 * @param {object} node - The node the panel belongs to, for its widgets and its redraws.
 * @param {object} [options] - What the panel reads and writes.
 * @param {string} [options.widgetName] - The widget the arrangement is written into.
 * @param {string} [options.slot] - The slot the layer thumbnails were published under.
 * @param {string} [options.bodyName] - The name the layer table was published under.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   refresh: Function, dispose: Function}} The panel, for `appendInterfaceWidget`.
 */
export function createLayerArrangePanel(node, options = {}) {
  const widgetName = options.widgetName || "arrangement";
  const slot = options.slot || "layers";
  const bodyName = options.bodyName || "layers";

  let disposed = false;
  let rows = [];
  let report = null;
  let label = RUN_LABELS[PREVIEW_STATE.WAITING] || "";
  let order = null;
  let drag = null;
  let written = null;
  let generation = 0;
  let drawnRun = -1;
  const thumbs = new Map();

  const root = document.createElement("div");
  root.className = "was-layer-arrange";
  root.tabIndex = -1;
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "gap:4px",
    "overflow:hidden",
    "padding:6px 7px",
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
    "border-radius:4px",
  ].join(";");

  const summary = document.createElement("div");
  summary.style.cssText = "flex:0 0 auto;font-weight:600;overflow:hidden;"
    + "text-overflow:ellipsis;white-space:nowrap";
  root.appendChild(summary);

  const list = document.createElement("div");
  list.style.cssText = "flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;"
    + `display:flex;flex-direction:column;gap:${ROW_GAP}px;scrollbar-width:thin`;
  root.appendChild(list);

  const footer = document.createElement("div");
  footer.style.cssText = "flex:0 0 auto;display:flex;gap:10px;align-items:baseline;"
    + "overflow:hidden;white-space:nowrap";
  footer.title = iconTitle(ICON.APPROXIMATE, "rotation and blend not drawn");
  root.appendChild(footer);

  /**
   * The widget the arrangement is written into.
   *
   * @returns {object|null} The widget, resolved by name at each use.
   */
  function widget() {
    const held = Array.isArray(node?.widgets) ? node.widgets : [];
    return held.find((candidate) => candidate?.name === widgetName) ?? null;
  }

  /**
   * The arrangement the widget holds.
   *
   * @returns {object|null} The decoded object, `{}` for an empty widget, and null where the
   *   text is not a JSON object, which is text the panel never overwrites.
   */
  function readArrangement() {
    const text = String(widget()?.value ?? "").trim();
    if (!text) return {};
    try {
      const decoded = JSON.parse(text);
      const usable = decoded && typeof decoded === "object" && !Array.isArray(decoded);
      return usable ? decoded : null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Merge changes into the widget, as one undo step.
   *
   * @param {object} changes - Layer index to the fields to set on it.
   * @returns {void}
   */
  function writeChanges(changes) {
    const target = widget();
    const held = readArrangement();
    // Unreadable text is left exactly as it stands.
    if (!target || held === null) return;
    const next = { ...held };
    for (const [key, fields] of Object.entries(changes)) {
      const entry = next[key];
      const kept = entry && typeof entry === "object" && !Array.isArray(entry) ? entry : {};
      next[key] = { ...kept, ...fields };
    }
    const text = JSON.stringify(next);
    if (text === String(target.value ?? "")) return;
    written = text;
    withGraphChange(() => {
      target.value = text;
      target.callback?.(text);
    });
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * The rows in the order they are drawn.
   *
   * @param {object[]} placed - The rows with the arrangement over them.
   * @returns {number[]} Layer indices, front of the stack first.
   */
  function stacked(placed) {
    return placed
      .slice()
      .sort((a, b) => b.z - a.z || b.index - a.index)
      .map((row) => row.index);
  }

  /**
   * Write the drawn order onto the widget as a stacking for every layer.
   *
   * @param {number[]} drawn - Layer indices, front of the stack first.
   * @returns {void}
   */
  function commitOrder(drawn) {
    const changes = {};
    drawn.forEach((index, position) => {
      changes[String(index)] = { z_index: drawn.length - 1 - position };
    });
    writeChanges(changes);
  }

  /**
   * Follow the pointer while a row is being dragged.
   *
   * @param {PointerEvent} event - The move.
   * @returns {void}
   */
  function onPointerMove(event) {
    if (!drag || !order || event.pointerId !== drag.pointerId) return;
    // A pointer arriving with no button held has already ended the gesture.
    if (event.buttons === 0) {
      endDrag(true);
      return;
    }
    const step = drag.step > 0 ? drag.step : ROW_HEIGHT + ROW_GAP;
    const moved = Math.round((event.clientY - drag.startY) / step);
    const target = clamp(drag.from + moved, 0, order.length - 1);
    if (target === drag.at) return;
    const [held] = order.splice(drag.at, 1);
    order.splice(target, 0, held);
    drag.at = target;
    paint();
  }

  /**
   * End the drag on a release.
   *
   * @param {PointerEvent} event - The release.
   * @returns {void}
   */
  function onPointerUp(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    endDrag(true);
  }

  /**
   * End the drag on anything that takes the pointer away.
   *
   * @returns {void}
   */
  function onPointerLost() {
    endDrag(false);
  }

  /**
   * Stop following the pointer.
   *
   * @returns {void}
   */
  function unwatch(pointerId) {
    if (pointerId !== undefined) {
      try {
        if (root.hasPointerCapture?.(pointerId)) root.releasePointerCapture(pointerId);
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to let the pointer go:`, error);
      }
    }
    root.removeEventListener("pointermove", onPointerMove);
    root.removeEventListener("pointerup", onPointerUp);
    root.removeEventListener("pointercancel", onPointerLost);
    window.removeEventListener("blur", onPointerLost);
  }

  /**
   * End whatever drag is running and redraw from the widget.
   *
   * @param {boolean} commit - True to write the order the row was dragged to.
   * @returns {void}
   */
  function endDrag(commit) {
    if (!drag) return;
    const drawn = order ? order.slice() : null;
    const { pointerId } = drag;
    drag = null;
    order = null;
    unwatch(pointerId);
    if (commit && drawn) commitOrder(drawn);
    paint();
  }

  /**
   * Take a row and follow the pointer until it is let go.
   *
   * @param {PointerEvent} event - The press anywhere on the row but its visibility toggle.
   * @param {HTMLElement} row - The row itself.
   * @param {number} position - Where the row sits in the drawn order.
   * @param {number[]} drawn - The drawn order at the moment of the press.
   * @returns {void}
   */
  function startDrag(event, row, position, drawn) {
    if (event.button !== 0 || drawn.length < 2 || drag) return;
    event.preventDefault();
    event.stopPropagation();
    order = drawn.slice();
    drag = {
      pointerId: event.pointerId,
      startY: event.clientY,
      from: position,
      at: position,
      step: row.getBoundingClientRect().height + ROW_GAP,
    };
    // Captured on the panel, which paint() never rebuilds, so redrawing the row under the
    // pointer does not end the gesture and nothing between the row and the window can keep a
    // move from arriving.
    try {
      root.setPointerCapture(event.pointerId);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to take the pointer:`, error);
    }
    root.addEventListener("pointermove", onPointerMove);
    root.addEventListener("pointerup", onPointerUp);
    root.addEventListener("pointercancel", onPointerLost);
    window.addEventListener("blur", onPointerLost);
    paint();
  }

  /**
   * Draw one layer's row.
   *
   * @param {object} row - The layer as the next run will place it.
   * @param {number} position - Where it sits in the drawn order.
   * @param {number[]} drawn - The whole drawn order.
   * @returns {HTMLElement} The row.
   */
  function buildRow(row, position, drawn) {
    const held = drag?.at === position;
    const line = document.createElement("div");
    line.style.cssText = [
      "flex:0 0 auto",
      "display:flex",
      "gap:6px",
      "align-items:center",
      `height:${ROW_HEIGHT}px`,
      "padding:0 4px",
      "border-radius:3px",
      "box-sizing:border-box",
      `background:${themeVar(held ? "accentBg" : "bg")}`,
      `border:1px solid ${themeVar(held ? "accent" : "border")}`,
      `opacity:${row.visible ? 1 : 0.55}`,
      "cursor:grab",
      "touch-action:none",
    ].join(";");
    line.title = "Drag to restack";
    line.addEventListener("pointerdown", (event) => {
      startDrag(event, line, position, drawn);
    });

    const handle = document.createElement("div");
    handle.textContent = HANDLE_GLYPH;
    handle.style.cssText = [
      "flex:0 0 auto",
      "width:14px",
      "height:100%",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "cursor:grab",
      "touch-action:none",
      "user-select:none",
      `color:${themeVar("fgMuted")}`,
    ].join(";");
    line.appendChild(handle);

    const frame = document.createElement("div");
    frame.style.cssText = [
      "flex:0 0 auto",
      `width:${THUMB_WIDTH}px`,
      `height:${ROW_HEIGHT - 8}px`,
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "overflow:hidden",
      "border-radius:2px",
      `background:${themeVar("bgDark")}`,
      `border:1px solid ${themeVar("border")}`,
    ].join(";");
    const picture = thumbs.get(row.index);
    if (picture) {
      picture.style.cssText = "max-width:100%;max-height:100%;display:block";
      frame.appendChild(picture);
    }
    line.appendChild(frame);

    const meta = document.createElement("div");
    meta.style.cssText = "flex:1 1 auto;min-width:0;display:flex;flex-direction:column;"
      + "gap:1px;line-height:1.25";
    const name = document.createElement("div");
    name.style.cssText = `color:${themeVar(row.edited ? "accent" : "fg")};overflow:hidden;`
      + "text-overflow:ellipsis;white-space:nowrap";
    name.textContent = row.name || `layer ${row.index}`;
    const detail = document.createElement("div");
    detail.style.cssText = `color:${themeVar("fgMuted")};font-size:9px;overflow:hidden;`
      + "text-overflow:ellipsis;white-space:nowrap";
    detail.textContent = `${row.w}x${row.h} at ${row.x},${row.y}  z ${row.z}`
      + (row.opacity < 1 ? `  ${Math.round(row.opacity * 100)}%` : "");
    meta.append(name, detail);
    line.appendChild(meta);

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.textContent = row.visible ? SHOWN_GLYPH : HIDDEN_GLYPH;
    toggle.title = row.visible ? "Hide this layer" : "Show this layer";
    toggle.style.cssText = [
      "flex:0 0 auto",
      "width:20px",
      "height:20px",
      "padding:0",
      "cursor:pointer",
      "border-radius:3px",
      `background:${themeVar("panelBg")}`,
      `border:1px solid ${themeVar("border")}`,
      `color:${themeVar(row.visible ? "fg" : "fgMuted")}`,
    ].join(";");
    toggle.addEventListener("pointerdown", (event) => event.stopPropagation());
    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      writeChanges({ [String(row.index)]: { visible: !row.visible } });
      paint();
    });
    line.appendChild(toggle);
    return line;
  }

  /**
   * Redraw the whole panel from the report and the widget.
   *
   * @returns {void}
   */
  function paint() {
    if (disposed) return;
    if (!rows.length) {
      summary.textContent = label;
      summary.style.color = themeVar("fgMuted");
      list.textContent = "";
      footer.textContent = "";
      node.setDirtyCanvas?.(true, false);
      return;
    }

    const held = readArrangement();
    const placed = rows.map((row) => effective(row, held ?? {}));
    const drawn = drag && order ? order.slice() : stacked(placed);
    const byIndex = new Map(placed.map((row) => [row.index, row]));

    summary.textContent = report?.summary || "";
    summary.style.color = statusColour(report?.status);

    list.textContent = "";
    drawn.forEach((index, position) => {
      const row = byIndex.get(index);
      if (row) list.appendChild(buildRow(row, position, drawn));
    });

    footer.textContent = "";
    const canvas = (report?.facts ?? []).find((fact) => fact.name === "canvas");
    const state = document.createElement("span");
    state.style.color = themeVar(held === null ? "warning" : "fgMuted");
    state.textContent = held === null
      ? "arrangement is not readable"
      : `canvas ${canvas?.value ?? "unknown"}`;
    const count = document.createElement("span");
    count.style.color = themeVar("fgMuted");
    count.textContent = `${drawn.length} layers, front first`;
    footer.append(state, count);
    node.setDirtyCanvas?.(true, false);
  }

  /**
   * Ask for one thumbnail per layer.
   *
   * @param {number} mine - The generation this load belongs to.
   * @param {object[]} found - The rows to fetch pictures for.
   * @returns {Promise<void>} Resolved once every row has been asked for.
   */
  async function loadThumbs(mine, found) {
    for (const row of found) {
      if (disposed || mine !== generation) return;
      const answer = await fetchInputPreview(node, slot, row.index);
      if (disposed || mine !== generation) return;
      if (answer?.image) {
        thumbs.set(row.index, answer.image);
        paint();
      }
    }
  }

  let pending = false;
  let again = false;

  /**
   * Read what the node last published and draw it.
   *
   * @returns {Promise<void>} Resolved once the panel has been drawn.
   */
  async function refresh() {
    if (disposed) return;
    // Asked again while a read is in flight, the answer on its way describes an older run.
    if (pending) {
      again = true;
      return;
    }
    pending = true;
    try {
      do {
        again = false;
        const answer = await fetchRunResult(node);
        if (disposed) return;
        report = answer?.result ?? null;
        label = answer?.label || RUN_LABELS[answer?.state] || "";
        rows = readRows(report, bodyName);
        const run = whole(report?.run, -1);
        if (run !== drawnRun) {
          drawnRun = run;
          thumbs.clear();
          generation += 1;
          loadThumbs(generation, rows).catch((error) => {
            console.error(`[${LOG_NAME}] Failed to read the layer thumbnails:`, error);
          });
        }
        paint();
      } while (again && !disposed);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the arranged stack:`, error);
    } finally {
      pending = false;
    }
  }

  // A hand edit is read and drawn, and never written back.
  chainWidgetCallback(node, widgetName, () => {
    if (String(widget()?.value ?? "") !== written) endDrag(false);
    written = null;
    paint();
  }, LOG_NAME);

  // The rows are the only thing here that scrolls. At either end of the list the gesture is
  // the graph's and zooms the node under the pointer.
  const releaseWheel = captureWheel(root, (event) => {
    if (list.scrollHeight <= list.clientHeight || !list.contains(event.target)) return false;
    const before = list.scrollTop;
    list.scrollTop += wheelPixels(event, list).y;
    return list.scrollTop !== before;
  });

  root.addEventListener("contextmenu", (event) => event.preventDefault());

  root.addEventListener("pointerdown", (event) => {
    // The middle button is handed back to the canvas.
    if (event.button === 1) app.canvas?.processMouseDown?.(event);
  });

  root.addEventListener("keydown", (event) => {
    // Delete and Backspace are consumed rather than passed to the canvas.
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    event.preventDefault();
    event.stopPropagation();
  });

  const stopFinished = onNodeFinished(node, () => refresh());
  const stopEnded = onRunEnded(() => refresh());

  paint();
  refresh();

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    refresh,
    dispose() {
      if (disposed) return;
      endDrag(false);
      disposed = true;
      releaseWheel();
      generation += 1;
      if (typeof stopFinished === "function") stopFinished();
      if (typeof stopEnded === "function") stopEnded();
      thumbs.clear();
    },
  };
}
