/**
 * The frame panel on Resolution Selector (Advanced), and only the measurement size_by reads.
 *
 * The panel draws the shape the node answers and sets aspect_ratio by dragging the frame's corner.
 */

import { app } from "../../scripts/app.js";
import { mirrored, parseRatio, resolve } from "./interface/image_resolution.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { withGraphChange } from "./interface/region.js";
import { refreshRelevantWidgets, watchRelevantWidgets } from "./interface/relevant_widgets.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.js";
import { appendInterfaceWidget, chainWidgetCallback, graphLoading } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ResolutionSelectorUI";
const NODE_NAME = "WASResolutionSelector";
const SETTING_ID = "WAS.ResolutionSelector.ShowInterface";
const CONTROL_WIDGET = "size_by";

const UI_WIDGET_NAME = "was_resolution_ui";
const UI_WIDGET_TYPE = "was_resolution_frame";

/** size_by -> the measurement widget it reads, under the name drawn for it. */
const MODES = {
  "megapixels": { megapixels: { label: "megapixels" } },
  "long edge": { size: { label: "long_edge" } },
  "short edge": { size: { label: "short_edge" } },
  "width": { size: { label: "width" } },
  "height": { size: { label: "height" } },
};

// Widgets the readout and the frame are worked out from.
const READ_WIDGETS = ["aspect_ratio", "size_by", "size", "megapixels", "multiple_of"];

// Panel height in node units, and the room kept round the frame in layout pixels.
const PANEL_HEIGHT = 170;
const FRAME_PAD = 12;
const HANDLE = 6;
// A drag closer than this to the middle reads no shape.
const DEAD_ZONE = 4;

/**
 * Read whether the panel is drawn at all.
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
 * A node's widget by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget.
 */
function widgetOf(node, name) {
  return (node?.widgets ?? []).find((widget) => widget.name === name);
}

/**
 * The entries a combo widget offers.
 *
 * @param {object} widget - The widget.
 * @returns {string[]} Its options.
 */
function optionsOf(widget) {
  const values = widget?.options?.values;
  const listed = typeof values === "function" ? values(widget) : values;
  return Array.isArray(listed) ? listed : [];
}

/**
 * Whether a link replaces a widget's value.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {boolean} True when an input of that name is wired.
 */
function linked(node, name) {
  const input = (node?.inputs ?? []).find((socket) => socket?.name === name);
  return input?.link !== null && input?.link !== undefined;
}

/**
 * The listed shape nearest a width over height, compared on a log scale.
 *
 * @param {string[]} options - The aspect_ratio entries.
 * @param {number} shape - Width over height.
 * @returns {string|null} The nearest entry, or null when none reads.
 */
function nearestRatio(options, shape) {
  let best = null;
  let gap = Number.POSITIVE_INFINITY;
  for (const option of options) {
    const [wide, high] = parseRatio(option);
    const distance = Math.abs(Math.log(wide / high) - Math.log(shape));
    if (distance < gap) {
      gap = distance;
      best = option;
    }
  }
  return best;
}

/**
 * The largest rectangle of one shape that fits a box, centred in it.
 *
 * @param {number} shape - Width over height.
 * @param {number} width - Box width.
 * @param {number} height - Box height.
 * @returns {{x: number, y: number, w: number, h: number}} The rectangle.
 */
function fitShape(shape, width, height) {
  const roomW = Math.max(1, width - FRAME_PAD * 2);
  const roomH = Math.max(1, height - FRAME_PAD * 2);
  const w = Math.min(roomW, roomH * shape);
  const h = w / shape;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

/**
 * Build the frame panel for one node.
 *
 * @param {object} node - The node the panel is drawn on.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, refresh: () => void,
 *   schedulePaint: () => void, dispose: () => void}} The panel.
 */
function createFramePanel(node) {
  const root = document.createElement("div");
  root.style.cssText = [
    "display:flex", "flex-direction:column", "gap:4px", "box-sizing:border-box", "width:100%",
    "height:100%", "padding:4px", "overflow:hidden", "user-select:none", "touch-action:none",
    "font:11px sans-serif", `color:${themeVar("fg")}`,
  ].join(";");

  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;flex:1 1 auto;min-height:60px;border-radius:4px;"
    + `background:${themeVar("inputBg")};border:1px solid ${themeVar("border")};cursor:crosshair`;
  stage.title = "Drag a corner to reshape";
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  stage.appendChild(canvas);
  root.appendChild(stage);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;min-width:0";
  const readout = document.createElement("div");
  readout.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  const flip = document.createElement("button");
  flip.type = "button";
  flip.textContent = "flip";
  flip.title = "Swap portrait and landscape";
  flip.style.cssText = "flex:0 0 auto;padding:1px 8px;border-radius:3px;cursor:pointer;font:inherit;"
    + `background:${themeVar("inputBg")};color:${themeVar("fg")};border:1px solid ${themeVar("border")}`;
  footer.append(readout, flip);
  root.appendChild(footer);

  const state = { dragging: null, paintHandle: 0, disposed: false };

  /**
   * The shape on show: the one being dragged to, or the widget's.
   *
   * @returns {string} A ratio entry.
   */
  function shownRatio() {
    return state.dragging ?? String(widgetOf(node, "aspect_ratio")?.value ?? "1:1");
  }

  /**
   * Write the readout under the frame.
   *
   * @returns {void}
   */
  function writeReadout() {
    const value = (name, fallback) => widgetOf(node, name)?.value ?? fallback;
    const ratio = shownRatio();
    const [width, height] = resolve(
      ratio, value("size_by", "megapixels"), value("size", 1024), value("megapixels", 1), value("multiple_of", 64),
    );
    readout.textContent = "";
    const size = document.createElement("span");
    size.style.cssText = `font-weight:600;color:${themeVar("fg")}`;
    size.textContent = `${width} × ${height}`;
    const rest = document.createElement("span");
    rest.style.cssText = `color:${themeVar("fgMuted")}`;
    const area = (width * height) / 1_000_000;
    const note = linked(node, "aspect_ratio") ? "  ·  aspect_ratio is wired in" : "";
    rest.textContent = `  ·  ${ratio}  ·  ${area.toFixed(2)} MP${note}`;
    readout.append(size, rest);
    const [wide, high] = parseRatio(ratio);
    flip.disabled = wide === high || linked(node, "aspect_ratio");
    flip.style.opacity = flip.disabled ? "0.45" : "1";
  }

  /**
   * Draw the frame, the square it is measured against and its mirror.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    writeReadout();
    if (!width || !height) return;
    const ratio = surfaceRatio(canvas);
    const deviceW = Math.max(1, Math.round(width * ratio));
    const deviceH = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== deviceW) canvas.width = deviceW;
    if (canvas.height !== deviceH) canvas.height = deviceH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const theme = readTheme();

    const [wide, high] = parseRatio(shownRatio());
    const shape = wide / high;
    const square = fitShape(1, width, height);
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(square.x + 0.5, square.y + 0.5, square.w - 1, square.h - 1);
    if (wide !== high) {
      const ghost = fitShape(1 / shape, width, height);
      ctx.globalAlpha = 0.5;
      ctx.strokeRect(ghost.x + 0.5, ghost.y + 0.5, ghost.w - 1, ghost.h - 1);
      ctx.globalAlpha = 1;
    }
    ctx.setLineDash([]);

    const frame = fitShape(shape, width, height);
    ctx.fillStyle = theme.accentBg;
    ctx.fillRect(frame.x, frame.y, frame.w, frame.h);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = state.dragging ? theme.accentHover : theme.accent;
    ctx.strokeRect(frame.x + 0.75, frame.y + 0.75, frame.w - 1.5, frame.h - 1.5);
    ctx.fillStyle = theme.accent;
    for (const [cx, cy] of [[frame.x, frame.y], [frame.x + frame.w, frame.y],
      [frame.x, frame.y + frame.h], [frame.x + frame.w, frame.y + frame.h]]) {
      ctx.fillRect(cx - HANDLE / 2, cy - HANDLE / 2, HANDLE, HANDLE);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.fg;
    ctx.font = "600 13px sans-serif";
    ctx.fillText(shownRatio(), width / 2, height / 2 - 7, Math.max(1, frame.w - 8));
    ctx.fillStyle = theme.fgMuted;
    ctx.font = "10px sans-serif";
    const orientation = wide === high ? "square" : wide > high ? "landscape" : "portrait";
    ctx.fillText(orientation, width / 2, height / 2 + 9, Math.max(1, frame.w - 8));
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
        console.error(`[${EXT_NAME}] Failed to draw the frame:`, error);
      }
    });
  }

  /**
   * Write a shape to aspect_ratio as one undoable change.
   *
   * @param {string} ratio - A listed entry.
   * @returns {void}
   */
  function commit(ratio) {
    const widget = widgetOf(node, "aspect_ratio");
    if (!widget || widget.value === ratio || !optionsOf(widget).includes(ratio)) return;
    withGraphChange(() => {
      widget.value = ratio;
    });
    widget.callback?.(ratio);
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * The listed shape a pointer position asks for, from the stage's middle.
   *
   * @param {PointerEvent} event - The pointer event.
   * @returns {string|null} An entry, or null inside the dead zone.
   */
  function ratioAt(event) {
    const at = elementPoint(stage, event);
    const dx = Math.abs(at.x - stage.clientWidth / 2);
    const dy = Math.abs(at.y - stage.clientHeight / 2);
    if (dx < DEAD_ZONE && dy < DEAD_ZONE) return null;
    return nearestRatio(optionsOf(widgetOf(node, "aspect_ratio")), Math.max(dx, 0.5) / Math.max(dy, 0.5));
  }

  /**
   * Finish a drag, writing the shape it ended on.
   *
   * @returns {void}
   */
  function endDrag() {
    if (state.dragging === null) return;
    const ratio = state.dragging;
    state.dragging = null;
    commit(ratio);
    schedulePaint();
  }

  /**
   * Run a pointer handler, logging a failure rather than throwing at the browser.
   *
   * @param {(event: Event) => void} handler - The handler.
   * @returns {(event: Event) => void} The guarded handler.
   */
  const guard = (handler) => (event) => {
    try {
      handler(event);
    } catch (error) {
      console.error(`[${EXT_NAME}] Pointer handling failed:`, error);
    }
  };

  stage.addEventListener("pointerdown", guard((event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0 || linked(node, "aspect_ratio")) return;
    event.preventDefault();
    event.stopPropagation();
    stage.setPointerCapture?.(event.pointerId);
    state.dragging = ratioAt(event) ?? shownRatio();
    schedulePaint();
  }));
  stage.addEventListener("pointermove", guard((event) => {
    if (event.buttons & 4) app.canvas?.processMouseMove?.(event);
    if (state.dragging === null) return;
    if (!(event.buttons & 1)) {
      endDrag();
      return;
    }
    const next = ratioAt(event);
    if (next && next !== state.dragging) {
      state.dragging = next;
      schedulePaint();
    }
  }));
  stage.addEventListener("pointerup", guard((event) => {
    if (event.button === 1) app.canvas?.processMouseUp?.(event);
    else endDrag();
  }));
  for (const name of ["pointercancel", "lostpointercapture"]) {
    stage.addEventListener(name, guard(() => endDrag()));
  }
  root.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  flip.addEventListener("click", guard((event) => {
    event.preventDefault();
    event.stopPropagation();
    commit(mirrored(String(widgetOf(node, "aspect_ratio")?.value ?? "1:1")));
    schedulePaint();
  }));
  flip.addEventListener("pointerdown", (event) => event.stopPropagation());

  const releaseWheel = captureWheel(root);
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => schedulePaint()) : null;
  observer?.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, () => schedulePaint());
  const stopTheme = onThemeChange(() => schedulePaint());

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    refresh: schedulePaint,
    schedulePaint,
    dispose() {
      state.disposed = true;
      if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
      releaseWheel();
      observer?.disconnect();
      stopRatio?.();
      stopTheme?.();
    },
  };
}

/**
 * Append the frame panel to a node and repaint it whenever what it reads changes.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPanel(node) {
  const panel = createFramePanel(node);
  appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
  for (const name of READ_WIDGETS) chainWidgetCallback(node, name, panel.schedulePaint, EXT_NAME);

  // A link replaces a widget's value without touching the widget.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    panel.schedulePaint();
    return result;
  };
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    if (!graphLoading()) panel.schedulePaint();
    return result;
  };
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      panel.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
    }
    return result;
  };
  panel.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Resolution Selector", "Frame"],
      name: "Show the frame panel",
      tooltip:
        "Draw the shape Resolution Selector (Advanced) answers under its widgets, with a corner "
        + "to drag and a flip button. The widgets work either way. This applies to nodes added "
        + "after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    const proto = nodeType.prototype;
    if (proto.__was_resolution_selector_wrapped) return;
    proto.__was_resolution_selector_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        watchRelevantWidgets(this, CONTROL_WIDGET, MODES);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to lay the node out:`, error);
      }
      try {
        if (interfaceEnabled()) attachPanel(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the frame panel:`, error);
      }
      return result;
    };

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      try {
        refreshRelevantWidgets(this, CONTROL_WIDGET, MODES);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to lay the node out after loading:`, error);
      }
      return result;
    };
  },
});
