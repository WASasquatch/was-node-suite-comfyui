/**
 * The frame panel on Scale Image to Total Pixels (Advanced).
 *
 * The panel draws the source frame against the scaled one and sets megapixels by dragging the
 * scaled frame's corner.
 */

import { app } from "../../scripts/app.js";
import { drawLabel } from "./interface/canvas_label.js";
import { CARD_LOOK, cardChip, cardRule } from "./interface/card.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { megapixelUnit, scaleToTotal } from "./interface/image_resolution.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { PREVIEW_STATE } from "./interface/preview.js";
import { withGraphChange } from "./interface/region.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeDone } from "./interface/run_events.js";
import { fetchRunResult, whenReadable } from "./interface/run_result.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { inputValue, isWired, sourceOf } from "./interface/upstream.js";
import { appendInterfaceWidget, chainWidgetCallback, graphLoading } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ScaleToTotalPixelsUI";
const NODE_NAME = "WASImageScaleToTotalPixels";
const SETTING_ID = "WAS.ScaleToTotalPixels.ShowInterface";

const UI_WIDGET_NAME = "was_scale_total_ui";
const UI_WIDGET_TYPE = "was_scale_total_frame";

// Widgets the frames and the readout are worked out from.
const READ_WIDGETS = ["megapixels", "resolution_steps", "scale_mode", "crop", "realworld_mp"];

// Panel height in node units, and the room kept round the frames in layout pixels.
const PANEL_HEIGHT = 196;
const FRAME_PAD = 14;
const HANDLE = 6;

// The legend's font, as `drawLabel` sets it, and what its plate adds round the widest text.
const LEGEND_FONT = "10px system-ui,sans-serif";
const LEGEND_EXTRA = 8 + 9 + 8;

// The megapixels widget's range, read from the widget where it carries one.
const MIN_MEGAPIXELS = 0.01;
const MAX_MEGAPIXELS = 64;

// Longest side the node builds, from `MAX_CANVAS_SIDE` in `modules/image/sizing.py`.
const MAX_SIDE = 30000;

// How often the panel looks for a changed upstream picture or wired value, in milliseconds.
const WATCH_MS = 1000;

// What a size fact opens with, as `modules/interface/size_report.py` writes it.
const SIZE_PATTERN = /^\s*(\d+)\s*x\s*(\d+)/;

const STAGE_HINT = "Drag a corner of the scaled frame to set megapixels";

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
 * One named fact out of a report, read as a size.
 *
 * @param {object} report - A run report, whose `facts` is an array of `{name, value}`.
 * @param {string} name - The fact.
 * @returns {number[]|null} `[width, height]`, or null where the report carries no such size.
 */
function sizeFact(report, name) {
  for (const row of report?.facts ?? []) {
    if (row?.name !== name) continue;
    const found = SIZE_PATTERN.exec(String(row.value ?? ""));
    if (!found) return null;
    const width = Number(found[1]);
    const height = Number(found[2]);
    return width > 0 && height > 0 ? [width, height] : null;
  }
  return null;
}

/**
 * A pair of sides as the readout writes them.
 *
 * @param {number[]} size - `[width, height]`.
 * @returns {string} `1024 × 1536`.
 */
function spell(size) {
  return `${size[0]} × ${size[1]}`;
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
    "display:flex", "flex-direction:column", "gap:5px", "box-sizing:border-box", "width:100%",
    "height:100%", "padding:6px 8px 7px", "overflow:hidden", "user-select:none", "touch-action:none",
    "font:11.5px/16px system-ui,sans-serif", "font-variant-numeric:tabular-nums", `color:${CARD_LOOK.text}`,
  ].join(";");

  const head = document.createElement("div");
  head.style.cssText = "flex:none;display:flex;align-items:center;gap:6px;min-width:0";
  const sizeLine = document.createElement("div");
  sizeLine.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  const chips = document.createElement("div");
  chips.style.cssText = "flex:none;display:flex;gap:4px";
  head.append(sizeLine, chips);

  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;flex:1 1 auto;min-height:60px;border-radius:6px;"
    + `background:${CARD_LOOK.inset};border:1px solid ${CARD_LOOK.edge};cursor:nwse-resize`;
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  stage.appendChild(canvas);

  const areaLine = document.createElement("div");
  areaLine.style.cssText = "flex:none;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
    + `color:${CARD_LOOK.muted}`;
  root.append(head, cardRule(), stage, areaLine);

  const titles = hoverTitles(stage);
  const state = {
    dragging: null,
    report: null,
    paintHandle: 0,
    seen: "",
    disposed: false,
  };

  /**
   * The settings the node will run with, read through any wires feeding them.
   *
   * @returns {{known: boolean, megapixels: number, steps: number, mode: string, crop: string,
   *   unit: number, wired: string}} `known` is false where a wired value only a run can tell
   *   feeds one of them, which `wired` then names.
   */
  function readSettings() {
    const read = {};
    let wired = "";
    for (const name of READ_WIDGETS) {
      const answer = inputValue(node, name);
      if (!answer.known && !wired) wired = name;
      read[name] = answer.value;
    }
    return {
      known: !wired,
      megapixels: state.dragging?.megapixels ?? Number(read.megapixels ?? 1),
      steps: Number(read.resolution_steps ?? 1),
      mode: String(read.scale_mode ?? "always"),
      crop: String(read.crop ?? "disabled"),
      unit: megapixelUnit(read.realworld_mp === true),
      wired,
    };
  }

  /**
   * The size of the pictures the node will be handed, and where it was read.
   *
   * @returns {{size: number[], from: string, live: boolean}|null} `live` is true for a picture
   *   on the node upstream, false for the last run. Null where neither has one.
   */
  function readSource() {
    if (isWired(node, "image")) {
      const source = sourceOf(node, "image");
      const picture = source?.node?.imgs?.[0];
      if (picture?.complete && picture.naturalWidth > 0 && picture.naturalHeight > 0) {
        return {
          size: [picture.naturalWidth, picture.naturalHeight],
          from: String(source.node.title || source.node.type || "the image"),
          live: true,
        };
      }
    }
    const reported = sizeFact(state.report, "in");
    return reported ? { size: reported, from: "the last run", live: false } : null;
  }

  /**
   * Everything one repaint draws, worked out the way the node works it out.
   *
   * @returns {object} The source, the settings, the scaled size and how far each can be trusted.
   */
  function model() {
    const settings = readSettings();
    const source = readSource();
    let out = null;
    let scaled = true;
    if (source && settings.known) {
      const answer = scaleToTotal(
        source.size[0], source.size[1], settings.megapixels, settings.steps, settings.unit, settings.mode,
      );
      out = [answer.width, answer.height];
      scaled = answer.scaled;
    } else if (source) {
      // A wired value only a run can tell leaves the last run's sizes as the best answer.
      const reported = sizeFact(state.report, "in");
      out = sizeFact(state.report, "out");
      if (out && reported) return { settings, source: { ...source, size: reported }, out, scaled };
      out = null;
    }
    return { settings, source, out, scaled };
  }

  /**
   * The words drawn while there are no frames to draw.
   *
   * @param {object} m - What `model` answered.
   * @returns {string} The blocking state.
   */
  function emptyWords(m) {
    if (!isWired(node, "image")) return "no image connected";
    if (!m.source) return "no source size yet";
    return "no scaled size until the next run";
  }

  /**
   * The glyph the frames carry and the words behind it.
   *
   * @param {object} m - What `model` answered.
   * @returns {{icon: string, detail: string}} The claim.
   */
  function claimOf(m) {
    if (!m.source) {
      return {
        icon: ICON.WARNING,
        detail: isWired(node, "image")
          ? "the source size is read from a picture upstream or from the last run"
          : "connect the image input to read the source size",
      };
    }
    if (!m.settings.known) {
      return { icon: ICON.APPROXIMATE, detail: `${m.settings.wired} is wired in, so the sizes are the last run's` };
    }
    if (!m.source.live) return { icon: ICON.APPROXIMATE, detail: "source size from the last run" };
    return { icon: ICON.EXACT, detail: `source size read from ${m.source.from}` };
  }

  /**
   * Whether a drag on the stage can set megapixels.
   *
   * @param {object} m - What `model` answered.
   * @returns {boolean} True with both frames drawn from known settings and megapixels unwired.
   */
  function canDrag(m) {
    return Boolean(m.source && m.out && m.settings.known) && !isWired(node, "megapixels");
  }

  /**
   * Where the frames are drawn: frozen while a drag is held, fitted beside the legend otherwise.
   *
   * @param {number[]} source - The source size.
   * @param {number[]} out - The scaled size.
   * @param {number} width - Stage width.
   * @param {number} height - Stage height.
   * @returns {{scale: number, cx: number, cy: number}} Layout pixels per image pixel, and the
   *   centre both frames share.
   */
  function frameLayout(source, out, width, height) {
    if (state.dragging) return state.dragging.layout;
    const ctx = canvas.getContext("2d");
    let legend = 0;
    if (ctx) {
      ctx.font = LEGEND_FONT;
      legend = Math.ceil(Math.max(ctx.measureText(`out ${spell(out)}`).width,
        ctx.measureText(`in ${spell(source)}`).width)) + LEGEND_EXTRA;
    }
    // The legend gets a column of its own where the stage is wide enough to spare one.
    const gutter = width - legend > height ? legend : 0;
    const roomW = Math.max(1, width - gutter - FRAME_PAD * 2);
    const roomH = Math.max(1, height - FRAME_PAD * 2);
    return {
      scale: Math.min(roomW / Math.max(source[0], out[0]), roomH / Math.max(source[1], out[1])),
      cx: gutter + (width - gutter) / 2,
      cy: height / 2,
    };
  }

  /**
   * Add one pill to the header.
   *
   * @param {string} text - The words in the pill.
   * @param {string} tone - A key of `CHIP_TONES`.
   * @param {string} hint - Its hover text.
   * @returns {void}
   */
  function addChip(text, tone, hint) {
    const chip = cardChip(text, tone);
    chip.title = hint;
    chips.appendChild(chip);
  }

  /**
   * Write the header, its pills and the line under the stage.
   *
   * @param {object} m - What `model` answered.
   * @returns {void}
   */
  function writeReadout(m) {
    sizeLine.textContent = "";
    chips.textContent = "";
    areaLine.textContent = "";
    areaLine.style.color = CARD_LOOK.muted;
    if (!m.settings.known) addChip("wired", "accent", `${m.settings.wired} is wired in and read on the next run`);
    if (!m.out || !m.source) {
      const empty = document.createElement("span");
      empty.style.cssText = `color:${CARD_LOOK.muted}`;
      empty.textContent = emptyWords(m);
      sizeLine.appendChild(empty);
      areaLine.textContent = m.settings.known ? `${m.settings.megapixels.toFixed(2)} MP asked` : "";
      return;
    }
    const size = document.createElement("span");
    size.style.cssText = `font-weight:650;color:${CARD_LOOK.title}`;
    size.textContent = spell(m.out);
    const from = document.createElement("span");
    from.style.cssText = `color:${CARD_LOOK.muted}`;
    from.textContent = `  from ${spell(m.source.size)}`;
    sizeLine.append(size, from);

    const unit = m.settings.unit;
    const area = (m.out[0] * m.out[1]) / unit;
    const before = (m.source.size[0] * m.source.size[1]) / unit;
    const ratio = (m.out[0] * m.out[1]) / (m.source.size[0] * m.source.size[1]);
    addChip(`×${ratio.toFixed(2)}`, "accent", "mp_ratio: pixels out divided by pixels in");
    if (!m.scaled) addChip("left as is", "warning", `scale_mode is ${m.settings.mode}, so the picture passes through`);
    if (Math.max(m.out[0], m.out[1]) > MAX_SIDE) {
      addChip("too large", "error", `Past ${MAX_SIDE} pixels a side: lower megapixels`);
    }
    areaLine.textContent = `${area.toFixed(2)} MP from ${before.toFixed(2)} MP`;
  }

  /**
   * Draw both frames, their labels and the glyph.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const m = model();
    writeReadout(m);
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (!width || !height) {
      titles.set([]);
      return;
    }
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
    const frame = { width, height, top: 0, theme };
    const claim = claimOf(m);
    const regions = [];

    if (m.source && m.out) {
      const { scale, cx, cy } = frameLayout(m.source.size, m.out, width, height);
      const rect = (size) => {
        const w = Math.max(2, size[0] * scale);
        const h = Math.max(2, size[1] * scale);
        return { x: cx - w / 2, y: cy - h / 2, w, h };
      };
      const inRect = rect(m.source.size);
      const outRect = rect(m.out);

      const accent = theme.cardAccent || theme.accent;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, width, height);
      ctx.clip();
      // The scaled frame is washed from the card's accent at its top left to the deep ground.
      const wash = ctx.createLinearGradient(outRect.x, outRect.y, outRect.x + outRect.w, outRect.y + outRect.h);
      wash.addColorStop(0, accent);
      wash.addColorStop(1, theme.bgDark);
      ctx.globalAlpha = state.dragging ? 0.42 : 0.3;
      ctx.fillStyle = wash;
      ctx.fillRect(outRect.x, outRect.y, outRect.w, outRect.h);
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = state.dragging ? theme.accentHover : accent;
      ctx.strokeRect(outRect.x + 0.75, outRect.y + 0.75, outRect.w - 1.5, outRect.h - 1.5);
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = theme.fgMuted;
      ctx.strokeRect(inRect.x + 0.5, inRect.y + 0.5, inRect.w - 1, inRect.h - 1);
      ctx.setLineDash([]);
      if (m.settings.known) {
        ctx.fillStyle = accent;
        ctx.strokeStyle = theme.bgDark;
        for (const [hx, hy] of [[outRect.x, outRect.y], [outRect.x + outRect.w, outRect.y],
          [outRect.x, outRect.y + outRect.h], [outRect.x + outRect.w, outRect.y + outRect.h]]) {
          ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
          ctx.strokeRect(hx - HANDLE / 2 - 0.5, hy - HANDLE / 2 - 0.5, HANDLE + 1, HANDLE + 1);
        }
      }
      ctx.restore();

      // A legend in the top left corner names the two frames by their swatches.
      drawLabel(ctx, frame, `out ${spell(m.out)}`, 4, 11, { swatch: accent });
      drawLabel(ctx, frame, `in ${spell(m.source.size)}`, 4, 28, { swatch: theme.fgMuted, muted: true });
    } else {
      const side = Math.max(8, Math.min(width, height) - FRAME_PAD * 2);
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = theme.border;
      ctx.strokeRect((width - side) / 2 + 0.5, (height - side) / 2 + 0.5, side - 1, side - 1);
      ctx.setLineDash([]);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = theme.fgMuted;
      ctx.font = "11px sans-serif";
      ctx.fillText(emptyWords(m), width / 2, height / 2, Math.max(1, side - 8));
    }

    const glyph = drawIcon(ctx, claim.icon, width - ICON_SIZE - 5, 5, ICON_SIZE, theme.fgMuted);
    regions.push({ ...glyph, title: iconTitle(claim.icon, claim.detail) });
    const draggable = canDrag(m);
    if (draggable) regions.push({ x: 0, y: 0, width, height, title: STAGE_HINT });
    stage.style.cursor = state.dragging || draggable ? "nwse-resize" : "default";
    titles.set(regions);
    state.seen = watchKey();
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
        console.error(`[${EXT_NAME}] Failed to draw the frames:`, error);
      }
    });
  }

  /**
   * What the panel was last drawn from that nothing on this node announces a change to.
   *
   * @returns {string} The upstream picture's size and every value read through a wire.
   */
  function watchKey() {
    const source = readSource();
    const settings = readSettings();
    return JSON.stringify([source?.size, source?.from, settings.megapixels, settings.steps,
      settings.mode, settings.crop, settings.unit, settings.wired]);
  }

  /**
   * Ask for the node's last run report and repaint from it.
   *
   * @returns {Promise<void>}
   */
  async function readReport() {
    if (state.disposed) return;
    try {
      const answer = await fetchRunResult(node);
      if (answer.state === PREVIEW_STATE.READY) state.report = answer.result;
      else if (answer.state === PREVIEW_STATE.WAITING) state.report = null;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the last run:`, error);
    }
    schedulePaint();
  }

  /**
   * The megapixels a pointer position asks for, measured along the source frame's diagonal.
   *
   * @param {PointerEvent} event - The pointer event.
   * @returns {number|null} Megapixels on the widget's step, or null with no frame to measure.
   */
  function megapixelsAt(event) {
    const drag = state.dragging;
    if (!drag) return null;
    const at = elementPoint(stage, event);
    const dx = Math.abs(at.x - drag.layout.cx);
    const dy = Math.abs(at.y - drag.layout.cy);
    const halfW = (drag.source[0] * drag.layout.scale) / 2;
    const halfH = (drag.source[1] * drag.layout.scale) / 2;
    const k = (dx * halfW + dy * halfH) / Math.max(1e-6, halfW * halfW + halfH * halfH);
    const widget = widgetOf(node, "megapixels");
    const low = Number(widget?.options?.min ?? MIN_MEGAPIXELS);
    const high = Number(widget?.options?.max ?? MAX_MEGAPIXELS);
    const asked = (k * k * drag.source[0] * drag.source[1]) / drag.unit;
    return Number(Math.min(high, Math.max(low, asked)).toFixed(2));
  }

  /**
   * Write megapixels as one undoable change.
   *
   * @param {number} value - The new value.
   * @returns {void}
   */
  function commit(value) {
    const widget = widgetOf(node, "megapixels");
    if (!widget || !Number.isFinite(value) || widget.value === value) return;
    withGraphChange(() => {
      widget.value = value;
    });
    widget.callback?.(value);
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Finish a drag, writing the megapixels it ended on.
   *
   * @returns {void}
   */
  function endDrag() {
    if (!state.dragging) return;
    const value = state.dragging.megapixels;
    state.dragging = null;
    commit(value);
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
    if (event.button !== 0) return;
    const m = model();
    if (!canDrag(m)) return;
    event.preventDefault();
    event.stopPropagation();
    stage.setPointerCapture?.(event.pointerId);
    state.dragging = {
      source: m.source.size,
      unit: m.settings.unit,
      layout: frameLayout(m.source.size, m.out, stage.clientWidth, stage.clientHeight),
      megapixels: m.settings.megapixels,
    };
    state.dragging.megapixels = megapixelsAt(event) ?? m.settings.megapixels;
    schedulePaint();
  }));
  stage.addEventListener("pointermove", guard((event) => {
    if (event.buttons & 4) app.canvas?.processMouseMove?.(event);
    if (!state.dragging) return;
    if (!(event.buttons & 1)) {
      endDrag();
      return;
    }
    const next = megapixelsAt(event);
    if (next !== null && next !== state.dragging.megapixels) {
      state.dragging.megapixels = next;
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

  const releaseWheel = captureWheel(root);
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => schedulePaint()) : null;
  observer?.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, () => schedulePaint());
  const stopTheme = onThemeChange(() => schedulePaint());
  const stopDone = onNodeDone(node, () => readReport());
  const stopSettling = whenReadable(node, () => readReport());
  // A picture chosen upstream or a wired value changing announces nothing to this node.
  const watch = setInterval(() => {
    try {
      if (!state.disposed && !state.dragging && root.isConnected && watchKey() !== state.seen) schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to look upstream:`, error);
    }
  }, WATCH_MS);

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    refresh() {
      readReport();
    },
    schedulePaint,
    dispose() {
      state.disposed = true;
      if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
      clearInterval(watch);
      releaseWheel();
      observer?.disconnect();
      stopRatio?.();
      stopTheme?.();
      stopDone?.();
      stopSettling?.();
      titles.dispose();
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
      category: ["WAS Node Suite", "Scale Image to Total Pixels", "Frames"],
      name: "Show the frame panel",
      tooltip:
        "Draw the source frame against the scaled frame under Scale Image to Total Pixels "
        + "(Advanced)'s widgets, with a corner to drag that sets megapixels. The widgets work "
        + "either way. This applies to nodes added after the setting changes, so a reload shows "
        + "it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    const proto = nodeType.prototype;
    if (proto.__was_scale_total_wrapped) return;
    proto.__was_scale_total_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachPanel(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the frame panel:`, error);
      }
      return result;
    };
  },
});
