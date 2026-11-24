/**
 * The action timeline drawn on H3 World Actions, and the sentences its latents read.
 *
 * Two lanes, Move and Camera, hold spans of keys. The lanes are read from and written to the
 * node's `actions` widget, as `start-end: words` entries.
 */

import { app } from "../../scripts/app.js";
import { captureWheel } from "./interface/pointer.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget, boundTextBoxes, chainWidgetCallback } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.H3WorldActionsUI";
const LOG_NAME = "WASNodeSuite.H3WorldActions";
const SETTING_ID = "WAS.H3.ShowWorldTimeline";
const NODE_ID = "WASH3WorldActions";
const WIDGET = "actions";

const UI_WIDGET_NAME = "was_h3_world_ui";
const UI_WIDGET_TYPE = "was_h3_world";

// Height in node units the panel opens at: the summary, the counts, the lanes, the keys and a
// few sentence rows.
const PANEL_HEIGHT = 250;

// Height of one lane in CSS pixels, and of the time scale under both.
const LANE_HEIGHT = 26;
const SCALE_HEIGHT = 14;

// How wide an edge is to grab, in CSS pixels either side.
const GRAB = 6;

// Seconds a time snaps to, and the length a new span opens at.
const SNAP = 0.1;
const NEW_SPAN = 1.0;

// The axis a node opens on: 124 frames at 24 fps.
const DEFAULT_LENGTH = 5.2;

// The largest backing store the strip is given.
const MAX_EDGE = 4096;

// Keys in the order the official table lists them, with the word an entry writes for each.
const KEY_WORDS = {
  W: "forward", A: "strafe-left", S: "back", D: "strafe-right",
  I: "tilt-down", J: "pan-left", K: "tilt-up", L: "pan-right", F: "fast",
};
const KEYS = Object.keys(KEY_WORDS);

// Words an entry takes, each the keys it holds.
const PRESETS = {
  still: [], forward: ["W"], back: ["S"], "strafe-left": ["A"], "strafe-right": ["D"],
  "tilt-down": ["I"], "tilt-up": ["K"], "pan-left": ["J"], "pan-right": ["L"],
  "pan-left-fast": ["J", "F"], "pan-right-fast": ["L", "F"], fast: ["F"],
};

// The two lanes, the keys each holds and what a new span in it holds.
const LANES = [
  { name: "Move", keys: ["W", "A", "S", "D"], fresh: ["W"], tint: "#5fae6e" },
  { name: "Camera", keys: ["J", "L", "I", "K", "F"], fresh: ["J"], tint: "#5b8fe0" },
];

// A short label per key, drawn on a span.
const KEY_LABEL = {
  W: "fwd", A: "left", S: "back", D: "right", J: "pan L", L: "pan R", I: "tilt dn", K: "tilt up", F: "fast",
};

const ENTRY = /^\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*-\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*:?\s*(.*?)\s*$/;

/**
 * Read an action script as the node reads it.
 *
 * @param {string} text - The widget's text.
 * @returns {{entries: Array<{start: number, end: number, keys: Set<string>}>, problem: string}}
 *   The entries read, and the first entry that could not be, or an empty string.
 */
export function parseActions(text) {
  const entries = [];
  for (const part of String(text ?? "").split(/[;\n\r]+/)) {
    if (!part.trim()) continue;
    const found = ENTRY.exec(part);
    if (!found) return { entries, problem: part.trim() };
    const start = Number(found[1]);
    const end = Number(found[2]);
    if (!(end > start)) return { entries, problem: part.trim() };
    const keys = new Set();
    for (const word of found[3].trim().toLowerCase().split(/[\s,+]+/)) {
      if (!word) continue;
      if (word in PRESETS) PRESETS[word].forEach((key) => keys.add(key));
      else if (word.length === 1 && KEYS.includes(word.toUpperCase())) keys.add(word.toUpperCase());
      else return { entries, problem: part.trim() };
    }
    entries.push({ start, end, keys });
  }
  return { entries, problem: "" };
}

/**
 * One lane's spans: the entries' keys in that lane, joined where they overlap.
 *
 * @param {Array} entries - What `parseActions` answered.
 * @param {object} lane - One of `LANES`.
 * @returns {Array<{start: number, end: number, keys: Set<string>}>} Spans in time order, none
 *   overlapping.
 */
function laneSpans(entries, lane) {
  const own = entries
    .map((entry) => ({ ...entry, keys: new Set([...entry.keys].filter((key) => lane.keys.includes(key))) }))
    .filter((entry) => entry.keys.size > 0);
  const edges = [...new Set(own.flatMap((entry) => [entry.start, entry.end]))].sort((a, b) => a - b);
  const spans = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const keys = new Set();
    for (const entry of own) {
      if (entry.start <= edges[i] && entry.end >= edges[i + 1]) entry.keys.forEach((key) => keys.add(key));
    }
    if (!keys.size) continue;
    const last = spans[spans.length - 1];
    const same = last && last.end === edges[i] && last.keys.size === keys.size
      && [...keys].every((key) => last.keys.has(key));
    if (same) last.end = edges[i + 1];
    else spans.push({ start: edges[i], end: edges[i + 1], keys });
  }
  return spans;
}

/**
 * Write both lanes back as a script, the Move entries first.
 *
 * @param {Array<Array>} lanes - Each lane's spans, in `LANES` order.
 * @returns {string} The script.
 */
function writeActions(lanes) {
  const time = (seconds) => String(Math.round(seconds * 1000) / 1000);
  const lines = [];
  for (const spans of lanes) {
    for (const span of [...spans].sort((a, b) => a.start - b.start)) {
      const words = KEYS.filter((key) => span.keys.has(key)).map((key) => KEY_WORDS[key]);
      lines.push(`${time(span.start)}-${time(span.end)}: ${words.join(" ") || "still"}`);
    }
  }
  return lines.join("\n");
}

/**
 * Whether the timeline is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * Build the two lanes and the key chips under them.
 *
 * @param {object} node - The node the timeline edits.
 * @returns {{element: HTMLElement, update: Function, clear: Function, refresh: Function,
 *   dispose: Function}} The band, in the shape `createReportPanel` takes as its sketch.
 */
function createActionTimeline(node) {
  const widget = () => (node.widgets ?? []).find((candidate) => candidate?.name === WIDGET);

  const element = document.createElement("div");
  element.tabIndex = 0;
  element.style.cssText = "flex:0 0 auto;display:flex;flex-direction:column;gap:4px;outline:none";

  const frame = document.createElement("div");
  frame.style.cssText = `position:relative;height:${LANES.length * LANE_HEIGHT + SCALE_HEIGHT}px;flex:0 0 auto`;
  element.appendChild(frame);
  const strip = document.createElement("canvas");
  strip.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;border-radius:3px;"
    + "touch-action:none";
  frame.appendChild(strip);

  const chips = document.createElement("div");
  chips.style.cssText = "display:flex;flex-wrap:wrap;gap:3px;align-items:center;flex:0 0 auto";
  element.appendChild(chips);

  const note = document.createElement("div");
  note.style.cssText = `font-size:10px;color:${themeVar("fgMuted")};flex:0 0 auto;white-space:nowrap;`
    + "overflow:hidden;text-overflow:ellipsis";
  element.appendChild(note);

  let disposed = false;
  let lanes = [[], []];
  let problem = "";
  let selected = null;
  let grip = null;
  let reported = 0;

  /** Read both lanes from the widget. */
  const load = () => {
    const parsed = parseActions(widget()?.value ?? "");
    problem = parsed.problem;
    lanes = LANES.map((lane) => laneSpans(parsed.entries, lane));
    if (selected && !lanes[selected.lane]?.[selected.index]) selected = null;
  };

  /** How far the axis runs, in seconds. */
  const axisLength = () => {
    const last = Math.max(0, ...lanes.flat().map((span) => span.end));
    return Math.max(DEFAULT_LENGTH, reported, last + NEW_SPAN);
  };

  /** Put both lanes on the widget and repaint. */
  const commit = () => {
    const target = widget();
    if (!target) return;
    const text = writeActions(lanes);
    if (target.value !== text) {
      target.value = text;
      target.callback?.(text);
    }
    node.setDirtyCanvas?.(true, true);
    paint();
  };

  const button = (label, title, onClick) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = label;
    chip.title = title;
    chip.style.cssText = `font:600 10px system-ui,sans-serif;padding:2px 6px;border-radius:3px;cursor:pointer;`
      + `border:1px solid ${themeVar("border")};background:${themeVar("bgLight")};color:${themeVar("fg")}`;
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      onClick();
      element.focus();
    });
    chips.appendChild(chip);
    return chip;
  };

  /** Hold or release a key on the selected span. */
  const toggle = (key) => {
    if (!selected) return;
    const span = lanes[selected.lane][selected.index];
    if (!LANES[selected.lane].keys.includes(key)) return;
    if (span.keys.has(key)) span.keys.delete(key);
    else span.keys.add(key);
    if (!span.keys.size) {
      lanes[selected.lane].splice(selected.index, 1);
      selected = null;
    }
    commit();
  };

  const keyChips = KEYS.map((key) => [key, button(key, `${KEY_WORDS[key]} (press ${key})`, () => toggle(key))]);
  /** Cut the selected span in two at its middle. */
  const split = () => {
    if (!selected) return;
    const spans = lanes[selected.lane];
    const span = spans[selected.index];
    const middle = Math.round((span.start + span.end) / 2 / SNAP) * SNAP;
    if (middle <= span.start || middle >= span.end) return;
    spans.splice(selected.index + 1, 0, { start: middle, end: span.end, keys: new Set(span.keys) });
    span.end = middle;
    commit();
  };
  button("split", "Split the selected span in two (press /)", split);
  button("delete", "Remove the selected span (press Delete)", () => {
    if (!selected) return;
    lanes[selected.lane].splice(selected.index, 1);
    selected = null;
    commit();
  });

  /** Light the chips the selected span can take, and those it holds. */
  const paintChips = (theme) => {
    for (const [key, chip] of keyChips) {
      const usable = selected && LANES[selected.lane].keys.includes(key);
      const held = usable && lanes[selected.lane][selected.index].keys.has(key);
      chip.disabled = !usable;
      chip.style.opacity = usable ? "1" : "0.4";
      chip.style.background = held ? theme.accent : theme.bgLight;
      chip.style.color = held ? theme.selectionText : theme.fg;
    }
  };

  /** Draw the lanes, the spans and the scale. */
  const paint = () => {
    if (disposed) return;
    const theme = readTheme();
    const wide0 = Math.max(1, strip.clientWidth);
    const tall0 = Math.max(1, strip.clientHeight);
    const wanted = surfaceRatio(strip);
    const ratio = wanted * Math.min(1, MAX_EDGE / (Math.max(wide0, tall0) * wanted));
    const wide = Math.max(1, Math.round(wide0 * ratio));
    const tall = Math.max(1, Math.round(tall0 * ratio));
    if (strip.width !== wide) strip.width = wide;
    if (strip.height !== tall) strip.height = tall;
    const pen = strip.getContext("2d");
    if (!pen) return;
    pen.setTransform(ratio, 0, 0, ratio, 0, 0);
    pen.clearRect(0, 0, wide0, tall0);
    pen.fillStyle = theme.bgDark;
    pen.fillRect(0, 0, wide0, tall0);

    const length = axisLength();
    const x = (seconds) => (seconds / length) * wide0;
    pen.font = "9px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";
    pen.textBaseline = "middle";
    LANES.forEach((lane, index) => {
      const top = index * LANE_HEIGHT;
      pen.fillStyle = index % 2 ? theme.bg : theme.bgLight;
      pen.fillRect(0, top, wide0, LANE_HEIGHT);
      lanes[index].forEach((span, spanIndex) => {
        const left = x(span.start);
        const right = x(span.end);
        const picked = selected && selected.lane === index && selected.index === spanIndex;
        pen.fillStyle = lane.tint;
        pen.globalAlpha = picked ? 0.95 : 0.6;
        pen.fillRect(left, top + 3, Math.max(2, right - left), LANE_HEIGHT - 6);
        pen.globalAlpha = 1;
        if (picked) {
          pen.strokeStyle = theme.fg;
          pen.lineWidth = 1;
          pen.strokeRect(left + 0.5, top + 3.5, Math.max(2, right - left) - 1, LANE_HEIGHT - 7);
        }
        const label = KEYS.filter((key) => span.keys.has(key)).map((key) => KEY_LABEL[key]).join(" + ");
        pen.save();
        pen.beginPath();
        pen.rect(left, top, Math.max(0, right - left), LANE_HEIGHT);
        pen.clip();
        pen.fillStyle = theme.fg;
        pen.textAlign = "left";
        pen.fillText(label, left + 4, top + LANE_HEIGHT / 2);
        pen.restore();
      });
      pen.fillStyle = theme.fgMuted;
      pen.textAlign = "right";
      pen.fillText(lane.name, wide0 - 4, top + 8);
    });

    const scaleTop = LANES.length * LANE_HEIGHT;
    const step = length > 12 ? 2 : length > 6 ? 1 : 0.5;
    pen.strokeStyle = theme.borderLight;
    pen.fillStyle = theme.fgMuted;
    pen.textBaseline = "top";
    for (let mark = 0; mark <= length + 1e-9; mark += step) {
      const at = Math.round(x(mark)) + 0.5;
      pen.beginPath();
      pen.moveTo(at, 0);
      pen.lineTo(at, scaleTop + 3);
      pen.stroke();
      pen.textAlign = mark === 0 ? "left" : "center";
      pen.fillText(`${mark}s`, Math.min(at + 1, wide0 - 10), scaleTop + 2);
    }
    if (reported > 0) {
      pen.strokeStyle = theme.warning;
      pen.setLineDash([3, 3]);
      const end = Math.round(x(reported)) + 0.5;
      pen.beginPath();
      pen.moveTo(end, 0);
      pen.lineTo(end, scaleTop);
      pen.stroke();
      pen.setLineDash([]);
    }
    paintChips(theme);
    note.style.color = problem ? theme.warning : theme.fgMuted;
    note.textContent = problem
      ? `cannot read "${problem}"; the node refuses it until it is fixed`
      : "click a lane to add a span, drag its edges, press its keys; gaps stand still";
  };

  /** Which lane, span and part a pointer is over. */
  const hit = (event) => {
    const box = strip.getBoundingClientRect();
    const scale = box.width ? strip.clientWidth / box.width : 1;
    const px = (event.clientX - box.left) * scale;
    const py = (event.clientY - box.top) * (box.height ? strip.clientHeight / box.height : 1);
    const lane = Math.floor(py / LANE_HEIGHT);
    const length = axisLength();
    const seconds = Math.max(0, Math.min(length, (px / Math.max(1, strip.clientWidth)) * length));
    if (lane < 0 || lane >= LANES.length) return { lane: -1, seconds };
    const perSecond = strip.clientWidth / length;
    for (let index = 0; index < lanes[lane].length; index++) {
      const span = lanes[lane][index];
      if (Math.abs(seconds - span.start) * perSecond <= GRAB) return { lane, index, part: "start", seconds };
      if (Math.abs(seconds - span.end) * perSecond <= GRAB) return { lane, index, part: "end", seconds };
      if (seconds > span.start && seconds < span.end) return { lane, index, part: "body", seconds };
    }
    return { lane, seconds };
  };

  const snap = (seconds) => Math.max(0, Math.round(seconds / SNAP) * SNAP);

  strip.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    const at = hit(event);
    if (at.lane < 0) return;
    if (at.index === undefined) {
      const spans = lanes[at.lane];
      const start = snap(at.seconds);
      const next = spans.find((span) => span.start >= start);
      const end = Math.min(start + NEW_SPAN, next ? next.start : Infinity);
      if (end - start < SNAP) return;
      spans.push({ start, end, keys: new Set(LANES[at.lane].fresh) });
      spans.sort((a, b) => a.start - b.start);
      selected = { lane: at.lane, index: spans.findIndex((span) => span.start === start) };
      commit();
      return;
    }
    selected = { lane: at.lane, index: at.index };
    const span = lanes[at.lane][at.index];
    grip = { part: at.part, from: at.seconds, start: span.start, end: span.end };
    strip.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    element.focus();
    paint();
  });

  strip.addEventListener("pointermove", (event) => {
    if (!grip || !selected) {
      const at = hit(event);
      strip.style.cursor = at.part === "start" || at.part === "end" ? "ew-resize"
        : at.part === "body" ? "grab" : at.lane >= 0 ? "copy" : "default";
      return;
    }
    const spans = lanes[selected.lane];
    const span = spans[selected.index];
    const before = spans[selected.index - 1];
    const after = spans[selected.index + 1];
    const floor = before ? before.end : 0;
    const ceiling = after ? after.start : axisLength() + NEW_SPAN;
    const at = hit(event).seconds;
    if (grip.part === "start") span.start = Math.min(Math.max(floor, snap(at)), span.end - SNAP);
    else if (grip.part === "end") span.end = Math.max(Math.min(ceiling, snap(at)), span.start + SNAP);
    else {
      const width = grip.end - grip.start;
      const start = Math.min(Math.max(floor, snap(grip.start + at - grip.from)), ceiling - width);
      span.start = start;
      span.end = start + width;
    }
    commit();
  });

  const release = (event) => {
    if (!grip) return;
    grip = null;
    strip.releasePointerCapture?.(event.pointerId);
    paint();
  };
  strip.addEventListener("pointerup", release);
  strip.addEventListener("pointercancel", release);

  element.addEventListener("keydown", (event) => {
    const key = event.key.toUpperCase();
    if (KEYS.includes(key) && !event.ctrlKey && !event.metaKey && !event.altKey) {
      toggle(key);
    } else if ((event.key === "Delete" || event.key === "Backspace") && selected) {
      lanes[selected.lane].splice(selected.index, 1);
      selected = null;
      commit();
    } else if (event.key === "/" && selected) {
      split();
    } else if (event.key === "Escape") {
      selected = null;
      paint();
    } else {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
  });

  for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel"]) {
    element.addEventListener(type, (event) => event.stopPropagation());
  }
  const releaseWheel = captureWheel(element);
  const stopRatio = watchSurfaceRatio(strip, () => paint());
  const stopTheme = onThemeChange(() => paint());
  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => paint()) : null;
  observer?.observe(frame);
  chainWidgetCallback(node, WIDGET, () => { load(); paint(); }, LOG_NAME);
  load();
  paint();

  return {
    element,
    update(report) {
      const seconds = (report?.counts ?? []).find((entry) => entry.name === "seconds");
      reported = seconds ? Number(seconds.value) || 0 : 0;
      load();
      paint();
    },
    clear() {
      reported = 0;
      paint();
    },
    refresh() {
      load();
      paint();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel();
      observer?.disconnect();
      stopRatio?.();
      stopTheme?.();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "MiniMax H3", "Show the world timeline"],
      name: "Draw the action timeline on H3 World Actions",
      tooltip:
        "Draw a Move lane and a Camera lane on H3 World Actions to set held keys by dragging, with "
        + "the sentences each latent read after a run. The node reads only its actions text either way. "
        + "Applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second panel.
    if (proto.__was_h3_world_wrapped) return;
    proto.__was_h3_world_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const node = this;
        const panel = createReportPanel(node, {
          className: "was-h3-world-actions",
          height: PANEL_HEIGHT,
          emptyLabel: "set the keys, then run to see each latent's sentence",
          sketch: () => createActionTimeline(node),
          logName: LOG_NAME,
          failure: "Failed to read what the actions encoded:",
        });
        appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(node);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the action timeline:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the action timeline:`, error);
      }
      return result;
    };
  },
});
