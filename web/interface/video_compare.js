/**
 * Two videos under a divider, played from one clock.
 *
 * The panel stacks both clips and clips the right-hand one, so dragging the divider reveals
 * one against the other at the same frame.
 */

import { app } from "../../../scripts/app.js";
import { onRunEnded } from "./run_events.js";
import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.VideoCompare";

// Height the panel is drawn at before the node is resized: a 16:9 clip and its controls.
const PANEL_HEIGHT = 320;

// The narrowest the panel is worth drawing in, in node units.
const PANEL_MIN_WIDTH = 300;

// Where the divider starts, as a percentage across.
const START_SPLIT = 50;

// How far the two clocks may drift before the right-hand clip is pulled back into step.
const DRIFT_SECONDS = 0.06;

// Width of the divider, at rest and under the pointer.
const LINE_WIDTH = 2;
const LINE_HOVER = 3;

// Width of the transparent strip the divider is dragged by.
const GRIP_WIDTH = 14;

/**
 * The address a written side is served from.
 *
 * @param {object} entry - A side's filename, subfolder and type.
 * @returns {string} The view address, or an empty string.
 */
function addressOf(entry) {
  if (!entry?.filename) return "";
  const parts = new URLSearchParams({
    filename: entry.filename,
    subfolder: entry.subfolder ?? "",
    type: entry.type ?? "temp",
  });
  return `/api/view?${parts.toString()}`;
}

/**
 * Build the two-video panel for one node.
 *
 * @param {object} node - The node the panel belongs to.
 * @returns {object} The element, its height, its minimum width, `dispose` and `refresh`.
 */
export function createVideoComparePanel(node) {
  const root = document.createElement("div");
  root.className = "was-video-compare";
  Object.assign(root.style, {
    position: "relative", width: "100%", height: "100%",
    display: "flex", flexDirection: "column", gap: "4px",
    background: themeVar("panelBg"),
    borderRadius: "4px", overflow: "hidden", boxSizing: "border-box", padding: "4px",
  });

  const stage = document.createElement("div");
  Object.assign(stage.style, {
    position: "relative", flex: "1 1 auto", minHeight: "0", overflow: "hidden",
    background: "#000", borderRadius: "3px",
  });

  const make = () => {
    const el = document.createElement("video");
    el.muted = true;
    el.loop = true;
    el.playsInline = true;
    el.preload = "auto";
    Object.assign(el.style, {
      position: "absolute", inset: "0", width: "100%", height: "100%",
      objectFit: "contain", display: "none",
    });
    return el;
  };
  const left = make();
  const right = make();
  stage.append(left, right);

  const divider = document.createElement("div");
  Object.assign(divider.style, {
    position: "absolute", top: "0", bottom: "0", width: `${LINE_WIDTH}px`,
    background: themeVar("fg"), display: "none", pointerEvents: "none",
  });
  const grip = document.createElement("div");
  Object.assign(grip.style, {
    position: "absolute", top: "0", bottom: "0", width: `${GRIP_WIDTH}px`,
    cursor: "ew-resize", display: "none", background: "transparent",
  });
  stage.append(divider, grip);

  const empty = document.createElement("div");
  empty.textContent = "run once to compare";
  Object.assign(empty.style, {
    position: "absolute", inset: "0", display: "flex",
    alignItems: "center", justifyContent: "center",
    color: themeVar("fgMuted"), fontSize: "11px", textAlign: "center",
  });
  stage.append(empty);

  const bar = document.createElement("div");
  Object.assign(bar.style, {
    display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto",
    fontSize: "11px", color: themeVar("fg"),
  });
  const toggle = document.createElement("button");
  toggle.textContent = "play";
  Object.assign(toggle.style, {
    fontSize: "11px", padding: "1px 8px", cursor: "pointer",
    background: themeVar("panelBg"),
    color: themeVar("fg"),
    border: `1px solid ${themeVar("border")}`, borderRadius: "3px",
  });
  const scrub = document.createElement("input");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = "1000";
  scrub.value = "0";
  Object.assign(scrub.style, { flex: "1 1 auto", minWidth: "0" });
  const readout = document.createElement("span");
  readout.textContent = "0.00 / 0.00 s";
  Object.assign(readout.style, { flex: "0 0 auto", fontVariantNumeric: "tabular-nums" });
  bar.append(toggle, scrub, readout);

  root.append(stage, bar);

  let split = START_SPLIT;
  let hovering = false;
  const dragging = { on: false };
  const paint = () => {
    right.style.clipPath = `inset(0 0 0 ${split}%)`;
    const line = hovering ? LINE_HOVER : LINE_WIDTH;
    divider.style.width = `${line}px`;
    // Placed against the same percentage the clip-path uses, so the canvas zoom cannot put
    // the two out of step. The clamp keeps both inside the stage at either end.
    const centre = (span) => `clamp(0px, calc(${split}% - ${span / 2}px), calc(100% - ${span}px))`;
    divider.style.left = centre(line);
    grip.style.left = centre(GRIP_WIDTH);
  };

  grip.addEventListener("pointerenter", () => { hovering = true; paint(); });
  grip.addEventListener("pointerleave", () => { if (!dragging.on) { hovering = false; paint(); } });

  const positionFrom = (event) => {
    const box = stage.getBoundingClientRect();
    if (!(box.width > 0)) return;
    split = Math.max(0, Math.min(100, ((event.clientX - box.left) / box.width) * 100));
    paint();
  };
  // Bubble phase: the widget root claims a pointer on capture, which would otherwise take
  // the event before the panel sees it.
  stage.addEventListener("pointerdown", (event) => {
    dragging.on = true;
    hovering = true;
    stage.setPointerCapture?.(event.pointerId);
    positionFrom(event);
    event.stopPropagation();
  });
  stage.addEventListener("pointermove", (event) => {
    if (!dragging.on) return;
    positionFrom(event);
    event.stopPropagation();
  });
  stage.addEventListener("pointerup", (event) => {
    dragging.on = false;
    hovering = false;
    paint();
    stage.releasePointerCapture?.(event.pointerId);
    event.stopPropagation();
  });

  const both = () => [left, right].filter((el) => el.style.display !== "none");
  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const playing = !left.paused && left.currentSrc;
    for (const el of both()) {
      if (playing) el.pause();
      else el.play().catch(() => {});
    }
    toggle.textContent = playing ? "play" : "pause";
  });
  scrub.addEventListener("input", (event) => {
    event.stopPropagation();
    const span = left.duration || right.duration || 0;
    if (!(span > 0)) return;
    const at = (Number(scrub.value) / 1000) * span;
    for (const el of both()) el.currentTime = at;
  });

  left.addEventListener("timeupdate", () => {
    const span = left.duration || 0;
    if (span > 0) {
      scrub.value = String(Math.round((left.currentTime / span) * 1000));
      readout.textContent = `${left.currentTime.toFixed(2)} / ${span.toFixed(2)} s`;
    }
    if (right.currentSrc && Math.abs(right.currentTime - left.currentTime) > DRIFT_SECONDS) {
      right.currentTime = left.currentTime;
    }
  });

  /**
   * Point both players at whatever the last run wrote.
   *
   * @returns {void}
   */
  function refresh() {
    try {
      const output = app?.nodeOutputs?.[node.id] ?? app?.nodeOutputs?.[String(node.id)];
      const a = addressOf(output?.a_video?.[0]);
      const b = addressOf(output?.b_video?.[0]);
      const stamp = `&rand=${Math.random()}`;
      for (const [el, address] of [[left, a], [right, b]]) {
        if (address) {
          el.src = address + stamp;
          el.style.display = "block";
        } else {
          el.removeAttribute("src");
          el.style.display = "none";
        }
      }
      const any = Boolean(a || b);
      empty.style.display = any ? "none" : "flex";
      divider.style.display = a && b ? "block" : "none";
      grip.style.display = a && b ? "block" : "none";
      paint();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the written sides:`, error);
    }
  }

  const stop = onRunEnded(() => refresh());
  paint();
  refresh();

  const dispose = () => {
    try {
      stop?.();
      for (const el of [left, right]) {
        el.pause();
        el.removeAttribute("src");
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to release the players:`, error);
    }
  };

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    dispose,
    refresh,
  };
}
