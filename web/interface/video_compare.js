/**
 * Two videos under a divider, played from one clock.
 *
 * Both clips are drawn into one canvas, the right-hand one cut at the divider, so dragging the
 * divider reveals one against the other at the same frame.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { executionId } from "./preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onRunEnded } from "./run_events.js";
import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.VideoCompare";

// Height the panel is drawn at before the node is resized: a 16:9 clip and its controls.
const PANEL_HEIGHT = 320;

// The narrowest the panel is worth drawing in, in node units.
const PANEL_MIN_WIDTH = 300;

// Where the divider starts, as a percentage across.
const START_SPLIT = 50;

// How far the following clip may sit from the leading one before it is seeked rather than steered.
const SEEK_SECONDS = 0.5;

// How far the following clip may sit from the leading one and still play at the same rate.
const SETTLE_SECONDS = 0.02;

// Seconds a drift is closed over by changing the following clip's rate.
const CATCH_UP_SECONDS = 0.5;

// The furthest the following clip's rate is moved from the leading one's, as a share.
const RATE_REACH = 0.15;

// How far before its end a shorter clip is parked, so its last frame stays on screen.
const HOLD_SECONDS = 0.01;

// Width of the divider, at rest and under the pointer.
const LINE_WIDTH = 2;
const LINE_HOVER = 3;

// Width of the transparent strip the divider is dragged by.
const GRIP_WIDTH = 14;

// Where the last pair of written sides is kept. `properties` is serialised with the node and no
// python reads it.
const REMEMBERED_KEY = "was_compare_sides";

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
      objectFit: "contain", display: "none", opacity: "0", pointerEvents: "none",
    });
    return el;
  };
  const left = make();
  const right = make();

  // Both clips are drawn here, the right-hand one cut at the divider.
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, {
    position: "absolute", inset: "0", width: "100%", height: "100%", display: "block",
  });
  const context = canvas.getContext("2d");
  stage.append(left, right, canvas);

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
  let paired = false;
  const dragging = { on: false };

  const ready = (el) => el.style.display !== "none" && el.readyState >= 2 && el.videoWidth > 0 && !el.seeking;

  // The last frame each clip presented, kept so a clip that is seeking shows that frame.
  const held = new Map([[left, null], [right, null]]);
  const keep = (el, width, height) => {
    let copy = held.get(el);
    if (!copy || copy.width !== width || copy.height !== height) {
      const resized = document.createElement("canvas");
      resized.width = width;
      resized.height = height;
      if (copy) resized.getContext("2d")?.drawImage(copy, 0, 0, width, height);
      held.set(el, resized);
      copy = resized;
    }
    if (ready(el)) {
      const pen = copy.getContext("2d");
      pen?.clearRect(0, 0, width, height);
      pen?.drawImage(el, ...fitted(el, width, height));
    }
    return el.style.display !== "none" && el.videoWidth > 0 ? copy : null;
  };
  // The clip's own aspect, centred in the canvas, as `object-fit: contain` places it.
  const fitted = (el, width, height) => {
    const scale = Math.min(width / el.videoWidth, height / el.videoHeight);
    const w = el.videoWidth * scale;
    const h = el.videoHeight * scale;
    return [(width - w) / 2, (height - h) / 2, w, h];
  };
  const draw = () => {
    if (!context) return;
    const ratio = surfaceRatio(canvas);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const leftFrame = keep(left, width, height);
    const rightFrame = keep(right, width, height);
    context.clearRect(0, 0, width, height);
    if (leftFrame) context.drawImage(leftFrame, 0, 0);
    if (!rightFrame) return;
    context.save();
    if (paired) {
      context.beginPath();
      context.rect((width * split) / 100, 0, width, height);
      context.clip();
    }
    context.drawImage(rightFrame, 0, 0);
    context.restore();
  };
  let pending = 0;
  const playing = () => [left, right].some((el) => !el.paused && !el.ended);
  // One draw per display frame while either clip plays, and one per event while both rest.
  const tick = () => {
    pending = 0;
    steer();
    draw();
    if (playing()) pending = requestAnimationFrame(tick);
  };
  const schedule = () => {
    if (!pending) pending = requestAnimationFrame(tick);
  };
  for (const el of [left, right]) {
    for (const name of ["play", "seeked", "loadeddata", "timeupdate"]) el.addEventListener(name, schedule);
  }
  const resized = new ResizeObserver(() => schedule());
  resized.observe(stage);
  const stopRatio = watchSurfaceRatio(canvas, schedule);

  const paint = () => {
    const line = hovering ? LINE_HOVER : LINE_WIDTH;
    divider.style.width = `${line}px`;
    // Placed against the same percentage the clip-path uses, so the canvas zoom cannot put
    // the two out of step. The clamp keeps both inside the stage at either end.
    const centre = (span) => `clamp(0px, calc(${split}% - ${span / 2}px), calc(100% - ${span}px))`;
    divider.style.left = centre(line);
    grip.style.left = centre(GRIP_WIDTH);
    schedule();
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
  const length = (el) => (el.currentSrc && el.duration > 0 ? el.duration : 0);
  // The longer clip keeps the time and loops; the shorter one holds its last frame until the
  // longer one wraps round.
  const clock = () => (length(right) > length(left) ? right : left);
  const lastFrame = (el) => Math.max(0, length(el) - HOLD_SECONDS);
  toggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const lead = clock();
    const playing = !lead.paused && lead.currentSrc;
    for (const el of both()) {
      if (playing) el.pause();
      else if (el === lead || el.currentTime < lastFrame(el)) el.play().catch(() => {});
    }
    toggle.textContent = playing ? "play" : "pause";
  });
  scrub.addEventListener("input", (event) => {
    event.stopPropagation();
    const span = length(clock());
    if (!(span > 0)) return;
    const at = (Number(scrub.value) / 1000) * span;
    for (const el of both()) el.currentTime = Math.min(at, lastFrame(el));
  });

  // The following clip is held to the leading one by its playback rate, and seeked when far out.
  function steer() {
    const lead = clock();
    const other = lead === left ? right : left;
    lead.loop = true;
    other.loop = false;
    if (!other.currentSrc || !(length(other) > 0) || other.seeking) return;
    if (lead.currentTime >= lastFrame(other)) {
      if (!other.paused) other.pause();
      other.playbackRate = 1;
      if (Math.abs(other.currentTime - lastFrame(other)) > SEEK_SECONDS) {
        other.currentTime = lastFrame(other);
      }
      return;
    }
    const drift = other.currentTime - lead.currentTime;
    if (lead.paused || Math.abs(drift) > SEEK_SECONDS) {
      other.playbackRate = 1;
      if (Math.abs(drift) > SETTLE_SECONDS) other.currentTime = lead.currentTime;
    } else if (Math.abs(drift) > SETTLE_SECONDS) {
      const nudge = Math.max(-RATE_REACH, Math.min(RATE_REACH, drift / CATCH_UP_SECONDS));
      other.playbackRate = lead.playbackRate * (1 - nudge);
    } else {
      other.playbackRate = lead.playbackRate;
    }
    if (other.paused && !lead.paused) other.play().catch(() => {});
  }

  const followed = () => {
    const lead = clock();
    const span = length(lead);
    if (span > 0) {
      scrub.value = String(Math.round((lead.currentTime / span) * 1000));
      readout.textContent = `${lead.currentTime.toFixed(2)} / ${span.toFixed(2)} s`;
    }
    steer();
  };
  left.addEventListener("timeupdate", followed);
  right.addEventListener("timeupdate", followed);
  left.addEventListener("loadedmetadata", followed);
  right.addEventListener("loadedmetadata", followed);

  let loaded = "";

  /**
   * Point both players at one pair of written sides.
   *
   * @param {object} sides - `a` and `b`, each a written file's entry or nothing.
   * @returns {boolean} True when the pair named at least one file.
   */
  function show(sides) {
    const a = addressOf(sides?.a);
    const b = addressOf(sides?.b);
    if (!a && !b) return false;
    const pair = `${a}|${b}`;
    if (pair === loaded) return true;
    loaded = pair;
    held.set(left, null);
    held.set(right, null);
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
    paired = Boolean(a && b);
    empty.style.display = "none";
    divider.style.display = paired ? "block" : "none";
    grip.style.display = paired ? "block" : "none";
    readout.textContent = "0.00 / 0.00 s";
    scrub.value = "0";
    paint();
    return true;
  }

  /**
   * Keep a pair on the node, so a reloaded or reopened workflow shows it again.
   *
   * @param {object} sides - `a` and `b`.
   * @returns {void}
   */
  function remember(sides) {
    node.properties ??= {};
    node.properties[REMEMBERED_KEY] = { a: sides?.a ?? null, b: sides?.b ?? null };
  }

  const sidesOf = (output) => ({ a: output?.a_video?.[0], b: output?.b_video?.[0] });

  /**
   * Point both players at whatever the last run wrote, or the pair kept on the node.
   *
   * @returns {void}
   */
  function refresh() {
    try {
      const store = app?.nodeOutputs ?? {};
      const sides = sidesOf(store[executionId(node)] ?? store[node.id] ?? store[String(node.id)]);
      if (show(sides)) {
        remember(sides);
        return;
      }
      show(node.properties?.[REMEMBERED_KEY]);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the written sides:`, error);
    }
  }

  // The node's own report of what it wrote, read as it arrives rather than from the store.
  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      const sides = sidesOf(detail?.output);
      if (show(sides)) remember(sides);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's report:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  // A side that no longer exists, such as a temp file cleared by a restart, leaves the prompt.
  for (const el of [left, right]) {
    el.addEventListener("error", () => {
      el.style.display = "none";
      if (left.style.display === "none" && right.style.display === "none") {
        loaded = "";
        empty.style.display = "flex";
        divider.style.display = "none";
        grip.style.display = "none";
      }
      schedule();
    });
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const configured = originalOnConfigure?.apply(this, args);
    refresh();
    return configured;
  };

  const stop = onRunEnded(() => refresh());
  paint();
  refresh();

  const dispose = () => {
    try {
      stop?.();
      api.removeEventListener?.("executed", onExecuted);
      stopRatio();
      resized.disconnect();
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      for (const el of [left, right]) {
        el.pause();
        el.removeAttribute("src");
      }
      held.clear();
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
