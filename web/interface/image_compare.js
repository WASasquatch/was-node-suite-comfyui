/**
 * Two images under a divider, drawn on the node that received them.
 *
 * A canvas sized to its surface, one tab per pair of a batch, drawn from the files the node's
 * run output names.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addressOf, hold, release } from "./compare_media.js";
import { createMetricsSection } from "./metrics_panel.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { executionId, nodeLocator, ownOutput } from "./preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onRunEnded } from "./run_events.js";
import { createFrameTabs, readableBytes } from "./report_panel.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";
import { graphLoading, keepOnNode } from "./widget.js";

const LOG_NAME = "WASNodeSuite.ImageCompare";

// The height the panel is drawn at before the node is resized. Enough for a tab strip, a
// picture and the footer without scrolling.
const PANEL_HEIGHT = 380;

// The narrowest the panel is worth drawing in, in node units: a 16:9 picture still wide enough to
// judge a difference across, and roughly ten tabs before the strip has to be scrolled. Without it
// the node refits to its sockets and leaves the panel standing outside itself.
const PANEL_MIN_WIDTH = 320;

// The panel's own padding and the gap between its rows, in CSS pixels, which the picture's box
// is measured against.
const PANEL_PADDING = 4;
const PANEL_GAP = 3;

// What the measurements are left, in CSS pixels, before the picture may take any more room:
// the figures, the histogram and its legend. The picture gets everything above this.
const METRICS_MIN = 104;

// The run output keys the node names each side's written frames under.
const OUTPUT_A = "a_images";
const OUTPUT_B = "b_images";

// Where the last pair of written sides is kept. `properties` is serialised with the node and no
// python reads it.
const REMEMBERED_KEY = "was_compare_images";

/**
 * Decode one written frame, held in the page while it decodes.
 *
 * @param {string} address - Where it is served from.
 * @returns {Promise<HTMLImageElement|null>} The image, or null when it could not be read or
 *   decoded.
 */
async function loadImage(address) {
  const shown = await hold(address);
  if (!shown) return null;
  try {
    return await new Promise((resolve) => {
      const picture = new Image();
      picture.onload = () => resolve(picture);
      picture.onerror = () => resolve(null);
      picture.src = shown;
    });
  } finally {
    release(address);
  }
}

/**
 * Build the panel one Image Compare node draws in.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @returns {{element: HTMLElement, height: number, refresh: () => void, dispose: () => void}}
 *   The panel, for `appendInterfaceWidget`.
 */
export function createImageComparePanel(node) {
  const root = document.createElement("div");
  root.className = "was-image-compare";
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "justify-content:flex-start",
    "gap:3px",
    "padding:4px",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
    "border-radius:4px",
    "overflow:hidden",
  ].join(";");

  const stage = document.createElement("div");
  stage.style.cssText = [
    "position:relative",
    "flex:0 0 auto",
    "align-self:center",
    "min-height:0",
    "overflow:hidden",
    "border-radius:3px",
  ].join(";");
  root.appendChild(stage);

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  stage.appendChild(canvas);

  // Over the picture rather than above it. The tabs are only wanted while choosing a pair, and
  // a row of its own costs the picture that height on every node, batch or not. `show` is
  // declared below, so the pick is wrapped rather than passed: reading it here would reach it
  // before it is initialised.
  const tabs = createFrameTabs((index) => show(index), { overlay: true });
  const strip = tabs.element;
  stage.appendChild(strip);

  const footer = document.createElement("div");
  footer.style.cssText = "font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    + `flex:0 0 auto;color:${themeVar("fgMuted")}`;
  root.appendChild(footer);

  const metrics = createMetricsSection();
  root.appendChild(metrics.element);

  // Where the divider sits, as a fraction of the picture's width.
  let split = 0.5;
  let pairs = 1;
  let current = 0;
  let held = { a: null, b: null };
  // The frames each side was written as, in batch order.
  let sides = { a: [], b: [] };
  // Address -> the decoding of it, for the frames of the pair of sides on show.
  const decoded = new Map();
  // Counts every read, so an answer that lands after a newer read began is dropped.
  let reading = 0;
  let disposed = false;
  let dragging = false;

  /**
   * The largest box of the picture's shape that fits the room the panel has for it.
   *
   * @param {number} room - The width available, in CSS pixels.
   * @param {number} ceiling - The height available, in CSS pixels.
   * @returns {{width: number, height: number}} The box the stage is sized to.
   */
  const boxFor = (room, ceiling) => {
    const image = held.a || held.b;
    const aspect = image ? (image.naturalWidth || 1) / (image.naturalHeight || 1) : 1;
    const width = Math.max(1, Math.min(room, ceiling * aspect));
    return { width, height: Math.max(1, width / aspect) };
  };

  /** Draw the pair, with image_b revealed to the right of the divider. */
  const draw = () => {
    if (disposed) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;
    const theme = readTheme();
    ctx.clearRect(0, 0, width, height);

    const a = held.a;
    const b = held.b;
    if (!a && !b) return;
    // The stage is already the picture's shape, so both fill it corner to corner and the
    // divider means the same thing on each even when the pair differ in size.
    if (a) ctx.drawImage(a, 0, 0, width, height);
    if (b) {
      const cut = width * split;
      ctx.save();
      ctx.beginPath();
      ctx.rect(cut, 0, width - cut, height);
      ctx.clip();
      ctx.drawImage(b, 0, 0, width, height);
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = theme.fg;
      ctx.lineWidth = Math.max(1, Math.round(width / 600));
      ctx.beginPath();
      ctx.moveTo(cut, 0);
      ctx.lineTo(cut, height);
      ctx.stroke();
      // A handle, so the divider reads as something to drag rather than a seam.
      ctx.fillStyle = theme.fg;
      const r = Math.max(4, Math.round(width / 160));
      ctx.beginPath();
      ctx.arc(cut, height / 2, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  };

  /**
   * Move the divider to where a pointer is.
   *
   * @param {PointerEvent} event - The pointer event on the stage.
   * @returns {void}
   */
  const moveSplit = (event) => {
    const rect = stage.getBoundingClientRect();
    if (!rect.width) return;
    split = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    draw();
  };

  stage.addEventListener("pointerdown", (event) => {
    // Middle button panning belongs to the canvas underneath, and so does anything that is
    // not a plain left press.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    dragging = true;
    stage.setPointerCapture?.(event.pointerId);
    moveSplit(event);
    event.stopPropagation();
    event.preventDefault();
  });
  stage.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    moveSplit(event);
    event.stopPropagation();
  });
  const endDrag = (event) => {
    if (!dragging) return;
    dragging = false;
    stage.releasePointerCapture?.(event.pointerId);
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);

  /** Draw the tab strip for the current pair count. */
  const drawStrip = () => {
    tabs.draw(pairs, current);
  };

  /**
   * One side of a pair, held at its last frame once the tab is past the end of it.
   *
   * @param {object[]} frames - The side's written frames.
   * @param {number} index - The pair on show, counting from 0.
   * @returns {Promise<object>} The image and what the frame says about the picture it was
   *   written from: `sourceWidth`, `sourceHeight`, `mode`, `bytes` and `frameCount`.
   */
  const readSide = async (frames, index) => {
    if (!frames.length) return { image: null, frameCount: 0 };
    const entry = frames[Math.min(index, frames.length - 1)];
    const address = addressOf(entry);
    if (!decoded.has(address)) decoded.set(address, loadImage(address));
    return {
      image: await decoded.get(address),
      sourceWidth: Number(entry.width) || 0,
      sourceHeight: Number(entry.height) || 0,
      mode: String(entry.mode ?? ""),
      bytes: Number(entry.bytes) || 0,
      frameCount: frames.length,
    };
  };

  /**
   * Load one pair and draw it.
   *
   * @param {number} index - Which pair to show.
   * @returns {Promise<void>}
   */
  const show = async (index) => {
    if (disposed) return;
    const read = ++reading;
    pairs = Math.max(sides.a.length, sides.b.length, 1);
    current = Math.max(0, Math.min(index, pairs - 1));
    try {
      const [a, b] = await Promise.all([readSide(sides.a, current), readSide(sides.b, current)]);
      if (disposed || read !== reading) return;
      held = { a: a.image, b: b.image };
      const standing = {
        a: Boolean(a.image) && current > sides.a.length - 1,
        b: Boolean(b.image) && current > sides.b.length - 1,
      };
      footer.textContent = a.image || b.image ? describe(a, b, current, pairs, standing) : "";
      drawStrip();
      sizeCanvas();
      metrics.update(a.image, b.image, {
        aWidth: a.sourceWidth, aHeight: a.sourceHeight,
        bWidth: b.sourceWidth, bHeight: b.sourceHeight,
      });
      node.setDirtyCanvas?.(true, false);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the comparison:`, error);
    }
  };

  /**
   * The written frames a run output names, when this workflow's run wrote them.
   *
   * @param {object} output - A node's run output, or nothing.
   * @returns {{a: object[], b: object[]}|null} Both sides, or null when the output names no
   *   frame or names one another workflow's run wrote.
   */
  const sidesOf = (output) => {
    const a = Array.isArray(output?.[OUTPUT_A]) ? output[OUTPUT_A] : [];
    const b = Array.isArray(output?.[OUTPUT_B]) ? output[OUTPUT_B] : [];
    if (!a.length && !b.length) return null;
    return [...a, ...b].every(ownOutput) ? { a, b } : null;
  };

  /**
   * Show a pair of written sides and keep it on the node.
   *
   * @param {{a: object[], b: object[]}} next - Both sides.
   * @param {boolean} keep - Whether to write it into the node's properties.
   * @returns {void}
   */
  const take = (next, keep) => {
    const before = `${sides.a.map(addressOf).join()}|${sides.b.map(addressOf).join()}`;
    const after = `${next.a.map(addressOf).join()}|${next.b.map(addressOf).join()}`;
    sides = { a: next.a, b: next.b };
    if (before !== after) {
      const wanted = new Set([...sides.a, ...sides.b].map(addressOf));
      for (const address of [...decoded.keys()]) if (!wanted.has(address)) decoded.delete(address);
    }
    const kept = node.properties?.[REMEMBERED_KEY];
    if (keep && JSON.stringify(kept) !== JSON.stringify({ a: sides.a, b: sides.b })) {
      keepOnNode(node, REMEMBERED_KEY, { a: sides.a, b: sides.b });
    }
    show(current);
  };

  /**
   * Read whatever the last run of this workflow wrote, or the pair kept on the node.
   *
   * @returns {void}
   */
  const refresh = () => {
    try {
      const store = app?.nodeOutputs ?? {};
      const output = store[nodeLocator(node)] ?? store[executionId(node)] ?? store[String(node.id)];
      const fresh = sidesOf(output);
      if (fresh) {
        take(fresh, true);
        return;
      }
      const kept = node.properties?.[REMEMBERED_KEY];
      take({
        a: Array.isArray(kept?.a) ? kept.a : [],
        b: Array.isArray(kept?.b) ? kept.b : [],
      }, false);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the written sides:`, error);
    }
  };

  // The node's own report of what it wrote, read as it arrives rather than from the store.
  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      const fresh = sidesOf(detail?.output);
      if (fresh) take(fresh, true);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's report:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  // A pasted node reads its kept pair here; a loaded graph is read once its outputs are back.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const configured = originalOnConfigure?.apply(this, args);
    if (!graphLoading()) refresh();
    return configured;
  };

  // The backing store is sized from the stage, which is laid out by flex and so has no size at
  // all until the panel is on the page. Two things change it: the graph's zoom, which
  // `watchSurfaceRatio` reports, and the node being resized, which only a size observer sees.
  const sizeCanvas = () => {
    // Layout pixels, not `getBoundingClientRect`. The graph draws the whole panel through a CSS
    // transform, so a measured rectangle is already multiplied by the zoom; writing one back as
    // a CSS size would scale it a second time and the picture would shrink as the graph is
    // zoomed out. `clientWidth` and `offsetHeight` are untransformed.
    const room = root.clientWidth - PANEL_PADDING * 2;
    if (room <= 0) return;
    // What the picture may have, once the metadata line, the measurements, the gaps and the
    // padding have taken theirs. The strip is not counted: it is drawn over the picture.
    const taken = footer.offsetHeight
      + METRICS_MIN
      + PANEL_PADDING * 2
      + PANEL_GAP * 2;
    const box = boxFor(room, Math.max(1, root.clientHeight - taken));
    stage.style.width = `${Math.round(box.width)}px`;
    stage.style.height = `${Math.round(box.height)}px`;

    // Read here rather than carried from the watcher, which calls back with no arguments. The
    // ratio counts the graph's zoom as well as the display's density, so the backing store is
    // rebuilt at the resolution the picture is actually being drawn at and a zoomed in
    // comparison is sharp rather than a magnified bitmap.
    const ratio = surfaceRatio(canvas);
    const width = Math.max(1, Math.round(box.width * ratio));
    const height = Math.max(1, Math.round(box.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    draw();
  };

  const stopWatching = watchSurfaceRatio(canvas, () => sizeCanvas());

  // The panel is watched rather than the stage, since the stage is now sized from the panel
  // and watching it would react to its own change.
  const observer = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => sizeCanvas())
    : null;
  observer?.observe(root);

  // The strip is the only thing here that scrolls, and only sideways, so a wheel over it walks
  // the tabs. Everywhere else the gesture is the graph's and zooms the node under the pointer.
  const releaseWheel = captureWheel(root, (event) => {
    if (!strip.contains(event.target) || strip.scrollWidth <= strip.clientWidth) return false;
    const step = wheelPixels(event, strip);
    const before = strip.scrollLeft;
    strip.scrollLeft += step.x + step.y;
    return strip.scrollLeft !== before;
  });

  // The store is read again whenever a run ends, however it ended.
  const stopWatchingRuns = onRunEnded(() => refresh());

  // A frame the server could not answer while it was away is read again once it is back.
  const onReconnected = async () => {
    let missing = false;
    for (const [address, decoding] of [...decoded.entries()]) {
      if (!(await decoding)) {
        decoded.delete(address);
        missing = true;
      }
    }
    if (missing && !disposed) show(current);
  };
  api.addEventListener("reconnected", onReconnected);

  // The divider is drawn into a canvas, where a custom property means nothing, so the palette
  // is subscribed to and the pair drawn again.
  const stopTheme = onThemeChange(() => draw());

  refresh();

  return {
    element: root,
    height: PANEL_HEIGHT,
    // No ceiling worth naming: the node's own height is the bound, so dragging the node taller
    // gives every extra pixel to the picture rather than leaving it empty below the panel.
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel();
      try {
        if (typeof stopWatchingRuns === "function") stopWatchingRuns();
        api.removeEventListener?.("executed", onExecuted);
        api.removeEventListener?.("reconnected", onReconnected);
        decoded.clear();
        metrics.dispose();
        observer?.disconnect();
        stopWatching?.();
        stopTheme();
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to release the comparison:`, error);
      }
    },
  };
}

/**
 * One picture as its size, its channel mode and the bytes it encoded to.
 *
 * @param {object} answer - What `readSide` answered for one side.
 * @param {boolean} standing - Whether this is the side's last frame drawn for a pair the side
 *   does not reach.
 * @returns {string} A phrase such as `1856x2254 RGB 3.4 MB`, or `none` for no picture.
 */
function measure(answer, standing) {
  if (!answer?.image) return "none";
  const width = answer.sourceWidth || answer.image.naturalWidth;
  const height = answer.sourceHeight || answer.image.naturalHeight;
  const mode = answer.mode || "RGB";
  const size = answer.bytes > 0 ? ` ${readableBytes(answer.bytes)}` : "";
  const stand = standing ? " last frame" : "";
  return `${width}x${height} ${mode}${size}${stand}`;
}

/**
 * The footer line for one pair.
 *
 * @param {object} a - The answer for image_a.
 * @param {object} b - The answer for image_b.
 * @param {number} index - The pair on show, counting from 0.
 * @param {number} pairs - How many pairs there are.
 * @param {object} standing - Which sides are drawing their last frame for a pair they do not
 *   reach, as `{a, b}`.
 * @returns {string} A line describing both pictures, and which of the pairs they are.
 */
function describe(a, b, index, pairs, standing = {}) {
  const where = pairs > 1 ? `${index + 1} of ${pairs}   ` : "";
  return `${where}A: ${measure(a, standing.a)}   B: ${measure(b, standing.b)}`;
}
