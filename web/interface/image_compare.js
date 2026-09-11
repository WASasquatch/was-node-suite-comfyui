/**
 * Two images under a divider, drawn on the node that received them.
 *
 * The panel is a canvas sized to the surface it is drawn on. A batch is compared pair by pair,
 * one tab per pair along the top.
 */

import { app } from "../../../scripts/app.js";
import { createMetricsSection } from "./metrics_panel.js";
import { captureWheel, wheelPixels } from "./pointer.js";
import { PREVIEW_STATE, fetchInputPreview } from "./preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onRunEnded } from "./run_events.js";
import { createFrameTabs, readableBytes } from "./report_panel.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";

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

const SLOT_A = "image_a";
const SLOT_B = "image_b";

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
  // How many frames each side last reported holding, which is what the fallback for a shorter
  // batch is measured against.
  let counts = { [SLOT_A]: 0, [SLOT_B]: 0 };
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
   * @param {string} slot - Which input to read.
   * @param {number} index - The pair on show, counting from 0.
   * @returns {Promise<object>} What `fetchInputPreview` resolved to.
   */
  const readSide = async (slot, index) => {
    const known = counts[slot];
    let answer = await fetchInputPreview(node, slot, known > 0 ? Math.min(index, known - 1) : index);
    if (answer.frameCount) counts[slot] = answer.frameCount;
    if (!answer.image && index > 0) {
      // The stored batch is shorter than the count this panel was holding.
      const head = await fetchInputPreview(node, slot, 0);
      counts[slot] = head.frameCount || 0;
      answer = counts[slot] > 1
        ? await fetchInputPreview(node, slot, counts[slot] - 1)
        : head;
    }
    return answer;
  };

  /**
   * Load one pair and draw it.
   *
   * @param {number} index - Which pair to show.
   * @returns {Promise<void>}
   */
  const show = async (index) => {
    if (disposed) return;
    current = Math.max(0, Math.min(index, Math.max(0, pairs - 1)));
    try {
      const [a, b] = await Promise.all([
        readSide(SLOT_A, current),
        readSide(SLOT_B, current),
      ]);
      if (disposed) return;
      held = { a: a.image, b: b.image };
      pairs = Math.max(a.frameCount || 1, b.frameCount || 1);
      // A batch that came back shorter than the last one leaves the tab past its end, so the
      // pair on show is the last one there is rather than a number no side holds.
      current = Math.max(0, Math.min(current, pairs - 1));
      const standing = {
        a: Boolean(a.image) && current > (counts[SLOT_A] || 1) - 1,
        b: Boolean(b.image) && current > (counts[SLOT_B] || 1) - 1,
      };
      const ready = a.state === PREVIEW_STATE.READY || b.state === PREVIEW_STATE.READY;
      footer.textContent = ready
        ? describe(a, b, current, pairs, standing)
        : (a.label || b.label || "");
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

  const refresh = () => { show(current); };

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

  // A run replaces what the node published, so the panel asks again rather than holding a
  // picture from a graph that has since changed.
  const stopWatchingRuns = onRunEnded(() => refresh());

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
 * @param {object} answer - What `fetchInputPreview` resolved to.
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
