/**
 * A band drawing one picture a node published, for a report panel to embed.
 *
 * Answers the `{element, update, clear, dispose}` a report panel's `sketch` factory takes,
 * so a node whose answer is worth looking at shows it beside its figures.
 */

import { LABELS, PREVIEW_STATE, fetchOutputPreview, watchPreviews } from "./preview.js";
import { onNodeFinished, onRunEnded } from "./run_events.js";
import { whenReadable } from "./run_result.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.PictureBand";

// The shortest the band is drawn in CSS pixels. Past that it takes the node's spare room.
const BAND_HEIGHT = 92;

// The checkerboard behind a picture carrying transparency.
const CHECKER = 8;

// The largest backing store either side is given, whatever a stray layout measures.
const MAX_EDGE = 4096;

/**
 * Build a band that draws one picture a node answered with.
 *
 * @param {object} node - The node the panel is on.
 * @param {object} [options] - `slot` names the output the picture left on, `label` is the
 *   word drawn under it, `height` overrides the band height in CSS pixels, and `overlay` is
 *   called as `overlay(pen, {x, y, width, height, ratio})` after the picture is drawn.
 * @returns {{element: HTMLElement, update: Function, clear: Function, dispose: Function,
 *   show: Function, refresh: Function}} The band, in the shape a report panel's sketch takes;
 *   `show(image, text)` draws a picture handed to it until the node next answers, and
 *   `refresh` reads the node's picture again.
 */
export function createPictureBand(node, options = {}) {
  const slot = String(options.slot ?? "");
  const label = String(options.label ?? "result");
  const height = Number(options.height) > 0 ? Number(options.height) : BAND_HEIGHT;

  const band = document.createElement("div");
  band.className = "was-picture-band";
  // Grows faster than the fact rows beside it, so dragging the node taller shows more
  // picture rather than more empty table.
  band.style.cssText = "display:flex;flex-direction:column;gap:4px;flex:3 1 auto;"
    + `min-height:${height}px;overflow:hidden`;

  // The canvas is taken out of flow inside a frame of its own. In flow, a full-height
  // canvas feeds its own height back into the band and the two grow without stopping.
  const frame = document.createElement("div");
  frame.style.cssText = "position:relative;flex:1 1 auto;min-width:0;min-height:0;"
    + "align-self:stretch";
  band.appendChild(frame);

  const stage = document.createElement("canvas");
  stage.style.cssText = "position:absolute;inset:0;width:100%;height:100%;"
    + "border-radius:3px;display:block";
  frame.appendChild(stage);

  // One line under the picture, so the picture keeps the band's whole width.
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;align-items:baseline;gap:8px;min-width:0;"
    + "flex:0 0 auto;overflow:hidden;white-space:nowrap";
  band.appendChild(legend);

  const title = document.createElement("div");
  title.style.cssText = `flex:0 0 auto;color:${themeVar("fgMuted")}`;
  title.textContent = label;
  legend.appendChild(title);

  const detail = document.createElement("div");
  detail.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `color:${themeVar("fg")}`;
  legend.appendChild(detail);

  let picture = null;
  let words = "";
  let disposed = false;
  let expectedPromptId = "";

  /**
   * Paint the picture, or the words standing in for it, at the graph's own resolution.
   *
   * @returns {void}
   */
  const paint = () => {
    if (disposed) return;
    const theme = readTheme();

    // Layout pixels times the ratio, which already holds the graph's zoom; both sides shrink
    // together past the largest edge so the picture keeps its shape.
    const layoutWide = Math.max(1, stage.clientWidth);
    const layoutTall = Math.max(1, stage.clientHeight);
    const wanted = surfaceRatio(stage);
    const ratio = wanted * Math.min(1, MAX_EDGE / (Math.max(layoutWide, layoutTall) * wanted));
    const wide = Math.max(1, Math.round(layoutWide * ratio));
    const tall = Math.max(1, Math.round(layoutTall * ratio));
    if (stage.width !== wide) stage.width = wide;
    if (stage.height !== tall) stage.height = tall;

    const pen = stage.getContext("2d");
    if (!pen) return;
    pen.clearRect(0, 0, wide, tall);

    // A checkerboard, so a cut-out reads as transparent rather than as black.
    const step = CHECKER * ratio;
    for (let y = 0; y < tall; y += step) {
      for (let x = 0; x < wide; x += step) {
        pen.fillStyle = ((x / step | 0) + (y / step | 0)) % 2 ? theme.bg : theme.borderLight;
        pen.fillRect(x, y, step, step);
      }
    }

    if (!picture) {
      pen.fillStyle = theme.fgMuted;
      pen.font = `${11 * ratio}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace`;
      pen.textAlign = "center";
      pen.textBaseline = "middle";
      pen.fillText(words || "no picture yet", wide / 2, tall / 2);
      return;
    }

    const scale = Math.min(wide / picture.naturalWidth, tall / picture.naturalHeight);
    const drawWide = Math.max(1, Math.round(picture.naturalWidth * scale));
    const drawTall = Math.max(1, Math.round(picture.naturalHeight * scale));
    const left = Math.round((wide - drawWide) / 2);
    const top = Math.round((tall - drawTall) / 2);
    pen.imageSmoothingEnabled = true;
    pen.drawImage(picture, left, top, drawWide, drawTall);
    if (typeof options.overlay === "function") {
      try {
        options.overlay(pen, { x: left, y: top, width: drawWide, height: drawTall, ratio });
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw over the picture:`, error);
      }
    }
  };

  /**
   * Ask for the picture this node last answered with, and paint whatever comes back.
   *
   * @returns {Promise<void>}
   */
  const load = async () => {
    if (disposed) return;
    try {
      const answer = await fetchOutputPreview(node, slot, 0, expectedPromptId);
      if (disposed) return;
      if (answer.state === PREVIEW_STATE.READY && answer.image) {
        picture = answer.image;
        words = "";
        const source = answer.sourceWidth && answer.sourceHeight
          ? `${answer.sourceWidth} x ${answer.sourceHeight}`
          : "";
        detail.textContent = [source, answer.mode].filter(Boolean).join("  ");
      } else {
        picture = null;
        words = LABELS[answer.state] ?? "";
        detail.textContent = "";
      }
      paint();
      node.setDirtyCanvas?.(true, false);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the picture:`, error);
    }
  };

  const release = watchPreviews(node);
  const stopFinished = onNodeFinished(node, (info) => {
    expectedPromptId = info.promptId;
    // A cached node published nothing new, so the last drawing is still what the graph did.
    if (info.cached) return;
    load();
  });
  const stopEnded = onRunEnded(() => load());
  const stopRatio = watchSurfaceRatio(stage, () => paint());
  // The picture is drawn into a canvas, which takes literal colours, so a palette change repaints.
  const stopTheme = onThemeChange(() => paint());
  const resize = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => paint())
    : null;
  resize?.observe(frame);

  load();
  // A band built while its workflow loads has no node id yet, so it reads again once it has one.
  const stopSettling = whenReadable(node, () => load());

  return {
    element: band,
    update() {
      paint();
    },
    refresh: load,
    clear() {
      picture = null;
      words = "";
      detail.textContent = "";
      paint();
    },
    show(image, text = "") {
      if (disposed || !image) return;
      picture = image;
      words = "";
      detail.textContent = String(text ?? "");
      paint();
    },
    dispose() {
      disposed = true;
      try { release(); } catch (error) { /* the subscription is already gone */ }
      try { stopFinished(); } catch (error) { /* the listener is already gone */ }
      try { stopEnded(); } catch (error) { /* the listener is already gone */ }
      try { stopSettling(); } catch (error) { /* the wait is already over */ }
      try { stopRatio(); } catch (error) { /* the watcher is already gone */ }
      try { stopTheme(); } catch (error) { /* the listener is already gone */ }
      resize?.disconnect();
    },
  };
}
