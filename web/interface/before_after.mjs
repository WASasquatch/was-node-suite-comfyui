/**
 * The picture a filter was handed against the picture it answered with.
 *
 * `createBeforeAfterPanel` draws a strip of thumbnails, a band of measurements and one line
 * describing both sides. The A side is always the before.
 */

import { ICON, ICON_SIZE, drawFidelityGlyph, iconTitle } from "./icons.mjs";
import { createMetricsSection, workingSize } from "./metrics_panel.mjs";
import { loadPlaceholder, placeholderPicture, standInDetail } from "./placeholder.mjs";
import {
  PREVIEW_STATE,
  fetchInputPreview,
  fetchPreviewPair,
  watchPreviews,
} from "./preview.mjs";
import { captureWheel } from "./pointer.mjs";
import { readableBytes } from "./report_panel.mjs";
import { onNodeFinished, onRunEnded } from "./run_events.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.mjs";
import { onThemeChange, themeVar } from "./theme.mjs";

const LOG_NAME = "WASNodeSuite.BeforeAfter";

/** Height in node units the panel opens at: the strip, the measurements and the footer. */
export const PANEL_HEIGHT = 200;

/** The narrowest the band is worth drawing in, which is what five figures and a plot need. */
export const MIN_WIDTH = 260;

// The longest edge the pair is measured at.
const WORKING_EDGE = 512;

// The shortest a thumbnail is drawn, in CSS pixels. Below this a picture says nothing at all,
// so the strip keeps its room rather than collapsing into the measurements.
//: The two greys a transparent area is drawn against, and the side of one square in CSS
// pixels, matching the swatches drawn elsewhere in the pack.
const CHECKER_LIGHT = "#999999";
const CHECKER_DARK = "#666666";
const CHECKER_SIDE = 8;

/**
 * Fill a canvas with the checkerboard a transparent area is read against.
 *
 * @param {CanvasRenderingContext2D} ctx - The context to fill.
 * @param {number} width - Backing store width in pixels.
 * @param {number} height - Backing store height in pixels.
 * @param {number} ratio - Backing store pixels per CSS pixel.
 * @returns {void}
 */
function paintChecker(ctx, width, height, ratio) {
  const side = Math.max(2, Math.round(CHECKER_SIDE * ratio));
  ctx.fillStyle = CHECKER_LIGHT;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = CHECKER_DARK;
  for (let y = 0; y < height; y += side) {
    for (let x = ((y / side) % 2) * side; x < width; x += side * 2) {
      ctx.fillRect(x, y, side, side);
    }
  }
}

const THUMB_MIN = 40;

/**
 * The size a preview answer says the node itself held.
 *
 * @param {object|null} answer - An answer from `preview.mjs`.
 * @returns {{width: number, height: number}} The source size, falling back to the decoded one.
 */
function sourceSize(answer) {
  const image = answer?.image;
  return {
    width: Number(answer?.sourceWidth) || image?.naturalWidth || 0,
    height: Number(answer?.sourceHeight) || image?.naturalHeight || 0,
  };
}

/**
 * Whether one side's picture arrived at the size the node held it at.
 *
 * @param {object|null} answer - An answer from `preview.mjs`.
 * @returns {boolean} True when the decoded picture is the source size.
 */
function unreduced(answer) {
  const image = answer?.image;
  if (!image) return false;
  const source = sourceSize(answer);
  return image.naturalWidth === source.width && image.naturalHeight === source.height;
}

/**
 * What the figures drawn are worth as a measurement of what the node did.
 *
 * @param {object|null} before - The before answer.
 * @param {object|null} after - The after answer, or null when it is not measured.
 * @returns {{icon: string, detail: string}} The glyph to draw and what it says on hover.
 */
export function pairFidelity(before, after) {
  if (!before?.image) {
    return { icon: ICON.WARNING, detail: standInDetail(before?.state) };
  }
  const a = sourceSize(before);
  const work = workingSize(a.width, a.height, WORKING_EDGE);
  const measured = after?.image ? [before, after] : [before];
  if (!work.scaled && measured.every((one) => unreduced(one))) {
    return {
      icon: ICON.EXACT,
      detail: measured.length === 2
        ? `both sides counted at every one of their ${a.width}x${a.height} pixels`
        : `the before counted at every one of its ${a.width}x${a.height} pixels, `
          + "and the after is not measured",
    };
  }
  const b = after?.image ? sourceSize(after) : null;
  return {
    icon: ICON.APPROXIMATE,
    detail: `counted at ${work.width}x${work.height}, from a before of ${a.width}x${a.height}`
      + (b ? ` and an after of ${b.width}x${b.height}` : " and no after"),
  };
}

/**
 * One side written out: the size the node held, the channels and the encoded length.
 *
 * @param {string} name - What the side is called.
 * @param {object|null} answer - An answer from `preview.mjs`.
 * @returns {string} The description, or the state's own words where no picture arrived.
 */
function describeSide(name, answer) {
  if (!answer?.image) {
    return `${name} ${answer?.label || "not published yet"}`;
  }
  const size = sourceSize(answer);
  const frames = Number(answer.frameTotal) || 1;
  return [
    `${name} ${size.width}x${size.height}`,
    answer.mode || "",
    readableBytes(answer.bytes),
    frames > 1 ? `${frames} frames` : "",
  ].filter(Boolean).join(" ");
}

/**
 * One captioned cell of the thumbnail strip.
 *
 * @param {string} caption - What the cell is called, drawn under the picture.
 * @returns {{element: HTMLElement, show: Function, predict: Function}} The cell, a `show`
 *   taking a preview answer, which falls back to the pack's stand-in where no picture
 *   arrived, and a `predict` drawing another cell's picture through a CSS filter.
 */
function createCell(caption) {
  const element = document.createElement("div");
  element.style.cssText = "display:flex;flex-direction:column;gap:2px;flex:1 1 0;"
    + `min-width:${THUMB_MIN}px;min-height:0;overflow:hidden`;

  const frame = document.createElement("div");
  frame.style.cssText = `flex:1 1 auto;min-height:${THUMB_MIN}px;display:flex;`
    + "align-items:center;justify-content:center;border-radius:2px;overflow:hidden;"
    + `background:${themeVar("bgDark")}`;
  const picture = document.createElement("canvas");
  // Contained, never stretched.
  picture.style.cssText = "max-width:100%;max-height:100%;display:block;object-fit:contain";
  frame.appendChild(picture);

  // The last picture drawn.
  let painted = null;

  // A CSS filter the browser applies as the picture is drawn, and the words under it while
  // that is what is on screen. Both empty once a run has answered.
  let predicted = "";
  let predictedLabel = "";

  /**
   * Draw one decoded picture into the cell at its own aspect.
   *
   * @param {HTMLImageElement} image - A decoded picture.
   * @returns {void}
   */
  function paint(image) {
    painted = image;
    const wide = image.naturalWidth || image.width;
    const tall = image.naturalHeight || image.height;
    if (!(wide > 0) || !(tall > 0)) return;
    const ratio = surfaceRatio(frame);
    // Layout pixels, not the zoom-transformed box.
    const roomWide = frame.clientWidth || THUMB_MIN;
    const roomTall = frame.clientHeight || THUMB_MIN;
    const room = Math.max(1, Math.min(roomWide, roomTall * wide / tall));
    const width = Math.max(1, Math.round(room * ratio));
    const height = Math.max(1, Math.round(room * tall / wide * ratio));
    if (picture.width !== width) picture.width = width;
    if (picture.height !== height) picture.height = height;
    picture.style.width = `${Math.round(width / ratio)}px`;
    picture.style.height = `${Math.round(height / ratio)}px`;
    const ctx = picture.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    // A cut-out is drawn over a checkerboard, so what it does not cover reads as nothing
    // rather than as the colour of whatever sits behind the panel.
    paintChecker(ctx, width, height, ratio);
    ctx.filter = predicted || "none";
    ctx.drawImage(image, 0, 0, width, height);
    ctx.filter = "none";
  }

  const label = document.createElement("div");
  label.style.cssText = "flex:0 0 auto;font-size:9px;white-space:nowrap;overflow:hidden;"
    + `text-overflow:ellipsis;color:${themeVar("fgMuted")}`;
  element.append(frame, label);

  return {
    element,
    /**
     * Draw the picture already in the cell again, at whatever size the cell is now.
     *
     * @returns {void}
     */
    repaint() {
      if (painted) paint(painted);
    },
    /**
     * Put one side's picture in the cell.
     *
     * @param {object|null} answer - An answer from `preview.mjs`.
     * @returns {void}
     */
    show(answer) {
      predicted = "";
      predictedLabel = "";
      if (answer?.image) {
        // Drawn rather than pointed at: preview.mjs releases the object URL once the bytes
        // are decoded, so the address no longer loads while the element it decoded into
        // still holds the picture. Copying .src onto a second element gets a dead URL.
        paint(answer.image);
        picture.style.opacity = "1";
        picture.title = "";
        label.textContent = caption;
        return;
      }
      const standIn = placeholderPicture();
      if (standIn) paint(standIn);
      picture.style.opacity = "0.35";
      picture.title = iconTitle(ICON.WARNING, standInDetail(answer?.state));
      label.textContent = `${caption}  ${answer?.label || "waiting"}`.trim();
    },
    /**
     * Draw another cell's picture through a CSS filter, as what the node would answer.
     *
     * @param {HTMLImageElement|null} image - The picture to draw, usually the before side.
     * @param {string} filter - A CSS filter list, or empty to draw it untouched.
     * @param {string} caption - Words under the picture saying it is not a run.
     * @returns {boolean} Whether anything was drawn.
     */
    predict(image, filter, caption) {
      if (!image) return false;
      predicted = String(filter || "");
      predictedLabel = String(caption || "");
      paint(image);
      picture.style.opacity = "1";
      picture.title = "";
      label.textContent = predictedLabel;
      return true;
    },
  };
}

/**
 * Build the before and after band a pixels node carries.
 *
 * @param {object} node - The node the panel belongs to, for its id and its redraws.
 * @param {object} [options] - How the band is drawn.
 * @param {string} [options.slot] - The socket both sides are filed under, which is the node's
 *   first picture input.
 * @param {string[]} [options.controls] - The node's other picture inputs, each drawn as its own
 *   thumbnail and never measured.
 * @param {number} [options.height] - Height in node units the panel opens at.
 * @param {number} [options.minWidth] - The narrowest it is worth drawing in.
 * @param {string} [options.logName] - The name a failure is logged under.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   refresh: Function, dispose: Function}} The panel, for `appendInterfaceWidget`.
 */
export function createBeforeAfterPanel(node, options = {}) {
  const slot = String(options.slot ?? "");
  const controls = Array.isArray(options.controls) ? options.controls.filter(Boolean) : [];
  const predicting = typeof options.predict === "function" ? options.predict : null;
  const height = Number(options.height) > 0 ? Number(options.height) : PANEL_HEIGHT;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : MIN_WIDTH;
  const logName = options.logName || LOG_NAME;

  const root = document.createElement("div");
  root.className = "was-before-after";
  root.style.cssText = [
    "box-sizing:border-box", "width:100%", "height:100%",
    "display:flex", "flex-direction:column", "gap:6px", "overflow:hidden",
    "padding:8px 10px", "border-radius:4px",
    "font:10px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
  ].join(";");

  const strip = document.createElement("div");
  strip.style.cssText = `display:flex;gap:6px;flex:1 1 auto;min-height:${THUMB_MIN}px`;
  root.appendChild(strip);

  const cells = [createCell("before"), createCell("after")];
  for (const name of controls) cells.push(createCell(name));
  for (const cell of cells) strip.appendChild(cell.element);

  // Refused rather than resampled: a figure taken over a B the browser stretched is a figure
  // about the resampler as much as about the node, which is the one thing this band is for.
  const metrics = createMetricsSection({ mismatch: "refuse", workingEdge: WORKING_EDGE });
  metrics.element.style.flex = "0 0 auto";
  root.appendChild(metrics.element);

  const footer = document.createElement("div");
  footer.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;font-size:9px";
  const glyph = document.createElement("canvas");
  glyph.style.cssText = `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`;
  const note = document.createElement("span");
  note.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${themeVar("fgMuted")}`;
  footer.append(glyph, note);
  root.appendChild(footer);

  let disposed = false;
  let expectedPromptId = "";
  let claim = { icon: ICON.WARNING, detail: standInDetail(PREVIEW_STATE.WAITING) };
  // What the last read answered, held so the arithmetic can be put off until the panel is seen.
  let held = null;
  let dirty = false;
  let onScreen = false;

  /** Draw the fidelity glyph the last reading earned. */
  const drawGlyph = () => {
    drawFidelityGlyph(glyph, claim);
  };

  /**
   * Whether the two sides may be read against each other, and why not where they may not.
   *
   * @param {object} answer - What `fetchPreviewPair` resolved to.
   * @returns {{pair: boolean, reason: string}} Whether to measure the pair, and the words for
   *   a refusal that is not about the two sizes.
   */
  const pairable = (answer) => {
    const { before, after, sameRun } = answer;
    if (before?.state !== PREVIEW_STATE.READY || after?.state !== PREVIEW_STATE.READY) {
      // A side carrying a picture from a run this page did not ask for is a state of its own,
      // and a description that only gave its size would read as an ordinary side.
      const foreign = [before, after].some((one) => one?.state === PREVIEW_STATE.FOREIGN);
      return {
        pair: false,
        reason: foreign
          ? "not compared: a side is from a run this page did not ask for"
          : "",
      };
    }
    if (!sameRun) {
      return { pair: false, reason: "not compared: the two sides are from different runs" };
    }
    const one = Number(before.frameTotal) || 1;
    const two = Number(after.frameTotal) || 1;
    if (one !== two) {
      return {
        pair: false,
        reason: `not compared: the before is ${one} frame${one === 1 ? "" : "s"} and the after `
          + `is ${two}`,
      };
    }
    return { pair: true, reason: "" };
  };

  /** Take the measurements the last reading is owed, once the panel is somewhere to be seen. */
  const measure = () => {
    if (disposed || !held) return;
    dirty = false;
    const { before, after, pair } = held;
    const a = sourceSize(before);
    const b = sourceSize(after);
    metrics.update(
      before.image || null,
      pair ? after.image || null : null,
      { aWidth: a.width, aHeight: a.height, bWidth: b.width, bHeight: b.height },
    );
    node.setDirtyCanvas?.(true, false);
  };

  /**
   * Draw one reading: the thumbnails, the words and the glyph, and queue the arithmetic.
   *
   * @param {object} answer - What `fetchPreviewPair` resolved to.
   * @param {object[]} steering - One answer per control input, in the order they are drawn.
   * @returns {void}
   */
  const draw = (answer, steering) => {
    if (disposed) return;
    const { pair, reason } = pairable(answer);
    held = { before: answer.before, after: answer.after, pair };
    cells[0].show(answer.before);
    cells[1].show(answer.after);
    for (let index = 0; index < controls.length; index += 1) {
      cells[index + 2]?.show(steering[index]);
    }
    // The band refuses a pair of different sizes on its own, so the after counts towards the
    // fidelity claim only where the band is really going to measure it.
    const a = sourceSize(answer.before);
    const b = sourceSize(answer.after);
    const bothMeasured = pair && a.width === b.width && a.height === b.height;
    claim = pairFidelity(answer.before, bothMeasured ? answer.after : null);
    note.textContent = [
      describeSide("before", answer.before),
      describeSide("after", answer.after),
      reason,
    ].filter(Boolean).join("   ");
    drawGlyph();
    node.setDirtyCanvas?.(true, false);
    dirty = true;
    // A pair at the working edge is around a tenth of a second, so ten panels would freeze the
    // canvas for a second after every run. The panels nobody is looking at wait their turn.
    if (onScreen) measure();
  };

  let fetching = false;
  let again = false;

  /** Read both sides and every control, coalescing an ask that arrives while one is in flight. */
  const load = async () => {
    if (disposed) return;
    if (fetching) {
      again = true;
      return;
    }
    fetching = true;
    try {
      do {
        again = false;
        const [answer, steering] = await Promise.all([
          fetchPreviewPair(node, slot, 0, expectedPromptId),
          Promise.all(controls.map((name) => fetchInputPreview(node, name, 0, expectedPromptId))),
        ]);
        draw(answer, steering);
      } while (again && !disposed);
    } catch (error) {
      console.error(`[${logName}] Failed to read what the node did to the picture:`, error);
    } finally {
      fetching = false;
    }
  };

  /**
   * Draw the before side through the node's own filter, as what a run would answer.
   *
   * @returns {boolean} Whether a prediction was drawn.
   */
  // Called as a control moves, so the after side answers the widget rather than the last run.
  function showPrediction() {
    if (disposed || !predicting) return false;
    const source = held?.before?.image || null;
    if (!source) return false;
    let filter = "";
    try {
      filter = String(predicting() || "");
    } catch (error) {
      console.error(`[${logName}] Failed to read the preview filter:`, error);
      return false;
    }
    const drawn = cells[1].predict(source, filter, "after, as the widget stands");
    if (drawn) node.setDirtyCanvas?.(true, false);
    return drawn;
  }

  const release = watchPreviews(node);
  const stopFinished = onNodeFinished(node, (info) => {
    expectedPromptId = info.promptId;
    // A cached node published nothing new, so the last drawing is still what the graph did.
    // Clearing it would report a node the canvas shows as done as one that has not run.
    if (info.cached) return;
    load();
  });
  const stopEnded = onRunEnded(() => load());

  const resize = typeof ResizeObserver === "function"
    ? new ResizeObserver(() => {
      drawGlyph();
      for (const cell of cells) cell.repaint();
    })
    : null;
  resize?.observe(root);
  const stopWatching = watchSurfaceRatio(glyph, () => drawGlyph());
  // The glyph is drawn into a canvas, where a custom property means nothing, so the palette is
  // subscribed to and the glyph drawn again.
  const stopTheme = onThemeChange(() => drawGlyph());

  // A browser without one measures at once, which is what every panel did before the deferral.
  const seen = typeof IntersectionObserver === "function"
    ? new IntersectionObserver((entries) => {
      onScreen = entries.some((entry) => entry.isIntersecting);
      if (onScreen && dirty) measure();
    })
    : null;
  if (seen) seen.observe(root); else onScreen = true;

  // Nothing in the band scrolls, so the panel takes every wheel gesture over it and the graph
  // zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);

  drawGlyph();
  for (const cell of cells) cell.show(null);
  // The stand-in is one decoded picture for the page, so the cells are filled again once it
  // lands rather than each panel asking for its own.
  loadPlaceholder().then(() => {
    if (disposed || held) return;
    for (const cell of cells) cell.show(null);
  });
  load();

  return {
    element: root,
    height,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth,
    refresh: load,
    showPrediction,
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel();
      release();
      stopFinished();
      stopEnded();
      resize?.disconnect();
      seen?.disconnect();
      stopWatching();
      stopTheme();
      metrics.dispose();
    },
  };
}
