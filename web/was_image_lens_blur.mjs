/**
 * Image Lens Blur's picture: the result, the input or the depth map, the focus point to click,
 * and the sharp zone the current settings give, worked out from the depth map the node read.
 */

import { app } from "../../scripts/app.js";
import { findWidget } from "./interface/audio_view.mjs";
import { CARD_LOOK, CHIP_TONES, cardChip, cardRule } from "./interface/card.mjs";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.mjs";
import { drawStandIn, loadPlaceholder, standInDetail } from "./interface/placeholder.mjs";
import { captureWheel, elementPoint } from "./interface/pointer.mjs";
import { LABELS, PREVIEW_STATE, fetchInputPreview, fetchOutputPreview } from "./interface/preview.mjs";
import { withGraphChange } from "./interface/region.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onNodeFinished } from "./interface/run_events.mjs";
import { fetchRunResult, whenReadable } from "./interface/run_result.mjs";
import { onThemeChange, readTheme } from "./interface/theme.mjs";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.mjs";
import { SANS, el } from "./interface/window_kit.mjs";

const EXT_NAME = "WASNodeSuite.ImageLensBlurUI";
const NODE_ID = "WASImageLensBlur";

const UI_WIDGET_NAME = "was_lens_blur_ui";
const UI_WIDGET_TYPE = "was_lens_blur";

// The slots the node files its pictures under.
const PICTURE_SLOT = "images";
const DEPTH_SLOT = "depth";

// The widgets the panel reads, and the ones a click writes.
const FOCUS_X = "focus_x";
const FOCUS_Y = "focus_y";
const FOCUS_FROM = "focus_from";
const FOCUS_DEPTH = "focus_depth";
const BLUR_AMOUNT = "blur_amount";
const FOCUS_RANGE = "focus_range";
const NEAR_IS = "near_is";
const APERTURE = "aperture";
const PICKED_POINT = "picked point";
const DARK_IS_NEAR = "dark";
const REPAINT_ON = [FOCUS_X, FOCUS_Y, FOCUS_FROM, FOCUS_DEPTH, BLUR_AMOUNT, FOCUS_RANGE, NEAR_IS, APERTURE];

// Widgets whose value the last run reports, compared to tell a stale picture.
const REPORTED = [FOCUS_X, FOCUS_Y, FOCUS_FROM, FOCUS_DEPTH, BLUR_AMOUNT, FOCUS_RANGE, NEAR_IS, APERTURE];

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 380;
const PANEL_WIDTH = 270;

// Room left round the picture inside its well, in CSS pixels.
const MARGIN = 6;

// How far a press may move and still count as a click, in CSS pixels.
const CLICK_SLOP = 4;

// Decimals a written focus point keeps.
const POINT_DECIMALS = 3;

// Share of the short side the depth under the focus point is read over, as the node reads it.
const FOCUS_WINDOW = 0.01;

// Diameters under this many pixels are sharp, as the node draws them.
const SHARP_DIAMETER = 1;

// Opacity of the sharp zone's tint, and the most the veil over blurred areas darkens.
const SHARP_ALPHA = 0.42;
const VEIL_ALPHA = 0.62;

// Half the size of the focus point's corner brackets, and the length of each arm, in CSS pixels.
const RETICLE = 11;
const RETICLE_ARM = 5;

// The three pictures the well can show.
const VIEW = { AFTER: "after", BEFORE: "before", DEPTH: "depth" };

/**
 * A number held inside a range.
 *
 * @param {number} value - The number.
 * @param {number} low - The least it may be.
 * @param {number} high - The most it may be.
 * @returns {number} The number, held to the range.
 */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * The red, green and blue a CSS colour resolves to.
 *
 * @param {string} colour - Any colour a canvas accepts.
 * @returns {number[]} Three channels, 0 to 255.
 */
function channels(colour) {
  const probe = document.createElement("canvas");
  probe.width = 1;
  probe.height = 1;
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [80, 200, 120];
  ctx.fillStyle = colour;
  ctx.fillRect(0, 0, 1, 1);
  const data = ctx.getImageData(0, 0, 1, 1).data;
  return [data[0], data[1], data[2]];
}

/**
 * The grey levels of a published depth map, read once.
 *
 * @param {HTMLImageElement} image - The depth thumbnail.
 * @returns {{values: Float32Array, width: number, height: number}|null} Levels on 0 to 1 as the
 *   map holds them, or null when the picture cannot be read.
 */
function readDepth(image) {
  const width = image?.naturalWidth || 0;
  const height = image?.naturalHeight || 0;
  if (!(width > 0) || !(height > 0)) return null;
  const probe = document.createElement("canvas");
  probe.width = width;
  probe.height = height;
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(image, 0, 0);
  const data = ctx.getImageData(0, 0, width, height).data;
  const values = new Float32Array(width * height);
  for (let index = 0; index < values.length; index += 1) values[index] = data[index * 4] / 255;
  return { values, width, height };
}

/**
 * The depth under a point as the node reads it: the lower median of a small window.
 *
 * @param {{values: Float32Array, width: number, height: number}} depth - The levels, 1 nearest.
 * @param {number} x - Share of the width.
 * @param {number} y - Share of the height.
 * @returns {number} The depth, 0 to 1.
 */
function depthAt(depth, x, y) {
  const { values, width, height } = depth;
  const column = clamp(Math.floor(x * width), 0, width - 1);
  const row = clamp(Math.floor(y * height), 0, height - 1);
  const reach = Math.max(2, Math.round(FOCUS_WINDOW * Math.min(width, height)));
  const held = [];
  for (let r = Math.max(0, row - reach); r <= Math.min(height - 1, row + reach); r += 1) {
    for (let c = Math.max(0, column - reach); c <= Math.min(width - 1, column + reach); c += 1) {
      held.push(values[r * width + c]);
    }
  }
  held.sort((a, b) => a - b);
  return held[(held.length - 1) >> 1];
}

/**
 * A small rounded button in the card's colours.
 *
 * @param {string} label - Its words.
 * @param {string} hint - Hover text.
 * @param {() => void} onPress - Called on a press.
 * @returns {{element: HTMLButtonElement, setOn: (on: boolean) => void}} The button, and a way to
 *   show it switched on.
 */
function pillButton(label, hint, onPress) {
  const press = document.createElement("button");
  press.type = "button";
  press.textContent = label;
  press.title = hint;
  let on = false;
  const style = (hovered) => {
    const tone = on ? CHIP_TONES.accent : null;
    press.style.cssText = "flex:none;padding:2px 8px;border-radius:999px;cursor:pointer;white-space:nowrap;"
      + `font:600 11px/18px ${SANS};border:1px solid ${hovered || on ? CARD_LOOK.rule : CARD_LOOK.edge};`
      + `background:${tone ? tone.background : CARD_LOOK.inset};`
      + `color:${tone ? tone.color : hovered ? CARD_LOOK.title : CARD_LOOK.muted}`;
  };
  style(false);
  press.onmouseenter = () => style(true);
  press.onmouseleave = () => style(false);
  press.addEventListener("pointerdown", (event) => event.stopPropagation());
  press.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      onPress();
    } catch (error) {
      console.error(`[${EXT_NAME}] ${label} failed:`, error);
    }
  });
  return {
    element: press,
    setOn(next) {
      on = !!next;
      style(false);
    },
  };
}

/**
 * Build the panel for one node.
 *
 * @param {object} node - The Image Lens Blur node.
 * @returns {object} The panel `appendInterfaceWidget` hosts.
 */
function createPanel(node) {
  let disposed = false;
  let serial = 0;
  let frameHandle = 0;
  let view = VIEW.AFTER;
  let showZone = true;
  let press = null;
  let hover = null;
  let report = null;
  let zone = null;
  let zoneKey = "";
  let tones = null;
  const pictures = {
    [VIEW.BEFORE]: { state: PREVIEW_STATE.WAITING, image: null },
    [VIEW.AFTER]: { state: PREVIEW_STATE.WAITING, image: null },
    [VIEW.DEPTH]: { state: PREVIEW_STATE.WAITING, image: null },
  };
  let depth = null;
  let source = { width: 0, height: 0 };

  const root = el("div", "position:relative;box-sizing:border-box;overflow:hidden;width:100%;height:100%;display:flex;"
    + "flex-direction:column;gap:6px;padding:7px 8px 8px;min-width:0;min-height:0;"
    + `font:11px/1.35 ${SANS};user-select:none`);

  const header = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0;overflow:hidden");
  const title = el("span", `flex:none;font:650 12px/1.4 ${SANS};color:${CARD_LOOK.title};white-space:nowrap`,
    "Lens Blur");
  const chips = el("div", "flex:1 1 auto;display:flex;flex-wrap:wrap;gap:4px;min-width:0");
  const controls = el("div", "flex:0 1 auto;display:flex;flex-wrap:wrap;gap:4px;align-items:center;min-width:0");
  const segment = el("div", "flex:none;display:flex;gap:2px;padding:2px;border-radius:999px;"
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well}`);
  const afterButton = pillButton("After", "The blurred picture", () => setView(VIEW.AFTER));
  const beforeButton = pillButton("Before", "The picture as it arrived", () => setView(VIEW.BEFORE));
  const depthButton = pillButton("Depth", "The depth map, bright near", () => setView(VIEW.DEPTH));
  segment.append(afterButton.element, beforeButton.element, depthButton.element);
  const zoneButton = pillButton("Sharp zone", "Tint what the settings keep sharp", () => {
    showZone = !showZone;
    zoneButton.setOn(showZone);
    redraw();
  });
  controls.append(segment, zoneButton.element);
  header.append(title, chips, controls);

  const stage = el("div", "position:relative;flex:1 1 auto;min-height:0;min-width:0;border-radius:6px;"
    + `overflow:hidden;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:crosshair`);
  const canvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  stage.append(canvas);

  const footer = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:6px;min-width:0");
  const glyph = el("canvas", `flex:none;width:${ICON_SIZE}px;height:${ICON_SIZE}px`);
  const status = el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${CARD_LOOK.muted}`);
  footer.append(glyph, status);

  root.append(header, cardRule(), stage, footer);

  /**
   * A widget's value.
   *
   * @param {string} name - The widget's name.
   * @returns {*} Its value, or undefined.
   */
  function value(name) {
    return findWidget(node, name)?.value;
  }

  /**
   * A fact the last run reported.
   *
   * @param {string} name - The fact's name.
   * @returns {string} Its text, or an empty string.
   */
  function fact(name) {
    return report?.facts.get(name) ?? "";
  }

  /**
   * The depth map's levels with 1 nearest, as the current near_is setting reads them.
   *
   * @param {number} level - A level as the map holds it.
   * @returns {number} The level read with 1 nearest.
   */
  function nearness(level) {
    return value(NEAR_IS) === DARK_IS_NEAR ? 1 - level : level;
  }

  /**
   * The settings the sharp zone is worked out from.
   *
   * @returns {{focus: number, largest: number, half: number}|null} The focus depth, the widest
   *   blur in source pixels and half the sharp zone's depth, or null with no depth map read.
   */
  function settings() {
    if (!depth) return null;
    const fromPoint = value(FOCUS_FROM) === PICKED_POINT;
    const focus = fromPoint
      ? nearness(depthAt(depth, Number(value(FOCUS_X)) || 0, Number(value(FOCUS_Y)) || 0))
      : clamp(Number(value(FOCUS_DEPTH)) || 0, 0, 1);
    const shortSide = Math.min(source.width || depth.width, source.height || depth.height);
    const largest = Math.max(0, Number(value(BLUR_AMOUNT)) || 0) * shortSide;
    const half = Math.max(0, Number(value(FOCUS_RANGE)) || 0) / 2;
    return { focus, largest, half };
  }

  /**
   * The blur diameter the node gives a depth, in source pixels.
   *
   * @param {number} level - The depth, 1 nearest.
   * @param {{focus: number, largest: number, half: number}} given - From `settings`.
   * @returns {number} The diameter.
   */
  function diameterOf(level, given) {
    return Math.max(0, Math.abs(level - given.focus) - given.half) * given.largest;
  }

  /**
   * The sharp zone over the depth map: a tint where it stays sharp, a veil by how far the rest blurs.
   *
   * @returns {{canvas: HTMLCanvasElement, share: number}|null} The overlay at the depth map's size
   *   and the share of the frame kept sharp, or null with no depth map read.
   */
  function zoneOverlay() {
    const given = settings();
    if (!given) return null;
    const key = [given.focus, given.largest, given.half, value(NEAR_IS)].join("|");
    if (zone && zoneKey === key) return zone;
    tones ??= { sharp: channels(readTheme().success) };
    const { values, width, height } = depth;
    const surface = document.createElement("canvas");
    surface.width = width;
    surface.height = height;
    const ctx = surface.getContext("2d");
    if (!ctx) return null;
    const pixels = ctx.createImageData(width, height);
    const data = pixels.data;
    const [red, green, blue] = tones.sharp;
    let sharp = 0;
    for (let index = 0; index < values.length; index += 1) {
      const diameter = diameterOf(nearness(values[index]), given);
      const at = index * 4;
      if (diameter < SHARP_DIAMETER) {
        sharp += 1;
        data[at] = red;
        data[at + 1] = green;
        data[at + 2] = blue;
        data[at + 3] = Math.round(255 * SHARP_ALPHA);
      } else {
        const veil = given.largest > 0 ? clamp(diameter / (given.largest * 0.5), 0, 1) : 0;
        data[at + 3] = Math.round(255 * VEIL_ALPHA * veil);
      }
    }
    ctx.putImageData(pixels, 0, 0);
    zone = { canvas: surface, share: sharp / Math.max(1, values.length) };
    zoneKey = key;
    return zone;
  }

  /**
   * Choose which picture the well shows.
   *
   * @param {string} next - A value of `VIEW`.
   * @returns {void}
   */
  function setView(next) {
    view = next;
    afterButton.setOn(view === VIEW.AFTER);
    beforeButton.setOn(view === VIEW.BEFORE);
    depthButton.setOn(view === VIEW.DEPTH);
    redraw();
  }

  /**
   * Repaint once, on the next frame.
   *
   * @returns {void}
   */
  function redraw() {
    if (disposed || frameHandle) return;
    frameHandle = requestAnimationFrame(() => {
      frameHandle = 0;
      try {
        paint();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to draw the panel:`, error);
      }
    });
  }

  /**
   * The picture the well shows now, falling back to the input before a result exists.
   *
   * @returns {{state: string, image: HTMLImageElement|null}} The entry.
   */
  function shown() {
    const wanted = pictures[view];
    if (wanted.image || view !== VIEW.AFTER) return wanted;
    return pictures[VIEW.BEFORE].image ? pictures[VIEW.BEFORE] : wanted;
  }

  /**
   * Where the picture sits inside the well.
   *
   * @returns {{x: number, y: number, width: number, height: number}} The rectangle, in layout
   *   pixels, fitted to the picture's own shape.
   */
  function pictureRect() {
    const image = shown().image ?? pictures[VIEW.BEFORE].image;
    const across = image?.naturalWidth || source.width || 4;
    const down = image?.naturalHeight || source.height || 3;
    const width = Math.max(1, canvas.clientWidth - MARGIN * 2);
    const height = Math.max(1, canvas.clientHeight - MARGIN * 2);
    const fit = Math.min(width / across, height / down);
    return {
      x: MARGIN + (width - across * fit) / 2,
      y: MARGIN + (height - down * fit) / 2,
      width: across * fit,
      height: down * fit,
    };
  }

  /**
   * Draw the focus point as a camera's focus brackets.
   *
   * @param {CanvasRenderingContext2D} ctx - The well's context.
   * @param {{x: number, y: number, width: number, height: number}} rect - Where the picture is.
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintReticle(ctx, rect, theme) {
    const x = rect.x + clamp(Number(value(FOCUS_X)) || 0, 0, 1) * rect.width;
    const y = rect.y + clamp(Number(value(FOCUS_Y)) || 0, 0, 1) * rect.height;
    const live = value(FOCUS_FROM) === PICKED_POINT;
    ctx.save();
    ctx.lineCap = "round";
    ctx.setLineDash(live ? [] : [2, 3]);
    for (const [width, colour] of [[4, theme.bgDark], [2, live ? theme.accent : theme.fgMuted]]) {
      ctx.lineWidth = width;
      ctx.strokeStyle = colour;
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const cornerX = x + sx * RETICLE;
        const cornerY = y + sy * RETICLE;
        ctx.beginPath();
        ctx.moveTo(cornerX - sx * RETICLE_ARM, cornerY);
        ctx.lineTo(cornerX, cornerY);
        ctx.lineTo(cornerX, cornerY - sy * RETICLE_ARM);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  /**
   * Draw the well.
   *
   * @param {object} theme - The palette.
   * @returns {{icon: string, detail: string}} What the well claims about its fidelity.
   */
  function paintStage(theme) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!(width > 0) || !(height > 0)) return { icon: ICON.WARNING, detail: standInDetail(PREVIEW_STATE.WAITING) };
    const ratio = surfaceRatio(canvas);
    const backingWidth = Math.max(1, Math.round(width * ratio));
    const backingHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== backingWidth) canvas.width = backingWidth;
    if (canvas.height !== backingHeight) canvas.height = backingHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return { icon: ICON.WARNING, detail: "" };
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const rect = pictureRect();
    const entry = shown();
    if (!entry.image) {
      drawStandIn(ctx, { x: rect.x, y: rect.y, w: rect.width, h: rect.height });
      const words = entry.state === PREVIEW_STATE.WAITING ? "Run once to see the picture" : LABELS[entry.state] ?? "";
      if (words) {
        ctx.fillStyle = theme.fgMuted;
        ctx.font = `600 12px ${SANS}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(words, rect.x + rect.width / 2, rect.y + rect.height - 6);
      }
      return { icon: ICON.WARNING, detail: standInDetail(entry.state) };
    }
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(entry.image, rect.x, rect.y, rect.width, rect.height);
    let claim = { icon: ICON.EXACT, detail: `${view} picture` };
    if (showZone) {
      const overlay = zoneOverlay();
      if (overlay) {
        ctx.drawImage(overlay.canvas, rect.x, rect.y, rect.width, rect.height);
        claim = { icon: ICON.APPROXIMATE, detail: "sharp zone worked out from the depth map's thumbnail" };
      }
    }
    paintReticle(ctx, rect, theme);
    return claim;
  }

  /**
   * Whether a widget still holds the value the last run reported for it.
   *
   * @param {string} name - The widget's name.
   * @returns {boolean} True when it matches, or when nothing was reported.
   */
  function unchanged(name) {
    const reported = fact(name);
    if (reported === "") return true;
    const current = value(name);
    if (typeof current === "number") return Math.abs(current - Number(reported)) < 5e-4;
    return String(current) === reported;
  }

  /**
   * Fill the header's chips.
   *
   * @returns {void}
   */
  function paintChips() {
    chips.replaceChildren();
    const current = !report || REPORTED.every(unchanged);
    const ran = Number(fact("focus"));
    const given = settings();
    const focus = report && current && Number.isFinite(ran) && fact("focus") !== "" ? ran : given?.focus;
    if (Number.isFinite(focus)) chips.append(cardChip(`focus ${focus.toFixed(2)}`, "accent"));
    const widest = Number(report?.counts.get("widest"));
    if (current && Number.isFinite(widest)) chips.append(cardChip(`blur up to ${Math.round(widest)} px`));
    if (!current) chips.append(cardChip("changed since the run", "warning"));
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    if (disposed) return;
    const theme = readTheme();
    const claim = paintStage(theme);
    paintChips();
    drawFidelityGlyph(glyph, claim);
    const parts = [];
    const given = settings();
    if (hover && given && depth) {
      const level = nearness(depthAt(depth, hover.u, hover.v));
      const diameter = diameterOf(level, given);
      parts.push(`depth ${level.toFixed(2)}`, diameter < SHARP_DIAMETER ? "sharp" : `blur ${Math.round(diameter)} px`);
    } else {
      if (showZone && zone && zoneKey) parts.push(`${Math.round(zone.share * 100)}% sharp`);
      parts.push(value(FOCUS_FROM) === PICKED_POINT ? "click the picture to focus" : "focus set by focus_depth");
    }
    status.textContent = parts.join("  ·  ");
  }

  /**
   * Read the last run's report and its three pictures.
   *
   * @returns {Promise<void>} Settles once the answers are in.
   */
  async function load() {
    const ticket = ++serial;
    try {
      const [run, before, after, map] = await Promise.all([
        fetchRunResult(node),
        fetchInputPreview(node, PICTURE_SLOT),
        fetchOutputPreview(node, PICTURE_SLOT),
        fetchInputPreview(node, DEPTH_SLOT),
      ]);
      if (ticket !== serial || disposed) return;
      report = run?.state === PREVIEW_STATE.READY && run.result ? {
        facts: new Map((run.result.facts ?? []).map((row) => [row.name, String(row.value)])),
        counts: new Map((run.result.counts ?? []).map((row) => [row.name, row.value])),
      } : null;
      for (const [key, answer] of [[VIEW.BEFORE, before], [VIEW.AFTER, after], [VIEW.DEPTH, map]]) {
        const ready = answer?.state === PREVIEW_STATE.READY;
        pictures[key] = { state: answer?.state ?? PREVIEW_STATE.FAILED, image: ready ? answer.image : null };
      }
      source = before?.state === PREVIEW_STATE.READY
        ? { width: before.sourceWidth || 0, height: before.sourceHeight || 0 }
        : { width: 0, height: 0 };
      depth = pictures[VIEW.DEPTH].image ? readDepth(pictures[VIEW.DEPTH].image) : null;
      zone = null;
      zoneKey = "";
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the last run:`, error);
    }
    redraw();
  }

  /**
   * Where a pointer is on the picture.
   *
   * @param {PointerEvent} event - The event.
   * @returns {{u: number, v: number}|null} Shares of the picture, or null outside it.
   */
  function picturePoint(event) {
    const rect = pictureRect();
    const point = elementPoint(canvas, event);
    if (!point) return null;
    const u = (point.x - rect.x) / rect.width;
    const v = (point.y - rect.y) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { u, v };
  }

  /**
   * Set the focus point where the picture was clicked, as one undoable edit.
   *
   * @param {{u: number, v: number}} point - Where the click landed.
   * @returns {void}
   */
  function focusAt(point) {
    const round = (share) => Number(clamp(share, 0, 1).toFixed(POINT_DECIMALS));
    const values = { [FOCUS_X]: round(point.u), [FOCUS_Y]: round(point.v), [FOCUS_FROM]: PICKED_POINT };
    const writes = [];
    for (const [name, next] of Object.entries(values)) {
      const widget = findWidget(node, name);
      if (widget && widget.value !== next) writes.push([widget, next]);
    }
    if (!writes.length) return;
    withGraphChange(() => {
      for (const [widget, next] of writes) {
        widget.value = next;
        widget.callback?.(next);
      }
    });
    node.setDirtyCanvas?.(true, true);
    redraw();
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    press = { x: event.clientX, y: event.clientY };
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
  });
  const endPress = (event, cancelled) => {
    const pressed = press;
    press = null;
    try {
      canvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
    }
    if (!pressed || cancelled) return;
    if (Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) > CLICK_SLOP) return;
    const point = picturePoint(event);
    if (point && (pictures[VIEW.BEFORE].image || pictures[VIEW.AFTER].image)) focusAt(point);
  };
  canvas.addEventListener("pointerup", (event) => endPress(event, false));
  canvas.addEventListener("pointercancel", (event) => endPress(event, true));
  canvas.addEventListener("lostpointercapture", () => { press = null; });
  canvas.addEventListener("pointermove", (event) => {
    if (press && event.buttons === 0) press = null;
    const point = picturePoint(event);
    const moved = (point === null) !== (hover === null)
      || (point && hover && (point.u !== hover.u || point.v !== hover.v));
    hover = point;
    if (moved && depth) redraw();
  });
  canvas.addEventListener("pointerleave", () => {
    if (!hover) return;
    hover = null;
    redraw();
  });
  root.addEventListener("contextmenu", (event) => event.preventDefault());

  const releaseWheel = captureWheel(root);
  const resizeWatch = new ResizeObserver(() => redraw());
  resizeWatch.observe(root);
  const stopRatio = watchSurfaceRatio(canvas, () => redraw());
  const stopTheme = onThemeChange(() => {
    tones = null;
    zone = null;
    zoneKey = "";
    redraw();
  });
  const stopFinished = onNodeFinished(node, () => load());
  const stopReadable = whenReadable(node, () => load());
  for (const name of REPAINT_ON) chainWidgetCallback(node, name, () => redraw(), EXT_NAME);
  loadPlaceholder().then(() => redraw());
  setView(VIEW.AFTER);
  zoneButton.setOn(showZone);

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: () => load(),
    dispose() {
      disposed = true;
      serial += 1;
      if (frameHandle) cancelAnimationFrame(frameHandle);
      frameHandle = 0;
      resizeWatch.disconnect();
      stopRatio?.();
      stopTheme?.();
      stopFinished?.();
      stopReadable?.();
      releaseWheel?.();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_lens_blur_wrapped) return;
    proto.__was_lens_blur_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
