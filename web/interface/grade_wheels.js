/**
 * The colour grading wheels drawn on Image Color Grading.
 *
 * Four wheels, for the shadows, the midtones, the highlights and the whole picture, each set
 * by dragging a puck for hue and saturation, with a slider under it for luminance.
 */

import { app } from "../../../scripts/app.js";
import { CARD_LOOK, CHIP_TONES, cardRule } from "./card.js";
import { captureWheel, elementPoint } from "./pointer.js";
import { withGraphChange } from "./region.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme, themeVar } from "./theme.js";
import { chainWidgetCallback } from "./widget.js";

const LOG_NAME = "WASNodeSuite.GradeWheels";

// Height of the appended widget in node units, and the narrowest the wheels stay usable in.
const DEFAULT_HEIGHT = 380;
const DEFAULT_MIN_WIDTH = 300;

// The wheels, in the order they are drawn, as the prefix of their widgets and their title.
const WHEELS = [
  { key: "shadows", title: "Shadows" },
  { key: "midtones", title: "Midtones" },
  { key: "highlights", title: "Highlights" },
  { key: "global", title: "Global" },
];

// The column counts the wheels may be laid out in: one row of four, or two rows of two.
const LAYOUTS = [4, 2];

// Ranges of the three widgets a wheel writes.
const HUE_MAX = 360;
const SATURATION_MAX = 100;
const LUMINANCE_MIN = -100;
const LUMINANCE_MAX = 100;

// Spacing between the wheel cards, and inside one, in layout pixels.
const GAP = 6;
const CELL_PAD = 6;

// Height a card spends on its title, rule, slider and readout, in layout pixels.
const CHROME = 58;

// Room kept around a disc inside its box, in layout pixels.
const DISC_MARGIN = 6;

// The smallest disc radius drawn, in layout pixels.
const MIN_RADIUS = 10;

// Saturation and lightness of the rim, matching the pack's hue dial.
const RIM_SATURATION = 85;
const RIM_LIGHTNESS = 52;

// Colour stops the conic rim is built from.
const RIM_STOPS = 24;

// Wedges drawn where the canvas has no conic gradient.
const WEDGES = 120;

// The puck, in layout pixels.
const PUCK_RADIUS = 6;
const PUCK_LINE = 2;

// Height of the luminance track and its thumb, in layout pixels.
const TRACK_HEIGHT = 6;
const THUMB_SIZE = 11;

// Hue names the chip reads, as the upper edge of each span in degrees.
const HUE_NAMES = [
  [15, "red"],
  [40, "orange"],
  [65, "yellow"],
  [95, "lime"],
  [150, "green"],
  [175, "teal"],
  [195, "cyan"],
  [220, "azure"],
  [255, "blue"],
  [285, "violet"],
  [320, "magenta"],
  [345, "rose"],
  [360, "red"],
];

/**
 * A number held inside a range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} low - Lower bound.
 * @param {number} high - Upper bound.
 * @returns {number} The value, held inside the bounds.
 */
function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * One widget of a node, by name.
 *
 * @param {object} node - The node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function findWidget(node, name) {
  return (node?.widgets || []).find(widget => widget?.name === name) || null;
}

/**
 * The name of a hue, for the chip beside a wheel's title.
 *
 * @param {number} hue - Degrees, 0 to 360.
 * @returns {string} A colour name.
 */
function hueName(hue) {
  const turned = ((hue % HUE_MAX) + HUE_MAX) % HUE_MAX;
  const found = HUE_NAMES.find(([edge]) => turned < edge);
  return found ? found[1] : "red";
}

/**
 * The colour a wheel shows at a hue and saturation.
 *
 * @param {number} hue - Degrees.
 * @param {number} saturation - 0 to 100.
 * @returns {string} A CSS colour.
 */
function wheelColour(hue, saturation) {
  const share = clamp(saturation, 0, SATURATION_MAX) / SATURATION_MAX;
  const lightness = 50 + (RIM_LIGHTNESS - 50) * share;
  return `hsl(${Math.round(hue)} ${Math.round(RIM_SATURATION * share)}% ${lightness}%)`;
}

/**
 * Put colour grading wheels on a node, wired to its hue, saturation and luminance widgets.
 *
 * @param {object} node - The node being created.
 * @param {object} [options] - Everything below.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {number} [options.minWidth] - The narrowest the wheels are drawn in, in node units.
 * @returns {{element: HTMLElement, height: number, minWidth: number, refresh: () => void,
 *   dispose: () => void}} The element to hand to `appendInterfaceWidget`, the height it was
 *   built for, the narrowest it draws in, a repaint, and teardown.
 */
export function createGradeWheels(node, options = {}) {
  const height = Number(options.height) > 0 ? Number(options.height) : DEFAULT_HEIGHT;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : DEFAULT_MIN_WIDTH;

  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${height - GAP * 2}px`,
    "min-width:0",
    "display:flex",
    "flex-wrap:wrap",
    "align-content:stretch",
    `gap:${GAP}px`,
    `padding:${GAP}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
    `color:${CARD_LOOK.text}`,
    "font:11px/1.3 system-ui,sans-serif",
  ].join(";");

  const state = { paintHandle: 0, disposed: false, columns: 0, dragging: null, gesture: false };
  const views = WHEELS.map(wheel => buildWheel(wheel));

  /**
   * One wheel's card: title, disc, luminance slider and readout.
   *
   * @param {{key: string, title: string}} wheel - The wheel to build.
   * @returns {object} The card's elements and its key.
   */
  function buildWheel(wheel) {
    const card = document.createElement("div");
    card.style.cssText = [
      "box-sizing:border-box",
      "display:flex",
      "flex-direction:column",
      "min-width:0",
      "min-height:0",
      `padding:${CELL_PAD - 2}px ${CELL_PAD}px ${CELL_PAD}px`,
      `background:${CARD_LOOK.surface}`,
      `border:1px solid ${CARD_LOOK.edge}`,
      "border-radius:8px",
      `box-shadow:${CARD_LOOK.shine}`,
      "gap:3px",
    ].join(";");

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:6px;min-width:0;cursor:pointer";
    header.title = "Double-click to reset this wheel.";
    const title = document.createElement("span");
    title.style.cssText = [
      "flex:1 0 auto",
      "min-width:0",
      "overflow:hidden",
      "text-overflow:ellipsis",
      "white-space:nowrap",
      `color:${CARD_LOOK.title}`,
      "font-weight:650",
      "font-variant-caps:all-small-caps",
      "letter-spacing:0.06em",
      "font-size:12px",
    ].join(";");
    title.textContent = wheel.title;
    const chip = document.createElement("span");
    chip.style.cssText = "flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
      + "font:600 9.5px/1.45 system-ui,sans-serif;letter-spacing:0.04em;"
      + "text-transform:uppercase;padding:1px 6px;border-radius:999px;white-space:nowrap";
    header.appendChild(title);
    header.appendChild(chip);

    const discBox = document.createElement("div");
    discBox.style.cssText = "position:relative;flex:1 1 auto;min-height:0;min-width:0;cursor:crosshair";
    discBox.title = "Hue and saturation";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;display:block";
    discBox.appendChild(canvas);

    const slider = document.createElement("div");
    slider.style.cssText = `position:relative;flex:0 0 ${THUMB_SIZE + 3}px;min-width:0;cursor:ew-resize`;
    slider.title = "Luminance";
    const track = document.createElement("div");
    track.style.cssText = [
      "position:absolute",
      "left:0",
      "right:0",
      `top:${(THUMB_SIZE + 3 - TRACK_HEIGHT) / 2}px`,
      `height:${TRACK_HEIGHT}px`,
      "border-radius:999px",
      `background:linear-gradient(90deg, ${themeVar("bgDark")}, ${themeVar("fgMuted")} 50%, ${themeVar("fg")})`,
      `border:1px solid ${CARD_LOOK.edge}`,
      `box-shadow:${CARD_LOOK.well}`,
      "box-sizing:border-box",
    ].join(";");
    const tick = document.createElement("div");
    tick.style.cssText = `position:absolute;left:50%;top:1px;bottom:1px;width:1px;background:${CARD_LOOK.rule}`;
    const thumb = document.createElement("div");
    thumb.style.cssText = [
      "position:absolute",
      "top:1px",
      `width:${THUMB_SIZE}px`,
      `height:${THUMB_SIZE}px`,
      "border-radius:50%",
      "box-sizing:border-box",
      `background:${CARD_LOOK.title}`,
      `border:1px solid ${CARD_LOOK.edge}`,
      `box-shadow:${CARD_LOOK.shine}`,
      `margin-left:${-THUMB_SIZE / 2}px`,
      "pointer-events:none",
    ].join(";");
    slider.appendChild(track);
    slider.appendChild(tick);
    slider.appendChild(thumb);

    const readout = document.createElement("div");
    readout.style.cssText = "display:flex;justify-content:center;gap:8px;min-width:0;overflow:hidden;"
      + "white-space:nowrap;font-variant-numeric:tabular-nums";
    const figures = {};
    for (const [letter, name] of [["H", "hue"], ["S", "saturation"], ["L", "luminance"]]) {
      const group = document.createElement("span");
      group.style.cssText = "min-width:0";
      const label = document.createElement("span");
      label.style.cssText = `color:${CARD_LOOK.muted};margin-right:3px`;
      label.textContent = letter;
      const value = document.createElement("span");
      value.style.cssText = `color:${CARD_LOOK.title};font-weight:650`;
      group.appendChild(label);
      group.appendChild(value);
      readout.appendChild(group);
      figures[name] = value;
    }

    card.appendChild(header);
    card.appendChild(cardRule());
    card.appendChild(discBox);
    card.appendChild(slider);
    card.appendChild(readout);
    root.appendChild(card);

    return {
      key: wheel.key, card, header, chip, discBox, canvas, slider, thumb, figures,
      geometry: { cx: 0, cy: 0, radius: MIN_RADIUS },
    };
  }

  /**
   * The three values a wheel's widgets hold.
   *
   * @param {string} key - The wheel's widget prefix.
   * @returns {{hue: number, saturation: number, luminance: number}} The values, clamped.
   */
  function readWheel(key) {
    const number = (name, low, high) => {
      const value = Number(findWidget(node, `${key}_${name}`)?.value);
      return Number.isFinite(value) ? clamp(value, low, high) : 0;
    };
    return {
      hue: number("hue", 0, HUE_MAX),
      saturation: number("saturation", 0, SATURATION_MAX),
      luminance: number("luminance", LUMINANCE_MIN, LUMINANCE_MAX),
    };
  }

  /**
   * Write some of a wheel's values onto its widgets, as one undo step.
   *
   * @param {string} key - The wheel's widget prefix.
   * @param {{hue?: number, saturation?: number, luminance?: number}} values - What to write.
   * @returns {boolean} True where any widget moved.
   */
  function writeWheel(key, values) {
    const writes = [];
    for (const [name, raw] of Object.entries(values)) {
      const widget = findWidget(node, `${key}_${name}`);
      if (!widget || !Number.isFinite(raw)) continue;
      let next = Math.round(raw);
      if (name === "hue") next = ((next % HUE_MAX) + HUE_MAX) % HUE_MAX;
      if (name === "saturation") next = clamp(next, 0, SATURATION_MAX);
      if (name === "luminance") next = clamp(next, LUMINANCE_MIN, LUMINANCE_MAX);
      const low = Number(widget.options?.min);
      const high = Number(widget.options?.max);
      if (Number.isFinite(low)) next = Math.max(low, next);
      if (Number.isFinite(high)) next = Math.min(high, next);
      if (Number(widget.value) !== next) writes.push([widget, next]);
    }
    if (!writes.length) return false;
    const apply = () => {
      for (const [widget, next] of writes) {
        widget.value = next;
        widget.callback?.(next);
      }
    };
    // A drag is already inside one change, opened on the press and closed on the release.
    if (state.gesture) apply();
    else withGraphChange(apply);
    node.setDirtyCanvas?.(true, true);
    schedulePaint();
    return true;
  }

  /**
   * Lay the cards out in whichever column count gives the largest disc.
   *
   * @returns {void}
   */
  function arrange() {
    const width = root.clientWidth || minWidth;
    const tall = root.clientHeight || height;
    let best = LAYOUTS[0];
    let bestDisc = -Infinity;
    for (const columns of LAYOUTS) {
      const rows = Math.ceil(WHEELS.length / columns);
      const cellWidth = (width - GAP * 2 - GAP * (columns - 1)) / columns;
      const cellHeight = (tall - GAP * 2 - GAP * (rows - 1)) / rows;
      const disc = Math.min(cellWidth - CELL_PAD * 2, cellHeight - CHROME);
      if (disc > bestDisc) {
        bestDisc = disc;
        best = columns;
      }
    }
    if (best === state.columns) return;
    state.columns = best;
    // Each card takes an equal share of its row, so the wrap lands on the chosen column count.
    const share = `calc(${100 / best}% - ${(GAP * (best - 1)) / best}px)`;
    for (const view of views) view.card.style.flex = `1 1 ${share}`;
  }

  /**
   * Draw the hue rim and the saturation fade of one disc.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {{cx: number, cy: number, radius: number}} place - Centre and radius.
   * @returns {void}
   */
  function drawDisc(ctx, place) {
    const { cx, cy, radius } = place;
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.clip();
    if (typeof ctx.createConicGradient === "function") {
      // Twelve o'clock is a hue of zero and the hues run clockwise, as on the hue dial.
      const rim = ctx.createConicGradient(-Math.PI / 2, cx, cy);
      for (let index = 0; index <= RIM_STOPS; index += 1) {
        const hue = (index / RIM_STOPS) * HUE_MAX;
        rim.addColorStop(index / RIM_STOPS, `hsl(${hue} ${RIM_SATURATION}% ${RIM_LIGHTNESS}%)`);
      }
      ctx.fillStyle = rim;
      ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
    } else {
      const sweep = (Math.PI * 2) / WEDGES;
      for (let index = 0; index < WEDGES; index += 1) {
        const from = (index / WEDGES) * Math.PI * 2 - Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, from, from + sweep + 0.01);
        ctx.closePath();
        ctx.fillStyle = `hsl(${(index / WEDGES) * HUE_MAX} ${RIM_SATURATION}% ${RIM_LIGHTNESS}%)`;
        ctx.fill();
      }
    }
    // The centre is neutral grey, fading out to the full rim colour.
    const fade = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    fade.addColorStop(0, "hsl(0 0% 50%)");
    fade.addColorStop(1, "hsl(0 0% 50% / 0)");
    ctx.fillStyle = fade;
    ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
    ctx.restore();
  }

  /**
   * Draw one wheel and bring its slider and readout up to date.
   *
   * @param {object} view - The wheel's card, from `buildWheel`.
   * @param {object} theme - Palette tokens from `readTheme`.
   * @returns {void}
   */
  function paintWheel(view, theme) {
    const { hue, saturation, luminance } = readWheel(view.key);

    const width = view.discBox.clientWidth;
    const tall = view.discBox.clientHeight;
    if (width > 0 && tall > 0) {
      const ratio = surfaceRatio(root);
      view.canvas.width = Math.max(1, Math.round(width * ratio));
      view.canvas.height = Math.max(1, Math.round(tall * ratio));
      const ctx = view.canvas.getContext("2d");
      if (ctx) {
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, tall);
        const radius = Math.max(MIN_RADIUS, Math.min(width, tall) / 2 - DISC_MARGIN);
        const place = { cx: width / 2, cy: tall / 2, radius };
        view.geometry = place;
        drawDisc(ctx, place);

        // A guide ring at half saturation, a short cross at the centre and the outline.
        ctx.strokeStyle = theme.fg;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.22;
        ctx.beginPath();
        ctx.arc(place.cx, place.cy, radius / 2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(place.cx - 4, place.cy);
        ctx.lineTo(place.cx + 4, place.cy);
        ctx.moveTo(place.cx, place.cy - 4);
        ctx.lineTo(place.cx, place.cy + 4);
        ctx.stroke();
        ctx.globalAlpha = 0.6;
        ctx.strokeStyle = theme.border;
        ctx.beginPath();
        ctx.arc(place.cx, place.cy, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;

        const angle = (hue / HUE_MAX) * Math.PI * 2 - Math.PI / 2;
        const reach = (radius * saturation) / SATURATION_MAX;
        const px = place.cx + Math.cos(angle) * reach;
        const py = place.cy + Math.sin(angle) * reach;
        if (saturation > 0) {
          ctx.strokeStyle = theme.fg;
          ctx.globalAlpha = 0.45;
          ctx.beginPath();
          ctx.moveTo(place.cx, place.cy);
          ctx.lineTo(px, py);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
        ctx.beginPath();
        ctx.arc(px, py, PUCK_RADIUS + 1, 0, Math.PI * 2);
        ctx.fillStyle = theme.bgDark;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, PUCK_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = wheelColour(hue, saturation);
        ctx.fill();
        ctx.lineWidth = PUCK_LINE;
        ctx.strokeStyle = theme.fg;
        ctx.stroke();
      }
    }

    const share = (luminance - LUMINANCE_MIN) / (LUMINANCE_MAX - LUMINANCE_MIN);
    view.thumb.style.left = `${(share * 100).toFixed(2)}%`;

    view.figures.hue.textContent = String(Math.round(hue));
    view.figures.saturation.textContent = String(Math.round(saturation));
    view.figures.luminance.textContent = String(Math.round(luminance));

    if (saturation > 0) {
      view.chip.textContent = hueName(hue);
      view.chip.style.background = `hsl(${Math.round(hue)} 75% 50% / 0.32)`;
      view.chip.style.color = CARD_LOOK.title;
    } else {
      view.chip.textContent = "neutral";
      view.chip.style.background = CHIP_TONES.plain.background;
      view.chip.style.color = CHIP_TONES.plain.color;
    }
  }

  /**
   * Lay out and draw every wheel.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    arrange();
    const theme = readTheme();
    for (const view of views) paintWheel(view, theme);
  }

  /**
   * Repaint once, on the next frame.
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
        console.error(`[${LOG_NAME}] Failed to draw the colour grading wheels:`, error);
      }
    });
  }

  /**
   * The wheel whose card holds an element.
   *
   * @param {EventTarget} target - Where an event landed.
   * @returns {object|null} The wheel's card, or null outside every card.
   */
  function viewOf(target) {
    return views.find(view => view.card.contains(target)) || null;
  }

  /**
   * Set a wheel's hue and saturation from a pointer over its disc.
   *
   * @param {object} view - The wheel's card.
   * @param {PointerEvent} event - Pointer event over the disc.
   * @param {boolean} keepHue - Whether only the saturation moves.
   * @returns {void}
   */
  function writeFromDisc(view, event, keepHue) {
    const point = elementPoint(view.discBox, event);
    const { cx, cy, radius } = view.geometry;
    const dx = point.x - cx;
    const dy = point.y - cy;
    const saturation = clamp((Math.hypot(dx, dy) / Math.max(radius, 1)) * SATURATION_MAX, 0, SATURATION_MAX);
    if (keepHue) {
      writeWheel(view.key, { saturation });
      return;
    }
    // Clockwise from twelve o'clock, in degrees.
    const hue = ((Math.atan2(dx, -dy) * 180) / Math.PI + HUE_MAX) % HUE_MAX;
    writeWheel(view.key, { hue, saturation });
  }

  /**
   * Set a wheel's luminance from a pointer over its slider.
   *
   * @param {object} view - The wheel's card.
   * @param {PointerEvent} event - Pointer event over the slider.
   * @returns {void}
   */
  function writeFromSlider(view, event) {
    const point = elementPoint(view.slider, event);
    const width = Math.max(1, view.slider.clientWidth);
    const share = clamp(point.x / width, 0, 1);
    writeWheel(view.key, { luminance: LUMINANCE_MIN + share * (LUMINANCE_MAX - LUMINANCE_MIN) });
  }

  /**
   * Start a drag on a disc or a slider.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerDown(event) {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    const view = viewOf(event.target);
    if (!view) return;
    let part = null;
    if (view.discBox.contains(event.target)) part = "disc";
    else if (view.slider.contains(event.target)) part = "slider";
    if (!part) return;
    state.dragging = { view, part, pointerId: event.pointerId };
    openGesture();
    try {
      root.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] Pointer capture refused:`, error);
    }
    if (part === "disc") writeFromDisc(view, event, event.shiftKey);
    else writeFromSlider(view, event);
    event.stopPropagation();
    event.preventDefault();
  }

  /**
   * Follow the pointer while a disc or slider is being dragged.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerMove(event) {
    const drag = state.dragging;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.part === "disc") writeFromDisc(drag.view, event, event.shiftKey);
    else writeFromSlider(drag.view, event);
    event.stopPropagation();
  }

  /**
   * End a drag.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerUp(event) {
    const drag = state.dragging;
    if (!drag || drag.pointerId !== event.pointerId) return;
    state.dragging = null;
    closeGesture();
    try {
      root.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] Pointer release refused:`, error);
    }
    schedulePaint();
    event.stopPropagation();
  }

  /**
   * Open the one graph change a whole drag is recorded as.
   *
   * @returns {void}
   */
  function openGesture() {
    if (state.gesture) return;
    const canvas = app.canvas;
    if (typeof canvas?.emitBeforeChange !== "function" || typeof canvas?.emitAfterChange !== "function") return;
    canvas.emitBeforeChange();
    state.gesture = true;
  }

  /**
   * Close the graph change a drag opened.
   *
   * @returns {void}
   */
  function closeGesture() {
    if (!state.gesture) return;
    state.gesture = false;
    app.canvas?.emitAfterChange?.();
  }

  /**
   * Reset a wheel, or only its luminance when the slider was double-clicked.
   *
   * @param {MouseEvent} event - Double-click event.
   * @returns {void}
   */
  function onDoubleClick(event) {
    const view = viewOf(event.target);
    if (!view) return;
    if (view.slider.contains(event.target)) {
      writeWheel(view.key, { luminance: 0 });
    } else if (view.discBox.contains(event.target) || view.header.contains(event.target)) {
      writeWheel(view.key, { hue: 0, saturation: 0, luminance: 0 });
    } else {
      return;
    }
    event.stopPropagation();
    event.preventDefault();
  }

  /**
   * Nudge a wheel's saturation, or its luminance over the slider, by one per notch.
   *
   * @param {WheelEvent} event - Wheel event.
   * @returns {boolean} True where the notch moved a value.
   */
  function onWheel(event) {
    if (!event.deltaY) return false;
    const view = viewOf(event.target);
    if (!view) return false;
    const notch = event.deltaY > 0 ? -1 : 1;
    const current = readWheel(view.key);
    if (view.slider.contains(event.target)) {
      return writeWheel(view.key, { luminance: current.luminance + notch });
    }
    return writeWheel(view.key, { saturation: current.saturation + notch });
  }

  root.addEventListener("pointerdown", onPointerDown);
  root.addEventListener("pointermove", onPointerMove);
  root.addEventListener("pointerup", onPointerUp);
  root.addEventListener("pointercancel", onPointerUp);
  root.addEventListener("dblclick", onDoubleClick);
  const releaseWheel = captureWheel(root, onWheel);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
    for (const view of views) observer.observe(view.discBox);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here.
  const unwatch = watchSurfaceRatio(root, schedulePaint);

  // The discs are drawn into canvases, which take literal colours, so a palette change repaints.
  const unwatchTheme = onThemeChange(schedulePaint);

  // Typing in any of the twelve widgets moves its wheel with it.
  for (const wheel of WHEELS) {
    for (const name of ["hue", "saturation", "luminance"]) {
      chainWidgetCallback(node, `${wheel.key}_${name}`, schedulePaint, LOG_NAME);
    }
  }

  schedulePaint();

  return {
    element: root,
    height,
    minWidth,
    refresh: schedulePaint,
    dispose() {
      closeGesture();
      state.disposed = true;
      if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
      state.paintHandle = 0;
      root.removeEventListener("pointerdown", onPointerDown);
      root.removeEventListener("pointermove", onPointerMove);
      root.removeEventListener("pointerup", onPointerUp);
      root.removeEventListener("pointercancel", onPointerUp);
      root.removeEventListener("dblclick", onDoubleClick);
      releaseWheel();
      observer?.disconnect();
      unwatch?.();
      unwatchTheme?.();
    },
  };
}
