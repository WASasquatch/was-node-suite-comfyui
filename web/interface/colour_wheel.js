/**
 * The wheel a colour is picked on, drawn on the node.
 *
 * A hue ring around a saturation and brightness square. Releasing the pointer writes the
 * node's three level widgets.
 */

import { drawIcon, hoverTitles, ICON, ICON_SIZE, iconTitle } from "./icons.js";
import { captureWheel, elementPoint } from "./pointer.js";
import { withGraphChange } from "./region.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme } from "./theme.js";

const LOG_NAME = "WASNodeSuite.ColourWheel";

// Height of the appended widget in node units, and the narrowest the wheel stays usable in.
const DEFAULT_HEIGHT = 206;
const DEFAULT_MIN_WIDTH = 210;

// The widgets a node holds its colour in, in the order the levels are read.
const DEFAULT_CHANNELS = ["red", "green", "blue"];

// Wedges the ring is drawn from. 180 puts one every two degrees, which reads as continuous.
const WEDGES = 180;

// Room left around the wheel and the thickness of the ring, as a share of its radius.
const MARGIN = 8;
const RING_SHARE = 0.2;

// Gap between the ring and the square inside it, in layout pixels.
const GAP = 4;

// The handles, in layout pixels.
const HANDLE_RADIUS = 5;
const HANDLE_LINE = 2;

// The readout strip under the wheel: its height, the swatch beside it and the text sizes.
// Tall enough for both lines of text at the panel's own default height.
const STRIP_HEIGHT = 42;
const SWATCH_WIDTH = 46;
const HEX_SIZE = 13;
const LEVEL_SIZE = 10;
const LINE_GAP = 3;

// Levels one channel spans, and a full turn in degrees.
const FULL = 255;
const DEGREES = 360;

// Which part of the wheel a drag started on.
const ON_RING = "ring";
const ON_SQUARE = "square";

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
 * Red, green and blue as hue, saturation and brightness.
 *
 * @param {number[]} rgb - Three levels, each 0 to 255.
 * @returns {number[]} Hue as a fraction of the wheel, then saturation and brightness, 0 to 1.
 */
export function rgbToHsv(rgb) {
  const [r, g, b] = rgb.map((level) => clamp(Number(level) || 0, 0, FULL) / FULL);
  const high = Math.max(r, g, b);
  const low = Math.min(r, g, b);
  const spread = high - low;
  let hue = 0;
  if (spread > 0) {
    if (high === r) hue = ((g - b) / spread + 6) % 6;
    else if (high === g) hue = (b - r) / spread + 2;
    else hue = (r - g) / spread + 4;
    hue /= 6;
  }
  return [hue, high > 0 ? spread / high : 0, high];
}

/**
 * Hue, saturation and brightness as red, green and blue.
 *
 * @param {number[]} hsv - Hue as a fraction of the wheel, then saturation and brightness.
 * @returns {number[]} Three levels, each a whole number 0 to 255.
 */
export function hsvToRgb(hsv) {
  const [hue, saturation, value] = hsv;
  const sector = (((hue % 1) + 1) % 1) * 6;
  const slice = Math.floor(sector);
  const rest = sector - slice;
  const p = value * (1 - saturation);
  const q = value * (1 - saturation * rest);
  const t = value * (1 - saturation * (1 - rest));
  const table = [
    [value, t, p], [q, value, p], [p, value, t],
    [p, q, value], [t, p, value], [value, p, q],
  ];
  return table[slice % 6].map((level) => clamp(Math.round(level * FULL), 0, FULL));
}

/**
 * Three levels as the code a person reads.
 *
 * @param {number[]} rgb - Three levels, each 0 to 255.
 * @returns {string} A `#rrggbb` colour.
 */
export function rgbToHex(rgb) {
  return `#${rgb.map((level) => clamp(Math.round(level), 0, FULL).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Put a colour wheel on a node, wired to the widgets holding the three levels.
 *
 * @param {object} node - The node being created.
 * @param {object} [options] - Everything below.
 * @param {string[]} [options.channels] - The three level widgets, red first.
 * @param {number} [options.height] - Height of the appended widget in node units.
 * @param {number} [options.minWidth] - The narrowest the wheel is drawn in, in node units.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   refresh: () => void, dispose: () => void}} The element to hand to
 *   `appendInterfaceWidget`, the height it was built for, its ceiling, the narrowest it
 *   draws in, a repaint, and teardown.
 */
export function createColourWheel(node, options = {}) {
  const channels = Array.isArray(options.channels) && options.channels.length === 3
    ? options.channels.slice()
    : DEFAULT_CHANNELS.slice();
  const height = Number(options.height) > 0 ? Number(options.height) : DEFAULT_HEIGHT;
  const minWidth = Number(options.minWidth) > 0 ? Number(options.minWidth) : DEFAULT_MIN_WIDTH;

  const root = document.createElement("div");
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${height - MARGIN}px`,
    "overflow:hidden",
    "touch-action:none",
    "user-select:none",
    "cursor:crosshair",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  const titles = hoverTitles(root);
  const state = { grab: null, draft: null, paintHandle: 0, disposed: false };

  /**
   * One widget of the node, by name.
   *
   * @param {string} name - Widget name.
   * @returns {object|null} The widget, or null where the node carries none of that name.
   */
  function findWidget(name) {
    return (node?.widgets || []).find((widget) => widget?.name === name) || null;
  }

  /**
   * The colour the widgets hold, or the one being dragged.
   *
   * @returns {number[]} Three levels, each a whole number 0 to 255.
   */
  function readColour() {
    if (state.draft) return state.draft.slice();
    return channels.map((name) => {
      const value = Number(findWidget(name)?.value);
      return Number.isFinite(value) ? clamp(Math.round(value), 0, FULL) : 0;
    });
  }

  /**
   * Write three levels onto the widgets, as one undo step.
   *
   * @param {number[]} rgb - Three levels, each 0 to 255.
   * @returns {void}
   */
  function writeColour(rgb) {
    const found = channels.map((name) => findWidget(name));
    if (found.some((widget) => !widget)) return;
    const next = rgb.map((level) => clamp(Math.round(level), 0, FULL));
    if (found.every((widget, index) => Number(widget.value) === next[index])) return;
    withGraphChange(() => {
      found.forEach((widget, index) => {
        widget.value = next[index];
        widget.callback?.(next[index]);
      });
    });
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Where the wheel and its readout sit in the element, in layout pixels.
   *
   * @returns {object} The centre, the ring's radii, the square and the readout strip.
   */
  function layout() {
    const width = root.clientWidth || minWidth;
    const tall = root.clientHeight || height;
    const wheelHeight = Math.max(40, tall - STRIP_HEIGHT);
    const radius = Math.max(22, Math.min(width, wheelHeight) / 2 - MARGIN);
    const thickness = radius * RING_SHARE;
    const inner = radius - thickness - GAP;
    // The square is the largest one that fits inside the ring.
    const side = Math.max(12, inner * Math.SQRT1_2 * 2);
    const cx = width / 2;
    const cy = wheelHeight / 2;
    return {
      width,
      tall,
      cx,
      cy,
      radius,
      thickness,
      inner,
      square: { x: cx - side / 2, y: cy - side / 2, side },
      strip: { x: 0, y: wheelHeight, width, height: tall - wheelHeight },
    };
  }

  /**
   * Draw the hue ring.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {object} place - The layout.
   * @returns {void}
   */
  function ring(ctx, place) {
    const sweep = (Math.PI * 2) / WEDGES;
    for (let index = 0; index < WEDGES; index += 1) {
      const turn = index / WEDGES;
      // Twelve o'clock is a hue of zero and the ring runs clockwise, as a picker does.
      const from = turn * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(place.cx, place.cy, place.radius, from, from + sweep + 0.01);
      ctx.arc(place.cx, place.cy, place.radius - place.thickness, from + sweep + 0.01, from, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${turn * DEGREES} 100% 50%)`;
      ctx.fill();
    }
  }

  /**
   * Draw the saturation and brightness square for one hue.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {object} box - The square, as `{x, y, side}`.
   * @param {number} hue - Hue as a fraction of the wheel.
   * @returns {void}
   */
  function square(ctx, box, hue) {
    const pure = `hsl(${hue * DEGREES} 100% 50%)`;
    ctx.fillStyle = pure;
    ctx.fillRect(box.x, box.y, box.side, box.side);

    // Saturation runs left to right over white, brightness top to bottom into black.
    const across = ctx.createLinearGradient(box.x, 0, box.x + box.side, 0);
    across.addColorStop(0, "rgba(255,255,255,1)");
    across.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = across;
    ctx.fillRect(box.x, box.y, box.side, box.side);

    const down = ctx.createLinearGradient(0, box.y, 0, box.y + box.side);
    down.addColorStop(0, "rgba(0,0,0,0)");
    down.addColorStop(1, "rgba(0,0,0,1)");
    ctx.fillStyle = down;
    ctx.fillRect(box.x, box.y, box.side, box.side);
  }

  /**
   * Draw one handle, outlined so it reads on any colour under it.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {number} x - Centre, in element pixels.
   * @param {number} y - Centre, in element pixels.
   * @returns {void}
   */
  function handle(ctx, x, y) {
    ctx.beginPath();
    ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = HANDLE_LINE + 1.5;
    ctx.stroke();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = HANDLE_LINE;
    ctx.stroke();
  }

  /**
   * Draw the swatch and the levels under the wheel.
   *
   * @param {CanvasRenderingContext2D} ctx - Context to draw on.
   * @param {object} place - The layout.
   * @param {number[]} rgb - The colour being drawn.
   * @param {object} theme - The palette.
   * @returns {object} The glyph's area, ready to hand to `hoverTitles`.
   */
  function strip(ctx, place, rgb, theme) {
    const { x, y, width, height: tall } = place.strip;
    const hex = rgbToHex(rgb);
    const block = HEX_SIZE + LINE_GAP + LEVEL_SIZE;
    const top = y + Math.max(2, (tall - block) / 2);
    const swatchTall = Math.max(8, Math.min(block, tall - 6));

    ctx.fillStyle = hex;
    ctx.fillRect(x + MARGIN, top, SWATCH_WIDTH, swatchTall);
    ctx.strokeStyle = theme.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + MARGIN + 0.5, top + 0.5, SWATCH_WIDTH - 1, swatchTall - 1);

    const textLeft = x + MARGIN + SWATCH_WIDTH + 8;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = theme.fg;
    ctx.font = `600 ${HEX_SIZE}px monospace`;
    ctx.fillText(hex, textLeft, top + HEX_SIZE);
    ctx.fillStyle = theme.fgMuted;
    ctx.font = `${LEVEL_SIZE}px sans-serif`;
    ctx.fillText(
      `${rgb[0]}, ${rgb[1]}, ${rgb[2]}`, textLeft, top + HEX_SIZE + LINE_GAP + LEVEL_SIZE,
    );

    return drawIcon(
      ctx, ICON.EXACT, width - MARGIN - ICON_SIZE, top + (swatchTall - ICON_SIZE) / 2,
      ICON_SIZE, theme.fgMuted,
    );
  }

  /**
   * Draw the whole wheel.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const theme = readTheme();
    const place = layout();
    const ratio = surfaceRatio(root);
    canvas.width = Math.max(1, Math.round(place.width * ratio));
    canvas.height = Math.max(1, Math.round(place.tall * ratio));
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      titles.set([]);
      return;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, place.width, place.tall);

    const rgb = readColour();
    const [hue, saturation, value] = rgbToHsv(rgb);

    ring(ctx, place);
    square(ctx, place.square, hue);

    const angle = hue * Math.PI * 2 - Math.PI / 2;
    const reach = place.radius - place.thickness / 2;
    handle(ctx, place.cx + Math.cos(angle) * reach, place.cy + Math.sin(angle) * reach);
    handle(
      ctx,
      place.square.x + saturation * place.square.side,
      place.square.y + (1 - value) * place.square.side,
    );

    const box = strip(ctx, place, rgb, theme);
    titles.set([
      {
        ...box,
        title: iconTitle(
          ICON.EXACT,
          `the swatch is filled with ${rgbToHex(rgb)}, the levels the node fills with`,
        ),
      },
    ]);
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
        console.error(`[${LOG_NAME}] Failed to draw the colour wheel:`, error);
      }
    });
  }

  /**
   * Which part of the wheel one pointer position is over.
   *
   * @param {PointerEvent} event - Event to read.
   * @returns {string|null} `ON_RING`, `ON_SQUARE`, or null for neither.
   */
  function partAt(event) {
    const point = elementPoint(root, event);
    const place = layout();
    const box = place.square;
    if (
      point.x >= box.x && point.x <= box.x + box.side
      && point.y >= box.y && point.y <= box.y + box.side
    ) {
      return ON_SQUARE;
    }
    const reach = Math.hypot(point.x - place.cx, point.y - place.cy);
    return reach <= place.radius + HANDLE_RADIUS ? ON_RING : null;
  }

  /**
   * The colour one pointer position stands for, on the part the drag started on.
   *
   * @param {PointerEvent} event - Event to read.
   * @param {string} part - `ON_RING` or `ON_SQUARE`.
   * @param {number[]} from - The colour the drag started from.
   * @returns {number[]} Three levels, each a whole number 0 to 255.
   */
  function colourAt(event, part, from) {
    const point = elementPoint(root, event);
    const place = layout();
    const held = rgbToHsv(from);
    if (part === ON_RING) {
      const angle = Math.atan2(point.x - place.cx, place.cy - point.y);
      const hue = ((angle / (Math.PI * 2)) + 1) % 1;
      // A colourless start takes full saturation and value.
      return hsvToRgb([hue, held[1] || 1, held[2] || 1]);
    }
    const box = place.square;
    return hsvToRgb([
      held[0],
      clamp((point.x - box.x) / box.side, 0, 1),
      clamp(1 - (point.y - box.y) / box.side, 0, 1),
    ]);
  }

  /**
   * Start a drag on whichever part the pointer went down on.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerDown(event) {
    // The middle button goes to the graph.
    if (event.button === 1) {
      window.app?.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    const part = partAt(event);
    if (!part) return;
    state.grab = part;
    root.setPointerCapture?.(event.pointerId);
    // Held in a draft until the drag ends.
    state.draft = colourAt(event, part, readColour());
    schedulePaint();
    event.stopPropagation();
    event.preventDefault();
  }

  /**
   * Follow the pointer while the colour is being dragged.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function onPointerMove(event) {
    if (!state.grab) return;
    // A drag whose release landed in another window arrives here with nothing held.
    if (typeof event.buttons === "number" && !(event.buttons & 1)) {
      finish(event);
      return;
    }
    state.draft = colourAt(event, state.grab, state.draft || readColour());
    schedulePaint();
    event.stopPropagation();
  }

  /**
   * Commit the drag and end it.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function finish(event) {
    if (!state.grab) return;
    const picked = state.draft;
    state.grab = null;
    state.draft = null;
    try {
      root.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to release the pointer:`, error);
    }
    if (picked) writeColour(picked);
    schedulePaint();
    event.stopPropagation?.();
  }

  /**
   * Abandon the drag, leaving the widgets as they were.
   *
   * @param {PointerEvent} event - Pointer event.
   * @returns {void}
   */
  function abandon(event) {
    if (!state.grab) return;
    state.grab = null;
    state.draft = null;
    schedulePaint();
    event.stopPropagation?.();
  }

  /**
   * Keep the browser's own menu off the wheel.
   *
   * @param {MouseEvent} event - Context menu event.
   * @returns {void}
   */
  function onContextMenu(event) {
    event.preventDefault();
  }

  root.addEventListener("pointerdown", onPointerDown);
  root.addEventListener("pointermove", onPointerMove);
  root.addEventListener("pointerup", finish);
  root.addEventListener("pointercancel", abandon);
  root.addEventListener("lostpointercapture", abandon);
  root.addEventListener("contextmenu", onContextMenu);
  // Every gesture goes to the graph.
  const releaseWheel = captureWheel(root);

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
  }

  const unwatch = watchSurfaceRatio(root, schedulePaint);

  const unwatchTheme = onThemeChange(schedulePaint);

  schedulePaint();

  return {
    element: root,
    height,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth,
    refresh: schedulePaint,
    dispose() {
      state.disposed = true;
      if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
      state.paintHandle = 0;
      root.removeEventListener("pointerdown", onPointerDown);
      root.removeEventListener("pointermove", onPointerMove);
      root.removeEventListener("pointerup", finish);
      root.removeEventListener("pointercancel", abandon);
      root.removeEventListener("lostpointercapture", abandon);
      root.removeEventListener("contextmenu", onContextMenu);
      releaseWheel();
      observer?.disconnect();
      unwatch?.();
      unwatchTheme?.();
      titles.dispose();
    },
  };
}
