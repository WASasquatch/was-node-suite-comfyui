/**
 * Local adjustment masks drawn on the picture: a linear gradient's three lines, a radial
 * gradient's ellipse, and colour or lightness samples, over the mask each node made.
 */

import { app } from "../../scripts/app.js";
import { findWidget } from "./interface/audio_view.js";
import { drawLabel } from "./interface/canvas_label.js";
import { CARD_LOOK, CHIP_TONES, cardChip, cardRule } from "./interface/card.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.js";
import { drawStandIn, loadPlaceholder, standInDetail } from "./interface/placeholder.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { LABELS, PREVIEW_STATE, fetchInputPreview, fetchOutputPreview } from "./interface/preview.js";
import { withGraphChange } from "./interface/region.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onNodeFinished } from "./interface/run_events.js";
import { fetchRunResult, whenReadable } from "./interface/run_result.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";
import { SANS, el } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.LocalMasksUI";

const GRADUATED_NODE = "WASImageGraduatedMask";
const RADIAL_NODE = "WASImageRadialMask";
const COLOR_RANGE_NODE = "WASImageColorRangeMask";

const UI_WIDGET_NAME = "was_local_mask_ui";
const UI_WIDGET_TYPE = "was_local_mask";

// The slots the nodes file their frames and masks under.
const FRAMES_SLOT = "images";
const MASK_SLOT = "mask";

// The optional input a wired mask arrives on.
const MASK_INPUT = "mask";

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 420;
const PANEL_WIDTH = 340;

// Room round the picture, so a handle just past its edge stays reachable, in CSS pixels.
const MARGIN = 16;

// How near a press has to land to take a handle, a line, or a marker, in CSS pixels.
const HIT = 8;
const LINE_HIT = 6;

// How far a press may move and still count as a click, in CSS pixels.
const CLICK_SLOP = 4;

// Distance from the gradient's pin to its turn knob, and from the ellipse's top to its knob.
const KNOB_OFFSET = 34;
const KNOB_GAP = 22;

// Closest the gradient's two lines may be dragged, and the least a new shape spans, in CSS pixels.
const MIN_SPAN = 6;

// Degrees a turn snaps to with shift held.
const SNAP_DEGREES = 15;

// Decimals a written position keeps.
const DECIMALS = 4;

// The range a position widget accepts.
const POSITION = { min: -1, max: 2 };

// The range a radius widget accepts.
const RADIUS = { min: 0.001, max: 4 };

// Opacity of the overlay at full mask, and of its line at half.
const OVERLAY_ALPHA = 0.55;
const CONTOUR_ALPHA = 0.9;

// Most colour samples one mask reads, and the area a click averages.
const MAX_SAMPLES = 16;
const SAMPLE_SIDE = 0.01;
const SAMPLE_LEAST = 3;

// Lightness a smoothness of 1 fades over past a luminance band, and a clicked band's width.
const LUMINANCE_FADE = 0.25;
const LUMINANCE_BAND = 0.1;

// Height of the luminance bar and of the frame track, in CSS pixels.
const BAR_HEIGHT = 34;
const TRACK_HEIGHT = 22;

// Side room inside the luminance bar, in CSS pixels.
const BAR_INSET = 9;

// Bins of the lightness histogram drawn behind the bar.
const HISTOGRAM_BINS = 64;

// Requests for frames and masks in flight at once.
const CONCURRENCY = 3;

// What the picture well shows.
const VIEW = { PICTURE: "picture", OVERLAY: "overlay", MASK: "mask" };

const COMBINES = ["replace", "add", "subtract", "intersect"];

// A typed colour sample.
const HEX = /^#?([0-9a-fA-F]{6})$/;

// Linear sRGB to cone response, and cone response to Oklab.
const TO_LMS = [
  [0.4122214708, 0.5363325363, 0.0514459929],
  [0.2119034982, 0.6806995451, 0.1073969566],
  [0.0883024619, 0.2817188376, 0.6299787005],
];
const LMS_TO_LAB = [
  [0.2104542553, 0.7936177850, -0.0040720468],
  [1.9779984951, -2.4285922050, 0.4505937099],
  [0.0259040371, 0.7827717662, -0.8086757660],
];

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
 * A number cut to the decimals a widget is written with.
 *
 * @param {number} value - The number.
 * @returns {number} The number at `DECIMALS` places.
 */
function rounded(value) {
  return Number(Number(value).toFixed(DECIMALS));
}

/**
 * A rise from 0 to 1, in the profile the nodes use.
 *
 * @param {number} t - Position, held to 0 to 1.
 * @param {string} profile - `linear`, or `smooth` for `3t^2 - 2t^3`.
 * @returns {number} 0 to 1.
 */
function ramp(t, profile) {
  const held = clamp(t, 0, 1);
  return profile === "linear" ? held : held * held * (3 - 2 * held);
}

/**
 * 1 up to `full`, 0 from `zero` on, and the profile's fall between.
 *
 * @param {number} distance - The distance measured.
 * @param {number} full - Where the fall starts.
 * @param {number} zero - Where it ends.
 * @param {string} profile - The fall's shape.
 * @returns {number} 0 to 1.
 */
function fade(distance, full, zero, profile) {
  if (zero - full <= 1e-12) return distance <= zero ? 1 : 0;
  return 1 - ramp((distance - full) / (zero - full), profile);
}

/**
 * Light from one sRGB picture code.
 *
 * @param {number} code - 0 to 1.
 * @returns {number} Linear light.
 */
function linearise(code) {
  const held = Math.max(0, code);
  return held <= 0.04045 ? held / 12.92 : ((held + 0.055) / 1.055) ** 2.4;
}

/**
 * The sRGB picture code for linear light.
 *
 * @param {number} light - Linear light.
 * @returns {number} 0 to 1.
 */
function encode(light) {
  const held = Math.max(0, light);
  return clamp(held <= 0.0031308 ? held * 12.92 : 1.055 * held ** (1 / 2.4) - 0.055, 0, 1);
}

/**
 * Oklab of linear light.
 *
 * @param {number} red - Linear red.
 * @param {number} green - Linear green.
 * @param {number} blue - Linear blue.
 * @returns {number[]} Lightness, a and b.
 */
function oklab(red, green, blue) {
  const cone = TO_LMS.map((row) => Math.cbrt(row[0] * red + row[1] * green + row[2] * blue));
  return LMS_TO_LAB.map((row) => row[0] * cone[0] + row[1] * cone[1] + row[2] * cone[2]);
}

/**
 * The linear light of a typed `#rrggbb` colour.
 *
 * @param {string} text - The colour.
 * @returns {number[]|null} Three channels, or null where it is not a colour.
 */
function hexLight(text) {
  const match = HEX.exec(String(text ?? "").trim());
  if (!match) return null;
  return [0, 2, 4].map((at) => linearise(parseInt(match[1].slice(at, at + 2), 16) / 255));
}

/**
 * The `#rrggbb` code of linear light.
 *
 * @param {number[]} light - Three channels.
 * @returns {string} The code.
 */
function hexOf(light) {
  return `#${light.map((value) => Math.round(encode(value) * 255).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Turn a point about a centre.
 *
 * @param {{x: number, y: number}} point - The point.
 * @param {{x: number, y: number}} centre - The centre.
 * @param {number} turn - Radians, clockwise on screen.
 * @returns {{x: number, y: number}} The turned point.
 */
function turned(point, centre, turn) {
  const cos = Math.cos(turn);
  const sin = Math.sin(turn);
  const dx = point.x - centre.x;
  const dy = point.y - centre.y;
  return { x: centre.x + dx * cos - dy * sin, y: centre.y + dx * sin + dy * cos };
}

/**
 * An angle snapped to `SNAP_DEGREES`.
 *
 * @param {number} radians - The angle.
 * @returns {number} The nearest snapped angle, in radians.
 */
function snapped(radians) {
  const step = (SNAP_DEGREES * Math.PI) / 180;
  return Math.round(radians / step) * step;
}

/**
 * Degrees held to -180 to 180.
 *
 * @param {number} degrees - Any angle.
 * @returns {number} The same turn, from -180 to 180.
 */
function wrapDegrees(degrees) {
  let held = ((degrees + 180) % 360 + 360) % 360 - 180;
  if (held === -180 && degrees > 0) held = 180;
  return held;
}

/**
 * The colour samples a widget holds.
 *
 * @param {*} value - The `samples` widget's value.
 * @returns {{entries: Array<object>, readable: boolean}} Each sample as a click, an area or a
 *   typed colour, and whether every entry could be read, which allows the panel to rewrite it.
 */
function readSamples(value) {
  const entries = [];
  let readable = true;
  for (const piece of String(value ?? "").split(/[;\n]+/)) {
    const text = piece.trim();
    if (!text) continue;
    if (entries.length >= MAX_SAMPLES) {
      readable = false;
      break;
    }
    const typed = HEX.exec(text);
    if (typed) {
      entries.push({ kind: "typed", hex: `#${typed[1].toLowerCase()}` });
      continue;
    }
    const [place, frame = ""] = text.split("@");
    const numbers = place.split(",").map((part) => Number(part.trim()));
    const index = frame.trim() ? Number(frame.trim()) : 0;
    if ((numbers.length !== 2 && numbers.length !== 4) || !numbers.every(Number.isFinite)
      || !Number.isInteger(index) || index < 0) {
      readable = false;
      continue;
    }
    const [x, y] = numbers.slice(0, 2).map((number) => clamp(number, 0, 1));
    const [w, h] = (numbers.length === 4 ? numbers.slice(2) : [0, 0]).map((number) => clamp(number, 0, 1));
    entries.push({ kind: w > 0 || h > 0 ? "area" : "point", x, y, w, h, frame: index });
  }
  return { entries, readable };
}

/**
 * Samples as the text the `samples` widget holds.
 *
 * @param {Array<object>} entries - Samples as `readSamples` answers them.
 * @returns {string} The text, empty for none.
 */
function writeSamples(entries) {
  return entries.map((entry) => {
    if (entry.kind === "typed") return entry.hex;
    const place = entry.kind === "area"
      ? [entry.x, entry.y, entry.w, entry.h]
      : [entry.x, entry.y];
    return `${place.map(rounded).join(",")}@${entry.frame}`;
  }).join("; ");
}

/**
 * Samples in the form a run reports them, for comparing.
 *
 * @param {Array<object>} entries - Samples as `readSamples` answers them.
 * @returns {Array<*>} A typed colour as its code, anything else as `[x, y, w, h, frame]`.
 */
function sampleKeys(entries) {
  return entries.map((entry) => (entry.kind === "typed"
    ? entry.hex
    : [rounded(entry.x), rounded(entry.y), rounded(entry.w ?? 0), rounded(entry.h ?? 0), entry.frame]));
}

/**
 * Whether two reported values agree, numbers to the written decimals.
 *
 * @param {*} first - One value.
 * @param {*} second - The other.
 * @returns {boolean} True when they agree.
 */
function same(first, second) {
  if (typeof first === "number" && typeof second === "number") return Math.abs(first - second) < 1e-6;
  if (Array.isArray(first) && Array.isArray(second)) {
    return first.length === second.length && first.every((item, index) => same(item, second[index]));
  }
  return first === second;
}

/**
 * A JSON value from report text, or null where it is not JSON.
 *
 * @param {*} text - The text.
 * @returns {*} The parsed value, or null.
 */
function parsed(text) {
  if (typeof text !== "string" || !text.trim()) return null;
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
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
  if (!ctx) return [240, 70, 60];
  ctx.fillStyle = colour;
  ctx.fillRect(0, 0, 1, 1);
  const data = ctx.getImageData(0, 0, 1, 1).data;
  return [data[0], data[1], data[2]];
}

/**
 * A mask drawn as a tint with a line where it crosses one half, or as grey.
 *
 * @param {Float32Array} levels - One value per pixel, row by row.
 * @param {number} width - Pixels across.
 * @param {number} height - Pixels down.
 * @param {number[]} tint - Channels of the overlay colour.
 * @param {boolean} plain - Grey levels rather than a tint.
 * @returns {HTMLCanvasElement|null} The picture, or null where a canvas was refused.
 */
function maskPicture(levels, width, height, tint, plain) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const image = ctx.createImageData(width, height);
  const data = image.data;
  for (let index = 0; index < width * height; index += 1) {
    const level = clamp(levels[index], 0, 1);
    const at = index * 4;
    if (plain) {
      const grey = Math.round(level * 255);
      data[at] = grey;
      data[at + 1] = grey;
      data[at + 2] = grey;
      data[at + 3] = 255;
      continue;
    }
    data[at] = tint[0];
    data[at + 1] = tint[1];
    data[at + 2] = tint[2];
    data[at + 3] = Math.round(level * OVERLAY_ALPHA * 255);
  }
  if (!plain) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const inside = levels[index] >= 0.5;
        const across = x + 1 < width && (levels[index + 1] >= 0.5) !== inside;
        const down = y + 1 < height && (levels[index + width] >= 0.5) !== inside;
        if (across || down) data[index * 4 + 3] = Math.round(CONTOUR_ALPHA * 255);
      }
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

/**
 * A pill button in the card's colours.
 *
 * @param {string} label - Its words.
 * @param {string} hint - Hover text.
 * @param {() => void} onPress - Called on a press.
 * @returns {{element: HTMLButtonElement, setOn: (on: boolean) => void}} The button, and a way to
 *   show it chosen.
 */
function pill(label, hint, onPress) {
  const press = document.createElement("button");
  press.type = "button";
  press.textContent = label;
  press.title = hint;
  let on = false;
  const style = (hovered) => {
    const tone = on ? CHIP_TONES.accent : null;
    press.style.cssText = "flex:none;padding:2px 9px;border-radius:999px;cursor:pointer;white-space:nowrap;"
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
 * A row of pills of which one is chosen, in an inset well.
 *
 * @param {Array<{value: string, label: string, hint: string}>} options - The choices.
 * @param {(value: string) => void} onPick - Called with the value pressed.
 * @returns {{element: HTMLElement, set: (value: string) => void}} The row, and a way to show the
 *   chosen one.
 */
function segment(options, onPick) {
  const element = el("div", "flex:none;display:flex;gap:2px;padding:2px;border-radius:999px;"
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well}`);
  const buttons = options.map((option) => {
    const button = pill(option.label, option.hint, () => onPick(option.value));
    element.append(button.element);
    return [option.value, button];
  });
  return {
    element,
    set(value) {
      for (const [option, button] of buttons) button.setOn(option === value);
    },
  };
}

/**
 * Draw a line through a point, far enough each way to cross the stage, with a dark halo.
 *
 * @param {CanvasRenderingContext2D} ctx - The stage's context.
 * @param {{x: number, y: number}} through - A point on the line.
 * @param {{x: number, y: number}} along - The line's unit direction.
 * @param {number} reach - How far it runs each way.
 * @param {object} theme - The palette.
 * @param {{width?: number, dash?: number[], hot?: boolean}} style - Its look.
 * @returns {void}
 */
function strokeLine(ctx, through, along, reach, theme, style) {
  const width = style.width ?? 1.5;
  ctx.save();
  ctx.lineCap = "round";
  ctx.setLineDash(style.dash ?? []);
  ctx.beginPath();
  ctx.moveTo(through.x - along.x * reach, through.y - along.y * reach);
  ctx.lineTo(through.x + along.x * reach, through.y + along.y * reach);
  ctx.globalAlpha = 0.7;
  ctx.strokeStyle = theme.bgDark;
  ctx.lineWidth = width + 2;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = style.hot ? theme.accent : theme.fg;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

/**
 * Draw a round handle.
 *
 * @param {CanvasRenderingContext2D} ctx - The stage's context.
 * @param {{x: number, y: number}} at - Its centre.
 * @param {number} radius - Its radius.
 * @param {object} theme - The palette.
 * @param {{filled?: boolean, hot?: boolean}} style - Filled or hollow, and whether it is under
 *   the pointer.
 * @returns {void}
 */
function roundHandle(ctx, at, radius, theme, style) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(at.x, at.y, radius, 0, Math.PI * 2);
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = theme.bgDark;
  ctx.stroke();
  ctx.fillStyle = style.hot ? theme.accent : style.filled ? theme.fg : theme.bgDark;
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = style.hot ? theme.fg : style.filled ? theme.bgDark : theme.fg;
  ctx.stroke();
  ctx.restore();
}

/**
 * Draw a square handle turned with its shape.
 *
 * @param {CanvasRenderingContext2D} ctx - The stage's context.
 * @param {{x: number, y: number}} at - Its centre.
 * @param {number} turn - Radians it is turned.
 * @param {object} theme - The palette.
 * @param {boolean} hot - Whether it is under the pointer.
 * @param {boolean} [diamond] - Drawn a further eighth of a turn round.
 * @returns {void}
 */
function squareHandle(ctx, at, turn, theme, hot, diamond = false) {
  ctx.save();
  ctx.translate(at.x, at.y);
  ctx.rotate(turn + (diamond ? Math.PI / 4 : 0));
  ctx.lineWidth = 3.5;
  ctx.strokeStyle = theme.bgDark;
  ctx.strokeRect(-4, -4, 8, 8);
  ctx.fillStyle = hot ? theme.accent : theme.fg;
  ctx.fillRect(-4, -4, 8, 8);
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.bgDark;
  ctx.strokeRect(-4, -4, 8, 8);
  ctx.restore();
}

/**
 * Draw a label centred on a point, on a plate held inside the stage.
 *
 * @param {CanvasRenderingContext2D} ctx - The stage's context.
 * @param {{width: number, height: number}} size - The stage's size.
 * @param {object} theme - The palette.
 * @param {string} text - What it says.
 * @param {{x: number, y: number}} at - Its centre.
 * @returns {void}
 */
function centredLabel(ctx, size, theme, text, at) {
  ctx.save();
  ctx.font = "10px system-ui,sans-serif";
  const width = ctx.measureText(text).width + 8;
  ctx.restore();
  drawLabel(ctx, { width: size.width, height: size.height, top: 0, theme }, text, at.x - width / 2, at.y);
}

/**
 * The linear gradient's lines, pin and knob.
 *
 * @param {object} context - What the panel offers a tool.
 * @returns {object} The tool.
 */
function createGradientTool(context) {
  /**
   * Where the lines and handles sit for a set of values.
   *
   * @param {object} values - The node's values.
   * @returns {object} The picture rectangle, both line points, the pin, the knob and directions.
   */
  function geometry(values) {
    const rect = context.rect();
    const start = { x: rect.x + values.start_x * rect.width, y: rect.y + values.start_y * rect.height };
    const end = { x: rect.x + values.end_x * rect.width, y: rect.y + values.end_y * rect.height };
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    const normal = length > 1e-9
      ? { x: (end.x - start.x) / length, y: (end.y - start.y) / length }
      : { x: 0, y: 1 };
    const along = { x: -normal.y, y: normal.x };
    const centre = { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 };
    const stage = context.stageSize();
    let knob = { x: centre.x + along.x * KNOB_OFFSET, y: centre.y + along.y * KNOB_OFFSET };
    if (knob.x < 0 || knob.y < 0 || knob.x > stage.width || knob.y > stage.height) {
      knob = { x: centre.x - along.x * KNOB_OFFSET, y: centre.y - along.y * KNOB_OFFSET };
    }
    return { rect, start, end, centre, normal, along, length, knob };
  }

  /**
   * The values for a start and end point in stage pixels.
   *
   * @param {object} rect - The picture rectangle.
   * @param {{x: number, y: number}} start - The start line's point.
   * @param {{x: number, y: number}} end - The end line's point.
   * @returns {object} The four position widgets' values.
   */
  function toValues(rect, start, end) {
    const share = (value, origin, span) => rounded(clamp((value - origin) / span, POSITION.min, POSITION.max));
    return {
      start_x: share(start.x, rect.x, rect.width),
      start_y: share(start.y, rect.y, rect.height),
      end_x: share(end.x, rect.x, rect.width),
      end_y: share(end.y, rect.y, rect.height),
    };
  }

  return {
    tip(handle) {
      return {
        move: "Drag to move",
        start: "Drag the full-effect line",
        end: "Drag the no-effect line",
        turn: "Drag to turn; shift snaps",
        knob: "Drag to turn; shift snaps",
      }[handle] ?? "Drag to draw a gradient";
    },

    levels(width, height, values, source) {
      const across = source.width;
      const down = source.height;
      const sx = values.start_x * across;
      const sy = values.start_y * down;
      const dx = values.end_x * across - sx;
      const dy = values.end_y * down - sy;
      const span = dx * dx + dy * dy;
      if (span < 0.25) return null;
      const levels = new Float32Array(width * height);
      for (let y = 0; y < height; y += 1) {
        const py = ((y + 0.5) / height) * down - sy;
        for (let x = 0; x < width; x += 1) {
          const px = ((x + 0.5) / width) * across - sx;
          levels[y * width + x] = 1 - ramp((px * dx + py * dy) / span, values.profile);
        }
      }
      return levels;
    },

    paint(ctx, theme, values, hovered) {
      const g = geometry(values);
      const stage = context.stageSize();
      const reach = Math.hypot(stage.width, stage.height) * 2;
      strokeLine(ctx, g.start, g.along, reach, theme, { width: 1.8, hot: hovered === "start" });
      strokeLine(ctx, g.centre, g.along, reach, theme, { width: 1, dash: [3, 4], hot: hovered === "turn" });
      strokeLine(ctx, g.end, g.along, reach, theme, { width: 1.4, dash: [8, 5], hot: hovered === "end" });
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = theme.fg;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(g.centre.x, g.centre.y);
      ctx.lineTo(g.knob.x, g.knob.y);
      ctx.stroke();
      ctx.restore();
      roundHandle(ctx, g.knob, 4.5, theme, { hot: hovered === "knob" });
      roundHandle(ctx, g.centre, 6, theme, { filled: true, hot: hovered === "move" });
      centredLabel(ctx, stage, theme, "100%", { x: g.start.x - g.normal.x * 13, y: g.start.y - g.normal.y * 13 });
      centredLabel(ctx, stage, theme, "0%", { x: g.end.x + g.normal.x * 13, y: g.end.y + g.normal.y * 13 });
    },

    hit(point, values) {
      const g = geometry(values);
      if (Math.hypot(point.x - g.knob.x, point.y - g.knob.y) <= HIT) return "knob";
      if (Math.hypot(point.x - g.centre.x, point.y - g.centre.y) <= HIT + 1) return "move";
      const off = (through) => Math.abs((point.x - through.x) * g.normal.x + (point.y - through.y) * g.normal.y);
      if (off(g.start) <= LINE_HIT) return "start";
      if (off(g.end) <= LINE_HIT) return "end";
      if (off(g.centre) <= LINE_HIT) return "turn";
      return null;
    },

    cursor(handle) {
      if (handle === "move") return "move";
      if (handle) return "grab";
      return "crosshair";
    },

    begin(point, event, values) {
      if (event.button !== 0) return null;
      return { handle: this.hit(point, values) ?? "draw", geometry: geometry(values) };
    },

    drag(gesture, point, event) {
      const g = gesture.geometry;
      const origin = gesture.origin;
      let start = g.start;
      let end = g.end;
      if (gesture.handle === "move") {
        const dx = point.x - origin.x;
        const dy = point.y - origin.y;
        start = { x: g.start.x + dx, y: g.start.y + dy };
        end = { x: g.end.x + dx, y: g.end.y + dy };
      } else if (gesture.handle === "start") {
        const t = Math.min((point.x - g.start.x) * g.normal.x + (point.y - g.start.y) * g.normal.y, g.length - MIN_SPAN);
        start = { x: g.start.x + g.normal.x * t, y: g.start.y + g.normal.y * t };
      } else if (gesture.handle === "end") {
        const t = Math.max((point.x - g.end.x) * g.normal.x + (point.y - g.end.y) * g.normal.y, MIN_SPAN - g.length);
        end = { x: g.end.x + g.normal.x * t, y: g.end.y + g.normal.y * t };
      } else if (gesture.handle === "turn" || gesture.handle === "knob") {
        const before = Math.atan2(origin.y - g.centre.y, origin.x - g.centre.x);
        const now = Math.atan2(point.y - g.centre.y, point.x - g.centre.x);
        const direction = Math.atan2(g.normal.y, g.normal.x);
        let next = direction + now - before;
        if (event.shiftKey) next = snapped(next);
        start = turned(g.start, g.centre, next - direction);
        end = turned(g.end, g.centre, next - direction);
      } else {
        start = origin;
        const length = Math.hypot(point.x - origin.x, point.y - origin.y);
        let direction = Math.atan2(point.y - origin.y, point.x - origin.x);
        if (event.shiftKey) direction = snapped(direction);
        end = { x: origin.x + Math.cos(direction) * length, y: origin.y + Math.sin(direction) * length };
        if (length < MIN_SPAN) return;
      }
      context.setDraft(toValues(g.rect, start, end));
    },

    finish(gesture) {
      if (!gesture.moved) return null;
      return context.draft();
    },

    readout(values, source) {
      if (!source) return "";
      const dx = (values.end_x - values.start_x) * source.width;
      const dy = (values.end_y - values.start_y) * source.height;
      const tilt = wrapDegrees((Math.atan2(dy, dx) * 180) / Math.PI - 90);
      const held = tilt > 90 ? tilt - 180 : tilt <= -90 ? tilt + 180 : tilt;
      return `lines at ${held.toFixed(1)}°  ·  fades over ${Math.round(Math.hypot(dx, dy))} px`;
    },
  };
}

/**
 * The radial gradient's ellipse, feather ring and handles.
 *
 * @param {object} context - What the panel offers a tool.
 * @returns {object} The tool.
 */
function createRadialTool(context) {
  /**
   * Where the ellipse and its handles sit for a set of values.
   *
   * @param {object} values - The node's values.
   * @returns {object} The picture rectangle, centre, radii, turn, axes and every handle.
   */
  function geometry(values) {
    const rect = context.rect();
    const centre = { x: rect.x + values.centre_x * rect.width, y: rect.y + values.centre_y * rect.height };
    const across = Math.max(1e-6, values.radius_x * rect.width);
    const down = Math.max(1e-6, values.radius_y * rect.height);
    const turn = (values.angle * Math.PI) / 180;
    const u = { x: Math.cos(turn), y: Math.sin(turn) };
    const v = { x: -Math.sin(turn), y: Math.cos(turn) };
    const at = (a, b) => ({ x: centre.x + u.x * a + v.x * b, y: centre.y + u.y * a + v.y * b });
    const inner = 1 - clamp(values.feather, 0, 100) / 100;
    const corner = Math.SQRT1_2;
    const stage = context.stageSize();
    const above = at(0, -down - KNOB_GAP);
    const offStage = above.x < 0 || above.y < 0 || above.x > stage.width || above.y > stage.height;
    return {
      rect, centre, across, down, turn, u, v, inner,
      right: at(across, 0), left: at(-across, 0), bottom: at(0, down), top: at(0, -down),
      knob: offStage ? at(0, down + KNOB_GAP) : above,
      stem: offStage ? at(0, down) : at(0, -down),
      feather: at(across * inner * corner, -down * inner * corner),
    };
  }

  /**
   * How far a point sits out from the centre, 1 on the ellipse.
   *
   * @param {object} g - The geometry.
   * @param {{x: number, y: number}} point - The point.
   * @returns {number} The elliptical radius.
   */
  function reachOf(g, point) {
    const dx = point.x - g.centre.x;
    const dy = point.y - g.centre.y;
    const a = (dx * g.u.x + dy * g.u.y) / g.across;
    const b = (dx * g.v.x + dy * g.v.y) / g.down;
    return Math.hypot(a, b);
  }

  /**
   * Draw an ellipse.
   *
   * @param {CanvasRenderingContext2D} ctx - The stage's context.
   * @param {object} g - The geometry.
   * @param {number} scale - Its size against the outer ellipse.
   * @param {object} theme - The palette.
   * @param {{width: number, dash?: number[], hot?: boolean}} style - Its look.
   * @returns {void}
   */
  function ellipse(ctx, g, scale, theme, style) {
    ctx.save();
    ctx.setLineDash(style.dash ?? []);
    ctx.beginPath();
    ctx.ellipse(g.centre.x, g.centre.y, g.across * scale, g.down * scale, g.turn, 0, Math.PI * 2);
    ctx.globalAlpha = 0.7;
    ctx.strokeStyle = theme.bgDark;
    ctx.lineWidth = style.width + 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = style.hot ? theme.accent : theme.fg;
    ctx.lineWidth = style.width;
    ctx.stroke();
    ctx.restore();
  }

  return {
    tip(handle) {
      if (["right", "left", "top", "bottom"].includes(handle)) return "Resize; shift keeps shape";
      return {
        move: "Drag to move",
        turn: "Drag to turn; shift snaps",
        knob: "Drag to turn; shift snaps",
        feather: "Drag to set feather",
      }[handle] ?? "Drag to draw; shift circles";
    },

    levels(width, height, values, source) {
      const across = Math.max(values.radius_x * source.width, 1e-6);
      const down = Math.max(values.radius_y * source.height, 1e-6);
      const cx = values.centre_x * source.width;
      const cy = values.centre_y * source.height;
      const turn = (values.angle * Math.PI) / 180;
      const cos = Math.cos(turn);
      const sin = Math.sin(turn);
      const share = clamp(values.feather, 0, 100) / 100;
      const levels = new Float32Array(width * height);
      for (let y = 0; y < height; y += 1) {
        const dy = ((y + 0.5) / height) * source.height - cy;
        for (let x = 0; x < width; x += 1) {
          const dx = ((x + 0.5) / width) * source.width - cx;
          const a = (dx * cos + dy * sin) / across;
          const b = (-dx * sin + dy * cos) / down;
          const reach = Math.sqrt(a * a + b * b);
          levels[y * width + x] = share <= 0 ? (reach < 1 ? 1 : 0) : fade(reach, 1 - share, 1, values.profile);
        }
      }
      return levels;
    },

    paint(ctx, theme, values, hovered) {
      const g = geometry(values);
      ellipse(ctx, g, 1, theme, { width: 1.6, hot: hovered === "turn" });
      if (g.inner > 0 && g.inner < 1) ellipse(ctx, g, g.inner, theme, { width: 1, dash: [4, 4], hot: hovered === "feather" });
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = theme.fg;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(g.stem.x, g.stem.y);
      ctx.lineTo(g.knob.x, g.knob.y);
      ctx.stroke();
      ctx.restore();
      for (const name of ["right", "left", "top", "bottom"]) {
        squareHandle(ctx, g[name], g.turn, theme, hovered === name);
      }
      squareHandle(ctx, g.feather, g.turn, theme, hovered === "feather", true);
      roundHandle(ctx, g.knob, 4.5, theme, { hot: hovered === "knob" });
      roundHandle(ctx, g.centre, 5, theme, { filled: true, hot: hovered === "move" });
    },

    hit(point, values) {
      const g = geometry(values);
      const near = (at, room = HIT) => Math.hypot(point.x - at.x, point.y - at.y) <= room;
      if (near(g.knob)) return "knob";
      for (const name of ["right", "left", "top", "bottom"]) if (near(g[name])) return name;
      if (near(g.feather)) return "feather";
      if (near(g.centre, HIT + 1)) return "move";
      const reach = reachOf(g, point);
      if (Math.abs(reach - 1) * Math.min(g.across, g.down) <= LINE_HIT) return "turn";
      if (reach < 1) return "move";
      return null;
    },

    cursor(handle) {
      if (handle === "move") return "move";
      if (handle) return "grab";
      return "crosshair";
    },

    begin(point, event, values) {
      if (event.button !== 0) return null;
      return { handle: this.hit(point, values) ?? "draw", geometry: geometry(values), values };
    },

    drag(gesture, point, event) {
      const g = gesture.geometry;
      const rect = g.rect;
      const origin = gesture.origin;
      const values = gesture.values;
      const share = (value, base, span, low, high) => rounded(clamp((value - base) / span, low, high));
      const dx = point.x - g.centre.x;
      const dy = point.y - g.centre.y;
      if (gesture.handle === "move") {
        const centre = { x: g.centre.x + point.x - origin.x, y: g.centre.y + point.y - origin.y };
        context.setDraft({
          centre_x: share(centre.x, rect.x, rect.width, POSITION.min, POSITION.max),
          centre_y: share(centre.y, rect.y, rect.height, POSITION.min, POSITION.max),
        });
      } else if (gesture.handle === "right" || gesture.handle === "left") {
        const across = Math.abs(dx * g.u.x + dy * g.u.y);
        const next = { radius_x: share(across, 0, rect.width, RADIUS.min, RADIUS.max) };
        if (event.shiftKey) next.radius_y = rounded(clamp(values.radius_y * (across / g.across), RADIUS.min, RADIUS.max));
        context.setDraft(next);
      } else if (gesture.handle === "top" || gesture.handle === "bottom") {
        const down = Math.abs(dx * g.v.x + dy * g.v.y);
        const next = { radius_y: share(down, 0, rect.height, RADIUS.min, RADIUS.max) };
        if (event.shiftKey) next.radius_x = rounded(clamp(values.radius_x * (down / g.down), RADIUS.min, RADIUS.max));
        context.setDraft(next);
      } else if (gesture.handle === "turn" || gesture.handle === "knob") {
        const before = Math.atan2(origin.y - g.centre.y, origin.x - g.centre.x);
        const now = Math.atan2(dy, dx);
        let next = g.turn + now - before;
        if (event.shiftKey) next = snapped(next);
        context.setDraft({ angle: Number(wrapDegrees((next * 180) / Math.PI).toFixed(1)) });
      } else if (gesture.handle === "feather") {
        context.setDraft({ feather: Math.round(clamp(1 - reachOf(g, point), 0, 1) * 100) });
      } else {
        const spanX = Math.abs(point.x - origin.x);
        const spanY = Math.abs(point.y - origin.y);
        if (Math.hypot(spanX, spanY) < MIN_SPAN) return;
        const round = Math.hypot(spanX, spanY);
        context.setDraft({
          centre_x: share(origin.x, rect.x, rect.width, POSITION.min, POSITION.max),
          centre_y: share(origin.y, rect.y, rect.height, POSITION.min, POSITION.max),
          radius_x: share(event.shiftKey ? round : Math.max(spanX, MIN_SPAN), 0, rect.width, RADIUS.min, RADIUS.max),
          radius_y: share(event.shiftKey ? round : Math.max(spanY, MIN_SPAN), 0, rect.height, RADIUS.min, RADIUS.max),
          angle: 0,
        });
      }
    },

    finish(gesture) {
      if (!gesture.moved) return null;
      return context.draft();
    },

    readout(values, source) {
      if (!source) return "";
      const across = Math.round(values.radius_x * 2 * source.width);
      const down = Math.round(values.radius_y * 2 * source.height);
      return `${across} x ${down} px  ·  turned ${Number(values.angle).toFixed(1)}°  ·  feather ${Math.round(values.feather)}`;
    },
  };
}

/**
 * The colour samples and the luminance band of a range mask.
 *
 * @param {object} context - What the panel offers a tool.
 * @returns {object} The tool, with its own controls and the strip under the picture.
 */
function createRangeTool(context) {
  let bar = null;
  let swatchKey = "";

  const modes = segment([
    { value: "color", label: "Color", hint: "Select clicked colours" },
    { value: "luminance", label: "Luminance", hint: "Select a lightness band" },
  ], (value) => context.write({ mode: value }));
  const clearButton = pill("Clear", "Remove every sample", () => context.write({ samples: "" }));
  const swatches = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;gap:4px;align-items:center;min-width:0;"
    + "min-height:22px");
  const barWell = el("div", `position:relative;flex:0 0 ${BAR_HEIGHT}px;min-width:0;border-radius:6px;overflow:hidden;`
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:ew-resize`);
  const barCanvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  barWell.title = "Drag the band's ends";
  barWell.append(barCanvas);
  const below = el("div", "flex:0 0 auto;display:flex;flex-direction:column;gap:4px;min-width:0");
  below.append(swatches, barWell);

  /**
   * The samples the widget holds, as the draft has them during a gesture.
   *
   * @param {object} values - The node's values.
   * @returns {{entries: Array<object>, readable: boolean}} The samples.
   */
  function samplesOf(values) {
    return readSamples(values.samples);
  }

  /**
   * Oklab of a stored frame's pixels, worked out once.
   *
   * @param {number} index - A stored frame.
   * @returns {{width: number, height: number, lab: Float32Array}|null} The picture's Oklab, or
   *   null while it has not arrived.
   */
  function labOf(index) {
    const pixels = context.pixels(index);
    if (!pixels) return null;
    if (pixels.lab) return pixels;
    const { width, height, data } = pixels;
    const lab = new Float32Array(width * height * 3);
    for (let at = 0; at < width * height; at += 1) {
      const [l, a, b] = oklab(linearise(data[at * 4] / 255), linearise(data[at * 4 + 1] / 255),
        linearise(data[at * 4 + 2] / 255));
      lab[at * 3] = l;
      lab[at * 3 + 1] = a;
      lab[at * 3 + 2] = b;
    }
    pixels.lab = lab;
    return pixels;
  }

  /**
   * The pixel box a sample averages on a stored frame, as the node places it.
   *
   * @param {object} entry - A click or area sample.
   * @param {number} width - The stored frame's width.
   * @param {number} height - Its height.
   * @param {{width: number, height: number}} source - The picture the node read.
   * @returns {number[]} Left, top, right and bottom, right and bottom exclusive.
   */
  function areaOf(entry, width, height, source) {
    const scale = width / source.width;
    const side = Math.max(SAMPLE_LEAST, Math.round(SAMPLE_SIDE * Math.min(source.width, source.height)));
    const across = Math.min(source.width, entry.w > 0 ? Math.max(side, Math.round(entry.w * source.width)) : side);
    const down = Math.min(source.height, entry.h > 0 ? Math.max(side, Math.round(entry.h * source.height)) : side);
    const spanX = Math.max(1, Math.round(across * scale));
    const spanY = Math.max(1, Math.round(down * scale));
    const left = clamp(Math.round(entry.x * width - spanX / 2), 0, width - spanX);
    const top = clamp(Math.round(entry.y * height - spanY / 2), 0, height - spanY);
    return [left, top, left + spanX, top + spanY];
  }

  /**
   * The light a sample averages on the panel's copy of its frame.
   *
   * @param {object} entry - A click or area sample.
   * @returns {number[]|null} Linear light, or null where its frame is not on the panel.
   */
  function sampledLight(entry) {
    const stored = context.storedIndex(entry.frame);
    const pixels = stored === null ? null : context.pixels(stored);
    const source = context.source();
    if (!pixels || !source) return null;
    const [left, top, right, bottom] = areaOf(entry, pixels.width, pixels.height, source);
    const sum = [0, 0, 0];
    let count = 0;
    for (let y = top; y < bottom; y += 1) {
      for (let x = left; x < right; x += 1) {
        const at = (y * pixels.width + x) * 4;
        for (let channel = 0; channel < 3; channel += 1) sum[channel] += linearise(pixels.data[at + channel] / 255);
        count += 1;
      }
    }
    return count ? sum.map((value) => value / count) : null;
  }

  /**
   * The colour each sample stands for: the run's own where the sample is unchanged since it.
   *
   * @param {Array<object>} entries - The samples.
   * @returns {Array<{light: number[]|null, exact: boolean}>} One per sample.
   */
  function coloursOf(entries) {
    const settings = context.runSettings();
    const reported = parsed(context.fact("colors"));
    const keys = sampleKeys(entries);
    return entries.map((entry, index) => {
      if (entry.kind === "typed") return { light: hexLight(entry.hex), exact: true };
      const ran = Array.isArray(settings?.samples) ? settings.samples[index] : undefined;
      const hex = Array.isArray(reported) && Array.isArray(reported[index]) ? reported[index][0] : null;
      if (ran !== undefined && same(ran, keys[index]) && hexLight(hex)) return { light: hexLight(hex), exact: true };
      return { light: sampledLight(entry), exact: false };
    });
  }

  /**
   * The Oklab lightness round a point of the shown frame, over a few pixels.
   *
   * @param {{u: number, v: number}} share - The point, in shares of the picture.
   * @param {{u: number, v: number}} [other] - The far corner of an area, for its darkest and lightest.
   * @returns {number[]|null} The least and most lightness found, or null with no picture.
   */
  function lightnessAt(share, other) {
    const pixels = labOf(context.current());
    if (!pixels) return null;
    const { width, height, lab } = pixels;
    const x0 = Math.floor(clamp(Math.min(share.u, other?.u ?? share.u), 0, 1) * (width - 1));
    const x1 = Math.ceil(clamp(Math.max(share.u, other?.u ?? share.u), 0, 1) * (width - 1));
    const y0 = Math.floor(clamp(Math.min(share.v, other?.v ?? share.v), 0, 1) * (height - 1));
    const y1 = Math.ceil(clamp(Math.max(share.v, other?.v ?? share.v), 0, 1) * (height - 1));
    if (!other) {
      let sum = 0;
      let count = 0;
      for (let y = Math.max(0, y0 - 1); y <= Math.min(height - 1, y1 + 1); y += 1) {
        for (let x = Math.max(0, x0 - 1); x <= Math.min(width - 1, x1 + 1); x += 1) {
          sum += lab[(y * width + x) * 3];
          count += 1;
        }
      }
      const mean = sum / Math.max(1, count);
      return [mean, mean];
    }
    let least = Infinity;
    let most = -Infinity;
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        const value = lab[(y * width + x) * 3];
        least = Math.min(least, value);
        most = Math.max(most, value);
      }
    }
    return Number.isFinite(least) ? [least, most] : null;
  }

  /**
   * Where each picture sample sits on the stage.
   *
   * @param {Array<object>} entries - The samples.
   * @returns {Array<{index: number, entry: object, x: number, y: number, box: number[]|null}>} One
   *   per click or area, typed colours left out.
   */
  function markers(entries) {
    const rect = context.rect();
    const placed = [];
    entries.forEach((entry, index) => {
      if (entry.kind === "typed") return;
      const x = rect.x + entry.x * rect.width;
      const y = rect.y + entry.y * rect.height;
      const box = entry.kind === "area"
        ? [x - (entry.w * rect.width) / 2, y - (entry.h * rect.height) / 2, entry.w * rect.width, entry.h * rect.height]
        : null;
      placed.push({ index, entry, x, y, box });
    });
    return placed;
  }

  /**
   * The sample under a point.
   *
   * @param {{x: number, y: number}} point - The point, in stage pixels.
   * @param {Array<object>} entries - The samples.
   * @returns {number} Its index, or -1.
   */
  function markerAt(point, entries) {
    let found = -1;
    for (const marker of markers(entries)) {
      if (Math.hypot(point.x - marker.x, point.y - marker.y) <= HIT) return marker.index;
      if (marker.box && point.x >= marker.box[0] && point.x <= marker.box[0] + marker.box[2]
        && point.y >= marker.box[1] && point.y <= marker.box[1] + marker.box[3]) found = marker.index;
    }
    return found;
  }

  /**
   * Fill the row of swatches under the picture, rebuilt only when it changes.
   *
   * @param {object} values - The node's values.
   * @returns {void}
   */
  function paintSwatches(values) {
    const { entries } = samplesOf(values);
    const colours = coloursOf(entries);
    const key = JSON.stringify([entries, colours.map((one) => one.light && hexOf(one.light)), context.batchIndex()]);
    if (key === swatchKey) return;
    swatchKey = key;
    swatches.replaceChildren();
    if (!entries.length) {
      swatches.append(el("span", `color:${CARD_LOOK.muted};font:11px/1.4 ${SANS};min-width:0`,
        "Click the picture to sample"));
      return;
    }
    entries.forEach((entry, index) => {
      const chip = el("span", "flex:none;display:inline-flex;align-items:center;gap:4px;padding:1px 3px 1px 2px;"
        + `border-radius:999px;border:1px solid ${CARD_LOOK.edge};background:${CARD_LOOK.inset};`
        + `font:600 10.5px/16px ${SANS};color:${CARD_LOOK.text}`);
      const swatch = el("span", `flex:none;width:14px;height:14px;border-radius:999px;border:1px solid ${CARD_LOOK.edge}`);
      const light = colours[index].light;
      if (light) swatch.style.backgroundColor = hexOf(light);
      if (!colours[index].exact) swatch.style.borderStyle = "dashed";
      const where = entry.kind === "typed" ? entry.hex : entry.frame === context.batchIndex() ? "" : `f${entry.frame}`;
      chip.title = entry.kind === "typed"
        ? `Typed ${entry.hex}`
        : `Frame ${entry.frame}${light ? ` ${hexOf(light)}` : ""}${colours[index].exact ? "" : " estimated"}`;
      const remove = el("span", `flex:none;cursor:pointer;padding:0 3px;color:${CARD_LOOK.muted}`, "×");
      remove.title = "Remove this sample";
      remove.addEventListener("pointerdown", (event) => event.stopPropagation());
      remove.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const current = samplesOf(context.values());
        if (!current.readable) return;
        context.write({ samples: writeSamples(current.entries.filter((_, at) => at !== index)) });
      });
      chip.append(swatch, el("span", "", `${index + 1}${where ? ` ${where}` : ""}`), remove);
      swatches.append(chip);
    });
  }

  /**
   * Draw the luminance bar: the lightness ramp, the picture's histogram, the band and its fade.
   *
   * @param {object} theme - The palette.
   * @param {object} values - The node's values.
   * @returns {void}
   */
  function paintBar(theme, values) {
    const surface = context.sized(barCanvas);
    if (!surface) return;
    const { ctx, width, height } = surface;
    const usable = Math.max(1, width - BAR_INSET * 2);
    const xOf = (lightness) => BAR_INSET + clamp(lightness, 0, 1) * usable;
    for (let x = 0; x < usable; x += 2) {
      const grey = Math.round(encode((x / usable) ** 3) * 255);
      ctx.fillStyle = `rgb(${grey},${grey},${grey})`;
      ctx.fillRect(BAR_INSET + x, 4, 2, height - 8);
    }
    const pixels = labOf(context.current());
    if (pixels) {
      if (!pixels.bins) {
        pixels.bins = new Float32Array(HISTOGRAM_BINS);
        for (let at = 0; at < pixels.width * pixels.height; at += 1) {
          pixels.bins[clamp(Math.floor(pixels.lab[at * 3] * HISTOGRAM_BINS), 0, HISTOGRAM_BINS - 1)] += 1;
        }
      }
      const bins = pixels.bins;
      const tallest = Math.max(...bins) || 1;
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = theme.accent;
      for (let bin = 0; bin < HISTOGRAM_BINS; bin += 1) {
        const tall = (Math.sqrt(bins[bin] / tallest)) * (height - 10);
        ctx.fillRect(BAR_INSET + (bin / HISTOGRAM_BINS) * usable, height - 5 - tall, usable / HISTOGRAM_BINS - 0.5, tall);
      }
      ctx.restore();
    }
    const low = Math.min(values.luminance_low, values.luminance_high);
    const high = Math.max(values.luminance_low, values.luminance_high);
    const reach = clamp(values.smoothness, 0, 1) * LUMINANCE_FADE;
    ctx.save();
    ctx.fillStyle = theme.bgDark;
    ctx.globalAlpha = 0.62;
    ctx.fillRect(BAR_INSET, 0, xOf(low - reach) - BAR_INSET, height);
    ctx.fillRect(xOf(high + reach), 0, width - BAR_INSET - xOf(high + reach), height);
    for (const [from, to] of [[low - reach, low], [high + reach, high]]) {
      if (Math.abs(xOf(to) - xOf(from)) < 1) continue;
      const shade = ctx.createLinearGradient(xOf(from), 0, xOf(to), 0);
      shade.addColorStop(0, "rgba(0,0,0,0.62)");
      shade.addColorStop(1, "rgba(0,0,0,0)");
      ctx.globalAlpha = 1;
      ctx.fillStyle = shade;
      ctx.fillRect(Math.min(xOf(from), xOf(to)), 0, Math.abs(xOf(to) - xOf(from)), height);
    }
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xOf(low) + 0.5, 1.5, Math.max(1, xOf(high) - xOf(low) - 1), height - 3);
    for (const edge of [low, high]) {
      const x = xOf(edge);
      ctx.fillStyle = theme.bgDark;
      ctx.fillRect(x - 3, 0, 6, height);
      ctx.fillStyle = bar?.handle === (edge === low ? "low" : "high") ? theme.fg : theme.accent;
      ctx.fillRect(x - 2, 0, 4, height);
    }
    ctx.restore();
    const frame = { width, height, top: 0, theme };
    drawLabel(ctx, frame, low.toFixed(2), xOf(low) + 5, height / 2);
    drawLabel(ctx, frame, high.toFixed(2), xOf(high) - 5, height / 2, { align: "right" });
  }

  /**
   * The lightness under a point on the bar.
   *
   * @param {PointerEvent} event - The event.
   * @returns {number} 0 to 1.
   */
  function barLightness(event) {
    const point = elementPoint(barCanvas, event);
    return clamp((point.x - BAR_INSET) / Math.max(1, barCanvas.clientWidth - BAR_INSET * 2), 0, 1);
  }

  barCanvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    context.focus();
    const values = context.values();
    const low = Math.min(values.luminance_low, values.luminance_high);
    const high = Math.max(values.luminance_low, values.luminance_high);
    const at = barLightness(event);
    const room = HIT / Math.max(1, barCanvas.clientWidth - BAR_INSET * 2);
    let handle = "move";
    if (Math.abs(at - low) <= room && Math.abs(at - low) <= Math.abs(at - high)) handle = "low";
    else if (Math.abs(at - high) <= room) handle = "high";
    else if (at < low || at > high) handle = Math.abs(at - low) < Math.abs(at - high) ? "low" : "high";
    bar = { handle, origin: at, low, high, pointerId: event.pointerId };
    try {
      barCanvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
    moveBar(event);
  });

  /**
   * Follow the pointer along the bar.
   *
   * @param {PointerEvent} event - The event.
   * @returns {void}
   */
  function moveBar(event) {
    if (!bar) return;
    const at = barLightness(event);
    let low = bar.low;
    let high = bar.high;
    if (bar.handle === "low") low = Math.min(at, high);
    else if (bar.handle === "high") high = Math.max(at, low);
    else {
      const shift = clamp(at - bar.origin, -low, 1 - high);
      low += shift;
      high += shift;
    }
    context.setDraft({ luminance_low: rounded(low), luminance_high: rounded(high) });
  }

  /**
   * End a drag on the bar.
   *
   * @param {PointerEvent} event - The event.
   * @param {boolean} keep - Whether what was dragged is written.
   * @returns {void}
   */
  function endBar(event, keep) {
    const ended = bar;
    bar = null;
    try {
      barCanvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
    }
    if (!ended) return;
    const values = keep ? context.draft() : null;
    context.clearDraft();
    if (values) context.write(values);
  }

  barCanvas.addEventListener("pointermove", (event) => {
    if (!bar) return;
    if (event.buttons === 0) {
      endBar(event, true);
      return;
    }
    moveBar(event);
  });
  barCanvas.addEventListener("pointerup", (event) => endBar(event, true));
  barCanvas.addEventListener("pointercancel", (event) => endBar(event, false));
  barCanvas.addEventListener("lostpointercapture", (event) => {
    if (bar) endBar(event, false);
  });

  return {
    tip(handle, values) {
      if (values.mode === "luminance") return "Click to set the band";
      return handle === null || handle === undefined ? "Click or drag; shift adds" : "Click to remove";
    },
    controls: [modes.element, clearButton.element],
    below,

    levels(width, height, values, source, index) {
      const pixels = labOf(index);
      if (!pixels || pixels.width !== width || pixels.height !== height) return null;
      const { lab } = pixels;
      const levels = new Float32Array(width * height);
      const share = clamp(values.smoothness, 0, 1);
      if (values.mode === "luminance") {
        const low = Math.min(values.luminance_low, values.luminance_high);
        const high = Math.max(values.luminance_low, values.luminance_high);
        const reach = share * LUMINANCE_FADE;
        for (let at = 0; at < width * height; at += 1) {
          const lightness = lab[at * 3];
          levels[at] = fade(Math.max(low - lightness, lightness - high, 0), 0, reach, "smooth");
        }
        return levels;
      }
      const { entries, readable } = samplesOf(values);
      if (!readable && !entries.length) return null;
      const targets = coloursOf(entries).filter((one) => one.light).map((one) => oklab(...one.light));
      if (!targets.length) return entries.length ? null : levels;
      const weight = clamp(values.lightness_weight, 0, 1);
      const full = values.range * (1 - share);
      for (let at = 0; at < width * height; at += 1) {
        let nearest = Infinity;
        for (const target of targets) {
          const dl = lab[at * 3] - target[0];
          const da = lab[at * 3 + 1] - target[1];
          const db = lab[at * 3 + 2] - target[2];
          nearest = Math.min(nearest, weight * dl * dl + da * da + db * db);
        }
        levels[at] = fade(Math.sqrt(nearest), full, values.range, "smooth");
      }
      return levels;
    },

    liveDetail() {
      return "worked out here, before refine";
    },

    paint(ctx, theme, values, hovered) {
      if (values.mode === "luminance") return;
      const { entries } = samplesOf(values);
      const shownFrame = context.batchIndex();
      const stage = context.stageSize();
      ctx.save();
      for (const marker of markers(entries)) {
        const elsewhere = marker.entry.frame !== shownFrame;
        const hot = hovered === marker.index;
        ctx.setLineDash(elsewhere ? [3, 3] : []);
        for (const [width, colour] of [[3.5, theme.bgDark], [1.5, hot ? theme.accent : theme.fg]]) {
          ctx.lineWidth = width;
          ctx.strokeStyle = colour;
          if (marker.box) {
            ctx.strokeRect(marker.box[0], marker.box[1], marker.box[2], marker.box[3]);
          } else {
            ctx.beginPath();
            ctx.arc(marker.x, marker.y, 6, 0, Math.PI * 2);
            ctx.moveTo(marker.x - 10, marker.y);
            ctx.lineTo(marker.x - 3, marker.y);
            ctx.moveTo(marker.x + 3, marker.y);
            ctx.lineTo(marker.x + 10, marker.y);
            ctx.moveTo(marker.x, marker.y - 10);
            ctx.lineTo(marker.x, marker.y - 3);
            ctx.moveTo(marker.x, marker.y + 3);
            ctx.lineTo(marker.x, marker.y + 10);
            ctx.stroke();
          }
          if (marker.box) continue;
        }
        const tag = `${marker.index + 1}${elsewhere ? ` f${marker.entry.frame}` : ""}`;
        const corner = marker.box ? { x: marker.box[0] + 4, y: marker.box[1] - 9 } : { x: marker.x + 9, y: marker.y - 11 };
        drawLabel(ctx, { width: stage.width, height: stage.height, top: 0, theme }, tag, corner.x, corner.y,
          { muted: elsewhere });
      }
      ctx.restore();
      if (context.pending()) {
        const box = context.pending();
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = theme.accent;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        ctx.restore();
      }
    },

    paintBelow(theme, values) {
      const luminance = values.mode === "luminance";
      modes.set(values.mode);
      clearButton.element.style.display = luminance ? "none" : "";
      swatches.style.display = luminance ? "none" : "flex";
      barWell.style.display = luminance ? "block" : "none";
      if (luminance) paintBar(theme, values);
      else paintSwatches(values);
    },

    hit(point, values) {
      if (values.mode === "luminance") return null;
      const index = markerAt(point, samplesOf(values).entries);
      return index >= 0 ? index : null;
    },

    cursor(handle) {
      return handle === null || handle === undefined ? "crosshair" : "pointer";
    },

    begin(point, event, values) {
      if (event.button !== 0 && event.button !== 2) return null;
      return { handle: "sample", shift: event.shiftKey, right: event.button === 2, values };
    },

    drag(gesture, point) {
      if (gesture.right) return;
      const origin = gesture.origin;
      const box = {
        x: Math.min(origin.x, point.x),
        y: Math.min(origin.y, point.y),
        width: Math.abs(point.x - origin.x),
        height: Math.abs(point.y - origin.y),
      };
      context.setPending(box);
      const start = context.share(origin);
      const end = context.share(point);
      if (gesture.values.mode === "luminance") {
        const span = lightnessAt(start, end);
        if (span) context.setDraft({ luminance_low: rounded(span[0]), luminance_high: rounded(span[1]) });
        return;
      }
      const { entries, readable } = samplesOf(gesture.values);
      if (!readable) return;
      const area = {
        kind: "area",
        x: clamp((start.u + end.u) / 2, 0, 1),
        y: clamp((start.v + end.v) / 2, 0, 1),
        w: clamp(Math.abs(end.u - start.u), 0, 1),
        h: clamp(Math.abs(end.v - start.v), 0, 1),
        frame: context.batchIndex(),
      };
      const kept = gesture.shift ? entries.slice(0, MAX_SAMPLES - 1) : [];
      context.setDraft({ samples: writeSamples([...kept, area]) });
    },

    finish(gesture, point) {
      context.setPending(null);
      if (gesture.moved && !gesture.right) return context.draft();
      const share = context.share(point);
      if (share.u < 0 || share.u > 1 || share.v < 0 || share.v > 1) return null;
      const values = gesture.values;
      if (values.mode === "luminance") {
        const span = lightnessAt(share);
        if (!span) return null;
        const low = Math.min(values.luminance_low, values.luminance_high);
        const high = Math.max(values.luminance_low, values.luminance_high);
        if (gesture.shift) {
          return { luminance_low: rounded(Math.min(low, span[0])), luminance_high: rounded(Math.max(high, span[1])) };
        }
        return {
          luminance_low: rounded(clamp(span[0] - LUMINANCE_BAND / 2, 0, 1)),
          luminance_high: rounded(clamp(span[1] + LUMINANCE_BAND / 2, 0, 1)),
        };
      }
      const { entries, readable } = samplesOf(values);
      if (!readable) return null;
      const hit = markerAt(point, entries);
      if (hit >= 0) return { samples: writeSamples(entries.filter((_, index) => index !== hit)) };
      if (gesture.right) return null;
      const sample = { kind: "point", x: clamp(share.u, 0, 1), y: clamp(share.v, 0, 1), w: 0, h: 0, frame: context.batchIndex() };
      const kept = gesture.shift ? entries.slice(0, MAX_SAMPLES - 1) : [];
      return { samples: writeSamples([...kept, sample]) };
    },

    chips(values) {
      if (values.mode === "luminance") return [];
      const { entries, readable } = samplesOf(values);
      const chips = [];
      if (!readable) chips.push(["samples partly unreadable", "warning"]);
      else if (!entries.length) chips.push(["no colour sampled", "warning"]);
      return chips;
    },

    readout(values) {
      if (values.mode === "luminance") {
        const low = Math.min(values.luminance_low, values.luminance_high);
        const high = Math.max(values.luminance_low, values.luminance_high);
        const reach = clamp(values.smoothness, 0, 1) * LUMINANCE_FADE;
        return `lightness ${low.toFixed(2)} to ${high.toFixed(2)}  ·  fades over ${reach.toFixed(3)}`;
      }
      const { entries } = samplesOf(values);
      const noun = entries.length === 1 ? "colour" : "colours";
      return `${entries.length} ${noun}  ·  range ${Number(values.range).toFixed(3)}  ·  smoothness ${Number(values.smoothness).toFixed(2)}`;
    },
  };
}

// Each node's title, its widgets and their defaults, and the tool drawn on its picture.
const SPECS = new Map([
  [GRADUATED_NODE, {
    title: "Graduated Mask",
    fields: { profile: "smooth", invert: false, combine: "replace", start_x: 0.5, start_y: 0.2, end_x: 0.5, end_y: 0.6 },
    tool: createGradientTool,
  }],
  [RADIAL_NODE, {
    title: "Radial Mask",
    fields: {
      feather: 50, profile: "smooth", invert: false, combine: "replace", centre_x: 0.5, centre_y: 0.5,
      radius_x: 0.3, radius_y: 0.3, angle: 0,
    },
    tool: createRadialTool,
  }],
  [COLOR_RANGE_NODE, {
    title: "Color Range Mask",
    fields: {
      mode: "color", range: 0.15, smoothness: 0.5, lightness_weight: 0.5, refine: 0.3, invert: false,
      combine: "replace", samples: "", luminance_low: 0.5, luminance_high: 1.0,
    },
    tool: createRangeTool,
  }],
]);

/**
 * Build the panel for one node.
 *
 * @param {object} node - A Graduated, Radial or Color Range mask node.
 * @param {object} spec - Its entry in `SPECS`.
 * @returns {object} The panel `appendInterfaceWidget` hosts.
 */
function createPanel(node, spec) {
  let disposed = false;
  let serial = 0;
  let frameHandle = 0;
  let pictureState = PREVIEW_STATE.WAITING;
  let promptId = "";
  let held = 0;
  let total = 0;
  let shown = [];
  let current = 0;
  let report = null;
  let view = VIEW.OVERLAY;
  let draft = null;
  let gesture = null;
  let hovered = null;
  let pending = null;
  let tint = null;
  let scrubbing = false;
  let overlay = { key: "", picture: null, claim: null };

  const frames = new Map();
  const outputs = new Map();
  const givens = new Map();
  const queue = [];
  let inflight = 0;

  const root = el("div", "position:relative;box-sizing:border-box;width:100%;height:100%;display:flex;"
    + "flex-direction:column;gap:6px;padding:7px 8px 8px;min-width:0;min-height:0;outline:none;"
    + `font:11px/1.35 ${SANS};user-select:none`);
  root.tabIndex = 0;

  const header = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0");
  const title = el("span", `flex:none;font:650 12px/1.4 ${SANS};color:${CARD_LOOK.title};white-space:nowrap`, spec.title);
  const chips = el("div", "flex:1 1 auto;display:flex;flex-wrap:wrap;gap:4px;min-width:0");
  header.append(title, chips);

  const toolbar = el("div", "flex:0 0 auto;display:flex;flex-wrap:wrap;gap:5px;align-items:center;min-width:0");
  const views = segment([
    { value: VIEW.PICTURE, label: "Picture", hint: "The picture alone" },
    { value: VIEW.OVERLAY, label: "Overlay", hint: "Mask tinted over the picture" },
    { value: VIEW.MASK, label: "Mask", hint: "Mask in grey" },
  ], (value) => {
    view = value;
    redraw();
  });
  const invertButton = pill("Invert", "Turn the mask over", () => write({ invert: !values().invert }));
  const combines = segment([
    { value: "replace", label: "Replace", hint: "This mask alone" },
    { value: "add", label: "Add", hint: "Larger of the two" },
    { value: "subtract", label: "Subtract", hint: "Wired mask minus this" },
    { value: "intersect", label: "Intersect", hint: "Wired mask times this" },
  ], (value) => write({ combine: value }));
  toolbar.append(views.element, invertButton.element, combines.element);

  const stage = el("div", "position:relative;flex:1 1 auto;min-height:0;min-width:0;border-radius:6px;"
    + `overflow:hidden;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:crosshair`);
  const canvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  stage.append(canvas);

  const frameRow = el("div", `flex:0 0 ${TRACK_HEIGHT}px;display:none;align-items:center;gap:4px;min-width:0`);
  const track = el("div", "position:relative;flex:1 1 auto;align-self:stretch;min-width:0;border-radius:6px;"
    + `overflow:hidden;background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};cursor:ew-resize`);
  const trackCanvas = el("canvas", "position:absolute;left:0;top:0;width:100%;height:100%;display:block");
  track.append(trackCanvas);
  const backButton = pill("‹", "Previous frame", () => showFrame(current - 1));
  const forwardButton = pill("›", "Next frame", () => showFrame(current + 1));
  const frameLabel = el("span", `flex:none;color:${CARD_LOOK.muted};font:600 10.5px/1 ${SANS};white-space:nowrap`);
  frameRow.append(backButton.element, track, forwardButton.element, frameLabel);

  const footer = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:6px;min-width:0");
  const glyph = el("canvas", `flex:none;width:${ICON_SIZE}px;height:${ICON_SIZE}px`);
  const status = el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${CARD_LOOK.muted}`);
  footer.append(glyph, status);

  const context = {
    rect: () => pictureRect(),
    stageSize: () => ({ width: canvas.clientWidth, height: canvas.clientHeight }),
    values: () => values(),
    draft: () => (draft ? { ...draft } : null),
    setDraft(next) {
      draft = { ...(draft ?? {}), ...next };
      redraw();
    },
    clearDraft() {
      draft = null;
      redraw();
    },
    pending: () => pending,
    setPending(box) {
      pending = box;
    },
    write: (next) => write(next),
    focus: () => root.focus({ preventScroll: true }),
    source: () => sourceSize(),
    current: () => current,
    batchIndex: () => batchIndex(current) ?? 0,
    storedIndex: (batch) => {
      const at = shown.indexOf(batch);
      return at >= 0 ? at : null;
    },
    pixels: (index) => pixelsOf(frames.get(index)),
    runSettings: () => parsed(fact("settings")),
    fact: (name) => fact(name),
    share(point) {
      const rect = pictureRect();
      return { u: (point.x - rect.x) / rect.width, v: (point.y - rect.y) / rect.height };
    },
    sized: (target) => sized(target),
  };
  const tool = spec.tool(context);
  if (tool.controls) toolbar.append(...tool.controls);

  root.append(header, toolbar, cardRule(), stage);
  if (tool.below) root.append(tool.below);
  root.append(frameRow, footer);

  /**
   * The values the node holds, with a drag in progress laid over them.
   *
   * @returns {object} Widget name to value.
   */
  function values() {
    const read = {};
    for (const [name, fallback] of Object.entries(spec.fields)) {
      const value = findWidget(node, name)?.value;
      if (typeof fallback === "number") read[name] = Number.isFinite(Number(value)) ? Number(value) : fallback;
      else if (typeof fallback === "boolean") read[name] = typeof value === "boolean" ? value : fallback;
      else read[name] = value === undefined || value === null ? fallback : String(value);
    }
    return draft ? { ...read, ...draft } : read;
  }

  /**
   * Write widgets as one undoable edit.
   *
   * @param {Object<string, *>} next - Widget name to value.
   * @returns {boolean} True when any widget took a new value.
   */
  function write(next) {
    const writes = [];
    for (const [name, raw] of Object.entries(next ?? {})) {
      const widget = findWidget(node, name);
      if (!widget) continue;
      let value = raw;
      if (typeof raw === "number") {
        const low = Number(widget.options?.min);
        const high = Number(widget.options?.max);
        if (Number.isFinite(low)) value = Math.max(low, value);
        if (Number.isFinite(high)) value = Math.min(high, value);
      }
      if (widget.value !== value) writes.push([widget, value]);
    }
    if (!writes.length) {
      redraw();
      return false;
    }
    withGraphChange(() => {
      for (const [widget, value] of writes) {
        widget.value = value;
        widget.callback?.(value);
      }
    });
    node.setDirtyCanvas?.(true, true);
    redraw();
    return true;
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
   * A count the last run reported.
   *
   * @param {string} name - The count's name.
   * @returns {number} Its value, or NaN.
   */
  function count(name) {
    return Number(report?.counts.get(name));
  }

  /**
   * The size of the pictures the node read.
   *
   * @returns {{width: number, height: number}|null} Pixels, or null before a run.
   */
  function sourceSize() {
    const width = count("width");
    const height = count("height");
    if (width > 0 && height > 0) return { width, height };
    const image = frames.get(current)?.image;
    return image ? { width: image.naturalWidth, height: image.naturalHeight } : null;
  }

  /**
   * The batch frame a stored frame stands for.
   *
   * @param {number} index - A stored frame.
   * @returns {number|null} Its batch index, or null where nothing is stored there.
   */
  function batchIndex(index) {
    return index >= 0 && index < shown.length ? shown[index] : null;
  }

  /**
   * Whether a mask is wired into the node.
   *
   * @returns {boolean} True while the `mask` input has a link.
   */
  function maskWired() {
    const input = (node.inputs ?? []).find((slot) => slot?.name === MASK_INPUT);
    return input?.link !== null && input?.link !== undefined;
  }

  /**
   * Whether the node holds the values its last run used, with no drag in progress.
   *
   * @returns {boolean} True when the run's own mask stands for the settings.
   */
  function runMatches() {
    if (draft) return false;
    const settings = parsed(fact("settings"));
    if (!settings || typeof settings !== "object") return false;
    const now = values();
    for (const name of Object.keys(spec.fields)) {
      if (name === "samples") {
        if (now.mode === "luminance" && settings.mode === "luminance") continue;
        const read = readSamples(now.samples);
        if (!read.readable || !same(sampleKeys(read.entries), settings.samples ?? [])) return false;
        continue;
      }
      if (!same(now[name], settings[name])) return false;
    }
    return (fact("mask_wired") === "yes") === maskWired();
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
   * Size a canvas to its element at the graph's zoom.
   *
   * @param {HTMLCanvasElement} target - The canvas.
   * @returns {{ctx: CanvasRenderingContext2D, width: number, height: number}|null} Its context,
   *   scaled to layout pixels, and its size in them.
   */
  function sized(target) {
    const width = target.clientWidth;
    const height = target.clientHeight;
    if (!(width > 0) || !(height > 0)) return null;
    const ratio = surfaceRatio(target);
    const backingWidth = Math.max(1, Math.round(width * ratio));
    const backingHeight = Math.max(1, Math.round(height * ratio));
    if (target.width !== backingWidth) target.width = backingWidth;
    if (target.height !== backingHeight) target.height = backingHeight;
    const ctx = target.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }

  /**
   * Where the shown frame sits inside the well, at the shape of the pictures the node read.
   *
   * @returns {{x: number, y: number, width: number, height: number}} The rectangle, in layout pixels.
   */
  function pictureRect() {
    const source = sourceSize() ?? { width: 16, height: 9 };
    const width = Math.max(1, canvas.clientWidth - MARGIN * 2);
    const height = Math.max(1, canvas.clientHeight - MARGIN * 2);
    const fit = Math.min(width / source.width, height / source.height);
    const drawWidth = source.width * fit;
    const drawHeight = source.height * fit;
    return { x: MARGIN + (width - drawWidth) / 2, y: MARGIN + (height - drawHeight) / 2, width: drawWidth, height: drawHeight };
  }

  /**
   * The pixels of a stored picture, read once.
   *
   * @param {{image: HTMLImageElement|null}|undefined} entry - A stored frame or mask.
   * @returns {{width: number, height: number, data: Uint8ClampedArray}|null} Its pixels.
   */
  function pixelsOf(entry) {
    const image = entry?.image;
    if (!image) return null;
    if (entry.pixels) return entry.pixels;
    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const probe = document.createElement("canvas");
    probe.width = width;
    probe.height = height;
    const ctx = probe.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0);
    entry.pixels = { width, height, data: ctx.getImageData(0, 0, width, height).data };
    return entry.pixels;
  }

  /**
   * A stored mask's levels at a given size.
   *
   * @param {{image: HTMLImageElement|null}|undefined} entry - A stored mask.
   * @param {number} width - Pixels across wanted.
   * @param {number} height - Pixels down wanted.
   * @returns {Float32Array|null} One level per pixel, or null while it has not arrived.
   */
  function levelsOf(entry, width, height) {
    const pixels = pixelsOf(entry);
    if (!pixels) return null;
    const levels = new Float32Array(width * height);
    for (let y = 0; y < height; y += 1) {
      const row = Math.min(pixels.height - 1, Math.floor(((y + 0.5) / height) * pixels.height));
      for (let x = 0; x < width; x += 1) {
        const column = Math.min(pixels.width - 1, Math.floor(((x + 0.5) / width) * pixels.width));
        levels[y * width + x] = pixels.data[(row * pixels.width + column) * 4] / 255;
      }
    }
    return levels;
  }

  /**
   * The mask for the shown frame as the settings on the node would make it, worked out here.
   *
   * @param {object} now - The node's values.
   * @param {number} width - The stored frame's width.
   * @param {number} height - Its height.
   * @returns {{levels: Float32Array, detail: string}|null} The levels and what they leave out, or
   *   null where they cannot be worked out on the panel.
   */
  function liveLevels(now, width, height) {
    const source = sourceSize();
    if (!source) return null;
    const levels = tool.levels(width, height, now, source, current);
    if (!levels) return null;
    if (now.invert) for (let at = 0; at < levels.length; at += 1) levels[at] = 1 - levels[at];
    let detail = tool.liveDetail?.(now) ?? "worked out here at panel size";
    if (maskWired() && now.combine !== "replace") {
      const given = givens.get(current)?.image ? levelsOf(givens.get(current), width, height) : null;
      if (given) {
        for (let at = 0; at < levels.length; at += 1) {
          const made = levels[at];
          const wired = given[at];
          levels[at] = now.combine === "add" ? Math.max(wired, made)
            : now.combine === "subtract" ? wired * (1 - made) : wired * made;
        }
      } else {
        detail = "wired mask joins after a run";
      }
    }
    return { levels, detail };
  }

  /**
   * The mask picture to draw over the shown frame, rebuilt only when what it stands for changes.
   *
   * @param {object} now - The node's values.
   * @param {object} theme - The palette.
   * @returns {{picture: HTMLCanvasElement|null, claim: {icon: string, detail: string}}} The picture
   *   and what it claims.
   */
  function maskLayer(now, theme) {
    const frame = frames.get(current)?.image;
    if (!frame || view === VIEW.PICTURE) {
      return { picture: null, claim: { icon: ICON.EXACT, detail: "picture the node read" } };
    }
    tint ??= channels(theme.error);
    const plain = view === VIEW.MASK;
    const width = frame.naturalWidth;
    const height = frame.naturalHeight;
    const own = runMatches() && outputs.get(current)?.image;
    const key = JSON.stringify([own ? "run" : "live", current, view, own ? promptId : now, tint, maskWired(),
      !!givens.get(current)?.image]);
    if (key === overlay.key) return overlay;
    if (own) {
      const levels = levelsOf(outputs.get(current), width, height);
      overlay = {
        key,
        picture: levels ? maskPicture(levels, width, height, tint, plain) : null,
        claim: { icon: ICON.EXACT, detail: "mask from the last run" },
      };
      return overlay;
    }
    const live = liveLevels(now, width, height);
    overlay = live
      ? { key, picture: maskPicture(live.levels, width, height, tint, plain), claim: { icon: ICON.APPROXIMATE, detail: live.detail } }
      : { key, picture: null, claim: { icon: ICON.WARNING, detail: "mask shows after a run" } };
    return overlay;
  }

  /**
   * Draw the picture well.
   *
   * @param {object} theme - The palette.
   * @param {object} now - The node's values.
   * @returns {{icon: string, detail: string}} What the well claims about its fidelity.
   */
  function paintStage(theme, now) {
    const surface = sized(canvas);
    if (!surface) return { icon: ICON.WARNING, detail: standInDetail(pictureState) };
    const { ctx, width, height } = surface;
    const rect = pictureRect();
    const image = frames.get(current)?.image;
    if (!image) {
      const state = frames.get(current)?.state ?? pictureState;
      drawStandIn(ctx, { x: rect.x, y: rect.y, w: rect.width, h: rect.height });
      const words = state === PREVIEW_STATE.WAITING ? "Run once to load picture" : LABELS[state] ?? "";
      if (words) {
        ctx.fillStyle = theme.fgMuted;
        ctx.font = `600 12px ${SANS}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(words, rect.x + rect.width / 2, rect.y + rect.height - 6, Math.max(10, rect.width - 12));
      }
      return { icon: ICON.WARNING, detail: standInDetail(state) };
    }
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, rect.x, rect.y, rect.width, rect.height);
    const layer = maskLayer(now, theme);
    if (layer.picture) {
      ctx.save();
      ctx.imageSmoothingEnabled = view !== VIEW.MASK;
      ctx.drawImage(layer.picture, rect.x, rect.y, rect.width, rect.height);
      ctx.restore();
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, width, height);
    ctx.clip();
    try {
      tool.paint(ctx, theme, now, hovered);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to draw the handles:`, error);
    }
    ctx.restore();
    return layer.claim;
  }

  /**
   * Draw the frame track, with a tick per stored frame and the shown one marked.
   *
   * @param {object} theme - The palette.
   * @returns {void}
   */
  function paintTrack(theme) {
    const many = held > 1;
    frameRow.style.display = many ? "flex" : "none";
    if (!many) return;
    const batch = batchIndex(current);
    const frameCount = Number.isFinite(count("frames")) ? count("frames") : total;
    frameLabel.textContent = batch === null ? "" : `frame ${batch} of ${frameCount}`;
    const surface = sized(trackCanvas);
    if (!surface) return;
    const { ctx, width, height } = surface;
    const slice = width / held;
    ctx.fillStyle = theme.fgMuted;
    for (let index = 0; index < held; index += 1) {
      ctx.fillRect(Math.round((index + 0.5) * slice), height / 2 - 3, 1, 6);
    }
    ctx.fillStyle = theme.accent;
    ctx.fillRect(Math.round((current + 0.5) * slice) - 1.5, 3, 3, height - 6);
  }

  /**
   * Fill the header's chips.
   *
   * @param {object} now - The node's values.
   * @returns {void}
   */
  function paintChips(now) {
    chips.replaceChildren();
    const coverage = count("coverage");
    if (report && Number.isFinite(coverage)) {
      const tone = report.status === "warning" ? "warning" : report.status === "error" ? "error" : "success";
      chips.append(cardChip(`${coverage.toFixed(1)}% covered`, tone));
    }
    if (report && !runMatches()) chips.append(cardChip("changed since the run", "accent"));
    if (now.combine !== "replace" && !maskWired()) chips.append(cardChip("no mask wired", "warning"));
    for (const [text, tone] of tool.chips?.(now) ?? []) chips.append(cardChip(text, tone));
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    if (disposed) return;
    const theme = readTheme();
    const now = values();
    views.set(view);
    combines.set(now.combine);
    combines.element.style.opacity = maskWired() || now.combine !== "replace" ? "1" : "0.6";
    combines.element.title = maskWired() ? "" : "Needs a wired mask";
    invertButton.setOn(!!now.invert);
    if (!gesture) canvas.title = frames.get(current)?.image ? tool.tip(hovered, now) : "";
    const claim = paintStage(theme, now);
    tool.paintBelow?.(theme, now);
    paintTrack(theme);
    paintChips(now);
    drawFidelityGlyph(glyph, claim);
    const words = tool.readout(now, sourceSize());
    const summary = report?.summary && runMatches() ? report.summary : "";
    status.textContent = [words, summary].filter(Boolean).join("  ·  ");
    status.title = status.textContent;
  }

  /**
   * Ask for a stored frame and its masks, ahead of anything already waiting.
   *
   * @param {number} index - A stored frame.
   * @returns {void}
   */
  function wantNow(index) {
    queue.unshift({ store: outputs, side: "output", index }, { store: givens, side: "given", index },
      { store: frames, side: "input", index });
    pump();
  }

  /**
   * Start waiting requests, up to `CONCURRENCY` at once.
   *
   * @returns {void}
   */
  function pump() {
    while (!disposed && inflight < CONCURRENCY && queue.length) {
      const job = queue.shift();
      if (job.store.has(job.index) || job.index < 0 || job.index >= held) continue;
      if (job.side === "given" && fact("mask_wired") !== "yes") continue;
      job.store.set(job.index, { state: PREVIEW_STATE.LOADING, image: null });
      inflight += 1;
      const ticket = serial;
      const ask = job.side === "input" ? fetchInputPreview(node, FRAMES_SLOT, job.index, promptId)
        : job.side === "given" ? fetchInputPreview(node, MASK_SLOT, job.index, promptId)
          : fetchOutputPreview(node, MASK_SLOT, job.index, promptId);
      ask.then((answer) => {
        if (ticket !== serial || disposed) return;
        const ready = answer?.state === PREVIEW_STATE.READY;
        job.store.set(job.index, { state: answer?.state ?? PREVIEW_STATE.FAILED, image: ready ? answer.image : null });
        if (job.index === current) overlay = { key: "", picture: null, claim: null };
        redraw();
      }).catch((error) => {
        console.error(`[${EXT_NAME}] Failed to read a frame:`, error);
        if (ticket === serial) job.store.set(job.index, { state: PREVIEW_STATE.FAILED, image: null });
      }).finally(() => {
        if (ticket === serial) inflight -= 1;
        pump();
      });
    }
  }

  /**
   * Read the last run's report and its first frame, then queue the rest.
   *
   * @returns {Promise<void>} Settles once the first answers are in.
   */
  async function load() {
    const ticket = ++serial;
    inflight = 0;
    queue.length = 0;
    pictureState = PREVIEW_STATE.LOADING;
    redraw();
    try {
      const [run, first] = await Promise.all([fetchRunResult(node), fetchInputPreview(node, FRAMES_SLOT, 0)]);
      if (ticket !== serial || disposed) return;
      frames.clear();
      outputs.clear();
      givens.clear();
      overlay = { key: "", picture: null, claim: null };
      report = run?.state === PREVIEW_STATE.READY && run.result ? {
        status: run.result.status,
        summary: run.result.summary ?? "",
        facts: new Map((run.result.facts ?? []).map((row) => [row.name, row.value])),
        counts: new Map((run.result.counts ?? []).map((row) => [row.name, row.value])),
      } : null;
      pictureState = first?.state ?? PREVIEW_STATE.FAILED;
      if (first?.state !== PREVIEW_STATE.READY) {
        held = 0;
        total = 0;
        shown = [];
        redraw();
        return;
      }
      promptId = first.promptId ?? "";
      held = Math.max(1, Number(first.frameCount) || 1);
      total = Math.max(held, Number(first.frameTotal) || held);
      const listed = parsed(fact("shown"));
      shown = Array.isArray(listed) && listed.length >= held && listed.slice(0, held).every(Number.isInteger)
        ? listed.slice(0, held)
        : Array.from({ length: held }, (_, index) => index);
      frames.set(0, { state: PREVIEW_STATE.READY, image: first.image });
      current = clamp(current, 0, held - 1);
      wantNow(current);
      for (let index = 0; index < held; index += 1) {
        queue.push({ store: frames, side: "input", index }, { store: outputs, side: "output", index },
          { store: givens, side: "given", index });
      }
      pump();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the last run:`, error);
      pictureState = PREVIEW_STATE.FAILED;
    }
    redraw();
  }

  /**
   * Show another stored frame.
   *
   * @param {number} index - The stored frame wanted.
   * @returns {boolean} True when the shown frame moved.
   */
  function showFrame(index) {
    if (!held) return false;
    const next = clamp(Math.round(index), 0, held - 1);
    if (next === current) return false;
    current = next;
    overlay = { key: "", picture: null, claim: null };
    wantNow(current);
    redraw();
    return true;
  }

  /**
   * End a gesture on the picture.
   *
   * @param {PointerEvent|null} event - The event that ended it.
   * @param {boolean} keep - Whether what was dragged is written.
   * @returns {void}
   */
  function endGesture(event, keep) {
    const ended = gesture;
    gesture = null;
    if (event) {
      try {
        canvas.releasePointerCapture?.(event.pointerId);
      } catch (error) {
        console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
      }
    }
    if (!ended) return;
    let next = null;
    if (keep) {
      try {
        next = tool.finish(ended, event ? elementPoint(canvas, event) : ended.last, event ?? {});
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to finish the gesture:`, error);
      }
    }
    draft = null;
    pending = null;
    if (next) write(next);
    else redraw();
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    event.stopPropagation();
    root.focus({ preventScroll: true });
    if (!frames.get(current)?.image || gesture) return;
    const point = elementPoint(canvas, event);
    const begun = tool.begin(point, event, values());
    if (!begun) return;
    gesture = { ...begun, origin: point, last: point, moved: false };
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
  });
  canvas.addEventListener("pointermove", (event) => {
    const point = elementPoint(canvas, event);
    if (!gesture) {
      const ready = !!frames.get(current)?.image;
      const now = values();
      const next = ready ? tool.hit(point, now) : null;
      canvas.title = ready ? tool.tip(next, now) : "";
      if (next !== hovered) {
        hovered = next;
        canvas.style.cursor = ready ? tool.cursor(hovered) : "default";
        redraw();
      }
      return;
    }
    if (event.buttons === 0) {
      endGesture(event, true);
      return;
    }
    gesture.last = point;
    if (!gesture.moved && Math.hypot(point.x - gesture.origin.x, point.y - gesture.origin.y) < CLICK_SLOP) return;
    gesture.moved = true;
    tool.drag(gesture, point, event);
    redraw();
  });
  canvas.addEventListener("pointerup", (event) => endGesture(event, true));
  canvas.addEventListener("pointercancel", (event) => endGesture(event, false));
  canvas.addEventListener("lostpointercapture", () => {
    if (gesture) endGesture(null, false);
  });
  canvas.addEventListener("pointerleave", () => {
    if (hovered === null || gesture) return;
    hovered = null;
    redraw();
  });

  /**
   * The stored frame under a pointer on the track.
   *
   * @param {PointerEvent} event - The event.
   * @returns {number} The stored frame.
   */
  function trackIndex(event) {
    const point = elementPoint(trackCanvas, event);
    return Math.floor(clamp(point.x / (trackCanvas.clientWidth || 1), 0, 0.9999) * Math.max(1, held));
  }

  trackCanvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    root.focus({ preventScroll: true });
    scrubbing = true;
    try {
      trackCanvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer capture refused:`, error);
    }
    showFrame(trackIndex(event));
  });
  trackCanvas.addEventListener("pointermove", (event) => {
    if (!scrubbing) return;
    if (event.buttons === 0) {
      scrubbing = false;
      return;
    }
    showFrame(trackIndex(event));
  });
  const endScrub = (event) => {
    scrubbing = false;
    try {
      trackCanvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${EXT_NAME}] Pointer release refused:`, error);
    }
  };
  trackCanvas.addEventListener("pointerup", endScrub);
  trackCanvas.addEventListener("pointercancel", endScrub);
  trackCanvas.addEventListener("lostpointercapture", () => { scrubbing = false; });

  root.addEventListener("contextmenu", (event) => event.preventDefault());
  root.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && gesture) {
      event.preventDefault();
      event.stopPropagation();
      endGesture(null, false);
      return;
    }
    const steps = { ArrowLeft: -1, ArrowRight: 1 };
    if (held > 1 && event.key in steps) {
      event.preventDefault();
      event.stopPropagation();
      showFrame(current + steps[event.key]);
    }
  });

  const releaseWheel = captureWheel(root, (event) => {
    if (!frameRow.contains(event.target) || !event.deltaY) return false;
    return showFrame(current + (event.deltaY > 0 ? 1 : -1));
  });

  const resizeWatch = new ResizeObserver(() => redraw());
  resizeWatch.observe(root);
  const stopStageRatio = watchSurfaceRatio(canvas, () => redraw());
  const stopTrackRatio = watchSurfaceRatio(trackCanvas, () => redraw());
  const stopTheme = onThemeChange(() => {
    tint = null;
    overlay = { key: "", picture: null, claim: null };
    redraw();
  });
  const stopFinished = onNodeFinished(node, () => load());
  const stopReadable = whenReadable(node, () => load());
  for (const name of Object.keys(spec.fields)) chainWidgetCallback(node, name, () => redraw(), EXT_NAME);
  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    redraw();
    return result;
  };
  loadPlaceholder().then(() => redraw());

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: () => load(),
    dispose() {
      disposed = true;
      serial += 1;
      queue.length = 0;
      gesture = null;
      if (frameHandle) cancelAnimationFrame(frameHandle);
      frameHandle = 0;
      resizeWatch.disconnect();
      stopStageRatio?.();
      stopTrackRatio?.();
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
    const id = nodeData?.name;
    const spec = SPECS.get(id);
    if (!spec) return;
    const proto = nodeType.prototype;
    if (proto.__was_local_mask_wrapped) return;
    proto.__was_local_mask_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createPanel(this, spec);
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
