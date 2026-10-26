/**
 * The Bounding Box Editor's layout in the browser, mirroring `modules/image/box_layout.py`:
 * reading the layout text, refitting it to a frame, and writing it.
 *
 * Coordinates are whole pixels from the frame's top left. No DOM.
 */

import { roundHalfEven } from "./python_arithmetic.js";

/** Box kinds, in menu order. */
export const KINDS = ["object", "character", "text"];

/** What happens to a layout drawn at another frame size. */
export const RESIZE_MODES = ["fit", "fill", "stretch", "keep size"];

/** Display colours handed to boxes in turn. */
export const PALETTE = [
  "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
  "#D55E00", "#CC79A7", "#A6D854", "#E78AC3", "#B3B3B3",
];

/** Most boxes one layout holds. */
export const MAX_BOXES = 64;

/** Longest layout text read, in characters. */
export const MAX_TEXT = 262144;

/** Longest caption or text of one box, in characters. */
export const MAX_CAPTION = 2000;

/** Most colours one box asks of the model. */
export const MAX_PALETTE = 5;

/** Largest coordinate read, either side of zero. */
export const MAX_COORDINATE = 65536;

/** Largest frame side. */
export const MAX_EDGE = 16384;

/** Ratio locks a box may carry, as `[value, label]`. */
export const RATIOS = [
  ["", "Free"], ["frame", "Frame"], ["1:1", "1:1"], ["4:3", "4:3"], ["3:4", "3:4"],
  ["3:2", "3:2"], ["2:3", "2:3"], ["16:9", "16:9"], ["9:16", "9:16"], ["21:9", "21:9"],
];

const HEX = /^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i;

/**
 * A colour as lowercase six digit hex.
 *
 * @param {*} value - A colour as written.
 * @returns {string|null} `#rrggbb`, or null when it is not a hex colour.
 */
export function normaliseColour(value) {
  if (typeof value !== "string") return null;
  const text = value.trim().toLowerCase();
  if (!HEX.test(text)) return null;
  if (text.length === 4) return `#${text[1]}${text[1]}${text[2]}${text[2]}${text[3]}${text[3]}`;
  return text.slice(0, 7);
}

/**
 * A ratio lock as a number, width over height.
 *
 * @param {string} ratio - A value from `RATIOS`.
 * @param {number[]} frame - `[width, height]`, for `frame`.
 * @returns {number|null} The ratio, or null for a free box.
 */
export function ratioValue(ratio, frame) {
  if (ratio === "frame") return frame[0] / frame[1];
  const match = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(String(ratio ?? ""));
  if (!match) return null;
  const value = Number(match[1]) / Number(match[2]);
  return value > 0 && Number.isFinite(value) ? value : null;
}

const PYTHON_FLOAT = /^\s*[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?\s*$/i;

/**
 * Whether a value is a plain object.
 *
 * @param {*} value - The value.
 * @returns {boolean} True for an object that is not a list.
 */
function isRecord(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

/**
 * Read one coordinate as python's `float` and `round` would.
 *
 * @param {*} value - The number as written.
 * @param {string} what - Its name, for the message.
 * @returns {number} The value rounded to a whole pixel.
 */
function coordinate(value, what) {
  if (value === undefined || value === null) throw new Error(`a box's ${what} is missing`);
  let number = Number.NaN;
  if (typeof value === "number") number = value;
  else if (typeof value === "string" && PYTHON_FLOAT.test(value)) number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`a box's ${what} is not a number`);
  if (Math.abs(number) > MAX_COORDINATE) throw new Error(`a box's ${what} is past ${MAX_COORDINATE}`);
  return roundHalfEven(number);
}

/**
 * Text as python's `str` gives it for a caption.
 *
 * @param {*} value - The value as written.
 * @param {string} what - Its name, for the message.
 * @returns {string} The text, held to `MAX_CAPTION`.
 */
function words(value, what) {
  let text = "";
  if (typeof value === "string") text = value;
  else if (typeof value === "boolean") text = value ? "True" : "False";
  else if (typeof value === "number") text = String(value);
  if (text.length > MAX_CAPTION) throw new Error(`a box's ${what} is longer than ${MAX_CAPTION} characters`);
  return text;
}

/**
 * The colours of a palette, at most `MAX_PALETTE`.
 *
 * @param {*} value - A list of colours, an object whose values are colours, or one colour.
 * @returns {string[]} The valid colours in order.
 */
export function paletteOf(value) {
  let list = [];
  if (typeof value === "string") list = [value];
  else if (Array.isArray(value)) list = value;
  else if (isRecord(value)) list = Object.values(value);
  return list.map(normaliseColour).filter(Boolean).slice(0, MAX_PALETTE);
}

/**
 * A core bounding box carrying metadata, or a detector's label, in this editor's shape.
 *
 * @param {object} raw - The box as written.
 * @returns {object} The box with its metadata read into it.
 */
function fromMetadata(raw) {
  const label = typeof raw.label === "string" ? raw.label : "";
  if (!isRecord(raw.metadata)) return label && !("caption" in raw) ? { ...raw, caption: label } : raw;
  const meta = raw.metadata;
  const desc = meta.desc;
  return {
    x: raw.x,
    y: raw.y,
    width: raw.width,
    height: raw.height,
    kind: KINDS.includes(meta.kind) ? meta.kind : (meta.type === "text" ? "text" : "object"),
    caption: typeof desc === "string" && desc ? desc : label,
    text: meta.text ?? "",
    palette: meta.palette ?? [],
    color: meta.color,
  };
}

/**
 * Read one box, in this editor's shape or a core one.
 *
 * @param {object} entry - The box as written.
 * @param {number} index - Its place, for its default colour.
 * @returns {object} The box with every key present.
 */
function readBox(entry, index) {
  const raw = fromMetadata(entry);
  const x = coordinate(raw.x, "x");
  const y = coordinate(raw.y, "y");
  const width = coordinate(raw.width, "width");
  const height = coordinate(raw.height, "height");
  if (width < 1 || height < 1) throw new Error("a box is narrower or shorter than one pixel");
  return {
    kind: KINDS.includes(raw.kind) ? raw.kind : "object",
    x,
    y,
    width,
    height,
    caption: words(raw.caption, "caption"),
    text: words(raw.text, "text"),
    palette: paletteOf(raw.palette),
    color: normaliseColour(raw.color) ?? PALETTE[index % PALETTE.length],
    ratio: typeof raw.ratio === "string" ? raw.ratio : "",
    locked: raw.locked === true,
    hidden: raw.hidden === true,
  };
}

/**
 * Whether an entry is a prompt element, holding a four-value `bbox`.
 *
 * @param {*} value - The entry.
 * @returns {boolean} True for an element.
 */
function isElement(value) {
  return isRecord(value) && Array.isArray(value.bbox) && value.bbox.length === 4;
}

/**
 * Read a prompt element, `bbox` on a 0 to 1000 grid as `[ymin, xmin, ymax, xmax]`, as a box.
 *
 * @param {object} raw - The element.
 * @param {number[]} frame - `[width, height]` the grid is laid over.
 * @returns {object} The element as a box, before it is read.
 */
function elementBox(raw, frame) {
  if (!isElement(raw)) throw new Error("an element has no bbox of four numbers");
  const grid = raw.bbox.map((value) => (typeof value === "number" ? value
    : typeof value === "string" && PYTHON_FLOAT.test(value) ? Number(value) : Number.NaN) / 1000);
  if (!grid.every(Number.isFinite)) throw new Error("an element's bbox holds a value that is not a number");
  const [ymin, xmin, ymax, xmax] = grid;
  const kind = raw.type === "text" ? "text" : "object";
  return {
    x: coordinate(Math.min(xmin, xmax) * frame[0], "x"),
    y: coordinate(Math.min(ymin, ymax) * frame[1], "y"),
    width: coordinate(Math.abs(xmax - xmin) * frame[0], "width"),
    height: coordinate(Math.abs(ymax - ymin) * frame[1], "height"),
    kind,
    caption: raw.desc ?? "",
    text: kind === "text" ? raw.text ?? "" : "",
    palette: raw.color_palette || [],
  };
}

/**
 * Read a frame.
 *
 * @param {*} value - `[width, height]` as written.
 * @returns {number[]} The frame.
 */
function frameOf(value) {
  if (!Array.isArray(value) || value.length !== 2) throw new Error("frame is not [width, height]");
  if (!value.every((side) => typeof side === "number" && Number.isFinite(side))) {
    throw new Error("frame holds a side that is not a number");
  }
  const frame = value.map((side) => roundHalfEven(side));
  if (!frame.every((side) => side >= 1 && side <= MAX_EDGE)) throw new Error(`frame holds a side outside 1 to ${MAX_EDGE}`);
  return frame;
}

/**
 * Read the layout text.
 *
 * @param {string} text - What the layout widget holds.
 * @param {number[]} frame - `[width, height]` of the run, for a layout naming no frame of its own.
 * @returns {{ok: boolean, frame: number[], boxes: object[], error: string}} The layout. `ok` is
 *   false when the text cannot be read, and `error` says why.
 */
export function parseLayout(text, frame) {
  const body = String(text ?? "");
  if (!body.trim()) return { ok: true, frame: [...frame], boxes: [], error: "" };
  try {
    if (body.length > MAX_TEXT) throw new Error(`the layout is longer than ${MAX_TEXT} characters`);
    let data = JSON.parse(body);
    let drawnAt = null;
    if (isRecord(data)) {
      if ("boxes" in data) {
        if (data.frame !== undefined && data.frame !== null) drawnAt = frameOf(data.frame);
        data = data.boxes ?? [];
        if (!Array.isArray(data)) throw new Error("the layout's boxes is not a list");
      } else if (isElement(data) || ["x", "y", "width", "height"].every((key) => key in data)) {
        data = [data];
      } else {
        throw new Error("the layout holds an object with no boxes");
      }
    }
    if (!Array.isArray(data)) throw new Error("the layout holds no list of boxes");
    // A detector's per-frame lists: the first frame is read.
    if (data.length && Array.isArray(data[0])) data = data[0];
    let entries = data.filter(isRecord);
    if (entries.length > MAX_BOXES) throw new Error(`the layout holds more than ${MAX_BOXES} boxes`);
    const placedIn = drawnAt ?? [...frame];
    if (entries.length && isElement(entries[0])) entries = entries.map((entry) => elementBox(entry, placedIn));
    const boxes = entries.map((entry, index) => readBox(entry, index));
    return { ok: true, frame: placedIn, boxes, error: "" };
  } catch (error) {
    return { ok: false, frame: [...frame], boxes: [], error: String(error?.message ?? error) };
  }
}

/**
 * Write a layout as the text the layout widget holds, one box per line.
 *
 * @param {number[]} frame - `[width, height]` the boxes are measured in.
 * @param {object[]} boxes - Boxes in this editor's shape.
 * @returns {string} The layout text.
 */
export function writeLayout(frame, boxes) {
  if (!boxes.length) return "";
  const lines = boxes.map((box) => {
    const written = { kind: box.kind, x: box.x, y: box.y, width: box.width, height: box.height };
    if (box.caption) written.caption = box.caption;
    if (box.kind === "text" && box.text) written.text = box.text;
    if (box.palette?.length) written.palette = box.palette;
    written.color = box.color;
    if (box.ratio) written.ratio = box.ratio;
    if (box.locked) written.locked = true;
    if (box.hidden) written.hidden = true;
    return `  ${JSON.stringify(written)}`;
  });
  return `{"version": 1, "frame": [${frame[0]}, ${frame[1]}], "boxes": [\n${lines.join(",\n")}\n]}`;
}

/**
 * How a layout drawn at one frame lands on another.
 *
 * @param {number[]} from - The frame the layout was drawn at.
 * @param {number[]} to - The frame of the run.
 * @param {string} mode - A value of `RESIZE_MODES`.
 * @returns {{same: boolean, sx: number, sy: number, ox: number, oy: number}} The scale and offset.
 */
export function mapping(from, to, mode) {
  const [fw, fh] = from;
  const [W, H] = to;
  if (fw === W && fh === H) return { same: true, sx: 1, sy: 1, ox: 0, oy: 0 };
  if (mode === "stretch") return { same: false, sx: W / fw, sy: H / fh, ox: 0, oy: 0 };
  if (mode === "keep size") return { same: false, sx: 1, sy: 1, ox: (W - fw) / 2, oy: (H - fh) / 2 };
  const s = mode === "fill" ? Math.max(W / fw, H / fh) : Math.min(W / fw, H / fh);
  return { same: false, sx: s, sy: s, ox: (W - fw * s) / 2, oy: (H - fh * s) / 2 };
}

/**
 * One box placed on the run's frame, as the node places it.
 *
 * @param {object} box - The box in the layout's frame.
 * @param {object} map - What `mapping` answered.
 * @param {number[]} to - The run's frame.
 * @param {number} grid - Pixels a refitted edge snaps to.
 * @returns {object|null} The placed box, or null when nothing of it is left inside the frame.
 */
export function placed(box, map, to, grid) {
  let x0 = box.x;
  let y0 = box.y;
  let x1 = box.x + box.width;
  let y1 = box.y + box.height;
  if (!map.same) {
    // Corner and size snap separately, so equal sides stay equal.
    const corner = grid > 1 ? (value) => roundHalfEven(value / grid) * grid : (value) => roundHalfEven(value);
    const span = grid > 1
      ? (value) => Math.max(grid, roundHalfEven(value / grid) * grid)
      : (value) => Math.max(1, roundHalfEven(value));
    x0 = corner(box.x * map.sx + map.ox);
    y0 = corner(box.y * map.sy + map.oy);
    x1 = x0 + span(box.width * map.sx);
    y1 = y0 + span(box.height * map.sy);
  }
  x0 = Math.max(0, Math.min(to[0], x0));
  x1 = Math.max(0, Math.min(to[0], x1));
  y0 = Math.max(0, Math.min(to[1], y0));
  y1 = Math.max(0, Math.min(to[1], y1));
  if (x1 - x0 < 1 || y1 - y0 < 1) return null;
  return { ...box, x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

/**
 * A rectangle on the run's frame carried back into the layout's frame.
 *
 * @param {{x: number, y: number, width: number, height: number}} rect - On the run's frame.
 * @param {object} map - What `mapping` answered.
 * @returns {{x: number, y: number, width: number, height: number}} In whole layout pixels.
 */
export function unplaced(rect, map) {
  if (map.same) return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  return {
    x: roundHalfEven((rect.x - map.ox) / map.sx),
    y: roundHalfEven((rect.y - map.oy) / map.sy),
    width: Math.max(1, roundHalfEven(rect.width / map.sx)),
    height: Math.max(1, roundHalfEven(rect.height / map.sy)),
  };
}

/**
 * The first palette colour no box uses yet.
 *
 * @param {object[]} boxes - The boxes already drawn.
 * @returns {string} A colour from `PALETTE`.
 */
export function nextColour(boxes) {
  const used = new Set(boxes.map((box) => String(box.color).toLowerCase()));
  return PALETTE.find((colour) => !used.has(colour.toLowerCase())) ?? PALETTE[boxes.length % PALETTE.length];
}

/**
 * Black or white, whichever reads on a colour.
 *
 * @param {string} colour - `#rrggbb`.
 * @returns {string} `#111111` or `#ffffff`.
 */
export function inkOn(colour) {
  const hex = normaliseColour(colour) ?? "#888888";
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#111111" : "#ffffff";
}
