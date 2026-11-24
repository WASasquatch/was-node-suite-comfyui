/**
 * The colour cell node interfaces draw colours with.
 *
 * A cell is one colour laid over a checkerboard, or a marker where there is no colour to lay.
 * `parseColor` reads a widget string and `drawCell` paints the result.
 */

const LOG_NAME = "WASNodeSuite.ColourCell";

/**
 * What a parse found. The keys of a `tallyColours` count are these values.
 */
export const STATUS = {
  COLOUR: "colour",
  EMPTY: "empty",
  DECLINED: "declined",
  INVALID: "invalid",
};

// Glyph drawn in the cell when there is no colour to draw. A function form gets its own
// glyph, apart from the one an unreadable value gets.
const MARK_DECLINED = "()";
const MARK_INVALID = "?";

// PIL refuses a specifier longer than this before it looks at the text at all.
const TEXT_LIMIT = 100;

// The characters Python's `str.strip` takes off a value, which is the trim both node
// functions apply before they hand the text to PIL. JavaScript's `trim` reads a different
// set at five code points: it leaves the ASCII separators `\x1c` to `\x1f` and the next
// line `\x85` in place, and it takes off a byte order mark, which Python leaves and PIL
// then refuses.
const PYTHON_SPACE =
  "[\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const PYTHON_TRIM = new RegExp(`^${PYTHON_SPACE}+|${PYTHON_SPACE}+$`, "g");

// The checkerboard alpha is shown against, and the side of one square in element pixels. Two
// mid greys read as a checkerboard against both a light and a dark panel.
const CHECKER_LIGHT = [153, 153, 153];
const CHECKER_DARK = [102, 102, 102];
const CHECKER_SIZE = 4;

const MARK_FONT = "9px sans-serif";

// Used when no marker colour is given. ComfyUI's own warning colour is what an interface
// passes in, read from the palette rather than hardcoded.
const MARK_FALLBACK = "#ff9800";

// The function forms PIL resolves, with the grammar it resolves them by, in the order
// `getrgb` tries them. `hs[bv]` covers `hsv()` and `hsb()`, which share one pattern.
//
// None of these is handed to `ctx.fillStyle` to be resolved instead. CSS and PIL
// disagree in both directions: `rgba(0,0,0,0.5)` is a half transparent black to the browser
// and unreadable to PIL, which answers it with the caller's fallback, so a resolved swatch
// would show a colour the node never draws.
//
// One narrow divergence from PIL is left standing here. PIL's own patterns accept any Unicode
// decimal digit where `\d` below accepts ASCII, so a function form written in another numeral
// system is marked unreadable rather than not previewed. That case ends at a marker either
// way, so nothing on screen claims a colour the node does not draw.
const FUNCTION_FORMS = [
  { name: "rgb()", pattern: /^rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$/ },
  { name: "rgb(%)", pattern: /^rgb\(\s*\d+%\s*,\s*\d+%\s*,\s*\d+%\s*\)$/ },
  { name: "hsl()", pattern: /^hsl\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*%\s*,\s*\d+\.?\d*%\s*\)$/ },
  { name: "hsv()", pattern: /^hs[bv]\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*%\s*,\s*\d+\.?\d*%\s*\)$/ },
  { name: "rgba()", pattern: /^rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$/ },
];

// PIL's own colour table, in the order `PIL.ImageColor.colormap` lists it: 148 names, which
// are the X11 set CSS uses with both spellings of every grey. A name is looked up after the
// text has been lowered, so any capitalisation reads.
const NAMED_COLOURS = {
  aliceblue: "#f0f8ff",
  antiquewhite: "#faebd7",
  aqua: "#00ffff",
  aquamarine: "#7fffd4",
  azure: "#f0ffff",
  beige: "#f5f5dc",
  bisque: "#ffe4c4",
  black: "#000000",
  blanchedalmond: "#ffebcd",
  blue: "#0000ff",
  blueviolet: "#8a2be2",
  brown: "#a52a2a",
  burlywood: "#deb887",
  cadetblue: "#5f9ea0",
  chartreuse: "#7fff00",
  chocolate: "#d2691e",
  coral: "#ff7f50",
  cornflowerblue: "#6495ed",
  cornsilk: "#fff8dc",
  crimson: "#dc143c",
  cyan: "#00ffff",
  darkblue: "#00008b",
  darkcyan: "#008b8b",
  darkgoldenrod: "#b8860b",
  darkgray: "#a9a9a9",
  darkgrey: "#a9a9a9",
  darkgreen: "#006400",
  darkkhaki: "#bdb76b",
  darkmagenta: "#8b008b",
  darkolivegreen: "#556b2f",
  darkorange: "#ff8c00",
  darkorchid: "#9932cc",
  darkred: "#8b0000",
  darksalmon: "#e9967a",
  darkseagreen: "#8fbc8f",
  darkslateblue: "#483d8b",
  darkslategray: "#2f4f4f",
  darkslategrey: "#2f4f4f",
  darkturquoise: "#00ced1",
  darkviolet: "#9400d3",
  deeppink: "#ff1493",
  deepskyblue: "#00bfff",
  dimgray: "#696969",
  dimgrey: "#696969",
  dodgerblue: "#1e90ff",
  firebrick: "#b22222",
  floralwhite: "#fffaf0",
  forestgreen: "#228b22",
  fuchsia: "#ff00ff",
  gainsboro: "#dcdcdc",
  ghostwhite: "#f8f8ff",
  gold: "#ffd700",
  goldenrod: "#daa520",
  gray: "#808080",
  grey: "#808080",
  green: "#008000",
  greenyellow: "#adff2f",
  honeydew: "#f0fff0",
  hotpink: "#ff69b4",
  indianred: "#cd5c5c",
  indigo: "#4b0082",
  ivory: "#fffff0",
  khaki: "#f0e68c",
  lavender: "#e6e6fa",
  lavenderblush: "#fff0f5",
  lawngreen: "#7cfc00",
  lemonchiffon: "#fffacd",
  lightblue: "#add8e6",
  lightcoral: "#f08080",
  lightcyan: "#e0ffff",
  lightgoldenrodyellow: "#fafad2",
  lightgreen: "#90ee90",
  lightgray: "#d3d3d3",
  lightgrey: "#d3d3d3",
  lightpink: "#ffb6c1",
  lightsalmon: "#ffa07a",
  lightseagreen: "#20b2aa",
  lightskyblue: "#87cefa",
  lightslategray: "#778899",
  lightslategrey: "#778899",
  lightsteelblue: "#b0c4de",
  lightyellow: "#ffffe0",
  lime: "#00ff00",
  limegreen: "#32cd32",
  linen: "#faf0e6",
  magenta: "#ff00ff",
  maroon: "#800000",
  mediumaquamarine: "#66cdaa",
  mediumblue: "#0000cd",
  mediumorchid: "#ba55d3",
  mediumpurple: "#9370db",
  mediumseagreen: "#3cb371",
  mediumslateblue: "#7b68ee",
  mediumspringgreen: "#00fa9a",
  mediumturquoise: "#48d1cc",
  mediumvioletred: "#c71585",
  midnightblue: "#191970",
  mintcream: "#f5fffa",
  mistyrose: "#ffe4e1",
  moccasin: "#ffe4b5",
  navajowhite: "#ffdead",
  navy: "#000080",
  oldlace: "#fdf5e6",
  olive: "#808000",
  olivedrab: "#6b8e23",
  orange: "#ffa500",
  orangered: "#ff4500",
  orchid: "#da70d6",
  palegoldenrod: "#eee8aa",
  palegreen: "#98fb98",
  paleturquoise: "#afeeee",
  palevioletred: "#db7093",
  papayawhip: "#ffefd5",
  peachpuff: "#ffdab9",
  peru: "#cd853f",
  pink: "#ffc0cb",
  plum: "#dda0dd",
  powderblue: "#b0e0e6",
  purple: "#800080",
  rebeccapurple: "#663399",
  red: "#ff0000",
  rosybrown: "#bc8f8f",
  royalblue: "#4169e1",
  saddlebrown: "#8b4513",
  salmon: "#fa8072",
  sandybrown: "#f4a460",
  seagreen: "#2e8b57",
  seashell: "#fff5ee",
  sienna: "#a0522d",
  silver: "#c0c0c0",
  skyblue: "#87ceeb",
  slateblue: "#6a5acd",
  slategray: "#708090",
  slategrey: "#708090",
  snow: "#fffafa",
  springgreen: "#00ff7f",
  steelblue: "#4682b4",
  tan: "#d2b48c",
  teal: "#008080",
  thistle: "#d8bfd8",
  tomato: "#ff6347",
  turquoise: "#40e0d0",
  violet: "#ee82ee",
  wheat: "#f5deb3",
  white: "#ffffff",
  whitesmoke: "#f5f5f5",
  yellow: "#ffff00",
  yellowgreen: "#9acd32",
};

/**
 * Clamp a number into a range.
 *
 * @param {number} value - Value to clamp.
 * @param {number} low - Lower bound.
 * @param {number} high - Upper bound.
 * @returns {number} The value, held inside the bounds.
 */
function clamp(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

/**
 * Hold a value to a whole channel.
 *
 * @param {number} value - Channel value.
 * @returns {number} A whole number, 0 to 255.
 */
function channel(value) {
  return clamp(Math.round(Number(value) || 0), 0, 255);
}

/**
 * Trim a value the way Python's `str.strip` trims one.
 *
 * @param {string} text - Text as the widget holds it.
 * @returns {string} The text without the whitespace Python strips.
 */
function pythonTrim(text) {
  return String(text ?? "").replace(PYTHON_TRIM, "");
}

/**
 * Read one hex digit as the channel PIL reads it from a short form.
 *
 * @param {string} digit - One hex digit.
 * @returns {number} The digit doubled, so `f` is 255 rather than 15.
 */
function nibble(digit) {
  return parseInt(`${digit}${digit}`, 16);
}

/**
 * Build the result for a colour that resolved.
 *
 * @param {string} form - The spelling recognised.
 * @param {number[]} rgba - Red, green, blue and alpha, 0 to 255.
 * @param {string} text - The trimmed text.
 * @returns {object} A parse result.
 */
function resolvedColour(form, rgba, text) {
  return { status: STATUS.COLOUR, form, rgba, text, mark: "", note: "" };
}

/**
 * Build the result for text holding nothing.
 *
 * @returns {object} A parse result. `parse_color` reads this as fully transparent and
 *   `parse_palette` skips it, so the caller decides which.
 */
function emptyColour() {
  return { status: STATUS.EMPTY, form: "", rgba: null, text: "", mark: "", note: "" };
}

/**
 * Build the result for a spelling the node resolves and the cell does not preview.
 *
 * @param {string} form - The function form recognised.
 * @param {string} text - The trimmed text.
 * @returns {object} A parse result.
 */
function declinedColour(form, text) {
  return {
    status: STATUS.DECLINED,
    form,
    rgba: null,
    text,
    mark: MARK_DECLINED,
    note: `${form} is not previewed`,
  };
}

/**
 * Build the result for text that is not a colour.
 *
 * @param {string} text - The trimmed text.
 * @param {string} note - Wording for a footer or a tooltip.
 * @returns {object} A parse result.
 */
function unreadableColour(text, note = "unreadable") {
  return { status: STATUS.INVALID, form: "", rgba: null, text, mark: MARK_INVALID, note };
}

/**
 * Read a colour out of a widget string, with the rules the nodes read one by.
 *
 * @param {string} text - Text as the widget holds it, trimmed here.
 * @returns {{status: string, form: string, rgba: number[]|null, text: string, mark: string,
 *   note: string}} The status as one of `STATUS`, the spelling recognised, the colour with
 *   its alpha where there is one, the trimmed text, the glyph a cell draws when there is no
 *   colour, and a note for a footer.
 */
export function parseColor(text) {
  const value = pythonTrim(text);
  if (value === "") return emptyColour();
  if (value.length > TEXT_LIMIT) {
    return unreadableColour(value, `unreadable, over ${TEXT_LIMIT} characters`);
  }

  const lower = value.toLowerCase();

  if (Object.hasOwn(NAMED_COLOURS, lower)) {
    const hex = NAMED_COLOURS[lower];
    return resolvedColour(
      "name",
      [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16),
        255,
      ],
      value,
    );
  }

  // PIL's hex grammar is these four lengths and no others. The 9 and 12 digit forms X11
  // accepts are read by neither PIL nor CSS, so they fall through to the marker below.
  if (/^#[a-f0-9]{3}$/.test(lower)) {
    return resolvedColour(
      "#rgb",
      [nibble(lower[1]), nibble(lower[2]), nibble(lower[3]), 255],
      value,
    );
  }

  if (/^#[a-f0-9]{4}$/.test(lower)) {
    return resolvedColour(
      "#rgba",
      [nibble(lower[1]), nibble(lower[2]), nibble(lower[3]), nibble(lower[4])],
      value,
    );
  }

  if (/^#[a-f0-9]{6}$/.test(lower)) {
    return resolvedColour(
      "#rrggbb",
      [
        parseInt(lower.slice(1, 3), 16),
        parseInt(lower.slice(3, 5), 16),
        parseInt(lower.slice(5, 7), 16),
        255,
      ],
      value,
    );
  }

  if (/^#[a-f0-9]{8}$/.test(lower)) {
    return resolvedColour(
      "#rrggbbaa",
      [
        parseInt(lower.slice(1, 3), 16),
        parseInt(lower.slice(3, 5), 16),
        parseInt(lower.slice(5, 7), 16),
        parseInt(lower.slice(7, 9), 16),
      ],
      value,
    );
  }

  for (const form of FUNCTION_FORMS) {
    if (form.pattern.test(lower)) return declinedColour(form.name, value);
  }

  return unreadableColour(value);
}

/**
 * Write a colour as the hex a widget stores.
 *
 * Six digits while the colour is opaque, eight once it carries an alpha.
 *
 * @param {number[]} rgba - Red, green and blue, with an optional alpha. A missing alpha is
 *   opaque.
 * @returns {string} A `#rrggbb` or `#rrggbbaa` colour.
 */
export function formatColour(rgba) {
  const values = Array.isArray(rgba) ? rgba : [0, 0, 0, 255];
  const alpha = values.length > 3 ? channel(values[3]) : 255;
  const hex = (value) => channel(value).toString(16).padStart(2, "0");
  const opaque = `#${hex(values[0])}${hex(values[1])}${hex(values[2])}`;
  return alpha === 255 ? opaque : `${opaque}${hex(alpha)}`;
}

/**
 * Read a `#rrggbb` colour into an RGB triple.
 *
 * @param {string} hex - Colour as written by a native colour input.
 * @returns {number[]} Three channels, 0 to 255.
 */
function hexToRgb(hex) {
  const text = String(hex ?? "").replace("#", "");
  if (text.length !== 6) return [0, 0, 0];
  return [
    parseInt(text.slice(0, 2), 16) || 0,
    parseInt(text.slice(2, 4), 16) || 0,
    parseInt(text.slice(4, 6), 16) || 0,
  ];
}

/**
 * Format a colour for a native colour input, which takes six digits and no alpha.
 *
 * @param {number[]} rgb - Red, green and blue. Any alpha is left off.
 * @returns {string} A `#rrggbb` colour.
 */
function inputHex(rgb) {
  const values = Array.isArray(rgb) ? rgb : [0, 0, 0];
  return formatColour([values[0], values[1], values[2], 255]);
}

/**
 * Paint one colour over the checkerboard.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {number} ratio - Device pixel ratio the canvas is scaled by.
 * @param {{x: number, y: number, width: number, height: number}} rect - Cell in element
 *   pixels.
 * @param {number[]|null} rgba - Colour to lay over the checkerboard, with an optional alpha.
 *   Null leaves the checkerboard bare.
 * @param {{checkerLight?: number[], checkerDark?: number[], checkerSize?: number}} [options]
 *   - Checkerboard colours as RGB triples and the side of one square in element pixels.
 * @returns {void}
 */
export function drawSwatch(ctx, ratio, rect, rgba, options = {}) {
  if (!ctx || !rect) return;

  // `putImageData` is not affected by the context transform, so the rectangle is converted to
  // device pixels here rather than left in the element pixels the caller works in. Each edge is
  // rounded on its own, so a row of adjacent cells tiles with no seam and no overlap.
  const scale = Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
  const left = Math.round(rect.x * scale);
  const top = Math.round(rect.y * scale);
  const width = Math.max(1, Math.round((rect.x + rect.width) * scale) - left);
  const height = Math.max(1, Math.round((rect.y + rect.height) * scale) - top);
  if (![left, top, width, height].every(Number.isFinite)) return;

  const square = Math.max(1, Math.round((options.checkerSize ?? CHECKER_SIZE) * scale));
  const colour = Array.isArray(rgba) ? rgba : null;
  const alpha = colour ? (colour.length > 3 ? channel(colour[3]) : 255) : 0;
  const source = colour ? [channel(colour[0]), channel(colour[1]), channel(colour[2])] : [0, 0, 0];

  // Straight alpha, over the checkerboard the interface draws for itself. The node
  // composites onto the real image instead, so this board reaches no output.
  const blend = (board) => [
    Math.round((source[0] * alpha + channel(board[0]) * (255 - alpha)) / 255),
    Math.round((source[1] * alpha + channel(board[1]) * (255 - alpha)) / 255),
    Math.round((source[2] * alpha + channel(board[2]) * (255 - alpha)) / 255),
  ];
  const light = blend(options.checkerLight ?? CHECKER_LIGHT);
  const dark = blend(options.checkerDark ?? CHECKER_DARK);

  // A checkerboard holds two rows, so both are built once and each line of the cell takes
  // whichever its band calls for.
  const rows = [new Uint8ClampedArray(width * 4), new Uint8ClampedArray(width * 4)];
  for (let column = 0; column < width; column++) {
    const offset = column * 4;
    const even = Math.floor(column / square) % 2 === 0;
    const pair = even ? [light, dark] : [dark, light];
    for (let row = 0; row < 2; row++) {
      rows[row][offset] = pair[row][0];
      rows[row][offset + 1] = pair[row][1];
      rows[row][offset + 2] = pair[row][2];
      rows[row][offset + 3] = 255;
    }
  }

  const image = ctx.createImageData(width, height);
  for (let line = 0; line < height; line++) {
    const even = Math.floor(line / square) % 2 === 0;
    image.data.set(rows[even ? 0 : 1], line * width * 4);
  }
  // `putImageData` replaces the pixels it covers instead of compositing onto them.
  ctx.putImageData(image, left, top);
}

/**
 * Paint the cell a parse result calls for.
 *
 * A resolved colour is drawn over the checkerboard. Text holding nothing leaves the
 * checkerboard bare.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {number} ratio - Device pixel ratio the canvas is scaled by.
 * @param {{x: number, y: number, width: number, height: number}} rect - Cell in element
 *   pixels.
 * @param {object} parsed - Result from `parseColor`.
 * @param {{markColour?: string, markFont?: string, checkerLight?: number[],
 *   checkerDark?: number[], checkerSize?: number}} [options] - Marker colour and font, taken
 *   from the interface's own theme, and the checkerboard `drawSwatch` reads.
 * @returns {void}
 */
export function drawCell(ctx, ratio, rect, parsed, options = {}) {
  if (!ctx || !rect) return;

  drawSwatch(ctx, ratio, rect, parsed?.status === STATUS.COLOUR ? parsed.rgba : null, options);

  const mark = parsed?.mark ?? "";
  if (!mark) return;

  // The marker is text under the context transform the interface set, in element pixels, while
  // the swatch beneath it was written in device pixels.
  ctx.save();
  ctx.font = options.markFont ?? MARK_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = options.markColour ?? MARK_FALLBACK;
  ctx.fillText(
    mark,
    rect.x + rect.width / 2,
    rect.y + rect.height / 2,
    Math.max(1, rect.width - 2),
  );
  ctx.restore();
}

/**
 * Draw the one pixel frame that separates a cell from the node behind it.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {{x: number, y: number, width: number, height: number}} rect - Cell in element
 *   pixels.
 * @param {string} colour - Border colour, read from the interface's own theme.
 * @returns {void}
 */
export function outlineCell(ctx, rect, colour) {
  if (!ctx || !rect) return;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = colour;
  ctx.strokeRect(
    rect.x + 0.5,
    rect.y + 0.5,
    Math.max(1, rect.width - 1),
    Math.max(1, rect.height - 1),
  );
  ctx.restore();
}

// The one hidden input, created on first use.
let colorInput = null;

/**
 * Get the one hidden colour input every cell opens.
 *
 * @returns {HTMLInputElement} The shared colour input.
 */
function getColorInput() {
  if (!colorInput) {
    colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.style.cssText = "position:absolute;opacity:0;pointer-events:none;z-index:-999";
    document.body.appendChild(colorInput);
  }
  return colorInput;
}

/**
 * Open the native colour picker at a point on screen.
 *
 * @param {number} clientX - Horizontal position on screen.
 * @param {number} clientY - Vertical position on screen.
 * @param {number[]} rgb - Colour the picker opens on. Any alpha is left off.
 * @param {(rgb: number[]) => void} onPicked - Called with the chosen colour.
 * @returns {void}
 */
export function pickColour(clientX, clientY, rgb, onPicked) {
  const input = getColorInput();
  input.value = inputHex(rgb);
  input.style.left = `${Math.round(clientX)}px`;
  input.style.top = `${Math.round(clientY)}px`;
  // Assigned rather than added, so at most one pick is ever armed. A cancelled picker fires no
  // event, so a handler that was added would stay attached and would recolour the cell it was
  // opened for the next time any pick completed.
  input.onchange = () => {
    input.onchange = null;
    try {
      // A native colour input carries no alpha, so this is three channels. An interface keeping
      // an alpha the widget already held carries it over itself.
      onPicked(hexToRgb(input.value));
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to apply the picked colour:`, error);
    }
  };
  // Deferred by one frame, which puts the picker under the pointer in Chrome and out of the
  // screen corner in Firefox on Windows.
  requestAnimationFrame(() => input.click());
}

/**
 * Count parse results by status.
 *
 * @param {object[]} results - Results from `parseColor`.
 * @returns {{colour: number, empty: number, declined: number, invalid: number}} How many
 *   results carried each status, keyed on the values of `STATUS`.
 */
export function tallyColours(results) {
  const tally = { colour: 0, empty: 0, declined: 0, invalid: 0 };
  for (const parsed of results ?? []) {
    const status = parsed?.status;
    if (typeof status === "string" && Object.hasOwn(tally, status)) tally[status] += 1;
  }
  return tally;
}

/**
 * Word a tally as the residual line a footer carries.
 *
 * @param {{invalid?: number, declined?: number}} tally - Tally from `tallyColours`.
 * @param {string} [singular] - What one entry is called.
 * @param {string} [plural] - What several are called.
 * @returns {string} The footer line, empty when nothing was left over.
 */
export function residualNote(tally, singular = "line", plural = `${singular}s`) {
  const count = (value) => Math.max(0, Math.trunc(Number(value) || 0));
  const invalid = count(tally?.invalid);
  const declined = count(tally?.declined);
  const parts = [];
  if (invalid > 0) parts.push(`${invalid} ${invalid === 1 ? singular : plural} unreadable`);
  if (declined > 0) {
    parts.push(`${declined} ${declined === 1 ? singular : plural} not previewed`);
  }
  return parts.join(", ");
}
