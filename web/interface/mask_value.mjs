/**
 * The drawn mask, as one string a widget holds.
 *
 * The format is `was-mask-1 <width>x<height> <base64 png>`, one line, with the empty string
 * standing for nothing drawn. The header states the size the pixels are at.
 */

const LOG_NAME = "WASNodeSuite.MaskValue";

/** The format tag every value starts with. A later format takes a later tag. */
export const MASK_VALUE_TAG = "was-mask-1";

/** What a widget holds when nothing is drawn. */
export const EMPTY_MASK_VALUE = "";

/**
 * Longest edge the drawing is stored at, in pixels.
 */
export const MASK_MAX_EDGE = 2048;

/**
 * The size a drawing over a frame is stored at.
 *
 * @param {number} width - Frame width in pixels.
 * @param {number} height - Frame height in pixels.
 * @returns {{width: number, height: number}} The stored size, at most `MASK_MAX_EDGE` on its
 *   long edge and never below one pixel on either.
 */
export function maskStoreSize(width, height) {
  const across = Math.max(1, Math.round(Number(width) || 0));
  const down = Math.max(1, Math.round(Number(height) || 0));
  const longest = Math.max(across, down);
  if (longest <= MASK_MAX_EDGE) return { width: across, height: down };
  const fit = MASK_MAX_EDGE / longest;
  return {
    width: Math.max(1, Math.round(across * fit)),
    height: Math.max(1, Math.round(down * fit)),
  };
}

/**
 * Read the header of a stored drawing without decoding its pixels.
 *
 * @param {string} value - What the widget holds.
 * @returns {{width: number, height: number, data: string}|null} The size the drawing was made
 *   at and its base64 body, or null when the value is empty or is not this format.
 */
export function readMaskHeader(value) {
  if (typeof value !== "string") return null;
  const text = value.trim();
  if (!text) return null;

  const parts = text.split(/\s+/);
  if (parts.length !== 3 || parts[0] !== MASK_VALUE_TAG) return null;

  const size = /^(\d+)x(\d+)$/.exec(parts[1]);
  if (!size) return null;

  const width = Number(size[1]);
  const height = Number(size[2]);
  if (!(width > 0) || !(height > 0)) return null;
  if (!parts[2]) return null;

  return { width, height, data: parts[2] };
}

/**
 * Whether a value holds a drawing.
 *
 * @param {string} value - What the widget holds.
 * @returns {boolean} True when the value parses as a drawing.
 */
export function hasDrawnMask(value) {
  return readMaskHeader(value) !== null;
}

/**
 * How many bytes a stored drawing adds to the workflow.
 *
 * @param {string} value - What the widget holds.
 * @returns {number} The value's length in bytes, 0 for an empty one.
 */
export function maskValueBytes(value) {
  if (typeof value !== "string" || !value) return 0;
  if (typeof TextEncoder === "function") return new TextEncoder().encode(value).length;
  return value.length;
}

/**
 * Encode a drawing as the value a widget holds.
 *
 * @param {HTMLCanvasElement} source - The drawing, at the size it is stored at. Coverage may be
 *   carried as alpha over white, which is what a strokes canvas holds, or as a grey level on an
 *   opaque canvas. Compositing onto black turns either into the level the value stores.
 * @returns {string} The value, or the empty string when the source cannot be encoded.
 */
export function encodeMask(source) {
  try {
    const width = Math.max(1, Math.round(source?.width || 0));
    const height = Math.max(1, Math.round(source?.height || 0));

    const flat = document.createElement("canvas");
    flat.width = width;
    flat.height = height;
    const ctx = flat.getContext("2d");
    if (!ctx) return EMPTY_MASK_VALUE;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(source, 0, 0, width, height);

    const url = flat.toDataURL("image/png");
    const comma = url.indexOf(",");
    if (comma < 0) return EMPTY_MASK_VALUE;
    const data = url.slice(comma + 1);
    if (!data) return EMPTY_MASK_VALUE;

    return `${MASK_VALUE_TAG} ${width}x${height} ${data}`;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to encode the drawing:`, error);
    return EMPTY_MASK_VALUE;
  }
}

/**
 * Decode a stored drawing back into a canvas.
 *
 * @param {string} value - What the widget holds.
 * @returns {Promise<{canvas: HTMLCanvasElement, width: number, height: number}|null>} The
 *   drawing at the size it was made, white everywhere with coverage in the alpha channel, or
 *   null when the value holds none or cannot be decoded.
 */
export async function decodeMask(value) {
  const header = readMaskHeader(value);
  if (!header) return null;

  try {
    const image = await loadPng(header.data);
    const canvas = document.createElement("canvas");
    canvas.width = header.width;
    canvas.height = header.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0, header.width, header.height);

    // The grey level is moved into the alpha channel.
    const pixels = ctx.getImageData(0, 0, header.width, header.height);
    const data = pixels.data;
    for (let at = 0; at < data.length; at += 4) {
      // Red alone, since the three colour channels agree.
      const level = data[at];
      data[at] = 255;
      data[at + 1] = 255;
      data[at + 2] = 255;
      data[at + 3] = level;
    }
    ctx.putImageData(pixels, 0, 0);

    return { canvas, width: header.width, height: header.height };
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to decode the drawing:`, error);
    return null;
  }
}

/**
 * Load a base64 PNG body as an image.
 *
 * @param {string} data - The base64 body, with no data URL prefix.
 * @returns {Promise<HTMLImageElement>} The decoded picture.
 */
function loadPng(data) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("the stored drawing is not a readable PNG"));
    image.src = `data:image/png;base64,${data}`;
  });
}
