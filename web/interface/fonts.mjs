/**
 * The typeface a node renders with, loaded into the page.
 *
 * `loadFont` resolves to the family for `ctx.font`, or null for a name the pack does not serve.
 * `faceMetrics` answers a face's ascent and descent at a size.
 */

import { api } from "../../../scripts/api.js";
import { fetchWithin } from "./request.mjs";

const LOG_PREFIX = "[WASNodeSuite.Fonts]";

const ROUTE = "/was/interface/api/font";

/** Font name to the promise of its family, or of null. One entry per name for the page. */
const pending = new Map();

/** Font name to the family it was added under, once it is ready to draw with. */
const families = new Map();

/** Font name to `{unitsPerEm, ascender, descender}` in design units, from the face's own tables. */
const designMetrics = new Map();

// The four things an sfnt file can start with: TrueType outlines, CFF outlines, the Apple spelling
// of the first, and a collection holding several faces.
const SFNT_TRUETYPE = 0x00010000;
const SFNT_CFF = 0x4f54544f;
const SFNT_APPLE = 0x74727565;
const SFNT_COLLECTION = 0x74746366;

// Where each number sits in the table that holds it, and the bit that says a face wants its
// typographic metrics used instead of its horizontal header ones.
const HEAD_UNITS_PER_EM = 18;
const HHEA_ASCENDER = 4;
const OS2_SELECTION = 62;
const OS2_TYPO_ASCENDER = 68;
const USE_TYPO_METRICS = 128;

/**
 * How many faces have been added, which makes each family name unique.
 */
let added = 0;

/**
 * The family a font is already loaded under.
 *
 * @param {string} name - The name the `font` menu offers, as the widget stores it.
 * @returns {string|null} The family to put in `ctx.font`, or null when this font is not loaded.
 *   A repaint asks with this and draws its fallback while a load is still in flight.
 */
export function fontFamily(name) {
  return families.get(String(name ?? "").trim()) ?? null;
}

/**
 * Load one font the pack serves, and add it to the document.
 *
 * @param {string} name - The name the `font` menu offers, as the widget stores it. A path is
 *   not a name and is never served.
 * @returns {Promise<string|null>} The family to put in `ctx.font`, or null when the pack does
 *   not serve that name or the bytes could not be parsed. Called again for a name already
 *   loaded or in flight, the same promise comes back and no second request is made.
 */
export function loadFont(name) {
  const key = String(name ?? "").trim();
  if (!key) return Promise.resolve(null);
  let promise = pending.get(key);
  if (!promise) {
    promise = load(key);
    pending.set(key, promise);
  }
  return promise;
}

/**
 * Fetch one font's bytes and add the face.
 *
 * @param {string} name - The font name, already trimmed and known to be non-empty.
 * @returns {Promise<string|null>} The family, or null when the font is not available.
 */
async function load(name) {
  let bytes = null;
  try {
    // The bytes are asked for rather than handed to FontFace as a URL: the answer carries an
    // ETag and a max-age, so the browser holds its own copy across page loads, and a refusal
    // is a status to read here rather than a load error with nothing in it.
    const response = await fetchWithin(`${ROUTE}?name=${encodeURIComponent(name)}`);
    // 404 is the answer for a font the pack does not list, which is a value and not a fault:
    // an interface draws its approximate state and says so.
    if (response?.status === 404) return null;
    if (!response?.ok) {
      console.error(`${LOG_PREFIX} The font ${name} was answered with ${response?.status}.`);
      return null;
    }
    bytes = await response.arrayBuffer();
  } catch (error) {
    console.error(`${LOG_PREFIX} Failed to ask for the font ${name}:`, error);
    return null;
  }
  if (!bytes?.byteLength) return null;

  // Read before the face is built, since the metrics are wanted whether or not this browser can
  // parse the outlines, and a caller drawing in a substitute face still needs to know it has one.
  const units = readDesignMetrics(bytes, name);
  if (units) designMetrics.set(name, units);

  const family = `WASFont${++added}`;
  try {
    const face = new FontFace(family, bytes);
    await face.load();
    document.fonts.add(face);
  } catch (error) {
    console.error(`${LOG_PREFIX} The font ${name} could not be read by this browser:`, error);
    return null;
  }
  families.set(name, family);
  return family;
}

/**
 * The ascent and descent a face gives at a size, in whole pixels.
 *
 * @param {string} name - The name the `font` menu offers, as the widget stores it.
 * @param {number} size - Point size, which is the `font_size` widget.
 * @returns {{ascent: number, descent: number}|null} The two metrics, or null for a font whose
 *   bytes have not arrived or whose tables could not be read.
 */
export function faceMetrics(name, size) {
  const units = designMetrics.get(String(name ?? "").trim());
  const points = Math.max(1, Math.trunc(Number(size) || 0));
  if (!units) return null;

  // FreeType's own arithmetic, in the order it does it.
  const scale = divFix(points * 64, units.unitsPerEm);
  return {
    ascent: Math.ceil(mulFix(units.ascender, scale) / 64),
    descent: -Math.floor(mulFix(units.descender, scale) / 64),
  };
}

/**
 * FreeType's `FT_DivFix`: a 16.16 fixed point quotient, rounded.
 *
 * @param {number} a - Numerator.
 * @param {number} b - Denominator.
 * @returns {number} The quotient in 16.16 fixed point.
 */
function divFix(a, b) {
  if (!b) return 0;
  const sign = Math.sign(a) * Math.sign(b);
  const value = Math.floor((Math.abs(a) * 65536 + Math.floor(Math.abs(b) / 2)) / Math.abs(b));
  return sign * value;
}

/**
 * FreeType's `FT_MulFix`: a value taken through a 16.16 fixed point factor, rounded.
 *
 * @param {number} a - Value in design units.
 * @param {number} b - Factor in 16.16 fixed point.
 * @returns {number} The product in 26.6 fixed point.
 */
function mulFix(a, b) {
  const sign = Math.sign(a) * Math.sign(b);
  // Divided rather than shifted: a face at a large size reaches past 32 bits, where a shift folds
  // the value into that width and answers a metric off by a multiple of 65536.
  return sign * Math.floor((Math.abs(a) * Math.abs(b) + 32768) / 65536);
}

/**
 * Read the vertical design metrics out of one font file.
 *
 * @param {ArrayBuffer} bytes - The whole file, as the route answered it.
 * @param {string} name - The font's name, for the log line when the tables cannot be read.
 * @returns {{unitsPerEm: number, ascender: number, descender: number}|null} The metrics in design
 *   units, or null when this is not a font file the tables can be found in.
 */
function readDesignMetrics(bytes, name) {
  try {
    const view = new DataView(bytes);
    const tables = sfntTables(view);
    const head = tables.get("head");
    const hhea = tables.get("hhea");
    if (!head || !hhea) return null;

    const unitsPerEm = view.getUint16(head + HEAD_UNITS_PER_EM);
    if (!(unitsPerEm > 0)) return null;
    let ascender = view.getInt16(hhea + HHEA_ASCENDER);
    let descender = view.getInt16(hhea + HHEA_ASCENDER + 2);

    // A face that sets this bit is asking for its typographic metrics, and FreeType obliges, so a
    // line height read off the horizontal header would be the wrong one for it. Of the faces this
    // pack ships, the one WAS Node Suite 2 drew with sets it.
    const os2 = tables.get("OS/2");
    if (os2 !== undefined && view.getUint16(os2 + OS2_SELECTION) & USE_TYPO_METRICS) {
      ascender = view.getInt16(os2 + OS2_TYPO_ASCENDER);
      descender = view.getInt16(os2 + OS2_TYPO_ASCENDER + 2);
    }
    return { unitsPerEm, ascender, descender };
  } catch (error) {
    console.error(`${LOG_PREFIX} The metrics of ${name} could not be read:`, error);
    return null;
  }
}

/**
 * Where each table of one sfnt face begins.
 *
 * @param {DataView} view - The whole file.
 * @returns {Map<string, number>} Table tag to its offset in the file. Empty for anything that is
 *   not an sfnt file, and the first face of a collection.
 */
function sfntTables(view) {
  const tables = new Map();
  let start = 0;
  if (view.getUint32(0) === SFNT_COLLECTION) {
    // A collection holds several faces behind one file, and the route serves it whole. The `font`
    // menu names the file rather than a face inside it, so the first face is the one meant.
    if (view.getUint32(8) < 1) return tables;
    start = view.getUint32(12);
  }
  const version = view.getUint32(start);
  if (version !== SFNT_TRUETYPE && version !== SFNT_CFF && version !== SFNT_APPLE) return tables;

  const count = view.getUint16(start + 4);
  for (let index = 0; index < count; index++) {
    const entry = start + 12 + index * 16;
    if (entry + 16 > view.byteLength) break;
    let tag = "";
    for (let byte = 0; byte < 4; byte++) tag += String.fromCharCode(view.getUint8(entry + byte));
    tables.set(tag, view.getUint32(entry + 8));
  }
  return tables;
}
