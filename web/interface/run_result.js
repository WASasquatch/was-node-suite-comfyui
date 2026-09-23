/**
 * What a node's last run did, for the interface on that node.
 *
 * `fetchRunResult` answers a state, its words, and the report. Spans arrive counted in
 * characters and leave in the code units a browser indexes by.
 */

import { api } from "../../../scripts/api.js";
import { fetchWithin } from "./request.js";
import { LABELS, PREVIEW_STATE, connected, executionId, placed } from "./preview.js";

const LOG_PREFIX = "[WASNodeSuite.RunResult]";

const ROUTE = "/was/interface/api/run_result";

const PAGE_ROUTE = "/was/interface/api/run_result_page";

// Lines one page is asked for when the caller names no number. `modules/interface/run_result.py`
// answers at most 500 whatever is asked for.
const PAGE_LINES = 200;

// Rows of each kind read from one answer. `modules/interface/run_result.py` holds a report to
// eight of each, so this only bounds what a malformed body can cost.
const MAX_ROWS = 16;

// Marked spans read from one body. `modules/interface/run_result.py` holds a body to 64, so
// this only bounds what a malformed body can cost.
const MAX_MARKS = 128;

// The narrowest a block is wrapped to, so a box squeezed to nothing wraps one
// character to a line rather than looping forever.
const MIN_WRAP_WIDTH = 12;

// The reversed CRC-32 polynomial, which is the one zlib uses.
const CRC_POLYNOMIAL = 0xedb88320;

let CRC_TABLE = null;
let ENCODER = null;

/** The statuses a report carries, spelled as `modules/interface/run_result.py` spells them. */
export const RUN_STATUS = {
  // The run did what the node is for.
  OK: "ok",
  // The run finished and produced something worth seeing before the output is used.
  WARNING: "warning",
  // The run finished and part of what it was asked to do did not happen.
  ERROR: "error",
};

/** The parts of a report `truncated` can name, so an adopter reads them by constant. */
export const TRUNCATED = {
  SUMMARY: "summary",
  COUNTS: "counts",
  // A row of a breakdown was left out, which a report also says by counting more of them in
  // `talliesTotal` than it carries.
  TALLIES: "tallies",
  FACTS: "facts",
  ITEMS: "items",
  TEXT: "text",
  // A value the run was handed was left out of the report.
  INPUTS: "inputs",
  // A whole body was left out of the report.
  BODIES: "bodies",
  // A body carries a window of its text rather than all of it.
  BODY_TEXT: "body_text",
  // A body carries fewer marked spans than its text holds.
  MARKS: "marks",
};

/**
 * The words an interface draws for each state of a report.
 */
export const RUN_LABELS = {
  [PREVIEW_STATE.CONNECTING]: LABELS[PREVIEW_STATE.CONNECTING],
  [PREVIEW_STATE.WAITING]: "No run to report yet",
  [PREVIEW_STATE.LOADING]: LABELS[PREVIEW_STATE.LOADING],
  [PREVIEW_STATE.READY]: "",
  [PREVIEW_STATE.FAILED]: "The run report could not be read",
};

/**
 * Fetch the report a node published on its last run.
 *
 * @param {object|string|number} nodeOrId - The node itself, preferred so a node inside a
 *   subgraph resolves its execution id, or a bare id.
 * @returns {Promise<{state: string, label: string, result: object|null}>} The state, the words
 *   to draw for it, and on `READY` the report.
 */
export async function fetchRunResult(nodeOrId) {
  const node = typeof nodeOrId === "object" && nodeOrId !== null ? nodeOrId : null;
  const id = (node ? executionId(node) : String(nodeOrId ?? "")).trim();
  if (!placed(id)) return answer(PREVIEW_STATE.WAITING);
  if (!connected()) return answer(PREVIEW_STATE.CONNECTING);

  // A graph id is handed out again after a graph is cleared, so the kind of node asking goes
  // with the id and a report published by another kind is answered as no report at all.
  const kind = String(node?.comfyClass ?? node?.type ?? "").trim();
  const asks = `${ROUTE}?node_id=${encodeURIComponent(id)}`
    + (kind ? `&node_type=${encodeURIComponent(kind)}` : "");

  try {
    const response = await fetchWithin(asks, {
      cache: "no-store",
    });
    // 404 is the answer for a node that has published nothing.
    if (response?.status === 404) return answer(PREVIEW_STATE.WAITING);
    if (!response?.ok) return answer(PREVIEW_STATE.FAILED);
    const report = normalise(await response.json());
    return report ? answer(PREVIEW_STATE.READY, report) : answer(PREVIEW_STATE.FAILED);
  } catch (error) {
    console.error(`${LOG_PREFIX} Failed to ask what node ${id} last did:`, error);
    return answer(PREVIEW_STATE.FAILED);
  }
}

/**
 * Fetch a range of lines from one body of the report a node published.
 *
 * @param {object|string|number} nodeOrId - The node itself, preferred so a node inside a
 *   subgraph resolves its execution id, or a bare id.
 * @param {number} body - Which body of the report, counting from zero in the order it
 *   carries them.
 * @param {number} start - The first line wanted, counting from zero.
 * @param {number} count - How many lines to ask for, held to `MAX_PAGE_LINES` by the server.
 * @returns {Promise<{state: string, label: string, page: object|null}>} The state, the words
 *   to draw for it, and on `READY` `{name, start, lines, total, held, clipped, run}`.
 */
export async function fetchRunResultPage(nodeOrId, body = 0, start = 0, count = PAGE_LINES) {
  const node = typeof nodeOrId === "object" && nodeOrId !== null ? nodeOrId : null;
  const id = (node ? executionId(node) : String(nodeOrId ?? "")).trim();
  if (!placed(id)) return paged(PREVIEW_STATE.WAITING);
  if (!connected()) return paged(PREVIEW_STATE.CONNECTING);

  const kind = String(node?.comfyClass ?? node?.type ?? "").trim();
  const asks = `${PAGE_ROUTE}?node_id=${encodeURIComponent(id)}`
    + `&body=${whole(body)}&start=${whole(start)}&count=${whole(count)}`
    + (kind ? `&node_type=${encodeURIComponent(kind)}` : "");

  try {
    const response = await fetchWithin(asks, { cache: "no-store" });
    // 404 is the answer for a node that has published nothing.
    if (response?.status === 404) return paged(PREVIEW_STATE.WAITING);
    if (!response?.ok) return paged(PREVIEW_STATE.FAILED);
    const page = normalisePage(await response.json());
    return page ? paged(PREVIEW_STATE.READY, page) : paged(PREVIEW_STATE.FAILED);
  } catch (error) {
    console.error(`${LOG_PREFIX} Failed to ask node ${id} for lines ${start}:`, error);
    return paged(PREVIEW_STATE.FAILED);
  }
}

/**
 * One piece of a report's text as it can be drawn on a single line.
 *
 * Control characters become spaces, one for one.
 *
 * @param {string} text - An item's text.
 * @param {number[]|null} mark - Its `[start, end]` span, or null.
 * @returns {{text: string, mark: number[]|null}} The drawable text and the span inside it.
 */
export function visibleText(text, mark) {
  const drawn = drawableBody(text).split("\n").join(" ");
  if (!Array.isArray(mark) || mark.length !== 2) return { text: drawn, mark: null };
  const first = bound(mark[0], drawn.length);
  return { text: drawn, mark: [first, Math.max(first, bound(mark[1], drawn.length))] };
}

/**
 * One body's text as it can be drawn as a block, its line breaks kept.
 *
 * @param {string} text - A body's text.
 * @returns {string} The drawable text, the same length as what came in.
 */
export function drawableBody(text) {
  const source = typeof text === "string" ? text : "";
  let drawn = "";
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === "\n") {
      drawn += char;
      continue;
    }
    const code = source.charCodeAt(index);
    drawn += code >= 32 && code !== 127 ? char : " ";
  }
  return drawn;
}

/**
 * Break a body's text into the lines a block draws it on.
 *
 * Line breaks are kept; the rest is wrapped at spaces.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the font the block is drawn in.
 * @param {string} text - The body's text, from `drawableBody`.
 * @param {number} width - Pixels to wrap to.
 * @returns {Array<{text: string, start: number}>} Each line and where it starts in the text.
 */
export function wrapBody(ctx, text, width) {
  const room = Math.max(MIN_WRAP_WIDTH, width);
  const lines = [];
  let start = 0;
  for (const paragraph of text.split("\n")) {
    let cursor = 0;
    while (cursor < paragraph.length) {
      let taken = fitting(ctx, paragraph, cursor, room);
      if (cursor + taken < paragraph.length) {
        // The break goes at the last space inside what fits.
        const space = paragraph.lastIndexOf(" ", cursor + taken);
        if (space > cursor) taken = space - cursor + 1;
      }
      lines.push({ text: paragraph.slice(cursor, cursor + taken), start: start + cursor });
      cursor += taken;
    }
    if (!paragraph.length) lines.push({ text: "", start });
    // One past the paragraph is the line break itself, which no line draws.
    start += paragraph.length + 1;
  }
  return lines;
}

/**
 * How many characters from an index fit a width.
 *
 * @param {CanvasRenderingContext2D} ctx - Context carrying the font the block is drawn in.
 * @param {string} paragraph - The text between two line breaks.
 * @param {number} cursor - Where the line starts in it.
 * @param {number} room - Pixels to fit.
 * @returns {number} Characters that fit, at least one.
 */
function fitting(ctx, paragraph, cursor, room) {
  const rest = paragraph.length - cursor;
  if (ctx.measureText(paragraph.slice(cursor)).width <= room) return rest;
  let low = 1;
  let high = rest;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (ctx.measureText(paragraph.slice(cursor, cursor + middle)).width <= room) low = middle;
    else high = middle - 1;
  }
  return Math.max(1, low);
}

/**
 * Build the answer for one state.
 *
 * @param {string} state - A value of `PREVIEW_STATE`.
 * @param {object|null} [result] - The report, for `READY`.
 * @returns {{state: string, label: string, result: object|null}} The answer.
 */
function answer(state, result = null) {
  return { state, label: RUN_LABELS[state] ?? "", result };
}

/**
 * Build the answer for one state of a page read.
 *
 * @param {string} state - A value of `PREVIEW_STATE`.
 * @param {object|null} [page] - The page, for `READY`.
 * @returns {{state: string, label: string, page: object|null}} The answer.
 */
function paged(state, page = null) {
  return { state, label: RUN_LABELS[state] ?? "", page };
}

/**
 * Read one page of a body into the shape a listing draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {{name: string, start: number, lines: string[], total: number, held: number,
 *   clipped: boolean, run: number}|null} The page, or null when the body is not one.
 */
function normalisePage(data) {
  if (!data || typeof data !== "object") return null;
  const lines = Array.isArray(data.lines) ? data.lines.map((line) => text(line)) : [];
  return {
    name: text(data.name),
    start: whole(data.start),
    lines,
    // Lines of the whole body, and of the part of it still reachable, neither below the page.
    total: Math.max(lines.length, whole(data.total)),
    held: Math.max(lines.length, whole(data.held)),
    clipped: data.clipped === true,
    run: whole(data.run),
  };
}

/**
 * Read one answer from the route into the shape an interface draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The report, or null when the body is not one.
 */
function normalise(data) {
  if (!data || typeof data !== "object") return null;
  const items = rows(data.items, item);
  const tallies = rows(data.tallies, count);
  return {
    status: Object.values(RUN_STATUS).includes(data.status) ? data.status : RUN_STATUS.OK,
    summary: text(data.summary),
    counts: rows(data.counts, count),
    // A tally is named and numbered exactly as a count is, so the same reader answers for both.
    tallies,
    // The node states how many its breakdown held before it took the first few, which is what a
    // readout drawing eight of them reports. It is never below what arrived.
    talliesTotal: Math.max(tallies.length, whole(data.tallies_total)),
    facts: rows(data.facts, fact),
    inputs: rows(data.inputs, handed),
    bodies: rows(data.bodies, block),
    items,
    // The node states how many rows it walked before taking a sample, which is the number a
    // readout holding eight of them reports. It is never below what arrived.
    itemsTotal: Math.max(items.length, whole(data.items_total)),
    truncated: rows(data.truncated, (value) => (typeof value === "string" ? value : null)),
    run: whole(data.run),
  };
}

/**
 * Read one array of a report, dropping whatever does not belong in it.
 *
 * @param {*} values - The array from the body.
 * @param {(value: *) => *} read - Reads one entry, answering null to drop it.
 * @returns {Array<*>} The entries that were readable, at most `MAX_ROWS` of them.
 */
function rows(values, read) {
  if (!Array.isArray(values)) return [];
  const kept = [];
  for (const value of values.slice(0, MAX_ROWS)) {
    const entry = read(value);
    if (entry !== null) kept.push(entry);
  }
  return kept;
}

/**
 * Read one named number.
 *
 * @param {*} value - The entry from the body.
 * @returns {{name: string, value: number}|null} The count, or null when it is not one.
 */
function count(value) {
  const name = text(value?.name);
  const number = Number(value?.value);
  return name && Number.isFinite(number) ? { name, value: number } : null;
}

/**
 * Read one named string.
 *
 * @param {*} value - The entry from the body.
 * @returns {{name: string, value: string}|null} The fact, or null when it is not one.
 */
function fact(value) {
  const name = text(value?.name);
  const said = text(value?.value);
  return name && said ? { name, value: said } : null;
}

/**
 * Read one value the run was handed.
 *
 * @param {*} value - The entry from the body.
 * @returns {{name: string, linked: boolean|null, bytes: number, checksum: string}|null} The
 *   entry, or null when it does not name a value well enough to measure a box against.
 */
function handed(value) {
  const name = text(value?.name);
  const sum = text(value?.checksum);
  if (!name || !sum) return null;
  return {
    name,
    // True for a link, false for the widget beside the input, null where the run could not
    // tell, which is not the same as the widget and is never read as it.
    linked: typeof value?.linked === "boolean" ? value.linked : null,
    bytes: whole(value?.bytes),
    checksum: sum,
  };
}

/**
 * Whether a value is the one a run was handed.
 *
 * @param {{bytes: number, checksum: string}} entry - One entry of a report's `inputs`.
 * @param {*} value - What the box holds now.
 * @returns {boolean} True when the value's UTF-8 encoding is the length the run published and
 *   carries the same CRC-32, so a change is missed only where a value keeps both.
 */
export function sameValue(entry, value) {
  const bytes = encoder().encode(typeof value === "string" ? value : "");
  // The length is measured first, so a text that is megabytes long is walked only when an edit
  // left it exactly as long as the run read it.
  if (bytes.length !== entry.bytes) return false;
  return crc32(bytes) === entry.checksum;
}

/**
 * The one text encoder, built when a value is first measured.
 *
 * @returns {TextEncoder} The encoder.
 */
function encoder() {
  ENCODER ??= new TextEncoder();
  return ENCODER;
}

/**
 * The CRC-32 of some bytes, as `modules/interface/run_result.py` writes it.
 *
 * @param {Uint8Array} bytes - The bytes.
 * @returns {string} Eight lowercase hexadecimal characters.
 */
function crc32(bytes) {
  const table = crcTable();
  let crc = -1;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = (crc >>> 8) ^ table[(crc ^ bytes[index]) & 0xff];
  }
  return ((crc ^ -1) >>> 0).toString(16).padStart(8, "0");
}

/**
 * The CRC-32 table, built once for the page.
 *
 * @returns {Int32Array} One entry per byte value.
 */
function crcTable() {
  if (CRC_TABLE) return CRC_TABLE;
  CRC_TABLE = new Int32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? (value >>> 1) ^ CRC_POLYNOMIAL : value >>> 1;
    }
    CRC_TABLE[index] = value;
  }
  return CRC_TABLE;
}

/**
 * Read one sample row.
 *
 * @param {*} value - The entry from the body.
 * @returns {{text: string, mark: number[]|null, note: string, clipped: boolean}|null} The row,
 *   or null when it carries nothing to draw.
 */
function item(value) {
  const said = text(value?.text);
  if (!said) return null;
  const spans = marks(value?.mark === undefined ? [] : [value.mark], said);
  return {
    text: said,
    mark: spans.length ? spans[0] : null,
    note: text(value?.note),
    clipped: value?.clipped === true,
  };
}

/**
 * Read one body of text.
 *
 * @param {*} value - The entry from the body.
 * @returns {{name: string, text: string, marks: number[][], marksTotal: number,
 *   offset: number, length: number, lines: number, whole: boolean}|null} The body, or null when
 *   it is not one.
 */
function block(value) {
  const name = text(value?.name);
  if (!name) return null;
  const said = text(value?.text);
  const spans = marks(Array.isArray(value?.marks) ? value.marks : [], said);
  const offset = whole(value?.offset);
  return {
    name,
    text: said,
    marks: spans,
    marksTotal: Math.max(spans.length, whole(value?.marks_total)),
    offset,
    // Characters of the text the body is a piece of, never fewer than the piece it carries.
    length: Math.max(whole(value?.length), offset + said.length),
    // Lines of that whole text, which is what a listing sizes its scrollbar against before it
    // has asked for a line past the ones the report carried.
    lines: Math.max(whole(value?.lines), said ? said.split("\n").length : 0),
    whole: value?.whole === true,
  };
}

/**
 * Read a report's spans onto the indices a browser counts.
 *
 * @param {Array<*>} spans - Pairs of indices, each counted in characters.
 * @param {string} said - The text they point into.
 * @returns {number[][]} The pairs that were spans, in code units, held inside the text.
 */
function marks(spans, said) {
  // Python counts a character where a browser counts a UTF-16 code unit, so a span sitting
  // after an astral character is two indices out per character of it unless it is moved.
  const units = /[\uD800-\uDBFF]/.test(said) ? codeUnits(said) : null;
  const kept = [];
  for (const span of spans.slice(0, MAX_MARKS)) {
    if (!Array.isArray(span) || span.length !== 2) continue;
    const points = units ? units.length - 1 : said.length;
    const first = bound(span[0], points);
    const last = Math.max(first, bound(span[1], points));
    kept.push(units ? [units[first], units[last]] : [first, last]);
  }
  return kept;
}

/**
 * Where each character of a text starts, counted in code units.
 *
 * @param {string} said - The text.
 * @returns {number[]} One index per character, and the text's own length after the last.
 */
function codeUnits(said) {
  const starts = [];
  for (let unit = 0; unit < said.length; unit += 1) {
    starts.push(unit);
    const high = said.charCodeAt(unit);
    if (high >= 0xd800 && high <= 0xdbff && unit + 1 < said.length) {
      const low = said.charCodeAt(unit + 1);
      if (low >= 0xdc00 && low <= 0xdfff) unit += 1;
    }
  }
  starts.push(said.length);
  return starts;
}

/**
 * A value as a string.
 *
 * @param {*} value - Whatever the body carried.
 * @returns {string} The string, empty for anything that is not one.
 */
function text(value) {
  return typeof value === "string" ? value : "";
}

/**
 * A value as a whole number of at least zero.
 *
 * @param {*} value - Whatever the body carried.
 * @returns {number} The number, 0 for anything that is not one.
 */
function whole(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.trunc(number)) : 0;
}

/**
 * An index held inside a length.
 *
 * @param {*} value - The index from the body.
 * @param {number} length - Characters it points into.
 * @returns {number} The index, between 0 and `length`.
 */
function bound(value, length) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(Math.trunc(number), length));
}
