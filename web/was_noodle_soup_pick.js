/**
 * Word picker for the Noodle Soup Pick node.
 *
 * Lists the stored terminology and opens one to its words, several to a row, a window at a
 * time. It writes the `picked` widget, one pick per line.
 */

import { app } from "../../scripts/app.js";
import { elideText } from "./interface/canvas_text.js";
import {
  buildGrid,
  cellColumn,
  cellLeft,
  cellPitch,
  cellWidth,
  gridCell,
  gridColumns,
  gridPosition,
  gridRow,
} from "./interface/grid.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { LABELS, PREVIEW_STATE } from "./interface/preview.js";
import { fetchWithin } from "./interface/request.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes, chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.NoodleSoupPickUI";
const NODE_NAME = "WASNoodleSoupPick";
const SETTING_ID = "WAS.NoodleSoupPick.ShowInterface";

const PICKED_WIDGET = "picked";
const TERM_WIDGET = "term";

const UI_WIDGET_NAME = "was_noodle_soup_pick_ui";
const UI_WIDGET_TYPE = "was_term_picker";

const ROUTE = "/was/interface/api/nsp_pantry";

// The word standing for every word of a terminology, and what divides a name from a word.
const WHOLE = "*";
const SEPARATOR = ":";

// What divides the parts of a key the panel holds its own state under.
const KEY_GAP = "\u0000";

// Rows one answer carries, and how far a window start moves between requests. A window covers
// every position the view can reach before the start moves again, so scrolling costs one
// request per STRIDE rows rather than one per row.
const PAGE = 500;
const STRIDE = 250;

// Word windows held at once, across every terminology. A wide panel in many columns reads
// two or three of them at a time; the rest is room to scroll back.
const HELD_WINDOWS = 12;

// Characters typed before the words themselves are searched rather than the names.
const SEARCH_CHARS = 2;

// How long the listing stands before the pointer arriving over the panel asks again. A run
// writes terminology while the page is open, which is the whole reason this is short.
const STALE_MS = 3000;

const UI_HEIGHT = 232;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// The narrowest the footer stays readable in.
const PANEL_MIN_WIDTH = 280;

// Layout bands, in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const CHIP_HEIGHT = 14;
const CHIP_GAP = 4;
const CHIP_PAD = 5;
const ROW_HEIGHT = 15;
const FOOTER_HEIGHT = 13;
const SCROLLBAR_WIDTH = 5;
const MIN_THUMB = 12;
const MARK_BOX = 8;
const MARK_GAP = 5;
const TWIST_WIDTH = 10;
const WORD_INDENT = 12;
const COUNT_COLUMN = 62;
const GLYPH_GAP = 4;

// Cells a run of words spreads across at most, the gap kept at the right edge of one, the
// room a word added here marks itself in, and the share of a cell a match gives its name.
const MAX_COLUMNS = 8;
const CELL_GAP = 8;
const OWN_WIDTH = 8;
const TAG_SHARE = 0.45;

const BODY_FONT = "10px sans-serif";
const SMALL_FONT = "9px sans-serif";

const MESSAGE_TIMEOUT = 4000;

// Rows one wheel notch moves, and how far the arrow keys move with Shift held.
const WHEEL_ROWS = 3;
const COARSE_STEP = 10;

// Characters a typed filter holds, matching what the route reads.
const FILTER_CHARS = 64;

/** Which terminologies the list shows. One at a time, widest first. */
const MODES = ["all", "yours", "picked"];

/** What the footer's hover says, which is what the panel is and is not responsible for. */
const FOOTER_TITLE =
  "Every tick is a line of the picked box, and clicking writes that box and nothing else: no "
  + "word is added to or taken out of the pantry here. The run's own count is in the band "
  + "below, after a word named twice is dropped and after limit cuts the list.";

/** What the mode chips say on hover. */
const CHIP_TITLE =
  "Which terminologies the list shows. 'yours' keeps the ones holding a word you added, "
  + "'picked' the ones you have ticked something in. This only filters what is on screen; a "
  + "word already ticked stays ticked while its terminology is hidden.";

/** What the pantry being empty means, and the two ways to fill it. */
const NO_PANTRY =
  "The pantry holds no terminology yet. Run Noodle Soup Pantry Refresh, or turn on "
  + "features.network in config.yaml";

/**
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * Test whether one of a node's inputs is linked.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  for (const input of inputs) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

/**
 * Hold a value inside a range.
 *
 * @param {number} value - Value to bound.
 * @param {number} low - Lowest allowed.
 * @param {number} high - Highest allowed.
 * @returns {number} The bounded value.
 */
function clamp(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

/**
 * Read a grid row's kind in the panel's own words.
 *
 * @param {string} kind - `header`, `run` or `tail`.
 * @returns {string} `term`, `word` or `match`.
 */
function rowKind(kind) {
  if (kind === "header") return "term";
  return kind === "run" ? "word" : "match";
}

/**
 * Read whether the panel is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * The first row of the window that covers a view position.
 *
 * @param {number} position - Row wanted, counting from the start of what is being paged.
 * @returns {number} Where the request starts, on a fixed stride so a few rows of scrolling
 *   reuse the window already loaded.
 */
function windowStart(position) {
  return Math.max(0, Math.floor(position / STRIDE) * STRIDE);
}

/**
 * Whether a line of the widget is one the panel accounts for.
 *
 * @param {string} line - One line of the widget.
 * @returns {boolean} True when the line names a pick.
 */
function namesPick(line) {
  const text = String(line ?? "").trim();
  return text !== "" && !text.startsWith("#");
}

/**
 * Read one trimmed line as a terminology and one of its words.
 *
 * @param {string} line - The line, already trimmed.
 * @param {Set<string>} names - The terminology names the pantry holds.
 * @returns {{term: string, entry: string}} The pick, `entry` empty for a whole terminology.
 *   A line whose whole text names a terminology takes it whole, otherwise the split falls at
 *   the first colon whose prefix names one, and failing that at the first colon.
 */
function readPick(line, names) {
  if (names.has(line)) return { term: line, entry: "" };
  let at = -1;
  for (;;) {
    at = line.indexOf(SEPARATOR, at + 1);
    if (at < 0) break;
    if (names.has(line.slice(0, at).trim())) return splitPick(line, at);
  }
  const first = line.indexOf(SEPARATOR);
  if (first < 0) return { term: line, entry: "" };
  return splitPick(line, first);
}

/**
 * Divide one line at a colon.
 *
 * @param {string} line - The line, already trimmed.
 * @param {number} at - Where the dividing colon sits.
 * @returns {{term: string, entry: string}} The pick, `entry` empty for the whole word.
 */
function splitPick(line, at) {
  const term = line.slice(0, at).trim();
  const entry = line.slice(at + 1).trim();
  return { term, entry: entry === WHOLE ? "" : entry };
}

/**
 * Read one listing answer into the shape the panel draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The listing, or null when the body is not one.
 */
function normaliseListing(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.terms)) return null;
  const terms = [];
  for (const row of data.terms) {
    const name = typeof row?.name === "string" ? row.name : "";
    if (!name) continue;
    terms.push({
      name,
      entries: Math.max(0, Math.trunc(Number(row?.entries) || 0)),
      own: Math.max(0, Math.trunc(Number(row?.own) || 0)),
    });
  }
  return {
    terms,
    entries: Math.max(0, Math.trunc(Number(data.entries) || 0)),
    generation: typeof data.generation === "string" ? data.generation : "",
  };
}

/**
 * Read one terminology answer into the shape the panel draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The window, or null when the body is not one. A word is trimmed,
 *   which is the text a picked line carries and the text the run reads back out of it.
 */
function normaliseWords(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.entries)) return null;
  return {
    name: typeof data.name === "string" ? data.name : "",
    start: Math.max(0, Math.trunc(Number(data.start) || 0)),
    rows: data.entries.map((row) => ({
      text: typeof row?.text === "string" ? row.text.trim() : "",
      own: row?.own === true,
    })),
    total: Math.max(0, Math.trunc(Number(data.total) || 0)),
  };
}

/**
 * Read one search answer into the shape the panel draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The window, or null when the body is not one. A match is trimmed,
 *   which is the text a picked line carries and the text the run reads back out of it.
 */
function normaliseMatches(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.matches)) return null;
  return {
    search: typeof data.search === "string" ? data.search : "",
    start: Math.max(0, Math.trunc(Number(data.start) || 0)),
    rows: data.matches.map((row) => ({
      term: typeof row?.term === "string" ? row.term : "",
      text: typeof row?.text === "string" ? row.text.trim() : "",
      own: row?.own === true,
    })),
    total: Math.max(0, Math.trunc(Number(data.total) || 0)),
    truncated: data.truncated === true,
  };
}

/**
 * Work out where each band of the panel sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the chips, the list, the word columns and the footer,
 *   with the number of whole rows the list holds.
 */
function computeLayout(width, height) {
  const x0 = PAD_X;
  const x1 = Math.max(x0 + 1, width - PAD_X);
  const chipsY = PAD_Y;
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT);
  const rowsY = chipsY + CHIP_HEIGHT + CHIP_GAP;
  const rowsHeight = Math.max(ROW_HEIGHT, footerY - rowsY - 2);
  return {
    width,
    height,
    x0,
    x1,
    chipsY,
    chipHeight: CHIP_HEIGHT,
    rowsY,
    rowsHeight,
    gridX: x0 + WORD_INDENT,
    // The scrollbar's gutter is left out of the columns whether the bar is drawn or not.
    gridRight: Math.max(x0 + WORD_INDENT + 1, x1 - SCROLLBAR_WIDTH - 2),
    // Whole rows only, so a word is never cut across the middle by the band's edge.
    rows: Math.max(1, Math.floor(rowsHeight / ROW_HEIGHT)),
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * Build the word picker for one node.
 *
 * @param {object} node - The node the panel decorates.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   schedulePaint: () => void, handlePickedChanged: () => void, refresh: () => void,
 *   dispose: () => void}} The panel for `appendInterfaceWidget`, a repaint, a reload, and
 *   teardown.
 */
function createTermPicker(node) {
  const root = document.createElement("div");
  root.tabIndex = 0;
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    `min-height:${ELEMENT_MIN_HEIGHT}px`,
    "overflow:hidden",
    "outline:none",
    "touch-action:none",
    "user-select:none",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%";
  root.appendChild(canvas);

  // The glyph, the chips and the footer state what they are worth through the element's own
  // title. The regions are handed over again on every repaint, since they move whenever the
  // node is resized.
  const titles = hoverTitles(root);

  const state = {
    listing: null,
    status: PREVIEW_STATE.LOADING,
    fetchedAt: 0,
    token: 0,
    listingPending: false,
    // Which terminologies are drawn open, the windows their words arrived in, and the
    // requests still in flight. All of it is the panel's own and reaches neither the widget
    // nor the workflow, so a reopened workflow draws every terminology collapsed.
    open: new Set(),
    words: new Map(),
    wordsNeeded: new Set(),
    wordsPending: new Set(),
    matches: null,
    matchesKey: "",
    matchesPending: "",
    view: 0,
    // Which cell the keyboard is on, and the column a move up or down keeps aiming at.
    caret: { position: 0, column: 0 },
    caretWant: 0,
    hover: null,
    press: null,
    chips: [],
    mode: MODES[0],
    filter: "",
    message: "",
    noteAction: null,
    noteBox: null,
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    // Cells a run of words spreads across, the width one of them wants, and what that width
    // was measured for.
    columns: 1,
    natural: 0,
    naturalKey: "",
    picks: null,
    picksFrom: null,
    disposed: false,
  };

  /**
   * Read the `picked` widget.
   *
   * @returns {string} Its value, empty when it cannot be read.
   */
  function pickedValue() {
    const value = findWidget(node, PICKED_WIDGET)?.value;
    return typeof value === "string" ? value : "";
  }

  /**
   * The lines of the widget, exactly as it holds them.
   *
   * @returns {string[]} One entry per line, nothing trimmed.
   */
  function widgetLines() {
    return pickedValue().split("\n");
  }

  /**
   * The terminology names the listing holds.
   *
   * @returns {Set<string>} The names, empty while no listing has arrived.
   */
  function knownNames() {
    return new Set((state.listing?.terms ?? []).map((row) => row.name));
  }

  /**
   * What the picked box names, read against the listing.
   *
   * @returns {{byTerm: Map<string, {whole: number[], words: Map<string, number>}>,
   *   lines: number, unknown: number[], words: number, terms: number}} Where each pick sits
   *   in the widget, how many lines name a pick at all, which line numbers name a
   *   terminology the listing does not hold, and how many words and terminologies are
   *   ticked.
   */
  function picked() {
    const value = pickedValue();
    if (state.picks && state.picksFrom === value) return state.picks;
    const names = knownNames();
    const counts = new Map((state.listing?.terms ?? []).map((row) => [row.name, row.entries]));
    const byTerm = new Map();
    const unknown = [];
    let lines = 0;
    widgetLines().forEach((line, index) => {
      if (!namesPick(line)) return;
      lines += 1;
      const pick = readPick(line.trim(), names);
      if (!names.has(pick.term) && state.listing) unknown.push(index);
      let held = byTerm.get(pick.term);
      if (!held) {
        held = { whole: [], words: new Map() };
        byTerm.set(pick.term, held);
      }
      if (pick.entry) {
        if (!held.words.has(pick.entry)) held.words.set(pick.entry, index);
      } else {
        held.whole.push(index);
      }
    });
    let words = 0;
    for (const [term, held] of byTerm) {
      words += held.whole.length ? (counts.get(term) ?? 0) : held.words.size;
    }
    state.picks = { byTerm, lines, unknown, words, terms: byTerm.size };
    state.picksFrom = value;
    return state.picks;
  }

  /**
   * How many words of one terminology are ticked.
   *
   * @param {object} chosen - The reading from `picked`.
   * @param {object} term - One row of the listing.
   * @returns {number} The count, which equals the terminology's own count while it is whole.
   */
  function tickedIn(chosen, term) {
    const held = chosen.byTerm.get(term.name);
    if (!held) return 0;
    return held.whole.length ? term.entries : held.words.size;
  }

  /**
   * Whether the typed filter is long enough to search the words themselves.
   *
   * @returns {boolean} True at :data:`SEARCH_CHARS` characters or more.
   */
  function searching() {
    return state.filter.length >= SEARCH_CHARS;
  }

  /**
   * The terminologies on screen, after the mode chip and the typed filter.
   *
   * @returns {object[]} Rows of the listing, in the order the route gave them.
   */
  function visibleTerms() {
    const terms = state.listing?.terms ?? [];
    const chosen = picked();
    const needle = state.filter.toLowerCase();
    return terms.filter((term) => {
      if (state.mode === "yours" && !term.own) return false;
      if (state.mode === "picked" && !chosen.byTerm.has(term.name)) return false;
      return !needle || term.name.toLowerCase().includes(needle);
    });
  }

  /**
   * The flat list of rows, as the sections it is built from rather than as the rows.
   *
   * @param {number} [columns] - Cells a run of words spreads across.
   * @returns {{search: boolean, terms: object[], sections: object[], grid: object,
   *   total: number, matchTotal: number, names: number}} The terminologies on screen, where
   *   each one's rows begin, the row and column arithmetic, and how many rows there are
   *   altogether.
   */
  function model(columns = state.columns) {
    // A terminology holding 2265 words is one section carrying its count, so nothing here
    // allocates a row per word.
    const terms = visibleTerms();
    const search = searching();
    const opens = terms.map((term) => !search && state.open.has(term.name));
    const runs = terms.map((term, at) => ({ count: opens[at] ? term.entries : 0 }));
    const matchTotal = search ? (state.matches?.total ?? 0) : 0;
    const grid = buildGrid(runs, matchTotal, columns);
    const sections = grid.sections.map((row, at) => ({
      ...row,
      term: terms[at],
      open: opens[at],
    }));
    return { search, terms, sections, grid, total: grid.total, matchTotal, names: terms.length };
  }

  /**
   * The row at one flat position.
   *
   * @param {object} shape - The reading from `model`.
   * @param {number} position - Flat row position.
   * @returns {object|null} The row from `gridRow` with `kind` as `term`, `word` or `match`,
   *   the terminology it belongs to, and `first` and `span` naming the words on it. Null when
   *   the position is on no row.
   */
  function rowAt(shape, position) {
    const row = gridRow(shape.grid, position);
    if (!row) return null;
    return { ...row, kind: rowKind(row.kind), position, term: termOf(shape, row.section) };
  }

  /**
   * The word at one row and column.
   *
   * @param {object} shape - The reading from `model`.
   * @param {number} position - Flat row position.
   * @param {number} column - Which cell across, ignored on a terminology row.
   * @returns {object|null} The row with `index` naming the word, or null when the cell holds
   *   nothing.
   */
  function cellAt(shape, position, column) {
    const cell = gridCell(shape.grid, position, column);
    if (!cell) return null;
    return { ...cell, kind: rowKind(cell.kind), term: termOf(shape, cell.section) };
  }

  /**
   * Where one cell sits in another reading of the same list.
   *
   * @param {object} shape - The reading from `model` to look in.
   * @param {object|null} cell - The cell from `cellAt`.
   * @returns {{position: number, column: number}|null} Its row and column, or null when that
   *   reading does not hold it.
   */
  function positionOf(shape, cell) {
    if (!cell) return null;
    if (cell.kind === "term") {
      return gridPosition(shape.grid, { kind: "header", section: cell.section });
    }
    if (cell.kind === "match") {
      return gridPosition(shape.grid, { kind: "tail", index: cell.index });
    }
    return gridPosition(shape.grid, { kind: "run", section: cell.section, index: cell.index });
  }

  /**
   * The terminology a section stands for.
   *
   * @param {object} shape - The reading from `model`.
   * @param {number} section - Which section, or -1 for the search matches.
   * @returns {object|null} One row of the listing, or null for a match.
   */
  function termOf(shape, section) {
    return section >= 0 ? (shape.terms[section] ?? null) : null;
  }

  /**
   * What one window of a terminology's words is held under.
   *
   * @param {string} name - The terminology name.
   * @param {number} start - First word of the window.
   * @returns {string} The key into `state.words`.
   */
  function windowKey(name, start) {
    return `${name}${KEY_GAP}${start}`;
  }

  /**
   * One word of an open terminology, from the window it arrived in.
   *
   * @param {string} name - The terminology name.
   * @param {number} index - Which word, counting from the start of the terminology.
   * @returns {{text: string, own: boolean}|null} The word, or null while its window is
   *   still in flight.
   */
  function wordAt(name, index) {
    const base = windowStart(index);
    // Windows overlap by a stride, so the one before this word's own also covers it.
    for (const start of [base, base - STRIDE]) {
      if (start < 0) continue;
      const held = state.words.get(windowKey(name, start));
      const row = held?.rows[index - held.start];
      if (row) return row;
    }
    return null;
  }

  /**
   * The windows held for the terminologies drawn open.
   *
   * @param {object} shape - The reading from `model`.
   * @returns {Array<[string, object]>} Each window's key with the answer it arrived in.
   */
  function openWindows(shape) {
    const open = new Set();
    for (const section of shape.sections) {
      if (section.open) open.add(section.term.name);
    }
    if (!open.size) return [];
    return [...state.words].filter(([, held]) => open.has(held.name));
  }

  /**
   * One match of the search, from the window it arrived in.
   *
   * @param {number} index - Which match, counting from the first.
   * @returns {{term: string, text: string, own: boolean}|null} The match, or null while
   *   its window is still in flight.
   */
  function matchAt(index) {
    const held = state.matches;
    if (!held) return null;
    return held.rows[index - held.start] ?? null;
  }

  /**
   * What the column width was measured for.
   *
   * @param {object} shape - The reading from `model`.
   * @returns {string} A key naming the search, or the terminologies drawn open.
   */
  function measuredFrom(shape) {
    if (shape.search) return `${KEY_GAP}search ${state.filter}`;
    return shape.sections
      .filter((section) => section.open)
      .map((section) => section.term.name)
      .join(KEY_GAP);
  }

  /**
   * The width one word cell wants, measured once for the words on show.
   *
   * @param {CanvasRenderingContext2D} ctx - Context carrying the fonts the cells draw in.
   * @param {object} shape - The reading from `model`.
   * @returns {number} Element pixels, zero while no word has arrived. The measurement stands
   *   until the search or the terminologies drawn open change.
   */
  function naturalCell(ctx, shape) {
    const key = measuredFrom(shape);
    if (key === state.naturalKey) return state.natural;
    let text = 0;
    let tag = 0;
    ctx.font = BODY_FONT;
    if (shape.search) {
      for (const row of state.matches?.rows ?? []) {
        text = Math.max(text, ctx.measureText(row.text).width);
      }
      ctx.font = SMALL_FONT;
      for (const row of state.matches?.rows ?? []) {
        tag = Math.max(tag, ctx.measureText(row.term).width);
      }
    } else {
      for (const [, held] of openWindows(shape)) {
        for (const row of held.rows) text = Math.max(text, ctx.measureText(row.text).width);
      }
    }
    if (text <= 0) return 0;
    state.naturalKey = key;
    state.natural = MARK_BOX + MARK_GAP + Math.ceil(text) + OWN_WIDTH + CELL_GAP
      + (tag > 0 ? MARK_GAP + Math.min(COUNT_COLUMN, Math.ceil(tag)) : 0);
    return state.natural;
  }

  /**
   * How many cells a run of words spreads across at the width the panel now has.
   *
   * @param {CanvasRenderingContext2D} ctx - Context carrying the fonts the cells draw in.
   * @param {object} shape - The reading from `model`.
   * @returns {number} Columns, the count already in use while nothing has been measured.
   */
  function measureColumns(ctx, shape) {
    const natural = naturalCell(ctx, shape);
    if (natural <= 0) return state.columns;
    return gridColumns(state.layout.gridRight - state.layout.gridX, natural, MAX_COLUMNS);
  }

  /**
   * Where one cell of a row sits.
   *
   * @param {object} row - The row from `rowAt`.
   * @param {number} column - Which cell across.
   * @param {number} y - Top of the row in element pixels.
   * @param {number} right - Where a full width row ends.
   * @returns {{x: number, y: number, width: number, height: number}} Its rectangle.
   */
  function cellBox(row, column, y, right) {
    const layout = state.layout;
    if (row.kind === "term") {
      return { x: layout.x0, y, width: Math.max(1, right - layout.x0), height: ROW_HEIGHT };
    }
    const pitch = cellPitch(layout.gridRight - layout.gridX, state.columns);
    return {
      x: cellLeft(layout.gridX, pitch, column),
      y,
      width: cellWidth(layout.gridX, pitch, column),
      height: ROW_HEIGHT,
    };
  }

  /**
   * Whether two cells are the same one.
   *
   * @param {{position: number, column: number}|null} one - A cell, or nothing.
   * @param {{position: number, column: number}|null} other - The other.
   * @returns {boolean} True when both are nothing, or both name the same cell.
   */
  function sameCell(one, other) {
    if (!one || !other) return !one && !other;
    return one.position === other.position && one.column === other.column;
  }

  /**
   * Whether two cells name the same thing to pick.
   *
   * @param {object|null} one - A cell from `cellAt`, or nothing.
   * @param {object|null} other - The other.
   * @returns {boolean} True when both name the same terminology, word or match.
   */
  function sameItem(one, other) {
    if (!one || !other) return false;
    return one.kind === other.kind && one.section === other.section && one.index === other.index;
  }

  /**
   * Show a short note in the footer.
   *
   * @param {string} text - Note to show.
   * @returns {void}
   */
  function setMessage(text) {
    state.message = text;
    if (state.messageTimer) clearTimeout(state.messageTimer);
    state.messageTimer = setTimeout(() => {
      state.messageTimer = 0;
      state.message = "";
      schedulePaint();
    }, MESSAGE_TIMEOUT);
    schedulePaint();
  }

  /**
   * Ask the route for the terminology listing.
   *
   * @param {boolean} [force] - Ask again for a listing already held, which is how a
   *   terminology a run has just added reaches the panel.
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function loadListing(force = false) {
    if (state.disposed || state.listingPending) return;
    if (!force && state.listing) return;
    state.listingPending = true;
    if (!state.listing) state.status = PREVIEW_STATE.LOADING;
    const token = (state.token += 1);
    try {
      const response = await fetchWithin(ROUTE, {
        // A run writes terminology while the page is open, so a copy held by the browser
        // would claim to be what the pantry holds and would not be.
        cache: "no-store",
      });
      if (state.disposed || token !== state.token) return;
      const payload = response.ok ? normaliseListing(await response.json()) : null;
      if (state.disposed || token !== state.token) return;
      if (!payload) {
        state.status = PREVIEW_STATE.FAILED;
        return;
      }
      // A pantry written since the last answer numbers the same words differently, so every
      // window held against the older stamp is dropped rather than drawn.
      if (state.listing && state.listing.generation !== payload.generation) {
        state.words.clear();
        state.matches = null;
        state.matchesKey = "";
        state.naturalKey = "";
      }
      state.listing = payload;
      state.status = PREVIEW_STATE.READY;
      state.picks = null;
      state.view = clampView(state.view);
    } catch (error) {
      if (state.disposed || token !== state.token) return;
      console.error(`[${EXT_NAME}] Failed to read the terminology listing:`, error);
      state.status = PREVIEW_STATE.FAILED;
    } finally {
      if (!state.disposed) {
        state.listingPending = false;
        state.fetchedAt = Date.now();
        schedulePaint();
      }
    }
  }

  /**
   * Ask the route for one window of a terminology's words.
   *
   * @param {string} name - The terminology name.
   * @param {number} start - First word of the window.
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function loadWords(name, start) {
    const key = windowKey(name, start);
    if (state.disposed || state.wordsPending.has(key)) return;
    state.wordsPending.add(key);
    const token = state.token;
    const query = `${ROUTE}?term=${encodeURIComponent(name)}&start=${start}&limit=${PAGE}`;
    try {
      const response = await fetchWithin(query, { cache: "no-store" });
      if (state.disposed || token !== state.token) return;
      const payload = response.ok ? normaliseWords(await response.json()) : null;
      if (state.disposed || token !== state.token || !payload) return;
      state.words.set(key, payload);
      releaseWindows();
    } catch (error) {
      if (state.disposed || token !== state.token) return;
      console.error(`[${EXT_NAME}] Failed to read the words of \`${name}\`:`, error);
    } finally {
      if (!state.disposed) {
        state.wordsPending.delete(key);
        schedulePaint();
      }
    }
  }

  /**
   * Ask the route for one window of the search.
   *
   * @param {string} needle - What was typed.
   * @param {number} start - First match of the window.
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function loadMatches(needle, start) {
    const key = `${needle} ${start}`;
    if (state.disposed || state.matchesPending === key) return;
    state.matchesPending = key;
    const token = state.token;
    const query = `${ROUTE}?search=${encodeURIComponent(needle)}&start=${start}&limit=${PAGE}`;
    try {
      const response = await fetchWithin(query, { cache: "no-store" });
      if (state.disposed || token !== state.token) return;
      const payload = response.ok ? normaliseMatches(await response.json()) : null;
      if (state.disposed || token !== state.token || !payload) return;
      if (payload.search !== needle) return;
      state.matches = payload;
      state.matchesKey = key;
    } catch (error) {
      if (state.disposed || token !== state.token) return;
      console.error(`[${EXT_NAME}] Failed to search the pantry:`, error);
    } finally {
      if (!state.disposed) {
        if (state.matchesPending === key) state.matchesPending = "";
        schedulePaint();
      }
    }
  }

  /**
   * Drop the windows held past the bound, oldest first, keeping every one on screen.
   *
   * @returns {void}
   */
  function releaseWindows() {
    while (state.words.size > HELD_WINDOWS) {
      let oldest = null;
      for (const key of state.words.keys()) {
        if (state.wordsNeeded.has(key)) continue;
        oldest = key;
        break;
      }
      if (oldest === null) return;
      state.words.delete(oldest);
    }
  }

  /**
   * Ask for whatever window the rows on screen need, and nothing else.
   *
   * @param {object} shape - The reading from `model`.
   * @returns {void}
   */
  function ensureWindows(shape) {
    if (!state.listing) return;
    const wanted = new Map();
    const needed = new Set();
    for (let offset = 0; offset < state.layout.rows; offset += 1) {
      const row = rowAt(shape, state.view + offset);
      if (!row) break;
      if (row.kind === "word") {
        for (let at = row.first; at < row.first + row.span; at += 1) {
          const start = windowStart(at);
          needed.add(windowKey(row.term.name, start));
          if (start >= STRIDE) needed.add(windowKey(row.term.name, start - STRIDE));
          if (wordAt(row.term.name, at)) continue;
          wanted.set(windowKey(row.term.name, start), { name: row.term.name, start });
        }
      } else if (row.kind === "match") {
        for (let at = row.first; at < row.first + row.span; at += 1) {
          if (matchAt(at)) continue;
          const start = windowStart(at);
          const key = `${state.filter} ${start}`;
          if (state.matchesKey !== key) {
            loadMatches(state.filter, start).catch((error) => {
              console.error(`[${EXT_NAME}] Failed to ask for the search:`, error);
            });
          }
          break;
        }
      }
    }
    state.wordsNeeded = needed;
    releaseWindows();
    for (const [, ask] of wanted) {
      loadWords(ask.name, ask.start).catch((error) => {
        console.error(`[${EXT_NAME}] Failed to ask for a terminology's words:`, error);
      });
    }
    // A search whose first window has never been asked for draws nothing at all, so the ask
    // does not wait for a row to be missing from a window that is not there.
    if (shape.search && !state.matches) {
      const key = `${state.filter} 0`;
      if (state.matchesKey !== key && state.matchesPending !== key) {
        loadMatches(state.filter, 0).catch((error) => {
          console.error(`[${EXT_NAME}] Failed to ask for the search:`, error);
        });
      }
    }
  }

  /**
   * Ask for the listing, reporting a failure rather than throwing it.
   *
   * @returns {void}
   */
  function refresh() {
    loadListing(true).catch((error) => {
      console.error(`[${EXT_NAME}] Failed to ask for the terminology listing:`, error);
    });
  }

  /**
   * Hold a view position inside the rows on screen.
   *
   * @param {number} view - Wanted first row on screen.
   * @returns {number} The bounded position.
   */
  function clampView(view) {
    const total = model().total;
    return clamp(Math.trunc(view), 0, Math.max(0, total - state.layout.rows));
  }

  /**
   * Scroll the list.
   *
   * @param {number} view - Wanted first row on screen.
   * @returns {boolean} True when the list moved, which decides whether a wheel gesture
   *   belongs here or to the graph underneath.
   */
  function setView(view) {
    const next = clampView(view);
    if (next === state.view) return false;
    state.view = next;
    schedulePaint();
    return true;
  }

  /**
   * Bring the caret into view.
   *
   * @returns {void}
   */
  function reveal() {
    const rows = state.layout.rows;
    const at = state.caret.position;
    if (at < state.view) setView(at);
    else if (at > state.view + rows - 1) setView(at - rows + 1);
  }

  /**
   * Put the caret on one cell, holding the column a move up or down aims at.
   *
   * @param {object} shape - The reading from `model`.
   * @param {number} position - Flat row position.
   * @param {number} want - Column the caret is aiming at.
   * @returns {void}
   */
  function setCaret(shape, position, want) {
    state.caret = { position, column: caretColumn(rowAt(shape, position), want) };
    reveal();
    schedulePaint();
  }

  /**
   * Which cell of a row a caret aiming at one column lands on.
   *
   * @param {object|null} row - The row from `rowAt`.
   * @param {number} want - Column the caret is aiming at.
   * @returns {number} The column, zero on a terminology row and on no row at all.
   */
  function caretColumn(row, want) {
    if (!row || row.kind === "term") return 0;
    return clamp(want, 0, Math.max(0, row.span - 1));
  }

  /**
   * The cell the caret is on.
   *
   * @param {object} shape - The reading from `model`.
   * @returns {object|null} The cell from `cellAt`, or null when the caret is on no row.
   */
  function caretCell(shape) {
    const row = rowAt(shape, state.caret.position);
    if (!row) return null;
    return cellAt(shape, state.caret.position, caretColumn(row, state.caret.column));
  }

  /**
   * Write the `picked` widget once.
   *
   * @param {string[]} lines - The lines the widget should hold.
   * @returns {void}
   */
  function writeLines(lines) {
    const widget = findWidget(node, PICKED_WIDGET);
    if (!widget) return;
    const value = lines.join("\n");
    if (String(widget.value ?? "") === value) return;
    // The write is bracketed in the canvas change events the graph's change tracker listens
    // for, so a tick made from the keyboard gets an undo entry of its own. A commit that
    // rides on a pointerup is snapshotted by the tracker's own mouseup as well, and finds
    // nothing left to record.
    const canvas = app.canvas;
    const transactional = typeof canvas?.emitBeforeChange === "function"
      && typeof canvas?.emitAfterChange === "function";
    if (transactional) canvas.emitBeforeChange();
    try {
      widget.value = value;
    } finally {
      if (transactional) canvas.emitAfterChange();
    }
    state.picks = null;
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Put one line into the widget, writing over the empty last line where there is one.
   *
   * @param {string[]} lines - The lines as the widget holds them.
   * @param {string} line - The line to add.
   * @returns {string[]} The lines with it added.
   */
  function withLine(lines, line) {
    // The one line that is written over is the empty last line a text box holds the moment
    // somebody has pressed Enter in it.
    if (lines.length && lines[lines.length - 1].trim() === "") {
      const next = lines.slice();
      next[next.length - 1] = line;
      return next;
    }
    return [...lines, line];
  }

  /**
   * Drop lines from the widget by position.
   *
   * @param {string[]} lines - The lines as the widget holds them.
   * @param {Iterable<number>} positions - Which lines to drop.
   * @returns {string[]} The lines that are left, in order.
   */
  function withoutLines(lines, positions) {
    const dropping = new Set(positions);
    return lines.filter((_line, index) => !dropping.has(index));
  }

  /**
   * Tick or untick one word of one terminology.
   *
   * @param {string} name - The terminology name.
   * @param {string} word - The word.
   * @returns {void}
   */
  function toggleWord(name, word) {
    if (state.disposed || !word) return;
    const chosen = picked();
    const held = chosen.byTerm.get(name);
    if (held?.whole.length) {
      setMessage(`${name} is taken whole; untick it on its own row to pick words`);
      return;
    }
    const lines = widgetLines();
    if (held?.words.has(word)) {
      writeLines(withoutLines(lines, [held.words.get(word)]));
      setMessage(`removed ${word}`);
      return;
    }
    const entries = (state.listing?.terms ?? []).find((row) => row.name === name)?.entries ?? 0;
    const wanted = (held?.words.size ?? 0) + 1;
    if (entries > 0 && wanted >= entries) {
      // Every word of the terminology is now ticked, so the whole set is one line rather
      // than one line per word.
      const rest = withoutLines(lines, held?.words.values() ?? []);
      writeLines(withLine(rest, `${name}${SEPARATOR} ${WHOLE}`));
      setMessage(`added every word of ${name}`);
      return;
    }
    writeLines(withLine(lines, `${name}${SEPARATOR} ${word}`));
    setMessage(`added ${word}`);
  }

  /**
   * Take a whole terminology, or clear whatever was ticked in it.
   *
   * @param {object} term - One row of the listing.
   * @returns {void}
   */
  function toggleTerm(term) {
    if (state.disposed) return;
    const chosen = picked();
    const held = chosen.byTerm.get(term.name);
    const lines = widgetLines();
    if (held?.whole.length) {
      writeLines(withoutLines(lines, held.whole));
      setMessage(`removed ${term.name}`);
      return;
    }
    if (held?.words.size) {
      writeLines(withoutLines(lines, held.words.values()));
      setMessage(`cleared ${held.words.size} word(s) of ${term.name}`);
      return;
    }
    writeLines(withLine(lines, `${term.name}${SEPARATOR} ${WHOLE}`));
    setMessage(`added every word of ${term.name}`);
  }

  /**
   * Drop every picked line naming a terminology the pantry does not hold.
   *
   * @returns {void}
   */
  function dropUnknown() {
    const chosen = picked();
    if (!chosen.unknown.length) return;
    const count = chosen.unknown.length;
    writeLines(withoutLines(widgetLines(), chosen.unknown));
    setMessage(`removed ${count} line(s) naming no terminology`);
  }

  /**
   * Open or close one terminology.
   *
   * @param {string} name - The terminology name.
   * @param {boolean} [open] - What to set it to. Left out, it is flipped.
   * @returns {void}
   */
  function setOpen(name, open) {
    const wanted = open === undefined ? !state.open.has(name) : open;
    if (wanted) state.open.add(name);
    else state.open.delete(name);
    state.view = clampView(state.view);
    schedulePaint();
  }

  /**
   * What the panel is worth against the run, as a glyph and the measurement behind it.
   *
   * @param {object} chosen - The reading from `picked`.
   * @returns {{icon: string, detail: string}} The claim for `iconTitle`.
   */
  function readClaim(chosen) {
    if (inputLinked(node, PICKED_WIDGET)) {
      return {
        icon: ICON.WARNING,
        detail: "the picked input is filled by a link, so the run reads whatever arrives "
          + "there and the ticks here are not read at all",
      };
    }
    const notes = [];
    if (!state.listing) {
      notes.push(
        "the terminology listing has not arrived, so a tick cannot be drawn against a "
        + "terminology this panel does not hold",
      );
    }
    if (chosen.unknown.length) {
      notes.push(
        `${chosen.unknown.length} picked line(s) name a terminology the pantry does not `
        + "hold, and they are left exactly as they are",
      );
    }
    if (inputLinked(node, TERM_WIDGET)) {
      notes.push(
        "the term input is linked, so the run also takes whatever terminology arrives there, "
        + "which is not known until the run",
      );
    }
    if (notes.length) return { icon: ICON.APPROXIMATE, detail: notes.join("; ") };
    return {
      icon: ICON.EXACT,
      detail: "every tick is a line of the picked box the run reads, in the order the run "
        + "reads them",
    };
  }

  /**
   * The state drawn in place of the rows.
   *
   * @param {object} shape - The reading from `model`.
   * @returns {string} The words, empty when there are rows to draw.
   */
  function blockingText(shape) {
    if (inputLinked(node, PICKED_WIDGET)) {
      return "picked is linked, so the ticks here are not read";
    }
    if (state.status === PREVIEW_STATE.FAILED) return "The pantry could not be read";
    if (!state.listing) return LABELS[PREVIEW_STATE.LOADING];
    if (!state.listing.terms.length) return NO_PANTRY;
    if (!shape.total) {
      if (shape.search) return `No word or terminology matches "${state.filter}"`;
      if (state.filter) return `No terminology matches "${state.filter}"`;
      if (state.mode === "yours") return "No terminology holds a word added here";
      if (state.mode === "picked") return "Nothing is ticked yet; click 'all' to pick a word";
      return NO_PANTRY;
    }
    return "";
  }

  /**
   * Where the scrollbar and its thumb sit.
   *
   * @param {number} total - Rows in the list.
   * @returns {object|null} The track, the thumb and the range they cover, or null when every
   *   row is already visible.
   */
  function scrollGeometry(total) {
    const layout = state.layout;
    const rows = layout.rows;
    if (total <= rows) return null;
    const height = rows * ROW_HEIGHT;
    const thumb = Math.max(MIN_THUMB, Math.round((rows / total) * height));
    const span = Math.max(1, height - thumb);
    const maxView = Math.max(1, total - rows);
    return {
      x: layout.x1 - SCROLLBAR_WIDTH,
      y: layout.rowsY,
      width: SCROLLBAR_WIDTH,
      height,
      thumbY: layout.rowsY + Math.round((clamp(state.view, 0, maxView) / maxView) * span),
      thumbHeight: thumb,
      span,
      maxView,
    };
  }

  /**
   * Read the pointer position in element pixels.
   *
   * @param {PointerEvent|MouseEvent} event - Event to read.
   * @returns {{x: number, y: number}} Position inside the element.
   */
  function localPoint(event) {
    return elementPoint(root, event);
  }

  /**
   * Find the cell under a point.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @param {object} shape - The reading from `model`.
   * @returns {object|null} The cell from `cellAt`, or null when the point is on none.
   */
  function hitCell(point, shape) {
    const layout = state.layout;
    if (point.x < layout.x0 || point.x > layout.x1) return null;
    if (point.y < layout.rowsY || point.y >= layout.rowsY + layout.rows * ROW_HEIGHT) return null;
    const position = state.view + Math.floor((point.y - layout.rowsY) / ROW_HEIGHT);
    const row = rowAt(shape, position);
    if (!row) return null;
    if (row.kind === "term") return cellAt(shape, position, 0);
    const pitch = cellPitch(layout.gridRight - layout.gridX, state.columns);
    const column = cellColumn(point.x, layout.gridX, pitch, state.columns);
    return column === null ? null : cellAt(shape, position, column);
  }

  /**
   * Whether a point is on the footer note, which is the only clickable part of the footer.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {boolean} True while the note is drawn there.
   */
  function onNote(point) {
    const box = state.noteBox;
    if (!box) return false;
    return point.x >= box.x && point.x <= box.x + box.width
      && point.y >= box.y && point.y <= box.y + box.height;
  }

  /**
   * Find the chip under a point.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @param {object[]} chips - The chips from `drawChips`.
   * @returns {object|null} The chip, or null.
   */
  function hitChip(point, chips) {
    for (const chip of chips) {
      if (
        point.x >= chip.x && point.x <= chip.x + chip.width
        && point.y >= chip.y && point.y <= chip.y + chip.height
      ) {
        return chip;
      }
    }
    return null;
  }

  /**
   * Draw the mode chips, and answer where each one is.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @returns {object[]} One box per chip, for hit testing and hover text.
   */
  function drawChips(ctx, theme, chosen) {
    const layout = state.layout;
    const terms = state.listing?.terms ?? [];
    const counts = {
      all: terms.length,
      yours: terms.filter((term) => term.own).length,
      picked: terms.filter((term) => chosen.byTerm.has(term.name)).length,
    };
    const chips = [];
    ctx.font = SMALL_FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    let x = layout.x0;
    for (const mode of MODES) {
      const text = `${mode} ${counts[mode]}`;
      const width = Math.ceil(ctx.measureText(text).width) + CHIP_PAD * 2;
      if (x + width > layout.x1) break;
      const on = state.mode === mode;
      ctx.globalAlpha = on ? 0.22 : 0.08;
      ctx.fillStyle = on ? theme.accent : theme.fg;
      ctx.fillRect(x, layout.chipsY, width, CHIP_HEIGHT);
      ctx.globalAlpha = 1;
      ctx.fillStyle = on ? theme.fg : theme.fgDisabled;
      ctx.fillText(text, x + CHIP_PAD, layout.chipsY + CHIP_HEIGHT / 2);
      chips.push({ mode, x, y: layout.chipsY, width, height: CHIP_HEIGHT });
      x += width + CHIP_GAP;
    }
    if (state.filter) {
      ctx.textAlign = "right";
      ctx.fillStyle = searching() ? theme.accent : theme.fgMuted;
      ctx.fillText(`"${state.filter}"`, layout.x1, layout.chipsY + CHIP_HEIGHT / 2);
    }
    return chips;
  }

  /**
   * Draw a tick box.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {number} x - Left edge in element pixels.
   * @param {number} y - Top edge in element pixels.
   * @param {string} fill - `none`, `part` or `full`.
   * @returns {void}
   */
  function drawBox(ctx, theme, x, y, fill) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = fill === "none" ? theme.border : theme.accent;
    ctx.strokeRect(x + 0.5, y + 0.5, MARK_BOX - 1, MARK_BOX - 1);
    if (fill === "full") {
      ctx.fillStyle = theme.accent;
      ctx.fillRect(x + 2, y + 2, MARK_BOX - 4, MARK_BOX - 4);
    } else if (fill === "part") {
      ctx.fillStyle = theme.accent;
      ctx.fillRect(x + 2, y + MARK_BOX / 2 - 1, MARK_BOX - 4, 2);
    }
  }

  /**
   * Draw a disclosure triangle.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {number} x - Left edge in element pixels.
   * @param {number} y - Middle of the row in element pixels.
   * @param {boolean} open - Whether the terminology is open.
   * @returns {void}
   */
  function drawTwist(ctx, theme, x, y, open) {
    ctx.fillStyle = theme.fgMuted;
    ctx.beginPath();
    if (open) {
      ctx.moveTo(x, y - 2);
      ctx.lineTo(x + 6, y - 2);
      ctx.lineTo(x + 3, y + 3);
    } else {
      ctx.moveTo(x + 1, y - 3);
      ctx.lineTo(x + 6, y);
      ctx.lineTo(x + 1, y + 3);
    }
    ctx.closePath();
    ctx.fill();
  }

  /**
   * Draw the rows.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @param {object} shape - The reading from `model`.
   * @param {number} right - Where the text column ends.
   * @returns {void}
   */
  function drawRows(ctx, theme, chosen, shape, right) {
    const layout = state.layout;
    const focused = document.activeElement === root;
    for (let offset = 0; offset < layout.rows; offset += 1) {
      const position = state.view + offset;
      const row = rowAt(shape, position);
      if (!row) break;
      const y = layout.rowsY + offset * ROW_HEIGHT;
      const cells = row.kind === "term" ? 1 : row.span;
      const onCaret = position === state.caret.position
        ? caretColumn(row, state.caret.column)
        : -1;

      for (let column = 0; column < cells; column += 1) {
        const box = cellBox(row, column, y, right);
        if (sameCell(state.hover, { position, column })) {
          ctx.globalAlpha = 0.10;
          ctx.fillStyle = theme.fg;
          ctx.fillRect(box.x, y, box.width, ROW_HEIGHT);
          ctx.globalAlpha = 1;
        }

        if (row.kind === "term") {
          drawTermRow(ctx, theme, chosen, row.term, y, y + ROW_HEIGHT / 2, right, shape);
        } else if (row.kind === "word") {
          drawWordCell(ctx, theme, chosen, row.term, row.first + column, box);
        } else {
          drawMatchCell(ctx, theme, chosen, row.first + column, box);
        }

        if (focused && column === onCaret) {
          ctx.save();
          ctx.setLineDash([2, 2]);
          ctx.lineWidth = 1;
          ctx.strokeStyle = theme.accent;
          ctx.strokeRect(box.x + 0.5, y + 0.5, Math.max(1, box.width - 1), ROW_HEIGHT - 1);
          ctx.restore();
        }
      }
    }
  }

  /**
   * Draw one terminology row.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @param {object} term - One row of the listing.
   * @param {number} y - Top of the row in element pixels.
   * @param {number} middle - Middle of the row in element pixels.
   * @param {number} right - Where the text column ends.
   * @param {object} shape - The reading from `model`.
   * @returns {void}
   */
  function drawTermRow(ctx, theme, chosen, term, y, middle, right, shape) {
    const layout = state.layout;
    const held = chosen.byTerm.get(term.name);
    const ticks = tickedIn(chosen, term);
    const fill = held?.whole.length ? "full" : ticks ? "part" : "none";

    if (fill !== "none") {
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = theme.accent;
      ctx.fillRect(layout.x0, y, right - layout.x0, ROW_HEIGHT);
      ctx.globalAlpha = 1;
    }
    if (!shape.search) {
      drawTwist(ctx, theme, layout.x0 + 2, middle, state.open.has(term.name));
    }
    const boxX = layout.x0 + TWIST_WIDTH;
    drawBox(ctx, theme, boxX, y + (ROW_HEIGHT - MARK_BOX) / 2, fill);

    const textX = boxX + MARK_BOX + MARK_GAP;
    const textRight = Math.max(textX + 1, right - COUNT_COLUMN);
    ctx.save();
    ctx.beginPath();
    ctx.rect(textX, y, textRight - textX, ROW_HEIGHT);
    ctx.clip();
    ctx.font = BODY_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.fg;
    ctx.fillText(term.name, textX, middle);
    ctx.restore();

    ctx.font = SMALL_FONT;
    ctx.textAlign = "right";
    ctx.fillStyle = ticks ? theme.accent : theme.fgMuted;
    const tail = ticks ? `${ticks} of ${term.entries}` : String(term.entries);
    ctx.fillText(tail, right - 2, middle);
    if (term.own) {
      ctx.textAlign = "left";
      ctx.fillStyle = theme.success;
      ctx.fillText(`+${term.own}`, textRight + 2, middle);
    }
  }

  /**
   * Draw one word of an open terminology.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @param {object} term - One row of the listing.
   * @param {number} index - Which word, counting from the start of the terminology.
   * @param {{x: number, y: number, width: number}} box - The cell from `cellBox`.
   * @returns {void}
   */
  function drawWordCell(ctx, theme, chosen, term, index, box) {
    const word = wordAt(term.name, index);
    const held = chosen.byTerm.get(term.name);
    const ticked = Boolean(
      held && (held.whole.length || (word && held.words.has(word.text))),
    );
    drawWord(ctx, theme, {
      ...box,
      middle: box.y + ROW_HEIGHT / 2,
      ticked,
      text: word?.text ?? "",
      own: word?.own ?? false,
      tag: "",
      waiting: !word,
    });
  }

  /**
   * Draw one match of a search.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @param {number} index - Which match, counting from the first.
   * @param {{x: number, y: number, width: number}} box - The cell from `cellBox`.
   * @returns {void}
   */
  function drawMatchCell(ctx, theme, chosen, index, box) {
    const match = matchAt(index);
    const held = match ? chosen.byTerm.get(match.term) : null;
    const ticked = Boolean(held && (held.whole.length || held.words.has(match.text)));
    drawWord(ctx, theme, {
      ...box,
      middle: box.y + ROW_HEIGHT / 2,
      ticked,
      text: match?.text ?? "",
      own: match?.own ?? false,
      tag: match?.term ?? "",
      waiting: !match,
    });
  }

  /**
   * Draw a word, from an open terminology or from a search.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} cell - What to draw: `x`, `y`, `width`, `middle`, `ticked`, `text`,
   *   `own`, `tag` and `waiting`.
   * @returns {void}
   */
  function drawWord(ctx, theme, cell) {
    const boxX = cell.x;
    const cellRight = cell.x + cell.width - CELL_GAP;
    if (cell.waiting) {
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = theme.fgMuted;
      ctx.fillRect(boxX, cell.middle - 1, Math.max(8, (cellRight - boxX) * 0.6), 2);
      ctx.globalAlpha = 1;
      return;
    }
    if (cell.ticked) {
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = theme.accent;
      ctx.fillRect(cell.x, cell.y, cell.width, ROW_HEIGHT);
      ctx.globalAlpha = 1;
    }
    drawBox(ctx, theme, boxX, cell.y + (ROW_HEIGHT - MARK_BOX) / 2, cell.ticked ? "full" : "none");

    const textX = boxX + MARK_BOX + MARK_GAP;
    const tagRoom = cell.tag
      ? Math.min(COUNT_COLUMN, Math.max(0, Math.floor((cellRight - textX) * TAG_SHARE)))
      : 0;
    const textRight = Math.max(textX, cellRight - tagRoom - (cell.own ? OWN_WIDTH : 0));

    // The word is cut at its own column edge rather than clipped there.
    ctx.font = BODY_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = cell.ticked ? theme.fg : theme.fgMuted;
    ctx.fillText(elideText(ctx, cell.text, textRight - textX, 1), textX, cell.middle);

    ctx.font = SMALL_FONT;
    if (cell.own) {
      ctx.textAlign = "left";
      ctx.fillStyle = theme.success;
      ctx.fillText("+", textRight + 1, cell.middle, OWN_WIDTH);
    }
    if (tagRoom > 0) {
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgDisabled;
      ctx.fillText(elideText(ctx, cell.tag, tagRoom, 1), cellRight, cell.middle);
    }
  }

  /**
   * Draw the words that stand in for the rows.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {string} text - The state to draw.
   * @returns {void}
   */
  function drawNotice(ctx, theme, text) {
    const layout = state.layout;
    ctx.font = BODY_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(
      text,
      (layout.x0 + layout.x1) / 2,
      layout.rowsY + (layout.rows * ROW_HEIGHT) / 2,
      layout.x1 - layout.x0 - 8,
    );
  }

  /**
   * Draw the scrollbar.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object|null} bar - Geometry from `scrollGeometry`.
   * @returns {void}
   */
  function drawScrollbar(ctx, theme, bar) {
    if (!bar) return;
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = theme.border;
    ctx.fillRect(bar.x, bar.y, bar.width, bar.height);
    ctx.globalAlpha = 1;
    ctx.fillStyle = theme.fgMuted;
    ctx.fillRect(bar.x, bar.thumbY, bar.width, bar.thumbHeight);
  }

  /**
   * The standing note in the footer, which is a state rather than an explanation.
   *
   * @param {object} chosen - The reading from `picked`.
   * @returns {{text: string, warn: boolean, action: (() => void)|null}} What to draw, whether
   *   it is drawn as a warning, and what clicking it does.
   */
  function standingNote(chosen) {
    if (inputLinked(node, PICKED_WIDGET)) {
      return { text: "picked is linked", warn: true, action: null };
    }
    if (chosen.unknown.length) {
      return {
        text: `${chosen.unknown.length} line(s) name no terminology; click to remove`,
        warn: true,
        action: dropUnknown,
      };
    }
    if (!chosen.lines) {
      return { text: "nothing ticked, so the names go out", warn: false, action: null };
    }
    if (searching()) return { text: "Escape clears the search", warn: false, action: null };
    return { text: "type to filter, Enter ticks", warn: false, action: null };
  }

  /**
   * Draw the footer, and collect the regions its hover text sits in.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} chosen - The reading from `picked`.
   * @param {object} shape - The reading from `model`.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawFooter(ctx, theme, chosen, shape, regions) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    const claim = readClaim(chosen);
    const box = drawIcon(
      ctx,
      claim.icon,
      layout.x0,
      layout.footerY + (layout.footerHeight - ICON_SIZE) / 2,
      ICON_SIZE,
      claim.icon === ICON.WARNING ? theme.warning : theme.fgMuted,
    );
    regions.push({ ...box, title: iconTitle(claim.icon, claim.detail) });
    const glyphWidth = ICON_SIZE + GLYPH_GAP;

    const count = shape.search
      ? `${chosen.words} ticked, ${shape.matchTotal}${state.matches?.truncated ? "+" : ""} found`
      : `${chosen.words} ticked in ${chosen.terms} terminology(s)`;
    ctx.textAlign = "right";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(count, layout.x1, middle);
    const rightWidth = ctx.measureText(count).width;

    const note = state.message
      ? { text: state.message, warn: false, action: null }
      : standingNote(chosen);
    state.noteAction = state.message ? null : note.action;
    state.noteBox = null;
    const available = layout.x1 - layout.x0 - glyphWidth - rightWidth - 8;
    if (note.text && available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = note.warn ? theme.warning : theme.fgMuted;
      const noteX = layout.x0 + glyphWidth;
      ctx.fillText(note.text, noteX, middle, available);
      if (state.noteAction) {
        state.noteBox = {
          x: noteX,
          y: layout.footerY,
          width: Math.min(available, ctx.measureText(note.text).width),
          height: layout.footerHeight,
        };
      }
    }
    regions.push({
      x: layout.x0,
      y: layout.footerY,
      width: layout.x1 - layout.x0,
      height: layout.footerHeight,
      title: FOOTER_TITLE,
    });
  }

  /**
   * Draw the whole panel.
   *
   * @returns {void}
   */
  function paint() {
    if (state.disposed) return;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return;

    // The graph's zoom is in here as well as the screen's density, so a magnified node is
    // drawn at the resolution it is shown at. Everything below `setTransform` stays in layout
    // units.
    const ratio = surfaceRatio(root);
    const deviceWidth = Math.max(1, Math.round(width * ratio));
    const deviceHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== deviceWidth) canvas.width = deviceWidth;
    if (canvas.height !== deviceHeight) canvas.height = deviceHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    state.layout = computeLayout(width, height);
    const layout = state.layout;
    const theme = readTheme();
    const chosen = picked();
    let shape = model();

    const columns = measureColumns(ctx, shape);
    if (columns !== state.columns) {
      // The word at the top of the view and the one under the caret keep their place: a flat
      // position on its own names a different word once the arithmetic under it has moved.
      const after = model(columns);
      const top = positionOf(after, cellAt(shape, state.view, 0));
      const caret = positionOf(after, caretCell(shape));
      state.columns = columns;
      shape = after;
      state.view = clamp(top?.position ?? state.view, 0, Math.max(0, after.total - layout.rows));
      state.caret = caret
        ?? { position: clamp(state.caret.position, 0, Math.max(0, after.total - 1)), column: 0 };
      state.hover = null;
    }
    const regions = [];

    const chips = drawChips(ctx, theme, chosen);
    if (chips.length) {
      const last = chips[chips.length - 1];
      regions.push({
        x: chips[0].x,
        y: layout.chipsY,
        width: Math.max(1, last.x + last.width - chips[0].x),
        height: CHIP_HEIGHT,
        title: CHIP_TITLE,
      });
    }
    state.chips = chips;

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.x0, layout.rowsY, layout.x1 - layout.x0, layout.rows * ROW_HEIGHT);

    const blocking = blockingText(shape);
    if (blocking) {
      drawNotice(ctx, theme, blocking);
    } else {
      const bar = scrollGeometry(shape.total);
      const right = bar ? bar.x - 2 : layout.x1;
      drawRows(ctx, theme, chosen, shape, right);
      drawScrollbar(ctx, theme, bar);
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      layout.x0 + 0.5,
      layout.rowsY + 0.5,
      Math.max(1, layout.x1 - layout.x0 - 1),
      Math.max(1, layout.rows * ROW_HEIGHT - 1),
    );

    drawFooter(ctx, theme, chosen, shape, regions);
    titles.set(regions);

    if (document.activeElement === root) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = theme.accent;
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    }

    // Asked for after the rows are drawn, so a window that has not arrived is a dim row now
    // and a real one on the frame the answer lands.
    ensureWindows(shape);
  }

  /**
   * Repaint on the next frame, coalescing repeated requests into one.
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
        console.error(`[${EXT_NAME}] Failed to draw the word picker:`, error);
      }
    });
  }

  /**
   * Wrap an event handler so a failure is logged rather than thrown at the browser.
   *
   * @param {(event: Event) => void} handler - Handler to wrap.
   * @returns {(event: Event) => void} The wrapped handler.
   */
  function guard(handler) {
    return (event) => {
      try {
        return handler(event);
      } catch (error) {
        console.error(`[${EXT_NAME}] Word picker input failed:`, error);
      }
      return undefined;
    };
  }

  /**
   * Repaint after the widget was edited by hand or by this panel.
   *
   * @returns {void}
   */
  function handlePickedChanged() {
    state.picks = null;
    schedulePaint();
  }

  /**
   * Set the typed filter, dropping whatever it addressed.
   *
   * @param {string} text - The filter as it should now read.
   * @returns {void}
   */
  function setFilter(text) {
    if (text === state.filter) return;
    state.filter = text;
    state.matches = null;
    state.matchesKey = "";
    state.view = 0;
    state.caret = { position: 0, column: 0 };
    state.caretWant = 0;
    schedulePaint();
  }

  /**
   * Act on one cell, which is a tick for a word and a whole terminology for a name.
   *
   * @param {object} cell - The cell from `cellAt`.
   * @returns {void}
   */
  function activate(cell) {
    if (!cell) return;
    if (cell.kind === "term") {
      toggleTerm(cell.term);
      return;
    }
    if (cell.kind === "word") {
      const word = wordAt(cell.term.name, cell.index);
      if (word) toggleWord(cell.term.name, word.text);
      else setMessage("that word has not arrived yet");
      return;
    }
    const match = matchAt(cell.index);
    if (match) toggleWord(match.term, match.text);
    else setMessage("that match has not arrived yet");
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;

    root.focus?.({ preventScroll: true });
    state.press = null;
    const point = localPoint(event);

    const chip = hitChip(point, state.chips ?? []);
    if (chip) {
      state.mode = chip.mode;
      state.view = clampView(state.view);
      state.caret = { position: 0, column: 0 };
      state.caretWant = 0;
      schedulePaint();
      return;
    }

    const shape = model();
    const bar = scrollGeometry(shape.total);
    if (bar && point.x >= bar.x - 2 && point.y >= bar.y && point.y <= bar.y + bar.height) {
      const onThumb = point.y >= bar.thumbY && point.y <= bar.thumbY + bar.thumbHeight;
      const grip = onThumb ? point.y - bar.thumbY : bar.thumbHeight / 2;
      state.press = { pointerId: event.pointerId, kind: "thumb", grip };
      // The pointer is captured so a drag that leaves the panel keeps scrolling it, and the
      // release outside still arrives here rather than being lost.
      try {
        root.setPointerCapture?.(event.pointerId);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to capture the pointer:`, error);
      }
      if (!onThumb) dragThumb(point, bar, grip);
      return;
    }

    const layout = state.layout;
    if (state.noteAction && onNote(point)) {
      state.press = { pointerId: event.pointerId, kind: "footer" };
      return;
    }

    // A notice is drawn in place of the rows, so there is no row under the pointer to act on.
    if (blockingText(shape)) return;

    const cell = hitCell(point, shape);
    if (!cell) return;
    // On a terminology row only the tick box ticks; the triangle, the name and the counts
    // open it, so the two gestures never fight over the same pixels.
    const boxEnd = layout.x0 + TWIST_WIDTH + MARK_BOX + MARK_GAP;
    const opens = cell.kind === "term" && !shape.search
      && (point.x < layout.x0 + TWIST_WIDTH || point.x >= boxEnd);
    state.press = {
      pointerId: event.pointerId,
      kind: opens ? "twist" : "row",
      item: cell,
    };
    state.caret = { position: cell.position, column: cell.column };
    state.caretWant = cell.column;
    schedulePaint();
  };

  /**
   * Scroll to where a thumb drag has put the thumb.
   *
   * @param {{x: number, y: number}} point - Pointer position in element pixels.
   * @param {object} bar - Geometry from `scrollGeometry`.
   * @param {number} grip - Where inside the thumb the drag started.
   * @returns {void}
   */
  function dragThumb(point, bar, grip) {
    const travel = clamp(point.y - grip - bar.y, 0, bar.span);
    setView(Math.round((travel / bar.span) * bar.maxView));
  }

  const onPointerMove = (event) => {
    if (event.buttons & 4) {
      app.canvas?.processMouseMove?.(event);
      return;
    }
    // A button released off the element delivers no pointerup for a press that captured
    // nothing, so a press left armed would act on the next release from any gesture at all.
    if (state.press && !(event.buttons & 1)) {
      endPress();
      return;
    }

    const point = localPoint(event);
    if (state.press?.kind === "thumb") {
      const bar = scrollGeometry(model().total);
      if (bar) dragThumb(point, bar, state.press.grip);
      return;
    }

    const shape = model();
    const cell = blockingText(shape) ? null : hitCell(point, shape);
    const overChip = hitChip(point, state.chips ?? []) !== null;
    const overFooter = Boolean(state.noteAction) && onNote(point);
    root.style.cursor = cell || overChip || overFooter ? "pointer" : "default";
    const hover = cell ? { position: cell.position, column: cell.column } : null;
    if (!sameCell(hover, state.hover)) {
      state.hover = hover;
      schedulePaint();
    }
  };

  const onPointerUp = (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseUp?.(event);
      return;
    }
    if (event.button !== 0) return;
    const press = state.press;
    endPress();
    if (!press || press.pointerId !== event.pointerId) return;

    if (press.kind === "footer") {
      const action = state.noteAction;
      if (action && onNote(localPoint(event))) action();
      return;
    }
    if (press.kind !== "row" && press.kind !== "twist") return;

    // The listing can have been rebuilt between the two halves of the gesture, so the cell
    // under the pointer now is the one acted on, and only while it names what it started on.
    const shape = model();
    const cell = hitCell(localPoint(event), shape);
    if (!sameItem(cell, press.item)) return;
    if (press.kind === "twist") {
      if (cell.kind === "term") setOpen(cell.term.name);
      return;
    }
    // A name clicked while searching leaves the search on that terminology, since the words
    // under it in a search are matches rather than its own list.
    if (cell.kind === "term" && shape.search) {
      const name = cell.term.name;
      setFilter("");
      setOpen(name, true);
      return;
    }
    activate(cell);
  };

  /**
   * End a press, releasing anything it captured.
   *
   * @returns {void}
   */
  function endPress() {
    const press = state.press;
    state.press = null;
    if (!press || press.kind !== "thumb") return;
    try {
      root.releasePointerCapture?.(press.pointerId);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the pointer:`, error);
    }
  }

  const onContextMenu = (event) => {
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  };

  const onWheel = (event) => {
    if (!event.deltaY) return false;
    const step = event.deltaY > 0 ? WHEEL_ROWS : -WHEEL_ROWS;
    // At either end of the list, and for a list that fits on screen, the view does not move and
    // the gesture is the graph's.
    const before = state.view;
    setView(state.view + step);
    return state.view !== before;
  };

  /**
   * Move the caret up or down, keeping the column it aims at.
   *
   * @param {number} step - Rows to move, negative toward the top.
   * @returns {void}
   */
  function moveCaret(step) {
    const shape = model();
    if (!shape.total) {
      setMessage("nothing to move through");
      return;
    }
    setCaret(shape, clamp(state.caret.position + step, 0, shape.total - 1), state.caretWant);
  }

  /**
   * Move the caret one word along the run it is in, wrapping at the end of a row.
   *
   * @param {object} shape - The reading from `model`.
   * @param {number} step - Words to move, negative toward the first.
   * @returns {void}
   */
  function moveAcross(shape, step) {
    const cell = caretCell(shape);
    if (!cell || cell.kind === "term") return;
    const limit = cell.kind === "match"
      ? shape.grid.tailCount
      : (shape.grid.sections[cell.section]?.count ?? 0);
    const index = cell.index + step;
    if (index < 0 || index >= limit) return;
    const at = positionOf(shape, { ...cell, index });
    if (!at) return;
    state.caret = at;
    state.caretWant = at.column;
    reveal();
    schedulePaint();
  }

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const shape = model();
    const page = Math.max(1, state.layout.rows);
    const row = rowAt(shape, state.caret.position);
    let handled = true;

    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown":
        moveCaret((event.key === "ArrowUp" ? -1 : 1) * (event.shiftKey ? COARSE_STEP : 1));
        break;
      case "ArrowRight":
      case "ArrowLeft": {
        const forward = event.key === "ArrowRight";
        if (row?.kind === "term" && !shape.search) setOpen(row.term.name, forward);
        else if (row) moveAcross(shape, forward ? 1 : -1);
        else setMessage("nothing to move through");
        break;
      }
      case "PageUp":
      case "PageDown":
        moveCaret(event.key === "PageUp" ? -page : page);
        break;
      case "Home":
      case "End": {
        if (!shape.total) {
          setMessage("nothing to move through");
          break;
        }
        const last = shape.total - 1;
        const edge = rowAt(shape, last);
        const column = edge && edge.kind !== "term" ? Math.max(0, edge.span - 1) : 0;
        state.caretWant = event.key === "Home" ? 0 : column;
        state.caret = event.key === "Home"
          ? { position: 0, column: 0 }
          : { position: last, column };
        reveal();
        schedulePaint();
        break;
      }
      case "Enter":
      case " ": {
        const cell = caretCell(shape);
        if (cell) activate(cell);
        else setMessage("no row to pick");
        break;
      }
      case "Backspace":
        // Consumed whatever is on screen. Left unhandled these reach ComfyUI's own binding,
        // which deletes the node the panel is drawn on.
        if (state.filter) setFilter(state.filter.slice(0, -1));
        else setMessage("nothing to delete here; Space ticks and unticks a word");
        break;
      case "Delete":
        if (row?.kind === "term" && tickedIn(picked(), row.term)) toggleTerm(row.term);
        else setMessage("the pantry is never changed here; Space ticks and unticks a word");
        break;
      case "Escape":
        // The filter is the only thing this key drops. Unticking every word is an edit to the
        // picked box and a keystroke there, where an accident is undone by looking at it.
        setFilter("");
        endPress();
        break;
      default:
        // Anything printable filters the list, and at two characters the words themselves are
        // searched. The filter is this panel's own and reaches no widget.
        if (event.key.length === 1) {
          if (state.filter.length < FILTER_CHARS) setFilter(state.filter + event.key);
        } else {
          handled = false;
        }
    }

    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const onBlur = () => {
    // Focus can only leave mid-press when the gesture was interrupted, by another window
    // taking the pointer for instance, so the press is dropped.
    endPress();
    state.hover = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(endPress));
  root.addEventListener("lostpointercapture", guard(endPress));
  root.addEventListener("pointerenter", guard(() => {
    // A run writes terminology while the page is open, so the panel asks again for a listing
    // it has been holding for a while.
    if (Date.now() - state.fetchedAt > STALE_MS) refresh();
  }));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hover === null) return;
    state.hover = null;
    schedulePaint();
  }));
  root.addEventListener("contextmenu", guard(onContextMenu));
  const releaseWheel = captureWheel(root, guard(onWheel));
  root.addEventListener("keydown", guard(onKeyDown));
  root.addEventListener("focus", guard(schedulePaint));
  root.addEventListener("blur", guard(onBlur));

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => schedulePaint());
    observer.observe(root);
  }

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the
  // repaint that follows a zoom comes from here.
  let unwatchRatio = watchSurfaceRatio(root, schedulePaint);

  // The panel is drawn into a canvas, which takes literal colours, so a palette change
  // repaints.
  let unwatchTheme = onThemeChange(schedulePaint);

  /**
   * Release the timers, observers, listeners and hover text the panel holds.
   *
   * @returns {void}
   */
  function dispose() {
    state.disposed = true;
    releaseWheel();
    if (state.paintHandle) cancelAnimationFrame(state.paintHandle);
    if (state.messageTimer) clearTimeout(state.messageTimer);
    state.paintHandle = 0;
    state.messageTimer = 0;
    // The token moves past every answer still in flight, so none of them is taken after this.
    state.token += 1;
    state.words.clear();
    state.wordsNeeded.clear();
    state.wordsPending.clear();
    observer?.disconnect();
    observer = null;
    unwatchRatio?.();
    unwatchRatio = null;
    unwatchTheme?.();
    unwatchTheme = null;
    titles.dispose();
  }

  return {
    element: root,
    height: UI_HEIGHT,
    // Unbounded, so the node's spare room reaches the interface rather than stopping at it.
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    schedulePaint,
    handlePickedChanged,
    refresh,
    dispose,
  };
}

/**
 * Append the panel to a node and wire it to the widget it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachTermPicker(node) {
  if (!findWidget(node, PICKED_WIDGET)) return;

  const picker = createTermPicker(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, picker, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Every multiline box on the node bounded the same way, so the panel above takes the room
  // past their ceiling instead of losing all of it to them.
  boundTextBoxes(node);

  chainWidgetCallback(node, PICKED_WIDGET, picker.handlePickedChanged, EXT_NAME);
  chainWidgetCallback(node, TERM_WIDGET, picker.schedulePaint, EXT_NAME);

  // A widget value is the default until `configure` has run, so what a saved workflow picked
  // is drawn from here rather than on creation.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      picker.handlePickedChanged();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to redraw after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered
  // and its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      picker.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the word picker:`, error);
    }
    return result;
  };

  picker.refresh();
  picker.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Noodle Soup Pick", "Word picker"],
      name: "Show the terminology word picker",
      tooltip:
        "Draw the stored terminology under the widgets of Noodle Soup Pick, with a tick "
        + "against each word the run will emit. The picked box itself is always available and "
        + "holds the same lines either way. This applies to nodes added after the setting "
        + "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise
    // wrap the prototype a second time and append a second panel.
    if (proto.__was_noodle_soup_pick_wrapped) return;
    proto.__was_noodle_soup_pick_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachTermPicker(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the word picker:`, error);
      }
      return result;
    };
  },
});
