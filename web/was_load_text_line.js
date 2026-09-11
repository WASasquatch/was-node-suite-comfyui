/**
 * Line browser for the Load Text Line node.
 *
 * Draws the lines of the file the `file` widget names and marks the line the run takes. The one
 * value it writes is the `index` widget.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fetchWithin } from "./interface/request.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { LABELS, PREVIEW_STATE } from "./interface/preview.js";
import { floorMod, truncate } from "./interface/python_arithmetic.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LoadTextLineUI";
const NODE_NAME = "WASLoadTextLine";
const SETTING_ID = "WAS.LoadTextLine.ShowInterface";

const FILE_WIDGET = "file";
const MODE_WIDGET = "mode";
const INDEX_WIDGET = "index";
const RANGE_WIDGET = "out_of_range";
const COMMENTS_WIDGET = "skip_comment_lines";
const SEED_WIDGET = "seed";

// The combo entry `modules/util/text_files.NO_FILES` offers when neither folder holds a text
// file. The node reads nothing for it, so the panel draws nothing for it either.
const NO_FILES = "No Text Files";

const MODE_FILE = "file";
const MODE_INDEX = "index";
const MODE_RANDOM = "random";

const RANGE_WRAP = "wrap";
const RANGE_CLAMP = "clamp";
const RANGE_ERROR = "error";

const UI_WIDGET_NAME = "was_load_text_line_ui";
const UI_WIDGET_TYPE = "was_line_browser";

const ROUTE = "/was/interface/api/text_lines";

// Lines asked for at once, which is the cap `modules/interface/lines.MAX_LINES` holds the
// answer to, and how far the window start moves between requests. A window covers every
// position the view can reach before the start moves again, so scrolling costs one request per
// STRIDE lines rather than one per row.
const PAGE = 500;
const STRIDE = 250;

// How long an answer stands before the pointer arriving over the panel asks again. A file can
// be rewritten in place between runs, and the node caches none either.
const STALE_MS = 3000;

// Characters of the route's refusal that are drawn. Its two refusals are short constants; the
// cut bounds anything a future one might carry.
const REFUSAL_CHARS = 120;

// The states this panel adds to the three it shares with every other interface. A file that is
// not chosen, and one the route will not serve, are states of their own rather than failures.
const NO_FILE = "was_no_file";
const REFUSED = "was_refused";

const UI_HEIGHT = 180;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout bands, measured in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const ROW_HEIGHT = 14;
const FOOTER_HEIGHT = 13;
const SCROLLBAR_WIDTH = 5;
const MIN_THUMB = 12;
const GUTTER_GAP = 6;
const MIN_GUTTER = 12;
const MARK_BAR = 2;
const GLYPH_GAP = 4;
const CLIP_MARK_WIDTH = 8;

const BODY_FONT = "10px sans-serif";
const NUMBER_FONT = "9px sans-serif";

const MESSAGE_TIMEOUT = 4000;

// Rows one wheel notch moves, and how many lines a page key covers beyond the rows on screen.
const WHEEL_ROWS = 3;

// How far the arrow keys move the index with Shift held.
const COARSE_STEP = 10;

/** What the footer's hover says about each mode, which is what the mark means in that mode. */
const MODE_TITLES = {
  [MODE_FILE]: "Every line goes out together in file mode, so no single line is marked and "
    + "neither index nor seed is read.",
  [MODE_INDEX]: "The marked line is the one index selects, after out_of_range has been applied "
    + "to an index outside the file. Click a line or use the arrow keys to move index.",
  [MODE_RANDOM]: "The marked line is the one seed draws. Index is not read in random mode, and "
    + "a click still moves it, for the mode being changed back.",
};

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
 * Which of the inputs deciding the line are filled by a link.
 *
 * @param {object} node - The node the panel is drawn on.
 * @returns {string[]} Their names, in the order the node declares them.
 */
function linkedDeciders(node) {
  return [MODE_WIDGET, INDEX_WIDGET, SEED_WIDGET, RANGE_WIDGET, COMMENTS_WIDGET]
    .filter((name) => inputLinked(node, name));
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
 * Work out where each band of the panel sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the list and the footer, with the number of whole rows
 *   the list holds.
 */
function computeLayout(width, height) {
  const x0 = PAD_X;
  const x1 = Math.max(x0 + 1, width - PAD_X);
  const footerY = Math.max(0, height - PAD_Y - FOOTER_HEIGHT);
  const rowsY = PAD_Y;
  const rowsHeight = Math.max(ROW_HEIGHT, footerY - rowsY - 2);
  return {
    width,
    height,
    x0,
    x1,
    rowsY,
    rowsHeight,
    rows: Math.max(1, Math.floor(rowsHeight / ROW_HEIGHT)),
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * The first line of the window that covers a view position.
 *
 * @param {number} view - First line on screen, counting every line in the file.
 * @returns {number} Where the request starts, on a fixed stride so a few rows of scrolling
 *   reuse the window already loaded.
 */
function windowStart(view) {
  return Math.max(0, Math.floor(view / STRIDE) * STRIDE);
}

/**
 * Read one answer from the route into the shape the panel draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The window, or null when the body is not one.
 */
function normalise(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.lines)) return null;
  const whole = (value, fallback) => {
    const number = Number(value);
    return Number.isFinite(number) ? Math.trunc(number) : fallback;
  };
  const optional = (value) => {
    if (value === null || value === undefined) return null;
    const number = Number(value);
    return Number.isFinite(number) ? Math.trunc(number) : null;
  };
  return {
    total: Math.max(0, whole(data.total, 0)),
    kept: Math.max(0, whole(data.kept, 0)),
    start: Math.max(0, whole(data.start, 0)),
    lines: data.lines.map((row) => ({
      text: typeof row?.text === "string" ? row.text : "",
      index: optional(row?.index),
      comment: row?.comment === true,
      clipped: row?.clipped === true,
    })),
    randomIndex: optional(data.random_index),
    clipped: Math.max(0, whole(data.clipped, 0)),
    lossy: data.lossy === true,
    truncated: data.truncated === true,
  };
}

/**
 * One line as it is drawn.
 *
 * @param {string} text - The line as the file holds it.
 * @returns {string} The same characters, with tabs opened out. A canvas draws a tab as nothing
 *   at all, which runs the columns of a .tsv together.
 */
function displayText(text) {
  return text.replace(/\t/g, "    ");
}

/**
 * Which line the run takes, mirroring `LoadTextLine.select`.
 *
 * @param {string} mode - The `mode` widget.
 * @param {number} index - The `index` widget, already truncated.
 * @param {number} kept - Lines the node counts, as far as the route read.
 * @param {string} rule - The `out_of_range` widget.
 * @param {number|null} random - The line the seed draws, from the route.
 * @param {boolean} truncated - Whether the file is longer than the route read.
 * @returns {{whole: boolean, taken: number|null, note: string, warn: boolean}} Whether every
 *   line goes out, the one line that does otherwise, the words for it, and whether those words
 *   are about a run that will not do what was asked.
 */
function selectLine(mode, index, kept, rule, random, truncated) {
  const answer = { whole: false, taken: null, note: "", warn: false };
  if (mode === MODE_FILE) {
    answer.whole = true;
    return answer;
  }
  if (!kept) return answer;

  if (mode === MODE_RANDOM) {
    if (truncated) {
      answer.warn = true;
      answer.note = "the seed draws from every line, past the lines read here";
      return answer;
    }
    answer.taken = random !== null && random >= 0 && random < kept ? random : null;
    if (answer.taken !== null) answer.note = `seed draws line ${answer.taken}`;
    return answer;
  }

  if (truncated && index < 0) {
    answer.warn = true;
    answer.note = `index ${index} counts back from the end of a longer file`;
    return answer;
  }
  const position = index < 0 ? index + kept : index;
  if (position >= 0 && position < kept) {
    // A positive index inside the lines read names the same line however many follow it, so
    // this one is marked whether or not the file was read whole.
    answer.taken = position;
    answer.note = index === position ? `line ${position}` : `index ${index} is line ${position}`;
    return answer;
  }
  if (truncated) {
    answer.warn = true;
    answer.note = `index ${index} is past the lines read here, and the file is longer`;
    return answer;
  }
  if (rule === RANGE_WRAP) {
    answer.taken = floorMod(position, kept);
    answer.note = `index ${index} wraps to line ${answer.taken}`;
    return answer;
  }
  if (rule === RANGE_CLAMP) {
    answer.taken = position < 0 ? 0 : kept - 1;
    answer.note = `index ${index} clamps to line ${answer.taken}`;
    return answer;
  }
  answer.warn = true;
  answer.note = rule === RANGE_ERROR
    ? `index ${index} is outside the file, and the run stops there`
    : `index ${index} is outside the file, and the line comes out empty`;
  return answer;
}

/**
 * Build the line browser for one node.
 *
 * @param {object} node - The node the panel decorates.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   handleFileChanged: () => void, handleWindowChanged: () => void,
 *   handleIndexChanged: () => void, refresh: () => void, dispose: () => void}} The panel for
 *   `appendInterfaceWidget`, the repaints for each widget it follows, a reload, and teardown.
 */
function createLineBrowser(node) {
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

  // The glyph, the numbers and the footer state what they are worth through the element's own
  // title. The regions are handed over again on every repaint, since they move whenever the
  // node is resized.
  const titles = hoverTitles(root);

  const state = {
    // The window the route answered, the key it was asked for under, and what the panel has to
    // draw instead of lines.
    window: null,
    key: "",
    meaning: "",
    pendingKey: "",
    token: 0,
    fetchedAt: 0,
    status: NO_FILE,
    refusal: "",
    // View state, which is the panel's own and reaches neither the widget nor the workflow.
    view: 0,
    chase: null,
    hover: null,
    press: null,
    // An index the arrow keys are moving, written when the key is released. Held here rather
    // than in the widget so one key gesture is one undo entry, whatever its repeat rate.
    pending: null,
    lastWritten: null,
    message: "",
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Read the `file` widget.
   *
   * @returns {string} The label it holds, empty when it cannot be read.
   */
  function fileValue() {
    const value = findWidget(node, FILE_WIDGET)?.value;
    return typeof value === "string" ? value.trim() : "";
  }

  /**
   * Read the `mode` widget.
   *
   * @returns {string} `file`, `index` or `random`, taking the schema's default for anything
   *   else, which is what the node does with a value it does not recognise.
   */
  function modeValue() {
    const value = findWidget(node, MODE_WIDGET)?.value;
    return value === MODE_INDEX || value === MODE_RANDOM ? value : MODE_FILE;
  }

  /**
   * Read the `out_of_range` widget.
   *
   * @returns {string} The stored value, which the selection compares by name.
   */
  function rangeValue() {
    const value = findWidget(node, RANGE_WIDGET)?.value;
    return typeof value === "string" ? value : RANGE_WRAP;
  }

  /**
   * Read the `skip_comment_lines` widget.
   *
   * @returns {boolean} True while comment lines are dropped, which is the schema's default.
   */
  function skipComments() {
    return findWidget(node, COMMENTS_WIDGET)?.value !== false;
  }

  /**
   * Read the `seed` widget.
   *
   * @returns {number} The seed, 0 when it cannot be read.
   */
  function seedValue() {
    const value = Number(findWidget(node, SEED_WIDGET)?.value);
    return Number.isFinite(value) ? Math.trunc(value) : 0;
  }

  /**
   * The seed as the route reads it.
   *
   * @returns {string} Its digits, spelled exactly as the prompt spells them. The widget's
   *   range reaches past the largest whole number JavaScript holds exactly, and the two
   *   spellings of such a number disagree: `toFixed` writes the double's own binary value
   *   while the prompt carries the shortest decimal that round-trips, and a seed of
   *   18446744073709552000 drew one line here and another on the run. `String` is what the
   *   prompt uses, and it reaches an exponent only at 1e21, above the seed's own maximum.
   */
  function seedText() {
    return String(seedValue());
  }

  /**
   * Read the `index` widget, or the value a key gesture is moving.
   *
   * @returns {number} The index the node would take, truncated the way Python's `int()`
   *   truncates the number the prompt carries.
   */
  function caretIndex() {
    if (state.pending !== null) return state.pending;
    const value = Number(findWidget(node, INDEX_WIDGET)?.value);
    return Number.isFinite(value) ? truncate(value) : 0;
  }

  /**
   * Whether the answer has to carry the line a seed draws.
   *
   * @returns {boolean} True in random mode, which is the only mode that reads the seed.
   */
  function wantsSeed() {
    return modeValue() === MODE_RANDOM;
  }

  /**
   * What the answer means, as against which part of the file it covers.
   *
   * @returns {string} A key over the widgets that change what a line is numbered and which one
   *   the seed draws. Written as JSON rather than joined on a separator, since a file label is
   *   somebody's own path and any character picked to divide the parts is one a label may hold.
   */
  function meaningKey() {
    return JSON.stringify([fileValue(), skipComments(), wantsSeed() ? seedText() : ""]);
  }

  /**
   * The request the panel wants answered.
   *
   * @returns {string} A key covering every part of the request, so a widget change that cannot
   *   alter the answer costs nothing.
   */
  function requestKey() {
    return JSON.stringify([meaningKey(), windowStart(state.view)]);
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
   * Ask the route for the window the view needs.
   *
   * @param {boolean} [force] - Ask again for a window already held, which is how a file
   *   rewritten in place reaches the panel.
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function load(force = false) {
    if (state.disposed) return;
    const file = fileValue();
    if (!file || file === NO_FILES) {
      state.window = null;
      state.key = "";
      state.pendingKey = "";
      state.status = NO_FILE;
      state.refusal = "";
      schedulePaint();
      return;
    }

    const key = requestKey();
    if (key === state.pendingKey) return;
    if (!force && key === state.key) return;
    state.pendingKey = key;
    // A window already on screen is left there while the next one is fetched, so scrolling and
    // a reload do not blank the panel between frames. What is held back is the mark, since a
    // seed or a comment rule that has moved on numbers those same lines differently.
    if (!state.window) state.status = PREVIEW_STATE.LOADING;
    const token = (state.token += 1);
    const query = `${ROUTE}?file=${encodeURIComponent(file)}`
      + `&start=${windowStart(state.view)}&limit=${PAGE}`
      + `&skip_comments=${skipComments() ? 1 : 0}`
      + (wantsSeed() ? `&seed=${encodeURIComponent(seedText())}` : "");

    try {
      const response = await fetchWithin(query, {
        // The file can be rewritten between two runs, so a copy held by the browser would
        // claim to be the file on disk while showing something else.
        cache: "no-store",
      });
      if (state.disposed || token !== state.token) return;

      if (response.status === 404) {
        // The route separates a label nobody lists from a listed file it could not read, and
        // its words are drawn rather than reworded, so the panel and the node's own log name
        // the same condition the same way.
        const words = (await response.text()).trim();
        if (state.disposed || token !== state.token) return;
        state.window = null;
        state.key = key;
        state.status = REFUSED;
        state.refusal = words.slice(0, REFUSAL_CHARS) || "that file could not be read";
        return;
      }
      if (!response.ok) {
        state.window = null;
        state.key = key;
        state.status = PREVIEW_STATE.FAILED;
        return;
      }

      const payload = normalise(await response.json());
      if (state.disposed || token !== state.token) return;
      if (!payload) {
        state.window = null;
        state.key = key;
        state.status = PREVIEW_STATE.FAILED;
        return;
      }
      accept(payload, key);
    } catch (error) {
      if (state.disposed || token !== state.token) return;
      console.error(`[${EXT_NAME}] Failed to read the lines of \`${file}\`:`, error);
      state.window = null;
      state.key = key;
      state.status = PREVIEW_STATE.FAILED;
    } finally {
      if (!state.disposed) {
        if (state.pendingKey === key) state.pendingKey = "";
        state.fetchedAt = Date.now();
        schedulePaint();
      }
    }
  }

  /**
   * Take a window the route answered.
   *
   * @param {object} payload - The window from `normalise`.
   * @param {string} key - The key it was asked for under.
   * @returns {void}
   */
  function accept(payload, key) {
    state.window = payload;
    state.key = key;
    // What the answer is an answer to, so the frame after a seed or a comment rule changed
    // draws no mark rather than the mark the answer before it carried.
    state.meaning = meaningKey();
    state.status = PREVIEW_STATE.READY;
    state.refusal = "";
    // The route clamps a start past the end of the file, so a view that asked for one is
    // brought back to where the answer actually begins.
    state.view = clampView(state.view, payload.total);
    if (state.chase === null) return;
    const chase = state.chase;
    state.chase = null;
    if (payload.lines.some((row) => row.index === chase)) reveal(chase);
  }

  /**
   * Ask for whatever the current widgets and view need.
   *
   * @returns {void}
   */
  function refresh() {
    load(false).catch((error) => {
      console.error(`[${EXT_NAME}] Failed to ask for the file's lines:`, error);
    });
  }

  /**
   * Hold a view position inside the file.
   *
   * @param {number} view - Wanted first line on screen.
   * @param {number} total - Lines the file has, as far as the route counted.
   * @returns {number} The bounded position.
   */
  function clampView(view, total) {
    return clamp(Math.trunc(view), 0, Math.max(0, total - state.layout.rows));
  }

  /**
   * Scroll the list.
   *
   * @param {number} view - Wanted first line on screen.
   * @returns {boolean} True when the list moved, which is what decides whether a wheel gesture
   *   belongs here or to the graph underneath.
   */
  function setView(view) {
    const next = state.window ? clampView(view, state.window.total) : Math.max(0, Math.trunc(view));
    if (next === state.view) return false;
    state.view = next;
    refresh();
    schedulePaint();
    return true;
  }

  /**
   * Scroll a line into view by the number the node counts it by.
   *
   * @param {number|null} index - The node's index for the line, or null for no line.
   * @returns {void}
   */
  function reveal(index) {
    if (index === null || !Number.isFinite(index)) return;
    const payload = state.window;
    const visible = state.layout.rows;
    if (payload) {
      const at = payload.lines.findIndex((row) => row.index === index);
      if (at >= 0) {
        const position = payload.start + at;
        if (position < state.view) setView(position);
        else if (position > state.view + visible - 1) setView(position - visible + 1);
        state.chase = null;
        return;
      }
    }
    // Only a comment line above a line pushes it further down the file, so its index is the
    // lowest position it can sit at. The window that arrives for that position is looked in
    // once, and the panel stops chasing either way rather than walking the file.
    state.chase = index;
    setView(index);
  }

  /**
   * Write the `index` widget once.
   *
   * @param {number} index - The line number to store.
   * @returns {void}
   */
  function writeIndex(index) {
    if (state.disposed) return;
    const widget = findWidget(node, INDEX_WIDGET);
    if (!widget) return;
    const value = truncate(Number(index));
    if (!Number.isFinite(value)) return;
    state.lastWritten = value;
    if (Number(widget.value) === value) return;

    // The write is bracketed in the canvas change events the graph's change tracker listens
    // for, so an edit made from the keyboard gets an undo entry of its own. A commit that
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
    node.setDirtyCanvas?.(true, true);
  }

  /**
   * Write whatever a key gesture was moving.
   *
   * @returns {void}
   */
  function commitPending() {
    if (state.pending === null) return;
    const value = state.pending;
    state.pending = null;
    writeIndex(value);
    schedulePaint();
  }

  /**
   * Move the index a line at a time.
   *
   * @param {number} step - Lines to move, negative toward the start of the file.
   * @param {object} model - Model from `readModel`.
   * @returns {void}
   */
  function moveCaret(step, model) {
    if (!model.kept) {
      setMessage("no line to select");
      return;
    }
    const from = state.pending === null ? caretIndex() : state.pending;
    // A negative index counts back from the end, so a move starts from the line it names
    // rather than from the number itself, and the arrows leave the widget holding a position.
    const position = from < 0 ? from + model.kept : from;
    const next = clamp(position + step, 0, model.kept - 1);
    state.pending = next;
    reveal(next);
    schedulePaint();
  }

  /**
   * Choose a line, which is what a click and Enter both do.
   *
   * @param {number|null} index - The node's index for the line, or null for a line it drops.
   * @returns {void}
   */
  function chooseLine(index) {
    if (index === null) {
      setMessage("that line is a comment, and comment lines are skipped");
      return;
    }
    state.pending = null;
    writeIndex(index);
    const mode = modeValue();
    // The index is written whatever the mode, since it is the value the widget holds for when
    // index mode comes back. Saying which mode reads it is the honest answer to a click whose
    // mark does not move.
    if (mode !== MODE_INDEX) setMessage(`index ${index}, which index mode reads`);
    schedulePaint();
  }

  /**
   * Work out what the run takes, and what the footer says about it.
   *
   * @param {object|null} payload - The window from the route, or null.
   * @returns {object} The line marked, the caret, the words for both, and the blocking state
   *   drawn in place of the lines when there are none to draw.
   */
  function readModel(payload) {
    const mode = modeValue();
    const kept = payload ? payload.kept : 0;
    const total = payload ? payload.total : 0;
    const truncatedFile = !!payload?.truncated;
    const model = {
      mode,
      kept,
      total,
      caret: null,
      taken: null,
      whole: false,
      note: "",
      warn: false,
      count: "",
      claim: null,
      blocking: blockingText(payload),
    };
    if (model.blocking) return model;

    // A linked decider changes what somebody should do here, not just how truly the panel
    // reads, so it is drawn rather than left to the glyph's hover: clicking a row writes a
    // widget the run will not read, and without this the panel marks a line and is wrong.
    const linked = linkedDeciders(node);
    if (linked.length) {
      model.note = `${linked.join(" and ")} ${linked.length > 1 ? "are" : "is"} linked`;
      model.warn = true;
    }

    // The lines on screen were counted under the seed and the comment rule of the moment they
    // were asked for. Once either has moved on they are still the same lines, and which of
    // them the run takes is not known again until the next answer.
    if (state.meaning !== meaningKey()) {
      model.note = LABELS[PREVIEW_STATE.LOADING];
      return model;
    }

    const caret = caretIndex();
    // The caret is drawn where the index widget points, which is a line only when it lands on
    // one. Nothing is drawn for an index outside the file, and nothing for one counting back
    // from an end this panel never read: that is the same line count the mark is held back for.
    const caretPosition = caret < 0 ? (truncatedFile ? -1 : caret + kept) : caret;
    model.caret = caretPosition >= 0 && caretPosition < kept ? caretPosition : null;
    model.claim = readClaim(payload);
    model.count = countText(payload);
    return Object.assign(
      model,
      selectLine(mode, caret, kept, rangeValue(), payload.randomIndex, truncatedFile),
    );
  }

  /**
   * The state drawn in place of the lines.
   *
   * @param {object|null} payload - The window from the route, or null.
   * @returns {string} The words, empty when there are lines to draw.
   */
  function blockingText(payload) {
    if (state.status === NO_FILE) return "Pick a file to see its lines";
    if (state.status === REFUSED) return state.refusal;
    if (state.status === PREVIEW_STATE.FAILED) return "The lines could not be loaded";
    if (!payload) return LABELS[PREVIEW_STATE.LOADING];
    // A file that is not UTF-8 is read by the node as nothing at all, so the lines the route
    // recovered from it stand for no output and are not drawn.
    if (payload.lossy) return "This file is not UTF-8, so the node reads nothing from it";
    if (!payload.total) return "The file is empty";
    if (!payload.kept) return "Every line is a comment, and comment lines are skipped";
    return "";
  }

  /**
   * What the panel is worth against the run, as a glyph and the measurement behind it.
   *
   * @param {object} payload - The window from the route.
   * @returns {{icon: string, detail: string}} The claim for `iconTitle`.
   */
  function readClaim(payload) {
    // A link beats everything else this claim could say. The panel reads widgets, the run
    // reads the link, and the marked row is then the widget's line rather than the run's, so
    // saying so outranks reporting how faithfully the file itself was read.
    const linked = linkedDeciders(node);
    if (linked.length) {
      return {
        icon: ICON.WARNING,
        detail: `${linked.join(" and ")} ${linked.length > 1 ? "are" : "is"} filled by a link, `
          + "so the run reads that instead of the widget here and the marked line is not "
          + "necessarily the one it takes",
      };
    }
    if (payload.truncated) {
      return {
        icon: ICON.WARNING,
        detail: `the first ${payload.total} lines of a longer file, which is all this panel `
          + "reads; the node counts, wraps and draws over the whole of it",
      };
    }
    if (payload.clipped) {
      return {
        icon: ICON.APPROXIMATE,
        detail: `${payload.clipped} line(s) on screen cut to 400 characters; the node reads `
          + "them whole, and every line number here is the node's own",
      };
    }
    return {
      icon: ICON.EXACT,
      detail: "every line of the file, numbered as the node numbers them, and the line this "
        + "run takes",
    };
  }

  /**
   * Word the count on the right of the footer.
   *
   * @param {object} payload - The window from the route.
   * @returns {string} The lines the node will index, and how many the comment rule dropped.
   */
  function countText(payload) {
    const kept = payload.kept;
    const skipped = Math.max(0, payload.total - kept);
    const lead = payload.truncated
      ? `first ${kept} lines`
      : `${kept} ${kept === 1 ? "line" : "lines"}`;
    return skipped ? `${lead}, ${skipped} skipped` : lead;
  }

  /**
   * Where the scrollbar and its thumb sit.
   *
   * @param {number} total - Lines the file has, as far as the route counted.
   * @returns {object|null} The track, the thumb and the range they cover, or null when every
   *   line is already on screen.
   */
  function scrollGeometry(total) {
    const layout = state.layout;
    const visible = layout.rows;
    if (total <= visible) return null;
    const height = visible * ROW_HEIGHT;
    const thumb = Math.max(MIN_THUMB, Math.round((visible / total) * height));
    const span = Math.max(1, height - thumb);
    const maxView = Math.max(1, total - visible);
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
   * The width the line numbers are given.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context, with the number font set.
   * @param {number} kept - How many lines the node counts.
   * @returns {number} The column width in element pixels.
   */
  function gutterWidth(ctx, kept) {
    const widest = String(Math.max(0, kept - 1));
    return Math.max(MIN_GUTTER, Math.ceil(ctx.measureText(widest).width) + 2);
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
   * Find the line under a point.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {number|null} The line's position in the file, or null when the point is not on
   *   a line that was loaded.
   */
  function hitRow(point) {
    const layout = state.layout;
    const payload = state.window;
    if (!payload) return null;
    if (point.x < layout.x0 || point.x > layout.x1) return null;
    if (point.y < layout.rowsY || point.y >= layout.rowsY + layout.rows * ROW_HEIGHT) return null;
    const position = state.view + Math.floor((point.y - layout.rowsY) / ROW_HEIGHT);
    const row = payload.lines[position - payload.start];
    return row ? position : null;
  }

  /**
   * The row a position in the file holds, in the window on screen.
   *
   * @param {number|null} position - Position in the file.
   * @returns {object|null} The row, or null when it is not in the window.
   */
  function rowAt(position) {
    const payload = state.window;
    if (!payload || position === null) return null;
    return payload.lines[position - payload.start] ?? null;
  }

  /**
   * Draw the lines.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {number} right - Where the text column ends.
   * @returns {number} Where the number column ends, for the hover region over it.
   */
  function drawRows(ctx, theme, model, right) {
    const layout = state.layout;
    const payload = state.window;
    ctx.font = NUMBER_FONT;
    const gutter = gutterWidth(ctx, model.kept);
    const textX = layout.x0 + gutter + GUTTER_GAP;

    if (model.whole) {
      // Every line goes out together, so the band is marked once down its edge rather than a
      // line at a time: there is no line here that the run leaves behind. It covers the lines
      // on screen and not the room below the last of them, which is not a line.
      const drawnRows = Math.max(0, Math.min(layout.rows, payload.total - state.view));
      ctx.globalAlpha = 0.08;
      ctx.fillStyle = theme.accent;
      ctx.fillRect(layout.x0, layout.rowsY, right - layout.x0, drawnRows * ROW_HEIGHT);
      ctx.globalAlpha = 1;
      ctx.fillStyle = theme.accent;
      ctx.fillRect(layout.x0, layout.rowsY, MARK_BAR, drawnRows * ROW_HEIGHT);
    }

    for (let offset = 0; offset < layout.rows; offset += 1) {
      const position = state.view + offset;
      const row = payload.lines[position - payload.start];
      if (!row) continue;
      const y = layout.rowsY + offset * ROW_HEIGHT;
      const taken = model.taken !== null && row.index === model.taken;
      const caret = !taken && model.caret !== null && row.index === model.caret;

      if (taken) {
        ctx.globalAlpha = 0.22;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(layout.x0, y, right - layout.x0, ROW_HEIGHT);
        ctx.globalAlpha = 1;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(layout.x0, y, MARK_BAR, ROW_HEIGHT);
      } else if (state.hover === position && row.index !== null) {
        ctx.globalAlpha = 0.10;
        ctx.fillStyle = theme.fg;
        ctx.fillRect(layout.x0, y, right - layout.x0, ROW_HEIGHT);
        ctx.globalAlpha = 1;
      }

      if (caret) {
        // Dashed, and never the mark a taken line carries: in random mode the index is not
        // what decides the line, so drawing it filled would name the wrong line.
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = theme.accent;
        ctx.strokeRect(layout.x0 + 0.5, y + 0.5, right - layout.x0 - 1, ROW_HEIGHT - 1);
        ctx.restore();
      }

      if (row.index !== null) {
        ctx.font = NUMBER_FONT;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = taken ? theme.fg : theme.fgMuted;
        ctx.fillText(String(row.index), layout.x0 + gutter, y + ROW_HEIGHT / 2);
      }

      const cut = row.clipped ? CLIP_MARK_WIDTH : 0;
      ctx.save();
      ctx.beginPath();
      ctx.rect(textX, y, Math.max(1, right - textX - cut), ROW_HEIGHT);
      ctx.clip();
      ctx.font = BODY_FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = row.comment ? theme.fgDisabled : theme.fg;
      ctx.fillText(displayText(row.text), textX, y + ROW_HEIGHT / 2);
      ctx.restore();

      if (row.clipped) {
        ctx.font = BODY_FONT;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText("...", right, y + ROW_HEIGHT / 2);
      }
    }
    return layout.x0 + gutter;
  }

  /**
   * Draw the words that stand in for the lines.
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
   * Draw the footer, and collect the regions its hover text sits in.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} model - Model from `readModel`.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawFooter(ctx, theme, model, regions) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    let glyphWidth = 0;
    if (model.claim) {
      const box = drawIcon(
        ctx,
        model.claim.icon,
        layout.x0,
        layout.footerY + (layout.footerHeight - ICON_SIZE) / 2,
        ICON_SIZE,
        model.claim.icon === ICON.WARNING ? theme.warning : theme.fgMuted,
      );
      regions.push({ ...box, title: iconTitle(model.claim.icon, model.claim.detail) });
      glyphWidth = ICON_SIZE + GLYPH_GAP;
    }

    let rightWidth = 0;
    if (model.count) {
      // The count and the note share one line, so the count gives up its second half before
      // the note is cut into.
      const half = (layout.x1 - layout.x0) / 2;
      const short = model.count.split(",")[0];
      const text = ctx.measureText(model.count).width > half ? short : model.count;
      rightWidth = ctx.measureText(text).width;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(text, layout.x1, middle);
    }

    const note = state.message || model.note;
    const available = layout.x1 - layout.x0 - glyphWidth - rightWidth - 8;
    if (note && available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = model.warn && !state.message ? theme.warning : theme.fgMuted;
      ctx.fillText(note, layout.x0 + glyphWidth, middle, available);
    }

    regions.push({
      x: layout.x0,
      y: layout.footerY,
      width: layout.x1 - layout.x0,
      height: layout.footerHeight,
      title: MODE_TITLES[model.mode] ?? "",
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

    // The graph's zoom is in here as well as the screen's density, so a magnified node is drawn
    // at the resolution it is shown at. Everything below `setTransform` stays in layout units.
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
    const model = readModel(state.window);
    const regions = [];

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.x0, layout.rowsY, layout.x1 - layout.x0, layout.rows * ROW_HEIGHT);

    if (model.blocking) {
      drawNotice(ctx, theme, model.blocking);
    } else {
      const bar = scrollGeometry(model.total);
      const right = bar ? bar.x - 2 : layout.x1;
      const gutterX1 = drawRows(ctx, theme, model, right);
      drawScrollbar(ctx, theme, bar);
      regions.push({
        x: layout.x0,
        y: layout.rowsY,
        width: Math.max(1, gutterX1 - layout.x0),
        height: layout.rows * ROW_HEIGHT,
        title: skipComments()
          ? "The numbers are the value index takes, counting from 0. A comment line carries "
            + "none: the node skips it and never counts it."
          : "The numbers are the value index takes, counting from 0.",
      });
    }

    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.border;
    ctx.strokeRect(
      layout.x0 + 0.5,
      layout.rowsY + 0.5,
      Math.max(1, layout.x1 - layout.x0 - 1),
      Math.max(1, layout.rows * ROW_HEIGHT - 1),
    );

    drawFooter(ctx, theme, model, regions);
    titles.set(regions);

    if (document.activeElement === root) {
      ctx.lineWidth = 1;
      ctx.strokeStyle = theme.accent;
      ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
    }
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
        console.error(`[${EXT_NAME}] Failed to draw the line browser:`, error);
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
        console.error(`[${EXT_NAME}] Line browser input failed:`, error);
      }
      return undefined;
    };
  }

  /**
   * Repaint after the file changed, dropping the lines of the file that was there before.
   *
   * @returns {void}
   */
  function handleFileChanged() {
    state.window = null;
    state.key = "";
    state.meaning = "";
    state.view = 0;
    state.chase = null;
    state.hover = null;
    state.status = PREVIEW_STATE.LOADING;
    state.refusal = "";
    refresh();
    schedulePaint();
  }

  /**
   * Repaint after a widget that changes what the route answers.
   *
   * @returns {void}
   */
  function handleWindowChanged() {
    refresh();
    schedulePaint();
  }

  /**
   * Repaint after the index changed, dropping a key gesture somebody else's edit overtook.
   *
   * @returns {void}
   */
  function handleIndexChanged() {
    const widget = findWidget(node, INDEX_WIDGET);
    const value = Number(widget?.value);
    if (state.pending !== null && (!Number.isFinite(value) || value !== state.lastWritten)) {
      state.pending = null;
    }
    schedulePaint();
  }

  const onPointerDown = (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;

    root.focus?.({ preventScroll: true });
    commitPending();
    state.press = null;
    const point = localPoint(event);
    const model = readModel(state.window);
    if (model.blocking) return;

    const bar = scrollGeometry(model.total);
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

    const position = hitRow(point);
    if (position === null) return;
    state.press = { pointerId: event.pointerId, kind: "row", position };
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
      const bar = scrollGeometry(state.window?.total ?? 0);
      if (bar) dragThumb(point, bar, state.press.grip);
      return;
    }

    const position = hitRow(point);
    const row = rowAt(position);
    root.style.cursor = row && row.index !== null ? "pointer" : "default";
    if (position !== state.hover) {
      state.hover = position;
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
    if (!press || press.pointerId !== event.pointerId || press.kind !== "row") return;

    // The file can have been rewritten between the two halves of the gesture, so the line
    // under the pointer now is the one chosen, and only while it is the one it started on.
    const position = hitRow(localPoint(event));
    if (position === null || position !== press.position) return;
    chooseLine(rowAt(position)?.index ?? null);
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
    if (!event.deltaY || !state.window) return false;
    const step = event.deltaY > 0 ? WHEEL_ROWS : -WHEEL_ROWS;
    // At either end of the file, and for a file that fits on screen, the view does not move and
    // the gesture is the graph's.
    const before = state.view;
    setView(state.view + step);
    return state.view !== before;
  };

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const model = readModel(state.window);
    const page = Math.max(1, state.layout.rows);
    let handled = true;

    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown": {
        const step = event.key === "ArrowUp" ? -1 : 1;
        moveCaret(step * (event.shiftKey ? COARSE_STEP : 1), model);
        break;
      }
      case "PageUp":
      case "PageDown": {
        moveCaret(event.key === "PageUp" ? -page : page, model);
        break;
      }
      case "Home":
      case "End": {
        if (!model.kept) {
          setMessage("no line to select");
          break;
        }
        state.pending = event.key === "Home" ? 0 : model.kept - 1;
        reveal(state.pending);
        schedulePaint();
        break;
      }
      case "Enter":
      case " ": {
        // The arrows write the line as the key comes up, so what is left for Enter is a move
        // still being made. It never rewrites an index that already names a line: -1 names the
        // last line, and writing that line's number back would name a different line as soon
        // as the file grows.
        if (state.pending !== null) commitPending();
        else setMessage("no move to keep");
        break;
      }
      case "Escape": {
        // An unfinished key gesture is dropped rather than written, so the widget holds what
        // it held before the first key press.
        endPress();
        state.pending = null;
        schedulePaint();
        break;
      }
      case "Delete":
      case "Backspace": {
        // Consumed whatever is on screen. Left unhandled these reach ComfyUI's own binding,
        // which deletes the node the panel is drawn on.
        setMessage("the file is not edited here");
        break;
      }
      default:
        handled = false;
    }

    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  const onKeyUp = (event) => {
    const moves = event.key === "ArrowUp" || event.key === "ArrowDown"
      || event.key === "PageUp" || event.key === "PageDown"
      || event.key === "Home" || event.key === "End";
    if (moves) commitPending();
  };

  const onBlur = () => {
    // Focus can only leave mid-press when the gesture was interrupted, by another window
    // taking the pointer for instance, so the press is dropped and the move is written.
    endPress();
    commitPending();
    state.hover = null;
    schedulePaint();
  };

  root.addEventListener("pointerdown", guard(onPointerDown));
  root.addEventListener("pointermove", guard(onPointerMove));
  root.addEventListener("pointerup", guard(onPointerUp));
  root.addEventListener("pointercancel", guard(endPress));
  root.addEventListener("lostpointercapture", guard(endPress));
  root.addEventListener("pointerenter", guard(() => {
    // A file is rewritten in place more often than it is renamed, and the node rereads it on
    // every run, so the panel asks again for one it has been holding for a while.
    if (Date.now() - state.fetchedAt > STALE_MS) {
      load(true).catch((error) => {
        console.error(`[${EXT_NAME}] Failed to ask for the file's lines again:`, error);
      });
    }
  }));
  root.addEventListener("pointerleave", guard(() => {
    if (state.hover === null) return;
    state.hover = null;
    schedulePaint();
  }));
  root.addEventListener("contextmenu", guard(onContextMenu));
  const releaseWheel = captureWheel(root, guard(onWheel));
  root.addEventListener("keydown", guard(onKeyDown));
  root.addEventListener("keyup", guard(onKeyUp));
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

  // The panel is drawn into a canvas, which takes literal colours, so a palette change repaints.
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
    schedulePaint,
    handleFileChanged,
    handleWindowChanged,
    handleIndexChanged,
    refresh,
    dispose,
  };
}

/**
 * Chain a repaint onto a widget's callback.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {() => void} onChange - Called after the original callback.
 * @returns {void}
 */
function chainWidgetCallback(node, name, onChange) {
  const widget = findWidget(node, name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to repaint after a widget change:`, error);
    }
    return result;
  };
}

/**
 * Append the panel to a node and wire it to the widgets it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachLineBrowser(node) {
  if (!findWidget(node, FILE_WIDGET)) return;

  const browser = createLineBrowser(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, browser, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  chainWidgetCallback(node, FILE_WIDGET, browser.handleFileChanged);
  chainWidgetCallback(node, COMMENTS_WIDGET, browser.handleWindowChanged);
  chainWidgetCallback(node, SEED_WIDGET, browser.handleWindowChanged);
  chainWidgetCallback(node, MODE_WIDGET, browser.handleWindowChanged);
  chainWidgetCallback(node, INDEX_WIDGET, browser.handleIndexChanged);
  chainWidgetCallback(node, RANGE_WIDGET, browser.schedulePaint);

  // A widget value is the default until `configure` has run, so the file a saved workflow
  // names is asked for from here rather than on creation.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      browser.handleFileChanged();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the file after a workflow load:`, error);
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
      browser.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the line browser:`, error);
    }
    return result;
  };

  browser.refresh();
  browser.schedulePaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Load Text Line", "Line browser"],
      name: "Show the line browser",
      tooltip:
        "Draw the lines of the chosen file under the widgets of Load Text Line. The widgets "
        + "themselves are always available. This applies to nodes added after the setting "
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
    if (proto.__was_load_text_line_wrapped) return;
    proto.__was_load_text_line_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachLineBrowser(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the line browser:`, error);
      }
      return result;
    };
  },
});
