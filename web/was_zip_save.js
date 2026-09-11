/**
 * File browser for the Zip Save node.
 *
 * Lists what ComfyUI's input, output and temp folders hold and marks the files the archive will
 * carry. The one value it writes is the `files` widget, one label per line.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fetchWithin } from "./interface/request.js";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.js";
import { captureWheel, elementPoint } from "./interface/pointer.js";
import { LABELS, PREVIEW_STATE } from "./interface/preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme } from "./interface/theme.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ZipSaveUI";
const NODE_NAME = "WASZipSave";
const SETTING_ID = "WAS.ZipSave.ShowInterface";

const FILES_WIDGET = "files";
const NAMING_WIDGET = "entry_paths";
const PATHS_INPUT = "paths";

const UI_WIDGET_NAME = "was_zip_save_ui";
const UI_WIDGET_TYPE = "was_file_picker";

const ROUTE = "/was/interface/api/file_listing";

// The three folders the route lists, in the order it lists them, and how a label spells each.
const TAGS = ["input", "output", "temp"];

// How long an answer stands before the pointer arriving over the panel asks again. A run writes
// files while the page is open, which is the whole reason this is short.
const STALE_MS = 3000;

const UI_HEIGHT = 208;
const UI_MARGIN = 10;
const ELEMENT_MIN_HEIGHT = UI_HEIGHT - UI_MARGIN * 2;

// Layout bands, in element pixels.
const PAD_X = 4;
const PAD_Y = 4;
const CHIP_HEIGHT = 14;
const CHIP_GAP = 4;
const CHIP_PAD = 5;
const ROW_HEIGHT = 14;
const FOOTER_HEIGHT = 13;
const SCROLLBAR_WIDTH = 5;
const MIN_THUMB = 12;
const MARK_BOX = 8;
const MARK_GAP = 5;
const SIZE_COLUMN = 52;
const GLYPH_GAP = 4;

const BODY_FONT = "10px sans-serif";
const SMALL_FONT = "9px sans-serif";

const MESSAGE_TIMEOUT = 4000;

// Rows one wheel notch moves, and how far the arrow keys move with Shift held.
const WHEEL_ROWS = 3;
const COARSE_STEP = 10;

// Characters a typed filter holds. Long enough to name a file, short enough to draw.
const FILTER_CHARS = 40;

/** What the footer's hover says, which is what the panel is and is not responsible for. */
const FOOTER_TITLE =
  "The ticked files are the ones the archive will hold, in the order they were ticked, which "
  + "is the order the lines sit in the files box. Clicking writes that box and nothing else: "
  + "the files themselves are never moved, renamed or deleted here.";

/** What the tag chips say on hover. */
const CHIP_TITLE =
  "Which of the three folders the list shows. This only filters what is on screen; a file "
  + "already ticked stays ticked while its folder is hidden.";

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
 * A byte count as the node's own report writes it.
 *
 * @param {number} count - Bytes.
 * @returns {string} The size, in B, KB, MB or GB.
 */
function sizeText(count) {
  const bytes = Number(count);
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${Math.trunc(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * Whether a line of the widget is one the panel accounts for.
 *
 * @param {string} line - One line of the widget.
 * @returns {boolean} True when the line names a file.
 */
function namesFile(line) {
  const text = String(line ?? "").trim();
  return text !== "" && !text.startsWith("#");
}

/**
 * Work out where each band of the panel sits inside the element.
 *
 * @param {number} width - Element width in pixels.
 * @param {number} height - Element height in pixels.
 * @returns {object} Pixel geometry of the chips, the list and the footer, with the number of
 *   whole rows the list holds.
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
    rows: Math.max(1, Math.floor(rowsHeight / ROW_HEIGHT)),
    footerY,
    footerHeight: FOOTER_HEIGHT,
  };
}

/**
 * Read one answer from the route into the shape the panel draws from.
 *
 * @param {object} data - The parsed JSON body.
 * @returns {object|null} The listing, or null when the body is not one.
 */
function normalise(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.entries)) return null;
  const entries = [];
  for (const row of data.entries) {
    const label = typeof row?.label === "string" ? row.label : "";
    if (!label) continue;
    entries.push({
      label,
      relative: typeof row?.relative === "string" ? row.relative : label,
      tag: TAGS.includes(row?.tag) ? row.tag : "",
      size: Number.isFinite(Number(row?.size)) ? Math.max(0, Math.trunc(Number(row.size))) : 0,
      mtime: Number.isFinite(Number(row?.mtime)) ? Number(row.mtime) : 0,
    });
  }
  const roots = [];
  if (Array.isArray(data.roots)) {
    for (const root of data.roots) {
      if (TAGS.includes(root?.tag)) {
        roots.push({ tag: root.tag, files: Math.max(0, Math.trunc(Number(root.files) || 0)) });
      }
    }
  }
  return { entries, roots, truncated: data.truncated === true };
}

/**
 * Build the file browser for one node.
 *
 * @param {object} node - The node the panel decorates.
 * @returns {{element: HTMLElement, height: number, schedulePaint: () => void,
 *   handleFilesChanged: () => void, refresh: () => void, dispose: () => void}} The panel for
 *   `appendInterfaceWidget`, a repaint, a reload, and teardown.
 */
function createFilePicker(node) {
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
    pending: false,
    // View state, which is the panel's own and reaches neither the widget nor the workflow.
    view: 0,
    caret: 0,
    hover: null,
    press: null,
    chips: [],
    shown: new Set(TAGS),
    filter: "",
    message: "",
    messageTimer: 0,
    paintHandle: 0,
    layout: computeLayout(1, 1),
    disposed: false,
  };

  /**
   * Read the `files` widget.
   *
   * @returns {string} Its value, empty when it cannot be read.
   */
  function filesValue() {
    const value = findWidget(node, FILES_WIDGET)?.value;
    return typeof value === "string" ? value : "";
  }

  /**
   * The lines of the widget, exactly as it holds them.
   *
   * @returns {string[]} One entry per line, nothing trimmed.
   */
  function widgetLines() {
    return filesValue().split("\n");
  }

  /**
   * Which labels are chosen, and where each sits in the widget.
   *
   * @returns {{order: Map<string, number>, unknown: number, count: number}} The position each
   *   chosen line holds among the chosen ones, how many chosen lines the listing does not
   *   hold, and how many lines name a file at all.
   */
  function chosen() {
    const listed = new Set((state.listing?.entries ?? []).map((row) => row.label));
    const order = new Map();
    let unknown = 0;
    let count = 0;
    for (const line of widgetLines()) {
      if (!namesFile(line)) continue;
      const label = line.trim();
      count += 1;
      if (listed.has(label)) {
        if (!order.has(label)) order.set(label, order.size + 1);
      } else {
        unknown += 1;
      }
    }
    return { order, unknown, count };
  }

  /**
   * The rows on screen, after the chips and the typed filter.
   *
   * @returns {object[]} The entries, in the order the route gave them.
   */
  function visible() {
    const entries = state.listing?.entries ?? [];
    const needle = state.filter.toLowerCase();
    return entries.filter(
      (row) => state.shown.has(row.tag) && (!needle || row.label.toLowerCase().includes(needle)),
    );
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
   * Ask the route for the listing.
   *
   * @param {boolean} [force] - Ask again for a listing already held, which is how a file a run
   *   has just written reaches the panel.
   * @returns {Promise<void>} Resolved once the answer has been taken or dropped.
   */
  async function load(force = false) {
    if (state.disposed || state.pending) return;
    if (!force && state.listing) return;
    state.pending = true;
    if (!state.listing) state.status = PREVIEW_STATE.LOADING;
    const token = (state.token += 1);
    try {
      const response = await fetchWithin(ROUTE, {
        // A run writes files while the page is open, so a copy held by the browser would claim
        // to be what the three folders hold and would not be.
        cache: "no-store",
      });
      if (state.disposed || token !== state.token) return;
      if (!response.ok) {
        state.status = PREVIEW_STATE.FAILED;
        return;
      }
      const payload = normalise(await response.json());
      if (state.disposed || token !== state.token) return;
      if (!payload) {
        state.status = PREVIEW_STATE.FAILED;
        return;
      }
      state.listing = payload;
      state.status = PREVIEW_STATE.READY;
      state.view = clampView(state.view);
      state.caret = clamp(state.caret, 0, Math.max(0, visible().length - 1));
    } catch (error) {
      if (state.disposed || token !== state.token) return;
      console.error(`[${EXT_NAME}] Failed to read the file listing:`, error);
      state.status = PREVIEW_STATE.FAILED;
    } finally {
      if (!state.disposed) {
        state.pending = false;
        state.fetchedAt = Date.now();
        schedulePaint();
      }
    }
  }

  /**
   * Ask for the listing, reporting a failure rather than throwing it.
   *
   * @returns {void}
   */
  function refresh() {
    load(true).catch((error) => {
      console.error(`[${EXT_NAME}] Failed to ask for the file listing:`, error);
    });
  }

  /**
   * Hold a view position inside the rows on screen.
   *
   * @param {number} view - Wanted first row on screen.
   * @returns {number} The bounded position.
   */
  function clampView(view) {
    return clamp(Math.trunc(view), 0, Math.max(0, visible().length - state.layout.rows));
  }

  /**
   * Scroll the list.
   *
   * @param {number} view - Wanted first row on screen.
   * @returns {boolean} True when the list moved, which decides whether a wheel gesture belongs
   *   here or to the graph underneath.
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
    if (state.caret < state.view) setView(state.caret);
    else if (state.caret > state.view + rows - 1) setView(state.caret - rows + 1);
  }

  /**
   * Write the `files` widget once.
   *
   * @param {string} label - The label to tick or untick.
   * @returns {void}
   */
  function toggle(label) {
    if (state.disposed) return;
    const widget = findWidget(node, FILES_WIDGET);
    if (!widget) return;
    const lines = widgetLines();
    const at = lines.findIndex((line) => namesFile(line) && line.trim() === label);
    let ticked;
    if (at >= 0) {
      lines.splice(at, 1);
      ticked = false;
    } else if (lines.length && lines[lines.length - 1].trim() === "") {
      // The one line that is written over is the empty last line a text box holds the moment
      // somebody has pressed Enter in it.
      lines[lines.length - 1] = label;
      ticked = true;
    } else {
      lines.push(label);
      ticked = true;
    }
    write(widget, lines.join("\n"));
    setMessage(ticked ? `added ${label}` : `removed ${label}`);
  }

  /**
   * Write a value to the widget inside one undo entry.
   *
   * @param {object} widget - The `files` widget.
   * @param {string} value - The new value.
   * @returns {void}
   */
  function write(widget, value) {
    if (String(widget.value ?? "") === value) return;
    // The write is bracketed in the canvas change events the graph's change tracker listens
    // for, so a tick made from the keyboard gets an undo entry of its own. A commit that rides
    // on a pointerup is snapshotted by the tracker's own mouseup as well, and finds nothing
    // left to record.
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
   * What the panel is worth against the run, as a glyph and the measurement behind it.
   *
   * @param {object} picked - The reading from `chosen`.
   * @returns {{icon: string, detail: string}} The claim for `iconTitle`.
   */
  function readClaim(picked) {
    if (inputLinked(node, FILES_WIDGET)) {
      return {
        icon: ICON.WARNING,
        detail: "the files input is filled by a link, so the run archives whatever arrives "
          + "there and the ticks here are not read at all",
      };
    }
    const notes = [];
    if (inputLinked(node, PATHS_INPUT)) {
      notes.push(
        "the paths input is linked, so the archive also holds whatever arrives there, which "
        + "is not known until the run",
      );
    }
    if (picked.unknown) {
      notes.push(
        `${picked.unknown} picked line(s) name something this panel does not list, a typed `
        + "path or a file that has gone, and they are left exactly as they are",
      );
    }
    if (state.listing?.truncated) {
      notes.push(
        "the three folders hold more files than this listing offers, so a file may be "
        + "archivable without being on screen",
      );
    }
    if (notes.length) return { icon: ICON.APPROXIMATE, detail: notes.join("; ") };
    return {
      icon: ICON.EXACT,
      detail: "the files the archive will hold, in the order it will hold them, read from the "
        + "files box the run reads",
    };
  }

  /**
   * The state drawn in place of the rows.
   *
   * @returns {string} The words, empty when there are rows to draw.
   */
  function blockingText() {
    if (inputLinked(node, FILES_WIDGET)) return "files is linked, so the ticks here are not read";
    if (state.status === PREVIEW_STATE.FAILED) return "The file listing could not be loaded";
    if (!state.listing) return LABELS[PREVIEW_STATE.LOADING];
    if (!state.listing.entries.length) {
      return "The input, output and temp folders hold no files yet";
    }
    if (!visible().length) {
      return state.filter
        ? `No file matches "${state.filter}"`
        : "No folder is shown; click a folder above";
    }
    return "";
  }

  /**
   * Where the scrollbar and its thumb sit.
   *
   * @param {number} total - Rows on screen.
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
   * Find the row under a point.
   *
   * @param {{x: number, y: number}} point - Position in element pixels.
   * @returns {number|null} Its position among the rows on screen, or null when the point is on
   *   no row.
   */
  function hitRow(point) {
    const layout = state.layout;
    if (point.x < layout.x0 || point.x > layout.x1) return null;
    if (point.y < layout.rowsY || point.y >= layout.rowsY + layout.rows * ROW_HEIGHT) return null;
    const position = state.view + Math.floor((point.y - layout.rowsY) / ROW_HEIGHT);
    return position < visible().length ? position : null;
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
   * Draw the folder chips, and answer where each one is.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {object[]} One box per chip, for hit testing and hover text.
   */
  function drawChips(ctx, theme) {
    const layout = state.layout;
    const counts = new Map((state.listing?.roots ?? []).map((root) => [root.tag, root.files]));
    const chips = [];
    ctx.font = SMALL_FONT;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    let x = layout.x0;
    for (const tag of TAGS) {
      const count = counts.get(tag);
      const text = count === undefined ? `${tag} (none)` : `${tag} ${count}`;
      const width = Math.ceil(ctx.measureText(text).width) + CHIP_PAD * 2;
      if (x + width > layout.x1) break;
      const on = state.shown.has(tag);
      ctx.globalAlpha = on ? 0.22 : 0.08;
      ctx.fillStyle = on ? theme.accent : theme.fg;
      ctx.fillRect(x, layout.chipsY, width, CHIP_HEIGHT);
      ctx.globalAlpha = 1;
      ctx.fillStyle = on ? theme.fg : theme.fgDisabled;
      ctx.fillText(text, x + CHIP_PAD, layout.chipsY + CHIP_HEIGHT / 2);
      chips.push({ tag, x, y: layout.chipsY, width, height: CHIP_HEIGHT });
      x += width + CHIP_GAP;
    }
    if (state.filter) {
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(`"${state.filter}"`, layout.x1, layout.chipsY + CHIP_HEIGHT / 2);
    }
    return chips;
  }

  /**
   * Draw the rows.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} picked - The reading from `chosen`.
   * @param {object[]} rows - The rows on screen.
   * @param {number} right - Where the text column ends.
   * @returns {void}
   */
  function drawRows(ctx, theme, picked, rows, right) {
    const layout = state.layout;
    const focused = document.activeElement === root;
    for (let offset = 0; offset < layout.rows; offset += 1) {
      const position = state.view + offset;
      const row = rows[position];
      if (!row) break;
      const y = layout.rowsY + offset * ROW_HEIGHT;
      const order = picked.order.get(row.label);
      const ticked = order !== undefined;

      if (ticked) {
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = theme.accent;
        ctx.fillRect(layout.x0, y, right - layout.x0, ROW_HEIGHT);
        ctx.globalAlpha = 1;
      } else if (state.hover === position) {
        ctx.globalAlpha = 0.10;
        ctx.fillStyle = theme.fg;
        ctx.fillRect(layout.x0, y, right - layout.x0, ROW_HEIGHT);
        ctx.globalAlpha = 1;
      }
      if (focused && position === state.caret) {
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = theme.accent;
        ctx.strokeRect(layout.x0 + 0.5, y + 0.5, right - layout.x0 - 1, ROW_HEIGHT - 1);
        ctx.restore();
      }

      const boxX = layout.x0 + MARK_GAP;
      const boxY = y + (ROW_HEIGHT - MARK_BOX) / 2;
      ctx.lineWidth = 1;
      ctx.strokeStyle = ticked ? theme.accent : theme.border;
      ctx.strokeRect(boxX + 0.5, boxY + 0.5, MARK_BOX - 1, MARK_BOX - 1);
      if (ticked) {
        ctx.fillStyle = theme.accent;
        ctx.fillRect(boxX + 2, boxY + 2, MARK_BOX - 4, MARK_BOX - 4);
      }

      const textX = boxX + MARK_BOX + MARK_GAP;
      const sizeX = right - 2;
      const textRight = Math.max(textX + 1, sizeX - SIZE_COLUMN);
      ctx.save();
      ctx.beginPath();
      ctx.rect(textX, y, textRight - textX, ROW_HEIGHT);
      ctx.clip();
      ctx.font = BODY_FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = ticked ? theme.fg : theme.fgMuted;
      ctx.fillText(`${row.relative} [${row.tag}]`, textX, y + ROW_HEIGHT / 2);
      ctx.restore();

      ctx.font = SMALL_FONT;
      ctx.textAlign = "right";
      ctx.fillStyle = theme.fgMuted;
      ctx.fillText(sizeText(row.size), sizeX, y + ROW_HEIGHT / 2);
      if (ticked) {
        ctx.textAlign = "left";
        ctx.fillStyle = theme.accent;
        ctx.fillText(String(order), textRight + 2, y + ROW_HEIGHT / 2);
      }
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
   * Draw the footer, and collect the regions its hover text sits in.
   *
   * @param {CanvasRenderingContext2D} ctx - Target context.
   * @param {object} theme - Tokens from `readTheme`.
   * @param {object} picked - The reading from `chosen`.
   * @param {object[]} rows - The rows on screen.
   * @param {Array<object>} regions - Hover regions, appended to.
   * @returns {void}
   */
  function drawFooter(ctx, theme, picked, rows, regions) {
    const layout = state.layout;
    const middle = layout.footerY + layout.footerHeight / 2;
    ctx.font = BODY_FONT;
    ctx.textBaseline = "middle";

    const claim = readClaim(picked);
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

    const total = state.listing?.entries.length ?? 0;
    const count = `${picked.order.size} picked, ${rows.length} of ${total} shown`;
    ctx.textAlign = "right";
    ctx.fillStyle = theme.fgMuted;
    ctx.fillText(count, layout.x1, middle);
    const rightWidth = ctx.measureText(count).width;

    const note = state.message || standingNote(picked);
    const available = layout.x1 - layout.x0 - glyphWidth - rightWidth - 8;
    if (note && available > 12) {
      ctx.textAlign = "left";
      ctx.fillStyle = !state.message && picked.unknown ? theme.warning : theme.fgMuted;
      ctx.fillText(note, layout.x0 + glyphWidth, middle, available);
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
   * The standing note in the footer, which is a state rather than an explanation.
   *
   * @param {object} picked - The reading from `chosen`.
   * @returns {string} The words, empty when there is nothing to say.
   */
  function standingNote(picked) {
    if (inputLinked(node, FILES_WIDGET)) return "files is linked";
    if (picked.unknown) {
      return `${picked.unknown} picked line(s) are not in this list`;
    }
    if (!picked.count) return "nothing picked, so the node writes no archive";
    return "type to filter";
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
    const picked = chosen();
    const rows = visible();
    const regions = [];

    const chips = drawChips(ctx, theme);
    if (chips.length) {
      regions.push({
        x: chips[0].x,
        y: layout.chipsY,
        width: Math.max(1, chips[chips.length - 1].x + chips[chips.length - 1].width - chips[0].x),
        height: CHIP_HEIGHT,
        title: CHIP_TITLE,
      });
    }
    state.chips = chips;

    ctx.fillStyle = theme.inputBg;
    ctx.fillRect(layout.x0, layout.rowsY, layout.x1 - layout.x0, layout.rows * ROW_HEIGHT);

    const blocking = blockingText();
    if (blocking) {
      drawNotice(ctx, theme, blocking);
    } else {
      const bar = scrollGeometry(rows.length);
      const right = bar ? bar.x - 2 : layout.x1;
      drawRows(ctx, theme, picked, rows, right);
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

    drawFooter(ctx, theme, picked, rows, regions);
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
        console.error(`[${EXT_NAME}] Failed to draw the file browser:`, error);
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
        console.error(`[${EXT_NAME}] File browser input failed:`, error);
      }
      return undefined;
    };
  }

  /**
   * Repaint after the widget was edited by hand or by this panel.
   *
   * @returns {void}
   */
  function handleFilesChanged() {
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
    state.press = null;
    const point = localPoint(event);

    const chip = hitChip(point, state.chips ?? []);
    if (chip) {
      if (state.shown.has(chip.tag)) state.shown.delete(chip.tag);
      else state.shown.add(chip.tag);
      state.view = clampView(state.view);
      state.caret = clamp(state.caret, 0, Math.max(0, visible().length - 1));
      schedulePaint();
      return;
    }

    const rows = visible();
    const bar = scrollGeometry(rows.length);
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
    state.caret = position;
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
      const bar = scrollGeometry(visible().length);
      if (bar) dragThumb(point, bar, state.press.grip);
      return;
    }

    const position = hitRow(point);
    const overChip = hitChip(point, state.chips ?? []) !== null;
    root.style.cursor = position !== null || overChip ? "pointer" : "default";
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

    // The listing can have been rebuilt between the two halves of the gesture, so the row under
    // the pointer now is the one ticked, and only while it is the one it started on.
    const position = hitRow(localPoint(event));
    if (position === null || position !== press.position) return;
    const row = visible()[position];
    if (row) toggle(row.label);
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
    // At either end of the listing, and for a listing that fits on screen, the view does not
    // move and the gesture is the graph's.
    const before = state.view;
    setView(state.view + step);
    return state.view !== before;
  };

  /**
   * Move the caret.
   *
   * @param {number} step - Rows to move, negative toward the top.
   * @returns {void}
   */
  function moveCaret(step) {
    const rows = visible();
    if (!rows.length) {
      setMessage("no files to move through");
      return;
    }
    state.caret = clamp(state.caret + step, 0, rows.length - 1);
    reveal();
    schedulePaint();
  }

  const onKeyDown = (event) => {
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const rows = visible();
    const page = Math.max(1, state.layout.rows);
    let handled = true;

    switch (event.key) {
      case "ArrowUp":
      case "ArrowDown":
        moveCaret((event.key === "ArrowUp" ? -1 : 1) * (event.shiftKey ? COARSE_STEP : 1));
        break;
      case "PageUp":
      case "PageDown":
        moveCaret(event.key === "PageUp" ? -page : page);
        break;
      case "Home":
      case "End":
        if (!rows.length) {
          setMessage("no files to move through");
          break;
        }
        state.caret = event.key === "Home" ? 0 : rows.length - 1;
        reveal();
        schedulePaint();
        break;
      case "Enter":
      case " ": {
        const row = rows[state.caret];
        if (row) toggle(row.label);
        else setMessage("no file to pick");
        break;
      }
      case "Backspace":
        // Consumed whatever is on screen. Left unhandled these reach ComfyUI's own binding,
        // which deletes the node the panel is drawn on.
        if (state.filter) {
          state.filter = state.filter.slice(0, -1);
          state.view = 0;
          state.caret = 0;
          schedulePaint();
        } else {
          setMessage("nothing to delete here; Space picks and unpicks a file");
        }
        break;
      case "Delete": {
        const row = rows[state.caret];
        if (row && chosen().order.has(row.label)) toggle(row.label);
        else setMessage("the file itself is never deleted here; Space picks and unpicks it");
        break;
      }
      case "Escape":
        // The filter is the only thing this key drops. Unpicking every file is a selection in
        // the files box and a keystroke there, where an accident is undone by looking at it.
        state.filter = "";
        state.view = 0;
        state.caret = 0;
        endPress();
        schedulePaint();
        break;
      default:
        // Anything printable filters the list, which is how a folder of two thousand renders is
        // reached without a scrollbar. The filter is this panel's own and reaches no widget.
        if (event.key.length === 1) {
          if (state.filter.length < FILTER_CHARS) state.filter += event.key;
          state.view = 0;
          state.caret = 0;
          schedulePaint();
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
    // Focus can only leave mid-press when the gesture was interrupted, by another window taking
    // the pointer for instance, so the press is dropped.
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
    // A run writes files while the page is open, so the panel asks again for a listing it has
    // been holding for a while.
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

  // A ResizeObserver watches the border box, which the graph's zoom leaves alone, so the repaint
  // that follows a zoom comes from here.
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
    handleFilesChanged,
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
 * Append the panel to a node and wire it to the widget it draws.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachFilePicker(node) {
  if (!findWidget(node, FILES_WIDGET)) return;

  const picker = createFilePicker(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, picker, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Every multiline box on the node bounded the same way, so the panel above takes
  // the room past their ceiling instead of losing all of it to them.
  boundTextBoxes(node);

  chainWidgetCallback(node, FILES_WIDGET, picker.handleFilesChanged);
  chainWidgetCallback(node, NAMING_WIDGET, picker.schedulePaint);

  // A widget value is the default until `configure` has run, so what a saved workflow picked is
  // drawn from here rather than on creation.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      picker.schedulePaint();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to redraw after a workflow load:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      picker.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the file browser:`, error);
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
      category: ["WAS Node Suite", "Zip Save", "File browser"],
      name: "Show the source file picker",
      tooltip:
        "Draw the input, output and temp folders under the widgets of Zip Save, with a tick "
        + "against each file the archive will hold. The files box itself is always available "
        + "and holds the same lines either way. This applies to nodes added after the setting "
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
    if (proto.__was_zip_save_wrapped) return;
    proto.__was_zip_save_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachFilePicker(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the file browser:`, error);
      }
      return result;
    };
  },
});
