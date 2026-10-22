/**
 * The system prompt library on Load System Prompt: search, folders, model families, the prompts
 * and the chosen one's text, and the same picker in a window on Expand.
 *
 * It writes the node's `system_prompt` widget and folds that widget away.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createFloatingWindow } from "./interface/floating_window.js";
import { captureWheel } from "./interface/pointer.js";
import { readableBytes } from "./interface/report_panel.js";
import { fetchWithin } from "./interface/request.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";
import {
  FIELD, LABEL, LOOK, SANS, SCROLLBARS, SELECTED_FILL, button, el, focusRing, pill,
} from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.SystemPromptLibrary";
const SETTING_ID = "WAS.SystemPromptLibrary.ShowPicker";

const LOADER = "WASLoadSystemPrompt";
const SAVER = "WASSaveSystemPrompt";
const PROMPT_WIDGET = "system_prompt";

// The menu entry the node offers while the library is empty.
const NO_PROMPTS = "No System Prompts";

const ROUTE = "/was/interface/api/system_prompts";

// Where the library sits, as a portable install lays it out.
const LIBRARY_PATH = "ComfyUI/user/was-node-suite/system_prompts";

const UI_WIDGET_NAME = "was_system_prompt_ui";
const UI_WIDGET_TYPE = "was_system_prompt_picker";

// The picker's height on the node in node units, and the narrowest it stays readable at.
const PANEL_HEIGHT = 420;
const PANEL_MIN_WIDTH = 460;

// How long a listing stands before a picker asks again on its own.
const STALE_MS = 3000;

// Characters of a prompt's text drawn in a preview.
const PREVIEW_TEXT_CHARS = 262144;

// Family pills drawn on a row before the rest are counted.
const ROW_FAMILIES = 2;

// The sidebar's width on the node and in the window, in pixels.
const NODE_SIDEBAR = 132;
const WINDOW_SIDEBAR = 210;

// The window's opening size and the smallest a resize allows.
const WINDOW_WIDTH = 1000;
const WINDOW_HEIGHT = 640;
const WINDOW_MIN_WIDTH = 680;
const WINDOW_MIN_HEIGHT = 380;
const STORAGE_KEY = "was.system_prompt_library";

// Rows a page key moves the pick by.
const PAGE_ROWS = 8;

// Hues a family's pill is drawn in, chosen by its name.
const FAMILY_HUES = [212, 268, 328, 28, 148, 188, 352, 88];

// A book, as path data in a 24 unit box, for the title bar.
const BOOK_ICON = [
  "M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z",
  "M4 21.5A2.5 2.5 0 0 0 6.5 24H20v-5",
  "M8 7h8M8 11h6",
];

// A family filter that keeps the prompts naming no family.
const NO_FAMILY = "\u0000";

const STATUS = Object.freeze({
  LOADING: "loading",
  READY: "ready",
  MISSING: "missing",
  FAILED: "failed",
});

/**
 * Read whether the picker is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function pickerEnabled() {
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
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  return widgets.find((widget) => widget?.name === name) ?? null;
}

/**
 * The key a node's widget holds.
 *
 * @param {object} node - A Load System Prompt node.
 * @returns {string} The key, empty when the widget cannot be read.
 */
function keyOf(node) {
  const value = findWidget(node, PROMPT_WIDGET)?.value;
  return typeof value === "string" ? value : "";
}

/**
 * Whether the widget's input is filled by a link instead of the picker.
 *
 * @param {object} node - A Load System Prompt node.
 * @returns {boolean} True while a link is attached.
 */
function promptLinked(node) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  const input = inputs.find((each) => each?.name === PROMPT_WIDGET);
  return input?.link !== null && input?.link !== undefined;
}

/**
 * A string list read off an answer.
 *
 * @param {unknown} value - What the answer held.
 * @returns {string[]} The strings in it.
 */
function strings(value) {
  return Array.isArray(value) ? value.filter((item) => typeof item === "string" && item) : [];
}

/**
 * One entry of an answer, in the shape a picker draws.
 *
 * @param {object} raw - One entry as the route answered it.
 * @returns {object|null} The entry, or null when it carries no key.
 */
function readEntry(raw) {
  if (!raw || typeof raw !== "object" || typeof raw.key !== "string" || !raw.key) return null;
  const text = (value) => (typeof value === "string" ? value : "");
  const number = (value) => (Number.isFinite(Number(value)) ? Number(value) : 0);
  return {
    key: raw.key,
    name: text(raw.name) || raw.key,
    folder: text(raw.folder),
    families: strings(raw.families),
    description: text(raw.description),
    tags: strings(raw.tags),
    size: number(raw.size),
    modified: number(raw.modified),
  };
}

/**
 * The colour a family's pill is drawn in.
 *
 * @param {string} family - The family's name.
 * @returns {string} A CSS colour, the same for the same name in any case.
 */
function familyTone(family) {
  let hash = 0;
  for (const char of family.toLowerCase()) hash = (hash * 31 + char.codePointAt(0)) >>> 0;
  const hue = FAMILY_HUES[hash % FAMILY_HUES.length];
  return `color-mix(in srgb, hsl(${hue} 72% 52%) 78%, ${LOOK.text})`;
}

/**
 * A row of family pills, the rest counted in a last one.
 *
 * @param {string[]} families - The families.
 * @param {number} most - Pills drawn before the rest are counted.
 * @returns {HTMLElement} The row.
 */
function familyPills(families, most) {
  const row = el("span", "display:inline-flex;align-items:center;gap:4px;flex:0 0 auto;min-width:0");
  for (const family of families.slice(0, most)) row.appendChild(pill(family, familyTone(family)));
  if (families.length > most) {
    const more = pill(`+${families.length - most}`);
    more.title = families.slice(most).join(", ");
    row.appendChild(more);
  }
  return row;
}

/**
 * How long ago a time was, in the largest whole unit.
 *
 * @param {number} seconds - Seconds since the epoch.
 * @returns {string} Such as `5 min ago` or `3 d ago`, empty for no time.
 */
function ago(seconds) {
  if (!(seconds > 0)) return "";
  const elapsed = Math.max(0, Date.now() / 1000 - seconds);
  if (elapsed < 60) return "just now";
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)} min ago`;
  if (elapsed < 86400) return `${Math.floor(elapsed / 3600)} h ago`;
  if (elapsed < 86400 * 30) return `${Math.floor(elapsed / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

// The library as last read, shared by every picker on the page.
const library = {
  entries: [],
  status: STATUS.LOADING,
  fetchedAt: 0,
  token: 0,
  pending: null,
};

// Each entry's text, by key, as `{status, modified, text}`, and the requests in flight.
const texts = new Map();
const textRequests = new Map();

// Every picker on the page.
const PICKERS = new Set();

/**
 * Redraw every picker, or every picker on one node.
 *
 * @param {object} [node] - The node whose pickers to redraw. Left out, all of them.
 * @returns {void}
 */
function redraw(node) {
  for (const picker of PICKERS) {
    if (node && picker.node !== node) continue;
    try {
      picker.render();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to draw the picker:`, error);
    }
  }
}

/**
 * Bring a node's menu in line with the library.
 *
 * @param {object} node - A Load System Prompt node.
 * @returns {void}
 */
function syncOptions(node) {
  const values = findWidget(node, PROMPT_WIDGET)?.options?.values;
  if (!Array.isArray(values) || library.status !== STATUS.READY) return;
  const keys = library.entries.map((entry) => entry.key);
  const next = keys.length ? keys : [NO_PROMPTS];
  if (next.length === values.length && next.every((key, index) => values[index] === key)) return;
  values.splice(0, values.length, ...next);
}

/**
 * Read the library again, sharing one request between every picker that asks.
 *
 * @param {boolean} [force] - Ask even while the last answer is fresh.
 * @returns {Promise<void>} Resolved once every picker has been redrawn.
 */
function loadLibrary(force = false) {
  if (library.pending) return library.pending;
  if (!force && library.status === STATUS.READY && Date.now() - library.fetchedAt < STALE_MS) {
    return Promise.resolve();
  }
  const token = (library.token += 1);
  library.pending = (async () => {
    try {
      const response = await fetchWithin(ROUTE, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (token !== library.token) return;
      const entries = Array.isArray(data?.entries) ? data.entries : [];
      library.entries = entries.map(readEntry).filter(Boolean);
      library.status = STATUS.READY;
      for (const [key, held] of texts) {
        const entry = library.entries.find((each) => each.key === key);
        if (!entry || entry.modified !== held.modified) texts.delete(key);
      }
    } catch (error) {
      if (token !== library.token) return;
      console.error(`[${EXT_NAME}] Failed to read the system prompt library:`, error);
      library.status = library.entries.length ? STATUS.READY : STATUS.FAILED;
    } finally {
      if (token === library.token) {
        library.fetchedAt = Date.now();
        library.pending = null;
      }
    }
    for (const picker of PICKERS) syncOptions(picker.node);
    redraw();
  })();
  return library.pending;
}

/**
 * Ask for one entry's text, sharing one request per key.
 *
 * @param {object} entry - The entry.
 * @returns {void}
 */
function loadText(entry) {
  if (textRequests.has(entry.key)) return;
  const request = (async () => {
    try {
      const response = await fetchWithin(`${ROUTE}?key=${encodeURIComponent(entry.key)}`, { cache: "no-store" });
      if (response.status === 204) {
        texts.set(entry.key, { status: STATUS.MISSING, modified: entry.modified, text: "" });
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const answered = readEntry(data?.entry);
      const text = typeof data?.text === "string" ? data.text.slice(0, PREVIEW_TEXT_CHARS) : "";
      texts.set(entry.key, { status: STATUS.READY, modified: answered?.modified ?? entry.modified, text });
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the system prompt \`${entry.key}\`:`, error);
      texts.set(entry.key, { status: STATUS.FAILED, modified: entry.modified, text: "" });
    } finally {
      textRequests.delete(entry.key);
      redraw();
    }
  })();
  textRequests.set(entry.key, request);
}

/**
 * Write a key into a node's widget as one undoable edit.
 *
 * @param {object} node - A Load System Prompt node.
 * @param {string} key - The key to store.
 * @returns {boolean} True when the widget now holds the key.
 */
function writeKey(node, key) {
  const widget = findWidget(node, PROMPT_WIDGET);
  if (!widget || typeof key !== "string" || !key) return false;
  const values = widget.options?.values;
  if (Array.isArray(values) && !values.includes(key)) {
    values.splice(0, values.length, ...values.filter((value) => value !== NO_PROMPTS), key);
  }
  if (widget.value === key) return true;
  const canvas = app.canvas;
  const transactional = typeof canvas?.emitBeforeChange === "function"
    && typeof canvas?.emitAfterChange === "function";
  if (transactional) canvas.emitBeforeChange();
  try {
    widget.value = key;
  } finally {
    if (transactional) canvas.emitAfterChange();
  }
  node.setDirtyCanvas?.(true, true);
  return widget.value === key;
}

/**
 * Build a picker over the library for one node.
 *
 * @param {object} node - The Load System Prompt node it writes.
 * @param {object} options - `{layout, onExpand, onDone}`; `layout` is `node` for the panel on the
 *   node, with the preview under the list, or `window` for the larger view, with it beside.
 * @returns {{element: HTMLElement, node: object, render: () => void, focusSearch: () => void,
 *   dispose: () => void}} The picker.
 */
function createPicker(node, { layout = "node", onExpand = null, onDone = null } = {}) {
  const wide = layout === "window";
  const root = el("div", "box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;"
    + `min-height:0;overflow:hidden;outline:none;color:${LOOK.text};font:12px/1.45 ${SANS};`
    + (wide ? "" : `border-radius:8px;border:1px solid ${LOOK.border};background:${LOOK.body}`));
  root.tabIndex = 0;

  const toolbar = el("div", `flex:0 0 auto;display:flex;align-items:center;gap:8px;`
    + `padding:${wide ? "10px 14px" : "7px 8px"};border-bottom:1px solid ${LOOK.border};`
    + `background:${LOOK.surface}`);
  const search = focusRing(el("input", `${FIELD};flex:1 1 auto;min-width:80px;`
    + (wide ? "max-width:420px" : "padding:4px 8px;font-size:12px")));
  search.type = "search";
  search.placeholder = "Search names, descriptions, families";
  search.spellcheck = false;
  const count = el("span", `flex:0 0 auto;color:${LOOK.muted};font-variant-numeric:tabular-nums;white-space:nowrap`);
  toolbar.append(search, count);
  if (wide) {
    const where = el("span", `flex:1 1 auto;min-width:0;text-align:right;color:${LOOK.muted};`
      + "font:12px ui-monospace,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
    LIBRARY_PATH);
    where.title = "Library folder";
    toolbar.appendChild(where);
  }
  toolbar.appendChild(button("↻", () => loadLibrary(true), { hint: "Read the folder again", kind: "tool", logName: EXT_NAME }));
  if (onExpand) toolbar.appendChild(button("Expand", () => onExpand(node), { hint: "Open a larger view", kind: "tool", logName: EXT_NAME }));
  if (onDone) toolbar.appendChild(button("Done", () => onDone(), { hint: "Close the larger view", kind: "go", logName: EXT_NAME }));

  const main = el("div", `display:flex;min-height:0;${wide ? "flex:1 1 auto" : "flex:3 1 0"}`);
  const sidebar = el("div", `flex:0 0 ${wide ? WINDOW_SIDEBAR : NODE_SIDEBAR}px;display:flex;flex-direction:column;`
    + `gap:2px;padding:${wide ? "12px 10px" : "6px 5px"};overflow:auto;border-right:1px solid ${LOOK.border};`
    + `box-sizing:border-box;${SCROLLBARS}`);
  sidebar.dataset.scroll = "1";
  const list = el("div", `flex:1 1 auto;min-width:0;overflow:auto;display:flex;flex-direction:column;`
    + `gap:${wide ? 4 : 3}px;padding:${wide ? "10px" : "6px"};box-sizing:border-box;${SCROLLBARS}`);
  list.dataset.scroll = "1";
  const preview = el("div", "display:flex;flex-direction:column;min-width:0;min-height:0;box-sizing:border-box;"
    + (wide
      ? `flex:1 1 54%;gap:8px;padding:14px 16px;border-left:1px solid ${LOOK.border}`
      : `flex:2 1 0;gap:4px;padding:7px 9px 8px;border-top:1px solid ${LOOK.border};background:${LOOK.surface}`));
  if (wide) {
    const listColumn = el("div", "flex:1 1 46%;display:flex;min-width:0;min-height:0");
    listColumn.appendChild(list);
    main.append(sidebar, listColumn, preview);
    root.append(toolbar, main);
  } else {
    main.append(sidebar, list);
    root.append(toolbar, main, preview);
  }

  const state = { folder: "", family: "", search: "", disposed: false };

  /**
   * The entries the filters keep.
   *
   * @returns {object[]} Entries in library order.
   */
  function visibleEntries() {
    const needle = state.search.trim().toLowerCase();
    return library.entries.filter((entry) => {
      if (state.folder && entry.folder !== state.folder && !entry.folder.startsWith(`${state.folder}/`)) return false;
      if (state.family === NO_FAMILY && entry.families.length) return false;
      if (state.family && state.family !== NO_FAMILY
        && !entry.families.some((family) => family.toLowerCase() === state.family)) return false;
      if (!needle) return true;
      return [entry.name, entry.description, entry.key, ...entry.families, ...entry.tags]
        .join("\n").toLowerCase().includes(needle);
    });
  }

  /**
   * One pressable row of the sidebar.
   *
   * @param {string} label - What it says.
   * @param {number} number - The entries under it.
   * @param {boolean} chosen - Whether it is the filter in force.
   * @param {() => void} onPick - What pressing it does.
   * @param {number} [depth] - Folder levels it is indented by.
   * @param {string} [tone] - A colour for a dot before the label.
   * @returns {HTMLElement} The row.
   */
  function sideRow(label, number, chosen, onPick, depth = 0, tone = "") {
    const row = el("button", "display:flex;align-items:center;gap:6px;width:100%;box-sizing:border-box;"
      + `padding:${wide ? "5px 8px" : "3px 6px"};padding-left:${(wide ? 8 : 6) + depth * (wide ? 14 : 10)}px;`
      + `border-radius:5px;border:0;cursor:pointer;text-align:left;font:${wide ? 13 : 12}px/1.4 ${SANS};`
      + `color:${chosen ? LOOK.text : LOOK.muted};background:${chosen ? SELECTED_FILL : "transparent"}`);
    row.type = "button";
    if (tone) row.appendChild(el("span", `flex:0 0 auto;width:7px;height:7px;border-radius:50%;background:${tone}`));
    row.appendChild(el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap", label));
    row.appendChild(el("span", `flex:0 0 auto;font-size:11px;color:${LOOK.muted};font-variant-numeric:tabular-nums`, String(number)));
    row.addEventListener("mouseenter", () => { if (!chosen) row.style.color = LOOK.text; });
    row.addEventListener("mouseleave", () => { if (!chosen) row.style.color = LOOK.muted; });
    row.addEventListener("click", (event) => {
      event.preventDefault();
      onPick();
    });
    return row;
  }

  /**
   * Draw the folder tree and the family list.
   *
   * @returns {void}
   */
  function renderSidebar() {
    sidebar.replaceChildren();
    const heading = (text, top) => el("div", `${LABEL};font-size:10px;padding:${top}px 6px 3px`, text);
    sidebar.appendChild(heading("Folders", 0));
    sidebar.appendChild(sideRow("All prompts", library.entries.length, state.folder === "", () => setFolder("")));
    const counts = new Map();
    for (const entry of library.entries) {
      const parts = entry.folder ? entry.folder.split("/") : [];
      for (let depth = 1; depth <= parts.length; depth += 1) {
        const path = parts.slice(0, depth).join("/");
        counts.set(path, (counts.get(path) ?? 0) + 1);
      }
    }
    const folders = [...counts.keys()].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
    for (const path of folders) {
      const row = sideRow(path.split("/").pop(), counts.get(path), state.folder === path, () => setFolder(path),
        path.split("/").length);
      row.title = path;
      sidebar.appendChild(row);
    }

    const families = new Map();
    let unassigned = 0;
    for (const entry of library.entries) {
      if (!entry.families.length) unassigned += 1;
      for (const family of entry.families) {
        const id = family.toLowerCase();
        const known = families.get(id);
        if (known) known.count += 1;
        else families.set(id, { label: family, count: 1 });
      }
    }
    sidebar.appendChild(heading("Model families", wide ? 14 : 10));
    sidebar.appendChild(sideRow("Any", library.entries.length, state.family === "", () => setFamily("")));
    const sorted = [...families.entries()]
      .sort((a, b) => a[1].label.localeCompare(b[1].label, undefined, { sensitivity: "base" }));
    for (const [id, family] of sorted) {
      sidebar.appendChild(sideRow(family.label, family.count, state.family === id, () => setFamily(id), 0,
        familyTone(family.label)));
    }
    if (unassigned && families.size) {
      sidebar.appendChild(sideRow("No family", unassigned, state.family === NO_FAMILY, () => setFamily(NO_FAMILY)));
    }
  }

  /**
   * A centred message in a column, with an optional action under it.
   *
   * @param {HTMLElement} host - The column.
   * @param {string} words - The message.
   * @param {string[]} [facts] - Short lines under it.
   * @param {HTMLElement} [action] - A button under them.
   * @returns {void}
   */
  function notice(host, words, facts = [], action = null) {
    const box = el("div", "margin:auto;display:flex;flex-direction:column;align-items:center;gap:5px;"
      + "padding:16px;text-align:center;max-width:340px");
    box.appendChild(el("div", `font:600 ${wide ? 14 : 13}px ${SANS};color:${LOOK.text}`, words));
    for (const fact of facts) {
      box.appendChild(el("div", `font:12px/1.5 ${SANS};color:${LOOK.muted};overflow-wrap:anywhere`, fact));
    }
    if (action) {
      action.style.marginTop = "6px";
      box.appendChild(action);
    }
    host.appendChild(box);
  }

  /**
   * Draw the prompts the filters keep, the chosen one marked.
   *
   * @returns {void}
   */
  function renderList() {
    list.replaceChildren();
    const shown = visibleEntries();
    const total = library.entries.length;
    count.textContent = library.status === STATUS.READY
      ? (shown.length === total ? `${total} ${total === 1 ? "prompt" : "prompts"}` : `${shown.length} of ${total}`)
      : "";
    if (library.status === STATUS.LOADING && !total) {
      notice(list, "Loading");
      return;
    }
    if (library.status === STATUS.FAILED) {
      notice(list, "Library could not be read", [], button("Try again", () => loadLibrary(true), { kind: "plain", logName: EXT_NAME }));
      return;
    }
    if (!total) {
      notice(list, "No system prompts yet", ["Save System Prompt adds one", LIBRARY_PATH],
        button("Refresh", () => loadLibrary(true), { kind: "plain", logName: EXT_NAME }));
      return;
    }
    if (!shown.length) {
      notice(list, "No prompts match", [], button("Clear filters", () => {
        state.search = "";
        search.value = "";
        state.folder = "";
        state.family = "";
        render();
      }, { kind: "plain", logName: EXT_NAME }));
      return;
    }
    const chosen = keyOf(node);
    for (const entry of shown) {
      const on = entry.key === chosen;
      const row = el("div", `display:flex;flex-direction:column;gap:1px;padding:${wide ? "8px 10px" : "5px 8px"};`
        + `border-radius:6px;cursor:pointer;border:1px solid ${on ? LOOK.accent : LOOK.border};`
        + `background:${on ? SELECTED_FILL : LOOK.surface}`);
      row.dataset.key = entry.key;
      const top = el("div", "display:flex;align-items:center;gap:8px;min-width:0");
      top.appendChild(el("span", "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
        + `font:600 ${wide ? 13 : 12}px/1.4 ${SANS}`, entry.name));
      top.appendChild(familyPills(entry.families, ROW_FAMILIES));
      row.appendChild(top);
      const sub = [entry.description, state.folder === entry.folder ? "" : entry.folder].filter(Boolean).join(" · ");
      if (sub) {
        row.appendChild(el("div", `font:${wide ? 12 : 11}px/1.4 ${SANS};color:${LOOK.muted};overflow:hidden;`
          + "text-overflow:ellipsis;white-space:nowrap", sub));
      }
      row.title = entry.key;
      row.addEventListener("click", () => pick(entry.key));
      row.addEventListener("dblclick", () => onDone?.());
      row.addEventListener("mouseenter", () => { if (entry.key !== keyOf(node)) row.style.borderColor = LOOK.muted; });
      row.addEventListener("mouseleave", () => { if (entry.key !== keyOf(node)) row.style.borderColor = LOOK.border; });
      list.appendChild(row);
    }
  }

  /**
   * Draw the chosen prompt: its name, families, place and text.
   *
   * @returns {void}
   */
  function renderPreview() {
    preview.replaceChildren();
    const key = keyOf(node);
    if (promptLinked(node)) {
      notice(preview, "Set by the link");
      return;
    }
    if (!key || key === NO_PROMPTS) {
      if (library.entries.length) notice(preview, "Pick a prompt");
      return;
    }
    const entry = library.entries.find((each) => each.key === key);
    if (!entry) {
      if (library.status !== STATUS.READY) {
        notice(preview, "Loading");
        return;
      }
      preview.appendChild(el("div", `font:600 13px ${SANS};overflow-wrap:anywhere`, key));
      preview.appendChild(el("div", `font:12px ${SANS};color:${LOOK.warning}`, "Not in the library"));
      return;
    }
    const head = el("div", "flex:0 0 auto;display:flex;align-items:center;gap:8px;min-width:0;flex-wrap:wrap");
    head.appendChild(el("span", `font:600 ${wide ? 17 : 13}px/1.3 ${SANS};overflow-wrap:anywhere;min-width:0`, entry.name));
    if (entry.families.length) head.appendChild(familyPills(entry.families, entry.families.length));
    preview.appendChild(head);
    preview.appendChild(el("div", `flex:0 0 auto;font:${wide ? 12 : 11}px/1.4 ${SANS};color:${LOOK.muted};`
      + "font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap",
    [entry.folder || "top level", readableBytes(entry.size), ago(entry.modified)].filter(Boolean).join(" · ")));
    if (entry.description) {
      preview.appendChild(el("div", `flex:0 0 auto;font:${wide ? 13 : 12}px/1.45 ${SANS};color:${LOOK.muted};`
        + (wide ? "" : "overflow:hidden;text-overflow:ellipsis;white-space:nowrap"), entry.description));
    }
    const body = el("div", "flex:1 1 auto;min-height:0;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;"
      + `padding:${wide ? "10px 12px" : "6px 8px"};border-radius:6px;border:1px solid ${LOOK.border};`
      + `background:${wide ? LOOK.surface : LOOK.body};font:${wide ? 13 : 12}px/1.55 ${SANS};${SCROLLBARS}`);
    body.dataset.scroll = "1";
    const held = texts.get(entry.key);
    if (held?.status === STATUS.READY && held.modified === entry.modified) {
      body.textContent = held.text || "This prompt holds no text";
      body.style.color = held.text ? LOOK.text : LOOK.muted;
    } else if (held?.status === STATUS.MISSING) {
      body.textContent = "No longer in the library";
      body.style.color = LOOK.warning;
    } else if (held?.status === STATUS.FAILED) {
      body.textContent = "File could not be read";
      body.style.color = LOOK.danger;
    } else {
      body.textContent = "Loading";
      body.style.color = LOOK.muted;
      loadText(entry);
    }
    preview.appendChild(body);
    if (wide) {
      preview.appendChild(el("div", `flex:0 0 auto;font:12px ui-monospace,Consolas,monospace;color:${LOOK.muted};`
        + "overflow:hidden;text-overflow:ellipsis;white-space:nowrap", entry.key));
    }
  }

  /**
   * Redraw every part of the picker.
   *
   * @returns {void}
   */
  function render() {
    if (state.disposed) return;
    renderSidebar();
    renderList();
    renderPreview();
  }

  /**
   * Scroll the chosen row into view.
   *
   * @returns {void}
   */
  function revealChosen() {
    const key = keyOf(node);
    for (const row of list.children) {
      if (row.dataset?.key === key) {
        row.scrollIntoView({ block: "nearest" });
        return;
      }
    }
  }

  /**
   * Load a prompt on the node.
   *
   * @param {string} key - Its key.
   * @returns {void}
   */
  function pick(key) {
    if (!node.graph) return;
    if (writeKey(node, key)) {
      redraw(node);
      revealChosen();
    }
  }

  /**
   * Narrow the list to a folder.
   *
   * @param {string} folder - The folder, empty for all.
   * @returns {void}
   */
  function setFolder(folder) {
    state.folder = folder;
    render();
  }

  /**
   * Narrow the list to a family.
   *
   * @param {string} family - The family in lower case, empty for any, NUL for none.
   * @returns {void}
   */
  function setFamily(family) {
    state.family = family;
    render();
  }

  /**
   * Move the pick through the rows on screen.
   *
   * @param {number} step - Rows to move, negative upward.
   * @returns {void}
   */
  function movePick(step) {
    const shown = visibleEntries();
    if (!shown.length) return;
    const at = shown.findIndex((entry) => entry.key === keyOf(node));
    const next = Math.max(0, Math.min(shown.length - 1, at < 0 ? 0 : at + step));
    pick(shown[next].key);
  }

  const onSearchInput = () => {
    state.search = search.value;
    renderList();
  };
  const onSearchKey = (event) => {
    let handled = true;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") movePick(event.key === "ArrowDown" ? 1 : -1);
    else if (event.key === "Enter") onDone?.();
    else if (event.key === "Escape" && search.value) {
      search.value = "";
      state.search = "";
      render();
    } else if (event.key === "Escape" && onDone) onDone();
    else handled = false;
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };
  const onRootKey = (event) => {
    if (event.target === search || event.ctrlKey || event.altKey || event.metaKey) return;
    let handled = true;
    switch (event.key) {
      case "ArrowDown": movePick(1); break;
      case "ArrowUp": movePick(-1); break;
      case "PageDown": movePick(PAGE_ROWS); break;
      case "PageUp": movePick(-PAGE_ROWS); break;
      case "Home": movePick(-Infinity); break;
      case "End": movePick(Infinity); break;
      case "/": search.focus(); break;
      case "Enter": onDone?.(); break;
      case "Delete":
      case "Backspace":
        break;
      default: handled = false;
    }
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };
  const onPointerDown = (event) => {
    if (event.button === 1) app.canvas?.processMouseDown?.(event);
  };
  const onContextMenu = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  search.addEventListener("input", onSearchInput);
  search.addEventListener("keydown", onSearchKey);
  root.addEventListener("keydown", onRootKey);
  root.addEventListener("pointerdown", onPointerDown);
  root.addEventListener("contextmenu", onContextMenu);
  const onEnter = () => loadLibrary(false);
  root.addEventListener("pointerenter", onEnter);
  const releaseWheel = wide ? () => {} : captureWheel(root, (event) => {
    const scroller = event.target?.closest?.("[data-scroll]");
    if (!scroller || !root.contains(scroller) || scroller.scrollHeight <= scroller.clientHeight) return false;
    const before = scroller.scrollTop;
    scroller.scrollTop += event.deltaY;
    return scroller.scrollTop !== before;
  });

  const picker = {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    node,
    render,
    revealChosen,
    focusSearch() {
      search.focus({ preventScroll: true });
    },
    dispose() {
      state.disposed = true;
      releaseWheel();
      search.removeEventListener("input", onSearchInput);
      search.removeEventListener("keydown", onSearchKey);
      root.removeEventListener("keydown", onRootKey);
      root.removeEventListener("pointerdown", onPointerDown);
      root.removeEventListener("contextmenu", onContextMenu);
      root.removeEventListener("pointerenter", onEnter);
      PICKERS.delete(picker);
      root.remove();
    },
  };
  PICKERS.add(picker);
  render();
  return picker;
}

let held = null;

/**
 * The larger view, built on first use and shared by every node.
 *
 * @returns {{open: (node: object) => void, close: () => void, nodeRemoved: (node: object) => void}}
 *   The window.
 */
function libraryWindow() {
  if (held) return held;
  let picker = null;
  const win = createFloatingWindow({
    title: "System Prompts",
    icon: BOOK_ICON,
    width: WINDOW_WIDTH,
    height: WINDOW_HEIGHT,
    minWidth: WINDOW_MIN_WIDTH,
    minHeight: WINDOW_MIN_HEIGHT,
    storageKey: STORAGE_KEY,
    logName: EXT_NAME,
    onClose: () => release(),
  });

  /**
   * Take the picker out of the window.
   *
   * @returns {void}
   */
  function release() {
    picker?.dispose();
    picker = null;
  }

  held = {
    open(node) {
      release();
      picker = createPicker(node, { layout: "window", onDone: () => held.close() });
      win.body.appendChild(picker.element);
      win.setBadge(`#${node.id} ${node.title || "Load System Prompt"}`);
      win.open();
      win.expand();
      loadLibrary(true).then(() => picker?.revealChosen());
      picker.focusSearch();
    },
    close() {
      release();
      win.close();
    },
    nodeRemoved(node) {
      if (picker?.node === node) held.close();
    },
  };
  return held;
}

/**
 * Bring every picker and menu in line with the library after a save.
 *
 * @returns {void}
 */
function librarySaved() {
  loadLibrary(true).catch((error) => console.error(`[${EXT_NAME}] Failed to refresh after a save:`, error));
}

/**
 * Put the picker on a node, fold the menu it replaces, and follow the widget.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPicker(node) {
  const widget = findWidget(node, PROMPT_WIDGET);
  if (!widget) return;
  const picker = createPicker(node, { layout: "node", onExpand: (target) => libraryWindow().open(target) });
  appendInterfaceWidget(node, picker, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
  setWidgetHidden(widget, true);
  chainWidgetCallback(node, PROMPT_WIDGET, () => redraw(node), EXT_NAME);

  // Read again once a saved workflow has set the widget.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      setWidgetHidden(findWidget(this, PROMPT_WIDGET), true);
      redraw(this);
      picker.revealChosen();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the prompt after a workflow load:`, error);
    }
    return result;
  };

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    redraw(this);
    return result;
  };

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      picker.dispose();
      held?.nodeRemoved(this);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the picker:`, error);
    }
    return result;
  };

  loadLibrary(false);
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "System prompt library", "Picker"],
      name: "Show the system prompt library on the node",
      tooltip:
        "Draw the library on Load System Prompt: search, folders, model families and the chosen "
        + "prompt's text, in place of its menu. The node runs the same either way. This applies "
        + "to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  setup() {
    // A run that starts Save System Prompt refreshes every picker once the next node starts or
    // the run ends.
    let saving = false;
    const settle = () => {
      if (!saving) return;
      saving = false;
      librarySaved();
    };
    api.addEventListener("executing", (event) => {
      try {
        const detail = event?.detail;
        const id = detail && typeof detail === "object" ? (detail.display_node ?? detail.node) : detail;
        settle();
        if (id === null || id === undefined) return;
        const node = app.graph?.getNodeById?.(id) ?? app.graph?.getNodeById?.(Number(id));
        saving = node?.type === SAVER || node?.comfyClass === SAVER;
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read a running node:`, error);
      }
    });
    api.addEventListener("execution_success", settle);
    api.addEventListener("execution_error", settle);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== LOADER) return;
    const proto = nodeType.prototype;
    // Wrapped once, however often the definitions are registered.
    if (proto.__was_system_prompt_wrapped) return;
    proto.__was_system_prompt_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (pickerEnabled()) attachPicker(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the system prompt picker:`, error);
      }
      return result;
    };
  },
});
