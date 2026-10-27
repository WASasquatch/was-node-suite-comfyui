/**
 * Titled, colourable bars above a node's repeated rows and above single widgets.
 *
 * Bars are never saved. Colours and names are kept in `node.properties` under `ROW_COLOURS_KEY`
 * and `ROW_NAMES_KEY`, by row number from 1 or by label key.
 */

import { app } from "../../../scripts/app.js";
import { addSectionHeader, redrawHeaders } from "./decoration.js";
import { withGraphChange } from "./region.js";

const LOG_NAME = "WASNodeSuite.RowHeaders";

// Property holding each header's colour, as `{ "<row or label key>": "<palette name>" }`.
export const ROW_COLOURS_KEY = "was_row_colours";

// Property holding each renamed header's name, as `{ "<row or label key>": "<name>" }`.
export const ROW_NAMES_KEY = "was_row_names";

// Colours a row header may take, in menu order.
export const ROW_PALETTE = {
  red: { label: "Red", fill: "rgba(200, 70, 70, 0.38)", stripe: "#d65a5a", text: "#ffe8e8" },
  orange: { label: "Orange", fill: "rgba(215, 125, 45, 0.38)", stripe: "#e38a3c", text: "#fff0e0" },
  yellow: { label: "Yellow", fill: "rgba(200, 175, 50, 0.36)", stripe: "#dcc143", text: "#fff8d8" },
  green: { label: "Green", fill: "rgba(70, 160, 90, 0.38)", stripe: "#4fb368", text: "#e6ffea" },
  teal: { label: "Teal", fill: "rgba(50, 160, 160, 0.38)", stripe: "#3db6b6", text: "#e0ffff" },
  blue: { label: "Blue", fill: "rgba(70, 120, 210, 0.42)", stripe: "#5b8ce3", text: "#e6eeff" },
  purple: { label: "Purple", fill: "rgba(140, 90, 200, 0.40)", stripe: "#9d6ddb", text: "#f2e8ff" },
  pink: { label: "Pink", fill: "rgba(210, 90, 160, 0.38)", stripe: "#e272b3", text: "#ffe8f5" },
  grey: { label: "Grey", fill: "rgba(140, 140, 140, 0.32)", stripe: "#a0a0a0", text: "#f2f2f2" },
};

// Swatch drawn before each colour's name in the menu.
const SWATCH = {
  red: "🟥", orange: "🟧", yellow: "🟨", green: "🟩", teal: "🟦",
  blue: "🔵", purple: "🟪", pink: "🩷", grey: "⬜",
};

// Node -> its colourable headers as `{widget, key, label, renamable}`, labels first, then rows.
const HEADERS = new WeakMap();

/**
 * Record one colourable header on a node.
 *
 * @param {object} node - The node the header is on.
 * @param {object} entry - `{widget, key, label, renamable}`.
 * @returns {void}
 */
function remember(node, entry) {
  const known = HEADERS.get(node) ?? [];
  known.push(entry);
  HEADERS.set(node, known);
}

/**
 * The palette name a header is coloured with.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @returns {string|null} A key of `ROW_PALETTE`, or null for the plain bar.
 */
export function rowColour(node, row) {
  const stored = node?.properties?.[ROW_COLOURS_KEY]?.[String(row)];
  return Object.hasOwn(ROW_PALETTE, stored ?? "") ? stored : null;
}

/**
 * Colour several headers at once, leaving every other header as it is.
 *
 * @param {object} node - The node the headers belong to.
 * @param {Object<string, string|null>} colours - A key of `ROW_PALETTE`, or null to clear, by
 *   row number or label key.
 * @returns {void}
 */
export function setRowColours(node, colours) {
  if (!node) return;
  withGraphChange(() => {
    node.properties ??= {};
    const stored = { ...(node.properties[ROW_COLOURS_KEY] ?? {}) };
    for (const [row, colour] of Object.entries(colours ?? {})) {
      if (colour && Object.hasOwn(ROW_PALETTE, colour)) stored[String(row)] = colour;
      else delete stored[String(row)];
    }
    if (Object.keys(stored).length) node.properties[ROW_COLOURS_KEY] = stored;
    else delete node.properties[ROW_COLOURS_KEY];
  });
  redrawHeaders(node);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Colour one header, or clear it back to the plain bar.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @param {string|null} colour - A key of `ROW_PALETTE`, or null to clear.
 * @returns {void}
 */
export function setRowColour(node, row, colour) {
  setRowColours(node, { [String(row)]: colour });
}

/**
 * Menu entries that colour one header.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @returns {object[]} One entry per colour, then one clearing it.
 */
export function colourEntries(node, row) {
  const current = rowColour(node, row);
  const entries = Object.entries(ROW_PALETTE).map(([key, colour]) => ({
    content: menuText(`${SWATCH[key] ?? ""} ${colour.label}${key === current ? " ✓" : ""}`),
    callback: () => setRowColour(node, row, key),
  }));
  entries.push({
    content: "Default",
    disabled: current === null,
    callback: () => setRowColour(node, row, null),
  });
  return entries;
}

/**
 * Open the colour list for one header wherever the pointer is.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @param {string} title - Menu title.
 * @param {Event} [event] - What opened the menu, used to place it.
 * @returns {void}
 */
export function askRowColour(node, row, title, event) {
  const ContextMenu = window.LiteGraph?.ContextMenu;
  if (!ContextMenu) return;
  new ContextMenu(colourEntries(node, row), { title, event: event ?? window.event });
}

/**
 * Text for a context menu entry, with every `<` turned into `‹`.
 *
 * @param {*} value - The text, as a workflow or a widget holds it.
 * @returns {string} The text with no `<` left in it.
 */
export function menuText(value) {
  return String(value ?? "").replaceAll("<", "\u2039");
}

/**
 * The name a header was given.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @returns {string|null} The name, or null where the header keeps its own text.
 */
export function rowName(node, row) {
  const stored = node?.properties?.[ROW_NAMES_KEY]?.[String(row)];
  return typeof stored === "string" && stored.trim() ? stored.trim() : null;
}

/**
 * Name several headers at once, leaving every other header as it is.
 *
 * @param {object} node - The node the headers belong to.
 * @param {Object<string, string|null>} names - A name, or null or empty to clear, by row
 *   number or label key.
 * @returns {void}
 */
export function setRowNames(node, names) {
  if (!node) return;
  withGraphChange(() => {
    node.properties ??= {};
    const stored = { ...(node.properties[ROW_NAMES_KEY] ?? {}) };
    for (const [row, name] of Object.entries(names ?? {})) {
      const text = typeof name === "string" ? name.trim() : "";
      if (text) stored[String(row)] = text;
      else delete stored[String(row)];
    }
    if (Object.keys(stored).length) node.properties[ROW_NAMES_KEY] = stored;
    else delete node.properties[ROW_NAMES_KEY];
  });
  redrawHeaders(node);
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Name one header, or clear it back to its own text.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @param {string|null} name - The name, or null or empty to clear.
 * @returns {void}
 */
export function setRowName(node, row, name) {
  setRowNames(node, { [String(row)]: name });
}

/**
 * Ask for a header's name and keep the answer.
 *
 * @param {object} node - The node the header belongs to.
 * @param {number|string} row - Row number from 1, or a label's key.
 * @param {string} label - What the header is called in the dialog, as `LoRA 2`.
 * @returns {Promise<void>} Settled once the name is kept or the dialog dismissed.
 */
export async function askRowName(node, row, label) {
  const current = rowName(node, row) ?? "";
  let answer = null;
  try {
    const dialog = app?.extensionManager?.dialog;
    answer = typeof dialog?.prompt === "function"
      ? await dialog.prompt({ title: `Rename ${label}`, message: "Name", defaultValue: current })
      : window.prompt(`Rename ${label}`, current);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to ask for ${label}'s name:`, error);
    return;
  }
  if (answer === null || answer === undefined) return;
  setRowName(node, row, String(answer));
}

/**
 * Entries for one header's own menu: renaming where it can be renamed, then its colours.
 *
 * @param {object} node - The node the header belongs to.
 * @param {object} entry - The header, as `{key, label, renamable}`.
 * @returns {Array<object|null>} Menu entries, null drawing a rule.
 */
function headerEntries(node, entry) {
  const entries = [];
  if (entry.renamable) {
    entries.push(
      { content: "✏️ Rename", callback: () => askRowName(node, entry.key, entry.label) },
      {
        content: "↩️ Default Name",
        disabled: rowName(node, entry.key) === null,
        callback: () => setRowName(node, entry.key, null),
      },
      null,
    );
  }
  return [...entries, ...colourEntries(node, entry.key)];
}

/**
 * Open one header's menu wherever the pointer is.
 *
 * @param {object} node - The node the header belongs to.
 * @param {object} entry - The header, as `{key, label, renamable}`.
 * @param {Event} [event] - What opened the menu, used to place it.
 * @returns {void}
 */
function askHeader(node, entry, event) {
  const ContextMenu = window.LiteGraph?.ContextMenu;
  if (!ContextMenu) return;
  const title = entry.renamable ? entry.label : `${entry.label} colour`;
  new ContextMenu(headerEntries(node, entry), { title, event: event ?? window.event });
}

/**
 * Put a colourable bar above one widget.
 *
 * @param {object} node - The node to draw on.
 * @param {object} options - Settings.
 * @param {string} options.name - Header widget name.
 * @param {string} options.key - Key its colour is kept under.
 * @param {string} options.title - Text of the bar.
 * @param {string} options.before - Name of the widget the bar sits above.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addLabelHeader(node, { name, key, title, before }) {
  const entry = { key, label: title, renamable: false };
  entry.widget = addSectionHeader(node, {
    name,
    title,
    before,
    colour: () => ROW_PALETTE[rowColour(node, key)] ?? null,
    onClick: (host, event) => askHeader(host, entry, event),
  });
  if (entry.widget) remember(node, entry);
  return entry.widget;
}

/**
 * Put a header above every row of a node.
 *
 * @param {object} node - The node to draw on.
 * @param {object} options - Settings.
 * @param {string[][]} options.groups - Widget names, one array per row, in row order.
 * @param {(row: number) => string} options.name - Header widget name for a row.
 * @param {(node: object, row: number) => string} options.title - Text of a row's header.
 * @param {string} options.noun - What a row is called in menus, as `Segment`.
 * @param {number} [options.before] - Position within a row of the widget the header sits
 *   above, the first by default.
 * @param {(node: object, row: number) => object|null} [options.tint] - The `{fill, stripe,
 *   text}` a row takes where no colour is chosen for it.
 * @param {boolean} [options.renamable] - Offer a name per row, read back with `rowName`.
 * @returns {string[]} Header widget names, one per row, for folding with their rows.
 */
export function addRowHeaders(node, { groups, name, title, noun, before = 0, tint, renamable = false }) {
  groups.forEach((names, index) => {
    const row = index + 1;
    const entry = { key: row, label: `${noun} ${row}`, renamable };
    entry.widget = addSectionHeader(node, {
      name: name(row),
      title: () => title(node, row),
      before: names[before],
      colour: () => ROW_PALETTE[rowColour(node, row)] ?? tint?.(node, row) ?? null,
      onClick: (host, event) => askHeader(host, entry, event),
    });
    if (entry.widget) remember(node, entry);
  });
  return groups.map((unused, index) => name(index + 1));
}

/**
 * The header the pointer is over, on the canvas renderer.
 *
 * @param {object} node - The node the menu was opened on.
 * @returns {{key: number|string, label: string}|null} The header, or null when none is under
 *   the pointer.
 */
export function headerUnderPointer(node) {
  const headers = HEADERS.get(node);
  const mouse = app?.canvas?.graph_mouse;
  // Nodes 2.0 draws widgets in their own elements and never updates `last_y`.
  if (window.LiteGraph?.vueNodesMode) return null;
  const y = Number(mouse?.[1]) - Number(node?.pos?.[1]);
  if (!headers || !Number.isFinite(y)) return null;
  for (const entry of headers) {
    const widget = entry.widget;
    if (!widget || widget.hidden || !Number.isFinite(widget.last_y)) continue;
    const height = widget.computedHeight ?? widget.computeSize?.()[1] ?? 0;
    if (y >= widget.last_y && y <= widget.last_y + height) return entry;
  }
  return null;
}

/**
 * Headers that are drawn.
 *
 * @param {object} node - The node to read.
 * @returns {Array<{key: number|string, label: string}>} Labels first, then rows in order.
 */
export function drawnHeaders(node) {
  return (HEADERS.get(node) ?? []).filter((entry) => entry.widget && !entry.widget.hidden);
}

/**
 * Node menu entries colouring and renaming its headers: the one under the pointer directly, any
 * otherwise.
 *
 * @param {object} node - The node the menu was opened on.
 * @returns {object[]} Entries for `getNodeMenuItems`.
 */
export function rowColourMenu(node) {
  try {
    const under = headerUnderPointer(node);
    if (under !== null) {
      const entries = [
        null,
        {
          content: menuText(`🎨 ${under.label} Colour`),
          has_submenu: true,
          submenu: { options: colourEntries(node, under.key) },
        },
      ];
      if (under.renamable) {
        entries.push({
          content: menuText(`✏️ Rename ${under.label}`),
          callback: () => askRowName(node, under.key, under.label),
        });
      }
      return entries;
    }
    const drawn = drawnHeaders(node);
    if (!drawn.length) return [];
    const entries = [
      null,
      {
        content: "🎨 Header Colour",
        has_submenu: true,
        submenu: {
          options: drawn.map(({ key, label }) => {
            const colour = rowColour(node, key);
            return {
              content: menuText(`${colour ? SWATCH[colour] + " " : ""}${label}`),
              callback: (value, options, event) =>
                askRowColour(node, key, `${label} colour`, event),
            };
          }),
        },
      },
    ];
    const renamable = drawn.filter((entry) => entry.renamable);
    if (renamable.length) {
      entries.push({
        content: "✏️ Rename Header",
        has_submenu: true,
        submenu: {
          options: renamable.map(({ key, label }) => {
            const named = rowName(node, key);
            return {
              content: menuText(named ? `${label} · ${named}` : label),
              callback: () => askRowName(node, key, label),
            };
          }),
        },
      });
    }
    return entries;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to build the header colour menu:`, error);
    return [];
  }
}
