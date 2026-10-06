/**
 * A titled, colourable bar above each repeated row of a node.
 *
 * Bars are decorations and never saved. A row's colour is kept in `node.properties` under
 * `ROW_COLOURS_KEY`, as a palette name per row number counted from 1.
 */

import { app } from "../../../scripts/app.js";
import { addSectionHeader } from "./decoration.js";
import { withGraphChange } from "./region.js";

const LOG_NAME = "WASNodeSuite.RowHeaders";

// Property holding each row's colour, as `{ "<row>": "<palette name>" }`.
export const ROW_COLOURS_KEY = "was_row_colours";

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

// Node -> its header widgets, one per row, in row order.
const HEADERS = new WeakMap();

/**
 * The palette name a row is coloured with.
 *
 * @param {object} node - The node the row belongs to.
 * @param {number} row - Row number, from 1.
 * @returns {string|null} A key of `ROW_PALETTE`, or null for the plain bar.
 */
export function rowColour(node, row) {
  const stored = node?.properties?.[ROW_COLOURS_KEY]?.[String(row)];
  return Object.hasOwn(ROW_PALETTE, stored ?? "") ? stored : null;
}

/**
 * Colour one row's header, or clear it back to the plain bar.
 *
 * @param {object} node - The node the row belongs to.
 * @param {number} row - Row number, from 1.
 * @param {string|null} colour - A key of `ROW_PALETTE`, or null to clear.
 * @returns {void}
 */
export function setRowColour(node, row, colour) {
  if (!node) return;
  withGraphChange(() => {
    node.properties ??= {};
    const colours = { ...(node.properties[ROW_COLOURS_KEY] ?? {}) };
    if (colour && Object.hasOwn(ROW_PALETTE, colour)) colours[String(row)] = colour;
    else delete colours[String(row)];
    if (Object.keys(colours).length) node.properties[ROW_COLOURS_KEY] = colours;
    else delete node.properties[ROW_COLOURS_KEY];
  });
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Menu entries that colour one row.
 *
 * @param {object} node - The node the row belongs to.
 * @param {number} row - Row number, from 1.
 * @returns {object[]} One entry per colour, then one clearing it.
 */
export function colourEntries(node, row) {
  const current = rowColour(node, row);
  const entries = Object.entries(ROW_PALETTE).map(([key, colour]) => ({
    content: `${SWATCH[key] ?? ""} ${colour.label}${key === current ? " ✓" : ""}`,
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
 * Open the colour list for one row wherever the pointer is.
 *
 * @param {object} node - The node the row belongs to.
 * @param {number} row - Row number, from 1.
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
 * @returns {string[]} Header widget names, one per row, for folding with their rows.
 */
export function addRowHeaders(node, { groups, name, title, noun, before = 0, tint }) {
  const headers = [];
  groups.forEach((names, index) => {
    const row = index + 1;
    const widget = addSectionHeader(node, {
      name: name(row),
      title: () => title(node, row),
      before: names[before],
      colour: () => ROW_PALETTE[rowColour(node, row)] ?? tint?.(node, row) ?? null,
      onClick: (host, event) => askRowColour(host, row, `${noun} ${row} colour`, event),
    });
    headers.push(widget);
  });
  HEADERS.set(node, headers);
  return groups.map((unused, index) => name(index + 1));
}

/**
 * The row whose header the pointer is over, on the canvas renderer.
 *
 * @param {object} node - The node the menu was opened on.
 * @returns {number|null} Row number, from 1, or null when no header is under the pointer.
 */
export function rowUnderPointer(node) {
  const headers = HEADERS.get(node);
  const mouse = app?.canvas?.graph_mouse;
  // Nodes 2.0 draws widgets in their own elements and never updates `last_y`.
  if (window.LiteGraph?.vueNodesMode) return null;
  const y = Number(mouse?.[1]) - Number(node?.pos?.[1]);
  if (!headers || !Number.isFinite(y)) return null;
  for (let index = 0; index < headers.length; index += 1) {
    const widget = headers[index];
    if (!widget || widget.hidden || !Number.isFinite(widget.last_y)) continue;
    const height = widget.computedHeight ?? widget.computeSize?.()[1] ?? 0;
    if (y >= widget.last_y && y <= widget.last_y + height) return index + 1;
  }
  return null;
}

/**
 * Rows whose headers are drawn.
 *
 * @param {object} node - The node to read.
 * @returns {number[]} Row numbers, from 1.
 */
export function drawnRows(node) {
  const headers = HEADERS.get(node) ?? [];
  const rows = [];
  headers.forEach((widget, index) => {
    if (widget && !widget.hidden) rows.push(index + 1);
  });
  return rows;
}

/**
 * Node menu entries colouring its rows: the row under the pointer directly, any row otherwise.
 *
 * @param {object} node - The node the menu was opened on.
 * @param {string} noun - What a row is called, as `Segment`.
 * @returns {object[]} Entries for `getNodeMenuItems`.
 */
export function rowColourMenu(node, noun) {
  try {
    const row = rowUnderPointer(node);
    if (row !== null) {
      return [
        null,
        {
          content: `🎨 ${noun} ${row} Colour`,
          has_submenu: true,
          submenu: { options: colourEntries(node, row) },
        },
      ];
    }
    const rows = drawnRows(node);
    if (!rows.length) return [];
    return [
      null,
      {
        content: `🎨 ${noun} Colour`,
        has_submenu: true,
        submenu: {
          options: rows.map((each) => {
            const colour = ROW_PALETTE[rowColour(node, each)];
            return {
              content: `${colour ? SWATCH[rowColour(node, each)] + " " : ""}${noun} ${each}`,
              callback: (value, options, event) =>
                askRowColour(node, each, `${noun} ${each} colour`, event),
            };
          }),
        },
      },
    ];
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to build the row colour menu:`, error);
    return [];
  }
}
