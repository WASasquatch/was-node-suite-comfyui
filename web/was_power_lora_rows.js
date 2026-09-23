/**
 * Adding, removing and clearing the LoRA rows on the two Power LoRA nodes.
 */

import { app } from "../../scripts/app.js";
import { addButton } from "./interface/decoration.js";

const EXT_NAME = "WASNodeSuite.PowerLoraRows";

// The nodes this attaches to.
const NODES = ["WASPowerLoraLoader", "WASNSPowerLoraMerger"];

// Rows either node declares, matching MAX_ROWS in nodes/extras/lora/power_lora_loader.py.
const MAX_ROWS = 26;

// ComfyUI's own "Refresh Node Definitions".
const REFRESH_COMMAND = "Comfy.RefreshNodeDefinitions";

// What a row's file widget holds when the row carries no LoRA.
const EMPTY = "None";

const DEFAULT_ROW = { on: true, name: EMPTY, weight: 1.0 };

/**
 * One of a node's widgets by name.
 *
 * @param {object} node - The node to look on.
 * @param {string} name - The widget's name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function widget(node, name) {
  return (node?.widgets ?? []).find((found) => found?.name === name) ?? null;
}

/**
 * The three widgets making up one row.
 *
 * @param {object} node - The node the row is on.
 * @param {number} index - The row's number, from one.
 * @returns {{on: object, name: object, weight: object}} The row's widgets.
 */
function rowWidgets(node, index) {
  return {
    on: widget(node, `lora_${index}_enabled`),
    name: widget(node, `lora_${index}`),
    weight: widget(node, `lora_${index}_weight`),
  };
}

/**
 * Every row carrying a LoRA, in the order they are drawn.
 *
 * @param {object} node - The node to read.
 * @returns {Array<{on: boolean, name: string, weight: number}>} The rows in use.
 */
function readRows(node) {
  const rows = [];
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const row = rowWidgets(node, index);
    const name = row.name?.value;
    if (!name || name === EMPTY) continue;
    rows.push({ on: row.on?.value ?? true, name, weight: row.weight?.value ?? 1.0 });
  }
  return rows;
}

/**
 * How many rows the node is drawing, filled or not.
 *
 * @param {object} node - The node to read.
 * @returns {number} Rows on screen, up to every row the node declares.
 */
function visibleRows(node) {
  let count = 0;
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const row = rowWidgets(node, index).name;
    if (row && !row.hidden) count += 1;
  }
  return count;
}

/**
 * Whether the node is already drawing every row it has.
 *
 * @param {object} node - The node to read.
 * @returns {boolean} True when there is no row left to open.
 */
function atCeiling(node) {
  return visibleRows(node) >= MAX_ROWS;
}

/**
 * Whether the node carries no LoRA at all.
 *
 * @param {object} node - The node to read.
 * @returns {boolean} True when there is nothing to remove or clear.
 */
function isEmpty(node) {
  return readRows(node).length === 0;
}

/**
 * Whether the node is already showing as little as it can.
 *
 * @param {object} node - The node to read.
 * @returns {boolean} True when there is neither a LoRA to drop nor an opened row to close.
 */
function nothingToClear(node) {
  return isEmpty(node) && visibleRows(node) <= 1;
}

/**
 * Lay a list of LoRAs back over the rows, packed to the top.
 *
 * @param {object} node - The node to write to.
 * @param {Array<{on: boolean, name: string, weight: number}>} rows - The LoRAs to keep.
 * @returns {void}
 */
function writeRows(node, rows) {
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const row = rowWidgets(node, index);
    const wanted = rows[index - 1] ?? DEFAULT_ROW;
    if (row.on) row.on.value = wanted.on;
    if (row.name) row.name.value = wanted.name;
    if (row.weight) row.weight.value = wanted.weight;
  }
}

/**
 * Draw the node again with however many rows its contents now call for.
 *
 * @param {object} node - The node to redraw.
 * @param {number} [forced] - Rows to hold open regardless of what they hold.
 * @returns {void}
 */
function refold(node, forced = 0) {
  node.__was_forced_groups = forced;
  node.__was_refold?.();
  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.setDirtyCanvas(true, true);
}

/**
 * Hold one more empty row open, so there is somewhere to put another LoRA.
 *
 * @param {object} node - The node to add a row to.
 * @returns {void}
 */
function addRow(node) {
  if (atCeiling(node)) return;
  refold(node, Math.min(MAX_ROWS, visibleRows(node) + 1));
}

/**
 * Take one LoRA out of the list and close the gap behind it.
 *
 * @param {object} node - The node to remove from.
 * @param {number} at - Position in the list of LoRAs in use, from zero.
 * @returns {void}
 */
function removeRow(node, at) {
  const rows = readRows(node);
  if (at < 0 || at >= rows.length) return;
  rows.splice(at, 1);
  writeRows(node, rows);
  refold(node);
}

/**
 * Take every LoRA off the node.
 *
 * @param {object} node - The node to clear.
 * @returns {void}
 */
function clearRows(node) {
  writeRows(node, []);
  refold(node);
}

/**
 * Read the LoRA files again, so one added since the page loaded is offered in every row.
 *
 * @param {object} node - The node to redraw once the lists are new.
 * @returns {Promise<void>} Settled once the rows carry the new list.
 */
async function refreshCatalog(node) {
  const command = app?.extensionManager?.command;
  try {
    if (typeof command?.execute === "function") {
      await command.execute(REFRESH_COMMAND);
    } else if (typeof app?.refreshComboInNodes === "function") {
      await app.refreshComboInNodes();
    }
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to refresh the LoRA list:`, error);
  }
  refold(node);
}

/**
 * The LoRAs currently on the node, each entry removing the one it names.
 *
 * @param {object} node - The node the menu belongs to.
 * @returns {object[]} One entry per LoRA in use, or a note when there are none.
 */
function removeEntries(node) {
  const rows = readRows(node);
  if (!rows.length) return [{ content: "No LoRAs to remove", disabled: true }];
  return rows.map((row, at) => ({
    content: `${at + 1}. ${row.name}`,
    callback: () => removeRow(node, at),
  }));
}

/**
 * Offer the list of LoRAs to take out, wherever the pointer is.
 *
 * @param {object} node - The node to remove from.
 * @param {Event} [event] - What opened the menu, used to place it.
 * @returns {void}
 */
function askWhichToRemove(node, event) {
  const ContextMenu = window.LiteGraph?.ContextMenu;
  if (!ContextMenu) return;
  new ContextMenu(removeEntries(node), { title: "Remove LoRA", event });
}

app.registerExtension({
  name: EXT_NAME,

  getNodeMenuItems(node) {
    if (!NODES.includes(node?.comfyClass)) return [];
    const rows = readRows(node);
    return [
      null,
      { content: "➕ Add LoRA", disabled: atCeiling(node), callback: () => addRow(node) },
      {
        content: "➖ Remove LoRA",
        disabled: rows.length === 0,
        has_submenu: rows.length > 0,
        submenu: { options: removeEntries(node) },
      },
      {
        content: "❌ Clear LoRAs",
        disabled: nothingToClear(node),
        callback: () => clearRows(node),
      },
      { content: "♻️ Refresh LoRA List", callback: () => refreshCatalog(node) },
    ];
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    if (proto.__was_power_lora_rows_wrapped) return;
    proto.__was_power_lora_rows_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        addButton(this, {
          name: "was_row_add",
          label: "➕ Add LoRA",
          onClick: addRow,
          disabled: atCeiling,
        });
        addButton(this, {
          name: "was_row_remove",
          label: "➖ Remove LoRA",
          onClick: (node) => askWhichToRemove(node, window.event),
          disabled: isEmpty,
        });
        addButton(this, {
          name: "was_row_clear",
          label: "❌ Clear LoRAs",
          onClick: clearRows,
          disabled: nothingToClear,
        });
        addButton(this, {
          name: "was_row_refresh",
          label: "♻️ Refresh LoRA List",
          onClick: refreshCatalog,
        });
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add ${nodeData.name}'s row buttons:`, error);
      }
      return result;
    };
  },
});
