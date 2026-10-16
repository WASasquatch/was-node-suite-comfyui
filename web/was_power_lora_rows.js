/**
 * Adding, removing and clearing the LoRA rows on the two Power LoRA nodes.
 */

import { app } from "../../scripts/app.js";
import { addButton, addSectionHeader, refreshButtons } from "./interface/decoration.js";
import { menuText, rowColour, rowName, setRowColours, setRowNames } from "./interface/row_headers.js";

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

// The Power LoRA Merger input whose LoRAs replace the rows while it is wired.
const STACK_INPUT = "stack_data";

// The bar above the rows, and what it reads with stack_data wired and without.
const ROWS_HEADER = "was_rows_header";
const ROWS_TITLE = "LoRAs";
const STACK_TITLE = "Using stack_data (rows ignored)";

/**
 * Whether the node's stack_data input is wired, so its rows take no part.
 *
 * @param {object} node - The node to read.
 * @returns {boolean} True while a link reaches stack_data.
 */
function stackLinked(node) {
  return (node?.inputs ?? []).some((input) => input?.name === STACK_INPUT && input.link != null);
}

/**
 * Grey the row widgets out while stack_data is wired, and back in once it is not.
 *
 * @param {object} node - The node whose rows to set.
 * @returns {void}
 */
function applyStackState(node) {
  const linked = stackLinked(node);
  let changed = false;
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    for (const row of Object.values(rowWidgets(node, index))) {
      if (!row || Boolean(row.disabled) === linked) continue;
      row.disabled = linked;
      changed = true;
    }
  }
  if (changed) node.setDirtyCanvas?.(true, true);
}

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
 * @returns {Array<{on: boolean, name: string, weight: number, colour: string|null,
 *   title: string|null}>} The rows in use, each with its header colour and name.
 */
function readRows(node) {
  const rows = [];
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const row = rowWidgets(node, index);
    const name = row.name?.value;
    if (!name || name === EMPTY) continue;
    rows.push({
      on: row.on?.value ?? true,
      name,
      weight: row.weight?.value ?? 1.0,
      colour: rowColour(node, index),
      title: rowName(node, index),
    });
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
 * How many rows the node's contents call for: every row up to the last LoRA, and one empty.
 *
 * @param {object} node - The node to read.
 * @returns {number} Rows drawn with none held open.
 */
function naturalRows(node) {
  let last = 0;
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const name = rowWidgets(node, index).name?.value;
    if (name && name !== EMPTY) last = index;
  }
  return Math.min(MAX_ROWS, last + 1);
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
 * Lay a list of LoRAs back over the rows, packed to the top, each header's colour and name with
 * its LoRA.
 *
 * @param {object} node - The node to write to.
 * @param {Array<{on: boolean, name: string, weight: number, colour: string|null,
 *   title: string|null}>} rows - The LoRAs to keep.
 * @returns {void}
 */
function writeRows(node, rows) {
  const colours = {};
  const titles = {};
  for (let index = 1; index <= MAX_ROWS; index += 1) {
    const row = rowWidgets(node, index);
    const wanted = rows[index - 1] ?? DEFAULT_ROW;
    if (row.on) row.on.value = wanted.on;
    if (row.name) row.name.value = wanted.name;
    if (row.weight) row.weight.value = wanted.weight;
    colours[index] = wanted.colour ?? null;
    titles[index] = wanted.title ?? null;
  }
  setRowColours(node, colours);
  setRowNames(node, titles);
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
  refreshButtons(node);
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
 * Take away the bottom row: an empty one held open, or else the last LoRA.
 *
 * @param {object} node - The node to remove from.
 * @returns {void}
 */
function removeLast(node) {
  const shown = visibleRows(node);
  if (shown > naturalRows(node)) {
    refold(node, shown - 1);
    return;
  }
  const rows = readRows(node);
  if (rows.length) removeRow(node, rows.length - 1);
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
    content: menuText(`${at + 1}. ${row.title ?? row.name}`),
    callback: () => removeRow(node, at),
  }));
}

/**
 * Grey the row buttons out or back in once the change in hand has been drawn.
 *
 * @param {object} node - The node whose buttons to read again.
 * @returns {void}
 */
function queueButtonRefresh(node) {
  if (node.__was_buttons_queued) return;
  node.__was_buttons_queued = true;
  setTimeout(() => {
    node.__was_buttons_queued = false;
    applyStackState(node);
    refreshButtons(node);
  }, 0);
}

app.registerExtension({
  name: EXT_NAME,

  getNodeMenuItems(node) {
    if (!NODES.includes(node?.comfyClass)) return [];
    const rows = readRows(node);
    const linked = stackLinked(node);
    return [
      null,
      { content: "➕ Add LoRA", disabled: linked || atCeiling(node), callback: () => addRow(node) },
      {
        content: "➖ Remove LoRA",
        disabled: linked || rows.length === 0,
        has_submenu: !linked && rows.length > 0,
        submenu: { options: removeEntries(node) },
      },
      {
        content: "❌ Clear LoRAs",
        disabled: linked || nothingToClear(node),
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
        if (this.inputs?.some((input) => input?.name === STACK_INPUT)) {
          addSectionHeader(this, {
            name: ROWS_HEADER,
            title: (node) => (stackLinked(node) ? STACK_TITLE : ROWS_TITLE),
            before: "lora_1_enabled",
          });
        }
        addButton(this, {
          name: "was_row_add",
          label: "➕ Add LoRA",
          onClick: addRow,
          disabled: (node) => stackLinked(node) || atCeiling(node),
        });
        addButton(this, {
          name: "was_row_remove",
          label: "➖ Remove LoRA",
          onClick: removeLast,
          disabled: (node) => stackLinked(node) || nothingToClear(node),
        });
        addButton(this, {
          name: "was_row_clear",
          label: "❌ Clear LoRAs",
          onClick: clearRows,
          disabled: (node) => stackLinked(node) || nothingToClear(node),
        });
        addButton(this, {
          name: "was_row_refresh",
          label: "♻️ Refresh LoRA List",
          onClick: refreshCatalog,
        });
        // Picking, switching or wiring a row changes what the buttons can do.
        for (let index = 1; index <= MAX_ROWS; index += 1) {
          for (const row of Object.values(rowWidgets(this, index))) {
            if (!row) continue;
            const original = row.callback;
            const node = this;
            row.callback = function (...args) {
              const answer = original?.apply(this, args);
              queueButtonRefresh(node);
              return answer;
            };
          }
        }
        const originalConnections = this.onConnectionsChange;
        this.onConnectionsChange = function (...args) {
          const answer = originalConnections?.apply(this, args);
          queueButtonRefresh(this);
          return answer;
        };
        queueButtonRefresh(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add ${nodeData.name}'s row buttons:`, error);
      }
      return result;
    };

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      queueButtonRefresh(this);
      return result;
    };
  },
});
