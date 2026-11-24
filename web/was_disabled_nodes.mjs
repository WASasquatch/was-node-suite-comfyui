/**
 * The warning shown when a workflow opens with WAS Node Suite nodes the server did not load:
 * a console line, and a dialog once per workflow naming the change that loads each one.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "WASNodeSuite.DisabledNodes";
const SETTING_ID = "WASNodeSuite.DisabledNodes.Warn";
const ROUTE = "/was/interface/api/disabled_nodes";
const DIALOG_KEY = "was-disabled-nodes";
const TITLE = "Disabled WAS Node Suite nodes";

// The name Open Manager finds this pack under, whatever its folder is called.
const OPEN_MANAGER_PACK = "was-node-suite-comfyui";

// Node ids named in one line before the rest are counted.
const SHOWN_IDS = 6;

// LiteGraph node modes left out of the count: muted and bypassed.
const MUTED = 2;
const BYPASSED = 4;

// How long after the graph is configured a frontend with no `afterLoadGraph` hook is given to
// finish the load, in milliseconds.
const FALLBACK_MS = 400;

let table = null;
let pending = [];
let fallback = 0;
const warned = new Set();

/**
 * Whether the dialog is shown. The console line is written either way.
 *
 * @returns {boolean}
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * Open Manager's call that opens a pack's page, where its Optional tab installs requirement
 * groups.
 *
 * @returns {Function|null} Null when Open Manager is not installed.
 */
function openPack() {
  const open = globalThis.openManager?.openPack;
  return typeof open === "function" ? open : null;
}

/**
 * The server's table of held-back node ids, fetched once per page.
 *
 * @returns {Promise<object|null>} The table, or null when the route did not answer.
 */
function heldBack() {
  table ??= api
    .fetchApi(ROUTE, { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : null))
    .catch((error) => {
      console.debug(`[${EXT_NAME}] ${ROUTE} did not answer:`, error);
      return null;
    });
  return table;
}

/**
 * The node types a load reported missing, or found missing on the graph when it reported none.
 *
 * @param {Array<string|object>|undefined} missing - What `afterConfigureGraph` was given.
 * @returns {string[]} Each type once.
 */
function missingTypes(missing) {
  const types = new Set();
  if (Array.isArray(missing)) {
    for (const entry of missing) {
      const type = typeof entry === "string" ? entry : entry?.type;
      if (typeof type === "string" && type) types.add(type);
    }
    return [...types];
  }
  const registered = globalThis.LiteGraph?.registered_node_types ?? {};
  for (const node of app.graph?._nodes ?? []) {
    if (node?.mode === MUTED || node?.mode === BYPASSED) continue;
    if (typeof node?.type === "string" && !(node.type in registered)) types.add(node.type);
  }
  return [...types];
}

/**
 * Node ids joined for one line, the rest counted.
 *
 * @param {string[]} ids
 * @returns {string}
 */
function named(ids) {
  if (ids.length <= SHOWN_IDS) return ids.join(", ");
  return `${ids.slice(0, SHOWN_IDS).join(", ")} and ${ids.length - SHOWN_IDS} more`;
}

/**
 * What the dialog says about the held-back types this workflow uses.
 *
 * @param {object} held - The server's table.
 * @param {string[]} types - Missing node types.
 * @param {boolean} [manager=false] - Whether Open Manager is installed.
 * @returns {{count: number, message: string, items: string[], hint: string,
 *   install: boolean}|null} Null when no missing type is one this pack held back. `install`
 *   is true when a named group has packages to install and Open Manager can install them.
 */
function report(held, types, manager = false) {
  const wanted = new Set(types);
  const pick = (ids) => (Array.isArray(ids) ? ids.filter((id) => wanted.has(id)) : []);
  const counted = new Set();
  const items = [];
  let settings = false;
  let generic = false;
  let packages = false;

  for (const group of held.groups ?? []) {
    const ids = pick(group.nodes);
    if (!ids.length) continue;
    ids.forEach((id) => counted.add(id));
    settings = true;
    let line = `Set ${group.group}: true in the ${group.section} block (${named(ids)}).`;
    if (group.install?.command) {
      packages = true;
      if (group.install.command.startsWith("python ")) generic = true;
      const where = `from the ${group.install.folder} folder`;
      line += ` Install its packages first, ${manager ? `with Open Manager or ${where}` : where}: `
        + group.install.command;
    }
    items.push(line);
  }

  const disabled = pick(held.disabled);
  if (disabled.length) {
    disabled.forEach((id) => counted.add(id));
    settings = true;
    items.push(`Remove ${named(disabled)} from disable in the nodes block.`);
  }

  const failed = pick(held.failed).filter((id) => !counted.has(id));
  if (failed.length) {
    failed.forEach((id) => counted.add(id));
    items.push(
      `${named(failed)} failed to load. The ComfyUI log names the reason.`,
    );
  }

  const count = counted.size;
  if (!count) return null;
  const nodes = count === 1 ? "1 WAS Node Suite node" : `${count} WAS Node Suite nodes`;
  const shows = count === 1 ? "it shows" : "they show";
  const message = settings
    ? `This workflow uses ${nodes} that the server did not load, so ${shows} as missing. `
      + `Make the changes below in ${held.config}, then restart ComfyUI and reload this page.`
    : `This workflow uses ${nodes} that failed to load, so ${shows} as missing.`;
  const hints = [];
  if (settings) {
    hints.push(
      "To load single nodes without their whole group, list their ids under enable in the "
      + "nodes block instead.",
    );
  }
  if (generic) hints.push("Activate the python environment ComfyUI runs in before installing.");
  if (settings) hints.push("docs/CONFIG.md in the WAS Node Suite folder lists what each group needs.");
  const install = packages && manager;
  if (install) {
    hints.push(
      "Confirm opens WAS Node Suite in Open Manager, where the Optional tab installs each "
      + "group's packages.",
    );
  }
  return { count, message, items, hint: hints.join(" "), install };
}

/**
 * The key one warning is remembered under: the open workflow and the types it was for.
 *
 * @param {string[]} types
 * @returns {string}
 */
function warningKey(types) {
  let workflow = "";
  try {
    const active = app?.extensionManager?.workflow?.activeWorkflow;
    workflow = active?.key ?? active?.path ?? "";
  } catch (error) {
    console.debug(`[${EXT_NAME}] The open workflow could not be read:`, error);
  }
  return `${workflow}\n${[...types].sort().join("\n")}`;
}

/**
 * Show the dialog, or a toast without a dialog service. Confirm opens this pack in Open
 * Manager when `found.install` is set.
 *
 * @param {{message: string, items: string[], hint: string, install: boolean}} found
 * @returns {void}
 */
function show(found) {
  const dialog = app?.extensionManager?.dialog;
  if (typeof dialog?.confirm === "function") {
    Promise.resolve(
      dialog.confirm({
        key: DIALOG_KEY,
        title: TITLE,
        type: found.install ? "default" : "info",
        message: found.message,
        itemList: found.items,
        hint: found.hint,
      }),
    )
      .then((confirmed) => {
        if (confirmed === true && found.install) return openPack()?.(OPEN_MANAGER_PACK);
      })
      .catch((error) => console.error(`[${EXT_NAME}] The dialog failed:`, error));
    return;
  }
  app?.extensionManager?.toast?.add?.({
    severity: "warn",
    summary: TITLE,
    detail: [found.message, ...found.items].join("\n"),
    life: 30000,
  });
}

/**
 * Warn about the held-back node types a finished load left missing.
 *
 * @param {string[]} types - Missing node types.
 * @returns {Promise<void>}
 */
async function warn(types) {
  if (!types.length) return;
  const held = await heldBack();
  if (!held) return;
  const found = report(held, types, openPack() !== null);
  if (!found) return;
  const key = warningKey(types);
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`[WAS Node Suite] ${found.message}\n${found.items.map((item) => `  - ${item}`).join("\n")}`);
  if (enabled()) show(found);
}

/**
 * Hand the types the last load left missing to `warn`, once.
 *
 * @returns {void}
 */
function finishLoad() {
  clearTimeout(fallback);
  const types = pending;
  pending = [];
  warn(types).catch((error) => console.error(`[${EXT_NAME}] Failed to check the workflow:`, error));
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Workflows", "Warn about disabled nodes"],
      name: "Explain WAS Node Suite nodes a workflow uses that are turned off",
      tooltip:
        "On: opening a workflow that uses nodes turned off in config.yaml, or nodes that failed "
        + "to load, shows a dialog naming the setting and install command that bring them back, "
        + "once per workflow. Off: the same text goes to the browser console only.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async setup() {
    heldBack();
  },

  async afterConfigureGraph(missing) {
    pending = missingTypes(missing);
    clearTimeout(fallback);
    if (pending.length) fallback = setTimeout(finishLoad, FALLBACK_MS);
  },

  async afterLoadGraph() {
    finishLoad();
  },
});
