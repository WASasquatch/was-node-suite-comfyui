/**
 * The entry picker on ZIP Manage.
 *
 * Lists the entries the held node published, with a tick against each. Resuming sends the
 * ticked ones back as the entries to keep.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fetchWithin } from "./interface/request.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ZipManage";
const NODE = "WASZipManage";
const ROUTE = "/was/interface/api/pause";

/** What the node says it is holding for, so another node's hold is left alone. */
const HOLD_KIND = "zip_entries";

const UI_WIDGET_NAME = "was_zip_manage_ui";
const UI_WIDGET_TYPE = "was_zip_manage";

const PANEL_HEIGHT = 240;
const PANEL_MAX_HEIGHT = 900;
const PANEL_MIN_WIDTH = 320;

// What a row spends on everything but the name, in CSS pixels: the tick box, the gap beside it
// and the row's own padding.
const TICK_WIDTH = 30;

const IDLE_LABEL = "Run the node to list what the archive holds.";

/**
 * Send one resume or cancel for a held node.
 *
 * @param {string} nodeId - The node holding the run.
 * @param {string} action - `resume` or `cancel`.
 * @param {string} value - The entries to keep, one per line. Ignored for a cancel.
 * @returns {Promise<void>}
 */
async function release(nodeId, action, value) {
  try {
    const answer = await fetchWithin(ROUTE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: String(nodeId),
        action,
        value: value ?? "",
        client_id: String(api?.clientId || api?.initialClientId || ""),
      }),
    });
    if (!answer.ok) {
      const reply = await answer.json().catch(() => ({}));
      console.error(`[${EXT_NAME}] ${nodeId} was not released:`, reply?.error ?? answer.status);
    }
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to ${action} the held run:`, error);
  }
}

/**
 * Draw whatever a node is already waiting for.
 *
 * @param {object} node - The node the picker belongs to.
 * @param {object} picker - The picker to draw into.
 * @returns {Promise<void>}
 */
async function adoptExistingHold(node, picker) {
  // `was-pause` is announced once, when the hold begins. A node added after that, as a page
  // reload or opening the workflow adds it, would otherwise draw its placeholder while the
  // run stays held with nothing on the page to resume it from.
  const hold = await heldFor(node.id);
  if (!hold || hold.kind !== HOLD_KIND) return;
  const names = String(hold.content ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (names.length) picker.show(names);
  else picker.idle("The archive holds nothing to choose from.");
}

/**
 * What a node is currently waiting for, or null when it is not waiting.
 *
 * @param {string} nodeId - The node to ask about.
 * @returns {Promise<object|null>} The hold record, or null.
 */
async function heldFor(nodeId) {
  try {
    const response = await fetchWithin(ROUTE);
    const body = await response.json();
    const wanted = String(nodeId);
    return (body?.waiting ?? []).find((hold) => String(hold.node_id) === wanted) ?? null;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read what the node is waiting for:`, error);
    return null;
  }
}

/**
 * Build the picker.
 *
 * @param {object} node - The node the picker is drawn on.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   show: (names: string[]) => void, idle: (label?: string) => void}} The panel.
 */
function createPicker(node) {
  const element = document.createElement("div");
  element.className = "was-zip-manage";
  Object.assign(element.style, {
    display: "flex", flexDirection: "column", gap: "6px",
    width: "100%", height: "100%", boxSizing: "border-box",
    padding: "6px", font: "12px var(--comfy-font, sans-serif)",
    color: "var(--fg-color, #ddd)", background: "var(--comfy-menu-bg, #202020)",
    border: "1px solid var(--border-color, #444)", borderRadius: "4px",
    overflow: "hidden",
  });

  const status = document.createElement("div");
  status.textContent = IDLE_LABEL;
  Object.assign(status.style, { opacity: "0.75", flex: "0 0 auto" });

  const list = document.createElement("div");
  // The rows run in as many columns as the panel's width holds. `show` sets the column width
  // from the longest name, so an archive of long paths keeps one column.
  Object.assign(list.style, {
    flex: "1 1 auto", overflowY: "auto", overflowX: "hidden", minHeight: "0",
    display: "grid", gridTemplateColumns: "1fr", alignContent: "start", columnGap: "10px",
    border: "1px solid var(--border-color, #444)", borderRadius: "3px",
    padding: "4px", background: "var(--comfy-input-bg, #181818)",
  });

  const bar = document.createElement("div");
  Object.assign(bar.style, { display: "flex", gap: "6px", flex: "0 0 auto" });

  const button = (label, grow) => {
    const element = document.createElement("button");
    element.textContent = label;
    Object.assign(element.style, {
      flex: grow ? "1 1 auto" : "0 0 auto", padding: "4px 8px", cursor: "pointer",
      font: "inherit", color: "var(--fg-color, #ddd)",
      background: "var(--comfy-input-bg, #181818)",
      border: "1px solid var(--border-color, #444)", borderRadius: "3px",
    });
    return element;
  };

  const all = button("All");
  const none = button("None");
  const keep = button("Keep ticked", true);
  const cancel = button("Cancel run");
  bar.append(all, none, keep, cancel);
  element.append(status, list, bar);

  /** @returns {HTMLInputElement[]} Every tick box currently listed. */
  const boxes = () => [...list.querySelectorAll("input[type=checkbox]")];

  const setEnabled = (on) => {
    for (const control of [all, none, keep, cancel]) {
      control.disabled = !on;
      control.style.opacity = on ? "1" : "0.5";
      control.style.cursor = on ? "pointer" : "default";
    }
  };

  const count = () => {
    const ticked = boxes().filter((box) => box.checked).length;
    status.textContent = `${ticked} of ${boxes().length} kept, resume when ready`;
  };

  const idle = (label) => {
    list.replaceChildren();
    list.style.gridTemplateColumns = "1fr";
    status.textContent = label ?? IDLE_LABEL;
    setEnabled(false);
    node.setDirtyCanvas(true, true);
  };

  const show = (names) => {
    list.replaceChildren();
    const longest = names.reduce((wide, name) => Math.max(wide, name.length), 0);
    const column = `min(100%,calc(${longest + 2}ch + ${TICK_WIDTH}px))`;
    list.style.gridTemplateColumns = `repeat(auto-fill,minmax(${column},1fr))`;
    for (const name of names) {
      const row = document.createElement("label");
      Object.assign(row.style, {
        display: "flex", alignItems: "center", gap: "6px", minWidth: "0",
        padding: "2px 4px", cursor: "pointer", borderRadius: "2px",
      });
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.dataset.name = name;
      box.addEventListener("change", count);
      const text = document.createElement("span");
      text.textContent = name;
      Object.assign(text.style, {
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
      });
      row.append(box, text);
      list.append(row);
    }
    setEnabled(true);
    count();
    node.setDirtyCanvas(true, true);
  };

  all.addEventListener("click", () => { boxes().forEach((b) => { b.checked = true; }); count(); });
  none.addEventListener("click", () => { boxes().forEach((b) => { b.checked = false; }); count(); });
  keep.addEventListener("click", () => {
    const chosen = boxes().filter((b) => b.checked).map((b) => b.dataset.name);
    release(node.id, "resume", chosen.length ? chosen.join("\n") : "#");
    idle("Resumed.");
  });
  cancel.addEventListener("click", () => {
    release(node.id, "cancel", "");
    idle("Run cancelled.");
  });

  setEnabled(false);
  return {
    element,
    height: PANEL_HEIGHT,
    maxHeight: PANEL_MAX_HEIGHT,
    minWidth: PANEL_MIN_WIDTH,
    show,
    idle,
  };
}

const pickers = new Map();

app.registerExtension({
  name: EXT_NAME,

  async setup() {
    api.addEventListener("was-pause", async ({ detail }) => {
      if (detail?.kind !== HOLD_KIND) return;
      const picker = pickers.get(String(detail.node_id));
      if (!picker) return;
      const hold = await heldFor(detail.node_id);
      const names = String(hold?.content ?? "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      if (names.length) picker.show(names);
      else picker.idle("The archive holds nothing to choose from.");
    });

    api.addEventListener("was-pause-done", ({ detail }) => {
      pickers.get(String(detail?.node_id))?.idle("Nothing is waiting.");
    });

    api.addEventListener("execution_start", () => {
      for (const picker of pickers.values()) picker.idle("Running…");
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const proto = nodeType.prototype;
    if (proto.__was_zip_manage_wrapped) return;
    proto.__was_zip_manage_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const picker = createPicker(this);
        appendInterfaceWidget(this, picker, {
          name: UI_WIDGET_NAME,
          type: UI_WIDGET_TYPE,
        });
        // A node has no graph id until it is added, so both the registration and the
        // lookup of an existing hold are repeated once it does.
        const register = () => {
          pickers.set(String(this.id), picker);
          adoptExistingHold(this, picker).catch((error) => {
            console.error(`[${EXT_NAME}] Failed to adopt a hold already in progress:`, error);
          });
        };
        register();
        const originalOnAdded = this.onAdded;
        this.onAdded = function (...args) {
          const added = originalOnAdded?.apply(this, args);
          register();
          return added;
        };
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          pickers.delete(String(this.id));
          return originalOnRemoved?.apply(this, args);
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add the entry picker:`, error);
      }
      return result;
    };
  },
});
