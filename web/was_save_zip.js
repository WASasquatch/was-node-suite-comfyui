/**
 * What Save ZIP wrote, drawn on the node.
 *
 * Lists the file the node wrote, the folder it landed in, and every entry the archive holds.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createListing } from "./interface/listing.js";
import { fetchRunResult, fetchRunResultPage, whenReadable } from "./interface/run_result.js";
import { PREVIEW_STATE } from "./interface/preview.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.SaveZip";
const NODE = "WASSaveZip";

const UI_WIDGET_NAME = "was_save_zip_ui";
const UI_WIDGET_TYPE = "was_save_zip";

const PANEL_HEIGHT = 210;
const PANEL_MAX_HEIGHT = 900;
const PANEL_MIN_WIDTH = 300;

const IDLE_LABEL = "Run the node to see what it wrote.";
const NOTHING_LABEL = "This node has written nothing this session.";

/**
 * Read a range of lines from one body of what a node published.
 *
 * @param {object} node - The node the panel is drawn on.
 * @param {number} index - Which body, counting from zero in the order the report carries them.
 * @param {number} start - The first line wanted, counting from zero.
 * @param {number} wanted - How many lines to ask for.
 * @returns {Promise<object|null>} The page, or null where it could not be read.
 */
async function page(node, index, start, wanted) {
  const answer = await fetchRunResultPage(node, index, start, wanted);
  return answer?.page ?? null;
}

/**
 * Build the contents panel.
 *
 * @param {object} node - The node the panel is drawn on.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   show: (result: object) => void, idle: (label?: string) => void}} The panel.
 */
function createContents(node) {
  const element = document.createElement("div");
  Object.assign(element.style, {
    display: "flex", flexDirection: "column", gap: "6px",
    width: "100%", height: "100%", boxSizing: "border-box", padding: "6px",
    font: "11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    color: "var(--fg-color, #ddd)", background: "var(--comfy-menu-bg, #202020)",
    border: "1px solid var(--border-color, #444)", borderRadius: "4px",
    overflow: "hidden",
  });

  const summary = document.createElement("div");
  summary.textContent = IDLE_LABEL;
  Object.assign(summary.style, { fontWeight: "600", flex: "0 0 auto" });

  const where = document.createElement("div");
  Object.assign(where.style, {
    opacity: "0.75", flex: "0 0 auto",
    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
  });

  const list = document.createElement("div");
  Object.assign(list.style, {
    flex: "1 1 auto", overflowY: "auto", minHeight: "0",
    border: "1px solid var(--border-color, #444)", borderRadius: "3px",
    padding: "4px", background: "var(--comfy-input-bg, #181818)",
  });

  element.append(summary, where, list);

  let listing = null;

  const idle = (label) => {
    summary.textContent = label ?? IDLE_LABEL;
    where.textContent = "";
    listing?.dispose?.();
    listing = null;
    list.textContent = "";
    node.setDirtyCanvas(true, true);
  };

  const show = (result) => {
    summary.textContent = result?.summary || "wrote an archive";
    const folder = (result?.facts ?? []).find((fact) => fact?.name === "folder")?.value ?? "";
    where.textContent = folder;
    where.title = folder;

    const carried = result?.bodies ?? [];
    const index = carried.findIndex((one) => one?.name === "contents");
    const body = index >= 0 ? carried[index] : null;
    const text = String(body?.text ?? "");
    listing?.dispose?.();
    // The entries run across the panel in as many columns as its width holds while they are
    // short names, and down it as written where one of them is a long path. An archive holding
    // more entries than the report carries reads the rest as the panel is scrolled.
    listing = createListing(text || "(the archive holds nothing)", {
      lines: body?.lines,
      whole: body?.whole,
      offset: body?.offset,
      name: body?.name,
      run: result?.run,
      page: (start, wanted) => page(node, index, start, wanted),
    });
    list.replaceChildren(listing);
    if (body?.whole === false && listing.dataset.wasPaged !== "1") {
      const short = document.createElement("div");
      short.style.cssText = "opacity:0.7;padding-top:2px";
      short.textContent = "listing shortened";
      list.appendChild(short);
    }
    node.setDirtyCanvas(true, true);
  };

  return {
    element,
    height: PANEL_HEIGHT,
    maxHeight: PANEL_MAX_HEIGHT,
    minWidth: PANEL_MIN_WIDTH,
    show,
    idle,
  };
}

// Node id to the node and the panel drawn on it.
const panels = new Map();

/**
 * Ask the server what one node wrote and draw it.
 *
 * @param {string} nodeId - The node to refresh.
 * @returns {Promise<void>}
 */
async function refresh(nodeId) {
  const held = panels.get(String(nodeId));
  if (!held) return;
  const answer = await fetchRunResult(held.node);
  // A closed socket keeps whatever the panel already shows.
  if (answer.state === PREVIEW_STATE.CONNECTING) return;
  if (answer.state === PREVIEW_STATE.READY && answer.result) held.panel.show(answer.result);
  else if (answer.state === PREVIEW_STATE.WAITING) held.panel.idle(NOTHING_LABEL);
  else held.panel.idle(answer.label || NOTHING_LABEL);
}

app.registerExtension({
  name: EXT_NAME,

  async setup() {
    api.addEventListener("execution_start", () => {
      for (const { panel } of panels.values()) panel.idle("Running…");
    });
    api.addEventListener("executed", ({ detail }) => {
      if (panels.has(String(detail?.node))) refresh(detail.node);
    });
    api.addEventListener("execution_success", () => {
      for (const id of panels.keys()) refresh(id);
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const proto = nodeType.prototype;
    if (proto.__was_save_zip_wrapped) return;
    proto.__was_save_zip_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createContents(this);
        appendInterfaceWidget(this, panel, {
          name: UI_WIDGET_NAME,
          type: UI_WIDGET_TYPE,
        });
        const register = () => panels.set(String(this.id), { node: this, panel });
        register();
        let cancelRead = () => {};
        const originalOnAdded = this.onAdded;
        this.onAdded = function (...args) {
          const added = originalOnAdded?.apply(this, args);
          register();
          cancelRead();
          cancelRead = whenReadable(this, () => refresh(this.id));
          return added;
        };
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          cancelRead();
          panels.delete(String(this.id));
          return originalOnRemoved?.apply(this, args);
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add the contents panel:`, error);
      }
      return result;
    };
  },
});
