/**
 * Resume and Cancel on a node holding a run still.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fetchWithin } from "./interface/request.js";
import { addButton } from "./interface/decoration.js";

const EXT_NAME = "WASNodeSuite.Pause";

// The node this attaches to.
const NODE = "WASPause";

// The route a held run is released through.
const ROUTE = "/was/interface/api/pause";

// Node id -> the message it is waiting with, while it waits.
const held = new Map();

/**
 * Whether a node is holding a run still.
 *
 * @param {object} node - The node to ask about.
 * @returns {boolean} True while it waits.
 */
function waiting(node) {
  return held.has(String(node?.id));
}

/**
 * Publish the held state to both canvas and reactive widget renderers.
 *
 * @param {object} node - The node whose buttons need updating.
 * @returns {void}
 */
function updateButtons(node) {
  const disabled = !waiting(node);
  for (const widget of node.widgets ?? []) {
    if (widget.name === "was_pause_resume" || widget.name === "was_pause_cancel") {
      widget.disabled = disabled;
    }
  }
}

/**
 * Let a held node carry on.
 *
 * @param {object} node - The node to release.
 * @param {string} action - `resume` or `cancel`.
 * @returns {Promise<void>} Settled once the server has answered.
 */
async function release(node, action) {
  try {
    await fetchWithin(ROUTE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: String(node.id), action }),
    });
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to ${action} ${node?.id}:`, error);
  }
}

/**
 * Show a held viewer the content it is offering, which nothing else can deliver.
 *
 * @param {object} node - The held node.
 * @param {string} nodeId - Its id, as the route reports it.
 * @returns {Promise<void>} Settled once the content is on screen.
 */
async function showHeldContent(node, nodeId) {
  if (typeof node?.onExecuted !== "function") return;
  try {
    const answer = await (await fetchWithin(ROUTE)).json();
    const entry = (answer?.waiting ?? []).find((w) => String(w.node_id) === String(nodeId));
    if (!entry?.content) return;
    node.onExecuted({
      text: [entry.content],
      source_content: [entry.content],
      content_hash: [`held_${nodeId}`],
    });
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to show what ${nodeId} is holding:`, error);
  }
}

/**
 * Draw the node again with its waiting state.
 *
 * @param {string} nodeId - The node to redraw.
 * @returns {void}
 */
function redraw(nodeId) {
  const node = app.graph?.getNodeById?.(Number(nodeId))
    ?? app.graph?.getNodeById?.(nodeId);
  if (node) {
    const held = waiting(node);
    updateButtons(node);
    node.color = held ? "#7a5a1e" : undefined;
    node.__was_viewer_held = held;
    node.__was_paint_hold?.();
    node.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: EXT_NAME,

  async setup() {
    api.addEventListener("was-pause", ({ detail }) => {
      held.set(String(detail?.node_id), detail?.message ?? "");
      const node = app.graph?.getNodeById?.(Number(detail?.node_id));
      if (node) node.__was_viewer_kind = detail?.kind ?? "none";
      redraw(detail?.node_id);
      // A held node never reaches its own `ui` payload, so the content it wants edited is
      // fetched and pushed through the same path a finished run would use.
      if (node && detail?.kind && detail.kind !== "none") {
        showHeldContent(node, detail.node_id);
      }
    });
    api.addEventListener("was-pause-done", ({ detail }) => {
      held.delete(String(detail?.node_id));
      redraw(detail?.node_id);
    });
    api.addEventListener("execution_start", () => {
      const ids = [...held.keys()];
      held.clear();
      for (const id of ids) redraw(id);
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const proto = nodeType.prototype;
    if (proto.__was_pause_wrapped) return;
    proto.__was_pause_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        addButton(this, {
          name: "was_pause_resume",
          label: "▶ Resume",
          onClick: (node) => release(node, "resume"),
        });
        addButton(this, {
          name: "was_pause_cancel",
          label: "✕ Cancel run",
          onClick: (node) => release(node, "cancel"),
        });
        updateButtons(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add the resume controls:`, error);
      }
      return result;
    };
  },
});
