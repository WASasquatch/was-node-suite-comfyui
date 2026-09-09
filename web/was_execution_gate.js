/**
 * Setting the nodes after a closed Execution Gate to bypass, on the canvas.
 *
 * Everything downstream is switched between bypass and always as the gate's own `open`
 * widget turns. A gate whose `open` is wired is left alone.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.ExecutionGate";
const NODE_ID = "WASExecutionGate";

// litegraph's mode numbers. 4 is what the canvas draws as a bypassed node.
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

// Node ids this extension put into bypass, which are the only ones it takes back out.
const ours = new Map();

/**
 * Every node reachable by following links out of one node.
 *
 * @param {object} node - Where the walk starts.
 * @returns {object[]} The nodes downstream of it, the starting node excluded.
 */
function downstreamOf(node) {
  const graph = node?.graph;
  if (!graph) return [];
  const found = new Map();
  const queue = [node];
  while (queue.length) {
    const current = queue.shift();
    for (const output of current.outputs || []) {
      for (const linkId of output.links || []) {
        const link = graph.links?.[linkId];
        if (!link) continue;
        const target = graph.getNodeById?.(link.target_id);
        if (!target || found.has(target.id) || target === node) continue;
        found.set(target.id, target);
        queue.push(target);
      }
    }
  }
  return [...found.values()];
}

/**
 * Whether the gate's own widget decides it, rather than something wired in.
 *
 * @param {object} node - The gate.
 * @returns {object|null} The widget where the gate decides for itself, otherwise null.
 */
function ownSwitch(node) {
  const input = (node.inputs || []).find((slot) => slot.name === "open");
  if (input && input.link != null) return null;
  return (node.widgets || []).find((w) => w.name === "open") || null;
}

/**
 * Match the nodes after one gate to whether it is open.
 *
 * @param {object} gate - The gate to read.
 * @returns {void}
 */
function apply(gate) {
  const wantBypass = (node) => {
    const marker = (gate.widgets || []).find((w) => w.name === "bypass_downstream");
    if (!marker?.value) return false;
    const openWidget = ownSwitch(gate);
    if (!openWidget) return false;
    return openWidget.value === false;
  };

  const targets = downstreamOf(gate);
  const held = ours.get(gate.id) || new Set();

  if (wantBypass(gate)) {
    for (const node of targets) {
      if (node.mode === MODE_BYPASS) continue;
      node.mode = MODE_BYPASS;
      held.add(node.id);
    }
    ours.set(gate.id, held);
  } else {
    for (const id of held) {
      const node = gate.graph?.getNodeById?.(id);
      if (node && node.mode === MODE_BYPASS) node.mode = MODE_ALWAYS;
    }
    ours.set(gate.id, new Set());
  }
  app?.canvas?.setDirty?.(true, true);
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_execution_gate_wrapped) return;
    proto.__was_execution_gate_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      for (const name of ["open", "bypass_downstream"]) {
        const widget = (this.widgets || []).find((w) => w.name === name);
        if (!widget) continue;
        const originalCallback = widget.callback;
        widget.callback = (...args) => {
          const out = originalCallback?.apply(widget, args);
          try {
            apply(this);
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to set the nodes after the gate:`, error);
          }
          return out;
        };
      }
      return result;
    };

    // A wire added or dropped after the gate changes what is downstream of it.
    const originalOnConnectionsChange = proto.onConnectionsChange;
    proto.onConnectionsChange = function (...args) {
      const out = originalOnConnectionsChange?.apply(this, args);
      try {
        apply(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to set the nodes after the gate:`, error);
      }
      return out;
    };
  },
});
