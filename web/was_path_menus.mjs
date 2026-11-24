/**
 * File and folder menus that also take a wired path, refilled when node definitions are
 * refreshed.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.PathMenus";

// How many subgraphs deep the walk goes.
const MAX_DEPTH = 16;

/**
 * The entries of one input that is a menu or a wired value, as its definition lists them.
 *
 * @param {object} definition - The node's definition, as `/object_info` answers it.
 * @param {string} name - The input's name.
 * @returns {Array|null} The entries, or null where the input is not such a menu.
 */
function menuEntries(definition, name) {
  const inputs = definition?.input;
  const spec = inputs?.required?.[name] ?? inputs?.optional?.[name];
  if (!Array.isArray(spec) || typeof spec[0] !== "string") return null;
  const types = spec[0].split(",").map((type) => type.trim());
  if (types.length < 2 || !types.includes("COMBO")) return null;
  const entries = spec[1]?.options;
  return Array.isArray(entries) ? entries : null;
}

/**
 * The workflow's graphs: the root and every subgraph inside it.
 *
 * @param {object} root - The root graph.
 * @returns {object[]} Each graph once.
 */
function allGraphs(root) {
  const found = new Set();
  const walk = (graph, depth) => {
    if (!graph || found.has(graph) || depth > MAX_DEPTH) return;
    found.add(graph);
    for (const node of graph.nodes ?? []) {
      if (node?.isSubgraphNode?.() && node.subgraph) walk(node.subgraph, depth + 1);
    }
  };
  walk(root, 0);
  for (const graph of root?.subgraphs?.values?.() ?? []) walk(graph, 1);
  return [...found];
}

/**
 * Give every such menu in the workflow the entries its definition now lists.
 *
 * @param {Object<string, object>} definitions - Node definitions by node id.
 * @returns {number} How many menus were refilled.
 */
function refillMenus(definitions) {
  let refilled = 0;
  for (const graph of allGraphs(app.rootGraph ?? app.graph)) {
    for (const node of graph.nodes ?? []) {
      const definition = definitions?.[node?.type];
      if (!definition) continue;
      for (const widget of node.widgets ?? []) {
        if (widget?.type !== "combo") continue;
        const entries = menuEntries(definition, widget.name);
        if (!entries) continue;
        widget.options ??= {};
        widget.options.values = entries;
        refilled += 1;
      }
    }
  }
  return refilled;
}

app.registerExtension({
  name: EXT_NAME,

  async refreshComboInNodes(definitions) {
    try {
      refillMenus(definitions);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to refill the file menus:`, error);
    }
  },
});
