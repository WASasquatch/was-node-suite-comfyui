/**
 * Combo Primitive's dropdown, listing the options of the dropdown its output feeds.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.ComboPrimitive";
const NODE = "WASComboPrimitive";
const WIDGET = "value";

// What the dropdown holds before its output reaches another dropdown. Matches the schema.
const UNCONNECTED = "connect combo to a dropdown";

// The node property the last copied options are kept under, so a saved workflow opens on them.
const KEPT = "was_combo_options";

/**
 * One link of a graph by id.
 *
 * @param {object} graph - The graph.
 * @param {number} id - The link id.
 * @returns {object|null} The link.
 */
function linkOf(graph, id) {
  return graph?.links?.get?.(id) ?? graph?.links?.[id] ?? null;
}

/**
 * A dropdown's options as a list.
 *
 * @param {object} widget - A widget.
 * @returns {string[]|null} Its options, or null for a widget that is not a dropdown.
 */
function optionsOf(widget) {
  let values = widget?.options?.values;
  if (typeof values === "function") values = values(widget);
  return Array.isArray(values) && values.length ? values.slice() : null;
}

/**
 * The first dropdown this node's combo output feeds.
 *
 * @param {object} node - The Combo Primitive.
 * @returns {{node: object, widget: object, options: string[]}|null} The target, or null.
 */
function targetOf(node) {
  const graph = node.graph ?? app.graph;
  for (const id of node.outputs?.[0]?.links ?? []) {
    const link = linkOf(graph, id);
    const target = link ? graph.getNodeById?.(link.target_id) : null;
    const input = target?.inputs?.[link.target_slot];
    const name = input?.widget?.name ?? input?.name;
    const widget = target?.widgets?.find((candidate) => candidate.name === name);
    if (!widget || widget.name === WIDGET && target.type === NODE) continue;
    const options = optionsOf(widget);
    if (options) return { node: target, widget, options };
  }
  return null;
}

/**
 * Copy the target dropdown's options, name and, where the current choice is not among them, its
 * choice.
 *
 * @param {object} node - The Combo Primitive.
 * @returns {boolean} True when a target was found.
 */
function copyTarget(node) {
  const widget = node.widgets?.find((candidate) => candidate.name === WIDGET);
  const target = targetOf(node);
  if (!widget || !target) return false;
  const label = target.widget.label || target.widget.name;
  widget.label = label;
  if (!target.options.includes(widget.value)) {
    widget.value = target.options.includes(target.widget.value) ? target.widget.value : target.options[0];
    widget.callback?.(widget.value);
  }
  node.properties ??= {};
  const kept = node.properties[KEPT];
  if (JSON.stringify(kept?.options) !== JSON.stringify(target.options) || kept?.label !== label) {
    node.properties[KEPT] = { options: target.options, label };
  }
  node.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * Make the dropdown list what the target offers now, or what was last copied.
 *
 * @param {object} node - The Combo Primitive.
 * @returns {void}
 */
function followTarget(node) {
  const widget = node.widgets?.find((candidate) => candidate.name === WIDGET);
  if (!widget?.options) return;
  Object.defineProperty(widget.options, "values", {
    configurable: true,
    enumerable: true,
    get() {
      const live = targetOf(node)?.options;
      if (live) return live;
      const kept = node.properties?.[KEPT]?.options;
      return Array.isArray(kept) && kept.length ? kept : [UNCONNECTED];
    },
    set() {},
  });
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const proto = nodeType.prototype;
    if (proto.__was_combo_primitive_wrapped) return;
    proto.__was_combo_primitive_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      try {
        followTarget(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to set up the dropdown:`, error);
      }
      return result;
    };

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      const kept = this.properties?.[KEPT];
      const widget = this.widgets?.find((candidate) => candidate.name === WIDGET);
      if (widget && kept?.label) widget.label = kept.label;
      // Runs once the load has wired every node, so the target's dropdown is there to read.
      queueMicrotask(() => {
        try {
          copyTarget(this);
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to read the connected dropdown:`, error);
        }
      });
      return result;
    };

    const originalOnConnectionsChange = proto.onConnectionsChange;
    proto.onConnectionsChange = function (...args) {
      const result = originalOnConnectionsChange?.apply(this, args);
      try {
        copyTarget(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read the connected dropdown:`, error);
      }
      return result;
    };
  },
});
