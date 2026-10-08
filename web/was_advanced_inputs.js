/**
 * The advanced inputs of the pack's nodes, folded under a bar that opens them.
 *
 * The bar's state is kept in the node's properties. `widgets_values` carries every widget
 * whether it is drawn or not.
 */
import { app } from "../../scripts/app.js";
import { addSectionHeader } from "./interface/decoration.js";
import { setWidgetHidden } from "./interface/visibility.js";

const EXT_NAME = "WASNodeSuite.AdvancedInputs";

// Only this pack's own nodes.
const OWNED = "WAS Suite";

const HEADER_NAME = "was_advanced_inputs";
const OPEN_PROPERTY = "was_show_advanced";

/**
 * The names of a definition's inputs flagged advanced, in declaration order.
 *
 * @param {object} nodeData - The node definition the frontend registered.
 * @returns {string[]} Input names.
 */
function advancedNames(nodeData) {
  const names = [];
  for (const group of ["required", "optional"]) {
    for (const [name, spec] of Object.entries(nodeData?.input?.[group] ?? {})) {
      if (Array.isArray(spec) && spec[1]?.advanced === true) names.push(name);
    }
  }
  return names;
}

/**
 * The widgets drawn for advanced inputs, with the controls that ride on them.
 *
 * @param {object} node - The node.
 * @param {Set<string>} names - Advanced input names.
 * @returns {object[]} Widgets in node order.
 */
function advancedWidgets(node, names) {
  const found = [];
  for (const widget of node.widgets ?? []) {
    if (!names.has(widget.name)) continue;
    found.push(widget);
    for (const linked of widget.linkedWidgets ?? []) found.push(linked);
  }
  return found;
}

/**
 * Draw or fold a node's advanced widgets to match its bar.
 *
 * @param {object} node - The node.
 * @param {Set<string>} names - Advanced input names.
 * @returns {void}
 */
function fold(node, names) {
  const open = Boolean(node.properties?.[OPEN_PROPERTY]);
  let changed = false;
  for (const widget of advancedWidgets(node, names)) {
    // The bar decides, on either renderer.
    widget.advanced = false;
    if (widget.options) widget.options.advanced = false;
    if (setWidgetHidden(widget, !open)) changed = true;
  }
  if (!changed) return;
  // Only the height is taken from the recomputed size.
  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData?.category ?? "").startsWith(OWNED)) return;
    const declared = advancedNames(nodeData);
    if (!declared.length) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap twice.
    if (proto.__was_advanced_inputs_wrapped) return;
    proto.__was_advanced_inputs_wrapped = true;
    const names = new Set(declared);

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        this.properties = this.properties ?? {};
        if (!(OPEN_PROPERTY in this.properties)) this.properties[OPEN_PROPERTY] = false;
        const first = (this.widgets ?? []).find((widget) => names.has(widget.name));
        if (first) {
          addSectionHeader(this, {
            name: HEADER_NAME,
            before: first.name,
            title: (node) => {
              const open = Boolean(node.properties?.[OPEN_PROPERTY]);
              return `${open ? "▾" : "▸"} advanced inputs (${names.size})`;
            },
            onClick: (node) => {
              node.properties[OPEN_PROPERTY] = !node.properties[OPEN_PROPERTY];
              fold(node, names);
            },
          });
        }
        fold(this, names);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to fold ${this.type}:`, error);
      }
      return result;
    };

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      try {
        fold(this, names);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to fold ${this.type}:`, error);
      }
      return result;
    };
  },
});
