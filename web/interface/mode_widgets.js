/**
 * Widgets drawn only while a mode widget asks for them.
 *
 * Hiding is presentation only: `widgets_values` carries every widget whether it is drawn or not.
 */
import { startingInput } from "./node_preview.js";
import { setWidgetHidden } from "./visibility.js";

const LOG_NAME = "WASNodeSuite.ModeWidgets";

/**
 * Draw the widgets the current mode uses, and hide the others.
 *
 * @param {object} node - The node to fold.
 * @param {object} controller - The widget naming the mode.
 * @param {Object<string, string[]>} byMode - Widget names to draw, keyed on the mode's value.
 * @param {Set<string>} governed - Every name any mode may draw.
 * @returns {void}
 */
function fold(node, controller, byMode, governed) {
  const shown = new Set(byMode[controller?.value] ?? []);
  let changed = false;
  for (const widget of node.widgets ?? []) {
    if (!governed.has(widget.name)) continue;
    if (setWidgetHidden(widget, !shown.has(widget.name))) changed = true;
  }
  if (!changed) return;

  // Only the height is taken from the recomputed size.
  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

/**
 * Fold a node's widgets to follow one of its combo widgets.
 *
 * @param {object} node - The node to fold.
 * @param {string} controllerName - The widget whose value chooses the mode.
 * @param {Object<string, string[]>} byMode - Widget names to draw, keyed on the mode's value. A
 *   name listed under every mode is always drawn; one listed under none never is.
 * @returns {() => void} A function that folds again, for a caller with its own reason to.
 */
export function followMode(node, controllerName, byMode) {
  const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
  const controller = widgets.get(controllerName);
  const governed = new Set(Object.values(byMode).flat());

  const refold = () => {
    try {
      if (controller) fold(node, controller, byMode, governed);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to fold ${node?.type}:`, error);
    }
  };

  if (controller) {
    const original = controller.callback;
    controller.callback = function (...args) {
      const result = original?.apply(this, args);
      refold();
      return result;
    };
  }

  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    refold();
    return result;
  };

  refold();
  return refold;
}

/**
 * The widgets a newly placed node folds away for the mode it starts on.
 *
 * @param {object} definition - The node definition the frontend files.
 * @param {string} controllerName - The widget whose value chooses the mode.
 * @param {Object<string, string[]>} byMode - Widget names to draw, keyed on the mode's value.
 * @returns {{widgets: string[]}} The names folded away.
 */
export function foldedByMode(definition, controllerName, byMode) {
  const shown = new Set(byMode[startingInput(definition, controllerName)] ?? []);
  const governed = new Set(Object.values(byMode).flat());
  return { widgets: [...governed].filter((name) => !shown.has(name)) };
}
