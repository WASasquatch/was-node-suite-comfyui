/**
 * Presenting a node's shared widgets as the chosen mode uses them.
 *
 * A mode draws the widgets it lists, under the names and ranges it gives them, and hides the
 * rest.
 */
// A hidden widget is still serialised and still configured.
import { startingInput } from "./node_preview.js";
import { setWidgetHidden } from "./visibility.js";

/**
 * Re-range one widget and hold its value inside the new bounds.
 *
 * @param {object} widget - The widget to present.
 * @param {object} spec - `{label, min, max, start, step, precision, values}`, each optional.
 * @param {boolean} starting - True to take the mode's own start, false to keep a value the
 *     mode accepts. Loading a workflow keeps; changing mode by hand starts.
 * @returns {boolean} True where the drawn value changed.
 */
function present(widget, spec, starting) {
  widget.label = spec.label ?? widget.name;
  const options = widget.options ?? (widget.options = {});
  for (const key of ["min", "max", "step", "precision"]) {
    if (spec[key] !== undefined) options[key] = spec[key];
  }
  if (Array.isArray(spec.values)) {
    options.values = spec.values;
    // A value the narrowed list no longer offers is moved to the first that it does.
    if (!spec.values.includes(widget.value)) {
      widget.value = spec.values[0];
      return true;
    }
    return false;
  }
  // step2 and round are what the number widget actually drags and snaps by.
  if (spec.step !== undefined) options.step2 = spec.step;
  if (spec.step !== undefined) options.round = spec.step;

  if (typeof widget.value !== "number") return false;
  const low = spec.min ?? Number.NEGATIVE_INFINITY;
  const high = spec.max ?? Number.POSITIVE_INFINITY;
  const inside = widget.value >= low && widget.value <= high;
  // A shared widget means something different under each mode, so a number carried across
  // means nothing under the new one however well it fits. The nearest bound is as likely to
  // be the useless end of the range as the useful one, so the mode's own start is taken.
  if (inside && !(starting && spec.start !== undefined)) return false;
  const wanted = spec.start ?? Math.min(high, Math.max(low, widget.value));
  if (wanted === widget.value) return false;
  widget.value = wanted;
  return true;
}

/**
 * Draw one mode's widgets and hide the rest.
 *
 * @param {object} node - The node to lay out.
 * @param {object} control - The widget holding the mode.
 * @param {Record<string, Record<string, object>>} modes - Mode name to widget name to spec.
 * @param {Set<string>} governed - Every widget name any mode lists.
 * @param {boolean} starting - True to take each mode's own start.
 * @returns {void}
 */
function apply(node, control, modes, governed, starting) {
  const specs = modes[control?.value] ?? {};
  let moved = false;
  for (const widget of node.widgets ?? []) {
    if (widget === control) continue;
    if (!governed.has(widget.name)) continue;
    const spec = specs[widget.name];
    const hide = spec === undefined;
    if (setWidgetHidden(widget, hide)) moved = true;
    if (!hide) moved = present(widget, spec, starting) || moved;
  }
  if (!moved) return;
  // The node keeps whatever height it was given, so a mode with fewer widgets would leave
  // dead space and one with more would clip. Both are corrected by re-measuring.
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

/**
 * Every widget name any mode lists.
 *
 * @param {Record<string, Record<string, object>>} modes - The mode table.
 * @returns {Set<string>} The governed names.
 */
function governedBy(modes) {
  const names = new Set();
  for (const specs of Object.values(modes)) for (const name of Object.keys(specs)) names.add(name);
  return names;
}

/**
 * Present the chosen mode's widgets, and keep presenting them as the mode changes.
 *
 * @param {object} node - The node to govern.
 * @param {string} controlName - Name of the widget holding the mode.
 * @param {Record<string, Record<string, object>>} modes - Mode name to widget name to
 *     `{label, min, max, step, precision}`. A widget no mode lists is never touched.
 * @returns {void}
 */
export function watchRelevantWidgets(node, controlName, modes) {
  const control = (node.widgets ?? []).find((widget) => widget.name === controlName);
  if (!control) return;
  const governed = governedBy(modes);

  const previous = control.callback;
  control.callback = function callback(...args) {
    const answer = previous?.apply(this, args);
    apply(node, control, modes, governed, true);
    return answer;
  };

  apply(node, control, modes, governed, true);
}

/**
 * Present the mode's widgets again after a workflow has been loaded into the node.
 *
 * @param {object} node - The node that was configured.
 * @param {string} controlName - Name of the widget holding the mode.
 * @param {Record<string, Record<string, object>>} modes - The mode table.
 * @returns {void}
 */
export function refreshRelevantWidgets(node, controlName, modes) {
  // configure writes every value at once and calls no widget callback.
  const control = (node.widgets ?? []).find((widget) => widget.name === controlName);
  if (!control) return;
  // What a workflow stored is what the user chose, so nothing is moved to a start here.
  apply(node, control, modes, governedBy(modes), false);
}

/**
 * The widgets a newly placed node folds away for the mode it starts on.
 *
 * @param {object} definition - The node definition the frontend files.
 * @param {string} controlName - Name of the widget holding the mode.
 * @param {Record<string, Record<string, object>>} modes - The mode table.
 * @returns {{widgets: string[]}} The names folded away.
 */
export function foldedByRelevance(definition, controlName, modes) {
  const specs = modes[startingInput(definition, controlName)] ?? {};
  const governed = [...governedBy(modes)];
  return { widgets: governed.filter((name) => name !== controlName && specs[name] === undefined) };
}
