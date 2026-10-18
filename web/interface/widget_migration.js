/**
 * Saved widget values, put back where they belong after a node grew a widget.
 *
 * `widgets_values` is a positional array over a node's widgets. This puts the values back by
 * name.
 */

const LOG_NAME = "WASNodeSuite.WidgetMigration";

/**
 * Whether a widget contributes an entry to `widgets_values`.
 *
 * @param {object} widget - The widget to test.
 * @returns {boolean} True when the widget is serialised.
 */
function serialised(widget) {
  return widget?.serialize !== false;
}

/**
 * Expand a v2 widget order with the widgets the frontend attaches to them.
 *
 * @param {object[]} widgets - The node's widgets.
 * @param {string[]} order - The declared v2 widget order.
 * @returns {string[]} The same order with each widget's linked widgets after it.
 */
function withLinked(widgets, order) {
  const byName = new Map(widgets.map((widget) => [widget.name, widget]));
  const expanded = [];
  for (const name of order) {
    expanded.push(name);
    for (const linked of byName.get(name)?.linkedWidgets ?? []) {
      if (serialised(linked)) expanded.push(linked.name);
    }
  }
  return expanded;
}

/**
 * Whether a value is a plain `{name: value}` record.
 *
 * @param {*} value - The value to test.
 * @returns {boolean} True for a non-array object.
 */
function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/**
 * Whether a save's `{name: value}` record holds the same values as its array, in the same order.
 *
 * @param {object} named - The saved `widgets_values_named`.
 * @param {Array} saved - The saved `widgets_values`.
 * @returns {boolean} True when the record names the array's values.
 */
function namesArray(named, saved) {
  const values = Object.values(named);
  return values.length === saved.length
    && values.every((value, index) => JSON.stringify(value) === JSON.stringify(saved[index]));
}

/**
 * The first input spec named `key` anywhere under a part of a node definition.
 *
 * @param {*} scope - Part of the definition's `input`.
 * @param {string} key - The input's name.
 * @returns {Array|null} The `[type, options]` spec, or null.
 */
function findSpec(scope, key) {
  const stack = [scope];
  while (stack.length) {
    const value = stack.shift();
    if (!value || typeof value !== "object") continue;
    if (!Array.isArray(value) && Array.isArray(value[key])) return value[key];
    for (const child of Object.values(value)) {
      if (child && typeof child === "object") stack.push(child);
    }
  }
  return null;
}

/**
 * The default a node's definition declares for one widget, a dynamic group's included.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name, as `affine.affine_keep_colour` inside a group.
 * @returns {*} The default, or undefined where none is declared.
 */
function declaredDefault(node, name) {
  let scope = node?.constructor?.nodeData?.input;
  let spec = null;
  for (const part of String(name).split(".")) {
    spec = findSpec(scope, part);
    if (!spec) return undefined;
    scope = spec;
  }
  const options = spec[1] ?? {};
  if ("default" in options) return options.default;
  if (Array.isArray(spec[0])) return spec[0][0];
  return Array.isArray(options.options) ? options.options[0] : undefined;
}

/**
 * Restore an earlier save's widget values onto a node that has since gained or lost widgets.
 *
 * @param {object} node - The node being created.
 * @param {string[]|string[][]} order - The widgets an earlier save's array holds, in the order
 *   it holds them, or one such order per earlier layout.
 * @returns {void}
 */
export function migrateWidgetValues(node, order) {
  const orders = Array.isArray(order[0]) ? order : [order];
  const defaults = new Map((node.widgets ?? []).map((widget) => [widget.name, widget.value]));

  // The frontend pads `widgets_values` to the current widgets before `onConfigure`, so the array
  // as saved is kept from the moment `configure` is handed it.
  let arrived = null;
  let arrivedNamed = null;
  const originalConfigure = node.configure;
  node.configure = function (info, ...rest) {
    arrived = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : null;
    arrivedNamed = isRecord(info?.widgets_values_named) ? { ...info.widgets_values_named } : null;
    try {
      return originalConfigure.apply(this, [info, ...rest]);
    } finally {
      arrived = null;
      arrivedNamed = null;
    }
  };

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (info, ...rest) {
    try {
      const saved = arrived ?? info?.widgets_values;
      const named = arrivedNamed ?? (isRecord(info?.widgets_values_named) ? info.widgets_values_named : null);
      const current = (this.widgets ?? []).filter(serialised);
      const byNames = Array.isArray(saved) && named && saved.length !== current.length
        && namesArray(named, saved);
      const candidates = orders.flatMap((each) => [withLinked(current, each), each]);
      const matched = Array.isArray(saved) && !byNames
        ? candidates.find((names) => names.length === saved.length && current.length !== names.length)
        : null;
      const target = byNames ? new Map(Object.entries(named))
        : matched ? new Map(matched.map((name, index) => [name, saved[index]])) : null;
      if (target) {
        // Writes only the values that differ, leaving a dynamic group's switch and its widgets as built.
        for (const name of current.map((widget) => widget.name)) {
          const widget = (this.widgets ?? []).find((each) => each.name === name);
          if (!widget) continue;
          const wanted = target.has(name) ? target.get(name)
            : defaults.has(name) ? defaults.get(name) : declaredDefault(this, name);
          if (wanted === undefined || JSON.stringify(widget.value) === JSON.stringify(wanted)) continue;
          widget.value = wanted;
        }
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to migrate ${node?.type}'s saved values:`, error);
    }
    return originalOnConfigure?.apply(this, [info, ...rest]);
  };
}
