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
 * Restore a v2 workflow's widget values onto a node that has since gained widgets.
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
      const byName = new Map(current.map((widget) => [widget.name, widget]));
      const restoreDefaults = () => {
        for (const widget of current) {
          if (defaults.has(widget.name)) widget.value = defaults.get(widget.name);
        }
      };
      const byNames = Array.isArray(saved) && named && saved.length !== current.length
        && namesArray(named, saved);
      if (byNames) {
        restoreDefaults();
        for (const [name, value] of Object.entries(named)) {
          const widget = byName.get(name);
          if (widget) widget.value = value;
        }
      }
      const candidates = orders.flatMap((each) => [withLinked(current, each), each]);
      const matched = Array.isArray(saved) && !byNames
        ? candidates.find((names) => names.length === saved.length && current.length > names.length)
        : null;
      if (matched) {
        restoreDefaults();
        matched.forEach((name, index) => {
          const widget = byName.get(name);
          if (widget) widget.value = saved[index];
        });
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to migrate ${node?.type}'s saved values:`, error);
    }
    return originalOnConfigure?.apply(this, [info, ...rest]);
  };
}
