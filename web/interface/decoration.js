/**
 * Widgets a node draws but never saves: headers, buttons, anything that holds no value.
 */

const LOG_NAME = "WASNodeSuite.Decoration";

// Margin the frontend insets a widget from each edge of a node.
const HEADER_INSET = 15;

// A header bar stands as tall as a widget row.
const HEADER_HEIGHT = 22;

// Node -> the decorations on it, in no particular order.
const DECORATIONS = new WeakMap();

// Nodes with a repair already queued for this turn of the event loop.
const SCHEDULED = new WeakSet();

/**
 * Arm a node so its decorations are lifted out of the widget list while anything walks it.
 *
 * @param {object} node - The node to arm.
 * @returns {Set<object>} The node's decorations, to add to.
 */
function armed(node) {
  const existing = DECORATIONS.get(node);
  if (existing) return existing;

  const decorations = new Set();
  DECORATIONS.set(node, decorations);

  // Lift the decorations out while anything walks the widget list.
  const without = (run) => {
    const lifted = [];
    for (const widget of decorations) {
      const index = node.widgets?.indexOf(widget) ?? -1;
      if (index >= 0) lifted.push([index, widget]);
    }
    // Back to front.
    lifted.sort((left, right) => right[0] - left[0]);
    for (const [index] of lifted) node.widgets.splice(index, 1);
    try {
      return run();
    } finally {
      // Front to back.
      lifted.sort((left, right) => left[0] - right[0]);
      for (const [index, widget] of lifted) node.widgets.splice(index, 0, widget);
    }
  };

  const originalSerialize = node.serialize?.bind(node);
  if (originalSerialize) {
    node.serialize = (...args) => without(() => originalSerialize(...args));
  }
  const originalConfigure = node.configure?.bind(node);
  if (originalConfigure) {
    node.configure = (info, ...rest) =>
      without(() => originalConfigure(withoutDecorationValues(node, info, decorations), ...rest));
  }
  return decorations;
}

/**
 * A node's saved values with any decoration slots taken back out.
 *
 * @param {object} node - The node being configured.
 * @param {object} info - The saved node, as the workflow holds it.
 * @param {Set<object>} decorations - The decorations on the node.
 * @returns {object} The same info, or a copy whose `widgets_values` holds values only.
 */
function withoutDecorationValues(node, info, decorations) {
  const values = info?.widgets_values;
  const widgets = node?.widgets;
  if (!Array.isArray(values) || !Array.isArray(widgets) || decorations.size === 0) return info;
  const spots = new Set();
  widgets.forEach((widget, index) => {
    if (decorations.has(widget)) spots.add(index);
  });
  // A save that counted the decorations holds one value per widget, decorations included.
  if (spots.size === 0 || values.length !== widgets.length) return info;
  return { ...info, widgets_values: values.filter((unused, index) => !spots.has(index)) };
}

/**
 * Put a workflow's values back on the right widgets where its save counted the decorations.
 *
 * @param {object} node - The decorated node.
 * @param {Set<object>} decorations - The decorations on it.
 * @returns {boolean} Whether anything was moved.
 */
function repairShiftedValues(node, decorations) {
  const raw = node?.widgets_values;
  const widgets = node?.widgets;
  if (!Array.isArray(raw) || !Array.isArray(widgets) || decorations.size === 0) return false;
  if (raw.length !== widgets.length) return false;

  const values = raw.filter((unused, index) => !decorations.has(widgets[index]));
  let at = 0;
  for (const widget of widgets) {
    if (decorations.has(widget)) continue;
    if (at < values.length) widget.value = values[at];
    at += 1;
  }
  node.widgets_values = values;
  node.graph?.setDirtyCanvas(true, true);
  return true;
}

/**
 * Put a widget on a node without it ever being saved.
 *
 * @param {object} node - The node to draw it on.
 * @param {object} widget - The widget.
 * @param {string} [before] - Name of the widget it sits above. Appended when left out.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addDecoration(node, widget, before) {
  const widgets = node?.widgets;
  if (!Array.isArray(widgets) || !widget?.name) return null;
  if (widgets.some((existing) => existing?.name === widget.name)) return null;

  let at = widgets.length;
  if (before) {
    at = widgets.findIndex((existing) => existing?.name === before);
    if (at < 0) return null;
  }
  // Serialisers that walk the widget list themselves read this rather than `node.serialize`.
  widget.serialize = false;
  widgets.splice(at, 0, widget);
  const decorations = armed(node);
  decorations.add(widget);
  // Once every decoration for this node is in place, after the workflow has been applied.
  if (!SCHEDULED.has(node)) {
    SCHEDULED.add(node);
    setTimeout(() => {
      SCHEDULED.delete(node);
      try {
        if (repairShiftedValues(node, decorations)) {
          console.warn(`[${LOG_NAME}] Put ${node?.type}'s saved values back on their widgets.`);
        }
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to repair ${node?.type}'s saved values:`, error);
      }
    }, 0);
  }
  return widget;
}

/**
 * Draw a titled bar immediately above one of a node's widgets.
 *
 * @param {object} node - The node to draw it on.
 * @param {object} header - Settings.
 * @param {string} header.name - Widget name, unique on the node.
 * @param {string} header.title - Text drawn in the bar.
 * @param {string} header.before - Name of the widget the bar sits above.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addSectionHeader(node, { name, title, before }) {
  return addDecoration(node, {
    name,
    type: "custom",
    value: title,
    computeSize(width) {
      return [width ?? 0, HEADER_HEIGHT];
    },
    draw(ctx, host, width, y, height) {
      try {
        const h = height ?? HEADER_HEIGHT;
        // Clamped to the node's own width.
        const room = Math.max(0, Math.min(width ?? 0, host?.size?.[0] ?? width ?? 0));
        const inset = Math.min(HEADER_INSET, room / 2);
        const bar = Math.max(0, room - inset * 2);
        ctx.save();
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "rgba(120, 170, 255, 0.15)";
        ctx.fillRect(inset, y, bar, h);
        ctx.fillStyle = "rgba(210, 230, 255, 0.95)";
        ctx.font = "12px sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(title, inset + 6, y + h / 2);
        ctx.restore();
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw ${name}:`, error);
      }
    },
  }, before);
}

/**
 * Put a button on a node without it ever being saved.
 *
 * @param {object} node - The node to draw it on.
 * @param {object} button - Settings.
 * @param {string} button.name - Widget name, unique on the node.
 * @param {string} button.label - Text drawn on the button.
 * @param {Function} button.onClick - Called with the node when it is pressed.
 * @param {Function} [button.disabled] - Called with the node; true greys the button out.
 * @param {string} [button.before] - Name of the widget it sits above. Appended when left out.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addButton(node, { name, label, onClick, disabled, before }) {
  const widget = {
    name,
    type: "button",
    label,
    value: null,
    callback: () => {
      if (widget.disabled) return;
      try {
        onClick(node);
      } catch (error) {
        console.error(`[${LOG_NAME}] ${name} failed:`, error);
      }
    },
  };
  if (typeof disabled === "function") {
    Object.defineProperty(widget, "disabled", {
      configurable: true,
      enumerable: true,
      // Read on every draw.
      get() {
        try {
          return Boolean(disabled(node));
        } catch (error) {
          console.error(`[${LOG_NAME}] ${name} could not say whether it is disabled:`, error);
          return false;
        }
      },
      set() {},
    });
  }
  return addDecoration(node, widget, before);
}
