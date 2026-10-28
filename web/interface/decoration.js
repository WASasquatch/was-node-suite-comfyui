/**
 * Widgets a node draws but never saves: headers, buttons, anything that holds no value.
 */
import { app } from "../../../scripts/app.js";

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

  // Lift the decorations out while anything walks the widget list, one splice each way.
  const without = (run) => {
    const widgets = node.widgets;
    if (!Array.isArray(widgets)) return run();
    const lifted = [];
    const kept = [];
    widgets.forEach((widget, index) => {
      if (decorations.has(widget)) lifted.push([index, widget]);
      else kept.push(widget);
    });
    if (lifted.length === 0) return run();
    widgets.splice(0, widgets.length, ...kept);
    try {
      return run();
    } finally {
      // Front to back, each at the place it was lifted from.
      const restored = node.widgets.slice();
      for (const [index, widget] of lifted) restored.splice(Math.min(index, restored.length), 0, widget);
      node.widgets.splice(0, node.widgets.length, ...restored);
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
  // Keeps the decoration out of the API prompt, which reads the option rather than the flag.
  widget.options = { ...(widget.options ?? {}), serialize: false };
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

// Bar and text colours of a header with no colour of its own.
const HEADER_FILL = "rgba(120, 170, 255, 0.15)";
const HEADER_TEXT = "rgba(210, 230, 255, 0.95)";

// Width of the solid edge down the left of a coloured header.
const HEADER_STRIPE = 4;

// Node -> its headers, as header name -> a function answering `{label, tint, key}` now.
const HEADERS = new WeakMap();

// Header widget -> the `key` of what it last painted.
const PAINTED = new WeakMap();

// `<node id>\0<header name>` -> weak references to every header made under that key, newest last.
const TWINS = new Map();

// Headers kept per key.
const MAX_TWINS = 8;

// Renderer copies one header repaints at most, the oldest dropped first.
const MAX_PAINTERS = 8;

// What a renderer copy hands back as `triggerDraw` when it closes.
const CLOSED_PAINTER = /^\(\)\s*=>\s*\{\s*\}$/;

/**
 * Make `triggerDraw` on a drawn widget repaint every renderer copy bound to it, not only the last.
 *
 * @param {object} widget - A widget with its own `draw`.
 * @returns {object} The same widget.
 */
function paintEveryCopy(widget) {
  const painters = [];
  const repaint = () => {
    for (const paint of painters.slice()) {
      try {
        paint();
      } catch {
        // A copy whose canvas has gone throws, and is dropped.
        painters.splice(painters.indexOf(paint), 1);
      }
    }
  };
  Object.defineProperty(widget, "triggerDraw", {
    configurable: true,
    enumerable: false,
    get: () => (painters.length ? repaint : undefined),
    set: (paint) => {
      if (typeof paint !== "function" || paint === repaint || painters.includes(paint)) return;
      if (CLOSED_PAINTER.test(String(paint))) return;
      painters.push(paint);
      if (painters.length > MAX_PAINTERS) painters.shift();
    },
  });
  return widget;
}

/**
 * The node on screen in place of one that is no longer in a graph.
 *
 * @param {object} node - The node a renderer handed over.
 * @returns {object} The node with that id in the graph on screen, or the node itself.
 */
function liveNode(node) {
  if (!node || node.graph) return node;
  try {
    return app?.canvas?.graph?.getNodeById?.(node.id) ?? node;
  } catch {
    return node;
  }
}

/**
 * Note a header under its node id and name, for repainting a copy Nodes 2.0 still shows.
 *
 * @param {object} node - The node it was made for.
 * @param {object} widget - The header widget.
 * @returns {void}
 */
function rememberTwin(node, widget) {
  const key = `${node?.id}\u0000${widget.name}`;
  const kept = (TWINS.get(key) ?? []).filter((ref) => ref.deref());
  kept.push(new WeakRef(widget));
  TWINS.set(key, kept.slice(-MAX_TWINS));
}

/**
 * Repaint every earlier copy of a header that a renderer may still be showing.
 *
 * @param {object} node - The node the header is on now.
 * @param {object} widget - The header on that node.
 * @returns {void}
 */
function repaintTwins(node, widget) {
  for (const ref of TWINS.get(`${node?.id}\u0000${widget.name}`) ?? []) {
    const twin = ref.deref();
    if (!twin || twin === widget) continue;
    PAINTED.delete(twin);
    try {
      twin.triggerDraw?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to repaint ${widget.name}:`, error);
    }
  }
}

/**
 * Paint a node's headers again where their text or colour has moved since they were drawn.
 *
 * @param {object} node - The node whose headers to read again.
 * @returns {void}
 */
export function redrawHeaders(node) {
  const headers = HEADERS.get(node);
  if (!headers) return;
  let changed = false;
  for (const [name, look] of headers) {
    const widget = node.widgets?.find((each) => each?.name === name);
    if (!widget) continue;
    try {
      const now = look(node);
      if (PAINTED.get(widget) === now.key) continue;
      changed = true;
      if (typeof widget.triggerDraw === "function") widget.triggerDraw();
      else repaintTwins(node, widget);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to repaint ${name}:`, error);
    }
  }
  if (changed) node.setDirtyCanvas?.(true, false);
}

/**
 * Text cut short with an ellipsis to fit a width, in the context's current font.
 *
 * @param {CanvasRenderingContext2D} ctx - The context it will be drawn with.
 * @param {string} text - The text.
 * @param {number} room - Width available, in canvas units.
 * @returns {string} The text whole, or as much of it as fits followed by `…`.
 */
function fitted(ctx, text, room) {
  if (ctx.measureText(text).width <= room) return text;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= room) low = mid;
    else high = mid - 1;
  }
  return low > 0 ? `${text.slice(0, low).trimEnd()}…` : "";
}

/**
 * The width a header bar spans: the node's own on the graph canvas, the given width on a
 * widget canvas inside a Nodes 2.0 node.
 *
 * @param {CanvasRenderingContext2D} ctx - The context being drawn with.
 * @param {object} host - The node.
 * @param {number} width - Width the renderer passed to the widget.
 * @returns {number} Width in canvas units.
 */
function barRoom(ctx, host, width) {
  const given = Math.max(0, Number(width) || 0);
  if (ctx?.canvas?.closest?.("[data-node-id]")) return given;
  const node = Number(host?.size?.[0]);
  return Number.isFinite(node) && node > 0 ? Math.min(given, node) : given;
}

/**
 * Draw a titled bar immediately above one of a node's widgets.
 *
 * @param {object} node - The node to draw it on.
 * @param {object} header - Settings.
 * @param {string} header.name - Widget name, unique on the node.
 * @param {string|Function} header.title - Text drawn in the bar, or a function of the node
 *   answering it on every draw.
 * @param {string} header.before - Name of the widget the bar sits above.
 * @param {Function} [header.colour] - Called with the node on every draw; answers
 *   `{fill, stripe, text}` CSS colours, or null for the plain bar.
 * @param {Function} [header.onClick] - Called with the node and the pointer event when the bar
 *   is pressed.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addSectionHeader(node, { name, title, before, colour, onClick }) {
  const text = (host) => {
    const value = typeof title === "function" ? title(host) : title;
    return String(value ?? "");
  };
  const look = (host) => {
    const label = text(host);
    const tint = typeof colour === "function" ? colour(host) : null;
    const key = [label, tint?.fill ?? "", tint?.stripe ?? "", tint?.text ?? ""].join("\u0000");
    return { label, tint, key };
  };
  const widget = {
    name,
    type: "custom",
    // Fixed, so a press never reads as a change of value.
    value: name,
    computeSize(width) {
      return [width ?? 0, HEADER_HEIGHT];
    },
    draw(ctx, host, width, y, height) {
      let saved = false;
      try {
        const h = height ?? HEADER_HEIGHT;
        const shown = liveNode(host ?? node);
        const room = barRoom(ctx, shown, width);
        const inset = Math.min(HEADER_INSET, room / 2);
        const bar = Math.max(0, room - inset * 2);
        const { label, tint, key } = look(shown);
        PAINTED.set(widget, key);
        ctx.save();
        saved = true;
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = tint?.fill ?? HEADER_FILL;
        ctx.fillRect(inset, y, bar, h);
        let at = inset + 6;
        if (tint?.stripe) {
          ctx.fillStyle = tint.stripe;
          ctx.fillRect(inset, y, Math.min(HEADER_STRIPE, bar), h);
          at += HEADER_STRIPE;
        }
        ctx.fillStyle = tint?.text ?? HEADER_TEXT;
        ctx.font = "12px sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(fitted(ctx, label, Math.max(0, bar - (at - inset) - 6)), at, y + h / 2);
      } catch (error) {
        // Once, not per frame.
        if (!widget.drawFailed) {
          widget.drawFailed = true;
          console.error(`[${LOG_NAME}] Failed to draw ${name}:`, error);
        }
      } finally {
        if (saved) ctx.restore();
      }
    },
  };
  if (typeof onClick === "function") {
    widget.mouse = (event, pos, host) => {
      if (event?.type !== "pointerdown" && event?.type !== "mousedown") return false;
      const shown = liveNode(host ?? node);
      try {
        onClick(shown, event);
      } catch (error) {
        console.error(`[${LOG_NAME}] ${name} failed:`, error);
      }
      redrawHeaders(shown);
      // The copy that was pressed, where the renderer still shows an earlier one.
      PAINTED.delete(widget);
      widget.triggerDraw?.();
      return true;
    };
  }
  // On the node and in the properties panel alike.
  paintEveryCopy(widget);
  const added = addDecoration(node, widget, before);
  if (added) {
    const known = HEADERS.get(node) ?? new Map();
    known.set(name, look);
    HEADERS.set(node, known);
    rememberTwin(node, widget);
  }
  return added;
}

// Node -> its buttons that grey out, as `{name, greyed}`.
const GREYING = new WeakMap();

/**
 * Put a button on a node without it ever being saved.
 *
 * @param {object} node - The node to draw it on.
 * @param {object} button - Settings.
 * @param {string} button.name - Widget name, unique on the node.
 * @param {string} button.label - Text drawn on the button.
 * @param {Function} button.onClick - Called with the node when it is pressed.
 * @param {Function} [button.disabled] - Called with the node; true greys the button out.
 *   Read when the button is made, when it is pressed and on `refreshButtons`.
 * @param {string} [button.before] - Name of the widget it sits above. Appended when left out.
 * @returns {object|null} The widget added, or null when there was nowhere to put it.
 */
export function addButton(node, { name, label, onClick, disabled, before }) {
  const greyed = () => {
    if (typeof disabled !== "function") return false;
    try {
      return Boolean(disabled(node));
    } catch (error) {
      console.error(`[${LOG_NAME}] ${name} could not say whether it is disabled:`, error);
      return false;
    }
  };
  const widget = {
    name,
    type: "button",
    label,
    value: null,
    disabled: greyed(),
    callback: () => {
      if (greyed()) return;
      try {
        onClick(node);
      } catch (error) {
        console.error(`[${LOG_NAME}] ${name} failed:`, error);
      }
      refreshButtons(node);
    },
  };
  if (typeof disabled === "function") {
    const known = GREYING.get(node) ?? [];
    known.push({ name, greyed });
    GREYING.set(node, known);
  }
  return addDecoration(node, widget, before);
}

/**
 * Grey out or restore a node's buttons to match what they act on.
 *
 * @param {object} node - The node whose buttons to read again.
 * @returns {void}
 */
export function refreshButtons(node) {
  redrawHeaders(node);
  const known = GREYING.get(node);
  if (!known) return;
  let changed = false;
  for (const { name, greyed } of known) {
    // The frontend draws its own copy of the widget, found by name.
    const widget = node.widgets?.find((each) => each?.name === name);
    if (!widget) continue;
    const value = greyed();
    if (widget.disabled !== value) {
      widget.disabled = value;
      changed = true;
    }
  }
  if (changed) node.setDirtyCanvas?.(true, true);
}
