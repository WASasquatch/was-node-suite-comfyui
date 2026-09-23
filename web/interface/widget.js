/**
 * How an interface is attached to a node.
 *
 * `appendInterfaceWidget` adds one DOM widget carrying no data, after the last schema widget.
 * The panel handed in is an element and the height the widget is pinned to.
 */

import { app } from "../../../scripts/app.js";
import { nodeLocator, watchPreviews } from "./preview.js";

/**
 * The top of the first widget laid out below this one, in node units.
 *
 * @param {object} node - The node the interface belongs to.
 * @param {object} widget - The DOM widget the panel was added as.
 * @returns {number|null} The widget's `y`, or null when nothing is drawn below it.
 */
function nextWidgetTop(node, widget) {
  const widgets = node.widgets ?? [];
  const index = widgets.indexOf(widget);
  if (index < 0 || !(typeof widget.y === "number")) return null;
  // The first visible widget after this one, such as the frontend's picture preview.
  for (const later of widgets.slice(index + 1)) {
    if (later?.hidden || later?.options?.hidden) continue;
    if (typeof later?.y === "number" && later.y > widget.y) return later.y;
  }
  return null;
}

/**
 * Hold the host element's box to its slot in the node, for a node the frontend has stopped
 * laying out.
 *
 * @param {object} node - The node the interface belongs to.
 * @param {object} widget - The DOM widget the panel was added as.
 * @param {HTMLElement} element - The panel the interface put in the widget.
 * @returns {void}
 */
function reconcile(node, widget, element) {
  const host = element?.parentElement;
  const canvas = app?.canvas;
  if (!host || !canvas?.canvas) return;

  // The frontend sizes the host from the widget's own `width`, and on a node it has stopped
  // laying out that number keeps whatever it held when it last did. It is then written back
  // over anything else on every frame, so resizing the node does not recover the panel: the
  // node is the right size and the width it is drawn from is not. Corrected
  // first, so the frontend's own write lands on the right number instead of being fought.
  if (typeof widget?.width === "number" && Math.abs(widget.width - node.size[0]) > 0.5) {
    widget.width = node.size[0];
  }

  const box = host.getBoundingClientRect();
  // Culled, collapsed or hidden: there is no geometry to measure against, and writing one
  // would fight whatever hid it.
  if (!(box.width > 0) || !(box.height > 0) || node.flags?.collapsed) return;

  const scale = canvas.ds?.scale || 1;
  const origin = canvas.ds?.offset ?? [0, 0];
  const surface = canvas.canvas.getBoundingClientRect();
  // The gaps between the node's edges and the panel's, in node units. Measured rather than
  // assumed: they are the frontend's own padding and the height of the sockets above, which
  // differ per node and change with the release.
  const inset = (box.x - (surface.x + (node.pos[0] + origin[0]) * scale)) / scale;
  const above = (box.y - (surface.y + (node.pos[1] + origin[1]) * scale)) / scale;
  if (!(inset >= 0) || !(above >= 0)) return;

  // The panel runs down to the next widget, or to the node's bottom edge when it is the last.
  const floor = Math.min(node.size[1], nextWidgetTop(node, widget) ?? node.size[1]);
  const width = Math.max(0, node.size[0] - inset * 2);
  const height = Math.max(0, floor - above - inset);
  // Only when they actually disagree, so a node the frontend is still laying out is left
  // alone and no write happens on a frame where nothing moved.
  if (Math.abs(box.width / scale - width) > 0.5) host.style.width = `${width}px`;
  if (Math.abs(box.height / scale - height) > 0.5) host.style.height = `${height}px`;
}


/**
 * Forget any run output cached against this node's id.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
export function dropForeignOutputs(node) {
  // The frontend keys `app.nodeOutputs` on the graph id and never clears an entry when the node
  // that made it goes away, while a pasted node is handed a recycled id. A node carrying an
  // interface draws its own report and emits no pictures, so an `images` entry under its id is
  // another node's and would be painted onto this one: copying one shows a stray thumbnail.
  try {
    const app = window.app;
    // The store is keyed on a node locator, not on a bare canvas id: inside a subgraph the id
    // alone names a root-graph node of the same number, so clearing it leaves the stray
    // thumbnail this guard exists to prevent and deletes another node's entry besides.
    const id = nodeLocator(node);
    if (id && app?.nodeOutputs?.[id]) delete app.nodeOutputs[id];
    // Set by the frontend from those outputs, and drawn from then on whatever the store says.
    if (node?.imgs) node.imgs = undefined;
  } catch (error) {
    console.error("[WASNodeSuite.Interface] Failed to clear stale outputs:", error);
  }
}

/**
 * Append an interface to a node as a DOM widget that carries no data.
 *
 * @param {object} node - The node being created.
 * @param {{element: HTMLElement, height: number, maxHeight?: number, minWidth?: number}} panel -
 *   What the interface factory answered: the element to host, the height in node units the widget
 *   is held to, optionally a maximum above it, which lets the node be dragged taller and gives the
 *   panel the extra room, and optionally the narrowest the panel can be drawn in.
 * @param {{name: string, type: string}} names - What to call the widget and its type. Both are
 *   the caller's to choose, since a name is a fact about the node rather than about this module.
 *   The type has to collide with no key of the frontend's widget registry, and several nodes may
 *   share one that does not; a node carrying two interfaces gives each its own name.
 * @returns {object} The widget that was added.
 * @throws {Error} When the panel carries no element or height, or the widget is unnamed. Every
 *   interface wraps its own attachment in a try/catch that logs and leaves the plain widgets, so
 *   a mistake here shows up as a node with no decoration rather than as a widget called
 *   `undefined` in the graph.
 */
export function appendInterfaceWidget(node, panel, names) {
  // Hooked on `onAdded` rather than called here: a node is built before it joins a graph, so at
  // this point its id is still a placeholder, the entry to drop cannot be found, and the id the
  // channel is asked to publish under is not the one this node will run as.
  let release = null;
  const originalOnAdded = node.onAdded;
  node.onAdded = function (...args) {
    const added = originalOnAdded?.apply(this, args);
    dropForeignOutputs(this);
    try {
      release?.();
      release = watchPreviews(this);
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to register for this node's pictures:", error);
    }
    return added;
  };
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    try {
      release?.();
      release = null;
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to release this node's pictures:", error);
    }
    return originalOnRemoved?.apply(this, args);
  };
  // A node that is already in a graph will never fire `onAdded` again, which is the case for an
  // interface attached from anything later than node creation.
  if (node.graph && String(node.id ?? "") && String(node.id) !== "-1") {
    try {
      release = watchPreviews(node);
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to register for this node's pictures:", error);
    }
  }
  const height = panel?.height;
  if (!panel?.element || !(height > 0) || !names?.name || !names?.type) {
    throw new Error(
      "An interface widget takes an element and a height from the interface factory, and a name"
        + " and a type from the caller: appendInterfaceWidget(node, panel, { name, type }).",
    );
  }
  // A maximum below the height would be a panel that cannot be drawn at the size it asked for,
  // so the height is the floor of the range whatever the panel said. A maximum above it is only
  // half the story: the frontend hands the slack to whichever widgets will take it, so a growing
  // panel next to an uncapped multiline box splits the room with it rather than taking it.
  const maxHeight = Math.max(height, Number(panel.maxHeight) || 0);
  // A panel that names no width is one the node may be collapsed under. The frontend sizes a node
  // to its content whenever it refits one, and a widget answering no width at all is not counted,
  // so the node shrinks to its sockets while the panel keeps the geometry it already had and is
  // left standing outside the node, over whatever is behind it.
  const minWidth = Math.max(0, Number(panel.minWidth) || 0);

  // Appended after every schema widget and never inserted: `serialize` writes `widgets_values`
  // by absolute index while `configure` reads it with a compacted counter, so a widget placed
  // before a serialising one loads every later value into the wrong widget.
  const widget = node.addDOMWidget(names.name, names.type, panel.element, {
    hideOnZoom: true,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: () => height,
    getMaxHeight: () => maxHeight,
    getHeight: () => height,
  });

  // The frontend reads the range off `computeLayoutSize` when a widget defines one, and off the
  // getters when it does not, so a panel that grows has to answer the same range both ways or
  // the layout it gets depends on which path the release takes.
  widget.computeLayoutSize = () => ({ minHeight: height, maxHeight, minWidth });

  // Both flags are set here, since neither implies the other and passing one into
  // `addDOMWidget` sets only `options.serialize`. `widget.serialize` keeps the interface out of
  // the saved workflow and `widget.options.serialize` keeps it out of the API prompt.
  widget.serialize = false;
  widget.options.serialize = false;

  // Installed once the widget exists, since the box is derived from it. `onDrawForeground` is
  // the one callback that still runs for every node the canvas draws, so it is where a panel
  // that has drifted out of its node is put back.
  const originalOnDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function (...args) {
    const drawn = originalOnDrawForeground?.apply(this, args);
    try {
      reconcile(this, widget, panel.element);
    } catch (error) {
      // Once, not per frame: this runs on every draw and a broken measurement would otherwise
      // fill the console faster than it could be read.
      if (!this.__was_reconcile_failed) {
        this.__was_reconcile_failed = true;
        console.error("[WASNodeSuite.Interface] Failed to fit the panel to its node:", error);
      }
    }
    return drawn;
  };
  return widget;
}

// The widget type ComfyUI gives a multiline string input. Boxes are found by type rather than by
// name, so a schema that adds or renames one cannot leave a box behaving unlike its neighbours.
const TEXT_WIDGET_TYPE = "customtext";

// What a text box grows to before it stops taking room, in node units. Roughly five lines: past
// any pattern or path, and enough of a body to work in.
export const TEXT_BOX_CEILING = 160;

/**
 * Give every multiline box on a node the same ceiling, so an interface beside them can grow.
 *
 * @param {object} node - The node holding the boxes.
 * @param {number} [ceiling] - The most a box grows to, in node units.
 * @returns {number} How many boxes were bounded.
 */
/**
 * Run something after a widget's own callback, whenever that widget changes.
 *
 * @param {object} node - The node holding the widget.
 * @param {string} name - The widget's name. A name the node does not carry is ignored.
 * @param {() => void} onChange - Called after the widget's own callback has run.
 * @param {string} [logName] - What a failure inside `onChange` is logged under.
 * @returns {void}
 */
export function chainWidgetCallback(node, name, onChange, logName = "WASNodeSuite.Widget") {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  const widget = widgets.find((candidate) => candidate?.name === name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${logName}] Failed to repaint after a widget change:`, error);
    }
    return result;
  };
}

export function boundTextBoxes(node, ceiling = TEXT_BOX_CEILING) {
  // The frontend divides a node's spare room between every widget whose maximum is above its
  // minimum, and a multiline string widget declares no maximum at all. Left alone the boxes take
  // all of it and the interface beside them never grows, however open its own maximum is.
  //
  // Bounded rather than pinned, and every box alike: each grows with the node until it reaches the
  // ceiling and then stops competing, so a small drag feeds the boxes, a larger one feeds the
  // interface, and no box behaves unlike the box above it.
  const boxes = (node?.widgets ?? []).filter((widget) => widget.type === TEXT_WIDGET_TYPE);
  for (const widget of boxes) {
    // Its own minimum, read before the override replaces the callback that answers it, so a box
    // keeps the height it asks for and only its maximum is decided here.
    const natural = widget.computeLayoutSize?.bind(widget);
    const floor = Number(natural?.()?.minHeight) || Number(widget.options?.getMinHeight?.()) || 0;
    const top = Math.max(floor, ceiling);
    widget.computeLayoutSize = () => ({ minHeight: floor, maxHeight: top, minWidth: 0 });
    widget.options = widget.options ?? {};
    widget.options.getMinHeight = () => floor;
    widget.options.getMaxHeight = () => top;
  }
  return boxes.length;
}
