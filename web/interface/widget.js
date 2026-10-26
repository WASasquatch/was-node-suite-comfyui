/**
 * How an interface is attached to a node.
 *
 * `appendInterfaceWidget` adds one DOM widget carrying no data, after the last schema widget.
 * The panel handed in is an element and the height the widget is pinned to.
 */

import { app } from "../../../scripts/app.js";
import { dressPanel, unwatchCut, watchCut } from "./card.js";
import { nodeLocator, watchPreviews } from "./preview.js";

// The width a core node with widgets opens at on the canvas. No interface opens a node past it.
const NODE_WIDTH_CAP = 270;

// The height a panel can be dragged down to, in node units.
export const PANEL_FLOOR = 24;

// The node property holding the height each panel was dragged to, by widget name.
const HELD_KEY = "was_panel_heights";

// The id of the node whose corner is being dragged in Nodes 2.0, while it is.
let draggedNodeId = null;

// Whether the page-wide pointer listeners are in place.
let watchingDrags = false;

// Widget -> what its node, the zoom and its host's box were when its panel was last fitted.
const reconciled = new WeakMap();

// Every panel on a node in a graph, as `{node, panel}`, so a finished graph load can ask each
// one to read its node again.
const mounted = new Set();

// True from the moment a graph load starts until it has finished and restored its outputs.
let loading = false;

/**
 * Whether a graph is being loaded, which is every workflow switch, undo, redo and page load.
 *
 * @returns {boolean} True between `beginGraphLoad` and `endGraphLoad`.
 */
export function graphLoading() {
  return loading;
}

/**
 * Mark a graph load as started.
 *
 * @returns {void}
 */
export function beginGraphLoad() {
  loading = true;
}

// How long a value written onto a node waits before the workflow is told it changed, and how
// many times that is tried while a graph load still holds the workflow's state shut.
const KEEP_SETTLE_MS = 500;
const KEEP_ATTEMPTS = 12;

/**
 * Whether the open workflow's saved state carries a value on a node.
 *
 * @param {object} node - The node the value was written on.
 * @param {string} key - The property name.
 * @param {*} value - The value written.
 * @returns {boolean} True when a node of that id in the saved state, at the root or in a
 *   subgraph definition, holds the same value.
 */
function stateCarries(node, key, value) {
  const state = app?.extensionManager?.workflow?.activeWorkflow?.activeState;
  if (!state) return false;
  const wanted = JSON.stringify(value);
  const nodes = [
    ...(state.nodes ?? []),
    ...(state.definitions?.subgraphs ?? []).flatMap((graph) => graph?.nodes ?? []),
  ];
  return nodes.some((saved) => String(saved?.id) === String(node.id)
    && JSON.stringify(saved?.properties?.[key]) === wanted);
}

/**
 * Write a value onto a node's properties and have the open workflow's saved state take it.
 *
 * @param {object} node - The node to write on.
 * @param {string} key - The property name.
 * @param {*} value - A JSON value.
 * @returns {void}
 */
export function keepOnNode(node, key, value) {
  node.properties ??= {};
  node.properties[key] = value;
  // Compared as text: the frontend hands a stored object back wrapped in a proxy.
  const written = JSON.stringify(value);
  let attempts = 0;
  const settle = () => {
    attempts += 1;
    if (!node.graph || JSON.stringify(node.properties?.[key]) !== written) return;
    try {
      app?.canvas?.emitBeforeChange?.();
      app?.canvas?.emitAfterChange?.();
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to mark the workflow changed:", error);
      return;
    }
    if (!stateCarries(node, key, value) && attempts < KEEP_ATTEMPTS) {
      setTimeout(settle, KEEP_SETTLE_MS);
    }
  };
  setTimeout(settle, KEEP_SETTLE_MS);
}

/**
 * Mark a graph load as finished and have every panel on the new graph read its node again.
 *
 * @returns {number} How many panels were asked.
 */
export function endGraphLoad() {
  loading = false;
  holdPanels();
  let asked = 0;
  for (const entry of [...mounted]) {
    if (!entry.node?.graph) continue;
    try {
      entry.panel.refresh?.();
      asked += 1;
    } catch (error) {
      console.error("[WASNodeSuite.Interface] A panel failed to read its node after a load:", error);
    }
  }
  remountStale();
  return asked;
}

// The attribute an interface's element carries, naming its widget.
const INTERFACE_ATTRIBUTE = "data-was-interface";

// Frames a stale panel is looked for after a load, while the node renderer catches up.
const REMOUNT_FRAMES = 30;

/**
 * Put each panel back where the node renderer still shows its copy from before a graph reload.
 *
 * @param {number} [frames] - Frames left to look in.
 * @returns {void}
 */
function remountStale(frames = REMOUNT_FRAMES) {
  let waiting = 0;
  for (const entry of mounted) {
    const element = entry.panel?.element;
    const node = entry.node;
    if (!element || element.isConnected || !node?.graph || !entry.name) continue;
    if (node.graph !== app?.canvas?.graph) continue;
    const shell = document.querySelector(`[data-node-id="${CSS.escape(String(node.id))}"]`);
    const stale = shell?.querySelector?.(`[${INTERFACE_ATTRIBUTE}="${CSS.escape(entry.name)}"]`);
    if (stale && stale !== element) {
      try {
        stale.replaceWith(element);
        entry.panel.refresh?.();
      } catch (error) {
        console.error("[WASNodeSuite.Interface] Failed to put a panel back on its node:", error);
      }
    } else if (shell) {
      waiting += 1;
    }
  }
  if (waiting && frames > 0) requestAnimationFrame(() => remountStale(frames - 1));
}

/**
 * Whether a node's corner is being dragged, in either node renderer.
 *
 * @param {object} node - The node.
 * @returns {boolean} True from the press on its resize handle until the release.
 */
function beingResized(node) {
  if (app?.canvas?.resizing_node === node) return true;
  return draggedNodeId !== null && draggedNodeId === String(node?.id);
}

/**
 * The least height a panel keeps: its floor mid-drag, else its last dragged height, else the
 * height it opens at.
 *
 * @param {object} node - The node the interface belongs to.
 * @param {string} name - The interface widget's name.
 * @param {number} height - The height the panel opens at.
 * @param {number} floor - The panel's floor in node units.
 * @param {number} ceiling - The tallest the panel grows to.
 * @returns {number} The minimum in node units.
 */
function panelMinimum(node, name, height, floor, ceiling) {
  if (beingResized(node)) return floor;
  const held = Number(node?.properties?.[HELD_KEY]?.[name]);
  if (Number.isFinite(held) && held > 0) return Math.min(Math.max(held, floor), ceiling);
  return height;
}

/**
 * Remember the height every panel on a node was dragged to, once the drag is over.
 *
 * @param {object} node - The node that was resized.
 * @returns {void}
 */
function keepDraggedHeights(node) {
  const held = { ...(node?.properties?.[HELD_KEY] ?? {}) };
  let changed = false;
  for (const entry of mounted) {
    if (entry.node !== node || !entry.widget) continue;
    // Nodes 2.0 lays the panel out itself, so its box is read rather than the canvas layout.
    const drawn = entry.panel?.element?.offsetHeight || 0;
    const measured = globalThis.LiteGraph?.vueNodesMode
      ? drawn : Number(entry.widget.computedHeight) || drawn;
    if (!(measured > 0)) continue;
    const kept = Math.round(Math.min(Math.max(measured, entry.floor), entry.ceiling));
    if (held[entry.name] === kept) continue;
    held[entry.name] = kept;
    changed = true;
  }
  if (changed) keepOnNode(node, HELD_KEY, held);
}

/**
 * Size a panel's root for its renderer: under Nodes 2.0 to its dragged height, or its floor
 * mid-drag; on the canvas, to its box.
 *
 * @param {{node: object, name: string, panel: object, floor: number, ceiling: number}} entry -
 *   One mounted panel.
 * @returns {void}
 */
function holdPanel(entry) {
  const element = entry?.panel?.element;
  if (!element?.style || !entry.widget) return;
  let minHeight = "0";
  let contain = "";
  if (globalThis.LiteGraph?.vueNodesMode) {
    // Content-sized until the node is dragged; held to a height once it has been.
    const held = Number(entry.node?.properties?.[HELD_KEY]?.[entry.name]);
    const dragged = beingResized(entry.node);
    if (dragged || (Number.isFinite(held) && held > 0)) {
      const height = dragged ? entry.floor : Math.min(Math.max(held, entry.floor), entry.ceiling);
      minHeight = `${height}px`;
      contain = "size";
    } else {
      minHeight = "";
    }
  }
  if (element.style.minHeight !== minHeight) element.style.minHeight = minHeight;
  if (element.style.contain !== contain) element.style.contain = contain;
}

/**
 * Size the root of every panel on one node, or on every node when none is named.
 *
 * @param {object} [node] - The node.
 * @returns {void}
 */
function holdPanels(node) {
  for (const entry of mounted) {
    if (node && entry.node !== node) continue;
    try {
      holdPanel(entry);
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to size a panel for its renderer:", error);
    }
  }
}

/**
 * Listen, once for the page, for a drag on a node's resize handle, in either node renderer.
 *
 * @returns {void}
 */
function watchDrags() {
  if (watchingDrags || typeof document === "undefined") return;
  watchingDrags = true;
  document.addEventListener("pointerdown", (event) => {
    try {
      const target = event.target instanceof Element ? event.target : null;
      const shell = target?.closest?.("[data-node-id]");
      if (shell && /-resize$/.test(getComputedStyle(target).cursor)) {
        draggedNodeId = String(shell.dataset.nodeId);
        holdPanels(app?.canvas?.graph?.getNodeById?.(draggedNodeId));
      }
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to read a pointer press:", error);
    }
  }, true);
  // Read before the frontend ends the drag, while the node still holds the size it was dragged to.
  const release = () => {
    const dragged = app?.canvas?.resizing_node
      ?? (draggedNodeId !== null ? app?.canvas?.graph?.getNodeById?.(draggedNodeId) : null);
    try {
      if (dragged) keepDraggedHeights(dragged);
    } catch (error) {
      console.error("[WASNodeSuite.Interface] Failed to keep a dragged panel's height:", error);
    } finally {
      draggedNodeId = null;
      if (dragged) holdPanels(dragged);
    }
  };
  document.addEventListener("pointerup", release, true);
  document.addEventListener("pointercancel", release, true);
}

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

  // Measured again only when the node, the zoom or the host's own box changed since the last
  // fit; panning moves none of them.
  if (reconciled.get(widget) === fitState(node, widget, host, canvas)) return;

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
  reconciled.set(widget, fitState(node, widget, host, canvas));
}

/**
 * What a panel's fit depends on, as one string.
 *
 * @param {object} node - The node the interface belongs to.
 * @param {object} widget - The DOM widget the panel was added as.
 * @param {HTMLElement} host - The element the frontend positions for the widget.
 * @param {object} canvas - The graph canvas.
 * @returns {string} The node's size, the widget's top, the zoom and the host's own box.
 */
function fitState(node, widget, host, canvas) {
  return [node.size[0], node.size[1], widget?.y, canvas.ds?.scale, node.flags?.collapsed,
    host.style.width, host.style.height, host.isConnected].join("|");
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
 *   opens at, the tallest it grows to as the node is dragged taller, unbounded when left out, and
 *   optionally the node width it opens at. Dragging the node's corner takes a panel down to
 *   `PANEL_FLOOR` and the node's narrowest width, its content clipped and its cut edge faded;
 *   the height it is left at is kept on the node.
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
  const entry = { node, panel, name: names?.name };
  panel?.element?.setAttribute?.(INTERFACE_ATTRIBUTE, String(names?.name ?? ""));
  dressPanel(panel?.element);
  mounted.add(entry);
  const originalOnAdded = node.onAdded;
  node.onAdded = function (...args) {
    const added = originalOnAdded?.apply(this, args);
    dropForeignOutputs(this);
    mounted.add(entry);
    watchCut(panel?.element);
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
    mounted.delete(entry);
    unwatchCut(panel?.element);
    try {
      // A load clears the graph it replaces; that workflow's runs still publish.
      release?.({ keepRegistration: loading });
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
  // A panel naming no maximum grows with its node; one that must hold its height names it.
  const maxHeight = panel.maxHeight === undefined || panel.maxHeight === null
    ? Number.MAX_SAFE_INTEGER
    : Math.max(height, Number(panel.maxHeight) || 0);
  // The width a panel names is where its node opens, never a floor: `reconcile` holds the panel
  // to whatever width the node is dragged to, and what no longer fits is clipped.
  const openWidth = Math.min(Math.max(0, Number(panel.minWidth) || 0), NODE_WIDTH_CAP);
  const floor = Math.min(height, PANEL_FLOOR);

  // Appended after every schema widget and never inserted: `serialize` writes `widgets_values`
  // by absolute index while `configure` reads it with a compacted counter, so a widget placed
  // before a serialising one loads every later value into the wrong widget.
  const widthBefore = Number(node.size?.[0]) || 0;
  watchDrags();
  const minimum = () => panelMinimum(node, names.name, height, floor, maxHeight);
  const widget = node.addDOMWidget(names.name, names.type, panel.element, {
    hideOnZoom: true,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: minimum,
    getMaxHeight: () => maxHeight,
    getHeight: () => height,
  });

  // The frontend reads the range off `computeLayoutSize` when a widget defines one, and off the
  // getters when it does not, so a panel has to answer the same range both ways or the layout
  // it gets depends on which path the release takes.
  widget.computeLayoutSize = () => ({ minHeight: minimum(), maxHeight, minWidth: 0 });

  // Both flags are set here, since neither implies the other and passing one into
  // `addDOMWidget` sets only `options.serialize`. `widget.serialize` keeps the interface out of
  // the saved workflow and `widget.options.serialize` keeps it out of the API prompt.
  widget.serialize = false;
  widget.options.serialize = false;
  Object.assign(entry, { widget, floor, ceiling: maxHeight });
  holdPanel(entry);

  // The node opens at the width the panel names, since the panel itself asks for none; a
  // workflow being loaded sets its own size after this.
  try {
    const needed = node.computeSize?.();
    if (needed && Number.isFinite(needed[1])) {
      node.setSize([
        Math.max(widthBefore || 0, openWidth, Number(needed[0]) || 0),
        Math.max(Number(node.size?.[1]) || 0, needed[1]),
      ]);
    }
  } catch (error) {
    console.error("[WASNodeSuite.Interface] Failed to open the node at the panel's size:", error);
  }

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

/**
 * Give every multiline box on a node the same ceiling, so an interface beside them can grow.
 *
 * @param {object} node - The node holding the boxes.
 * @param {number} [ceiling] - The most a box grows to, in node units.
 * @returns {number} How many boxes were bounded.
 */
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
