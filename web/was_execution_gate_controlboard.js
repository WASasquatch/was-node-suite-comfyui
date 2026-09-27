/**
 * Every execution gate in the graph, listed on one node with a switch each.
 *
 * A switch writes the widget deciding the gate: its own `open`, a subgraph input or a Boolean
 * node. Subgraph gates are listed once per placement.
 */

import { app } from "../../scripts/app.js";
import { captureWheel, wheelPixels } from "./interface/pointer.js";
import { withGraphChange } from "./interface/region.js";
import { createChip, createPip, joinTicking } from "./interface/switch_board.js";
import { themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ExecutionGateControlboardUI";

// The node this file draws on, declared in nodes/logic/execution_gate_controlboard.py.
const NODE_ID = "WASExecutionGateControlboard";
const UI_WIDGET_NAME = "was_gate_board_ui";
const UI_WIDGET_TYPE = "was_gate_board";

// The nodes listed, and the short word each row carries for its kind.
const GATE_KINDS = { WASExecutionGate: "gate", WASAnyGate: "any" };

// The widget and input every listed gate is switched through.
const OPEN_NAME = "open";

// The size a fresh node is placed at, and what the panel inside it asks for.
const NODE_SIZE = [270, 240];
const PANEL_HEIGHT = 176;
const PANEL_MIN_WIDTH = 208;

// The order the rows are drawn in, and the word on the chip that chooses it.
const ORDERS = ["graph", "name", "position"];
const ORDER_LABELS = { graph: "Graph", name: "Name", position: "Position" };

// Where the order is kept. `properties` is serialised with the node and no python reads it.
const ORDER_KEY = "was_gates_order";

// Row geometry, in CSS pixels.
const ROW_HEIGHT = 20;

// What a row spends on everything but its title, in CSS pixels: the kind and id, the gaps, the
// switch, the padding and the border.
const ROW_CHROME = 100;

// Separates a subgraph's title from the gate's own in a row.
const PATH_JOIN = " / ";

/**
 * The order the rows are drawn in.
 *
 * @param {object} node - The node the choice belongs to.
 * @returns {string} One of `ORDERS`, the first when the node holds anything else.
 */
function orderOf(node) {
  const held = node?.properties?.[ORDER_KEY];
  return ORDERS.includes(held) ? held : ORDERS[0];
}

// The id litegraph gives the node a subgraph's own inputs arrive from.
const SUBGRAPH_INPUT_ID = -10;

// How many subgraphs deep the walk goes.
const MAX_DEPTH = 16;

/**
 * One link of a graph by its id.
 *
 * @param {object} graph - The graph holding the link.
 * @param {number} id - The link id.
 * @returns {object|null} The link, or null where the graph holds none by that id.
 */
function linkOf(graph, id) {
  const links = graph?.links;
  if (!links) return null;
  return (typeof links.get === "function" ? links.get(id) : links[id]) || null;
}

/**
 * The widget that decides one input, followed out through subgraph inputs and to a lone
 * Boolean source node.
 *
 * @param {object} node - The node the input belongs to.
 * @param {string} name - The input's name.
 * @param {object[]} hosts - The subgraph nodes enclosing `node`, outermost first.
 * @returns {{widget: object, owner: object, via: string}|null} The widget, the node it sits
 *   on and what it was reached through, or null where a computed value decides the input.
 */
function controlOf(node, name, hosts) {
  const input = (node?.inputs || []).find((slot) => slot.name === name);
  if (!input || input.link == null) {
    const widgetName = input?.widget?.name ?? name;
    const widget = (node?.widgets || []).find((candidate) => candidate.name === widgetName);
    return widget ? { widget, owner: node, via: "" } : null;
  }
  const graph = node.graph;
  const link = linkOf(graph, input.link);
  if (!link) return null;
  if (Number(link.origin_id) === SUBGRAPH_INPUT_ID) {
    const host = hosts[hosts.length - 1];
    const outer = graph?.inputs?.[link.origin_slot]?.name;
    if (!host || !outer) return null;
    const found = controlOf(host, outer, hosts.slice(0, -1));
    return found && { ...found, via: found.via || `the ${host.title ?? "subgraph"} input` };
  }
  const source = graph?.getNodeById?.(link.origin_id);
  if (!source || (source.inputs || []).some((slot) => slot.link != null)) return null;
  const toggles = (source.widgets || []).filter((widget) => typeof widget.value === "boolean");
  if (toggles.length !== 1) return null;
  return { widget: toggles[0], owner: source, via: String(source.title ?? source.type) };
}

/**
 * Read every gate in the workflow, one entry per subgraph placement.
 *
 * @param {object} graph - The graph the panel's node sits in, read when no root is known.
 * @param {string} order - Which of `ORDERS` the rows are drawn in.
 * @returns {Array<{gate: object, kind: string, label: string, depth: number, key: string,
 *   control: object|null, wired: boolean, open: boolean}>} One entry per gate, sorted.
 */
function readGates(graph, order) {
  const root = app?.rootGraph ?? graph;
  const entries = [];

  const walk = (current, path, hosts) => {
    if (!current || hosts.length > MAX_DEPTH) return;
    for (const node of current.nodes ?? []) {
      const kind = GATE_KINDS[node?.type];
      if (kind) {
        const control = controlOf(node, OPEN_NAME, hosts);
        const title = String(node.title ?? node.type);
        entries.push({
          gate: node,
          kind,
          label: [...path, title].join(PATH_JOIN),
          depth: path.length,
          key: [...hosts.map((host) => host.id), node.id].join(":"),
          control,
          wired: !control,
          open: control ? control.widget.value !== false : true,
        });
      }
      if (node?.isSubgraphNode?.() && node.subgraph) {
        walk(node.subgraph, [...path, String(node.title ?? "Subgraph")], [...hosts, node]);
      }
    }
  };
  walk(root, [], []);

  if (order === "name") {
    entries.sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
  } else if (order === "position") {
    entries.sort((a, b) => {
      const first = a.gate.pos ?? [0, 0];
      const second = b.gate.pos ?? [0, 0];
      return a.depth - b.depth || first[1] - second[1] || first[0] - second[0];
    });
  }
  return entries;
}

/**
 * Open or close one gate through whichever widget decides it.
 *
 * @param {object} entry - One entry from `readGates`.
 * @param {boolean} open - The value the deciding widget is set to.
 * @returns {boolean} True when the widget changed.
 */
function setGate(entry, open) {
  const control = entry.control;
  if (!control || control.widget.value === open) return false;
  control.widget.value = open;
  // The owner's callback runs as a click on the widget would, which is where a gate applies
  // bypass_downstream.
  control.widget.callback?.call(control.widget, open, app?.canvas, control.owner);
  control.owner.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * Build the gate list for one node.
 *
 * @param {object} node - The node the list is drawn on.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   refresh: () => void, dispose: () => void}} The panel, the room it asks for, the pass the
 *   page timer calls, and its teardown.
 */
function createGateBoard(node) {
  const root = document.createElement("div");
  root.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    "display:flex",
    "flex-direction:column",
    "gap:5px",
    "overflow:hidden",
    "padding:6px 7px",
    "border-radius:4px",
    "font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace",
    `background:${themeVar("panelBg")}`,
    `color:${themeVar("fg")}`,
    `border:1px solid ${themeVar("border")}`,
  ].join(";");

  const header = document.createElement("div");
  header.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:5px";
  const summary = document.createElement("span");
  summary.style.cssText = "flex:1 1 auto;font-weight:600;overflow:hidden;text-overflow:ellipsis";
  const orderChip = createChip("Row order", () => {
    const next = ORDERS[(ORDERS.indexOf(orderOf(node)) + 1) % ORDERS.length];
    withGraphChange(() => {
      node.properties = node.properties ?? {};
      node.properties[ORDER_KEY] = next;
    });
    refresh();
  });
  header.append(summary, orderChip);

  const list = document.createElement("div");
  list.style.cssText =
    "flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;display:grid;"
    + "grid-template-columns:1fr;align-content:start;gap:2px;scrollbar-width:thin";

  const footer = document.createElement("div");
  footer.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:5px";
  const note = document.createElement("span");
  note.style.cssText =
    `flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;color:${themeVar("fgMuted")}`;
  const allOpen = createChip("Open every gate", () => setEvery(true));
  const allClosed = createChip("Close every gate", () => setEvery(false));
  allOpen.textContent = "All open";
  allClosed.textContent = "All closed";
  footer.append(note, allOpen, allClosed);

  root.append(header, list, footer);

  let entries = [];
  let signature = "";
  let disposed = false;

  /**
   * Set every gate the panel can switch at once.
   *
   * @param {boolean} open - The value every `open` widget is set to.
   * @returns {void}
   */
  function setEvery(open) {
    withGraphChange(() => {
      for (const entry of entries) setGate(entry, open);
    });
    app?.canvas?.setDirty?.(true, true);
    refresh();
  }

  /**
   * Draw one gate as a row that presses.
   *
   * @param {object} entry - One entry from `readGates`.
   * @returns {HTMLButtonElement} The row, for the caller to append.
   */
  function createRow(entry) {
    const { open, wired } = entry;
    const row = document.createElement("button");
    row.type = "button";
    const via = entry.control?.via ? `, switched through ${entry.control.via}` : "";
    row.title = wired
      ? `${entry.label}: open is computed by the graph, so the graph decides this gate`
      : `${entry.label}: ${open ? "open" : "closed"}${via}`;
    row.disabled = wired;
    row.style.cssText = [
      "box-sizing:border-box",
      "min-width:0",
      `height:${ROW_HEIGHT}px`,
      "display:flex",
      "align-items:center",
      "gap:6px",
      "padding:0 5px",
      "border-radius:3px",
      "font:inherit",
      "text-align:left",
      "overflow:hidden",
      `background:${themeVar(open && !wired ? "accentBg" : "bgDark")}`,
      `border:1px solid ${themeVar(open && !wired ? "accent" : "border")}`,
      `color:${themeVar(wired ? "fgDisabled" : "fg")}`,
      `cursor:${wired ? "default" : "pointer"}`,
    ].join(";");

    const title = document.createElement("span");
    title.style.cssText = "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    title.textContent = entry.label;

    const kind = document.createElement("span");
    kind.style.cssText = `flex:0 0 auto;color:${themeVar("fgMuted")}`;
    // The node id tells apart gates that share a title.
    kind.textContent = `${wired ? "wired" : entry.kind} #${entry.gate.id}`;

    // A wired gate's widget is not what decides it, so it carries no switch.
    row.append(title, kind);
    if (!wired) row.append(createPip(open));
    row.addEventListener("click", (event) => {
      event.stopPropagation();
      if (wired) return;
      // The gate can be replaced by a reload between the row being drawn and pressed.
      if (!entry.gate.graph || !entry.control?.owner?.graph) {
        signature = "";
        refresh();
        return;
      }
      withGraphChange(() => setGate(entry, !open));
      app?.canvas?.setDirty?.(true, true);
      refresh();
    });
    return row;
  }

  /**
   * Draw the header, the rows and the footer from what the graph holds now.
   *
   * @returns {void}
   */
  function draw() {
    orderChip.textContent = ORDER_LABELS[orderOf(node)];

    const switchable = entries.filter((entry) => !entry.wired);
    const open = switchable.filter((entry) => entry.open).length;
    summary.textContent = entries.length
      ? `${open} of ${switchable.length} open`
      : "No gates in the graph";
    const wired = entries.length - switchable.length;
    note.textContent = wired ? `${wired} wired` : "";
    allOpen.disabled = switchable.length === 0;
    allClosed.disabled = switchable.length === 0;

    const longest = entries.reduce((wide, entry) => Math.max(wide, entry.label.length), 0);
    const column = `min(100%,calc(${longest + 2}ch + ${ROW_CHROME}px))`;
    list.style.gridTemplateColumns = `repeat(auto-fill,minmax(${column},1fr))`;

    // The pressed row is replaced by the redraw, so the keyboard goes back to its successor.
    const focused = [...list.children].indexOf(document.activeElement);
    list.replaceChildren(...entries.map((entry) => createRow(entry)));
    if (focused >= 0) list.children[focused]?.focus?.({ preventScroll: true });
  }

  /**
   * Look at the graph, and redraw when anything a row shows has moved.
   *
   * @returns {void}
   */
  function refresh() {
    if (disposed) return;
    const order = orderOf(node);
    const next = readGates(node.graph, order);
    const marks = [order];
    for (const entry of next) {
      marks.push(entry.key, entry.label, entry.wired ? 1 : 0, entry.open ? 1 : 0);
    }
    entries = next;
    const drawn = marks.join(" | ");
    if (drawn === signature) return;
    signature = drawn;
    draw();
  }

  root.addEventListener("pointerdown", (event) => {
    // Middle button panning belongs to the canvas underneath.
    if (event.button === 1) app?.canvas?.processMouseDown?.(event);
  });

  root.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });

  const onWheel = (event) => {
    if (list.scrollHeight <= list.clientHeight || !list.contains(event.target)) return false;
    const before = list.scrollTop;
    list.scrollTop += wheelPixels(event, list).y;
    return list.scrollTop !== before;
  };
  let releaseWheel = captureWheel(root, onWheel);

  root.addEventListener("keydown", (event) => {
    // A focused row would otherwise pass these to the frontend, which deletes the selection.
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    event.preventDefault();
    event.stopPropagation();
  });

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const configured = originalOnConfigure?.apply(this, args);
    refresh();
    return configured;
  };

  let leaveTicking = null;
  const originalOnAdded = node.onAdded;
  node.onAdded = function (...args) {
    const added = originalOnAdded?.apply(this, args);
    try {
      // The same node object can join a graph again after its teardown ran.
      disposed = false;
      releaseWheel ??= captureWheel(root, onWheel);
      signature = "";
      leaveTicking?.();
      leaveTicking = joinTicking(node, refresh, EXT_NAME);
      refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to start reading the graph's gates:`, error);
    }
    return added;
  };

  draw();

  return {
    element: root,
    height: PANEL_HEIGHT,
    maxHeight: Number.MAX_SAFE_INTEGER,
    minWidth: PANEL_MIN_WIDTH,
    refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      releaseWheel?.();
      releaseWheel = null;
      leaveTicking?.();
      leaveTicking = null;
      list.replaceChildren();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_gate_board_wrapped) return;
    proto.__was_gate_board_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      this.serialize_widgets = false;
      this.properties = this.properties ?? {};
      this.size = [
        Math.max(this.size?.[0] ?? 0, NODE_SIZE[0]),
        Math.max(this.size?.[1] ?? 0, NODE_SIZE[1]),
      ];

      try {
        const panel = createGateBoard(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the gate list:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the gate list:`, error);
      }
      return result;
    };
  },
});
