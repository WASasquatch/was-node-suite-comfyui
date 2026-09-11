/**
 * Every group in the graph, listed on one node with a switch each.
 *
 * The node is registered in the browser alone. A switch reads and writes `mode` on the graph's
 * own nodes and holds nothing itself.
 */

import { app } from "../../scripts/app.js";
import { captureWheel, wheelPixels } from "./interface/pointer.js";
import { withGraphChange } from "./interface/region.js";
import { themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.FastGroupsUI";
const SETTING_ID = "WAS.FastGroups.ShowGroups";

// The node this file draws on, declared in nodes/utility/fast_groups.py. A saved workflow
// holding it loads with no interface present.
const NODE_ID = "WASFastGroups";
const UI_WIDGET_NAME = "was_fast_groups_ui";
const UI_WIDGET_TYPE = "was_fast_groups";

// The size a fresh node is placed at, and what the panel inside it asks for. The panel grows
// with the node, so a graph with many groups is read by dragging the node taller.
const NODE_SIZE = [284, 240];
const PANEL_HEIGHT = 176;
const PANEL_MIN_WIDTH = 208;

// How often the panel looks at the graph, in milliseconds. Groups are renamed, moved, added and
// muted from the canvas, and none of those raise an event a panel can listen for.
const REFRESH_MS = 250;

// LiteGraph's execution modes. `MODE_MUTE` is `LiteGraph.NEVER`; the global carries no name for
// bypass, which the frontend's own enum numbers 4.
const MODE_ALWAYS = 0;
const MODE_MUTE = 2;
const MODE_BYPASS = 4;

// What a switch does to the nodes in its group, and the word on the chip that chooses it.
const ACTIONS = ["mute", "bypass"];
const ACTION_LABELS = { mute: "Mute", bypass: "Bypass" };
const ACTION_MODES = { mute: MODE_MUTE, bypass: MODE_BYPASS };

// The order the rows are drawn in, and the word on the chip that chooses it.
const ORDERS = ["graph", "name", "position"];
const ORDER_LABELS = { graph: "Graph", name: "Name", position: "Position" };

// Where each choice is kept. `properties` is serialised with the node, which is what a per node
// choice needs, and no python reads it.
const ACTION_KEY = "was_groups_action";
const ORDER_KEY = "was_groups_order";

// Row geometry, in CSS pixels.
const ROW_HEIGHT = 20;
const SWATCH_WIDTH = 3;
const PIP_WIDTH = 22;
const PIP_HEIGHT = 11;

// What a row spends on everything but its title, in CSS pixels: the colour band, the three
// gaps, the node count, the switch, the padding and the border.
const ROW_CHROME = 72;

/**
 * Whether the group list is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

// Every open panel, as `{node, refresh}`, and the one timer that calls them. A panel joins when
// its node joins a graph and leaves when the node does, so a node built for the clipboard and
// then thrown away never holds the timer open.
const ticking = new Set();
let tickHandle = 0;

/**
 * Stop the timer once no panel is left to call.
 *
 * @returns {void}
 */
function stopTicking() {
  if (ticking.size || !tickHandle) return;
  clearInterval(tickHandle);
  tickHandle = 0;
}

/**
 * Call every open panel's refresh once, and drop the ones whose node has left the graph.
 *
 * @returns {void}
 */
function tick() {
  // A hidden tab draws nothing, so there is nothing for a refresh to correct until it is shown.
  if (document.hidden) return;
  for (const entry of [...ticking]) {
    // A graph cleared rather than emptied node by node takes its nodes away without telling
    // each one, which would otherwise leave an entry reading a graph nobody can see.
    if (!entry.node?.graph) {
      ticking.delete(entry);
      continue;
    }
    try {
      entry.refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the graph's groups:`, error);
    }
  }
  stopTicking();
}

/**
 * Put one panel in the timer, starting the timer when it was stopped.
 *
 * @param {object} node - The node the panel is drawn on.
 * @param {() => void} refresh - What to call on every pass.
 * @returns {() => void} Release, which does nothing the second time it is called.
 */
function joinTicking(node, refresh) {
  const entry = { node, refresh };
  ticking.add(entry);
  if (!tickHandle) tickHandle = setInterval(tick, REFRESH_MS);
  let live = true;
  return () => {
    if (!live) return;
    live = false;
    ticking.delete(entry);
    stopTicking();
  };
}

/**
 * One of a fixed list of choices, held on the node.
 *
 * @param {object} node - The node the choice belongs to.
 * @param {string} key - Which property holds it.
 * @param {string[]} choices - Every value it may take, the first being the default.
 * @returns {string} The stored value, or the first choice when it holds anything else.
 */
function choice(node, key, choices) {
  const held = node?.properties?.[key];
  return choices.includes(held) ? held : choices[0];
}

/**
 * Move one choice on to the next value and keep it on the node.
 *
 * @param {object} node - The node the choice belongs to.
 * @param {string} key - Which property holds it.
 * @param {string[]} choices - Every value it may take, in the order they cycle.
 * @returns {void}
 */
function cycle(node, key, choices) {
  const next = choices[(choices.indexOf(choice(node, key, choices)) + 1) % choices.length];
  withGraphChange(() => {
    node.properties = node.properties ?? {};
    node.properties[key] = next;
  });
}

/**
 * The rectangle one item covers, as `[x, y, width, height]`.
 *
 * @param {object} item - A node or a group.
 * @returns {number[]|null} The rectangle, or null for an item with no measurable box.
 */
function boxOf(item) {
  const bounds = item?.boundingRect ?? item?._bounding;
  if (bounds && bounds.length >= 4) {
    return [Number(bounds[0]), Number(bounds[1]), Number(bounds[2]), Number(bounds[3])];
  }
  const pos = item?.pos;
  const size = item?.size;
  if (!pos || !size || pos.length < 2 || size.length < 2) return null;
  return [Number(pos[0]), Number(pos[1]), Number(size[0]), Number(size[1])];
}

/**
 * Whether a group holds a node, by the rule LiteGraph itself uses.
 *
 * @param {number[]} group - The group's rectangle.
 * @param {number[]} node - The node's rectangle.
 * @returns {boolean} True when the node's centre point lies inside the group.
 */
function holds(group, node) {
  const x = node[0] + node[2] * 0.5;
  const y = node[1] + node[3] * 0.5;
  return x >= group[0] && x < group[0] + group[2] && y >= group[1] && y < group[1] + group[3];
}

/**
 * Read every group in one graph, with the nodes each of them covers.
 *
 * @param {object} graph - The graph the panel's node sits in.
 * @param {string} order - Which of `ORDERS` the rows are drawn in.
 * @returns {Array<{group: object, title: string, colour: string, nodes: object[]}>} One entry
 *   per group, already sorted. Nodes carrying this same panel are left out of every entry, so a
 *   switch never mutes the node it was pressed on.
 */
function readGroups(graph, order) {
  const groups = Array.isArray(graph?.groups) ? [...graph.groups] : [];
  const entries = [];
  const boxes = [];
  for (const group of groups) {
    const box = boxOf(group);
    if (!box) continue;
    boxes.push(box);
    entries.push({
      group,
      title: String(group.title ?? ""),
      colour: String(group.color ?? ""),
      nodes: [],
    });
  }

  for (const node of graph?.nodes ?? []) {
    if (node?.type === NODE_ID) continue;
    const box = boxOf(node);
    if (!box) continue;
    for (let index = 0; index < entries.length; index += 1) {
      if (holds(boxes[index], box)) entries[index].nodes.push(node);
    }
  }

  if (order === "name") {
    entries.sort((a, b) => a.title.localeCompare(b.title, undefined, { numeric: true }));
  } else if (order === "position") {
    entries.sort((a, b) => {
      const first = boxOf(a.group) ?? [0, 0];
      const second = boxOf(b.group) ?? [0, 0];
      return first[1] - second[1] || first[0] - second[0];
    });
  }
  return entries;
}

/**
 * Whether a group counts as on, by the rule the selection toolbox uses.
 *
 * @param {object[]} nodes - The nodes the group covers.
 * @param {number} mode - The mode the switch puts them in.
 * @returns {boolean} True while at least one of them is not in that mode.
 */
function isOn(nodes, mode) {
  return nodes.length > 0 && !nodes.every((node) => node.mode === mode);
}

/**
 * Put every node in one group into the mode a switch chose, or back to always.
 *
 * @param {object[]} nodes - The nodes the group covers.
 * @param {number} mode - The mode the switch puts them in.
 * @returns {void}
 */
function toggleNodes(nodes, mode) {
  const wanted = isOn(nodes, mode) ? mode : MODE_ALWAYS;
  withGraphChange(() => {
    for (const node of nodes) node.mode = wanted;
  });
  app?.canvas?.setDirty?.(true, true);
}

/**
 * A small pressable label, drawn the same wherever the panel uses one.
 *
 * @param {string} hint - What the hover says, five words at most.
 * @param {() => void} onPress - What a left click or a keyboard press does.
 * @returns {HTMLButtonElement} The chip, for the caller to append and to paint.
 */
function createChip(hint, onPress) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.title = hint;
  chip.style.cssText = [
    "flex:0 0 auto",
    "padding:1px 6px",
    "border-radius:3px",
    "font:inherit",
    "line-height:15px",
    "cursor:pointer",
    "white-space:nowrap",
    `background:${themeVar("bgLight")}`,
    `border:1px solid ${themeVar("border")}`,
    `color:${themeVar("fg")}`,
  ].join(";");
  chip.addEventListener("click", (event) => {
    event.stopPropagation();
    onPress();
  });
  return chip;
}

/**
 * Build the group list for one node.
 *
 * @param {object} node - The node the list is drawn on.
 * @returns {{element: HTMLElement, height: number, maxHeight: number, minWidth: number,
 *   refresh: () => void, dispose: () => void}} The panel, the room it asks for, the pass the
 *   page timer calls, and its teardown.
 */
function createFastGroupsPanel(node) {
  const root = document.createElement("div");
  root.className = "was-fast-groups";
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
  const actionChip = createChip("Mute or bypass", () => {
    cycle(node, ACTION_KEY, ACTIONS);
    refresh();
  });
  const orderChip = createChip("Row order", () => {
    cycle(node, ORDER_KEY, ORDERS);
    refresh();
  });
  header.append(summary, actionChip, orderChip);

  const list = document.createElement("div");
  // The rows run in as many columns as the panel's width holds, so a wide node lists the graph
  // across rather than down one edge of it. `draw` sets the column width from the longest title.
  list.style.cssText =
    "flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;display:grid;"
    + "grid-template-columns:1fr;align-content:start;gap:2px;scrollbar-width:thin";

  const footer = document.createElement("div");
  footer.style.cssText = "flex:0 0 auto;display:flex;align-items:center;gap:5px";
  const note = document.createElement("span");
  note.style.cssText =
    `flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;color:${themeVar("fgMuted")}`;
  const allOn = createChip("Turn every group on", () => setEvery(true));
  const allOff = createChip("Turn every group off", () => setEvery(false));
  footer.append(note, allOn, allOff);

  root.append(header, list, footer);

  let entries = [];
  let signature = "";
  let disposed = false;

  /**
   * Drop the record of what was drawn, so the next pass draws whatever it finds.
   *
   * @returns {void}
   */
  function invalidate() {
    signature = "";
  }

  /**
   * Put every group in the graph on or off at once.
   *
   * @param {boolean} on - True to set every node back to always, false to apply the action.
   * @returns {void}
   */
  function setEvery(on) {
    const mode = ACTION_MODES[choice(node, ACTION_KEY, ACTIONS)];
    const wanted = on ? MODE_ALWAYS : mode;
    withGraphChange(() => {
      for (const entry of entries) {
        for (const member of entry.nodes) member.mode = wanted;
      }
    });
    app?.canvas?.setDirty?.(true, true);
    refresh();
  }

  /**
   * Draw one group as a row that presses.
   *
   * @param {object} entry - One entry from `readGroups`.
   * @param {number} mode - The mode the switch puts the group's nodes in.
   * @returns {HTMLButtonElement} The row, for the caller to append.
   */
  function createRow(entry, mode) {
    const empty = entry.nodes.length === 0;
    const on = isOn(entry.nodes, mode);

    const row = document.createElement("button");
    row.type = "button";
    row.title = entry.title;
    row.disabled = empty;
    row.style.cssText = [
      "box-sizing:border-box",
      "min-width:0",
      `height:${ROW_HEIGHT}px`,
      "display:flex",
      "align-items:center",
      "gap:6px",
      "padding:0 5px 0 0",
      "border-radius:3px",
      "font:inherit",
      "text-align:left",
      "overflow:hidden",
      `background:${themeVar(on ? "accentBg" : "bgDark")}`,
      `border:1px solid ${themeVar(on ? "accent" : "border")}`,
      `color:${themeVar(empty ? "fgDisabled" : "fg")}`,
      `cursor:${empty ? "default" : "pointer"}`,
    ].join(";");

    // The group's own colour, so a row is found by the colour of the band it names.
    const swatch = document.createElement("span");
    swatch.style.cssText = [
      "flex:0 0 auto",
      "align-self:stretch",
      `width:${SWATCH_WIDTH}px`,
      "border-radius:2px 0 0 2px",
      `background:${entry.colour || themeVar("border")}`,
    ].join(";");

    const title = document.createElement("span");
    title.style.cssText = "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    title.textContent = entry.title;

    const count = document.createElement("span");
    count.style.cssText = `flex:0 0 auto;color:${themeVar("fgMuted")}`;
    count.textContent = String(entry.nodes.length);

    const pip = document.createElement("span");
    pip.style.cssText = [
      "flex:0 0 auto",
      "box-sizing:border-box",
      `width:${PIP_WIDTH}px`,
      `height:${PIP_HEIGHT}px`,
      `border-radius:${PIP_HEIGHT}px`,
      "position:relative",
      `background:${themeVar(on ? "accent" : "inputBg")}`,
      `border:1px solid ${themeVar(on ? "accent" : "inputBorder")}`,
    ].join(";");
    const knob = document.createElement("span");
    const travel = PIP_WIDTH - PIP_HEIGHT;
    knob.style.cssText = [
      "position:absolute",
      "top:0",
      `left:${on ? travel - 1 : 0}px`,
      `width:${PIP_HEIGHT - 2}px`,
      `height:${PIP_HEIGHT - 2}px`,
      "border-radius:50%",
      `background:${themeVar(on ? "selectionText" : "fgMuted")}`,
    ].join(";");
    pip.appendChild(knob);

    row.append(swatch, title, count, pip);
    row.addEventListener("click", (event) => {
      event.stopPropagation();
      if (empty) return;
      // The group object can be replaced by a reload between the row being drawn and pressed.
      // Invalidated as well, since the replacement can read exactly as the row already drew.
      if (!node.graph?.groups?.includes(entry.group)) {
        invalidate();
        refresh();
        return;
      }
      toggleNodes(entry.nodes, mode);
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
    const action = choice(node, ACTION_KEY, ACTIONS);
    actionChip.textContent = ACTION_LABELS[action];
    orderChip.textContent = ORDER_LABELS[choice(node, ORDER_KEY, ORDERS)];
    allOn.textContent = "All on";
    allOff.textContent = "All off";

    if (!enabled()) {
      const off = document.createElement("span");
      off.style.cssText = `color:${themeVar("fgMuted")}`;
      off.textContent = "Group list off";
      list.style.gridTemplateColumns = "1fr";
      list.replaceChildren(off);
      header.style.display = "none";
      footer.style.display = "none";
      return;
    }
    header.style.display = "flex";
    footer.style.display = "flex";

    const mode = ACTION_MODES[action];
    const on = entries.filter((entry) => isOn(entry.nodes, mode)).length;
    summary.textContent = entries.length ? `${on} of ${entries.length} on` : "No groups yet";

    // Counted once each. A group drawn inside another holds its nodes twice over.
    const members = new Set();
    for (const entry of entries) for (const member of entry.nodes) members.add(member);
    const covered = members.size;
    note.textContent = entries.length ? `${covered} node${covered === 1 ? "" : "s"}` : "";

    // The column is as wide as the longest title needs and never wider than the panel, so a
    // narrow node keeps one column and a wide one splits into as many as fit.
    const longest = entries.reduce((wide, entry) => Math.max(wide, entry.title.length), 0);
    const column = `min(100%,calc(${longest + 2}ch + ${ROW_CHROME}px))`;
    list.style.gridTemplateColumns = `repeat(auto-fill,minmax(${column},1fr))`;

    // A row is replaced by the redraw its own press caused, so the keyboard is put back on the
    // row that is now in its place rather than being dropped onto the page.
    const focused = [...list.children].indexOf(document.activeElement);
    list.replaceChildren(...entries.map((entry) => createRow(entry, mode)));
    if (focused >= 0) list.children[focused]?.focus?.({ preventScroll: true });
  }

  /**
   * Look at the graph, and redraw when anything a row shows has moved.
   *
   * @returns {void}
   */
  function refresh() {
    if (disposed) return;
    if (!enabled()) {
      entries = [];
      if (signature === "off") return;
      signature = "off";
      draw();
      return;
    }
    const action = choice(node, ACTION_KEY, ACTIONS);
    const order = choice(node, ORDER_KEY, ORDERS);
    const next = readGroups(node.graph, order);
    const mode = ACTION_MODES[action];
    const marks = [action, order];
    for (const entry of next) {
      // The members are named rather than counted, so a node dragged out of one group and into
      // another is a change even where both counts stayed where they were.
      marks.push(
        entry.title,
        entry.colour,
        entry.nodes.map((member) => member.id).join(","),
        isOn(entry.nodes, mode) ? 1 : 0,
      );
    }
    // Held whether or not the rows are redrawn, so a press never reaches a group or a node the
    // graph has already replaced.
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
    // The graph canvas suppresses its own context menu on its own element, and this is a
    // separate element, so the browser menu would otherwise open over the node.
    event.preventDefault();
    event.stopPropagation();
  });

  // The rows are the only thing here that scrolls. At either end of the list the gesture is the
  // graph's and zooms the node under the pointer.
  const onWheel = (event) => {
    if (list.scrollHeight <= list.clientHeight || !list.contains(event.target)) return false;
    const before = list.scrollTop;
    list.scrollTop += wheelPixels(event, list).y;
    return list.scrollTop !== before;
  };
  let releaseWheel = captureWheel(root, onWheel);

  root.addEventListener("keydown", (event) => {
    // Clicking a row focuses it and selects the node, and the frontend binds both of these to
    // deleting the selection, so the node a switch was just pressed on would be deleted.
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
      // The same node object joins a graph again after leaving one, and its teardown emptied
      // the rows, released the wheel and stopped the pass that fills them.
      disposed = false;
      releaseWheel ??= captureWheel(root, onWheel);
      invalidate();
      leaveTicking?.();
      leaveTicking = joinTicking(node, refresh);
      refresh();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to start reading the graph's groups:`, error);
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
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Fast Groups", "Show the groups"],
      name: "List the graph's groups",
      tooltip:
        "Draw every group in the graph on the Fast Groups node, each with a switch that mutes "
        + "or bypasses the nodes inside it. Turning this off leaves the node in place and empty, "
        + "so a saved workflow still loads. It applies to every one of these nodes at once.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_fast_groups_wrapped) return;
    proto.__was_fast_groups_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      // The two choices are drawn on the panel and read by nothing in python, so they belong
      // to the node rather than to a widget the prompt would carry.
      this.serialize_widgets = false;
      this.properties = this.properties ?? {};
      this.size = [
        Math.max(this.size?.[0] ?? 0, NODE_SIZE[0]),
        Math.max(this.size?.[1] ?? 0, NODE_SIZE[1]),
      ];

      try {
        const panel = createFastGroupsPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the group list:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the group list:`, error);
      }
      return result;
    };
  },

});
