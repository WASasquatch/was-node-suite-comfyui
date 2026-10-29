/**
 * The exposed inputs of a saved app workflow, drawn on App Workflow as its own widgets.
 *
 * One widget per exposed input, of the type the node behind it declares. Their values are
 * sent in the `overrides` widget.
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { fetchExposure } from "./interface/app_exposure.js";
import { COLOUR_WIDGETS, attachSwatch } from "./was_colour_swatch.js";

const EXT_NAME = "WASNodeSuite.AppWorkflow";
const NODE = "WASAppWorkflow";

// Marks a widget this extension added, so a change of workflow removes only its own.
const OWNED = "__was_app_input";

// Marks the chooser as already watched, so a refresh does not stack watchers.
const WATCHED = "__was_app_watched";

// Holds the rebuild in flight, and the wires a restored graph had on its widget sockets.
const PENDING = "__was_app_pending";

// Set while a saved graph is being restored, so the rebuild waits for its wires.
const LOADING = "__was_app_loading";
const STARTER = "__was_app_starter";
const SAVED_JOINS = "__was_app_joins";

// The link table of the graph being restored, read before its nodes are configured.
let restoring = new Map();

// The node definition the canvas reads a socket's hint from, found once.
let held;

// Widgets added for one workflow. An app exposing more than this draws the rest as JSON.
const MAX_WIDGETS = 32;

// Widgets the schema declares, which this extension never removes.
const DECLARED_WIDGETS = new Set(["app", "overrides"]);

// Hints for a socket past what the chosen workflow exposes.
const UNBOUND_INPUT = "The workflow exposes no input for this slot; any type.";
const UNBOUND_OUTPUT = "The workflow presents no result for this slot; any type.";

// Hints for a socket while a link names the workflow, which is read only when the node runs.
const WIRED_INPUT = "Feeds the wired workflow's exposed input in this place; any type.";
const WIRED_OUTPUT = "The wired workflow's result in this place; any type.";

// Whether the `app` input was filled by a link at the node's last rebuild.
const LINKED = "__was_app_linked";

/**
 * The widget on a node, by name.
 *
 * @param {object} node - The node to look on.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget, or undefined when there is none.
 */
function widgetNamed(node, name) {
  return (node.widgets ?? []).find((widget) => widget.name === name);
}

/**
 * Whether a link fills the node's `app` input in place of its menu.
 *
 * @param {object} node - The App Workflow node.
 * @returns {boolean} True while a link is attached.
 */
function appLinked(node) {
  const slot = (node.inputs ?? []).find((each) => each?.name === "app");
  return slot?.link !== null && slot?.link !== undefined;
}

/**
 * What is wired into each exposed input's own socket.
 *
 * @param {object} node - The App Workflow node.
 * @returns {object} `{widget name: {id, slot}}` for every one carrying a wire.
 */
function joinedWidgets(node) {
  const graph = node.graph;
  const found = {};
  if (!graph) return found;
  for (const slot of node.inputs ?? []) {
    const name = slot.widget?.name;
    if (!name || slot.link == null) continue;
    const link = graph.links?.[slot.link] ?? graph._links?.get(slot.link);
    if (link) found[name] = { id: link.origin_id, slot: link.origin_slot };
  }
  return found;
}

/**
 * Put back what was wired into each exposed input's own socket.
 *
 * @param {object} node - The App Workflow node.
 * @param {object} joined - What `joinedWidgets` answered before the rebuild.
 * @returns {void}
 */
function rejoinWidgets(node, joined) {
  const graph = node.graph;
  if (!graph) return;
  for (const [name, upstream] of Object.entries(joined)) {
    const index = (node.inputs ?? []).findIndex((slot) => slot.widget?.name === name);
    if (index < 0) continue;
    graph.getNodeById(upstream.id)?.connect(upstream.slot, node, index);
  }
}

/**
 * Put back what a saved graph wired into each exposed input's own socket.
 *
 * @param {object} node - The App Workflow node.
 * @returns {void}
 */
function rejoinSaved(node) {
  const graph = node.graph;
  const saved = node[SAVED_JOINS];
  delete node[SAVED_JOINS];
  if (!graph || !saved?.length) return;
  for (const { name, link: id } of saved) {
    const index = (node.inputs ?? []).findIndex((slot) => slot.widget?.name === name);
    const row = restoring.get(id);
    if (index < 0 || !row || node.inputs[index].link != null) continue;
    graph.getNodeById(row.origin)?.connect(row.slot, node, index);
  }
}

/**
 * Drop every widget this extension added to a node.
 *
 * @param {object} node - The node to clear.
 * @returns {void}
 */
function clearOwned(node) {
  for (const widget of [...(node.widgets ?? [])]) {
    if (!widget[OWNED]) continue;
    widget.onRemove?.();
    node.widgets.splice(node.widgets.indexOf(widget), 1);
  }
  // A restored graph brings back these slots before their widgets exist, so every widget
  // input this extension is responsible for goes, whether its widget was found or not.
  for (let index = (node.inputs ?? []).length - 1; index >= 0; index--) {
    const bound = node.inputs[index].widget?.name;
    if (!bound || DECLARED_WIDGETS.has(bound)) continue;
    node.removeInput(index);
  }
}

/**
 * Build the widget a text input is typed into, in whichever graph is drawing.
 *
 * @param {object} node - The App Workflow node.
 * @param {string} label - The exposed input's name.
 * @param {object} entry - One entry of the exposure answer.
 * @param {Function} record - Called with the value whenever it changes.
 * @returns {object|null} The widget, or null when none could be built.
 */
function stringWidget(node, label, entry, record) {
  if (entry.multiline) {
    try {
      const built = ComfyWidgets.STRING(node, label, ["STRING", { multiline: true }], app);
      const widget = built?.widget ?? built;
      if (widget) {
        widget.callback = record;
        return widget;
      }
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to build the ${label} text box:`, error);
    }
  }
  return node.addWidget("text", label, String(entry.value ?? entry.default ?? ""), record);
}

/**
 * The value one exposed input starts at.
 *
 * @param {object} entry - One entry of the exposure answer.
 * @returns {*} The value the workflow saved, or the input's own default.
 */
function seed(entry) {
  const value = entry.value ?? entry.default;
  if (entry.kind === "INT" || entry.kind === "FLOAT") return Number(value ?? 0);
  if (entry.kind === "BOOLEAN") return Boolean(value ?? false);
  if (entry.kind === "STRING") return String(value ?? "");
  return value ?? (Array.isArray(entry.options) ? entry.options[0] : "");
}

/**
 * Write one exposed input's value into the overrides text.
 *
 * @param {object} node - The App Workflow node.
 * @param {string} label - The exposed input's name.
 * @param {*} value - What the widget now holds.
 * @returns {void}
 */
function recordValue(node, label, value) {
  const overrides = widgetNamed(node, "overrides");
  if (!overrides) return;
  let held = {};
  try {
    const parsed = JSON.parse((overrides.value ?? "").trim() || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) held = parsed;
  } catch (error) {
    // Text that will not read is left alone rather than replaced.
    return;
  }
  held[label] = value;
  overrides.value = JSON.stringify(held, null, 2);
}

/**
 * Add one widget for an exposed input, of the type the node behind it declares.
 *
 * @param {object} node - The App Workflow node.
 * @param {object} entry - One entry of the exposure answer.
 * @returns {object|null} The widget added, or null for an input with no drawable type.
 */
function addInputWidget(node, entry) {
  const label = entry.label;
  const settings = {};
  for (const key of ["min", "max", "step", "round", "multiline"]) {
    if (entry[key] !== undefined && entry[key] !== null) settings[key] = entry[key];
  }
  const record = (value) => recordValue(node, label, value);
  let widget = null;
  if (entry.kind === "COMBO" && Array.isArray(entry.options)) {
    widget = node.addWidget("combo", label, entry.value ?? entry.options[0], record,
      { values: entry.options });
  } else if (entry.kind === "INT" || entry.kind === "FLOAT") {
    const step = entry.kind === "INT" ? 1 : (entry.step ?? 0.01);
    widget = node.addWidget("number", label, Number(entry.value ?? entry.default ?? 0), record,
      { ...settings, step: step * 10, precision: entry.kind === "INT" ? 0 : 2 });
  } else if (entry.kind === "BOOLEAN") {
    widget = node.addWidget("toggle", label, Boolean(entry.value ?? entry.default ?? false), record);
  } else if (entry.kind === "STRING") {
    widget = stringWidget(node, label, entry, record);
  }
  if (!widget) return null;
  // A widget of this name may be reused rather than built, so the value is put on after.
  widget.value = seed(entry);
  widget[OWNED] = true;
  // The widget takes a wire on its own row, the way a declared widget does.
  node.addInput(label, entry.kind === "COMBO" ? "COMBO" : (entry.kind ?? "*"),
    { widget: { name: label }, tooltip: inputHint(entry) });
  widget.serialize = false;
  if ((COLOUR_WIDGETS[entry.node] ?? []).includes(entry.widget)) attachSwatch(node, widget);
  widget.tooltip = entry.slot
    ? `${inputHint(entry)} Wiring the ${entry.label} socket replaces it.`
    : inputHint(entry);
  return widget;
}

/**
 * Rebuild whenever the chosen workflow changes, however it was set.
 *
 * @param {object} node - The App Workflow node.
 * @param {object} chooser - The `app` widget.
 * @returns {void}
 */
function watchChoice(node, chooser) {
  if (!chooser || chooser[WATCHED]) return;
  chooser[WATCHED] = true;
  let held = chooser.value;
  const start = () => {
    const wanted = held;
    node[PENDING] = rebuild(node, wanted).catch((error) => {
      console.error(`[${EXT_NAME}] Failed to read what the workflow offers:`, error);
    });
  };
  Object.defineProperty(chooser, "value", {
    configurable: true,
    get: () => held,
    // A restored value is assigned rather than picked, so the change is caught here.
    set(next) {
      const changed = next !== held;
      held = next;
      if (changed && !node[LOADING]) start();
    },
  });
  node[STARTER] = start;
  // A node being restored is configured on the same tick it is made, so the first rebuild
  // waits a tick to find out whether it is about to be.
  setTimeout(() => {
    if (!node[LOADING] && !node[PENDING]) start();
  }, 0);
}

/**
 * Rebuild a node's exposed-input widgets for the workflow it now names.
 *
 * @param {object} node - The App Workflow node.
 * @returns {Promise<void>} Resolved once the widgets are in place.
 */
async function rebuild(node, chosen) {
  const linked = appLinked(node);
  node[LINKED] = linked;
  // A wired workflow is named only when the node runs, so the menu's choice offers nothing.
  const answer = linked ? null : await fetchExposure(chosen);
  // A later choice may have overtaken this one while its answer was on the way.
  if (widgetNamed(node, "app")?.value !== chosen || appLinked(node) !== linked) return;
  const saved = node[OWNED + "_values"] ?? {};
  delete node[OWNED + "_values"];
  const joined = joinedWidgets(node);
  clearOwned(node);
  // The text holds this workflow's inputs and no other's, so a change of workflow starts it over.
  const overrides = widgetNamed(node, "overrides");
  if (overrides) overrides.value = "{}";
  node.__was_app_exposure = answer;
  const typed = (answer?.inputs ?? []).filter((entry) => !entry.wire);
  for (const entry of typed.slice(0, MAX_WIDGETS)) {
    const widget = addInputWidget(node, entry);
    // A value the node was loaded with wins over the one the workflow saved.
    if (widget && saved[entry.label] !== undefined) widget.value = saved[entry.label];
  }
  mutateSockets(node, answer, linked);
  rejoinWidgets(node, joined);
  for (const widget of node.widgets ?? []) {
    if (widget[OWNED]) recordValue(node, widget.name, widget.value);
  }
  node.setSize(node.computeSize());
  node.setDirtyCanvas?.(true, true);
}

/**
 * A hint for one exposed input, from the input it stands for.
 *
 * @param {object} entry - One entry of the exposure answer.
 * @returns {string} The hint, ending in what the input accepts.
 */
function inputHint(entry) {
  const range = entry.min !== undefined && entry.max !== undefined
    ? `, ${entry.min} to ${entry.max}` : "";
  const carries = entry.wire
    ? `${entry.wire}, standing in for what ${entry.node ?? "it"} reads`
    : (entry.kind === "COMBO" && entry.options
      ? `one of ${entry.options.length} choices`
      : `${entry.kind ?? "any type"}${range}`);
  const own = entry.tooltip ? `${entry.tooltip} ` : "";
  return `${own}${entry.label} on ${entry.node ?? "the workflow"}; ${carries}.`;
}

/**
 * Every definition of this node the canvas reads a socket's hint from.
 *
 * @param {object} node - The App Workflow node.
 * @returns {Array<object>} The definitions found, which may be none.
 */
function definitions(node) {
  const found = [];
  if (node.constructor?.nodeData) found.push(node.constructor.nodeData);
  if (held !== undefined) {
    if (held) found.push(held);
    return found;
  }
  held = null;
  try {
    const root = app.vueApp ?? document.querySelector("#vue-app")?.__vue_app__;
    const provides = root?._context?.provides ?? {};
    for (const key of Object.getOwnPropertySymbols(provides)) {
      const stores = provides[key]?._s;
      if (!(stores instanceof Map)) continue;
      for (const store of stores.values()) {
        const found_ = store?.nodeDefsByName?.[NODE];
        if (found_) held = found_;
      }
    }
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to reach the node definitions:`, error);
  }
  if (held) found.push(held);
  return found;
}

/**
 * Point the node definition's hints at one node's chosen workflow.
 *
 * @param {object} node - The App Workflow node.
 * @returns {void}
 */
function describeSlots(node) {
  // The hint is read off the definition every node of this type shares, so it is written
  // from whichever node is being read.
  for (const definition of definitions(node)) {
    for (const slot of node.inputs ?? []) {
      const declared = definition.inputs?.[slot.name];
      if (declared && slot.tooltip) declared.tooltip = slot.tooltip;
    }
    (node.outputs ?? []).forEach((slot, index) => {
      const declared = definition.outputs?.[index];
      if (declared && slot.tooltip) declared.tooltip = slot.tooltip;
    });
    // A socket the workflow has nothing for keeps wording of its own rather than the last
    // workflow's, since the definition is shared by every node of this type.
    for (const [name, declared] of Object.entries(definition.inputs ?? {})) {
      if (/^input_\d+$/.test(name) && !(node.inputs ?? []).some((slot) => slot.name === name)) {
        declared.tooltip = UNBOUND_INPUT;
      }
    }
    (definition.outputs ?? []).forEach((declared, index) => {
      if (/^output_\d+$/.test(declared?.name ?? "") && index >= (node.outputs ?? []).length) {
        declared.tooltip = UNBOUND_OUTPUT;
      }
    });
  }
}

/**
 * Rebuild a node's slots to the ones the chosen workflow actually uses.
 *
 * @param {object} node - The App Workflow node.
 * @param {number} usedInputs - Exposed inputs a socket is kept for.
 * @param {number} usedOutputs - Results a socket is kept for.
 * @returns {void}
 */
function refreshSlots(node, usedInputs, usedOutputs) {
  const graph = node.graph;
  if (!graph) return;
  const linkOf = (id) => (id == null ? null : (graph.links?.[id] ?? graph._links?.get(id)));
  const spare = (name, kept) => {
    const match = /^(?:input|output)_(\d+)$/.exec(name);
    return match ? Number(match[1]) > kept : false;
  };

  const inputs = (node.inputs ?? []).filter((slot) => !spare(slot.name, usedInputs)).map((slot) => {
    const link = linkOf(slot.link);
    return {
      name: slot.name, type: slot.type, label: slot.label, localised: slot.localized_name,
      tooltip: slot.tooltip, shape: slot.shape,
      widget: slot.widget ? { name: slot.widget.name } : undefined,
      upstream: link ? { id: link.origin_id, slot: link.origin_slot } : null,
    };
  });
  const outputs = (node.outputs ?? []).filter((slot) => !spare(slot.name, usedOutputs)).map((slot) => ({
    name: slot.name, type: slot.type, label: slot.label, localised: slot.localized_name,
    tooltip: slot.tooltip, shape: slot.shape,
    downstream: (slot.links ?? []).map(linkOf).filter(Boolean)
      .map((link) => ({ id: link.target_id, slot: link.target_slot })),
  }));

  // A name is read when its slot is built, so every slot is rebuilt in the order it held.
  for (let index = (node.inputs ?? []).length - 1; index >= 0; index--) node.removeInput(index);
  for (const entry of inputs) {
    node.addInput(entry.name, entry.type, {
      label: entry.label, localized_name: entry.localised, tooltip: entry.tooltip,
      shape: entry.shape, widget: entry.widget,
    });
  }
  for (let index = (node.outputs ?? []).length - 1; index >= 0; index--) node.removeOutput(index);
  for (const entry of outputs) {
    node.addOutput(entry.name, entry.type, {
      label: entry.label, localized_name: entry.localised, tooltip: entry.tooltip,
      shape: entry.shape,
    });
  }

  inputs.forEach((entry, index) => {
    if (entry.upstream) graph.getNodeById(entry.upstream.id)?.connect(entry.upstream.slot, node, index);
  });
  outputs.forEach((entry, index) => {
    for (const target of entry.downstream) {
      const downstream = graph.getNodeById(target.id);
      if (downstream) node.connect(index, downstream, target.slot);
    }
  });
}

/**
 * Put back every slot the schema declares, so a rebuild has them all to choose from.
 *
 * @param {object} node - The App Workflow node.
 * @returns {void}
 */
function restoreSlots(node) {
  const declared = node.constructor?.nodeData;
  const optional = declared?.input?.optional ?? {};
  for (const name of Object.keys(optional)) {
    if (!/^input_\d+$/.test(name) || (node.inputs ?? []).some((slot) => slot.name === name)) continue;
    const at = (node.inputs ?? []).findLastIndex((slot) => /^input_\d+$/.test(slot.name));
    node.addInput(name, "*");
    // A restored socket belongs beside its fellows, not after the widget inputs.
    const added = node.inputs.pop();
    node.inputs.splice(at + 1, 0, added);
  }
  const names = declared?.output_name ?? [];
  names.forEach((name, index) => {
    if (!/^output_\d+$/.test(name) || (node.outputs ?? []).some((slot) => slot.name === name)) return;
    node.addOutput(name, declared.output?.[index] ?? "*");
    const added = node.outputs.pop();
    node.outputs.splice(index, 0, added);
  });
}

/**
 * Name and type each socket after the input or result it stands for.
 *
 * @param {object} node - The App Workflow node.
 * @param {object|null} answer - The exposure answer, or null for no workflow.
 * @param {boolean} [wired] - A link names the workflow, so every declared slot is kept.
 * @returns {void}
 */
function mutateSockets(node, answer, wired = false) {
  restoreSlots(node);
  // Only an input a wire stands in for is given a socket; the rest are widgets.
  const inputs = (answer?.inputs ?? []).filter((entry) => entry.slot);
  for (const slot of node.inputs ?? []) {
    const match = /^input_(\d+)$/.exec(slot.name);
    if (!match) continue;
    const entry = inputs[Number(match[1]) - 1];
    // The canvas draws localized_name, and falls back to the slot's own name without it.
    slot.localized_name = entry ? entry.label : slot.name;
    slot.label = entry ? entry.label : undefined;
    // The declared type stays `*`, so the socket only narrows what the canvas will join.
    slot.type = entry?.wire ?? (entry?.kind && entry.kind !== "COMBO" ? entry.kind : "*");
    slot.tooltip = entry ? inputHint(entry) : (wired ? WIRED_INPUT : UNBOUND_INPUT);
  }

  const results = answer?.outputs ?? [];
  for (const slot of node.outputs ?? []) {
    const match = /^output_(\d+)$/.exec(slot.name);
    if (!match) continue;
    const entry = results[Number(match[1]) - 1];
    slot.localized_name = entry?.name ?? slot.name;
    slot.label = entry?.name ?? undefined;
    slot.type = entry?.type ?? "*";
    slot.tooltip = entry
      ? `${entry.tooltip ? `${entry.tooltip} ` : ""}${entry.name} from `
        + `${entry.node ?? "the workflow"}, which ${entry.presented_by ?? "it"} presents; `
        + `${entry.type ?? "any type"}.`
      : (wired ? WIRED_OUTPUT : UNBOUND_OUTPUT);
  }
  refreshSlots(node, wired ? Infinity : inputs.length, wired ? Infinity : results.length);
  describeSlots(node);
}

/**
 * The overrides JSON a node sends, built from its exposed-input widgets.
 *
 * @param {object} node - The App Workflow node.
 * @param {string} typed - What the overrides widget itself holds.
 * @returns {string} The JSON to send.
 */
function overridesFor(node, typed) {
  let base = {};
  try {
    const parsed = JSON.parse((typed ?? "").trim() || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) base = parsed;
  } catch (error) {
    // Text that will not read is sent as it stands, so the node reports it rather than
    // this quietly replacing it.
    return typed;
  }
  const collected = {};
  for (const widget of node.widgets ?? []) {
    if (widget[OWNED]) collected[widget.name] = widget.value;
  }
  return JSON.stringify({ ...collected, ...base });
}

app.registerExtension({
  name: EXT_NAME,

  /**
   * Keep the link table of a graph about to be restored.
   *
   * @param {object} graphData - The workflow being loaded.
   * @returns {void}
   */
  beforeConfigureGraph(graphData) {
    restoring = new Map();
    for (const link of graphData?.links ?? []) {
      if (Array.isArray(link)) restoring.set(link[0], { origin: link[1], slot: link[2] });
      else if (link?.id !== undefined) restoring.set(link.id, { origin: link.origin_id, slot: link.origin_slot });
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const proto = nodeType.prototype;
    if (proto.__was_app_workflow_wrapped) return;
    proto.__was_app_workflow_wrapped = true;

    const originalMouseEnter = proto.onMouseEnter;
    proto.onMouseEnter = function (...args) {
      try {
        describeSlots(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to name the sockets:`, error);
      }
      return originalMouseEnter?.apply(this, args);
    };

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const node = this;
        watchChoice(node, widgetNamed(node, "app"));
        const overrides = widgetNamed(node, "overrides");
        if (overrides) {
          overrides.serializeValue = () => overridesFor(node, overrides.value);
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to prepare the node:`, error);
      }
      return result;
    };

    // A link joining or leaving `app` changes what the node can know before it runs. The
    // check waits a tick, so a graph being restored is read once its links are all in.
    const originalOnConnectionsChange = proto.onConnectionsChange;
    proto.onConnectionsChange = function (...args) {
      const result = originalOnConnectionsChange?.apply(this, args);
      const node = this;
      setTimeout(() => {
        try {
          if (!node[LOADING] && node.graph && appLinked(node) !== Boolean(node[LINKED])) {
            node[STARTER]?.();
          }
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to follow the app link:`, error);
        }
      }, 0);
      return result;
    };

    // Links are restored after every node is configured, so the rebuild that reads them
    // waits until then rather than running as the node is made.
    const originalAfterConfigured = proto.onAfterGraphConfigured;
    proto.onAfterGraphConfigured = function (...args) {
      const result = originalAfterConfigured?.apply(this, args);
      delete this[LOADING];
      const node = this;
      node[STARTER]?.();
      Promise.resolve(node[PENDING]).then(() => rejoinSaved(node)).catch((error) => {
        console.error(`[${EXT_NAME}] Failed to put the wires back:`, error);
      });
      return result;
    };

    // A saved graph carries the exposed values in its overrides text, which is read back
    // onto the widgets once they exist.
    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function (info) {
      this[LOADING] = true;
      const result = originalOnConfigure?.apply(this, arguments);
      try {
        const text = (this.widgets_values ?? []).find(
          (value) => typeof value === "string" && value.trim().startsWith("{"),
        );
        const parsed = text ? JSON.parse(text) : null;
        if (parsed && typeof parsed === "object") this[OWNED + "_values"] = parsed;
        // Slots naming a widget are this extension's, and litegraph drops them when it
        // rebuilds the node from its definition, so what they carried is kept here.
        this[SAVED_JOINS] = (info?.inputs ?? [])
          .filter((slot) => slot.widget?.name && slot.link != null
            && !DECLARED_WIDGETS.has(slot.widget.name))
          .map((slot) => ({ name: slot.widget.name, link: slot.link }));
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read the saved values back:`, error);
      }
      return result;
    };
  },
});
