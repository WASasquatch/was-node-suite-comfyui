/**
 * Number Sequence's list drawn on the node, its values set by dragging them.
 *
 * A dragged point writes `start` or `end`, or its entry of the values box. Widgets the mode
 * does not read are folded away.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ICON } from "./interface/icons.js";
import { followMode } from "./interface/mode_widgets.js";
import { createNumberPlot } from "./interface/number_plot.js";
import {
  CURVE_NAMES,
  LOOPS,
  MAX_COUNT,
  buildRounded,
  keyframeIndex,
  lengthOf,
  readEntries,
} from "./interface/number_sequence.js";
import { executionId, nodeLocator, ownOutput } from "./interface/preview.js";
import {
  appendInterfaceWidget,
  boundTextBoxes,
  chainWidgetCallback,
  graphLoading,
} from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.NumberSequenceUI";
const LOG_NAME = "WASNodeSuite.NumberSequence";
const NODE_ID = "WASNumberSequence";
const SETTING_ID = "WAS.NumberSequence.ShowPlot";

const UI_WIDGET_NAME = "was_number_sequence_ui";
const UI_WIDGET_TYPE = "was_number_plot";

// The key the node's run output carries its values under.
const OUTPUT_KEY = "was_number_sequence";

// The widgets each mode reads; the rest of the node reads in both.
const BY_MODE = {
  range: ["start", "end"],
  values: ["values"],
};

// Every widget the list is worked out from.
const READ = ["mode", "start", "end", "values", "count", "curve", "curve_on", "repeat", "loop",
  "reverse", "decimals"];

/**
 * Read whether the plot is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
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

/**
 * A widget on a node, by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget.
 */
function widgetOf(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

// Nodes whose output is the value of their one widget, read on the canvas before a run.
const PRIMITIVES = new Set([
  "PrimitiveInt",
  "PrimitiveFloat",
  "PrimitiveBoolean",
  "PrimitiveString",
  "PrimitiveStringMultiline",
  "PrimitiveNode",
]);

// Primitive widgets already reporting their changes, to the repaints that read them.
const watchedPrimitives = new WeakMap();

/**
 * Whether an input arrives on a wire rather than from its widget.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {boolean} True when the input has a link.
 */
function wired(node, name) {
  return (node.inputs ?? []).some((input) => input.name === name && input.link != null);
}

/**
 * The widget of the primitive node wired into an input, when that is what feeds it.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {object|null} The primitive's widget, or null for an unwired input or another node.
 */
function primitiveSource(node, name) {
  const input = (node.inputs ?? []).find((candidate) => candidate.name === name);
  if (input?.link == null) return null;
  const links = node.graph?.links;
  const link = links?.get?.(input.link) ?? links?.[input.link];
  const origin = link ? node.graph.getNodeById(link.origin_id) : null;
  if (!origin || !PRIMITIVES.has(origin.type)) return null;
  return origin.widgets?.find((widget) => widget.name === "value") ?? origin.widgets?.[0] ?? null;
}

/**
 * Repaint whenever a primitive's widget changes.
 *
 * @param {object} widget - The primitive's widget.
 * @param {() => void} repaint - The repaint to run.
 * @returns {void}
 */
function followPrimitive(widget, repaint) {
  let repaints = watchedPrimitives.get(widget);
  if (!repaints) {
    repaints = new Set();
    watchedPrimitives.set(widget, repaints);
    const original = widget.callback;
    widget.callback = function (...args) {
      const result = original?.apply(this, args);
      for (const run of repaints) {
        try {
          run();
        } catch (error) {
          console.error(`[${LOG_NAME}] Failed to repaint after a primitive changed:`, error);
        }
      }
      return result;
    };
  }
  repaints.add(repaint);
}

/**
 * What one input holds on the canvas: its widget, or the primitive wired into it.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @param {() => void} [repaint] - Run when a wired primitive changes.
 * @returns {{known: boolean, value: *}} `known` false when another node's run decides it.
 */
function inputValue(node, name, repaint) {
  if (!wired(node, name)) return { known: true, value: widgetOf(node, name)?.value };
  const source = primitiveSource(node, name);
  if (!source) return { known: false, value: widgetOf(node, name)?.value };
  if (repaint) followPrimitive(source, repaint);
  return { known: true, value: source.value };
}

/**
 * Write one widget as a single undo step.
 *
 * @param {object} widget - The widget.
 * @param {*} value - Its new value.
 * @returns {void}
 */
function writeWidget(widget, value) {
  const canvas = app.canvas;
  const bracketed = typeof canvas?.emitBeforeChange === "function"
    && typeof canvas?.emitAfterChange === "function";
  if (bracketed) canvas.emitBeforeChange();
  try {
    widget.value = value;
  } finally {
    if (bracketed) canvas.emitAfterChange();
  }
}

/**
 * The lowest and highest of a list.
 *
 * @param {number[]} values - The list.
 * @returns {[number, number]} Both ends.
 */
function extent(values) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  return [low, high];
}

/**
 * The words for a list's length and spread.
 *
 * @param {number[]} values - The list.
 * @returns {string} As `20 entries from 1 to 0`, or `between 0 and 1` where it turns back.
 */
function described(values) {
  const [low, high] = extent(values);
  const count = `${values.length} entr${values.length === 1 ? "y" : "ies"}`;
  if (low === high) return `${count}, all ${low}`;
  let rising = true;
  let falling = true;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[index - 1]) rising = false;
    if (values[index] > values[index - 1]) falling = false;
  }
  return rising || falling
    ? `${count} from ${values[0]} to ${values[values.length - 1]}`
    : `${count} between ${low} and ${high}`;
}

/**
 * What the plot shows for the node as its widgets stand.
 *
 * @param {object} node - The node.
 * @param {number[]|null} lastRun - The values the node emitted on its last run, or null.
 * @param {() => void} [repaint] - Run when a primitive wired into the node changes.
 * @returns {object} A model for `createNumberPlot`'s `show`.
 */
function modelOf(node, lastRun, repaint) {
  const held = new Map();
  const read = (name) => {
    if (!held.has(name)) held.set(name, inputValue(node, name, repaint));
    return held.get(name).value;
  };
  const mode = read("mode") === "values" ? "values" : "range";
  const used = [...BY_MODE[mode], ...READ.filter((name) => !BY_MODE.range.includes(name)
    && !BY_MODE.values.includes(name))];
  const arriving = used.filter((name) => {
    read(name);
    return !held.get(name).known;
  });

  if (arriving.length) {
    if (!lastRun?.length) {
      return {
        empty: `${arriving.join(", ")} arrive on a wire: queue the graph to draw the list`,
        summary: "",
        claim: { icon: ICON.WARNING, detail: "nothing is drawn until the node runs" },
      };
    }
    return {
      series: [{ name: "values", values: lastRun }],
      summary: described(lastRun),
      claim: {
        icon: ICON.EXACT,
        detail: `the values of the last run; ${arriving.join(", ")} arrive on a wire`,
      },
    };
  }

  let keys;
  if (mode === "range") {
    keys = [Number(read("start")), Number(read("end"))];
  } else {
    const entries = readEntries(read("values"));
    if (!entries.length) {
      return { blocking: "the values box is empty", summary: "nothing to build",
        claim: { icon: ICON.WARNING, detail: "the node stops with an error as it stands" } };
    }
    const unreadable = entries.findIndex((entry) => entry.value === null);
    if (unreadable >= 0) {
      return {
        blocking: `value ${unreadable + 1}, "${entries[unreadable].text.slice(0, 24)}", is not a number`,
        summary: "nothing to build",
        claim: { icon: ICON.WARNING, detail: "the node stops with an error as it stands" },
      };
    }
    if (entries.length > MAX_COUNT) {
      return { blocking: `${entries.length} values; the node takes at most ${MAX_COUNT}`,
        summary: "nothing to build",
        claim: { icon: ICON.WARNING, detail: "the node stops with an error as it stands" } };
    }
    keys = entries.map((entry) => entry.value);
  }

  const settings = {
    count: Math.trunc(Number(read("count")) || 0),
    curve: String(read("curve") ?? "linear"),
    target: String(read("curve_on") ?? "transitions"),
    repeat: Math.max(1, Math.trunc(Number(read("repeat")) || 1)),
    loop: String(read("loop") ?? "restart"),
    reverse: Boolean(read("reverse")),
    decimals: Math.trunc(Number(read("decimals")) || 0),
  };
  const refusal = (text) => ({ blocking: text, summary: "nothing to build",
    claim: { icon: ICON.WARNING, detail: "the node stops with an error as it stands" } });
  if (!CURVE_NAMES.includes(settings.curve)) return refusal(`no curve called ${settings.curve}`);
  if (!LOOPS.includes(settings.loop)) return refusal(`no loop called ${settings.loop}`);
  const length = lengthOf(keys.length, settings.count, settings.repeat, settings.loop);
  if (length > MAX_COUNT) return refusal(`${length} entries; the node holds at most ${MAX_COUNT}`);
  if (settings.loop === "restart" && length < settings.repeat) {
    return refusal(`${length} entries cannot fill ${settings.repeat} repeats`);
  }

  const values = buildRounded(keys, settings);
  const handles = keys.map((value, index) => ({
    key: index,
    index: keyframeIndex(index, keys.length, values.length, settings.repeat, settings.loop,
      settings.reverse),
    value,
    label: mode === "range" ? (index === 0 ? "start" : "end") : `value ${index + 1}`,
  }));
  return {
    series: [{ name: "values", values }],
    handles,
    editable: mode === "range" ? !wired(node, "start") && !wired(node, "end") : !wired(node, "values"),
    summary: described(values),
    claim: { icon: ICON.EXACT, detail: "worked out in the browser with the node's own arithmetic" },
  };
}

/**
 * Append the plot to a node and wire it to the widgets it reads and writes.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPlot(node) {
  let lastRun = null;

  const plot = createNumberPlot({
    unit: "entry",
    logName: LOG_NAME,
    onCommit: (handle, value) => {
      const mode = widgetOf(node, "mode")?.value === "values" ? "values" : "range";
      if (mode === "range") {
        const widget = widgetOf(node, handle.key === 0 ? "start" : "end");
        if (widget && !wired(node, widget.name)) writeWidget(widget, value);
      } else {
        const widget = widgetOf(node, "values");
        if (!widget || wired(node, "values")) return;
        const text = String(widget.value ?? "");
        const entry = readEntries(text)[handle.key];
        if (!entry) return;
        writeWidget(widget, `${text.slice(0, entry.start)}${value}${text.slice(entry.end)}`);
      }
      repaint();
    },
  });

  let pending = 0;
  let released = false;

  /** Draw the list as the widgets stand on the next frame, once however often it is asked. */
  const repaint = () => {
    if (pending || released) return;
    pending = requestAnimationFrame(() => {
      pending = 0;
      try {
        plot.show(modelOf(node, lastRun, repaint));
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to draw the list:`, error);
      }
    });
  };

  /** Take the values a run output names, when this workflow's run wrote them. */
  const take = (output) => {
    const report = output?.[OUTPUT_KEY]?.[0];
    if (report && ownOutput(report) && Array.isArray(report.values)) lastRun = report.values;
  };

  /** Read the last run of this workflow back, then draw. */
  plot.refresh = () => {
    try {
      const store = app?.nodeOutputs ?? {};
      take(store[nodeLocator(node)] ?? store[executionId(node)]);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the last run:`, error);
    }
    repaint();
  };

  appendInterfaceWidget(node, plot, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
  boundTextBoxes(node);
  followMode(node, "mode", BY_MODE);
  for (const name of READ) chainWidgetCallback(node, name, repaint, LOG_NAME);

  const onExecuted = (event) => {
    try {
      const detail = event?.detail;
      const shown = String(detail?.display_node ?? detail?.node ?? "");
      if (!shown || shown !== executionId(node)) return;
      take(detail?.output);
      repaint();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the node's run:`, error);
    }
  };
  api.addEventListener("executed", onExecuted);

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    if (!graphLoading()) plot.refresh();
    return result;
  };

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    repaint();
    return result;
  };

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      released = true;
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      api.removeEventListener("executed", onExecuted);
      plot.dispose();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to release the plot:`, error);
    }
    return result;
  };

  repaint();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Number Sequence", "Plot"],
      name: "Show the sequence plot",
      tooltip:
        "Draw the list Number Sequence builds on the node, with its values dragged into place. "
        + "The widgets are always available. This applies to nodes added after the setting "
        + "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_number_sequence_wrapped) return;
    proto.__was_number_sequence_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachPlot(this);
        else followMode(this, "mode", BY_MODE);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the sequence plot:`, error);
      }
      return result;
    };
  },
});
