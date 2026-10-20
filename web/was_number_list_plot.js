/**
 * The lists Number List Plot was handed, drawn on the node.
 *
 * Each series is a line in its own colour, with its length and spread in the legend, and the
 * entry under the pointer read out below.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ICON } from "./interface/icons.js";
import { SERIES_TOKENS, createNumberPlot } from "./interface/number_plot.js";
import { PREVIEW_STATE, executionId, nodeLocator, ownOutput } from "./interface/preview.js";
import { RUN_LABELS } from "./interface/run_result.js";
import { appendInterfaceWidget, graphLoading } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.NumberListPlotUI";
const LOG_NAME = "WASNodeSuite.NumberListPlot";
const NODE_ID = "WASNumberListPlot";
const SETTING_ID = "WAS.NumberListPlot.ShowPlot";

const UI_WIDGET_NAME = "was_number_list_plot_ui";
const UI_WIDGET_TYPE = "was_number_plot";

// The key the node's run output carries its series under.
const OUTPUT_KEY = "was_number_plot";

// The size a freshly placed node opens at, in node units.
const OPEN_SIZE = [300, 250];

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
 * What the plot shows for one run report.
 *
 * @param {object|null} report - The node's last run output, or null before it has run.
 * @returns {object} A model for `createNumberPlot`'s `show`.
 */
function modelOf(report) {
  const series = Array.isArray(report?.series) ? report.series : [];
  if (!series.length) {
    return {
      empty: RUN_LABELS[PREVIEW_STATE.WAITING],
      summary: "",
      claim: { icon: ICON.WARNING, detail: "nothing has been drawn: queue the graph" },
    };
  }
  const drawn = series.map((entry, position) => ({
    name: entry.name,
    values: Array.isArray(entry.values) ? entry.values : [],
    total: Number(entry.total) || 0,
    skipped: Number(entry.skipped) || 0,
    token: SERIES_TOKENS[position % SERIES_TOKENS.length],
  }));
  const legend = drawn.map((entry) => {
    const [low, high] = extent(entry.values);
    const spread = entry.values.length ? (low === high ? `all ${low}` : `${low} to ${high}`) : "empty";
    const notes = [
      entry.skipped ? `${entry.skipped} not numbers` : "",
      entry.total > entry.values.length ? `first ${entry.values.length} drawn` : "",
    ].filter(Boolean);
    return {
      text: `${entry.name}: ${entry.total}, ${spread}${notes.length ? ` (${notes.join(", ")})` : ""}`,
      token: entry.token,
    };
  });
  const cut = drawn.filter((entry) => entry.total > entry.values.length);
  const longest = Math.max(...drawn.map((entry) => entry.total));
  return {
    series: drawn,
    legend,
    summary: drawn.length === 1
      ? `${longest} entr${longest === 1 ? "y" : "ies"}`
      : `${drawn.length} lists, up to ${longest} entries`,
    claim: cut.length
      ? {
        icon: ICON.APPROXIMATE,
        detail: cut.map((entry) => `${entry.name}: the first ${entry.values.length} of ${entry.total}`)
          .join("; "),
      }
      : { icon: ICON.EXACT, detail: "the numbers the node was handed on its last run" },
  };
}

/**
 * Append the plot to a node and keep it on the node's last run.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachPlot(node) {
  let report = null;
  const plot = createNumberPlot({ unit: "entry", logName: LOG_NAME });

  /** Draw the last run. */
  const repaint = () => {
    try {
      plot.show(modelOf(report));
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to draw the lists:`, error);
    }
  };

  /** Take the report a run output names, when this workflow's run wrote it. */
  const take = (output) => {
    const entry = output?.[OUTPUT_KEY]?.[0];
    if (entry && ownOutput(entry)) report = entry;
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
  // A saved node takes its own size back when it is configured.
  node.size = [Math.max(node.size[0], OPEN_SIZE[0]), Math.max(node.size[1], OPEN_SIZE[1])];

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

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
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
      category: ["WAS Node Suite", "Number List Plot", "Plot"],
      name: "Show the list plot",
      tooltip:
        "Draw the lists Number List Plot was handed on the node. This applies to nodes added "
        + "after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_number_list_plot_wrapped) return;
    proto.__was_number_list_plot_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachPlot(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the list plot:`, error);
      }
      return result;
    };
  },
});
