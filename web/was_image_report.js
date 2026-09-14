/**
 * What a picture holds, drawn on the nodes that read or show one.
 *
 * Draws the mean, the contrast and the clipped share as figures, and the size, the range and
 * how far the picture moved as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ImageReportUI";
const SETTING_ID = "WAS.Pixels.ShowImageReport";
const LOG_NAME = "WASNodeSuite.ImageReport";

// The nodes that file a picture's measurements. Image Load reports what it read and how far
// the tensor moved from the file; Image Preview reports what it drew and how far the view is
// from the numbers; HDR VAE Decode reports the peak and the share above white.
const NODES = [
  "Image Load",
  "Load Image Batch",
  "WASImagePreview",
  "WASHDRVAEDecode",
];

// Node id -> the widget a run's own reading is written back onto, and the fact it is read
// from. `incremental_image` walks the folder on its own, so the index widget is the only
// place a user can see where that walk has got to.
const READBACK = {
  "Load Image Batch": { widget: "index", fact: "at", onlyWhen: { mode: "incremental_image" } },
};

// Height of the panel in node units: the summary line, the three figures and the eight fact
// rows, with nothing scrolling.
const PANEL_HEIGHT = 236;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 250;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 74;

const EMPTY_LABEL = "run the node to see what the picture holds";

const UI_WIDGET_NAME = "was_image_report_ui";
const UI_WIDGET_TYPE = "was_image_report";

/**
 * Whether the report is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * Put a run's own reading back on the widget that shows it.
 *
 * @param {object} node - The node the report was drawn for.
 * @param {object|null} report - The report `run_result` answered with, already unwrapped.
 * @returns {void}
 */
function readBack(node, report) {
  const rule = READBACK[node?.constructor?.comfyClass ?? node?.type];
  if (!rule) return;
  const widgets = Array.isArray(node.widgets) ? node.widgets : [];
  for (const [name, wanted] of Object.entries(rule.onlyWhen ?? {})) {
    if (widgets.find((candidate) => candidate?.name === name)?.value !== wanted) return;
  }
  const facts = report?.facts ?? [];
  const fact = facts.find((entry) => entry?.name === rule.fact);
  const found = String(fact?.value ?? "").match(/\d+/);
  if (!found) return;
  const widget = widgets.find((candidate) => candidate?.name === rule.widget);
  if (!widget) return;
  // Shown, not chosen: the value is what the last run read, so writing it through the
  // widget's callback would look like the user reaching for it.
  const shown = Number(found[0]);
  if (widget.value !== shown) {
    widget.value = shown;
    node.setDirtyCanvas?.(true, false);
  }
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Pixels", "Show what a picture holds"],
      name: "Draw the picture report",
      tooltip:
        "Draw the mean, the contrast, the clipped share, the size, the channels, the range "
        + "and the entropy on Image Load and Image Preview. Image Load also reports how far "
        + "its tensor moved from the file it read, per channel and at worst, which is what a "
        + "colour profile conversion shows up as; a picture that moved with no conversion "
        + "behind it is drawn in the warning colour. The nodes run the same either way. This "
        + "applies to nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const id = nodeData?.name;
    if (!NODES.includes(id)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_image_report_wrapped) return;
    proto.__was_image_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-image-report",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the picture report:",
          onAnswer: (report) => readBack(this, report),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the picture report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the picture report:`, error);
      }
      return result;
    };
  },
});
