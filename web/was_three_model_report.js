/**
 * What a loaded model holds, drawn on Three Load Model.
 *
 * Draws the triangle, mesh, material, texture, clip and bone counts as figures, and the file,
 * the size, the clip names, the compression and the extensions as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ThreeModelReport";
const SETTING_ID = "WAS.Three.ShowModelReport";
const LOG_NAME = "WASNodeSuite.ThreeModelReport";

const NODES = ["WASThreeLoadModel"];

// Tall enough for the summary, the figures and six fact rows; longer reports scroll.
const PANEL_HEIGHT = 196;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 280;

// The widest fact name the report writes, "largest texture".
const LABEL_WIDTH = 96;

const EMPTY_LABEL = "run the node to see what the model holds";

const UI_WIDGET_NAME = "was_three_model_report_ui";
const UI_WIDGET_TYPE = "was_three_model_report";

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

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Three", "Show what a model holds"],
      name: "Draw the model report",
      tooltip:
        "Draw the triangle, texture, clip and bone counts, the size, the clip names and the "
        + "compression of the model Three Load Model read. The node runs the same either way. "
        + "This applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_three_model_report_wrapped) return;
    proto.__was_three_model_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-three-model-report",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the model report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the model report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the model report:`, error);
      }
      return result;
    };
  },
});
