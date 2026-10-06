/**
 * What H3 Load Clip and the MiniMax H3 writing nodes did, drawn on the node.
 *
 * Draws each node's own counts as figures and its facts as rows, from its last run.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3ReportsUI";
const SETTING_ID = "WAS.H3.ShowReports";
const LOG_NAME = "WASNodeSuite.H3Reports";

// The nodes the panel is drawn on.
const NODES = ["WASH3LoadClip", "WASH3SceneWriter", "WASH3PromptRewrite", "WASH3PlanTransitions"];

// Height of the panel in node units: the summary line, three figures and three fact rows.
const PANEL_HEIGHT = 168;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 240;

// The widest name a report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 78;

const EMPTY_LABEL = "run the node to see its report";

const UI_WIDGET_NAME = "was_h3_report_ui";
const UI_WIDGET_TYPE = "was_h3_report";

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
      category: ["WAS Node Suite", "MiniMax H3", "Show clip reports"],
      name: "Draw the H3 clip report",
      tooltip:
        "Draw the scenes, frames and next index H3 Load Clip read, and what the MiniMax H3 writing "
        + "nodes wrote, on each node. The nodes run the same either way. This applies to nodes "
        + "added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_h3_report_wrapped) return;
    proto.__was_h3_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-h3-report",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the H3 clip report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the H3 clip report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the H3 clip report:`, error);
      }
      return result;
    };
  },
});
