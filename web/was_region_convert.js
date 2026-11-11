/**
 * How many regions a conversion carried across, drawn on the node.
 *
 * The node publishes the count and the first rectangle through `run_result`, and the shared
 * report panel draws them.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.RegionConvert";
const LOG_NAME = "WASNodeSuite.RegionConvert";
const SETTING_ID = "WAS.RegionConvert.ShowCount";

// The nodes this draws on.
const NODES = [
  "WASBoundingBoxesToBounds",
  "WASBoundsToBoundingBoxes",
  "WASBoundingBoxesFilter",
];

// Tall enough for the summary, the count tile and the one fact row under it.
const PANEL_HEIGHT = 104;
const LABEL_WIDTH = 60;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 230;

const EMPTY_LABEL = "run the node to see the regions";

const UI_WIDGET_NAME = "was_region_convert_ui";
const UI_WIDGET_TYPE = "was_region_convert";

/**
 * Read whether the panel is drawn at all.
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

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Bounds", "Show the region count"],
      name: "Show the converted region count",
      tooltip:
        "Draw how many regions a bounds conversion carried across, and the first rectangle " +
        "it produced, on the node itself. The node converts either way. This applies to " +
        "nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_region_convert_wrapped) return;
    proto.__was_region_convert_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-region-convert",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the region count:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the region panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the region panel:`, error);
      }
      return result;
    };
  },
});
