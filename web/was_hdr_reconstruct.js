/**
 * What HDR Reconstruct recovered, drawn on the node.
 *
 * Draws the peak value, the share of the frame above one and the frame count as figures,
 * and the size and what the network was fed as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.HDRReconstructUI";
const SETTING_ID = "WAS.HDR.ShowRecovered";
const LOG_NAME = "WASNodeSuite.HDRReconstruct";

const NODE_ID = "WASHDRReconstruct";

// Height of the panel in node units: the summary line, the three figures and the two fact
// rows, with nothing scrolling.
const PANEL_HEIGHT = 116;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 240;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 40;

const EMPTY_LABEL = "run the node to see the headroom";

const UI_WIDGET_NAME = "was_hdr_reconstruct_ui";
const UI_WIDGET_TYPE = "was_hdr_reconstruct";

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
      category: ["WAS Node Suite", "HDR", "Show what was recovered"],
      name: "Draw the reconstruction report",
      tooltip:
        "Draw the peak value, the share of the frame above one, the frame count, the size "
        + "and whether the frames were dequantised on HDR Reconstruct. A frame that came "
        + "back with nothing above one is drawn in the warning colour, which is the reading "
        + "that says the picture held no clipped highlight. The node runs the same either "
        + "way. This applies to nodes added after the setting changes, so a reload shows it "
        + "everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_hdr_reconstruct_wrapped) return;
    proto.__was_hdr_reconstruct_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-hdr-reconstruct",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the reconstruction report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the reconstruction report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the reconstruction report:`, error);
      }
      return result;
    };
  },
});
