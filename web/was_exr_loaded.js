/**
 * What EXR Load read off disk, drawn on the node.
 *
 * Draws the peak value and the share of the frame above one as figures, and the size, the
 * channels, the compression, the depth and the file name as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.EXRLoadedUI";
const SETTING_ID = "WAS.HDR.ShowLoaded";
const LOG_NAME = "WASNodeSuite.EXRLoaded";

const NODE_ID = "WASEXRLoad";

// Height of the panel in node units: the summary line, the two figures and the five fact
// rows, with nothing scrolling.
const PANEL_HEIGHT = 168;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 240;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 78;

const EMPTY_LABEL = "run the node to see what the file holds";

const UI_WIDGET_NAME = "was_exr_loaded_ui";
const UI_WIDGET_TYPE = "was_exr_loaded";

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
      category: ["WAS Node Suite", "HDR", "Show what was read"],
      name: "Draw the EXR reading report",
      tooltip:
        "Draw the peak value, the share of the frame above one, the frame size, the channel "
        + "names, the compression, the bit depth and the file name on EXR Load. The channel "
        + "row is the reading that says whether the file carried an alpha, and the peak is "
        + "the one that says whether it holds light above white. The node runs the same "
        + "either way. This applies to nodes added after the setting changes, so a reload "
        + "shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_exr_loaded_wrapped) return;
    proto.__was_exr_loaded_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-exr-loaded",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the EXR reading report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the EXR reading report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the EXR reading report:`, error);
      }
      return result;
    };
  },
});
