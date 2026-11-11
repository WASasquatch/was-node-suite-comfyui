/**
 * The curve editor for the Image Curves node.
 *
 * Drag a point to bend the channel, click empty grid to add one, right-click a point to drop it.
 * The plot writes `curve_points` over the input picture's levels.
 */

import { app } from "../../scripts/app.js";
import { mountCurveEditor } from "./interface/curve_editor.js";

const EXT_NAME = "WASNodeSuite.ImageCurvesUI";
const LOG_NAME = "WASNodeSuite.ImageCurves";
const NODE_NAME = "WASImageCurves";
const SETTING_ID = "WAS.ImageCurves.ShowInterface";

const POINTS_WIDGET = "curve_points";

/**
 * Read whether the editor is drawn at all.
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

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Curves", "Curve editor"],
      name: "Show the curve editor",
      tooltip:
        "Draw the curve editor under the curve_points widget of Image Curves. The widget " +
        "itself is always available. This applies to nodes added after the setting changes, " +
        "so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;
    if (proto.__was_curves_wrapped) return;
    proto.__was_curves_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) {
          mountCurveEditor(this, {
            pointsWidget: POINTS_WIDGET,
            histogram: true,
            logName: LOG_NAME,
          });
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the curve editor:`, error);
      }
      return result;
    };
  },
});
