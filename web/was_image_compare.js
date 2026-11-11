/**
 * The comparison panel on Image Compare (Advanced).
 *
 * Draws one image over the other under a divider, with a tab per pair.
 */

import { app } from "../../scripts/app.js";
import { createImageComparePanel } from "./interface/image_compare.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ImageCompare";
const SETTING_ID = "WAS.ImageCompare.ShowInterface";

const NODE = "WASImageCompare";

const UI_WIDGET_NAME = "was_image_compare_ui";
const UI_WIDGET_TYPE = "was_image_compare";

/**
 * Whether the panel is drawn at all.
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
      category: ["WAS Node Suite", "Image Compare", "Show the comparison"],
      name: "Draw the comparison on Image Compare (Advanced)",
      tooltip:
        "Draw the two images under a divider on the node, with a tab per pair for a batch. The "
        + "node runs the same either way. This applies to nodes added after the setting changes, "
        + "so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_image_compare_wrapped) return;
    proto.__was_image_compare_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createImageComparePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the comparison:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the comparison:`, error);
      }
      return result;
    };
  },
});
