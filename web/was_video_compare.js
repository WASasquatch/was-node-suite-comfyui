/**
 * The two-video player drawn on Compare Video.
 *
 * Both sides are written to the temp folder by the node and played from one clock, split by
 * a divider that drags across.
 */

import { app } from "../../scripts/app.js";
import { createVideoComparePanel } from "./interface/video_compare.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.VideoCompare";
const SETTING_ID = "WAS.Animation.ShowVideoCompare";

const NODES = ["WASVideoCompare"];

const UI_WIDGET_NAME = "was_video_compare_ui";
const UI_WIDGET_TYPE = "was_video_compare";

/**
 * Whether the player is drawn at all.
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
      category: ["WAS Node Suite", "Animation", "Show the video comparison"],
      name: "Draw the two-video player",
      tooltip:
        "Draw both videos on Compare Video, one over the other, split by a divider that drags "
        + "left and right. Both play from one clock, so the same frame is shown on each side. "
        + "Off, the node writes both clips and draws nothing. This applies to nodes added after "
        + "the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // player to every node of this type.
    if (proto.__was_video_compare_wrapped) return;
    proto.__was_video_compare_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createVideoComparePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the player:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the player:`, error);
      }
      return result;
    };
  },
});
