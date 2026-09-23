/**
 * The report drawn on the batching nodes.
 *
 * Draws the frame count, the frame size, the mode and the memory on the node itself.
 */

import { app } from "../../scripts/app.js";
import { createBatchStatePanel } from "./interface/batch_state.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.BatchReadout";
const SETTING_ID = "WAS.Batch.ShowState";

const NODES = [
  "WASImageBatchAdvanced",
  "Image Batch",
  "Mask Batch",
  "Latent Batch",
  "WASLoadImageSequence",
  "WASLayersToImageBatch",
  "WASEMAVFIFrameInterpolation",
  "WASNSCameraMotionTrajectory",
];

const UI_WIDGET_NAME = "was_batch_state_ui";
const UI_WIDGET_TYPE = "was_batch_state";

/**
 * Whether the readout is drawn at all.
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
      category: ["WAS Node Suite", "Batching", "Show what was batched"],
      name: "Draw the batch report",
      tooltip:
        "Draw the frame count, the frame size, the channel mode and the memory the batch costs "
        + "on the image batchers, Mask Batch and Latent Batch, and name the slot when sizes "
        + "disagree. "
        + "The nodes run the same either way. This applies to nodes added after the setting "
        + "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_batch_readout_wrapped) return;
    proto.__was_batch_readout_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createBatchStatePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the batch readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the batch readout:`, error);
      }
      return result;
    };
  },
});
