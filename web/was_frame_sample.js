/**
 * The report and the frame strip drawn on the samplers and on the frame picker.
 *
 * Draws the counts, the frame size and the strategy, with a block per frame above them and the
 * kept ones lit.
 */

import { app } from "../../scripts/app.js";
import { createFrameTimelinePanel } from "./interface/frame_timeline.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.FrameSample";
const SETTING_ID = "WAS.FrameSample.ShowTimeline";

const NODES = ["WASImageFrameSample", "WASVideoFrameSample", "Tensor Batch to Image"];

// Tensor Batch to Image is a picker rather than a sampler: it takes one frame by index and
// holds an index past the end to the last frame. It carries none of the sampler's four
// widgets, so it supplies its own pick and the strip lights the frame it will return.
const PICKERS = {
  "Tensor Batch to Image": {
    asked: (read) => ({ index: Number(read("batch_image_number", 0)) || 0 }),
    select: (frames, { index }) => [Math.min(Math.max(index, 0), frames - 1)],
  },
};

const UI_WIDGET_NAME = "was_frame_timeline_ui";
const UI_WIDGET_TYPE = "was_frame_timeline";

/**
 * Whether the strip is drawn at all.
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
      category: ["WAS Node Suite", "Frame sampling", "Show the frame strip"],
      name: "Draw the frame strip",
      tooltip:
        "Draw a block per frame on Image Frame Sample, Video Frame Sample and Tensor Batch to "
        + "Image, with the kept frames lit, plus the counts and the frame size. The strip "
        + "follows the widgets, so a strategy or an index can be judged before running. The "
        + "nodes run the same either way. This applies to nodes added after the setting "
        + "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    const id = nodeData?.name;
    if (!NODES.includes(id)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_frame_sample_wrapped) return;
    proto.__was_frame_sample_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createFrameTimelinePanel(this, PICKERS[id] ?? {});
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the frame strip:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the frame strip:`, error);
      }
      return result;
    };
  },
});
