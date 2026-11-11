/**
 * The measurement band drawn on the mask nodes.
 *
 * Draws the coverage before and after, what was set and cleared, the connected regions, the
 * value range and the box the mask fills.
 */

import { app } from "../../scripts/app.js";
import { createMaskStatePanel } from "./interface/mask_state.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.MaskReadout";
const SETTING_ID = "WAS.Mask.ShowReadout";

// Every node that takes a mask and answers one. Mask Batch already draws a batch report, and
// Mask Rect Area and Mask Rect Area (Advanced) state their mask in their own widgets, so those
// three are left out: one node publishes one run result, and a second report would overwrite
// the first.
//
// Mask Threshold Region, Mask Fill Holes, Mask Dominant Region and Mask Minority Region also
// carry the brush in `web/was_mask_brush.js`. That is not a second report: the brush draws
// on the picture channel and publishes no run result, so the band and the brush stack rather
// than overwrite.
//
// The four segmenters at the end read a picture and answer a mask with no mask to compare
// against, which `modules/interface/mask_report.publish` takes as a before of None: the band
// then draws the coverage, the regions and the frames on their own.
const NODES = [
  "Mask Invert",
  "Mask Threshold Region",
  "Mask Floor Region",
  "Mask Ceiling Region",
  "Mask Gaussian Region",
  "Mask Smooth Region",
  "Mask Dilate Region",
  "Mask Erode Region",
  "Mask Fill Holes",
  "Mask Dominant Region",
  "Mask Minority Region",
  "Mask Arbitrary Region",
  "WASMaskFeather",
  "WASMaskGrow",
  "WASMaskGuidedFilter",
  "Masks Add",
  "Masks Subtract",
  "Masks Combine Regions",
  "Masks Combine Batch",
  "Mask Batch to Mask",
  "Mask Crop Region",
  "Mask Crop Dominant Region",
  "Mask Crop Minority Region",
  "Mask Paste Region",
  "WASImageCropByMask",
  "CLIPSeg Masking",
  "CLIPSeg Batch Masking",
  "SAM Image Mask",
  "Image to Latent Mask",
  "WASImageMatte",
];

const UI_WIDGET_NAME = "was_mask_state_ui";
const UI_WIDGET_TYPE = "was_mask_state";

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
      category: ["WAS Node Suite", "Masking", "Show what the mask operation did"],
      name: "Draw the measurement band",
      tooltip:
        "Draw the coverage before and after, what was set and cleared, the connected regions, "
        + "the value range and the box the mask fills, on Mask Invert, the region operations, "
        + "the mask arithmetic and the mask crops. The nodes run the same either way, and "
        + "nothing is measured while no browser is connected. This applies to nodes added "
        + "after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // band to every node of this type.
    if (proto.__was_mask_readout_wrapped) return;
    proto.__was_mask_readout_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createMaskStatePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the mask readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the mask readout:`, error);
      }
      return result;
    };
  },
});
