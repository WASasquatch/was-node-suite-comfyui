/**
 * What Image Skin Mask and Image Simple Skin Mask found, drawn on the node.
 *
 * The node publishes its people, faces and coverage through `run_result`, and the first
 * picture with its skin tinted, which the band draws.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ImageSkinMaskUI";
const NODE_IDS = new Set(["WASImageSkinMask", "WASImageSimpleSkinMask"]);

// Height of the panel in node units: the summary, the counts, the facts and the picture.
const PANEL_HEIGHT = 300;

// Height of the picture in CSS pixels when the panel opens.
const PICTURE_HEIGHT = 180;

// The widest fact name the node writes, which opens the fact column.
const LABEL_WIDTH = 52;

// The narrowest the summary stays readable in.
const PANEL_MIN_WIDTH = 260;

const EMPTY_LABEL = "run the node to see the skin it found";

const UI_WIDGET_NAME = "was_image_skin_mask_ui";
const UI_WIDGET_TYPE = "was_image_skin_mask";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData?.name)) return;

    const proto = nodeType.prototype;
    if (proto.__was_image_skin_mask_wrapped) return;
    proto.__was_image_skin_mask_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-image-skin-mask",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the skin report:",
          sketch: () => createPictureBand(this, { label: "skin", height: PICTURE_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the skin panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the skin panel:`, error);
      }
      return result;
    };
  },
});
