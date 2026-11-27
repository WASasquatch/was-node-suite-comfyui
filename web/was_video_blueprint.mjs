/**
 * The shots drawn on Video Blueprint.
 *
 * Draws how many shots were found, each shot's measured camera move and the first frame of every shot.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.VideoBlueprintUI";
const NODE_ID = "WASVideoBlueprint";

// Height of the panel in node units: the summary, one row of figures, the shot list and the
// contact sheet.
const PANEL_HEIGHT = 380;

// Height of the contact sheet in CSS pixels when the panel opens.
const SHEET_HEIGHT = 180;

// The widest figure name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 84;

const EMPTY_LABEL = "run the node once to see its shots";

const UI_WIDGET_NAME = "was_video_blueprint_ui";
const UI_WIDGET_TYPE = "was_video_blueprint";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_video_blueprint_wrapped) return;
    proto.__was_video_blueprint_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-video-blueprint",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the shots:",
          sketch: () => createPictureBand(this, { label: "the first frame of every shot", height: SHEET_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the shots:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the shots:`, error);
      }
      return result;
    };
  },
});
