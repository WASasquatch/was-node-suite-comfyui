/**
 * The scenes drawn on Video Split Scenes.
 *
 * Draws how many scenes were found, where the cuts fall and the first frame of every scene.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.VideoSplitScenesUI";
const NODE_ID = "WASVideoSplitScenes";

// Height of the panel in node units: the summary, one row of figures, the facts and the
// contact sheet.
const PANEL_HEIGHT = 320;

// Height of the contact sheet in CSS pixels when the panel opens.
const SHEET_HEIGHT = 180;

// The widest figure name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 84;

const EMPTY_LABEL = "run the node once to see its scenes";

const UI_WIDGET_NAME = "was_video_split_scenes_ui";
const UI_WIDGET_TYPE = "was_video_split_scenes";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_video_split_scenes_wrapped) return;
    proto.__was_video_split_scenes_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-video-split-scenes",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the scenes:",
          sketch: () => createPictureBand(this, { label: "the first frame of every scene", height: SHEET_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the scenes:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the scenes:`, error);
      }
      return result;
    };
  },
});
