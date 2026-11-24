/**
 * What Video Subtitles, Video Timecode and Video Lower Third drew, shown on each node.
 *
 * Draws the run's figures and one frame of the result.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.VideoTextUI";

// Each node, the word under its picture, the line shown before it has run, and the panel's
// height in node units: the summary, a row of figures, the facts and the frame.
const NODES = {
  WASVideoSubtitles: { label: "a frame with its cue in full", empty: "run the node once to see its cues", height: 420 },
  WASVideoTimecode: { label: "the first stamped frame", empty: "run the node once to see the timecode", height: 340 },
  WASVideoLowerThird: { label: "the lower third settled", empty: "run the node once to see the lower third", height: 340 },
};

// Height of the frame in CSS pixels when the panel opens.
const PICTURE_HEIGHT = 170;

// The widest fact name the reports write, which opens the fact column.
const LABEL_WIDTH = 104;

// The narrowest the facts stay readable in.
const PANEL_MIN_WIDTH = 300;

const UI_WIDGET_NAME = "was_video_text_ui";
const UI_WIDGET_TYPE = "was_video_text";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    const spec = NODES[nodeData?.name];
    if (!spec) return;

    const proto = nodeType.prototype;
    if (proto.__was_video_text_wrapped) return;
    proto.__was_video_text_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-video-text",
          height: spec.height,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: spec.empty,
          logName: EXT_NAME,
          failure: "Failed to read the report:",
          sketch: () => createPictureBand(this, { label: spec.label, height: PICTURE_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
