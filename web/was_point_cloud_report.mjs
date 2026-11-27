/**
 * What a point cloud node made, drawn on the node: a strip of views and the run's figures.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.PointCloudReport";

// Each node that publishes views after a run, and the word drawn under its strip.
const NODES = {
  WASPointCloudFromDepth: "turned either side",
  WASPointCloudCameraPath: "first, middle, last",
  WASPointCloudRender: "first, middle, last",
  WASPointCloudRefine: "picture, before, after",
  WASPointCloudFromPictures: "picture, splat",
};

// Height of the panel in node units, and of the strip in CSS pixels.
const PANEL_HEIGHT = 420;
const PICTURE_HEIGHT = 170;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 300;

// The widest row name the nodes write.
const LABEL_WIDTH = 110;

const EMPTY_LABEL = "run the node to see the views";

const UI_WIDGET_NAME = "was_point_cloud_report_ui";
const UI_WIDGET_TYPE = "was_point_cloud_report";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    const label = NODES[nodeData?.name];
    if (!label) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise add a second panel.
    if (proto.__was_point_cloud_report_wrapped) return;
    proto.__was_point_cloud_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-point-cloud-report",
          height: PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the point cloud report:",
          sketch: () => createPictureBand(this, { label, height: PICTURE_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the point cloud report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the point cloud report:`, error);
      }
      return result;
    };
  },
});
