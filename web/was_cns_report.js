/**
 * What CNS Model Patch did in the last sampling run, drawn on the node.
 *
 * The sampler files the draws it coloured and the profile it used through `run_result`, or a
 * warning when it adds no noise.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.CNSReportUI";
const NODE_ID = "WASCNSModelPatch";

// Height of the panel in node units: the summary and four facts.
const PANEL_HEIGHT = 180;

// The widest fact name the sampler writes, which opens the fact column.
const LABEL_WIDTH = 72;

// The narrowest the sampler advice stays readable in.
const PANEL_MIN_WIDTH = 320;

const EMPTY_LABEL = "sample with the model to see what was coloured";

const UI_WIDGET_NAME = "was_cns_report_ui";
const UI_WIDGET_TYPE = "was_cns_report";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_cns_report_wrapped) return;
    proto.__was_cns_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-cns-report",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the CNS report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the CNS panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the CNS panel:`, error);
      }
      return result;
    };
  },
});
