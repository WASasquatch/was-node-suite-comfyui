/**
 * What Adaptive Detail did in the last sampling run, drawn on the node.
 *
 * The sampler files the steps it pushed toward detail, the steps it refined with an extra pass
 * and the time it took through `run_result`.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.AdaptiveDetailUI";
const NODE_ID = "WASAdaptiveDetail";

// Height of the panel in node units: the summary and five facts.
const PANEL_HEIGHT = 230;

// The widest fact name the sampler writes, which opens the fact column.
const LABEL_WIDTH = 92;

// The narrowest a 50-step pattern stays readable in.
const PANEL_MIN_WIDTH = 340;

const EMPTY_LABEL = "sample with the model to see which steps were refined";

const UI_WIDGET_NAME = "was_adaptive_detail_ui";
const UI_WIDGET_TYPE = "was_adaptive_detail";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_adaptive_detail_wrapped) return;
    proto.__was_adaptive_detail_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-adaptive-detail",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the Adaptive Detail report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the Adaptive Detail panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the Adaptive Detail panel:`, error);
      }
      return result;
    };
  },
});
