/**
 * What Quantize Model did, drawn on the node.
 *
 * The node publishes its layer counts per format, the size before and after, and the layers
 * with the largest error through `run_result`.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.QuantizeModelUI";
const NODE_ID = "WASQuantizeModel";

// Height of the panel in node units: the summary, the counts, the facts and the error list.
const PANEL_HEIGHT = 440;

// The widest fact name the node writes, which opens the fact column.
const LABEL_WIDTH = 64;

// The narrowest the summary stays readable in.
const PANEL_MIN_WIDTH = 300;

const EMPTY_LABEL = "run the node to see what was quantized";

const UI_WIDGET_NAME = "was_quantize_model_ui";
const UI_WIDGET_TYPE = "was_quantize_model";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_quantize_model_wrapped) return;
    proto.__was_quantize_model_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-quantize-model",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the quantize report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the quantize panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the quantize panel:`, error);
      }
      return result;
    };
  },
});
