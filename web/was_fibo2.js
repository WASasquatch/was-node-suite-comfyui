/**
 * The reports drawn on Text Encode Fibo 2 and Generate Fibo 2 Prompt.
 *
 * The encoder shows its mode, size and token counts; the writer shows its task, decode speed
 * and caption.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.Fibo2UI";

// Node id -> how its report is drawn. Heights are node units.
const PANELS = {
  WASTextEncodeFibo2: {
    className: "was-fibo2-encode",
    height: 150,
    labelWidth: 96,
    emptyLabel: "run the node once to see what it encoded",
    failure: "Failed to read the encoding report:",
  },
  WASGenerateFibo2Prompt: {
    className: "was-fibo2-prompt",
    height: 260,
    labelWidth: 72,
    emptyLabel: "run the node once to see the caption",
    failure: "Failed to read the caption report:",
  },
};

const UI_WIDGET_NAME = "was_fibo2_ui";
const UI_WIDGET_TYPE = "was_fibo2";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const options = PANELS[nodeData?.name];
    if (!options) return;

    const proto = nodeType.prototype;
    if (proto.__was_fibo2_wrapped) return;
    proto.__was_fibo2_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, { ...options, logName: EXT_NAME });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the report:`, error);
      }
      return result;
    };
  },
});
