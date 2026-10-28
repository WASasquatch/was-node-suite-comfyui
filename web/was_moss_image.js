/**
 * The report drawn on MOSS-Image Compile Prompt: what the draft was, its caption and its boxes.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.MOSSImageUI";

// Node id -> how its report is drawn. Heights are node units.
const PANELS = {
  WASMOSSImageCompilePrompt: {
    className: "was-moss-image-compile",
    height: 340,
    labelWidth: 84,
    emptyLabel: "run the node once to see the compiled prompt",
    failure: "Failed to read the compile report:",
  },
};

const UI_WIDGET_NAME = "was_moss_image_ui";
const UI_WIDGET_TYPE = "was_moss_image";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const options = PANELS[nodeData?.name];
    if (!options) return;

    const proto = nodeType.prototype;
    if (proto.__was_moss_image_wrapped) return;
    proto.__was_moss_image_wrapped = true;

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
