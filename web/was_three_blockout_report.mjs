/**
 * The report drawn on Three Blockout Scene: what it built and its shot list.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.ThreeBlockoutReport";

const PANEL = {
  className: "was-three-blockout-report",
  height: 200,
  labelWidth: 70,
  emptyLabel: "run to build the scene",
  failure: "Failed to read the scene report:",
};

const UI_WIDGET_NAME = "was_three_blockout_ui";
const UI_WIDGET_TYPE = "was_three_blockout";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "WASThreeBlockoutScene") return;
    const proto = nodeType.prototype;
    // One panel per node type, however often the definitions are registered.
    if (proto.__was_three_blockout_wrapped) return;
    proto.__was_three_blockout_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, { ...PANEL, logName: EXT_NAME });
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
