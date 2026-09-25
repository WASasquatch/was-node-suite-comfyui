/**
 * The report drawn on Fast Generate Text.
 *
 * Draws the token count and the seconds as figures, the decode path, the prompt length and the
 * graph blocks as rows, and the text the model wrote.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.FastGenerateTextUI";
const NODE_ID = "WASFastGenerateText";

// Height of the panel in node units: the summary, the two figures, four rows and a few lines
// of the text, which scrolls.
const PANEL_HEIGHT = 220;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 96;

const EMPTY_LABEL = "run the node once to see the decode speed";

const UI_WIDGET_NAME = "was_fast_generate_text_ui";
const UI_WIDGET_TYPE = "was_fast_generate_text";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_fast_generate_text_wrapped) return;
    proto.__was_fast_generate_text_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-fast-generate-text",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the generation report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the generation report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the generation report:`, error);
      }
      return result;
    };
  },
});
