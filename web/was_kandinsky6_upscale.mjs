/**
 * The reports drawn on the Kandinsky 6 upscale loaders: what each one built.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.Kandinsky6UpscaleUI";

// Node id -> how its report is drawn. Heights are node units.
const PANELS = {
  WASKandinsky6UpscaleModelLoader: {
    className: "was-k6-upscale-model",
    height: 170,
    labelWidth: 104,
    emptyLabel: "run to load the model",
    failure: "Failed to read the model report:",
  },
  WASKandinsky6UpscaleVAELoader: {
    className: "was-k6-upscale-vae",
    height: 140,
    labelWidth: 64,
    emptyLabel: "run to load the autoencoder",
    failure: "Failed to read the autoencoder report:",
  },
};

const UI_WIDGET_NAME = "was_k6_upscale_ui";
const UI_WIDGET_TYPE = "was_k6_upscale";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    const options = PANELS[nodeData?.name];
    if (!options) return;

    const proto = nodeType.prototype;
    // One panel per node type, however often the definitions are registered.
    if (proto.__was_k6_upscale_wrapped) return;
    proto.__was_k6_upscale_wrapped = true;

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
