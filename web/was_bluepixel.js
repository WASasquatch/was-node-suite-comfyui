/**
 * The reports drawn on the BluePixel loaders and scheduler.
 *
 * The model loader shows the file and how its weights are stored, the VAE loader the
 * autoencoder it built, and the scheduler the shift and the sigmas.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.BluePixelUI";

// Node id -> how its report is drawn. Heights are node units.
const PANELS = {
  WASBluePixelModelLoader: {
    className: "was-bluepixel-model",
    height: 170,
    labelWidth: 96,
    emptyLabel: "run to load the model",
    failure: "Failed to read the model report:",
  },
  WASBluePixelVAELoader: {
    className: "was-bluepixel-vae",
    height: 160,
    labelWidth: 96,
    emptyLabel: "run to load the autoencoder",
    failure: "Failed to read the autoencoder report:",
  },
  WASBluePixelScheduler: {
    className: "was-bluepixel-schedule",
    height: 130,
    labelWidth: 64,
    emptyLabel: "run to see the sigmas",
    failure: "Failed to read the schedule report:",
  },
};

const UI_WIDGET_NAME = "was_bluepixel_ui";
const UI_WIDGET_TYPE = "was_bluepixel";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const options = PANELS[nodeData?.name];
    if (!options) return;

    const proto = nodeType.prototype;
    // One panel per node type, however often the definitions are registered.
    if (proto.__was_bluepixel_wrapped) return;
    proto.__was_bluepixel_wrapped = true;

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
