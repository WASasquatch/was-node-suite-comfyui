/**
 * CNS Model Patch drawing its tuning widgets only in `manual`, the one mode that reads them.
 */

import { app } from "../../scripts/app.js";
import { followMode } from "./interface/mode_widgets.js";

const EXT_NAME = "WASNodeSuite.CNSMode";

const NODE = "WASCNSModelPatch";
const CONTROLLER = "mode";

// Mode value -> the widgets that mode reads.
const BY_MODE = {
  auto: [],
  manual: ["bands", "divider", "power", "tilt_start", "tilt_end", "sharpness", "energy"],
};

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap the mode
    // widget's callback a second time.
    if (proto.__was_cns_mode_wrapped) return;
    proto.__was_cns_mode_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        followMode(this, CONTROLLER, BY_MODE);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to follow ${NODE}'s mode:`, error);
      }
      return result;
    };
  },
});
