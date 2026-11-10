/**
 * For Loop Open drawing only the widgets its chosen mode reads.
 *
 * The widgets of the mode that is not chosen are folded away.
 */

import { app } from "../../scripts/app.js";
import { foldedByMode, followMode } from "./interface/mode_widgets.js";
import { foldedOnPlacing } from "./interface/node_preview.js";

const EXT_NAME = "WASNodeSuite.LoopMode";

const NODE = "WASForLoopOpen";
const CONTROLLER = "mode";

// Mode value -> the widgets that mode reads. `total_frames` carries its own safety ceiling,
// having no iteration count of its own to stop at.
const BY_MODE = {
  iterations: ["iterations"],
  total_frames: ["total_frames", "max_iterations"],
};

foldedOnPlacing(NODE, (definition) => foldedByMode(definition, CONTROLLER, BY_MODE));

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap the mode
    // widget's callback a second time.
    if (proto.__was_loop_mode_wrapped) return;
    proto.__was_loop_mode_wrapped = true;

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
