/**
 * What H3 PTD Patch applied, and the steps H3 PTD Sigmas read, drawn on the node.
 *
 * The nodes publish the release, the step count, the trained sigmas, the sampler settings and
 * what was checked about the transformer through `run_result`.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3PTDPatchUI";
// The nodes the panel is drawn on.
const NODE_IDS = new Set(["WASH3PTDPatch", "WASH3PTDSigmas"]);

// Height of the panel in node units: the summary and the facts.
const PANEL_HEIGHT = 230;

// The widest fact name the node writes, which opens the fact column.
const LABEL_WIDTH = 64;

// The narrowest the facts stay readable in.
const PANEL_MIN_WIDTH = 320;

const EMPTY_LABEL = "run the node to see the steps and sampler settings";

const UI_WIDGET_NAME = "was_h3_ptd_patch_ui";
const UI_WIDGET_TYPE = "was_h3_ptd_patch";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData?.name)) return;

    const proto = nodeType.prototype;
    if (proto.__was_h3_ptd_patch_wrapped) return;
    proto.__was_h3_ptd_patch_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-h3-ptd-patch",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the PTD report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the PTD panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the PTD panel:`, error);
      }
      return result;
    };
  },
});
