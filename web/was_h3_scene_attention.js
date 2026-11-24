/**
 * The scenes H3 Scene Attention found, drawn on the node.
 *
 * The node publishes the scenes, the tokens each holds, the shared frames and the share of full
 * attention the run computed, through `run_result`.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3SceneAttentionUI";
const NODE_ID = "WASH3SceneAttention";

// Height of the panel in node units: the summary, the counts and one row per scene.
const PANEL_HEIGHT = 220;

// The widest fact name the node writes, which opens the fact column.
const LABEL_WIDTH = 78;

// The narrowest the facts stay readable in.
const PANEL_MIN_WIDTH = 300;

const EMPTY_LABEL = "run a sampler on this model to see its scenes";

const UI_WIDGET_NAME = "was_h3_scene_attention_ui";
const UI_WIDGET_TYPE = "was_h3_scene_attention";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_h3_scene_attention_wrapped) return;
    proto.__was_h3_scene_attention_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-h3-scene-attention",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the scene report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the scene panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the scene panel:`, error);
      }
      return result;
    };
  },
});
