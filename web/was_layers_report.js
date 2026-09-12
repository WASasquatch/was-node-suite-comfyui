/**
 * What an edit did to a layer stack, drawn on the node.
 *
 * Every node in the Layers family publishes its canvas, its layer count and what it changed
 * through `run_result`, and the shared report panel draws them.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LayersReport";
const LOG_NAME = "WASNodeSuite.LayersReport";
const SETTING_ID = "WAS.Layers.ShowStackReport";

// The nodes this draws on.
const NODES = [
  "WASLayerAlign",
  "WASLayerBevel",
  "WASLayerEdit",
  "WASLayerFit",
  "WASLayerGlow",
  "WASLayerMask",
  "WASLayerOrder",
  "WASLayerOverlay",
  "WASLayerRemove",
  "WASLayerReplaceImage",
  "WASLayerSelect",
  "WASLayerShadow",
  "WASLayerStroke",
  "WASLayerTrim",
  "WASLayersCanvas",
  "WASLayersFromImageBatch",
  "WASLayersLoad",
  "WASLayersMerge",
];

// Tall enough for the summary, a row of count tiles and two fact rows under them.
const PANEL_HEIGHT = 132;
const LABEL_WIDTH = 74;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 250;

const EMPTY_LABEL = "run the node to see the stack";

const UI_WIDGET_NAME = "was_layers_report_ui";
const UI_WIDGET_TYPE = "was_layers_report";

/**
 * Read whether the panel is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Layers", "Show what an edit did"],
      name: "Show the layer stack report",
      tooltip:
        "Draw the canvas size, the layer count and what the edit changed on the node itself. " +
        "Covers the stack edits and the layer effects. This applies to nodes added after the " +
        "setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_layers_report_wrapped) return;
    proto.__was_layers_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-layers-report",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the layer stack report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the layer panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the layer panel:`, error);
      }
      return result;
    };
  },
});
