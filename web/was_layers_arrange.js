/**
 * The layer list drawn on Layers Arrange.
 *
 * Lists the stack the node last held, front first, and writes every move, restack and
 * visibility change into the node's own arrangement widget.
 */

import { app } from "../../scripts/app.js";
import { createLayerArrangePanel } from "./interface/layer_arrange.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LayerArrangeUI";
const SETTING_ID = "WAS.Layers.ShowArrange";

const NODES = ["WASLayersArrange"];

const UI_WIDGET_NAME = "was_layer_arrange_ui";
const UI_WIDGET_TYPE = "was_layer_arrange";

/**
 * Whether the panel is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Layers", "Arrange panel"],
      name: "Show the layer arrange panel",
      tooltip:
        "Draw the stack Layers Arrange last held under the node, one row per layer with its "
        + "thumbnail, a visibility toggle and a handle to restack it. The arrangement widget "
        + "is the whole of what the node reads either way. This applies to nodes added after "
        + "the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_layer_arrange_wrapped) return;
    proto.__was_layer_arrange_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createLayerArrangePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        // The arrangement box is a growable widget too, so it is capped rather than left to
        // take the room the panel asked for.
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the layer list:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the layer list:`, error);
      }
      return result;
    };
  },
});
