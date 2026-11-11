/**
 * The live state readout on For Loop Close and While Loop Close.
 *
 * Shows the iteration, the limit, the frames collected and what each slot holds, updated as
 * each iteration finishes.
 */

import { app } from "../../scripts/app.js";
import { createLoopStatePanel } from "./interface/loop_state.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LoopReadout";
const SETTING_ID = "WAS.Loop.ShowState";

const NODES = ["WASForLoopClose", "WASWhileLoopClose"];

const UI_WIDGET_NAME = "was_loop_state_ui";
const UI_WIDGET_TYPE = "was_loop_state";

/**
 * Whether the readout is drawn at all.
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
      category: ["WAS Node Suite", "Loops", "Live state"],
      name: "Show the loop state readout",
      tooltip:
        "Draw the iteration, the limit and what each slot holds under For Loop Close and While "
        + "Loop Close, updated as each iteration finishes. The loop runs the same either way. This "
        + "applies to nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_loop_readout_wrapped) return;
    proto.__was_loop_readout_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createLoopStatePanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the loop readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the loop readout:`, error);
      }
      return result;
    };
  },
});
