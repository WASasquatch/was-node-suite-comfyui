/**
 * The value a Display Any node was handed, drawn on the node.
 *
 * The node publishes what reached it through `run_result`, and the shared report panel draws
 * it. Nothing here reads the value itself.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.DisplayAny";
const LOG_NAME = "WASNodeSuite.DisplayAny";
const SETTING_ID = "WAS.DisplayAny.ShowValue";

// The node this draws on.
const NODES = ["WASDisplayAny"];

const UI_WIDGET_NAME = "was_display_any_ui";
const UI_WIDGET_TYPE = "was_display_any";

// Tall enough for a short value and the footer under it. No maximum: the value can be any
// size, so the panel takes whatever room the node is dragged to.
const PANEL_HEIGHT = 132;
const LABEL_WIDTH = 74;

// What it says before anything has reached it.
const EMPTY_LABEL = "Run node";

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
      category: ["WAS Node Suite", "Display Any", "Show the value"],
      name: "Show the value readout",
      tooltip:
        "Draw what reached Display Any on the node itself: its type, size, range and a " +
        "rendering of the value. The node passes the value through either way. This " +
        "applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_display_any_wrapped) return;
    proto.__was_display_any_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-display-any",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the value:",
          // The value fills the panel and what it is sits under it, on one line.
          footer: true,
          tiles: false,
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the value panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the value panel:`, error);
      }
      return result;
    };
  },
});
