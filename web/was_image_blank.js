/**
 * The colour wheel on Image Blank.
 *
 * Dragging the ring or the square writes `red`, `green` and `blue`, and typing a level moves
 * the wheel with it.
 */

import { app } from "../../scripts/app.js";
import { createColourWheel } from "./interface/colour_wheel.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ImageBlankUI";
const SETTING_ID = "WAS.ImageBlank.ShowColourWheel";

const UI_WIDGET_NAME = "was_colour_wheel_ui";
const UI_WIDGET_TYPE = "was_colour_wheel";

const NODE_ID = "Image Blank";

// The widgets the node holds its fill colour in, red first.
const CHANNELS = ["red", "green", "blue"];

/**
 * Read whether the wheel is drawn at all.
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

/**
 * Put a wheel on one node and wire its teardown.
 *
 * @param {object} node - The node being created.
 * @param {string[]} channels - The three level widgets, red first.
 * @returns {void}
 */
function attachWheel(node, channels) {
  const wheel = createColourWheel(node, { channels });
  appendInterfaceWidget(node, wheel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  // Read and repaint only.
  for (const name of channels) {
    chainWidgetCallback(node, name, () => wheel.refresh(), EXT_NAME);
  }

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    try {
      wheel.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to dispose the colour wheel:`, error);
    }
    return originalOnRemoved?.apply(this, args);
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Image Blank", "Colour wheel"],
      name: "Show the colour wheel",
      tooltip:
        "Draw a colour wheel on Image Blank. The red, green and blue widgets are always "
        + "available. This applies to nodes added after the setting changes, so a reload "
        + "shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;

    if (proto.__was_colour_wheel_wrapped) return;
    proto.__was_colour_wheel_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (enabled()) attachWheel(this, CHANNELS);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the colour wheel:`, error);
      }
      return result;
    };
  },
});
