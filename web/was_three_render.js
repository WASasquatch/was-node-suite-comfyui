/**
 * The window Three Render captures, set by dragging a strip on the node.
 *
 * The strip reads and writes the node's own `start`, `num_frames` and `fps` widgets, and takes
 * the length of its axis from the Three App wired into it.
 */

import { app } from "../../scripts/app.js";
import { createTimeSpanPanel } from "./interface/time_span.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ThreeRenderTimeline";
const SETTING_ID = "WAS.ThreeRender.ShowTimeline";

const NODE_ID = "WASThreeRender";
const APP_NODE_ID = "WASThreeApp";

const START = "start";
const FRAMES = "num_frames";
const FPS = "fps";
const LOOP_SECONDS = "loop_seconds";

const UI_WIDGET_NAME = "was_three_render_ui";
const UI_WIDGET_TYPE = "was_time_span";

// What the axis runs to with no Three App to ask, in seconds.
const FALLBACK_LENGTH = 4;

// Room the strip and its two lines take, in node units.
const PANEL_HEIGHT = 88;

// The narrowest the strip is worth drawing in, in node units.
const PANEL_MIN_WIDTH = 260;

/**
 * Whether the strip is drawn.
 *
 * @returns {boolean} The setting, true where it has never been set.
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

/**
 * One widget's value.
 *
 * @param {object} node - The node holding it.
 * @param {string} name - The widget's name.
 * @param {number} fallback - Answered where the widget is absent or unreadable.
 * @returns {number} The value.
 */
function widgetValue(node, name, fallback) {
  const widget = node?.widgets?.find((one) => one.name === name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * How long the app wired into a render loops for.
 *
 * @param {object} node - The Three Render node.
 * @returns {number} The seconds, or 0 with no Three App upstream.
 */
function loopSeconds(node) {
  try {
    const slot = node.inputs?.findIndex((one) => one.name === "app");
    if (slot === undefined || slot < 0) return 0;
    const source = node.getInputNode?.(slot);
    if (source?.comfyClass !== APP_NODE_ID) return 0;
    return Math.max(0, widgetValue(source, LOOP_SECONDS, 0));
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read the app's loop length:`, error);
    return 0;
  }
}

/**
 * The accessor the strip reads the window through.
 *
 * @param {object} node - The Three Render node.
 * @returns {{read: Function, write: Function}} The pair.
 */
function spanAccessor(node) {
  return {
    read() {
      const fps = Math.max(0.01, widgetValue(node, FPS, 24));
      const frames = Math.max(1, Math.round(widgetValue(node, FRAMES, 1)));
      const start = Math.max(0, widgetValue(node, START, 0));
      const loop = loopSeconds(node);
      return {
        start,
        frames,
        fps,
        length: loop > 0 ? loop : Math.max(FALLBACK_LENGTH, start + frames / fps),
      };
    },
    write({ start, frames }) {
      for (const [name, value] of [[START, start], [FRAMES, frames]]) {
        const widget = node.widgets?.find((one) => one.name === name);
        if (!widget) continue;
        const low = Number(widget.options?.min);
        const high = Number(widget.options?.max);
        let held = value;
        if (Number.isFinite(low)) held = Math.max(low, held);
        if (Number.isFinite(high)) held = Math.min(high, held);
        widget.value = name === FRAMES ? Math.round(held) : held;
        widget.callback?.(widget.value, app.canvas, node);
      }
    },
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Three.js", "Show the capture strip"],
      name: "Draw the capture strip",
      tooltip:
        "Draw the captured window on Three Render as a strip, with a handle at each end and a "
        + "rule per frame. Drag an end to move it, drag the middle to slide the whole window, "
        + "and the axis runs to the loop_seconds of the Three App wired in. The node renders "
        + "the same either way. This applies to nodes added after the setting changes, so a "
        + "reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // strip to every node of this type.
    if (proto.__was_three_render_wrapped) return;
    proto.__was_three_render_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createTimeSpanPanel(this, {
          span: spanAccessor(this),
          label: "captured window",
        });
        appendInterfaceWidget(
          this,
          { element: panel.element, height: PANEL_HEIGHT, minWidth: PANEL_MIN_WIDTH },
          { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE },
        );
        this.__wasThreeRenderStrip = panel;

        // A widget typed into, and a link into or out of the app socket, both move the window.
        for (const name of [START, FRAMES, FPS]) {
          const widget = this.widgets?.find((one) => one.name === name);
          if (!widget) continue;
          const original = widget.callback;
          widget.callback = function (...args) {
            const answer = original?.apply(this, args);
            panel.refresh();
            return answer;
          };
        }
        const originalConnections = this.onConnectionsChange;
        this.onConnectionsChange = function (...args) {
          const answer = originalConnections?.apply(this, args);
          panel.refresh();
          return answer;
        };

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the capture strip:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the capture strip:`, error);
      }
      return result;
    };
  },
});
