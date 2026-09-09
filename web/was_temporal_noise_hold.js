/**
 * The report drawn on Temporal Noise Hold.
 *
 * Draws the hold, the coefficient it works out to and the measured correlation, frame change
 * and spread as figures, and the seed and each stream as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.TemporalNoiseHold";
const SETTING_ID = "WAS.Sampling.ShowNoiseHold";
const LOG_NAME = "WASNodeSuite.TemporalNoiseHold";

const NODES = ["WASTemporalNoiseHold"];

// Height of the panel in node units: the summary line, the five figures and the rows for the
// seed and each stream of a nested latent, with nothing scrolling.
const PANEL_HEIGHT = 204;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 78;

const EMPTY_LABEL = "sample once to see what the noise came out as";

const UI_WIDGET_NAME = "was_temporal_noise_hold_ui";
const UI_WIDGET_TYPE = "was_temporal_noise_hold";

/**
 * Whether the report is drawn at all.
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
      category: ["WAS Node Suite", "Sampling", "Show the temporal noise hold report"],
      name: "Draw the temporal noise hold report",
      tooltip:
        "Draw the hold, the measured correlation between neighbouring frames, the frame-to-frame "
        + "change as a share of ordinary noise, the spread and each stream of the latent on "
        + "Temporal Noise Hold. A latent with no time axis, which the node passes through "
        + "untouched, is drawn in the warning colour. The node runs the same either way. This "
        + "applies to nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_temporal_noise_hold_wrapped) return;
    proto.__was_temporal_noise_hold_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-temporal-noise-hold",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the temporal noise hold report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the temporal noise hold report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the temporal noise hold report:`, error);
      }
      return result;
    };
  },
});
