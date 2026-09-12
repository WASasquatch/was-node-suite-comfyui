/**
 * The report drawn on the nodes that write a file.
 *
 * Draws the file count and the write count as figures, and the folder, the format, the first
 * and last names and the bytes as rows.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.FileWritten";
const SETTING_ID = "WAS.Files.ShowWritten";
const LOG_NAME = "WASNodeSuite.FileWritten";

// Every node in the pack that writes a file. Image Save is the one of the seven the core
// frontend draws anything for, and what it draws is the pictures, not where they landed; the
// other six draw nothing at all.
const NODES = [
  "Image Save",
  "WASDNGSave",
  "WASEXRSave",
  "WASLayersSave",
  "Write to GIF",
  "Write to Video",
  "Create Morph Image",
];

// Height of the panel in node units: the summary line, the two figures, the five fact rows the
// report writes itself and the three a node adds, with nothing scrolling.
const PANEL_HEIGHT = 228;

// The widest name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 78;

const EMPTY_LABEL = "run the node once to see what it wrote";

const UI_WIDGET_NAME = "was_file_written_ui";
const UI_WIDGET_TYPE = "was_file_written";

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
      category: ["WAS Node Suite", "Files", "Show what was written"],
      name: "Draw the file report",
      tooltip:
        "Draw the number of files, the number of writes, the folder, the format, the first and "
        + "last names and the bytes on Image Save, EXR Save, Write to GIF, Write to Video and "
        + "Create Morph Image. A naming scheme that replaces its own output on every frame, and a "
        + "write that failed, are drawn in the warning colour. The nodes run the same either "
        + "way. This applies to nodes added after the setting changes, so a reload shows it "
        + "everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_file_written_wrapped) return;
    proto.__was_file_written_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-file-written",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the file report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the file report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the file report:`, error);
      }
      return result;
    };
  },
});
