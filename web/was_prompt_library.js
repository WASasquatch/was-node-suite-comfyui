/**
 * What an edit did to the terminology pantry or the style library, drawn on the node.
 *
 * Both families publish their counts, the term or file, and the entries through
 * `run_result`, and the shared report panel draws them.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.PromptLibrary";
const LOG_NAME = "WASNodeSuite.PromptLibrary";
const SETTING_ID = "WAS.PromptLibrary.ShowReport";

// The nodes this draws on.
const NODES = [
  "WASNoodleSoupTermEdit",
  "WASNoodleSoupPantryImport",
  "WASNoodleSoupPantryExport",
  "WASPromptStyleSave",
  "WASPromptStylesImport",
  "WASPromptStylesExport",
];

// Tall enough for the summary, a row of count tiles, two fact rows and a listing under them.
const PANEL_HEIGHT = 236;

const LABEL_WIDTH = 76;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 260;

const EMPTY_LABEL = "run the node to see the list";

const UI_WIDGET_NAME = "was_prompt_library_ui";
const UI_WIDGET_TYPE = "was_prompt_library";

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
      category: ["WAS Node Suite", "Prompt library", "Show what an edit did"],
      name: "Show the pantry and style report",
      tooltip:
        "Draw what was added, removed and stored, and list the entries themselves, on the "
        + "Noodle Soup terminology and prompt style nodes. The nodes run the same either way. "
        + "This applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;


    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_prompt_library_wrapped) return;
    proto.__was_prompt_library_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const panel = createReportPanel(this, {
          className: "was-prompt-library",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: LOG_NAME,
          failure: "Failed to read the prompt library report:",
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the library panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the library panel:`, error);
      }
      return result;
    };
  },
});
