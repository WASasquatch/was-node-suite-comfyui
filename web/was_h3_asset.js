/**
 * What a MiniMax H3 Asset, RefMod or Save RefMod read, drawn on the node.
 *
 * Each node publishes its summary, counts and facts through `run_result`, and a picture under
 * its entry's slot, which the band draws.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3AssetUI";

// Each node drawn, the picture slot it publishes and the band's label.
const NODES = {
  WASMiniMaxH3Asset: { slot: "image", label: "what it holds" },
  WASMiniMaxH3RefMod: { slot: "refmod", label: "what it holds" },
  WASMiniMaxH3SaveRefMod: { slot: "refmod", label: "what was saved" },
};

// Height of the panel in node units: the summary, the counts, the facts and the picture.
const PANEL_HEIGHT = 230;

// Height of the picture in CSS pixels when the panel opens.
const PICTURE_HEIGHT = 110;

// The widest fact name the node writes, which opens the fact column.
const LABEL_WIDTH = 60;

// The narrowest the summary stays readable in.
const PANEL_MIN_WIDTH = 240;

const EMPTY_LABEL = "run the node to see what it holds";

const UI_WIDGET_NAME = "was_h3_asset_ui";
const UI_WIDGET_TYPE = "was_h3_asset";

// The node property a MiniMax H3 RefMod keeps its last run's references in, for the Prompt Timeline.
const REFMOD_NODE_ID = "WASMiniMaxH3RefMod";
const REFMOD_PROPERTY = "was_refmods";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    const drawn = NODES[nodeData?.name];
    if (!drawn) return;

    const proto = nodeType.prototype;
    if (proto.__was_h3_asset_wrapped) return;
    proto.__was_h3_asset_wrapped = true;

    if (nodeData.name === REFMOD_NODE_ID) {
      const originalOnExecuted = proto.onExecuted;
      proto.onExecuted = function (message) {
        const result = originalOnExecuted?.apply(this, arguments);
        try {
          const held = message?.[REFMOD_PROPERTY]?.[0];
          if (held && Array.isArray(held.members)) {
            this.properties = this.properties ?? {};
            this.properties[REFMOD_PROPERTY] = {
              file: String(held.file ?? ""),
              members: held.members.map((member) => ({ kind: String(member?.kind ?? ""), label: String(member?.label ?? "") })),
            };
          }
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to keep the RefMods a run added:`, error);
        }
        return result;
      };
    }

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-h3-asset",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the asset report:",
          sketch: () => createPictureBand(this, {
            slot: drawn.slot, label: drawn.label, height: PICTURE_HEIGHT,
          }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the asset panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the asset panel:`, error);
      }
      return result;
    };
  },
});
