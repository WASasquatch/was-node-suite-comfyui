/**
 * What a MiniMax H3 Asset read, drawn on the node.
 *
 * The node publishes its summary, its frame and second counts and its facts through
 * `run_result`, and the first picture it holds on its `image` output, which the band draws.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3AssetUI";
const NODE_ID = "WASMiniMaxH3Asset";

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

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_h3_asset_wrapped) return;
    proto.__was_h3_asset_wrapped = true;

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
            slot: "image", label: "what it holds", height: PICTURE_HEIGHT,
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
