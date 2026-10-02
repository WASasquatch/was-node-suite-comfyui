/**
 * The tile plan drawn on H3 Tiled Sampler [WIP].
 *
 * Draws where the tiles and the time windows sit and the sigma the run started from, in a panel
 * of its own beside the image the sampler's step previews draw in.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.H3TiledSamplerWIPUI";
const NODE_ID = "WASH3TiledSamplerWIP";

// Height of the panel in node units: the summary, one row of figures, the frame fact and the
// picture of the tiles.
const PANEL_HEIGHT = 300;

// Height of the picture of the tiles in CSS pixels when the panel opens.
const PLAN_HEIGHT = 170;

// The widest figure name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 96;

const EMPTY_LABEL = "run the node once to see where the tiles sit";

const UI_WIDGET_NAME = "was_h3_tiled_sampler_wip_ui";
const UI_WIDGET_TYPE = "was_h3_tiled_sampler_wip";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_h3_tiled_sampler_wip_wrapped) return;
    proto.__was_h3_tiled_sampler_wip_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createReportPanel(this, {
          className: "was-h3-tiled-sampler",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the tile plan:",
          sketch: () => createPictureBand(this, { label: "the tile plan", height: PLAN_HEIGHT }),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the tile plan:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the tile plan:`, error);
      }
      return result;
    };
  },
});
