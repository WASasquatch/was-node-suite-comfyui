/**
 * What a 3D node made of the model, drawn on the mesh, model and character nodes.
 *
 * Figures: parts, vertices, triangles, quads, joints, height. Rows: the node's own numbers and
 * the model's before it. Facts below.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.Model3DReport";

// The nodes that publish a model's figures after a run.
const NODES = [
  "WASModel3DLoad",
  "WASModel3DSave",
  "WASModel3DMeta",
  "WASModel3DParts",
  "WASModel3DTransform",
  "WASModel3DMaterial",
  "WASMeshDecimate",
  "WASMeshRemesh",
  "WASMeshSmooth",
  "WASMeshSubdivide",
  "WASMeshRetopology",
  "WASMeshClean",
  "WASMeshSymmetrize",
  "WASMeshCut",
  "WASMeshInspect",
  "WASMeshUnwrap",
  "WASMeshBakeTextures",
  "WASCharacterAssemble",
  "WASCharacterRig",
  "WASCharacterSprings",
  "WASCharacterNeckJoin",
  "WASMeshRenderViews",
  "WASMeshProjectViews",
];

// The nodes that draw a picture under their figures, and the word drawn under it.
const PICTURES = {
  WASMeshRenderViews: "views",
  WASMeshProjectViews: "texture",
};

// Height of the panel in node units when it carries a picture, and of the picture in CSS pixels.
const PICTURE_PANEL_HEIGHT = 460;
const PICTURE_HEIGHT = 180;

// Height of the panel in node units: the summary, eight figures, eight rows and the facts.
const PANEL_HEIGHT = 300;

// The narrowest the summary line stays readable in.
const PANEL_MIN_WIDTH = 260;

// The widest row name the nodes write, which is what the name column is opened at.
const LABEL_WIDTH = 110;

const EMPTY_LABEL = "run the node to see the model's figures";

const UI_WIDGET_NAME = "was_model3d_report_ui";
const UI_WIDGET_TYPE = "was_model3d_report";

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData?.name)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_model3d_report_wrapped) return;
    proto.__was_model3d_report_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const picture = PICTURES[nodeData.name];
        const panel = createReportPanel(this, {
          className: "was-model3d-report",
          height: picture ? PICTURE_PANEL_HEIGHT : PANEL_HEIGHT,
          minWidth: PANEL_MIN_WIDTH,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the 3D report:",
          sketch: picture ? () => createPictureBand(this, { label: picture, height: PICTURE_HEIGHT }) : undefined,
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the 3D report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the 3D report:`, error);
      }
      return result;
    };
  },
});
