/**
 * The before and after drawn on the filters, the adjustments and the colour work.
 *
 * Draws the two thumbnails, one more for every picture input that only steers the result, and
 * the measurements between them.
 */

import { app } from "../../scripts/app.js";
import { createBeforeAfterPanel } from "./interface/before_after.js";
import { createHueDial } from "./interface/hue_dial.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.PixelsBeforeAfterUI";
const LOG_NAME = "WASNodeSuite.PixelsBeforeAfter";
const SETTING_ID = "WAS.Pixels.ShowBeforeAfter";
const CONTROL_SETTING_ID = "WAS.Pixels.ShowControls";

const UI_WIDGET_NAME = "was_pixels_before_after_ui";
const UI_WIDGET_TYPE = "was_pixels_before_after";

// The socket type the pair is read from. Everything else on a node is a control.
const IMAGE_TYPE = "IMAGE";

// Nodes whose band carries a control as well as the readout, as `node id -> builder`. The
// control is drawn above the thumbnails, inside the one panel the node carries.
const CONTROLS = {
  "Image Rotate Hue": node => createHueDial(node, { widgetName: "hue_shift" }),
};

// Node id -> the CSS filter that stands for what the node does, read from its widgets. The
// after thumbnail is drawn from the before one through this as a control moves, so the strip
// answers the widget rather than the last run.
const PREDICTIONS = {
  "Image Rotate Hue": node => {
    const widget = (node.widgets ?? []).find((candidate) => candidate.name === "hue_shift");
    const shift = Number(widget?.value);
    if (!Number.isFinite(shift) || shift === 0) return "";
    return `hue-rotate(${(shift * 360).toFixed(1)}deg)`;
  },
};

// Node id -> the widgets whose movement redraws the prediction.
const PREDICTION_WIDGETS = {
  "Image Rotate Hue": ["hue_shift"],
};

// The node ids the band is drawn on. `modules/interface/pixels.py` holds the same list read
// from the other end, so the two are one list with two readers.
//
// `Image Seamless Texture` is deliberately absent although the edge blend it does is a member's
// job. Its `tiled` combo repeats the answer `tiles` times along each side, so the pair is two
// frames of different sizes, which the measurement section refuses and the fidelity glyph can
// then claim nothing about. `web/was_geometry_readout.js` draws that multiplier as a size
// instead, and one node carrying two panels leaves neither of them room.
//
// `Image Flip` is here rather than among the sizes. A mirror moves every pixel and changes no
// size, so a size band could only ever report the frame as unchanged, while the pair names the
// axis and lays one histogram exactly on the other where the pixel figures read as two
// different pictures.
const NODES = [
  "CLIPSEG2",
  "Image Blend",
  "Image Blend by Mask",
  "Image Blending Mode",
  "Image Bloom Filter",
  "Image Canny Filter",
  "Image Chromatic Aberration",
  "Image Color Match",
  "Image Displacement Warp",
  "Image Dragan Photography Filter",
  "Image Edge Detection Filter",
  "Image Film Grain",
  "Image Flip",
  "Image High Pass Filter",
  "Image Levels Adjustment",
  "Image Lucy Sharpen",
  "Image Median Filter",
  "Image Mix RGB Channels",
  "Image Monitor Effects Filter",
  "Image Pixelate",
  "Image Rembg (Remove Background)",
  "Image Remove Background (Alpha)",
  "Image Remove Color",
  "Image Rotate Hue",
  "Image SSAO (Ambient Occlusion)",
  "Image SSDO (Direct Occlusion)",
  "Image Select Channel",
  "Image Select Color",
  "Image Threshold",
  "Image fDOF Filter",
  "Image to Noise",
  "Images to Linear",
  "Images to RGB",
  "MiDaS Depth Approximation",
  "MiDaS Mask Image",
  "WASVividSharpen",
  "WASVividSharpenV2",
  "WASNSApplyLUT",
  "WASDrawImageBounds",
  "WASImageAutoLevels",
  "WASImageColorBalance",
  "WASImageCompositeMasked",
  "WASImageDequantise",
  "WASImageDirectionalBlur",
  "WASImageFrequencyBlend",
  "WASImageGuidedFilter",
  "WASImageLensDistortion",
  "WASImageTemporalEqualize",
  "WASImageToneMap",
  "WASImageVignette",
  "WASImageWhiteBalance",
];

/**
 * Whether the band is drawn at all.
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

/**
 * Whether a node's control is drawn on the band.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function controlsEnabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(CONTROL_SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(CONTROL_SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${CONTROL_SETTING_ID}:`, error);
    return true;
  }
}

/**
 * One element holding a control above a band, as one interface widget.
 *
 * @param {object} control - What a builder in `CONTROLS` answered.
 * @param {object} band - What `createBeforeAfterPanel` answered.
 * @returns {object} A panel for `appendInterfaceWidget`, holding both.
 */
function stacked(control, band) {
  const wrap = document.createElement("div");
  wrap.style.cssText = [
    "box-sizing:border-box", "width:100%", "height:100%",
    "display:flex", "flex-direction:column", "overflow:hidden",
  ].join(";");
  control.element.style.flex = `0 0 ${control.height}px`;
  control.element.style.height = "auto";
  band.element.style.flex = "1 1 auto";
  band.element.style.height = "auto";
  band.element.style.minHeight = "0";
  wrap.appendChild(control.element);
  wrap.appendChild(band.element);
  return {
    element: wrap,
    height: control.height + band.height,
    maxHeight: band.maxHeight,
    minWidth: Math.max(control.minWidth || 0, band.minWidth || 0),
    refresh() {
      control.refresh?.();
      band.refresh?.();
    },
    dispose() {
      control.dispose?.();
      band.dispose?.();
    },
  };
}

/**
 * The node's picture sockets, in the order the frontend drew them.
 *
 * @param {object} node - The node the band is going on.
 * @returns {string[]} Socket names. The first is the side the answer is compared with and the
 *   rest only steer it, which is the pairing `modules/interface/pixels.py` publishes under.
 */
export function pictureSockets(node) {
  const sockets = Array.isArray(node?.inputs) ? node.inputs : [];
  return sockets
    .filter((socket) => socket?.type === IMAGE_TYPE && socket?.name)
    .map((socket) => String(socket.name));
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Pixels", "Show the before and after"],
      name: "Draw the before and after comparison",
      tooltip:
        "Draw the picture that went in against the picture that came out, with PSNR, SSIM, "
        + `RMSE, MAE, colour shift and both histograms, on the ${NODES.length} filters, `
        + "adjustments and colour nodes. The node sends the browser a copy of each side for "
        + "this, which is two PNG encodes per run for every one of these nodes with a panel "
        + "open, plus one more for each extra picture input. Nothing is encoded while "
        + "the setting is off, and the nodes run the same either way. This applies to nodes "
        + "added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
    {
      id: CONTROL_SETTING_ID,
      category: ["WAS Node Suite", "Pixels", "Show the control on the band"],
      name: "Draw the control a node's band carries",
      tooltip:
        "Draw the control that belongs to the node above its before and after, where the node "
        + "has one: a hue wheel on Image Rotate Hue. The wheel sets hue_shift by being "
        + "dragged, and its two rings hold the hues as they arrive against where the turn "
        + "sends them. The number widget stays and can be typed into either way. This applies "
        + "to nodes added after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const id = nodeData?.name;
    if (!NODES.includes(id)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_pixels_before_after_wrapped) return;
    proto.__was_pixels_before_after_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const sockets = pictureSockets(this);
        if (!sockets.length) {
          console.error(
            `[${EXT_NAME}] ${id} declares no picture input, so there is no pair to draw.`,
          );
          return result;
        }
        const predicting = PREDICTIONS[id];
        const band = createBeforeAfterPanel(this, {
          slot: sockets[0],
          controls: sockets.slice(1),
          logName: LOG_NAME,
          predict: predicting ? () => predicting(this) : undefined,
        });
        const builder = CONTROLS[id];
        const panel = builder && controlsEnabled() ? stacked(builder(this), band) : band;
        for (const name of PREDICTION_WIDGETS[id] ?? []) {
          chainWidgetCallback(this, name, () => band.showPrediction());
        }
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the before and after:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the before and after:`, error);
      }
      return result;
    };
  },
});
