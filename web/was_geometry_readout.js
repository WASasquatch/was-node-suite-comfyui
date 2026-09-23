/**
 * The size report drawn on the nodes whose job is the size.
 *
 * Draws both sizes, the scale and the pixel count on the node itself.
 */

import { app } from "../../scripts/app.js";
import { SHORT_HEIGHT, createSizePanel } from "./interface/size_panel.js";
import { createPictureBand } from "./interface/picture_band.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.GeometryReadoutUI";
const SETTING_ID = "WAS.Geometry.ShowSizes";

// The node ids the readout is drawn on. `modules/interface/size_report.py` publishes from each
// of these and from nothing else, so the two lists are one list read from two ends.
//
// Image Seamless Texture is here rather than among the filters: its `tiled` combo repeats
// the answer `tiles` times along each side, so the size is a widget on this node.
//
// Image Flip is absent. A mirror answers the frame it took, so the band could only ever report
// the size as unchanged, which is a figure that cannot fail; the pair it does answer is drawn by
// `web/was_pixels_before_after.js` instead.
const NODES = [
  "Bounded Image Blend",
  "Bounded Image Blend with Mask",
  "Bounded Image Crop",
  "Bounded Image Crop with Mask",
  "Create Grid Image from Batch",
  "Image Padding",
  "Image Paste Crop",
  "Image Paste Crop by Location",
  "Image Paste Face",
  "Image Resize",
  "Image Rotate",
  "Image Seamless Texture",
  "Image Stitch",
  "Image Tiled",
  "Image Transpose",
  "Latent Upscale by Factor (WAS)",
  "WASCanvasComposeBatch",
  "WASImageCropFaceNative",
  "WASImageCropFaceYuNet",
  "WASImageCropRegion",
  "WASImagePadForOutpaint",
  "WASNSImageTileExtract",
  "WASImageTileExtractGrid",
  "WASNSImageTileShuffle",
  "WASLatentScaleToMaxDimension",
  "WASPSSRSuperResolution",
  "WASNSTiledImageUpscaleWithModel",
];

// The face crops answer a picture whose whole point is what it looks like, so they draw the
// crop itself where the other nodes draw two rectangles. The size figures stay above it.
const SHOWS_RESULT = new Set([
  "Image Paste Face",
  "WASImageCropFaceNative",
  "WASImageCropFaceYuNet",
]);

// The height a band drawing a real picture opens at, against the sketch's shorter one.
const PICTURE_HEIGHT = 200;

// Nodes that already carry an interface of their own take the summary line only, and declare no
// room above it, so every spare unit the node is dragged into still goes to that interface.
const SHORT = {
  "Image Paste Crop by Location": {
    height: SHORT_HEIGHT,
    tiles: false,
    facts: false,
    sketch: false,
    grows: false,
  },
};

const UI_WIDGET_NAME = "was_size_report_ui";
const UI_WIDGET_TYPE = "was_size_report";

/**
 * Whether the readout is drawn at all.
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
      category: ["WAS Node Suite", "Geometry", "Show what the size became"],
      name: "Draw the size report",
      tooltip:
        "Draw the frame that went in against the frame that came out, with the scale, the "
        + "pixel count and a sketch of the two, on Image Resize, the crops, the pastes, the "
        + "tilers and the latent scalers. The face crops draw the face they cut instead of "
        + "the sketch. A size that is not the one asked for, such as a "
        + "width rounded up to a multiple of 8, and a crop resampled to fit its window, are "
        + "drawn in the warning colour. The nodes run the same either way. This applies to "
        + "nodes added after the setting changes, so a reload shows it everywhere.",
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
    if (proto.__was_geometry_readout_wrapped) return;
    proto.__was_geometry_readout_wrapped = true;

    const options = SHORT[id] ?? (SHOWS_RESULT.has(id)
      ? {
        height: PICTURE_HEIGHT,
        sketch: (node) => createPictureBand(node, { label: "what came out" }),
      }
      : {});

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        const built = typeof options.sketch === "function"
          ? { ...options, sketch: () => options.sketch(this) }
          : options;
        const panel = createSizePanel(this, built);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the size readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the size readout:`, error);
      }
      return result;
    };
  },
});
