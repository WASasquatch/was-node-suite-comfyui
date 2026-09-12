/**
 * Drawing only the settings and models the chosen preprocessor reads, on Power Preprocessor.
 */
// Every setting is saved and sent whether it is drawn or not.

import { app } from "../../scripts/app.js";
import { refreshRelevantWidgets, watchRelevantWidgets } from "./interface/relevant_widgets.js";
import { growSockets } from "./interface/grow_sockets.js";
import { setWidgetHidden } from "./interface/visibility.js";

const EXT_NAME = "WASNodeSuite.PowerPreprocessorUI";
const NODE_NAME = "WASPowerPreprocessor";
const SETTING_ID = "WAS.PowerPreprocessor.HideIrrelevant";

const CONTROL_WIDGET = "preprocessor";

const CUT = (label, start) => ({ label, min: 0, max: 255, start, step: 1, precision: 0 });
// Name of the control widget the frontend draws under a seed.
const SEED_CONTROL = "fixed";
const SEED_PAIR = {
  seed: { label: "seed", min: 0, max: 0xffffffffffffffff, start: 0, step: 1, precision: 0 },
  [SEED_CONTROL]: { label: SEED_CONTROL },
};
const TILE = {
  tile: { label: "tile", min: 0, max: 4096, start: 0, step: 64, precision: 0 },
};
const INTRINSIC_STEPS = {
  steps: { label: "steps", min: 1, max: 20, start: 4, step: 1, precision: 0 },
};
// The model read from the node's own inputs.
const WIRED_MODEL = "Marigold v2";
const DEPTH = [
  "Depth Anything V2 Small", "Depth Anything V2 Base", "Depth Anything V2 Large",
  "DPT SwinV2 Tiny", "DPT Large", WIRED_MODEL,
];
const model = (values) => ({ model_name: { label: "model", values } });

/** Preprocessor -> the shared widgets it reads, each under its label, bounds and models. */
// A widget no preprocessor lists is left alone, `resolution` among them.
const MODES = {
  canny_pyramid: {
    threshold_low: { label: "low_threshold", min: 1, max: 255, start: 100, step: 1, precision: 0 },
    threshold_high: CUT("high_threshold", 200),
  },
  lineart_simple: {
    radius: { label: "blur_radius", min: 0.5, max: 32.0, start: 6.0, step: 0.1, precision: 1 },
    threshold_low: { label: "noise_floor", min: 0, max: 64, start: 8, step: 1, precision: 0 },
  },
  scribble_xdog: {
    threshold_low: { label: "stroke_threshold", min: 1, max: 64, start: 32, step: 1, precision: 0 },
  },
  binary: { threshold_low: CUT("split_level", 100) },
  shuffle: SEED_PAIR,
  depth_map: model(DEPTH),
  normal_map: {
    ...model(DEPTH),
    strength: { label: "relief", min: 0.5, max: 64.0, start: 16.0, step: 0.5, precision: 1 },
    radius: { label: "smoothing", min: 0, max: 8, start: 3, step: 1, precision: 0 },
  },
  openpose: {
    ...model(["ViTPose Base", "ViTPose Small", "ViTPose Wholebody"]),
    threshold_low: {
      label: "detection_threshold", min: 0.05, max: 0.95, start: 0.3, step: 0.05, precision: 2,
    },
  },
  animal_pose: {
    ...model(["ViTPose Animal"]),
    threshold_low: {
      label: "detection_threshold", min: 0.05, max: 0.95, start: 0.3, step: 0.05, precision: 2,
    },
  },
  ade20k_segments: model(["SegFormer B0 ADE20K", "SegFormer B2 ADE20K", "SegFormer B4 ADE20K"]),
  soft_edge: model(["HED Soft Edge", "PiDiNet Soft Edge", "TEED Soft Edge"]),
  lineart_model: model(["Lineart", "Lineart Coarse", "Lineart Anime", "Manga Line"]),
  denoise: { ...model(["SCUNet", "NAFNet SIDD width32", "NAFNet SIDD width64"]), ...TILE },
  low_light: { ...model([
    "DarkIR", "Retinexformer NTIRE", "Retinexformer LOL v1", "Retinexformer LOL v2 Real",
    "Retinexformer LOL v2 Synthetic", "Retinexformer FiveK", "Retinexformer Extreme Dark",
    "Retinexformer Dark Motion", "Retinexformer Indoor Night", "Retinexformer Outdoor Night",
    "HVI-CIDNet Generalization", "HVI-CIDNet FiveK", "HVI-CIDNet SICE",
    "HVI-CIDNet Extreme Dark",
  ]), ...TILE },
  albedo: {
    ...model(["Marigold IID Appearance", "Marigold IID Lighting", WIRED_MODEL]),
    ...INTRINSIC_STEPS,
    ...SEED_PAIR,
  },
  roughness: { ...model(["Marigold IID Appearance"]), ...INTRINSIC_STEPS, ...SEED_PAIR },
  metallicity: { ...model(["Marigold IID Appearance"]), ...INTRINSIC_STEPS, ...SEED_PAIR },
  material: { ...model(["Marigold IID Appearance"]), ...INTRINSIC_STEPS, ...SEED_PAIR },
  shading: { ...model(["Marigold IID Lighting"]), ...INTRINSIC_STEPS, ...SEED_PAIR },
  residual: { ...model(["Marigold IID Lighting"]), ...INTRINSIC_STEPS, ...SEED_PAIR },
  anyline: {
    ...model(["AnyLine"]),
    threshold_low: { label: "speck_size", min: 1, max: 256, start: 36, step: 1, precision: 0 },
  },
  line_segments: {
    ...model(["MLSD Line Segments"]),
    threshold_low: {
      label: "score_threshold", min: 0.01, max: 0.4, start: 0.1, step: 0.01, precision: 2,
    },
    threshold_high: {
      label: "shortest_segment", min: 1, max: 60, start: 20, step: 1, precision: 0,
    },
  },
};

/**
 * Read whether irrelevant settings are hidden at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function hidingEnabled() {
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

// What the wired model reads.
const WIRED_INPUTS = ["model", "vae", "conditioning"];
const WIRED_WIDGETS = ["adapter_name", "conditioning_name"];
const MODEL_WIDGET = "model_name";

/**
 * Draw the wired model's own inputs only while that model is chosen.
 *
 * A socket carrying a link stays drawn whatever is chosen.
 *
 * @param {object} node - The node to lay out.
 * @returns {void}
 */
function wiredModelChosen(node) {
  const widgets = node.widgets || [];
  const chosen = widgets.find(widget => widget.name === MODEL_WIDGET)?.value;
  const question = widgets.find(widget => widget.name === CONTROL_WIDGET)?.value;
  // The question is read as well as the model.
  const offered = MODES[question]?.model_name?.values || [];
  return chosen === WIRED_MODEL && offered.includes(WIRED_MODEL);
}

/**
 * Draw the wired model's own widget only while that model is chosen.
 *
 * @param {object} node - The node to lay out.
 * @returns {void}
 */
function presentWiredWidget(node) {
  const wanted = wiredModelChosen(node);
  let moved = false;
  for (const name of WIRED_WIDGETS) {
    const widget = (node.widgets || []).find(entry => entry.name === name);
    moved = setWidgetHidden(widget, !wanted) || moved;
  }
  if (moved) {
    node.setSize?.(node.computeSize());
    node.setDirtyCanvas?.(true, true);
  }
}

/**
 * Present the wired inputs now, and again whenever the model changes.
 *
 * @param {object} node - The node to govern.
 * @returns {void}
 */
function watchWiredInputs(node) {
  // One group, so the sockets appear together.
  const refit = growSockets(node, [WIRED_INPUTS], {
    exactCount: () => (wiredModelChosen(node) ? 1 : 0),
  });
  const present = () => {
    refit();
    presentWiredWidget(node);
  };
  // The question can change the model with it.
  for (const name of [MODEL_WIDGET, CONTROL_WIDGET]) {
    const widget = (node.widgets || []).find(entry => entry.name === name);
    if (!widget) continue;
    const previous = widget.callback;
    widget.callback = function callback(...args) {
      const answer = previous?.apply(this, args);
      present();
      return answer;
    };
  }
  present();
  return present;
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Power Preprocessor", "Relevant settings"],
      name: "Show only the chosen preprocessor's settings",
      tooltip:
        "Draw only the settings and models the preprocessor chosen on Power Preprocessor " +
        "can use, and hide the rest. Every setting is still saved with the workflow and " +
        "still sent with an API prompt either way, so this changes what the node looks " +
        "like and nothing about what it does. Turn it off to see all of them at once. " +
        "This applies to nodes added after the setting changes, so a reload shows it " +
        "everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;
    if (proto.__was_power_preprocessor_wrapped) return;
    proto.__was_power_preprocessor_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function onNodeCreated() {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!hidingEnabled()) return result;
      try {
        watchRelevantWidgets(this, CONTROL_WIDGET, MODES);
        watchWiredInputs(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to lay the node out:`, error);
      }
      return result;
    };

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function onConfigure() {
      const result = originalOnConfigure?.apply(this, arguments);
      if (!hidingEnabled()) return result;
      try {
        refreshRelevantWidgets(this, CONTROL_WIDGET, MODES);
        presentWiredWidget(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to lay the node out after loading:`, error);
      }
      return result;
    };
  },
});
