/**
 * Drawing only the settings and models the chosen preprocessor reads, on Power Preprocessor,
 * and wiring the transformer and naming the files `Marigold v2` reads when it is chosen.
 */
// Every setting is saved and sent whether it is drawn or not.

import { app } from "../../scripts/app.js";
import { clearSpot, inputLink, slotOf } from "./interface/h3_asset_chain.js";
import { withGraphChange } from "./interface/region.js";
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
const WIRED_WIDGETS = ["adapter_name", "vae_name", "conditioning_name"];
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

// Load Diffusion Model, and the widget naming its file.
const LOADER_TYPE = "UNETLoader";
const LOADER_WIDGET = "unet_name";
const LOADER_FOLDER = "diffusion_models";
// The socket the transformer arrives on.
const TRANSFORMER_INPUT = "model";
// The transformer `Marigold v2` is published on, and the stem every build of it starts with.
const TRANSFORMER = "qwen_image_edit_2509_int8_convrot.safetensors";
const TRANSFORMER_FAMILY = "qwen_image_edit_2509";
// Where every `Marigold v2` file is published, under the name of its models folder.
const PUBLISHED_AT = "https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main";
// Space between a new loader and the node it feeds, in graph units.
const LOADER_GAP = 40;
// The file widgets' automatic entry.
const AUTOMATIC = "auto";
// Node modes that never run: muted and bypassed.
const IDLE_MODES = new Set([2, 4]);

/** Preprocessor -> the map `Marigold v2` reads for it. */
const MARIGOLD_MAPS = { depth_map: "depth", normal_map: "normals", albedo: "albedo" };

/** File widget -> its models folder, how a file there is recognised, and each map's own file. */
const MARIGOLD_FILES = {
  adapter_name: {
    folder: "loras",
    matches: (stem, map) => stem === `marigold_v2_${map}` || stem.startsWith(`marigold_v2_${map}_`),
    published: {
      depth: "marigold_v2_depth_log_stage2.safetensors",
      normals: "marigold_v2_normals.safetensors",
      albedo: "marigold_v2_albedo.safetensors",
    },
  },
  vae_name: {
    folder: "vae",
    matches: (stem, map) => stem.startsWith(`marigold_v2_${map}`),
    published: {
      depth: "marigold_v2_depth_log_stage2_vae.safetensors",
      normals: "marigold_v2_normals_vae.safetensors",
      albedo: "marigold_v2_albedo_vae.safetensors",
    },
  },
  conditioning_name: {
    folder: "embeddings",
    matches: (stem, map) => stem === `marigold_v2_${map}_conditioning`,
    published: {
      depth: "marigold_v2_depth_conditioning.safetensors",
      normals: "marigold_v2_normals_conditioning.safetensors",
      albedo: "marigold_v2_albedo_conditioning.safetensors",
    },
  },
};

/**
 * A file name without its folders.
 *
 * @param {*} name - A menu entry, folders separated by either slash.
 * @returns {string} The last part.
 */
function baseOf(name) {
  return String(name ?? "").replace(/\\/g, "/").split("/").pop();
}

/**
 * A file name without its folders or its extension.
 *
 * @param {*} name - A menu entry.
 * @returns {string} The stem.
 */
function stemOf(name) {
  const text = String(name ?? "");
  const dot = text.lastIndexOf(".");
  return baseOf(dot === -1 ? text : text.slice(0, dot));
}

/**
 * A node's widget by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget.
 */
function widgetOf(node, name) {
  return (node?.widgets ?? []).find((widget) => widget.name === name);
}

/**
 * The entries a combo widget offers.
 *
 * @param {object} widget - The widget.
 * @returns {string[]} Its options, or none.
 */
function optionsOf(widget) {
  const values = widget?.options?.values;
  const listed = typeof values === "function" ? values(widget) : values;
  return Array.isArray(listed) ? listed : [];
}

/**
 * The map `Marigold v2` is reading, when it is the chosen model.
 *
 * @param {object} node - A Power Preprocessor.
 * @returns {string|null} `depth`, `normals` or `albedo`, or null.
 */
function marigoldMap(node) {
  if (widgetOf(node, MODEL_WIDGET)?.value !== WIRED_MODEL) return null;
  return MARIGOLD_MAPS[widgetOf(node, CONTROL_WIDGET)?.value] ?? null;
}

/**
 * Whether a file name is a build of the transformer `Marigold v2` runs on.
 *
 * @param {*} name - A menu entry.
 * @returns {boolean} True for any `qwen_image_edit_2509` file.
 */
function transformerBuild(name) {
  return stemOf(name).toLowerCase().startsWith(TRANSFORMER_FAMILY);
}

/**
 * One link of a graph, by id.
 *
 * @param {object} graph - The graph.
 * @param {number} id - The link's id.
 * @returns {object|null} The link.
 */
function linkOf(graph, id) {
  const links = graph?.links;
  return (links instanceof Map ? links.get(id) : links?.[id]) ?? null;
}

/**
 * A Load Diffusion Model already in the graph on a build of the transformer.
 *
 * @param {object} node - The Power Preprocessor to feed.
 * @returns {object|null} One already feeding a Power Preprocessor, else the nearest, else null.
 */
function existingLoader(node) {
  const graph = node.graph;
  const loaders = (graph?._nodes ?? graph?.nodes ?? []).filter((other) =>
    other?.type === LOADER_TYPE
    && !IDLE_MODES.has(other.mode)
    && transformerBuild(widgetOf(other, LOADER_WIDGET)?.value));
  const feeding = loaders.find((loader) => (loader.outputs?.[0]?.links ?? []).some((id) =>
    graph.getNodeById?.(linkOf(graph, id)?.target_id)?.type === NODE_NAME));
  if (feeding) return feeding;
  const distance = (other) => Math.hypot(other.pos[0] - node.pos[0], other.pos[1] - node.pos[1]);
  return loaders.sort((a, b) => distance(a) - distance(b))[0] ?? null;
}

/**
 * Add a Load Diffusion Model on the transformer, left of the node, clear of other nodes.
 *
 * @param {object} node - The Power Preprocessor it feeds.
 * @returns {{loader: object, missing: boolean}} The new node, and whether the file it names is
 *   absent from the folder.
 * @throws {Error} Load Diffusion Model is not registered.
 */
function addLoader(node) {
  const loader = window.LiteGraph?.createNode?.(LOADER_TYPE);
  if (!loader) throw new Error("Load Diffusion Model is not registered on this server");
  const widget = widgetOf(loader, LOADER_WIDGET);
  const offered = optionsOf(widget);
  const found = offered.find((option) => baseOf(option) === TRANSFORMER)
    ?? offered.find(transformerBuild);
  if (widget) widget.value = found ?? TRANSFORMER;
  if (!found) {
    // Read by the frontend's missing-model list, which offers the download.
    loader.properties = {
      ...(loader.properties ?? {}),
      models: [{ name: TRANSFORMER, url: `${PUBLISHED_AT}/${LOADER_FOLDER}/${TRANSFORMER}`, directory: LOADER_FOLDER }],
    };
  }
  const width = loader.computeSize?.()?.[0] ?? loader.size?.[0] ?? 0;
  loader.pos = clearSpot(node.graph, loader, [node.pos[0] - width - LOADER_GAP, node.pos[1]]);
  node.graph.add(loader);
  return { loader, missing: !found };
}

/**
 * Wire a transformer into the node, reusing a loader already in the graph.
 *
 * @param {object} node - A Power Preprocessor with `Marigold v2` chosen.
 * @returns {boolean} True where a loader was added naming a file the folder lacks.
 */
function wireTransformer(node) {
  const slot = slotOf(node, TRANSFORMER_INPUT);
  if (slot < 0 || inputLink(node, slot)) return false;
  let loader = existingLoader(node);
  let missing = false;
  if (!loader) ({ loader, missing } = addLoader(node));
  loader.connect(0, node, slot);
  return missing;
}

/**
 * Take a transformer loader's link off the node once `Marigold v2` is no longer chosen.
 *
 * @param {object} node - A Power Preprocessor.
 * @returns {boolean} True where a link was removed. Anything other than a Load Diffusion Model
 *   on the transformer stays wired.
 */
function unwireTransformer(node) {
  const slot = slotOf(node, TRANSFORMER_INPUT);
  const link = inputLink(node, slot);
  if (!link) return false;
  const source = node.graph?.getNodeById?.(link.origin_id);
  if (source?.type !== LOADER_TYPE || !transformerBuild(widgetOf(source, LOADER_WIDGET)?.value)) {
    return false;
  }
  node.disconnectInput(slot);
  return true;
}

/**
 * Point each file widget on `auto` or a published file at the map's own, or name the absent file.
 *
 * @param {object} node - A Power Preprocessor.
 * @param {string|null} map - The map read, from `marigoldMap`.
 * @returns {{changed: boolean, named: boolean}} Whether any value moved, and whether a missing
 *   file was newly named.
 */
function followFiles(node, map) {
  let changed = false;
  let named = false;
  for (const [name, spec] of Object.entries(MARIGOLD_FILES)) {
    const widget = widgetOf(node, name);
    if (!widget) continue;
    const offered = optionsOf(widget);
    const current = widget.value;
    const published = Object.values(spec.published);
    if (current !== AUTOMATIC && !published.includes(baseOf(current))) continue;

    let wanted = current;
    if (map) {
      const held = offered.some((option) => option !== AUTOMATIC && spec.matches(stemOf(option), map));
      const kept = offered.includes(current) && spec.matches(stemOf(current), map);
      if (!held) wanted = spec.published[map];
      else if (!kept) wanted = AUTOMATIC;
    } else if (current !== AUTOMATIC && !offered.includes(current)) {
      wanted = AUTOMATIC;
    }
    if (wanted === current) continue;
    widget.value = wanted;
    changed = true;
    if (!offered.includes(wanted)) named = true;
  }
  return { changed, named };
}

/**
 * List the node's named but absent files where the frontend's missing-model list reads them.
 *
 * @param {object} node - A Power Preprocessor.
 * @returns {void}
 */
function recordMissing(node) {
  const listed = [];
  for (const [name, spec] of Object.entries(MARIGOLD_FILES)) {
    const widget = widgetOf(node, name);
    const value = widget?.value;
    if (!Object.values(spec.published).includes(value) || optionsOf(widget).includes(value)) continue;
    listed.push({ name: value, url: `${PUBLISHED_AT}/${spec.folder}/${value}`, directory: spec.folder });
  }
  const properties = node.properties ?? (node.properties = {});
  const kept = (properties.models ?? []).filter((entry) => !String(entry?.url ?? "").startsWith(PUBLISHED_AT));
  if (kept.length || listed.length) properties.models = [...kept, ...listed];
  else delete properties.models;
}

/**
 * Bring the transformer link and the file widgets in line with the chosen model and map.
 *
 * @param {object} node - A Power Preprocessor.
 * @returns {void}
 */
function followMarigold(node) {
  const map = marigoldMap(node);
  let changed = false;
  let named = false;
  withGraphChange(() => {
    if (map) named = wireTransformer(node);
    else changed = unwireTransformer(node);
    const files = followFiles(node, map);
    changed = changed || files.changed;
    named = named || files.named;
    recordMissing(node);
  });
  node.setDirtyCanvas?.(true, true);
  if (!changed && !named) return;
  // Opens the frontend's error panel only where this change named a missing file.
  queueMicrotask(() => {
    Promise.resolve(app.refreshMissingModels?.({ silent: !named, reloadDefs: false }))
      .catch((error) => console.error(`[${EXT_NAME}] Failed to list missing models:`, error));
  });
}

/**
 * Follow `Marigold v2` whenever the model or the preprocessor is changed by hand.
 *
 * @param {object} node - The node to govern.
 * @returns {void}
 */
function watchMarigold(node) {
  for (const name of [MODEL_WIDGET, CONTROL_WIDGET]) {
    const widget = widgetOf(node, name);
    if (!widget) continue;
    const previous = widget.callback;
    widget.callback = function callback(...args) {
      const answer = previous?.apply(this, args);
      try {
        followMarigold(node);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to wire Marigold v2:`, error);
      }
      return answer;
    };
  }
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
      try {
        if (hidingEnabled()) {
          watchRelevantWidgets(this, CONTROL_WIDGET, MODES);
          watchWiredInputs(this);
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to lay the node out:`, error);
      }
      // Wrapped after the layout callbacks above, which draw the socket it wires.
      watchMarigold(this);
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
