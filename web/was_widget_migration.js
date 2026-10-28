/**
 * Saved v2 widget values, put back on the widget they were written for: earlier widget orders,
 * former combos and sliders that are now checkboxes, and renamed options.
 */

import { app } from "../../scripts/app.js";
import { migrateWidgetValues } from "./interface/widget_migration.js";

const EXT_NAME = "WASNodeSuite.WidgetMigration";

// MiniMax H3 Conditioning's row widgets, in the order each row took them.
const H3_ROW_FIELDS = ["prompt", "duration", "overlap", "continuity", "source", "header_footer", "model"];

/**
 * One earlier MiniMax H3 Conditioning widget order.
 *
 * @param {number} fields - How many of each row's widgets the layout held.
 * @param {boolean} refSize - Whether the layout ends on `ref_image_size`.
 * @returns {string[]} The widget names, in saved order.
 */
function h3Layout(fields, refSize) {
  return [
    "mode", "aspect_ratio", "megapixels", "width", "height", "prompt_header", "prompt_footer",
    ...Array.from({ length: 24 }, (unused, index) =>
      H3_ROW_FIELDS.slice(0, fields).map((field) => `${field}_${index + 1}`)).flat(),
    ...(refSize ? ["ref_image_size"] : []),
  ];
}

// The Tiled Captioned samplers' `tiling: manual` and `affine: on` widgets, as their groups hold them.
const TILED_TILING = ["tiling.tile_width", "tiling.tile_height", "tiling.overlap"];
const TILED_AFFINE = [
  "affine.affine_interval", "affine.max_scale", "affine.max_bias", "affine.pattern", "affine.affine_seed",
  "affine.affine_seed_increment", "affine.temporal_mode", "affine.affine_space", "affine.affine_acts_on",
];

/**
 * A Tiled Captioned sampler's widget orders from before `affine_keep_colour`, its affine group open.
 *
 * @param {string[]} lead - The widgets ahead of `tiling`.
 * @returns {string[][]} The order with tiling `auto`, then with tiling `manual`.
 */
function tiledLayouts(lead) {
  return [false, true].map((manual) => [
    ...lead, "tiling", ...(manual ? TILED_TILING : []), "affine", ...TILED_AFFINE,
    "prompt", "style", "system_prompt",
  ]);
}

// Node id -> the widgets an earlier save's `widgets_values` holds, in the order it holds them. Read
// left to right against the node's current widgets: anything the node has that is not named here
// is new, and takes its default rather than a value that was meant for the widget before it.
const V2_WIDGET_ORDER = {
  // The Tiled Captioned samplers' affine group gained `affine_keep_colour`. A save made with the
  // group open puts its prompt, style and system prompt back on those widgets.
  // `control_after_generate` rides on `seed`, and the migration adds it after `seed` itself.
  "WASTiledCaptionedKSampler": tiledLayouts([
    "upscale_by", "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise",
  ]),
  "WASTiledCaptionedCustomSampler": tiledLayouts(["upscale_by"]),
  // Three Custom Material took a typed body before it took a module file. The saved body is
  // dropped rather than landing on the module menu.
  "WASThreeCustomMaterial": ["javascript", "module"],
  "WASImageGradientMapNative": ["flip_left_right"],
  // Resolution Selector (Advanced) took the shape's orientation from a widget of its own. A save
  // holding portrait opens on the landscape entry of the same shape.
  "WASResolutionSelector": ["aspect_ratio", "orientation", "size_by", "size", "multiple_of", "megapixels"],
  // Film Grain's v2 settings name no current widget, so a v2 save opens on the defaults.
  "Image Film Grain": ["density", "intensity", "highlights", "supersample_factor"],
  "Image SSAO (Ambient Occlusion)": [
    "strength",
    "radius",
    "ao_blur",
    "specular_threshold",
    "enable_specular_masking",
    "tile_size",
  ],
  "Load Image Batch": [
    "mode",
    "seed",
    "index",
    "label",
    "path",
    "pattern",
    "allow_RGBA_output",
    "filename_text_extension",
  ],
  "Save Text File": [
    "path",
    "filename_prefix",
    "filename_delimiter",
    "filename_number_padding",
    "file_extension",
    "encoding",
    "filename_suffix",
  ],
  "Text Add Token by Input": ["print_current_tokens"],
  "Text Dictionary New": [
    "key_1", "value_1", "key_2", "value_2", "key_3", "value_3", "key_4", "value_4",
    "key_5", "value_5", "key_6", "value_6", "key_7", "value_7", "key_8", "value_8",
    "list_values",
  ],
  "Text Compare": ["mode", "tolerance"],
  "Text Find": ["substring", "pattern"],
  "Text Find and Replace by Dictionary": ["replacement_key", "seed"],
  "Text Random Line": ["seed"],
  "Text Shuffle": ["separator", "seed"],
  "Text Sort": ["separator"],
  "Text String Truncate": ["truncate_by", "truncate_from", "truncate_to"],
  // CNS Model Patch replaced its manual widgets with the published allocation's own knobs.
  WASCNSModelPatch: ["mode", "strength", "bands", "slope", "start", "end"],
  "Text to Console": ["label"],
  // MiniMax H3 Conditioning gained a continuity menu on each row, after that row's overlap,
  // then a source, a header and footer menu, a model menu, and then a hold, a sound and a seed.
  // One earlier order per layout a save may hold.
  WASMiniMaxH3Conditioning: [
    h3Layout(3, false),
    h3Layout(4, false),
    h3Layout(4, true),
    h3Layout(5, true),
    h3Layout(6, true),
    h3Layout(7, true),
  ],
};

// Node id -> widgets that took the strings "false" and "true" before becoming a checkbox. A
// saved "false" is truthy, so it would arrive ticked without this.
const V2_BOOLEAN_WIDGETS = {
  "Create Grid Image": ["include_subfolders"],
  "Image Canny Filter": ["enable_threshold"],
  "Image Dragan Photography Filter": ["colorize"],
  "Image Filter Adjustments": ["detail_enhance"],
  "WASImageGradientMapNative": ["flip_left_right"],
  "Image High Pass Filter": ["color_output", "neutral_background"],
  "Image Load": ["RGBA", "filename_text_extension"],
  "Image Padding": ["feather_second_pass"],
  "Image Pixelate": ["dither", "reverse_palette"],
  "Image Resize": ["supersample"],
  "Image SSAO (Ambient Occlusion)": ["enable_specular_masking"],
  "Image SSDO (Direct Occlusion)": ["colored_occlusion"],
  "Image Save": [
    "filename_number_start",
    "optimize_image",
    "lossless_webp",
    { name: "overwrite_mode", on: "prefix_as_filename" },
    "show_history",
    "show_history_by_prefix",
    "embed_workflow",
    "show_previews",
  ],
  "Image Seamless Texture": ["tiled"],
  "Image Voronoi Noise Filter": ["flat", "RGB_output"],
  "Latent Upscale by Factor (WAS)": ["align"],
  "Load Image Batch": ["allow_RGBA_output", "filename_text_extension"],
  "MiDaS Depth Approximation": ["use_cpu", "invert_depth"],
  "MiDaS Mask Image": ["use_cpu", "threshold"],
  "Text Add Token by Input": ["print_current_tokens"],
  "Text Add Tokens": ["print_current_tokens"],
  "Text Concatenate": ["clean_whitespace"],
};

// Node id -> widgets that took a number from 0 to 1 before becoming a checkbox, read as ticked
// above 0.5. A saved 0.3 is truthy, so it would arrive ticked without this.
const V2_NUMBER_BOOLEAN_WIDGETS = {
  "Logic Boolean": ["boolean"],
};

// Continuity menus on the H3 nodes, and the options each once offered under another name.
const H3_CONTINUITY_RENAMED = { reference: "reference (video)" };

// Node id -> `[widget names, {saved option: current option}]` for combos whose options were renamed.
const RENAMED_OPTIONS = {
  WASH3ExtendWindow: [["continuity"], H3_CONTINUITY_RENAMED],
  WASMiniMaxH3Conditioning: [
    Array.from({ length: 24 }, (unused, index) => `continuity_${index + 1}`),
    { ...H3_CONTINUITY_RENAMED, "as set": "carry" },
  ],
  WASRembgModelLoader: [
    ["model"],
    {
      "BiRefNet General": "birefnet/General.safetensors",
      "BiRefNet General HR": "birefnet/General-HR.safetensors",
      "BiRefNet General Dynamic": "birefnet/General-dynamic.safetensors",
      "BiRefNet General 512": "birefnet/General-reso_512.safetensors",
      "BiRefNet Portrait": "birefnet/Portrait.safetensors",
      "BiRefNet Matting HR": "birefnet/Matting-HR.safetensors",
      "BiRefNet Fine Detail": "birefnet/DIS.safetensors",
      "BiRefNet Fine Detail Extended": "birefnet/DIS-TR_TEs.safetensors",
      "BiRefNet Camouflage": "birefnet/COD.safetensors",
      "BiRefNet Salient Object": "birefnet/HRSOD.safetensors",
      BEN2: "ben2/ben2-base.safetensors",
    },
  ],
};

/**
 * Put a renamed option's current name on each combo still holding the old one.
 *
 * @param {object} node - The node being configured.
 * @param {[string[], Object<string, string>]} entry - Widget names and the renames.
 * @returns {void}
 */
function renameOptions(node, [names, renamed]) {
  for (const widget of node.widgets ?? []) {
    if (!names.includes(widget?.name)) continue;
    if (Object.hasOwn(renamed, widget.value)) widget.value = renamed[widget.value];
  }
}

/**
 * Read the saved strings of a node's former combo widgets back as booleans.
 *
 * @param {object} node - The node being configured.
 * @param {(string|{name: string, on: string})[]} entries - The widgets to coerce.
 * @returns {void}
 */
function coerceBooleans(node, entries) {
  // A bare name ticked on "true"; a `{name, on}` pair names the string that stood for ticked.
  for (const entry of entries) {
    const name = typeof entry === "string" ? entry : entry.name;
    const ticked = (typeof entry === "string" ? "true" : entry.on).toLowerCase();
    const widget = (node.widgets ?? []).find((candidate) => candidate.name === name);
    if (typeof widget?.value === "string") widget.value = widget.value.toLowerCase() === ticked;
  }
}

/**
 * Read the saved numbers of a node's former 0 to 1 sliders back as booleans.
 *
 * @param {object} node - The node being configured.
 * @param {string[]} names - The widgets to coerce.
 * @returns {void}
 */
function numbersToBooleans(node, names) {
  for (const widget of node.widgets ?? []) {
    if (names.includes(widget?.name) && typeof widget.value === "number") widget.value = widget.value > 0.5;
  }
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const order = V2_WIDGET_ORDER[nodeData?.name];
    const booleans = V2_BOOLEAN_WIDGETS[nodeData?.name];
    const numbers = V2_NUMBER_BOOLEAN_WIDGETS[nodeData?.name];
    const renamed = RENAMED_OPTIONS[nodeData?.name];
    if (!order && !booleans && !numbers && !renamed) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap `onConfigure`
    // twice and migrate an already migrated array a second time.
    if (proto.__was_widget_migration_wrapped) return;
    proto.__was_widget_migration_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (order) migrateWidgetValues(this, order);
        if (booleans || numbers || renamed) {
          const originalOnConfigure = this.onConfigure;
          this.onConfigure = function (...args) {
            const configured = originalOnConfigure?.apply(this, args);
            if (booleans) coerceBooleans(this, booleans);
            if (numbers) numbersToBooleans(this, numbers);
            if (renamed) renameOptions(this, renamed);
            return configured;
          };
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to arm ${nodeData.name}'s migration:`, error);
      }
      return result;
    };
  },
});
