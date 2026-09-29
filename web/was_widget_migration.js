/**
 * Saved v2 widget values, put back on the widget they were written for.
 *
 * `V2_WIDGET_ORDER` holds each node's earlier widget order. `V2_BOOLEAN_WIDGETS` names widgets that
 * took "false" and "true" before becoming a checkbox. `RENAMED_OPTIONS` maps renamed options.
 */

import { app } from "../../scripts/app.js";
import { migrateWidgetValues } from "./interface/widget_migration.js";

const EXT_NAME = "WASNodeSuite.WidgetMigration";

// Node id -> the widgets an earlier save's `widgets_values` holds, in the order it holds them. Read
// left to right against the node's current widgets: anything the node has that is not named here
// is new, and takes its default rather than a value that was meant for the widget before it.
const V2_WIDGET_ORDER = {
  // Three Custom Material took a typed body before it took a module file. The saved body is
  // dropped rather than landing on the module menu.
  "WASThreeCustomMaterial": ["javascript", "module"],
  "WASImageGradientMapNative": ["flip_left_right"],
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
  // then a source, then a header and footer menu. One earlier order per layout a save may hold.
  "WASMiniMaxH3Conditioning": [
    [
      "mode", "aspect_ratio", "megapixels", "width", "height",
      "prompt_header", "prompt_footer",
      ...Array.from({ length: 24 }, (unused, index) => [
        `prompt_${index + 1}`, `duration_${index + 1}`, `overlap_${index + 1}`,
      ]).flat(),
    ],
    [
      "mode", "aspect_ratio", "megapixels", "width", "height",
      "prompt_header", "prompt_footer",
      ...Array.from({ length: 24 }, (unused, index) => [
        `prompt_${index + 1}`, `duration_${index + 1}`, `overlap_${index + 1}`,
        `continuity_${index + 1}`,
      ]).flat(),
    ],
    [
      "mode", "aspect_ratio", "megapixels", "width", "height",
      "prompt_header", "prompt_footer",
      ...Array.from({ length: 24 }, (unused, index) => [
        `prompt_${index + 1}`, `duration_${index + 1}`, `overlap_${index + 1}`,
        `continuity_${index + 1}`,
      ]).flat(),
      "ref_image_size",
    ],
    [
      "mode", "aspect_ratio", "megapixels", "width", "height",
      "prompt_header", "prompt_footer",
      ...Array.from({ length: 24 }, (unused, index) => [
        `prompt_${index + 1}`, `duration_${index + 1}`, `overlap_${index + 1}`,
        `continuity_${index + 1}`, `source_${index + 1}`,
      ]).flat(),
      "ref_image_size",
    ],
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
  "Text Dictionary New": [
    "key_1", "value_1", "key_2", "value_2", "key_3", "value_3", "key_4", "value_4",
    "key_5", "value_5", "key_6", "value_6", "key_7", "value_7", "key_8", "value_8",
    "list_values",
  ],
  "Text Add Tokens": ["print_current_tokens"],
  "Text Concatenate": ["clean_whitespace"],
};

// Continuity menus on the H3 nodes, and the options each once offered under another name.
const H3_CONTINUITY_RENAMED = { reference: "reference (video)" };

// Node id -> `[widget names, {saved option: current option}]` for combos whose options were renamed.
const RENAMED_OPTIONS = {
  WASH3ExtendWindow: [["continuity"], H3_CONTINUITY_RENAMED],
  WASMiniMaxH3Conditioning: [
    Array.from({ length: 24 }, (unused, index) => `continuity_${index + 1}`),
    H3_CONTINUITY_RENAMED,
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

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const order = V2_WIDGET_ORDER[nodeData?.name];
    const booleans = V2_BOOLEAN_WIDGETS[nodeData?.name];
    const renamed = RENAMED_OPTIONS[nodeData?.name];
    if (!order && !booleans && !renamed) return;

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
        if (booleans || renamed) {
          const originalOnConfigure = this.onConfigure;
          this.onConfigure = function (...args) {
            const configured = originalOnConfigure?.apply(this, args);
            if (booleans) coerceBooleans(this, booleans);
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
