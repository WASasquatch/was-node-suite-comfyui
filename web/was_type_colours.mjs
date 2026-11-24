/**
 * A colour for each wire type this pack declares.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.TypeColours";
const SETTING_ID = "WAS.TypeColours.Enabled";

// Type -> colour.
const COLOURS = {
  // Numbers and containers.
  NUMBER: "#4DB6AC",
  SEED: "#26A69A",
  LIST: "#9CCC65",
  DICT: "#AED581",

  // Regions measured on a picture.
  CROP_DATA: "#4DD0E1",
  IMAGE_BOUNDS: "#00ACC1",

  // Documents and archives.
  DOC: "#BCAAA4",
  ZIP: "#8D6E63",

  // Colour handling.
  LUT: "#CE93D8",

  // Graph plumbing.
  BUS: "#78909C",
  WAS_LOOP: "#90A4AE",

  // What a read measured.
  WAS_VIDEO_METADATA: "#7986CB",

  // Settings handed to a node rather than data.
  SAM_PARAMETERS: "#A1887F",
  WAS_LORA_MERGE_OPTIONS: "#B0BEC5",
  CONDITIONING_SEQ: "#FFB74D",

  // Loaded weights.
  BLIP_MODEL: "#F06292",
  CLIPSEG_MODEL: "#BA68C8",
  MIDAS_MODEL: "#9575CD",
  SAM_MODEL: "#7E57C2",
  YUNET_MODEL: "#5C6BC0",
};

/**
 * Whether the colours are applied at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * The wire type colours the active palette leaves blank.
 *
 * @returns {object|null} The palette's slot colours, or null when no palette is readable.
 */
function paletteSlots() {
  try {
    return app?.extensionManager?.colorPalette?.getActiveColorPalette?.()?.colors?.node_slot
      ?? null;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read the active colour palette:`, error);
    return null;
  }
}

/**
 * Name the colours on the palette and write them into the tables the canvas draws from.
 *
 * @returns {{palette: number, tables: number}} How many entries each took.
 */
function apply() {
  const slots = paletteSlots();
  const tables = [
    app?.canvas?.default_connection_color_byType,
    window.LiteGraph?.LGraphCanvas?.link_type_colors,
  ];
  let palette = 0;
  let written = 0;
  for (const [type, colour] of Object.entries(COLOURS)) {
    if (slots && !slots[type]) {
      slots[type] = colour;
      palette += 1;
    }
    for (const table of tables) {
      if (table && !table[type]) {
        table[type] = colour;
        written += 1;
      }
    }
  }
  if (written) app?.canvas?.setDirty?.(true, true);
  return { palette, tables: written };
}

/**
 * Re-apply the colours after each palette load, since another palette names none of them.
 *
 * @returns {boolean} True once the palette store is wrapped.
 */
function reapplyOnPaletteChange() {
  const store = app?.extensionManager?.colorPalette;
  if (typeof store?.loadColorPalette !== "function") return false;
  const load = store.loadColorPalette.bind(store);
  store.loadColorPalette = async (...args) => {
    const result = await load(...args);
    try {
      apply();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to re-colour after a palette change:`, error);
    }
    return result;
  };
  return true;
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Appearance", "Colour the pack's wires"],
      name: "Enable custom data type colours",
      tooltip:
        "Draw each of this pack's own wire types in its own colour, so a crop window is told "
        + "from a document or a lookup table at a glance. Off, they all draw in the palette's "
        + "default. A colour palette naming one of these types keeps its own colour either "
        + "way. Applies on the next page load.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async setup() {
    if (!enabled()) return;
    try {
      apply();
      const watching = reapplyOnPaletteChange();
      if (!watching) console.debug(`[${EXT_NAME}] palette changes are unwatched`);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to colour the pack's wire types:`, error);
    }
  },

  async afterConfigureGraph() {
    if (!enabled()) return;
    try {
      const { palette, tables } = apply();
      if (palette || tables) {
        console.debug(
          `[${EXT_NAME}] named ${palette} colour(s) on the palette, `
          + `filled ${tables} table entr(ies)`,
        );
      }
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to colour the pack's wire types:`, error);
    }
  },
});
