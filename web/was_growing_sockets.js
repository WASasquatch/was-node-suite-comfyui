/**
 * Which nodes draw their repeated sockets only as they are wired.
 *
 * An entry grows as its slots are wired, or takes its count from `count` or `exactCount`,
 * or its visible set from `select`.
 */

import { app } from "../../scripts/app.js";
import { growSockets } from "./interface/grow_sockets.js";

const EXT_NAME = "WASNodeSuite.GrowingSockets";

const SLOTS = 8;

// One name per slot, on the input and the output of both nodes, so a slot is the same thing
// wherever it is read.
const VALUE_SLOTS = Array.from({ length: SLOTS }, (unused, index) => `value_${index + 1}`);

const TILE_SLOTS = Array.from({ length: 16 }, (unused, index) => `tile_${index + 1}`);

// How many slots a lettered series declares by default, and the letters it uses.
const LETTERED_SLOTS = 24;

/**
 * One group per slot of a lettered series, `stem_a` onwards.
 *
 * @param {string} stem - What each name starts with.
 * @param {number} [count] - How many slots the node declares, 24 by default.
 * @returns {string[][]} One single-name group per slot.
 */
function lettered(stem, count = LETTERED_SLOTS) {
  return Array.from({ length: count }, (unused, index) => [
    `${stem}_${String.fromCharCode(97 + index)}`,
  ]);
}

// One group per bare-letter slot, `a` to `x`, which is how Number Expression names its inputs.
const BARE_LETTER_SLOTS = Array.from({ length: LETTERED_SLOTS }, (unused, index) => [
  String.fromCharCode(97 + index),
]);

/**
 * Read a widget's number off a node.
 *
 * @param {object} node - The node to read.
 * @param {string} name - Widget name.
 * @param {number} fallback - Used when the widget is missing or unreadable.
 * @returns {number} The widget's value.
 */
function widgetNumber(node, name, fallback) {
  const widget = node.widgets?.find((candidate) => candidate.name === name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * How many of a node's repeated slots are in use, counting a typed box as readily as a wire.
 *
 * @param {object} node - The node holding them.
 * @param {string[][]} pairs - The slot groups, each naming the widgets and sockets that appear
 *   together.
 * @returns {number} One past the last group holding anything, so the next empty one is drawn,
 *   and never fewer than two, which is the floor `growWidgets` draws the boxes to. The two
 *   growers have to agree or a fresh node shows a box with no socket beside it.
 */
// A widget-backed input is not this grower's to move. Re-adding one builds a plain socket with
// no widget bound to it, which draws a second row beside the box it belongs to, so what is
// listed here is only ever what carries no widget: the outputs, and the picture inputs.
//
// Hiding the box is `growWidgets`'s job and it takes the socket with it. The two growers
// divide the work along that line.

// The outputs each text box feeds. Counted from the boxes, listed as outputs.
const TEXT_OUTPUTS = [["TEXT"], ...lettered("TEXT").slice(1)].map((group) =>
  group.map((name) => name.toUpperCase()),
);
const TEXT_BOXES = [["text"], ...lettered("text").slice(1)];

// The pictures, and the prompts that decide how many are drawn.
const CLIPSEG_IMAGES = lettered("image");
const CLIPSEG_PAIRS = lettered("image").map(([name], index) => [
  name,
  lettered("text")[index][0],
]);

function usedSlots(node, pairs) {
  let last = -1;
  pairs.forEach((names, index) => {
    const used = names.some((name) => {
      const widget = node.widgets?.find((candidate) => candidate.name === name);
      if (widget !== undefined) {
        const value = widget.value;
        return value !== "" && value !== null && value !== undefined;
      }
      // No widget of that name, so it is a plain socket and a link is what makes it used. An
      // output is never counted: its input decides whether it is drawn.
      const input = node.inputs?.find((candidate) => candidate.name === name);
      return Boolean(input?.link != null);
    });
    if (used) last = index;
  });
  return Math.max(2, last + 2);
}

// Every conditioning slot, as one group each. Link-only sockets with no widget, so this
// grower owns them rather than growWidgets.
const CONDITIONING_SLOTS = [
  ["conditioning_a"], ["conditioning_b"], ["conditioning_c"], ["conditioning_d"], ["conditioning_e"], ["conditioning_f"],
  ["conditioning_g"], ["conditioning_h"], ["conditioning_i"], ["conditioning_j"], ["conditioning_k"], ["conditioning_l"],
  ["conditioning_m"], ["conditioning_n"], ["conditioning_o"], ["conditioning_p"], ["conditioning_q"], ["conditioning_r"],
  ["conditioning_s"], ["conditioning_t"], ["conditioning_u"], ["conditioning_v"], ["conditioning_w"], ["conditioning_x"],
  ["conditioning_y"], ["conditioning_z"],
];

// One name output per LoRA row, matching MAX_ROWS in nodes/extras/lora/power_lora_loader.py.
// The count comes from the row widgets, not from wiring, so an unwired name still appears once
// its row names a file.
const LORA_MAX_ROWS = 26;
const LORA_NAME_SLOTS = Array.from({ length: LORA_MAX_ROWS }, (unused, index) => [
  `name_${index + 1}`,
]);

// The row widgets the count is read from, in slot order.
const LORA_ROW_WIDGETS = Array.from({ length: LORA_MAX_ROWS }, (unused, index) => [
  `lora_${index + 1}`,
  `lora_${index + 1}_enabled`,
]).flat();

/**
 * How many LoRA rows name a file and are switched on.
 *
 * @param {object} node - The Power LoRA Loader node.
 * @returns {number} Rows that will be applied. 0 where none is, which draws no name sockets.
 */
function enabledLoraRows(node) {
  const values = new Map((node?.widgets || []).map((widget) => [widget.name, widget.value]));
  let count = 0;
  for (let slot = 1; slot <= LORA_MAX_ROWS; slot += 1) {
    const file = values.get(`lora_${slot}`);
    if (typeof file !== "string" || !file || file === "None") continue;
    if (values.get(`lora_${slot}_enabled`) === false) continue;
    count += 1;
  }
  return count;
}

// The references `ref2va` reads. A reference video and its soundtrack are drawn as one pair.
const H3_REF_IMAGES = Array.from({ length: 9 }, (unused, index) => [`ref_image_${index + 1}`]);
const H3_REF_VIDEOS = Array.from({ length: 3 }, (unused, index) => [
  `ref_video_${index + 1}`,
  `ref_video_audio_${index + 1}`,
]);
const H3_REF_AUDIOS = Array.from({ length: 3 }, (unused, index) => [`ref_audio_${index + 1}`]);

// The pictures and sounds MiniMax H3 Conditioning reads, and which of them each mode draws. A
// mode that reads none draws no socket for them at all.
const H3_FRAME_SLOTS = [
  ["first_frame"],
  ["last_frame"],
  ["images"],
  ["audio_vae"],
  ...H3_REF_IMAGES,
  ...H3_REF_VIDEOS,
  ...H3_REF_AUDIOS,
];
// Which of those the chosen mode reads. `fl2va_batched` takes a whole batch and draws that
// instead of the two single frames.
const H3_MODE_SLOTS = {
  t2va: [],
  i2va: [0],
  fl2va: [0, 1],
  fl2va_batched: [2],
  ref2va: [3],
};
// Where each reference series starts in H3_FRAME_SLOTS.
const H3_REF_SERIES = [
  [4, H3_REF_IMAGES],
  [4 + H3_REF_IMAGES.length, H3_REF_VIDEOS],
  [4 + H3_REF_IMAGES.length + H3_REF_VIDEOS.length, H3_REF_AUDIOS],
];

/**
 * The slots of one reference series in use, plus one spare below the last of them.
 *
 * @param {object} node - The MiniMax H3 Conditioning node.
 * @param {number} start - Where the series starts in H3_FRAME_SLOTS.
 * @param {string[][]} series - Its groups, in order.
 * @returns {number[]} Indexes into H3_FRAME_SLOTS.
 */
function h3RefSlots(node, start, series) {
  const linked = new Set(
    (node?.inputs ?? [])
      .filter((socket) => socket.link !== null && socket.link !== undefined)
      .map((socket) => socket.name),
  );
  let lastUsed = -1;
  series.forEach((names, index) => {
    if (names.some((name) => linked.has(name))) lastUsed = index;
  });
  const count = Math.min(lastUsed + 2, series.length);
  return Array.from({ length: count }, (unused, index) => start + index);
}

/**
 * Which inputs the chosen mode reads.
 *
 * @param {object} node - The MiniMax H3 Conditioning node.
 * @returns {number[]} Indexes into H3_FRAME_SLOTS.
 */
function h3ModeSlots(node) {
  const widget = node?.widgets?.find((candidate) => candidate.name === "mode");
  const slots = [...(H3_MODE_SLOTS[widget?.value] ?? [])];
  if (widget?.value === "ref2va") {
    for (const [start, series] of H3_REF_SERIES) slots.push(...h3RefSlots(node, start, series));
  }
  return slots;
}

// The index switches' inputs, matching SLOT_NAMES in modules/logic/switch_index.py.
const INDEX_SWITCH_SLOTS = Array.from({ length: 26 }, (unused, index) => [
  `input_${String.fromCharCode(97 + index)}`,
]);

// Node id -> the slots that may come and go, in declared order. `growSockets` fits whichever
// side actually declares each name. An entry may be a bare array, or an object naming the
// widgets that decide the count.
const GROWING = {
  // Blends any number of conditionings; the slots are sockets, counted from what is wired.
  WASConditioningBlend: CONDITIONING_SLOTS,
  // One name output per switched-on row, counted from the row widgets rather than from wiring,
  // so a name appears as soon as its row names a file. One empty slot rests below that, as
  // everywhere else.
  WASPowerLoraLoader: {
    slots: LORA_NAME_SLOTS,
    watch: LORA_ROW_WIDGETS,
    count: (node) => enabledLoraRows(node),
  },
  // Picture inputs drawn by the mode that reads them rather than by wiring, so `t2va` shows
  // none and `i2va` shows only the frame it opens on. `ref2va` draws each reference series
  // as it is wired, one spare below the last.
  WASMiniMaxH3Conditioning: {
    slots: H3_FRAME_SLOTS,
    watch: ["mode"],
    select: (node) => h3ModeSlots(node),
  },
  // The index switches declare 26 link-only slots each, folded to the ones wired plus one.
  WASAnyIndexSwitch: INDEX_SWITCH_SLOTS,
  WASAnyFirstSwitch: INDEX_SWITCH_SLOTS,
  WASTensorImageIndexSwitch: INDEX_SWITCH_SLOTS,
  WASModelIndexSwitch: INDEX_SWITCH_SLOTS,
  WASForLoopOpen: VALUE_SLOTS,
  WASForLoopClose: VALUE_SLOTS,
  WASWhileLoopOpen: VALUE_SLOTS,
  WASWhileLoopClose: VALUE_SLOTS,
  // Repeated link-only inputs. These carry no widget, so `growWidgets` never sees them and it is
  // this grower, which works on sockets, that draws them as they are wired.
  // The batchers declare a slot per letter, so their lists run two further than the rest.
  "Image Batch": lettered("images", 26),
  "Mask Batch": lettered("masks", 26),
  "Latent Batch": lettered("latent", 26),
  // The expression variables, drawn as they are wired. They are link-only, so the widget
  // grower never sees them.
  WASNumberExpression: BARE_LETTER_SLOTS,
  "Masks Combine Regions": lettered("mask"),
  "Text List Concatenate": lettered("list"),
  "Text Dictionary Update": lettered("dictionary"),
  // A text field and the output it feeds, drawn as a pair: revealing the box without the socket
  // would leave the text with nowhere to be read from. Widget-backed inputs sit in `node.inputs`
  // alongside real sockets, so one group names both sides.
  "Text String": {
    slots: TEXT_OUTPUTS,
    watch: TEXT_BOXES.flat(),
    count: (node) => usedSlots(node, TEXT_BOXES),
  },
  "Text String Truncate": {
    slots: TEXT_OUTPUTS,
    watch: TEXT_BOXES.flat(),
    count: (node) => usedSlots(node, TEXT_BOXES),
  },
  // A picture and the prompt that segments it. Neither is any use alone, so the pair appears
  // together, counted from the wire on one and the text in the other. The three outputs are
  // fixed, being batches of whatever was filled in.
  "CLIPSeg Batch Masking": {
    slots: CLIPSEG_IMAGES,
    watch: lettered("text").flat(),
    count: (node) => usedSlots(node, CLIPSEG_PAIRS),
  },
  // One output per tile, so the grid decides how many are drawn rather than the wiring, and
  // the sockets are redrawn whenever either widget named in `watch` changes.
  WASImageTileExtractGrid: {
    slots: TILE_SLOTS,
    watch: ["columns", "rows"],
    count: (node) => widgetNumber(node, "columns", 2) * widgetNumber(node, "rows", 2),
  },
};

/**
 * Grow one node's sockets, taking the count from its widgets when it has one.
 *
 * @param {object} node - The node to grow.
 * @param {object} entry - Its `GROWING` entry, already normalised to the object form.
 * @returns {() => void} The refit to call again when a watched widget changes.
 */
function apply(node, entry) {
  const options = {};
  if (entry.select) {
    // A node whose visible set changes rather than only its length, as a mode swapping one
    // picture input for another.
    options.select = () => entry.select(node);
  }
  if (entry.exactCount) {
    // A count that may be nought, where a widget rather than the wiring says how many sockets
    // belong on the node at all. A wired socket is still kept.
    options.exactCount = () => entry.exactCount(node);
  } else if (entry.count) {
    // Passed as a function, not a number: `growSockets` reads the declaration off the node when
    // it is called, so calling it a second time would capture a node it had already shrunk and
    // lose the sockets past that point for good. A wired socket is never hidden either way, so
    // this is a floor rather than a limit.
    options.minVisible = () => Math.max(1, entry.count(node));
  }
  return growSockets(node, entry.slots, options);
}

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const declared = GROWING[nodeData?.name];
    if (!declared) return;
    const entry = Array.isArray(declared) ? { slots: declared } : declared;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise capture the
    // declaration a second time from a node that has already been shrunk.
    if (proto.__was_growing_sockets_wrapped) return;
    proto.__was_growing_sockets_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const refit = apply(this, entry);
        for (const name of entry.watch ?? []) {
          const widget = this.widgets?.find((candidate) => candidate.name === name);
          if (!widget) continue;
          const originalCallback = widget.callback;
          widget.callback = (...args) => {
            const answer = originalCallback?.apply(widget, args);
            try {
              refit();
              this.setDirtyCanvas?.(true, true);
            } catch (error) {
              console.error(`[${EXT_NAME}] Failed to regrow ${nodeData.name}:`, error);
            }
            return answer;
          };
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to grow ${nodeData.name}:`, error);
      }
      return result;
    };
  },
});
