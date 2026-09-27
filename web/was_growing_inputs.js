/**
 * Repeated inputs that appear as they are filled.
 *
 * Registers the nodes that declare the same input several times with `growWidgets`, which draws
 * the ones in use and the next empty one.
 */

import { app } from "../../scripts/app.js";
import { growWidgets } from "./interface/grow.js";
import { addSectionHeader } from "./interface/decoration.js";
import { addRowHeaders, rowColourMenu } from "./interface/row_headers.js";

const EXT_NAME = "WASNodeSuite.GrowingInputs";

// Node id -> the repeated inputs, in the order they are drawn, one per group. Adding a node here
// is the whole of wiring it up.
// The condition inputs on Condition Chain and Boolean Reduce.
const CONDITION_SLOTS = Array.from({ length: 26 }, (unused, index) => `condition_${String.fromCharCode(97 + index)}`);

// Rows either Power LoRA node declares, matching MAX_ROWS in
// nodes/extras/lora/power_lora_loader.py.
const POWER_LORA_MAX_ROWS = 26;

const POWER_LORA_ROWS = {
  groups: Array.from({ length: POWER_LORA_MAX_ROWS }, (unused, index) => [
    `lora_${index + 1}_enabled`,
    `lora_${index + 1}`,
    `lora_${index + 1}_weight`,
  ]),
  empty: ["None"],
  minVisible: 1,
  decidesAt: 1,
  header: { name: "was_row_header", title: "Selected LoRA's", before: "lora_1_enabled" },
};

/**
 * The value one of a node's widgets holds.
 *
 * @param {object} node - The node to read.
 * @param {string} name - The widget's name.
 * @returns {*} Its value, or undefined when the node has no such widget.
 */
function widgetValue(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name)?.value;
}

/**
 * Text of one segment's header on MiniMax H3 Conditioning.
 *
 * @param {object} node - The node to read.
 * @param {number} row - Segment number, from 1.
 * @returns {string} As `Segment 2 · 7.0 s · carry` or `Segment 3 · 8.0 s · cut · footer only`.
 */
function h3SegmentTitle(node, row) {
  const parts = [`Segment ${row}`];
  const seconds = Number(widgetValue(node, `duration_${row}`));
  if (Number.isFinite(seconds)) parts.push(`${seconds.toFixed(1)} s`);
  const continuity = widgetValue(node, `continuity_${row}`);
  if (row === 1) parts.push("opening");
  else if (continuity) parts.push(String(continuity));
  const source = Number(widgetValue(node, `source_${row}`));
  if (row > 1 && Number.isFinite(source) && source !== -1 && source !== 0) {
    parts.push(`from ${source > 0 ? source : row + source}`);
  }
  const wrap = widgetValue(node, `header_footer_${row}`);
  if (wrap && wrap !== "both") parts.push(String(wrap));
  return parts.join(" · ");
}

// A prompt, a frame count, an overlap, a continuity, a source and the shared text it takes to a
// row on MiniMax H3 Conditioning.
// `decidesAt` names the prompt as the widget that says whether a row is in use.
const H3_PROMPT_ROWS = {
  groups: Array.from({ length: 24 }, (unused, index) => [
    `prompt_${index + 1}`,
    `duration_${index + 1}`,
    `overlap_${index + 1}`,
    `continuity_${index + 1}`,
    `source_${index + 1}`,
    `header_footer_${index + 1}`,
  ]),
  minVisible: 2,
  decidesAt: 0,
  // A coloured bar above each segment, folded away with it.
  rowHeaders: {
    name: (row) => `was_segment_header_${row}`,
    title: h3SegmentTitle,
    noun: "Segment",
  },
};

// How many slots a lettered series declares.
const LETTERED_SLOTS = 24;

/**
 * The names of a lettered series, `stem_a` to `stem_x`.
 *
 * @param {string} stem - What each name starts with.
 * @returns {string[]} One name per slot.
 */
function lettered(stem) {
  return Array.from(
    { length: LETTERED_SLOTS },
    (unused, index) => `${stem}_${String.fromCharCode(97 + index)}`,
  );
}

const GROWING = {
  "Text List": lettered("text"),
  "Text Concatenate": lettered("text"),
  // These three also grow sockets, in `was_growing_sockets.js`, which draws the output each box
  // feeds. Both growers count a box as used the same way, so the two sides agree.
  // The frozen pair plus the slots appended for v3, revealed as each one fills.
  "Image Stitch": ["image_a", "image_b", "image_c", "image_d", "image_e", "image_f", "image_g", "image_h", "image_i", "image_j", "image_k", "image_l", "image_m", "image_n", "image_o", "image_p", "image_q", "image_r", "image_s", "image_t", "image_u", "image_v", "image_w", "image_x", "image_y", "image_z"],
  "Text String": ["text", ...lettered("text").slice(1)],
  "Text String Truncate": ["text", ...lettered("text").slice(1)],
  "CLIPSeg Batch Masking": lettered("text"),
  // Combos, not boxes. A combo always holds one of its options, so the node names the one that
  // means unused and this passes it on; without that every slot looks filled from the moment the
  // node is dropped and nothing ever folds.
  "Prompt Multiple Styles Selector": {
    names: Array.from({ length: LETTERED_SLOTS }, (unused, index) => `style${index + 1}`),
    empty: ["None"],
  },
  // The condition slots on the two reducers, revealed as each one is wired. `empty` names the
  // unticked box: a boolean widget always holds something, so without it every slot reads as
  // filled from the moment the node is dropped and none of them ever folds away.
  WASConditionChain: { names: CONDITION_SLOTS, minVisible: 2, empty: [false] },
  WASBooleanReduce: { names: CONDITION_SLOTS, minVisible: 2, empty: [false] },
  // Three widgets to a row rather than one. `decidesAt` names the file widget as the one that
  // says whether a row is in use: the switch and the strength always hold something, so a row
  // would never fold away if they had a vote. These also grow the per-row name outputs, in
  // `was_growing_sockets.js`, which counts a row used the same way.
  WASPowerLoraLoader: POWER_LORA_ROWS,
  WASNSPowerLoraMerger: POWER_LORA_ROWS,
  // Row 1 is the clip and every row after it is one extension pass.
  WASMiniMaxH3Conditioning: H3_PROMPT_ROWS,
};

// Nodes whose repeated inputs come in pairs rather than singly, as the pair names in the
// order they are drawn. A pair counts as used when either half holds something, so a name
// typed with no value yet still keeps the next pair from appearing.
const GROWING_PAIRS = {
  "Text Dictionary New": Array.from({ length: LETTERED_SLOTS }, (unused, index) => [
    `key_${index + 1}`,
    `value_${index + 1}`,
  ]),
};

app.registerExtension({
  name: EXT_NAME,

  getNodeMenuItems(node) {
    const rowHeaders = GROWING[node?.comfyClass]?.rowHeaders;
    if (!rowHeaders) return [];
    return rowColourMenu(node, rowHeaders.noun);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const entry = GROWING[nodeData?.name];
    const listed = Array.isArray(entry) ? entry : entry?.names;
    const groups = entry?.groups ?? GROWING_PAIRS[nodeData?.name] ?? listed?.map((name) => [name]);
    if (!groups) return;
    const options = {};
    if (entry && !Array.isArray(entry)) {
      if (entry.empty) options.empty = entry.empty;
      if (Number.isFinite(entry.minVisible)) options.minVisible = entry.minVisible;
      if (Number.isFinite(entry.decidesAt)) options.decidesAt = entry.decidesAt;
    }
    const header = Array.isArray(entry) ? null : entry?.header;
    const rowHeaders = Array.isArray(entry) ? null : entry?.rowHeaders;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap the prototype a
    // second time and fold twice on every keystroke.
    if (proto.__was_growing_inputs_wrapped) return;
    proto.__was_growing_inputs_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (header) addSectionHeader(this, header);
        let grown = groups;
        if (rowHeaders) {
          // Each header joins its row's group, after the widget that decides it, to fold with it.
          const names = addRowHeaders(this, { groups, ...rowHeaders });
          grown = groups.map((group, index) => [...group, names[index]]);
        }
        this.__was_refold = growWidgets(this, grown, options);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to grow ${nodeData.name}:`, error);
      }
      return result;
    };
  },
});
