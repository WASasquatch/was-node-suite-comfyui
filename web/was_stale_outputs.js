/**
 * Another node's pictures, cleared off this pack's nodes that cannot have made them.
 */

// The frontend keys `app.nodeOutputs` on the node id and keeps an entry after the node that
// made it is gone, while clearing a graph restarts the numbering. A node handed a recycled id
// inherits the previous occupant's pictures and draws them. Reproduced with core nodes alone,
// so this guard covers only what this pack registers.

import { app } from "../../scripts/app.js";
import { dropForeignOutputs } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.StaleOutputs";

// Only this pack's own nodes. The same thing happens to a core node, and correcting those
// would reach past what this pack is answerable for.
const OWNED = "WAS Suite";

/**
 * Whether a node could have produced the pictures held under its id.
 *
 * @param {object} nodeData - The node definition the frontend registered.
 * @returns {boolean} True where an `images` entry can only be another node's.
 */
function cannotDraw(nodeData) {
  // An output node publishes its pictures through the same store, so its entry is its own.
  if (nodeData?.output_node) return false;
  const outputs = nodeData?.output ?? [];
  return !outputs.some((entry) => {
    const type = Array.isArray(entry) ? entry[0] : entry;
    return type === "IMAGE" || type === "MASK";
  });
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData?.category ?? "").startsWith(OWNED)) return;
    if (!cannotDraw(nodeData)) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise wrap twice.
    if (proto.__was_stale_outputs_wrapped) return;
    proto.__was_stale_outputs_wrapped = true;

    const originalOnAdded = proto.onAdded;
    proto.onAdded = function (...args) {
      const added = originalOnAdded?.apply(this, args);
      try {
        dropForeignOutputs(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to clear another node's pictures:`, error);
      }
      return added;
    };
  },
});
