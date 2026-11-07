/**
 * Node previews in search and the node library, listing repeated rows and slots as a newly
 * placed node draws them.
 *
 * A folded input stays on the definition under its name and a folded output at its slot.
 */
import { app } from "../../scripts/app.js";
import { foldedNames, foldsOnPlacing } from "./interface/node_preview.js";

const EXT_NAME = "WASNodeSuite.NodePreviews";

/**
 * The class the frontend files node definitions as, read off a registered node.
 *
 * @param {string} nodeId - A registered node id.
 * @returns {Function|null} The class, or null where the node carries none.
 */
function definitionClass(nodeId) {
  const found = globalThis.LiteGraph?.registered_node_types?.[nodeId]?.nodeData?.constructor;
  return typeof found === "function" && found !== Object ? found : null;
}

/**
 * Leave a node's folded inputs and outputs out of every listing of its definition.
 *
 * @param {object} definition - The definition the frontend files, as built from the server's.
 * @param {{inputs: Set<string>, outputs: Set<string>}} folded - The names to leave out.
 * @returns {void}
 */
function listAsPlaced(definition, folded) {
  const inputs = definition.inputs;
  if (inputs && typeof inputs === "object") {
    for (const name of folded.inputs) {
      const held = Object.getOwnPropertyDescriptor(inputs, name);
      // Kept under its name, skipped by anything that walks the inputs.
      if (held?.configurable) Object.defineProperty(inputs, name, { enumerable: false });
    }
  }
  const outputs = definition.outputs;
  if (!Array.isArray(outputs) || !folded.outputs.size) return;
  const listed = outputs.filter((output) => !folded.outputs.has(output?.name));
  if (listed.length === outputs.length) return;
  // Every slot is still at its index; only a mapping over the outputs skips the folded ones.
  Object.defineProperty(outputs, "map", {
    configurable: true,
    enumerable: false,
    writable: true,
    value: (callback, thisArg) => listed.map(callback, thisArg),
  });
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterVueAppNodeDefs(definitions) {
    if (!Array.isArray(definitions)) return;
    definitions.forEach((served, index) => {
      const nodeId = served?.name;
      if (!nodeId || !foldsOnPlacing(nodeId)) return;
      try {
        const Definition = definitionClass(nodeId);
        if (!Definition) return;
        const definition = served instanceof Definition ? served : new Definition(served);
        listAsPlaced(definition, foldedNames(nodeId, definition));
        definitions[index] = definition;
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to fit the preview of ${nodeId}:`, error);
      }
    });
  },
});
