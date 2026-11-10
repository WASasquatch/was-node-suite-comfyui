/**
 * Node previews in search and the node library, drawn as a newly placed node draws itself.
 *
 * Folded inputs and outputs keep their names and slots. Folded widgets stay listed. A widget
 * taking other wires is listed under its widget's type.
 */
import { app } from "../../scripts/app.js";
import { foldedNames, foldsOnPlacing } from "./interface/node_preview.js";

const EXT_NAME = "WASNodeSuite.NodePreviews";

// Only this pack's own nodes.
const OWNED = "WAS Suite";

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
 * Whether an input is drawn as a widget on a placed node while its type names other wires too.
 *
 * @param {string|string[]} type - The input's type, such as `COMBO,STRING`.
 * @param {object|undefined} options - Its options.
 * @returns {boolean} True for a widget input that also takes another type of wire.
 */
function widenedWidget(type, options) {
  return typeof type === "string" && type.includes(",") && typeof options?.widgetType === "string"
    && !options.forceInput;
}

/**
 * Whether a served definition holds a widget input that also takes other wires.
 *
 * @param {object} served - A node definition as the server sends it.
 * @returns {boolean} True where any input is such a widget.
 */
function holdsWidenedWidget(served) {
  for (const group of ["required", "optional"]) {
    for (const spec of Object.values(served?.input?.[group] ?? {})) {
      if (Array.isArray(spec) && widenedWidget(spec[0], spec[1])) return true;
    }
  }
  return false;
}

/**
 * Lay a node's definition out as a newly placed node draws it.
 *
 * @param {object} definition - The definition the frontend files, as built from the server's.
 * @param {{inputs: Set<string>, outputs: Set<string>, widgets: Set<string>}} folded - The names
 *   to leave out.
 * @returns {void}
 */
function listAsPlaced(definition, folded) {
  const inputs = definition.inputs;
  if (inputs && typeof inputs === "object") {
    for (const name of Object.keys(inputs)) {
      const held = Object.getOwnPropertyDescriptor(inputs, name);
      const spec = held?.value;
      if (!held?.configurable || !spec || typeof spec !== "object") continue;
      const widened = widenedWidget(spec.type, spec);
      const hidden = folded.widgets.has(name);
      if (!widened && !hidden) continue;
      // A copy typed as its widget, and flagged advanced where folded: the drawn preview skips
      // an advanced input and the input lists keep it.
      const fitted = { ...spec };
      if (widened) fitted.type = spec.widgetType;
      if (hidden) fitted.advanced = true;
      Object.defineProperty(inputs, name, { value: fitted });
    }
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
      if (!nodeId) return;
      const owned = String(served?.category ?? "").startsWith(OWNED);
      if (!foldsOnPlacing(nodeId) && !(owned && holdsWidenedWidget(served))) return;
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
