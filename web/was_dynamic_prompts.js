/**
 * The `dynamic_prompts` switch on a text box, honoured where the value is serialised.
 *
 * ComfyUI resolves a `{red|blue}` alternation on the way out of the canvas, before the
 * backend sees anything. Switched off, the text is sent exactly as typed.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.DynamicPrompts";

// The core extension that resolves a widget declaring `dynamicPrompts`.
const CORE_EXT_NAME = "Comfy.DynamicPrompts";

// Node id -> the text widget the switch beside it governs.
const GOVERNED = {
  "Text Multiline": { text: "text", toggle: "dynamic_prompts" },
};

let coreResolve;

/**
 * Core's resolver for a `{a|b}` text, taken from its extension through a stand-in widget.
 *
 * @returns {((text: string) => Promise<string>)|null} The resolver, or null where this frontend
 *   has none.
 */
function coreResolver() {
  if (coreResolve !== undefined) return coreResolve;
  coreResolve = null;
  try {
    const core = (app.extensions ?? []).find((extension) => extension?.name === CORE_EXT_NAME);
    const stand = { name: "text", value: "", dynamicPrompts: true };
    core?.nodeCreated?.({ widgets: [stand] });
    const serialize = stand.serializeValue;
    if (typeof serialize === "function") {
      coreResolve = async (text) => {
        stand.value = text;
        return serialize.call(stand, null, 0);
      };
    }
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to reach ${CORE_EXT_NAME}:`, error);
  }
  return coreResolve;
}

/**
 * Read a widget's value off a node.
 *
 * @param {object} node - The node to read.
 * @param {string} name - The widget's name.
 * @returns {*} Its value, or undefined where there is no such widget.
 */
function widgetValue(node, name) {
  return (node.widgets ?? []).find((candidate) => candidate.name === name)?.value;
}

app.registerExtension({
  name: EXT_NAME,

  nodeCreated(node) {
    const governed = GOVERNED[node?.comfyClass ?? node?.type];
    if (!governed) return;
    const widget = (node.widgets ?? []).find((candidate) => candidate.name === governed.text);
    if (!widget || widget.__was_dynamic_prompts) return;
    widget.__was_dynamic_prompts = true;

    // A serializer held behind a setter runs this one as its next step and is never called back.
    const descriptor = Object.getOwnPropertyDescriptor(widget, "serializeValue");
    const earlier = descriptor?.set ? null : widget.serializeValue?.bind(widget);

    widget.serializeValue = async (workflowNode, widgetIndex) => {
      const on = widgetValue(node, governed.toggle) !== false;
      if (on && earlier) return earlier(workflowNode, widgetIndex);
      const text = widget.value;
      let prompt = text;
      if (on && typeof text === "string") {
        const resolve = coreResolver();
        if (resolve) prompt = await resolve(text);
      }
      if (workflowNode?.widgets_values) workflowNode.widgets_values[widgetIndex] = prompt;
      return prompt;
    };
  },
});
