/**
 * Presenting a script node as the module it picked.
 *
 * The slots a module's header asks for take its names, and the rest are folded away.
 */
// Every widget is saved and sent whether it is drawn or not.

import { app } from "../../../scripts/app.js";
import { growSockets } from "../interface/grow_sockets.js";

const EXT_NAME = "WASNodeSuite.ThreeModuleUI";
const ROUTE = "/was/threejs/api/module";

// The widget holding the module.
const CONTROL = "module";

// Slot names a module's declared inputs are mapped onto, by kind, in declared order.
const SLOTS = {
  texture: ["texture1", "texture2", "texture3", "texture4"],
  geometry: ["geometry1", "geometry2"],
  material: ["material1", "material2"],
  object: ["object1", "object2"],
};

// Every node drawing a module picker.
const NODE_NAMES = ["WASThreeCustomMaterial"];

/** What each module label declared, so switching back and forth costs one fetch. */
const known = new Map();

// What the menu says when the read roots hold no module.
const NO_MODULES = "no module files found";

/**
 * Whether a menu value names a module.
 *
 * @param {object} widget - The module widget.
 * @returns {boolean} True when a module is picked.
 */
function picked(widget) {
  const value = String(widget?.value ?? "").trim();
  return Boolean(value) && value !== NO_MODULES;
}

/**
 * What one module declares, from the server or from what was already asked.
 *
 * @param {string} label - The module menu's value.
 * @returns {Promise<object|null>} The declaration, or null where it could not be read.
 */
async function declarationFor(label) {
  if (known.has(label)) return known.get(label);
  let answer = null;
  try {
    const response = await fetch(`${ROUTE}?label=${encodeURIComponent(label)}`, {
      cache: "no-store",
    });
    if (response.ok) {
      const body = await response.json();
      answer = body?.ok ? body : null;
      if (!body?.ok) {
        console.warn(`[WAS ThreeJS] ${label} could not be read: ${body?.error ?? "unknown"}`);
      }
    }
  } catch (error) {
    console.warn(`[WAS ThreeJS] ${label} could not be read.`, error);
  }
  known.set(label, answer);
  return answer;
}

/**
 * Put every slot back under its own name and draw it.
 *
 * @param {object} node - The node to reset.
 * @returns {void}
 */
function plainSlots(node) {
  for (const names of Object.values(SLOTS)) {
    for (const name of names) {
      const slot = (node.inputs ?? []).find((input) => input.name === name);
      if (slot) slot.label = undefined;
    }
  }
}

/**
 * Name the slots a module asked for and fold away the ones it did not.
 *
 * @param {object} node - The node to lay out.
 * @param {object} declared - The module's declaration.
 * @returns {void}
 */
function namedSlots(node, declared) {
  const wanted = new Map();
  const counts = {};
  for (const wire of declared.wires ?? []) {
    const names = SLOTS[wire.kind];
    if (!names) continue;
    const index = counts[wire.kind] ?? 0;
    counts[wire.kind] = index + 1;
    if (index < names.length) wanted.set(names[index], wire.name);
  }
  for (const names of Object.values(SLOTS)) {
    for (const name of names) {
      const slot = (node.inputs ?? []).find((input) => input.name === name);
      if (!slot) continue;
      slot.label = wanted.get(name);
    }
  }
}

/**
 * Draw the node as its module asks, or as itself where none is picked.
 *
 * @param {object} node - The node to present.
 * @returns {Promise<void>} Resolves once the layout is settled.
 */
async function present(node) {
  const control = (node.widgets ?? []).find((widget) => widget.name === CONTROL);
  if (!control) return;

  const declared = picked(control) ? await declarationFor(control.value) : null;
  if (declared) {
    namedSlots(node, declared);
    node.__was_module_textures = (declared.wires ?? []).filter(
      (wire) => wire.kind === "texture"
    ).length;
  } else {
    plainSlots(node);
    node.__was_module_textures = SLOTS.texture.length;
  }
  node.__was_module_refit?.();
  node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: EXT_NAME,
  async nodeCreated(node) {
    if (!NODE_NAMES.includes(node.comfyClass ?? node.type)) return;
    const control = (node.widgets ?? []).find((widget) => widget.name === CONTROL);
    if (!control) return;

    node.__was_module_textures = SLOTS.texture.length;
    node.__was_module_refit = growSockets(node, SLOTS.texture, {
      minVisible: 1,
      exactCount: () => node.__was_module_textures ?? SLOTS.texture.length,
    });

    const originalCallback = control.callback;
    control.callback = function (...args) {
      const answer = originalCallback?.apply(this, args);
      present(node);
      return answer;
    };

    // configure writes every value at once and calls no widget callback.
    const originalConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const answer = originalConfigure?.apply(this, args);
      present(node);
      return answer;
    };

    present(node);
  },
});
