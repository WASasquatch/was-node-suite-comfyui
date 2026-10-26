/**
 * What a wired input will receive, worked out on the page from the nodes upstream of it.
 *
 * Primitives, constants, Resolution Selector and number operations are read from their widgets;
 * reroutes are followed through.
 */

import { resolve as resolveResolution } from "./image_resolution.js";
import { floorMod, truncate } from "./python_arithmetic.js";

const LOG_NAME = "WASNodeSuite.Upstream";

// Links followed before the walk gives up, which also bounds a cycle.
const MAX_DEPTH = 24;

// Nodes one read works out at most; past it the value reads as unknown.
const MAX_VISITS = 512;

// One read's visits and the answers it has worked out, by the path set it started with.
const READ = new WeakMap();

// The id a link carries when it starts at a subgraph's own input.
const SUBGRAPH_INPUT_ID = -10;

// Frontend nodes that hand their one input straight on.
const PASS_THROUGH = new Set(["Reroute"]);

// Frontend nodes whose first widget is the value they answer.
const WIDGET_PRIMITIVES = new Set(["PrimitiveNode"]);

/** The answer for a value nothing on the page can tell. */
export const UNKNOWN = Object.freeze({ known: false, value: undefined, from: "" });

/**
 * A known answer.
 *
 * @param {*} value - The value.
 * @param {string} from - The node it was read from, as its title.
 * @returns {{known: boolean, value: *, from: string}} The answer.
 */
function known(value, from) {
  return { known: true, value, from };
}

/**
 * A node's widget by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|null} The widget.
 */
function widgetOf(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name) ?? null;
}

/**
 * The link feeding one input slot.
 *
 * @param {object} node - The node.
 * @param {number} slot - The input's index.
 * @returns {object|null} The link, or null when nothing is wired.
 */
function linkInto(node, slot) {
  if (typeof node?.getInputLink === "function") return node.getInputLink(slot) ?? null;
  const id = node?.inputs?.[slot]?.link;
  if (id == null) return null;
  const links = node?.graph?.links;
  return (typeof links?.get === "function" ? links.get(id) : links?.[id]) ?? null;
}

/**
 * Whether an input is wired.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {boolean} True when a link feeds it.
 */
export function isWired(node, name) {
  const slot = node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
  return slot >= 0 && node.inputs[slot].link != null;
}

/**
 * The node a wired input comes from, following reroutes, and the output slot it leaves on.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {{node: object, slot: number}|null} The source, or null when unwired or unreachable.
 */
export function sourceOf(node, name) {
  const slot = node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
  if (slot < 0) return null;
  let link = linkInto(node, slot);
  for (let depth = 0; link && depth < MAX_DEPTH; depth += 1) {
    if (link.origin_id === SUBGRAPH_INPUT_ID) return null;
    const origin = node.graph?.getNodeById?.(link.origin_id);
    if (!origin) return null;
    if (!isReroute(origin)) return { node: origin, slot: link.origin_slot };
    link = linkInto(origin, 0);
  }
  return null;
}

/**
 * Whether a node only passes its one input through.
 *
 * @param {object} node - The node.
 * @returns {boolean} True for the frontend's reroute node.
 */
function isReroute(node) {
  return PASS_THROUGH.has(node?.type);
}

/**
 * A number read as python's `float` would take it from a text box.
 *
 * @param {*} text - The text.
 * @returns {number} The number, or NaN.
 */
function pythonNumber(text) {
  const cleaned = String(text ?? "").trim();
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(cleaned)) return Number.NaN;
  return Number(cleaned);
}

// What each known node answers on each output slot, from the values on its inputs.
const READERS = {
  PrimitiveInt: (read) => [read("value")],
  PrimitiveFloat: (read) => [read("value")],
  WASResolutionSelector: (read) => {
    const [width, height] = resolveResolution(
      read("aspect_ratio"), read("size_by"), read("size"), read("megapixels"), read("multiple_of"),
    );
    return [width, height, width / height, (width * height) / 1_000_000];
  },
  "Constant Number": (read) => {
    const type = read("number_type");
    let number = read("number");
    const text = read("number_as_text");
    if (typeof text === "string" && text.trim()) {
      if (type === "bool") return undefined;
      number = pythonNumber(text);
      if (!Number.isFinite(number)) return undefined;
      if (type === "integer") number = truncate(number);
    }
    if (typeof number !== "number" || !Number.isFinite(number)) return undefined;
    if (type === "integer") return [truncate(number), number, truncate(number)];
    if (type === "bool") {
      const flag = number > 0.5 ? 1 : 0;
      return [flag, flag, flag];
    }
    return [number, number, truncate(number)];
  },
  "Number Operation": (read) => {
    const a = read("number_a");
    const b = read("number_b");
    if (typeof a !== "number" || typeof b !== "number") return undefined;
    const result = operate(read("operation"), a, b);
    return Number.isFinite(result) ? [result, result, truncate(result)] : undefined;
  },
  "Number Multiple Of": (read) => {
    let number = read("number");
    const multiple = read("multiple");
    if (typeof number !== "number" || typeof multiple !== "number" || multiple === 0) return undefined;
    if (floorMod(number, multiple) !== 0) number = Math.floor(number / multiple) * multiple + multiple;
    return [number, number, truncate(number)];
  },
  "Number to Int": (read) => {
    const number = read("number");
    return typeof number === "number" ? [truncate(number)] : undefined;
  },
};

/**
 * One of Number Operation's operations.
 *
 * @param {string} name - The operation's menu entry.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The result, or NaN where python would raise.
 */
function operate(name, a, b) {
  switch (name) {
    case "addition": return a + b;
    case "subtraction": return a - b;
    case "multiplication": return a * b;
    case "division": return b === 0 ? Number.NaN : a / b;
    case "floor division": return b === 0 ? Number.NaN : Math.floor(a / b);
    case "modulus": return b === 0 ? Number.NaN : floorMod(a, b);
    case "exponentiation": return a ** b;
    case "greater-than": return a > b ? 1 : 0;
    case "greater-than or equals": return a >= b ? 1 : 0;
    case "less-than": return a < b ? 1 : 0;
    case "less-than or equals": return a <= b ? 1 : 0;
    case "equals": return a === b ? 1 : 0;
    case "does not equal": return a !== b ? 1 : 0;
    default: return a;
  }
}

/**
 * The value a node with one number widget and nothing wired answers, for constants of other packs.
 *
 * @param {object} node - The node.
 * @returns {number|undefined} The widget's value, or undefined when the node is not that shape.
 */
function constantOf(node) {
  if ((node.outputs?.length ?? 0) !== 1) return undefined;
  if ((node.inputs ?? []).some((input) => input?.link != null)) return undefined;
  const numbers = (node.widgets ?? []).filter((widget) => typeof widget?.value === "number"
    && widget.name !== "control_after_generate" && !widget.hidden);
  return numbers.length === 1 ? numbers[0].value : undefined;
}

/**
 * The value a node answers on one output slot.
 *
 * @param {object} node - The node.
 * @param {number} slot - The output's index.
 * @param {Set<object>} walking - Nodes on the current path, which stops a cycle.
 * @param {number} depth - Links followed so far.
 * @returns {{known: boolean, value: *, from: string}} The answer.
 */
function outputValue(node, slot, walking, depth) {
  if (!node || depth > MAX_DEPTH || walking.has(node)) return UNKNOWN;
  let read = READ.get(walking);
  if (!read) {
    read = { visits: 0, answers: new Map() };
    READ.set(walking, read);
  }
  const key = `${node.id}:${slot}`;
  if (read.answers.has(key)) return read.answers.get(key);
  read.visits += 1;
  if (read.visits > MAX_VISITS) return UNKNOWN;
  const answer = readOutput(node, slot, walking, depth);
  read.answers.set(key, answer);
  return answer;
}

/**
 * The value a node answers on one output slot, worked out from its inputs and widgets.
 *
 * @param {object} node - The node.
 * @param {number} slot - The output's index.
 * @param {Set<object>} walking - Nodes on the current path, which stops a cycle.
 * @param {number} depth - Links followed so far.
 * @returns {{known: boolean, value: *, from: string}} The answer.
 */
function readOutput(node, slot, walking, depth) {
  if (isReroute(node)) return valueAt(node, 0, walking, depth + 1);
  const from = String(node.title || node.type || "");
  walking.add(node);
  try {
    const type = node.comfyClass ?? node.type;
    if (WIDGET_PRIMITIVES.has(type)) {
      const value = node.widgets?.[0]?.value;
      return value === undefined ? UNKNOWN : known(value, from);
    }
    const reader = READERS[type];
    if (reader) {
      let missing = false;
      const read = (name) => {
        const answer = inputValue(node, name, walking, depth + 1);
        if (!answer.known) missing = true;
        return answer.value;
      };
      const values = reader(read);
      if (missing || !Array.isArray(values) || values[slot] === undefined) return UNKNOWN;
      return known(values[slot], from);
    }
    const constant = constantOf(node);
    return constant === undefined ? UNKNOWN : known(constant, from);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read ${from || "a node"} upstream:`, error);
    return UNKNOWN;
  } finally {
    walking.delete(node);
  }
}

/**
 * The value arriving at one input slot.
 *
 * @param {object} node - The node.
 * @param {number} slot - The input's index.
 * @param {Set<object>} walking - Nodes on the current path.
 * @param {number} depth - Links followed so far.
 * @returns {{known: boolean, value: *, from: string}} The answer.
 */
function valueAt(node, slot, walking, depth) {
  const link = linkInto(node, slot);
  if (!link) {
    const widget = widgetOf(node, node.inputs?.[slot]?.name);
    return widget ? known(widget.value, "") : UNKNOWN;
  }
  if (link.origin_id === SUBGRAPH_INPUT_ID) return UNKNOWN;
  return outputValue(node.graph?.getNodeById?.(link.origin_id), link.origin_slot, walking, depth);
}

/**
 * What one input of a node holds or will receive.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @param {Set<object>} [walking] - Nodes already on the path, for a nested read.
 * @param {number} [depth] - Links already followed.
 * @returns {{known: boolean, value: *, from: string}} `from` names the node a wired value came from,
 *   and is empty for the node's own widget. `known` is false where only a run can tell.
 */
export function inputValue(node, name, walking = new Set(), depth = 0) {
  const slot = node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
  if (slot >= 0 && node.inputs[slot].link != null) return valueAt(node, slot, walking, depth);
  const widget = widgetOf(node, name);
  return widget ? known(widget.value, "") : UNKNOWN;
}
