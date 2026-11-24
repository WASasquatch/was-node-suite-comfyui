/**
 * What a newly placed node folds away, for node previews.
 *
 * A rule per node id reads the node definition and names the repeated `inputs` and `outputs` a
 * new node leaves off and the `widgets` it folds.
 */

// Node id -> the rules naming what a new node of that id folds away.
const RULES = new Map();

/**
 * Register what a newly placed node of one id folds away.
 *
 * @param {string} nodeId - The node id.
 * @param {(definition: object) => {inputs?: Iterable<string>, outputs?: Iterable<string>,
 *   widgets?: Iterable<string>}} rule - The names of the repeated inputs and outputs a new node
 *   does not draw, and of the widgets it folds away.
 * @returns {void}
 */
export function foldedOnPlacing(nodeId, rule) {
  if (!nodeId || typeof rule !== "function") return;
  const key = String(nodeId);
  if (!RULES.has(key)) RULES.set(key, []);
  RULES.get(key).push(rule);
}

/**
 * Whether any rule names a node id.
 *
 * @param {string} nodeId - The node id.
 * @returns {boolean} True where a rule is registered for it.
 */
export function foldsOnPlacing(nodeId) {
  return RULES.has(String(nodeId));
}

/**
 * Every input, output and widget a newly placed node of one id folds away.
 *
 * @param {string} nodeId - The node id.
 * @param {object} definition - The node definition the frontend files.
 * @returns {{inputs: Set<string>, outputs: Set<string>, widgets: Set<string>}} The names, empty
 *   where no rule names the id.
 */
export function foldedNames(nodeId, definition) {
  const inputs = new Set();
  const outputs = new Set();
  const widgets = new Set();
  for (const rule of RULES.get(String(nodeId)) ?? []) {
    const named = rule(definition) ?? {};
    for (const name of named.inputs ?? []) inputs.add(String(name));
    for (const name of named.outputs ?? []) outputs.add(String(name));
    for (const name of named.widgets ?? []) widgets.add(String(name));
  }
  return { inputs, outputs, widgets };
}

/**
 * The value a new node's widget starts on.
 *
 * @param {object|undefined} spec - The input's spec from the node definition.
 * @returns {*} Its default, the first option of a list, or undefined where it names neither.
 */
export function startingValue(spec) {
  if (!spec) return undefined;
  if (spec.default !== undefined && spec.default !== null) return spec.default;
  if (Array.isArray(spec.options) && spec.options.length) return spec.options[0];
  return undefined;
}

/**
 * The value a new node's input starts on, read off its definition.
 *
 * @param {object} definition - The node definition the frontend files.
 * @param {string} name - The input's name.
 * @returns {*} Its starting value, or undefined where it names none.
 */
export function startingInput(definition, name) {
  return startingValue(definition?.inputs?.[name]);
}

/**
 * A stand-in for a newly placed node: each widget on its starting value and nothing wired.
 *
 * @param {object} definition - The node definition the frontend files.
 * @returns {{widgets: Array<{name: string, value: *}>, inputs: object[], outputs: object[]}}
 *   Enough of a node for a rule that reads widget values.
 */
export function placedStandIn(definition) {
  const widgets = Object.entries(definition?.inputs ?? {}).map(([name, spec]) => ({
    name,
    value: startingValue(spec),
  }));
  return { widgets, inputs: [], outputs: [] };
}
