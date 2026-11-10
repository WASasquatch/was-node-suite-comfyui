/**
 * Widget groups that appear as they are needed.
 *
 * The groups in use and the next empty one are drawn, and the rest hidden. Hiding is
 * presentation only: `widgets_values` carries every widget whether it is drawn or not.
 */
import { redrawHeaders } from "./decoration.js";
import { startingInput } from "./node_preview.js";
import { setWidgetHidden } from "./visibility.js";

const LOG_NAME = "WASNodeSuite.Grow";

// Groups always drawn, however empty. One would leave nowhere to type without a reveal step
// first, and it is the second box that shows what the node is for.
const MIN_VISIBLE = 2;

/**
 * Whether a widget value is something a user put there.
 *
 * @param {*} value - The value to read.
 * @param {Set<*>|null} [empty] - Values that count as empty besides `""`, null and undefined.
 * @returns {boolean} True for any other value.
 */
export function holdsValue(value, empty) {
  if (value === "" || value === undefined || value === null) return false;
  return !empty?.has(value);
}

/**
 * How many of a node's groups are drawn: every group up to the last one in use, one more, and
 * never fewer than `minVisible`.
 *
 * @param {string[][]} groups - Widget names, one array per group, in the order they are drawn.
 * @param {(names: string[]) => boolean} used - Whether a group is in use, given its voting names.
 * @param {object} [options] - Settings.
 * @param {number} [options.minVisible] - Groups drawn however empty, two by default.
 * @param {number} [options.forced] - Groups held open regardless of what they hold.
 * @param {number|null} [options.decidesAt] - Position within a group of the one widget that
 *   votes. Every widget in the group votes when it is left out.
 * @returns {number} How many groups are drawn, counted from the first.
 */
export function drawnGroupCount(groups, used, options = {}) {
  const minVisible = Number.isFinite(options.minVisible) ? options.minVisible : MIN_VISIBLE;
  const forced = Number.isFinite(options.forced) ? options.forced : 0;
  const decidesAt = Number.isFinite(options.decidesAt) ? options.decidesAt : null;

  // The last group holding anything, rather than the first empty one, so a gap in the middle
  // stays open instead of folding away a box somebody filled in below an empty one. It starts
  // two below the minimum, with one more than it drawn: an empty node shows `minVisible`
  // boxes, and the next appears only once one of those is filled.
  let lastUsed = Math.max(minVisible - 2, forced - 2);
  groups.forEach((names, index) => {
    const voting = decidesAt === null ? names : names.slice(decidesAt, decidesAt + 1);
    if (used(voting)) lastUsed = Math.max(lastUsed, index);
  });
  return Math.min(groups.length, lastUsed + 2);
}

/**
 * The groups a newly placed node folds away: every group past the ones its starting values
 * draw.
 *
 * @param {string[][]} groups - Widget names, one array per group, in the order they are drawn.
 * @param {object} definition - The node definition the frontend files.
 * @param {object} [options] - The `growWidgets` options.
 * @returns {{inputs: string[]}} The input names.
 */
export function foldedGroups(groups, definition, options = {}) {
  const empty = options.empty ? new Set(options.empty) : null;
  const drawn = drawnGroupCount(
    groups,
    (voting) => voting.some((name) => holdsValue(startingInput(definition, name), empty)),
    options,
  );
  return { inputs: groups.slice(drawn).flat() };
}

/**
 * Whether one of a node's inputs carries a link.
 *
 * @param {object} node - The node the input belongs to.
 * @param {string} name - The input's name.
 * @returns {boolean} True when something is wired into it.
 */
function linked(node, name) {
  for (const input of node?.inputs ?? []) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

/**
 * Show the groups in use and the next empty one, and hide the rest.
 *
 * @param {object} node - The node to fold.
 * @param {string[][]} groups - Widget names, one array per group, in the order they are drawn.
 * @param {number} minVisible - How many groups are drawn however empty.
 * @returns {void}
 */
function fold(node, groups, minVisible, empty, decidesAt) {
  const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
  const drawn = drawnGroupCount(
    groups,
    (voting) => voting.some((name) => holdsValue(widgets.get(name)?.value, empty) || linked(node, name)),
    // Groups held open regardless of what they hold.
    { minVisible, decidesAt, forced: node.__was_forced_groups },
  );

  let changed = false;
  groups.forEach((names, index) => {
    const hide = index >= drawn;
    for (const name of names) {
      if (setWidgetHidden(widgets.get(name), hide)) changed = true;
    }
  });
  if (!changed) return;

  // Only the height is taken from the recomputed size.
  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

/**
 * Draw a node's repeated widget groups as they are filled.
 *
 * @param {object} node - The node to grow.
 * @param {string[][]} groups - Widget names, one array per group, in the order they are drawn.
 * @param {object} [options] - Settings.
 * @param {number} [options.minVisible] - Groups drawn however empty, two by default.
 * @param {number} [options.decidesAt] - Position within a group of the one widget that says
 *   whether the group is in use. Every widget in the group votes by default, which is wrong
 *   where a group carries a switch or a number that is never empty.
 * @returns {() => void} A function that folds again, for a caller with its own reason to.
 *   Set `node.__was_forced_groups` first to hold that many groups open regardless of content.
 */
export function growWidgets(node, groups, options = {}) {
  const minVisible = Number.isFinite(options.minVisible) ? options.minVisible : MIN_VISIBLE;
  const empty = options.empty ? new Set(options.empty) : null;
  const decidesAt = Number.isFinite(options.decidesAt) ? options.decidesAt : null;
  const names = new Set(groups.flat());

  const refold = () => {
    try {
      fold(node, groups, minVisible, empty, decidesAt);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to fold ${node?.type}:`, error);
    }
    redrawHeaders(node);
  };

  for (const widget of node.widgets ?? []) {
    if (!names.has(widget.name)) continue;
    const original = widget.callback;
    widget.callback = function (...args) {
      const result = original?.apply(this, args);
      refold();
      return result;
    };
    // A multiline widget is a textarea typed into directly, and typing does not go through the
    // callback, so without this the next group would not appear until something else repainted.
    const field = widget.element?.querySelector?.("textarea") ?? widget.element;
    field?.addEventListener?.("input", refold);
  }

  // A link arriving at or leaving a hidden input changes what has to be drawn, and neither one
  // touches a widget value. Every link a workflow restores lands here, so they share one fold.
  let refoldQueued = false;
  const queueRefold = () => {
    if (refoldQueued) return;
    refoldQueued = true;
    queueMicrotask(() => {
      refoldQueued = false;
      refold();
    });
  };
  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    queueRefold();
    return result;
  };

  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    refold();
    // Again once configure returns, when decorations are back in the widget list.
    setTimeout(refold, 0);
    return result;
  };

  refold();
  return refold;
}
