/**
 * Repeated sockets that appear as they are wired.
 *
 * A socket is taken off the node and put back when it is wanted, keeping the declared order and
 * repairing the links whose slot numbers move.
 */

const LOG_NAME = "WASNodeSuite.GrowSockets";

// Sockets always drawn, however empty. Two, so the list reads as a list rather than as one
// socket that happens to repeat.
const MIN_VISIBLE = 2;

/**
 * Whether a socket carries a link.
 *
 * @param {object} socket - An entry of `node.inputs` or `node.outputs`.
 * @returns {boolean} True when anything is wired to it.
 */
function wired(socket) {
  if (!socket) return false;
  if (Array.isArray(socket.links)) return socket.links.length > 0;
  return socket.link !== null && socket.link !== undefined;
}

/**
 * A growable list as groups, so one entry may name several sockets that appear together.
 *
 * @param {Array<string|string[]>} growable - Socket names, or arrays of names drawn as one step.
 * @returns {string[][]} One array per step.
 */
function asGroups(growable) {
  return (growable ?? []).map((entry) => (Array.isArray(entry) ? entry : [entry]));
}

/**
 * How many of a growable list to draw: those in use, plus one spare.
 *
 * @param {object[]} sockets - The node's current `inputs` or `outputs`.
 * @param {string[][]} groups - The growable groups, in declared order.
 * @param {number} minVisible - The fewest to draw.
 * @returns {number} A count between `minVisible` and `groups.length`.
 */
function wantedCount(sockets, groups, minVisible) {
  const byName = new Map(sockets.map((socket) => [socket.name, socket]));
  let lastUsed = -1;
  groups.forEach((names, index) => {
    if (names.some((name) => wired(byName.get(name)))) lastUsed = index;
  });
  return Math.max(minVisible, Math.min(lastUsed + 2, groups.length));
}

/**
 * Drop any socket whose name is already on the node, keeping the one that carries links.
 *
 * @param {object} node - The node to clean up.
 * @param {"inputs"|"outputs"} side - Which list to clean.
 * @returns {boolean} Whether anything was removed.
 */
function dedupe(node, side) {
  // Loading a workflow restores sockets onto a node this has already shrunk, and one it cannot
  // line up is appended rather than matched, so a node drawing fewer sockets than the saved file
  // lists comes back with its trailing sockets twice. Two sockets of one name is a node whose
  // slot numbers no longer describe it, and re-saving writes that back out.
  const sockets = node[side];
  if (!Array.isArray(sockets)) return false;

  const keep = new Map();
  sockets.forEach((socket, slot) => {
    const held = keep.get(socket.name);
    // The wired copy is the one worth keeping: dropping it would take a link with it.
    if (held === undefined || (!wired(sockets[held]) && wired(socket))) keep.set(socket.name, slot);
  });

  const doomed = [];
  sockets.forEach((socket, slot) => {
    if (keep.get(socket.name) !== slot) doomed.push(slot);
  });
  if (doomed.length === 0) return false;

  // Back to front, so a removal never shifts a slot still waiting to be removed.
  for (let index = doomed.length - 1; index >= 0; index -= 1) {
    if (side === "inputs") node.removeInput(doomed[index]);
    else node.removeOutput(doomed[index]);
  }
  return true;
}

/**
 * Put a node's sockets back into their declared order and repair the links that moved.
 *
 * @param {object} node - The node to fix up.
 * @param {"inputs"|"outputs"} side - Which list to order.
 * @param {string[]} order - Every socket name on that side, in declared order.
 * @returns {void}
 */
function reorder(node, side, order) {
  // A link records the slot it lands on as a number, so the numbers are recomputed here.
  const sockets = node[side];
  if (!Array.isArray(sockets)) return;
  const rank = new Map(order.map((name, index) => [name, index]));
  sockets.sort((a, b) => (rank.get(a.name) ?? 0) - (rank.get(b.name) ?? 0));

  const graph = node.graph;
  if (!graph) return;
  sockets.forEach((socket, slot) => {
    if (side === "inputs") {
      const link = graph.links?.[socket.link];
      if (link) link.target_slot = slot;
      return;
    }
    for (const id of socket.links ?? []) {
      const link = graph.links?.[id];
      if (link) link.origin_slot = slot;
    }
  });
}

/**
 * Bring one side of a node to a given number of growable sockets.
 *
 * @param {object} node - The node to grow or shrink.
 * @param {"inputs"|"outputs"} side - Which list to work on.
 * @param {object} plan - The captured declaration for this side.
 * @param {number} wanted - How many growable sockets this side should draw.
 * @returns {boolean} Whether anything changed.
 */
function fitSide(node, side, plan, wanted) {
  const sockets = node[side];
  if (!Array.isArray(sockets) || plan.types.size === 0) return false;

  const present = new Set(sockets.map((socket) => socket.name));
  let changed = false;

  // Drop from the end of the growable list inward, and never drop one that is wired: taking a
  // socket off the node takes its link with it, which would quietly delete a connection the
  // user made. A group is dropped socket by socket for the same reason, so a wired output keeps
  // its place while an unused socket beside it folds away.
  for (let index = plan.groups.length - 1; index >= wanted; index -= 1) {
    for (const name of plan.groups[index]) {
      if (!present.has(name)) continue;
      const slot = sockets.findIndex((socket) => socket.name === name);
      if (slot === -1 || wired(sockets[slot])) continue;
      if (side === "inputs") node.removeInput(slot);
      else node.removeOutput(slot);
      present.delete(name);
      changed = true;
    }
  }

  for (let index = 0; index < wanted; index += 1) {
    for (const name of plan.groups[index]) {
      if (present.has(name)) continue;
      const declared = plan.types.get(name);
      if (!declared) continue;
      if (side === "inputs") node.addInput(name, declared.type, declared.options);
      else node.addOutput(name, declared.type, declared.options);
      present.add(name);
      changed = true;
    }
  }

  if (changed) reorder(node, side, plan.order);
  return changed;
}

/**
 * Capture how one side of a node is declared, before anything is removed from it.
 *
 * @param {object} node - The freshly created node.
 * @param {"inputs"|"outputs"} side - Which list to capture.
 * @param {string[][]} groups - The groups that may come and go.
 * @returns {object} The declared order, this side's groups, and each socket's type.
 */
function capture(node, side, groups) {
  const sockets = Array.isArray(node[side]) ? node[side] : [];
  const declared = new Set(sockets.map((socket) => socket.name));
  const sideGroups = groups.map((names) => names.filter((name) => declared.has(name)));
  const names = sideGroups.flat();
  const types = new Map();
  for (const socket of sockets) {
    if (!names.includes(socket.name)) continue;
    // `shape` and `label` are carried over to the re-added socket.
    types.set(socket.name, {
      type: socket.type,
      options: { shape: socket.shape, label: socket.label, localized_name: socket.localized_name },
    });
  }
  return { order: sockets.map((socket) => socket.name), groups: sideGroups, types };
}

/**
 * Draw a node's repeated sockets as they are wired.
 *
 * @param {object} node - The node to grow.
 * @param {Array<string|string[]>} growable - One entry per step, in declared order. An entry may
 *   be a name, or an array of names that appear together, such as an input and the outputs
 *   reporting it. Names absent from a side are ignored for that side.
 * @param {object} [options] - Settings.
 * @param {number|(() => number)} [options.minVisible] - The fewest to draw, two by default. A
 *   function is read on every fit, which is what a count driven by a widget needs: the captured
 *   declaration must not be taken again, so the caller keeps the returned refit and calls it.
 * @param {() => number} [options.exactCount] - How many entries to draw. A wired socket is
 *   kept whatever this answers.
 * @returns {() => void} A function that re-fits, for a caller with its own reason to.
 */
export function growSockets(node, growable, options = {}) {
  // Read per fit rather than once, so a caller whose count comes from a widget can keep the
  // returned refit instead of calling this again, which would re-capture an already shrunk node.
  const readMinVisible = typeof options.minVisible === "function"
    ? () => {
        const value = Number(options.minVisible());
        return Number.isFinite(value) ? value : MIN_VISIBLE;
      }
    : () => (Number.isFinite(options.minVisible) ? options.minVisible : MIN_VISIBLE);
  // A caller naming the count itself.
  const readExactCount = typeof options.exactCount === "function" ? options.exactCount : null;
  const groups = asGroups(growable);
  const plans = {
    inputs: capture(node, "inputs", groups),
    outputs: capture(node, "outputs", groups),
  };

  const refit = () => {
    try {
      // One count for both sides, taken from whichever is further along. On a loop's Open node
      // a carried value arrives as an input and is read as an output, so revealing the input
      // alone would leave the value with nowhere to be read from.
      const minVisible = readMinVisible();
      const asked = readExactCount ? Number(readExactCount()) : null;
      const wanted = Number.isFinite(asked)
        ? Math.max(0, Math.min(asked, groups.length))
        : Math.max(
            wantedCount(node.inputs ?? [], plans.inputs.groups, minVisible),
            wantedCount(node.outputs ?? [], plans.outputs.groups, minVisible),
          );
      // Before anything is counted or moved, since a duplicate makes both meaningless.
      const dedupedIn = dedupe(node, "inputs");
      const dedupedOut = dedupe(node, "outputs");
      const changedIn = fitSide(node, "inputs", plans.inputs, wanted);
      const changedOut = fitSide(node, "outputs", plans.outputs, wanted);
      // Ordered every time rather than only after a change made here. Loading a workflow
      // restores its sockets by name onto a node this has already shrunk, and appends any it
      // does not find, so a saved socket can arrive after the widget inputs without this code
      // having touched anything.
      reorder(node, "inputs", plans.inputs.order);
      reorder(node, "outputs", plans.outputs.order);
      if (!changedIn && !changedOut && !dedupedIn && !dedupedOut) return;
      // Height only: a node somebody widened keeps its width, which is theirs to choose.
      const computed = node.computeSize?.();
      if (computed) node.setSize([node.size[0], computed[1]]);
      node.graph?.setDirtyCanvas(true, true);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to fit ${node?.type}'s sockets:`, error);
    }
  };

  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    refit();
    return result;
  };

  // A workflow restores its links after the node is built, so the fit is run again once they
  // are in place. Without this a saved workflow opens with every socket drawn.
  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    refit();
    return result;
  };

  refit();
  return refit;
}
