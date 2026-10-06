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
 * Whether one of a node's sockets carries a link.
 *
 * @param {object} node - The node.
 * @param {"inputs"|"outputs"} side - Which list the socket is on.
 * @param {number} slot - Its index on that list, or -1 for none.
 * @returns {boolean} True when anything is wired to it.
 */
function wired(node, side, slot) {
  if (!node || slot === undefined || slot < 0) return false;
  if (side === "outputs") {
    if (typeof node.isOutputConnected === "function") return node.isOutputConnected(slot);
    return (node.outputs?.[slot]?.links ?? []).length > 0;
  }
  if (typeof node.isInputConnected === "function") return node.isInputConnected(slot);
  const link = node.inputs?.[slot]?.link;
  return link !== null && link !== undefined;
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
 * @param {object} node - The node.
 * @param {"inputs"|"outputs"} side - Which list to count.
 * @param {string[][]} groups - The growable groups, in declared order.
 * @param {number} minVisible - The fewest to draw.
 * @returns {number} A count between `minVisible` and `groups.length`.
 */
function wantedCount(node, side, groups, minVisible) {
  const byName = new Map((node[side] ?? []).map((socket, slot) => [socket.name, slot]));
  let lastUsed = -1;
  groups.forEach((names, index) => {
    if (names.some((name) => wired(node, side, byName.get(name) ?? -1))) lastUsed = index;
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
    if (held === undefined || (!wired(node, side, held) && wired(node, side, slot))) keep.set(socket.name, slot);
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
  // A socket the declaration does not name sorts after every one it does.
  const place = (socket) => rank.get(socket.name) ?? order.length;
  if (sockets.every((socket, slot) => slot === 0 || place(sockets[slot - 1]) <= place(socket))) {
    return;
  }

  const graph = node.graph;
  // Tied to their sockets by name before the sort.
  const held = graph ? linksBySocket(graph, node, side) : new Map();
  sockets.sort((a, b) => place(a) - place(b));
  const moves = new Map();
  sockets.forEach((socket, slot) => {
    for (const link of held.get(socket.name) ?? []) moves.set(link, slot);
  });
  if (side === "inputs") {
    const stranded = moveTargets(moves, graph ? linksEndingAt(graph, node, side) : []);
    if (stranded.length) {
      console.error(
        `[${LOG_NAME}] ${node?.type} #${node?.id}: links ${stranded.map((link) => link.id).join(", ")} `
          + "could not be moved with their sockets. Reconnect them by hand.",
      );
    }
  } else {
    for (const [link, slot] of moves) link.origin_slot = slot;
  }
}

/**
 * Point input links at their new slots without two ever landing on one slot at once.
 *
 * @param {Map<object, number>} moves - Link to the slot it belongs on.
 * @param {object[]} resident - Every link ending at the node, moving or not.
 * @returns {object[]} The links that did not reach their slot.
 */
function moveTargets(moves, resident) {
  const pending = new Map([...moves].filter(([link, slot]) => link.target_slot !== slot));
  while (pending.size > 0) {
    let progressed = false;
    for (const [link, slot] of pending) {
      // A move waits until its target slot has been vacated.
      const taken = [...pending.keys()].some((other) => other !== link && other.target_slot === slot);
      if (taken) continue;
      link.target_slot = slot;
      pending.delete(link);
      progressed = true;
    }
    if (progressed) continue;
    // A cycle of moves: one link steps aside to the lowest slot no link on the node holds.
    const occupied = new Set([...resident, ...moves.keys()].map((link) => link.target_slot));
    let free = 0;
    while (occupied.has(free)) free += 1;
    const [link] = pending.keys();
    link.target_slot = free;
    if (link.target_slot !== free) break;
  }
  return [...moves].filter(([link, slot]) => link.target_slot !== slot).map(([link]) => link);
}

/**
 * Every link ending at one side of a node.
 *
 * @param {object} graph - The graph holding the node.
 * @param {object} node - The node whose links are read.
 * @param {"inputs"|"outputs"} side - Which end of the link is the node.
 * @returns {object[]} The link objects.
 */
function linksEndingAt(graph, node, side) {
  const links = graph.links instanceof Map
    ? [...graph.links.values()]
    : Object.values(graph.links ?? {});
  return links.filter((link) => {
    if (!link) return false;
    const end = side === "inputs" ? link.target_id : link.origin_id;
    return String(end) === String(node.id);
  });
}

/**
 * The links on one side of a node, by the name of the socket each lands on.
 *
 * @param {object} graph - The graph holding the node.
 * @param {object} node - The node whose links are read.
 * @param {"inputs"|"outputs"} side - Which list the sockets are on.
 * @returns {Map<string, object[]>} Socket name to the link objects on it.
 */
function linksBySocket(graph, node, side) {
  const held = new Map();
  const hold = (name, link) => {
    if (name === undefined || !link) return;
    if (!held.has(name)) held.set(name, []);
    held.get(name).push(link);
  };
  // An input's link as the frontend itself resolves it, by slot or by the socket's own id.
  if (side === "inputs" && typeof node.getInputLink === "function") {
    node.inputs.forEach((socket, slot) => hold(socket.name, node.getInputLink(slot)));
    return held;
  }
  const names = node[side].map((socket) => socket.name);
  for (const link of linksEndingAt(graph, node, side)) {
    hold(names[side === "inputs" ? link.target_slot : link.origin_slot], link);
  }
  return held;
}

/**
 * Bring one side of a node to a given number of growable sockets.
 *
 * @param {object} node - The node to grow or shrink.
 * @param {"inputs"|"outputs"} side - Which list to work on.
 * @param {object} plan - The captured declaration for this side.
 * @param {number} wanted - How many growable sockets this side should draw.
 * @param {Set<number>|null} chosen - The group indices to draw, or null to draw the first `wanted`.
 * @returns {boolean} Whether anything changed.
 */
function fitSide(node, side, plan, wanted, chosen) {
  const sockets = node[side];
  if (!Array.isArray(sockets) || plan.types.size === 0) return false;

  const present = new Set(sockets.map((socket) => socket.name));
  let changed = false;

  // Drop from the end of the growable list inward, and never drop one that is wired: taking a
  // socket off the node takes its link with it, which would quietly delete a connection the
  // user made. A group is dropped socket by socket for the same reason, so a wired output keeps
  // its place while an unused socket beside it folds away.
  const drawn = (index) => (chosen ? chosen.has(index) : index < wanted);
  for (let index = plan.groups.length - 1; index >= 0; index -= 1) {
    if (drawn(index)) continue;
    for (const name of plan.groups[index]) {
      if (!present.has(name)) continue;
      const slot = sockets.findIndex((socket) => socket.name === name);
      if (slot === -1 || wired(node, side, slot)) continue;
      if (side === "inputs") node.removeInput(slot);
      else node.removeOutput(slot);
      present.delete(name);
      changed = true;
    }
  }

  for (let index = 0; index < plan.groups.length; index += 1) {
    if (!drawn(index)) continue;
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
 * Lay one side of a node out as a saved copy of it lists its sockets, moving no link.
 *
 * @param {object} node - The node about to be configured.
 * @param {"inputs"|"outputs"} side - Which list to lay out.
 * @param {object} plan - The captured declaration for this side.
 * @param {object[]} saved - The saved sockets, in saved order.
 * @returns {void}
 */
function matchSaved(node, side, plan, saved) {
  const sockets = node[side];
  if (!Array.isArray(sockets) || !Array.isArray(saved) || plan.types.size === 0) return;
  const growable = new Set(plan.groups.flat());
  const savedNames = saved.map((socket) => socket?.name);

  for (const name of savedNames) {
    if (!growable.has(name) || sockets.some((socket) => socket.name === name)) continue;
    const declared = plan.types.get(name);
    if (!declared) continue;
    if (side === "inputs") node.addInput(name, declared.type, declared.options);
    else node.addOutput(name, declared.type, declared.options);
  }

  // The saved sockets in saved order, then every other declared socket that is always drawn.
  const placed = new Set();
  const ordered = [];
  for (const name of savedNames) {
    const socket = sockets.find((candidate) => candidate.name === name && !placed.has(candidate));
    if (!socket) continue;
    placed.add(socket);
    ordered.push(socket);
  }
  for (const socket of sockets) {
    if (placed.has(socket) || growable.has(socket.name)) continue;
    ordered.push(socket);
  }
  sockets.splice(0, sockets.length, ...ordered);
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
 * @param {() => number[]} [options.select] - Which entries to draw, by index, for a node whose
 *   visible set changes rather than only its length. Takes precedence over exactCount.
 * @returns {() => void} A function that re-fits, for a caller with its own reason to. While the
 *   node is being built or configured the fit is queued until that has finished.
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
  const readSelect = typeof options.select === "function" ? options.select : null;
  const groups = asGroups(growable);
  const plans = {
    inputs: capture(node, "inputs", groups),
    outputs: capture(node, "outputs", groups),
  };

  const fit = () => {
    try {
      // One count for both sides, taken from whichever is further along. On a loop's Open node
      // a carried value arrives as an input and is read as an output, so revealing the input
      // alone would leave the value with nowhere to be read from.
      const minVisible = readMinVisible();
      const picked = readSelect ? readSelect() : null;
      const chosen = Array.isArray(picked) ? new Set(picked.map(Number)) : null;
      const asked = readExactCount ? Number(readExactCount()) : null;
      const wanted = Number.isFinite(asked)
        ? Math.max(0, Math.min(asked, groups.length))
        : Math.max(
            wantedCount(node, "inputs", plans.inputs.groups, minVisible),
            wantedCount(node, "outputs", plans.outputs.groups, minVisible),
          );
      // Before anything is counted or moved, since a duplicate makes both meaningless.
      const dedupedIn = dedupe(node, "inputs");
      const dedupedOut = dedupe(node, "outputs");
      const changedIn = fitSide(node, "inputs", plans.inputs, wanted, chosen);
      const changedOut = fitSide(node, "outputs", plans.outputs, wanted, chosen);
      // Ordered on every fit, whether or not this fit changed anything.
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

  // Every declared socket stays on the node until it is built and any configure has returned.
  let settled = false;
  let configuring = 0;
  let fitQueued = false;
  const queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    queueMicrotask(() => {
      fitQueued = false;
      if (configuring > 0) return;
      settled = true;
      fit();
    });
  };
  const refit = () => {
    if (!settled || configuring > 0) queueFit();
    else fit();
  };

  // Every link a workflow restores lands here, so they share one fit.
  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    queueFit();
    return result;
  };

  // A node with no links yet takes the saved socket layout before it is configured. Fits asked
  // for while it is configured, by this or by another extension's hook, run once that returns.
  const originalConfigureMethod = node.configure;
  if (typeof originalConfigureMethod === "function") {
    node.configure = function (...args) {
      configuring += 1;
      try {
        const info = args[0];
        const linked = node.graph
          && (linksEndingAt(node.graph, node, "inputs").length
            || linksEndingAt(node.graph, node, "outputs").length);
        if (info && (!settled || !linked)) {
          matchSaved(node, "inputs", plans.inputs, info.inputs);
          matchSaved(node, "outputs", plans.outputs, info.outputs);
        }
        return originalConfigureMethod.apply(this, args);
      } finally {
        configuring -= 1;
        queueFit();
      }
    };
  }
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    queueFit();
    return result;
  };

  queueFit();
  return refit;
}
