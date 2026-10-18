/**
 * The chain of MiniMax H3 Asset nodes feeding an `assets` input, read and rewired on the graph.
 *
 * A chain runs upstream first: each asset node's `assets` output feeds the next one's `assets`
 * input, and the last feeds the conditioning node.
 */

const LOG_NAME = "WASNodeSuite.H3AssetChain";

/** The node id of MiniMax H3 Asset. */
export const ASSET_NODE_ID = "WASMiniMaxH3Asset";

/** The input a chain arrives on, and the output it leaves by, on every node it passes. */
export const CHAIN_INPUT = "assets";
export const CHAIN_OUTPUT = 0;

// Longest chain followed, so a loop in the graph cannot hold the page.
const MOST_LINKS = 512;

/** Node modes a run reads differently: a muted node answers nothing, a bypassed one passes an
 * input through in place of its output. */
export const MUTED_MODE = 2;
export const BYPASSED_MODE = 4;

// Space left between a placed node and its neighbours, in graph units.
const PLACE_GAP = 40;

// Title bar height when LiteGraph does not say, in graph units.
const TITLE_HEIGHT = 30;

// Most nodes a placement steps past before it settles.
const MOST_STEPS = 400;

/**
 * The input slot a node declares under a name.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {number} The slot, or -1.
 */
export function slotOf(node, name) {
  return (node?.inputs ?? []).findIndex((input) => input?.name === name);
}

/**
 * One link of a graph, by id.
 *
 * @param {object} graph - The graph.
 * @param {number} id - The link's id.
 * @returns {object|null} The link, or null.
 */
function linkById(graph, id) {
  if (id === null || id === undefined || !graph) return null;
  const links = graph.links;
  return (links instanceof Map ? links.get(id) : links?.[id]) ?? null;
}

/**
 * The links leaving one of a node's output slots.
 *
 * @param {object} node - The node.
 * @param {number} slot - The output slot.
 * @returns {object[]} The links, in the graph's order.
 */
function outgoing(node, slot) {
  const links = node?.graph?.links;
  const all = links instanceof Map ? [...links.values()] : Object.values(links ?? {});
  return all.filter((link) => link && String(link.origin_id) === String(node.id)
    && Number(link.origin_slot) === slot);
}

/**
 * Whether anything reads one of a node's outputs.
 *
 * @param {object} node - The node.
 * @param {number} slot - The output slot.
 * @returns {boolean} True when a link leaves it.
 */
function outputWired(node, slot) {
  if (typeof node?.isOutputConnected === "function") return node.isOutputConnected(slot);
  return outgoing(node, slot).length > 0;
}

/**
 * The link into one of a node's input slots.
 *
 * @param {object} node - The node.
 * @param {number} slot - The slot.
 * @returns {object|null} The link, or null.
 */
export function inputLink(node, slot) {
  if (!node || slot < 0) return null;
  try {
    if (typeof node.getInputLink === "function") return node.getInputLink(slot) ?? null;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read a link:`, error);
  }
  return linkById(node.graph, node.inputs?.[slot]?.link);
}

/**
 * Whether two socket types may be joined, as LiteGraph reads them.
 *
 * @param {string} a - One type.
 * @param {string} b - The other.
 * @returns {boolean} True when a link between them is valid.
 */
function fits(a, b) {
  const check = window.LiteGraph?.isValidConnection;
  if (typeof check === "function") return check.call(window.LiteGraph, a, b);
  return a === b || a === "*" || b === "*" || !a || !b;
}

/**
 * The input a bypassed node passes through in place of one of its outputs.
 *
 * @param {object} node - The bypassed node.
 * @param {number} slot - The output being read.
 * @param {string} type - The type the reader asks for.
 * @returns {number} The input slot, or -1 where none fits.
 */
function bypassInput(node, slot, type) {
  const inputs = node.inputs ?? [];
  if (type === "*" || type === "") return inputs.length > slot ? slot : 0;
  const same = inputs[slot];
  const given = node.outputs?.[slot]?.type;
  if (same && fits(same.type, given) && fits(same.type, type)) return slot;
  const named = inputs.findIndex((input) => input.type === type);
  return named >= 0 ? named : inputs.findIndex((input) => fits(input.type, given) && fits(input.type, type));
}

/**
 * The node a link carries a run's value from, once muted and bypassed nodes are resolved.
 *
 * @param {object} graph - The graph the link is in.
 * @param {object|null} link - The link.
 * @param {string} type - The type the reading input takes.
 * @returns {{node: object, slot: number}|null} The node and output a run reads, or null where a
 *   muted node, or a bypassed one with nothing to pass, ends the link.
 */
export function liveOrigin(graph, link, type = "*") {
  let current = link;
  for (let step = 0; step < MOST_LINKS && current; step += 1) {
    const origin = graph?.getNodeById?.(current.origin_id);
    if (!origin || origin.mode === MUTED_MODE) return null;
    if (origin.mode !== BYPASSED_MODE) return { node: origin, slot: current.origin_slot };
    const through = bypassInput(origin, current.origin_slot, type);
    if (through < 0) return null;
    current = inputLink(origin, through);
  }
  return null;
}

/**
 * Whether a run receives anything on one of a node's inputs.
 *
 * @param {object} node - The node.
 * @param {string} name - The input's name.
 * @returns {boolean} True when a link lands on it from a node that is neither muted nor bypassed
 *   with nothing to pass.
 */
export function wiredInto(node, name) {
  const slot = slotOf(node, name);
  const link = inputLink(node, slot);
  return link !== null && liveOrigin(node.graph, link, node.inputs?.[slot]?.type ?? "*") !== null;
}

/**
 * The asset nodes whose assets a run hands to a node's `assets` input.
 *
 * @param {object} node - The node the chain ends on.
 * @returns {object[]} The asset nodes, upstream first. A bypassed one is passed over; a muted one
 *   ends the chain, and every node before it is cut off with it.
 */
export function liveChainOf(node) {
  const found = [];
  const seen = new Set();
  let current = node;
  for (let step = 0; step < MOST_LINKS && current; step += 1) {
    const slot = slotOf(current, CHAIN_INPUT);
    const live = liveOrigin(current.graph, inputLink(current, slot), current.inputs?.[slot]?.type ?? "*");
    if (!live || live.node.type !== ASSET_NODE_ID || seen.has(live.node.id)) break;
    seen.add(live.node.id);
    found.push(live.node);
    current = live.node;
  }
  return found.reverse();
}

/**
 * The asset nodes chained into a node's `assets` input, whatever their mode.
 *
 * @param {object} node - The node the chain ends on.
 * @returns {object[]} The asset nodes, upstream first. The walk stops at anything else.
 */
export function chainOf(node) {
  const found = [];
  const seen = new Set();
  let current = node;
  for (let step = 0; step < MOST_LINKS && current; step += 1) {
    const link = inputLink(current, slotOf(current, CHAIN_INPUT));
    if (!link) break;
    const origin = current.graph?.getNodeById?.(link.origin_id);
    if (!origin || origin.type !== ASSET_NODE_ID || seen.has(origin.id)) break;
    seen.add(origin.id);
    found.push(origin);
    current = origin;
  }
  return found.reverse();
}

/**
 * Wire a new asset node onto the end of the chain feeding a node.
 *
 * @param {object} target - The node the chain ends on.
 * @param {object} made - The asset node, already in the graph.
 * @returns {void}
 * @throws {Error} Either node declares no `assets` input.
 */
export function appendToChain(target, made) {
  const into = slotOf(target, CHAIN_INPUT);
  const madeInto = slotOf(made, CHAIN_INPUT);
  if (into < 0 || madeInto < 0) throw new Error("a node on the chain has no assets input");
  const link = inputLink(target, into);
  if (link) {
    const origin = target.graph?.getNodeById?.(link.origin_id);
    origin?.connect?.(link.origin_slot, made, madeInto);
  }
  made.connect(CHAIN_OUTPUT, target, into);
}

/**
 * Take an asset node out of its chain, joining the nodes either side of it.
 *
 * @param {object} asset - The asset node.
 * @returns {boolean} True when the node left the graph, false when it was only unwired, its
 *   other outputs still being read elsewhere.
 */
export function removeFromChain(asset) {
  const graph = asset?.graph;
  if (!graph) return false;
  const into = slotOf(asset, CHAIN_INPUT);
  const upstream = inputLink(asset, into);
  const origin = upstream ? graph.getNodeById?.(upstream.origin_id) : null;
  const downstream = outgoing(asset, CHAIN_OUTPUT)
    .map((link) => ({ node: graph.getNodeById?.(link.target_id), slot: link.target_slot }))
    .filter((each) => each.node);
  for (const each of downstream) {
    if (origin) origin.connect(upstream.origin_slot, each.node, each.slot);
    else each.node.disconnectInput?.(each.slot);
  }
  const elsewhere = (asset.outputs ?? []).some((output, index) =>
    index !== CHAIN_OUTPUT && outputWired(asset, index));
  if (elsewhere) {
    if (into >= 0) asset.disconnectInput?.(into);
    return false;
  }
  graph.remove(asset);
  return true;
}

/**
 * Chain the asset nodes a saved workflow wired into numbered `asset_N` sockets.
 *
 * @param {object} node - A conditioning node, just configured.
 * @returns {number} How many asset nodes were chained, 0 where the node held no such socket.
 */
export function chainNumberedSockets(node) {
  const numbered = (node?.inputs ?? [])
    .map((input, slot) => ({ slot, match: /^asset_(\d+)$/.exec(input?.name ?? "") }))
    .filter((each) => each.match);
  if (!numbered.length) return 0;
  const origins = [...numbered]
    .sort((a, b) => Number(a.match[1]) - Number(b.match[1]))
    .map((each) => inputLink(node, each.slot))
    .map((link) => (link ? node.graph?.getNodeById?.(link.origin_id) : null))
    .filter((origin) => origin?.type === ASSET_NODE_ID);
  for (const each of [...numbered].sort((a, b) => b.slot - a.slot)) {
    node.disconnectInput?.(each.slot);
    node.removeInput?.(each.slot);
  }
  origins.forEach((origin, index) => {
    const target = index + 1 < origins.length ? origins[index + 1] : node;
    const into = slotOf(target, CHAIN_INPUT);
    if (into >= 0 && !inputLink(target, into)) origin.connect(CHAIN_OUTPUT, target, into);
  });
  return origins.length;
}

/**
 * A node's outline on the canvas, title bar included.
 *
 * @param {object} node - The node.
 * @returns {number[]} `[x, y, width, height]` in graph units.
 */
function outlineOf(node) {
  const title = Number(window.LiteGraph?.NODE_TITLE_HEIGHT) || TITLE_HEIGHT;
  const collapsed = Boolean(node?.flags?.collapsed);
  return [
    Number(node?.pos?.[0] ?? 0),
    Number(node?.pos?.[1] ?? 0) - title,
    Number(node?.size?.[0] ?? 0),
    (collapsed ? 0 : Number(node?.size?.[1] ?? 0)) + title,
  ];
}

/**
 * The spot nearest a wanted one where a node overlaps no other node.
 *
 * @param {object} graph - The graph the node goes in.
 * @param {object} made - The node being placed, measured at the size its widgets grow it to.
 * @param {number[]} wanted - `[x, y]` it would take on an empty canvas.
 * @returns {number[]} `[x, y]` in graph units: straight down or straight left of `wanted`,
 *   whichever is nearer, and `wanted` itself when it is clear.
 */
export function clearSpot(graph, made, wanted) {
  const title = Number(window.LiteGraph?.NODE_TITLE_HEIGHT) || TITLE_HEIGHT;
  const grown = typeof made?.computeSize === "function" ? made.computeSize() : null;
  const width = Math.max(Number(made?.size?.[0] ?? 0), Number(grown?.[0] ?? 0));
  const height = Math.max(Number(made?.size?.[1] ?? 0), Number(grown?.[1] ?? 0)) + title;
  const others = (graph?._nodes ?? graph?.nodes ?? []).filter((each) => each && each !== made).map(outlineOf);
  const blocking = (x, y) => others.find(([left, top, wide, high]) =>
    x < left + wide + PLACE_GAP && x + width + PLACE_GAP > left
    && y - title < top + high + PLACE_GAP && y - title + height + PLACE_GAP > top);
  const [x, y] = wanted;
  let down = y;
  let left = x;
  for (let step = 0, hit = blocking(x, down); hit && step < MOST_STEPS; step += 1, hit = blocking(x, down)) {
    down = hit[1] + hit[3] + PLACE_GAP + title;
  }
  for (let step = 0, hit = blocking(left, y); hit && step < MOST_STEPS; step += 1, hit = blocking(left, y)) {
    left = hit[0] - PLACE_GAP - width;
  }
  return down - y <= x - left ? [x, down] : [left, y];
}

/**
 * Where a new asset node goes: below the chain, or left of the node it feeds, clear of other nodes.
 *
 * @param {object} target - The node the chain ends on.
 * @param {object[]} chain - The asset nodes already chained, from `chainOf`.
 * @param {object} made - The new asset node.
 * @returns {number[]} `[x, y]` in graph units.
 */
export function placeInChain(target, chain, made) {
  const width = Number(made?.size?.[0]) || 320;
  const left = chain.length
    ? Math.min(...chain.map((each) => Number(each.pos?.[0] ?? 0)))
    : Number(target.pos?.[0] ?? 0) - width - 80;
  const top = chain.length
    ? Math.max(...chain.map((each) => Number(each.pos?.[1] ?? 0) + Number(each.size?.[1] ?? 0))) + PLACE_GAP
    : Number(target.pos?.[1] ?? 0);
  return clearSpot(target.graph, made, [left, top]);
}
