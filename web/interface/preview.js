/**
 * The pictures a node held, for the interface drawn on that node.
 *
 * `fetchInputPreview`, `fetchOutputPreview` and `fetchPreviewPair` ask for the thumbnail a node
 * published. `PREVIEW_STATE` names the states an answer carries and `LABELS` gives the words.
 */

import { api } from "../../../scripts/api.js";
import { fetchWithin } from "./request.js";

const LOG_PREFIX = "[WASNodeSuite.Preview]";

const ROUTE = "/was/interface/api/preview";
const SUBSCRIBE_ROUTE = "/was/interface/api/preview/subscribe";

const SOURCE_WIDTH_HEADER = "X-WAS-Source-Width";
const SOURCE_HEIGHT_HEADER = "X-WAS-Source-Height";
const FRAME_COUNT_HEADER = "X-WAS-Frame-Count";
const FRAME_TOTAL_HEADER = "X-WAS-Frame-Total";
const SOURCE_MODE_HEADER = "X-WAS-Source-Mode";
const KIND_HEADER = "X-WAS-Kind";
const ENCODED_BYTES_HEADER = "X-WAS-Encoded-Bytes";
const PROMPT_HEADER = "X-WAS-Prompt";

/**
 * Which side of a node a picture was on.
 */
export const PREVIEW_SIDE = {
  // What the node received. The parameter is left off the request for this one, so the query an
  // interface written before sides existed sends is the one it always sent.
  INPUT: "input",
  // What the node answered with.
  OUTPUT: "output",
};

/**
 * The id a node publishes under.
 *
 * A node the canvas calls `5` inside subgraph `12` publishes as `12:5`.
 *
 * @param {object} node - The node the interface is drawn on.
 * @returns {string} The colon joined execution id, or the node's own id at the root.
 */
export function executionId(node) {
  const id = String(node?.id ?? "");
  const root = window.app?.rootGraph;
  if (!root || !node?.graph || node.graph === root) return id;

  const path = [];
  const walk = (graph) => {
    for (const candidate of graph?.nodes ?? []) {
      if (!candidate?.isSubgraphNode?.()) continue;
      if (candidate.subgraph === node.graph) {
        path.unshift(candidate.id);
        return true;
      }
      if (candidate.subgraph && walk(candidate.subgraph)) {
        path.unshift(candidate.id);
        return true;
      }
    }
    return false;
  };
  // Measured both ways a subgraph gets placed twice, rather than assumed. Copying a subgraph node
  // clones its definition, and inserting the same *published blueprint* twice from the node search
  // deserialises a fresh one: in both cases the two placements hold separate inner nodes with
  // separate ids, and the string composed here is exactly what `graphToPrompt` sends for them,
  // `3:1` and `6:4` for a copy, `5:2` and `7:6` for two blueprint insertions. Each instance's
  // interface reads its own run.
  //
  return walk(root) ? [...path, id].join(":") : id;
}

/**
 * The key the frontend files a node's run output under.
 *
 * @param {object} node - The node the entry belongs to.
 * @returns {string} The node's id at the root, and `subgraphUuid:id` anywhere else, which is
 *   what the frontend's `nodeToNodeLocatorId` composes. Empty for an unplaced node.
 */
export function nodeLocator(node) {
  const id = String(node?.id ?? "");
  if (!id) return "";
  const graph = node?.graph;
  return graph?.isRootGraph === false ? `${graph.id}:${id}` : id;
}

/**
 * What came back, and why the backdrop is or is not there.
 */
export const PREVIEW_STATE = {
  // The socket is still opening, which is also the case for the first moments after a reload.
  CONNECTING: "connecting",
  // A connected page and a node that has not run since it opened, which is the ordinary state
  // of a freshly placed node.
  WAITING: "waiting",
  // The request is in flight.
  LOADING: "loading",
  // The answer carries the image.
  READY: "ready",
  // The store answered with a picture from a run this page did not ask for, which is what a
  // second tab publishing under the same node id produces. Not a fault, and not this run.
  FOREIGN: "foreign",
  // A request or a decode that did not work, which is the only one of these that is a fault.
  FAILED: "failed",
};

/**
 * The words an interface draws for each state.
 */
export const LABELS = {
  [PREVIEW_STATE.CONNECTING]: "Connecting...",
  [PREVIEW_STATE.WAITING]: "",
  [PREVIEW_STATE.LOADING]: "Loading...",
  [PREVIEW_STATE.READY]: "",
  [PREVIEW_STATE.FOREIGN]: "From another run",
  [PREVIEW_STATE.FAILED]: "The image could not be loaded",
};

/**
 * Whether this page is connected well enough for a node to have published anything.
 *
 * @returns {boolean} True while the socket is open. False while it is absent, opening,
 *   closing or closed, which are the moments a node publishes nothing. True when the
 *   frontend declares no socket at all, so a build that tracks its connection some other way
 *   is never left waiting forever on a property it does not have.
 */
export function connected() {
  if (!api || !("socket" in api)) return true;
  const socket = api.socket;
  return !!socket && socket.readyState === WebSocket.OPEN;
}

// Every node a mounted panel is drawn on, counted once per open panel.
const watched = new Map();

let refreshInstalled = false;

/**
 * Whether an execution id names a node the server could hold anything for.
 *
 * @param {string} id - What {@link executionId} answered.
 * @returns {boolean} False for a blank id and for one carrying a negative part, which is
 *   what a node litegraph has built but not placed in a graph reports.
 */
export function placed(id) {
  const text = String(id ?? "").trim();
  if (!text) return false;
  return text.split(":").every((part) => {
    const value = Number(part);
    return !Number.isFinite(value) || value >= 0;
  });
}

/**
 * Send one registration request, which never raises and never blocks the caller.
 *
 * @param {string[]} ids - Execution ids to register or release.
 * @param {boolean} keep - True to register, false to release.
 * @returns {void}
 */
function postSubscription(ids, keep) {
  const wanted = ids.filter((id) => placed(id));
  if (!wanted.length) return;
  const query = new URLSearchParams();
  for (const id of wanted) query.append("node_id", id);
  query.set("watch", keep ? "1" : "0");
  Promise.resolve()
    .then(() => fetchWithin(`${SUBSCRIBE_ROUTE}?${query}`, { method: "POST" }))
    .catch((error) => {
      console.error(`${LOG_PREFIX} Failed to register interest in a node's pictures:`, error);
    });
}

/**
 * Re-send every live registration at the start of a run.
 *
 * @returns {void}
 */
function installRefresh() {
  if (refreshInstalled) return;
  refreshInstalled = true;
  api.addEventListener("execution_start", () => {
    if (!watched.size) return;
    postSubscription([...watched.keys()].map((node) => executionId(node)), true);
  });
}

/**
 * Register a node as one a panel is open on, so its publishes are worth encoding.
 *
 * @param {object} node - The node the interface is drawn on.
 * @returns {() => void} Release. Safe to call more than once, and called on dispose.
 */
export function watchPreviews(node) {
  if (!node) return () => {};
  installRefresh();
  watched.set(node, (watched.get(node) ?? 0) + 1);
  postSubscription([executionId(node)], true);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const held = (watched.get(node) ?? 1) - 1;
    if (held > 0) {
      watched.set(node, held);
      return;
    }
    watched.delete(node);
    postSubscription([executionId(node)], false);
  };
}

/**
 * Fetch one thumbnail a node published.
 *
 * @param {object|string|number} nodeOrId - The node itself, preferred so a node inside a
 *   subgraph resolves its execution id, or a bare id.
 * @param {{slot?: string, frame?: number, side?: string, expectedPromptId?: string}} [options] -
 *   Which picture to ask for. `slot` names the input it arrived on or the output it left on,
 *   `frame` counts from 0, `side` is a value of `PREVIEW_SIDE`, and `expectedPromptId` is the
 *   run this page is asking about: an answer from any other run is `FOREIGN` rather than
 *   `READY`, which is what a second tab publishing under the same node id produces.
 * @returns {Promise<{state: string, label: string, image: HTMLImageElement|null,
 *   sourceWidth?: number, sourceHeight?: number, scale?: number, frameCount?: number,
 *   frameTotal?: number, mode?: string, kind?: string, bytes?: number, promptId?: string,
 *   side?: string}>} The state, the words to draw for it, and on `READY` the image, the size of
 *   the picture it was reduced from, the factor from picture pixels to source pixels, how many
 *   frames the slot holds and how many the batch had, the channel mode, whether it is an image
 *   or a mask, the encoded picture's length in bytes, and the run it was published in.
 */
async function fetchPreview(nodeOrId, options = {}) {
  const side = options.side === PREVIEW_SIDE.OUTPUT ? PREVIEW_SIDE.OUTPUT : PREVIEW_SIDE.INPUT;
  const id = (typeof nodeOrId === "object" && nodeOrId !== null
    ? executionId(nodeOrId)
    : String(nodeOrId ?? "")).trim();
  if (!placed(id)) return result(PREVIEW_STATE.WAITING, null, side);
  if (!connected()) return result(PREVIEW_STATE.CONNECTING, null, side);
  // The parameter is left off entirely for a node holding one image, so the request an
  // interface written before slots existed sends is the one it always sent.
  const name = String(options.slot ?? "").trim();
  // The frame parameter is left off for the first one, so the request an interface written
  // before batches existed sends is the one it always sent.
  const index = Number(options.frame) || 0;
  const query = `${ROUTE}?node_id=${encodeURIComponent(id)}`
    + (name ? `&slot=${encodeURIComponent(name)}` : "")
    + (index ? `&frame=${encodeURIComponent(index)}` : "")
    + (side === PREVIEW_SIDE.OUTPUT ? `&side=${side}` : "");

  let blob = null;
  let source = { width: null, height: null };
  let frameCount = 1;
  let frameTotal = 0;
  let mode = "";
  let kind = "";
  let bytes = 0;
  let promptId = "";
  try {
    const response = await fetchWithin(query, {
      // A node runs again and publishes again, so a picture held from an earlier run would sit
      // under a fresh overlay and say something untrue about the graph.
      cache: "no-store",
    });
    // 404 is the answer for a node that published nothing, so it is a value rather than a
    // failure and is not logged. A caller never remembers this answer: a node queued before
    // the socket opened publishes on its next run, and one that recorded the first answer and
    // stopped asking would stay empty for the life of the page.
    if (response?.status === 404) return result(PREVIEW_STATE.WAITING, null, side);
    if (!response?.ok) return result(PREVIEW_STATE.FAILED, null, side);
    source = {
      width: response.headers.get(SOURCE_WIDTH_HEADER),
      height: response.headers.get(SOURCE_HEIGHT_HEADER),
    };
    frameCount = Number(response.headers.get(FRAME_COUNT_HEADER)) || 1;
    frameTotal = Number(response.headers.get(FRAME_TOTAL_HEADER)) || frameCount;
    mode = response.headers.get(SOURCE_MODE_HEADER) || "";
    kind = response.headers.get(KIND_HEADER) || "";
    bytes = Number(response.headers.get(ENCODED_BYTES_HEADER)) || 0;
    promptId = response.headers.get(PROMPT_HEADER) || "";
    blob = await response.blob();
  } catch (error) {
    console.error(
      `${LOG_PREFIX} Failed to ask for the ${side} preview of node ${id}`
        + `${name ? ` slot ${name}` : ""}:`,
      error,
    );
    return result(PREVIEW_STATE.FAILED, null, side);
  }

  if (!blob?.size) return result(PREVIEW_STATE.WAITING, null, side);
  const image = await decode(blob);
  if (!image) return result(PREVIEW_STATE.FAILED, null, side);
  // A picture whose run is known and is not the one asked for is drawn as its own state. The
  // store has no per-page scope, so two tabs whose graphs both hold a node numbered 5 file
  // under the same key, and the picture that comes back is whichever ran last.
  const expected = String(options.expectedPromptId ?? "").trim();
  const foreign = !!expected && !!promptId && expected !== promptId;
  // The picture is reduced to fit inside the node, so a gesture on it means nothing in
  // widget pixels without the size it was reduced from.
  const answer = result(foreign ? PREVIEW_STATE.FOREIGN : PREVIEW_STATE.READY, image, side);
  answer.sourceWidth = Number(source.width) || image.naturalWidth;
  answer.sourceHeight = Number(source.height) || image.naturalHeight;
  answer.scale = answer.sourceWidth / (image.naturalWidth || 1);
  answer.frameCount = frameCount;
  answer.frameTotal = Math.max(frameTotal, frameCount);
  answer.mode = mode;
  answer.kind = kind;
  answer.bytes = bytes;
  answer.promptId = promptId;
  return answer;
}

/**
 * Fetch one thumbnail of a picture a node received.
 *
 * @param {object|string|number} nodeOrId - The node itself, or a bare id.
 * @param {string} [slot] - Which of the node's images to ask for, named after the input it
 *   arrived on. Left out, the picture a node holding one image publishes.
 * @param {number} [frame] - Which frame of that slot, for a node that published a whole batch.
 * @param {string} [expectedPromptId] - The run being asked about, if one is known.
 * @returns {Promise<object>} The answer `fetchPreview` describes.
 */
export function fetchInputPreview(nodeOrId, slot = "", frame = 0, expectedPromptId = "") {
  return fetchPreview(nodeOrId, { slot, frame, side: PREVIEW_SIDE.INPUT, expectedPromptId });
}

/**
 * Fetch one thumbnail of a picture a node answered with.
 *
 * @param {object|string|number} nodeOrId - The node itself, or a bare id.
 * @param {string} [slot] - Which of the node's images to ask for, named after the output it
 *   left on. An output may share the name of an input, since the side is part of the key.
 * @param {number} [frame] - Which frame of that slot, for a node that published a whole batch.
 * @param {string} [expectedPromptId] - The run being asked about, if one is known.
 * @returns {Promise<object>} The answer `fetchPreview` describes.
 */
export function fetchOutputPreview(nodeOrId, slot = "", frame = 0, expectedPromptId = "") {
  return fetchPreview(nodeOrId, { slot, frame, side: PREVIEW_SIDE.OUTPUT, expectedPromptId });
}

/**
 * Fetch both sides of one slot, and say whether they belong to the same run.
 *
 * @param {object|string|number} nodeOrId - The node itself, or a bare id.
 * @param {string} [slot] - Which of the node's pictures to ask for, on both sides.
 * @param {number} [frame] - Which frame of that slot.
 * @param {string} [expectedPromptId] - The run being asked about, if one is known. Each side is
 *   `FOREIGN` rather than `READY` when it names a different run.
 * @returns {Promise<{before: object, after: object, sameRun: boolean}>} The two answers, and
 *   whether both name the same run. `sameRun` is false whenever either run is unknown: an empty
 *   prompt id is a publish made outside a run, and two of those are not evidence of a pair.
 */
export async function fetchPreviewPair(nodeOrId, slot = "", frame = 0, expectedPromptId = "") {
  const [before, after] = await Promise.all([
    fetchPreview(nodeOrId, { slot, frame, side: PREVIEW_SIDE.INPUT, expectedPromptId }),
    fetchPreview(nodeOrId, { slot, frame, side: PREVIEW_SIDE.OUTPUT, expectedPromptId }),
  ]);
  const one = String(before.promptId ?? "");
  const two = String(after.promptId ?? "");
  return { before, after, sameRun: !!one && !!two && one === two };
}

/**
 * Build the answer for one state.
 *
 * @param {string} state - A value of `PREVIEW_STATE`.
 * @param {HTMLImageElement|null} [image] - The image, for `READY` and `FOREIGN`.
 * @param {string} [side] - A value of `PREVIEW_SIDE`.
 * @returns {{state: string, label: string, image: HTMLImageElement|null, side: string}} The
 *   answer.
 */
function result(state, image = null, side = PREVIEW_SIDE.INPUT) {
  return { state, label: LABELS[state] ?? "", image, side };
}

/**
 * Decode PNG bytes into an image element.
 *
 * @param {Blob} blob - The bytes the endpoint answered with.
 * @returns {Promise<HTMLImageElement|null>} The decoded image, or null when the bytes could
 *   not be decoded.
 */
function decode(blob) {
  return new Promise((resolve) => {
    let url = "";
    try {
      url = URL.createObjectURL(blob);
    } catch (error) {
      console.error(`${LOG_PREFIX} Failed to read the preview bytes:`, error);
      resolve(null);
      return;
    }
    const picture = new Image();
    // The URL is released on both paths: the decoded image keeps its own copy of the
    // pixels, and one left unreleased holds the blob for the life of the page.
    picture.onload = () => {
      URL.revokeObjectURL(url);
      resolve(picture);
    };
    picture.onerror = () => {
      URL.revokeObjectURL(url);
      console.error(`${LOG_PREFIX} The preview bytes could not be decoded as an image.`);
      resolve(null);
    };
    picture.src = url;
  });
}
