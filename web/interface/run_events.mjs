/**
 * When a node finished, when a run ended, and which nodes came from cache.
 *
 * `onNodeFinished` fires once per run, `onRunEnded` however a run stops, `onNodeDone` at the
 * first of the two, and `onRunCached` for nodes served from cache.
 */

import { api } from "../../../scripts/api.js";
import { executionId } from "./preview.mjs";

const LOG_PREFIX = "[WASNodeSuite.RunEvents]";

// The three ways a run stops. Every one of them refreshes: a panel that listens only for success
// keeps drawing the last good answer after a run that was cancelled or that raised.
const RUN_ENDED = ["execution_success", "execution_error", "execution_interrupted"];

// The node state that means the body has returned. `comfy_execution/progress.py` sends the same
// four values the frontend passes through untouched.
const FINISHED = "finished";

/**
 * The finished entries a progress report holds for one canvas node: its own, and every loop
 * iteration's copy drawn as it.
 *
 * @param {object} detail - A `progress_state` event's detail.
 * @param {string} id - The node's execution id.
 * @returns {string[]} The reported ids of the finished entries.
 */
function finishedAs(detail, id) {
  const nodes = detail?.nodes;
  if (!nodes || typeof nodes !== "object") return [];
  const found = [];
  for (const [key, entry] of Object.entries(nodes)) {
    if (entry?.state !== FINISHED) continue;
    if (key === id || String(entry?.display_node_id ?? "") === id) found.push(key);
  }
  return found;
}

/**
 * Call a handler the first time a node is reported as finished in a run, and again for each loop
 * iteration's copy of it.
 *
 * @param {object} node - The node the interface is drawn on. Its execution id is composed with
 *   `executionId`, so a node inside a subgraph is matched under its colon joined path.
 * @param {(info: {promptId: string, nodeId: string, cached: boolean}) => void} handler - Called
 *   once per run for the node and once for each copy a loop runs, with the prompt it finished in
 *   and whether that prompt reported it as served from cache.
 * @returns {() => void} Unsubscribe. Safe to call more than once.
 */
export function onNodeFinished(node, handler) {
  if (typeof handler !== "function") return () => {};

  // Node ids served from cache in the current run, so a handler can tell a fresh picture from
  // one the store is still holding. `execution_cached` arrives before any node finishes.
  let cached = new Set();
  // The (prompt, node) pairs this subscription has already been told about, one set per
  // subscription, cleared when a run starts.
  const latched = new Set();
  const onCached = (event) => {
    const ids = event?.detail?.nodes;
    cached = new Set(Array.isArray(ids) ? ids.map((one) => String(one)) : []);
  };
  const onStart = () => {
    cached = new Set();
    latched.clear();
  };

  const onState = (event) => {
    try {
      const detail = event?.detail;
      const id = executionId(node);
      if (!id) return;
      const promptId = String(detail?.prompt_id ?? "");
      for (const reported of finishedAs(detail, id)) {
        const key = `${promptId}\u0000${reported}`;
        if (latched.has(key)) continue;
        latched.add(key);
        handler({ promptId, nodeId: id, cached: cached.has(reported) });
      }
    } catch (error) {
      console.error(`${LOG_PREFIX} A node-finished handler failed:`, error);
    }
  };

  api.addEventListener("execution_cached", onCached);
  api.addEventListener("execution_start", onStart);
  api.addEventListener("progress_state", onState);
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    api.removeEventListener?.("execution_cached", onCached);
    api.removeEventListener?.("execution_start", onStart);
    api.removeEventListener?.("progress_state", onState);
  };
}

/**
 * Call a handler when a run stops, however it stopped.
 *
 * @param {(info: {reason: string}) => void} handler - Called once per event, with the name of
 *   the event that ended the run.
 * @returns {() => void} Unsubscribe. Safe to call more than once.
 */
export function onRunEnded(handler) {
  if (typeof handler !== "function") return () => {};
  const listeners = RUN_ENDED.map((name) => {
    const listener = () => {
      try {
        handler({ reason: name });
      } catch (error) {
        console.error(`${LOG_PREFIX} A run-ended handler failed:`, error);
      }
    };
    api.addEventListener(name, listener);
    return [name, listener];
  });
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    for (const [name, listener] of listeners) api.removeEventListener?.(name, listener);
  };
}

/**
 * Call a handler as soon as a node finishes, or when a run stops without it having finished.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {() => void} handler - Called once per run: when the node's body returns, or when the
 *   run ends with the node never reported finished in it.
 * @returns {() => void} Unsubscribe. Safe to call more than once.
 */
export function onNodeDone(node, handler) {
  if (typeof handler !== "function") return () => {};
  let finished = false;
  const onStart = () => {
    finished = false;
  };
  api.addEventListener("execution_start", onStart);
  const stopFinished = onNodeFinished(node, () => {
    finished = true;
    handler();
  });
  const stopEnded = onRunEnded(() => {
    const seen = finished;
    finished = false;
    if (!seen) handler();
  });
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    api.removeEventListener?.("execution_start", onStart);
    stopFinished();
    stopEnded();
  };
}

/**
 * Call a handler with the nodes a run served from cache.
 *
 * @param {(info: {promptId: string, nodes: Set<string>}) => void} handler - Called once per run
 *   that reused anything, with the execution ids that were not executed.
 * @returns {() => void} Unsubscribe. Safe to call more than once.
 */
export function onRunCached(handler) {
  if (typeof handler !== "function") return () => {};
  const listener = (event) => {
    try {
      const ids = event?.detail?.nodes;
      handler({
        promptId: String(event?.detail?.prompt_id ?? ""),
        nodes: new Set(Array.isArray(ids) ? ids.map((one) => String(one)) : []),
      });
    } catch (error) {
      console.error(`${LOG_PREFIX} A run-cached handler failed:`, error);
    }
  };
  api.addEventListener("execution_cached", listener);
  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    api.removeEventListener?.("execution_cached", listener);
  };
}
