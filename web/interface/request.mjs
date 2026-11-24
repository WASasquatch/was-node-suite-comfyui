/**
 * Requests that give up rather than holding a connection open.
 *
 * A browser allows six connections to one host, so a handful of requests that never settle
 * starve every later one, including the queue's own.
 */

import { api } from "../../../scripts/api.js";

const LOG_NAME = "WASNodeSuite.Request";

//: How long a panel's request may take before it is abandoned.
export const REQUEST_TIMEOUT = 15000;

//: How long a request carrying a file may take, which is a size the network decides.
export const UPLOAD_TIMEOUT = 120000;

/**
 * Whether a rejection is this module giving up rather than the network failing.
 *
 * @param {unknown} error - The rejection to test.
 * @returns {boolean} True when the request was abandoned.
 */
export function timedOut(error) {
  return error instanceof DOMException && error.name === "AbortError";
}

/**
 * Ask the server for something, abandoning the request if it does not answer.
 *
 * @param {string} route - Route to ask, as `api.fetchApi` takes it.
 * @param {object} [options] - Request options. An `AbortController` signal is honoured
 *   alongside the deadline, so a caller can still cancel early.
 * @param {number} [timeout] - Milliseconds to wait. 0 waits with no limit.
 * @returns {Promise<Response>} What the server answered.
 * @throws {DOMException} `AbortError` when the deadline passed or the caller cancelled.
 */
export async function fetchWithin(route, options = {}, timeout = REQUEST_TIMEOUT) {
  if (!timeout) return api.fetchApi(route, options);

  const controller = new AbortController();
  const abandon = setTimeout(() => controller.abort(), timeout);

  // A caller's own signal has to reach the same controller, or cancelling early would be
  // ignored for the life of the deadline.
  const caller = options.signal;
  const relay = () => controller.abort();
  if (caller) {
    if (caller.aborted) controller.abort();
    else caller.addEventListener("abort", relay, { once: true });
  }

  try {
    return await api.fetchApi(route, { ...options, signal: controller.signal });
  } catch (error) {
    if (timedOut(error) && !caller?.aborted) {
      console.error(`[${LOG_NAME}] ${route} did not answer within ${timeout}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(abandon);
    caller?.removeEventListener("abort", relay);
  }
}
