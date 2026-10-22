/**
 * Where a compare panel's written sides are served from, and the bytes of each held in the page.
 *
 * A held side plays from a blob URL, so the panel keeps it while the server is away.
 */

import { api } from "../../../scripts/api.js";

const LOG_NAME = "WASNodeSuite.CompareMedia";

// The `type` of a side kept where a restart leaves it.
export const KEPT_TYPE = "was_compare";

// The route serving a kept side by name.
const KEPT_ROUTE = "/was/interface/api/compare_file";

// Bytes held across every panel before the least recently used unwatched side is let go.
const HELD_BYTES = 1024 * 1024 * 1024;

// A side larger than this is played from the server rather than held.
const STREAM_BYTES = 512 * 1024 * 1024;

// `address -> {promise, url, bytes, users, used}`.
const held = new Map();

/**
 * The address a written side is served from.
 *
 * @param {object} entry - A side's `filename`, `subfolder` and `type`.
 * @returns {string} The address, or an empty string when the entry names no file.
 */
export function addressOf(entry) {
  if (!entry?.filename) return "";
  if (entry.type === KEPT_TYPE) {
    return api.apiURL(`${KEPT_ROUTE}?${new URLSearchParams({ name: entry.filename })}`);
  }
  const parts = new URLSearchParams({
    filename: entry.filename,
    subfolder: entry.subfolder ?? "",
    type: entry.type ?? "temp",
  });
  return api.apiURL(`/view?${parts.toString()}`);
}

/** Let go of the least recently used unwatched sides until the page holds under `HELD_BYTES`. */
function trim() {
  let total = 0;
  for (const entry of held.values()) total += entry.bytes;
  if (total <= HELD_BYTES) return;
  const idle = [...held.entries()]
    .filter(([, entry]) => entry.users === 0 && entry.url)
    .sort((a, b) => a[1].used - b[1].used);
  for (const [address, entry] of idle) {
    if (total <= HELD_BYTES) break;
    URL.revokeObjectURL(entry.url);
    held.delete(address);
    total -= entry.bytes;
  }
}

/**
 * Read one side's bytes into the page.
 *
 * @param {string} address - Where the side is served from.
 * @returns {Promise<{url: string, bytes: number}|null>} A blob URL and its size, the address
 *   itself with a size of 0 for a side past `STREAM_BYTES`, or null when it could not be read.
 */
async function fetchSide(address) {
  const answer = await fetch(address);
  if (answer.status !== 200 && answer.status !== 206) {
    answer.body?.cancel?.();
    return null;
  }
  const length = Number(answer.headers.get("content-length")) || 0;
  if (length > STREAM_BYTES) {
    answer.body?.cancel?.();
    return { url: address, bytes: 0 };
  }
  const blob = await answer.blob();
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

/**
 * Hold one side in the page for a panel that shows it.
 *
 * @param {string} address - Where the side is served from.
 * @returns {Promise<string|null>} The URL to show it from, or null when it could not be read.
 *   A failed read is forgotten, so a later call reads it again. Each call is paired with
 *   `release`.
 */
export function hold(address) {
  if (!address) return Promise.resolve(null);
  let entry = held.get(address);
  if (!entry) {
    entry = { promise: null, url: "", bytes: 0, users: 0, used: 0 };
    held.set(address, entry);
    const current = entry;
    entry.promise = fetchSide(address)
      .then((read) => {
        if (!read) {
          if (held.get(address) === current) held.delete(address);
          return null;
        }
        if (held.get(address) !== current) {
          if (read.url.startsWith("blob:")) URL.revokeObjectURL(read.url);
          return null;
        }
        current.url = read.url;
        current.bytes = read.bytes;
        trim();
        return read.url;
      })
      .catch((error) => {
        console.warn(`[${LOG_NAME}] A compare side could not be read:`, error);
        if (held.get(address) === current) held.delete(address);
        return null;
      });
  }
  entry.users += 1;
  entry.used = performance.now();
  return entry.promise;
}

/**
 * Stop showing a side one `hold` call asked for. It stays held for another panel until the
 * page needs the room.
 *
 * @param {string} address - The address passed to `hold`.
 * @returns {void}
 */
export function release(address) {
  const entry = held.get(address);
  if (!entry) return;
  entry.users = Math.max(0, entry.users - 1);
  entry.used = performance.now();
  trim();
}
