/**
 * The frames each segment of a run was last drawn at, as one sheet per segment.
 *
 * Sheets are announced by `was-segment-preview` and fetched from the segment preview route; the
 * newest version of each segment wins.
 */

import { api } from "../../../scripts/api.js";

const LOG_NAME = "WASNodeSuite.SegmentPreviews";

/** The message a new sheet is announced by. */
export const PREVIEW_EVENT = "was-segment-preview";

/** The node property a node's sheets are filed under. */
export const PREVIEW_KEY = "was_preview_key";

const ROUTE = "/was/interface/api/segment_preview";
const INDEX_ROUTE = `${ROUTE}/index`;

/**
 * This tab's client id, which the server serves a run's sheets to.
 *
 * @returns {string} The id, or an empty string before the socket has given one.
 */
function clientId() {
  return String(api?.clientId || api?.initialClientId || "");
}

/**
 * A random version 4 UUID, on pages served over plain HTTP as well as HTTPS.
 *
 * @returns {string} The UUID.
 */
function newPreviewKey() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Give a node a preview key of its own, or a new one where another node shares it.
 *
 * @param {object} node - The node.
 * @param {object[]} [workflows] - Other open workflows whose saved state may hold the key, as a
 *   workflow saved as a copy of another does.
 * @returns {boolean} True when a key was written.
 */
export function ensurePreviewKey(node, workflows = []) {
  if (!node) return false;
  node.properties ??= {};
  const key = node.properties[PREVIEW_KEY];
  const holds = (saved) => saved?.properties?.[PREVIEW_KEY] === key;
  const shared = Boolean(key) && (
    (node.graph?.nodes ?? []).some((other) => other !== node && holds(other))
    || workflows.some((other) => (other?.changeTracker?.activeState?.nodes ?? []).some(holds))
  );
  if (key && !shared) return false;
  node.properties[PREVIEW_KEY] = newPreviewKey();
  return true;
}

/**
 * Follow the sheets one node's segments are drawn on.
 *
 * @param {() => string} ownerId - The node's execution id, read on every message.
 * @param {(segment: number, preview: object) => void} onPreview - Called once a sheet is ready to
 *   draw, with `{fields, image}`.
 * @param {(segment: number, fields: object) => void} [onProblem] - Called when a segment could
 *   not be previewed, with `{problem, detail}` among the fields.
 * @returns {{previews: Map<number, object>, problems: Map<number, object>,
 *   reload: (owner?: string) => Promise<void>, dispose: () => void}} The held sheets by segment,
 *   why a segment has none, a re-read of everything the server holds for the node or for another
 *   key, and unsubscribe.
 */
export function followSegmentPreviews(ownerId, onPreview, onProblem) {
  const previews = new Map();
  const problems = new Map();
  const wanted = new Map();
  let heldFor = null;
  let disposed = false;

  // Sheets held for another key are dropped as soon as the node's key changes.
  const own = (owner) => {
    if (heldFor === owner) return;
    for (const preview of previews.values()) preview.image?.close?.();
    previews.clear();
    problems.clear();
    wanted.clear();
    heldFor = owner;
  };

  const load = async (fields) => {
    const segment = Number(fields.segment);
    if ((wanted.get(segment) ?? -1) > fields.version) return;
    wanted.set(segment, fields.version);
    if (fields.problem) {
      previews.get(segment)?.image?.close?.();
      previews.delete(segment);
      problems.set(segment, fields);
      onProblem?.(segment, fields);
      return;
    }
    try {
      const response = await api.fetchApi(
        `${ROUTE}?node_id=${encodeURIComponent(fields.node_id)}&segment=${segment}&v=${fields.version}` +
          `&client_id=${encodeURIComponent(clientId())}`,
        { cache: "no-store" },
      );
      if (response.status !== 200) return;
      const image = await createImageBitmap(await response.blob());
      if (disposed || wanted.get(segment) !== fields.version || heldFor !== String(ownerId())) {
        image.close?.();
        return;
      }
      previews.get(segment)?.image?.close?.();
      const preview = { fields, image };
      previews.set(segment, preview);
      problems.delete(segment);
      onPreview(segment, preview);
    } catch (error) {
      console.warn(`[${LOG_NAME}] Segment ${segment + 1}'s preview could not be read:`, error);
    }
  };

  const listener = (event) => {
    const fields = event?.detail;
    if (!fields || String(fields.node_id) !== String(ownerId())) return;
    own(String(ownerId()));
    load(fields);
  };
  api.addEventListener(PREVIEW_EVENT, listener);

  const reload = async (owner = ownerId()) => {
    own(String(ownerId()));
    try {
      const response = await api.fetchApi(
        `${INDEX_ROUTE}?node_id=${encodeURIComponent(owner)}&client_id=${encodeURIComponent(clientId())}`, {
        cache: "no-store",
      });
      if (response.status !== 200) return;
      const held = await response.json();
      await Promise.all((Array.isArray(held) ? held : []).map((fields) => load(fields)));
    } catch (error) {
      console.warn(`[${LOG_NAME}] The held previews could not be listed:`, error);
    }
  };

  return {
    previews,
    problems,
    reload,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      api.removeEventListener(PREVIEW_EVENT, listener);
      for (const preview of previews.values()) preview.image?.close?.();
      previews.clear();
      problems.clear();
    },
  };
}

/**
 * Draw one frame of a sheet into a box, filling it and cropping the overflow.
 *
 * @param {CanvasRenderingContext2D} pen - Where to draw.
 * @param {{fields: object, image: CanvasImageSource}} preview - The sheet.
 * @param {number} frame - Frame of the sheet, clamped to the ones it holds.
 * @param {number} x - Left edge.
 * @param {number} y - Top edge.
 * @param {number} w - Width.
 * @param {number} h - Height.
 * @param {boolean} [fit=false] - Letterbox inside the box rather than fill it.
 * @returns {void}
 */
export function drawSheetFrame(pen, preview, frame, x, y, w, h, fit = false) {
  const { fields, image } = preview;
  const count = Math.max(1, Number(fields.frames) || 1);
  const index = Math.max(0, Math.min(count - 1, Math.round(frame)));
  const columns = Math.max(1, Number(fields.columns) || 1);
  const cw = Number(fields.cell_width) || 1;
  const ch = Number(fields.cell_height) || 1;
  const sx = (index % columns) * cw;
  const sy = Math.floor(index / columns) * ch;
  const scale = fit ? Math.min(w / cw, h / ch) : Math.max(w / cw, h / ch);
  const dw = cw * scale;
  const dh = ch * scale;
  if (fit) {
    pen.drawImage(image, sx, sy, cw, ch, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh);
    return;
  }
  const cropW = w / scale;
  const cropH = h / scale;
  pen.drawImage(image, sx + (cw - cropW) / 2, sy + (ch - cropH) / 2, cropW, cropH, x, y, w, h);
}
