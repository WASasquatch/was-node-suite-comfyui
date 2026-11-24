/**
 * The picture an interface draws where the node's own has not arrived.
 *
 * One decoded image serves the page. `placeholderPicture` answers null until it lands and after
 * a failure.
 */

import { ICON, iconTitle } from "./icons.mjs";
import { PREVIEW_STATE } from "./preview.mjs";

const LOG_PREFIX = "[WASNodeSuite.Placeholder]";

// Resolved against this module's own URL. The pack's `web/` is served under the project name from
// `pyproject.toml`, and a name written out here as well is a name the two can disagree on.
const SOURCE = new URL("../images/was-ns-placeholder.png", import.meta.url).href;

/** The glyph a stand-in carries, since it stands for a picture the run has not produced. */
export const STAND_IN_ICON = ICON.WARNING;

// What the hover says after the glyph's own lead, for each state that leaves an interface without
// the node's picture, so every interface words the same condition alike.
const STAND_IN_DETAIL = {
  [PREVIEW_STATE.CONNECTING]: "Not connected",
  [PREVIEW_STATE.WAITING]: "Run node",
  [PREVIEW_STATE.LOADING]: "Loading",
  [PREVIEW_STATE.FAILED]: "Load failed",
};

// What it says where the state gives no reason of its own.
const NO_BACKDROP_DETAIL = "No picture";

// The decoded picture, and the one ask for it. Both are the page's rather than an interface's: the
// file is a fixed asset, so a second copy would be the same pixels again at the same cost.
let picture = null;
let request = null;

/**
 * The pack's hover text for a stand-in, in the words of whatever left the picture missing.
 *
 * @param {string} state - A value of `PREVIEW_STATE`.
 * @returns {string} The hover text, led by the words the glyph carries everywhere else.
 */
export function standInTitle(state) {
  return iconTitle(STAND_IN_ICON, standInDetail(state));
}

/**
 * What a stand-in says for one state, without the glyph's own lead.
 *
 * @param {string} state - A value of `PREVIEW_STATE`.
 * @returns {string} The clause, in the pack's words for that state.
 */
export function standInDetail(state) {
  return STAND_IN_DETAIL[state] ?? NO_BACKDROP_DETAIL;
}

/**
 * The picture, where it has arrived.
 *
 * @returns {HTMLImageElement|null} The picture, or null before it lands and after a failure.
 */
export function placeholderPicture() {
  return picture;
}

/**
 * Ask for the picture, once for the page.
 *
 * @returns {Promise<HTMLImageElement|null>} The same promise for every caller, resolving to the
 *   picture or to null where it could not be decoded. It never rejects, so an interface that
 *   repaints when it lands needs nothing around the call.
 */
export function loadPlaceholder() {
  if (!request) request = decode();
  return request;
}

/**
 * Draw the picture fitted inside a rectangle, keeping its own aspect.
 *
 * @param {CanvasRenderingContext2D} ctx - Context to draw into.
 * @param {{x: number, y: number, w: number, h: number}} rect - Where the node's own picture would
 *   go, in element pixels.
 * @returns {boolean} True when it was drawn, false while there is nothing to draw it with.
 */
export function drawStandIn(ctx, rect) {
  const image = picture;
  if (!image || !(rect?.w > 0) || !(rect?.h > 0)) return false;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!(width > 0) || !(height > 0)) return false;

  // Fitted rather than filled. The rectangle is the frame the adopter vouches for, and cropping
  // this picture to that shape would carry the wordmark's ends outside it on anything far from
  // square, so what is left over stays the surface's own background.
  const fit = Math.min(rect.w / width, rect.h / height);
  const drawWidth = Math.max(1, width * fit);
  const drawHeight = Math.max(1, height * fit);
  try {
    ctx.drawImage(
      image,
      rect.x + (rect.w - drawWidth) / 2,
      rect.y + (rect.h - drawHeight) / 2,
      drawWidth,
      drawHeight,
    );
  } catch (error) {
    console.error(`${LOG_PREFIX} Failed to draw the placeholder:`, error);
    return false;
  }
  return true;
}

/**
 * Decode the picture the pack ships.
 *
 * @returns {Promise<HTMLImageElement|null>} The picture, or null where it could not be loaded.
 */
function decode() {
  return new Promise((resolve) => {
    let element = null;
    try {
      element = new Image();
    } catch (error) {
      console.error(`${LOG_PREFIX} Failed to ask for the placeholder:`, error);
      resolve(null);
      return;
    }
    element.onload = () => {
      picture = element;
      resolve(element);
    };
    // A picture that cannot be had leaves every interface drawing what it drew before there was
    // one, so the failure is logged and answered rather than raised.
    element.onerror = () => {
      console.error(`${LOG_PREFIX} The placeholder could not be decoded as an image.`);
      resolve(null);
    };
    element.src = SOURCE;
  });
}
