/**
 * The picture an interface draws its widgets over, and the frame they are measured in.
 *
 * Every number written against a backdrop is in frame units: one pixel of the picture the node
 * held, not one pixel on screen.
 */

import { LABELS, PREVIEW_SIDE, PREVIEW_STATE, fetchInputPreview, fetchOutputPreview } from "./preview.mjs";

/**
 * Read a number, answering a fallback for anything that is not one.
 *
 * @param {*} value - Value to read.
 * @param {number} fallback - What to answer when the value is not a finite number.
 * @returns {number} The number, or the fallback.
 */
function toNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/**
 * Build the answer a backdrop resolves to, with anything missing filled in.
 *
 * @param {object} answer - Whatever the backdrop's `load` resolved to.
 * @returns {{state: string, label: string, image: HTMLImageElement|null, width: number,
 *   height: number, scale: number, frameSource: string}} The state, the words to draw for it,
 *   the picture where there is one, the frame's own size in frame units, the factor from picture
 *   pixels to frame units, and which side of the node the frame belongs to.
 */
export function normaliseFrame(answer) {
  const state = typeof answer?.state === "string" ? answer.state : PREVIEW_STATE.READY;
  const width = Math.max(0, toNumber(answer?.width, 0));
  const height = Math.max(0, toNumber(answer?.height, 0));
  const scale = Math.max(0, toNumber(answer?.scale, 1)) || 1;
  return {
    state,
    label: typeof answer?.label === "string" ? answer.label : (LABELS[state] ?? ""),
    image: answer?.image ?? null,
    width,
    height,
    scale,
    frameSource: answer?.frameSource === PREVIEW_SIDE.OUTPUT
      ? PREVIEW_SIDE.OUTPUT
      : PREVIEW_SIDE.INPUT,
  };
}

/**
 * The backdrop that draws the picture a node published on its last run.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {{width?: number, height?: number}} [fallback] - A frame to use until the picture
 *   arrives, in frame units. Stated, the interface is usable before the node has ever run,
 *   against the size given rather than against the image. Left out, it states no frame and
 *   refuses every gesture until a picture arrives, since it has no size to convert a gesture
 *   through. Neither way is the frame ever the size of whatever picture is standing in for the
 *   node's own, which would change what a stored number means once the real one arrived.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
export function imageBackdrop(node, fallback = {}) {
  return sideBackdrop(node, fallback, PREVIEW_SIDE.INPUT);
}

/**
 * The backdrop that draws the picture a node answered with on its last run.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {{width?: number, height?: number}} [fallback] - A frame to use until the picture
 *   arrives, in frame units, on the same terms as `imageBackdrop`.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
export function outputBackdrop(node, fallback = {}) {
  return sideBackdrop(node, fallback, PREVIEW_SIDE.OUTPUT);
}

/**
 * Build the backdrop for one side of a node.
 *
 * @param {object} node - The node the interface is drawn on.
 * @param {{width?: number, height?: number}} fallback - The frame to use until a picture arrives.
 * @param {string} side - A value of `PREVIEW_SIDE`.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
function sideBackdrop(node, fallback, side) {
  const width = Math.max(0, toNumber(fallback?.width, 0));
  const height = Math.max(0, toNumber(fallback?.height, 0));
  const fetchOne = side === PREVIEW_SIDE.OUTPUT ? fetchOutputPreview : fetchInputPreview;
  return {
    async load() {
      const answer = await fetchOne(node);
      if (answer?.state === PREVIEW_STATE.READY) {
        return {
          ...answer,
          width: answer.sourceWidth,
          height: answer.sourceHeight,
          frameSource: side,
        };
      }
      return { ...answer, width, height, scale: 1, frameSource: side };
    },
  };
}

/**
 * The backdrop that states a size and draws nothing behind the widgets.
 *
 * @param {{width: number, height: number, label?: string}} frame - The frame's size in frame
 *   units, and words to draw in the middle of it.
 * @returns {{load: () => Promise<object>}} A backdrop.
 */
export function blankBackdrop(frame = {}) {
  const answer = {
    state: PREVIEW_STATE.READY,
    label: typeof frame.label === "string" ? frame.label : "",
    image: null,
    width: Math.max(0, toNumber(frame.width, 0)),
    height: Math.max(0, toNumber(frame.height, 0)),
    scale: 1,
    frameSource: PREVIEW_SIDE.INPUT,
  };
  return {
    async load() {
      return answer;
    },
  };
}
