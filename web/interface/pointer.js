/**
 * Where a pointer is in the pixels an interface draws in, and which wheel gestures a panel keeps.
 *
 * A canvas inside a node is scaled by the graph's zoom, so client pixels are not its own pixels.
 */

import { app } from "../../../scripts/app.js";

/**
 * Read a pointer position in an element's own pixels.
 *
 * @param {HTMLElement} element - The element the interface is drawn on.
 * @param {PointerEvent|MouseEvent} event - Event to read.
 * @returns {{x: number, y: number}} Position inside the element.
 */
export function elementPoint(element, event) {
  const rect = element.getBoundingClientRect();
  const scaleX = rect.width ? element.clientWidth / rect.width : 1;
  const scaleY = rect.height ? element.clientHeight / rect.height : 1;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

// The units a wheel gesture is measured in beyond pixels.
const DELTA_LINE = 1;
const DELTA_PAGE = 2;

// Pixels one line covers.
const LINE_PIXELS = 16;

/**
 * A wheel gesture's distance in pixels.
 *
 * @param {WheelEvent} event - The gesture.
 * @param {HTMLElement} [element] - The element being scrolled, for a gesture measured in pages.
 * @returns {{x: number, y: number}} The distance, in CSS pixels.
 */
export function wheelPixels(event, element) {
  if (event.deltaMode === DELTA_LINE) {
    return { x: event.deltaX * LINE_PIXELS, y: event.deltaY * LINE_PIXELS };
  }
  if (event.deltaMode === DELTA_PAGE) {
    return {
      x: event.deltaX * (element?.clientWidth || 0),
      y: event.deltaY * (element?.clientHeight || 0),
    };
  }
  return { x: event.deltaX, y: event.deltaY };
}

// How long a panel goes on taking gestures it has no use for after the last one it did use, in
// milliseconds. Running a list to its end and holding the wheel down stays in the list.
const LATCH_MS = 400;

/**
 * Take the wheel gestures a panel uses, and leave the rest to the graph.
 *
 * @param {HTMLElement} element - The panel's own element.
 * @param {(event: WheelEvent) => boolean} [onWheel] - Called with each gesture, for a panel that
 *   scrolls or steps through something. Answers true where the panel used the gesture. Left out,
 *   the panel uses none of them.
 * @returns {() => void} Releases the listener.
 */
export function captureWheel(element, onWheel) {
  let usedAt = -Infinity;
  const handler = (event) => {
    // Ctrl and Cmd are the frontend's own zoom modifier, and they reach the graph from anywhere
    // on the panel, including one whose plain wheel means something else.
    const zoom = event.ctrlKey || event.metaKey;
    const used = !zoom && onWheel?.(event) === true;
    if (used) {
      usedAt = event.timeStamp;
    } else if (zoom || event.timeStamp - usedAt >= LATCH_MS) {
      app.canvas?.processMouseWheel?.(event);
    }
    event.preventDefault();
    event.stopPropagation();
  };
  element.addEventListener("wheel", handler, { passive: false });
  return () => element.removeEventListener("wheel", handler);
}
