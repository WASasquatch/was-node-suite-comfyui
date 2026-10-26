/**
 * A drawn section under an audio panel's waveform: a curve or a scope, sharp at any zoom, that
 * may take drags, the wheel and keys.
 *
 * Coordinates handed to a section are element pixels.
 */

import { elementPoint } from "./pointer.js";
import { surfaceRatio, watchSurfaceRatio } from "./resolution.js";
import { onThemeChange, readTheme } from "./theme.js";
import { el } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.AudioGraph";

/** Frequencies a log axis spans by default, in hertz. */
export const LOG_LOW = 20;
export const LOG_HIGH = 20000;

/**
 * A position on a log frequency axis.
 *
 * @param {number} hertz - The frequency.
 * @param {number} width - The axis length in pixels.
 * @param {number} [high] - The highest frequency drawn.
 * @returns {number} Pixels from the left.
 */
export function logX(hertz, width, high = LOG_HIGH) {
  return (Math.log(Math.max(LOG_LOW, hertz) / LOG_LOW) / Math.log(high / LOG_LOW)) * width;
}

/**
 * The frequency at a position on a log axis.
 *
 * @param {number} x - Pixels from the left.
 * @param {number} width - The axis length in pixels.
 * @param {number} [high] - The highest frequency drawn.
 * @returns {number} Hertz.
 */
export function logHz(x, width, high = LOG_HIGH) {
  return LOG_LOW * Math.pow(high / LOG_LOW, Math.max(0, Math.min(1, x / width)));
}

/**
 * Build a section.
 *
 * @param {object} options - What it draws and takes.
 * @param {number} [options.height] - Its height in node units.
 * @param {(ctx: CanvasRenderingContext2D, box: {width: number, height: number}, theme: object) => void} options.draw -
 *   Paints the section.
 * @param {object} [options.pointer] - `down(at, event)` answering true to take the gesture, `move(at, event)`,
 *   `up(at, event)`, `cursor(at)` and `doubleClick(at, event)`.
 * @param {(event: WheelEvent, at: object) => boolean} [options.wheel] - Answers whether it used a gesture.
 * @param {(event: KeyboardEvent) => boolean} [options.key] - Answers whether it used a key.
 * @param {string} [options.hint] - Hover text naming the gestures.
 * @returns {{element: HTMLElement, repaint: () => void, dispose: () => void}} The section.
 */
export function createGraphSection(options) {
  const element = el("div", `position:relative;flex:0 0 auto;height:${options.height ?? 110}px;`
    + "border-radius:3px;overflow:hidden;outline:none");
  element.tabIndex = 0;
  if (options.hint) element.title = options.hint;
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none");
  element.append(canvas);
  const context = canvas.getContext("2d");
  let theme = readTheme();
  let frame = 0;
  let gesture = false;

  const draw = () => {
    frame = 0;
    if (!context) return;
    const ratio = surfaceRatio(canvas);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!(width > 0) || !(height > 0)) return;
    const pixelWidth = Math.max(1, Math.round(width * ratio));
    const pixelHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, pixelWidth, pixelHeight);
    context.scale(pixelWidth / width, pixelHeight / height);
    context.fillStyle = theme.bgDark ?? theme.bg;
    context.fillRect(0, 0, width, height);
    try {
      options.draw(context, { width, height }, theme);
    } catch (error) {
      console.error(`[${LOG_NAME}] A section failed to draw:`, error);
    }
  };
  const repaint = () => {
    if (!frame) frame = requestAnimationFrame(draw);
  };

  const at = (event) => elementPoint(canvas, event);
  const end = (event) => {
    if (!gesture) return;
    gesture = false;
    try {
      options.pointer?.up?.(at(event), event);
    } catch (error) {
      console.error(`[${LOG_NAME}] A section gesture failed to end:`, error);
    }
    try {
      canvas.releasePointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] The pointer was already released:`, error);
    }
    repaint();
  };
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      window.app?.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.stopPropagation();
    element.focus({ preventScroll: true });
    if (!options.pointer?.down?.(at(event), event)) return;
    gesture = true;
    try {
      canvas.setPointerCapture?.(event.pointerId);
    } catch (error) {
      console.debug(`[${LOG_NAME}] The pointer could not be captured:`, error);
    }
    repaint();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (gesture && event.buttons === 0) {
      end(event);
      return;
    }
    if (!gesture) {
      canvas.style.cursor = options.pointer?.cursor?.(at(event)) || "default";
      return;
    }
    event.stopPropagation();
    options.pointer?.move?.(at(event), event);
    repaint();
  });
  canvas.addEventListener("pointerup", (event) => {
    event.stopPropagation();
    end(event);
  });
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("lostpointercapture", end);
  canvas.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    if (options.pointer?.doubleClick?.(at(event), event)) repaint();
  });
  element.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  if (options.wheel) {
    element.setAttribute("data-capture-wheel", "true");
    element.addEventListener("wheel", (event) => {
      const zoom = event.ctrlKey || event.metaKey;
      let used = false;
      if (!zoom) {
        try {
          used = options.wheel(event, at(event)) === true;
        } catch (error) {
          console.error(`[${LOG_NAME}] A section wheel handler failed:`, error);
        }
      }
      if (!used) window.app?.canvas?.processMouseWheel?.(event);
      event.preventDefault();
      event.stopPropagation();
      if (used) repaint();
    }, { passive: false });
  }
  element.addEventListener("keydown", (event) => {
    if (event.target !== element || !options.key) return;
    let used = false;
    try {
      used = options.key(event) === true;
    } catch (error) {
      console.error(`[${LOG_NAME}] A section key handler failed:`, error);
    }
    if (used) {
      event.preventDefault();
      event.stopPropagation();
      repaint();
    }
  });

  const resized = new ResizeObserver(() => repaint());
  resized.observe(element);
  const stopRatio = watchSurfaceRatio(canvas, () => repaint());
  const stopTheme = onThemeChange((next) => {
    theme = next;
    repaint();
  });
  repaint();

  return {
    element,
    repaint,
    dispose: () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      resized.disconnect();
      stopRatio();
      stopTheme();
    },
  };
}
