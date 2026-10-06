/**
 * A window a node opens beside the graph: dragged, resized, collapsed and remembered.
 *
 * `createFloatingWindow` builds one over `document.body` and answers the handle that opens,
 * closes, collapses, maximizes and disposes it. Sizes are in CSS pixels.
 */

import { LOOK, SANS } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.FloatingWindow";

// Where the windows stack: above the canvas action bar at 1300, below litegraph's prompt
// dialog at 1500 and the PrimeVue dialogs from 1800.
export const WINDOW_Z_INDEX = 1400;

// Steps the stack climbs above the base before it is renumbered.
const Z_BAND = 99;

// Pixels of a window kept inside the viewport.
const VIEWPORT_MARGIN = 48;

// The sizes a caller gets when it names none.
const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 480;
const DEFAULT_MIN_WIDTH = 240;
const DEFAULT_MIN_HEIGHT = 160;

// The title bar, its buttons, its icon, the resize edges and the corner grip.
const TITLE_HEIGHT = 44;
const BUTTON_SIZE = 24;
const ICON_SIZE = 22;
const EDGE_SIZE = 6;
const GRIP_SIZE = 16;

// Pixels a press on a maximized window's title moves before the window is dragged out of it.
const DRAG_THRESHOLD = 5;

// Keys the window gives a meaning to, held back from the graph while no text field is focused.
const WINDOW_KEYS = new Set([
  "Escape", "Enter", " ", "Tab", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Home", "End", "PageUp", "PageDown", "Delete", "Backspace",
]);

// Elements whose keystrokes and context menu are the browser's own.
const TEXT_FIELDS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

const SVG_NS = "http://www.w3.org/2000/svg";

// The marks on the title bar's buttons.
const FOLDED = "\u25B8";
const UNFOLDED = "\u25BE";
const CLOSE_MARK = "\u00D7";
const MAXIMIZE_MARK = "\u25A1";
const RESTORE_MARK = "\u2750";

// What a window behind the one on top fades its body to.
const INACTIVE_OPACITY = "0.82";

// The windows in the page, the one on top, and its z-index. Each window answers a call that
// draws it in front or behind.
const stacked = new Set();
const shading = new WeakMap();
let topWindow = null;
let topZ = WINDOW_Z_INDEX;

/**
 * A number above zero, or a fallback.
 *
 * @param {unknown} value - What the caller passed.
 * @param {number} fallback - What stands in for anything else.
 * @returns {number} The value when it is a number above zero, else the fallback.
 */
function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

/**
 * Whether an element takes typed text.
 *
 * @param {EventTarget|null} target - What an event was aimed at.
 * @returns {boolean} True for an input, a textarea, a select or an editable element.
 */
function isTextField(target) {
  if (!(target instanceof Element)) return false;
  return TEXT_FIELDS.has(target.tagName) || target.isContentEditable === true;
}

/**
 * Whether an event landed on a button.
 *
 * @param {EventTarget|null} target - What the event was aimed at.
 * @returns {boolean} True inside a button.
 */
function onButton(target) {
  return target instanceof Element && target.closest("button") !== null;
}

/**
 * A glyph filled in the current text colour.
 *
 * @param {string[]} paths - The glyph, as path data in a 24 unit box.
 * @param {number} size - Pixels a side.
 * @returns {SVGElement} The glyph.
 */
function svgIcon(paths, size) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.style.cssText = "display:block;flex:0 0 auto;fill:currentColor;opacity:0.85;pointer-events:none";
  for (const path of paths) {
    const shape = document.createElementNS(SVG_NS, "path");
    shape.setAttribute("d", path);
    svg.appendChild(shape);
  }
  return svg;
}

/**
 * One button on the title bar.
 *
 * @param {string} hint - What the hover and the accessible name say.
 * @param {string} mark - The character drawn on it.
 * @param {number} size - The character's size in pixels.
 * @returns {HTMLButtonElement} The button, for the caller to append and to wire.
 */
function createBarButton(hint, mark, size) {
  const button = document.createElement("button");
  button.type = "button";
  button.title = hint;
  button.setAttribute("aria-label", hint);
  button.textContent = mark;
  button.style.cssText = [
    "flex:0 0 auto",
    `width:${BUTTON_SIZE}px`,
    `height:${BUTTON_SIZE}px`,
    "padding:0",
    "display:inline-flex",
    "align-items:center",
    "justify-content:center",
    "border:0",
    "border-radius:5px",
    "background:none",
    `color:${LOOK.muted}`,
    "cursor:pointer",
    `font:${size}px/1 ${SANS}`,
  ].join(";");
  button.onmouseenter = () => {
    button.style.background = LOOK.hover;
    button.style.color = LOOK.text;
  };
  button.onmouseleave = () => {
    button.style.background = "none";
    button.style.color = LOOK.muted;
  };
  return button;
}

/**
 * The geometry stored under one key.
 *
 * @param {string} key - The storage key, or an empty string for none.
 * @param {string} logName - The prefix a failure is logged under.
 * @returns {object|null} The stored object, or null when there is none or it cannot be read.
 */
function readStored(key, logName) {
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch (error) {
    console.warn(`[${logName}] Could not read ${key} from storage:`, error);
    return null;
  }
}

/**
 * Store the geometry under one key.
 *
 * @param {string} key - The storage key, or an empty string for none.
 * @param {object} value - What to store.
 * @param {string} logName - The prefix a failure is logged under.
 * @returns {void}
 */
function writeStored(key, value, logName) {
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`[${logName}] Could not write ${key} to storage:`, error);
  }
}

/**
 * Put one window above every other window in the band.
 *
 * @param {HTMLElement} root - The window's element.
 * @returns {void}
 */
function bringToFront(root) {
  stacked.add(root);
  if (topWindow === root) return;
  topWindow = root;
  for (const one of stacked) {
    try {
      shading.get(one)?.(one === root);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to shade a window:`, error);
    }
  }
  if (topZ >= WINDOW_Z_INDEX + Z_BAND) {
    const order = Array.from(stacked)
      .filter((one) => one.isConnected)
      .sort((a, b) => Number(a.style.zIndex) - Number(b.style.zIndex));
    stacked.clear();
    topZ = WINDOW_Z_INDEX - 1;
    for (const one of order) {
      topZ += 1;
      one.style.zIndex = String(topZ);
      stacked.add(one);
    }
  }
  topZ += 1;
  root.style.zIndex = String(topZ);
}

/**
 * Follow one pointer from a press until it is released, cancelled or lost.
 *
 * @param {HTMLElement} element - The element that captures the pointer.
 * @param {PointerEvent} event - The press that starts the gesture.
 * @param {(event: PointerEvent) => void} onMove - Called with each move while a button is held.
 * @param {() => void} onEnd - Called once when the gesture ends, however it ends.
 * @param {string} logName - The prefix a failure is logged under.
 * @returns {void}
 */
function beginGesture(element, event, onMove, onEnd, logName) {
  const id = event.pointerId;
  let live = true;
  const finish = () => {
    if (!live) return;
    live = false;
    element.removeEventListener("pointermove", move);
    element.removeEventListener("pointerup", finish);
    element.removeEventListener("pointercancel", finish);
    element.removeEventListener("lostpointercapture", finish);
    try {
      if (element.hasPointerCapture?.(id)) element.releasePointerCapture(id);
    } catch (error) {
      console.error(`[${logName}] Failed to release the pointer:`, error);
    }
    onEnd();
  };
  const move = (moved) => {
    if (moved.pointerId !== id) return;
    if ((moved.buttons & 1) === 0) {
      finish();
      return;
    }
    onMove(moved);
  };
  element.addEventListener("pointermove", move);
  element.addEventListener("pointerup", finish);
  element.addEventListener("pointercancel", finish);
  element.addEventListener("lostpointercapture", finish);
  try {
    element.setPointerCapture(id);
  } catch (error) {
    console.error(`[${logName}] Failed to capture the pointer:`, error);
  }
}

/**
 * Build a draggable, resizable, collapsible window over the page.
 *
 * @param {object} [options] - How the window is built.
 * @param {string} [options.title] - The text on the title bar.
 * @param {string} [options.badge] - A quieter line beside the title, as what the window is on.
 * @param {string[]} [options.icon] - A glyph before the title, as path data in a 24 unit box.
 * @param {number} [options.width] - Width in CSS pixels, 640 by default.
 * @param {number} [options.height] - Height in CSS pixels, 480 by default.
 * @param {number} [options.minWidth] - Narrowest a resize allows, 240 by default.
 * @param {number} [options.minHeight] - Shortest a resize allows, 160 by default.
 * @param {string} [options.storageKey] - The `localStorage` key the position, the size, the
 *   collapsed state and the maximized state are remembered under. Left out, nothing is
 *   remembered.
 * @param {() => void} [options.onClose] - Called after the close button or Escape closes it.
 * @param {string} [options.logName] - The prefix a failure is logged under.
 * @returns {{element: HTMLElement, body: HTMLElement, titleBar: HTMLElement,
 *   setTitle: (text: string) => void, setBadge: (text: string) => void, open: () => void,
 *   close: () => void,
 *   toggle: () => void, isOpen: () => boolean, collapse: () => void, expand: () => void,
 *   isCollapsed: () => boolean, maximize: () => void, restore: () => void,
 *   isMaximized: () => boolean, dispose: () => void}} The window: its element, the body
 *   content goes in, the title bar, and the calls that drive it.
 */
export function createFloatingWindow(options = {}) {
  const logName = options.logName || LOG_NAME;
  const storageKey = typeof options.storageKey === "string" ? options.storageKey : "";
  const minWidth = positive(options.minWidth, DEFAULT_MIN_WIDTH);
  const minHeight = positive(options.minHeight, DEFAULT_MIN_HEIGHT);
  const onClose = typeof options.onClose === "function" ? options.onClose : null;

  // The expanded geometry. A collapsed window keeps its height here and draws the title bar.
  const rect = {
    left: 0,
    top: 0,
    width: Math.max(minWidth, positive(options.width, DEFAULT_WIDTH)),
    height: Math.max(minHeight, positive(options.height, DEFAULT_HEIGHT)),
  };
  let collapsed = false;
  let maximized = false;
  // The geometry a maximized window goes back to.
  let restored = null;
  let placed = false;
  let watching = false;
  let disposed = false;

  const root = document.createElement("div");
  root.tabIndex = -1;
  root.setAttribute("role", "dialog");
  root.style.cssText = [
    "position:fixed",
    "display:none",
    "flex-direction:column",
    "box-sizing:border-box",
    `z-index:${WINDOW_Z_INDEX}`,
    "overflow:hidden",
    "border-radius:10px",
    `background:${LOOK.body}`,
    `color:${LOOK.text}`,
    `border:1px solid ${LOOK.border}`,
    `box-shadow:0 10px 40px ${LOOK.shadow}`,
    `font:13px/1.5 ${SANS}`,
    "outline:none",
  ].join(";");

  const titleBar = document.createElement("div");
  titleBar.style.cssText = [
    "flex:0 0 auto",
    `min-height:${TITLE_HEIGHT}px`,
    "box-sizing:border-box",
    "display:flex",
    "align-items:center",
    "gap:8px",
    "padding:0 10px",
    `background:${LOOK.surface}`,
    `border-bottom:1px solid ${LOOK.border}`,
    "cursor:move",
    "user-select:none",
    "touch-action:none",
  ].join(";");

  const titleText = document.createElement("span");
  titleText.style.cssText = "flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + "white-space:nowrap;font-size:15px;font-weight:600";
  const badgeText = document.createElement("span");
  badgeText.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;font-size:13px;color:${LOOK.muted}`;

  const collapseButton = createBarButton("Collapse", UNFOLDED, 16);
  const maximizeButton = createBarButton("Maximize", MAXIMIZE_MARK, 15);
  const closeButton = createBarButton("Close", CLOSE_MARK, 20);
  titleBar.append(collapseButton);
  if (Array.isArray(options.icon) && options.icon.length) titleBar.append(svgIcon(options.icon, ICON_SIZE));
  titleBar.append(titleText, badgeText, maximizeButton, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "flex:1 1 auto;display:flex;flex-direction:column;min-height:0;"
    + "overflow:hidden";

  const rightEdge = createHandle(`top:0;right:0;bottom:0;width:${EDGE_SIZE}px;cursor:ew-resize`);
  const bottomEdge = createHandle(
    `left:0;right:0;bottom:0;height:${EDGE_SIZE}px;cursor:ns-resize`,
  );
  const grip = createHandle(
    `right:0;bottom:0;width:${GRIP_SIZE}px;height:${GRIP_SIZE}px;cursor:nwse-resize`,
  );
  const gripMark = document.createElement("span");
  gripMark.style.cssText = "position:absolute;right:4px;bottom:4px;width:7px;height:7px;"
    + `border-right:2px solid ${LOOK.muted};border-bottom:2px solid ${LOOK.muted};opacity:0.6;`
    + "pointer-events:none";
  grip.appendChild(gripMark);
  const handles = [[rightEdge, "block"], [bottomEdge, "block"], [grip, "flex"]];

  root.append(titleBar, body, rightEdge, bottomEdge, grip);

  /**
   * One resize edge or grip.
   *
   * @param {string} style - Where it sits and which cursor it shows.
   * @returns {HTMLElement} The handle.
   */
  function createHandle(style) {
    const handle = document.createElement("div");
    handle.style.cssText = "position:absolute;z-index:1;touch-action:none;user-select:none;"
      + style;
    return handle;
  }

  /**
   * Write the geometry onto the element.
   *
   * @returns {void}
   */
  function applyRect() {
    root.style.left = `${rect.left}px`;
    root.style.top = `${rect.top}px`;
    root.style.width = `${rect.width}px`;
    root.style.height = collapsed ? "auto" : `${rect.height}px`;
    root.style.borderRadius = maximized ? "0" : "10px";
  }

  /**
   * Fill the viewport with a maximized window.
   *
   * @returns {void}
   */
  function fill() {
    rect.left = 0;
    rect.top = 0;
    rect.width = Math.max(minWidth, Math.round(window.innerWidth || rect.width));
    rect.height = Math.max(minHeight, Math.round(window.innerHeight || rect.height));
    applyRect();
  }

  /**
   * Draw the maximize button for the state the window is in.
   *
   * @returns {void}
   */
  function markMaximized() {
    const hint = maximized ? "Restore" : "Maximize";
    maximizeButton.title = hint;
    maximizeButton.setAttribute("aria-label", hint);
    maximizeButton.textContent = maximized ? RESTORE_MARK : MAXIMIZE_MARK;
    for (const [handle, shown] of handles) handle.style.display = collapsed || maximized ? "none" : shown;
  }

  /**
   * Hold the window inside the viewport and draw it there.
   *
   * @returns {boolean} True when the geometry moved.
   */
  function place() {
    const before = `${rect.left},${rect.top},${rect.width},${rect.height}`;
    if (maximized) {
      fill();
      return before !== `${rect.left},${rect.top},${rect.width},${rect.height}`;
    }
    const viewWidth = Math.max(0, window.innerWidth || 0);
    const viewHeight = Math.max(0, window.innerHeight || 0);
    rect.width = Math.round(Math.max(minWidth, Math.min(rect.width, viewWidth || rect.width)));
    rect.height = Math.round(
      Math.max(minHeight, Math.min(rect.height, viewHeight || rect.height)),
    );
    const leftMost = VIEWPORT_MARGIN - rect.width;
    const rightMost = Math.max(leftMost, viewWidth - VIEWPORT_MARGIN);
    rect.left = Math.round(Math.min(Math.max(rect.left, leftMost), rightMost));
    const lowest = Math.max(0, viewHeight - VIEWPORT_MARGIN);
    rect.top = Math.round(Math.min(Math.max(rect.top, 0), lowest));
    applyRect();
    return before !== `${rect.left},${rect.top},${rect.width},${rect.height}`;
  }

  /**
   * Store the geometry under the storage key.
   *
   * @returns {void}
   */
  function remember() {
    const kept = maximized && restored ? restored : rect;
    writeStored(storageKey, {
      left: kept.left,
      top: kept.top,
      width: kept.width,
      height: kept.height,
      collapsed,
      maximized,
    }, logName);
  }

  /**
   * Draw the window collapsed to its title bar, or expanded.
   *
   * @param {boolean} next - True for collapsed.
   * @returns {void}
   */
  function setCollapsed(next) {
    collapsed = next === true;
    body.style.display = collapsed ? "none" : "flex";
    for (const [handle, shown] of handles) handle.style.display = collapsed || maximized ? "none" : shown;
    const hint = collapsed ? "Expand" : "Collapse";
    collapseButton.title = hint;
    collapseButton.setAttribute("aria-label", hint);
    collapseButton.setAttribute("aria-expanded", collapsed ? "false" : "true");
    collapseButton.textContent = collapsed ? FOLDED : UNFOLDED;
    applyRect();
  }

  /**
   * Whether the window is in the page and shown.
   *
   * @returns {boolean} True while open.
   */
  function isOpen() {
    return !disposed && root.isConnected && root.style.display !== "none";
  }

  /**
   * Show the window, on its remembered geometry or centred the first time.
   *
   * @returns {void}
   */
  function open() {
    if (disposed) return;
    if (!root.isConnected) document.body.appendChild(root);
    if (!placed) {
      placed = true;
      const stored = readStored(storageKey, logName);
      if (stored) {
        rect.width = Math.max(minWidth, positive(stored.width, rect.width));
        rect.height = Math.max(minHeight, positive(stored.height, rect.height));
        if (stored.collapsed === true) setCollapsed(true);
      }
      if (stored && Number.isFinite(stored.left) && Number.isFinite(stored.top)) {
        rect.left = Number(stored.left);
        rect.top = Number(stored.top);
      } else {
        rect.left = ((window.innerWidth || 0) - rect.width) / 2;
        rect.top = ((window.innerHeight || 0) - rect.height) / 2;
      }
      if (stored?.maximized === true) {
        restored = { ...rect };
        maximized = true;
        markMaximized();
      }
    }
    root.style.display = "flex";
    place();
    bringToFront(root);
    if (!watching) {
      watching = true;
      window.addEventListener("resize", onViewportResize);
    }
    root.focus({ preventScroll: true });
  }

  /**
   * Hide the window. Its geometry is kept.
   *
   * @returns {void}
   */
  function close() {
    if (disposed) return;
    root.style.display = "none";
  }

  /**
   * Open the window when it is closed, and close it when it is open.
   *
   * @returns {void}
   */
  function toggle() {
    if (isOpen()) close();
    else open();
  }

  /**
   * Close the window at the user's request and tell the caller.
   *
   * @returns {void}
   */
  function userClose() {
    close();
    if (!onClose) return;
    try {
      onClose();
    } catch (error) {
      console.error(`[${logName}] The close handler failed:`, error);
    }
  }

  /**
   * Draw only the title bar.
   *
   * @returns {void}
   */
  function collapse() {
    if (disposed || collapsed) return;
    setCollapsed(true);
    remember();
  }

  /**
   * Draw the body under the title bar again.
   *
   * @returns {void}
   */
  function expand() {
    if (disposed || !collapsed) return;
    setCollapsed(false);
    place();
    remember();
  }

  /**
   * Whether only the title bar is drawn.
   *
   * @returns {boolean} True while collapsed.
   */
  function isCollapsed() {
    return collapsed;
  }

  /**
   * Fill the viewport, keeping the geometry to go back to.
   *
   * @returns {void}
   */
  function maximize() {
    if (disposed || maximized) return;
    restored = { ...rect };
    maximized = true;
    if (collapsed) setCollapsed(false);
    markMaximized();
    fill();
    remember();
  }

  /**
   * Go back to the geometry the window had before it was maximized.
   *
   * @returns {void}
   */
  function restore() {
    if (disposed || !maximized) return;
    maximized = false;
    if (restored) Object.assign(rect, restored);
    restored = null;
    markMaximized();
    place();
    remember();
  }

  /**
   * Whether the window fills the viewport.
   *
   * @returns {boolean} True while maximized.
   */
  function isMaximized() {
    return maximized;
  }

  /**
   * Set the text on the title bar.
   *
   * @param {string} text - The title.
   * @returns {void}
   */
  function setTitle(text) {
    const title = String(text ?? "");
    titleText.textContent = title;
    titleText.title = title;
    root.setAttribute("aria-label", title);
  }

  /**
   * Set the quieter line beside the title.
   *
   * @param {string} text - What it says, or an empty string for none.
   * @returns {void}
   */
  function setBadge(text) {
    badgeText.textContent = String(text ?? "");
    badgeText.title = badgeText.textContent;
  }

  /**
   * Draw the window in front of the others, or receding behind them.
   *
   * @param {boolean} active - True for the window on top.
   * @returns {void}
   */
  function shade(active) {
    titleBar.style.background = active ? LOOK.surface : LOOK.hover;
    titleText.style.color = active ? LOOK.text : LOOK.muted;
    body.style.opacity = active ? "1" : INACTIVE_OPACITY;
    root.style.borderColor = active ? LOOK.border : `color-mix(in srgb, ${LOOK.border} 60%, transparent)`;
  }
  shading.set(root, shade);

  /**
   * Take the window out of the page and release its listener on the viewport.
   *
   * @returns {void}
   */
  function dispose() {
    if (disposed) return;
    disposed = true;
    if (watching) window.removeEventListener("resize", onViewportResize);
    stacked.delete(root);
    shading.delete(root);
    if (topWindow === root) topWindow = null;
    root.remove();
  }

  /**
   * Hold the window inside a viewport that changed size.
   *
   * @returns {void}
   */
  function onViewportResize() {
    if (!isOpen()) return;
    if (place()) remember();
  }

  /**
   * Start resizing from one edge or the grip.
   *
   * @param {PointerEvent} event - The press on the handle.
   * @param {HTMLElement} handle - The handle pressed.
   * @param {boolean} horizontal - Whether the width follows the pointer.
   * @param {boolean} vertical - Whether the height follows the pointer.
   * @returns {void}
   */
  function startResize(event, handle, horizontal, vertical) {
    if (event.button !== 0 || collapsed || disposed) return;
    event.preventDefault();
    const origin = { x: event.clientX, y: event.clientY, width: rect.width, height: rect.height };
    beginGesture(handle, event, (moved) => {
      if (horizontal) {
        const widest = Math.max(minWidth, (window.innerWidth || 0) - rect.left);
        const wanted = origin.width + moved.clientX - origin.x;
        rect.width = Math.round(Math.min(widest, Math.max(minWidth, wanted)));
      }
      if (vertical) {
        const tallest = Math.max(minHeight, (window.innerHeight || 0) - rect.top);
        const wanted = origin.height + moved.clientY - origin.y;
        rect.height = Math.round(Math.min(tallest, Math.max(minHeight, wanted)));
      }
      applyRect();
    }, () => {
      place();
      remember();
    }, logName);
  }

  rightEdge.addEventListener("pointerdown", (event) => {
    startResize(event, rightEdge, true, false);
  });
  bottomEdge.addEventListener("pointerdown", (event) => {
    startResize(event, bottomEdge, false, true);
  });
  grip.addEventListener("pointerdown", (event) => {
    startResize(event, grip, true, true);
  });

  titleBar.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || disposed || onButton(event.target)) return;
    const start = { x: event.clientX, y: event.clientY };
    let origin = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    beginGesture(titleBar, event, (moved) => {
      if (maximized) {
        if (Math.hypot(moved.clientX - start.x, moved.clientY - start.y) < DRAG_THRESHOLD) return;
        // Dragged out of maximized, the window takes back its size under the pointer.
        const across = rect.width > 0 ? (start.x - rect.left) / rect.width : 0.5;
        restore();
        origin = { x: Math.round(across * rect.width), y: Math.round(TITLE_HEIGHT / 2) };
      }
      rect.left = Math.round(moved.clientX - origin.x);
      rect.top = Math.round(moved.clientY - origin.y);
      applyRect();
    }, () => {
      if (maximized) return;
      place();
      remember();
    }, logName);
  });

  titleBar.addEventListener("dblclick", (event) => {
    if (onButton(event.target)) return;
    event.preventDefault();
    if (collapsed) expand();
    else collapse();
  });

  collapseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    if (collapsed) expand();
    else collapse();
  });

  maximizeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    if (maximized) restore();
    else maximize();
  });

  closeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    userClose();
  });

  // The window pressed last is drawn on top.
  root.addEventListener("pointerdown", () => bringToFront(root), true);

  root.addEventListener("keydown", (event) => {
    if (isTextField(event.target)) return;
    if (event.ctrlKey || event.metaKey) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      userClose();
      return;
    }
    if (WINDOW_KEYS.has(event.key)) event.stopPropagation();
  });

  root.addEventListener("contextmenu", (event) => {
    if (!isTextField(event.target)) event.preventDefault();
  });

  setTitle(options.title ?? "");
  setBadge(options.badge ?? "");
  setCollapsed(false);

  return {
    element: root,
    body,
    titleBar,
    setTitle,
    setBadge,
    open,
    close,
    toggle,
    isOpen,
    collapse,
    expand,
    isCollapsed,
    maximize,
    restore,
    isMaximized,
    dispose,
  };
}
