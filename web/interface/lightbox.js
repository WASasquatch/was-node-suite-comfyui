/**
 * A picture opened over the whole page, panned and zoomed, with its caption in a column beside it.
 *
 * `openLightbox` shows one entry of a list and steps through the rest, one lightbox at a time.
 */

import { CARD_LOOK, cardChip } from "./card.js";
import { WINDOW_Z_INDEX } from "./floating_window.js";
import { LOOK, SANS, SCROLLBARS, button, el } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.Lightbox";

// Above the floating windows and below litegraph's prompt dialog.
const Z_INDEX = WINDOW_Z_INDEX + 50;

// Width of the caption column, and the narrowest page that still gets one beside the picture.
const COLUMN_WIDTH = 340;
const SIDE_BY_SIDE = 720;

// Zoom bounds, the factor one zoom step applies, and the wheel distance that doubles the zoom.
const MIN_ZOOM = 0.02;
const MAX_ZOOM = 32;
const ZOOM_STEP = 1.25;
const WHEEL_DOUBLING = 360;

// Pixels a pointer moves before a press on the picture counts as a drag.
const DRAG_THRESHOLD = 3;

// Pixels an arrow key pans with Shift held.
const KEY_PAN = 80;

// How long the Copy button says what happened, in milliseconds.
const COPIED_MS = 1400;

const NO_CAPTION = "No caption saved with this picture.";

let current = null;

/**
 * Put text on the clipboard.
 *
 * @param {string} text - What to copy.
 * @returns {Promise<boolean>} True once the text is on the clipboard.
 */
async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (error) {
    console.error(`[${LOG_NAME}] The clipboard refused the caption:`, error);
  }
  // A page served over plain HTTP from another machine has no clipboard API.
  const holder = document.createElement("textarea");
  holder.value = text;
  holder.setAttribute("readonly", "");
  holder.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
  document.body.appendChild(holder);
  try {
    holder.select();
    return document.execCommand("copy");
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to copy the caption:`, error);
    return false;
  } finally {
    holder.remove();
  }
}

/**
 * Close the open lightbox, if there is one.
 *
 * @returns {void}
 */
export function closeLightbox() {
  current?.close();
}

/**
 * Open a list of pictures over the page at one of them.
 *
 * @param {object} options - What to show.
 * @param {object[]} options.items - The entries, each carrying `name` and `caption`.
 * @param {number} [options.index] - The entry shown first.
 * @param {(item: object) => {url: string, note?: string}} options.source - The address of an
 *   entry's picture, and a line saying what it is when it is not the file at full size.
 * @param {(index: number) => void} [options.onIndex] - Told the entry on show whenever it changes.
 * @returns {{close: () => void}} The open lightbox.
 */
export function openLightbox({ items, index = 0, source, onIndex } = {}) {
  closeLightbox();
  const list = Array.isArray(items) ? items : [];
  if (!list.length || typeof source !== "function") return { close() {} };

  const state = {
    index: Math.max(0, Math.min(list.length - 1, Number(index) || 0)),
    zoom: 1,
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    fitted: false,
    drag: null,
  };

  const root = el("div", `position:fixed;inset:0;z-index:${Z_INDEX};display:flex;`
    + `background:color-mix(in srgb, #000 82%, transparent);font:13px ${SANS};color:${LOOK.text}`);
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-modal", "true");
  root.tabIndex = -1;

  const stage = el("div", "position:relative;flex:1 1 auto;min-width:0;overflow:hidden;"
    + "cursor:grab;touch-action:none;user-select:none");
  const picture = document.createElement("img");
  picture.draggable = false;
  picture.decoding = "async";
  picture.style.cssText = "position:absolute;left:0;top:0;transform-origin:0 0;max-width:none;"
    + "max-height:none;image-rendering:auto;box-shadow:0 8px 40px rgba(0,0,0,0.6)";
  const status = el("div", "position:absolute;inset:0;display:flex;align-items:center;"
    + "justify-content:center;color:#cfcfcf;pointer-events:none");
  stage.append(picture, status);

  // The bar over the picture: where in the list, the zoom, and the zoom controls.
  const bar = el("div", "position:absolute;left:12px;top:12px;display:flex;align-items:center;"
    + `gap:6px;padding:5px 7px;border-radius:8px;background:${CARD_LOOK.surface};`
    + `border:1px solid ${CARD_LOOK.edge};box-shadow:${CARD_LOOK.shine}`);
  const counter = el("span", "font-variant-numeric:tabular-nums;color:" + LOOK.muted + ";padding:0 4px");
  const zoomLabel = el("span", "min-width:46px;text-align:center;font-variant-numeric:tabular-nums");
  bar.append(
    counter,
    button("−", () => zoomBy(1 / ZOOM_STEP), { hint: "Zoom out (-)", kind: "tool", logName: LOG_NAME }),
    zoomLabel,
    button("+", () => zoomBy(ZOOM_STEP), { hint: "Zoom in (+)", kind: "tool", logName: LOG_NAME }),
    button("Fit", () => fit(), { hint: "Fit the picture (0)", kind: "tool", logName: LOG_NAME }),
    button("1:1", () => actual(), { hint: "Full size (1)", kind: "tool", logName: LOG_NAME }),
  );
  stage.appendChild(bar);

  const arrow = (mark, step, side) => {
    const control = button(mark, () => go(step), {
      hint: step < 0 ? "Previous (Left arrow)" : "Next (Right arrow)", kind: "tool", logName: LOG_NAME,
    });
    control.style.position = "absolute";
    control.style.top = "50%";
    control.style[side] = "12px";
    control.style.transform = "translateY(-50%)";
    control.style.font = `20px/28px ${SANS}`;
    control.style.padding = "6px 10px";
    return control;
  };
  const previous = arrow("‹", -1, "left");
  const next = arrow("›", 1, "right");
  if (list.length > 1) stage.append(previous, next);

  // The column beside the picture: its name, its caption and the Copy button.
  const column = el("div", `flex:0 0 ${COLUMN_WIDTH}px;display:flex;flex-direction:column;gap:8px;`
    + `padding:12px;box-sizing:border-box;background:${CARD_LOOK.surface};`
    + `border-left:1px solid ${CARD_LOOK.edge};min-height:0`);
  const head = el("div", "display:flex;align-items:flex-start;gap:8px");
  const name = el("div", `flex:1 1 auto;min-width:0;font:600 13px/1.35 ${SANS};`
    + `color:${CARD_LOOK.title};overflow-wrap:anywhere`);
  const close = button("×", () => shut(), { hint: "Close (Esc)", kind: "tool", logName: LOG_NAME });
  close.style.font = `16px/18px ${SANS}`;
  head.append(name, close);
  const facts = el("div", "display:flex;flex-wrap:wrap;gap:4px");
  const label = el("div", `font:600 11px ${SANS};letter-spacing:0.05em;text-transform:uppercase;`
    + `color:${LOOK.muted}`, "Prompt");
  const caption = el("div", "flex:1 1 auto;min-height:0;overflow:auto;white-space:pre-wrap;"
    + `overflow-wrap:anywhere;padding:8px 10px;border-radius:6px;background:${CARD_LOOK.inset};`
    + `box-shadow:${CARD_LOOK.well};font:13px/1.5 ${SANS};user-select:text;${SCROLLBARS}`);
  const copy = button("Copy prompt", () => copyCaption(), {
    hint: "Copy the prompt to the clipboard", kind: "go", logName: LOG_NAME,
  });
  copy.style.alignSelf = "flex-start";
  column.append(head, facts, label, caption, copy);

  root.append(stage, column);

  let copiedTimer = 0;

  /**
   * Draw the picture where the pan and zoom put it.
   *
   * @returns {void}
   */
  function place() {
    picture.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.zoom})`;
    zoomLabel.textContent = state.width ? `${Math.round(state.zoom * 100)}%` : "";
  }

  /**
   * Keep some of the picture on the stage whatever the pan.
   *
   * @returns {void}
   */
  function hold() {
    const w = state.width * state.zoom;
    const h = state.height * state.zoom;
    const sw = stage.clientWidth;
    const sh = stage.clientHeight;
    // A picture smaller than the stage stays centred on that axis; a larger one keeps its edges
    // at or beyond the stage's.
    state.x = w <= sw ? (sw - w) / 2 : Math.min(0, Math.max(sw - w, state.x));
    state.y = h <= sh ? (sh - h) / 2 : Math.min(0, Math.max(sh - h, state.y));
  }

  /**
   * Zoom about one point of the stage.
   *
   * @param {number} zoom - The zoom wanted.
   * @param {number} [px] - The point's x on the stage, its middle when left out.
   * @param {number} [py] - The point's y on the stage.
   * @returns {void}
   */
  function zoomTo(zoom, px = stage.clientWidth / 2, py = stage.clientHeight / 2) {
    if (!state.width) return;
    const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
    state.x = px - ((px - state.x) * next) / state.zoom;
    state.y = py - ((py - state.y) * next) / state.zoom;
    state.zoom = next;
    state.fitted = false;
    hold();
    place();
  }

  function zoomBy(factor) {
    zoomTo(state.zoom * factor);
  }

  /**
   * Fit the whole picture on the stage, never past full size.
   *
   * @returns {void}
   */
  function fit() {
    if (!state.width) return;
    state.zoom = Math.min(1, stage.clientWidth / state.width, stage.clientHeight / state.height);
    state.fitted = true;
    hold();
    place();
  }

  /**
   * Show the picture at full size, centred.
   *
   * @returns {void}
   */
  function actual() {
    if (!state.width) return;
    state.zoom = 1;
    state.fitted = false;
    state.x = (stage.clientWidth - state.width) / 2;
    state.y = (stage.clientHeight - state.height) / 2;
    hold();
    place();
  }

  /**
   * Show one entry: its picture, its name and its caption.
   *
   * @returns {void}
   */
  function show() {
    const item = list[state.index] ?? {};
    counter.textContent = list.length > 1 ? `${state.index + 1} / ${list.length}` : "";
    previous.style.visibility = state.index > 0 ? "visible" : "hidden";
    next.style.visibility = state.index < list.length - 1 ? "visible" : "hidden";
    name.textContent = item.name ?? "";
    const text = typeof item.caption === "string" ? item.caption : "";
    caption.textContent = text || NO_CAPTION;
    caption.style.color = text ? LOOK.text : LOOK.muted;
    caption.style.fontStyle = text ? "normal" : "italic";
    caption.scrollTop = 0;
    copy.style.display = text ? "" : "none";
    copy.textContent = "Copy prompt";

    let address = { url: "" };
    try {
      address = source(item) ?? { url: "" };
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to find the picture's address:`, error);
    }
    facts.textContent = "";
    if (address.note) facts.appendChild(cardChip(address.note, "warning"));

    state.width = 0;
    state.height = 0;
    picture.style.visibility = "hidden";
    status.textContent = address.url ? "Loading" : "This picture is not in a folder the page can read.";
    place();
    if (!address.url) {
      picture.removeAttribute("src");
      return;
    }
    const shown = state.index;
    picture.onload = () => {
      if (shown !== state.index || !current) return;
      state.width = picture.naturalWidth;
      state.height = picture.naturalHeight;
      status.textContent = "";
      picture.style.visibility = "visible";
      // Full size where it fits or overflows the stage; fitted only where it would be lost.
      actual();
      facts.prepend(cardChip(`${state.width}x${state.height}`));
    };
    picture.onerror = () => {
      if (shown !== state.index) return;
      status.textContent = "Picture failed to load.";
    };
    picture.src = address.url;
    try {
      onIndex?.(state.index);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to report the entry on show:`, error);
    }
  }

  function go(step) {
    const target = state.index + step;
    if (target < 0 || target >= list.length) return;
    state.index = target;
    show();
  }

  async function copyCaption() {
    const text = list[state.index]?.caption;
    if (!text) return;
    const done = await copyText(String(text));
    copy.textContent = done ? "Copied" : "Copy failed";
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => { copy.textContent = "Copy prompt"; }, COPIED_MS);
  }

  /**
   * Lay the column beside the picture on a wide page and under it on a narrow one.
   *
   * @returns {void}
   */
  function layout() {
    const wide = window.innerWidth >= SIDE_BY_SIDE;
    root.style.flexDirection = wide ? "row" : "column";
    column.style.flex = wide ? `0 0 ${COLUMN_WIDTH}px` : "0 0 38%";
    column.style.borderLeft = wide ? `1px solid ${CARD_LOOK.edge}` : "none";
    column.style.borderTop = wide ? "none" : `1px solid ${CARD_LOOK.edge}`;
    if (state.fitted) fit();
    else {
      hold();
      place();
    }
  }

  const onWheel = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (!state.width) return;
    const rect = stage.getBoundingClientRect();
    const lines = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? stage.clientHeight : 1;
    // Ctrl, Cmd and a trackpad pinch zoom about the pointer; a plain wheel scrolls the picture.
    if (event.ctrlKey || event.metaKey) {
      zoomTo(
        state.zoom * 2 ** (-event.deltaY * lines / WHEEL_DOUBLING),
        event.clientX - rect.left,
        event.clientY - rect.top,
      );
      return;
    }
    const dx = (event.shiftKey && !event.deltaX ? event.deltaY : event.deltaX) * lines;
    const dy = (event.shiftKey && !event.deltaX ? 0 : event.deltaY) * lines;
    state.x -= dx;
    state.y -= dy;
    hold();
    place();
  };

  const onPointerDown = (event) => {
    if (event.button !== 0 || (event.target !== picture && event.target !== stage)) return;
    event.preventDefault();
    stage.setPointerCapture?.(event.pointerId);
    state.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, ox: state.x, oy: state.y, moved: false };
    stage.style.cursor = "grabbing";
  };
  const onPointerMove = (event) => {
    const drag = state.drag;
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    drag.moved = true;
    state.x = drag.ox + dx;
    state.y = drag.oy + dy;
    hold();
    place();
  };
  const onPointerUp = (event) => {
    const drag = state.drag;
    if (!drag || drag.id !== event.pointerId) return;
    state.drag = null;
    stage.style.cursor = "grab";
    // A click on the dark around the picture closes, as a click outside a dialog does.
    if (!drag.moved && event.target === stage) shut();
  };
  const onDoubleClick = (event) => {
    if (event.target !== picture) return;
    if (state.fitted || state.zoom < 1) {
      const rect = stage.getBoundingClientRect();
      zoomTo(1, event.clientX - rect.left, event.clientY - rect.top);
    } else {
      fit();
    }
  };

  const onKey = (event) => {
    const typing = event.target instanceof HTMLElement
      && (event.target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName));
    // Copy shortcuts in the caption are the browser's own, and the graph never sees them.
    if (typing || ((event.ctrlKey || event.metaKey) && !["0", "=", "+", "-"].includes(event.key))) {
      event.stopPropagation();
      return;
    }
    const keys = {
      Escape: () => shut(),
      ArrowLeft: () => (event.shiftKey ? pan(KEY_PAN, 0) : go(-1)),
      ArrowRight: () => (event.shiftKey ? pan(-KEY_PAN, 0) : go(1)),
      ArrowUp: () => pan(0, KEY_PAN),
      ArrowDown: () => pan(0, -KEY_PAN),
      Home: () => { state.index = 0; show(); },
      End: () => { state.index = list.length - 1; show(); },
      "+": () => zoomBy(ZOOM_STEP),
      "=": () => zoomBy(ZOOM_STEP),
      "-": () => zoomBy(1 / ZOOM_STEP),
      0: () => fit(),
      1: () => actual(),
    };
    const action = keys[event.key];
    // Every key stays out of the graph while the lightbox is open, so none deletes or queues.
    event.stopPropagation();
    if (!action) return;
    event.preventDefault();
    action();
  };

  function pan(dx, dy) {
    state.x += dx;
    state.y += dy;
    hold();
    place();
  }

  /**
   * Close the lightbox and release everything it listens to.
   *
   * @returns {void}
   */
  function shut() {
    if (current !== handle) return;
    current = null;
    clearTimeout(copiedTimer);
    window.removeEventListener("keydown", onKey, true);
    window.removeEventListener("resize", layout);
    picture.onload = null;
    picture.onerror = null;
    root.remove();
  }

  stage.addEventListener("wheel", onWheel, { passive: false });
  stage.addEventListener("pointerdown", onPointerDown);
  stage.addEventListener("pointermove", onPointerMove);
  stage.addEventListener("pointerup", onPointerUp);
  stage.addEventListener("pointercancel", onPointerUp);
  stage.addEventListener("dblclick", onDoubleClick);
  root.addEventListener("contextmenu", (event) => event.stopPropagation());
  // Text selected in the caption is copied as text, never as the graph's selected nodes.
  for (const kind of ["copy", "cut", "paste", "keyup", "keypress"]) {
    root.addEventListener(kind, (event) => event.stopPropagation());
  }
  // The column's own scrolling stays inside it rather than zooming the graph underneath.
  column.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  window.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", layout);

  const handle = { close: shut };
  current = handle;
  document.body.appendChild(root);
  layout();
  show();
  root.focus({ preventScroll: true });
  return handle;
}
