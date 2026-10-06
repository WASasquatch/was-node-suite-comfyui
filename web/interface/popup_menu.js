/**
 * A menu opened at the pointer over a floating window: entries, headers, rules and swatches.
 *
 * One menu is open on the page at a time. Positions are client pixels.
 */

import { WINDOW_Z_INDEX } from "./floating_window.js";
import { FLOATING_PANEL, LABEL, LOOK, SANS, SCROLLBARS } from "./window_kit.js";

const LOG_NAME = "WASNodeSuite.PopupMenu";

// Above every floating window, below ComfyUI's dialogs.
const MENU_Z_INDEX = WINDOW_Z_INDEX + 150;

// Pixels of a menu kept inside the viewport.
const MARGIN = 8;

// Pixels a side of one colour swatch.
const SWATCH = 18;

// The menu on the page, and how to close it.
let openMenu = null;

/**
 * Close whichever menu is open.
 *
 * @returns {void}
 */
export function closePopupMenu() {
  openMenu?.close();
}

/**
 * One pressable row.
 *
 * @param {object} item - `{label, detail, checked, disabled, danger, colour, onSelect}`, `colour`
 *   a CSS colour drawn as a dot before the label.
 * @param {() => void} close - Closes the menu.
 * @param {string} logName - The prefix a failure is logged under.
 * @returns {HTMLButtonElement} The row.
 */
function entryRow(item, close, logName) {
  const row = document.createElement("button");
  row.type = "button";
  row.setAttribute("role", "menuitem");
  row.disabled = item.disabled === true;
  row.style.cssText = [
    "display:flex",
    "align-items:center",
    "gap:8px",
    "width:100%",
    "box-sizing:border-box",
    "padding:7px 12px 7px 8px",
    "border:0",
    "border-radius:6px",
    "background:none",
    "text-align:left",
    `font:13px/1.5 ${SANS}`,
    `color:${item.danger ? LOOK.danger : LOOK.text}`,
    `cursor:${row.disabled ? "default" : "pointer"}`,
    `opacity:${row.disabled ? "0.45" : "1"}`,
    "white-space:nowrap",
  ].join(";");
  const tick = document.createElement("span");
  tick.textContent = item.checked ? "✓" : "";
  tick.style.cssText = `flex:0 0 14px;text-align:center;color:${LOOK.accent}`;
  const label = document.createElement("span");
  label.textContent = String(item.label ?? "");
  label.style.cssText = "flex:1 1 auto";
  row.append(tick);
  if (item.colour) {
    const dot = document.createElement("span");
    dot.style.cssText = `flex:0 0 9px;height:9px;border-radius:50%;background:${item.colour}`;
    row.appendChild(dot);
  }
  row.appendChild(label);
  if (item.detail) {
    const detail = document.createElement("span");
    detail.textContent = String(item.detail);
    detail.style.cssText = `flex:0 0 auto;margin-left:16px;font-size:11px;color:${LOOK.muted}`;
    row.appendChild(detail);
  }
  const light = () => {
    if (!row.disabled) row.style.background = LOOK.hover;
  };
  const dim = () => {
    row.style.background = "none";
  };
  row.onmouseenter = light;
  row.onmouseleave = dim;
  row.onfocus = light;
  row.onblur = dim;
  row.addEventListener("click", (event) => {
    event.stopPropagation();
    if (row.disabled) return;
    close();
    try {
      item.onSelect?.();
    } catch (error) {
      console.error(`[${logName}] A menu entry failed:`, error);
    }
  });
  return row;
}

/**
 * A row of colour swatches.
 *
 * @param {object} item - `{swatches: [{name, colour, title}], current, onPick}`.
 * @param {() => void} close - Closes the menu.
 * @param {string} logName - The prefix a failure is logged under.
 * @returns {HTMLElement} The row.
 */
function swatchRow(item, close, logName) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;padding:6px 10px 8px 30px";
  for (const swatch of item.swatches ?? []) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.title = String(swatch.title ?? swatch.name ?? "");
    chip.setAttribute("aria-label", chip.title);
    const chosen = swatch.name === item.current;
    chip.style.cssText = [
      `width:${SWATCH}px`,
      `height:${SWATCH}px`,
      "padding:0",
      "border-radius:50%",
      "cursor:pointer",
      `background:${swatch.colour}`,
      `border:2px solid ${chosen ? LOOK.text : "transparent"}`,
      `box-shadow:0 0 0 1px ${LOOK.border}`,
    ].join(";");
    chip.addEventListener("click", (event) => {
      event.stopPropagation();
      close();
      try {
        item.onPick?.(swatch.name);
      } catch (error) {
        console.error(`[${logName}] A swatch failed:`, error);
      }
    });
    row.appendChild(chip);
  }
  return row;
}

/**
 * Open a menu at a point, closing any other.
 *
 * @param {object} options - What the menu holds and where it opens.
 * @param {object[]} options.items - Rows, in order: `{label, detail, checked, disabled,
 *   danger, colour, onSelect}`, `{header}`, `{separator: true}` or `{swatches, current, onPick}`.
 * @param {number} options.x - Left edge, in client pixels.
 * @param {number} options.y - Top edge, in client pixels.
 * @param {string} [options.logName] - The prefix a failure is logged under.
 * @returns {{close: () => void}} The menu.
 */
export function openPopupMenu(options = {}) {
  closePopupMenu();
  const logName = options.logName || LOG_NAME;
  const root = document.createElement("div");
  root.setAttribute("role", "menu");
  root.tabIndex = -1;
  root.style.cssText = [
    "position:fixed",
    `z-index:${MENU_Z_INDEX}`,
    "min-width:200px",
    "max-width:460px",
    "max-height:70vh",
    "overflow:auto",
    "box-sizing:border-box",
    "padding:4px",
    FLOATING_PANEL,
    SCROLLBARS,
    "outline:none",
  ].join(";");

  let live = true;
  const close = () => {
    if (!live) return;
    live = false;
    document.removeEventListener("pointerdown", onOutside, true);
    window.removeEventListener("blur", close);
    root.remove();
    if (openMenu?.root === root) openMenu = null;
  };
  const onOutside = (event) => {
    if (!root.contains(event.target)) close();
  };

  for (const item of options.items ?? []) {
    if (!item) continue;
    if (item.separator) {
      const rule = document.createElement("div");
      rule.style.cssText = `height:1px;margin:4px 6px;background:${LOOK.border}`;
      root.appendChild(rule);
    } else if (item.header) {
      const header = document.createElement("div");
      header.textContent = String(item.header);
      header.style.cssText = `padding:7px 10px 3px 30px;${LABEL}`;
      root.appendChild(header);
    } else if (item.swatches) {
      root.appendChild(swatchRow(item, close, logName));
    } else {
      root.appendChild(entryRow(item, close, logName));
    }
  }

  root.addEventListener("keydown", (event) => {
    event.stopPropagation();
    // Enter and Space press the focused row through the button's own click.
    if (event.key === "Enter" || event.key === " ") return;
    const rows = Array.from(root.querySelectorAll("button:not([disabled])"));
    const at = rows.indexOf(document.activeElement);
    const step = event.key === "ArrowDown" || (event.key === "Tab" && !event.shiftKey) ? 1
      : event.key === "ArrowUp" || (event.key === "Tab" && event.shiftKey) ? -1 : 0;
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (step && rows.length) {
      event.preventDefault();
      const next = at < 0 ? (step > 0 ? 0 : rows.length - 1) : (at + step + rows.length) % rows.length;
      rows[next].focus({ preventScroll: true });
    }
  });
  root.addEventListener("contextmenu", (event) => event.preventDefault());

  document.body.appendChild(root);
  const width = root.offsetWidth;
  const height = root.offsetHeight;
  const left = Math.max(MARGIN, Math.min(Number(options.x) || 0, window.innerWidth - width - MARGIN));
  const top = Math.max(MARGIN, Math.min(Number(options.y) || 0, window.innerHeight - height - MARGIN));
  root.style.left = `${Math.round(left)}px`;
  root.style.top = `${Math.round(top)}px`;
  // Deferred, so the press that opened the menu does not also close it.
  setTimeout(() => {
    if (!live) return;
    document.addEventListener("pointerdown", onOutside, true);
    window.addEventListener("blur", close);
  }, 0);
  root.focus({ preventScroll: true });
  openMenu = { root, close };
  return { close };
}
