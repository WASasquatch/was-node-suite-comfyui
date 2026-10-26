/**
 * The card every interface is drawn on, and the fade over content a small node cuts off.
 *
 * `CARD_LOOK` colours are mixed from the palette's `--was-*` properties. `dressPanel` dresses a
 * root; `cardRule` and `cardChip` draw a title's rule and pills.
 */

import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.Card";

const ACCENT = themeVar("cardAccent");
const PANEL_BG = themeVar("panelBg");
const DEEP_BG = themeVar("bgDark");
const PAGE_BG = themeVar("bg");
const FG = themeVar("fg");

/** The card's colours, each a mix of palette properties. */
export const CARD_LOOK = Object.freeze({
  surface: `radial-gradient(130% 110% at 0% 0%, color-mix(in srgb, ${ACCENT} 16%, transparent), `
    + `transparent 55%), linear-gradient(160deg, color-mix(in srgb, ${ACCENT} 10%, `
    + `color-mix(in srgb, ${PANEL_BG} 70%, ${PAGE_BG})) 0%, color-mix(in srgb, ${PANEL_BG} 55%, ${PAGE_BG}) 45%, `
    + `color-mix(in srgb, ${PAGE_BG} 75%, ${DEEP_BG}) 100%)`,
  edge: `color-mix(in srgb, ${ACCENT} 28%, ${themeVar("border")})`,
  rule: `color-mix(in srgb, ${ACCENT} 60%, transparent)`,
  well: `inset 0 3px 9px -2px ${themeVar("shadow")}, inset 0 2px 3px ${themeVar("shadow")}, `
    + `inset 1px 0 2px -1px ${themeVar("shadow")}, inset 0 -1px 0 color-mix(in srgb, ${FG} 7%, transparent)`,
  shine: `inset 0 1px 0 color-mix(in srgb, ${FG} 6%, transparent), 0 1px 3px ${themeVar("shadow")}`,
  title: `color-mix(in srgb, ${ACCENT} 32%, ${FG})`,
  text: FG,
  muted: themeVar("fgMuted"),
  inset: `color-mix(in srgb, ${DEEP_BG} 45%, transparent)`,
});

/** The pills a header carries, as background and text. */
export const CHIP_TONES = Object.freeze({
  accent: { background: `color-mix(in srgb, ${ACCENT} 18%, transparent)`, color: CARD_LOOK.title },
  plain: { background: `color-mix(in srgb, ${FG} 8%, transparent)`, color: CARD_LOOK.muted },
  success: { background: `color-mix(in srgb, ${themeVar("success")} 18%, transparent)`, color: themeVar("success") },
  warning: { background: `color-mix(in srgb, ${themeVar("warning")} 18%, transparent)`, color: themeVar("warning") },
  error: { background: `color-mix(in srgb, ${themeVar("error")} 18%, transparent)`, color: themeVar("error") },
});

// The corner radius of a card, in CSS pixels.
const RADIUS = 8;

// How far the fade over cut content reaches, in CSS pixels.
const FADE = 14;

// Overflow values that mean the root scrolls its own content.
const SCROLLS = ["auto", "scroll", "overlay"];

// One observer serves every card on the page.
let observer = null;

// Card roots the observer watches, each with the children it watches beside it.
const watched = new WeakMap();

/**
 * Put the card on a panel's root: its surface, edge, corners and shine, and room to shrink.
 *
 * @param {HTMLElement} element - The root an interface factory answered.
 * @returns {void}
 */
export function dressPanel(element) {
  if (!element?.style) return;
  try {
    Object.assign(element.style, {
      boxSizing: "border-box",
      background: CARD_LOOK.surface,
      border: `1px solid ${CARD_LOOK.edge}`,
      borderRadius: `${RADIUS}px`,
      boxShadow: CARD_LOOK.well,
      minWidth: "0",
    });
    if (!element.style.color) element.style.color = CARD_LOOK.text;
    const overflow = element.style.overflowY || element.style.overflow;
    if (!SCROLLS.includes(overflow)) element.style.overflow = "hidden";
    watchCut(element);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to dress a panel:`, error);
  }
}

/**
 * The line under a card's title: the accent, fading out to the right.
 *
 * @returns {HTMLElement} The rule, for the caller to place.
 */
export function cardRule() {
  const rule = document.createElement("div");
  rule.style.cssText = `flex:none;height:1px;margin:1px 0;background:linear-gradient(90deg, ${CARD_LOOK.rule}, `
    + "transparent 92%)";
  return rule;
}

/**
 * A small uppercase pill, for a kind, a status or a family named beside a title.
 *
 * @param {string} text - The words in the pill.
 * @param {string} [tone] - A key of `CHIP_TONES`. `plain` when left out or unknown.
 * @returns {HTMLElement} The pill.
 */
export function cardChip(text, tone = "plain") {
  const pill = document.createElement("span");
  const look = CHIP_TONES[tone] ?? CHIP_TONES.plain;
  pill.style.cssText = "flex:none;font:600 9.5px/1.45 system-ui,sans-serif;letter-spacing:0.04em;"
    + `text-transform:uppercase;padding:1px 6px;border-radius:999px;background:${look.background};`
    + `color:${look.color};white-space:nowrap`;
  pill.textContent = String(text ?? "");
  return pill;
}

/**
 * Start marking a root whose content its box cuts off, and keep doing so as it changes size.
 *
 * @param {HTMLElement} element - A card root.
 * @returns {void}
 */
export function watchCut(element) {
  if (!element || typeof ResizeObserver !== "function") return;
  observer ??= new ResizeObserver((entries) => {
    const roots = new Set();
    for (const { target } of entries) {
      const root = watched.has(target) ? target : target.parentElement;
      if (root && watched.has(root)) roots.add(root);
    }
    for (const root of roots) markCut(root);
  });
  if (watched.has(element)) return;
  const children = new Set();
  watched.set(element, children);
  observer.observe(element);
  observeChildren(element);
  element.addEventListener("scroll", () => markCut(element), { passive: true });
}

/**
 * Stop watching a root, for a node taken off the graph.
 *
 * @param {HTMLElement} element - A card root `watchCut` was given.
 * @returns {void}
 */
export function unwatchCut(element) {
  const children = watched.get(element);
  if (!children || !observer) return;
  observer.unobserve(element);
  for (const child of children) observer.unobserve(child);
  watched.delete(element);
}

/**
 * Watch a root's direct children too, so content that grows inside an unchanged box is seen.
 *
 * @param {HTMLElement} root - A watched card root.
 * @returns {void}
 */
function observeChildren(root) {
  const children = watched.get(root);
  if (!children) return;
  for (const child of children) {
    if (child.parentElement === root) continue;
    observer.unobserve(child);
    children.delete(child);
  }
  for (const child of root.children) {
    if (children.has(child)) continue;
    children.add(child);
    observer.observe(child);
  }
}

/**
 * Fade the edge a root's content runs past, and clear the fade once it fits or is scrolled to.
 *
 * @param {HTMLElement} root - A watched card root.
 * @returns {void}
 */
function markCut(root) {
  try {
    observeChildren(root);
    const scrolls = SCROLLS.includes(getComputedStyle(root).overflowY);
    const below = root.scrollHeight - root.clientHeight - (scrolls ? root.scrollTop : 0);
    const right = root.scrollWidth - root.clientWidth;
    const masks = [];
    if (below > 1) masks.push(`linear-gradient(to bottom, ${FG} calc(100% - ${FADE}px), transparent)`);
    if (right > 1) masks.push(`linear-gradient(to right, ${FG} calc(100% - ${FADE}px), transparent)`);
    const mask = masks.length ? masks.join(", ") : "none";
    if (root.style.maskImage === mask) return;
    root.style.maskImage = mask;
    root.style.webkitMaskImage = mask;
    root.style.maskComposite = masks.length > 1 ? "intersect" : "";
    root.style.webkitMaskComposite = masks.length > 1 ? "source-in" : "";
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to mark a cut panel:`, error);
  }
}
