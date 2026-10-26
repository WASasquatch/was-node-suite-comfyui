/**
 * ComfyUI's own colours, as every interface in the pack reads them.
 *
 * `readTheme` answers the token object. The same tokens are published on the page root as
 * `--was-*` custom properties, named by `themeVar`. `onThemeChange` reports a palette change.
 */

import { getFullTheme } from "../viewer/utils/theme.js";

const LOG_NAME = "WASNodeSuite.Theme";

// How long an answer is held for, in milliseconds. Half a second is what keeps a drag from
// recomputing styles on every frame.
const CACHE_MS = 500;

// The stem every published custom property carries.
const VAR_PREFIX = "--was-";

// Attributes ComfyUI moves when it changes palette.
const WATCHED_ATTRIBUTES = ["class", "style", "data-theme", "data-color-scheme"];

// How often the palette is re-read regardless of what the page did, in milliseconds.
const POLL_MS = 1500;

// Every token `getFullTheme` answers, so each one resolves to a colour even on the read that
// fails. The values are the same literals that function falls back to when a custom property is
// missing, so a partial read and a failed read agree.
const FALLBACK_THEME = {
  bg: "#1a1a1a",
  bgLight: "#2a2a2a",
  bgDark: "#151515",
  fg: "#e0e0e0",
  fgMuted: "#888888",
  fgDisabled: "#666666",
  border: "#444444",
  borderLight: "#3a3a3a",
  accent: "#4a9eff",
  accentHover: "#5ab0ff",
  accentBg: "#4a9eff22",
  success: "#4caf50",
  warning: "#ff9800",
  error: "#f44336",
  inputBg: "#2a2a2a",
  inputBorder: "#555555",
  inputFocus: "#4a9eff",
  panelBg: "#252525",
  panelHeader: "#333333",
  shadow: "rgba(0,0,0,0.3)",
  scrollbarThumb: "#555555",
  scrollbarTrack: "#1a1a1a",
  selection: "#4a9eff44",
  selectionText: "#ffffff",
  cardAccent: "#4a9eff",
};

// How saturated a colour has to be, from 0 to 1, to tint a card.
const CARD_CHROMA = 0.12;

// A one-pixel surface that turns any CSS colour into its channels.
let colourProbe = null;

/** Every token name a palette carries, in the order it is published. */
export const THEME_TOKENS = Object.freeze(Object.keys(FALLBACK_THEME));

// One cache serves the page, since the tokens are the page's rather than a node's.
let themeCache = null;
let themeCacheTime = 0;

// One watch serves the page: one observer, one media query, one timer, one coalesced pass.
let watching = false;
let settlePending = false;
let signature = "";

// Callers told about a change, and the elements the palette is written onto besides the page
// root. An element is held weakly, so a node taken off the graph is collected.
const listeners = new Set();
const scopes = new Set();
const scoped = new WeakSet();

/**
 * Read ComfyUI's palette, cached briefly so a gesture does not recompute styles per frame.
 *
 * @returns {object} Theme tokens, falling back to a dark palette if they cannot be read.
 */
export function readTheme() {
  ensureThemeWatch();
  const now = Date.now();
  if (themeCache && now - themeCacheTime < CACHE_MS) return themeCache;
  try {
    themeCache = { ...FALLBACK_THEME, ...getFullTheme() };
    themeCache.cardAccent = cardAccent(themeCache);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read the theme:`, error);
    themeCache = themeCache ?? FALLBACK_THEME;
  }
  themeCacheTime = now;
  return themeCache;
}

/**
 * The colour a card is tinted with: the palette's node title colour, else its primary colour.
 *
 * @param {object} theme - The tokens read so far.
 * @returns {string} A CSS colour.
 */
function cardAccent(theme) {
  const root = document.documentElement ? getComputedStyle(document.documentElement) : null;
  const candidates = [
    globalThis.LiteGraph?.NODE_DEFAULT_COLOR,
    root?.getPropertyValue("--p-primary-color")?.trim(),
    theme.accent,
  ];
  for (const candidate of candidates) {
    if (candidate && chroma(candidate) >= CARD_CHROMA) return candidate;
  }
  return theme.accent || FALLBACK_THEME.cardAccent;
}

/**
 * How far a colour is from grey, from 0 to 1.
 *
 * @param {string} colour - Any CSS colour.
 * @returns {number} The spread between its strongest and weakest channel.
 */
function chroma(colour) {
  try {
    colourProbe ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    colourProbe.clearRect(0, 0, 1, 1);
    colourProbe.fillStyle = "#000000";
    colourProbe.fillStyle = colour;
    colourProbe.fillRect(0, 0, 1, 1);
    const [red, green, blue] = colourProbe.getImageData(0, 0, 1, 1).data;
    return (Math.max(red, green, blue) - Math.min(red, green, blue)) / 255;
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to read a colour:`, error);
    return 0;
  }
}

/**
 * The custom property one token is published under.
 *
 * @param {string} token - A key of the object `readTheme` answers, such as `fgMuted`.
 * @returns {string} The property name, such as `--was-fg-muted`.
 */
export function themeVarName(token) {
  return VAR_PREFIX + String(token).replace(/([A-Z])/g, "-$1").toLowerCase();
}

/**
 * One token as a CSS value, for a style string or a `setProperty` call.
 *
 * @param {string} token - A key of the object `readTheme` answers, such as `fgMuted`.
 * @returns {string} A `var()` reference, such as `var(--was-fg-muted)`.
 */
export function themeVar(token) {
  ensureThemeWatch();
  return `var(${themeVarName(token)})`;
}

/**
 * Write the palette onto one element as `--was-*` custom properties.
 *
 * Rewritten on every palette change until `clearThemeVars` takes the element back.
 *
 * @param {Element} element - What to write the properties on. The page root carries them
 *   already, so this is for a scope it does not reach, such as the document inside a frame.
 * @param {object} [theme] - Tokens to write. `readTheme`'s answer by default.
 * @returns {void}
 */
export function applyThemeVars(element, theme) {
  if (!element || !element.style) return;
  ensureThemeWatch();
  writeThemeVars(element, theme || readTheme());
  if (scoped.has(element)) return;
  scoped.add(element);
  scopes.add(new WeakRef(element));
}

/**
 * Take the `--was-*` properties back off one element and stop rewriting it.
 *
 * @param {Element} element - What `applyThemeVars` was called with.
 * @returns {void}
 */
export function clearThemeVars(element) {
  if (!element || !element.style) return;
  for (const token of THEME_TOKENS) element.style.removeProperty(themeVarName(token));
  scoped.delete(element);
  for (const ref of scopes) {
    if (ref.deref() === element) scopes.delete(ref);
  }
}

/**
 * Call a function once on every palette change, for drawing a custom property cannot reach.
 *
 * @param {Function} listener - Called with the new tokens, in `readTheme`'s shape.
 * @returns {Function} Releases the listener. Calling it more than once does nothing further.
 */
export function onThemeChange(listener) {
  if (typeof listener !== "function") return () => {};
  ensureThemeWatch();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Set every token as a custom property on one element.
 *
 * @param {Element} element - What to write on.
 * @param {object} theme - Tokens to write.
 * @returns {void}
 */
function writeThemeVars(element, theme) {
  for (const token of THEME_TOKENS) {
    const value = theme[token];
    if (value) element.style.setProperty(themeVarName(token), value);
  }
}

/**
 * Write the palette onto the page root and onto every element `applyThemeVars` was given.
 *
 * @param {object} theme - Tokens to publish.
 * @returns {void}
 */
function publishTheme(theme) {
  const root = document.documentElement;
  if (root) writeThemeVars(root, theme);
  for (const ref of scopes) {
    const element = ref.deref();
    if (element) writeThemeVars(element, theme);
    else scopes.delete(ref);
  }
}

/**
 * Re-read the palette and, where it moved, publish it and tell every listener.
 *
 * @returns {void}
 */
function settleTheme() {
  settlePending = false;
  // The cached answer is dropped before the palette is read again.
  themeCache = null;
  const theme = readTheme();
  const next = JSON.stringify(theme);
  if (next === signature) return;
  signature = next;
  publishTheme(theme);
  for (const listener of Array.from(listeners)) {
    try {
      listener(theme);
    } catch (error) {
      console.error(`[${LOG_NAME}] A theme listener failed:`, error);
    }
  }
}

/**
 * Ask for one settle pass, however many signals arrived before it runs.
 *
 * @returns {void}
 */
function scheduleSettle() {
  if (settlePending) return;
  settlePending = true;
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(settleTheme);
  else setTimeout(settleTheme, 0);
}

/**
 * Start the page's one palette watch, and publish the palette for the first time.
 *
 * @returns {void}
 */
function ensureThemeWatch() {
  if (watching) return;
  if (typeof document === "undefined" || !document.documentElement) return;
  watching = true;
  try {
    const theme = readTheme();
    signature = JSON.stringify(theme);
    publishTheme(theme);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to publish the palette:`, error);
  }
  try {
    const observer = new MutationObserver(scheduleSettle);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: WATCHED_ATTRIBUTES,
    });
    if (document.body) {
      observer.observe(document.body, { attributes: true, attributeFilter: WATCHED_ATTRIBUTES });
    }
    if (typeof matchMedia === "function") {
      matchMedia("(prefers-color-scheme: dark)")?.addEventListener?.("change", scheduleSettle);
    }
    // Read again on a timer and on a return to the tab, for a palette that arrives as a
    // stylesheet swap or while the tab is in the background.
    setInterval(scheduleSettle, POLL_MS);
    window.addEventListener("focus", scheduleSettle);
    document.addEventListener("visibilitychange", scheduleSettle);
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to watch for a palette change:`, error);
  }
}
