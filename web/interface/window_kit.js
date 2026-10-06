/**
 * The controls a floating window is built from, in one flat look: hairline borders, two
 * surface levels and compact type.
 *
 * Colours are the palette's own `--was-*` properties. Sizes are CSS pixels.
 */

import { themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.WindowKit";

/** The type a window is set in. */
export const SANS = "system-ui,sans-serif";
export const MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";

/** The colours a window is painted with, each a palette property. */
export const LOOK = Object.freeze({
  body: themeVar("panelBg"),
  surface: themeVar("inputBg"),
  border: themeVar("border"),
  hover: themeVar("border"),
  text: themeVar("fg"),
  muted: themeVar("fgMuted"),
  accent: themeVar("accent"),
  danger: themeVar("error"),
  success: themeVar("success"),
  warning: themeVar("warning"),
  shadow: themeVar("shadow"),
  onAccent: themeVar("selectionText"),
});

/** A border one step stronger than the hairline, for the control that matters most. */
export const STRONG_BORDER = `color-mix(in srgb, ${LOOK.text} 30%, ${LOOK.border})`;

/** A selected row's fill: the accent, mostly transparent. */
export const SELECTED_FILL = `color-mix(in srgb, ${LOOK.accent} 24%, transparent)`;

/** Thin scroll bars in the muted colour. */
export const SCROLLBARS = `scrollbar-width:thin;scrollbar-color:${LOOK.muted} transparent`;

/** The panel a menu, a popover or a tooltip floats in. */
export const FLOATING_PANEL = `background:${LOOK.surface};border:1px solid ${LOOK.border};`
  + `border-radius:8px;box-shadow:0 8px 24px ${LOOK.shadow}`;

/** A card: a block of related controls on the surface colour. */
export const CARD = `display:flex;flex-direction:column;gap:7px;padding:10px;border-radius:9px;`
  + `border:1px solid ${LOOK.border};background:${LOOK.surface}`;

/** A row in a list. */
export const ROW = `display:flex;align-items:center;gap:9px;padding:7px 8px;border-radius:7px;`
  + `border:1px solid ${LOOK.border};background:${LOOK.surface}`;

/** A text field, a number field or a select. */
export const FIELD = `box-sizing:border-box;min-width:0;padding:6px 9px;border-radius:6px;`
  + `font:13px ${SANS};color:${LOOK.text};background:${LOOK.surface};`
  + `border:1px solid ${LOOK.border};outline:none`;

/** A small uppercase label over a control. */
export const LABEL = `font:600 11px ${SANS};letter-spacing:0.05em;text-transform:uppercase;`
  + `color:${LOOK.muted}`;

// How each kind of button is drawn at rest and under the pointer.
const BUTTONS = {
  tool: {
    rest: `padding:4px 11px;border-radius:4px;font:12px/18px ${SANS};`
      + `background:${LOOK.surface};border:1px solid ${LOOK.border};color:${LOOK.text}`,
    hover: { background: LOOK.hover, borderColor: `color-mix(in srgb, ${LOOK.text} 26%, ${LOOK.border})` },
  },
  go: {
    rest: `padding:4px 11px;border-radius:4px;font:500 12px/18px ${SANS};`
      + `background:${LOOK.surface};border:1px solid ${STRONG_BORDER};color:${LOOK.text}`,
    hover: { background: LOOK.hover, borderColor: STRONG_BORDER },
  },
  danger: {
    rest: `padding:4px 11px;border-radius:4px;font:12px/18px ${SANS};`
      + `background:${LOOK.surface};border:1px solid ${LOOK.border};color:${LOOK.danger}`,
    hover: { background: `color-mix(in srgb, ${LOOK.danger} 14%, transparent)`, borderColor: LOOK.danger },
  },
  plain: {
    rest: `padding:6px 16px;border-radius:6px;font:13px ${SANS};`
      + `background:${LOOK.surface};border:1px solid ${LOOK.border};color:${LOOK.text}`,
    hover: { background: LOOK.hover },
  },
  primary: {
    rest: `padding:6px 16px;border-radius:6px;font:600 13px ${SANS};`
      + `background:${LOOK.success};border:1px solid ${LOOK.success};color:${LOOK.onAccent}`,
    hover: { background: `color-mix(in srgb, ${LOOK.success} 85%, ${LOOK.text})` },
  },
  ghost: {
    rest: `padding:3px 8px;border-radius:5px;font:12px ${SANS};`
      + `background:transparent;border:1px solid transparent;color:${LOOK.muted}`,
    hover: { background: LOOK.hover, color: LOOK.text },
  },
};

/**
 * An element with a style and a text.
 *
 * @param {string} tag - The tag.
 * @param {string} [css] - Its inline style.
 * @param {string} [text] - Its text.
 * @returns {HTMLElement} The element.
 */
export function el(tag, css = "", text = "") {
  const element = document.createElement(tag);
  if (css) element.style.cssText = css;
  if (text) element.textContent = text;
  return element;
}

/**
 * Light a field's border in the accent while it holds focus.
 *
 * @param {HTMLElement} control - The field.
 * @returns {HTMLElement} The same field.
 */
export function focusRing(control) {
  control.addEventListener("focus", () => { control.style.borderColor = LOOK.accent; });
  control.addEventListener("blur", () => { control.style.borderColor = LOOK.border; });
  return control;
}

/**
 * A button.
 *
 * @param {string} label - The text on it.
 * @param {(event: MouseEvent) => void} onPress - What it does.
 * @param {object} [options] - `{hint, kind, logName}`; `kind` is `tool`, `go` or `danger` for a
 *   toolbar, `plain` or `primary` in a window's body, or `ghost` for a quiet one.
 * @returns {HTMLButtonElement} The button.
 */
export function button(label, onPress, { hint = "", kind = "plain", logName = LOG_NAME } = {}) {
  const look = BUTTONS[kind] ?? BUTTONS.plain;
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  if (hint) element.title = hint;
  element.style.cssText = `${look.rest};cursor:pointer;white-space:nowrap;display:inline-flex;`
    + "align-items:center;justify-content:center;gap:6px;flex:0 0 auto";
  const rest = {
    background: element.style.background, borderColor: element.style.borderColor, color: element.style.color,
  };
  element.addEventListener("mouseenter", () => {
    if (!element.disabled) Object.assign(element.style, look.hover);
  });
  element.addEventListener("mouseleave", () => Object.assign(element.style, rest));
  element.addEventListener("click", (event) => {
    event.preventDefault();
    if (element.disabled) return;
    try {
      onPress(event);
    } catch (error) {
      console.error(`[${logName}] ${label} failed:`, error);
    }
  });
  return element;
}

/**
 * Grey a button out, or bring it back.
 *
 * @param {HTMLButtonElement} element - The button.
 * @param {boolean} off - True to disable it.
 * @returns {void}
 */
export function setDisabled(element, off) {
  element.disabled = off;
  element.style.opacity = off ? "0.45" : "1";
  element.style.cursor = off ? "default" : "pointer";
}

/**
 * A select over labelled values.
 *
 * @param {Array<[string, string]>} entries - `[value, label]` pairs.
 * @param {string} value - The chosen value.
 * @returns {HTMLSelectElement} The select.
 */
export function select(entries, value) {
  const element = focusRing(el("select", `${FIELD};width:100%;cursor:pointer`));
  for (const [option, label] of entries) {
    const entry = document.createElement("option");
    entry.value = String(option);
    entry.textContent = label;
    element.appendChild(entry);
  }
  element.value = String(value);
  return element;
}

/**
 * A number field with a unit after it.
 *
 * @param {number|string} value - The value.
 * @param {object} limits - `{min, max, step}`.
 * @param {string} [unit] - The unit drawn after it.
 * @returns {{box: HTMLElement, input: HTMLInputElement}} The row and its field.
 */
export function numberField(value, { min, max, step }, unit = "") {
  const box = el("div", "display:flex;align-items:center;gap:6px;min-width:0");
  const input = focusRing(el("input", `${FIELD};width:96px;font-variant-numeric:tabular-nums`));
  input.type = "number";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  box.appendChild(input);
  if (unit) box.appendChild(el("span", `font:12px ${SANS};color:${LOOK.muted}`, unit));
  return { box, input };
}

/**
 * A text box.
 *
 * @param {number} rows - Lines it opens at.
 * @param {string} [extra] - More style, as a minimum height.
 * @returns {HTMLTextAreaElement} The box.
 */
export function textBox(rows, extra = "") {
  const box = focusRing(el("textarea", `${FIELD};width:100%;resize:vertical;padding:9px 11px;`
    + `font:13px/1.55 ${SANS};${SCROLLBARS};${extra}`));
  box.rows = rows;
  box.spellcheck = false;
  return box;
}

/**
 * A labelled field.
 *
 * @param {string} label - The field's name.
 * @param {HTMLElement} control - The control.
 * @param {string} [meta] - A short fact under it, as `5.88 s · 141 frames`.
 * @param {string} [tip] - What the hover over the field says.
 * @returns {HTMLElement} The field.
 */
export function field(label, control, meta = "", tip = "") {
  const box = el("label", "display:flex;flex-direction:column;gap:6px;min-width:0");
  box.appendChild(el("span", LABEL, label));
  box.appendChild(control);
  if (meta) {
    box.appendChild(el("span", `font:12px/1.4 ${SANS};color:${LOOK.muted};font-variant-numeric:tabular-nums;`
      + "overflow:hidden;text-overflow:ellipsis;white-space:nowrap", meta));
  }
  if (tip) box.title = tip;
  return box;
}

/**
 * A section heading followed by a hairline.
 *
 * @param {string} text - The heading.
 * @param {HTMLElement} [aside] - Something drawn at its right.
 * @returns {HTMLElement} The heading row.
 */
export function heading(text, aside) {
  const row = el("div", "display:flex;align-items:center;gap:10px;margin-top:4px");
  row.appendChild(el("span", `font:700 12px ${SANS};letter-spacing:0.08em;text-transform:uppercase;`
    + `color:${LOOK.muted}`, text));
  row.appendChild(el("span", `flex:1 1 auto;height:1px;background:${LOOK.border}`));
  if (aside) row.appendChild(aside);
  return row;
}

/**
 * A pill holding a short word or a count.
 *
 * @param {string} text - What it says.
 * @param {string} [tone] - A CSS colour for its text and edge, muted by default.
 * @returns {HTMLSpanElement} The pill.
 */
export function pill(text, tone) {
  return el("span", `display:inline-flex;align-items:center;padding:0 8px;border-radius:999px;`
    + `font:12px/18px ${SANS};color:${tone ?? LOOK.muted};border:1px solid ${tone ?? LOOK.border};`
    + "font-variant-numeric:tabular-nums;white-space:nowrap", text);
}

/**
 * A strip of tabs underlined in the accent when chosen.
 *
 * @param {Array<[string, string]>} entries - `[id, label]` pairs.
 * @param {(id: string) => void} onPick - Called with the tab pressed.
 * @returns {{element: HTMLElement, show: (id: string) => void}} The strip, and the call that
 *   marks one tab chosen.
 */
export function tabStrip(entries, onPick) {
  const element = el("div", "display:flex;gap:4px;padding:0 16px;flex:0 0 auto;"
    + `background:${LOOK.surface};border-bottom:1px solid ${LOOK.border}`);
  const tabs = new Map();
  for (const [id, label] of entries) {
    const tab = el("button", `padding:9px 10px 7px;margin-bottom:-1px;border:0;`
      + `border-bottom:2px solid transparent;background:none;cursor:pointer;`
      + `font:13px/1.4 ${SANS};color:${LOOK.muted}`, label);
    tab.type = "button";
    tab.addEventListener("click", (event) => {
      event.preventDefault();
      onPick(id);
    });
    tab.addEventListener("mouseenter", () => { tab.style.color = LOOK.text; });
    tab.addEventListener("mouseleave", () => {
      if (tab.dataset.on !== "1") tab.style.color = LOOK.muted;
    });
    tabs.set(id, tab);
    element.appendChild(tab);
  }
  const show = (chosen) => {
    for (const [id, tab] of tabs) {
      const on = id === chosen;
      tab.dataset.on = on ? "1" : "0";
      tab.style.color = on ? LOOK.text : LOOK.muted;
      tab.style.borderBottomColor = on ? LOOK.accent : "transparent";
      tab.style.fontWeight = on ? "600" : "400";
    }
  };
  return { element, show };
}

/**
 * A notice that slides in at the bottom right of a host and leaves on its own.
 *
 * @param {HTMLElement} host - A positioned element to show it in.
 * @param {number} [bottom] - Pixels it sits above the host's bottom edge.
 * @returns {{say: (text: string, tone?: string) => void, dispose: () => void}} The notice.
 */
export function createNotice(host, bottom = 16) {
  const element = el("div", `position:absolute;right:16px;bottom:${bottom}px;z-index:5;display:none;`
    + `max-width:420px;padding:10px 14px;border-radius:8px;font:13px/1.4 ${SANS};color:${LOOK.text};`
    + `background:${LOOK.surface};border:1px solid ${LOOK.border};border-left:3px solid ${LOOK.accent};`
    + `box-shadow:0 6px 20px ${LOOK.shadow};pointer-events:none`);
  host.appendChild(element);
  let timer = 0;
  return {
    say(text, tone = "accent") {
      element.textContent = String(text ?? "");
      element.style.borderLeftColor = { warning: LOOK.warning, danger: LOOK.danger, success: LOOK.success }[tone]
        ?? LOOK.accent;
      element.style.display = text ? "block" : "none";
      clearTimeout(timer);
      timer = setTimeout(() => { element.style.display = "none"; }, 3600);
    },
    dispose() {
      clearTimeout(timer);
      element.remove();
    },
  };
}
