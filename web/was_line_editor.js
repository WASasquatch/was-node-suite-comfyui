/**
 * Line editor for String List, String List (Sections) and Multiline Text (Advanced).
 *
 * Each list is a box of coloured lines with switches, versions and hover tools; Sections stacks
 * several. Versions and lists that take turns move on after each run.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK, cardChip, cardRule } from "./interface/card.js";
import { ICON, ICON_SIZE, drawFidelityGlyph } from "./interface/icons.js";
import {
  addLine, addVersion, advance, advanceLists, groupAt, holdsChoice, leadOf, listsTakeTurns, pickVersion,
  setMode, switchLine, takesTurns,
} from "./interface/line_edits.js";
import { captureWheel, wheelPixels } from "./interface/pointer.js";
import { openPopupMenu } from "./interface/popup_menu.js";
import {
  ALL, BLANK, CYCLE, DECREMENT, FIXED, INCREMENT, MAX_COMBINED, MODE, MODES, NOTE, OFF_MARK, PROMPT,
  RANDOM, REFERENCE, SECTION, TERM, VERSION, classify, combine, glossary, groupLines, handedOn,
  hiddenLines, joinLines, listEntries, listLines, listSeed, parseChoices, readLists, resolve,
} from "./interface/prompt_lines.js";
import { withGraphChange } from "./interface/region.js";
import { watchSurfaceRatio } from "./interface/resolution.js";
import { ROW_PALETTE } from "./interface/row_headers.js";
import { themeVar } from "./interface/theme.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget, chainWidgetCallback } from "./interface/widget.js";
import { button } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.LineEditorUI";
const SETTING_ID = "WAS.LineEditor.ShowInterface";
const CONTROL_MODE_SETTING = "Comfy.WidgetControlMode";

const WIRE_INPUT = "text";
const BLANK_WIDGET = "skip_blank_lines";
const COMMENT_WIDGET = "skip_comment_lines";
const JOIN_WIDGET = "join_with";
const COMBINE_WIDGET = "combine";
const SEED_WIDGET = "seed";

// The nodes this draws on: the widget holding the text, what a kept line is called, the schema's
// blank-line default, whether kept lines are numbered, what the node hands on, and what the panel
// says over no lines.
const NODES = {
  WASStringList: {
    text: "lines", one: "entry", many: "entries", skipBlank: true, numbered: true, output: "list",
    hint: "One prompt per line · ~ switches a line off · # starts a note · ? lists the rest",
    all: "Every version, each its own entry",
  },
  WASMultilineTextAdvanced: {
    text: "lines", one: "line kept", many: "lines kept", skipBlank: false, numbered: false, output: "joined",
    hint: "One part per line · ~ switches a line off · # starts a note · ? lists the rest",
    all: "Every version, each its own line",
  },
  WASStringListSections: {
    text: "lists", one: "entry", many: "entries", skipBlank: true, numbered: true, output: "lists",
    hint: "Type or paste one line per item",
    all: "Every version, each its own line",
  },
};

const UI_WIDGET_NAME = "was_line_editor_ui";
const UI_WIDGET_TYPE = "was_line_editor";

// Characters of the joined text shown on hover.
const PREVIEW_CHARS = 2000;

// The height the panel opens at and the width its node opens at, in node units.
const PANEL_HEIGHT = 260;
const PANEL_WIDTH = 380;

// Layout, in CSS pixels.
const GUTTER = 62;
const PAD_X = 8;
const PAD_Y = 6;
const LINE_HEIGHT = 18;
const STRIPE = 3;
const BOX = 13;
const GAP = 3;

// How long the copy button shows its tick.
const COPIED_MS = 900;

// How many seeds a new draw picks from.
const SEED_SPAN = 4294967296;

// The kind of the empty line after a final line break, which is drawn and never handed on.
const END = "end";

// Keeps an empty row one line tall.
const ZERO_WIDTH = "​";

const FG = themeVar("fg");
const MUTED = themeVar("fgMuted");
const DISABLED = themeVar("fgDisabled");
const ACCENT = themeVar("accent");
const DEEP = themeVar("bgDark");
const WARNING = themeVar("warning");

// The type the text box and the lines under it both take. Kerning and ligatures are off so a
// coloured run sets exactly as the same letters do in the box.
const TEXT_LOOK = "font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;font-size:12px;"
  + `font-weight:400;font-style:normal;line-height:${LINE_HEIGHT}px;letter-spacing:normal;`
  + "word-spacing:normal;text-indent:0;text-transform:none;tab-size:4;white-space:pre-wrap;"
  + "overflow-wrap:break-word;word-break:normal;font-kerning:none;font-variant-ligatures:none;";

// The colours a prompt line cycles through, from the row header palette.
const PROMPT_HUES = ["green", "yellow", "orange", "pink", "purple", "red"]
  .map((name) => ROW_PALETTE[name]?.stripe)
  .filter(Boolean);

// Terms and their names, tokens, and the marks that set how versions take turns.
const TERM_HUE = ROW_PALETTE.teal?.stripe ?? ACCENT;
const TOKEN_HUE = ROW_PALETTE.blue?.stripe ?? ACCENT;
const MODE_HUE = `color-mix(in srgb, ${ACCENT} 70%, ${FG})`;

// A term's `$name =`, a `[token]`, and a character kept as text by a backslash.
const TERM_NAME = /^\$[A-Za-z_][A-Za-z0-9_]*[ \t]*=/;
const TOKEN = /(?<!\\)\[[^\]\n]*\]/g;
const ESCAPE = /\\[{}|$]/g;

// What the modes are called in the menus, and what each does, for a line's versions and for a
// whole list.
const MODE_NAMES = { fixed: "Fixed", increment: "Increment", decrement: "Decrement", random: "Random", all: "All" };
const MODE_HINTS = {
  fixed: "The version marked > every run",
  increment: "The next version after each run",
  decrement: "The one before after each run",
  random: "Another at random after each run",
};
const LIST_MODES = [ALL, FIXED, INCREMENT, DECREMENT, RANDOM];
const LIST_NAMES = { ...MODE_NAMES, all: "Whole list" };
const LIST_HINTS = {
  all: "Every line, each in turn",
  fixed: "One line, the one marked ●",
  increment: "One line, the next after each run",
  decrement: "One line, the one before each run",
  random: "One line, another at random each run",
};

// The syntax menu: what is typed, and what it does.
const SYNTAX = [
  ["~ line", "Switched off"],
  ["# note   // note", "A note, left out"],
  ["/* … */", "Notes over several lines"],
  ["== Name ==", "A heading; its switch turns the block off"],
  ["| version", "Another version of the line above"],
  ["> version", "The version in use"],
  ["@increment line", "Versions take turns: also @decrement, @random, @all"],
  ["$name = value", "A term"],
  ["$name   ${name}", "Replaced by the term"],
  ["{a|b|c}", "One option, drawn from the seed"],
  ["{a|}", "Sometimes left out"],
  ["{2$$a|b|c}", "Two options; {1-3$$…} one to three"],
  ["{3::a|b}", "a three times as likely"],
  ["\\{  \\|  \\$", "Kept as text"],
  ["[time]", "A token, filled in when it runs"],
];

// Icons, as 16 by 16 paths.
const SVG_NS = "http://www.w3.org/2000/svg";
const COPY_PATHS = ["M5.5 5.5h7a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1z", "M2.5 10.5v-7a1 1 0 0 1 1-1h7"];
const PLUS_PATHS = ["M8 3.5v9", "M3.5 8h9"];
const DONE_PATHS = ["M3.5 8.5l3 3 6-7"];
const REMOVE_PATHS = ["M4.5 4.5l7 7", "M11.5 4.5l-7 7"];
const UP_PATHS = ["M8 12.5v-9", "M4.5 7l3.5-3.5 3.5 3.5"];
const DOWN_PATHS = ["M8 3.5v9", "M4.5 9l3.5 3.5 3.5-3.5"];

/**
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  return (node?.widgets ?? []).find((widget) => widget?.name === name) ?? null;
}

/**
 * Whether one of a node's inputs has a link into it.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  const input = (node?.inputs ?? []).find((slot) => slot?.name === name);
  return input?.link !== null && input?.link !== undefined;
}

/**
 * Read a boolean widget.
 *
 * @param {object} node - Node holding it.
 * @param {string} name - Widget name.
 * @param {boolean} fallback - The schema's default, or what a node without it means.
 * @returns {boolean} The widget's value, or the default when it holds no boolean.
 */
function readSwitch(node, name, fallback) {
  const value = findWidget(node, name)?.value;
  return typeof value === "boolean" ? value : fallback;
}

/**
 * Read a widget holding a name.
 *
 * @param {object} node - Node holding it.
 * @param {string} name - Widget name.
 * @param {string} fallback - What a node without it means.
 * @returns {string} The value.
 */
function readName(node, name, fallback) {
  const value = findWidget(node, name)?.value;
  return typeof value === "string" ? value : fallback;
}

/**
 * Read the seed the choices draw from.
 *
 * @param {object} node - Node holding it.
 * @returns {number} The seed, 0 when there is none.
 */
function readSeed(node) {
  const value = Number(findWidget(node, SEED_WIDGET)?.value);
  return Number.isFinite(value) ? value : 0;
}

/**
 * Text with every line break written as `\n`, as a text box holds it.
 *
 * @param {*} value - A widget value.
 * @returns {string} The text.
 */
function normalise(value) {
  return String(value ?? "").replace(/\r\n?/g, "\n");
}

/**
 * Read a setting.
 *
 * @param {string} id - Its id.
 * @returns {*} Its value, or undefined when it cannot be read.
 */
function readSetting(id) {
  try {
    return app?.extensionManager?.setting?.get?.(id) ?? app?.ui?.settings?.getSettingValue?.(id);
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${id}:`, error);
    return undefined;
  }
}

/**
 * Read whether the editor is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  const value = readSetting(SETTING_ID);
  return typeof value === "boolean" ? value : true;
}

/**
 * Whether widgets move on before a prompt is queued rather than after.
 *
 * @returns {boolean} True for ComfyUI's `before` control mode.
 */
function runsBefore() {
  return readSetting(CONTROL_MODE_SETTING) === "before";
}

/**
 * Fold the text box away, the editor standing in for it.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget holding the text.
 * @returns {void}
 */
function foldTextBox(node, name) {
  if (setWidgetHidden(findWidget(node, name), true)) node.setDirtyCanvas?.(true, true);
}

/**
 * A colour at a share of full strength over the card.
 *
 * @param {string} colour - Any CSS colour.
 * @param {number} percent - How much of it shows.
 * @returns {string} A `color-mix` against transparent.
 */
function tint(colour, percent) {
  return `color-mix(in srgb, ${colour} ${percent}%, transparent)`;
}

// How a row looks, by what it is.
const lit = (colour) => ({ fill: tint(colour, 13), stripe: colour, text: FG, strike: false });
const dim = (colour) => ({ fill: tint(colour, 5), stripe: tint(colour, 40), text: MUTED, strike: false });
const OFF_LOOK = { fill: tint(FG, 3), stripe: DISABLED, text: DISABLED, strike: true };
const HIDDEN_LOOK = { fill: "transparent", stripe: tint(FG, 10), text: DISABLED, strike: false };
const NOTE_LOOK = {
  fill: tint(ACCENT, 15), stripe: ACCENT, text: `color-mix(in srgb, ${ACCENT} 55%, ${FG})`, strike: false,
};
const SECTION_LOOK = { fill: tint(FG, 10), stripe: CARD_LOOK.title, text: CARD_LOOK.title, strike: false };
const KEPT_BLANK_LOOK = { fill: tint(FG, 4), stripe: tint(FG, 18), text: FG, strike: false };
const PLAIN_LOOK = { fill: "transparent", stripe: "transparent", text: FG, strike: false };

/**
 * A 12 pixel icon from 16 by 16 paths, stroked in the text colour.
 *
 * @param {string[]} paths - Path data.
 * @returns {SVGSVGElement} The icon.
 */
function icon(paths) {
  const svg = document.createElementNS(SVG_NS, "svg");
  for (const [name, value] of Object.entries({
    viewBox: "0 0 16 16", width: "12", height: "12", fill: "none", stroke: "currentColor",
    "stroke-width": "1.6", "stroke-linecap": "round", "stroke-linejoin": "round",
  })) svg.setAttribute(name, value);
  svg.style.display = "block";
  for (const data of paths) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", data);
    svg.append(path);
  }
  return svg;
}

/**
 * Put text on the clipboard.
 *
 * @param {string} text - The text.
 * @returns {Promise<boolean>} True when it was copied.
 */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Pages served over plain http have no clipboard API.
  }
  const scratch = document.createElement("textarea");
  scratch.value = text;
  scratch.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
  document.body.append(scratch);
  scratch.select();
  let done = false;
  try {
    done = document.execCommand("copy");
  } catch {
    done = false;
  }
  scratch.remove();
  return done;
}

/**
 * Colour the marks of every `{a|b}` choice among some parts.
 *
 * @param {Array<string|object>} parts - From `parseChoices`.
 * @param {number} at - Where the parsed text starts in the line.
 * @param {(start: number, end: number, colour: string) => void} paint - Colours a span.
 * @returns {void}
 */
function markChoices(parts, at, paint) {
  for (const part of parts) {
    if (typeof part === "string") continue;
    if (!part.literal) {
      paint(at + part.start, at + part.start + 1, ACCENT);
      paint(at + part.end, at + part.end + 1, ACCENT);
      for (const bar of part.bars) paint(at + bar, at + bar + 1, ACCENT);
      if (part.head) paint(at + part.head[0], at + part.head[1], ACCENT);
      for (const option of part.options) paint(at + option.start, at + option.weightEnd, ACCENT);
    }
    for (const option of part.options) markChoices(option.parts, at, paint);
  }
}

/**
 * Cut a line into runs of one colour each: marks, choices, terms, tokens and escapes.
 *
 * @param {string} raw - The line as written.
 * @param {object|undefined} line - What it is, from `classify`.
 * @param {Map<string, string>} terms - The terms defined.
 * @param {string} hue - The colour of its group.
 * @param {Set<string>} unknown - Collects every name used with no term.
 * @returns {Array<[string, string|null]>} The runs, each text and a colour or null for the
 *   row's own.
 */
function highlight(raw, line, terms, hue, unknown) {
  if (!line || line.kind === NOTE || line.kind === BLANK) return [[raw, null]];
  const colours = new Array(raw.length).fill(null);
  const paint = (start, end, colour) => {
    for (let index = Math.max(0, start); index < Math.min(end, raw.length); index += 1) colours[index] = colour;
  };
  let at = leadOf(raw);
  if (line.kind === SECTION) {
    paint(at, raw.length, CARD_LOOK.title);
    for (let index = at; index < raw.length; index += 1) {
      if (raw[index] === "=" || raw[index] === OFF_MARK) paint(index, index + 1, MUTED);
    }
  } else {
    if (line.kind === VERSION) {
      paint(at, at + 1, hue);
      at += 1;
      if (raw.startsWith(OFF_MARK, at)) {
        paint(at, at + 1, MUTED);
        at += 1;
      }
    } else {
      if (raw.startsWith(OFF_MARK, at)) {
        paint(at, at + 1, MUTED);
        at += 1;
        at += leadOf(raw.slice(at));
      }
      const mode = MODE.exec(raw.slice(at));
      if (mode) {
        paint(at, at + mode[0].trimEnd().length, MODE_HUE);
        at += mode[0].length;
      }
      if (line.kind === TERM) {
        const name = TERM_NAME.exec(raw.slice(at));
        if (name) {
          paint(at, at + name[0].length, TERM_HUE);
          at += name[0].length;
        }
      }
    }
    const body = raw.slice(at);
    markChoices(parseChoices(body), at, paint);
    for (const found of body.matchAll(REFERENCE)) {
      const name = found[1] || found[2];
      const known = terms.has(name);
      if (!known) unknown.add(name);
      paint(at + found.index, at + found.index + found[0].length, known ? TERM_HUE : WARNING);
    }
    for (const found of body.matchAll(TOKEN)) paint(at + found.index, at + found.index + found[0].length, TOKEN_HUE);
    for (const found of body.matchAll(ESCAPE)) paint(at + found.index, at + found.index + 1, MUTED);
  }
  const runs = [];
  let start = 0;
  for (let index = 1; index <= raw.length; index += 1) {
    if (index === raw.length || colours[index] !== colours[start]) {
      runs.push([raw.slice(start, index), colours[start]]);
      start = index;
    }
  }
  return runs;
}

/**
 * Build one line's row: its switch, its glyph and its number, and its text.
 *
 * @returns {object} The row and its parts.
 */
function createRow() {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:stretch;min-width:0";
  const gutter = document.createElement("div");
  gutter.style.cssText = `flex:0 0 ${GUTTER}px;display:flex;align-items:flex-start;gap:${GAP}px;`
    + `box-sizing:border-box;padding:${(LINE_HEIGHT - BOX) / 2}px 0 0 ${STRIPE + 5}px`;
  const box = document.createElement("input");
  box.type = "checkbox";
  box.tabIndex = -1;
  box.style.cssText = `flex:none;margin:0;width:${BOX}px;height:${BOX}px;cursor:pointer;`
    + `accent-color:${ACCENT}`;
  const glyph = document.createElement("span");
  glyph.style.cssText = `flex:none;width:${BOX}px;height:${BOX}px;text-align:center;user-select:none;`
    + `font:600 11px/${BOX}px system-ui,sans-serif`;
  const number = document.createElement("span");
  number.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-align:right;"
    + `padding-right:4px;font:500 10px/${BOX}px system-ui,sans-serif;`
    + `font-variant-numeric:tabular-nums;color:${MUTED}`;
  gutter.append(box, glyph, number);
  const body = document.createElement("div");
  body.style.cssText = `flex:1 1 auto;min-width:0;box-sizing:border-box;padding:0 ${PAD_X}px;${TEXT_LOOK}`;
  row.append(gutter, body);
  return { row, box, glyph, number, body, key: "" };
}

/**
 * Draw one line's row from what it is.
 *
 * @param {object} row - The row's parts.
 * @param {object} info - Its look, runs, switch, glyph and number.
 * @returns {void}
 */
function paintRow(row, info) {
  const key = JSON.stringify(info);
  if (row.key === key) return;
  row.key = key;
  const ring = info.current ? `, inset 0 0 0 1px ${tint(ACCENT, 75)}` : "";
  const lift = info.current ? `linear-gradient(${tint(FG, 6)}, ${tint(FG, 6)}), ` : "";
  const rule = info.rule ? `, inset 0 1px 0 ${tint(FG, 22)}` : "";
  row.row.style.background = `${lift}${info.look.fill}`;
  row.row.style.boxShadow = `inset ${STRIPE}px 0 0 ${info.look.stripe}, `
    + `inset 0 -1px 0 ${tint(DEEP, 55)}${rule}${ring}`;
  row.body.replaceChildren();
  if (!info.runs.some(([text]) => text)) {
    row.body.textContent = ZERO_WIDTH;
  } else {
    for (const [text, colour] of info.runs) {
      if (!colour) {
        row.body.append(text);
        continue;
      }
      const run = document.createElement("span");
      run.style.color = colour;
      run.textContent = text;
      row.body.append(run);
    }
  }
  row.body.style.color = info.look.text;
  row.body.style.textDecoration = info.look.strike ? `line-through ${tint(FG, 45)}` : "none";
  row.box.style.visibility = info.box ? "visible" : "hidden";
  row.box.checked = info.box === "on";
  row.box.disabled = info.locked;
  row.box.title = info.box === "off" ? "Switch on" : "Switch off";
  row.glyph.style.visibility = info.glyph ? "visible" : "hidden";
  row.glyph.textContent = info.glyph?.text ?? "";
  row.glyph.style.color = info.glyph?.colour ?? MUTED;
  row.glyph.style.cursor = info.glyph?.pickable && !info.locked ? "pointer" : "default";
  row.glyph.title = info.glyph?.title ?? "";
  row.number.textContent = info.number ? String(info.number) : "";
  row.number.title = info.numberTitle ?? "";
}

/**
 * A small button in the hover tools or a list's header.
 *
 * @param {Node} content - Its icon or label.
 * @param {string} hint - What it does, on hover.
 * @param {(element: HTMLButtonElement) => void} onPress - What it does.
 * @returns {HTMLButtonElement} The button.
 */
function toolButton(content, hint, onPress) {
  const element = document.createElement("button");
  element.type = "button";
  element.tabIndex = -1;
  element.title = hint;
  element.style.cssText = "display:inline-flex;align-items:center;justify-content:center;gap:3px;"
    + `height:${LINE_HEIGHT - 4}px;min-width:${LINE_HEIGHT - 2}px;padding:0 4px;border:0;`
    + `border-radius:4px;background:transparent;color:${MUTED};font:500 10.5px/1 system-ui,sans-serif;`
    + "cursor:pointer;white-space:nowrap";
  element.append(content);
  element.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  element.addEventListener("mouseenter", () => {
    element.style.background = tint(FG, 12);
    element.style.color = FG;
  });
  element.addEventListener("mouseleave", () => {
    element.style.background = "transparent";
    element.style.color = MUTED;
  });
  element.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      onPress(element);
    } catch (error) {
      console.error(`[${EXT_NAME}] A line tool failed:`, error);
    }
  });
  return element;
}

/**
 * Show a tick on a copy button for a moment.
 *
 * @param {HTMLButtonElement} element - The button.
 * @param {boolean} done - Whether the copy worked.
 * @returns {void}
 */
function flashCopied(element, done) {
  if (!done) return;
  element.replaceChildren(icon(DONE_PATHS));
  setTimeout(() => element.replaceChildren(icon(COPY_PATHS)), COPIED_MS);
}

/**
 * Open a menu of modes under a button.
 *
 * @param {HTMLElement} anchor - What it opens under.
 * @param {string} heading - The menu's header.
 * @param {Array<[string, string, string]>} choices - `[mode, label, detail]` in order.
 * @param {string} current - The mode now.
 * @param {(mode: string) => void} onPick - What a pick does.
 * @returns {void}
 */
function openModeMenu(anchor, heading, choices, current, onPick) {
  const rect = anchor.getBoundingClientRect();
  openPopupMenu({
    x: rect.left,
    y: rect.bottom + 4,
    logName: EXT_NAME,
    items: [
      { header: heading },
      ...choices.map(([mode, label, detail]) => ({
        label, detail, checked: mode === current, onSelect: () => onPick(mode),
      })),
    ],
  });
}

/**
 * One box of coloured lines over a transparent text box: switches, versions and hover tools.
 *
 * @param {object} host - What the box edits: `read()` and `write(text)` the text, `locked()` while
 *   it is read-only, `context()` answering `{seed, skipBlank, skipComments, terms, numbering,
 *   joined, pick, all}`, `scroller` for keeping the caret in view, `hint`, `fill` to take the
 *   whole height, and `onStats(stats)` hearing what each draw counted.
 * @returns {object} `{element, box, render, schedule, sync, focus, dispose}`.
 */
function createLineBox(host) {
  const sheet = document.createElement("div");
  sheet.style.cssText = "position:relative;box-sizing:border-box;"
    + `min-height:${host.fill ? "100%" : `${LINE_HEIGHT * 2 + PAD_Y * 2}px`}`;
  const lines = document.createElement("div");
  lines.style.cssText = `position:relative;padding:${PAD_Y}px 0`;
  const hint = document.createElement("div");
  hint.style.cssText = `position:absolute;top:${PAD_Y}px;left:${GUTTER + PAD_X}px;right:${PAD_X}px;`
    + `pointer-events:none;color:${MUTED};${TEXT_LOOK}`;
  hint.textContent = host.hint;
  const box = document.createElement("textarea");
  box.spellcheck = false;
  box.setAttribute("aria-label", "Lines");
  box.style.cssText = `position:absolute;top:0;left:${GUTTER}px;width:calc(100% - ${GUTTER}px);height:100%;`
    + "margin:0;border:0;outline:none;resize:none;overflow:hidden;box-sizing:border-box;"
    + `padding:${PAD_Y}px ${PAD_X}px;background:transparent;color:transparent;`
    + `-webkit-text-fill-color:transparent;caret-color:${FG};${TEXT_LOOK}`;
  const tools = document.createElement("div");
  tools.style.cssText = `position:absolute;right:${PAD_X}px;z-index:2;display:none;align-items:center;`
    + `gap:1px;padding:1px;border-radius:5px;background:${themeVar("panelBg")};`
    + `box-shadow:0 1px 4px ${themeVar("shadow")}, 0 0 0 1px ${tint(FG, 16)}`;
  sheet.append(lines, hint, box, tools);

  const rows = [];
  let state = [];
  let current = -1;
  let frame = 0;
  let hovered = -1;

  /**
   * The line the caret is on while the box has focus.
   *
   * @returns {number} Line index from zero, or -1.
   */
  const caretLine = () => {
    if (document.activeElement !== box) return -1;
    const at = box.selectionStart ?? 0;
    let line = 0;
    for (let index = box.value.indexOf("\n"); index !== -1 && index < at; index = box.value.indexOf("\n", index + 1)) {
      line += 1;
    }
    return line;
  };

  /**
   * Scroll the caret's line into view.
   *
   * @returns {void}
   */
  const revealCaret = () => {
    const row = rows[current]?.row;
    const scroller = host.scroller;
    if (!row || !scroller) return;
    const top = row.offsetTop + sheet.offsetTop;
    const bottom = top + row.offsetHeight;
    if (top < scroller.scrollTop) scroller.scrollTop = top;
    else if (bottom > scroller.scrollTop + scroller.clientHeight) {
      scroller.scrollTop = Math.min(top, bottom - scroller.clientHeight);
    }
  };

  /**
   * Write the text box through the host, then draw.
   *
   * @returns {void}
   */
  const commit = () => {
    if (box.value !== host.read()) host.write(box.value);
    render();
    revealCaret();
  };

  /**
   * Replace the text with an edited copy as one edit of the text box, so Ctrl+Z takes it back.
   *
   * @param {string} next - The text after the edit.
   * @param {number[]|null} [selection] - Where the selection goes after, or null to keep it.
   * @returns {void}
   */
  const applyText = (next, selection = null) => {
    if (host.locked()) return;
    const before = box.value;
    if (next === before) return;
    let start = 0;
    while (start < before.length && start < next.length && before[start] === next[start]) start += 1;
    let endOld = before.length;
    let endNew = next.length;
    while (endOld > start && endNew > start && before[endOld - 1] === next[endNew - 1]) {
      endOld -= 1;
      endNew -= 1;
    }
    const focused = document.activeElement === box;
    const [from, to] = [box.selectionStart ?? 0, box.selectionEnd ?? 0];
    const moved = (position) => {
      if (position <= start) return position;
      if (position >= endOld) return position + (endNew - endOld);
      return endNew;
    };
    const insert = next.slice(start, endNew);
    withGraphChange(() => {
      box.focus({ preventScroll: true });
      box.setSelectionRange(start, endOld);
      let done = false;
      try {
        done = insert
          ? document.execCommand("insertText", false, insert)
          : document.execCommand("delete", false);
      } catch {
        done = false;
      }
      if (!done || box.value !== next) {
        box.value = next;
        commit();
      }
      if (selection) {
        box.setSelectionRange(selection[0], selection[1]);
      } else if (focused) {
        box.setSelectionRange(moved(from), moved(to));
      } else {
        const end = next.indexOf("\n", endNew);
        const caret = end < 0 ? next.length : end;
        box.setSelectionRange(caret, caret);
      }
    });
    render();
  };

  const modeLabel = document.createElement("span");
  const copyButton = toolButton(icon(COPY_PATHS), "Copy line", (element) => {
    const text = state[hovered]?.copy;
    if (typeof text !== "string") return;
    copyText(text).then((done) => flashCopied(element, done));
  });
  const addButton = toolButton(icon(PLUS_PATHS), "Add a version", () => {
    if (hovered < 0) return;
    const made = state[hovered]?.heading ? addLine(box.value, hovered) : addVersion(box.value, hovered);
    if (made) applyText(made.text, [made.start, made.end]);
  });
  const modeButton = toolButton(modeLabel, "How versions take turns", (element) => {
    const index = hovered;
    const found = groupAt(groupLines(classify(box.value)), index);
    if (!found) return;
    const modes = found.group.kind === TERM ? MODES.filter((mode) => mode !== ALL) : MODES;
    const all = host.context().all;
    openModeMenu(element, "Versions", modes.map((mode) => [mode, MODE_NAMES[mode], MODE_HINTS[mode] ?? all]),
      found.group.mode, (mode) => applyText(setMode(box.value, index, mode)));
  });
  tools.append(copyButton, addButton, modeButton);

  /**
   * Hide the hover tools.
   *
   * @returns {void}
   */
  const hideTools = () => {
    hovered = -1;
    tools.style.display = "none";
  };

  /**
   * Show the hover tools on one line.
   *
   * @param {number} index - Line index from zero.
   * @returns {void}
   */
  const showTools = (index) => {
    const spot = state[index];
    if (!spot || (typeof spot.copy !== "string" && !spot.addable) || host.locked()) {
      hideTools();
      return;
    }
    hovered = index;
    copyButton.style.display = typeof spot.copy === "string" ? "" : "none";
    copyButton.title = spot.heading ? "Copy the block" : "Copy line";
    addButton.style.display = spot.addable ? "" : "none";
    addButton.title = spot.heading ? "Add a line" : "Add a version";
    modeButton.style.display = spot.mode ? "" : "none";
    if (spot.mode) modeLabel.textContent = `${MODE_NAMES[spot.mode]} ▾`;
    tools.style.display = "flex";
    tools.style.top = `${rows[index].row.offsetTop + 2}px`;
  };

  /**
   * The line under the pointer.
   *
   * @param {PointerEvent} event - A pointer event over the lines.
   * @returns {number} Line index from zero, or -1.
   */
  const rowAtPointer = (event) => {
    const rect = lines.getBoundingClientRect();
    if (!rect.height || !lines.offsetHeight) return -1;
    const y = (event.clientY - rect.top) * (lines.offsetHeight / rect.height);
    for (let index = 0; index < rows.length; index += 1) {
      const top = rows[index].row.offsetTop;
      if (y >= top && y < top + rows[index].row.offsetHeight) return index;
    }
    return -1;
  };

  /**
   * Draw every line from the text box and what the host says about it.
   *
   * @returns {void}
   */
  const render = () => {
    cancelAnimationFrame(frame);
    frame = 0;
    const context = host.context();
    const locked = host.locked();
    const text = box.value;
    const { seed, skipBlank, skipComments } = context;
    const parsed = classify(text);
    const groups = groupLines(parsed);
    const own = glossary(parsed, groups, seed);
    const terms = new Map([...(context.terms ?? new Map()), ...own]);
    const hidden = hiddenLines(parsed);
    const resolved = resolve(text, seed, context.terms);
    const raw = text.split("\n");
    current = caretLine();
    while (rows.length < raw.length) {
      const row = createRow();
      const index = rows.length;
      row.box.addEventListener("pointerdown", (event) => event.preventDefault());
      row.box.addEventListener("click", (event) => {
        event.stopPropagation();
        applyText(switchLine(box.value, index));
      });
      row.glyph.addEventListener("pointerdown", (event) => event.preventDefault());
      row.glyph.addEventListener("click", (event) => {
        event.stopPropagation();
        const pick = state[index]?.pickable;
        if (pick === "version") applyText(pickVersion(box.value, index));
        else if (pick === "item" && !host.locked()) host.context().pick?.onPick(state[index].position + 1);
      });
      rows.push(row);
      lines.append(row.row);
    }
    while (rows.length > raw.length) rows.pop().row.remove();

    const place = new Map();
    let hue = 0;
    for (const group of groups) {
      const colour = group.kind === TERM ? TERM_HUE : (PROMPT_HUES[hue % PROMPT_HUES.length] ?? ACCENT);
      if (group.kind !== TERM) hue += 1;
      group.versions.forEach((line, position) => place.set(line, { group, position, colour }));
    }

    // Entry numbers, the prompts a list could hand on, and how many sit under each heading.
    const numberAt = new Map();
    const candidates = [];
    const underHeading = new Map();
    let heading = -1;
    let entries = 0;
    for (const item of resolved) {
      if (item.kind === SECTION) {
        heading = item.line;
        underHeading.set(heading, 0);
        continue;
      }
      const prompt = item.kind === PROMPT && Boolean(item.text.trim());
      if (prompt) {
        candidates.push(item.line);
        if (heading >= 0) underHeading.set(heading, underHeading.get(heading) + 1);
      }
      const kept = context.numbering === "items"
        ? prompt
        : handedOn(item.kind, skipBlank, skipComments) && !(context.joined && item.kind === BLANK);
      if (!kept) continue;
      entries += 1;
      if (context.numbering !== "none") numberAt.set(item.line, entries);
    }

    // Which line a list that hands on one is using.
    const picking = context.pick && context.pick.mode !== ALL && candidates.length ? context.pick : null;
    const inUseLine = picking ? candidates[(picking.pick - 1) % candidates.length] : -1;
    const itemAt = new Map();
    if (picking) candidates.forEach((line, position) => itemAt.set(line, { position, inUse: line === inUseLine }));

    const unknown = new Set();
    let off = 0;
    let notes = 0;
    state = raw.map((value, index) => {
      const line = parsed[index];
      const kind = line?.kind ?? END;
      const spot = place.get(index);
      const item = itemAt.get(index);
      const ownOff = Boolean(line?.off);
      const isOff = ownOff || Boolean(spot?.group.off);
      const under = Boolean(hidden[index]) && kind !== SECTION;
      if (ownOff) off += 1;
      const switchableNote = kind === NOTE && !skipComments && !line.block;
      if (kind === NOTE && !switchableNote) notes += 1;
      let look = PLAIN_LOOK;
      let glyph = null;
      let boxState = null;
      let pickable = false;
      if (kind === SECTION) {
        look = ownOff ? OFF_LOOK : SECTION_LOOK;
        boxState = ownOff ? "off" : "on";
        glyph = { text: "≡", colour: CARD_LOOK.title, title: "Heading" };
      } else if (spot) {
        const { group, position, colour } = spot;
        const chosen = group.mode === ALL ? group.live.includes(position) : position === group.pick;
        const inUse = chosen && (!picking || Boolean(item?.inUse));
        look = isOff ? OFF_LOOK : inUse ? lit(colour) : dim(colour);
        boxState = ownOff ? "off" : "on";
        if (group.versions.length > 1) {
          pickable = !isOff && !chosen && group.mode !== ALL ? "version" : false;
          glyph = chosen && !isOff
            ? { text: "●", colour, title: "In use" }
            : { text: "○", colour: pickable ? MUTED : DISABLED, title: pickable ? "Use this version" : "", pickable };
        } else if (item && !isOff) {
          pickable = item.inUse ? false : "item";
          glyph = item.inUse
            ? { text: "●", colour, title: "In use" }
            : { text: "○", colour: MUTED, title: "Use this line", pickable: true };
        } else if (group.kind === TERM) {
          glyph = { text: "$", colour: TERM_HUE, title: "Term" };
        }
      } else if (kind === NOTE) {
        look = NOTE_LOOK;
        if (switchableNote) boxState = "on";
        else glyph = { text: "i", colour: ACCENT, title: "Note, left out" };
      } else if (kind === BLANK && !skipBlank && context.numbering !== "items") {
        look = KEPT_BLANK_LOOK;
      }
      if (under) look = isOff ? OFF_LOOK : HIDDEN_LOOK;
      const plain = isOff || under || kind === END;
      const runs = plain ? [[value, null]] : highlight(value, line, terms, spot?.colour ?? ACCENT, unknown);
      let number = locked ? 0 : numberAt.get(index) ?? 0;
      let numberTitle = "";
      if (kind === SECTION && !ownOff && !locked) {
        number = underHeading.get(index) ?? 0;
        numberTitle = `${number} under this heading`;
      } else if (number) {
        numberTitle = context.numbering === "items" ? `Line ${number} of this list` : `Entry ${number} of the list`;
      }
      paintRow(rows[index], {
        look, runs, rule: kind === SECTION, current: index === current, box: boxState, locked, glyph,
        number, numberTitle,
      });
      let copy = null;
      if (kind === PROMPT || kind === VERSION) copy = line.body;
      else if (kind === TERM) copy = `$${line.name} = ${line.body}`;
      else if (kind === NOTE) copy = value;
      else if (kind === SECTION) {
        let end = index + 1;
        while (end < parsed.length && parsed[end].kind !== SECTION) end += 1;
        copy = raw.slice(index, end).join("\n");
      }
      return {
        copy,
        addable: kind === PROMPT || kind === TERM || kind === VERSION || kind === SECTION,
        heading: kind === SECTION,
        mode: spot && spot.group.versions.length > 1 ? spot.group.mode : null,
        pickable,
        position: item?.position ?? -1,
      };
    });

    hint.style.display = text ? "none" : "";
    box.readOnly = locked;
    sheet.style.opacity = locked ? "0.55" : "1";
    if (hovered >= 0) showTools(hovered);
    host.onStats({
      text, entries, off, notes, terms: own.size, unknown, candidates: candidates.length,
      choices: !locked && holdsChoice(text), turns: takesTurns(text),
    });
  };

  /**
   * Draw again on the next frame, once however many times it is asked.
   *
   * @returns {void}
   */
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(render);
  };

  /**
   * Load the host's text into the box where something else changed it, then draw.
   *
   * @returns {void}
   */
  const sync = () => {
    const value = host.read();
    if (value !== box.value) {
      const focused = document.activeElement === box;
      const [from, to] = [box.selectionStart ?? 0, box.selectionEnd ?? 0];
      box.value = value;
      if (focused) box.setSelectionRange(Math.min(from, value.length), Math.min(to, value.length));
    }
    schedule();
  };

  const onSelection = () => {
    if (document.activeElement !== box) return;
    if (caretLine() !== current) {
      render();
      revealCaret();
    }
  };
  box.addEventListener("input", commit);
  box.addEventListener("focus", () => {
    document.addEventListener("selectionchange", onSelection);
    schedule();
  });
  box.addEventListener("blur", () => {
    document.removeEventListener("selectionchange", onSelection);
    schedule();
  });
  sheet.addEventListener("pointermove", (event) => {
    if (host.locked()) {
      hideTools();
      return;
    }
    if (tools.contains(event.target)) return;
    const index = rowAtPointer(event);
    if (index < 0) hideTools();
    else if (index !== hovered) showTools(index);
  });
  sheet.addEventListener("pointerleave", hideTools);
  // A press under the last line puts the caret at the end.
  sheet.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target !== sheet) return;
    event.preventDefault();
    box.focus({ preventScroll: true });
    box.setSelectionRange(box.value.length, box.value.length);
  });
  lines.addEventListener("contextmenu", (event) => event.preventDefault());
  box.value = host.read();

  return {
    element: sheet,
    box,
    render,
    schedule,
    sync,
    focus() {
      box.focus({ preventScroll: true });
      box.setSelectionRange(box.value.length, box.value.length);
    },
    dispose() {
      cancelAnimationFrame(frame);
      document.removeEventListener("selectionchange", onSelection);
    },
  };
}

/**
 * The header over an editor: its summary, chips, and New draw, Preview and syntax buttons.
 *
 * @param {object} handlers - `{onDraw, onPreview}`.
 * @returns {object} `{element, summary, chips, drawButton, viewButton, paint}`, `paint` writing the
 *   summary from counts.
 */
function createHead(handlers) {
  const head = document.createElement("div");
  head.style.cssText = "flex:none;display:flex;align-items:center;gap:4px;min-width:0;padding:5px 6px 3px 8px";
  const summary = document.createElement("div");
  summary.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;font:11.5px/16px system-ui,sans-serif;color:${CARD_LOOK.muted}`;
  const chips = document.createElement("div");
  chips.style.cssText = "flex:none;display:flex;gap:4px";
  const drawButton = button("New draw", () => handlers.onDraw(), {
    kind: "ghost", hint: "Draw every {a|b} choice again", logName: EXT_NAME,
  });
  const viewButton = button("Preview", () => handlers.onPreview(), {
    kind: "ghost", hint: "What the next run hands on", logName: EXT_NAME,
  });
  const helpButton = button("?", () => {
    const rect = helpButton.getBoundingClientRect();
    openPopupMenu({
      x: rect.left,
      y: rect.bottom + 4,
      logName: EXT_NAME,
      items: [{ header: "Syntax" }, ...SYNTAX.map(([label, detail]) => ({ label, detail }))],
    });
  }, { kind: "ghost", hint: "Syntax", logName: EXT_NAME });
  for (const element of [drawButton, viewButton, helpButton]) {
    element.style.padding = "1px 6px";
    element.style.fontSize = "11.5px";
  }
  head.append(summary, chips, drawButton, viewButton, helpButton);
  head.addEventListener("contextmenu", (event) => event.preventDefault());

  /**
   * Write the summary and chips.
   *
   * @param {object} counts - `{wired, text, parts, title, over, empty, emptyChip, unknown}`.
   * @returns {void}
   */
  const paint = (counts) => {
    summary.replaceChildren();
    chips.replaceChildren();
    summary.title = counts.title ?? "";
    if (counts.wired) {
      summary.append("Using the wired text");
      chips.append(cardChip("wired", "accent"));
      return;
    }
    if (!counts.text) {
      summary.append("Nothing typed yet");
      return;
    }
    counts.parts.forEach(([value, words], index) => {
      if (index) summary.append(" · ");
      const figure = document.createElement("span");
      figure.style.cssText = `font-weight:650;color:${CARD_LOOK.title}`;
      figure.textContent = String(value);
      summary.append(figure, words);
    });
    if (counts.over) {
      const chip = cardChip(`over ${MAX_COMBINED}`, "error");
      chip.title = `Every combination makes more than ${MAX_COMBINED} entries`;
      chips.append(chip);
    } else if (counts.empty) {
      chips.append(cardChip(counts.emptyChip, "warning"));
    }
    if (counts.unknown.size) {
      const names = [...counts.unknown];
      const chip = cardChip(names.length === 1 ? `$${names[0]} unknown` : `${names.length} unknown terms`, "warning");
      chip.title = names.map((name) => `$${name}`).join(", ");
      chips.append(chip);
    }
  };

  return { element: head, summary, chips, drawButton, viewButton, paint };
}

/**
 * The preview of what the next run hands on, each entry with a copy button on hover.
 *
 * @returns {object} `{element, render(output, claim, caption, joined), clear, dispose}`.
 */
function createPreview() {
  const preview = document.createElement("div");
  preview.style.cssText = "display:none;box-sizing:border-box;min-height:100%;padding:4px 0 6px";
  let key = "";
  let releaseGlyph = () => {};

  const row = (number, text) => {
    const element = document.createElement("div");
    element.style.cssText = `position:relative;display:flex;min-width:0;box-shadow:inset 0 -1px 0 ${tint(DEEP, 55)}`;
    const gutter = document.createElement("div");
    gutter.style.cssText = `flex:0 0 ${GUTTER}px;box-sizing:border-box;padding-right:4px;text-align:right;`
      + `font:500 10px/${LINE_HEIGHT}px system-ui,sans-serif;font-variant-numeric:tabular-nums;color:${MUTED}`;
    gutter.textContent = String(number);
    const body = document.createElement("div");
    body.style.cssText = `flex:1 1 auto;min-width:0;padding:0 ${PAD_X}px;color:${FG};${TEXT_LOOK}`;
    body.textContent = text || ZERO_WIDTH;
    const copy = toolButton(icon(COPY_PATHS), "Copy", (target) => {
      copyText(text).then((done) => flashCopied(target, done));
    });
    copy.style.cssText += `;position:absolute;right:${PAD_X}px;top:2px;visibility:hidden;`
      + `background:${themeVar("panelBg")}`;
    element.addEventListener("mouseenter", () => { copy.style.visibility = "visible"; });
    element.addEventListener("mouseleave", () => { copy.style.visibility = "hidden"; });
    element.append(gutter, body, copy);
    return element;
  };

  return {
    element: preview,
    render(output, claim, caption, joined) {
      const next = JSON.stringify([output, claim, caption, joined]);
      if (next === key) return;
      key = next;
      preview.replaceChildren();
      const top = document.createElement("div");
      top.style.cssText = `display:flex;align-items:center;gap:6px;padding:2px ${PAD_X}px 4px ${STRIPE + 5}px;`
        + `font:11px/16px system-ui,sans-serif;color:${MUTED}`;
      const glyph = document.createElement("canvas");
      glyph.style.cssText = `flex:none;width:${ICON_SIZE}px;height:${ICON_SIZE}px`;
      const label = document.createElement("span");
      label.textContent = caption;
      top.append(glyph, label);
      preview.append(top);
      releaseGlyph();
      drawFidelityGlyph(glyph, claim);
      releaseGlyph = watchSurfaceRatio(glyph, () => drawFidelityGlyph(glyph, claim));
      const items = joined ? output.filter(Boolean) : output;
      if (!items.length) {
        const empty = document.createElement("div");
        empty.style.cssText = `padding:0 ${PAD_X}px 0 ${GUTTER + PAD_X}px;color:${MUTED};${TEXT_LOOK}`;
        empty.textContent = "Nothing to hand on";
        preview.append(empty);
        return;
      }
      if (joined) preview.append(row("", items[0]));
      else items.forEach((text, index) => preview.append(row(index + 1, text)));
    },
    clear() {
      if (!key) return;
      key = "";
      releaseGlyph();
      releaseGlyph = () => {};
    },
    dispose() {
      releaseGlyph();
    },
  };
}

/**
 * How close a preview is to the run.
 *
 * @param {string[]} output - What the preview shows.
 * @param {boolean} movesFirst - Whether something moves on before the run.
 * @returns {{icon: string, detail: string}} The claim the preview's glyph makes.
 */
function previewClaim(output, movesFirst) {
  if (output.some((item) => /(?<!\\)\[[^\]\n]*\]/.test(item))) {
    return { icon: ICON.APPROXIMATE, detail: "[tokens] fill in at run" };
  }
  if (movesFirst) return { icon: ICON.APPROXIMATE, detail: "moves on before run" };
  return { icon: ICON.EXACT, detail: "" };
}

/**
 * Whether the seed's control draws a new seed for the next run.
 *
 * @param {object} node - The node.
 * @returns {boolean} True when its control is anything but `fixed`.
 */
function seedMoves(node) {
  const control = findWidget(node, SEED_WIDGET)?.linkedWidgets?.[0]?.value;
  return Boolean(control) && control !== "fixed";
}

/**
 * Draw the seed again.
 *
 * @param {object} node - The node.
 * @returns {void}
 */
function newDraw(node) {
  const widget = findWidget(node, SEED_WIDGET);
  if (!widget) return;
  withGraphChange(() => {
    widget.value = Math.floor(Math.random() * SEED_SPAN);
    widget.callback?.(widget.value);
  });
  node.setDirtyCanvas?.(true, true);
}

/**
 * The panel shell every editor shares: header, rule, a scrolling body and the preview.
 *
 * @param {object} handlers - `{onDraw, onPreview}`.
 * @returns {object} `{root, head, scroller, preview, releaseWheel}`.
 */
function createShell(handlers) {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:0;"
    + "box-sizing:border-box;overflow:hidden";
  const head = createHead(handlers);
  const scroller = document.createElement("div");
  scroller.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;position:relative;"
    + `scrollbar-width:thin;scrollbar-color:${themeVar("scrollbarThumb")} transparent`;
  const preview = createPreview();
  root.append(head.element, cardRule(), scroller);
  root.addEventListener("pointerdown", (event) => {
    if (event.button === 1) app.canvas?.processMouseDown?.(event);
  });
  const releaseWheel = captureWheel(root, (event) => {
    if (scroller.scrollHeight <= scroller.clientHeight) return false;
    const before = scroller.scrollTop;
    scroller.scrollTop += wheelPixels(event, scroller).y;
    return scroller.scrollTop !== before;
  });
  return { root, head, scroller, preview, releaseWheel };
}

/**
 * The editor of String List and Multiline Text (Advanced): one box of lines.
 *
 * @param {object} node - The node it draws.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {object} The panel: its element and height, and `sync`, `schedule`, `refresh` and
 *   `dispose`.
 */
function createEditor(node, setup) {
  const widget = () => findWidget(node, setup.text);
  const linked = () => inputLinked(node, WIRE_INPUT);
  let previewing = false;
  let lines = null;

  const shell = createShell({
    onDraw: () => {
      newDraw(node);
      lines.render();
    },
    onPreview: () => {
      previewing = !previewing;
      lines.render();
    },
  });
  const { head, scroller, preview } = shell;

  /**
   * Write the summary and, when it is open, the preview, from what the box counted.
   *
   * @param {object} stats - From the box.
   * @returns {void}
   */
  const paint = (stats) => {
    const wired = linked();
    const seed = readSeed(node);
    const skipBlank = readSwitch(node, BLANK_WIDGET, setup.skipBlank);
    const skipComments = readSwitch(node, COMMENT_WIDGET, true);
    const output = setup.output === "joined"
      ? [joinLines(stats.text, skipBlank, skipComments, readName(node, JOIN_WIDGET, "line break"), seed)]
      : listLines(stats.text, skipBlank, skipComments, seed);
    const plural = (count, one, many) => (count === 1 ? one : many);
    const parts = [[stats.entries, ` ${plural(stats.entries, setup.one, setup.many)}`]];
    if (stats.terms) parts.push([stats.terms, plural(stats.terms, " term", " terms")]);
    if (stats.off) parts.push([stats.off, " off"]);
    if (stats.notes) parts.push([stats.notes, plural(stats.notes, " note", " notes")]);
    let title = "";
    if (setup.output === "joined") {
      const joined = output[0] ?? "";
      parts.push([joined.length, plural(joined.length, " character", " characters")]);
      title = joined.length > PREVIEW_CHARS ? `${joined.slice(0, PREVIEW_CHARS)}…` : joined;
    }
    head.paint({
      wired, text: stats.text, parts, title, over: false, empty: !stats.entries,
      emptyChip: setup.output === "joined" ? "empty text" : "nothing to hand on", unknown: stats.unknown,
    });
    head.drawButton.style.display = stats.choices && !wired ? "" : "none";
    head.viewButton.textContent = previewing ? "Edit" : "Preview";
    const showPreview = previewing && !wired;
    lines.element.style.display = showPreview ? "none" : "";
    preview.element.style.display = showPreview ? "block" : "none";
    if (!showPreview) {
      preview.clear();
      return;
    }
    const movesFirst = runsBefore() && (stats.turns || (stats.choices && seedMoves(node)));
    preview.render(output, previewClaim(output, movesFirst), stats.choices ? `Next run · seed ${seed}` : "Next run",
      setup.output === "joined");
  };

  lines = createLineBox({
    read: () => normalise(widget()?.value),
    write: (text) => {
      const target = widget();
      if (!target || normalise(target.value) === text) return;
      target.value = text;
      node.setDirtyCanvas?.(true, true);
    },
    locked: linked,
    context: () => ({
      seed: readSeed(node),
      skipBlank: readSwitch(node, BLANK_WIDGET, setup.skipBlank),
      skipComments: readSwitch(node, COMMENT_WIDGET, true),
      terms: null,
      numbering: setup.numbered ? "entries" : "none",
      joined: setup.output === "joined",
      pick: null,
      all: setup.all,
    }),
    scroller,
    hint: setup.hint,
    fill: true,
    onStats: paint,
  });
  scroller.append(lines.element, preview.element);

  /**
   * Draw again, folding the text box away first.
   *
   * @returns {void}
   */
  const schedule = () => {
    foldTextBox(node, setup.text);
    lines.schedule();
  };

  return {
    element: shell.root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    sync: () => {
      foldTextBox(node, setup.text);
      lines.sync();
    },
    schedule,
    refresh: () => {
      foldTextBox(node, setup.text);
      lines.sync();
    },
    dispose() {
      lines.dispose();
      preview.dispose();
      shell.releaseWheel();
    },
  };
}

/**
 * The editor of String List (Sections): a stack of lists, each its own box with a header.
 *
 * @param {object} node - The node it draws.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {object} The panel: its element and height, and `sync`, `schedule`, `refresh` and
 *   `dispose`.
 */
function createListsEditor(node, setup) {
  const widget = () => findWidget(node, setup.text);
  let lists = readLists(widget()?.value);
  let previewing = false;
  let frame = 0;
  const cards = [];

  const shell = createShell({
    onDraw: () => {
      newDraw(node);
      renderAll();
    },
    onPreview: () => {
      previewing = !previewing;
      paint();
    },
  });
  const { head, scroller, preview } = shell;
  const stack = document.createElement("div");
  stack.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:6px";
  const addRow = document.createElement("div");
  addRow.style.cssText = "display:flex;justify-content:center;padding:0 6px 8px";
  const addButton = button("+ Add list", () => addList(), { kind: "ghost", hint: "A new list under the last", logName: EXT_NAME });
  addButton.style.padding = "2px 10px";
  addButton.style.fontSize = "11.5px";
  addRow.append(addButton);
  const body = document.createElement("div");
  body.append(stack, addRow);
  scroller.append(body, preview.element);

  /**
   * Store the lists in the widget.
   *
   * @returns {void}
   */
  const store = () => {
    const target = widget();
    const value = JSON.stringify(lists);
    if (!target || target.value === value) return;
    target.value = value;
    node.setDirtyCanvas?.(true, true);
  };

  /**
   * Every term any switched-on list defines, by name.
   *
   * @returns {Map<string, string>} Name to value.
   */
  const sharedTerms = () => {
    const seed = readSeed(node);
    const terms = new Map();
    lists.forEach((entry, index) => {
      if (!entry.on) return;
      const parsed = classify(entry.lines);
      for (const [name, value] of glossary(parsed, groupLines(parsed), listSeed(seed, index))) terms.set(name, value);
    });
    return terms;
  };

  /**
   * Change one list and store it, as one step of the graph's undo.
   *
   * @param {number} index - The list's place.
   * @param {object} patch - The fields that change.
   * @returns {void}
   */
  const setList = (index, patch) => {
    if (!lists[index]) return;
    withGraphChange(() => {
      lists = lists.map((entry, at) => (at === index ? { ...entry, ...patch } : entry));
      store();
    });
    renderAll();
  };

  /**
   * Change the lists' order or number and draw the cards again.
   *
   * @param {object[]} next - The lists after the change.
   * @param {number} [focus] - The list whose box takes the caret after.
   * @returns {void}
   */
  const reshape = (next, focus = -1) => {
    withGraphChange(() => {
      lists = next;
      store();
    });
    rebuild();
    if (focus >= 0) cards[focus]?.lines.focus();
  };

  const addList = () => {
    const next = [...lists, { name: `List ${lists.length + 1}`, mode: ALL, pick: 1, on: true, lines: "" }];
    reshape(next, next.length - 1);
  };

  /**
   * Build one list's card: its header and its box of lines.
   *
   * @param {number} index - The list's place.
   * @returns {object} The card.
   */
  const createCard = (index) => {
    const card = document.createElement("div");
    card.style.cssText = `border-radius:6px;overflow:hidden;background:${tint(FG, 3)};`
      + `box-shadow:0 0 0 1px ${tint(FG, 12)}`;
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:4px;min-width:0;padding:3px 4px 3px 8px;"
      + `background:${tint(FG, 7)};box-shadow:inset 0 -1px 0 ${tint(DEEP, 55)}`;
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.tabIndex = -1;
    toggle.style.cssText = `flex:none;margin:0 2px 0 0;width:${BOX}px;height:${BOX}px;cursor:pointer;accent-color:${ACCENT}`;
    toggle.addEventListener("pointerdown", (event) => event.stopPropagation());
    toggle.addEventListener("change", () => setList(index, { on: toggle.checked }));
    const name = document.createElement("input");
    name.type = "text";
    name.spellcheck = false;
    name.placeholder = `List ${index + 1}`;
    name.style.cssText = "flex:1 1 auto;min-width:40px;margin:0;padding:1px 4px;border:0;border-radius:3px;"
      + `background:transparent;color:${CARD_LOOK.title};font:600 12px/16px system-ui,sans-serif;outline:none`;
    name.addEventListener("focus", () => { name.style.background = tint(FG, 8); });
    name.addEventListener("blur", () => { name.style.background = "transparent"; });
    name.addEventListener("input", () => {
      lists = lists.map((entry, at) => (at === index ? { ...entry, name: name.value } : entry));
      store();
      schedulePaint();
    });
    name.addEventListener("keydown", (event) => {
      if (event.key === "Enter") cards[index]?.lines.focus();
    });
    const count = document.createElement("span");
    count.style.cssText = `flex:none;font:500 10.5px/16px system-ui,sans-serif;font-variant-numeric:tabular-nums;color:${MUTED}`;
    const modeLabel = document.createElement("span");
    const mode = toolButton(modeLabel, "What this list hands on", (element) => {
      openModeMenu(element, "This list hands on", LIST_MODES.map((choice) => [choice, LIST_NAMES[choice], LIST_HINTS[choice]]),
        lists[index]?.mode, (choice) => setList(index, { mode: choice }));
    });
    mode.style.color = FG;
    const up = toolButton(icon(UP_PATHS), "Move up", () => {
      if (index < 1) return;
      const next = [...lists];
      [next[index - 1], next[index]] = [next[index], next[index - 1]];
      reshape(next);
    });
    const down = toolButton(icon(DOWN_PATHS), "Move down", () => {
      if (index >= lists.length - 1) return;
      const next = [...lists];
      [next[index + 1], next[index]] = [next[index], next[index + 1]];
      reshape(next);
    });
    const remove = toolButton(icon(REMOVE_PATHS), "Remove this list", () => {
      reshape(lists.filter((_, at) => at !== index));
    });
    header.append(toggle, name, count, mode, up, down, remove);
    header.addEventListener("contextmenu", (event) => event.preventDefault());

    let stats = null;
    const lines = createLineBox({
      read: () => normalise(lists[index]?.lines),
      write: (text) => {
        if (!lists[index] || lists[index].lines === text) return;
        lists = lists.map((entry, at) => (at === index ? { ...entry, lines: text } : entry));
        store();
        for (const other of cards) if (other.index !== index) other.lines.schedule();
      },
      locked: () => false,
      context: () => ({
        seed: listSeed(readSeed(node), index),
        skipBlank: true,
        skipComments: true,
        terms: sharedTerms(),
        numbering: "items",
        joined: false,
        pick: { mode: lists[index]?.mode ?? ALL, pick: lists[index]?.pick ?? 1, onPick: (pick) => setList(index, { pick }) },
        all: setup.all,
      }),
      scroller,
      hint: setup.hint,
      fill: false,
      onStats: (found) => {
        stats = found;
        paintHeader();
        schedulePaint();
      },
    });
    card.append(header, lines.element);

    /**
     * Write the header from the list and what its box counted.
     *
     * @returns {void}
     */
    const paintHeader = () => {
      const entry = lists[index];
      if (!entry) return;
      toggle.checked = entry.on;
      if (document.activeElement !== name) name.value = entry.name;
      const total = stats?.candidates ?? 0;
      count.textContent = entry.mode === ALL || !total
        ? `${total} ${total === 1 ? "line" : "lines"}`
        : `${((entry.pick - 1) % total) + 1} of ${total}`;
      modeLabel.textContent = `${LIST_NAMES[entry.mode]} ▾`;
      lines.element.style.opacity = entry.on ? "1" : "0.45";
      up.style.visibility = index > 0 ? "visible" : "hidden";
      down.style.visibility = index < lists.length - 1 ? "visible" : "hidden";
    };

    return { index, element: card, lines, paintHeader, stats: () => stats };
  };

  /**
   * Throw the cards away and build one per list.
   *
   * @returns {void}
   */
  const rebuild = () => {
    for (const card of cards.splice(0)) card.lines.dispose();
    stack.replaceChildren();
    lists.forEach((_, index) => {
      const card = createCard(index);
      cards.push(card);
      stack.append(card.element);
      card.paintHeader();
    });
    renderAll();
  };

  /**
   * Draw every list's box now.
   *
   * @returns {void}
   */
  const renderAll = () => {
    for (const card of cards) card.lines.render();
    paint();
  };

  /**
   * Write the summary and, when it is open, the preview, from every list.
   *
   * @returns {void}
   */
  const paint = () => {
    cancelAnimationFrame(frame);
    frame = 0;
    foldTextBox(node, setup.text);
    const seed = readSeed(node);
    const found = listEntries(lists, seed);
    const output = combine(found.map((entry) => entry.entries), readName(node, COMBINE_WIDGET, CYCLE),
      readName(node, JOIN_WIDGET, "comma"));
    const over = output === null;
    const entries = over ? [] : output;
    const unknown = new Set();
    let terms = 0;
    let choices = false;
    let turns = false;
    for (const card of cards) {
      const stats = card.stats();
      if (!stats || !lists[card.index]?.on) continue;
      for (const name of stats.unknown) unknown.add(name);
      terms += stats.terms;
      choices ||= stats.choices;
      turns ||= stats.turns;
    }
    const on = lists.filter((entry) => entry.on).length;
    const parts = [[entries.length, ` ${entries.length === 1 ? setup.one : setup.many}`],
      [on, on === 1 ? " list" : " lists"]];
    if (terms) parts.push([terms, terms === 1 ? " term" : " terms"]);
    const title = lists.map((entry, index) => {
      const size = found.find((item) => item.list === index)?.candidates.length ?? 0;
      const name = entry.name || `List ${index + 1}`;
      return `${name}: ${entry.on ? `${LIST_NAMES[entry.mode]}, ${size} ${size === 1 ? "line" : "lines"}` : "off"}`;
    }).join("\n");
    head.paint({
      wired: false, text: lists.some((entry) => entry.lines), parts, title, over, empty: !entries.length,
      emptyChip: "nothing to hand on", unknown,
    });
    head.drawButton.style.display = choices ? "" : "none";
    head.viewButton.textContent = previewing ? "Edit" : "Preview";
    body.style.display = previewing ? "none" : "";
    preview.element.style.display = previewing ? "block" : "none";
    if (!previewing) {
      preview.clear();
      return;
    }
    const movesFirst = runsBefore() && (listsTakeTurns(lists, seed) || turns || (choices && seedMoves(node)));
    preview.render(entries, previewClaim(entries, movesFirst), choices ? `Next run · seed ${seed}` : "Next run", false);
  };

  /**
   * Write the summary on the next frame, once however many times it is asked.
   *
   * @returns {void}
   */
  const schedulePaint = () => {
    if (!frame) frame = requestAnimationFrame(paint);
  };

  /**
   * Read the lists again where something else changed the widget, then draw.
   *
   * @returns {void}
   */
  const sync = () => {
    foldTextBox(node, setup.text);
    const next = readLists(widget()?.value);
    if (JSON.stringify(next) === JSON.stringify(lists) && cards.length === lists.length) {
      schedulePaint();
      return;
    }
    const sameShape = next.length === cards.length;
    lists = next;
    if (!sameShape) {
      rebuild();
      return;
    }
    for (const card of cards) {
      card.lines.sync();
      card.paintHeader();
    }
    schedulePaint();
  };

  rebuild();

  return {
    element: shell.root,
    height: PANEL_HEIGHT + 120,
    minWidth: PANEL_WIDTH,
    sync,
    schedule: () => {
      for (const card of cards) card.lines.schedule();
      schedulePaint();
    },
    refresh: sync,
    dispose() {
      cancelAnimationFrame(frame);
      for (const card of cards) card.lines.dispose();
      preview.dispose();
      shell.releaseWheel();
    },
  };
}

/**
 * Move lines and lists that take turns on after a run, and hold the seed while no line draws.
 *
 * @param {object} node - The node being created.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {void}
 */
function attachTurns(node, setup) {
  const text = findWidget(node, setup.text);
  if (!text || text.__was_line_turns) return;
  text.__was_line_turns = true;
  const several = setup.output === "lists";
  let primed = false;
  const turn = () => {
    if (inputLinked(node, WIRE_INPUT)) return;
    let next;
    if (several) {
      const lists = readLists(text.value);
      const moved = advanceLists(lists, readSeed(node));
      if (moved === lists) return;
      next = JSON.stringify(moved);
    } else {
      const value = normalise(text.value);
      next = advance(value);
      if (next === value) return;
    }
    text.value = next;
    text.callback?.(next);
    node.setDirtyCanvas?.(true, true);
  };
  const draws = () => {
    if (inputLinked(node, WIRE_INPUT)) return false;
    if (!several) return holdsChoice(normalise(text.value));
    return readLists(text.value).some((entry) => entry.on && holdsChoice(entry.lines));
  };
  const before = text.beforeQueued;
  text.beforeQueued = function (...args) {
    const result = before?.apply(this, args);
    try {
      if (runsBefore() && primed) turn();
      primed = true;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to move the lines on:`, error);
    }
    return result;
  };
  const after = text.afterQueued;
  text.afterQueued = function (...args) {
    const result = after?.apply(this, args);
    try {
      if (!runsBefore()) turn();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to move the lines on:`, error);
    }
    return result;
  };
  for (const control of findWidget(node, SEED_WIDGET)?.linkedWidgets ?? []) {
    for (const hook of ["beforeQueued", "afterQueued"]) {
      const original = control[hook];
      if (typeof original !== "function") continue;
      control[hook] = function (...args) {
        if (!draws()) return undefined;
        return original.apply(this, args);
      };
    }
  }
}

/**
 * Put the editor on a node and wire it to the widgets it reads.
 *
 * @param {object} node - The node being created.
 * @param {object} setup - Its entry in `NODES`.
 * @returns {void}
 */
function attachEditor(node, setup) {
  if (!findWidget(node, setup.text)) return;
  const panel = setup.output === "lists" ? createListsEditor(node, setup) : createEditor(node, setup);
  foldTextBox(node, setup.text);
  appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  chainWidgetCallback(node, setup.text, panel.sync, EXT_NAME);
  for (const name of [BLANK_WIDGET, COMMENT_WIDGET, JOIN_WIDGET, COMBINE_WIDGET, SEED_WIDGET]) {
    chainWidgetCallback(node, name, panel.schedule, EXT_NAME);
  }

  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      panel.sync();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the lines after a load:`, error);
    }
    return result;
  };

  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      panel.schedule();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to follow a connection change:`, error);
    }
    return result;
  };

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      panel.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the editor:`, error);
    }
    return result;
  };

  panel.sync();
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Line Editor", "Editor"],
      name: "Show the line editor",
      tooltip:
        "Draw the lines of String List, String List (Sections) and Multiline Text (Advanced) in "
        + "colour, each with a switch and tools on hover. Off, the plain text box is drawn "
        + "instead and reads the same way. This applies to nodes added after the setting "
        + "changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const setup = NODES[nodeData?.name];
    if (!setup) return;
    const proto = nodeType.prototype;
    if (proto.__was_line_editor_wrapped) return;
    proto.__was_line_editor_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        attachTurns(this, setup);
        if (interfaceEnabled()) attachEditor(this, setup);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the line editor:`, error);
      }
      return result;
    };
  },
});
